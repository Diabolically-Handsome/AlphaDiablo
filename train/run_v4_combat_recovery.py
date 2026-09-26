"""The one and only protocol-v4 combat-recovery launcher.

This is not an upgrade of the historical ``run_v33_content.py``. The old driver is bound to protocol v3,
old evaluation archives and an old training recipe, and may only be read for forensics; this file is the independent v4 entry point.

Supported closed loop:

* ``prepare-bc``: re-sample the BC-v1 bound to the current implementation and the BC-v2 bound to M29;
* ``train``: only allows the 249,856-step safe-prefix replay v28->M29/KING;
* ``eval-regression``: two-arm regression on 7000-7031;
* ``eval-fresh``: fresh two-arm final exam on 12000-12031;
* ``analyze`` / ``status``: re-verify and summarize the frozen archives.

This driver offers no arbitrary step counts, arbitrary managers, short-run publication or old-protocol compatibility knobs. To change
any frozen constant, open a new campaign instead of adding an escape flag here.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import math
import os
import pathlib
import stat
import subprocess
import sys
import sysconfig
import time
import types
import zipfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(TRAIN))

from eval_contract import (  # noqa: E402
    EvalContractError,
    OutputReservationError,
    PROTOCOL_VERSION,
    exclusive_lock,
    expected_eval_identity,
    freeze_eval_identity,
    read_eval_archive,
    runtime_identity,
    strict_json_loads,
    validate_eval_archive,
)


class CampaignError(RuntimeError):
    """Fail-closed error of the official campaign."""


class CommandFailed(CampaignError):
    """Non-zero subprocess exit."""


LEGACY_STATE_SCHEMA = "official-v4-combat-recovery-state/1"
STATE_SCHEMA = "official-v4-combat-recovery-state/2"
FRESH_LEDGER_SCHEMA = "official-v4-fresh-ledger/2"
FRESH_POOL_OPENED_SCHEMA = "official-v4-fresh-pool-opened/1"
FRESH_CANDIDATE_FIRED_SCHEMA = "official-v4-fresh-candidate-fired/1"
EXPECTATIONS_SCHEMA = "diablogym-publication-expectations/1"
PAIRED_ANALYSIS_SCHEMA = "official-v4-paired-analysis/3"
LAUNCHER_RULES_REVISION = "combat-recovery-fsm/7"
REGRESSION_GATE_REVISION = "combat-recovery-regression-gate/4"
CAMPAIGN_REVISION = 6
FRESH_CANDIDATE_MAX_ATTEMPTS = 1
EVAL_ARCHIVE_IDENTITY_KEYS = frozenset({
    "path",
    "sha256",
    "worker_sha256",
    "worker_receipt_sha256",
    "manager_sha256",
    "protocol_bundle_sha256",
})

# R6 replays the exact safe prefix selected after R5's root-anchor TV circuit
# correctly tripped.  R5 remains immutable forensic evidence: it did not
# publish a candidate or open the one-shot fresh pool.  The R6 target is the
# only periodic checkpoint before that trip and its known-seed forensic replay
# passed every registered combat/safety component.  R6 must reproduce the
# frozen policy head below; it may not resume the failed in-memory trajectory.
CONTROL_DIR = TRAIN / "runs" / "v4-combat-recovery-r6-control"
STATE_PATH = CONTROL_DIR / "status.json"
CONTROL_LOCK = CONTROL_DIR / ".launcher.lock"
EXPECTATIONS_PATH = CONTROL_DIR / "publication-expectations.json"
FRESH_LEDGER_PATH = CONTROL_DIR / "fresh-ledger.jsonl"
FRESH_POOL_OPENED_PATH = CONTROL_DIR / "fresh-12000-pool-opened.json"
FRESH_CANDIDATE_FIRED_PATH = (
    CONTROL_DIR / "fresh-12000-candidate-attempt-1-fired.json"
)
PREREG_PREFLIGHT_DIR = CONTROL_DIR / "preregistered-liveness"
PREREG_PREFLIGHT_PATH = (
    PREREG_PREFLIGHT_DIR / "bc_aux_liveness_preflight.json"
)

CANDIDATE_RUN_NAME = "v4-combat-recovery-r6-safe-prefix"
CANDIDATE_DIR = TRAIN / "runs" / CANDIDATE_RUN_NAME
CANDIDATE_ZIP = CANDIDATE_DIR / "model_final.zip"
CANDIDATE_RECEIPT = CANDIDATE_DIR / "bc_aux_behavior_receipt.json"
CANDIDATE_PREFLIGHT = CANDIDATE_DIR / "bc_aux_liveness_preflight.json"

V28_ZIP = TRAIN / "models" / "v28-worker-leg1" / "model_final.zip"
# 2026-09-23: re-saved with neutral local paths (content otherwise unchanged); was 2f7bc9dd810956c3
V28_SHA256 = "0c6f014da19c3bf27b208d55adc76be13fcad8bc5f744b95f4f743be97426434"
V28_POLICY_HEAD_SHA256 = (
    "627814498a5c6ab5819d3d7abaea7ce4d37cef1dbf2fd388d160434e61b71d40"
)
M29_NPZ = TRAIN / "models" / "v29-manager-mfresh" / "policy.npz"
M29_SHA256 = "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6"
KING_SD = TRAIN / "runs" / "v32" / "king_anchor_sd.pt"
KING_SHA256 = "009aaad29d2653cde3f4e8ed2fafd8861a0f1f572a140c64118df9e3fa3df35d"
KING_MANIFEST = KING_SD.with_name(f"{KING_SD.name}.manifest.json")
KING_MANIFEST_SHA256 = (
    "d4dfcc587bc1b4a303b0103e171e33b9a72abbd5bc3a3660e0c9fc8a3a491a7e"
)

BC_V1_DIR = TRAIN / "runs" / "bc-worker"
BC_V1_POLICY = BC_V1_DIR / "policy_sd.pt"
BC_V1_DEMOS = BC_V1_DIR / "demos.npz"
BC_V2_DIR = TRAIN / "runs" / "bc-worker-v2"
BC_V2_POLICY = BC_V2_DIR / "policy_sd.pt"
BC_V2_DEMOS = BC_V2_DIR / "demos.npz"

START_STEPS = 3_497_984
LEG_STEPS = 249_856
TARGET_STEPS = 3_747_840
N_STEPS = 512
NUM_ENVS = 4
ROLLOUT_QUANTUM = N_STEPS * NUM_ENVS
TRAIN_CALLS = 122
TRAIN_SEED = 304_000
LEARNING_RATE = 1e-4
ENT_COEF = 0.005
TARGET_KL = 0.02
DISTILL_BETA = 0.015625
BC_AUX_LAMBDA = 0.0
BC_AUX_MODE = "expanded-trainable-a12-contextual-mixture"
TRAINING_CONTRACT_REVISION = 12
BC_AUX_OBJECTIVE_REVISION = 10
BC_AUX_CIRCUIT_SCHEMA = "a12-onpolicy-contextual-mixture-adapter/1"
CURRICULUM = "linear:1.0:0.5:147,hold:0.5:97"
R5_SAFE_CHECKPOINT_SHA256 = (
    "64a336a699b3277d4ec4d2378f4227deeb9709b2916f4a3e980f333b182856c5"
)
R5_SAFE_CHECKPOINT = (
    TRAIN / "runs" / "v4-combat-recovery-r5-full"
    / "ckpt" / "model_3747840_steps.zip"
)
R6_EXPECTED_POLICY_HEAD_SHA256 = (
    "c488714c40a57f87bc56ba577cdb6e0eded89d40b619d9071afc2058c68051fd"
)

EVAL_DIR = TRAIN / "runs" / "eval-assembled"
EVAL_POOLS = {
    "regression": tuple(range(7000, 7032)),
    "fresh": tuple(range(12000, 12032)),
}
EVAL_TAGS = {
    ("regression", "baseline"): "official-v4-r6-m29-v28-regression-7000",
    ("regression", "candidate"): "official-v4-r6-m29-candidate-regression-7000",
    ("fresh", "baseline"): "official-v4-r6-m29-v28-fresh-12000",
    ("fresh", "candidate"): "official-v4-r6-m29-candidate-fresh-12000",
}


def _require_seed_pool_discipline() -> None:
    """Fail closed if any training/BC/evaluation pool can overlap."""
    from diablogym.worker_env import (  # imported lazily for CLI startup
        BC_RESERVED_SEED_RANGES,
        is_reserved_train_seed,
    )

    registered_bc_ranges = ((2000, 2128), (3000, 3384))
    _require(
        all(seed_range in tuple(BC_RESERVED_SEED_RANGES)
            for seed_range in registered_bc_ranges),
        "the historical BC pools of the official launcher have disappeared from the worker rejection table",
    )
    pools = {
        "bc-v1": set(range(*registered_bc_ranges[0])),
        "bc-v2": set(range(*registered_bc_ranges[1])),
        "regression": set(EVAL_POOLS["regression"]),
        "fresh": set(EVAL_POOLS["fresh"]),
    }
    names = tuple(pools)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            _require(
                pools[left_name].isdisjoint(pools[right_name]),
                f"official seed pools overlap: {left_name}/{right_name}",
            )
    _require(
        all(
            is_reserved_train_seed(seed)
            for pool in pools.values()
            for seed in pool
        ),
        "official BC/eval seeds are not fully rejected by the ordinary training sampler",
    )


PAIRED_METRIC_DIRECTIONS = {
    "farm_worker_wage": "higher",
    "farm_worker_kills": "higher",
    # The dry/fresh stratification is only for attributing distribution differences between the training course and the official full evaluation;
    # raw stratum totals also depend on window exposure, so they must not quietly become a hard model-selection gate.
    "farm_dry_n": "descriptive",
    "farm_fresh_n": "descriptive",
    "farm_dry_worker_wage": "descriptive",
    "farm_fresh_worker_wage": "descriptive",
    "farm_dry_worker_kills": "descriptive",
    "farm_fresh_worker_kills": "descriptive",
    "ret": "higher",
    "kills": "higher",
    "died": "lower",
    # The number of active drinks has no monotonic good or bad in itself; whether drinks repeat is judged separately by the two metrics below.
    "farm_voluntary_drinks": "descriptive",
    "farm_reflex_drains": "lower",
    "farm_multi_drink_windows": "lower",
    "farm_max_voluntary_drinks_per_window": "lower",
    "ending_belt_heals": "higher",
}

# The complete 17-key lineage frozen exactly before launch. The training contract can be precomputed by the single-source-of-truth
# pure function of the current strict trainer; the liveness receipt is produced by first running the same production preflight on an isolated clone
# (no environment started, final heldout not read), and then writing its byte SHA into
# expectations. The official trainer must reproduce this preflight verbatim, otherwise the candidate is refused publication.
EXPECTED_PROVENANCE_KEYS = frozenset({
    "protocol_version",
    "implementation_sha256",
    "manager_npz_sha256",
    "resume_checkpoint_sha256",
    "teacher_sha256",
    "bc_aux_demos_sha256",
    "bc_aux_liveness_preflight_sha256",
    "training_contract_sha256",
    "start_steps",
    "target_global_steps",
    "seed",
    "optimizer_reset",
    "target_kl",
    "distill_beta",
    "bc_aux_lambda",
    "bc_aux_mode",
    "calib_record_only",
})

CAMPAIGN_RECIPE = {
    "protocol_version": 4,
    "campaign_revision": CAMPAIGN_REVISION,
    "resume_checkpoint_sha256": V28_SHA256,
    "manager_npz_sha256": M29_SHA256,
    "teacher_sha256": KING_SHA256,
    "start_steps": START_STEPS,
    "leg_steps": LEG_STEPS,
    "target_global_steps": TARGET_STEPS,
    "rollout_quantum": ROLLOUT_QUANTUM,
    "seed": TRAIN_SEED,
    "learning_rate": LEARNING_RATE,
    "ent_coef": ENT_COEF,
    "target_kl": TARGET_KL,
    "distill_beta": DISTILL_BETA,
    "bc_aux_lambda": BC_AUX_LAMBDA,
    "bc_aux_mode": BC_AUX_MODE,
    "training_contract_revision": TRAINING_CONTRACT_REVISION,
    "bc_aux_objective_revision": BC_AUX_OBJECTIVE_REVISION,
    "bc_aux_circuit_schema": BC_AUX_CIRCUIT_SCHEMA,
    "curriculum": CURRICULUM,
    "manager": "M29",
    "safe_prefix_replay": {
        "source_campaign_revision": 5,
        "source_checkpoint_sha256": R5_SAFE_CHECKPOINT_SHA256,
        "expected_policy_head_sha256": R6_EXPECTED_POLICY_HEAD_SHA256,
        "source_target_global_steps": TARGET_STEPS,
        "selection_basis":
            "pre-tv-trip-checkpoint-plus-known-seed-forensic-pass",
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


CAMPAIGN_RECIPE_SHA256 = _canonical_sha256(CAMPAIGN_RECIPE)


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_read(path: pathlib.Path, label: str) -> bytes:
    """Read a stable set of bytes from the same regular file; reject symlinks/replacement during the read."""
    path = pathlib.Path(path)
    _require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    try:
        with open(path, "rb") as stream:
            before = os.fstat(stream.fileno())
            _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file: {path}")
            payload = stream.read()
            after = os.fstat(stream.fileno())
        current = path.stat()
    except OSError as exc:
        raise CampaignError(f"{label} missing/cannot be read stably: {path}: {exc}") from exc
    signature = _stat_signature(before)
    _require(
        signature == _stat_signature(after)
        and signature == _stat_signature(current)
        and len(payload) == before.st_size,
        f"{label} changed while being read: {path}",
    )
    return payload


def _stable_sha256(path: pathlib.Path, label: str) -> str:
    return hashlib.sha256(_stable_read(path, label)).hexdigest()


def _launcher_identity() -> dict:
    """Bring the source of the state machine actually executed into the campaign, evaluation and analysis identity."""
    return {
        "rules_revision": LAUNCHER_RULES_REVISION,
        "sha256": _stable_sha256(pathlib.Path(__file__).resolve(), "official launcher"),
    }


def _require_sha(path: pathlib.Path, expected: str, label: str) -> str:
    actual = _stable_sha256(path, label)
    _require(actual == expected, f"{label} SHA drift: {actual} != {expected}")
    return actual


def _atomic_write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with open(tmp, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        tmp.unlink(missing_ok=True)


def _exclusive_create_json(path: pathlib.Path, value: Any, label: str) -> None:
    """Create a non-reusable one-time marker; refuse if it already exists, never overwriting history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise CampaignError(f"{label} one-time marker already exists: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        # A successful creation means the pool is open. Never delete the marker even on error, or a second launch could happen.
        raise


def _run_lock_is_held() -> bool:
    """Read-only probe of the trainer's kernel flock; leftover PID text is not trusted."""
    path = CANDIDATE_DIR / ".run.lock"
    if not path.exists():
        return False
    try:
        stream = open(path, "r", encoding="utf-8")
    except OSError as exc:
        raise CampaignError(f"cannot probe trainer run lock: {path}: {exc}") from exc
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        stream.close()


def _lock_file_is_held(path: pathlib.Path) -> bool:
    if not path.exists():
        return False
    try:
        stream = open(path, "r", encoding="utf-8")
    except OSError as exc:
        raise CampaignError(f"cannot probe lock: {path}: {exc}") from exc
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        stream.close()


def _new_state() -> dict:
    return {
        "schema_version": STATE_SCHEMA,
        "campaign_recipe_sha256": CAMPAIGN_RECIPE_SHA256,
        "campaign_recipe": dict(CAMPAIGN_RECIPE),
        "launcher": _launcher_identity(),
        "migration": None,
        "updated_at_ns": time.time_ns(),
        "phases": {},
    }


def _migrate_pre_scientific_legacy_state(state: dict, payload: bytes) -> dict:
    """Migrate only old control records that have not yet trained/evaluated, embedding the original record in full in the new state.

    An old campaign may only have left prepare-bc operational records. Once any candidate, regression, fresh
    or analysis phase appears, the old launcher must seal it itself; a schema upgrade must not be used to
    rewrite scientific history.
    """
    _require(
        isinstance(state, dict)
        and set(state)
        == {
            "schema_version",
            "campaign_recipe_sha256",
            "campaign_recipe",
            "updated_at_ns",
            "phases",
        }
        and state.get("schema_version") == LEGACY_STATE_SCHEMA
        and isinstance(state.get("phases"), dict),
        "old campaign state schema/fields are malformed; refusing automatic migration",
    )
    _require(
        state["campaign_recipe_sha256"] == CAMPAIGN_RECIPE_SHA256
        and state["campaign_recipe"] == CAMPAIGN_RECIPE,
        "old campaign recipe/manager identity differs from the current campaign; refusing migration",
    )
    scientific_phases = set(state["phases"]) - {"prepare_bc"}
    scientific_residue = [
        path
        for path in (
            CANDIDATE_ZIP,
            CANDIDATE_RECEIPT,
            CANDIDATE_PREFLIGHT,
            EXPECTATIONS_PATH,
            PREREG_PREFLIGHT_PATH,
            FRESH_LEDGER_PATH,
            FRESH_POOL_OPENED_PATH,
            FRESH_CANDIDATE_FIRED_PATH,
            CONTROL_DIR / "analysis-regression.json",
            CONTROL_DIR / "analysis-fresh.json",
            *(
                _eval_archive_path(pool, arm)
                for pool in EVAL_POOLS
                for arm in ("baseline", "candidate")
            ),
        )
        if path.exists()
    ]
    if CANDIDATE_DIR.is_dir():
        scientific_residue.extend(
            path for path in CANDIDATE_DIR.iterdir()
            if path.name != ".run.lock"
        )
    _require(
        not scientific_phases
        and not scientific_residue,
        "the old launcher already produced training/evaluation scientific state or leftovers; automatic migration is forbidden: "
        f"phases={sorted(scientific_phases)},"
        f"files={[str(path) for path in scientific_residue]}",
    )
    migrated = _new_state()
    migrated["migration"] = {
        "source_schema_version": LEGACY_STATE_SCHEMA,
        "source_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "source_state_sha256": _canonical_sha256(state),
        "source_state": state,
    }
    return migrated


def _load_state(*, required: bool = False) -> dict:
    if not STATE_PATH.is_file():
        _require(not required, f"official-v4 state does not exist: {STATE_PATH}")
        return _new_state()
    try:
        payload = _stable_read(STATE_PATH, "campaign state")
        state = strict_json_loads(payload)
    except (EvalContractError, ValueError) as exc:
        raise CampaignError(f"campaign state corrupted: {STATE_PATH}") from exc
    if (
        isinstance(state, dict)
        and state.get("schema_version") == LEGACY_STATE_SCHEMA
    ):
        return _migrate_pre_scientific_legacy_state(state, payload)
    _require(
        isinstance(state, dict)
        and set(state)
        == {
            "schema_version",
            "campaign_recipe_sha256",
            "campaign_recipe",
            "launcher",
            "migration",
            "updated_at_ns",
            "phases",
        },
        "campaign state schema/fields are malformed",
    )
    _require(state["schema_version"] == STATE_SCHEMA, "campaign state schema is outdated")
    _require(
        state["campaign_recipe_sha256"] == CAMPAIGN_RECIPE_SHA256
        and state["campaign_recipe"] == CAMPAIGN_RECIPE,
        "campaign frozen recipe differs from the current launcher; in-place migration is forbidden",
    )
    _require(
        state["launcher"] == _launcher_identity(),
        "the launcher source/rules bound to the campaign state have drifted; further launches are forbidden",
    )
    _require(
        state["migration"] is None
        or (
            isinstance(state["migration"], dict)
            and set(state["migration"])
            == {
                "source_schema_version",
                "source_payload_sha256",
                "source_state_sha256",
                "source_state",
            }
            and state["migration"]["source_schema_version"]
            == LEGACY_STATE_SCHEMA
            and _is_sha256(state["migration"]["source_payload_sha256"])
            and _is_sha256(state["migration"]["source_state_sha256"])
            and isinstance(state["migration"]["source_state"], dict)
            and state["migration"]["source_state_sha256"]
            == _canonical_sha256(state["migration"]["source_state"])
        ),
        "campaign migration record is malformed",
    )
    _require(isinstance(state["phases"], dict), "campaign phases is not an object")
    return state


def _save_state(state: dict) -> None:
    state["updated_at_ns"] = time.time_ns()
    _atomic_write_json(STATE_PATH, state)


def _set_phase(state: dict, name: str, status: str, **fields: Any) -> None:
    record = dict(state["phases"].get(name, {}))
    record.update(fields)
    record["status"] = status
    record["updated_at_ns"] = time.time_ns()
    state["phases"][name] = record
    _save_state(state)


def _python_executable() -> str:
    executable = pathlib.Path(sys.executable)
    _require(executable.is_file(), f"current Python is not executable: {executable}")
    return str(executable)


def _invoke(command: list[str], label: str) -> None:
    """Run the single frozen command without a shell, inheriting the terminal so long tasks can be watched live."""
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise CommandFailed(
            f"{label} subprocess failed (returncode={completed.returncode}): "
            f"{command!r}"
        )


def _assert_native_build_fresh() -> dict:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    _require(bool(suffix), "current Python lacks EXT_SUFFIX")
    bridge = ROOT / "build" / f"_diablogym{suffix}"
    from eval_contract import engine_binary_path

    engine = engine_binary_path(ROOT)
    bridge_payload = _stable_read(bridge, "native bridge")
    engine_payload = _stable_read(engine, "native engine")
    bridge_mtime = bridge.stat().st_mtime_ns
    engine_mtime = engine.stat().st_mtime_ns
    bridge_inputs = (ROOT / "src" / "diablogym.cpp", ROOT / "CMakeLists.txt")
    engine_inputs = (ROOT / "CMakeLists.txt", *sorted((ROOT / "patches").glob("*.patch")))
    stale_bridge = [
        str(path.relative_to(ROOT))
        for path in bridge_inputs
        if path.is_file() and path.stat().st_mtime_ns > bridge_mtime
    ]
    stale_engine = [
        str(path.relative_to(ROOT))
        for path in engine_inputs
        if path.is_file() and path.stat().st_mtime_ns > engine_mtime
    ]
    _require(
        not stale_bridge and not stale_engine,
        "native source/patches are newer than the built binaries; run ./build.sh first:"
        f" bridge={stale_bridge}, engine={stale_engine}",
    )
    return {
        "bridge_sha256": hashlib.sha256(bridge_payload).hexdigest(),
        "engine_sha256": hashlib.sha256(engine_payload).hexdigest(),
    }


def _checkpoint_data(payload: bytes, label: str) -> dict:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            data = strict_json_loads(archive.read("data"))
    except (KeyError, ValueError, zipfile.BadZipFile, EvalContractError) as exc:
        raise CampaignError(f"{label} checkpoint data unreadable") from exc
    _require(isinstance(data, dict), f"{label} checkpoint data is not an object")
    return data


def _checkpoint_policy_heads(payload: bytes, label: str) -> dict:
    """Recompute the current six-tensor head and the persistent root anchor from the checkpoint on site."""
    import torch
    from sb3_contrib import MaskablePPO
    import train_ppo

    try:
        model = MaskablePPO.load(io.BytesIO(payload), env=None, device="cpu")
    except Exception as exc:
        raise CampaignError(f"{label} cannot safely load and recompute the policy head") from exc
    current = train_ppo._policy_head_snapshot(model.policy)
    _require(
        all(bool(tensor.detach().isfinite().all().item()) for tensor in current.values()),
        f"{label} current policy head contains NaN/Inf",
    )
    root = getattr(model, "bc_aux_root_anchor_sd", None)
    root_sha = None
    if root is not None:
        _require(
            isinstance(root, dict)
            and set(root) == set(train_ppo._POLICY_HEAD_KEYS)
            and all(isinstance(value, torch.Tensor) for value in root.values())
            and all(bool(value.detach().isfinite().all().item()) for value in root.values()),
            f"{label} persistent root anchor fields/tensors are malformed",
        )
        root_sha = train_ppo._policy_head_sha256(root)
    return {
        "current_policy_head_sha256": train_ppo._policy_head_sha256(current),
        "root_policy_head_sha256": root_sha,
    }


def _base_artifact_snapshot() -> dict:
    """Re-verify v28/M29/KING, the native build and the current training implementation."""
    _require(PROTOCOL_VERSION == 4, "the official launcher only accepts protocol v4")
    _require(
        LEG_STEPS == TRAIN_CALLS * ROLLOUT_QUANTUM,
        "249,856 steps/122 rollouts arithmetic drift",
    )
    _require(START_STEPS + LEG_STEPS == TARGET_STEPS, "total training steps do not close")
    native = _assert_native_build_fresh()
    v28_payload = _stable_read(V28_ZIP, "v28 resume checkpoint")
    _require(
        hashlib.sha256(v28_payload).hexdigest() == V28_SHA256,
        "v28 resume checkpoint SHA drift",
    )
    v28_data = _checkpoint_data(v28_payload, "v28")
    _require(
        v28_data.get("num_timesteps") == START_STEPS
        and v28_data.get("distill_beta") == DISTILL_BETA
        and v28_data.get("learning_rate") == 3e-4
        and v28_data.get("gamma") == 1.0
        and v28_data.get("ent_coef") == ENT_COEF
        and v28_data.get("n_steps") == N_STEPS
        and v28_data.get("batch_size") == 256,
        "v28 resume checkpoint step count/Leashed recipe is malformed",
    )
    v28_heads = _checkpoint_policy_heads(v28_payload, "v28")
    _require(
        v28_heads["current_policy_head_sha256"] == V28_POLICY_HEAD_SHA256,
        "v28 six-tensor policy head SHA drift",
    )
    _require_sha(M29_NPZ, M29_SHA256, "M29 manager")
    _require_sha(KING_SD, KING_SHA256, "KING teacher")
    _require_sha(KING_MANIFEST, KING_MANIFEST_SHA256, "KING manifest")

    import train_ppo

    # Also verify the source ZIP the manifest points to, the source SHA and all exported tensors, rather than trusting only the manifest text.
    manifest = train_ppo._validate_export_manifest(KING_SD)
    _require(
        manifest["artifact_sha256"] == KING_SHA256
        and manifest["source_checkpoint_sha256"] == V28_SHA256,
        "KING manifest is not exactly bound to v28",
    )
    implementation = train_ppo._implementation_bundle_sha256()
    return {
        **native,
        "implementation_sha256": implementation,
        "v28_sha256": V28_SHA256,
        "v28_policy_head_sha256": V28_POLICY_HEAD_SHA256,
        "manager_sha256": M29_SHA256,
        "teacher_sha256": KING_SHA256,
        "teacher_manifest_sha256": KING_MANIFEST_SHA256,
    }


def _validate_r5_safe_replay_anchor() -> dict:
    """Bind R6 to the exact safe R5 prefix selected before its TV trip."""
    payload = _stable_read(R5_SAFE_CHECKPOINT, "R5 safe-prefix checkpoint")
    _require(
        hashlib.sha256(payload).hexdigest() == R5_SAFE_CHECKPOINT_SHA256,
        "R5 safe-prefix checkpoint SHA drift",
    )
    heads = _checkpoint_policy_heads(payload, "R5 safe-prefix checkpoint")
    data = _checkpoint_data(payload, "R5 safe-prefix checkpoint")
    _require(
        heads["current_policy_head_sha256"]
        == R6_EXPECTED_POLICY_HEAD_SHA256
        and heads["root_policy_head_sha256"] == V28_POLICY_HEAD_SHA256
        and data.get("num_timesteps") == TARGET_STEPS
        and data.get("_last_completed_ppo_rollout_steps") == TARGET_STEPS
        and data.get("_ppo_optimizer_steps_completed")
        == TRAIN_CALLS * 80,
        "R5 safe-prefix policy head/root anchor/completed-update proof is not closed",
    )
    return {
        "checkpoint_sha256": R5_SAFE_CHECKPOINT_SHA256,
        "policy_head_sha256": R6_EXPECTED_POLICY_HEAD_SHA256,
        "num_timesteps": TARGET_STEPS,
        "ppo_optimizer_steps_completed": TRAIN_CALLS * 80,
    }


def _bc_v1_identity() -> dict:
    import train_ppo

    v1 = train_ppo._validate_bc_report(BC_V1_POLICY, "data_gate")
    v1_demos_sha = train_ppo._assert_bc_v1_demos_frozen(BC_V1_DEMOS)
    _require(
        v1.get("implementation_sha256") == train_ppo._implementation_bundle_sha256(),
        "BC-v1 implementation differs from the current one",
    )
    return {
        "v1_demos_sha256": v1_demos_sha,
        "v1_policy_sha256": v1["policy_sha256"],
        "v1_report_sha256": _stable_sha256(
            BC_V1_DIR / "bc_report.json", "BC-v1 report"
        ),
    }


def _bc_v2_identity() -> dict:
    import train_ppo

    _, _, _, _, v2_demos_sha = train_ppo._load_bc_aux_demos_v2(
        BC_V2_DEMOS,
        expected_manager_sha256=M29_SHA256,
    )
    v2_report = strict_json_loads(
        _stable_read(BC_V2_DIR / "bc_report_v2.json", "BC-v2 report")
    )
    _require(
        v2_report.get("implementation_sha256")
        == train_ppo._implementation_bundle_sha256()
        and v2_report.get("manager_npz_sha256") == M29_SHA256
        and v2_report.get("demos_sha256") == v2_demos_sha,
        "BC-v2 implementation/manager/demos chain is not closed",
    )
    _require(
        v2_report.get("preventive_threshold") == 0.65,
        "this campaign pre-registers only the BC-v2 main case threshold=0.65; "
        "refusing to mix OC items generated after seeing results into the same training lineage",
    )
    return {
        "v2_demos_sha256": v2_demos_sha,
        "v2_policy_sha256": v2_report["policy_sha256"],
        "v2_report_sha256": _stable_sha256(
            BC_V2_DIR / "bc_report_v2.json", "BC-v2 report"
        ),
    }


def _bc_identities() -> dict:
    """Re-verify the current canonical BC pair with the production validators of the training entry point."""
    return {**_bc_v1_identity(), **_bc_v2_identity()}


def _try_identity(validator) -> tuple[dict | None, str | None]:
    try:
        return validator(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _try_bc_identities() -> tuple[dict | None, str | None]:
    return _try_identity(_bc_identities)


def _reject_current_bc_scientific_failure(
    report_path: pathlib.Path,
    *,
    implementation_sha256: str,
    label: str,
) -> None:
    """Once the same implementation + generator has written a terminal receipt, repeated peeking without an identity change is forbidden."""
    if not report_path.is_file():
        return
    try:
        report = strict_json_loads(_stable_read(report_path, f"{label} report"))
    except Exception:
        return
    if (
        isinstance(report, dict)
        and report.get("protocol_version") == PROTOCOL_VERSION
        and report.get("implementation_sha256") == implementation_sha256
        and report.get("generator_sha256")
        == _stable_sha256(TRAIN / "bc_worker.py", "current BC generator")
        and report.get("data_gate") in {"PASS", "FAIL"}
    ):
        raise CampaignError(
            f"{label} already wrote terminal state {report.get('data_gate')} under the current implementation"
            " but the strict artifacts are incomplete; final may already have been consumed, so the launcher must not automatically re-sample/adjust the threshold."
            " Do forensics first and relaunch under a new implementation/campaign identity"
        )


def _prepare_bc_commands() -> tuple[list[str], list[str]]:
    python = _python_executable()
    return (
        [python, str(TRAIN / "bc_worker.py")],
        [
            python,
            str(TRAIN / "bc_worker.py"),
            "--v2",
            "--manager-npz",
            str(M29_NPZ),
        ],
    )


def command_prepare_bc() -> None:
    state = _load_state()
    snapshot = _base_artifact_snapshot()
    v1, v1_reason = _try_identity(_bc_v1_identity)
    v2, v2_reason = _try_identity(_bc_v2_identity)
    if v1 is not None and v2 is not None:
        current = {**v1, **v2}
        _set_phase(
            state,
            "prepare_bc",
            "complete",
            implementation_sha256=snapshot["implementation_sha256"],
            **current,
        )
        print("BC-v1/v2 are already strict PASS items for the current protocol-v4 + M29; no re-sampling needed.")
        return

    v1_command, v2_command = _prepare_bc_commands()
    commands = []
    if v1 is None:
        _reject_current_bc_scientific_failure(
            BC_V1_DIR / "bc_report.json",
            implementation_sha256=snapshot["implementation_sha256"],
            label="BC-v1",
        )
        commands.append(v1_command)
    if v2 is None:
        _reject_current_bc_scientific_failure(
            BC_V2_DIR / "bc_report_v2.json",
            implementation_sha256=snapshot["implementation_sha256"],
            label="BC-v2",
        )
        commands.append(v2_command)
    _set_phase(
        state,
        "prepare_bc",
        "running-v1" if v1 is None else "running-v2",
        initial_validation_errors={"v1": v1_reason, "v2": v2_reason},
        implementation_sha256=snapshot["implementation_sha256"],
        commands=commands,
    )
    try:
        # v1 uses the fixed H only for dry-anchor/telemetry; v2 must use M29 explicitly and never take the default H.
        if v1 is None:
            _invoke(v1_command, "BC-v1 re-sample")
            v1 = _bc_v1_identity()
        if v2 is None:
            _set_phase(state, "prepare_bc", "running-v2")
            _invoke(v2_command, "BC-v2 (M29, threshold=0.65) re-sample")
            v2 = _bc_v2_identity()
        _require(v1 is not None and v2 is not None, "the BC pair did not close together")
        identities = _bc_identities()
        final_snapshot = _base_artifact_snapshot()
        _require(
            final_snapshot["implementation_sha256"]
            == snapshot["implementation_sha256"],
            "implementation drifted during BC re-sampling",
        )
    except Exception as exc:
        _set_phase(
            state,
            "prepare_bc",
            "failed",
            error=f"{type(exc).__name__}: {exc}",
            note=(
                "the old 0.70 OC shares the final pool with 0.65 and is disabled since rev13. If n12 "
                "is insufficient, register an independent fresh pool first; no failure may tune on the same pool."
            ),
        )
        raise
    _set_phase(
        state,
        "prepare_bc",
        "complete",
        implementation_sha256=final_snapshot["implementation_sha256"],
        **identities,
    )
    print("BC-v1 and M29-BC-v2 re-sampled, re-verified and frozen.")


def _official_training_args() -> types.SimpleNamespace:
    """For precomputation by the trainer's single source of truth; the fields mirror the single CLI verbatim."""
    return types.SimpleNamespace(
        worker=True,
        options=False,
        flat_clock=False,
        arch="mlp",
        max_steps=3000,
        num_envs=NUM_ENVS,
        n_steps=N_STEPS,
        gamma=1.0,
        lr=LEARNING_RATE,
        ent_coef=ENT_COEF,
        distill_beta=DISTILL_BETA,
        calib_record_only=False,
        skip_dry=False,
        no_drink_sovereignty=False,
        dry_curriculum_schedule=CURRICULUM,
        bc_aux_lambda=BC_AUX_LAMBDA,
        bc_aux_demos=str(BC_V2_DEMOS),
        bc_aux_graft=True,
        bc_aux_liveness_preflight=True,
        target_kl=TARGET_KL,
        total_steps=LEG_STEPS,
        seed=TRAIN_SEED,
        device="cpu",
        reset_optimizer=True,
    )


def _expected_training_contract(snapshot: dict, bc: dict) -> tuple[dict, str]:
    """Precompute the exact rev12 with the current trainer's contract constructor before the environment launches."""
    import train_ppo

    args = _official_training_args()
    fake_model = types.SimpleNamespace(
        action_space=types.SimpleNamespace(n=15),
        observation_space=types.SimpleNamespace(shape=(298,)),
        teacher_sha256=KING_SHA256,
        device="cpu",
        max_grad_norm=0.5,
    )
    contract = train_ppo._training_contract(
        args,
        fake_model,
        train_ppo._select_batch_size(N_STEPS, NUM_ENVS),
        manager_npz_sha256=M29_SHA256,
        worker_npz_sha256=None,
        demos_sha256=bc["v1_demos_sha256"],
        implementation_sha256=snapshot["implementation_sha256"],
        bc_aux_demos_sha256=bc["v2_demos_sha256"],
    )
    _require(
        contract.get("contract_revision")
        == train_ppo._CONTRACT_REVISION
        == TRAINING_CONTRACT_REVISION
        and contract.get("mode") == "worker"
        and contract.get("manager_npz_sha256") == M29_SHA256
        and contract.get("teacher_sha256") == KING_SHA256
        and contract.get("demos_sha256") == bc["v1_demos_sha256"]
        and isinstance(contract.get("bc_aux"), dict)
        and contract["bc_aux"].get("mode") == BC_AUX_MODE
        # Plan A (design decision, 2026-07-27): era decoupling. This archived launcher checks only its own frozen
        # constants (rev10) and no longer chains to the live train_ppo constants (rev11 since 07-25).
        and contract["bc_aux"].get("objective_revision")
        == BC_AUX_OBJECTIVE_REVISION
        and isinstance(contract["bc_aux"].get("circuit"), dict)
        and contract["bc_aux"]["circuit"].get("schema_version")
        == BC_AUX_CIRCUIT_SCHEMA
        and contract["bc_aux"].get("trainable_adapter_parameters") == 5,
        "precomputed training contract does not match the official-v4-r6 rev10 fixed inputs",
    )
    return contract, train_ppo._canonical_json_sha256(contract)


def _validate_liveness_preflight(
    payload: bytes,
    *,
    snapshot: dict,
    bc: dict,
) -> dict:
    """Validate a producer-native rev10 contextual-mixture liveness receipt."""
    import train_ppo

    try:
        document = strict_json_loads(payload)
    except EvalContractError as exc:
        raise CampaignError("bc_aux liveness preflight cannot be parsed") from exc
    top_keys = {
        "schema_version",
        "protocol_version",
        "objective_revision",
        "inputs",
        "config",
        "status",
        "simulation",
        "installation",
        "evaluation_scope",
        "heldout_rows_consumed",
        "circuit",
        "optimizer",
        "policy",
        "calls",
        "policy_gradient_canary",
        "calibration",
        "metrics",
        "gate",
    }
    installation = (
        document.get("installation") if isinstance(document, dict) else None
    )
    _require(
        isinstance(document, dict)
        and set(document) == top_keys
        and document["schema_version"]
        == train_ppo._BC_AUX_LIVENESS_PREFLIGHT_SCHEMA_VERSION
        and document["protocol_version"] == PROTOCOL_VERSION == 4
        # Plan A (design decision, 2026-07-27): era decoupling, same note as _frozen_training_contract.
        and document["objective_revision"]
        == BC_AUX_OBJECTIVE_REVISION
        and document["status"] == "PASS"
        and document["simulation"]
        == "isolated-exact-mixture-with-policy-gradient-canary"
        and installation in {"first-install", "preserved-continuation"}
        and document["evaluation_scope"] == "bc-v2-nested-validation-only"
        and isinstance(document["heldout_rows_consumed"], int)
        and not isinstance(document["heldout_rows_consumed"], bool)
        and document["heldout_rows_consumed"] == 0,
        "liveness preflight top-level schema/status/isolation domain is not exact",
    )
    first_install = installation == "first-install"

    expected_inputs = {
        "resume_checkpoint_sha256": V28_SHA256,
        "demos_sha256": bc["v2_demos_sha256"],
        "manager_npz_sha256": M29_SHA256,
        "implementation_sha256": snapshot["implementation_sha256"],
    }
    _require(document["inputs"] == expected_inputs, "liveness input lineage is not exact")

    plan = {
        "rollout_quantum": ROLLOUT_QUANTUM,
        "train_calls": TRAIN_CALLS,
        "aux_optimizer_calls": 0,
        "policy_gradient_canary_calls": 1,
        "initial_adapter_calibrations": 1 if first_install else 0,
        "trainable_adapter_parameters": 5,
    }
    _require(
        LEG_STEPS // ROLLOUT_QUANTUM == plan["train_calls"],
        "trainer contextual-mixture call plan has deviated from 122 rollouts",
    )
    circuit = {
        **train_ppo._bc_aux_circuit_spec(),
        "king_support": train_ppo._BC_AUX_CIRCUIT_KING_SUPPORT,
    }
    _require(
        circuit.get("schema_version") == BC_AUX_CIRCUIT_SCHEMA
        and circuit.get("initial_probability") == 0.05
        and circuit.get("probability_min") == 0.001
        and circuit.get("probability_max") == 0.95
        and len(circuit.get("gate_feature_indices", ())) == 4
        and len(circuit.get("gate_parameter_columns", ())) == 4,
        "trainer rev10 contextual circuit constant drift",
    )
    expected_config = {
        "bc_aux_lambda": BC_AUX_LAMBDA,
        "seed": TRAIN_SEED,
        "device": "cpu",
        "learning_rate": LEARNING_RATE,
        "distill_beta": DISTILL_BETA,
        "target_kl": TARGET_KL,
        "reset_optimizer": first_install,
        "n_steps": N_STEPS,
        "num_envs": NUM_ENVS,
        "batch_size": train_ppo._select_batch_size(N_STEPS, NUM_ENVS),
        "total_steps": LEG_STEPS,
        "mechanism": BC_AUX_MODE,
        "circuit": circuit,
        **plan,
    }
    _require(
        document["config"] == expected_config,
        "liveness preflight config is not the exact official-v4 recipe",
    )
    config = document["config"]
    _require(
        isinstance(config["bc_aux_lambda"], (int, float))
        and not isinstance(config["bc_aux_lambda"], bool)
        and isinstance(config["reset_optimizer"], bool)
        and all(
            isinstance(config[key], int)
            and not isinstance(config[key], bool)
            for key in (
                "seed",
                "n_steps",
                "num_envs",
                "batch_size",
                "total_steps",
                "rollout_quantum",
                "train_calls",
                "aux_optimizer_calls",
                "policy_gradient_canary_calls",
                "initial_adapter_calibrations",
                "trainable_adapter_parameters",
            )
        ),
        "liveness config must not let a bool pose as a numeric count",
    )
    expected_calls = {
        "planned_train_calls": TRAIN_CALLS,
        "aux_optimizer_calls": 0,
        "policy_gradient_canary_calls": 1,
        "initial_adapter_calibrations": 1 if first_install else 0,
        "trainable_adapter_parameters": 5,
    }
    calls = document["calls"]
    _require(
        isinstance(calls, dict)
        and calls == expected_calls,
        "liveness exact-mixture planned/initial call counts are not exact",
    )
    _require(
        all(
            isinstance(calls[key], int)
            and not isinstance(calls[key], bool)
            for key in expected_calls
        ),
        "liveness calls must not let a bool pose as a numeric count",
    )

    policy = document["policy"]
    _require(
        isinstance(policy, dict)
        and set(policy)
        == {
            "start_head_sha256",
            "root_head_sha256",
            "grafted_head_sha256",
            "actor_width_before",
            "actor_width_after",
        }
        and policy["root_head_sha256"] == V28_POLICY_HEAD_SHA256
        and _is_sha256(policy["grafted_head_sha256"])
        and policy["actor_width_before"]
        == (
            train_ppo._BC_AUX_CIRCUIT_BASE_WIDTH
            if first_install
            else train_ppo._BC_AUX_CIRCUIT_EXPANDED_WIDTH
        )
        and policy["actor_width_after"]
        == train_ppo._BC_AUX_CIRCUIT_EXPANDED_WIDTH,
        "liveness policy head does not close from the frozen v28 root anchor",
    )
    if first_install:
        _require(
            policy["start_head_sha256"] == V28_POLICY_HEAD_SHA256,
            "liveness first-install starting point is not the frozen v28 root head",
        )
    else:
        _require(
            policy["start_head_sha256"] == policy["grafted_head_sha256"],
            "liveness preserved-continuation must not rewrite the policy head",
        )
    _require(document["circuit"] == circuit,
             "liveness circuit constants/KING support drift")
    canary = document["policy_gradient_canary"]
    canary_keys = {
        "schema_version",
        "scope",
        "pairs",
        "heldout_rows_consumed",
        "objective",
        "optimizer_steps",
        "movement_required",
        "probability_12_before",
        "probability_12_after",
        "probability_12_delta",
        "gate_bias_before",
        "gate_bias_after",
        "gate_bias_delta",
        "gate_bias_gradient",
        "gradient_norm_before_clip",
        "start_policy_head_sha256",
        "stepped_policy_head_sha256",
        "state_restored",
    }
    _require(
        isinstance(canary, dict)
        and set(canary) == canary_keys
        and canary["schema_version"] == "a12-policy-gradient-canary/1"
        and canary["scope"]
        == "bc-v2-nested-validation-positive-only"
        and isinstance(canary["pairs"], int)
        and not isinstance(canary["pairs"], bool)
        and canary["pairs"] > 0
        and canary["heldout_rows_consumed"] == 0
        and canary["objective"]
        == "negative-mean-log-probability-action12"
        and canary["optimizer_steps"] == 1
        and isinstance(canary["movement_required"], bool)
        and canary["state_restored"] is True
        and canary["start_policy_head_sha256"]
        == policy["grafted_head_sha256"]
        and _is_sha256(canary["stepped_policy_head_sha256"])
        and (
            not canary["movement_required"]
            or canary["stepped_policy_head_sha256"]
            != canary["start_policy_head_sha256"]
        ),
        "liveness policy-gradient canary schema/isolation identity is not exact",
    )
    for key in (
            "probability_12_before",
            "probability_12_after",
            "probability_12_delta",
            "gate_bias_before",
            "gate_bias_after",
            "gate_bias_delta",
            "gate_bias_gradient",
            "gradient_norm_before_clip"):
        _require(
            isinstance(canary[key], (int, float))
            and not isinstance(canary[key], bool)
            and math.isfinite(float(canary[key])),
            f"liveness policy-gradient canary {key} is not a finite number",
        )
    _require(
        0.0 < canary["probability_12_before"] < 1.0
        and canary["probability_12_after"]
        >= canary["probability_12_before"]
        and math.isclose(
            canary["probability_12_delta"],
            canary["probability_12_after"]
            - canary["probability_12_before"],
            rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(
            canary["gate_bias_delta"],
            canary["gate_bias_after"] - canary["gate_bias_before"],
            rel_tol=0.0, abs_tol=1e-15)
        and canary["gate_bias_gradient"] < 0.0
        and canary["gradient_norm_before_clip"] > 0.0
        and (
            not canary["movement_required"]
            or (
                canary["probability_12_delta"] > 0.0
                and canary["gate_bias_delta"] > 0.0
            )
        ),
        "liveness policy-gradient canary did not prove the real optimizer can raise p(a12)",
    )
    optimizer = document["optimizer"]
    state_entries = (
        optimizer.get("state_entries_at_start")
        if isinstance(optimizer, dict) else None
    )
    _require(
        isinstance(optimizer, dict)
        and set(optimizer)
        == {
            "class",
            "state_entries_at_start",
            "learning_rates_at_start",
            "reset_after_topology_change",
        }
        and isinstance(optimizer["class"], str)
        and bool(optimizer["class"])
        and isinstance(state_entries, int)
        and not isinstance(state_entries, bool)
        and state_entries >= 0
        and optimizer["learning_rates_at_start"] == [LEARNING_RATE]
        and optimizer["reset_after_topology_change"] is first_install
        and (state_entries == 0 if first_install else True),
        "liveness optimizer did not prove first-install reset/continuation preserved",
    )
    calibration = document["calibration"]
    target_probability = float(circuit["initial_probability"])
    if first_install:
        calibration_keys = {
            "fit_pairs",
            "validation_pairs",
            "fit_positive_a12",
            "validation_positive_a12",
            "initializer",
            "gate_feature_indices",
            "gate_parameter_columns",
            "target_probability_12",
            "fit_positive_probability_min_12",
            "fit_positive_probability_12",
            "fit_positive_probability_max_12",
            "validation_positive_probability_min_12",
            "validation_positive_probability_12",
            "validation_positive_probability_max_12",
            "initial_argmax_lower_bound",
            "initial_gate_bias",
            "gate_coefficients",
            "probability_min",
            "probability_max",
            "fit_metrics",
            "validation_metrics",
            "validation_gate",
            "candidate_policy_head_sha256",
        }
        positive_count_keys = (
            "fit_pairs",
            "validation_pairs",
            "fit_positive_a12",
            "validation_positive_a12",
        )
        probability_keys = (
            "fit_positive_probability_min_12",
            "fit_positive_probability_12",
            "fit_positive_probability_max_12",
            "validation_positive_probability_min_12",
            "validation_positive_probability_12",
            "validation_positive_probability_max_12",
        )
        expected_margin = (
            (1.0 - target_probability) / 14.0 - target_probability)
        _require(
            isinstance(calibration, dict)
            and set(calibration) == calibration_keys
            and all(
                isinstance(calibration[key], int)
                and not isinstance(calibration[key], bool)
                and calibration[key] > 0
                for key in positive_count_keys
            )
            and calibration["fit_positive_a12"]
            <= calibration["fit_pairs"]
            and calibration["validation_positive_a12"]
            <= calibration["validation_pairs"]
            and calibration["initializer"]
            == "exact-contextual-legal-support-mixture"
            and calibration["gate_feature_indices"]
            == circuit["gate_feature_indices"]
            and calibration["gate_parameter_columns"]
            == circuit["gate_parameter_columns"]
            and isinstance(calibration["gate_coefficients"], list)
            and len(calibration["gate_coefficients"]) == 4
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value) == 0.0
                for value in calibration["gate_coefficients"]
            )
            and all(
                isinstance(calibration[key], (int, float))
                and not isinstance(calibration[key], bool)
                and math.isfinite(float(calibration[key]))
                for key in (
                    *probability_keys,
                    "target_probability_12",
                    "initial_argmax_lower_bound",
                    "initial_gate_bias",
                    "probability_min",
                    "probability_max",
                )
            )
            and math.isclose(
                float(calibration["target_probability_12"]),
                target_probability, rel_tol=0.0, abs_tol=1e-12)
            and all(
                math.isclose(
                    float(calibration[key]), target_probability,
                    rel_tol=0.0, abs_tol=5e-7)
                for key in probability_keys
            )
            and math.isclose(
                float(calibration["initial_argmax_lower_bound"]),
                expected_margin, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(
                float(calibration["initial_gate_bias"]),
                float(circuit["initial_gate_bias"]),
                rel_tol=0.0, abs_tol=2e-7)
            and float(calibration["probability_min"])
            == float(circuit["probability_min"])
            and float(calibration["probability_max"])
            == float(circuit["probability_max"])
            and calibration["candidate_policy_head_sha256"]
            == policy["grafted_head_sha256"],
            "liveness contextual-mixture initializer evidence is not closed",
        )
    else:
        preserved_keys = {
            "initializer",
            "gate_coefficients",
            "gate_bias",
            "fit_pairs_excluded_from_retuning",
            "validation_pairs",
            "validation_metrics",
            "validation_gate",
            "candidate_policy_head_sha256",
        }
        coefficients = (
            calibration.get("gate_coefficients")
            if isinstance(calibration, dict) else None
        )
        gate_bias = (
            calibration.get("gate_bias")
            if isinstance(calibration, dict) else None
        )
        _require(
            isinstance(calibration, dict)
            and set(calibration) == preserved_keys
            and calibration["initializer"] == "preserved-continuation"
            and isinstance(coefficients, list)
            and len(coefficients) == 4
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and abs(float(value))
                <= float(circuit["gate_parameter_abs_max"])
                for value in coefficients
            )
            and isinstance(gate_bias, (int, float))
            and not isinstance(gate_bias, bool)
            and math.isfinite(float(gate_bias))
            and abs(float(gate_bias))
            <= float(circuit["gate_parameter_abs_max"])
            and all(
                isinstance(calibration[key], int)
                and not isinstance(calibration[key], bool)
                and calibration[key] > 0
                for key in (
                    "fit_pairs_excluded_from_retuning",
                    "validation_pairs",
                )
            )
            and calibration["candidate_policy_head_sha256"]
            == policy["grafted_head_sha256"],
            "liveness preserved-continuation gate/count evidence is not closed",
        )
    _require(
        isinstance(document["metrics"], dict)
        and set(document["metrics"]) == set(train_ppo._BC_AUX_BEHAVIOR_METRIC_KEYS),
        "liveness metrics schema is not exact",
    )
    recomputed_gate = train_ppo.bc_aux_behavior_gate(
        document["metrics"], require_root_anchor=True,
        require_teacher_recall=False,
    )
    fit_gate = train_ppo.bc_aux_behavior_gate(
        calibration["fit_metrics"], require_root_anchor=True,
        require_teacher_recall=False,
    ) if first_install else None
    _require(
        document["gate"] == recomputed_gate
        and calibration["validation_metrics"] == document["metrics"]
        and calibration["validation_gate"] == document["gate"]
        and (
            not first_install
            or (
                isinstance(calibration["fit_metrics"], dict)
                and set(calibration["fit_metrics"])
                == set(train_ppo._BC_AUX_BEHAVIOR_METRIC_KEYS)
                and fit_gate.get("verdict") == "PASS"
            )
        )
        and recomputed_gate.get("verdict") == "PASS",
        "liveness no-recall lower-bound safety gate was not recomputed exactly as PASS from the on-site metrics",
    )
    return document


def _validate_candidate_behavior_receipt(
    receipt: dict,
    *,
    expected_model_sha256: str,
    expected_policy_head_sha256: str,
    expected_demos_sha256: str,
    expected_provenance: dict,
) -> dict:
    """Recompute the strict rev10 publication safety gate.

    The evaluator performs its own independent capture.  The launcher still
    validates this receipt locally so a forged PASS string, bool-as-count, or
    self-selected exploration minimum cannot authorize the expensive paired
    campaign.  Deterministic action-12 deployment is deliberately not required:
    PPO decides from outcome gradients whether the explored action is useful.
    """
    import train_ppo

    top_keys = {
        "schema_version",
        "step",
        "demos_sha256",
        "objective_revision",
        "evaluation_scope",
        "mask_mode",
        "anchor",
        "candidate_policy_head_sha256",
        "provenance",
        "metrics",
        "gate",
        "exploration_evidence",
        "publication",
        "model_sha256",
        "save_error",
    }
    _require(
        isinstance(receipt, dict)
        and set(receipt) == top_keys
        and receipt["schema_version"]
        == train_ppo._BC_AUX_BEHAVIOR_RECEIPT_SCHEMA_VERSION
        # Plan A (design decision, 2026-07-27): era decoupling, same note as _frozen_training_contract.
        and receipt["objective_revision"]
        == BC_AUX_OBJECTIVE_REVISION
        and isinstance(receipt["step"], int)
        and not isinstance(receipt["step"], bool)
        and receipt["step"] == TARGET_STEPS
        and receipt["demos_sha256"] == expected_demos_sha256
        and receipt["evaluation_scope"]
        == "original-bc-v2-heldout-episodes"
        and receipt["mask_mode"] == "bc-v2-recorded"
        and receipt["provenance"] == expected_provenance
        and receipt["publication"] == "PUBLISHED"
        and receipt["model_sha256"] == expected_model_sha256
        and receipt["candidate_policy_head_sha256"]
        == expected_policy_head_sha256
        and receipt["save_error"] is None,
        "candidate behavior receipt schema/identity/publication status is not exact",
    )
    anchor = receipt["anchor"]
    _require(
        isinstance(anchor, dict)
        and set(anchor) == {"identity", "policy_head_sha256"}
        and anchor["identity"] == "bc-aux-root-policy"
        and anchor["policy_head_sha256"] == V28_POLICY_HEAD_SHA256,
        "candidate behavior receipt root policy anchor is not exact",
    )
    metrics = receipt["metrics"]
    _require(
        isinstance(metrics, dict)
        and set(metrics) == set(train_ppo._BC_AUX_BEHAVIOR_METRIC_KEYS)
        and metrics.get("scope") == "heldout",
        "candidate behavior metrics schema/scope is not exact",
    )
    recomputed_gate = train_ppo.bc_aux_behavior_gate(
        metrics,
        require_root_anchor=True,
        require_teacher_recall=False,
        require_deployable_a12=False,
    )
    _require(
        receipt["gate"] == recomputed_gate
        and recomputed_gate.get("verdict") == "PASS"
        and recomputed_gate.get("thresholds", {}).get(
            "deployable_a12_required"
        ) is False,
        "candidate behavior receipt did not pass the PPO autonomous action-12 publication safety gate",
    )

    evidence = receipt["exploration_evidence"]
    evidence_keys = {
        "eligible_states",
        "expected_a12_mass",
        "requested_a12",
        "sampled_a12",
        "rejected_a12",
        "unexpected_sampled_a12",
        "minimum_expected_a12_mass",
        "minimum_actual_a12_samples",
        "information_status",
        "reasons",
    }

    def plain_nonnegative_int(value: Any) -> bool:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        )

    expected_mass = (
        evidence.get("expected_a12_mass")
        if isinstance(evidence, dict) else None
    )
    minimum_mass = (
        evidence.get("minimum_expected_a12_mass")
        if isinstance(evidence, dict) else None
    )
    _require(
        isinstance(evidence, dict)
        and set(evidence) == evidence_keys
        and plain_nonnegative_int(evidence["eligible_states"])
        and evidence["eligible_states"] > 0
        and isinstance(expected_mass, (int, float))
        and not isinstance(expected_mass, bool)
        and math.isfinite(float(expected_mass))
        and float(expected_mass)
        >= train_ppo._BC_AUX_MIN_EXPECTED_A12_SAMPLES
        and float(expected_mass)
        <= float(evidence["eligible_states"]) + 1e-9
        and plain_nonnegative_int(evidence["requested_a12"])
        and evidence["requested_a12"] <= evidence["eligible_states"]
        and plain_nonnegative_int(evidence["sampled_a12"])
        and evidence["sampled_a12"]
        >= train_ppo._BC_AUX_MIN_ACTUAL_A12_SAMPLES
        and evidence["sampled_a12"] <= evidence["requested_a12"]
        and plain_nonnegative_int(evidence["rejected_a12"])
        and evidence["rejected_a12"] <= evidence["requested_a12"]
        and evidence["requested_a12"]
        == evidence["sampled_a12"] + evidence["rejected_a12"]
        and plain_nonnegative_int(evidence["unexpected_sampled_a12"])
        and evidence["unexpected_sampled_a12"] == 0
        and isinstance(minimum_mass, (int, float))
        and not isinstance(minimum_mass, bool)
        and math.isfinite(float(minimum_mass))
        and float(minimum_mass)
        == float(train_ppo._BC_AUX_MIN_EXPECTED_A12_SAMPLES)
        and isinstance(evidence["minimum_actual_a12_samples"], int)
        and not isinstance(evidence["minimum_actual_a12_samples"], bool)
        and evidence["minimum_actual_a12_samples"]
        == train_ppo._BC_AUX_MIN_ACTUAL_A12_SAMPLES
        and evidence["information_status"] == "INFORMATIVE"
        and evidence["reasons"] == [],
        "candidate exploration evidence does not meet the frozen 20/10 strict lower bounds",
    )
    return recomputed_gate


def _preregister_dynamic_provenance(snapshot: dict, bc: dict) -> dict:
    """Freeze the two dynamic SHAs of the contract and the isolated liveness before the real rollout."""
    import train_ppo

    _, contract_sha = _expected_training_contract(snapshot, bc)
    if not PREREG_PREFLIGHT_PATH.exists():
        args = _official_training_args()
        v28_payload = _stable_read(V28_ZIP, "v28 preflight resume")
        x, y, episode_id, masks, demos_sha = train_ppo._load_bc_aux_demos_v2(
            BC_V2_DEMOS,
            expected_manager_sha256=M29_SHA256,
        )
        _require(demos_sha == bc["v2_demos_sha256"], "preflight demos SHA drift")
        bank = train_ppo._build_bc_aux_training_bank(x, y, episode_id, masks)
        PREREG_PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
        train_ppo._run_bc_aux_liveness_preflight(
            run_dir=PREREG_PREFLIGHT_DIR,
            args=args,
            resume_checkpoint_bytes=v28_payload,
            resume_checkpoint_sha256=V28_SHA256,
            bank=bank,
            x=x,
            y=y,
            episode_id=episode_id,
            masks=masks,
            demos_sha256=demos_sha,
            manager_npz_sha256=M29_SHA256,
            implementation_sha256=snapshot["implementation_sha256"],
            batch_size=train_ppo._select_batch_size(N_STEPS, NUM_ENVS),
        )
    payload = _stable_read(PREREG_PREFLIGHT_PATH, "pre-registered liveness preflight")
    _validate_liveness_preflight(payload, snapshot=snapshot, bc=bc)
    return {
        "bc_aux_liveness_preflight_sha256": hashlib.sha256(payload).hexdigest(),
        "training_contract_sha256": contract_sha,
        "preflight_path": str(PREREG_PREFLIGHT_PATH),
    }


def _known_expected_provenance(
    snapshot: dict, bc: dict, dynamic: dict
) -> dict:
    expected = {
        "protocol_version": 4,
        "implementation_sha256": snapshot["implementation_sha256"],
        "manager_npz_sha256": M29_SHA256,
        "resume_checkpoint_sha256": V28_SHA256,
        "teacher_sha256": KING_SHA256,
        "bc_aux_demos_sha256": bc["v2_demos_sha256"],
        "bc_aux_liveness_preflight_sha256": dynamic[
            "bc_aux_liveness_preflight_sha256"
        ],
        "training_contract_sha256": dynamic["training_contract_sha256"],
        "start_steps": START_STEPS,
        "target_global_steps": TARGET_STEPS,
        "seed": TRAIN_SEED,
        "optimizer_reset": True,
        "target_kl": TARGET_KL,
        "distill_beta": DISTILL_BETA,
        "bc_aux_lambda": BC_AUX_LAMBDA,
        "bc_aux_mode": BC_AUX_MODE,
        "calib_record_only": False,
    }
    _require(
        set(expected) == EXPECTED_PROVENANCE_KEYS,
        "launcher expectations field set drift",
    )
    return expected


def _expectations_document(snapshot: dict, bc: dict, dynamic: dict) -> dict:
    return {
        "schema_version": EXPECTATIONS_SCHEMA,
        "expected_provenance": _known_expected_provenance(snapshot, bc, dynamic),
    }


def _read_expectations() -> tuple[dict, str]:
    payload = _stable_read(EXPECTATIONS_PATH, "publication expectations")
    try:
        document = strict_json_loads(payload)
    except EvalContractError as exc:
        raise CampaignError("publication expectations cannot be parsed") from exc
    _require(
        isinstance(document, dict)
        and set(document) == {"schema_version", "expected_provenance"}
        and document["schema_version"] == EXPECTATIONS_SCHEMA
        and isinstance(document["expected_provenance"], dict)
        and set(document["expected_provenance"]) == EXPECTED_PROVENANCE_KEYS,
        "publication expectations are not the exact official-v4 field set",
    )
    return document, hashlib.sha256(payload).hexdigest()


def _freeze_expectations(state: dict, snapshot: dict, bc: dict) -> tuple[dict, str]:
    dynamic = _preregister_dynamic_provenance(snapshot, bc)
    expected = _expectations_document(snapshot, bc, dynamic)
    if EXPECTATIONS_PATH.exists():
        existing, digest = _read_expectations()
        _require(
            existing == expected,
            "existing publication expectations differ from the current frozen inputs; overwriting the historical launch order is forbidden",
        )
        return existing, digest
    _atomic_write_json(EXPECTATIONS_PATH, expected)
    document, digest = _read_expectations()
    _require(document == expected, "publication expectations read-back after write failed")
    _set_phase(
        state,
        "expectations",
        "frozen",
        path=str(EXPECTATIONS_PATH),
        sha256=digest,
        expected_provenance=expected["expected_provenance"],
        preregistered_liveness_path=dynamic["preflight_path"],
    )
    return document, digest


def _training_command() -> list[str]:
    return [
        _python_executable(),
        str(TRAIN / "train_ppo.py"),
        "--worker",
        "--algo",
        "mppo",
        "--gamma",
        "1.0",
        "--max-steps",
        "3000",
        "--device",
        "cpu",
        "--n-steps",
        str(N_STEPS),
        "--num-envs",
        str(NUM_ENVS),
        "--lr",
        str(LEARNING_RATE),
        "--ent-coef",
        str(ENT_COEF),
        "--seed",
        str(TRAIN_SEED),
        "--total-steps",
        str(LEG_STEPS),
        "--run-name",
        CANDIDATE_RUN_NAME,
        "--distill-beta",
        str(DISTILL_BETA),
        "--teacher-override",
        str(KING_SD),
        "--manager-npz",
        str(M29_NPZ),
        "--resume-from",
        str(V28_ZIP),
        "--allow-legacy-resume",
        "--reset-optimizer",
        "--target-kl",
        str(TARGET_KL),
        "--dry-curriculum-schedule",
        CURRICULUM,
        "--bc-aux-lambda",
        str(BC_AUX_LAMBDA),
        "--bc-aux-demos",
        str(BC_V2_DEMOS),
        "--bc-aux-graft",
        "--bc-aux-liveness-preflight",
    ]


def _validate_candidate(*, state: dict | None = None) -> dict:
    """Re-verify the strict chain and fill the head/liveness gaps the evaluator does not cover yet."""
    from eval_assembled import (
        capture_published_worker,
        capture_publication_expectations,
        verify_publication_expectations,
    )

    snapshot = _base_artifact_snapshot()
    bc = _bc_identities()
    document, expectations_sha = _read_expectations()
    expected = document["expected_provenance"]
    _, expected_contract_sha = _expected_training_contract(snapshot, bc)
    prereg_preflight_payload = _stable_read(
        PREREG_PREFLIGHT_PATH, "pre-registered liveness preflight"
    )
    _validate_liveness_preflight(
        prereg_preflight_payload,
        snapshot=snapshot,
        bc=bc,
    )
    prereg_preflight_sha = hashlib.sha256(prereg_preflight_payload).hexdigest()
    _require(
        expected["implementation_sha256"] == snapshot["implementation_sha256"]
        and expected["manager_npz_sha256"] == M29_SHA256
        and expected["resume_checkpoint_sha256"] == V28_SHA256
        and expected["teacher_sha256"] == KING_SHA256
        and expected["bc_aux_demos_sha256"] == bc["v2_demos_sha256"]
        and expected["bc_aux_liveness_preflight_sha256"] == prereg_preflight_sha
        and expected["training_contract_sha256"] == expected_contract_sha
        and set(expected) == EXPECTED_PROVENANCE_KEYS,
        "publication expectations differ from the current fixed artifacts/implementation",
    )
    checkpoint_payload = _stable_read(CANDIDATE_ZIP, "official candidate checkpoint")
    receipt_before = _stable_read(
        CANDIDATE_RECEIPT, "candidate publication receipt"
    )
    preflight_before = _stable_read(
        CANDIDATE_PREFLIGHT, "candidate liveness preflight"
    )
    _require(
        hashlib.sha256(preflight_before).hexdigest() == prereg_preflight_sha
        and preflight_before == prereg_preflight_payload,
        "the official trainer did not reproduce the pre-launch pre-registered liveness preflight verbatim",
    )
    _validate_liveness_preflight(preflight_before, snapshot=snapshot, bc=bc)
    receipt_payload = capture_published_worker(
        CANDIDATE_ZIP,
        checkpoint_payload,
        expected_manager_sha256=M29_SHA256,
        expected_implementation_sha256=snapshot["implementation_sha256"],
        expected_provenance=expected,
    )
    _require(
        receipt_payload == receipt_before,
        "the receipt captured by the strict evaluator differs from the launcher's stable read",
    )
    receipt_sha = hashlib.sha256(receipt_payload).hexdigest()
    try:
        receipt = strict_json_loads(receipt_payload)
    except EvalContractError as exc:
        raise CampaignError("candidate publication receipt cannot be parsed") from exc
    _require(
        isinstance(receipt, dict)
        and receipt.get("provenance") == expected,
        "candidate receipt provenance is not exactly the 17-key pre-registered lineage",
    )
    heads = _checkpoint_policy_heads(checkpoint_payload, "official candidate")
    _require(
        heads["current_policy_head_sha256"]
        == receipt["candidate_policy_head_sha256"]
        == R6_EXPECTED_POLICY_HEAD_SHA256
        and heads["root_policy_head_sha256"] == V28_POLICY_HEAD_SHA256
        and receipt["anchor"]["policy_head_sha256"] == V28_POLICY_HEAD_SHA256,
        "candidate/receipt/R6 safe prefix/root-anchor six-tensor policy head SHAs do not close",
    )
    candidate_sha = hashlib.sha256(checkpoint_payload).hexdigest()
    _validate_candidate_behavior_receipt(
        receipt,
        expected_model_sha256=candidate_sha,
        expected_policy_head_sha256=heads["current_policy_head_sha256"],
        expected_demos_sha256=bc["v2_demos_sha256"],
        expected_provenance=expected,
    )
    checkpoint_data = _checkpoint_data(checkpoint_payload, "official candidate")
    _require(
        checkpoint_data.get("num_timesteps") == TARGET_STEPS
        and checkpoint_data.get("_last_completed_ppo_rollout_steps")
        == TARGET_STEPS
        and isinstance(
            checkpoint_data.get("_ppo_optimizer_steps_completed"), int)
        and not isinstance(
            checkpoint_data.get("_ppo_optimizer_steps_completed"), bool)
        and checkpoint_data["_ppo_optimizer_steps_completed"] > 0
        and checkpoint_data.get("diablogym_contract") is not None
        and expected_contract_sha
        == _canonical_sha256(checkpoint_data["diablogym_contract"]),
        "candidate checkpoint step count/optimizer update proof/training contract do not close",
    )
    _, expectations_identity = capture_publication_expectations(EXPECTATIONS_PATH)
    verify_publication_expectations(expectations_identity)
    _require(
        _stable_read(CANDIDATE_ZIP, "candidate post-capture checkpoint")
        == checkpoint_payload
        and _stable_read(CANDIDATE_RECEIPT, "candidate post-capture receipt")
        == receipt_payload
        and _stable_read(CANDIDATE_PREFLIGHT, "candidate post-capture preflight")
        == preflight_before,
        "the candidate triple changed during the strict capture",
    )
    info = {
        "candidate_sha256": candidate_sha,
        "receipt_sha256": receipt_sha,
        "preflight_sha256": prereg_preflight_sha,
        "expectations_sha256": expectations_sha,
        "training_contract_sha256": expected_contract_sha,
        "candidate_policy_head_sha256": heads["current_policy_head_sha256"],
        "root_policy_head_sha256": heads["root_policy_head_sha256"],
        "implementation_sha256": snapshot["implementation_sha256"],
        "bc_v2_demos_sha256": expected["bc_aux_demos_sha256"],
        "num_timesteps": TARGET_STEPS,
    }
    if state is not None:
        frozen = state["phases"].get("train", {})
        if frozen.get("status") == "frozen":
            _require(
                all(frozen.get(key) == value for key, value in info.items()),
                "candidate differs from the frozen identity in the campaign state",
            )
    return info


def command_train() -> None:
    state = _load_state()
    snapshot = _base_artifact_snapshot()
    bc = _bc_identities()
    _validate_r5_safe_replay_anchor()
    prepare = state["phases"].get("prepare_bc")
    _require(
        isinstance(prepare, dict)
        and prepare.get("status") == "complete"
        and prepare.get("implementation_sha256") == snapshot["implementation_sha256"]
        and all(prepare.get(key) == value for key, value in bc.items()),
        "before train, this launcher must complete prepare-bc for the current implementation",
    )
    _freeze_expectations(state, snapshot, bc)

    if CANDIDATE_ZIP.is_file() and CANDIDATE_RECEIPT.is_file():
        try:
            info = _validate_candidate(state=None)
        except Exception:
            # A final receipt that does not make an official candidate means heldout was already opened; never retrain.
            _set_phase(
                state,
                "train",
                "locked-failed",
                retry_forbidden=True,
                error="candidate/receipt exist but the strict publication chain does not pass",
            )
            raise
        _set_phase(state, "train", "frozen", retry_forbidden=True, **info)
        print("official candidate already frozen and the whole chain re-verified; no repeated training.")
        return

    prior = state["phases"].get("train", {})
    attempts = int(prior.get("attempts", 0))
    if _run_lock_is_held():
        raise CampaignError(
            "official-v4 trainer still holds .run.lock; only reporting the current state, repeated launch forbidden"
        )
    _require(
        not prior.get("retry_forbidden", False),
        "official training already consumed final heldout or left an unproven model; relaunch forbidden",
    )
    _require(
        attempts == 0,
        "official-v4 official training has already been launched. The launcher does not allow forking a retrain from v28"
        " after watching telemetry, nor passing an intermediate checkpoint off as the same pre-registered lineage;"
        " without a complete PUBLISHED candidate this campaign must be quarantined for audit",
    )
    _require(
        not CANDIDATE_ZIP.exists(),
        "found a model_final.zip without a complete PUBLISHED receipt; overwrite/retrain forbidden",
    )
    if CANDIDATE_RECEIPT.exists():
        _set_phase(
            state,
            "train",
            "locked-failed",
            retry_forbidden=True,
            receipt_sha256=_stable_sha256(
                CANDIDATE_RECEIPT, "candidate failed publication receipt"
            ),
        )
        raise CampaignError(
            "a final behavior receipt already exists (heldout consumed) but there is no strict official candidate; retrain forbidden"
        )
    if CANDIDATE_DIR.exists():
        residue = sorted(
            str(path.relative_to(CANDIDATE_DIR))
            for path in CANDIDATE_DIR.iterdir()
        )
        _require(
            not residue,
            "the candidate run directory has unregistered leftovers; the trainer must not archive them and relaunch: "
            f"{residue}",
        )

    command = _training_command()
    _set_phase(
        state,
        "train",
        "running",
        attempts=1,
        command=command,
        implementation_sha256=snapshot["implementation_sha256"],
        v2_demos_sha256=bc["v2_demos_sha256"],
        expectations_sha256=_stable_sha256(
            EXPECTATIONS_PATH, "publication expectations"
        ),
        retry_forbidden=True,
    )
    try:
        # Keep this launcher's BC generator from replacing canonical inputs during official training.
        with contextlib.ExitStack() as stack:
            stack.enter_context(exclusive_lock(BC_V1_DIR / ".bc.lock", "BC-v1"))
            stack.enter_context(exclusive_lock(BC_V2_DIR / ".bc.lock", "BC-v2"))
            locked_bc = _bc_identities()
            _require(locked_bc == bc, "canonical BC drifted before the BC lock was taken")
            _invoke(command, "official-v4 full training")
        info = _validate_candidate(state=None)
    except Exception as exc:
        _set_phase(
            state,
            "train",
            "locked-failed",
            attempts=1,
            retry_forbidden=True,
            error=f"{type(exc).__name__}: {exc}",
            quarantine_reason=(
                "cannot reliably distinguish environment launch, G-CAL, an uncertain target-gate window and a pure operational failure; "
                "automatic retraining or forking from an older checkpoint is forbidden"
            ),
        )
        raise
    _set_phase(
        state,
        "train",
        "frozen",
        attempts=1,
        retry_forbidden=True,
        **info,
    )
    print("official-v4 R6 candidate frozen at the complete update boundary of 3,747,840 steps.")


def _eval_archive_path(pool: str, arm: str) -> pathlib.Path:
    return EVAL_DIR / f"{EVAL_TAGS[(pool, arm)]}.json"


def _seed_arg(pool: str) -> str:
    seeds = EVAL_POOLS[pool]
    return f"{seeds[0]}-{seeds[-1]}"


def _eval_command(pool: str, arm: str) -> list[str]:
    _require(pool in EVAL_POOLS, f"unknown eval pool: {pool}")
    _require(arm in {"baseline", "candidate"}, f"unknown eval arm: {arm}")
    worker = V28_ZIP if arm == "baseline" else CANDIDATE_ZIP
    command = [
        _python_executable(),
        str(TRAIN / "eval_assembled.py"),
        "--worker",
        str(worker),
        "--manager-npz",
        str(M29_NPZ),
        "--seeds",
        _seed_arg(pool),
        "--tag",
        EVAL_TAGS[(pool, arm)],
    ]
    if arm == "candidate":
        command += [
            "--require-published-worker",
            "--publication-expectations",
            str(EXPECTATIONS_PATH),
        ]
    return command


def _eval_input_identity(candidate: dict) -> dict:
    """Freeze the common scientific inputs of both arms; re-hash them item by item before and after every long evaluation."""
    snapshot = _base_artifact_snapshot()
    bc = _bc_identities()
    expectations, expectations_sha = _read_expectations()
    runtime = freeze_eval_identity(
        ROOT,
        str(V28_ZIP),
        str(M29_NPZ),
    )["runtime"]
    identity = {
        "campaign_recipe_sha256": CAMPAIGN_RECIPE_SHA256,
        "launcher": _launcher_identity(),
        "snapshot": snapshot,
        "bc": bc,
        "candidate": dict(candidate),
        "candidate_checkpoint_sha256": _stable_sha256(
            CANDIDATE_ZIP, "eval candidate checkpoint"
        ),
        "candidate_receipt_sha256": _stable_sha256(
            CANDIDATE_RECEIPT, "eval candidate receipt"
        ),
        "candidate_preflight_sha256": _stable_sha256(
            CANDIDATE_PREFLIGHT, "eval candidate preflight"
        ),
        "preregistered_preflight_sha256": _stable_sha256(
            PREREG_PREFLIGHT_PATH, "eval preregistered preflight"
        ),
        "expectations_sha256": expectations_sha,
        "expected_provenance": expectations["expected_provenance"],
        "runtime": runtime,
    }
    _require(
        identity["candidate_checkpoint_sha256"] == candidate["candidate_sha256"]
        and identity["candidate_receipt_sha256"] == candidate["receipt_sha256"]
        and identity["candidate_preflight_sha256"] == candidate["preflight_sha256"]
        and identity["preregistered_preflight_sha256"]
        == candidate["preflight_sha256"]
        and expectations_sha == candidate["expectations_sha256"],
        "eval input identity does not close with the frozen candidate",
    )
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def _validate_official_archive(
    pool: str, arm: str, candidate: dict
) -> tuple[dict, dict]:
    worker = V28_ZIP if arm == "baseline" else CANDIDATE_ZIP
    snapshot = freeze_eval_identity(ROOT, str(worker), str(M29_NPZ))
    expected = expected_eval_identity(
        snapshot,
        tag=EVAL_TAGS[(pool, arm)],
        seeds=EVAL_POOLS[pool],
    )
    path = _eval_archive_path(pool, arm)
    try:
        payload = _stable_read(path, f"{pool}/{arm} eval archive")
        document = strict_json_loads(payload)
        document = validate_eval_archive(document, **expected)
    except (OSError, EvalContractError, ValueError) as exc:
        raise CampaignError(
            f"{pool}/{arm} evaluation archive failed re-verification against the current exact identity: {path}: {exc}"
        ) from exc
    archive_sha = hashlib.sha256(payload).hexdigest()
    _require(
        document["meta"]["manager"]["sha256"] == M29_SHA256,
        f"{pool}/{arm} archive did not use M29",
    )
    if arm == "baseline":
        _require(
            document["meta"]["worker"]["sha256"] == V28_SHA256
            and document["meta"]["worker"]["gate_report_sha256"] is None,
            f"{pool} baseline is not the bare v28",
        )
    else:
        _require(
            document["meta"]["worker"]["sha256"] == candidate["candidate_sha256"]
            and document["meta"]["worker"]["gate_report_sha256"]
            == candidate["receipt_sha256"],
            f"{pool} candidate archive is not bound to the frozen model/publication receipt",
        )
    return document, {
        "path": str(path),
        "sha256": archive_sha,
        "worker_sha256": document["meta"]["worker"]["sha256"],
        "worker_receipt_sha256": document["meta"]["worker"][
            "gate_report_sha256"
        ],
        "manager_sha256": document["meta"]["manager"]["sha256"],
        "protocol_bundle_sha256": document["meta"]["runtime"]["python_protocol"][
            "sha256"
        ],
    }


def _pair_documents(pool: str, baseline: dict, candidate: dict) -> None:
    _require(
        baseline["meta"]["protocol"]["seeds"]
        == candidate["meta"]["protocol"]["seeds"]
        == list(EVAL_POOLS[pool]),
        f"{pool} baseline/candidate seed columns are not exactly equal",
    )
    _require(
        baseline["meta"]["manager"] == candidate["meta"]["manager"],
        f"{pool} baseline/candidate manager identity differs",
    )
    _require(
        baseline["meta"]["runtime"] == candidate["meta"]["runtime"],
        f"{pool} baseline/candidate runtime/content/protocol differ",
    )


def _try_archive(
    pool: str, arm: str, candidate: dict
) -> tuple[dict, dict] | None:
    if not _eval_archive_path(pool, arm).is_file():
        return None
    try:
        return _validate_official_archive(pool, arm, candidate)
    except Exception:
        return None


def _run_eval_arm(
    state: dict, pool: str, arm: str, candidate: dict, *, allow_retry: bool
) -> tuple[dict, dict]:
    phase_name = f"eval_{pool}"
    phase = dict(state["phases"].get(phase_name, {}))
    arms = dict(phase.get("arms", {}))
    arm_state = dict(arms.get(arm, {}))
    existing = _try_archive(pool, arm, candidate)
    if existing is not None:
        document, identity = existing
        arm_state.update(status="complete", **identity)
        arms[arm] = arm_state
        _set_phase(state, phase_name, "running", arms=arms)
        return document, identity
    _require(
        not _eval_archive_path(pool, arm).exists(),
        f"{pool}/{arm} fixed tag already has an invalid archive; refusing to overwrite",
    )
    attempts = int(arm_state.get("attempts", 0))
    _require(allow_retry or attempts == 0, f"{pool}/{arm} a second launch is not allowed")
    command = _eval_command(pool, arm)
    frozen_inputs = _eval_input_identity(candidate)
    arm_state.update(
        status="running",
        attempts=attempts + 1,
        command=command,
        input_identity_sha256=frozen_inputs["identity_sha256"],
    )
    arms[arm] = arm_state
    _set_phase(state, phase_name, "running", arms=arms)
    try:
        _invoke(command, f"{pool}/{arm} evaluation")
        _require(
            _eval_input_identity(candidate) == frozen_inputs,
            f"{pool}/{arm} frozen inputs changed during evaluation",
        )
        document, identity = _validate_official_archive(pool, arm, candidate)
    except Exception as exc:
        try:
            unchanged = _eval_input_identity(candidate) == frozen_inputs
        except Exception:
            unchanged = False
        _require(unchanged, f"{pool}/{arm} frozen inputs changed during the failure")
        recovered = _try_archive(pool, arm, candidate)
        if recovered is None:
            arm_state.update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            arms[arm] = arm_state
            _set_phase(state, phase_name, "failed", arms=arms)
            raise
        document, identity = recovered
    arm_state.update(status="complete", **identity)
    arms[arm] = arm_state
    _set_phase(state, phase_name, "running", arms=arms)
    return document, identity


def command_eval_regression() -> None:
    state = _load_state(required=True)
    candidate = _validate_candidate(state=state)
    baseline_doc, _ = _run_eval_arm(
        state, "regression", "baseline", candidate, allow_retry=False
    )
    candidate_doc, _ = _run_eval_arm(
        state, "regression", "candidate", candidate, allow_retry=False
    )
    _pair_documents("regression", baseline_doc, candidate_doc)
    prior_phase = state["phases"].get("eval_regression", {})
    analysis, analysis_path, analysis_sha = _freeze_paired_analysis(
        "regression",
        baseline_doc,
        candidate_doc,
        candidate,
        allow_create=not bool(prior_phase.get("analysis_sha256")),
    )
    verdict = analysis["verdict"]
    phase = state["phases"]["eval_regression"]
    _set_phase(
        state,
        "eval_regression",
        "complete" if verdict["status"] == "PASS" else "scientific-fail",
        arms=phase["arms"],
        candidate_sha256=candidate["candidate_sha256"],
        expectations_sha256=candidate["expectations_sha256"],
        analysis_path=str(analysis_path),
        analysis_sha256=analysis_sha,
        verdict=verdict,
    )
    _require(
        verdict["status"] == "PASS",
        "7000-7031 paired regression failed the combat-recovery gate: "
        f"{verdict['failed_checks']}",
    )
    print("7000-7031 M29 paired regression completed; combat-recovery gate PASS.")


def _fresh_event_sha(record_without_sha: dict) -> str:
    return _canonical_sha256(record_without_sha)


def _read_fresh_ledger() -> list[dict]:
    if not FRESH_LEDGER_PATH.is_file():
        return []
    events: list[dict] = []
    previous = None
    payload = _stable_read(FRESH_LEDGER_PATH, "fresh append-only ledger")
    for index, raw in enumerate(payload.splitlines(), 1):
        _require(bool(raw), f"fresh ledger line {index} is empty")
        try:
            event = strict_json_loads(raw)
        except EvalContractError as exc:
            raise CampaignError(f"fresh ledger line {index} cannot be parsed") from exc
        _require(
            isinstance(event, dict)
            and set(event)
            == {
                "schema_version",
                "seq",
                "prev_event_sha256",
                "event",
                "time_ns",
                "payload",
                "event_sha256",
            },
            f"fresh ledger line {index} fields are malformed",
        )
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        _require(
            event["schema_version"] == FRESH_LEDGER_SCHEMA
            and event["seq"] == index
            and event["prev_event_sha256"] == previous
            and event["event_sha256"] == _fresh_event_sha(body),
            f"fresh ledger line {index} hash chain broken",
        )
        previous = event["event_sha256"]
        events.append(event)
    return events


def _append_fresh_event(event_name: str, payload: dict) -> dict:
    _require(
        event_name
        in {
            "BIND",
            "BASELINE_START",
            "BASELINE_SUCCESS",
            "BASELINE_FAIL",
            "CANDIDATE_START",
            "CANDIDATE_SUCCESS",
            "CANDIDATE_FAIL",
        }
        and isinstance(payload, dict),
        f"fresh ledger event name/payload illegal: {event_name}",
    )
    events = _read_fresh_ledger()
    body = {
        "schema_version": FRESH_LEDGER_SCHEMA,
        "seq": len(events) + 1,
        "prev_event_sha256": (
            events[-1]["event_sha256"] if events else None
        ),
        "event": event_name,
        "time_ns": time.time_ns(),
        "payload": payload,
    }
    record = {**body, "event_sha256": _fresh_event_sha(body)}
    FRESH_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FRESH_LEDGER_PATH, "ab") as stream:
        stream.write(_canonical_json_bytes(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    _require(_read_fresh_ledger()[-1] == record, "fresh ledger read-back after append failed")
    return record


def _validate_ledger_archive_identity(payload: Any, label: str) -> dict:
    _require(
        isinstance(payload, dict)
        and set(payload) == EVAL_ARCHIVE_IDENTITY_KEYS,
        f"{label} archive identity fields are malformed",
    )
    for key in (
        "sha256",
        "worker_sha256",
        "manager_sha256",
        "protocol_bundle_sha256",
    ):
        _require(_is_sha256(payload[key]), f"{label}.{key} is not a SHA256")
    _require(
        isinstance(payload["path"], str) and bool(payload["path"]),
        f"{label}.path is illegal",
    )
    _require(
        payload["worker_receipt_sha256"] is None
        or _is_sha256(payload["worker_receipt_sha256"]),
        f"{label}.worker_receipt_sha256 is illegal",
    )
    return payload


def _fresh_summary(events: list[dict]) -> dict:
    if not events:
        return {
            "bind": None,
            "baseline_started": False,
            "baseline_start_payload": None,
            "baseline_success": False,
            "baseline_failed": False,
            "baseline_success_payload": None,
            "candidate_attempts": 0,
            "candidate_open_attempt": None,
            "candidate_start_payload": None,
            "candidate_success": False,
            "candidate_failed": False,
            "candidate_success_payload": None,
        }
    _require(events[0]["event"] == "BIND", "fresh ledger first event must be BIND")
    bind = events[0]
    _require(
        isinstance(bind["payload"], dict) and bool(bind["payload"]),
        "fresh BIND payload must be a non-empty object",
    )
    stage = "bound"
    candidate_attempts = 0
    candidate_open_attempt = None
    candidate_start_payload = None
    baseline_start_payload = None
    baseline_success_payload = None
    candidate_success_payload = None
    for event in events[1:]:
        name = event["event"]
        payload = event["payload"]
        _require(isinstance(payload, dict), f"fresh {name} payload must be an object")
        if name == "BASELINE_START":
            _require(stage == "bound", "fresh BASELINE_START order/duplicate is malformed")
            _require(
                set(payload)
                == {
                    "command",
                    "input_identity_sha256",
                    "pool_opened_marker_sha256",
                }
                and isinstance(payload["command"], list)
                and _is_sha256(payload["input_identity_sha256"])
                and _is_sha256(payload["pool_opened_marker_sha256"]),
                "fresh BASELINE_START payload is malformed",
            )
            baseline_start_payload = dict(payload)
            stage = "baseline-started"
        elif name == "BASELINE_SUCCESS":
            _require(
                stage == "baseline-started",
                "fresh BASELINE_SUCCESS must immediately follow the single START",
            )
            baseline_success_payload = dict(
                _validate_ledger_archive_identity(
                    payload, "fresh BASELINE_SUCCESS"
                )
            )
            stage = "baseline-success"
        elif name == "BASELINE_FAIL":
            _require(
                stage == "baseline-started",
                "fresh BASELINE_FAIL must immediately follow the single START",
            )
            _require(
                set(payload) == {"error", "inputs_unchanged"}
                and isinstance(payload["error"], str)
                and isinstance(payload["inputs_unchanged"], bool),
                "fresh BASELINE_FAIL payload is malformed",
            )
            stage = "baseline-failed"
        elif name == "CANDIDATE_START":
            _require(
                stage == "baseline-success",
                "fresh CANDIDATE_START came before baseline SUCCESS or the previous attempt is not closed",
            )
            candidate_attempts += 1
            _require(
                set(payload)
                == {
                    "attempt",
                    "command",
                    "input_identity_sha256",
                    "candidate_fired_marker_sha256",
                }
                and payload.get("attempt") == candidate_attempts
                and isinstance(payload["command"], list)
                and _is_sha256(payload["input_identity_sha256"])
                and _is_sha256(payload["candidate_fired_marker_sha256"]),
                "fresh candidate attempt payload/number is malformed",
            )
            _require(
                candidate_attempts <= FRESH_CANDIDATE_MAX_ATTEMPTS,
                "fresh candidate attempts exceed the pre-registered limit",
            )
            candidate_open_attempt = candidate_attempts
            candidate_start_payload = dict(payload)
            stage = "candidate-started"
        elif name == "CANDIDATE_FAIL":
            _require(
                stage == "candidate-started"
                and payload.get("attempt") == candidate_open_attempt,
                "fresh CANDIDATE_FAIL has no matching START",
            )
            _require(
                set(payload)
                in (
                    {"attempt", "error", "inputs_unchanged"},
                    {"attempt", "error", "recovered_after_restart"},
                )
                and isinstance(payload["error"], str)
                and all(
                    isinstance(value, bool)
                    for key, value in payload.items()
                    if key in {"inputs_unchanged", "recovered_after_restart"}
                ),
                "fresh CANDIDATE_FAIL payload is malformed",
            )
            candidate_open_attempt = None
            stage = "candidate-failed"
        elif name == "CANDIDATE_SUCCESS":
            _require(
                stage == "candidate-started"
                and set(payload) == {"attempt", "archive"}
                and payload.get("attempt") == candidate_open_attempt,
                "fresh CANDIDATE_SUCCESS has no matching START",
            )
            candidate_success_payload = dict(
                _validate_ledger_archive_identity(
                    payload["archive"], "fresh CANDIDATE_SUCCESS"
                )
            )
            candidate_open_attempt = None
            stage = "candidate-success"
        else:
            raise CampaignError(f"fresh ledger unknown/duplicate event: {name}")
    return {
        "bind": bind,
        "baseline_started": stage
        in {
            "baseline-started",
            "baseline-success",
            "baseline-failed",
            "candidate-started",
            "candidate-failed",
            "candidate-success",
        },
        "baseline_start_payload": baseline_start_payload,
        "baseline_success": stage
        in {
            "baseline-success",
            "candidate-started",
            "candidate-failed",
            "candidate-success",
        },
        "baseline_failed": stage == "baseline-failed",
        "baseline_success_payload": baseline_success_payload,
        "candidate_attempts": candidate_attempts,
        "candidate_open_attempt": candidate_open_attempt,
        "candidate_start_payload": candidate_start_payload,
        "candidate_success": stage == "candidate-success",
        "candidate_failed": stage == "candidate-failed",
        "candidate_success_payload": candidate_success_payload,
    }


def _verify_fresh_ledger_checkpoint(state: dict, events: list[dict]) -> None:
    phase = state["phases"].get("eval_fresh", {})
    count = phase.get("ledger_event_count")
    head = phase.get("ledger_head_sha256")
    _require(
        (count is None) == (head is None),
        "fresh state ledger count/head exists only halfway",
    )
    if count is None:
        return
    _require(
        isinstance(count, int)
        and not isinstance(count, bool)
        and 1 <= count <= len(events)
        and _is_sha256(head)
        and events[count - 1]["event_sha256"] == head,
        "fresh ledger was truncated/rewritten and does not contain the prefix the state anchored",
    )


def _checkpoint_fresh_ledger(
    state: dict,
    status: str,
    **fields: Any,
) -> list[dict]:
    events = _read_fresh_ledger()
    _fresh_summary(events)
    _require(bool(events), "fresh ledger has no events yet and cannot be anchored")
    _set_phase(
        state,
        "eval_fresh",
        status,
        ledger_path=str(FRESH_LEDGER_PATH),
        ledger_event_count=len(events),
        ledger_head_sha256=events[-1]["event_sha256"],
        **fields,
    )
    return events


def _fresh_bind_payload(candidate: dict, frozen_inputs: dict) -> dict:
    return {
        "launcher": _launcher_identity(),
        "regression_gate_revision": REGRESSION_GATE_REVISION,
        "candidate_sha256": candidate["candidate_sha256"],
        "receipt_sha256": candidate["receipt_sha256"],
        "expectations_sha256": candidate["expectations_sha256"],
        "implementation_sha256": candidate["implementation_sha256"],
        "bc_v2_demos_sha256": candidate["bc_v2_demos_sha256"],
        "seeds": list(EVAL_POOLS["fresh"]),
        "manager_sha256": M29_SHA256,
        "baseline_tag": EVAL_TAGS[("fresh", "baseline")],
        "candidate_tag": EVAL_TAGS[("fresh", "candidate")],
        "input_identity_sha256": frozen_inputs["identity_sha256"],
        "frozen_inputs": frozen_inputs,
    }


def _require_regression_complete(state: dict, candidate: dict) -> None:
    phase = state["phases"].get("eval_regression", {})
    _require(
        phase.get("status") == "complete"
        and phase.get("candidate_sha256") == candidate["candidate_sha256"]
        and phase.get("expectations_sha256") == candidate["expectations_sha256"],
        "before the fresh final exam, the same candidate must first pass the 7000 paired regression",
    )
    baseline, _ = _validate_official_archive(
        "regression", "baseline", candidate
    )
    contender, _ = _validate_official_archive(
        "regression", "candidate", candidate
    )
    _pair_documents("regression", baseline, contender)
    analysis, analysis_path, analysis_sha = _freeze_paired_analysis(
        "regression",
        baseline,
        contender,
        candidate,
        allow_create=False,
    )
    _require(
        phase.get("analysis_path") == str(analysis_path)
        and phase.get("analysis_sha256") == analysis_sha
        and phase.get("verdict") == analysis["verdict"]
        and analysis["verdict"]["status"] == "PASS",
        "regression analysis/verdict missing, drifted or not passed; opening the fresh pool is forbidden",
    )


def _fresh_external_residue(
    *,
    allowed_eval_paths: tuple[pathlib.Path, ...] = (),
    include_control_markers: bool = True,
) -> list[str]:
    residues = _prior_fresh_control_residue()
    if include_control_markers:
        for marker in (
            FRESH_POOL_OPENED_PATH,
            FRESH_CANDIDATE_FIRED_PATH,
        ):
            if marker.exists():
                residues.append(str(marker))
    if not EVAL_DIR.is_dir():
        return residues
    allowed = {path.resolve() for path in allowed_eval_paths}
    fresh_seeds = set(EVAL_POOLS["fresh"])
    seed_literals = tuple(str(seed).encode("ascii") for seed in fresh_seeds)
    official_tags = tuple(
        EVAL_TAGS[("fresh", arm)].encode("utf-8")
        for arm in ("baseline", "candidate")
    )
    for path in sorted(EVAL_DIR.iterdir()):
        if not path.is_file() or path.name.endswith(".lock"):
            continue
        if path.resolve() in allowed:
            continue
        try:
            payload = _stable_read(path, "fresh contamination scan")
        except Exception:
            residues.append(f"{path} (cannot be read stably)")
            continue
        discovered: set[int] = set()
        try:
            document = strict_json_loads(payload)
            if isinstance(document, dict):
                meta = document.get("meta")
                if isinstance(meta, dict):
                    protocol = meta.get("protocol")
                    if isinstance(protocol, dict) and isinstance(
                        protocol.get("seeds"), list
                    ):
                        discovered.update(
                            seed
                            for seed in protocol["seeds"]
                            if isinstance(seed, int) and not isinstance(seed, bool)
                        )
                rows = document.get("rows")
                if isinstance(rows, list):
                    discovered.update(
                        row.get("seed")
                        for row in rows
                        if isinstance(row, dict)
                        and isinstance(row.get("seed"), int)
                        and not isinstance(row.get("seed"), bool)
                    )
        except EvalContractError:
            pass
        if discovered & fresh_seeds:
            residues.append(
                f"{path} (seeds={sorted(discovered & fresh_seeds)})"
            )
        elif any(token in payload for token in seed_literals + official_tags):
            # A half-written tmp/void/corrupt JSON that leaks a target seed/tag counts as the pool having been opened.
            residues.append(f"{path} (suspected 12000-pool leftover)")
    return residues


def _prior_fresh_control_residue() -> list[str]:
    """Find one-shot fresh-pool evidence left by any earlier campaign.

    An older launcher may have crashed after creating its control marker but
    before an evaluation archive became visible.  Scanning only ``EVAL_DIR``
    would then incorrectly declare the same 12000 pool fresh for a renamed
    campaign.
    """
    residues: list[str] = []
    control_root = CONTROL_DIR.parent
    if not control_root.is_dir():
        return residues
    current = CONTROL_DIR.resolve()
    evidence_names = (
        "fresh-ledger.jsonl",
        "fresh-12000-pool-opened.json",
        "fresh-12000-candidate-attempt-1-fired.json",
        "analysis-fresh.json",
    )
    for prior in sorted(control_root.glob("v4-combat-recovery*-control")):
        if not prior.is_dir() or prior.resolve() == current:
            continue
        for name in evidence_names:
            evidence = prior / name
            if evidence.exists():
                residues.append(f"{evidence} (prior campaign control evidence)")
        status_path = prior / "status.json"
        if not status_path.is_file():
            continue
        try:
            document = strict_json_loads(
                _stable_read(status_path, "prior campaign status")
            )
            phases = document.get("phases") if isinstance(document, dict) else None
            if not isinstance(phases, dict):
                raise CampaignError("prior campaign status lacks phases")
        except (CampaignError, EvalContractError, OSError, ValueError):
            residues.append(f"{status_path} (prior campaign status cannot be re-verified)")
            continue
        if "eval_fresh" in phases:
            residues.append(f"{status_path} (prior campaign eval_fresh phase)")
    return residues


def _require_fresh_pool_exclusive(summary: dict) -> None:
    allowed: list[pathlib.Path] = []
    if summary["baseline_started"]:
        allowed.append(_eval_archive_path("fresh", "baseline"))
    if summary["candidate_attempts"] > 0:
        allowed.append(_eval_archive_path("fresh", "candidate"))
    residues = _fresh_external_residue(
        allowed_eval_paths=tuple(allowed),
        include_control_markers=False,
    )
    _require(
        not residues,
        "after fresh BIND, an unregistered tag/leftover was found consuming the 12000 pool: "
        f"{residues}",
    )


def _fresh_output_lock_path(arm: str) -> pathlib.Path:
    output = _eval_archive_path("fresh", arm)
    return output.with_name(f".{output.name}.lock")


def _fresh_pool_marker(bind_payload: dict) -> dict | None:
    if not FRESH_POOL_OPENED_PATH.is_file():
        return None
    try:
        marker = strict_json_loads(
            _stable_read(FRESH_POOL_OPENED_PATH, "fresh pool-opened marker")
        )
    except EvalContractError as exc:
        raise CampaignError("fresh pool-opened marker cannot be parsed") from exc
    _require(
        isinstance(marker, dict)
        and set(marker)
        == {
            "schema_version",
            "created_at_ns",
            "bind",
            "baseline_command",
        }
        and marker["schema_version"] == FRESH_POOL_OPENED_SCHEMA
        and isinstance(marker["created_at_ns"], int)
        and not isinstance(marker["created_at_ns"], bool)
        and marker["created_at_ns"] > 0
        and marker["bind"] == bind_payload
        and marker["baseline_command"] == _eval_command("fresh", "baseline"),
        "fresh pool-opened marker differs from the current frozen candidate/command",
    )
    return marker


def _fresh_candidate_fired_marker(bind_payload: dict) -> dict | None:
    if not FRESH_CANDIDATE_FIRED_PATH.is_file():
        return None
    try:
        marker = strict_json_loads(
            _stable_read(
                FRESH_CANDIDATE_FIRED_PATH,
                "fresh candidate-fired marker",
            )
        )
    except EvalContractError as exc:
        raise CampaignError("fresh candidate-fired marker cannot be parsed") from exc
    _require(
        isinstance(marker, dict)
        and set(marker)
        == {
            "schema_version",
            "created_at_ns",
            "attempt",
            "bind",
            "candidate_command",
        }
        and marker["schema_version"] == FRESH_CANDIDATE_FIRED_SCHEMA
        and isinstance(marker["created_at_ns"], int)
        and not isinstance(marker["created_at_ns"], bool)
        and marker["created_at_ns"] > 0
        and marker["attempt"] == 1
        and marker["bind"] == bind_payload
        and marker["candidate_command"] == _eval_command("fresh", "candidate"),
        "fresh candidate-fired marker differs from the current BIND/single command",
    )
    return marker


def command_eval_fresh() -> None:
    state = _load_state(required=True)
    candidate = _validate_candidate(state=state)
    _require_regression_complete(state, candidate)
    frozen_inputs = _eval_input_identity(candidate)
    events = _read_fresh_ledger()
    _verify_fresh_ledger_checkpoint(state, events)
    summary = _fresh_summary(events)
    bind_payload = _fresh_bind_payload(candidate, frozen_inputs)
    if summary["bind"] is None:
        residues = _fresh_external_residue()
        _require(
            not residues,
            "the 12000 pool already had archives/void before the official BIND; treated as contaminated: "
            f"{residues}",
        )
        _append_fresh_event("BIND", bind_payload)
        events = _checkpoint_fresh_ledger(
            state,
            "bound",
            bind=bind_payload,
        )
        summary = _fresh_summary(events)
    else:
        _require(
            summary["bind"]["payload"] == bind_payload,
            "fresh ledger is already bound to another candidate/runtime/expectations",
        )
        _require(
            summary["bind"]["payload"]["frozen_inputs"] == frozen_inputs,
            "fresh scientific inputs changed after BIND",
        )
    marker = _fresh_pool_marker(bind_payload)
    candidate_fired_marker = _fresh_candidate_fired_marker(bind_payload)
    if summary["baseline_started"]:
        _require(
            marker is not None
            and _stable_sha256(
                FRESH_POOL_OPENED_PATH, "fresh pool-opened marker"
            )
            == summary["baseline_start_payload"]["pool_opened_marker_sha256"],
            "fresh BASELINE_START is not exactly bound to the current pool-opened marker",
        )
    if summary["candidate_attempts"] > 0:
        _require(
            candidate_fired_marker is not None
            and _stable_sha256(
                FRESH_CANDIDATE_FIRED_PATH,
                "fresh candidate-fired marker",
            )
            == summary["candidate_start_payload"][
                "candidate_fired_marker_sha256"
            ],
            "fresh CANDIDATE_START is not exactly bound to the current create-once fired marker",
        )
    else:
        _require(
            candidate_fired_marker is None,
            "fresh candidate-fired marker exists but the ledger has no START; "
            "treated as possibly launched, launch forbidden",
        )
    _require_fresh_pool_exclusive(summary)

    baseline_result = _try_archive("fresh", "baseline", candidate)
    if summary["baseline_success"]:
        _require(marker is not None, "fresh baseline SUCCESS but the pool-opened marker is missing")
        _require(baseline_result is not None, "fresh baseline SUCCESS archive disappeared/drifted")
        _require(
            baseline_result[1] == summary["baseline_success_payload"],
            "fresh baseline archive identity was replaced after SUCCESS",
        )
    elif summary["baseline_started"]:
        _require(marker is not None, "fresh baseline START but the pool-opened marker is missing")
        if baseline_result is not None and not summary["baseline_failed"]:
            _, identity = baseline_result
            _append_fresh_event("BASELINE_SUCCESS", identity)
            events = _checkpoint_fresh_ledger(
                state,
                "baseline-complete",
                bind=bind_payload,
                baseline=identity,
            )
            summary = _fresh_summary(events)
        else:
            _require(
                not _lock_file_is_held(_fresh_output_lock_path("baseline")),
                "fresh baseline process still holds the same-tag lock; wait for commit/failure, relaunch forbidden",
            )
            raise CampaignError(
                "fresh baseline was launched but has no re-verifiable successful archive; the 12000 blind pool is open, "
                "a second baseline launch is forbidden"
            )
    else:
        _require(
            marker is None,
            "fresh pool-opened marker exists but the ledger lacks BASELINE_START; "
            "treated as possibly launched, automatic baseline forbidden",
        )
        _require(
            _eval_input_identity(candidate) == frozen_inputs,
            "fresh frozen inputs changed before the baseline launch",
        )
        _exclusive_create_json(
            FRESH_POOL_OPENED_PATH,
            {
                "schema_version": FRESH_POOL_OPENED_SCHEMA,
                "created_at_ns": time.time_ns(),
                "bind": bind_payload,
                "baseline_command": _eval_command("fresh", "baseline"),
            },
            "fresh 12000 baseline",
        )
        marker = _fresh_pool_marker(bind_payload)
        _append_fresh_event(
            "BASELINE_START",
            {
                "command": _eval_command("fresh", "baseline"),
                "input_identity_sha256": frozen_inputs["identity_sha256"],
                "pool_opened_marker_sha256": _stable_sha256(
                    FRESH_POOL_OPENED_PATH, "fresh pool-opened marker"
                ),
            },
        )
        _checkpoint_fresh_ledger(
            state,
            "baseline-running",
            bind=bind_payload,
        )
        try:
            _invoke(
                _eval_command("fresh", "baseline"),
                "fresh/baseline evaluation",
            )
            _require(
                _eval_input_identity(candidate) == frozen_inputs,
                "fresh baseline frozen inputs changed during evaluation",
            )
            baseline_result = _validate_official_archive(
                "fresh", "baseline", candidate
            )
        except Exception as exc:
            try:
                unchanged = _eval_input_identity(candidate) == frozen_inputs
            except Exception:
                unchanged = False
            recovered = _try_archive("fresh", "baseline", candidate)
            if recovered is None or not unchanged:
                _append_fresh_event(
                    "BASELINE_FAIL",
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "inputs_unchanged": unchanged,
                    },
                )
                _checkpoint_fresh_ledger(
                    state,
                    "blocked-baseline",
                    error=f"{type(exc).__name__}: {exc}",
                    bind=bind_payload,
                )
                raise
            baseline_result = recovered
        _, baseline_identity = baseline_result
        _append_fresh_event("BASELINE_SUCCESS", baseline_identity)
        events = _checkpoint_fresh_ledger(
            state,
            "baseline-complete",
            baseline=baseline_identity,
            bind=bind_payload,
        )
        summary = _fresh_summary(events)

    _require(summary["baseline_success"], "fresh baseline was not frozen successfully")
    _require(
        _fresh_pool_marker(bind_payload) is not None
        and _stable_sha256(
            FRESH_POOL_OPENED_PATH, "fresh pool-opened marker"
        )
        == summary["baseline_start_payload"]["pool_opened_marker_sha256"],
        "fresh baseline pool-opened marker identity drifted at completion",
    )
    _require_fresh_pool_exclusive(summary)
    baseline_doc, baseline_identity = _validate_official_archive(
        "fresh", "baseline", candidate
    )
    _require(
        baseline_identity == summary["baseline_success_payload"],
        "fresh baseline ledger/archive identity differs",
    )

    candidate_result = _try_archive("fresh", "candidate", candidate)
    if summary["candidate_success"]:
        _require(candidate_result is not None, "fresh candidate SUCCESS archive disappeared/drifted")
        _require(
            candidate_result[1] == summary["candidate_success_payload"],
            "fresh candidate archive identity was replaced after SUCCESS",
        )
    elif candidate_result is not None:
        _, identity = candidate_result
        recovered_attempt = summary["candidate_open_attempt"]
        _require(
            recovered_attempt is not None,
            "fresh candidate archive appeared outside the officially open attempts",
        )
        _append_fresh_event(
            "CANDIDATE_SUCCESS",
            {"attempt": recovered_attempt, "archive": identity},
        )
        events = _checkpoint_fresh_ledger(
            state,
            "candidate-complete",
            candidate=identity,
            baseline=baseline_identity,
            bind=bind_payload,
        )
        summary = _fresh_summary(events)
    else:
        _require(
            not _eval_archive_path("fresh", "candidate").exists(),
            "fresh candidate fixed tag already has an invalid archive; overwrite/retry forbidden",
        )
        if summary["candidate_open_attempt"] is not None:
            _require(
                not _lock_file_is_held(_fresh_output_lock_path("candidate")),
                "fresh candidate process still holds the same-tag lock; wait for the current attempt to finish",
            )
            interrupted = summary["candidate_open_attempt"]
            _append_fresh_event(
                "CANDIDATE_FAIL",
                {
                    "attempt": interrupted,
                    "error": "launcher/evaluator interrupted, the same-tag lock released, no complete archive",
                    "recovered_after_restart": True,
                },
            )
            events = _checkpoint_fresh_ledger(
                state,
                "candidate-operational-failed",
                candidate_attempt=interrupted,
                baseline=baseline_identity,
                bind=bind_payload,
            )
            summary = _fresh_summary(events)
            raise CampaignError(
                "fresh candidate's first launch was interrupted with no complete archive; to prevent selective retries after seeing partial output,"
                " this pre-registered campaign forbids a second launch"
            )
        _require(
            summary["candidate_attempts"] < FRESH_CANDIDATE_MAX_ATTEMPTS,
            "fresh candidate has used its single pre-registered attempt; a second launch is forbidden",
        )
        _require(
            _eval_input_identity(candidate) == frozen_inputs,
            "fresh candidate frozen inputs differ from BIND before launch",
        )
        attempt = summary["candidate_attempts"] + 1
        command = _eval_command("fresh", "candidate")
        _exclusive_create_json(
            FRESH_CANDIDATE_FIRED_PATH,
            {
                "schema_version": FRESH_CANDIDATE_FIRED_SCHEMA,
                "created_at_ns": time.time_ns(),
                "attempt": attempt,
                "bind": bind_payload,
                "candidate_command": command,
            },
            "fresh 12000 candidate attempt 1",
        )
        candidate_fired_marker = _fresh_candidate_fired_marker(bind_payload)
        _require(
            candidate_fired_marker is not None,
            "fresh candidate-fired marker read-back after creation failed",
        )
        _append_fresh_event(
            "CANDIDATE_START",
            {
                "attempt": attempt,
                "command": command,
                "input_identity_sha256": frozen_inputs["identity_sha256"],
                "candidate_fired_marker_sha256": _stable_sha256(
                    FRESH_CANDIDATE_FIRED_PATH,
                    "fresh candidate-fired marker",
                ),
            },
        )
        _checkpoint_fresh_ledger(
            state,
            "candidate-running",
            candidate_attempt=attempt,
            baseline=baseline_identity,
            bind=bind_payload,
        )
        try:
            _invoke(command, f"fresh/candidate evaluation attempt {attempt}")
            _require(
                _eval_input_identity(candidate) == frozen_inputs,
                "fresh candidate frozen inputs changed during evaluation",
            )
            candidate_result = _validate_official_archive(
                "fresh", "candidate", candidate
            )
        except Exception as exc:
            try:
                unchanged = _eval_input_identity(candidate) == frozen_inputs
            except Exception:
                unchanged = False
            recovered = _try_archive("fresh", "candidate", candidate)
            if recovered is None or not unchanged:
                _append_fresh_event(
                    "CANDIDATE_FAIL",
                    {
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                        "inputs_unchanged": unchanged,
                    },
                )
                _checkpoint_fresh_ledger(
                    state,
                    "candidate-operational-failed",
                    candidate_attempt=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                    baseline=baseline_identity,
                    bind=bind_payload,
                )
                raise
            candidate_result = recovered
        _, candidate_identity = candidate_result
        _append_fresh_event(
            "CANDIDATE_SUCCESS",
            {"attempt": attempt, "archive": candidate_identity},
        )
        events = _checkpoint_fresh_ledger(
            state,
            "candidate-complete",
            candidate=candidate_identity,
            baseline=baseline_identity,
            bind=bind_payload,
        )
        summary = _fresh_summary(events)

    _require(summary["candidate_success"], "fresh candidate was not frozen successfully")
    candidate_doc, candidate_identity = _validate_official_archive(
        "fresh", "candidate", candidate
    )
    _require(
        candidate_identity == summary["candidate_success_payload"],
        "fresh candidate ledger/archive identity differs",
    )
    _require(
        _fresh_candidate_fired_marker(bind_payload) is not None
        and _stable_sha256(
            FRESH_CANDIDATE_FIRED_PATH,
            "fresh candidate-fired marker",
        )
        == summary["candidate_start_payload"][
            "candidate_fired_marker_sha256"
        ],
        "fresh candidate fired-marker identity drifted at completion",
    )
    _require_fresh_pool_exclusive(summary)
    _pair_documents("fresh", baseline_doc, candidate_doc)
    prior_fresh_phase = state["phases"].get("eval_fresh", {})
    analysis, analysis_path, analysis_sha = _freeze_paired_analysis(
        "fresh",
        baseline_doc,
        candidate_doc,
        candidate,
        allow_create=not bool(prior_fresh_phase.get("analysis_sha256")),
    )
    verdict = analysis["verdict"]
    _checkpoint_fresh_ledger(
        state,
        "complete" if verdict["status"] == "PASS" else "scientific-fail",
        baseline=baseline_identity,
        candidate=candidate_identity,
        bind=bind_payload,
        analysis_path=str(analysis_path),
        analysis_sha256=analysis_sha,
        verdict=verdict,
    )
    print(
        "12000-12031 fresh paired evaluation completed; the baseline was never launched twice, "
        f"candidate attempts={summary['candidate_attempts']}, "
        f"combat-recovery gate {verdict['status']}."
    )
    _require(
        verdict["status"] == "PASS",
        "fresh paired evaluation frozen but the combat-recovery gate FAILed: "
        f"{verdict['failed_checks']}",
    )


def _read_frozen_pair(pool: str, state: dict) -> tuple[dict, dict, dict]:
    candidate_info = _validate_candidate(state=state)
    phase = state["phases"].get(f"eval_{pool}", {})
    _require(
        phase.get("status") in {"complete", "scientific-fail"},
        f"{pool} paired eval not completed yet",
    )
    documents = []
    for arm in ("baseline", "candidate"):
        document, live_identity = _validate_official_archive(
            pool, arm, candidate_info
        )
        recorded = phase[arm] if arm in phase else phase.get("arms", {}).get(arm)
        _require(isinstance(recorded, dict), f"{pool}/{arm} state lacks an identity")
        _require(
            live_identity["sha256"] == recorded["sha256"]
            and live_identity["worker_sha256"] == recorded["worker_sha256"]
            and live_identity["manager_sha256"] == recorded["manager_sha256"],
            f"{pool}/{arm} archive identity has drifted",
        )
        documents.append(document)
    _pair_documents(pool, documents[0], documents[1])
    _require(
        documents[0]["meta"]["worker"]["sha256"] == V28_SHA256
        and documents[1]["meta"]["worker"]["sha256"]
        == candidate_info["candidate_sha256"],
        f"{pool} two-arm worker identity is malformed",
    )
    return documents[0], documents[1], candidate_info


def _paired_analysis(
    pool: str,
    baseline: dict,
    candidate: dict,
    *,
    archive_sha256s: tuple[str, str] | None = None,
) -> dict:
    left = {row["seed"]: row for row in baseline["rows"]}
    right = {row["seed"]: row for row in candidate["rows"]}
    _require(set(left) == set(right) == set(EVAL_POOLS[pool]), "paired rows seeds are malformed")
    metrics = tuple(PAIRED_METRIC_DIRECTIONS)
    paired_rows = []
    for seed in EVAL_POOLS[pool]:
        row = {"seed": seed}
        for metric in metrics:
            row[f"baseline_{metric}"] = left[seed][metric]
            row[f"candidate_{metric}"] = right[seed][metric]
            row[f"delta_{metric}"] = (
                int(right[seed][metric]) - int(left[seed][metric])
                if metric == "died"
                else right[seed][metric] - left[seed][metric]
            )
        paired_rows.append(row)

    summary: dict[str, Any] = {}
    for metric in metrics:
        deltas = [float(row[f"delta_{metric}"]) for row in paired_rows]
        direction = PAIRED_METRIC_DIRECTIONS[metric]
        desirability = (
            1.0 if direction == "higher"
            else -1.0 if direction == "lower"
            else None
        )
        signed = (
            [value * desirability for value in deltas]
            if desirability is not None
            else None
        )
        summary[metric] = {
            "baseline_mean": sum(
                float(int(row[f"baseline_{metric}"]))
                if metric == "died"
                else float(row[f"baseline_{metric}"])
                for row in paired_rows
            )
            / len(paired_rows),
            "candidate_mean": sum(
                float(int(row[f"candidate_{metric}"]))
                if metric == "died"
                else float(row[f"candidate_{metric}"])
                for row in paired_rows
            )
            / len(paired_rows),
            "delta_mean": sum(deltas) / len(deltas),
            "better_direction": direction,
            "paired_increases": sum(value > 0 for value in deltas),
            "paired_ties": sum(value == 0 for value in deltas),
            "paired_decreases": sum(value < 0 for value in deltas),
            "paired_better": (
                sum(value > 0 for value in signed)
                if signed is not None else None
            ),
            "paired_worse": (
                sum(value < 0 for value in signed)
                if signed is not None else None
            ),
        }
    if archive_sha256s is None:
        archive_sha256s = (
            _stable_sha256(
                _eval_archive_path(pool, "baseline"),
                f"{pool} baseline archive",
            ),
            _stable_sha256(
                _eval_archive_path(pool, "candidate"),
                f"{pool} candidate archive",
            ),
        )
    _require(
        len(archive_sha256s) == 2
        and all(_is_sha256(value) for value in archive_sha256s),
        f"{pool} analysis lacks the two-arm archive SHAs",
    )
    analysis = {
        "schema_version": PAIRED_ANALYSIS_SCHEMA,
        "launcher": _launcher_identity(),
        "pool": pool,
        "seeds": list(EVAL_POOLS[pool]),
        "primary_metrics": ["farm_worker_wage", "farm_worker_kills"],
        "secondary_metrics": ["ret", "kills", "died"],
        "farm_stratum_diagnostics": [
            "farm_dry_n",
            "farm_fresh_n",
            "farm_dry_worker_wage",
            "farm_fresh_worker_wage",
            "farm_dry_worker_kills",
            "farm_fresh_worker_kills",
        ],
        "potion_diagnostics": [
            "farm_voluntary_drinks",
            "farm_reflex_drains",
            "farm_multi_drink_windows",
            "farm_max_voluntary_drinks_per_window",
            "ending_belt_heals",
        ],
        "baseline_archive_sha256": archive_sha256s[0],
        "candidate_archive_sha256": archive_sha256s[1],
        "summary": summary,
        "rows": paired_rows,
    }
    analysis["verdict"] = _combat_recovery_verdict(analysis)
    return analysis


def _combat_recovery_verdict(analysis: dict) -> dict:
    """The frozen minimum combat-recovery gate; a failed regression never consumes the fresh pool."""
    summary = analysis["summary"]
    rows = analysis["rows"]
    checks = {
        "farm_worker_wage_mean_strictly_higher": (
            summary["farm_worker_wage"]["delta_mean"] > 0.0
        ),
        "farm_worker_wage_strict_seed_majority": (
            summary["farm_worker_wage"]["paired_better"]
            >= len(rows) // 2 + 1
        ),
        "farm_worker_kills_mean_strictly_higher": (
            summary["farm_worker_kills"]["delta_mean"] > 0.0
        ),
        "farm_worker_kills_strict_seed_majority": (
            summary["farm_worker_kills"]["paired_better"]
            >= len(rows) // 2 + 1
        ),
        "total_return_mean_not_lower": (
            summary["ret"]["delta_mean"] >= 0.0
        ),
        "total_kills_mean_not_lower": (
            summary["kills"]["delta_mean"] >= 0.0
        ),
        "candidate_deaths_not_higher": (
            sum(int(row["candidate_died"]) for row in rows)
            <= sum(int(row["baseline_died"]) for row in rows)
        ),
        "candidate_multi_drink_windows_zero": (
            sum(int(row["candidate_farm_multi_drink_windows"]) for row in rows)
            == 0
        ),
        "candidate_max_one_voluntary_drink_per_window": (
            max(
                int(row["candidate_farm_max_voluntary_drinks_per_window"])
                for row in rows
            )
            <= 1
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "revision": REGRESSION_GATE_REVISION,
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
    }


def _freeze_paired_analysis(
    pool: str,
    baseline: dict,
    candidate: dict,
    candidate_info: dict,
    *,
    allow_create: bool = True,
) -> tuple[dict, pathlib.Path, str]:
    live_baseline, baseline_identity = _validate_official_archive(
        pool, "baseline", candidate_info
    )
    live_candidate, candidate_identity = _validate_official_archive(
        pool, "candidate", candidate_info
    )
    _pair_documents(pool, live_baseline, live_candidate)
    _require(
        live_baseline == baseline and live_candidate == candidate,
        f"{pool} two-arm archives changed between reading and freezing the analysis",
    )
    analysis = _paired_analysis(
        pool,
        live_baseline,
        live_candidate,
        archive_sha256s=(
            baseline_identity["sha256"],
            candidate_identity["sha256"],
        ),
    )
    analysis["candidate_sha256"] = candidate_info["candidate_sha256"]
    destination = CONTROL_DIR / f"analysis-{pool}.json"
    if destination.is_file():
        try:
            recorded = strict_json_loads(
                _stable_read(destination, f"{pool} frozen analysis")
            )
        except EvalContractError as exc:
            raise CampaignError(f"{pool} analysis cannot be parsed") from exc
        _require(
            recorded == analysis,
            f"{pool} analysis exists but disagrees with the current frozen archives/launcher verdict",
        )
    else:
        _require(
            allow_create,
            f"{pool} analysis identity recorded but the analysis file is missing; silent rebuild forbidden",
        )
        _exclusive_create_json(destination, analysis, f"{pool} paired analysis")
    sha256 = _stable_sha256(destination, f"{pool} analysis")
    _require(
        sha256 == hashlib.sha256(
            json.dumps(
                analysis,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        f"{pool} analysis written bytes disagree with the in-memory verdict",
    )
    return analysis, destination, sha256


def command_analyze(pool: str) -> None:
    state = _load_state(required=True)
    baseline, candidate, candidate_info = _read_frozen_pair(pool, state)
    analysis, destination, analysis_sha = _freeze_paired_analysis(
        pool,
        baseline,
        candidate,
        candidate_info,
        allow_create=False,
    )
    eval_phase = state["phases"].get(f"eval_{pool}", {})
    if pool == "regression":
        _require(
            eval_phase.get("analysis_sha256") == analysis_sha
            and eval_phase.get("verdict") == analysis["verdict"],
            "regression eval phase disagrees with the frozen analysis verdict",
        )
    elif eval_phase.get("status") in {"complete", "scientific-fail"}:
        _require(
            eval_phase.get("analysis_sha256") == analysis_sha
            and eval_phase.get("verdict") == analysis["verdict"],
            "fresh eval phase disagrees with the frozen analysis verdict",
        )
    phase = state["phases"].get(f"eval_{pool}", {})
    _set_phase(
        state,
        f"analysis_{pool}",
        "complete",
        path=str(destination),
        sha256=analysis_sha,
        baseline_archive_sha256=analysis["baseline_archive_sha256"],
        candidate_archive_sha256=analysis["candidate_archive_sha256"],
        verdict=analysis["verdict"],
    )
    print(json.dumps(analysis["summary"], ensure_ascii=False, indent=2))
    print(f"full paired analysis: {destination}")


def command_status() -> None:
    state = _load_state()
    health: dict[str, Any] = {}
    try:
        health["artifacts"] = {"status": "PASS", **_base_artifact_snapshot()}
    except Exception as exc:
        health["artifacts"] = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
    bc, reason = _try_bc_identities()
    health["bc"] = (
        {"status": "PASS", **bc}
        if bc is not None
        else {"status": "FAIL", "error": reason}
    )
    candidate_info = None
    try:
        candidate_info = _validate_candidate(state=state)
        health["candidate"] = {"status": "PASS", **candidate_info}
    except Exception as exc:
        health["candidate"] = {
            "status": "NOT_FROZEN",
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        ledger = _read_fresh_ledger()
        _verify_fresh_ledger_checkpoint(state, ledger)
        residue = _fresh_external_residue() if not ledger else []
        summary = _fresh_summary(ledger)
        live_verdict = None
        if ledger:
            _require(candidate_info is not None, "fresh ledger exists but the candidate cannot be re-verified")
            _require_regression_complete(state, candidate_info)
            frozen_inputs = _eval_input_identity(candidate_info)
            bind_payload = _fresh_bind_payload(candidate_info, frozen_inputs)
            _require(
                summary["bind"]["payload"] == bind_payload,
                "fresh ledger BIND differs from the current frozen inputs",
            )
            marker = _fresh_pool_marker(bind_payload)
            candidate_marker = _fresh_candidate_fired_marker(bind_payload)
            if summary["baseline_started"]:
                _require(
                    marker is not None
                    and _stable_sha256(
                        FRESH_POOL_OPENED_PATH, "fresh pool-opened marker"
                    )
                    == summary["baseline_start_payload"][
                        "pool_opened_marker_sha256"
                    ],
                    "fresh status: baseline marker drift",
                )
            else:
                _require(
                    marker is None,
                    "fresh status: pool marker exists but the ledger has no BASELINE_START",
                )
            if summary["candidate_attempts"]:
                _require(
                    candidate_marker is not None
                    and _stable_sha256(
                        FRESH_CANDIDATE_FIRED_PATH,
                        "fresh candidate-fired marker",
                    )
                    == summary["candidate_start_payload"][
                        "candidate_fired_marker_sha256"
                    ],
                    "fresh status: candidate marker drift",
                )
            else:
                _require(
                    candidate_marker is None,
                    "fresh status: candidate marker exists but the ledger has no CANDIDATE_START",
                )
            _require_fresh_pool_exclusive(summary)
            baseline_document = None
            if summary["baseline_success"]:
                baseline_document, baseline_identity = (
                    _validate_official_archive(
                        "fresh", "baseline", candidate_info
                    )
                )
                _require(
                    baseline_identity == summary["baseline_success_payload"],
                    "fresh status: baseline SUCCESS identity drift",
                )
            if summary["candidate_success"]:
                _require(
                    baseline_document is not None,
                    "fresh status: candidate SUCCESS lacks the baseline",
                )
                candidate_document, candidate_identity = (
                    _validate_official_archive(
                        "fresh", "candidate", candidate_info
                    )
                )
                _require(
                    candidate_identity == summary["candidate_success_payload"],
                    "fresh status: candidate SUCCESS identity drift",
                )
                _pair_documents(
                    "fresh", baseline_document, candidate_document
                )
                analysis, analysis_path, analysis_sha = (
                    _freeze_paired_analysis(
                        "fresh",
                        baseline_document,
                        candidate_document,
                        candidate_info,
                        allow_create=False,
                    )
                )
                phase = state["phases"].get("eval_fresh", {})
                _require(
                    phase.get("status") in {"complete", "scientific-fail"}
                    and phase.get("analysis_path") == str(analysis_path)
                    and phase.get("analysis_sha256") == analysis_sha
                    and phase.get("verdict") == analysis["verdict"],
                    "fresh status: eval phase/analysis verdict does not close",
                )
                live_verdict = analysis["verdict"]
        if residue:
            ledger_status = "POOL_CONTAMINATED"
        elif summary["baseline_failed"] or summary["candidate_failed"]:
            ledger_status = "OPERATIONAL_FAIL"
        elif live_verdict is not None:
            ledger_status = (
                "PASS"
                if live_verdict["status"] == "PASS"
                else "SCIENTIFIC_FAIL"
            )
        elif ledger:
            ledger_status = "IN_PROGRESS"
        else:
            ledger_status = "CLEAN"
        health["fresh_ledger"] = {
            "status": ledger_status,
            "events": len(ledger),
            "head_sha256": ledger[-1]["event_sha256"] if ledger else None,
            "summary": summary,
            "pool_opened_marker_exists": FRESH_POOL_OPENED_PATH.exists(),
            "candidate_fired_marker_exists":
                FRESH_CANDIDATE_FIRED_PATH.exists(),
            "pre_bind_residue": residue,
            "verdict": live_verdict,
        }
    except Exception as exc:
        health["fresh_ledger"] = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(
        json.dumps(
            {"state": state, "health": health},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="the single official protocol-v4 M29/KING combat-recovery launcher"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-bc", help="re-sample/re-verify protocol-v4 BC-v1 + M29 BC-v2")
    sub.add_parser("train", help="run the single 249,856-step safe-prefix replay, which cannot be forked or relaunched")
    sub.add_parser("eval-regression", help="7000–7031 M29 paired regression")
    sub.add_parser("eval-fresh", help="12000-12031 one-time M29 paired fresh eval")
    analyze = sub.add_parser("analyze", help="summarize the frozen paired archives")
    analyze.add_argument(
        "--pool",
        choices=tuple(EVAL_POOLS),
        default="fresh",
    )
    sub.add_parser("status", help="read-only re-verification of the campaign state and artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _require_seed_pool_discipline()
        if args.command == "status":
            command_status()
            return 0
        with exclusive_lock(CONTROL_LOCK, "official-v4 launcher"):
            if args.command == "prepare-bc":
                command_prepare_bc()
            elif args.command == "train":
                command_train()
            elif args.command == "eval-regression":
                command_eval_regression()
            elif args.command == "eval-fresh":
                command_eval_fresh()
            elif args.command == "analyze":
                command_analyze(args.pool)
            else:  # argparse's required subparser makes this unreachable in theory.
                raise CampaignError(f"unknown command: {args.command}")
        return 0
    except (
        CampaignError,
        CommandFailed,
        EvalContractError,
        OutputReservationError,
        OSError,
        ValueError,
    ) as exc:
        print(f"official-v4 FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
