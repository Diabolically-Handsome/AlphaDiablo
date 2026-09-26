"""v23: BC warm start of the FARM operator brain (docs/prereg/PREREG-v23.md D4).

On-policy collection: frozen H manager + scripted teacher (dispatch farm branch; reflex steps belong to the wrapper
and never enter the set; fuse-forced steps are dropped whole). Current demo seeds 2102000-2102127, FARM
windows only; the consumed 2100000-2100127 are permanently burned.
Output: train/runs/bc-worker/policy_sd.pt (SB3 key names, for --bc-init) + bc_report.json.
Candidate selection: whole-episode validation carved out of the training episodes; validation top-1 < 0.95 or
validation recall < 0.85 for any class with >= 300 training-side samples -> retrain once with class-weighted CE
(the only BC retry). Only after the candidate is frozen is the final held-out used, once, for the same-threshold data gate and receipt.

E2 B1′ (docs/prereg/PREREG-v33-content-case.md E2): adds teacher v2 (bc-worker-v2) --
a preventive-drink branch ahead of dispatch("farm"): hp in [0.5, 0.65) and belt > 0 -> 12. The target must
be fully determined by the 298-dim visible state; the old hidden "once per window" latch labelled 75% of
legal states at the same HP line as non-12; measured, the lowest FPR at recall=.60 was still 3.5%: unlearnable.
Generation flag teacher_generation 1/2; v2 artifacts live in their own directory runs/bc-worker-v2/
(the v1 canonical path is untouched, avoiding the _previous archive exclusivity); v2 demos add per-sample
masks (captured live from env.action_masks(), the only on-manifold source of truth); the v2 receipt has its own
file name bc_report_v2.json + its own schema identifier + a dedicated validator (the v1 validator
fails loudly on v2 artifacts by construction); the n₁₂ gate and recall gate readings go into the receipt (fail-closed).
`python train/bc_worker.py` = v1 (unchanged); `--v2 [--preventive-threshold 0.7]` = v2.

Plan A (approved 2026-07-19): (1) v2 collects v1 x 3 episodes (_V2_COLLECTION_EPISODE_FACTOR;
the rev6 diagnosis had already looked at the old 100..483 pool, rev12 then found that the final domain of 1000..1383 was
read early by the coverage diagnosis; the later 2101000..2101383 was also opened by a one-shot producer,
so the current active v2 pool moved to the never-used, disjoint 2103000..2103383); (2) v2 primary training switches to class-balanced
weighted CE (w_c = N/(K·n_c), the standard balancing formula, class set = classes actually present). Both apply only to v2;
v1 episode count/seed discipline and the v1 training path (train_bc is called without weights) are unchanged.

Audit fix (2026-07-25): with fully class-balanced CE the very rare a12 (about 0.04% positives) is amplified
a thousandfold with no explicit constraint on false positives; the old artifact passed the recall_12 gate yet produced 867
false drinks on held-out. The final v2 model therefore adds a teacher-boundary calibration stage fitted on the training set only: the real masks
decide whether a12 is reachable, and the hp band plus the visible this-window drink latch of feature 297 enter a reserved
neural pathway inside the six SB3 tensors that is isolated from optimisation step 0; its bias is fitted only on nested-fit
episodes; the fixed validation/final-heldout episodes are used only for model selection/the final hard gate, never for tuning.
"""
import json
import hashlib
import math
import os
import pathlib
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import WorkerWindowEnv
from diablogym.options_env import (
    WORKER_DRINK_LATCH_FEATURE,
    WORKER_OBSERVATION_VIEW_A12_OVERLAY,
    dispatch,
)
from eval_contract import PROTOCOL_VERSION, exclusive_lock
from train_ppo import (_A12_FPR_MAX, _A12_HIGH_HP_FALSE_DRINK_MAX,
                       _A12_CALIBRATION_DRINK_LATCH_FEATURE,
                       _A12_CALIBRATION_HP_FEATURE,
                       _A12_CALIBRATION_PREDICATE,
                       _A12_CALIBRATION_SCHEMA_VERSION,
                       _A12_CALIBRATION_TRAIN_RECALL_TARGET,
                       _A12_LEGAL_NEGATIVE_PROBABILITY_MAX,
                       _A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX,
                       _A12_VISIBLE_HP_BOUNDARY_EPS,
                       _A12_PRECISION_MIN, _A12_PREDICTED_SHARE_MAX,
                       _A12_PREDICTED_SHARE_MIN, _A12_RECALL_MIN,
                       _A13_SPILLOVER_MAX, _BC_REPORT_SCHEMA_VERSION,
                       _BC_AUX_BEHAVIOR_METRIC_KEYS,
                       _BC_FINAL_HOLDOUT_MARKER_SCHEMA,
                       _BC_FINAL_SPLIT_SEED,
                       _BC_SELECTION_SPLIT_SEED,
                       _BC_SELECTION_VALIDATION_FRACTION,
                       _BC_V2_TEACHER_RECALL_MIN,
                       _BC_V2_DEMOS_SCHEMA_VERSION,
                       _BC_V2_PASS_KEYS,
                       _BC_V2_REPORT_SCHEMA_VERSION,
                       _BC_V2_COLLECTION_EPISODES,
                       _WORKER_BC_DEMO_SEEDS,
                       _WORKER_BC_FORBIDDEN_ACTIONS,
                       _WORKER_BC_REQUIRED_RECALL_ACTIONS,
                       _WORKER_BC_MIN_ACTION14_LABELS,
                       _WORKER_BC_MIN_ACTION14_EPISODES,
                       _assert_bc_final_holdout_pool_disjoint,
                       _implementation_bundle_sha256,
                       _bc_final_holdout_marker_path,
                       _bc_final_holdout_pool_spec,
                       _bc_v2_post_drink_coverage,
                       _validate_bc_final_holdout_marker,
                       bc_aux_behavior_gate, bc_aux_behavior_metrics)

OUT = ROOT / "train" / "runs" / "bc-worker"
OUT.mkdir(parents=True, exist_ok=True)
NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
DEMO_SEEDS = list(_WORKER_BC_DEMO_SEEDS)

# ---- E2 B1′ teacher v2 (bc-worker-v2) constant registration (PREREG-v33-content-case E2/D7) ----
OUT_V2 = ROOT / "train" / "runs" / "bc-worker-v2"  # separate v2 artifact directory; v1 canonical untouched
TEACHER_GENERATION_V1 = 1
TEACHER_GENERATION_V2 = 2
_PREVENTIVE_HP_LOW = 0.5                # lower bound of the preventive band (closed; just above the 0.5 brainstem reflex,
                                        # reflex states are drained, so the teacher never sees them by construction)
_PREVENTIVE_THRESHOLD_MAIN = 0.65       # main-case preventive threshold (D7 "preventive threshold" row)
# The historical 0.70 OC reused the same final episodes after observing 0.65.
# Rev13 disables it until an independent, training-reserved pool is registered.
_REGISTERED_PREVENTIVE_THRESHOLDS = (_PREVENTIVE_THRESHOLD_MAIN,)
_V2_FORBIDDEN_ACTIONS = (11,)           # v2 path forbids 11, allows 12 (the guard surface is not weakened;
                                        # v1 path _WORKER_BC_FORBIDDEN_ACTIONS untouched)
_N12_GATE_MIN = 122                     # n₁₂ gate >= 122 (= half the audited point estimate of 244, D7)
_RECALL12_GATE_MIN = _BC_V2_TEACHER_RECALL_MIN
_BC_V2_REPORT_NAME = "bc_report_v2.json"  # separate v2 receipt file name (schema isolation)
# Separate schema identifier: a non-int string -- the v1 validator's (train_ppo._validate_bc_report)
# exact key-set assertion + _is_plain_int(schema_version)==1 assertion both blow up on v2 artifacts.
# ---- v2 a12 training-set calibration (the export is still a standard SB3 64x64 MLP) ----
if _A12_CALIBRATION_DRINK_LATCH_FEATURE != WORKER_DRINK_LATCH_FEATURE:
    raise RuntimeError("BC-v2 calibration active-drink bit drifted from the Worker observation contract")

# ---- BC candidate-selection sets (the final held-out may be read only after selection) ----
# Inverse-frequency weighting converges more slowly than unweighted primary training. The old code still gave the retry only 8 epochs;
# under protocol-v4 validation stopped at .9017 / recall10=.816, while the same
# frozen target reached .9982 / .997 by 12 epochs. v2 primary/retry runs were even bit-identical fake retries
# because of the same objective, seed and 8 epochs.
_BC_PRIMARY_EPOCHS = 8
_BC_WEIGHTED_RETRY_EPOCHS = 12

# ---- append-only replacement collection registries ----
_V2_COLLECTION_EPISODE_FACTOR = 3   # plan A (a): v2 collection episodes = v1 x 3
# The current active registry is v2 2103000..2103383, v1
# 2102000..2102127; all old pools stay permanently in the burned/training refusal table.
if list(DEMO_SEEDS) != list(range(DEMO_SEEDS[0],
                                  DEMO_SEEDS[0] + len(DEMO_SEEDS))):
    raise RuntimeError("v1 demo seeds are not consecutive ascending integers: premise of the v2 seed-extension rule broken")
DEMO_SEEDS_V2 = list(_BC_V2_COLLECTION_EPISODES)
if len(DEMO_SEEDS_V2) != _V2_COLLECTION_EPISODE_FACTOR * len(DEMO_SEEDS):
    raise RuntimeError("rev6 v2 replacement seed count is not v1x3")

def artifact_provenance():
    return {
        "schema_version": _BC_REPORT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "implementation_sha256": _implementation_bundle_sha256(),
        "generator_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()).hexdigest(),
        "manager_npz_sha256": hashlib.sha256(NPZ.read_bytes()).hexdigest(),
    }


def _final_holdout_pool_spec(
        generation: int, seeds: list[int] | tuple[int, ...]) -> dict:
    """Immutable final-pool identity; excludes the implementation hash, so code changes cannot reopen the same pool."""
    return _bc_final_holdout_pool_spec(generation, seeds)


def _final_holdout_marker_path(
        out_dir: pathlib.Path, generation: int,
        seeds: list[int] | tuple[int, ...]) -> tuple[pathlib.Path, dict, str]:
    return _bc_final_holdout_marker_path(out_dir, generation, seeds)


def _fsync_directory(path: pathlib.Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _assert_final_holdout_unused(
        out_dir: pathlib.Path, generation: int,
        seeds: list[int] | tuple[int, ...]) -> None:
    marker, _, pool_sha256 = _final_holdout_marker_path(
        out_dir, generation, seeds)
    if marker.exists():
        raise RuntimeError(
            "BC registered pool already consumed once; collecting/scoring the same pool again is forbidden: "
            f"generation={generation},pool_sha256={pool_sha256},"
            f"marker={marker}")
    _assert_bc_final_holdout_pool_disjoint(
        out_dir, generation, seeds)


def _mark_final_holdout_started(
        out_dir: pathlib.Path, generation: int,
        seeds: list[int] | tuple[int, ...], provenance: dict) -> dict:
    """Before any episode reset, durably burn the whole registered pool."""
    marker, spec, pool_sha256 = _final_holdout_marker_path(
        out_dir, generation, seeds)
    marker_parent_was_missing = not marker.parent.exists()
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker_parent_was_missing:
        # fsync the runs directory as well as the registry directory: without
        # this, a crash can lose the newly-created registry entry
        # even after the marker file itself was synchronized.
        _fsync_directory(marker.parent.parent)
    # Exact-pool O_EXCL alone cannot serialize two *different* marker names
    # whose episode sets overlap.  Hold one registry-wide lock across the
    # overlap scan and durable marker creation to close that TOCTOU window.
    with exclusive_lock(
            marker.parent / ".registry.lock",
            "BC final heldout registry"):
        if marker.exists():
            raise RuntimeError(
                "BC pool one-shot marker already exists; re-collection/re-reading is forbidden: "
                f"{marker}")
        _assert_bc_final_holdout_pool_disjoint(
            out_dir, generation, seeds)
        record = {
            **spec,
            "pool_sha256": pool_sha256,
            "marker_schema_version": _BC_FINAL_HOLDOUT_MARKER_SCHEMA,
            "started_at_ns": time.time_ns(),
            "provenance": dict(provenance),
            "consumption_stage": "before_pool_collection",
        }
        payload = json.dumps(
            record, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")
        try:
            fd = os.open(
                str(marker), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(
                "BC pool one-shot marker already exists; re-collection/re-reading is forbidden: "
                f"{marker}") from exc
        try:
            with os.fdopen(fd, "wb", closefd=True) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(marker.parent)
        except Exception:
            # Keep the marker even if the write is interrupted; its existence is itself evidence that the pool "may have been opened".
            raise
    return {
        **record,
        "marker_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _reject_same_generator_terminal(
        out_dir: pathlib.Path, report_name: str, provenance: dict) -> None:
    """A direct CLI run must not move a same-generator final state into _previous and bypass the launcher either."""
    candidates = [pathlib.Path(out_dir) / report_name]
    candidates.extend(sorted(
        (pathlib.Path(out_dir) / "_previous").glob(
            f"*/{report_name}")))
    for report in candidates:
        if not report.is_file():
            continue
        try:
            record = json.loads(report.read_text())
        except (OSError, ValueError):
            continue
        if (
            isinstance(record, dict)
            and record.get("data_gate") in {"PASS", "FAIL"}
            and record.get("implementation_sha256")
            == provenance.get("implementation_sha256")
            and record.get("generator_sha256")
            == provenance.get("generator_sha256")
        ):
            raise RuntimeError(
                "the current implementation/generator already has a final BC scientific state; "
                f"direct CLI retry or archive bypass is forbidden: {report}")


def begin_output_attempt(provenance: dict | None = None):
    """Remove old canonical artifacts so that after a FAIL downstream cannot consume the previous weights by mistake."""
    provenance = artifact_provenance() if provenance is None else provenance
    _assert_final_holdout_unused(
        OUT, TEACHER_GENERATION_V1, DEMO_SEEDS)
    _reject_same_generator_terminal(
        OUT, "bc_report.json", provenance)
    old = [OUT / name for name in ("policy_sd.pt", "bc_report.json", "demos.npz")
           if (OUT / name).exists()]
    if old:
        archive = OUT / "_previous" / str(time.time_ns())
        archive.mkdir(parents=True)
        for path in old:
            path.replace(archive / path.name)
    (OUT / "bc_report.json").write_text(json.dumps({"data_gate": "RUNNING"}))


def write_report(record):
    tmp = OUT / "bc_report.tmp.json"
    tmp.write_text(json.dumps(record, ensure_ascii=False))
    tmp.replace(OUT / "bc_report.json")


def teacher_action(env: WorkerWindowEnv) -> int:
    raw = env.oe.env._raw
    context = getattr(env.oe, "_worker_masks_and_distance", None)
    if callable(context):
        masks, nearest = context()
        return dispatch(
            "farm", raw, bool(masks[14]), action_mask=masks,
            nearest_engageable_distance=nearest)
    # Narrow compatibility branch for pure collection fixtures.  Production
    # WorkerWindowEnv always exposes the exact context above.
    masks = np.asarray(env.action_masks(), dtype=np.bool_)
    return dispatch(
        "farm", raw, bool(masks[14]), action_mask=masks)


def collect():
    env = WorkerWindowEnv(
        str(NPZ), max_steps=3000, rng_seed=0, seed_scope="bc-v1",
        # BC-v1 initializes/supervises the ordinary frozen Worker actor.  Its
        # demonstrations must use the same canonical protocol-v3 view as
        # rev14 PPO and deployment.
        legacy_policy_observation_view=True)
    X, Y, groups = [], [], []
    dropped = 0
    dropped_no_effect = 0
    dropped_no_effect_by_action = Counter()
    for i, seed in enumerate(DEMO_SEEDS):
        obs, _ = env.reset(seed=seed)
        while obs is not None:
            a = teacher_action(env)
            pair = (np.asarray(obs, dtype=np.float32), a)
            obs2, w, term, trunc, info = env.step(a)
            if info.get("overridden"):
                dropped += 1          # steps rewritten by the fuse are dropped whole
            elif "executed_action" not in info:
                raise RuntimeError("BC-v1 step is missing its executed_action receipt")
            elif info["executed_action"] is None:
                # A legal proposal may still lose a dynamic native race or
                # degrade to an explicit wait.  Training that observation
                # against the unexecuted proposal recreates the action10
                # no-op attractor in supervised initialization.
                dropped_no_effect += 1
                dropped_no_effect_by_action[int(a)] += 1
            else:
                if int(info["executed_action"]) != int(a):
                    raise RuntimeError(
                        "BC-v1 executed_action disagrees with the teacher proposal: "
                        f"requested={a},executed={info['executed_action']}")
                X.append(pair[0]); Y.append(pair[1]); groups.append(seed)
            # at episode end next_window returns None (never rolls a new episode -- demo pool discipline)
            obs = env.next_window() if (term or trunc) else obs2
        if (i + 1) % 16 == 0:
            print(f"  collected {i+1}/{len(DEMO_SEEDS)} episodes, {len(Y)} pairs"
                  f" (fuse drops {dropped}, no-effect drops {dropped_no_effect}"
                  f"{dict(sorted(dropped_no_effect_by_action.items()))})",
                  flush=True)
    # demo pool discipline assertion: exactly one episode per demo seed, zero fallback rerolls (otherwise unknown seeds leak into the data)
    if env.stats["episodes"] != len(DEMO_SEEDS) or env.stats["reseeds"] != 0:
        raise RuntimeError(f"demo pool seed discipline broken: {env.stats}")
    print(f"demos: {len(Y)} decision pairs, dropped fuse steps {dropped}, "
          f"dropped no-effect steps {dropped_no_effect}, "
          f"by action {dict(sorted(dropped_no_effect_by_action.items()))}, "
          f"class distribution {dict(sorted(Counter(Y).items()))}", flush=True)
    env.close()
    groups_array = np.asarray(groups, dtype=np.int64)
    labels = np.asarray(Y, dtype=np.int64)
    if not np.array_equal(np.unique(groups_array), np.asarray(DEMO_SEEDS)):
        raise RuntimeError(
            "demo set does not exactly cover the current fixed seeds 2102000..2102127")
    if np.isin(labels, _WORKER_BC_FORBIDDEN_ACTIONS).any():
        raise RuntimeError("demo set contains forbidden actions 11/12 (11 is always masked and belongs to the manager; 12 is"
                           " never collected after the scripted teacher's drain -- autonomy-era demo pool discipline)")
    return np.stack(X), labels, groups_array


class PiHead(nn.Module):
    """Isomorphic to the policy side of SB3 MlpPolicy(64,64)."""

    def __init__(self, obs_dim=298, n_act=15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh())
        self.head = nn.Linear(64, n_act)

    def forward(self, x):
        return self.head(self.net(x))


def split_by_episode(groups):
    """Split by whole episodes so adjacent frames of one deterministic trajectory cannot leak into held-out."""
    episodes = np.unique(groups)
    if len(episodes) < 2:
        raise ValueError("BC held-out needs at least 2 independent episodes")
    order = np.random.default_rng(_BC_FINAL_SPLIT_SEED).permutation(episodes)
    n_holdout = max(1, int(round(len(order) * 0.1)))
    holdout_episodes = order[:n_holdout]
    ho_mask = np.isin(groups, holdout_episodes)
    tr, ho = np.flatnonzero(~ho_mask), np.flatnonzero(ho_mask)
    if len(tr) == 0 or len(ho) == 0:
        raise ValueError("BC episode split produced an empty training set or empty held-out")
    return tr, ho, holdout_episodes


def _split_fit_validation_by_episode(groups):
    """Carve validation out of the existing training episodes; the final held-out stays fully isolated.

    The outer ``split_by_episode`` still defines the canonical held-out used for the report/final gate;
    this function only makes a second deterministic whole-episode split of its training-side episodes. Validation is used only
    to decide whether the first training triggers the single retry, and never enters the gradient. At least three episodes are needed to keep
    three non-empty sets: fit / validation / final-held-out.
    """
    groups = np.asarray(groups)
    outer_train, outer_holdout, _ = split_by_episode(groups)
    training_episodes = np.unique(groups[outer_train])
    if len(training_episodes) < 2:
        raise ValueError(
            "BC candidate selection needs at least 3 independent episodes"
            " (fit/validation/final-held-out each non-empty)")
    order = np.random.default_rng(
        _BC_SELECTION_SPLIT_SEED).permutation(training_episodes)
    n_validation = max(
        1, int(round(len(order) * _BC_SELECTION_VALIDATION_FRACTION)))
    # round(0.1*n) cannot swallow all of fit on the current pool; still clamp explicitly in case a future small pool changes that.
    n_validation = min(n_validation, len(order) - 1)
    validation_episodes = order[:n_validation]
    validation_mask = np.isin(groups, validation_episodes)
    outer_train_mask = np.zeros(len(groups), dtype=np.bool_)
    outer_train_mask[outer_train] = True
    fit = np.flatnonzero(outer_train_mask & ~validation_mask)
    validation = np.flatnonzero(outer_train_mask & validation_mask)
    if len(fit) == 0 or len(validation) == 0:
        raise ValueError("BC nested episode split produced an empty fit or validation set")
    if (np.intersect1d(fit, validation).size
            or np.intersect1d(fit, outer_holdout).size
            or np.intersect1d(validation, outer_holdout).size):
        raise RuntimeError("BC nested episode split sets intersect")
    return fit, validation, validation_episodes


def _require_zero_exact_label_conflicts(X, Y) -> None:
    """A2 demos-validity hard assertion: the same exact observation must not carry conflicting labels."""
    import hashlib as _hashlib
    seen: dict = {}
    for i in range(len(X)):
        key = _hashlib.blake2b(X[i].tobytes(), digest_size=16).digest()
        prev = seen.get(key)
        if prev is None:
            seen[key] = int(Y[i])
        elif prev != int(Y[i]):
            raise RuntimeError(
                f"BC demos label conflict: the same observation row carries actions {prev} and {int(Y[i])}")


def _bc_quality_gate_passes(top1, recalls) -> bool:
    """Candidate quality condition shared by v1/v2; the caller decides the input domain."""
    return float(top1) >= 0.95 and all(
        float(recall) >= 0.85 for recall in recalls.values())


def _bc_v2_quality_gate_passes(top1, recalls) -> bool:
    """v2 generic classification gate; a12 goes only through its own safety/recall behaviour gates.

    a12 has separately registered recall/FPR/high-HP false-drink/share/13-spillover constraints and a training-set calibration.
    Also putting it under ``>=300-class recall>=.85`` would, once n12 crosses 300 as collection grows,
    suddenly raise its dedicated ``recall>=.5`` gate to .85, conflicting with the frozen calibration target of .75.
    """
    return float(top1) >= 0.95 and all(
        int(action) == 12 or float(recall) >= 0.85
        for action, recall in recalls.items())


def _require_v1_action14_coverage(labels, groups) -> dict:
    """Fail before candidate training when the upgrade target is too sparse.

    Action 14 is a high-value but naturally rare event.  Letting it inherit the
    generic ``>=300`` eligibility rule made it disappear from both candidate
    selection and the final receipt even when the aggregate classifier looked
    excellent.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if (labels.ndim != 1 or groups.shape != labels.shape
            or labels.dtype != np.int64 or groups.dtype != np.int64):
        raise RuntimeError(
            "BC-v1 a14 coverage check requires same-shape 1-D int64 labels/groups")
    selected = labels == 14
    readings = {
        "labels": int(selected.sum()),
        "episodes": int(np.unique(groups[selected]).size),
    }
    if (readings["labels"] < _WORKER_BC_MIN_ACTION14_LABELS
            or readings["episodes"] < _WORKER_BC_MIN_ACTION14_EPISODES):
        raise RuntimeError(
            "BC-v1 a14 strict-upgrade coverage insufficient: "
            f"labels={readings['labels']}"
            f"<{_WORKER_BC_MIN_ACTION14_LABELS},"
            f"episodes={readings['episodes']}"
            f"<{_WORKER_BC_MIN_ACTION14_EPISODES}")
    return readings


def _reserve_a12_calibration_path(model: PiHead) -> None:
    """Isolate 3+1 calibration neurons from optimisation step 0, so no ordinary class representation has to be deleted afterwards."""
    if not isinstance(model, PiHead):
        raise TypeError("a12 reserved pathway only accepts PiHead")
    w0, b0 = model.net[0].weight, model.net[0].bias
    w1, b1 = model.net[2].weight, model.net[2].bias
    wa = model.head.weight
    with torch.no_grad():
        w0[61:64].zero_()
        b0[61:64].zero_()
        w1[:, 61:64].zero_()
        w1[63].zero_()
        b1[63].zero_()
        wa[:, 63].zero_()

    masks = []
    for parameter in (w0, b0, w1, b1, wa):
        masks.append(torch.ones_like(parameter))
    masks[0][61:64].zero_()
    masks[1][61:64].zero_()
    masks[2][:, 61:64].zero_()
    masks[2][63].zero_()
    masks[3][63].zero_()
    masks[4][:, 63].zero_()
    model._a12_reservation_hooks = [
        parameter.register_hook(
            lambda gradient, live_mask=live_mask: gradient * live_mask)
        for parameter, live_mask in zip((w0, b0, w1, b1, wa), masks)
    ]


def train_bc(X, Y, groups, class_weights=None, *,
             epochs: int = _BC_PRIMARY_EPOCHS, masks=None,
             reserve_a12_path: bool = False,
             required_recall_actions=()):
    """Train only on fit and return candidate-selection readings only on validation.

    The final held-out is not on this function's training, per-class counting or scoring path; the caller must
    call ``_score_bc_model`` explicitly for the single final gate evaluation after all retry/selection is done.
    """
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("BC epochs must be a positive integer")
    if masks is not None:
        masks = np.asarray(masks)
        if masks.shape != (len(Y), 15) or masks.dtype != np.bool_:
            raise ValueError(
                f"BC validation masks shape/dtype invalid: {masks.shape}/{masks.dtype}")
    torch.manual_seed(23)
    fit, validation, _ = _split_fit_validation_by_episode(groups)
    outer_train, _, _ = split_by_episode(groups)
    model = PiHead(X.shape[1])
    if reserve_a12_path:
        _reserve_a12_calibration_path(model)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    wt = None
    if class_weights is not None:
        wt = torch.as_tensor(class_weights, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X[fit]), torch.from_numpy(Y[fit]))
    dl = torch.utils.data.DataLoader(
        ds, batch_size=512, shuffle=True,
        generator=torch.Generator().manual_seed(23))
    for epoch in range(epochs):
        tot = cnt = correct = 0
        for xb, yb in dl:
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb, weight=wt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(yb); cnt += len(yb)
            correct += int((logits.argmax(1) == yb).sum())
        print(f"BC epoch {epoch}: loss {tot/cnt:.4f} acc {correct/cnt:.3f}", flush=True)
    # Candidate selection reads validation only; per-class threshold counts look only at outer training episodes,
    # so final held-out labels cannot change the retry path through a "does it reach 300" side channel.
    top1, recalls = _score_bc_model_indices(
        model, X, Y, validation, eligibility_indices=outer_train,
        masks=masks, required_recall_actions=required_recall_actions)
    return model, top1, recalls


def export_sb3_sd(model):
    sd = {
        "mlp_extractor.policy_net.0.weight": model.net[0].weight,
        "mlp_extractor.policy_net.0.bias": model.net[0].bias,
        "mlp_extractor.policy_net.2.weight": model.net[2].weight,
        "mlp_extractor.policy_net.2.bias": model.net[2].bias,
        "action_net.weight": model.head.weight,
        "action_net.bias": model.head.bias,
    }
    return {k: v.detach().clone() for k, v in sd.items()}


def _masked_bc_predictions(model, X, masks=None) -> np.ndarray:
    """Model argmax; v2 must consume the real masks from collection, v1 passes None to keep the original semantics."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(X, dtype=np.float32)))
        if masks is not None:
            masks = np.asarray(masks)
            if masks.shape != tuple(logits.shape) or masks.dtype != np.bool_:
                raise RuntimeError(
                    f"BC scoring masks shape/dtype invalid: {masks.shape}/{masks.dtype}")
            logits = torch.where(
                torch.from_numpy(masks), logits,
                torch.full_like(logits, -1e8))
        return logits.argmax(1).cpu().numpy()


def _score_bc_model_indices(model, X, Y, score_indices, *,
                            eligibility_indices, masks=None,
                            required_recall_actions=()):
    """Score on an explicit row set; per-class threshold counts read only the explicit eligibility row set."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    score_indices = np.asarray(score_indices, dtype=np.int64)
    eligibility_indices = np.asarray(eligibility_indices, dtype=np.int64)
    if (score_indices.ndim != 1 or eligibility_indices.ndim != 1
            or len(score_indices) == 0 or len(eligibility_indices) == 0):
        raise ValueError("BC scoring set/class eligibility set must be 1-D non-empty indices")
    required = tuple(required_recall_actions)
    if (any(not isinstance(action, (int, np.integer))
            or isinstance(action, (bool, np.bool_))
            or not 0 <= int(action) < 15 for action in required)
            or len(set(map(int, required))) != len(required)):
        raise ValueError("BC required_recall_actions must be unique integers in 0..14")
    pred = _masked_bc_predictions(
        model, X[score_indices],
        None if masks is None else np.asarray(masks)[score_indices])
    yh = Y[score_indices]
    top1 = float((pred == yh).mean())
    eligible_counts = Counter(Y[eligibility_indices].tolist())
    recalls = {}
    gated_classes = {
        int(k) for k, v in eligible_counts.items() if v >= 300
    }
    gated_classes.update(map(int, required))
    for c in sorted(gated_classes):
        selected = yh == c
        recalls[int(c)] = (
            float((pred[selected] == c).mean())
            if selected.sum() > 0 else 0.0)
    return top1, recalls


def _score_bc_model(model, X, Y, groups, masks=None,
                    required_recall_actions=()):
    """After selection, recompute top-1/per-class recall on the fixed final held-out.

    Calibration rewrites the six exported tensors, so v2 must be re-scored after calibration; reusing
    the pre-calibration validation numbers is forbidden. v1 calls pass no masks, and the report schema is unchanged.
    """
    _, ho, _ = split_by_episode(groups)
    return _score_bc_model_indices(
        model, X, Y, ho, eligibility_indices=np.arange(len(Y)),
        masks=masks, required_recall_actions=required_recall_actions)


def main():
    provenance = artifact_provenance()
    begin_output_attempt(provenance)
    # ``collect`` itself traverses every registered episode and performs
    # whole-pool integrity checks.  Burn the pool before the first reset so a
    # crash after those reads can never leave an apparently unused final set.
    pool_marker = _mark_final_holdout_started(
        OUT, TEACHER_GENERATION_V1, DEMO_SEEDS, provenance)
    pool_evidence = {
        "final_pool_sha256": pool_marker["pool_sha256"],
        "final_holdout_marker_sha256": pool_marker["marker_sha256"],
    }
    write_report({
        "data_gate": "FAIL",
        "failure_stage": "pool_collection_started",
        "final_heldout_consumed": True,
        **pool_evidence,
        **provenance,
    })
    X, Y, groups = collect()
    action14_coverage = _require_v1_action14_coverage(Y, groups)
    print(f"v1 a14 strict-upgrade coverage {action14_coverage}", flush=True)
    demos_tmp = OUT / "demos.tmp.npz"
    np.savez_compressed(demos_tmp, X=X, Y=Y, episode_id=groups)
    demos_tmp.replace(OUT / "demos.npz")
    tr, ho, holdout_episodes = split_by_episode(groups)
    fit, _, _ = _split_fit_validation_by_episode(groups)
    # train_bc sends only fit/validation into the gradient or candidate readings; although the whole pool was collected,
    # it was already marked permanently consumed by the pre-collection marker.
    model, top1, recalls = train_bc(
        X, Y, groups,
        required_recall_actions=_WORKER_BC_REQUIRED_RECALL_ACTIONS)
    retrained = False
    if not _bc_quality_gate_passes(top1, recalls):
        print(f"first training missed validation targets (top1 {top1:.3f} recall {recalls})"
              " -> class-weighted retraining (the single retry)",
              flush=True)
        counts = np.bincount(Y[fit], minlength=15).astype(np.float64)
        weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
        weights = weights / weights[weights > 0].mean()
        model, top1, recalls = train_bc(
            X, Y, groups, class_weights=weights,
            epochs=_BC_WEIGHTED_RETRY_EPOCHS,
            required_recall_actions=_WORKER_BC_REQUIRED_RECALL_ACTIONS)
        retrained = True
    if not _bc_quality_gate_passes(top1, recalls):
        # Amendment A2 (approved 2026-07-27): the policy-quality line is downgraded to
        # record only, no verdict -- 0.95/0.85 are left over from the R5/R6 era (BC as the teacher anchor); the R7 training command
        # does not consume this policy (teacher=KING_SD); the demos are what gets consumed (dry-anchor).
        # Measured ceiling (200ep + class weights, offline diagnosis on burned-2_104 demos): top1 ~0.88-0.90,
        # rare keys <= 0.43 -- under the unfiltered v3 view of 07-25 the gate was structurally unreachable.
        print(f"candidate quality line not met (top1 {top1:.3f} recall {recalls}) -- A2 record only, no verdict; "
              "continuing the release flow (quality readings are filed with the report)", flush=True)
    # The candidate is frozen; only now does the final held-out enter the model scoring path.
    top1, recalls = _score_bc_model(
        model, X, Y, groups,
        required_recall_actions=_WORKER_BC_REQUIRED_RECALL_ACTIONS)
    # A2: release gate = demos-validity (pool coverage/episode discipline/a14 coverage are already hard-asserted by collect and
    # _require_v1_action14_coverage; here we add the zero-label-conflict hard assertion).
    # held_out_top1/class_recalls keep their original fields in the file; the identity and bitwise recomputation chain are untouched.
    _require_zero_exact_label_conflicts(X, Y)
    ok = True
    report = {
        "pairs": len(Y), "held_out_top1": round(top1, 4),
        "held_out_pairs": int(len(ho)),
        "held_out_episodes": [int(x) for x in sorted(holdout_episodes)],
        "class_recalls": recalls, "class_weighted_retry": retrained,
        **pool_evidence,
        "data_gate": "PASS" if ok else "FAIL", **provenance}
    if not ok:
        write_report(report)
        raise RuntimeError(
            f"BC data gate FAIL (top1={top1:.3f}, recalls={recalls}); "
            "refusing to overwrite policy_sd.pt")
    policy_tmp = OUT / "policy_sd.tmp.pt"
    if artifact_provenance() != provenance:
        raise RuntimeError("implementation/engine/content/manager drifted while BC worker was running")
    torch.save(export_sb3_sd(model), policy_tmp)
    policy_tmp.replace(OUT / "policy_sd.pt")
    report["policy_sha256"] = hashlib.sha256(
        (OUT / "policy_sd.pt").read_bytes()).hexdigest()
    report["demos_sha256"] = hashlib.sha256(
        (OUT / "demos.npz").read_bytes()).hexdigest()
    write_report(report)
    print(f"held-out top-1 {top1:.3f} recall {recalls} retry={retrained} "
          f"-> data gate PASS; saved {OUT}/policy_sd.pt", flush=True)


# ==== E2 B1′ v2 parts (everything below is new; the v1 surface above is unchanged) ====

def forbidden_actions_for_generation(generation: int) -> tuple[int, ...]:
    """Forbidden-action assertion conditioned on generation (E2): v1 forbids (11,12) unchanged; v2 forbids 11, allows 12."""
    if generation == TEACHER_GENERATION_V1:
        return tuple(_WORKER_BC_FORBIDDEN_ACTIONS)   # (11, 12) unchanged
    if generation == TEACHER_GENERATION_V2:
        return _V2_FORBIDDEN_ACTIONS                 # forbid 11, allow 12 (guard surface not weakened)
    raise ValueError(f"unknown teacher generation: {generation!r} (only 1/2)")


def teacher_v2_preventive_trigger(
        observation, masks,
        preventive_threshold: float = _PREVENTIVE_THRESHOLD_MAIN) -> bool:
    """Pure observable precondition predicate of E2 teacher v2.

    ``hp in [0.5, threshold) and live m12 and no active drink yet this window``. hp lower bound closed, upper open;
    belt/action legality is read only from the live mask at collection time; per-window drink history is read only from the
    published sign bit of feature 297. The target no longer reads teacher-internal state or window history outside the 298 dims.
    """
    observation = np.asarray(observation)
    masks = np.asarray(masks)
    if observation.shape != (298,) or not np.issubdtype(
            observation.dtype, np.floating):
        raise ValueError(
            f"TeacherV2 observation must be a (298,) float array: {observation.shape}/"
            f"{observation.dtype}")
    if masks.shape != (15,) or masks.dtype != np.bool_:
        raise ValueError(
            f"TeacherV2 masks must be (15,) bool: {masks.shape}/{masks.dtype}")
    hp = float(observation[_A12_CALIBRATION_HP_FEATURE])
    latch = float(observation[_A12_CALIBRATION_DRINK_LATCH_FEATURE])
    if not math.isfinite(hp) or not math.isfinite(latch):
        raise ValueError("TeacherV2 visible hp/active-drink bit must be finite")
    no_prior_window_drink = latch >= 0.0
    return bool(
        hp >= _PREVENTIVE_HP_LOW - _A12_VISIBLE_HP_BOUNDARY_EPS
        and hp < preventive_threshold - _A12_VISIBLE_HP_BOUNDARY_EPS
        and masks[12]
        and no_prior_window_drink)


class TeacherV2:
    """E2 B1′ teacher v2: the observable preventive-drink branch takes priority over dispatch.

    Each decision reads only the current 298-dim observation and the live mask; the object keeps no hidden
    "already drank this window" state. Feature 297 publishes that bit explicitly, so the same observation always gets the same label.
    """

    def __init__(self, preventive_threshold: float = _PREVENTIVE_THRESHOLD_MAIN):
        if preventive_threshold not in _REGISTERED_PREVENTIVE_THRESHOLDS:
            raise ValueError(
                f"preventive threshold {preventive_threshold!r} not registered: rev13 only allows "
                "the 0.65 main case; the old 0.70 OC has no independent fresh pool")
        self.preventive_threshold = float(preventive_threshold)

    def begin_window(self) -> None:
        """Window notification kept for collector compatibility; the teacher target itself stays stateless."""

    def action(self, env, observation=None, masks=None) -> int:
        raw = env.oe.env._raw
        if observation is None:
            builder = getattr(
                env.oe, "_worker_policy_observation", None)
            observation = (
                builder(WORKER_OBSERVATION_VIEW_A12_OVERLAY)
                if callable(builder)
                else env.oe._worker_obs()
            )
        if masks is None:
            masks = np.asarray(env.action_masks(), dtype=np.bool_)
        observed_hp = float(
            np.asarray(observation)[_A12_CALIBRATION_HP_FEATURE])
        raw_hp = raw["hp"] / max(1, raw["max_hp"])
        if not math.isclose(
                observed_hp, raw_hp, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(
                "TeacherV2 decision-state obs/raw hp misaligned: "
                f"{observed_hp} != {raw_hp}")
        visible_latch = float(
            np.asarray(observation)[
                _A12_CALIBRATION_DRINK_LATCH_FEATURE])
        raw_preventive_state = (
            _PREVENTIVE_HP_LOW <= raw_hp < self.preventive_threshold
            and int(raw.get("belt_heals", 0)) > 0
            and visible_latch >= 0.0
        )
        if raw_preventive_state and not bool(np.asarray(masks)[12]):
            raise RuntimeError(
                "teacher v2 visible preventive state: action 12 not in the live mask: "
                "on-manifold demo discipline broken")
        if teacher_v2_preventive_trigger(
                observation, masks, self.preventive_threshold):
            return 12
        context = getattr(env.oe, "_worker_masks_and_distance", None)
        if callable(context):
            masks, nearest = context()
            a = dispatch(
                "farm", raw, bool(masks[14]), action_mask=masks,
                nearest_engageable_distance=nearest)
        else:
            a = dispatch(
                "farm", raw, bool(masks[14]), action_mask=masks)
        if a == 12:
            # The 0.5 reflex branch embedded in dispatch should be dead code for the teacher: the window-open drain + reflex-tail
            # drain guarantee the worker only observes reflex-free states; reaching here means demo pool discipline is broken, never collect silently.
            raise RuntimeError("teacher v2 saw a reflex state (hp<0.5 and belt>0): drain breached; "
                               "real a12 labels may only come from the preventive precondition branch")
        return a


def collect_v2(preventive_threshold: float = _PREVENTIVE_THRESHOLD_MAIN,
               manager_npz: str | pathlib.Path = NPZ,
               manager_sha256: str | None = None):
    """E2: with autonomy on, the collection loop really executes and really fills the pool (on-manifold, no counterfactual labels).

    Uses the same real-execution discipline as v1 collect():
      - per-sample masks are captured live from env.action_masks() in the decision state (the single source of truth;
        inferring a12's belt bit from the obs belt dims would be a second source of truth, forbidden);
      - overridden steps and executed_action=None steps are dropped whole (refused/no-effect steps never enter the pool);
      - TeacherV2 labels depend only on the current 298-dim hp/active-drink bit and the live mask;
        window notifications never change the label of the same observation;
      - forbidden-action assertion conditioned on generation: v2 forbids 11, allows 12;
      - the current collection loop consumes DEMO_SEEDS_V2 (v1 x 3, fixed
        2103000..2103383); v1 consumes fresh 2102000..2102127.
    Returns (X, labels, groups, masks, belts); belts are per-sample belt readings in the decision state
    (source for the belt-economy receipt).
    """
    teacher = TeacherV2(preventive_threshold)
    manager_path = pathlib.Path(manager_npz)
    if not manager_path.is_file():
        raise RuntimeError(f"BC-v2 manager npz does not exist: {manager_path}")
    if manager_sha256 is None:
        manager_sha256 = hashlib.sha256(
            manager_path.read_bytes()).hexdigest()
    if (not isinstance(manager_sha256, str) or len(manager_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in manager_sha256)):
        raise RuntimeError("BC-v2 manager_sha256 invalid")
    env = WorkerWindowEnv(
        str(manager_path), max_steps=3000, rng_seed=0,
        manager_sha256=manager_sha256, seed_scope="bc-v2",
        # Teacher-v2 and the A12 gate require the registered overlay: an exact
        # v3 base plus reversible free-slot/drink-latch fields.  Keep this
        # explicit so defaults cannot rewrite targets or feed the inherited
        # actor an unrecoverable filtered v4 row.
        legacy_policy_observation_view=False)
    X, Y, M, groups, belts = [], [], [], [], []
    dropped = 0
    dropped_no_effect = 0
    dropped_no_effect_by_action = Counter()
    for i, seed in enumerate(DEMO_SEEDS_V2):
        obs, _ = env.reset(seed=seed)
        teacher.begin_window()            # reset is the first window; open the latch
        while obs is not None:
            masks = np.asarray(env.action_masks(), dtype=bool)  # captured live
            a = teacher.action(env, obs, masks)
            if not masks[a]:
                raise RuntimeError(
                    f"teacher v2 proposal {a} not in the live mask (a12 needs m[12]=True): "
                    "on-manifold demo discipline broken")
            belt = int(env.oe.env._raw.get("belt_heals", 0))
            pair = (np.asarray(obs, dtype=np.float32), a, masks, belt)
            obs2, w, term, trunc, info = env.step(a)
            if info.get("overridden"):
                dropped += 1              # steps rewritten by the fuse are dropped whole (v1 clause unchanged)
            elif "executed_action" not in info:
                raise RuntimeError("BC-v2 step is missing its executed_action receipt")
            elif info["executed_action"] is None:
                dropped_no_effect += 1
                dropped_no_effect_by_action[int(a)] += 1
            else:
                if int(info["executed_action"]) != int(a):
                    raise RuntimeError(
                        "BC-v2 executed_action disagrees with the teacher proposal: "
                        f"requested={a},executed={info['executed_action']}")
                X.append(pair[0]); Y.append(pair[1]); M.append(pair[2])
                groups.append(seed); belts.append(pair[3])
            farm_window_end = bool(info.get("farm_window_end", False))
            if farm_window_end:
                # Keep the explicit window-notification interface so a future stateless teacher can be audited per window;
                # it must not change the target action for the same visible state.
                teacher.begin_window()
            if term or trunc:
                # at episode end next_window returns None (never rolls a new episode -- demo pool discipline)
                obs = env.next_window()
            else:
                obs = obs2
        if (i + 1) % 16 == 0:
            print(f"  v2 collected {i+1}/{len(DEMO_SEEDS_V2)} episodes, {len(Y)} pairs"
                  f" (fuse drops {dropped}, "
                  f"no-effect drops {dropped_no_effect}"
                  f"{dict(sorted(dropped_no_effect_by_action.items()))})",
                  flush=True)
    if env.stats["episodes"] != len(DEMO_SEEDS_V2) or env.stats["reseeds"] != 0:
        raise RuntimeError(f"demo pool seed discipline broken: {env.stats}")
    print(f"v2 demos: {len(Y)} decision pairs, dropped fuse steps {dropped}, "
          f"dropped no-effect steps {dropped_no_effect}, "
          f"by action {dict(sorted(dropped_no_effect_by_action.items()))}, "
          f"class distribution {dict(sorted(Counter(Y).items()))}", flush=True)
    env.close()
    groups_array = np.asarray(groups, dtype=np.int64)
    labels = np.asarray(Y, dtype=np.int64)
    if not np.array_equal(np.unique(groups_array), np.asarray(DEMO_SEEDS_V2)):
        raise RuntimeError(
            "v2 demo set does not exactly cover the current fixed seeds "
            "2103000..2103383")
    if np.isin(labels, forbidden_actions_for_generation(
            TEACHER_GENERATION_V2)).any():
        raise RuntimeError("v2 demo set contains forbidden action 11 (11 is always masked and belongs to the manager; "
                           "12 is the real v2 preventive-drink label and may be collected)")
    masks_array = np.stack(M).astype(bool)
    belts_array = np.asarray(belts, dtype=np.int64)
    visible_targets = np.fromiter(
        (teacher_v2_preventive_trigger(
            row, mask, preventive_threshold)
         for row, mask in zip(X, masks_array)),
        dtype=np.bool_, count=len(labels))
    if not np.array_equal(labels == 12, visible_targets):
        mismatch = np.flatnonzero((labels == 12) != visible_targets)
        raise RuntimeError(
            "v2 whole-pool labels differ from the visible TeacherV2 predicate"
            f" (hp/m12/active-drink bit), first mismatched row={int(mismatch[0])}")
    return np.stack(X), labels, groups_array, masks_array, belts_array


def _save_demos_v2(out_dir: pathlib.Path, X, labels, groups, masks,
                   provenance: dict, *,
                   destination: pathlib.Path | None = None) -> pathlib.Path:
    """Write the v2/3 demos with their own provenance; the PASS report is the bundle commit marker."""
    n = len(labels)
    if masks.shape != (n, 15) or masks.dtype != np.bool_:
        raise RuntimeError(f"v2 masks shape/dtype invalid: {masks.shape}/{masks.dtype}"
                           " (must be (pairs, 15) bool)")
    if not masks[np.arange(n), labels].all():
        raise RuntimeError("v2 demos contain samples whose proposal is not in the mask"
                           " (a12 needs m[12]=True; on-manifold broken)")
    if masks[:, 11].any():
        raise RuntimeError("v2 demos m[11] must be always False (11 is always masked and belongs to the manager)")
    required_provenance = {
        "schema_version", "protocol_version", "implementation_sha256",
        "generator_sha256", "manager_npz_sha256", "teacher_generation",
        "preventive_threshold",
    }
    if not isinstance(provenance, dict) \
            or not required_provenance <= set(provenance):
        raise RuntimeError("v2 demos provenance missing fields: "
                           f"{sorted(required_provenance - set(provenance or {}))}")
    final = destination or (out_dir / "demos.npz")
    tmp = final.with_name(f".{final.stem}.{time.time_ns()}.tmp.npz")
    np.savez_compressed(
        tmp, X=X, Y=labels, episode_id=groups, masks=masks,
        schema_version=np.asarray(_BC_V2_DEMOS_SCHEMA_VERSION),
        protocol_version=np.asarray(provenance["protocol_version"],
                                    dtype=np.int64),
        implementation_sha256=np.asarray(provenance["implementation_sha256"]),
        generator_sha256=np.asarray(provenance["generator_sha256"]),
        manager_npz_sha256=np.asarray(provenance["manager_npz_sha256"]),
        teacher_generation=np.asarray(provenance["teacher_generation"],
                                      dtype=np.int64),
        preventive_threshold=np.asarray(provenance["preventive_threshold"],
                                        dtype=np.float64),
    )
    tmp.replace(final)
    return final


def _balanced_class_weights(labels, n_act: int = 15) -> np.ndarray:
    """Plan A (b) (approved 2026-07-19): class-balanced weights w_c = N/(K·n_c) (standard balancing formula).

    Class set = classes actually present (n_c > 0); K = number of present classes; N = total samples; absent classes get
    0.0 (a placeholder in the CE weight vector, never consumed by training). Pure integer-count arithmetic, deterministic.
    Applies only to the v2 primary training call; the v1 training path is never touched (train_bc defaults to None and v1
    calls never pass it).
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        raise RuntimeError("class-balanced weights: empty label set (plan A (b) premise broken)")
    counts = np.bincount(labels, minlength=n_act).astype(np.float64)
    present = counts > 0
    k = int(present.sum())
    weights = np.zeros(n_act, dtype=np.float64)
    weights[present] = float(labels.size) / (k * counts[present])
    return weights


def _wire_a12_teacher_boundary(
        model: PiHead, preventive_threshold: float) -> None:
    """Write the rare a12 discriminator into reserved neurons of the standard 64x64 policy MLP.

    Layer-1 neurons 61..63 encode hp >= 0.5, hp < preventive threshold, and no active drink yet this window; layer-2
    neuron 63 forms a soft AND. Belt availability is enforced exactly by m12 of the live action mask. This
    boundary is isomorphic to the TeacherV2 visible predicate; other actions do not read the reserved neurons, and a12 no longer
    reads the old miscalibrated all-class CE row. The export still has only the six native SB3 tensors.
    """
    if not isinstance(model, PiHead):
        raise RuntimeError("a12 calibration only accepts PiHead (SB3 64x64 isomorphic model)")
    if model.net[0].weight.shape[0] != 64 \
            or model.net[2].weight.shape != (64, 64) \
            or model.head.weight.shape != (15, 64):
        raise RuntimeError("a12 calibration: 64x64/15 policy head shape mismatch")
    if model.net[0].weight.shape[1] <= max(
            _A12_CALIBRATION_HP_FEATURE,
            _A12_CALIBRATION_DRINK_LATCH_FEATURE):
        raise RuntimeError("a12 calibration is missing the worker hp/active-drink visible bits")

    # Lower edge is closed and hp<0.5 is never live-m12, so a small outward
    # pad safely makes exact hp=0.5 a positive.  The upper edge is open:
    # its center must be the exact teacher threshold.  The old ``+0.002``
    # moved that center to 0.652 and placed the real 0.651163 negatives on
    # the positive side of the circuit.  A steeper exact-center transition
    # gives margin without changing the registered predicate.
    slope = 100.0
    lower_edge_pad = 0.002
    conjunction_slope = 8.0
    conjunction_floor = 2.0
    with torch.no_grad():
        w0, b0 = model.net[0].weight, model.net[0].bias
        w1, b1 = model.net[2].weight, model.net[2].bias
        wa, ba = model.head.weight, model.head.bias

        # train_bc(reserve_a12_path=True) must isolate these parameters from step 0;
        # otherwise zeroing them here would delete representations ordinary actions have already learned.
        reserved_zero = (
            int(torch.count_nonzero(w0[61:64])) == 0
            and int(torch.count_nonzero(b0[61:64])) == 0
            and int(torch.count_nonzero(w1[:, 61:64])) == 0
            and int(torch.count_nonzero(w1[63])) == 0
            and int(torch.count_nonzero(b1[63])) == 0
            and int(torch.count_nonzero(wa[:, 63])) == 0
        )
        if not reserved_zero:
            raise RuntimeError(
                "a12 calibration reserved pathway took part in ordinary CE; refusing destructive after-the-fact wiring")

        w0[61, _A12_CALIBRATION_HP_FEATURE] = slope
        b0[61] = -slope * (_PREVENTIVE_HP_LOW - lower_edge_pad)
        w0[62, _A12_CALIBRATION_HP_FEATURE] = -slope
        b0[62] = slope * float(preventive_threshold)
        w0[63, _A12_CALIBRATION_DRINK_LATCH_FEATURE] = slope
        # The not-drunk domain is at least 0, the drunk domain at most -1; a 0.05 margin separates them at saturation.
        b0[63] = slope * 0.05

        w1[63].zero_()
        b1[63] = -conjunction_slope * conjunction_floor
        w1[63, 61:64] = conjunction_slope

        wa[12].zero_()
        wa[12, 63] = 8.0
        ba[12].zero_()  # then fitted only from margin quantiles of training episodes


def _calibrate_a12_policy(model: PiHead, X, labels, groups, masks,
                          preventive_threshold: float) -> dict:
    """Calibrate a12 on nested-fit only and return the full fit record for the receipt.

    Neither validation nor final-heldout is read. The visible hp/per-window drink history boundary is fixed to
    the teacher predicate; the a12 bias is fitted only to a frozen 75% recall of fit positives, and the
    0.50 gate on independent validation/final leaves a sampling margin; the production probability gate
    also bounds soft leakage, and stricter fit FPR/high-HP lines screen candidates; tuning after peeking at
    any evaluation domain is forbidden.
    """
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    masks = np.asarray(masks)
    if X.ndim != 2 or X.shape[1] <= max(
            _A12_CALIBRATION_HP_FEATURE,
            _A12_CALIBRATION_DRINK_LATCH_FEATURE) \
            or labels.shape != (len(X),) or groups.shape != (len(X),) \
            or masks.shape != (len(X), 15) or masks.dtype != np.bool_:
        raise RuntimeError("a12 calibration input shape/dtype invalid")
    fit, validation, validation_episodes = (
        _split_fit_validation_by_episode(groups))
    _, final_heldout, final_heldout_episodes = split_by_episode(groups)
    fit_episode_count = int(np.unique(groups[fit]).size)
    true12 = labels[fit] == 12
    negative = ~true12
    legal12 = masks[fit, 12]
    if int(true12.sum()) < 2 or not bool(legal12[true12].all()):
        raise RuntimeError("a12 calibration fit split has too few positives, or positives are forbidden by the real mask")
    legal_negative = negative & legal12
    if int(legal_negative.sum()) < 3 * int(true12.sum()):
        raise RuntimeError("a12 calibration fit lacks real m12 hard negatives at a ratio of at least 3:1")

    target_count = int(math.ceil(
        _A12_CALIBRATION_TRAIN_RECALL_TARGET * int(true12.sum())))
    minimum_share = target_count / len(fit)
    if minimum_share > _A12_PREDICTED_SHARE_MAX:
        raise RuntimeError(
            "BC-v2 a12 fit gate arithmetically unreachable: "
            f"ceil({int(true12.sum())}×"
            f"{_A12_CALIBRATION_TRAIN_RECALL_TARGET})/{len(fit)}"
            f"={minimum_share:.8f} > predicted_share_max "
            f"{_A12_PREDICTED_SHARE_MAX}; refusing to hide it by lowering the safety gate or tuning the bias")
    _wire_a12_teacher_boundary(model, preventive_threshold)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X[fit]))
        mask_t = torch.from_numpy(masks[fit])
        other = torch.where(
            mask_t, logits, torch.full_like(logits, -1e8))
        other[:, 12] = -1e8
        margin = (logits[:, 12] - other.max(1).values).cpu().numpy()

    # Cross the margin of the frozen fit recall target as conservatively as possible; the bias is never set from an evaluation domain.
    positive_margin = np.sort(margin[true12])[::-1]
    cut = float(positive_margin[target_count - 1])
    bias = float(-cut + 1e-6)
    with torch.no_grad():
        model.head.bias[12] = bias
        # The parameter tensors are float32; the receipt must bind the value "actually deployed", not the Python float
        # from before assignment. Quantisation error for a large bias can reach 1e-6, enough for the producer's
        # own valid artifact to be rejected at the PPO entry.
        bias = float(model.head.bias[12].item())

    def deployed_fit_metrics() -> dict:
        # Must use the same six-tensor forward + mask + argmax as the PPO consumer. The old
        # ``margin+bias >= 0`` shortcut, under float32 addition rounding or a logit tie, would count
        # an action 9/12 tie as 12, while the real argmax picks the smaller index 9.
        behavior = bc_aux_behavior_metrics(
            export_sb3_sd(model), X[fit], labels[fit], groups[fit],
            masks[fit], heldout_only=False)
        return {
            "bias_12": float(model.head.bias[12].item()),
            "tp": int(behavior["tp"]),
            "fp": int(behavior["fp"]),
            "precision_12": float(behavior["precision_12"]),
            "recall_12": float(behavior["recall_12"]),
            "fpr_12": float(behavior["fpr_12"]),
            "predicted_share_12": float(
                behavior["predicted_share_12"]),
            "high_hp_false_drink_rate": float(
                behavior["high_hp_false_drink_rate"]),
            "legal_negative_probability_12_mean": float(
                behavior["legal_negative_probability_12_mean"]),
            "legal_negative_probability_12_max": float(
                behavior["legal_negative_probability_12_max"]),
            "a13_spillover": float(behavior["a13_spillover"]),
            "_high_hp_non_a12": int(behavior["high_hp_non_a12"]),
        }

    selected = deployed_fit_metrics()
    # The initial analytic bias crosses the target_count-th margin in theory; the final float32 forward
    # can still create a tie through rounding. Push up ULP by ULP on fit only until the real argmax reaches the target,
    # and recompute every subsequent FPR/share gate on the same final six tensors.
    for _ in range(64):
        if (selected["recall_12"]
                >= _A12_CALIBRATION_TRAIN_RECALL_TARGET):
            break
        with torch.no_grad():
            current = model.head.bias[12]
            current.copy_(torch.nextafter(
                current, torch.full_like(current, float("inf"))))
        selected = deployed_fit_metrics()
    else:
        raise RuntimeError(
            "BC-v2 a12 fit calibration still cannot reach the target recall within 64 float32 ULPs")

    bias = selected["bias_12"]
    tp = selected["tp"]
    fp = selected["fp"]
    precision = selected["precision_12"]
    recall = selected["recall_12"]
    fpr = selected["fpr_12"]
    share = selected["predicted_share_12"]
    high_hp_rate = selected["high_hp_false_drink_rate"]
    legal_negative_probability_mean = selected[
        "legal_negative_probability_12_mean"]
    legal_negative_probability_max = selected[
        "legal_negative_probability_12_max"]
    a13_spillover = selected["a13_spillover"]
    high_hp_count = selected.pop("_high_hp_non_a12")
    if not (
            recall >= _A12_CALIBRATION_TRAIN_RECALL_TARGET
            and precision >= max(0.10, _A12_PRECISION_MIN)
            and fpr <= _A12_FPR_MAX * 0.5
            and _A12_PREDICTED_SHARE_MIN <= share
            <= _A12_PREDICTED_SHARE_MAX
            and high_hp_count > 0
            and high_hp_rate <= _A12_HIGH_HP_FALSE_DRINK_MAX * 0.5
            and legal_negative_probability_mean
            <= _A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX
            and legal_negative_probability_max
            <= _A12_LEGAL_NEGATIVE_PROBABILITY_MAX
            and a13_spillover <= _A13_SPILLOVER_MAX):
        compact = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in selected.items() if k != "bias_12"
        }
        raise RuntimeError(
            "BC-v2 a12 fit calibration unreachable; refusing to let a miscalibrated classifier into validation: "
            f"{compact}")

    return {
        "schema_version": _A12_CALIBRATION_SCHEMA_VERSION,
        "fit_scope": "nested-fit-episodes-only",
        "fit_pairs": int(len(fit)),
        "fit_episodes": fit_episode_count,
        "validation_pairs_excluded": int(len(validation)),
        "validation_episodes_excluded": int(len(validation_episodes)),
        "final_heldout_pairs_excluded": int(len(final_heldout)),
        "final_heldout_episodes_excluded": int(
            len(final_heldout_episodes)),
        "hp_low": _PREVENTIVE_HP_LOW,
        "hp_high": float(preventive_threshold),
        "hp_feature": _A12_CALIBRATION_HP_FEATURE,
        "drink_latch_feature": _A12_CALIBRATION_DRINK_LATCH_FEATURE,
        "predicate": _A12_CALIBRATION_PREDICATE,
        "bias_12": selected["bias_12"],
        "target_recall_12": _A12_CALIBRATION_TRAIN_RECALL_TARGET,
        "fit_metrics": {
            key: (int(value) if key in {"tp", "fp"} else float(value))
            for key, value in selected.items()
            if key != "bias_12"
        },
    }


def _n12_readings(labels, groups, belts) -> dict:
    """n₁₂ gate reading: all a12 executed states inside the visible preventive band (overridden steps dropped)."""
    n = int(len(labels))
    a12 = labels == 12
    a13 = labels == 13
    n12 = int(a12.sum())
    by_episode = {str(int(seed)): int((a12 & (groups == seed)).sum())
                  for seed in np.unique(groups[a12])}   # zero-count episodes omitted; the rest are always 0
    return {
        "n12": n12,
        "n12_by_episode": by_episode,
        "class_share_12": round(n12 / n, 6) if n else 0.0,
        "class_share_13": round(int(a13.sum()) / n, 6) if n else 0.0,
        "belt_economy": {
            # Discretionary note: the design did not pin the fields of the "belt economy reading"; three are registered -- belt mean
            # in real-a12 states / belt mean over the whole set / a13 (pick up supplies) log count; zero coverage records 0.0 rather than vanishing.
            "belt_mean_at_a12": (round(float(belts[a12].mean()), 4)
                                 if n12 > 0 else 0.0),
            "belt_mean_overall": round(float(belts.mean()), 4) if n else 0.0,
            "a13_pairs": int(a13.sum()),
        },
    }


def _a12_behavior_from_model(model, X, labels, groups, masks) -> dict:
    """Evaluate the a12 behaviour surface on fixed held-out whole episodes with the real v2 masks."""
    model.eval()
    return bc_aux_behavior_metrics(
        export_sb3_sd(model), X, labels, groups, masks,
        heldout_only=True)


def _a12_behavior_from_indices(
        model, X, labels, groups, masks, indices) -> dict:
    """Evaluate a12 on an explicit selection row set; never switch implicitly to final-heldout."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("a12 validation indices must be 1-D non-empty indices")
    model.eval()
    return bc_aux_behavior_metrics(
        export_sb3_sd(model),
        np.asarray(X)[indices], np.asarray(labels)[indices],
        np.asarray(groups)[indices], np.asarray(masks)[indices],
        heldout_only=False)


def _bc_v2_a12_gate_passes(metrics: dict) -> tuple[bool, dict]:
    """BC-v2 dedicated a12 gate = generic behaviour safety gate + the registered 0.5 recall."""
    gate = bc_aux_behavior_gate(
        metrics, require_teacher_recall=False)
    recall = metrics.get("recall_12") if isinstance(metrics, dict) else None
    passes = (
        gate["verdict"] == "PASS"
        and isinstance(recall, (int, float))
        and not isinstance(recall, bool)
        and math.isfinite(float(recall))
        and float(recall) >= _RECALL12_GATE_MIN)
    return passes, gate


def _recall12_from_model(model, X, labels, groups, masks) -> tuple[float, int]:
    """Recall gate (pinned by E2/D7): denominator = real-a12 states in held-out under the current split
    (all on-manifold executed states produced by the visible hp/belt predicate);
    measure = held-out argmax hits; fail-closed, following the v1 class-recall precedent
    (train_bc zero coverage records 0.0 rather than vanishing). Returns (recall_12, denominator count)."""
    _, ho, _ = split_by_episode(groups)
    pred = _masked_bc_predictions(model, X[ho], masks[ho])
    true12 = labels[ho] == 12
    denominator = int(true12.sum())
    recall = (float((pred[true12] == 12).mean())
              if denominator else 0.0)
    return round(recall, 4), denominator


def artifact_provenance_v2(
        preventive_threshold: float,
        manager_npz: str | pathlib.Path = NPZ) -> dict:
    manager_path = pathlib.Path(manager_npz)
    try:
        manager_payload = manager_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"BC-v2 manager npz missing/unreadable: {manager_path}") from exc
    return {
        "schema_version": _BC_V2_REPORT_SCHEMA_VERSION,
        "teacher_generation": TEACHER_GENERATION_V2,
        "preventive_threshold": preventive_threshold,
        "protocol_version": PROTOCOL_VERSION,
        "implementation_sha256": _implementation_bundle_sha256(),
        "generator_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()).hexdigest(),
        "manager_npz_sha256": hashlib.sha256(manager_payload).hexdigest(),
    }


def begin_output_attempt_v2(provenance: dict | None = None):
    """Clear old v2 artifacts to prevent mistaken consumption (same as v1; separate directory avoids the _previous archive exclusivity)."""
    provenance = (
        artifact_provenance_v2(_PREVENTIVE_THRESHOLD_MAIN, NPZ)
        if provenance is None else provenance)
    OUT_V2.mkdir(parents=True, exist_ok=True)
    _assert_final_holdout_unused(
        OUT_V2, TEACHER_GENERATION_V2, DEMO_SEEDS_V2)
    _reject_same_generator_terminal(
        OUT_V2, _BC_V2_REPORT_NAME, provenance)
    old = [OUT_V2 / name
           for name in ("policy_sd.pt", _BC_V2_REPORT_NAME, "demos.npz")
           if (OUT_V2 / name).exists()]
    if old:
        archive = OUT_V2 / "_previous" / str(time.time_ns())
        archive.mkdir(parents=True)
        for path in old:
            path.replace(archive / path.name)
    # If the previous process died before committing the report, pending data must never survive into the next attempt
    # as seemingly usable input; the canonical three-file set was already archived above.
    for residue in OUT_V2.glob(".*.tmp.npz"):
        residue.unlink(missing_ok=True)
    (OUT_V2 / "demos.pending.npz").unlink(missing_ok=True)
    (OUT_V2 / _BC_V2_REPORT_NAME).write_text(json.dumps(
        {"data_gate": "RUNNING", "teacher_generation": TEACHER_GENERATION_V2}))


def write_report_v2(record):
    tmp = OUT_V2 / "bc_report_v2.tmp.json"
    tmp.write_text(json.dumps(record, ensure_ascii=False))
    tmp.replace(OUT_V2 / _BC_V2_REPORT_NAME)


def _validate_bc_v2_report(p: pathlib.Path,
                           expected_implementation_sha256: str | None = None,
                           *, policy_payload: bytes | None = None,
                           report_payload: bytes | None = None,
                           demos_payload: bytes | None = None,
                           expected_manager_sha256: str | None = None) -> dict:
    """Dedicated BC-v2 validator (E2 schema isolation, shaped after the v1 validator).

    Exact key-set assertion + separate schema identifier: always blows up on v1 artifacts (key sets differ);
    the v1 validator (train_ppo._validate_bc_report) also always blows up on v2 artifacts (exact key set
    + non-int schema_version, doubly incompatible) -- the v1/v2 validators are mutually exclusive.
    Measured demos-bytes assertion (added in rev4 addendum 12.2 (4)): sha256 of the demos.npz bytes ==
    receipt demos_sha256, mirroring the real-bytes assertion on the policy side.
    """
    from eval_contract import EvalContractError, strict_json_loads

    def _fail(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def _plain_int(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    def _plain_num(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    report = p.with_name(_BC_V2_REPORT_NAME)
    try:
        frozen_report = (report.read_bytes() if report_payload is None
                         else report_payload)
        rec = strict_json_loads(frozen_report)
    except (OSError, EvalContractError) as exc:
        raise ValueError(f"BC-v2 receipt missing/unreadable: {report}") from exc
    _fail(isinstance(rec, dict), f"BC-v2 receipt must be a JSON object: {report}")
    _fail(set(rec) == set(_BC_V2_PASS_KEYS),
          f"BC-v2 receipt fields/schema mismatch (v1/v2 artifacts are mutually exclusive): {report}")
    _fail(rec["schema_version"] == _BC_V2_REPORT_SCHEMA_VERSION,
          f"BC-v2 receipt schema identifier mismatch: {rec['schema_version']!r}")
    _fail(rec["teacher_generation"] == TEACHER_GENERATION_V2,
          f"BC-v2 receipt teacher_generation must be 2: {rec['teacher_generation']!r}")
    _fail(rec["preventive_threshold"] in _REGISTERED_PREVENTIVE_THRESHOLDS,
          f"BC-v2 receipt preventive threshold not registered: {rec['preventive_threshold']!r}")
    _fail(rec["data_gate"] == "PASS",
          f"refusing to accept a BC-v2 artifact that did not pass the data_gate: {rec['data_gate']!r}")
    _fail(_plain_num(rec["held_out_top1"])
          and math.isfinite(float(rec["held_out_top1"]))
          and 0.95 <= float(rec["held_out_top1"]) <= 1.0,
          f"BC-v2 held-out top1 gate not met: {rec['held_out_top1']!r}")
    class_recalls = rec["class_recalls"]
    _fail(isinstance(class_recalls, dict) and class_recalls,
          "BC-v2 class_recalls must be a non-empty object")
    for raw_action, raw_recall in class_recalls.items():
        try:
            action = int(raw_action)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"BC-v2 class_recalls keys must be action numbers: {raw_action!r}"
            ) from exc
        _fail(str(action) == str(raw_action) and 0 <= action < 15,
              f"BC-v2 class_recalls key invalid: {raw_action!r}")
        _fail(_plain_num(raw_recall)
              and math.isfinite(float(raw_recall))
              and 0.0 <= float(raw_recall) <= 1.0,
              f"BC-v2 class_recalls[{raw_action!r}] invalid: {raw_recall!r}")
        if action != 12:
            _fail(float(raw_recall) >= 0.85,
                  "BC-v2 non-a12 per-class recall gate not met:"
                  f" action={action}, recall={raw_recall!r}")
    _fail(_plain_int(rec["n12"]) and rec["n12"] >= _N12_GATE_MIN,
          f"BC-v2 n₁₂ gate not met (>={_N12_GATE_MIN}): {rec['n12']!r}")
    _fail(_plain_num(rec["recall_12"])
          and _RECALL12_GATE_MIN <= float(rec["recall_12"]) <= 1.0,
          f"BC-v2 recall gate not met (>={_RECALL12_GATE_MIN}): {rec['recall_12']!r}")
    behavior = rec["a12_behavior"]
    _fail(isinstance(behavior, dict)
          and set(behavior) == set(_BC_AUX_BEHAVIOR_METRIC_KEYS),
          f"BC-v2 a12_behavior fields/schema mismatch: {behavior!r}")
    _fail(behavior["pairs"] == rec["held_out_pairs"]
          and behavior["true_a12"] + behavior["all_non_a12"]
          == behavior["pairs"]
          and behavior["non_a12"] <= behavior["all_non_a12"]
          and behavior["tp"] + behavior["fn"] == behavior["true_a12"]
          and behavior["fp"] + behavior["tn"] == behavior["non_a12"]
          and behavior["tp"] + behavior["fp"]
          == behavior["predicted_a12"],
          "BC-v2 a12_behavior counts/denominators do not close")
    recomputed_gate = bc_aux_behavior_gate(
        behavior, require_teacher_recall=False)
    _fail(rec["a12_behavior_gate"] == recomputed_gate,
          "BC-v2 a12_behavior_gate disagrees with the behaviour readings/registered thresholds")
    _fail(recomputed_gate["verdict"] == "PASS",
          "BC-v2 a12 behaviour hard gate not passed: "
          f"{recomputed_gate['reasons']}")
    _fail(float(behavior["recall_12"]) >= _RECALL12_GATE_MIN,
          f"BC-v2 a12 dedicated recall gate not met (>={_RECALL12_GATE_MIN}):"
          f" {behavior['recall_12']!r}")
    _fail(isinstance(rec["held_out_episodes"], list)
          and _plain_int(rec["collection_episodes"])
          and rec["collection_episodes"]
          >= max(1, len(rec["held_out_episodes"])),
          f"BC-v2 receipt collection_episodes invalid: "
          f"{rec['collection_episodes']!r}")
    calibration = rec["a12_calibration"]
    calibration_keys = {
        "schema_version", "fit_scope", "fit_pairs", "fit_episodes",
        "validation_pairs_excluded", "validation_episodes_excluded",
        "final_heldout_pairs_excluded",
        "final_heldout_episodes_excluded", "hp_low", "hp_high",
        "hp_feature", "drink_latch_feature", "predicate", "bias_12",
        "target_recall_12", "fit_metrics",
    }
    _fail(isinstance(calibration, dict)
          and set(calibration) == calibration_keys,
          "BC-v2 a12_calibration fields/schema mismatch")
    _fail(calibration["schema_version"] == _A12_CALIBRATION_SCHEMA_VERSION
          and calibration["fit_scope"] == "nested-fit-episodes-only",
          "BC-v2 a12_calibration identity/fit domain mismatch")
    _fail(_plain_int(calibration["fit_pairs"])
          and _plain_int(calibration["validation_pairs_excluded"])
          and _plain_int(calibration["final_heldout_pairs_excluded"])
          and calibration["fit_pairs"] > 0
          and calibration["validation_pairs_excluded"] > 0
          and calibration["final_heldout_pairs_excluded"] > 0
          and calibration["fit_pairs"]
          + calibration["validation_pairs_excluded"]
          + calibration["final_heldout_pairs_excluded"]
          == rec["pairs"],
          "BC-v2 a12_calibration three-domain pairs split does not close")
    _fail(calibration["final_heldout_pairs_excluded"]
          == rec["held_out_pairs"],
          "BC-v2 a12_calibration final-heldout pairs disagree with the receipt")
    _fail(_plain_int(calibration["fit_episodes"])
          and _plain_int(calibration["validation_episodes_excluded"])
          and _plain_int(calibration["final_heldout_episodes_excluded"])
          and calibration["fit_episodes"] > 0
          and calibration["validation_episodes_excluded"] > 0
          and calibration["final_heldout_episodes_excluded"] > 0
          and calibration["fit_episodes"]
          + calibration["validation_episodes_excluded"]
          + calibration["final_heldout_episodes_excluded"]
          == rec["collection_episodes"],
          "BC-v2 a12_calibration three-domain episode split does not close")
    _fail(calibration["final_heldout_episodes_excluded"]
          == len(rec["held_out_episodes"]),
          "BC-v2 a12_calibration final-heldout episodes disagree with the receipt")
    _fail(calibration["hp_low"] == _PREVENTIVE_HP_LOW
          and calibration["hp_high"] == rec["preventive_threshold"]
          and calibration["hp_feature"] == _A12_CALIBRATION_HP_FEATURE
          and calibration["drink_latch_feature"]
          == _A12_CALIBRATION_DRINK_LATCH_FEATURE
          and calibration["predicate"] == _A12_CALIBRATION_PREDICATE
          and calibration["target_recall_12"]
          == _A12_CALIBRATION_TRAIN_RECALL_TARGET
          and _plain_num(calibration["bias_12"])
          and math.isfinite(float(calibration["bias_12"])),
          "BC-v2 a12_calibration parameters not registered")
    fit_metrics = calibration["fit_metrics"]
    metric_keys = {
        "tp", "fp", "precision_12", "recall_12", "fpr_12",
        "predicted_share_12", "high_hp_false_drink_rate",
        "legal_negative_probability_12_mean",
        "legal_negative_probability_12_max",
        "a13_spillover",
    }
    _fail(isinstance(fit_metrics, dict)
          and set(fit_metrics) == metric_keys
          and _plain_int(fit_metrics["tp"])
          and _plain_int(fit_metrics["fp"])
          and 0 <= fit_metrics["tp"] <= calibration["fit_pairs"]
          and 0 <= fit_metrics["fp"] <= calibration["fit_pairs"]
          and all(
              _plain_num(fit_metrics[key])
              and math.isfinite(float(fit_metrics[key]))
              and 0.0 <= float(fit_metrics[key]) <= 1.0
              for key in metric_keys - {"tp", "fp"})
          and math.isclose(
              float(fit_metrics["precision_12"]),
              fit_metrics["tp"] / max(
                  1, fit_metrics["tp"] + fit_metrics["fp"]),
              rel_tol=0.0, abs_tol=1e-15)
          and math.isclose(
              float(fit_metrics["predicted_share_12"]),
              (fit_metrics["tp"] + fit_metrics["fp"])
              / calibration["fit_pairs"],
              rel_tol=0.0, abs_tol=1e-15)
          and fit_metrics["recall_12"]
          >= _A12_CALIBRATION_TRAIN_RECALL_TARGET
          and fit_metrics["precision_12"] >= max(0.10, _A12_PRECISION_MIN)
          and fit_metrics["fpr_12"] <= _A12_FPR_MAX * 0.5
          and _A12_PREDICTED_SHARE_MIN
          <= fit_metrics["predicted_share_12"]
          <= _A12_PREDICTED_SHARE_MAX
          and fit_metrics["high_hp_false_drink_rate"]
          <= _A12_HIGH_HP_FALSE_DRINK_MAX * 0.5
          and fit_metrics["legal_negative_probability_12_mean"]
          <= _A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX
          and fit_metrics["legal_negative_probability_12_max"]
          <= _A12_LEGAL_NEGATIVE_PROBABILITY_MAX
          and fit_metrics["a13_spillover"] <= _A13_SPILLOVER_MAX,
          "BC-v2 a12_calibration fit safety readings not met")
    # The old recall field of the receipt and the new behaviour surface must share a source; one PASS and one FAIL is forbidden.
    _fail(math.isclose(float(rec["recall_12"]),
                       round(float(behavior["recall_12"]), 4),
                       rel_tol=0, abs_tol=1e-12)
          and rec["recall_12_denominator"] == behavior["true_a12"],
          "BC-v2 old recall field disagrees with a12_behavior")
    if "12" in class_recalls:
        _fail(math.isclose(
                  float(class_recalls["12"]),
                  float(behavior["recall_12"]),
                  rel_tol=0.0, abs_tol=1e-15),
              "BC-v2 class_recalls[12] disagrees with the dedicated behaviour recall")
    # Plan A new receipt field assertions (approved 2026-07-19)
    _fail(isinstance(rec["held_out_episodes"], list)
          and _plain_int(rec["collection_episodes"])
          and rec["collection_episodes"] >= max(1, len(rec["held_out_episodes"])),
          f"BC-v2 receipt collection_episodes invalid: {rec['collection_episodes']!r}")
    class_weights = rec["class_weights"]
    _fail(isinstance(class_weights, dict) and len(class_weights) > 0,
          f"BC-v2 receipt class_weights must be a non-empty object: {report}")
    for raw_class, raw_weight in class_weights.items():
        try:
            class_id = int(raw_class)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"BC-v2 receipt class_weights keys must be action numbers: {raw_class!r}"
            ) from exc
        _fail(str(class_id) == str(raw_class) and 0 <= class_id < 15,
              f"BC-v2 receipt class_weights key invalid: {raw_class!r}")
        _fail(_plain_num(raw_weight) and raw_weight > 0,
              f"BC-v2 receipt class_weights[{raw_class!r}] invalid: {raw_weight!r}")
    _fail(rec["protocol_version"] == PROTOCOL_VERSION,
          f"BC-v2 receipt protocol outdated: {rec['protocol_version']!r}")
    expected_impl = (expected_implementation_sha256
                     if expected_implementation_sha256 is not None
                     else _implementation_bundle_sha256())
    _fail(rec["implementation_sha256"] == expected_impl,
          "BC-v2 receipt implementation/engine/game-content identity disagrees with the current runtime")
    generator_sha = hashlib.sha256(
        pathlib.Path(__file__).read_bytes()).hexdigest()
    _fail(rec["generator_sha256"] == generator_sha,
          "BC-v2 receipt generator has drifted: train/bc_worker.py")
    _validate_bc_final_holdout_marker(
        p.parent, TEACHER_GENERATION_V2, DEMO_SEEDS_V2, rec)
    manager_sha = rec["manager_npz_sha256"]
    _fail(isinstance(manager_sha, str) and len(manager_sha) == 64
          and all(ch in "0123456789abcdef" for ch in manager_sha),
          f"BC-v2 receipt manager_npz_sha256 invalid: {manager_sha!r}")
    if expected_manager_sha256 is not None:
        _fail(isinstance(expected_manager_sha256, str)
              and len(expected_manager_sha256) == 64
              and all(ch in "0123456789abcdef"
                      for ch in expected_manager_sha256),
              "BC-v2 expected manager SHA invalid")
        _fail(manager_sha == expected_manager_sha256,
              "BC-v2 receipt manager identity disagrees with the training manager: "
              f"{manager_sha} != {expected_manager_sha256}")
    expected_sha = rec["policy_sha256"]
    _fail(isinstance(expected_sha, str) and len(expected_sha) == 64,
          f"BC-v2 receipt is missing its policy_sha256 binding: {report}")
    try:
        frozen_policy = (p.read_bytes() if policy_payload is None
                         else policy_payload)
    except OSError as exc:
        raise ValueError(f"BC-v2 weights missing/unreadable: {p}") from exc
    _fail(hashlib.sha256(frozen_policy).hexdigest() == expected_sha,
          "BC-v2 weights SHA does not match the receipt")
    _fail(isinstance(rec["demos_sha256"], str) and len(rec["demos_sha256"]) == 64,
          f"BC-v2 receipt is missing its demos_sha256 binding: {report}")
    demos = p.with_name("demos.npz")
    try:
        frozen_demos = (demos.read_bytes() if demos_payload is None
                        else demos_payload)
    except OSError as exc:
        raise ValueError(f"BC-v2 demos missing/unreadable: {demos}") from exc
    _fail(hashlib.sha256(frozen_demos).hexdigest() == rec["demos_sha256"],
          "BC-v2 demos SHA does not match the receipt")
    return rec


def main_v2(preventive_threshold: float = _PREVENTIVE_THRESHOLD_MAIN,
            manager_npz: str | pathlib.Path = NPZ):
    if preventive_threshold not in _REGISTERED_PREVENTIVE_THRESHOLDS:
        raise ValueError(f"preventive threshold {preventive_threshold!r} not registered"
                         " (rev13 only the 0.65 main case; 0.70 has no independent fresh pool)")
    manager_path = pathlib.Path(manager_npz)
    provenance = artifact_provenance_v2(
        preventive_threshold, manager_path)
    begin_output_attempt_v2(provenance)
    # collect_v2 validates and prints whole-pool labels/masks.  The stable
    # O_EXCL registry must therefore be committed before the first reset, not
    # merely before final scoring.
    pool_marker = _mark_final_holdout_started(
        OUT_V2, TEACHER_GENERATION_V2, DEMO_SEEDS_V2, provenance)
    pool_evidence = {
        "final_pool_sha256": pool_marker["pool_sha256"],
        "final_holdout_marker_sha256": pool_marker["marker_sha256"],
    }
    write_report_v2({
        "data_gate": "FAIL",
        "failure_stage": "pool_collection_started",
        "final_heldout_consumed": True,
        **pool_evidence,
        **provenance,
    })
    X, labels, groups, masks, belts = collect_v2(
        preventive_threshold, manager_path,
        manager_sha256=provenance["manager_npz_sha256"])
    post_drink_coverage = _bc_v2_post_drink_coverage(
        X, labels, groups, masks, scopes=("fit", "validation"))
    print(
        "v2 visible post-drink hard-negative pre-selection domain coverage "
        f"{post_drink_coverage}",
        flush=True,
    )
    pending_demos = OUT_V2 / "demos.pending.npz"
    _save_demos_v2(
        OUT_V2, X, labels, groups, masks, provenance,
        destination=pending_demos)
    tr, ho, holdout_episodes = split_by_episode(groups)
    fit, _, _ = _split_fit_validation_by_episode(groups)
    # Plan A (b) (approved 2026-07-19): v2 primary training is class-balanced weighted CE -- weights derived
    # deterministically from label counts of the training split (w_c = N/(K·n_c), class set = classes actually present); the v1 primary
    # training call train_bc(X, Y, groups) passes no weights, untouched. Class weights read fit only, so
    # validation/final-held-out labels cannot shape the candidate in reverse.
    class_weights_v2 = _balanced_class_weights(labels[fit])
    # train_bc sends only fit/validation into the gradient or candidate readings; whole-pool collection itself is already
    # permanently registered by the pre-collection marker.
    model, top1, recalls = train_bc(
        X, labels, groups, class_weights=class_weights_v2,
        masks=masks, reserve_a12_path=True)
    retrained = False
    if not _bc_v2_quality_gate_passes(top1, recalls):
        # The same single retry as v1, triggered only by the v1-surface quality gate (top1/>=300-class recall) --
        # n₁₂/recall_12 never trigger class-weighted retraining: class weighting as an N12 remedy was struck from the plan
        # and would need a separate approval (PREREG-v33-content-case D2-5/D5 P-N12/D7).
        print(f"v2 first training missed validation targets (top1 {top1:.3f} recall {recalls})"
              " -> class-weighted retraining (the single retry; n₁₂/recall_12 never trigger this retry)",
              flush=True)
        counts = np.bincount(labels[fit], minlength=15).astype(np.float64)
        weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
        weights = weights / weights[weights > 0].mean()
        model, top1, recalls = train_bc(
            X, labels, groups, class_weights=weights,
            epochs=_BC_WEIGHTED_RETRY_EPOCHS, masks=masks,
            reserve_a12_path=True)
        retrained = True
    if not _bc_v2_quality_gate_passes(top1, recalls):
        pending_demos.unlink(missing_ok=True)
        write_report_v2({
            "data_gate": "FAIL",
            "failure_stage": "candidate_selection",
            "validation_top1": float(top1),
            "validation_class_recalls": recalls,
            "class_weighted_retry": retrained,
            "final_heldout_consumed": True,
            **pool_evidence,
            **provenance,
        })
        raise RuntimeError(
            f"BC-v2 candidate validation FAIL(top1={top1:.3f}, "
            f"recalls={recalls}); final held-out did not enter the candidate metrics, "
            "but the whole pool was consumed once; refusing to release/retry on the same pool")
    # Class-balanced CE only improves recall on the 0.04% rare class, with no false-positive constraint. Before the final export,
    # calibrate the in-network a12 pathway with training episodes + real masks, then recompute held-out top1/recalls
    # on the final six tensors; reusing pre-calibration readings is forbidden.
    try:
        a12_calibration = _calibrate_a12_policy(
            model, X, labels, groups, masks, preventive_threshold)
    except Exception as exc:
        pending_demos.unlink(missing_ok=True)
        write_report_v2({
            "data_gate": "FAIL",
            "failure_stage": "training_calibration",
            "validation_top1": float(top1),
            "validation_class_recalls": recalls,
            "class_weighted_retry": retrained,
            "error": f"{type(exc).__name__}: {exc}",
            "final_heldout_consumed": True,
            **pool_evidence,
            **provenance,
        })
        raise
    # Calibration directly changes the final exported action head; it is not read-only post-processing. The final candidate must
    # pass the selection gate again with the same nested validation and the real collection masks; if it fails,
    # final-heldout still stays unread.
    outer_train, _, _ = split_by_episode(groups)
    _, validation, _ = _split_fit_validation_by_episode(groups)
    validation_top1, validation_recalls = _score_bc_model_indices(
        model, X, labels, validation,
        eligibility_indices=outer_train, masks=masks)
    validation_a12 = _a12_behavior_from_indices(
        model, X, labels, groups, masks, validation)
    validation_a12_passes, validation_a12_gate = (
        _bc_v2_a12_gate_passes(validation_a12))
    if (not _bc_v2_quality_gate_passes(
            validation_top1, validation_recalls)
            or not validation_a12_passes):
        pending_demos.unlink(missing_ok=True)
        write_report_v2({
            "data_gate": "FAIL",
            "failure_stage": "post_calibration_selection",
            "validation_top1": float(validation_top1),
            "validation_class_recalls": validation_recalls,
            "validation_a12_behavior": validation_a12,
            "validation_a12_behavior_gate": validation_a12_gate,
            "class_weighted_retry": retrained,
            "final_heldout_consumed": True,
            **pool_evidence,
            **provenance,
        })
        raise RuntimeError(
            "BC-v2 post-calibration validation FAIL"
            f"(top1={validation_top1:.3f}, recalls={validation_recalls});"
            "final held-out did not enter the candidate metrics, but the whole pool was consumed once; "
            "refusing to release/retry on the same pool")
    # Only from this point does the final split enter the candidate model's scoring/behaviour statistics path.
    final_post_drink_coverage = _bc_v2_post_drink_coverage(
        X, labels, groups, masks, scopes=("final",))
    print(
        "v2 visible post-drink hard-negative final domain coverage "
        f"{final_post_drink_coverage}",
        flush=True,
    )
    top1, recalls = _score_bc_model(
        model, X, labels, groups, masks=masks)
    a12_behavior = _a12_behavior_from_model(
        model, X, labels, groups, masks)
    a12_behavior_gate = bc_aux_behavior_gate(
        a12_behavior, require_teacher_recall=False)
    a12_behavior_passes, _ = _bc_v2_a12_gate_passes(a12_behavior)
    recall_12 = round(float(a12_behavior["recall_12"]), 4)
    recall_12_denominator = int(a12_behavior["true_a12"])
    readings = _n12_readings(labels, groups, belts)
    ok = (_bc_v2_quality_gate_passes(top1, recalls)
          and readings["n12"] >= _N12_GATE_MIN
          and a12_behavior_passes)
    report = {
        "pairs": len(labels), "held_out_top1": float(top1),
        "held_out_pairs": int(len(ho)),
        "held_out_episodes": [int(x) for x in sorted(holdout_episodes)],
        "class_recalls": recalls, "class_weighted_retry": retrained,
        "recall_12": recall_12,
        "recall_12_denominator": recall_12_denominator,
        "recall_12_gate_min": _RECALL12_GATE_MIN,
        "a12_behavior": a12_behavior,
        "a12_behavior_gate": a12_behavior_gate,
        "a12_calibration": a12_calibration,
        "n12_gate_min": _N12_GATE_MIN,
        # Plan A new receipt fields (approved 2026-07-19): measured collection episodes (the x3 discipline is already
        # pinned by a collect_v2 assertion) + per-class summary of the primary-training class-balanced weights (classes actually present only).
        "collection_episodes": int(np.unique(groups).size),
        "class_weights": {str(int(c)): round(float(class_weights_v2[c]), 6)
                          for c in np.flatnonzero(class_weights_v2 > 0)},
        **pool_evidence,
        **readings,
        "data_gate": "PASS" if ok else "FAIL", **provenance}
    if not ok:
        pending_demos.unlink(missing_ok=True)
        write_report_v2(report)
        raise RuntimeError(
            f"BC-v2 data gate FAIL (top1={top1:.3f}, n12={readings['n12']}, "
            f"recall_12={recall_12},"
            f" behavior={a12_behavior_gate['reasons']});"
            "refusing to overwrite policy_sd.pt. The old 0.70 OC shared the final pool with the main case, "
            "and rev13 disabled it; if n₁₂ is insufficient, register an independent fresh pool first; "
            "never re-collect on the same pool")
    policy_tmp = OUT_V2 / "policy_sd.tmp.pt"
    if artifact_provenance_v2(
            preventive_threshold, manager_path) != provenance:
        raise RuntimeError("implementation/engine/content/manager drifted while BC-v2 was running")
    torch.save(export_sb3_sd(model), policy_tmp)
    policy_tmp.replace(OUT_V2 / "policy_sd.pt")
    report["policy_sha256"] = hashlib.sha256(
        (OUT_V2 / "policy_sd.pt").read_bytes()).hexdigest()
    report["demos_sha256"] = hashlib.sha256(
        pending_demos.read_bytes()).hexdigest()
    # The report is the bundle's final commit marker: even if policy/demos were published and the process then crashed,
    # the sibling is still RUNNING and training consumers fail closed. Only after the bytes of all three files and the receipt
    # hashes are in place is the PASS report atomically replaced.
    pending_demos.replace(OUT_V2 / "demos.npz")
    write_report_v2(report)
    print(f"v2 held-out top-1 {top1:.3f} n12 {readings['n12']} "
          f"recall_12 {recall_12} retry={retrained} -> data gate PASS; "
          f"saved {OUT_V2}/policy_sd.pt", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BC worker collection/training (default: v1 surface unchanged; --v2 is the E2 B1′ path)")
    parser.add_argument("--v2", action="store_true",
                        help="teacher v2 collection + training -> runs/bc-worker-v2/"
                             " (PREREG-v33-content-case E2)")
    parser.add_argument("--preventive-threshold", type=float, default=None,
                        choices=_REGISTERED_PREVENTIVE_THRESHOLDS,
                        help="v2 preventive threshold: rev13 only allows the 0.65 main case"
                             " (only with --v2)")
    parser.add_argument(
        "--manager-npz", default=None,
        help="manager policy.npz used for BC-v2 collection; defaults to the canonical v22 H. "
             "The SHA of the selected manager bytes is written to demos/report, and training consumers must reconcile it with"
             " the actual --manager-npz (only with --v2)")
    cli = parser.parse_args()
    if cli.v2:
        threshold = (cli.preventive_threshold
                     if cli.preventive_threshold is not None
                     else _PREVENTIVE_THRESHOLD_MAIN)
        manager = pathlib.Path(cli.manager_npz) if cli.manager_npz else NPZ
        OUT_V2.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(OUT_V2 / ".bc.lock", "BC worker v2 artifacts"):
            main_v2(threshold, manager)
    else:
        if cli.preventive_threshold is not None:
            parser.error("--preventive-threshold only with --v2 (v1 surface unchanged)")
        if cli.manager_npz is not None:
            parser.error("--manager-npz only with --v2 (v1 surface unchanged)")
        with exclusive_lock(OUT / ".bc.lock", "BC worker artifacts"):
            main()
