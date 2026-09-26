"""R9 "manager re-education of the certified team" driver (sole executor of docs/prereg/PREREG-R9-manager-reeducation.md;
a targeted rework clone of run_v29_relection.py, not a thin wrapper).

Clone difference table (item by item against PREREG-R9):
- Team: worker = the R8 certified release model_final.zip (rev26, dual-v4-asymmetric-v3, sha 2837288d…)
  connected through the staging path without a receipt (0o444, borrowing the run_r8_certification._stage_eval_file pattern,
  avoiding the mandatory eval_assembled release-receipt gate); baseline manager = M29 npz (sha 89441388…, legacy-v3).
- CALIBRATED_PROTOCOL_VERSION = 4; the decision-line region (ABANDON/FLOOR) is no longer pinned as constants but derived
  on the spot from the new anchor (ABANDON = anchor mean×0.66, FLOOR = anchor mean×85/92; formula + values go into the ledger).
- the exam evaluation command adds --manager-policy-observation-view (M29 = legacy-v3; candidate arms = raw-v4)
  + --worker pointing at the staged zip.
- Seed sets: replication pool A = 2_123_000-127, pool B = 2_124_000-127, final exam = 2_125_000-255
  (virgin ranges of the evaluation bank; the only new-pool consumption, approved as "anchor folded into R9"); by_seed follows.
- ARMS: r9-mfresh (pure fresh) and r9-mcurr (same + --deep-start-curriculum p=0.5, target=2, cap=8);
  both arms --worker-zip <staged> --worker-zip-sha256 <sha> --manager-policy-observation-view
  raw-v4; 160k steps, 4h timeout per arm.
- Sequence: preflight → G0-6 full-table replay at the old endpoints (probe_efix_g0 with both knobs set to old, replaying the R8 final exam
  official-r8-final-{baseline,candidate}, 256 games each, reconciled bit for bit) → new anchor burn (M29 × certified worker,
  128 each on 2_123/2_124 + 256 on 2_125) → two arms trained → export_manager_npz+parity → arm exams
  (128 each on 2_123/2_124 per arm) → paired verdict → winner's 2_125 final exam, 256 pairs + final verdict.
- Verdict layer: imports r8_statistics (unchanged). The archive cross-check contract of its analyze_paired_archives pins
  the "manager" slot to numpy_policy and requires identical content on both sides (R8 geometry: variable = worker, shared = manager);
  R9 is exactly the dual (shared = certified worker zip[sb3_checkpoint], variable = manager npz), and any slot rewrite would
  fake the kind/num_timesteps identity fields. So this driver re-implements the same verdict maths formula by formula from its registered primitives (_student_t_upper_critical/
  _exact_sign_p_value/_clopper_pearson_upper)
  (phase=development/final minimum-pair semantics); tests/test_r9_machinery.py keeps numeric-equivalence assertions on the shared primitive layer
  (t critical value/sign test/Clopper-Pearson) against implementation drift.
- Gate limbs (design decision of 2026-07-31 on re-education: R9's scientific question is depth, so the gates follow; family-wise α=0.05
  re-split over 3 constraints):
  (1) depth superiority limb (primary metric, new): per-seed paired depth difference (candidate−baseline),
     both mean_lcb + exact_sign, minimum_effect=0; the key name 'depth' was verified on the spot against the official R8 archive
     rows (there is no dungeon_level key);
  (2) wage non-inferiority limb (changed from superiority to non-inferiority): mean LCB ≥ −0.10×anchor-pool wage mean (derived per pool from the anchor
     on the spot; formula + line value go into the ledger; exact_sign leaves the gate, sign information is recorded but not ruled on).
     A manager trading farm-level time for dives is the expected behaviour in this case, so it is not failed for wage not being superior; a wage collapse
     (falling more than 10% below the anchor mean) is still blocked;
  (3) death limb unchanged: exact conditional McNemar non-inferiority (margin 0.10) + an observed death line derived on the spot from the new anchor per the
     v31-D3-10 leftover obligation (line = anchor died count + margin×pairs); inheriting the absolute 6/32 line is forbidden;
  (4) ret/kills/worker_kills are demoted from gate limbs to record-only diagnostics: computed and archived,
     not part of the failed_checks verdict.
- Winner selection (final design decision): passing arms are compared by paired depth mean difference, within a 0.10 band counted as tied; tie-break
  (1) lower total per-seed died over the combined pools, (2) still tied, take r9-mfresh (Occam: the curriculum arm must prove itself
  with a visible depth advantage). The ret band leaves the selection (recorded in quals).
- The verdict must carry the depth histogram + per-seed died + the three fool-proofing instruments (median first forced hand-over / DIVE success rate /
  dlvl residence ratio; foundation-review baselines 1495/25%/16058:3512; instruments 2/3 are probe-level readings;
  the ledger records the probe re-check command, and the machine does not launch extra probes on its own).
- Gold pool 9000 and the 7000/8000/12000 held-out pools untouched; --board not used; this case burns no gold run.
Ledger: train/runs/r9-reeducation/gate_ledger.jsonl.
Usage: .venv/bin/python train/run_r9_reeducation.py (launch requires recorded approval).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import signal
import stat
import subprocess
import time
import traceback
import zipfile

import r8_statistics
from eval_contract import (PROTOCOL_VERSION, OperationalFailure, OutputReservationError,
                           exclusive_lock, expected_eval_identity,
                           freeze_eval_identity, read_eval_archive,
                           verify_eval_identity)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
R9 = RUNS / "r9-reeducation"          # campaign control directory (created lazily; implementation/import writes nothing)
LEDGER = R9 / "gate_ledger.jsonl"
STAGING = R9 / "staging"
STAGED_WORKER = STAGING / "worker.zip"
EVAL = RUNS / "eval-assembled"

# ---- team constants (full sha256sum taken on the spot, 2026-07-31) ----
W_ZIP_SRC = RUNS / "r8-certification-published" / "model_final.zip"
W_ZIP_SHA = "2837288dad19a685925558a0d86e1cecd951d5f065d1d2f367f667c13b9cf006"
M29_NPZ = ROOT / "train" / "models" / "v29-manager-mfresh" / "policy.npz"
M29_SHA = "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6"

# ---- G0-6 leftover obligation (E-fix): material for the full-table replay of the R8 final exam at the old endpoints ----
G0_PROBE = RUNS / "efix-g0-evidence" / "probe_efix_g0.py"
R8_FINAL = {
    "official-r8-final-baseline-2122000": {
        "archive_sha256":
            "e7c97a0aef1d3abd4eee18e574938575f853ee0f4b8d02b8118bb8917fc9d37d",
        "worker_zip": RUNS / "r8-certification-control" / "eval-inputs"
                      / "official-r8-final-baseline-2122000" / "worker.zip",
        "worker_sha256":
            "2f7bc9dd810956c3feeb330575c9a03ddff0b476333ac429a411935985b04f42",
    },
    "official-r8-final-candidate-2122000": {
        "archive_sha256":
            "42a8f5e2bfdda5dcc366e524e1d25d759eba303eed58b2caf4de4e25695ba032",
        "worker_zip": RUNS / "r8-certification-control" / "eval-inputs"
                      / "official-r8-final-candidate-2122000" / "worker.zip",
        "worker_sha256": W_ZIP_SHA,
    },
}
R8_FINAL_SEEDS = (2_122_000, 2_122_255)

# ---- pool allocation (PREREG-R9 §2; the only new-pool consumption of this case) ----
POOL_A = (2_123_000, 2_123_127)
POOL_B = (2_124_000, 2_124_127)
POOL_FINAL = (2_125_000, 2_125_255)

TAG_ANCHOR = {"a": "r9-anchor-a-2123000", "b": "r9-anchor-b-2124000",
              "final": "r9-anchor-final-2125000"}
POOLS = {"a": POOL_A, "b": POOL_B, "final": POOL_FINAL}

STEPS = 160_000
CALIBRATED_PROTOCOL_VERSION = 4
# decision-line derivation factors (base = combined mean of the new anchor A+B; values are plugged in from the anchor readings at launch and go into the ledger):
ABANDON_FACTOR = 0.66            # registered form of the v29 precedent 75/112.4≈0.667
FLOOR_NUM, FLOOR_DEN = 85.0, 92.0  # the v25/v29 ratio 85/92 kept, with the new anchor as base
DEATH_MARGIN = 0.10              # exact conditional McNemar non-inferiority margin (same value as PREREG-R8 §1.3)
FAMILYWISE_ALPHA = 0.05
# winner selection band (final design decision): paired depth mean difference, tied within a 0.10 band → (1) fewer deaths (2) mfresh.
WINNER_DEPTH_TIE_BAND = 0.10
# borderline-note thresholds (verdict discipline inherited from v31; registered here, mandatory with the verdict):
NEAR_LINE_DEATH_GAP = 1          # |candidate_deaths − derived line| ≤ 1 life
NEAR_LINE_LCB_BAND = 0.5         # any mean limb with LCB ∈ [0, 0.5)
NEAR_LINE_FLOOR_GAP = 1.0        # |winner mean − FLOOR| ≤ 1.0
NEAR_LINE_UCB_BAND = 0.01        # McNemar UCB within 0.01 of the margin

# ---- verdict gate limbs (design decision of 2026-07-31 on re-education; pre-registered in the module docstring) ----
# primary metric = depth superiority limb: the key name 'depth' was verified on the spot against the official R8 archive rows.
DEPTH_RULE = r8_statistics.MetricRule("depth")
# wage non-inferiority limb: line = −WAGE_NI_FRACTION×anchor-pool wage mean (derived per pool on the spot; the endpoint passes).
WAGE_KEY = "farm_worker_wage"
WAGE_NI_FRACTION = 0.10
# number of gate constraints = depth + wage non-inferiority + death = 3; family-wise α=0.05 is re-split over them.
GATE_CONSTRAINT_COUNT = 3
# ret/kills/worker_kills demoted to record-only diagnostics (computed on the 0-superiority definition, comparable with the old gate readings;
# they use no α and are not in failed_checks).
RECORD_ONLY_RULES = (
    r8_statistics.MetricRule("ret"),
    r8_statistics.MetricRule("kills"),
    r8_statistics.MetricRule("farm_worker_kills"),
)
R9_STATISTICS_SCHEMA = "diablogym-r9-paired-statistics/2"
R9_METHOD_REVISION = (r8_statistics.R8_METHOD_REVISION
                      + "+r9-role-dual-depth-gate/2")

# three fool-proofing instruments ("depth unlocked" requires all of them to move; baselines = foundation-review measurements, re-read against the new anchor after the fix):
GAUGE_BASELINES = {
    "first_forced_handover_median_micro_steps": 1495,   # probe level (decision stream)
    "dive_window_success_rate": 0.25,                    # computable from archives (mode_seq/depth)
    "dlvl_dwell_ratio_l1_l2": [16058, 3512],             # probe level (per-beat dlvl)
}

ARMS = {
    "r9-mfresh": ["--ent-coef", "0.02", "--lr", "3e-4", "--seed", "22"],
    "r9-mcurr": ["--ent-coef", "0.02", "--lr", "3e-4", "--seed", "22",
                 "--deep-start-curriculum", "p=0.5,target=2,cap=8"],
}


class CampaignError(RuntimeError):
    """R9 campaign inputs / file system do not meet the pre-registered contract."""


def log(event: dict):
    R9.mkdir(parents=True, exist_ok=True)
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    R9.mkdir(parents=True, exist_ok=True)
    with open(R9 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha16(p) -> str:
    return sha256(p)[:16]


# ---- the three staging functions (borrowing the run_r8_certification.py:741/1333/1360/3030 pattern) ----

def _stable_read(path: pathlib.Path) -> bytes:
    """Read one regular file identity and reject symlink/replace races."""
    path = pathlib.Path(path)
    try:
        before_path = path.lstat()
    except OSError as exc:
        raise CampaignError(f"file unreadable: {path}: {exc}") from exc
    require(not stat.S_ISLNK(before_path.st_mode), f"refusing symlink input: {path}")
    require(stat.S_ISREG(before_path.st_mode), f"input is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CampaignError(f"file cannot be opened stably: {path}: {exc}") from exc
    try:
        first = os.fstat(fd)
        require(stat.S_ISREG(first.st_mode), f"opened input is not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        second = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise CampaignError(f"file identity vanished after reading: {path}: {exc}") from exc
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_mode, item.st_size,
        item.st_mtime_ns, item.st_ctime_ns,
    )
    require(
        identity(before_path) == identity(first)
        == identity(second) == identity(after_path),
        f"file replaced or modified while being read: {path}",
    )
    return b"".join(chunks)


def sha256(p) -> str:
    return hashlib.sha256(_stable_read(pathlib.Path(p))).hexdigest()


def _fsync_directory(path: pathlib.Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise CampaignError(f"directory cannot be opened for fsync: {path}: {exc}") from exc
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_bytes_exclusive(
        path: pathlib.Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            mode,
        )
    except FileExistsError:
        require(_stable_read(path) == payload,
                f"immutable file already exists with content drift: {path}")
        return
    with os.fdopen(fd, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _stage_eval_file(
        source: pathlib.Path, destination: pathlib.Path,
        *, expected_sha256: str | None = None) -> str:
    payload = _stable_read(source)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        require(
            digest == expected_sha256,
            f"eval staging source SHA drift: {source}:{digest} != {expected_sha256}",
        )
    _write_bytes_exclusive(destination, payload, mode=0o444)
    try:
        os.chmod(destination, 0o444)
    except OSError as exc:
        raise CampaignError(f"eval staging cannot be made read-only: {destination}: {exc}") from exc
    _fsync_directory(destination.parent)
    require(sha256(destination) == digest,
            f"eval staging copy SHA drift: {destination}")
    return digest


# ---- operational primitives (v29 skeleton) ----

def run(cmd, logfile, timeout) -> int:
    R9.mkdir(parents=True, exist_ok=True)
    with open(R9 / logfile, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)   # kill the whole group: prevents orphaned SubprocVecEnv grandchildren
            except ProcessLookupError:
                pass
            proc.wait()
            return 124    # hang guard: recorded as a crash/failure (operational guard, not a verdict input)


def zip_steps(p: pathlib.Path) -> int:
    """SB3 real-chain reading (v29 panel blocker fix: the throttled status count always lags)."""
    try:
        with zipfile.ZipFile(p) as z:
            return int(json.loads(z.read("data"))["num_timesteps"])
    except Exception:
        return 0


def seeds_arg(pool: tuple[int, int]) -> str:
    lo, hi = pool
    return f"{lo}-{hi}"


def by_seed(rows, pool: tuple[int, int]) -> dict:
    lo, hi = pool
    m = {r["seed"]: r for r in rows}
    require(len(rows) == len(m), "abnormal seed set (duplicate seed)")
    require(set(m) == set(range(lo, hi + 1)),
            f"abnormal seed set (must be {lo}-{hi})")
    return m


def require_calibrated_protocol() -> None:
    if PROTOCOL_VERSION != CALIBRATED_PROTOCOL_VERSION:
        raise OperationalFailure(
            "R9's decision-line formulas / pool allocation are pre-registered only under protocol-v4 environment semantics; "
            f"current PROTOCOL_VERSION={PROTOCOL_VERSION}. A further protocol migration must reopen the pre-registration first; "
            "mixing old thresholds is forbidden")


# ---- evaluation (exam) ----

def exam(tag: str, pool: tuple[int, int], manager_npz: pathlib.Path,
         manager_view: str, timeout: int):
    """staged certified worker × given manager; returns (validated_doc, archive_sha256) or None."""
    require(manager_view in ("legacy-v3", "raw-v4"), f"invalid manager view: {manager_view}")
    out = EVAL / f"{tag}.json"
    require(not out.exists(), f"archive immutability: {out} already exists, refusing to overwrite")
    lo, hi = pool
    seed_values = list(range(lo, hi + 1))
    snapshot = freeze_eval_identity(ROOT, str(STAGED_WORKER), str(manager_npz))
    require(snapshot["worker"]["kind"] == "sb3_checkpoint"
            and snapshot["worker"]["sha256"] == W_ZIP_SHA,
            "staged certified worker identity drift")
    expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
    cmd = [PY, "train/eval_assembled.py",
           "--worker", snapshot["worker"]["path"],
           "--manager-npz", snapshot["manager"]["path"],
           "--manager-policy-observation-view", manager_view,
           "--seeds", seeds_arg(pool), "--tag", tag]
    if run(cmd, f"exam-{tag}.{time.time_ns()}.log", timeout=timeout) != 0:
        if out.exists():    # rotate the partial archive to make way for the re-exam
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    try:
        d = read_eval_archive(out, **expected)
        verify_eval_identity(snapshot, ROOT)
    except (OSError, KeyError, TypeError, ValueError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    return d, sha256(out)


def exam_retry(tag, pool, manager_npz, manager_view, timeout):
    result = exam(tag, pool, manager_npz, manager_view, timeout)
    if result is None:
        log({"event": "exam_crash", "tag": tag, "note": "evaluation failed; re-exam once under the crash clause"})
        result = exam(tag, pool, manager_npz, manager_view, timeout)
    return result


# ---- depth instruments (mandatory with the verdict) ----

def depth_hist(rows) -> dict:
    hist: dict[str, int] = {}
    for r in rows:
        hist[str(int(r["depth"]))] = hist.get(str(int(r["depth"])), 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: int(kv[0])))


def l3_plus(rows) -> int:
    return sum(1 for r in rows if r["depth"] >= 3)


def died_seeds(rows) -> list:
    return sorted(int(r["seed"]) for r in rows if r["died"])


def dive_per_ep(rows) -> float:
    return sum(r["mode_seq"].count("D") for r in rows) / max(1, len(rows))


def dive_gauges(rows) -> dict:
    """DIVE success rate (archive definition): Σ(depth−1)/Σ(D windows). The start is always level 1,
    and each successful dive moves exactly one level; the probe definition (per-window dlvl0→dlvl_end) is re-checked by the probe."""
    dives = sum(r["mode_seq"].count("D") for r in rows)
    descents = sum(max(0, int(r["depth"]) - 1) for r in rows)
    return {
        "dive_windows": int(dives),
        "descents": int(descents),
        "dive_window_success_rate": (descents / dives) if dives else None,
    }


def bonus_per_ep(rows) -> float:
    # stairs-bonus conversion: depth=d converts 8×(1+2+…+(d−1)); 0 for d≤1
    return sum(8 * sum(range(1, r["depth"])) for r in rows) / max(1, len(rows))


def depth_dashboard(rows) -> dict:
    return {
        "n": len(rows),
        "depth_hist": depth_hist(rows),
        "l3_plus": l3_plus(rows),
        "died": sum(1 for r in rows if r["died"]),
        "died_seeds": died_seeds(rows),
        "dive_per_ep": round(dive_per_ep(rows), 3),
        "bonus_per_ep": round(bonus_per_ep(rows), 3),
        **dive_gauges(rows),
    }


def gauge_report(anchor_rows, candidate_rows, anchor_tag: str) -> dict:
    """Three fool-proofing instruments: instrument 2 is computable from archives (read for anchor and candidate); instruments 1/3 are probe level:
    register the baseline values + the anchor-side re-check command (probe_r9_dive pins the manager to M29, which is exactly the anchor assembled agent;
    the candidate side needs a probe variant with a parameterisable manager; the machine does not launch extra probes on its own, and the obligation is recorded)."""
    probe_cmd = (f"{PY} {G0_PROBE.with_name('probe_r9_dive.py')} "
                 f"<out.json> <comma-separated seed list> {STAGED_WORKER} {anchor_tag}.json")
    return {
        "baselines_foundation_dossier": GAUGE_BASELINES,
        "dive_window_success_rate": {
            "anchor": dive_gauges(anchor_rows)["dive_window_success_rate"],
            "candidate": dive_gauges(candidate_rows)["dive_window_success_rate"],
        },
        "probe_level_gauges": {
            "gauges": ["first_forced_handover_median_micro_steps",
                       "dlvl_dwell_ratio_l1_l2"],
            "anchor_recheck_cmd": probe_cmd,
            "candidate_recheck_note":
                "probe_r9_dive hard-codes the M29 manager; a winner-side re-check needs a probe variant with a parameterisable manager,"
                " a manual obligation; this machine does not launch extra probes on its own",
        },
    }


# ---- verdict core (same machinery and formulas as r8_statistics, R9 role geometry) ----

def _shared_worker_identity(meta_worker: dict, label: str) -> dict:
    require(isinstance(meta_worker, dict), f"{label} worker identity invalid")
    require(meta_worker.get("kind") == "sb3_checkpoint",
            f"{label} shared worker kind must be sb3_checkpoint")
    require(meta_worker.get("sha256") == W_ZIP_SHA,
            f"{label} shared worker sha must equal the certified release: {meta_worker.get('sha256')!r}")
    require(meta_worker.get("gate_report_sha256") is None,
            f"{label} shared worker must not carry a release receipt (staging path without a receipt)")
    return {key: meta_worker.get(key)
            for key in ("kind", "sha256", "num_timesteps", "gate_report_sha256")}


def paired_judgment(baseline: dict, candidate: dict, *,
                    baseline_sha256: str, candidate_sha256: str,
                    phase: str) -> dict:
    """R9 paired verdict: the registered R8 primitives re-implemented formula by formula, with the gate limbs reset per the re-education design decision.

    baseline = new anchor archive (M29 × certified worker); candidate = candidate-manager archive (candidate × the same worker).
    Gates (3 constraints, family-wise α=0.05 split evenly): (1) depth superiority (primary metric, mean_lcb +
    exact_sign both, minimum_effect=0); (2) wage non-inferiority (mean LCB ≥
    −WAGE_NI_FRACTION×anchor-pool wage mean, derived on the spot, endpoint included; sign leaves the gate);
    (3) death: exact conditional McNemar non-inferiority (margin 0.10) + an observed line derived from the new anchor
    (line = anchor died count + margin×pairs; no absolute constant line).
    ret/kills/worker_kills are record-only diagnostics, computed and archived but not part of the verdict.
    """
    require(phase in ("development", "final"), f"invalid phase: {phase}")
    floor_pairs = (r8_statistics.MIN_DEVELOPMENT_PAIRS if phase == "development"
                   else r8_statistics.MIN_FINAL_PAIRS)
    for label, document in (("baseline", baseline), ("candidate", candidate)):
        require(isinstance(document, dict)
                and set(document) == {"schema_version", "meta", "agg", "rows"},
                f"{label} archive must be a validated schema-v5 archive")
        require(document["schema_version"]
                == r8_statistics.SUPPORTED_EVAL_ARCHIVE_SCHEMA,
                f"{label} archive schema must be v5")
    b_meta, c_meta = baseline["meta"], candidate["meta"]
    require(b_meta["protocol"] == c_meta["protocol"],
            "baseline/candidate protocol or seed table differ")
    require(b_meta["runtime"] == c_meta["runtime"],
            "baseline/candidate runtime/content identity differ")
    # R9 role geometry: shared = certified worker zip (content identity must be equal), variable = manager npz.
    require(_shared_worker_identity(b_meta["worker"], "baseline")
            == _shared_worker_identity(c_meta["worker"], "candidate"),
            "baseline/candidate shared worker content identity differ")
    for label, meta in (("baseline", b_meta), ("candidate", c_meta)):
        require(meta["manager"].get("kind") == "numpy_policy",
                f"{label} manager kind must be numpy_policy")
    b_mgr_sha = b_meta["manager"]["sha256"]
    c_mgr_sha = c_meta["manager"]["sha256"]
    require(b_mgr_sha == M29_SHA, f"baseline manager must be M29: {b_mgr_sha!r}")
    require(b_mgr_sha != c_mgr_sha, "the managers of the two paired sides must differ (variable = manager)")
    require(isinstance(baseline_sha256, str) and isinstance(candidate_sha256, str)
            and baseline_sha256 != candidate_sha256,
            "archive byte SHA must be given in full and must differ")

    protocol = b_meta["protocol"]
    require(protocol.get("deterministic") is True
            and isinstance(protocol.get("seeds"), list),
            "deterministic protocol seed table missing")
    seeds = protocol["seeds"]
    b_rows, c_rows = baseline["rows"], candidate["rows"]
    require([row.get("seed") for row in b_rows] == seeds
            and [row.get("seed") for row in c_rows] == seeds,
            "row order must match the protocol seed table exactly")
    n_pairs = len(seeds)
    require(n_pairs >= floor_pairs,
            f"{phase} verdict needs at least {floor_pairs} pairs (got {n_pairs})")

    all_metric_keys = ([DEPTH_RULE.key, WAGE_KEY]
                       + [rule.key for rule in RECORD_ONLY_RULES])
    per_constraint_alpha = FAMILYWISE_ALPHA / GATE_CONSTRAINT_COUNT
    t_critical = r8_statistics._student_t_upper_critical(
        per_constraint_alpha, n_pairs - 1)
    checks: dict = {}
    paired_hash_rows = []
    for b_row, c_row in zip(b_rows, c_rows):
        require(b_row.get("seed") == c_row.get("seed"), "paired row seed mismatch")
        hash_row = {"seed": b_row["seed"], "baseline": {}, "candidate": {}}
        for key in all_metric_keys:
            r8_statistics._finite_number(
                b_row.get(key), f"baseline seed {b_row['seed']} {key}")
            r8_statistics._finite_number(
                c_row.get(key), f"candidate seed {c_row['seed']} {key}")
            hash_row["baseline"][key] = b_row[key]
            hash_row["candidate"][key] = c_row[key]
        require(isinstance(b_row.get("died"), bool)
                and isinstance(c_row.get("died"), bool),
                "died must be a per-row bool")
        hash_row["baseline"]["died"] = b_row["died"]
        hash_row["candidate"]["died"] = c_row["died"]
        paired_hash_rows.append(hash_row)

    def limb_stats(key: str, minimum_effect: float) -> dict:
        """Paired statistics re-implemented verbatim from the registered r8 formulas (all limbs in the higher direction; pass/fail is decided by the gate layer)."""
        baseline_values = [float(row[key]) for row in b_rows]
        candidate_values = [float(row[key]) for row in c_rows]
        improvement_values = [c - b for b, c
                              in zip(baseline_values, candidate_values)]
        baseline_mean = math.fsum(baseline_values) / n_pairs
        candidate_mean = math.fsum(candidate_values) / n_pairs
        improvement_mean = math.fsum(improvement_values) / n_pairs
        squared = math.fsum((v - improvement_mean) ** 2
                            for v in improvement_values)
        sample_stddev = math.sqrt(squared / (n_pairs - 1))
        standard_error = sample_stddev / math.sqrt(n_pairs)
        lower_bound = improvement_mean - t_critical * standard_error
        for label, value in (("improvement_mean", improvement_mean),
                             ("lower_confidence_bound", lower_bound)):
            require(math.isfinite(value), f"{key}.{label} must be finite")
        centered = [v - float(minimum_effect) for v in improvement_values]
        wins = sum(v > 0.0 for v in centered)
        losses = sum(v < 0.0 for v in centered)
        sign_p = r8_statistics._exact_sign_p_value(wins, wins + losses)
        return {
            "direction": "higher",
            "minimum_effect": float(minimum_effect),
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "raw_candidate_minus_baseline_mean": improvement_mean,
            "improvement_mean": improvement_mean,
            "sample_stddev": sample_stddev,
            "standard_error": standard_error,
            "one_sided_t_critical": t_critical,
            "lower_confidence_bound": lower_bound,
            "sign_test": {
                "wins_above_minimum_effect": wins,
                "losses_below_minimum_effect": losses,
                "ties_at_minimum_effect": n_pairs - wins - losses,
                "non_ties": wins + losses,
                "one_sided_exact_p_value": sign_p,
            },
        }

    metrics: dict = {}
    # (1) depth superiority limb (primary metric; mean+sign share one α share, same form as r8).
    depth = limb_stats(DEPTH_RULE.key, float(DEPTH_RULE.minimum_effect))
    depth_mean_passed = (depth["lower_confidence_bound"]
                         > float(DEPTH_RULE.minimum_effect))
    depth_sign_passed = (depth["sign_test"]["one_sided_exact_p_value"]
                         <= per_constraint_alpha)
    checks["depth.mean_lcb"] = depth_mean_passed
    checks["depth.exact_sign"] = depth_sign_passed
    depth.update({
        "kind": "superiority",
        "mean_lcb_passed": depth_mean_passed,
        "sign_test": {**depth["sign_test"], "required": True,
                      "passed": depth_sign_passed},
        "passed": depth_mean_passed and depth_sign_passed,
    })
    metrics["depth"] = depth
    # (2) wage non-inferiority limb: line = −WAGE_NI_FRACTION×anchor-pool wage mean (derived per pool from the anchor on the spot,
    #    the endpoint passes); exact_sign leaves the gate (sign information recorded, not ruled on).
    anchor_wage_mean = math.fsum(
        float(row[WAGE_KEY]) for row in b_rows) / n_pairs
    wage_minimum_effect = -WAGE_NI_FRACTION * anchor_wage_mean
    wage = limb_stats(WAGE_KEY, wage_minimum_effect)
    wage_ni_passed = wage["lower_confidence_bound"] >= wage_minimum_effect
    checks["farm_worker_wage.noninferiority_lcb"] = wage_ni_passed
    wage.update({
        "kind": "noninferiority",
        "noninferiority": {
            "formula": "minimum_effect = -WAGE_NI_FRACTION × anchor_wage_mean"
                       " (derived per pool from the anchor on the spot; passes if LCB ≥ line, endpoint included)",
            "margin_fraction": WAGE_NI_FRACTION,
            "anchor_wage_mean": anchor_wage_mean,
            "minimum_effect": wage_minimum_effect,
            "lcb_passed": wage_ni_passed,
        },
        "sign_test": {**wage["sign_test"], "required": False, "passed": None},
        "passed": wage_ni_passed,
    })
    metrics[WAGE_KEY] = wage
    # (3) record-only diagnostic limbs: computed and archived, not in checks/failed_checks (re-education decision item 4).
    record_only_metrics: dict = {}
    for rule in RECORD_ONLY_RULES:
        stats = limb_stats(rule.key, float(rule.minimum_effect))
        stats.update({
            "record_only": True,
            "note": "computed on the 0-superiority definition (comparable with the old gate readings); uses no α, not part of the verdict",
            "mean_lcb_indicative": (stats["lower_confidence_bound"]
                                    > float(rule.minimum_effect)),
            "sign_test": {**stats["sign_test"], "required": False,
                          "passed": None},
        })
        record_only_metrics[rule.key] = stats

    baseline_deaths = sum(bool(row["died"]) for row in b_rows)
    candidate_deaths = sum(bool(row["died"]) for row in c_rows)
    candidate_only = sum((not bool(b["died"])) and bool(c["died"])
                         for b, c in zip(b_rows, c_rows))
    baseline_only = sum(bool(b["died"]) and (not bool(c["died"]))
                        for b, c in zip(b_rows, c_rows))
    component_alpha = per_constraint_alpha / 2.0
    discordant = candidate_only + baseline_only
    if discordant:
        theta_upper = r8_statistics._clopper_pearson_upper(
            candidate_only, discordant, component_alpha)
        risk_upper = (2.0 * theta_upper - 1.0) * discordant / n_pairs
    else:
        theta_upper = 0.0
        risk_upper = 0.0
    death_bound_passed = risk_upper <= DEATH_MARGIN
    # where v31-D3-10 lands: the observed death line is derived on the spot from the new anchor (the absolute 6/32 line is forbidden).
    derived_line = baseline_deaths + DEATH_MARGIN * n_pairs
    observed_within_line = candidate_deaths <= derived_line
    checks["deaths.noninferiority_upper_bound"] = death_bound_passed
    checks["deaths.observed_within_derived_line"] = observed_within_line
    death_report = {
        "key": "died",
        "method": "exact-conditional-mcnemar-clopper-pearson-risk-difference",
        "noninferiority_margin": DEATH_MARGIN,
        "baseline_deaths": baseline_deaths,
        "candidate_deaths": candidate_deaths,
        "baseline_death_rate": baseline_deaths / n_pairs,
        "candidate_death_rate": candidate_deaths / n_pairs,
        "observed_candidate_minus_baseline_risk":
            (candidate_deaths - baseline_deaths) / n_pairs,
        "candidate_only_deaths": candidate_only,
        "baseline_only_deaths": baseline_only,
        "concordant_pairs": n_pairs - discordant,
        "component_alpha": component_alpha,
        "discordant_pairs": discordant,
        "candidate_only_conditional_theta_upper_bound": theta_upper,
        "candidate_minus_baseline_risk_upper_bound": risk_upper,
        "derived_observed_line": {
            "formula": "line_deaths = anchor_deaths + margin*n_pairs"
                       " (v31-D3-10: derived from the anchor on the spot; inheriting the absolute 6/32 line is forbidden)",
            "anchor_deaths": baseline_deaths,
            "margin": DEATH_MARGIN,
            "n_pairs": n_pairs,
            "line_deaths": derived_line,
        },
        "observed_within_derived_line": observed_within_line,
        "noninferiority_passed": death_bound_passed,
        "passed": death_bound_passed and observed_within_line,
    }
    failed_checks = sorted(name for name, ok in checks.items() if not ok)
    near_line_notes = []
    if abs(candidate_deaths - derived_line) <= NEAR_LINE_DEATH_GAP:
        near_line_notes.append(
            f"observed deaths within {NEAR_LINE_DEATH_GAP} of the derived line"
            f"(candidate={candidate_deaths},line={derived_line:.1f})")
    if abs(DEATH_MARGIN - risk_upper) <= NEAR_LINE_UCB_BAND:
        near_line_notes.append(
            f"McNemar UCB within {NEAR_LINE_UCB_BAND} of the margin (UCB={risk_upper:.4f})")
    depth_lcb = metrics["depth"]["lower_confidence_bound"]
    if 0.0 <= depth_lcb < NEAR_LINE_LCB_BAND:
        near_line_notes.append(f"depth.LCB borderline ({depth_lcb:.3f})")
    wage_gap = metrics[WAGE_KEY]["lower_confidence_bound"] - wage_minimum_effect
    if 0.0 <= wage_gap < NEAR_LINE_LCB_BAND:
        near_line_notes.append(f"wage non-inferiority LCB borderline to the line (gap={wage_gap:.3f})")
    result = {
        "schema_version": R9_STATISTICS_SCHEMA,
        "method_revision": R9_METHOD_REVISION,
        "phase": phase,
        "n_pairs": n_pairs,
        "required_pairs": floor_pairs,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "simultaneous_confidence": 1.0 - FAMILYWISE_ALPHA,
        "constraint_count": GATE_CONSTRAINT_COUNT,
        "per_constraint_alpha": per_constraint_alpha,
        "source": {
            "eval_schema_version": baseline["schema_version"],
            "baseline_archive_sha256": baseline_sha256,
            "candidate_archive_sha256": candidate_sha256,
            "baseline_manager_sha256": b_mgr_sha,
            "candidate_manager_sha256": c_mgr_sha,
            "shared_worker_sha256": W_ZIP_SHA,
            "role_geometry": "r9-dual (shared = certified worker zip, variable = manager npz)",
        },
        "seeds_sha256": r8_statistics._canonical_sha256(seeds),
        "paired_data_sha256": r8_statistics._canonical_sha256(paired_hash_rows),
        "rules": {
            "gating": [
                {"key": "depth", "kind": "superiority", "direction": "higher",
                 "minimum_effect": float(DEPTH_RULE.minimum_effect),
                 "checks": ["mean_lcb", "exact_sign"]},
                {"key": WAGE_KEY, "kind": "noninferiority",
                 "direction": "higher",
                 "minimum_effect": wage_minimum_effect,
                 "formula": "minimum_effect = "
                            f"-{WAGE_NI_FRACTION} × anchor_wage_mean",
                 "checks": ["noninferiority_lcb"]},
                {"key": "died",
                 "kind": "mcnemar-noninferiority+derived-observed-line",
                 "noninferiority_margin": DEATH_MARGIN,
                 "checks": ["noninferiority_upper_bound",
                            "observed_within_derived_line"]},
            ],
            "record_only": [rule.key for rule in RECORD_ONLY_RULES],
        },
        "metrics": metrics,
        "record_only_metrics": record_only_metrics,
        "death_noninferiority": death_report,
        "near_line_notes": near_line_notes,
        "verdict": {
            "status": "PASS" if not failed_checks else "FAIL",
            "checks": checks,
            "failed_checks": failed_checks,
        },
    }
    json.dumps(result, allow_nan=False)
    return result


def select_winner(quals: dict, eligible: list) -> str:
    """Winner selection (final design decision): passing arms are compared by paired depth mean difference (primary metric).

    Within the band ``WINNER_DEPTH_TIE_BAND`` they count as tied; tie-break: (1) lower total per-seed
    died over the combined pools; (2) still tied, take r9-mfresh (Occam: the curriculum arm must prove itself with a visible
    depth advantage). The ret band has left the selection (recorded in quals, archived but not ruled on).
    """
    require(bool(eligible), "winner selection needs at least one passing arm")
    require(all(name in quals for name in eligible), "passing arm lacks quals readings")
    ms = {name: quals[name]["depth_paired_mean_ab"] for name in eligible}
    band = [name for name in eligible
            if max(ms.values()) - ms[name] <= WINNER_DEPTH_TIE_BAND]
    if len(band) > 1:
        dmin = min(quals[name]["pooled_died"] for name in band)
        band = [name for name in band if quals[name]["pooled_died"] == dmin]
        return "r9-mfresh" if "r9-mfresh" in band else band[0]
    return band[0]


def analysis_digest(analysis: dict) -> dict:
    """Ledger verdict summary: the main verdict line leads with the depth reading (re-education decision item 3)."""
    depth_m = analysis["metrics"]["depth"]
    wage_m = analysis["metrics"][WAGE_KEY]
    death = analysis["death_noninferiority"]
    return {
        "depth": {
            "paired_mean": round(depth_m["improvement_mean"], 3),
            "lcb": round(depth_m["lower_confidence_bound"], 3),
            "sign_p": depth_m["sign_test"]["one_sided_exact_p_value"],
            "wins": depth_m["sign_test"]["wins_above_minimum_effect"],
            "non_ties": depth_m["sign_test"]["non_ties"],
            "passed": depth_m["passed"],
        },
        "wage_ni": {
            "lcb": round(wage_m["lower_confidence_bound"], 3),
            "line": round(wage_m["noninferiority"]["minimum_effect"], 3),
            "anchor_wage_mean": round(
                wage_m["noninferiority"]["anchor_wage_mean"], 3),
            "passed": wage_m["passed"],
        },
        "death": {
            "baseline_deaths": death["baseline_deaths"],
            "candidate_deaths": death["candidate_deaths"],
            "mcnemar_ucb": round(
                death["candidate_minus_baseline_risk_upper_bound"], 4),
            "derived_line": death["derived_observed_line"]["line_deaths"],
            "passed": death["passed"],
        },
        "record_only": {
            key: round(value["improvement_mean"], 3)
            for key, value in analysis["record_only_metrics"].items()},
        "failed_checks": analysis["verdict"]["failed_checks"],
        "near_line_notes": analysis["near_line_notes"],
        "paired_data_sha16": analysis["paired_data_sha256"][:16],
    }


# ---- G0-6 leftover obligation: full-table replay at the old endpoints (E-fix REF_BITEQ) ----

def probe_mode_seq(windows) -> str:
    return "".join("FDR"[int(w["opt"])] + ("†" if w["reason"] == "death" else "")
                   for w in windows)


def g0_6_replay():
    lo, hi = R8_FINAL_SEEDS
    seeds = list(range(lo, hi + 1))
    for name, spec in R8_FINAL.items():
        archive_path = EVAL / f"{name}.json"
        actual_archive_sha = sha256(archive_path)
        require(actual_archive_sha == spec["archive_sha256"],
                f"R8 final-exam archive byte drift: {name}:{actual_archive_sha}")
        # the old archives are products of the pre-E-fix protocol bundle and may not be bound to the current runtime identity; read them with the pinned byte sha +
        # the expected tag/seeds/worker (the internal agg<->rows check still runs).
        doc = read_eval_archive(
            archive_path, expected_tag=name, expected_seeds=seeds,
            expected_worker_sha256=spec["worker_sha256"],
            expected_manager_sha256=M29_SHA)
        ref = by_seed(doc["rows"], R8_FINAL_SEEDS)
        worker_zip = spec["worker_zip"]
        require(sha256(worker_zip) == spec["worker_sha256"],
                f"G0-6 replay worker sha drift: {worker_zip}")
        out = R9 / "g0" / f"g0-{name}.{time.time_ns()}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [PY, str(G0_PROBE), str(out),
               ",".join(str(s) for s in seeds),
               str(worker_zip), f"{name}.json", "old"]
        rc = run(cmd, f"g0-{name}.{time.time_ns()}.log", timeout=5_400)
        if rc != 0 or not out.exists():
            log({"event": "g0_6_crash", "archive": name, "rc": rc,
                 "note": "replay process failed; retry once under the crash clause"})
            rc = run(cmd, f"g0-{name}.{time.time_ns()}.log", timeout=5_400)
            if rc != 0 or not out.exists():
                why = f"G0-6 replay process failed repeatedly: {name} (rc={rc})"
                log({"event": "STOP", "why": why})
                attention(why)
                raise OperationalFailure(why)
        records = json.loads(out.read_text())
        require(sorted(r["seed"] for r in records) == seeds,
                f"G0-6 replay seed set incomplete: {name}")
        bad = []
        for rec in records:
            row = ref[rec["seed"]]
            if (abs(rec["ret"] - row["ret"]) > 1e-9
                    or int(rec["depth"]) != int(row["depth"])
                    or bool(rec["died"]) != bool(row["died"])
                    or probe_mode_seq(rec["windows"]) != row["mode_seq"]):
                bad.append(int(rec["seed"]))
        log({"event": "g0_6_replay", "archive": name, "n": len(records),
             "n_exact": len(records) - len(bad),
             "mismatch_seeds": bad[:20],
             "mismatch_total": len(bad),
             "records_sha16": sha16(out),
             "verdict": "REF_BITEQ" if not bad else "FAIL"})
        if bad:
            why = (f"G0-6 bit-level reconciliation mismatch: {name}, {len(bad)} seeds -- "
                   "the old-endpoint replay did not reproduce the R8 final exam; manual review under the E-fix fallback clause")
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
    log({"event": "g0_6_verdict", "status": "REF_BITEQ",
         "note": "full-table bit-level reproduction of the 512 R8 final-exam games at the old endpoints; E-fix G0-6 leftover obligation settled"})


# ---- preflight ----

def preflight():
    require_calibrated_protocol()
    require(G0_PROBE.is_file(), f"G0-6 probe missing: {G0_PROBE}")
    require(W_ZIP_SRC.is_file() and sha256(W_ZIP_SRC) == W_ZIP_SHA,
            "certified worker release missing or sha drift")
    require(M29_NPZ.is_file() and sha256(M29_NPZ) == M29_SHA,
            "M29 manager npz missing or sha drift")
    for name, spec in R8_FINAL.items():
        require((EVAL / f"{name}.json").is_file(), f"R8 final-exam archive missing: {name}")
        require(spec["worker_zip"].is_file(), f"G0-6 replay worker missing: {spec['worker_zip']}")
    staged_sha = _stage_eval_file(W_ZIP_SRC, STAGED_WORKER,
                                  expected_sha256=W_ZIP_SHA)
    tags = list(TAG_ANCHOR.values())
    tags += [f"{arm}-{pool}-{POOLS[pool][0]}" for arm in ARMS for pool in ("a", "b")]
    tags += [f"r9-final-{arm}-{POOL_FINAL[0]}" for arm in ARMS]
    for t in tags:
        require(not (EVAL / f"{t}.json").exists(),
                f"target archive already exists: {t} (restart protocol: .void it first)")
    for arm in ARMS:
        require(not (RUNS / arm).exists(), f"run directory left over: {arm} (restart protocol: archive it first)")
    log({"event": "preflight_ok", "prereg": "docs/prereg/PREREG-R9-manager-reeducation.md",
         "protocol_version": PROTOCOL_VERSION,
         "worker_zip_sha16": W_ZIP_SHA[:16], "staged_sha16": staged_sha[:16],
         "m29_sha16": M29_SHA[:16],
         "r8_archives": {name: spec["archive_sha256"][:16]
                         for name, spec in R8_FINAL.items()},
         "target_tags": tags})


# ---- main sequence ----

def main():
    try:
        R9.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(R9 / ".driver.lock", "R9 driver"):
            _main()
    except (OperationalFailure, OutputReservationError) as e:
        log({"event": "OPERATIONAL_FAILURE", "why": str(e)})
        attention("operational failure:\n" + str(e))
        raise SystemExit(2) from e
    except Exception as e:   # catch-all clause: any unexpected exception must be recorded, never a silent death
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("driver died abnormally:\n" + traceback.format_exc())
        raise


def _main():
    preflight()
    log({"event": "start", "prereg": "docs/prereg/PREREG-R9-manager-reeducation.md",
         "steps": STEPS, "arms": {n: e for n, e in ARMS.items()},
         "pools": {"a": seeds_arg(POOL_A), "b": seeds_arg(POOL_B),
                   "final": seeds_arg(POOL_FINAL)},
         "death_margin": DEATH_MARGIN, "familywise_alpha": FAMILYWISE_ALPHA,
         "note": "gold pool 9000 / held-out pools untouched; --board not used; this case burns no gold run"})

    # ---- G0-6: E-fix leftover obligation (before any new-pool consumption) ----
    g0_6_replay()

    # ---- new anchor burn (the only new-pool consumption; candidates do not exist yet, so zero contact with the anchor pools) ----
    anchors: dict[str, tuple[dict, str]] = {}
    for pool in ("a", "b", "final"):
        tag = TAG_ANCHOR[pool]
        timeout = 3_600 if pool == "final" else 1_800
        result = exam_retry(tag, POOLS[pool], M29_NPZ, "legacy-v3", timeout)
        if result is None:
            why = f"anchor burn failed repeatedly: {tag}"
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
        doc, doc_sha = result
        anchors[pool] = (doc, doc_sha)
        log({"event": "anchor", "tag": tag, "mean": doc["agg"]["ret_mean"],
             "sha16": doc_sha[:16], **depth_dashboard(doc["rows"])})

    # ---- on-the-spot derivation of the decision lines (formula + values recorded; the v31-D3-10 death line is derived per pool from the anchor inside the verdict core) ----
    ab_rows = anchors["a"][0]["rows"] + anchors["b"][0]["rows"]
    anchor_ab_mean = sum(r["ret"] for r in ab_rows) / len(ab_rows)
    abandon = round(anchor_ab_mean * ABANDON_FACTOR, 1)
    floor_repro = round(anchor_ab_mean * FLOOR_NUM / FLOOR_DEN, 1)
    anchor_wage_means = {
        pool: round(math.fsum(float(r[WAGE_KEY]) for r in anchors[pool][0]["rows"])
                    / len(anchors[pool][0]["rows"]), 3)
        for pool in anchors}
    log({"event": "derived_lines",
         "anchor_ab_mean": round(anchor_ab_mean, 3),
         "abandon_formula": f"ABANDON = anchor_ab_mean × {ABANDON_FACTOR}",
         "abandon": abandon,
         "floor_formula": f"FLOOR_REPRO = anchor_ab_mean × {FLOOR_NUM}/{FLOOR_DEN}",
         "floor_repro": floor_repro,
         "death_line_formula": "per pool: line_deaths = anchor_deaths + "
                               f"{DEATH_MARGIN}×n_pairs (derived from the anchor inside the verdict core)",
         "anchor_deaths": {pool: sum(1 for r in anchors[pool][0]["rows"] if r["died"])
                           for pool in anchors},
         "wage_ni_formula": f"per pool: wage non-inferiority line = −{WAGE_NI_FRACTION}×anchor-pool "
                            f"{WAGE_KEY} mean (derived from the anchor inside the verdict core; LCB≥line passes, endpoint included)",
         "anchor_wage_means": anchor_wage_means,
         "wage_ni_lines": {pool: round(-WAGE_NI_FRACTION * mean, 3)
                           for pool, mean in anchor_wage_means.items()},
         "gate_note": "gate limbs = depth superiority (primary metric) + wage non-inferiority + death;"
                      " ret/kills/worker_kills are record-only (re-education design decision)"})

    # ---- two arms trained in series ----
    npz: dict[str, pathlib.Path] = {}
    for name, extra in ARMS.items():
        cmd = [PY, "train/train_ppo.py", "--options", "--algo", "mppo",
               "--gamma", "1.0", "--max-steps", "3000", "--n-steps", "64",
               "--num-envs", "4", "--total-steps", str(STEPS),
               "--worker-zip", str(STAGED_WORKER),
               "--worker-zip-sha256", W_ZIP_SHA,
               "--manager-policy-observation-view", "raw-v4",
               "--run-name", name] + extra
        log({"event": "arm_start", "arm": name, "cmd_extra": extra})
        t0 = time.time()
        rc = run(cmd, f"train-{name}.log", timeout=216_000)   # 60h hang backstop (design decision of 2026-08-24: launch directly, no time limit this time; measured need ~30h per arm, this wire only catches real deadlocks)
        sp = RUNS / name / "status.json"
        try:
            steps = json.loads(sp.read_text())["total_steps"] if sp.exists() else 0
        except Exception:
            steps = 0
        nt = zip_steps(RUNS / name / "model_final.zip")   # the only step source for the completion gate (SB3 real chain)
        log({"event": "arm_done", "arm": name, "rc": rc, "nt_zip": nt,
             "steps_status": steps, "dt_min": round((time.time() - t0) / 60, 1)})
        if rc != 0 or nt != STEPS:
            why = (f"{name} training fell short (rc={rc}, nt_zip={nt}, status={steps})"
                   " -- proposition not examined; this version does not add retraining (v25 clause inherited)")
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
        out = RUNS / name / "policy.npz"
        if run([PY, "train/export_manager_npz.py",
                str(RUNS / name / "model_final.zip"), str(out)],
               f"export-{name}.log", timeout=600) != 0 or not out.exists():
            why = f"{name} npz export/parity failed"
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
        npz[name] = out
        log({"event": "g_parity", "arm": name, "npz_sha16": sha16(out)})

    # ---- arm exams (128 each on 2_123/2_124 per arm; candidate manager view raw-v4) ----
    arm_docs: dict[tuple[str, str], tuple[dict, str]] = {}
    for name in ARMS:
        for pool in ("a", "b"):
            tag = f"{name}-{pool}-{POOLS[pool][0]}"
            result = exam_retry(tag, POOLS[pool], npz[name], "raw-v4", 1_800)
            if result is None:
                why = f"{name} arm exam failed repeatedly: {tag}"
                log({"event": "STOP", "why": why})
                attention(why)
                raise OperationalFailure(why)
            doc, doc_sha = result
            arm_docs[(name, pool)] = (doc, doc_sha)
            anchor_map = by_seed(anchors[pool][0]["rows"], POOLS[pool])
            cand_map = by_seed(doc["rows"], POOLS[pool])
            paired_mean = sum(cand_map[s]["ret"] - anchor_map[s]["ret"]
                              for s in anchor_map) / len(anchor_map)
            log({"event": "arm_exam", "arm": name, "pool": pool, "tag": tag,
                 "mean": doc["agg"]["ret_mean"],
                 "paired_mean_vs_anchor": round(paired_mean, 3),
                 "sha16": doc_sha[:16], **depth_dashboard(doc["rows"])})

    # ---- early abandon gate (derived line ABANDON; all four readings below the line → training failed, no statistics) ----
    exam_means = {f"{name}:{pool}": arm_docs[(name, pool)][0]["agg"]["ret_mean"]
                  for name in ARMS for pool in ("a", "b")}
    tripped = all(v < abandon for v in exam_means.values())
    log({"event": "abandon_check", "abandon": abandon, "means": exam_means,
         "tripped": tripped})
    if tripped:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"both arms on both pools <{abandon} (= anchor×{ABANDON_FACTOR}) -- training failed,"
                    " re-education proposition not examined (no final exam)"})
        attention("verdict: training failed, proposition not examined (ABANDON derived line)")
        return

    # ---- paired verdict (same machinery and formulas as r8; development semantics, two pools per arm) ----
    analyses: dict[tuple[str, str], dict] = {}
    for name in ARMS:
        for pool in ("a", "b"):
            doc, doc_sha = arm_docs[(name, pool)]
            anchor_doc, anchor_sha = anchors[pool]
            analysis = paired_judgment(
                anchor_doc, doc, baseline_sha256=anchor_sha,
                candidate_sha256=doc_sha, phase="development")
            analyses[(name, pool)] = analysis
            log({"event": "paired_analysis", "arm": name, "pool": pool,
                 "status": analysis["verdict"]["status"],
                 **analysis_digest(analysis)})

    # ---- replication gate / eligibility and winner (largest paired mean difference among passing arms; within the band fewer deaths, then mfresh) ----
    quals = {}
    for name in ARMS:
        both_pass = all(analyses[(name, pool)]["verdict"]["status"] == "PASS"
                        for pool in ("a", "b"))
        pooled_rows = (arm_docs[(name, "a")][0]["rows"]
                       + arm_docs[(name, "b")][0]["rows"])
        pooled_mean = sum(r["ret"] for r in pooled_rows) / len(pooled_rows)
        anchor_ab = {**by_seed(anchors["a"][0]["rows"], POOL_A),
                     **by_seed(anchors["b"][0]["rows"], POOL_B)}
        cand_ab = {**by_seed(arm_docs[(name, "a")][0]["rows"], POOL_A),
                   **by_seed(arm_docs[(name, "b")][0]["rows"], POOL_B)}
        paired_diffs = [cand_ab[s]["ret"] - anchor_ab[s]["ret"]
                        for s in sorted(anchor_ab)]
        depth_diffs = [cand_ab[s]["depth"] - anchor_ab[s]["depth"]
                       for s in sorted(anchor_ab)]
        quals[name] = {
            "both_pools_pass": both_pass,
            # the primary-metric reading leads the line (re-education decision item 3); also the winner-selection quantity (final design decision).
            "depth_paired_mean_ab": round(
                sum(depth_diffs) / len(depth_diffs), 3),
            "depth_paired_wins_ab": sum(d > 0 for d in depth_diffs),
            "pooled_mean": round(pooled_mean, 3),
            "floor_pass": pooled_mean >= floor_repro,
            "paired_mean_ab": round(sum(paired_diffs) / len(paired_diffs), 3),
            "paired_wins_ab": sum(d > 0 for d in paired_diffs),
            "pooled_died": sum(1 for r in pooled_rows if r["died"]),
        }
    log({"event": "quals", **{n: quals[n] for n in ARMS}})

    pool_pass = [n for n in ARMS if quals[n]["both_pools_pass"]]
    if not pool_pass:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": "both arms failed the replication gate (depth superiority / wage non-inferiority / death limbs did not all pass)"
                        " -- no winner; the re-education proposition is unanswered (outside the power)",
             "arms": quals})
        attention("verdict: both arms failed the replication gate, no winner (depth instruments already recorded with arm_exam)")
        return
    prelim = max(ARMS, key=lambda n: quals[n]["depth_paired_mean_ab"])
    if prelim not in pool_pass:
        log({"event": "substitution", "blocked": prelim,
             "why": quals[prelim],
             "note": "the depth paired-mean-difference winner was blocked by the replication gate; the next passing arm takes its place (same as v29 D3-2)"})
    winner = select_winner(quals, pool_pass)
    log({"event": "winner", "arm": winner, **quals[winner],
         "selection_rule": f"largest paired depth mean difference; within the {WINNER_DEPTH_TIE_BAND} band"
                           " ties → (1) fewer deaths (2) mfresh (final design decision; the ret band has left the selection,"
                           " recorded in quals, not ruled on)",
         "substituted": winner != prelim})

    # ---- reproduction floor (derived line FLOOR; winner's combined A+B mean) ----
    floor_note = (" (borderline: within 1.0 of FLOOR)"
                  if abs(quals[winner]["pooled_mean"] - floor_repro)
                  <= NEAR_LINE_FLOOR_GAP else "")
    if not quals[winner]["floor_pass"]:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"winner {quals[winner]['pooled_mean']} < {floor_repro}"
                    f" (= anchor×{FLOOR_NUM}/{FLOOR_DEN}){floor_note} -- retraining did not reproduce"
                    " the reference level; re-education proposition not examined"})
        attention("verdict: reference level not reproduced (FLOOR derived line)")
        return
    log({"event": "floor_check", "winner": winner,
         "pooled_mean": quals[winner]["pooled_mean"],
         "floor": floor_repro, "passed": True, "near_line": bool(floor_note)})

    # ---- winner's final exam (2_125, one-off 256 pairs) + final verdict (final semantics) ----
    final_tag = f"r9-final-{winner}-{POOL_FINAL[0]}"
    result = exam_retry(final_tag, POOL_FINAL, npz[winner], "raw-v4", 3_600)
    if result is None:
        why = f"final exam failed repeatedly: {final_tag}"
        log({"event": "STOP", "why": why})
        attention(why)
        raise OperationalFailure(why)
    final_doc, final_sha = result
    log({"event": "final_exam", "arm": winner, "tag": final_tag,
         "mean": final_doc["agg"]["ret_mean"], "sha16": final_sha[:16],
         **depth_dashboard(final_doc["rows"])})
    anchor_final_doc, anchor_final_sha = anchors["final"]
    final_analysis = paired_judgment(
        anchor_final_doc, final_doc, baseline_sha256=anchor_final_sha,
        candidate_sha256=final_sha, phase="final")
    log({"event": "final_analysis", "arm": winner,
         "status": final_analysis["verdict"]["status"],
         **analysis_digest(final_analysis)})

    # ---- depth side verdict (scientific conclusion; "depth unlocked" requires all three fool-proofing instruments to move) ----
    anchor_rows = anchor_final_doc["rows"]
    final_rows = final_doc["rows"]
    gauges = gauge_report(anchor_rows, final_rows, TAG_ANCHOR["final"])
    anchor_l3 = l3_plus(anchor_rows)
    cand_l3 = l3_plus(final_rows)
    anchor_rate = gauges["dive_window_success_rate"]["anchor"]
    cand_rate = gauges["dive_window_success_rate"]["candidate"]
    rate_moved = (anchor_rate is not None and cand_rate is not None
                  and cand_rate > anchor_rate)
    if cand_l3 > anchor_l3 and rate_moved:
        depth_verdict = ("the depth economy moved (L3+ up and success rate up); 'depth unlocked' may only be concluded"
                         " after fool-proofing instruments 1/3 (median hand-over / residence ratio) are re-checked by probe in the same direction")
    elif cand_l3 <= anchor_l3:
        depth_verdict = f"re-education did not unlock depth (L3+ {cand_l3} ≤ anchor {anchor_l3})"
    else:
        depth_verdict = (f"out of band (L3+ {anchor_l3}→{cand_l3}, success rate "
                         f"{anchor_rate}→{cand_rate}), recorded without narrative")
    log({"event": "depth_verdict", "anchor_l3": anchor_l3, "candidate_l3": cand_l3,
         "anchor_depth_hist": depth_hist(anchor_rows),
         "candidate_depth_hist": depth_hist(final_rows),
         "gauges": gauges, "verdict": depth_verdict,
         "note": "side verdict; throne/release decisions are not part of this case (against over-narration, B8: no gold run burned)"})

    # ---- final verdict recorded (the main verdict line leads with the depth reading; re-education decision item 3) ----
    final_status = final_analysis["verdict"]["status"]
    near = final_analysis["near_line_notes"]
    digest = analysis_digest(final_analysis)
    depth_r = digest["depth"]
    wage_r = digest["wage_ni"]
    death_r = digest["death"]
    verdict_text = (
        f"R9 final verdict {final_status}: depth paired mean difference {depth_r['paired_mean']:+.3f}"
        f" (LCB {depth_r['lcb']:+.3f}, sign {depth_r['wins']}/"
        f"{depth_r['non_ties']}, p={depth_r['sign_p']:.3g}; "
        f"L3+ {anchor_l3}→{cand_l3})"
        f"; wage non-inferiority {'passed' if wage_r['passed'] else 'failed'}"
        f" (LCB {wage_r['lcb']:+.3f} vs line {wage_r['line']:+.3f})"
        f"; death {'passed' if death_r['passed'] else 'failed'}"
        f" (candidate {death_r['candidate_deaths']} vs anchor "
        f"{death_r['baseline_deaths']}, derived line {death_r['derived_line']:.1f},"
        f" UCB {death_r['mcnemar_ucb']})"
        + ("" if final_status == "PASS"
           else f"; failed limbs: {final_analysis['verdict']['failed_checks']}")
        + f"; depth side verdict: {depth_verdict}"
        + (f"; borderline notes: {near}" if near else ""))
    log({"event": "VERDICT_FINAL", "status": final_status, "arm": winner,
         **digest,
         "verdict": verdict_text,
         "candidate_depth_hist": depth_hist(final_rows),
         "anchor_depth_hist": depth_hist(anchor_rows),
         "candidate_died_seeds": died_seeds(final_rows),
         "anchor_died_seeds": died_seeds(anchor_rows),
         "gauges": gauges,
         "note": "release/throne are decided in a separate case; losing/unexamined arms never see a new pool beyond 2_125;"
                 " gold pool 9000 untouched throughout"})
    attention(f"R9 final verdict {final_status} ({winner}); {verdict_text}")


if __name__ == "__main__":
    import sys
    # launch guard (2026-07-31 operational incident: --help was ignored and a run started; it was stopped in time,
    # zero pool consumption): this driver takes no CLI arguments and refuses any argv -- a launch must be
    # the explicit intent of a bare call, leaving no door open for a slip.
    if len(sys.argv) > 1:
        print("run_r9_reeducation takes no arguments; a bare call launches (launch requires recorded approval).",
              file=sys.stderr)
        raise SystemExit(2)
    main()
