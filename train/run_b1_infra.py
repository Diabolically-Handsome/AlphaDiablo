"""B1 "bundled evaluation infrastructure" driver (sole executor of docs/prereg/PREREG-B1-bundled-eval-infra.md;
cloned from the run_v32_sovereign.py skeleton; the constants inherit every v32 W7 constant + the four PRIORS
sha values + the five v32 CASE_RUNTIME sha values).

Stages (D2; the no-hindsight clause is enforced by stage order):
  S0 preflight (W1/W7/PRIOR_REFERENCE/BC_SD_WAIVER/W-E0/W-H8/304000+307000 ledger scan)
  S1 W-G0 zero-intrusion live run (seed 307000, 2x102,400 steps, tensor-level criteria)
  S2 V1 variance run (also RB.1 bit-level continuity; a mismatch follows RB.2 out-of-band triage)
  S3 K1/K2 two held-out reference runs (first exposure of the 8000 pool, HOLDOUT_EXPOSURE counted per run)
  S4 CANARY_SET + CRITERION_REGISTER (register first, open later)
  S5 P8 leg launch (ctrl recipe verbatim, five enumerated deviations; ledger budget 2)
  S6 offline canary sequence (all before L1) + dead-gate would-trip readings (record only, never ruled on)
  S7 five leg exams L1-L5 (dual-manager bundled channel + MS_REPORT)
  S8 CRITERION_VALIDATE (8000-pool catastrophe-criterion validation, schema pinned)
  S9 R-line scorecard (RB.1-RB.10; the verdict appendix is written by the operator)

**This is a measurement and instrumentation case: zero gold seeds, zero title changes, zero releases, zero environment intrusion;
every dead-gate reading is record-only; the M29 track is never a decision line.**

Usage:
  .venv/bin/python train/run_b1_infra.py            # whole case (S0-S9, idempotent resume)
  .venv/bin/python train/run_b1_infra.py --plan     # print the build/launch plan only, no side effects
  .venv/bin/python train/run_b1_infra.py --smoke    # run only S1 W-G0 (freeze prerequisite; may run live
                                                    # before the freeze commit; recorded in the ledger)
Exit codes: 0 case closed/idempotent; 2 budget exhausted; 3 preflight; 4 lock conflict; 5 W-E0 pre-launch drift;
6 runtime drift during the case; 7 CASE_HALT_G0; 8 REF_DIVERGENCE; anything else P1.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
import traceback
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from eval_contract import (PROTOCOL_VERSION, OperationalFailure, OutputReservationError,
                           exclusive_lock, expected_eval_identity,
                           freeze_eval_identity, read_eval_archive,
                           strict_json_loads, verify_eval_identity)
from extract_canary_set import extract as extract_depth2

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
B1 = RUNS / "infra-b1"
LEDGER = B1 / "gate_ledger.jsonl"
EVAL = RUNS / "eval-assembled"

# ---- W7 artifact pins (full-file sha, frozen driver constants; a mismatch means P4, no launch) ----
KING_ZIP = ROOT / "train" / "models" / "v28-worker-leg1" / "model_final.zip"
# 2026-09-23: re-saved with neutral local paths (content otherwise unchanged); was 2f7bc9dd810956c3
KING_ZIP_SHA = "0c6f014da19c3bf27b208d55adc76be13fcad8bc5f744b95f4f743be97426434"
KING_NPZ = ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz"
KING_NPZ_SHA = "976b6c05edaa0a32bb30bd372782e1201c72b029cedcbb3a5bf2361d34f27f8a"
KING_STEPS = 3_497_984
H_NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
H_NPZ_SHA = "0f2264860b0960e7951efd424836b90c09c002cebca7bf8109fd669b13be63d7"  # == DEFAULT_MANAGER_SHA256
M29_NPZ = ROOT / "train" / "models" / "v29-manager-mfresh" / "policy.npz"
M29_NPZ_SHA = "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6"
BC_SD = RUNS / "bc-worker" / "policy_sd.pt"
BC_SD_SHA = "f052067a589cfcdedaf1754ae6d241d736bb97f6fc798683f395c76cb0ff98e6"   # value settled by v32 BC_REGEN (BC_SD_WAIVER clause, D0-4)
KING_SD = RUNS / "v32" / "king_anchor_sd.pt"
KING_SD_SHA = "009aaad29d2653cde3f4e8ed2fafd8861a0f1f572a140c64118df9e3fa3df35d"  # value settled by v32 KING_SD_OK

# PRIORS archive pins (fingerprint baseline, sealed before use; PRIOR_REFERENCE event goes into the ledger)
PRIORS = {
    "v32-ref-launch": (EVAL / "v32-ref-launch.json",
                       "48033577f8f124ae81fc5436eb44e5aa0bf541a4437d7fec99d8f5b1209c71fa",
                       113.0),
    "v32-ref-science": (EVAL / "v32-ref-science.json",
                        "1736185e286c1f2a98f6d4f503c72b40e322dcc31151e0e83068714c1b29f5f0",
                        140.9),
    "v32-ctrl-full32": (EVAL / "v32-ctrl-full32.json",
                        "1c3f683476d1b6124ba2c85473e4f17a12d52984d55582c6f70d3f0b3306e86c",
                        92.3),
    "v32-ctrl-m29": (EVAL / "v32-ctrl-m29.json",
                     "cd198545a2723a114a8ceaf548ce7917ed8ea89355cd15c828895dad622652af",
                     125.2),
}

# W-E0 zero environment drift assertion: CASE_RUNTIME values settled in the v32 ledger (pinned as frozen driver constants)
V32_CASE_RT = {
    "bridge": "8c45da3ea4121eab13da2b1a62ba52a189a25e6ea36833fa98d58d0b46140ba1",
    "engine": "be59473ea7db0a122350b179d1509454bd05ebf0e7d07946f24b2f57ff1890ce",
    "game_data": "63fb47d9c76484024c7640d90ab6b7ec5e13f567a7e1a917b6c03a6631d3f2b0",
    "assets": "b344b24f7743a88b5bf3dcc9635d096e41ee14172f687383f46a9002882b954d",
    "protocol": "91beb61e08198f5e4e9f0d13cf01b8025867cd8a474601c9cbd23321edcfcba2",
}
CALIBRATED_PROTOCOL_VERSION = 3

# ---- P8 leg recipe constants (D2; ctrl recipe verbatim, five enumerated deviations) ----
LEG_STEPS = 499_712
NT_TARGET = 3_997_696
BETA = 0.015625
QUANTUM = 2048                        # 512 n-steps × 4 envs
SEED = 304_000                        # deviation 1: original lineage seed 303000 + 1000 (constant, no seed search)
CALIB_PROBES = tuple(KING_STEPS + 49_152 * k for k in range(1, 11))   # deviation 2: ten-point step table
CKPT_EVERY = 98_304                   # deviation 3 (=48x2048)
SENTINEL_EVERY = 49_152               # deviation 4
DRY_ANCHOR_EVERY = 49_152             # deviation 5
RUN_NAME = "b1-p8"
CANARY_STEPS = tuple(KING_STEPS + 98_304 * k for k in range(1, 5))
# the +491,520 ckpt produced mechanically is archived as usual but is not part of the canary sequence (the point table is closed by pre-registration)
EXTRA_CKPT_STEP = KING_STEPS + 491_520
CANARY_CONTROLS = (7003, 7011)        # W-C healthy controls (P5 spike-free, negative controls)

# ---- W-G0 smoke-test constants (dedicated step table fixed in the case) ----
SMOKE_SEED = 307_000                  # not a case-leg seed (304000 avoided so the leg's early telemetry is not previewed)
SMOKE_STEPS = 102_400                 # 50 rollouts, divisible by 2048
SMOKE_END = KING_STEPS + SMOKE_STEPS  # 3,600,384
SMOKE_CALIB = (KING_STEPS + 49_152, KING_STEPS + 98_304)   # 2 calib points inside the window
SMOKE_RUNS = {"bare": "b1-smoke-bare", "knobs": "b1-smoke-knobs"}
SMOKE_TIMEOUT = 3_600

# ---- dead-gate would-trip constants (E3; record only, never ruled on; thresholds are calibration inputs) ----
CE_LINE = 0.2                         # absolute gate (expected to saturate; a re-calibration target, not a discriminator)
RATIO_LINE = 30.0                     # gradient-ratio gate (g_pg/g_ce, β=0.015625 already folded in)
DRY_REF_THRONE = 0.7515               # throne-level absolute reference (zero point of the increment gate)
DRY_REF_LINEAGE = 0.6305              # lineage reference (v28-leg1 start point, recorded in F2-P9)
DRY_TRIP_PP = 2.0                     # would-trip: increment > +2pp

# ---- criterion constants (D3; all pre-registered, frozen for the case) ----
MS_FLAG_LINE = 20.0
REC_H_LINE = -20.0                    # REC first limb: Δ_H(s) ≤ −20
REC_RECOVERY = 20.0                   # REC second limb: Δ_M29(s) ≥ Δ_H(s) + 20
MIN_JUDGEABLE_HITS = 3
TAU_FLOOR_BAND = (25.0, 40.0)

# ---- R-line band constants (D3/D5; closed interval [lo, hi]) ----
RB3_BAND = {"point": 25.0, "band": (5.0, 50.0)}
RB4_BAND = {"point": -20.0, "band": (-45.0, -2.0),
            "median_line": -5.0, "neg_line": 21}
RB6_MEAN = {"point": 30.0, "band": (8.0, 55.0)}
RB6_MS = {"R": {"max": (60.0, (20.0, 130.0)), "over": (6, (2, 14))},
          "N": {"max": (10.0, (0.0, 30.0)), "over": (1, (0, 4))}}
RB7_BAND = {"point": -18.0, "band": (-50.0, 5.0),
            "d2_point": 7, "d2_band": (2, 14)}
RB8_BAND = {"R": (0.3, (0.0, 0.7)), "N": (0.9, (0.7, 1.0))}
RB9_BAND = {"ratio_median": (40.0, (10.0, 120.0)),
            "ratio_trips": (7, (3, 10)),
            "ce_over_share": (0.8, (0.3, 1.0)),
            "dry_increment_pp": (1.5, (0.0, 4.0))}
RB10_BAND = {"throne_d": (54.0, (40.0, 65.0)),
             "leg": {"R": (79.0, (65.0, 95.0)), "N": (55.0, (40.0, 70.0))}}
REC_PRECISION = {"point": 0.7, "band": (0.4, 1.0)}

POOL_PROBE = "7000-7031"
POOL_S16 = "7000-7015"
POOL_HOLD = "8000-8031"
HOLDOUT_TAGS = ("b1-ref8k-launch", "b1-ref8k-science", "p8-hold8k", "p8-hold8k-m29")
CANARY_TAGS = tuple(f"p8-canary{k}-{m}" for k in range(1, 5) for m in ("h", "m29"))
ALL_EXAM_TAGS = (("b1-varprobe-launch",) + HOLDOUT_TAGS
                 + ("p8-s16", "p8-full32", "p8-full32-m29") + CANARY_TAGS)

CASE_RT: dict | None = None


# ======================================================================
# Ledger and shared utilities (inherited verbatim from run_v32_sovereign + case-level extensions)
# ======================================================================

def log(event: dict):
    B1.mkdir(parents=True, exist_ok=True)
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    B1.mkdir(parents=True, exist_ok=True)
    with open(B1 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def sha16(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def sha256(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OperationalFailure(message)


class PreflightFailure(RuntimeError):
    """P4: preflight failed -> no launch, report for review (exit code 3)."""


def pre(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightFailure(message)


def runtime_five(snapshot) -> dict:
    rt = snapshot["runtime"]
    return {"bridge": rt["bridge"]["sha256"], "engine": rt["engine"]["sha256"],
            "game_data": rt["content"]["game_data"]["sha256"],
            "assets": rt["content"]["assets"]["sha256"],
            "protocol": rt["python_protocol"]["sha256"]}


def assert_case_runtime(snapshot, where: str):
    require(CASE_RT is not None, "CASE_RUNTIME not settled")
    current = runtime_five(snapshot)
    if current != CASE_RT:
        log({"event": "CASE_HALT_RUNTIME_DRIFT", "where": where,
             "case": CASE_RT, "current": current})
        attention(f"case-level runtime drift ({where}); halting for review")
        raise SystemExit(6)


def run(cmd, logfile, timeout) -> int:
    B1.mkdir(parents=True, exist_ok=True)
    with open(B1 / logfile, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return 124


def zip_steps(p: pathlib.Path) -> int:
    try:
        with zipfile.ZipFile(p) as z:
            return int(json.loads(z.read("data"))["num_timesteps"])
    except Exception:
        return 0


def read_ledger() -> list[dict]:
    if not LEDGER.is_file():
        return []
    out = []
    for i, line in enumerate(LEDGER.read_text().splitlines(), 1):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise OperationalFailure(f"ledger line {i} unparsable: {exc}") from exc
    return out


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def stage_done(events, name) -> bool:
    return any(e.get("event") == name for e in events)


def leg_starts(events, leg) -> int:
    return sum(1 for e in events
               if e.get("event") == "leg_start" and e.get("leg") == leg)


def firing_count(events, tag) -> int:
    return sum(1 for e in events
               if e.get("event") == "FIRING_START" and e.get("tag") == tag)


def by_seed(rows, lo, hi) -> dict:
    m = {r["seed"]: r for r in rows}
    require(len(rows) == len(m), "abnormal seed set (duplicate seed)")
    require(set(m) == set(range(lo, hi + 1)), f"abnormal seed set (must be {lo}-{hi})")
    return m


def depth2_count(rows) -> int:
    return sum(1 for r in rows if r["depth"] >= 2)


def d_windows(row) -> int:
    return str(row["mode_seq"]).count("D")


def impl_bundle_sha16() -> str:
    import train_ppo
    return train_ppo._implementation_bundle_sha256()[:16]


# ======================================================================
# Statistical discipline tools (D3 hard constraints; pure functions, unit-tested)
# ======================================================================

def median(xs) -> float:
    s = sorted(xs)
    n = len(s)
    require(n > 0, "median input is empty")
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def binom_tail_ge(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k), exact binomial."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def sign_test(diffs) -> dict:
    neg = sum(1 for d in diffs if d < 0)
    pos = sum(1 for d in diffs if d > 0)
    ties = len(diffs) - neg - pos
    n = neg + pos
    p = binom_tail_ge(max(neg, pos), n) if n else 1.0
    return {"neg": neg, "pos": pos, "ties": ties,
            "p_one_sided": round(p, 4)}


def deleveraged_mean(diff_by_seed: dict) -> dict:
    """Leave out the highest-leverage seed and recompute the mean, naming that seed (F2 headline correction 2)."""
    require(len(diff_by_seed) >= 2, "the deleveraged mean needs >=2 seeds")
    lever = max(diff_by_seed, key=lambda s: abs(diff_by_seed[s]))
    rest = [v for s, v in diff_by_seed.items() if s != lever]
    return {"dropped_seed": lever,
            "dropped_value": round(diff_by_seed[lever], 2),
            "mean": round(sum(rest) / len(rest), 2)}


def band_judge(x: float, lo: float, hi: float, integer: bool = False) -> dict:
    """Closed-band reading + borderline clause (D3: continuous |x-boundary|<=0.05 x bandwidth; counts within 1 of the boundary)."""
    in_band = lo <= x <= hi
    if integer:
        borderline = min(abs(x - lo), abs(x - hi)) <= 1
    else:
        borderline = min(abs(x - lo), abs(x - hi)) <= 0.05 * (hi - lo)
    out = {"x": round(float(x), 4), "band": [lo, hi], "in_band": in_band}
    if borderline:
        out["borderline_note"] = "borderline + line not recalibrated (mandatory note per the D3 borderline clause)"
    return out


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial 95% interval (D3: a precision verdict must carry it; no bare point estimates).
    Lower bound: solve cdf(k-1, p) = 1 - α/2; upper bound: solve cdf(k, p) = α/2;
    cdf(kk, p) is decreasing in p, so bisection suffices."""
    require(0 <= k <= n and n > 0, "clopper_pearson invalid input")

    def cdf(kk: int, p: float) -> float:      # P(X <= kk | n, p)
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                   for i in range(0, kk + 1))

    def solve(kk: int, target: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if cdf(kk, mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(k - 1, 1 - alpha / 2)
    upper = 1.0 if k == n else solve(k, alpha / 2)
    return (round(lower, 4), round(upper, 4))


def ms_vectors(leg_h: dict, leg_m29: dict, ref_h: dict, ref_m29: dict) -> dict:
    """MS(s) := |Δ_H(s) − Δ_M29(s)|, Δ_m(s) = ret_leg,m − ret_ref,m (paired by manager).
    The signed Δ_H−Δ_M29 is reported alongside (statistics m-7, so an 'M29 side worse' flip is not swallowed by the absolute value)."""
    seeds = sorted(leg_h)
    require(set(seeds) == set(leg_m29) == set(ref_h) == set(ref_m29),
            "MS: the four archives have different seed sets")
    signed = {}
    for s in seeds:
        dh = leg_h[s]["ret"] - ref_h[s]["ret"]
        dm = leg_m29[s]["ret"] - ref_m29[s]["ret"]
        signed[s] = round(dh - dm, 2)
    ms = {s: abs(v) for s, v in signed.items()}
    over = sorted(s for s, v in ms.items() if v > MS_FLAG_LINE)
    return {"signed_dh_minus_dm": signed,
            "ms_max": round(max(ms.values()), 2),
            "ms_median": round(median(list(ms.values())), 2),
            "over_line_seeds": over, "n_over_line": len(over),
            "flag_line": MS_FLAG_LINE}


# ======================================================================
# Evaluation machinery (exam_or_adopt skeleton inherited + P2 ledger budgets + E1 bundled channel)
# ======================================================================

def exam(worker, tag, seeds, manager_npz=None):
    out = EVAL / f"{tag}.json"
    require(not out.exists(), f"archive immutability: {out} already exists, refusing to overwrite")
    lo, hi = (int(x) for x in seeds.split("-", 1))
    seed_values = list(range(lo, hi + 1))
    snapshot = freeze_eval_identity(ROOT, worker, manager_npz)
    assert_case_runtime(snapshot, f"exam:{tag}")          # W11 identity restated for every run
    expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
    worker_arg = (worker if snapshot["worker"]["kind"] in {"script", "bc"}
                  else snapshot["worker"]["path"])
    # E1 discipline: --manager-npz explicit every time, no default fallback; no run in this case uses --board
    cmd = [PY, "train/eval_assembled.py", "--worker", str(worker_arg),
           "--manager-npz", snapshot["manager"]["path"],
           "--seeds", seeds, "--tag", tag]
    if run(cmd, f"exam-{tag}.{time.time_ns()}.log", timeout=1_800) != 0:
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    try:
        d = read_eval_archive(out, **expected)
        verify_eval_identity(snapshot, ROOT)
    except (OSError, KeyError, TypeError, ValueError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    d["agg"]["_sha"] = sha16(out)
    return d


def validate_adopted(tag, worker, seeds, manager_npz=None):
    out = EVAL / f"{tag}.json"
    lo, hi = (int(x) for x in seeds.split("-", 1))
    snapshot = freeze_eval_identity(ROOT, worker, manager_npz)
    assert_case_runtime(snapshot, f"adopt:{tag}")
    expected = expected_eval_identity(snapshot, tag=tag,
                                      seeds=list(range(lo, hi + 1)))
    d = read_eval_archive(out, **expected)
    verify_eval_identity(snapshot, ROOT)
    d["agg"]["_sha"] = sha16(out)
    return d


def exam_case(events, worker, tag, seeds, manager_npz=None,
              bundled_with=None, extra: dict | None = None):
    """Exam-run finality clause (v32 verbatim) + P2 ledger budget (each run's FIRING_START counts, 2 allowed)."""
    out = EVAL / f"{tag}.json"
    prior = [e for e in events
             if e.get("event") == "exam_ok" and e.get("tag") == tag]
    if out.exists():
        require(bool(prior), f"{tag} leftover archive present but no exam_ok in the ledger; halting for review")
        d = validate_adopted(tag, worker, seeds, manager_npz)
        require(d["agg"]["_sha"] == prior[-1]["sha"],
                f"{tag} archive and ledger exam_ok sha mismatch")
        log({"event": "exam_adopted", "tag": tag, "sha": d["agg"]["_sha"]})
        return d
    require(not prior, f"{tag} recorded in the ledger but the archive is missing (REF_INVALID type); halting for review")
    while True:
        fired = firing_count(events, tag)
        require(fired < 2, f"{tag} evaluation run budget exhausted (ledger budget 2) -- P5 plan A halt")
        ev = {"event": "FIRING_START", "tag": tag, "attempt": fired + 1}
        log(ev)
        events.append(ev)
        d = exam(worker, tag, seeds, manager_npz)
        if d is not None:
            a = d["agg"]
            ok = {"event": "exam_ok", "tag": tag, "mean": a["ret_mean"],
                  "died": a["died"], "sha": a["_sha"]}
            if bundled_with:
                ok["bundled_with"] = bundled_with     # E1: exam_ok must also carry the paired tag
            if extra:
                ok.update(extra)
            log(ok)
            events.append({"event": "exam_ok", "tag": tag, "sha": a["_sha"]})
            return d
        log({"event": "exam_crash", "tag": tag,
             "note": "evaluation failed; re-exam within the P2 budget (a failed pre-launch assertion always halts, no retry)"})


def holdout_account(events, tag):
    """HOLDOUT_EXPOSURE counted per run (W-H8 rule; the exposure budget does not regenerate)."""
    if any(e.get("event") == "HOLDOUT_EXPOSURE" and e.get("tag") == tag
           for e in events):
        return
    n = sum(1 for e in events if e.get("event") == "HOLDOUT_EXPOSURE") + 1
    ev = {"event": "HOLDOUT_EXPOSURE", "tag": tag, "cumulative_shots": n,
          "note": "counted per run; first exposure in this case is 2 pairs/4 runs (D0-2, approved 2026-07-17); "
                  "'at most one bundled pair per case' applies after this case closes"}
    log(ev)
    events.append(ev)


def bundled_exam(events, worker, tag_h, tag_m29, seeds, holdout=False):
    """E1 dual-manager bundled evaluation channel (driver level): every worker exam pairs H and M29 automatically."""
    d_h = exam_case(events, worker, tag_h, seeds, manager_npz=None,
                    bundled_with=tag_m29)
    if holdout:
        holdout_account(events, tag_h)
    d_m = exam_case(events, worker, tag_m29, seeds, manager_npz=str(M29_NPZ),
                    bundled_with=tag_h)
    if holdout:
        holdout_account(events, tag_m29)
    return d_h, d_m


def ms_report(events, pool, tag_h, tag_m29, leg_h, leg_m29, ref_h, ref_m29,
              ref_names):
    if any(e.get("event") == "MS_REPORT" and e.get("pool") == pool
           for e in events):
        return
    v = ms_vectors(leg_h, leg_m29, ref_h, ref_m29)
    var_zero = stage_done(events, "VARPROBE")
    ev = {"event": "MS_REPORT", "pool": pool, "tags": [tag_h, tag_m29],
          "refs": ref_names, **v,
          "family_note": "per-seed MS flags form a family of 32 comparisons (family accounting note, statistics M-6)"
                         + ("; RB.2 found variance=0 -> flags carry no measurement-noise false positives" if var_zero
                            else ""),
          "ms_limitation": "MS caveat (mandatory with the verdict): M29 and H share the exhausted blind spot"
                           " and the 7017-type window-choice blind spot; MS≈0 does not show the worker is undamaged -- MS is a"
                           " detector of 'manager-sensitive damage', not of all damage"}
    log(ev)
    events.append(ev)


# ======================================================================
# E4 replay channel (OBS_DRIFT) and E2 canaries
# ======================================================================

def run_obsdrift(worker_npz, archive: pathlib.Path, out: pathlib.Path,
                 manager=None) -> dict:
    """Archive first, then replay and reconcile (fidelity anchor order fixed, engineering m-8); an existing report file is adopted idempotently."""
    if out.exists():
        report = strict_json_loads(out.read_bytes())
        if (report.get("archive_sha256") == sha256(archive)
                and report.get("fidelity_ok")):
            return report
        out.rename(out.with_suffix(f".{time.time_ns()}.stale"))
    cmd = [PY, "train/probe_b1_obsdrift.py", "--worker", str(worker_npz),
           "--archive", str(archive), "--out", str(out)]
    if manager:
        cmd += ["--manager", str(manager)]
    rc = run(cmd, f"obsdrift-{out.stem}.{time.time_ns()}.log", 3_600)
    require(rc == 0 and out.exists(),
            f"E4 replay failed / fidelity mismatch: {archive.name} (fidelity anchor clause; halting for review)")
    return strict_json_loads(out.read_bytes())


def throne_replay() -> dict:
    return run_obsdrift(KING_NPZ, PRIORS["v32-ref-launch"][0],
                        B1 / "replay" / "throne-ref-launch.json")


def walk_subset_mean(report: dict, seeds) -> float | None:
    entries = [report["per_seed"][str(s)]["walk_end"] for s in seeds
               if str(s) in report["per_seed"]]
    total = sum(e["n"] for e in entries)
    if not total:
        return None
    return round(sum(e["walk_sum_over_121_mean"] * e["n"]
                     for e in entries if e["n"]) / total, 3)


def obs_drift_event(events, at: str, leg_report: dict, throne_report: dict,
                    d_c_seeds):
    if any(e.get("event") == "OBS_DRIFT" and e.get("at") == at
           for e in events):
        return
    diffs = {}
    for dim, leg_stats in leg_report["appended8_window_end"].items():
        th = throne_report["appended8_window_end"][dim]
        diffs[dim] = (None if leg_stats is None or th is None
                      else round(leg_stats["P50"] - th["P50"], 6))
    ev = {"event": "OBS_DRIFT", "at": at,
          "schema": "8 appended dims, each {mean,std,min,max,P5,P50,P95} + window-end walkable"
                    " shape {window-end mean of walkSum/121, south-west band mean} (local-map indices 44-164);"
                    " drift reading := per-dim P50 difference, leg side minus throne side (closed enumeration)",
          "appended8_leg": leg_report["appended8_window_end"],
          "appended8_throne": throne_report["appended8_window_end"],
          "p50_diff_leg_minus_throne": diffs,
          "walk": {"throne_d_decision": throne_report["walk_d_decision"],
                   "leg_window_end_all": leg_report["walk_window_end"],
                   "leg_window_end_dc_seeds": {
                       "seeds": sorted(d_c_seeds),
                       "walk_sum_over_121_mean": walk_subset_mean(
                           leg_report, d_c_seeds)}},
          "reports": {"leg_sha16": hashlib.sha256(json.dumps(
                          leg_report, sort_keys=True).encode()).hexdigest()[:16],
                      "throne_sha16": hashlib.sha256(json.dumps(
                          throne_report, sort_keys=True).encode()).hexdigest()[:16]},
          "note": "P1 revised-ruling caveat: the instrument dims (8 appended dims) only flip marginal windows; the shape dims are the main suspect;"
                  " the F-lock mechanism lives in the base 295-dim shape, so monitoring covers both; record only"}
    log(ev)
    events.append(ev)


def canary_eval_done(events, tag) -> dict | None:
    hits = [e for e in events
            if e.get("event") == "CANARY_EVAL" and e.get("tag") == tag]
    return hits[-1] if hits else None


def canary_exam(events, worker_npz, tag, manager_npz, l1_fired: bool):
    """E2 canary run: does not use the evaluation run budget; one archive per ckpt x manager is final;
    retries only for operational failures without an archive (counted in OPERATIONAL-canary); the make-up evaluation deadline = before the first L1
    FIRING_START; a failure is recorded only and never stops the leg (P-canary)."""
    out = EVAL / f"{tag}.json"
    prior = canary_eval_done(events, tag)
    if prior is not None:
        require(out.exists(), f"canary {tag} recorded in the ledger but the archive is missing; halting for review")
        d = validate_adopted(tag, worker_npz, POOL_PROBE, manager_npz)
        require(d["agg"]["_sha"] == prior["sha"],
                f"canary {tag} archive and ledger sha mismatch")
        return d, False
    if out.exists():
        # leftover archive from a driver crash between writing the archive and the ledger entry: adopted after identity re-check (finality clause)
        d = validate_adopted(tag, worker_npz, POOL_PROBE, manager_npz)
        return d, True
    if l1_fired:
        log({"event": "OPERATIONAL-canary", "tag": tag,
             "why": "make-up evaluation deadline passed (before the first L1 FIRING_START); no make-up -- "
                    "the R line for this point reads 'undecidable, recorded as is'"})
        return None, False
    retries = 0
    while retries < 2:
        d = exam(worker_npz, tag, POOL_PROBE, manager_npz)
        if d is not None:
            return d, True
        retries += 1
        log({"event": "OPERATIONAL-canary", "tag": tag, "retry": retries,
             "why": "operational failure without an archive; retry (counted in the ledger)"})
    log({"event": "OPERATIONAL-canary", "tag": tag,
         "why": "still failing after retry -- record only, the leg continues; the R line for this point reads 'undecidable'"})
    return None, False


def canary_stage(events, d_c_seeds):
    """S6: offline canary sequence (checkpoint offline channel; record only; all before L1)."""
    l1_fired = firing_count(events, "p8-s16") > 0
    throne_rep = throne_replay()
    canary_docs = {}
    for k, step in enumerate(CANARY_STEPS, 1):
        ckpt = RUNS / RUN_NAME / "ckpt" / f"model_{step}_steps.zip"
        npz = B1 / "canary" / f"policy_{step}.npz"
        if not npz.exists():
            require(ckpt.is_file(), f"canary ckpt missing: {ckpt}")
            # per-ckpt offline export in a separate out-of-case process (existing export_worker_npz, with built-in parity)
            rc = run([PY, "train/export_worker_npz.py", str(ckpt), str(npz)],
                     f"canary-export-{step}.log", 600)
            if rc != 0 or not npz.exists():
                log({"event": "OPERATIONAL-canary", "ckpt_step": step,
                     "why": f"npz export failed (rc={rc}); every reading at this point is 'undecidable'"})
                continue
        replay_rep = None
        for mtag, manager in (("h", None), ("m29", str(M29_NPZ))):
            tag = f"p8-canary{k}-{mtag}"
            d, fresh = canary_exam(events, str(npz), tag, manager, l1_fired)
            if d is None:
                continue
            canary_docs[tag] = d
            if mtag == "h":
                replay_rep = run_obsdrift(
                    npz, EVAL / f"{tag}.json",
                    B1 / "replay" / f"{tag}.json")
                obs_drift_event(events, tag, replay_rep, throne_rep, d_c_seeds)
            if fresh or canary_eval_done(events, tag) is None:
                rows = sorted(d["rows"], key=lambda r: r["seed"])
                tau_note = None
                if mtag == "h" and replay_rep is not None:
                    tau_note = {str(r["seed"]): replay_rep["per_seed"]
                                .get(str(r["seed"]), {}).get("tau_floor_distance")
                                for r in rows}
                ev = {"event": "CANARY_EVAL", "tag": tag, "k": k,
                      "ckpt_step": step,
                      "manager": "H" if mtag == "h" else "M29",
                      "sha": d["agg"]["_sha"],
                      "mean": d["agg"]["ret_mean"], "died": d["agg"]["died"],
                      "seeds": [r["seed"] for r in rows],
                      "ret": [r["ret"] for r in rows],
                      "died_vec": [int(r["died"]) for r in rows],
                      "depth": [r["depth"] for r in rows],
                      "d_windows": [d_windows(r) for r in rows],
                      "farm_tau_mean": [r["farm_tau_mean"] for r in rows],
                      "tau_floor_distance_footnote": tau_note,
                      "footnote": "near-miss distance of the τ median from the 25 floor (statistics m-9,"
                                  " record only; P5-style replay forensics explicitly out of scope;"
                                  " the τ median uses the H-side E4 replay definition)",
                      "discipline": "record only; never a basis for any in-leg intervention;"
                                    " the only end checkpoint = nt 3,997,696"}
                log(ev)
                events.append({"event": "CANARY_EVAL", "tag": tag,
                               "sha": d["agg"]["_sha"]})
    return canary_docs


def deadgate_stage(events):
    """S6b: dead-gate would-trip dual readings (E3; record only, never ruled on; thresholds are calibration inputs)."""
    leg_dir = RUNS / RUN_NAME
    calib_path = leg_dir / "calib.jsonl"
    require(calib_path.is_file(), "calib.jsonl missing (E3 ten-point step table not produced)")
    done_steps = {e.get("step") for e in events
                  if e.get("event") == "GATE_WOULD_TRIP"
                  and e.get("gate") == "calib"}
    ratios = []
    ce_values = []
    for line in calib_path.read_text().splitlines():
        rec = json.loads(line)
        ratio = (rec["g_pg"] / rec["g_ce"]) if rec["g_ce"] else None
        ratios.append(ratio)
        ce_values.append(rec["distill_ce"])
        if rec["step"] in done_steps:
            continue
        ev = {"event": "GATE_WOULD_TRIP", "gate": "calib", "step": rec["step"],
              "distill_ce": rec["distill_ce"], "ce_line": CE_LINE,
              "would_trip_ce": rec["distill_ce"] > CE_LINE,
              "g_pg": rec["g_pg"], "g_ce": rec["g_ce"],
              "grad_ratio": None if ratio is None else round(ratio, 2),
              "ratio_line": RATIO_LINE,
              "would_trip_ratio": (ratio is not None and ratio > RATIO_LINE),
              "definition": "ratio := g_pg/g_ce from calib.jsonl, g_ce = ‖∇(β·distill_ce)‖"
                     " (β=0.015625 folded in; reading it as the raw literal CE gradient is off by 64x, not allowed)",
              "note": "record only; the absolute gate is a re-calibration target, not a discriminator; saturation is expected,"
                      " and the verdict must not narrate it as an anomaly (engineering m-2)"}
        log(ev)
        events.append(ev)
    sentinel_path = leg_dir / "sentinel.jsonl"
    require(sentinel_path.is_file(), "sentinel.jsonl missing (E3 denser sentinel not produced)")
    dry_lines = [json.loads(x) for x in sentinel_path.read_text().splitlines()
                 if '"dry-anchor"' in x]
    done_dry = {e.get("step") for e in events
                if e.get("event") == "GATE_WOULD_TRIP"
                and e.get("gate") == "dry-anchor"}
    for rec in dry_lines:
        if rec["step"] in done_dry:
            continue
        inc_pp = round((rec["mismatch"] - DRY_REF_THRONE) * 100, 2)
        ev = {"event": "GATE_WOULD_TRIP", "gate": "dry-anchor",
              "step": rec["step"], "mismatch": rec["mismatch"],
              "ref_throne": DRY_REF_THRONE, "ref_lineage": DRY_REF_LINEAGE,
              "increment_pp": inc_pp, "trip_line_pp": DRY_TRIP_PP,
              "would_trip": inc_pp > DRY_TRIP_PP,
              "note": "both references pinned (statistics M-4 / engineering m-3): increment-gate zero point 0.7515"
                      " (throne level); lineage reference 0.6305 (v28-leg1 start) -- the drift predates"
                      " v32; the +12pp lineage fact is recorded with the calibration data so later cases do not gate on a misleading line;"
                      " first run after the callback crosses the boundary, first point is not the entry step, recorded as is; record only"}
        log(ev)
        events.append(ev)
    return {"ratios": ratios, "ce_values": ce_values, "dry_lines": dry_lines}


# ======================================================================
# S1 W-G0 zero-intrusion live run (freeze prerequisite; tensor-level criteria)
# ======================================================================

def _smoke_cmd(variant: str) -> list[str]:
    """Smoke-test commands (ctrl recipe verbatim, only seed/steps changed; the knobs side adds the full instrument knob set)."""
    cmd = [PY, "train/train_ppo.py", "--worker", "--algo", "mppo",
           "--gamma", "1.0", "--max-steps", "3000", "--n-steps", "512",
           "--num-envs", "4", "--lr", "3e-4", "--ent-coef", "0.005",
           "--seed", str(SMOKE_SEED), "--total-steps", str(SMOKE_STEPS),
           "--run-name", SMOKE_RUNS[variant],
           "--distill-beta", str(BETA),
           "--teacher-sd", str(BC_SD), "--teacher-override", str(KING_SD),
           "--skip-dry", "--manager-npz", str(M29_NPZ),
           "--resume-from", str(KING_ZIP), "--allow-legacy-resume",
           "--no-drink-sovereignty", "--calib-record-only"]
    if variant == "knobs":
        cmd += ["--calib-probes", ",".join(str(x) for x in SMOKE_CALIB),
                "--ckpt-every-steps", str(CKPT_EVERY),
                "--sentinel-every", str(SENTINEL_EVERY),
                "--dry-anchor-every", str(DRY_ANCHOR_EVERY)]
    return cmd


def _load_zip_states(path: pathlib.Path):
    import torch
    with zipfile.ZipFile(path) as z:
        policy = torch.load(io.BytesIO(z.read("policy.pth")),
                            map_location="cpu", weights_only=True)
        optim = torch.load(io.BytesIO(z.read("policy.optimizer.pth")),
                           map_location="cpu", weights_only=True)
    return policy, optim


def _tree_equal(a, b, path, diffs):
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.shape == b.shape and a.dtype == b.dtype
                and torch.equal(a, b)):
            diffs.append(path)
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append(f"{path}:keys")
            return
        for k in sorted(a, key=str):
            _tree_equal(a[k], b[k], f"{path}.{k}", diffs)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}:len")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _tree_equal(x, y, f"{path}[{i}]", diffs)
        return
    if a != b:
        diffs.append(path)


def _state_digest(policy, optim) -> str:
    import torch
    h = hashlib.sha256()

    def eat(node, path):
        if isinstance(node, torch.Tensor):
            h.update(path.encode())
            h.update(str(node.dtype).encode())
            h.update(str(tuple(node.shape)).encode())
            h.update(node.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(node, dict):
            for k in sorted(node, key=str):
                eat(node[k], f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, x in enumerate(node):
                eat(x, f"{path}[{i}]")
        else:
            h.update(f"{path}={node!r}".encode())

    eat(policy, "policy")
    eat(optim, "optim")
    return h.hexdigest()


def _progress_lines(run_dir: pathlib.Path) -> list[dict]:
    lines = []
    for raw in (run_dir / "progress.jsonl").read_text().splitlines():
        rec = json.loads(raw)
        rec.pop("t", None)                    # wall-clock field is not RNG-related; dropped
        lines.append(rec)
    return lines


def _final_sentinel_lines(run_dir: pathlib.Path) -> dict:
    out = {}
    p = run_dir / "sentinel.jsonl"
    if p.is_file():
        for raw in p.read_text().splitlines():
            rec = json.loads(raw)
            if rec.get("final"):
                out[rec["sentinel"]] = rec
    return out


def smoke_stage(events) -> None:
    """W-G0 zero intrusion (freeze prerequisite for the E-item knobs, live run). Idempotent: a PASS event whose impl bundle
    sha16 matches the current one skips the run; FAIL -> CASE_HALT_G0 (exit code 7), no freeze, no launch."""
    impl16 = impl_bundle_sha16()
    for e in events:
        if (e.get("event") == "G0_NULLINTRUSION" and e.get("verdict") == "PASS"
                and e.get("impl_sha16") == impl16):
            print(f"W-G0 already passed (impl {impl16}); idempotent skip", flush=True)
            return
    # smoke-test pre-assertions (artifact side; no W1 git assertion -- the smoke test is a freeze prerequisite and may run live on a dirty tree)
    pre(KING_ZIP.is_file() and sha256(KING_ZIP) == KING_ZIP_SHA, "king zip drift")
    pre(zip_steps(KING_ZIP) == KING_STEPS, "king zip step count abnormal")
    pre(sha256(M29_NPZ) == M29_NPZ_SHA, "M29 npz drift")
    pre(sha256(BC_SD) == BC_SD_SHA, "BC_SD differs from the value settled by v32 BC_REGEN")
    pre(sha256(KING_SD) == KING_SD_SHA, "KING_SD differs from the value settled in v32")
    scan_foreign_ledgers_for_seeds((SMOKE_SEED,))
    pre(not any(lo <= SMOKE_SEED + rank <= hi
                for rank in range(4)
                for lo, hi in ((7000, 7031), (8000, 8031), (9000, 9031))),
        "smoke seed collides with an evaluation pool")
    snapshot = freeze_eval_identity(ROOT, str(KING_NPZ), None)
    if runtime_five(snapshot) != V32_CASE_RT:
        log({"event": "CASE_HALT_ENV_DRIFT", "where": "smoke",
             "expected": V32_CASE_RT, "current": runtime_five(snapshot)})
        attention("W-E0 zero environment drift assertion failed (before smoke test); no freeze, no launch")
        raise SystemExit(5)

    results = {}
    for variant in ("bare", "knobs"):
        run_dir = RUNS / SMOKE_RUNS[variant]
        t0 = time.time()
        rc = run(_smoke_cmd(variant), f"smoke-{variant}.log", SMOKE_TIMEOUT)
        nt = zip_steps(run_dir / "model_final.zip")
        results[variant] = {"rc": rc, "nt": nt,
                            "dt_min": round((time.time() - t0) / 60, 1)}
        if rc != 0 or nt != SMOKE_END:
            log({"event": "G0_NULLINTRUSION", "verdict": "FAIL",
                 "impl_sha16": impl16, "runs": results,
                 "why": f"smoke {variant} fell short (rc={rc}, nt={nt}, "
                        f"target={SMOKE_END})"})
            attention("W-G0 smoke run failed; no freeze, no launch")
            raise SystemExit(7)

    # the instruments really ran ("instruments proven zero-intrusion" must not actually mean "instruments never ran")
    knobs_dir = RUNS / SMOKE_RUNS["knobs"]
    bare_dir = RUNS / SMOKE_RUNS["bare"]
    evidence = {
        "knobs_ckpt": sorted(p.name for p in (knobs_dir / "ckpt").glob("*.zip"))
        if (knobs_dir / "ckpt").is_dir() else [],
        "bare_ckpt": sorted(p.name for p in (bare_dir / "ckpt").glob("*.zip"))
        if (bare_dir / "ckpt").is_dir() else [],
        "knobs_calib_lines": (len((knobs_dir / "calib.jsonl").read_text()
                                  .splitlines())
                              if (knobs_dir / "calib.jsonl").is_file() else 0),
        "bare_calib_lines": (len((bare_dir / "calib.jsonl").read_text()
                                 .splitlines())
                             if (bare_dir / "calib.jsonl").is_file() else 0),
        "knobs_sentinel_lines": len((knobs_dir / "sentinel.jsonl").read_text()
                                    .splitlines()),
        "bare_sentinel_lines": len((bare_dir / "sentinel.jsonl").read_text()
                                   .splitlines()),
    }
    instrument_ok = (
        f"model_{KING_STEPS + CKPT_EVERY}_steps.zip" in evidence["knobs_ckpt"]
        and evidence["knobs_calib_lines"] >= 1
        and not evidence["bare_ckpt"] and evidence["bare_calib_lines"] == 0
        and sum(1 for raw in (knobs_dir / "sentinel.jsonl").read_text()
                .splitlines()
                if json.loads(raw).get("sentinel") == "v23"
                and json.loads(raw)["step"] < SMOKE_END) >= 1
        and sum(1 for raw in (knobs_dir / "sentinel.jsonl").read_text()
                .splitlines()
                if json.loads(raw).get("sentinel") == "dry-anchor"
                and json.loads(raw)["step"] < SMOKE_END) >= 1)

    # tensor-level criterion: policy state_dict + optimizer state, torch.equal per tensor
    pol_b, opt_b = _load_zip_states(bare_dir / "model_final.zip")
    pol_k, opt_k = _load_zip_states(knobs_dir / "model_final.zip")
    tensor_diffs: list[str] = []
    _tree_equal(pol_b, pol_k, "policy", tensor_diffs)
    _tree_equal(opt_b, opt_k, "optim", tensor_diffs)
    digest_bare = _state_digest(pol_b, opt_b)
    digest_knobs = _state_digest(pol_k, opt_k)

    # RNG-related telemetry field by field (step/loss/ret etc.; wall clock dropped)
    telemetry_diffs: list[str] = []
    prog_b, prog_k = _progress_lines(bare_dir), _progress_lines(knobs_dir)
    if len(prog_b) != len(prog_k):
        telemetry_diffs.append(
            f"progress line count {len(prog_b)} != {len(prog_k)}")
    else:
        for i, (a, b) in enumerate(zip(prog_b, prog_k)):
            if a != b:
                telemetry_diffs.append(f"progress[{i}]: {a} != {b}")
                if len(telemetry_diffs) >= 20:
                    break
    status_fields = ("total_steps", "leg_steps", "episodes", "target_reached")
    st_b = json.loads((bare_dir / "status.json").read_text())
    st_k = json.loads((knobs_dir / "status.json").read_text())
    for f in status_fields:
        if st_b[f] != st_k[f]:
            telemetry_diffs.append(f"status.{f}: {st_b[f]} != {st_k[f]}")
    fin_b, fin_k = _final_sentinel_lines(bare_dir), _final_sentinel_lines(knobs_dir)
    for name in ("v23", "dry-anchor"):
        if name in fin_b and name in fin_k:
            if fin_b[name] != fin_k[name]:
                telemetry_diffs.append(
                    f"final {name}: {fin_b[name]} != {fin_k[name]}")
        else:
            telemetry_diffs.append(f"final {name} line missing: "
                                   f"bare={name in fin_b}, knobs={name in fin_k}")

    verdict = ("PASS" if not tensor_diffs and not telemetry_diffs
               and instrument_ok else "FAIL")
    ev = {"event": "G0_NULLINTRUSION", "verdict": verdict,
          "impl_sha16": impl16, "seed": SMOKE_SEED,
          "steps": SMOKE_STEPS, "window": [KING_STEPS, SMOKE_END],
          "smoke_step_table": {
              "calib": list(SMOKE_CALIB),
              "ckpt": [KING_STEPS + CKPT_EVERY],
              "sentinel_first": ((KING_STEPS // SENTINEL_EVERY) + 1)
              * SENTINEL_EVERY,
              "dry_anchor_first": ((KING_STEPS // DRY_ANCHOR_EVERY) + 1)
              * DRY_ANCHOR_EVERY},
          "runs": results,
          "tensor_verdict": ("all tensors equal under torch.equal" if not tensor_diffs
                             else tensor_diffs[:20]),
          "state_digest_sha": {"bare": digest_bare, "knobs": digest_knobs},
          "telemetry_verdict": ("RNG-related fields equal field by field" if not telemetry_diffs
                                else telemetry_diffs[:20]),
          "instruments_actually_fired": instrument_ok,
          "instrument_evidence": evidence,
          "criteria_note": "file byte-level comparison dropped (SB3 zip member timestamps and pickle "
                           "serialization are not reproducible); smoke telemetry, beyond the bit-level criterion, must not feed any "
                           "go/no-go or re-education input; the IO/clock theory of extrapolating the short run to the full run"
                           " goes to residual 9"}
    log(ev)
    events.append(ev)
    if verdict != "PASS":
        attention("W-G0 zero intrusion failed: instruments go back for redesign; no freeze, no launch")
        raise SystemExit(7)


# ======================================================================
# S0 preflight (W lines)
# ======================================================================

def scan_foreign_ledgers_for_seeds(seeds: tuple[int, ...]):
    """Preflight assertion: the target seed appears in no leg_start of any earlier case ledger (hit -> halt for review)."""
    hits = []
    for ledger in sorted(RUNS.glob("*/gate_ledger.jsonl")):
        if ledger == LEDGER:
            continue
        for line in ledger.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "leg_start" and e.get("seed") in seeds:
                hits.append({"ledger": str(ledger), "seed": e["seed"]})
    pre(not hits, f"seed already appears in an earlier ledger leg_start; halting for review (no shifting to the next value): {hits}")


def holdout_virgin_scan(events):
    """W-H8 held-out pool virginity assertion (registered side; unregistered historical manual exposure is undetectable, residual 14)."""
    pattern = re.compile(r"\b80(?:[0-2][0-9]|3[01])\b")
    sanctioned = set()
    for tag in HOLDOUT_TAGS:
        if any(e.get("event") in ("exam_ok", "FIRING_START")
               and e.get("tag") == tag for e in events):
            sanctioned.add(f"{tag}.json")
    offenders = []
    for arch in sorted(EVAL.glob("*.json")):
        if arch.name in sanctioned:
            continue
        try:
            doc = json.loads(arch.read_text())
            seeds = doc.get("meta", {}).get("seeds", [])
        except (json.JSONDecodeError, UnicodeDecodeError):
            offenders.append(f"{arch.name}: unparsable")
            continue
        if any(8000 <= int(s) <= 8031 for s in seeds):
            offenders.append(arch.name)
    for board in sorted((ROOT / "train").glob("leaderboard*.md")):
        if pattern.search(board.read_text()):
            offenders.append(board.name)
    for ledger in sorted(RUNS.glob("*/gate_ledger.jsonl")):
        if ledger == LEDGER:
            continue
        for line in ledger.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") in ("leg_start", "exam_ok", "FIRING_START"):
                seed_val = e.get("seed")
                if isinstance(seed_val, int) and 8000 <= seed_val <= 8031:
                    offenders.append(f"{ledger.parent.name}: {e}")
    pre(not offenders, f"W-H8 held-out pool virginity assertion failed; halting for review: {offenders}")
    if not stage_done(events, "HOLDOUT_FIRSTBURN"):
        ev = {"event": "HOLDOUT_FIRSTBURN",
              "pairs": 2, "shots": 4, "tags": list(HOLDOUT_TAGS),
              "authorization": "D0-2, approved 2026-07-17; "
                               "counted per run in HOLDOUT_EXPOSURE; "
                               "'at most one bundled pair per case' applies after this case closes",
              "evidence_grade": "registered-side virginity assertion (statistics m-5, residual 14)"}
        log(ev)
        events.append(ev)


def preflight(events):
    global CASE_RT
    pre(PROTOCOL_VERSION == CALIBRATED_PROTOCOL_VERSION, "contract version drift")
    # ---- W1 freeze notarization ----
    dirty = [l for l in git("status", "--porcelain").splitlines()
             if l != "?? train/leaderboard-assembled-v3.md"]
    pre(not dirty, f"W1: working tree not clean {dirty}")
    head = git("rev-parse", "HEAD")
    for path in ("docs/prereg/PREREG-B1-bundled-eval-infra.md", "train/run_b1_infra.py"):
        touch = git("log", "-1", "--format=%H", "--", path)
        pre(touch == head, f"W1: {path} last touched != HEAD")
    freezes = [e for e in events if e.get("event") == "FREEZE_SHA"]
    if freezes and freezes[-1]["sha"] != head:
        reason_file = B1 / "REFREEZE_REASON"
        pre(reason_file.is_file(),
            "W1: HEAD != last FREEZE_SHA in the ledger (a chained re-freeze needs a REFREEZE_REASON file)")
        ev = {"event": "FREEZE_SHA", "sha": head,
              "prev_sha": freezes[-1]["sha"],
              "reason": reason_file.read_text().strip()}
        log(ev)
        events.append(ev)
        freezes.append(ev)
    # ---- W7 artifact pins ----
    pre(KING_ZIP.is_file() and sha256(KING_ZIP) == KING_ZIP_SHA, "king zip drift")
    pre(sha256(KING_NPZ) == KING_NPZ_SHA, "king npz drift")
    pre(sha256(H_NPZ) == H_NPZ_SHA, "H npz drift (!= DEFAULT_MANAGER_SHA256)")
    pre(sha256(M29_NPZ) == M29_NPZ_SHA, "M29 npz drift")
    pre(zip_steps(KING_ZIP) == KING_STEPS, "king zip step count abnormal")
    pre(KING_SD.is_file() and sha256(KING_SD) == KING_SD_SHA,
        "KING_SD drift (the v32 artifact is reused)")
    pre(run([PY, "train/check_teacher_parity.py", str(KING_SD),
             str(KING_NPZ)], "parity-king.log", 600) == 0,
        "G-KL-W: king anchor sd and npz attestation failed (in-case 0/1000 re-check)")
    # explicit discretionary clause for reusing the v32 regenerated BC_SD (compliance M-2; not inherited by default)
    pre(BC_SD.is_file() and sha256(BC_SD) == BC_SD_SHA,
        "BC_SD differs from the value settled by v32 BC_REGEN")
    if not stage_done(events, "BC_SD_WAIVER"):
        ev = {"event": "BC_SD_WAIVER", "bc_sd_sha256": BC_SD_SHA,
              "basis": "the BC generation chain does not pass through the training-driver knobs ∧ runtime_five has zero drift"
                       " (W-E0) -- case-level discretionary waiver, listed separately in the freeze report (D0-4)"}
        log(ev)
        events.append(ev)
    # PRIORS archive pins (sealed before use)
    for name, (path, expected, mean) in PRIORS.items():
        pre(path.is_file() and sha256(path) == expected,
            f"PRIORS archive pin mismatch: {name}")
        doc = strict_json_loads(path.read_bytes())
        pre(doc["agg"]["ret_mean"] == mean,
            f"PRIORS {name} ret_mean != value settled in the ledger {mean}")
    if not stage_done(events, "PRIOR_REFERENCE"):
        ev = {"event": "PRIOR_REFERENCE",
              "priors": {n: {"sha256": s, "ret_mean": m}
                         for n, (_, s, m) in PRIORS.items()},
              "note": "fingerprint baseline of the four v32 archives, sealed before use"}
        log(ev)
        events.append(ev)
    # ---- W-E0 zero environment drift assertion (runtime_five == value settled in v32 CASE_RUNTIME) ----
    snapshot = freeze_eval_identity(ROOT, str(KING_NPZ), None)
    five = runtime_five(snapshot)
    if five != V32_CASE_RT:
        log({"event": "CASE_HALT_ENV_DRIFT", "expected": V32_CASE_RT,
             "current": five})
        attention("W-E0 environment + evaluation-protocol bundle drift; no freeze, no launch; report for review")
        raise SystemExit(5)
    CASE_RT = five
    prior_rt = [e for e in events if e.get("event") == "CASE_RUNTIME"]
    if prior_rt:
        pre(prior_rt[0]["five"] == five, "W10: resume runtime reconciliation mismatch")
    else:
        log({"event": "CASE_RUNTIME", "five": five,
             "w_e0": "identical to v32 CASE_RUNTIME (zero drift in environment and evaluation-protocol bundle)"})
        events.append({"event": "CASE_RUNTIME", "five": five})
    # ---- seed discipline scan (304000/307000) + W-H8 ----
    scan_foreign_ledgers_for_seeds((SEED, SMOKE_SEED))
    pre(not any(lo <= SEED + rank <= hi for rank in range(4)
                for lo, hi in ((7000, 7031), (8000, 8031), (9000, 9031))),
        "W-H8: training seed collides with an evaluation pool (driver-side assertion; the train_ppo guard covers only the 7000/9000 ranges)")
    pre(SEED == 303_000 + 1_000, "seed selection rule violated (D2 deviation 1: original lineage seed + 1000)")
    holdout_virgin_scan(events)
    # ---- W9 target-archive prerequisite ----
    for t in ALL_EXAM_TAGS:
        has_ledger = any(e.get("event") in ("exam_ok", "CANARY_EVAL")
                         and e.get("tag") == t for e in events)
        if not has_ledger:
            adoptable = (t in CANARY_TAGS)   # leftover canary archives use the finality adoption clause
            if not adoptable:
                pre(not (EVAL / f"{t}.json").exists(),
                    f"W9: target archive already exists: {t} (restart protocol: .void it first)")
    if leg_starts(events, RUN_NAME) == 0:
        pre(not (RUNS / RUN_NAME).exists(), f"run directory left over: {RUN_NAME}")
    if not freezes:
        log({"event": "FREEZE_SHA", "sha": head})
    log({"event": "preflight_ok", "king_zip": KING_ZIP_SHA[:16],
         "bc_sd": BC_SD_SHA[:16], "king_sd": KING_SD_SHA[:16],
         "impl_sha16": impl_bundle_sha16()})
    return head


# ======================================================================
# S2 V1 variance run (RB.1 bit-level continuity + RB.2)
# ======================================================================

def varprobe_stage(events):
    tag = "b1-varprobe-launch"
    d = exam_case(events, str(KING_NPZ), tag, POOL_PROBE, manager_npz=None,
                  extra={"note": "V1 variance run: the independent re-evaluation authorization is recorded here; record only,"
                                 " no change of anchor or reference title; V1 never replaces the paired baseline;"
                                 " the final status of the original v32-ref-launch archive and its RB.4 baseline role are unchanged"})
    ref_doc = strict_json_loads(PRIORS["v32-ref-launch"][0].read_bytes())
    old_rows = sorted(ref_doc["rows"], key=lambda r: r["seed"])
    new_rows = sorted(d["rows"], key=lambda r: r["seed"])
    require(len(old_rows) == len(new_rows) == 32, "reference row count abnormal")
    diff_seeds = [o["seed"] for o, n in zip(old_rows, new_rows) if o != n]
    core = ("ret_mean", "ret_median", "died", "depth_median", "kills_mean",
            "farm_tau_mean", "override_rate", "cap_rate")
    agg_diff = [k for k in core
                if ref_doc["agg"].get(k) != d["agg"].get(k)]
    if diff_seeds or agg_diff:
        # RB.2 out-of-band: drift triage first (full re-check of W-E0/CASE_RUNTIME)
        snapshot = freeze_eval_identity(ROOT, str(KING_NPZ), None)
        if runtime_five(snapshot) != V32_CASE_RT:
            log({"event": "CASE_HALT_ENV_DRIFT", "where": "varprobe-triage",
                 "current": runtime_five(snapshot)})
            attention("V1 mismatch with runtime drift: CASE_HALT")
            raise SystemExit(6)
        max_dret = max(abs(n["ret"] - o["ret"])
                       for o, n in zip(old_rows, new_rows))
        log({"event": "REF_DIVERGENCE", "tag": tag,
             "row_diff_seeds": diff_seeds, "agg_diff": agg_diff,
             "max_abs_dret": round(max_dret, 2),
             "triage": "runtime_five all green and the mismatch is non-systematic -- halt for a design review"
                       "; a dedicated variance case (>=k repeats) would be set up separately; a 1-degree-of-freedom n=2"
                       " reading must not become a project-wide noise band, and 'publish as is and continue' is not allowed"})
        attention("V1 variance run mismatches v32-ref-launch: REF_DIVERGENCE halt for review")
        raise SystemExit(8)
    if not stage_done(events, "VARPROBE"):
        ev = {"event": "VARPROBE", "tag": tag, "rows": 32,
              "max_abs_dret": 0.0, "band": [0, 0], "in_band": True,
              "verdict": "within-seed measurement variance = 0 (deterministic protocol, limited to this runtime world"
                         " and to the H side -- the M29 side is presumed, the note goes with the verdict, measured in a separate case,"
                         " compliance m-4 / residual 15): significance of the paired Δ must use the cross-seed"
                         " sign test; no t test in the name of measurement noise; the 'variance unknown' gap"
                         " registered in the F2 statistics is closed",
              "rb1": "RB.1 bit-level continuity PASS (launch side biteq per seed per field;"
                     " science side carried over by the PRIORS full-file sha pin; no new run for 140.9)"}
        log(ev)
        events.append(ev)
    return d


# ======================================================================
# S4 CANARY_SET + CRITERION_REGISTER (register first, open later)
# ======================================================================

def canary_set_stage(events) -> dict:
    prior = [e for e in events if e.get("event") == "CANARY_SET"]
    if prior:
        return prior[0]
    info = extract_depth2(PRIORS["v32-ref-launch"][0], CANARY_CONTROLS)
    require(info["archive_sha256"] == PRIORS["v32-ref-launch"][1],
            "CANARY_SET extraction source archive sha drift")
    ev = {"event": "CANARY_SET", "n_D": info["n_D"],
          "depth2_seeds": info["depth2_seeds"],
          "controls": info["controls"], "C": info["C"],
          "source_archive_sha256": info["archive_sha256"],
          "extractor": "train/extract_canary_set.py (E8, committed with the freeze commit)",
          "note": "frozen before the leg launch; the scoring seed set constrains only RB.5/RB.8, not the exam surface"
                  " (canary evaluation seeds = the whole 7000-7031 pool)"}
    log(ev)
    events.append(ev)
    return ev


def criterion_register_stage(events):
    if stage_done(events, "CRITERION_REGISTER"):
        return [e for e in events if e.get("event") == "CRITERION_REGISTER"][0]
    info = extract_depth2(EVAL / "b1-ref8k-launch.json", ())
    ev = {"event": "CRITERION_REGISTER",
          "criterion": "REC(s) := [Δ_H(s) ≤ −20] ∧ [Δ_M29(s) ≥ Δ_H(s) + 20]"
                        " (manager-sensitive ground truth; each pair uses the same-manager reference K1/K2)",
          "holdout_depth2_seeds": info["depth2_seeds"],
          "n_holdout_depth2": info["n_D"],
          "source_archive_sha256": info["archive_sha256"],
          "min_judgeable_hits": MIN_JUDGEABLE_HITS,
          "precision_point": REC_PRECISION["point"],
          "precision_band": list(REC_PRECISION["band"]),
          "note": "register first, open later (statistics m-2) -- the list itself is instantiated here; this event is"
                  " a restating record of the D3 criterion, not an opening for a new in-case criterion (compliance m-7);"
                  " pool-level note (statistics m-6): the REC first limb is nearly always true when the whole pool shifts down"
                  " (RB.7 point estimate −18), so the discriminating burden falls mainly on the M29 recovery limb"}
    log(ev)
    events.append(ev)
    return ev


# ======================================================================
# S5 P8 leg (the only training launch; ledger budget 2; no seed change)
# ======================================================================

def leg_cmd() -> list[str]:
    return [PY, "train/train_ppo.py", "--worker", "--algo", "mppo",
            "--gamma", "1.0", "--max-steps", "3000", "--n-steps", "512",
            "--num-envs", "4", "--lr", "3e-4", "--ent-coef", "0.005",
            "--seed", str(SEED), "--total-steps", str(LEG_STEPS),
            "--run-name", RUN_NAME, "--distill-beta", str(BETA),
            "--teacher-sd", str(BC_SD), "--teacher-override", str(KING_SD),
            "--skip-dry", "--manager-npz", str(M29_NPZ),
            "--resume-from", str(KING_ZIP), "--allow-legacy-resume",
            "--no-drink-sovereignty",
            "--calib-probes", ",".join(str(x) for x in CALIB_PROBES),
            "--calib-record-only",
            "--ckpt-every-steps", str(CKPT_EVERY),
            "--sentinel-every", str(SENTINEL_EVERY),
            "--dry-anchor-every", str(DRY_ANCHOR_EVERY)]


def leg_stage(events) -> str:
    model_path = RUNS / RUN_NAME / "model_final.zip"
    out = RUNS / RUN_NAME / "policy.npz"
    if model_path.exists() and zip_steps(model_path) == NT_TARGET:
        if not out.exists():
            require(run([PY, "train/export_worker_npz.py", str(model_path),
                         str(out)], "export-p8.retry.log", 600) == 0
                    and out.exists(), "P8 make-up export failed; halting for review")
            log({"event": "npz_exported", "leg": RUN_NAME,
                 "sha256": sha256(out), "note": "make-up export (not a relaunch, uses no budget)"})
        log({"event": "leg_skip_complete", "leg": RUN_NAME})
        return str(out)
    require(leg_starts(events, RUN_NAME) < 2,
            "P8 leg launch budget exhausted (ledger budget 2) -- OPERATIONAL_FAILURE, whole case halts;"
            " never hand-edit the ledger to keep it alive")
    ev = {"event": "leg_start", "leg": RUN_NAME, "seed": SEED,
          "recipe": "v30/v32 ctrl recipe verbatim; five enumerated deviations"
                    " (seed/ten calib points/ckpt-every/sentinel-every/dry-anchor-every)",
          "resume_from": "king zip (a relaunch continues the same proposition; seed always 304000,"
                         " no seed change on relaunch)"}
    log(ev)
    events.append(ev)
    t0 = time.time()
    rc = run(leg_cmd(), f"train-{RUN_NAME}.log", timeout=21_600)
    nt = zip_steps(model_path)
    log({"event": "leg_done", "leg": RUN_NAME, "rc": rc, "nt_zip": nt,
         "dt_min": round((time.time() - t0) / 60, 1)})
    require(rc == 0 and nt == NT_TARGET,
            f"P8 leg fell short (rc={rc}, nt={nt}, target={NT_TARGET}) -- one of the three operational gates"
            " failed; relaunch within the P3 budget (this case has no abandon gate: every score is a reading)")
    require(run([PY, "train/export_worker_npz.py", str(model_path),
                 str(out)], "export-p8.log", 600) == 0 and out.exists(),
            "P8 npz export failed (the make-up export channel uses no budget; after restart the leg_skip branch runs)")
    log({"event": "npz_exported", "leg": RUN_NAME, "sha256": sha256(out)})
    return str(out)


# ======================================================================
# S8 CRITERION_VALIDATE
# ======================================================================

def criterion_validate_stage(events, reg_ev, k1, k2, l4, l5, l4_replay):
    if stage_done(events, "CRITERION_VALIDATE"):
        return
    d2_seeds = reg_ev["holdout_depth2_seeds"]
    k1_rows = by_seed(k1["rows"], 8000, 8031)
    k2_rows = by_seed(k2["rows"], 8000, 8031)
    l4_rows = by_seed(l4["rows"], 8000, 8031)
    l5_rows = by_seed(l5["rows"], 8000, 8031)
    tau = {int(s): v.get("farm_tau_median")
           for s, v in l4_replay.get("per_seed", {}).items()}
    from probe_composite_signature import composite_signature
    sig = composite_signature(k1_rows, l4_rows, tau)
    require(sorted(sig["ref_depth2_seeds"]) == sorted(d2_seeds),
            "CRITERION_VALIDATE denominator differs from the CRITERION_REGISTER list")
    confusion = {"hit_rec": [], "hit_norec": [], "miss_rec": [], "miss_norec": []}
    for s in d2_seeds:
        dh = l4_rows[s]["ret"] - k1_rows[s]["ret"]
        dm = l5_rows[s]["ret"] - k2_rows[s]["ret"]
        rec = (dh <= REC_H_LINE) and (dm >= dh + REC_RECOVERY)
        hit = sig["per_seed"][s]["signature_hit"]
        key = f"{'hit' if hit else 'miss'}_{'rec' if rec else 'norec'}"
        confusion[key].append(s)
    tp, fp = len(confusion["hit_rec"]), len(confusion["hit_norec"])
    fn, tn = len(confusion["miss_rec"]), len(confusion["miss_norec"])
    hits = tp + fp
    if hits == 0:
        precision_verdict = ("fingerprint not reproduced in the held-out pool (hits = 0) -- judged together with RB.5;"
                             " not a reason to rerun or add runs")
        precision = None
        ci = None
    elif hits < MIN_JUDGEABLE_HITS:
        precision_verdict = (f"undecidable (too few samples: hits {hits} < "
                             f"{MIN_JUDGEABLE_HITS}); not read against the band;"
                             " downgraded and reported like RB.7")
        precision = round(tp / hits, 4)
        ci = clopper_pearson(tp, hits)
    else:
        precision = round(tp / hits, 4)
        ci = clopper_pearson(tp, hits)
        precision_verdict = band_judge(precision, *REC_PRECISION["band"])
    recall = round(tp / (tp + fn), 4) if (tp + fn) else None
    ev = {"event": "CRITERION_VALIDATE",
          "denominator": {"seeds": d2_seeds, "n": len(d2_seeds),
                          "source": "the 8000-pool "
                                    "king x H depth>=2 list embedded in CRITERION_REGISTER"},
          "confusion_2x2": {k: sorted(v) for k, v in confusion.items()},
          "tp_fp_fn_tn": [tp, fp, fn, tn],
          "precision": precision,
          "precision_ci95_clopper_pearson": ci,
          "precision_verdict": precision_verdict,
          "recall": recall,
          "recall_note": "recall is a first-exposure calibration reading with no prior band; recorded as is",
          "pool_note": "the REC first limb is nearly always true when the whole pool shifts down, so the discriminating burden falls mainly on "
                       "the M29 recovery limb (statistics m-6; the verdict carries this note as is)",
          "caveats": ["signature specificity uncalibrated (D3, mandatory with the verdict)",
                      "horizon caveat (evaluation protocol max-steps 3000)"]}
    log(ev)
    events.append(ev)
    return ev


# ======================================================================
# S9 R-line scorecard
# ======================================================================

def rb4_grid(x: float, med: float, sign: dict, delev: dict) -> dict:
    lo, hi = RB4_BAND["band"]

    def region(v):
        return "in" if lo <= v <= hi else ("below" if v < lo else "above")

    limb_s1 = med <= RB4_BAND["median_line"]
    limb_s2 = sign["neg"] >= RB4_BAND["neg_line"]
    limb_s3 = region(delev["mean"]) == region(x)
    limbs = int(limb_s1) + int(limb_s2) + int(limb_s3)
    if lo <= x <= hi:
        cell = "reproduced" if limbs >= 2 else "leverage-driven candidate"
    elif x < lo:
        cell = "amplified" if limbs >= 2 else "leverage-driven candidate (negative variant)"
    elif med <= RB4_BAND["median_line"] or sign["neg"] >= RB4_BAND["neg_line"]:
        cell = "mean-masking candidate"
    else:
        cell = "not reproduced"
    return {"x": round(x, 2), **band_judge(x, lo, hi),
            "limbs": {"S1_median": [round(med, 2), limb_s1],
                      "S2_signs": [sign["neg"], limb_s2],
                      "S3_deleveraged": [delev, limb_s3]},
            "n_limbs": limbs, "cell": cell,
            "note": "every cell is a valid conclusion; the verdict is worded per cell, never lumped as 'pass/fail';"
                    " the mean alone never decides (statistics B-2)"}


def scorecard_stage(events, docs, canary_docs, gates, reg_ev, n_D, d_c_seeds,
                    leg_replay, throne_rep):
    if stage_done(events, "VERDICT_PATH"):
        return
    ref_launch = by_seed(strict_json_loads(
        PRIORS["v32-ref-launch"][0].read_bytes())["rows"], 7000, 7031)
    l2 = by_seed(docs["p8-full32"]["rows"], 7000, 7031)
    l3 = by_seed(docs["p8-full32-m29"]["rows"], 7000, 7031)
    k1 = by_seed(docs["b1-ref8k-launch"]["rows"], 8000, 8031)
    k2 = by_seed(docs["b1-ref8k-science"]["rows"], 8000, 8031)
    l4 = by_seed(docs["p8-hold8k"]["rows"], 8000, 8031)
    card = {}
    band_outcomes = []

    def add(name, judged):
        card[name] = judged
        if isinstance(judged, dict) and "in_band" in judged:
            band_outcomes.append((name, judged["in_band"]))

    # RB.1/RB.2 already recorded by the VARPROBE event
    card["RB1_RB2"] = ("VARPROBE recorded: bit-level continuity PASS, variance band [0,0] in band"
                       if stage_done(events, "VARPROBE") else "absent (verdict deferred)")
    # RB.3 dual-manager reading difference (8000-pool throne reference)
    diffs_k = [k2[s]["ret"] - k1[s]["ret"] for s in sorted(k1)]
    rb3 = band_judge(sum(diffs_k) / 32, *RB3_BAND["band"])
    rb3.update({"point": RB3_BAND["point"], "median": round(median(diffs_k), 2),
                "sign": sign_test(diffs_k),
                "deleveraged": deleveraged_mean(
                    {s: k2[s]["ret"] - k1[s]["ret"] for s in k1})})
    add("RB3_dual_manager_8k", rb3)
    # RB.4 main reading 1 (composite main verdict)
    diff4 = {s: l2[s]["ret"] - ref_launch[s]["ret"] for s in sorted(l2)}
    x4 = sum(diff4.values()) / 32
    rb4 = rb4_grid(x4, median(list(diff4.values())),
                   sign_test(list(diff4.values())), deleveraged_mean(diff4))
    add("RB4_f2_reproduction", rb4)
    cell = rb4["cell"]
    branch = ("R" if cell in ("reproduced", "amplified") else
              "N" if cell == "not reproduced" else None)
    # RB.5 F-lock fingerprint reproduction (per the actual CANARY_SET n_D)
    from probe_composite_signature import composite_signature
    tau = {int(s): v.get("farm_tau_median")
           for s, v in leg_replay.get("per_seed", {}).items()}
    sig7k = composite_signature(ref_launch, l2, tau)
    point5 = round(0.7 * n_D)
    lo5 = math.ceil(0.55 * n_D)
    rb5 = band_judge(sig7k["n_hits"], lo5, n_D, integer=True)
    rb5.update({"point": point5, "hits": sig7k["hits"],
                "not_reproduced_line": lo5 - 1,
                "verdict": ("fingerprint not reproduced (judged together with RB.4)"
                            if sig7k["n_hits"] <= lo5 - 1 else "read against the band"),
                "caveat": "signature specificity uncalibrated (mandatory caveat, D3); lower-edge basis: the band's lower edge"
                          " still counts as reproduced when the seed set decays by nearly half (statistics M-3)"})
    add("RB5_flock_fingerprint", rb5)
    # RB.6 P8 leg dual-manager reading difference
    diffs6 = [l3[s]["ret"] - l2[s]["ret"] for s in sorted(l2)]
    rb6 = band_judge(sum(diffs6) / 32, *RB6_MEAN["band"])
    rb6["point"] = RB6_MEAN["point"]
    rb6["branch_note"] = "mean-difference line is unconditional (healthy throne +27.9 and damaged legs +29.1/+32.9 are all in band)"
    add("RB6_mean_l3_minus_l2", rb6)
    ms_ev = [e for e in events if e.get("event") == "MS_REPORT"
             and e.get("pool") == "7000"]
    if branch and ms_ev:
        msb = RB6_MS[branch]
        m = ms_ev[-1]
        rb6ms_max = band_judge(m["ms_max"], *msb["max"][1])
        rb6ms_max["point"] = msb["max"][0]
        rb6ms_over = band_judge(m["n_over_line"], *msb["over"][1], integer=True)
        rb6ms_over["point"] = msb["over"][0]
        add(f"RB6_MS_max_branch_{branch}", rb6ms_max)
        add(f"RB6_MS_overline_branch_{branch}", rb6ms_over)
    else:
        card["RB6_MS"] = "conditional line undecidable (RB.4 landed on a candidate cell); recorded as is"
    # RB.7 held-out pool transfer
    diff7 = {s: l4[s]["ret"] - k1[s]["ret"] for s in sorted(l4)}
    rb7 = band_judge(sum(diff7.values()) / 32, *RB7_BAND["band"])
    rb7.update({"point": RB7_BAND["point"],
                "median": round(median(list(diff7.values())), 2),
                "sign": sign_test(list(diff7.values())),
                "deleveraged": deleveraged_mean(diff7)})
    add("RB7_holdout_transfer", rb7)
    d2_8k = reg_ev["n_holdout_depth2"]
    rb7d = band_judge(d2_8k, RB7_BAND["d2_band"][0], RB7_BAND["d2_band"][1],
                      integer=True)
    rb7d["point"] = RB7_BAND["d2_point"]
    if d2_8k < 2:
        rb7d["verdict"] = "the signature criterion is 'not testable' in this pool; CRITERION_VALIDATE downgraded and reported"
    add("RB7_holdout_depth2_count", rb7d)
    # RB.8 canary instrument (retention; metric formula pinned)
    c4h = canary_docs.get("p8-canary4-h")
    if c4h is not None and branch:
        c4_rows = by_seed(c4h["rows"], 7000, 7031)
        kept = [s for s in d_c_seeds if d_windows(c4_rows[s]) >= 1]
        retention = len(kept) / n_D
        pt, band8 = RB8_BAND[branch]
        rb8 = band_judge(retention, *band8)
        rb8.update({"point": pt, "kept_seeds": sorted(kept), "n_D": n_D,
                    "formula": "retention := |{s in D_C: leg x H canary (+393,216 point)"
                               " archive has >= 1 D window for the seed}| / n_D"})
        add(f"RB8_canary_retention_branch_{branch}", rb8)
    else:
        card["RB8_canary_retention"] = ("undecidable (canary data missing or RB.4 landed on a candidate cell);"
                                        " recorded as is (P-canary)")
    # RB.9 dead-gate would-trip (record only, calibration input)
    ratios = [r for r in gates["ratios"] if r is not None]
    if ratios:
        rb9a = band_judge(median(ratios), *RB9_BAND["ratio_median"][1])
        rb9a["point"] = RB9_BAND["ratio_median"][0]
        add("RB9_grad_ratio_median", rb9a)
        trips = sum(1 for r in ratios if r > RATIO_LINE)
        rb9b = band_judge(trips, *RB9_BAND["ratio_trips"][1], integer=True)
        rb9b.update({"point": RB9_BAND["ratio_trips"][0],
                     "n_points": len(gates["ratios"])})
        add("RB9_ratio_would_trips", rb9b)
    ce_share = (sum(1 for c in gates["ce_values"] if c > CE_LINE)
                / max(1, len(gates["ce_values"])))
    rb9c = band_judge(ce_share, *RB9_BAND["ce_over_share"][1])
    rb9c.update({"point": RB9_BAND["ce_over_share"][0],
                 "note": "saturation is expected; this gate is a re-calibration target, not a discriminator (note goes with the verdict)"})
    add("RB9_ce_over_share", rb9c)
    final_dry = [r for r in gates["dry_lines"] if r.get("final")] or \
        gates["dry_lines"][-1:]
    if final_dry:
        inc = (final_dry[-1]["mismatch"] - DRY_REF_THRONE) * 100
        rb9d = band_judge(inc, *RB9_BAND["dry_increment_pp"][1])
        rb9d.update({"point": RB9_BAND["dry_increment_pp"][0],
                     "reading": "increment (pp) of the leg-end dry-anchor mismatch over the throne-level zero point 0.7515;"
                                " lineage note: the lineage start 0.6305 goes with the calibration data"})
        add("RB9_dry_anchor_increment_pp", rb9d)
    # RB.10 manager observation shape drift
    th_walk = throne_rep["walk_d_decision"]["walk_sum_over_121_mean"]
    rb10t = band_judge(th_walk, *RB10_BAND["throne_d"][1])
    rb10t["point"] = RB10_BAND["throne_d"][0]
    add("RB10_throne_d_walk", rb10t)
    leg_walk = walk_subset_mean(leg_replay, d_c_seeds)
    if leg_walk is not None and branch:
        pt, band10 = RB10_BAND["leg"][branch]
        rb10l = band_judge(leg_walk, *band10)
        rb10l["point"] = pt
        add(f"RB10_leg_walk_branch_{branch}", rb10l)
    else:
        card["RB10_leg_walk"] = "undecidable (RB.4 landed on a candidate cell or replay data missing); recorded as is"
    card["RB10_p50_diffs_note"] = ("per-dim P50 differences of the 8 appended dims are listed in the OBS_DRIFT event;"
                                   " the P1 revised-ruling caveat applies: instrument dims only flip marginal windows,"
                                   " the shape dims are the main suspect")

    # family-level reading clause (statistics M-6): count in-band/out-of-band over the whole table, no cherry-picking
    n_out = sum(1 for _, ok in band_outcomes if not ok)
    out_names = [n for n, ok in band_outcomes if not ok]
    family = {"n_band_readings": len(band_outcomes), "n_out_of_band": n_out,
              "out_names": out_names,
              "clause": "intended coverage ≈85%; scattered out-of-band lines (<=3, with different mechanisms) are statistically"
                        " normal and must not be narrated as a systematic anomaly; clustered out-of-band lines (>=4 in the same direction or mechanism)"
                        " escalate to NEEDS_ATTENTION"}
    if n_out >= 4:
        attention(f"R lines clustered out of band ({n_out}): {out_names}")

    # conversion clause (wording limited to N=2, statistics M-7)
    if cell == "not reproduced":
        transform = ("RB.4 landed on 'not reproduced' -> the F2 case conclusion is narrowed to 'training-seed-specific"
                     " candidate', and dead-gate re-calibration thresholds use the measured P8 leg distribution -- that calibration"
                     " can only set specificity (no false alarms on healthy legs), not sensitivity (whether catastrophes are caught);"
                     " this caveat is fixed with the conversion clause; the P8 leg itself is the first healthy leg, and its signature hit"
                     " rate becomes the first measured false-positive rate, delivered with the calibration")
    elif cell in ("reproduced", "amplified"):
        transform = ("RB.4 landed on 'reproduced' -> verdict: 'F-lock reproduced on a second training seed"
                     " (2/2, same recipe, same teacher, same manager), replication evidence 1->2',"
                     " must carry residuals 1, 9 and 10; no population-level 'replicable proposition' claim; one prerequisite"
                     " of option 4 (succession case) on the manager-inquiry menu is met; reported for scheduling")
    else:
        transform = f"RB.4 landed on '{cell}' -- conditional line reads 'undecidable, recorded as is'"

    ev = {"event": "VERDICT_PATH", "case": "B1", "golden_authorized": False,
          "scorecard": card, "family_ledger": family,
          "rb4_cell": cell, "transform_clause": transform,
          "acceptance_three_limbs": {
              "1_prescription_1_confirmed": "E1 channel + template-obligation clause recorded (gold-evaluation scope per D0-1,"
                            " approved 2026-07-17: no new rule, gold evaluation stays single-H); the obligation explicitly excludes gold evaluation",
              "2_prescription_2_calibration_data": "10 calib points / two-gate would-trip / dry-anchor"
                               " dual reference / OBS_DRIFT / full canary sequence archived,"
                               " for the case that formally gates on the re-calibrated dead gates to cite",
              "3_P8_reproduction_verdict": f"RB.4/RB.5 joint verdict: {cell} (every direction is a legal conclusion,"
                             " no tier competition)"},
          "mandatory_notes": [
              "horizon sensitivity statement (from P5 (4)): every signature/criterion/catastrophe-list condition holds under the"
              " evaluation-protocol horizon; the horizon may not change within this case",
              "H-track default clause: no M29 reading ever enters any decision line",
              "non-retroactivity clause: all earlier verdicts, titles and anchor values stay as they are",
              "residuals 1-15: see the full pre-registration; carried into the verdict appendix"]}
    log(ev)
    events.append(ev)
    attention(f"B1 case-closing scorecard recorded: RB.4 = {cell}; the verdict appendix is written by the operator")


# ======================================================================
# --plan (no side effects)
# ======================================================================

def print_plan():
    def fmt_cmd(c):
        return " ".join(str(x).replace(str(ROOT) + "/", "") for x in c)

    plan = {
        "case": "B1 bundled evaluation infrastructure (measurement and instrumentation case; zero gold seeds/titles/releases/environment intrusion)",
        "driver": "train/run_b1_infra.py (cloned from the run_v32_sovereign.py skeleton)",
        "ledger": str(LEDGER.relative_to(ROOT)),
        "stages": [
            "S0 preflight: W1/W7 (king zip/npz, H, M29, KING_SD+parity, BC_SD_WAIVER)/"
            "PRIOR_REFERENCE four archives/W-E0 (runtime_five == v32 CASE_RUNTIME)/"
            "W-H8 held-out pool virginity + HOLDOUT_FIRSTBURN (2 pairs/4 runs)/304000+307000 ledger scan/"
            "W9/FREEZE_SHA",
            "S1 W-G0 zero-intrusion live run (freeze prerequisite)",
            "S2 V1 variance run b1-varprobe-launch (also RB.1 bit-level continuity; band [0,0])",
            "S3 K1 b1-ref8k-launch + K2 b1-ref8k-science (first exposure of the 8000 pool, counted per run)",
            "S4 CANARY_SET (E8 extracts depth>=2 ∪ {7003,7011}) + CRITERION_REGISTER"
            " (embeds K1's 8000-pool depth>=2 list)",
            "S5 P8 leg launch (budget 2; nt gate 3,997,696; timeout 6h; launcher per E6)",
            "S6 offline canary sequence (4 ckpt x {H,M29}, 8 runs in total, all before L1;"
            " OBS_DRIFT with each archive; the +3,989,504 ckpt is archived but not in the sequence) + GATE_WOULD_TRIP"
            " (ten calib points on both gates + dry-anchor dual reference; record only)",
            "S7 five leg exams: L1 p8-s16 (H only, E1 exemption 1) -> L2/L3 bundled + MS_REPORT"
            " -> L4/L5 held-out bundled + MS_REPORT",
            "S8 CRITERION_VALIDATE (2x2 confusion table + precision (CP 95%) + recall;"
            " hits <3 means undecidable)",
            "S9 R-line scorecard RB.1-RB.10 + conversion clause -> VERDICT_PATH (verdict appendix written by the operator)",
        ],
        "leg_cmd": fmt_cmd(leg_cmd()),
        "leg_deviations_closed_enum": {
            "1_seed": f"{SEED} (=303000+1000, no seed search/reselection)",
            "2_calib_probes": list(CALIB_PROBES),
            "3_ckpt_every_steps": CKPT_EVERY,
            "4_sentinel_every": SENTINEL_EVERY,
            "5_dry_anchor_every": DRY_ANCHOR_EVERY},
        "canary_points": {"mid4": list(CANARY_STEPS),
                          "terminal_npz": NT_TARGET,
                          "archived_not_in_sequence": EXTRA_CKPT_STEP},
        "smoke_w_g0": {
            "seed": SMOKE_SEED, "steps": SMOKE_STEPS,
            "window": [KING_STEPS, SMOKE_END],
            "bare_cmd": fmt_cmd(_smoke_cmd("bare")),
            "knobs_cmd": fmt_cmd(_smoke_cmd("knobs")),
            "guaranteed_in_window": {
                "calib_points": list(SMOKE_CALIB),
                "ckpt_saves": [KING_STEPS + CKPT_EVERY],
                "sentinel_first": ((KING_STEPS // SENTINEL_EVERY) + 1)
                * SENTINEL_EVERY,
                "dry_anchor_first": ((KING_STEPS // DRY_ANCHOR_EVERY) + 1)
                * DRY_ANCHOR_EVERY},
            "criteria": "policy state_dict + optimizer state, torch.equal per tensor"
                        " + RNG-related telemetry field by field; file byte-level comparison dropped"},
        "exam_table": [
            {"run": "V1", "tag": "b1-varprobe-launch", "worker": "king npz",
             "manager": "H", "seeds": POOL_PROBE},
            {"run": "K1", "tag": "b1-ref8k-launch", "worker": "king npz",
             "manager": "H", "seeds": POOL_HOLD},
            {"run": "K2", "tag": "b1-ref8k-science", "worker": "king npz",
             "manager": "M29", "seeds": POOL_HOLD},
            {"run": "L1", "tag": "p8-s16", "worker": "P8 npz", "manager": "H",
             "seeds": POOL_S16},
            {"run": "L2", "tag": "p8-full32", "worker": "P8 npz", "manager": "H",
             "seeds": POOL_PROBE},
            {"run": "L3", "tag": "p8-full32-m29", "worker": "P8 npz",
             "manager": "M29", "seeds": POOL_PROBE},
            {"run": "L4", "tag": "p8-hold8k", "worker": "P8 npz", "manager": "H",
             "seeds": POOL_HOLD},
            {"run": "L5", "tag": "p8-hold8k-m29", "worker": "P8 npz",
             "manager": "M29", "seeds": POOL_HOLD}],
        "would_trip_lines": {"distill_ce": CE_LINE, "grad_ratio": RATIO_LINE,
                             "dry_anchor": f"zero point {DRY_REF_THRONE},"
                                           f" +{DRY_TRIP_PP}pp; lineage reference "
                                           f"{DRY_REF_LINEAGE}",
                             "discipline": "record only throughout this case"},
        "exit_codes": {0: "case closed/idempotent", 2: "budget exhausted", 3: "preflight", 4: "lock conflict",
                       5: "W-E0 pre-launch drift", 6: "runtime drift during the case",
                       7: "CASE_HALT_G0", 8: "REF_DIVERGENCE"},
    }
    print(json.dumps(plan, ensure_ascii=False, indent=1))


# ======================================================================
# main flow
# ======================================================================

def _main():
    events = read_ledger()
    if stage_done(events, "VERDICT_PATH"):
        print("case already closed: idempotent exit", flush=True)
        return
    preflight(events)
    smoke_stage(events)                                        # S1
    varprobe_stage(events)                                     # S2
    k1, k2 = bundled_exam(events, str(KING_NPZ), "b1-ref8k-launch",
                          "b1-ref8k-science", POOL_HOLD, holdout=True)  # S3
    cs = canary_set_stage(events)                              # S4
    reg_ev = criterion_register_stage(events)
    n_D, d_c_depth2 = cs["n_D"], cs["depth2_seeds"]
    leg_npz = leg_stage(events)                                # S5
    canary_docs = canary_stage(events, d_c_depth2)             # S6
    gates = deadgate_stage(events)
    docs = {"b1-ref8k-launch": k1, "b1-ref8k-science": k2}
    docs["p8-s16"] = exam_case(events, leg_npz, "p8-s16", POOL_S16,
                               manager_npz=None,
                               extra={"note": "operational sanity reading (E1 exemption 1,"
                                              " half pool, H only, compliant)"})       # S7 L1
    l2, l3 = bundled_exam(events, leg_npz, "p8-full32", "p8-full32-m29",
                          POOL_PROBE)
    docs["p8-full32"], docs["p8-full32-m29"] = l2, l3
    ref_launch = by_seed(strict_json_loads(
        PRIORS["v32-ref-launch"][0].read_bytes())["rows"], 7000, 7031)
    ref_science = by_seed(strict_json_loads(
        PRIORS["v32-ref-science"][0].read_bytes())["rows"], 7000, 7031)
    ms_report(events, "7000", "p8-full32", "p8-full32-m29",
              by_seed(l2["rows"], 7000, 7031), by_seed(l3["rows"], 7000, 7031),
              ref_launch, ref_science,
              ["v32-ref-launch (PRIORS pinned)", "v32-ref-science (PRIORS pinned)"])
    l4, l5 = bundled_exam(events, leg_npz, "p8-hold8k", "p8-hold8k-m29",
                          POOL_HOLD, holdout=True)
    docs["p8-hold8k"], docs["p8-hold8k-m29"] = l4, l5
    ms_report(events, "8000", "p8-hold8k", "p8-hold8k-m29",
              by_seed(l4["rows"], 8000, 8031), by_seed(l5["rows"], 8000, 8031),
              by_seed(k1["rows"], 8000, 8031), by_seed(k2["rows"], 8000, 8031),
              ["b1-ref8k-launch", "b1-ref8k-science"])
    # S8: L4 replay (τ median; 8000 pool) + CRITERION_VALIDATE
    l4_replay = run_obsdrift(leg_npz, EVAL / "p8-hold8k.json",
                             B1 / "replay" / "p8-hold8k.json")
    criterion_validate_stage(events, reg_ev, k1, k2, l4, l5, l4_replay)
    # S9: final-leg replay (τ median + RB.10 leg side + final OBS_DRIFT) + scorecard
    throne_rep = throne_replay()
    leg_replay = run_obsdrift(leg_npz, EVAL / "p8-full32.json",
                              B1 / "replay" / "p8-full32.json")
    obs_drift_event(events, "terminal", leg_replay, throne_rep, d_c_depth2)
    scorecard_stage(events, docs, canary_docs, gates, reg_ev, n_D,
                    d_c_depth2, leg_replay, throne_rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print the plan, no side effects")
    ap.add_argument("--smoke", action="store_true",
                    help="run only the S1 W-G0 zero-intrusion smoke test (freeze prerequisite, dirty tree allowed)")
    args = ap.parse_args()
    if args.plan:
        print_plan()
        return
    try:
        with exclusive_lock(B1 / ".driver.lock", "B1 driver"):
            if args.smoke:
                smoke_stage(read_ledger())
                print("W-G0 smoke stage complete (verdict in ledger event G0_NULLINTRUSION)",
                      flush=True)
            else:
                _main()
    except OutputReservationError as e:
        log({"event": "OPERATIONAL_FAILURE", "why": f"W8 lock conflict: {e}"})
        attention("W8 not idle / lock conflict:\n" + str(e))
        raise SystemExit(4) from e
    except PreflightFailure as e:
        log({"event": "PREFLIGHT_FAIL", "why": str(e)})
        attention("P4 preflight failed, no launch; report for review:\n" + str(e))
        raise SystemExit(3) from e
    except OperationalFailure as e:
        log({"event": "OPERATIONAL_FAILURE", "why": str(e)})
        attention("operational failure:\n" + str(e))
        raise SystemExit(2) from e
    except SystemExit:
        raise
    except Exception as e:
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("driver died abnormally (P1):\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
