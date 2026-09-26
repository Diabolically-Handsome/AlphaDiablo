"""v24-KL leg driver (sole executor of clauses [leg-1] to [final-6] of docs/prereg/PREREG-v24.md).

Fixed annealing + two trip lines, no human discretion; every verdict step is written to train/runs/v24/gate_ledger.jsonl.
The gold-standard evaluation itself is not launched here: stop after G3 decides the winner and eligibility; it is started manually (gold-standard discipline).
All 22 items confirmed by the pre-launch review panel are implemented: crash interlock before G-CAL,
P* excludes the leg under review, G3 override 3% sentinel line + +/-0.05 tie band, recalibration reorders the whole table,
crash-burned steps count toward the budget, per-attempt autopsy archived, dual-probe wiring criterion, sps on the same ledger.
Usage: .venv/bin/python train/run_v24_legs.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import signal
import subprocess
import time
import traceback
import zipfile

from eval_contract import (PROTOCOL_VERSION, OutputReservationError,
                           exclusive_lock, expected_eval_identity,
                           freeze_eval_identity, read_eval_archive,
                           verify_eval_identity)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
V24 = RUNS / "v24"
V24.mkdir(parents=True, exist_ok=True)
LEDGER = V24 / "gate_ledger.jsonl"
DRIVER_LOCK = V24 / ".driver.lock"
DRIVER_LOCK_PURPOSE = "v24 driver"

LEG = 489 * 2048          # 1,001,472 ([leg-1] step quantization registered as is)
QUANTUM = 2048            # n_steps x num_envs; short legs must be quantized downward, over-sampling the hard budget is forbidden
N_LEGS = 8
BUDGET_STEPS = N_LEGS * LEG
BETA_SCHED = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0, 0.0]
HARD_LINE = 62.8          # [hard-3] (0.8xG1, derived from full 32; looser when applied to the 16-seed exam, kept unrelaxed)
SOFT_MULT = 0.97          # [soft-4]
SCRIPT_SUBSET = 93.9      # known constant of the script/BC on the 7000-7015 half pool (the seed element of the P* set)
SPS_FLOOR = 1_800_000     # real steps/hour (downgrade clause; numerator = real training steps including burned steps, on the same ledger as the denominator)
TAIL_CUT_STEPS = 244 * 2048
PROBES = (300_000, 600_000)
BC_SD = str(RUNS / "bc-worker" / "policy_sd.pt")
DEFAULT_MANAGER_NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"

# G3/gold-standard eligibility (numbers verbatim from the interpreted v23 appendix B; override: the 3% sentinel line passes the gate, 8% separately voids the data)
G3_MEAN = 74.6
G3_DEATHS = 6
R4 = {"farm_descend_rate": 0.0204, "override_sentinel": 0.03, "override_void": 0.08,
      "cap_rate": 0.05, "farm_tau_lo": 27.8, "farm_tau_hi": 46.4}
CALIBRATED_PROTOCOL_VERSION = 2


class OperationalFailure(RuntimeError):
    """Infrastructure/wiring failure; distinct from a normal scientific trip line, the process must exit non-zero."""


def budgeted_leg_steps(spent_steps: int, cap: int = LEG) -> int:
    """Deduct all observed real training steps from the hard budget live, instead of only shrinking the last leg."""
    remaining = max(0, BUDGET_STEPS - spent_steps)
    return min(cap, (remaining // QUANTUM) * QUANTUM)


def ensure_retry_budget(spent_steps: int, cap: int = LEG) -> None:
    if budgeted_leg_steps(spent_steps, cap) == 0:
        raise OperationalFailure("failed attempts exhausted the hard budget; the current leg cannot be completed")


def observed_attempt_steps(base: int, result: dict, allocated: int) -> int:
    """Conservatively take this attempt's observed increment from the two monotonic global counters, zip and status."""
    observed = max(int(result.get("global_steps", 0)),
                   int(result.get("status_steps", 0)))
    delta = max(0, observed - base)
    if delta > allocated:
        raise OperationalFailure(
            f"steps exceeded this allocation: base={base}, observed={observed}, allocated={allocated}")
    return delta


def failed_attempt_charge(observed: int, allocated: int) -> int:
    """status is only a lower bound; an abnormal attempt occupies the hard budget with its full granted allocation."""
    if not 0 <= observed <= allocated:
        raise OperationalFailure("observed steps of an abnormal attempt out of range")
    return allocated


def is_gcal_stop(k: int, result: dict, base: int, expected: int,
                 records: list[dict]) -> bool:
    """A G-CAL half-rollout early stop does not publish a final zip; identify it from status and the current records."""
    status = int(result.get("status_steps", 0))
    return (k == 1 and result.get("rc") == 0 and base < status <= expected
            and any(r.get("tripped") for r in records))


def reset_recalibration_attempts(attempts: dict[int, int]) -> None:
    attempts[1] = 0


def run_process(cmd, logfile, timeout: int) -> int:
    with open(logfile, "w") as lf:
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


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def sha16(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_calibrated_protocol() -> None:
    if PROTOCOL_VERSION != CALIBRATED_PROTOCOL_VERSION:
        raise OperationalFailure(
            "v24 HARD_LINE/SCRIPT_SUBSET/G3/R4 static thresholds are calibrated only under pre-v3 environment semantics; "
            "re-run the protocol-v3 baseline and update the pre-registration by hand first; mixing in the old thresholds is forbidden"
        )


def zip_steps(p: pathlib.Path) -> int:
    try:
        with zipfile.ZipFile(p) as zf:
            return int(json.loads(zf.read("data"))["num_timesteps"])
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return 0


def preflight() -> None:
    require_calibrated_protocol()
    require(pathlib.Path(BC_SD).is_file(), f"BC teacher missing: {BC_SD}")
    eval_dir = RUNS / "eval-assembled"
    for k in range(1, N_LEGS + 1):
        require(not (RUNS / f"v24-leg{k}").exists(), f"leftover run directory: v24-leg{k}")
        for tag in (f"v24-leg{k}", f"v24-G3-leg{k}"):
            require(not (eval_dir / f"{tag}.json").exists(), f"evaluation archive already exists: {tag}")
    require(not (RUNS / "v24-leg1r").exists(), "leftover run directory: v24-leg1r")


def run_leg(k: int, beta: float, resume_from: str | None, leg_steps: int,
            run_name: str, attempt: int) -> dict:
    run_dir = RUNS / run_name
    model_path = run_dir / "model_final.zip"
    old_mtime = model_path.stat().st_mtime_ns if model_path.exists() else None
    stale = run_dir / "status.json"
    if stale.exists():
        stale.unlink()            # a re-run must not read the previous attempt's step count
    for fn in ("calib.jsonl", "sentinel.jsonl"):
        p = run_dir / fn
        if p.exists():
            p.rename(p.with_suffix(f".pre{attempt}.{time.time_ns()}.void"))
    cmd = [PY, "train/train_ppo.py", "--worker", "--algo", "mppo", "--gamma", "1.0",
           "--max-steps", "3000", "--num-envs", "4", "--n-steps", "512", "--lr", "3e-4",
           "--ent-coef", "0.005", "--seed", str(100_000 + 1000 * k),
           "--total-steps", str(leg_steps), "--run-name", run_name,
           "--distill-beta", str(beta), "--teacher-sd", BC_SD]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    else:
        cmd += ["--bc-init", BC_SD, "--freeze-policy-steps", "200000",
                "--calib-probes", ",".join(str(p) for p in PROBES)]
    t0 = time.time()
    rc = run_process(cmd, V24 / f"{run_name}.try{attempt}.log", timeout=21_600)
    dt = time.time() - t0
    sp = run_dir / "status.json"
    try:
        status_steps = int(json.loads(sp.read_text())["total_steps"]) if sp.exists() else 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        status_steps = 0
    fresh_model = model_path.exists() and model_path.stat().st_mtime_ns != old_mtime
    model_steps = zip_steps(model_path) if fresh_model else 0
    return {"rc": rc, "dt_sec": round(dt), "global_steps": model_steps,
            "status_steps": status_steps, "model": model_path, "model_fresh": fresh_model}


def exam(model_path: pathlib.Path, tag: str, seeds: str) -> dict | None:
    worker = model_path.with_suffix("") if model_path.suffix == ".zip" else model_path
    j = RUNS / "eval-assembled" / f"{tag}.json"
    existed_before = j.exists()
    try:
        lo, hi = (int(x) for x in seeds.split("-", 1))
        seed_values = list(range(lo, hi + 1))
        require(seed_values and lo >= 0, f"illegal seed range: {seeds}")
        snapshot = freeze_eval_identity(ROOT, worker, DEFAULT_MANAGER_NPZ)
        expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
        rc = run_process(
            [PY, "train/eval_assembled.py",
             "--worker", snapshot["worker"]["path"],
             "--manager-npz", snapshot["manager"]["path"],
             "--seeds", seeds, "--tag", tag],
            V24 / f"exam-{tag}.{time.time_ns()}.log", timeout=1_800)
        if rc != 0:
            raise OSError(f"evaluation subprocess failed: rc={rc}")
        doc = read_eval_archive(j, **expected)
        verify_eval_identity(snapshot, ROOT)
        agg = doc["agg"]
        agg["_sha"] = sha16(j)
        return agg
    except (OSError, KeyError, TypeError, ValueError):
        if not existed_before and j.exists():
            j.rename(j.with_suffix(f".{time.time_ns()}.void"))
        return None


def second_largest(vals):
    s = sorted(vals, reverse=True)
    return s[1] if len(s) > 1 else s[0]


def main():
    try:
        with exclusive_lock(DRIVER_LOCK, DRIVER_LOCK_PURPOSE):
            _main()
    except (OperationalFailure, OutputReservationError) as exc:
        log({"event": "OPERATIONAL_FAILURE", "why": str(exc)})
        raise SystemExit(2) from exc
    except Exception as exc:
        log({"event": "DRIVER_EXCEPTION", "why": repr(exc),
             "traceback": traceback.format_exc()})
        raise


def _main():
    preflight()
    log({"event": "start", "leg_steps": LEG, "beta_sched": list(BETA_SCHED),
         "hard": HARD_LINE, "soft": SOFT_MULT})
    scores = []                 # completed leg-exam scores (1 decimal)
    sched_idx = 0               # soft-trip freeze = the pointer does not advance
    recalibrated = False
    burned = 0                  # discarded recalibration/crash steps (audit definition)
    spent_steps = 0             # observed new steps of all attempts; constrains BUDGET_STEPS one by one
    chain_steps = 0             # cumulative steps expected on the current count chain (reset to zero after recalibration)
    train_secs = 0.0
    prev_model = None
    leg_models = {}
    tail_cut = False
    attempts = {}
    k = 1
    while k <= N_LEGS:
        beta = BETA_SCHED[min(sched_idx, len(BETA_SCHED) - 1)]
        cap = TAIL_CUT_STEPS if (tail_cut and k >= 7) else LEG
        leg_steps = budgeted_leg_steps(spent_steps, cap)
        if leg_steps < cap:
            log({"event": "leg_budget_shrunk", "leg": k, "steps": leg_steps,
                 "spent": spent_steps, "remaining": BUDGET_STEPS - spent_steps,
                 "note": "all completed new steps and failed/recalibration burned steps are deducted from the hard budget, one by one"})
        if leg_steps == 0:
            log({"event": "budget_exhausted", "leg": k, "spent": spent_steps})
            break
        attempts[k] = attempts.get(k, 0) + 1
        run_name = f"v24-leg{k}" + ("r" if (k == 1 and recalibrated) else "")
        log({"event": "leg_start", "leg": k, "attempt": attempts[k], "beta": beta,
             "steps": leg_steps, "seed": 100_000 + 1000 * k, "resume": bool(prev_model)})
        res = run_leg(k, beta, prev_model, leg_steps, run_name, attempts[k])
        train_secs += res["dt_sec"]
        expected = chain_steps + leg_steps

        # ---- [exam-2] crash interlock (before any verdict: review panel blocker fix) ----
        calib_p = RUNS / run_name / "calib.jsonl"
        try:
            calib_recs = ([json.loads(l) for l in calib_p.read_text().splitlines()]
                          if k == 1 and calib_p.exists() else [])
        except (OSError, json.JSONDecodeError):
            calib_recs = []
        # A G-CAL trip ends learn normally, but a half rollout does not publish model_final; the current attempt's
        # three proofs (rc/status interval/tripped) suffice to identify it, so fresh_model can no longer be relied on.
        gcal_stop = is_gcal_stop(k, res, chain_steps, expected, calib_recs)
        clean = (res["rc"] == 0 and res["model_fresh"]
                 and res["global_steps"] == expected)
        sampled = observed_attempt_steps(chain_steps, res, leg_steps)
        if clean:
            require(sampled == leg_steps, "observed steps of a clean close disagree with the allocation")
            spent_steps += sampled
            chain_steps = res["global_steps"]
        elif gcal_stop:
            spent_steps += sampled
            burned += sampled
            log({"event": "gcal_early_stop", "leg": k, "observed_steps": sampled,
                 "status_steps": res["status_steps"], "burned_total": burned,
                 "spent_total": spent_steps})
        else:
            # An abnormal exit leaves only a status lower bound; samples after the last refresh are unobservable.
            # To make the hard budget a true upper bound, book this attempt's full allocation conservatively.
            partial = sampled
            charged = failed_attempt_charge(partial, leg_steps)
            spent_steps += charged
            burned += charged
            log({"event": "leg_crash", "leg": k, "attempt": attempts[k],
                 "rc": res["rc"], "global_steps": res["global_steps"],
                 "burned_observed": partial, "burned_charged": charged,
                 "burned_total": burned,
                 "spent_total": spent_steps,
                 "note": "re-run with the original config per [final-6]; burned steps are deducted live from the remaining allocatable quota"})
            if k == 1:
                calib = RUNS / run_name / "calib.jsonl"
                if calib.exists():   # rotate the probe records of a crashed attempt so they do not pollute the G-CAL verdict
                    calib.rename(calib.with_suffix(f".try{attempts[k]}.void"))
            ensure_retry_budget(spent_steps, cap)
            if attempts[k] >= 4:
                why = (f"leg {k} crashed {attempts[k]} times in a row: driver self-protection stop"
                       " (not a pre-registered scientific gate; needs a manual autopsy)")
                log({"event": "STOP", "why": why})
                raise OperationalFailure(why)
            continue

        # ---- G-CAL (leg 1 only; a complete close or a normal early stop with all three proofs) ----
        if k == 1:
            recs = calib_recs
            tripped = any(r.get("tripped") for r in recs)
            probes_ok = all(any(p <= r["step"] < p + 2048 and r["g_ce"] > 0
                                and r["distill_ce"] > 0 for r in recs)
                            for p in PROBES)
            log({"event": "g_cal", "records": recs, "tripped": tripped,
                 "probes_ok": probes_ok})
            if tripped:
                if recalibrated:
                    log({"event": "STOP", "why": "G-CAL triggered a second time = design judged dead; stopping and writing the verdict"})
                    return
                recalibrated = True
                # A trigger on a complete leg is not yet counted as burned steps; a normal early stop was already booked above from the status increment.
                if not gcal_stop:
                    burned += sampled
                BETA_SCHED[:] = [2.0 * 0.5 ** i for i in range(6)] + [0.0, 0.0]
                log({"event": "recalibrate", "beta0": 2.0,
                     "new_sched": list(BETA_SCHED), "burned": burned,
                     "note": "the one and only beta0 x4: the whole schedule is reordered as beta_k=beta0*2^{-(k-1)}"
                             " (legs 7/8 stay pinned at 0, noted in an addendum to the decision record); burned steps are deducted live from later quota"})
                prev_model = None
                chain_steps = 0
                sched_idx = 0
                reset_recalibration_attempts(attempts)  # the new schedule gets its own operational retry quota
                continue    # k stays 1
            if not probes_ok:
                why = ("G-CAL wiring failed (neither probe saw ce/g_ce>0)"
                       ": after fixing the code, re-run under the crash clause; needs manual intervention")
                log({"event": "STOP", "why": why})
                raise OperationalFailure(why)

        # ---- leg exam ----
        agg = exam(res["model"], f"v24-leg{k}", "7000-7015")
        if agg is None:
            log({"event": "exam_crash", "leg": k, "note": "exam process failed; retaking under the crash clause"})
            agg = exam(res["model"], f"v24-leg{k}", "7000-7015")
            if agg is None:
                why = "exam failed twice in a row: manual autopsy"
                log({"event": "STOP", "why": why})
                raise OperationalFailure(why)
        score = round(agg["ret_mean"], 1)
        p_star = second_largest([SCRIPT_SUBSET] + scores)   # excludes the leg under review (review panel fix:
        scores.append(score)                                 # leg 1 soft trip line = 0.97x93.9 = 91.1)
        leg_models[k] = (score, str(res["model"]), beta)
        log({"event": "leg_exam", "leg": k, "beta": beta, "score": score,
             "died": agg["died"], "diverge": agg.get("script_divergence_rate"),
             "sha": agg["_sha"], "model_sha": sha16(res["model"]),
             "p_star_prior": p_star, "global_steps": res["global_steps"]})

        # ---- [hard-3] ----
        if score < HARD_LINE:
            log({"event": "HARD_TRIP", "leg": k, "score": score,
                 "why": f"< {HARD_LINE}, training of this run permanently stopped (rollback-retrain limit = 0)"})
            break
        # ---- [soft-4] ----
        if score < round(SOFT_MULT * p_star, 1):
            log({"event": "soft_trip", "leg": k, "score": score,
                 "line": round(SOFT_MULT * p_star, 1), "note": "beta frozen, schedule shifted right"})
        else:
            sched_idx += 1
        # ---- sps downgrade (numerator includes burned steps, same ledger as the denominator; k<8 allows cutting leg 8 after leg 7) ----
        rate = spent_steps / max(1e-9, train_secs) * 3600
        if rate < SPS_FLOOR and k < 8 and not tail_cut:
            tail_cut = True
            log({"event": "sps_downshift", "rate_per_h": round(rate),
                 "note": "legs 7-8 each cut to 244x2048, the verdict downgraded accordingly (pre-registered)"})
        prev_model = str(res["model"])
        k += 1

    # ---- G3: candidates fixed = end-of-leg ckpts, top-2 by leg-exam score ----
    if not leg_models:
        why = "no completed leg at all"
        log({"event": "STOP", "why": why})
        raise OperationalFailure(why)
    top2 = sorted(leg_models.items(), key=lambda kv: kv[1][0], reverse=True)[:2]
    log({"event": "g3_candidates", "cands": [(kk, v[0], v[2]) for kk, v in top2]})
    finals = []
    for kk, (sc16, mp, bt) in top2:
        agg = exam(pathlib.Path(mp), f"v24-G3-leg{kk}", "7000-7031")
        if agg is None:
            why = f"G3 full-32 exam failed (leg {kk}): manual autopsy"
            log({"event": "STOP", "why": why})
            raise OperationalFailure(why)
        void = agg["override_rate"] >= R4["override_void"]
        r4_ok = (agg["farm_descend_rate"] <= R4["farm_descend_rate"]
                 and agg["override_rate"] < R4["override_sentinel"]   # the 3% sentinel line passes the gate
                 and agg["cap_rate"] < R4["cap_rate"]
                 and R4["farm_tau_lo"] <= agg["farm_tau_mean"] <= R4["farm_tau_hi"])
        ok = agg["ret_mean"] >= G3_MEAN and agg["died"] <= G3_DEATHS and r4_ok and not void
        finals.append((kk, agg["ret_mean"], agg["died"], ok, bt,
                       agg.get("script_divergence_rate"), mp))
        log({"event": "g3_full32", "leg": kk, "mean": agg["ret_mean"],
             "died": agg["died"], "r4_ok": r4_ok, "data_void": void, "qualified": ok,
             "diverge": agg.get("script_divergence_rate"),
             "override": agg["override_rate"], "descend_rate": agg["farm_descend_rate"],
             "tau": agg["farm_tau_mean"]})
    qual = [f for f in finals if f[3]]
    if not qual:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": "no candidate reached the gold-standard eligibility line: no gold seeds spent, the verdict stated as is (spirit of P2)"})
        return
    qual.sort(key=lambda f: -f[1])
    w = qual[0]
    tie = len(qual) == 2 and abs(qual[0][1] - qual[1][1]) <= 0.05
    if tie:
        w = min(qual, key=lambda f: f[4])    # +/-0.05 tie band: take the leg with the lower beta
    log({"event": "GOLDEN_AUTHORIZED", "leg": w[0], "probe32_mean": w[1],
         "died": w[2], "beta_of_leg": w[4], "diverge": w[5], "model": w[6],
         "tie_band_applied": tie,
         "note": "the gold-standard evaluation is started manually, single arm, once (gold-standard discipline)"})


if __name__ == "__main__":
    main()
