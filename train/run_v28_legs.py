"""v28 "oasis endurance" leg driver (final text of docs/prereg/PREREG-v28.md; a targeted rework of run_v26_legs.py).

Difference table vs the v26 driver (maps clause by clause to the PREREG-v28 D2 clone difference table; the panel's 24 decisions implemented):
- Endurance: all 8 legs resume, leg 1 starts from v26-leg6/model_final.zip; the bc-init branch is physically removed
- beta constant 0.015625: BETA_SCHED/sched_idx/soft trip 0.97xP*/SCRIPT_SUBSET/recalibrate all removed
- G-CAL: every leg passes absolute-step probes nt_chain+250k / +450k (trimmed and booked if unreachable),
  --calib-record-only records without judging; the probes_ok wiring gate runs every leg; calib rotation on crashed attempts every leg
- Single step counting: nt_chain starts at START=2,998,272 (zip num_timesteps, asserted at launch);
  the ~300-step lag of the status.json counter is covered by a +/-2048 slack (measured lag on v26-leg6: 268)
- Closing trip lines: hard trip <62.8 -> break into G3; two consecutive clean legs <103.1 (16-seed definition) ->
  break into G3 (crashed attempts neither count nor reset; any leg >=103.1 resets; the hard trip takes precedence)
- Sanity 2h/leg -> break into G3 (v26 was a bare STOP, which the panel judged inconsistent with the budget-protection philosophy)
- Archive immutability: leg exam tag v28-leg{k}, G3 tag v28-G3-leg{k}; exam() refuses to overwrite;
  anchor sha asserted at launch (the v24/v26 per-seed leg-exam archives were lost for two rounds through tag collisions; the incident is on record)
- Width probe: leg-exam JSON rows are paired by seed key against the first 16 of the anchor (seeds 7000-7015) to count wins;
  ties/missing rows do not count as wins; pure post-processing, zero extra evaluation
- G3 candidates: mean top-2 union width top-1 (deduplicated, <=3); the starting checkpoint itself is not in the pool; tie rule registered
- Launch: paired (joined on the seed key + assertion) mean difference >=+4 and wins >=18/32 and eligibility;
  candidates that entered the pool only through the width channel get an extra out-of-sample side line: after the full 32, the back 16 paired wins >=8/16 (baseline 6/16)
- VERDICT three-key dispatch is exhaustive (ineligible/side line blocked/width reached but magnitude not/intrinsic >=108.2/
  [103,108.2) unanswered/degraded <103); GOLDEN_AUTHORIZED carries model_sha + archive sha + the gold-standard evaluation command
- Every non-routine path writes runs/v28/NEEDS_ATTENTION (the first file to check)
The gold-standard evaluation is not launched here: stop after the winner and eligibility are decided; it is started manually (gold-standard discipline).
Usage: .venv/bin/python train/run_v28_legs.py
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
V28 = RUNS / "v28"
V28.mkdir(parents=True, exist_ok=True)
LEDGER = V28 / "gate_ledger.jsonl"
DRIVER_LOCK = V28 / ".driver.lock"
DRIVER_LOCK_PURPOSE = "v28 driver"
EVAL = RUNS / "eval-assembled"

LEG = 244 * 2048              # 499,712 (the v26 leg system unchanged)
QUANTUM = 2048
N_LEGS = 8
BUDGET_STEPS = N_LEGS * LEG     # counts only new steps after START
BETA = 0.015625               # constant, never let go (lesson 19 + v27 "cannot settle")
HARD_LINE = 62.8
FINISH_LINE = 103.1           # =round(0.9x114.5,1), 16-seed leg-exam definition (do not confuse with the full-32 definition 103)
INTRINSIC_LINE = 108.2        # full-32 definition: the full-32 mean of the starting checkpoint itself (intrinsic tier threshold)
PLATEAU_LINE = 103.0          # full-32 definition: [103,108.2) falls in the "endurance without mean gain" tier
START = 2_998_272             # v26-leg6 zip num_timesteps (6x499,712; asserted at launch)
BASE_CKPT = RUNS / "v26-leg6" / "model_final.zip"
BC_SD = str(RUNS / "bc-worker" / "policy_sd.pt")
ANCHOR = EVAL / "v24-G3-leg7.json"
ANCHOR_SHA = "22d9442257d3a3c79feb5b40918890917772e11036e4026ca4d3cc2005318359"
ANCHOR_WORKER = ROOT / "train" / "models" / "v24-worker-leg7" / "model.zip"
BASELINE = EVAL / "v26-G3-leg6.json"
BASELINE_SHA = "24a905a7baf0f70ab09a8103721757f10eec0692b64b81f9880abedd2e0325dd"
DEFAULT_MANAGER_SHA = "0f2264860b0960e7951efd424836b90c09c002cebca7bf8109fd669b13be63d7"

G3_MEAN = 74.6
G3_DEATHS = 6
R4 = {"farm_descend_rate": 0.0204, "override_sentinel": 0.03, "override_void": 0.08,
      "cap_rate": 0.05, "farm_tau_lo": 27.8, "farm_tau_hi": 46.4}
SIDELINE_BACK16 = 8           # out-of-sample side line for width-channel-only candidates (null hypothesis P(>=8)~16%)
CALIBRATED_PROTOCOL_VERSION = 2


class OperationalFailure(RuntimeError):
    """Infrastructure/wiring failure; distinct from a normal scientific trip line, the process must exit non-zero."""


def budgeted_leg_steps(spent_steps: int, cap: int = LEG) -> int:
    remaining = max(0, BUDGET_STEPS - spent_steps)
    return min(cap, (remaining // QUANTUM) * QUANTUM)


def ensure_retry_budget(spent_steps: int, cap: int = LEG) -> None:
    if budgeted_leg_steps(spent_steps, cap) == 0:
        raise OperationalFailure("failed attempts exhausted the hard budget; the current leg cannot be completed")


def observed_attempt_steps(base: int, result: dict, allocated: int) -> int:
    observed = max(int(result.get("global_steps", 0)),
                   int(result.get("status_steps", 0)))
    delta = max(0, observed - base)
    if delta > allocated:
        raise OperationalFailure(
            f"steps exceeded this allocation: base={base}, observed={observed}, allocated={allocated}")
    return delta


def failed_attempt_charge(observed: int, allocated: int) -> int:
    if not 0 <= observed <= allocated:
        raise OperationalFailure("observed steps of an abnormal attempt out of range")
    return allocated


def candidate_probe_eligible(probes_ok: bool | None) -> bool:
    """Only an explicit PASS may enter the candidates; the SKIPPED of an empty probe must fail closed."""
    return probes_ok is True


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    with open(V28 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha16(p: pathlib.Path) -> str:
    return sha256(p)[:16]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_calibrated_protocol() -> None:
    if PROTOCOL_VERSION != CALIBRATED_PROTOCOL_VERSION:
        raise OperationalFailure(
            "v28 HARD_LINE/G3/R4/plateau and candidate static thresholds are calibrated only under pre-v3 environment semantics; "
            "re-run the protocol-v3 baseline and update the pre-registration by hand first; mixing in the old thresholds is forbidden"
        )


def parse_seed_range(seeds: str) -> list[int]:
    """Parse by the LO-HI contract of eval_assembled; reject extra separators and negative numbers."""
    parts = seeds.split("-") if isinstance(seeds, str) else []
    require(len(parts) == 2 and all(p.isascii() and p.isdigit() for p in parts),
            f"illegal seed range: {seeds!r}")
    lo, hi = (int(p) for p in parts)
    require(lo <= hi, f"illegal seed range: {seeds!r}")
    return list(range(lo, hi + 1))


def read_comparable_reference(path: pathlib.Path, *, tag: str,
                              worker: pathlib.Path, label: str) -> dict:
    """Active anchors accept only the current semantics, a fixed assembly and a complete runtime/content identity."""
    try:
        snapshot = freeze_eval_identity(ROOT, worker, None)
        expected = expected_eval_identity(
            snapshot, tag=tag, seeds=range(7000, 7032))
        document = read_eval_archive(path, **expected)
        verify_eval_identity(snapshot, ROOT)
        return document
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise OperationalFailure(
            f"{label} does not satisfy the current schema-v2 comparability contract; "
            "after an environment-semantics change, re-run the baseline with the fixed worker + default manager"
        ) from exc


def read_comparable_anchor() -> dict:
    return read_comparable_reference(
        ANCHOR, tag="v24-G3-leg7", worker=ANCHOR_WORKER, label="endurance pairing anchor")


def read_comparable_baseline() -> dict:
    return read_comparable_reference(
        BASELINE, tag="v26-G3-leg6", worker=BASE_CKPT, label="endurance starting baseline")


def zip_steps(p: pathlib.Path) -> int:
    try:
        with zipfile.ZipFile(p) as zf:
            return int(json.loads(zf.read("data"))["num_timesteps"])
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return 0


def reachable_probes(start: int, leg_steps: int,
                     offsets=(250_000, 450_000)) -> list[int]:
    """Keep only the absolute probe steps that can fire at some rollout close point of this leg."""
    end = start + leg_steps
    probes = []
    for offset in offsets:
        target = start + offset
        first_rollout = start + ((offset + QUANTUM - 1) // QUANTUM) * QUANTUM
        if first_rollout <= end:
            probes.append(target)
    return probes


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


def preflight():
    require_calibrated_protocol()
    require(BASE_CKPT.exists(), f"starting checkpoint missing: {BASE_CKPT}")
    nt = zip_steps(BASE_CKPT)
    require(nt == START, f"START assertion failed: zip num_timesteps={nt} != {START}")
    st = json.loads((RUNS / "v26-leg6" / "status.json").read_text())["total_steps"]
    require(START - 2048 <= st <= START, f"status counter lag out of range: {st}")
    read_comparable_anchor()
    read_comparable_baseline()
    for k in range(1, N_LEGS + 1):
        for tag in (f"v28-leg{k}", f"v28-G3-leg{k}"):
            require(not (EVAL / f"{tag}.json").exists(), f"target archive already exists: {tag}")
        require(not (RUNS / f"v28-leg{k}").exists(),
                f"leftover run directory: v28-leg{k} (restart protocol: archive everything before launching)")
    require(not (EVAL / "v28-golden.json").exists(), "gold-standard target archive already exists (at most one gold run)")
    log({"event": "preflight_ok", "start_nt": nt, "status_lag": START - st,
         "anchor_sha": sha16(ANCHOR), "baseline_sha": sha16(BASELINE)})


def run_leg(k: int, resume_from: str, leg_steps: int, probes: list[int],
            run_name: str, attempt: int, seed_k: int) -> dict:
    run_dir = RUNS / run_name
    model_path = run_dir / "model_final.zip"
    old_mtime = model_path.stat().st_mtime_ns if model_path.exists() else None
    stale = run_dir / "status.json"
    if stale.exists():
        stale.unlink()            # a re-run must not read the previous attempt's step count
    for fn in ("calib.jsonl", "sentinel.jsonl"):
        p = run_dir / fn          # clear the table on entry: stale records from crash retries/restarts must not fake a gate pass
        if p.exists():
            p.rename(p.with_suffix(f".pre{attempt}.{time.strftime('%H%M%S')}.void"))
    cmd = [PY, "train/train_ppo.py", "--worker", "--algo", "mppo", "--gamma", "1.0",
           "--max-steps", "3000", "--num-envs", "4", "--n-steps", "512", "--lr", "3e-4",
           "--ent-coef", "0.005", "--seed", str(seed_k),
           "--total-steps", str(leg_steps), "--run-name", run_name,
           "--distill-beta", str(BETA), "--teacher-sd", BC_SD, "--skip-dry",
           "--resume-from", resume_from]
    if probes:
        cmd += ["--calib-probes", ",".join(str(p) for p in probes),
                "--calib-record-only"]
    t0 = time.time()
    # Hang guard 3h (> the 2h sanity line); terminate the whole process group so SubprocVecEnv orphans do not keep holding the machine.
    rc = run_process(cmd, V28 / f"{run_name}.try{attempt}.log", timeout=10_800)
    dt = time.time() - t0
    sp = run_dir / "status.json"
    try:
        gsteps = json.loads(sp.read_text())["total_steps"] if sp.exists() else 0
    except Exception:
        gsteps = 0                # half-written status (killed mid-write): booked under the crash clause
    fresh_model = model_path.exists() and model_path.stat().st_mtime_ns != old_mtime
    model_steps = zip_steps(model_path) if fresh_model else 0
    return {"rc": rc, "dt_sec": round(dt), "global_steps": model_steps,
            "status_steps": gsteps, "model": model_path, "model_fresh": fresh_model}


def exam(model_path: pathlib.Path, tag: str, seeds: str) -> tuple[dict, list] | None:
    out = EVAL / f"{tag}.json"
    require(not out.exists(), f"archive immutability: {out} already exists, refusing to overwrite")
    try:
        seed_values = parse_seed_range(seeds)
        snapshot = freeze_eval_identity(ROOT, model_path, None)
        require(snapshot["manager"]["sha256"] == DEFAULT_MANAGER_SHA,
                "default manager sha drift")
        expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
        command = [PY, "train/eval_assembled.py",
                   "--worker", snapshot["worker"]["path"],
                   "--manager-npz", snapshot["manager"]["path"],
                   "--seeds", seeds, "--tag", tag]
        rc = run_process(
            command, V28 / f"{tag}.eval.{time.time_ns()}.log", timeout=1_800)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    if rc != 0:
        if out.exists():      # rotate a half-written archive to make way for the retake
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    try:
        document = read_eval_archive(out, **expected)
        verify_eval_identity(snapshot, ROOT)
        agg = document["agg"]
        agg["_sha"] = sha16(out)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    return agg, document["rows"]


def exam_retry(model_path: pathlib.Path, tag: str, seeds: str, what: str):
    r = exam(model_path, tag, seeds)
    if r is None:
        log({"event": "exam_crash", "tag": tag, "note": f"{what} failed; retaking once under the crash clause"})
        r = exam(model_path, tag, seeds)
    return r


def breadth_wins(rows: list, anchor_by_seed: dict, lo: int, hi: int) -> int:
    by_seed = {r["seed"]: r["ret"] for r in rows}
    require(len(rows) == len(by_seed), "evaluation archive contains a duplicate seed")
    require(set(range(lo, hi + 1)) <= set(by_seed), "evaluation archive lacks required seeds")
    return sum(1 for s in range(lo, hi + 1)
               if s in by_seed and by_seed[s] > anchor_by_seed[s])


def main():
    try:
        with exclusive_lock(DRIVER_LOCK, DRIVER_LOCK_PURPOSE):
            _main()
    except (OperationalFailure, OutputReservationError) as e:
        log({"event": "OPERATIONAL_FAILURE", "why": str(e)})
        attention("operational failure:\n" + str(e))
        raise SystemExit(2) from e
    except Exception as e:   # clause 8 catch-all: every unexpected exception must be recorded; no silent death
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("driver died with an exception:\n" + traceback.format_exc())
        raise


def _main():
    preflight()
    anchor = read_comparable_anchor()
    anchor_by_seed = {r["seed"]: r["ret"] for r in anchor["rows"]}
    require(len(anchor["rows"]) == len(anchor_by_seed), "anchor archive contains duplicate seeds")
    require(set(anchor_by_seed) == set(range(7000, 7032)), "anchor seed set is malformed")
    baseline = read_comparable_baseline()
    baseline_front16 = sum(r["ret"] for r in baseline["rows"][:16]) / 16
    finish_line = round(0.9 * baseline_front16, 1)
    intrinsic_line = baseline["agg"]["ret_mean"]
    log({"event": "start", "leg_steps": LEG, "beta_const": BETA, "hard": HARD_LINE,
         "finish": finish_line, "intrinsic": intrinsic_line,
         "start_nt": START, "base": str(BASE_CKPT)})

    nt_chain = START            # real SB3 chain count (single count source)
    burned = 0                  # audit definition of partial crash steps; deducted live from the remaining allocatable quota
    spent_steps = 0             # observed new steps of all successful/failed attempts after START
    train_secs = 0.0
    prev_model = str(BASE_CKPT)
    leg_models = {}             # k -> (score16, model_path, breadth16)
    attempts = {}
    consec_low = 0
    stop_reason = None
    k = 1
    while k <= N_LEGS:
        cap = LEG
        leg_steps = budgeted_leg_steps(spent_steps, cap)
        if leg_steps < cap:
            log({"event": "leg_budget_shrunk", "leg": k, "steps": leg_steps,
                 "spent": spent_steps, "remaining": BUDGET_STEPS - spent_steps,
                 "note": "all completed new steps and failed burned steps are deducted from the hard budget, one by one"})
        if leg_steps == 0:
            log({"event": "budget_exhausted", "leg": k, "spent": spent_steps})
            break
        seed_k = 281_000 + 1_000 * (k - 1)      # single point of definition (cmd and log share the source)
        probes = reachable_probes(nt_chain, leg_steps)
        if len(probes) < 2:
            log({"event": "calib_trimmed", "leg": k, "probes": probes,
                 "note": "probe parts unreachable on a short leg are trimmed (pre-registered; probes_ok judged on the remainder)"})
        attempts[k] = attempts.get(k, 0) + 1
        run_name = f"v28-leg{k}"
        log({"event": "leg_start", "leg": k, "attempt": attempts[k], "beta": BETA,
             "steps": leg_steps, "seed": seed_k, "resume_from": prev_model,
             "probes": probes})
        res = run_leg(k, prev_model, leg_steps, probes, run_name, attempts[k], seed_k)
        train_secs += res["dt_sec"]
        expected = nt_chain + leg_steps

        # ---- crash interlock (before any verdict; count = real SB3 chain, +/-2048 slack covers the status lag) ----
        clean = (res["rc"] == 0 and res["model_fresh"]
                 and res["global_steps"] == expected)
        sampled = observed_attempt_steps(nt_chain, res, leg_steps)
        if not clean:
            partial = sampled
            charged = failed_attempt_charge(partial, leg_steps)
            spent_steps += charged
            burned += charged
            log({"event": "leg_crash", "leg": k, "attempt": attempts[k],
                 "rc": res["rc"], "global_steps": res["global_steps"],
                 "burned_observed": partial, "burned_charged": charged,
                 "burned_total": burned,
                 "spent_total": spent_steps,
                 "note": "re-run with the original config; burned steps are deducted live from later quota; the closing counter neither counts nor resets"})
            ensure_retry_budget(spent_steps, cap)
            if attempts[k] >= 4:  # stale calib/sentinel files are cleared when the next attempt enters; no rotation needed here
                stop_reason = (f"leg {k} crashed {attempts[k]} times in a row: training stops (the retry limit 4 is an"
                               " operational self-protection guard, not a pre-registered gate)")
                log({"event": "crash_halt", "why": stop_reason})
                attention(stop_reason)
                raise OperationalFailure(stop_reason)
            continue
        require(sampled == leg_steps, "observed steps of a clean close disagree with the allocation")
        spent_steps += sampled
        nt_chain = expected      # advance the real chain (the status lag does not enter the chain)

        # ---- G-oasis (leg 1 only; the round 3.0M point guarantees a sentinel row exists, no row = failure: fix for the v26 silent-skip incident) ----
        if k == 1:
            sent = RUNS / run_name / "sentinel.jsonl"
            lines = []
            if sent.exists():
                for l in sent.read_text().splitlines():
                    if '"sentinel": "v23"' in l:
                        try:
                            lines.append(json.loads(l))
                        except Exception:
                            pass  # skip half-written rows (process killed mid-write); must not crash the driver
            if not lines:
                stop_reason = "G-oasis failed: no sentinel row (v26 once skipped it silently; v28 requires it)"
                log({"event": "STOP", "why": stop_reason})
                attention(stop_reason)
                raise OperationalFailure(stop_reason)
            last = lines[-1]
            oasis_ok = last.get("dry", 1) == 0 and last.get("ff_dry", 0) > 0
            log({"event": "g_oasis", "dry": last.get("dry"), "ff_dry": last.get("ff_dry"),
                 "fresh": last.get("fresh"), "ok": oasis_ok})
            if not oasis_ok:
                stop_reason = "G-oasis failed: the learning window contains dry or ff_dry=0"
                log({"event": "STOP", "why": stop_reason})
                attention(stop_reason)
                raise OperationalFailure(stop_reason)

        # ---- G-CAL wiring gate (every leg; recorded only, not judged; the tripped bit is booked, not judged) ----
        calib_p = RUNS / run_name / "calib.jsonl"
        recs = ([json.loads(l) for l in calib_p.read_text().splitlines()]
                if calib_p.exists() else [])
        probes_ok = (all(any(p <= r["step"] < p + QUANTUM and r["g_ce"] > 0
                             and r["distill_ce"] > 0 for r in recs)
                         for p in probes) if probes else None)
        probe_status = ("SKIPPED" if probes_ok is None
                        else "PASS" if probes_ok else "FAIL")
        log({"event": "g_cal", "leg": k, "records": [
                {kk: r.get(kk) for kk in ("step", "g_pg", "g_ce", "teacher_diverge", "tripped")}
                for r in recs], "probes_ok": probes_ok, "probe_status": probe_status,
             "record_only": True})
        if probes_ok is False:
            stop_reason = f"G-CAL wiring failed (leg {k}: neither probe saw ce/g_ce>0): manual intervention"
            log({"event": "STOP", "why": stop_reason})
            attention(stop_reason)
            raise OperationalFailure(stop_reason)

        # ---- leg exam + width probe (pure post-processing) ----
        r = exam_retry(res["model"], f"v28-leg{k}", "7000-7015", f"leg {k} exam")
        if r is None:
            stop_reason = f"leg {k} exam failed twice in a row: manual autopsy"
            log({"event": "STOP", "why": stop_reason})
            attention(stop_reason)
            raise OperationalFailure(stop_reason)
        agg, rows = r
        score = round(agg["ret_mean"], 1)
        anchor_by_seed = {r["seed"]: r["ret"]
                          for r in read_comparable_anchor()["rows"]}
        bw = breadth_wins(rows, anchor_by_seed, 7000, 7015)
        if candidate_probe_eligible(probes_ok):
            leg_models[k] = (score, str(res["model"]), bw)
        else:
            log({"event": "candidate_ineligible", "leg": k,
                 "why": "G-CAL probes SKIPPED; this model must not enter the candidate pool"})
        log({"event": "leg_exam", "leg": k, "beta": BETA, "score": score,
             "died": agg["died"], "diverge": agg.get("script_divergence_rate"),
             "breadth16": bw, "sha": agg["_sha"], "model_sha": sha16(res["model"]),
             "global_steps": res["global_steps"], "nt_chain": nt_chain})

        # ---- trip lines (the hard trip takes precedence; all break into G3: budget protection, not punishment; trained legs keep their candidacy) ----
        if score < HARD_LINE:
            log({"event": "HARD_TRIP", "leg": k, "score": score,
                 "why": f"< {HARD_LINE}, training permanently stopped; completed legs enter the G3 candidate pool as usual"})
            attention(f"hard trip: leg {k} = {score}")
            break
        consec_low = consec_low + 1 if score < finish_line else 0
        if consec_low >= 2:
            log({"event": "early_finish", "leg": k, "score": score,
                 "why": f"{consec_low} consecutive clean legs < {finish_line}: early close, into G3"
                        " (budget protection, not punishment)"})
            attention(f"early close at leg {k}")
            break
        if res["dt_sec"] > 7200:
            log({"event": "sanity_finish", "leg": k, "dt_sec": res["dt_sec"],
                 "why": "leg wall clock >2h: sanity close, into G3 (the fix for v26's bare STOP)"})
            attention(f"sanity close: leg {k} wall clock {res['dt_sec']}s")
            break
        prev_model = str(res["model"])
        k += 1

    # ---- G3: mean top-2 union width top-1 (deduplicated, <=3; the starting checkpoint itself is not in the pool) ----
    if not leg_models:
        why = "no completed leg with a wiring PASS"
        log({"event": "STOP", "why": why})
        attention(why)
        raise OperationalFailure(why)
    by_mean = sorted(leg_models.items(), key=lambda kv: (-kv[1][0], kv[0]))
    by_breadth = sorted(leg_models.items(), key=lambda kv: (-kv[1][2], -kv[1][0], kv[0]))
    cand_legs = []
    for kk, _ in by_mean[:2] + [by_breadth[0]]:
        if kk not in cand_legs:
            cand_legs.append(kk)
    breadth_only = {by_breadth[0][0]} - {kk for kk, _ in by_mean[:2]}
    log({"event": "g3_candidates",
         "cands": [(kk, leg_models[kk][0], leg_models[kk][2]) for kk in cand_legs],
         "breadth_only": sorted(breadth_only)})

    finals = []
    for kk in cand_legs:
        sc16, mp, bw16 = leg_models[kk]
        r = exam_retry(pathlib.Path(mp), f"v28-G3-leg{kk}", "7000-7031", f"G3 leg {kk}")
        if r is None:
            stop_reason = f"G3 full-32 exam failed twice in a row (leg {kk}): manual autopsy"
            log({"event": "STOP", "why": stop_reason})
            attention(stop_reason)
            raise OperationalFailure(stop_reason)
        agg, rows = r
        anchor_by_seed = {row["seed"]: row["ret"]
                          for row in read_comparable_anchor()["rows"]}
        by_seed = {row["seed"]: row["ret"] for row in rows}
        require(len(rows) == len(by_seed), f"G3 leg {kk} contains duplicate seeds")
        require(set(by_seed) == set(range(7000, 7032)), f"G3 leg {kk} seed set is malformed")
        diffs = [by_seed[s] - anchor_by_seed[s] for s in range(7000, 7032)]
        mean_diff = sum(diffs) / 32
        wins = sum(d > 0 for d in diffs)
        back16 = breadth_wins(rows, anchor_by_seed, 7016, 7031)
        void = agg["override_rate"] >= R4["override_void"]
        r4_ok = (agg["farm_descend_rate"] <= R4["farm_descend_rate"]
                 and agg["override_rate"] < R4["override_sentinel"]
                 and agg["cap_rate"] < R4["cap_rate"]
                 and R4["farm_tau_lo"] <= agg["farm_tau_mean"] <= R4["farm_tau_hi"])
        qual_ok = agg["ret_mean"] >= G3_MEAN and agg["died"] <= G3_DEATHS and r4_ok and not void
        sideline_ok = (kk not in breadth_only) or (back16 >= SIDELINE_BACK16)
        launch = qual_ok and mean_diff >= 4.0 and wins >= 18 and sideline_ok
        finals.append({"leg": kk, "mean": agg["ret_mean"], "died": agg["died"],
                       "mean_diff": mean_diff, "wins": wins, "back16": back16,
                       "void": void, "qual_ok": qual_ok, "sideline_ok": sideline_ok,
                       "launch": launch, "diverge": agg.get("script_divergence_rate"),
                       "model": mp, "_sha": agg["_sha"]})
        log({"event": "g3_full32", "leg": kk, "mean": agg["ret_mean"], "died": agg["died"],
             "r4_ok": r4_ok, "data_void": void, "qualified": qual_ok,
             "mean_diff": round(mean_diff, 2), "wins": wins, "back16": back16,
             "sideline_ok": sideline_ok, "launch": launch,
             "diverge": agg.get("script_divergence_rate"),
             "override": agg["override_rate"], "descend_rate": agg["farm_descend_rate"],
             "tau": agg["farm_tau_mean"], "depth_median": agg.get("depth_median")})

    # ---- launch verdict ----
    launchers = [f for f in finals if f["launch"]]
    if launchers:
        launchers.sort(key=lambda f: -f["mean"])
        band = [f for f in launchers if launchers[0]["mean"] - f["mean"] <= 0.05]
        w = sorted(band, key=lambda f: (-f["wins"], f["leg"]))[0]
        golden_cmd = (f"{PY} {ROOT / 'train' / 'eval_assembled.py'} --worker "
                      f"{pathlib.Path(w['model']).with_suffix('')} --seeds 9000-9031 "
                      f"--tag v28-golden --board")
        log({"event": "GOLDEN_AUTHORIZED", "leg": w["leg"], "probe32_mean": w["mean"],
             "died": w["died"], "wins": w["wins"], "mean_diff": round(w["mean_diff"], 2),
             "diverge": w["diverge"], "model": w["model"],
             "model_sha": sha16(pathlib.Path(w["model"])), "full32_sha": w["_sha"],
             "golden_cmd": golden_cmd,
             "note": "the gold-standard evaluation is started manually, single arm, once; after the opening a golden_result event must be written back"})
        attention(f"gold-standard evaluation awaiting manual launch: leg {w['leg']} (command in the ledger)")
        return

    # ---- no launch: three-key exhaustive dispatch (panel blocker fix) ----
    nonvoid = [f for f in finals if not f["void"]]
    if not nonvoid:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": "all candidate data void: no winner, eligibility failed (outside statistical power)",
             "finals": [{k2: (round(v, 3) if isinstance(v, float) else v)
                         for k2, v in f.items() if k2 != "model"} for f in finals]})
        attention("verdict: all candidates void")
        return
    wv = sorted(nonvoid, key=lambda f: (-f["mean"], f["leg"]))[0]
    if not wv["qual_ok"]:
        verdict = ("eligibility failed, width question unanswered (outside statistical power): blocked by="
                   + ("death count" if wv["died"] > G3_DEATHS else "sentinel/mean eligibility"))
    elif wv["wins"] >= 18 and wv["mean_diff"] >= 4.0 and not wv["sideline_ok"]:
        verdict = (f"width-channel candidate failed the out-of-sample side line (back 16 {wv['back16']}/16 < "
                   f"{SIDELINE_BACK16}): does not spend the gold run; the guard against selection inflation worked")
    elif wv["wins"] >= 18 and wv["mean_diff"] < 4.0:
        verdict = "width reached but magnitude not: point-estimate width improvement, does not spend the gold run; left for the workstation line"
    elif wv["mean"] >= intrinsic_line:
        verdict = (f"width problem confirmed intrinsic (within statistical power): the mean held/exceeded the start {intrinsic_line} "
                   "while width was not reached: the under-training hypothesis is rejected; the mechanism prescriptions (anchor follows the king/course sampling) are promoted to the workstation line")
    elif wv["mean"] >= PLATEAU_LINE:
        verdict = (f"endurance without mean gain ([{PLATEAU_LINE},{intrinsic_line}) tier),"
                   " width question unanswered: neither intrinsic nor degradation is judged")
    else:
        verdict = "endurance degraded (<103); leg 6 is a local peak for this recipe"
    log({"event": "VERDICT_PATH", "golden_authorized": False, "verdict": verdict,
         "verdict_winner_leg": wv["leg"], "winner_mean": wv["mean"],
         "winner_wins": wv["wins"], "winner_mean_diff": round(wv["mean_diff"], 2),
         "finals": [{k2: (round(v, 3) if isinstance(v, float) else v)
                     for k2, v in f.items() if k2 != "model"} for f in finals]})
    attention(f"verdict (no launch): {verdict}")


if __name__ == "__main__":
    main()
