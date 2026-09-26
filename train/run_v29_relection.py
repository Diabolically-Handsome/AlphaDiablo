"""v29 "manager re-education" driver (sole executor of the clauses of docs/prereg/PREREG-v29.md; a targeted rework of run_v25_election.py).

Clone difference table (maps clause by clause to PREREG-v29 D2):
- Cast: worker = v28-worker-leg1 (zip+npz); anchor = v28-G3-leg1.json (112.4, sha pinned)
- The M-warm arm and the warm-sd export stage are physically removed; two arms = fresh (ent .02) / explore (ent .08), 160k decisions
- FLOOR_REPRO 85->103.9 (the panel's corrected final value); tags v29-*; pairing joined on the seed key + set assertion
- The full v28 operational guard set: top-level exception catch-all, training 4h/evaluation 30min timeouts, exam refuses to overwrite + .void rotation
  + evaluation logs, preflight (anchor sha/worker in place/target archives including v29-golden absent), NEEDS_ATTENTION
- Full-32 events gain depth instrumentation: seeds with depth>=2 / DIVE per episode / descent bonus paid per episode
- No-launch tiers are exhaustive (the v28 lesson): ineligible tier + substitution, point-estimate gain tier (>=+4 and <18 wins), probe level, incumbent stays
- GOLDEN_AUTHORIZED carries the exact gold-standard evaluation command (with --manager-npz) + both shas
The gold-standard evaluation is not launched here (started manually, single arm, once). Ledger: train/runs/v29/gate_ledger.jsonl.
Usage: .venv/bin/python train/run_v29_relection.py
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

from eval_contract import (PROTOCOL_VERSION, OperationalFailure, OutputReservationError,
                           exclusive_lock, expected_eval_identity,
                           freeze_eval_identity, read_eval_archive,
                           verify_eval_identity)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
V29 = RUNS / "v29"
V29.mkdir(parents=True, exist_ok=True)
LEDGER = V29 / "gate_ledger.jsonl"
EVAL = RUNS / "eval-assembled"

W_ZIP = str(ROOT / "train" / "models" / "v28-worker-leg1" / "model_final")  # the eval side appends .zip automatically
W_NPZ = ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz"
ARCHIVE = EVAL / "v28-G3-leg1.json"        # reference row 112.4 (incumbent v22-H x the same worker, per seed)
ARCHIVE_SHA = "6fc6a44c7862424ab5f71ff3a5031adfd34a3e33f9f4f2f8aee781a07711e59d"
# 2026-09-23: re-saved with neutral local paths (content otherwise unchanged); was 2f7bc9dd810956c3
W_ZIP_SHA = "0c6f014da19c3bf27b208d55adc76be13fcad8bc5f744b95f4f743be97426434"
W_NPZ_SHA = "976b6c05edaa0a32bb30bd372782e1201c72b029cedcbb3a5bf2361d34f27f8a"

STEPS = 160_000       # 4x v22-H's own budget (clock anchor 27.5 decisions/s ~ 97 minutes/arm)
ABANDON = 75.0        # early abandonment gate (both arms' 16-seed means < this value)
PAIRED_DIFF = 4.0     # launch line: paired mean difference >= +4
PAIRED_WINS = 18      # and paired wins >= 18/32
DEATHS_MAX = 6
FLOOR_REPRO = 103.9   # winner < this value -> "retraining did not reproduce the reference level, proposition not examined" (=0.9239x112.4, strictly following the v25 ratio 85/92)
R4 = {"descend": 0.0204, "override_sentinel": 0.03, "override_void": 0.08, "cap": 0.05}
CALIBRATED_PROTOCOL_VERSION = 2

ARMS = {
    "v29-mfresh": ["--ent-coef", "0.02", "--lr", "3e-4", "--seed", "22"],
    "v29-mexplore": ["--ent-coef", "0.08", "--lr", "3e-4", "--seed", "24"],
}


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    with open(V29 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def sha16(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def sha256(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_calibrated_protocol() -> None:
    if PROTOCOL_VERSION != CALIBRATED_PROTOCOL_VERSION:
        raise OperationalFailure(
            "v29 ABANDON/eligibility/R4 and other verdict lines are calibrated only under pre-v3 environment semantics; "
            "re-run the protocol-v3 baseline and update the pre-registration by hand first; mixing in the old thresholds is forbidden"
        )


def read_comparable_anchor() -> dict:
    """The v28 incumbent anchor is comparable only with the fixed worker/default manager under the current runtime."""
    try:
        snapshot = freeze_eval_identity(ROOT, W_ZIP, None)
        expected = expected_eval_identity(
            snapshot, tag="v28-G3-leg1", seeds=range(7000, 7032))
        document = read_eval_archive(ARCHIVE, **expected)
        verify_eval_identity(snapshot, ROOT)
        return document
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise OperationalFailure(
            "v28-G3-leg1 does not satisfy the current schema-v2 comparability contract; "
            "after an environment-semantics change, re-run the baseline with the fixed v28 leg1 worker + default manager"
        ) from exc


def run(cmd, logfile, timeout) -> int:
    with open(V29 / logfile, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)   # kill the whole group: keeps SubprocVecEnv grandchildren from being orphaned
            except ProcessLookupError:
                pass
            proc.wait()
            return 124    # hang guard: booked as crash/failure (operational guard, not a verdict input)


def zip_steps(p: pathlib.Path) -> int:
    """Real SB3 chain reading (panel blocker fix: 160000 is divisible by 256, the throttled status count always lags)."""
    try:
        with zipfile.ZipFile(p) as z:
            return int(json.loads(z.read("data"))["num_timesteps"])
    except Exception:
        return 0


def exam(worker, tag, seeds, manager_npz=None):
    out = EVAL / f"{tag}.json"
    require(not out.exists(), f"archive immutability: {out} already exists, refusing to overwrite")
    lo, hi = (int(x) for x in seeds.split("-", 1))
    seed_values = list(range(lo, hi + 1))
    require(seed_values and lo >= 0, f"illegal seed range: {seeds}")
    snapshot = freeze_eval_identity(ROOT, worker, manager_npz)
    expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
    worker_arg = (worker if snapshot["worker"]["kind"] in {"script", "bc"}
                  else snapshot["worker"]["path"])
    cmd = [PY, "train/eval_assembled.py", "--worker", str(worker_arg),
           "--manager-npz", snapshot["manager"]["path"],
           "--seeds", seeds, "--tag", tag]
    if run(cmd, f"exam-{tag}.{time.time_ns()}.log", timeout=1_800) != 0:
        if out.exists():    # rotate a half-written archive to make way for the retake
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


def exam_retry(worker, tag, seeds, manager_npz=None):
    d = exam(worker, tag, seeds, manager_npz)
    if d is None:
        log({"event": "exam_crash", "tag": tag, "note": "evaluation failed; retaking once under the crash clause"})
        d = exam(worker, tag, seeds, manager_npz)
    return d


def dive_per_ep(rows) -> float:
    return sum(r["mode_seq"].count("D") for r in rows) / max(1, len(rows))


def depth2_count(rows) -> int:
    return sum(1 for r in rows if r["depth"] >= 2)


def bonus_per_ep(rows) -> float:
    # Descent bonus payout: depth=d pays 8x(1+2+...+(d-1)); 0 for d<=1
    return sum(8 * sum(range(1, r["depth"])) for r in rows) / max(1, len(rows))


def by_seed(rows) -> dict:
    m = {r["seed"]: r for r in rows}
    require(len(rows) == len(m), "seed set is malformed (contains a duplicate seed)")
    require(set(m) == set(range(7000, 7032)), "seed set is malformed (must be 7000-7031)")
    return m


def preflight():
    require_calibrated_protocol()
    worker_zip = pathlib.Path(W_ZIP + ".zip")
    require(worker_zip.exists() and sha256(worker_zip) == W_ZIP_SHA,
            "worker zip missing or sha drift")
    require(W_NPZ.exists() and sha256(W_NPZ) == W_NPZ_SHA,
            "worker npz missing or sha drift (the item exported on launch day with parity 0/1000)")
    read_comparable_anchor()
    tags = ["v29-GA0", "v29-golden"] + [f"{a}-{s}" for a in ARMS for s in ("s16", "full32")]
    for t in tags:
        require(not (EVAL / f"{t}.json").exists(), f"target archive already exists: {t} (restart protocol: .void it first)")
    for a in ARMS:
        require(not (RUNS / a).exists(), f"leftover run directory: {a} (restart protocol: archive it first)")
    log({"event": "preflight_ok", "archive_sha": sha16(ARCHIVE),
         "worker_npz_sha": sha16(W_NPZ)})


def main():
    try:
        with exclusive_lock(V29 / ".driver.lock", "v29 driver"):
            _main()
    except (OperationalFailure, OutputReservationError) as e:
        log({"event": "OPERATIONAL_FAILURE", "why": str(e)})
        attention("operational failure:\n" + str(e))
        raise SystemExit(2) from e
    except Exception as e:   # catch-all clause: every unexpected exception must be recorded; no silent death
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("driver died with an exception:\n" + traceback.format_exc())
        raise


def _main():
    preflight()
    ref = read_comparable_anchor()
    floor_repro = round(ref["agg"]["ret_mean"] * 85.0 / 92.0, 1)
    log({"event": "start", "prereg": "docs/prereg/PREREG-v29.md", "steps": STEPS,
         "paired_line": PAIRED_DIFF, "wins_line": PAIRED_WINS,
         "floor": floor_repro})
    ref_rows = by_seed(ref["rows"])

    # ---- G-A0: instrument regression (npz worker + default manager == the 112.4 anchor, 32/32) ----
    ga0 = exam_retry(str(W_NPZ), "v29-GA0", "7000-7031")
    if ga0 is None:
        why = "G-A0 exam process failed repeatedly"
        log({"event": "STOP", "why": why})
        attention(why)
        raise OperationalFailure(why)
    bad = [s for s, r in by_seed(ga0["rows"]).items()
           if (abs(r["ret"] - ref_rows[s]["ret"]) > 0.01
               or r["died"] != ref_rows[s]["died"]
               or r["depth"] != ref_rows[s]["depth"]
               or r["mode_seq"] != ref_rows[s]["mode_seq"])]
    log({"event": "g_a0", "mismatch_seeds": bad, "n_ok": 32 - len(bad)})
    if bad:
        why = "G-A0 bit-level regression mismatch: re-anchor by hand under the pre-registered fallback clause"
        log({"event": "STOP", "why": why})
        attention(f"G-A0 mismatched seeds: {bad}")
        raise OperationalFailure(why)

    # ---- train the two arms in series ----
    npz = {}
    for name, extra in ARMS.items():
        cmd = [PY, "train/train_ppo.py", "--options", "--algo", "mppo", "--gamma", "1.0",
               "--max-steps", "3000", "--n-steps", "64", "--num-envs", "4",
               "--total-steps", str(STEPS), "--worker-npz", str(W_NPZ),
               "--run-name", name] + extra
        log({"event": "arm_start", "arm": name, "cmd_extra": extra})
        t0 = time.time()
        rc = run(cmd, f"train-{name}.log", timeout=14_400)   # 4h hang guard
        sp = RUNS / name / "status.json"
        try:
            steps = json.loads(sp.read_text())["total_steps"] if sp.exists() else 0
        except Exception:
            steps = 0
        nt = zip_steps(RUNS / name / "model_final.zip")   # the single step-count source for the target gate (real SB3 chain)
        log({"event": "arm_done", "arm": name, "rc": rc, "nt_zip": nt,
             "steps_status": steps, "dt_min": round((time.time() - t0) / 60, 1)})
        if rc != 0 or nt != STEPS:
            why = (f"{name} training did not reach its target (rc={rc}, nt_zip={nt}, "
                   f"status={steps}): proposition not examined; this version adds no retraining (v25 clause)")
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
        npz[name] = str(out)
        log({"event": "g_a0m", "arm": name, "npz_sha": sha16(out)})

    # ---- early abandonment gate (16 seeds) ----
    s16 = {}
    for name in ARMS:
        d = exam_retry(W_ZIP, f"{name}-s16", "7000-7015", manager_npz=npz[name])
        if d is None:
            why = f"{name} screen exam failed repeatedly"
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
        s16[name] = d["agg"]["ret_mean"]
        log({"event": "screen16", "arm": name, "score": d["agg"]["ret_mean"],
             "died": d["agg"]["died"]})
    if all(v < ABANDON for v in s16.values()):
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"both arms' screens < {ABANDON}: training failed, the re-education proposition was not examined (full 32 skipped)"})
        attention("verdict: training failed, proposition not examined")
        return

    # ---- both arms full 32 (with depth instrumentation) ----
    full = {}
    for name in ARMS:
        d = exam_retry(W_ZIP, f"{name}-full32", "7000-7031", manager_npz=npz[name])
        if d is None:
            why = f"{name} full-32 exam failed repeatedly"
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
        full[name] = d
        a = d["agg"]
        log({"event": "full32", "arm": name, "mean": a["ret_mean"], "died": a["died"],
             "dive_per_ep": round(dive_per_ep(d["rows"]), 2),
             "depth2_seeds": depth2_count(d["rows"]),
             "bonus_per_ep": round(bonus_per_ep(d["rows"]), 2),
             "depth_median": a.get("depth_median"), "tau": a["farm_tau_mean"],
             "override": a["override_rate"], "descend": a["farm_descend_rate"],
             "sha": a["_sha"]})
    fr = by_seed(full["v29-mfresh"]["rows"])
    ex = by_seed(full["v29-mexplore"]["rows"])
    r29_2 = sum(ex[s]["ret"] - fr[s]["ret"] for s in fr) / 32
    log({"event": "r29_2", "paired_explore_minus_fresh_mean": round(r29_2, 2)})

    # ---- per-arm eligibility (panel blocker fix: the winner is taken only from eligible arms) ----
    def qual_of(d):
        a = d["agg"]
        dpe_ = dive_per_ep(d["rows"])
        void_ = (a["override_rate"] >= R4["override_void"]
                 or (dpe_ > 1 and a["died"] > 6))
        hard_ok_ = a["farm_descend_rate"] <= R4["descend"] and a["cap_rate"] < R4["cap"]
        override_ok_ = a["override_rate"] < R4["override_sentinel"]
        dual_ = dpe_ > 1 and hard_ok_ and not override_ok_   # v25 dual-attribution clause
        ok_ = (a["died"] <= DEATHS_MAX and not void_
               and ((hard_ok_ and override_ok_) or dual_))
        return {"qual_ok": ok_, "void": void_, "dual_attr": dual_}

    quals = {n: qual_of(full[n]) for n in ARMS}
    log({"event": "quals", **{n: quals[n] for n in ARMS}})
    pool = [n for n in ARMS if quals[n]["qual_ok"]]
    if not pool:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": "both arms ineligible (deaths/sentinel/void): no winner, the re-education proposition unanswered (outside statistical power)",
             "arms": {n: {"mean": full[n]["agg"]["ret_mean"],
                          "died": full[n]["agg"]["died"], **quals[n]} for n in ARMS}})
        attention("verdict: both arms ineligible, no winner (depth instrumentation recorded with the full32 events)")
        return
    prelim = max(ARMS, key=lambda n: full[n]["agg"]["ret_mean"])
    if prelim not in pool:
        log({"event": "substitution", "blocked": prelim, "why": quals[prelim],
             "note": "the mean winner was blocked by eligibility; per D3-2 the eligible arm takes its place"})
    ms = {n: full[n]["agg"]["ret_mean"] for n in pool}
    band = [n for n in pool if max(ms.values()) - ms[n] <= 0.05]
    if len(band) > 1:
        dmin = min(full[n]["agg"]["died"] for n in band)
        band = [n for n in band if full[n]["agg"]["died"] == dmin]
        winner = "v29-mfresh" if "v29-mfresh" in band else band[0]
    else:
        winner = band[0]
    W = full[winner]
    wa = W["agg"]
    wrows = by_seed(W["rows"])
    dual_attr = quals[winner]["dual_attr"]
    log({"event": "winner", "arm": winner, "mean": wa["ret_mean"], "died": wa["died"],
         "substituted": winner != prelim})

    # ---- depth side verdict (a scientific conclusion that does not move the throne; PREREG-v29 D3-7) ----
    d2, dpe = depth2_count(W["rows"]), dive_per_ep(W["rows"])
    if d2 >= 12 and 0.5 <= dpe <= 3 and wa["died"] <= DEATHS_MAX:
        depth_verdict = "depth economy learned (>=12 maps reach level 2, DIVE share within the band)"
    elif d2 <= 7:
        depth_verdict = "re-education did not unlock depth (<= baseline 7)"
    else:
        depth_verdict = (f"out of band (depth2={d2}, dive={dpe:.2f}, died={wa['died']}),"
                         " recorded, no narrative")
    log({"event": "depth_verdict", "depth2_seeds": d2, "dive_per_ep": round(dpe, 2),
         "bonus_per_ep": round(bonus_per_ep(W["rows"]), 2), "verdict": depth_verdict,
         "note": "side verdict; the throne and Mark-I determinations follow D3-6 and the roadmap clause (docs/design/ROADMAP-course-plan.md) (guarding against over-narration)"})

    # ---- reproduction floor ----
    ref = read_comparable_anchor()
    ref_rows = by_seed(ref["rows"])
    floor_repro = round(ref["agg"]["ret_mean"] * 85.0 / 92.0, 1)
    if wa["ret_mean"] < floor_repro:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"winner {wa['ret_mean']} < {floor_repro}: retraining did not reproduce the reference level,"
                    f" the re-education proposition not examined; depth side verdict: {depth_verdict}"})
        attention("verdict: reference level not reproduced")
        return

    # ---- launch criterion (paired vs the 112.4 anchor, keyed by seed; the winner's eligibility is guaranteed by pool) ----
    diffs = [wrows[s]["ret"] - ref_rows[s]["ret"] for s in sorted(ref_rows)]
    pd_mean = sum(diffs) / 32
    pd_wins = sum(d > 0 for d in diffs)
    launch = pd_mean >= PAIRED_DIFF and pd_wins >= PAIRED_WINS
    log({"event": "launch_check", "paired_mean": round(pd_mean, 2), "paired_wins": pd_wins,
         "died": wa["died"], "dive_per_ep": round(dpe, 2),
         "dual_attribution": dual_attr, "tau_note": wa["farm_tau_mean"]})

    P_LINE = ("P line quick reference (decided in order): deaths>6 -> revert; gold>=101.2 and deaths<=4 -> P29 takes the throne;"
              " in (97.2,101.2) and deaths<=4 -> point-estimate gain, throne unchanged; >97.2 and deaths 5-6 -> tie (safety);"
              " in [93.9,97.2] -> tie; <93.9 -> revert")
    if launch:
        golden_cmd = (f"{PY} {ROOT / 'train' / 'eval_assembled.py'} --worker {W_ZIP} "
                      f"--manager-npz {npz[winner]} --seeds 9000-9031 "
                      f"--tag v29-golden --board")
        dual_note = ("[dual attribution undecided] override crossed its line and was let through on the dual-attribution path; before spending the gold run a human must decide"
                     " mix drift vs real degradation and write back a dual_attr_ruling event; decide first, then spend; "
                     if dual_attr else "")
        log({"event": "GOLDEN_AUTHORIZED", "arm": winner, "probe32_mean": wa["ret_mean"],
             "died": wa["died"], "wins": pd_wins, "mean_diff": round(pd_mean, 2),
             "manager_npz": npz[winner], "manager_npz_sha": sha16(npz[winner]),
             "full32_sha": wa["_sha"], "golden_cmd": golden_cmd, "p_line": P_LINE,
             "note": dual_note + "the gold-standard evaluation is started manually, single arm, once; losing/non-launched arms never see"
                     " the 9000 range; write back a golden_result event after the opening"})
        attention(dual_note + f"gold-standard evaluation awaiting manual launch: {winner} (command and P line quick reference in the ledger);"
                  f" depth side verdict: {depth_verdict}")
        return

    # ---- no launch: exhaustive dispatch (the winner is eligible; the no-winner tier comes first; the width-shift note goes along) ----
    wins_note = (f" (width-shift note: won {pd_wins}/32 >=14, the tier does not change)"
                 if pd_wins >= 14 else "")
    if pd_mean >= PAIRED_DIFF and pd_wins < PAIRED_WINS:
        verdict = (f"mean gain +{pd_mean:.2f} but width not reached (won {pd_wins}/32 < 18)"
                   ": a point-estimate gain, does not spend the gold run; rematch deferred to the workstation line")
    elif pd_mean >= 2.0:
        verdict = (f"paired mean diff {pd_mean:.2f} in [+2,+4): a probe-level improvement, does not spend the gold run"
                   f" (won {pd_wins}/32){wins_note}")
    else:
        verdict = (f"paired mean diff {pd_mean:.2f} <+2: the incumbent stays, re-education brings no gain (power-limited)"
                   f" (won {pd_wins}/32){wins_note}")
    log({"event": "VERDICT_PATH", "golden_authorized": False,
         "verdict": verdict, "depth_verdict": depth_verdict,
         "winner": winner, "winner_mean": wa["ret_mean"],
         "paired_mean": round(pd_mean, 2), "paired_wins": pd_wins})
    attention(f"verdict (no launch): {verdict}; depth side verdict: {depth_verdict}")


if __name__ == "__main__":
    main()
