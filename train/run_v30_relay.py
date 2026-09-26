"""v30 "worker relay" two-arm driver (sole executor of the final clauses of docs/prereg/PREREG-v30.md).

Structure: preflight (full sha chain) -> king anchor sd export + G-KL-W fidelity gate -> G-A0W (assembled
regression == the 140.3 archive) -> two arms in series (each a 2-leg resume chain; the single variable between arms = the leash teacher:
king self-anchor / old bc teacher) -> both arms full 32 (M29 assembly) -> eligibility/substitution (v29) -> scientific main verdict
(four tiers in order) -> floor -> dual-anchor launch (with the anti-free-riding prerequisite + the deep-level casualty conjunction) ->
GOLDEN_AUTHORIZED / exhaustive no-launch dispatch. The gold-standard evaluation is started manually (single arm, once).
All 33 panel decisions are implemented: three identity-chain latches, per-arm G-KL gates, G-oasis ff_dry downgraded,
sentinel = the v29 qual_of definition, clock recalibration (sanity 4h / timeout 4.5h), trip line 124.95
pinned, every evaluation uses --manager-npz, a new-evidence conjunction added to the launch line, the can't-learn tier named.
Usage: .venv/bin/python train/run_v30_relay.py
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
V30 = RUNS / "v30"
V30.mkdir(parents=True, exist_ok=True)
LEDGER = V30 / "gate_ledger.jsonl"
DRIVER_LOCK = V30 / ".driver.lock"
DRIVER_LOCK_PURPOSE = "v30 relay driver"
EVAL = RUNS / "eval-assembled"

M29_NPZ = RUNS / "v29-mfresh" / "policy.npz"
M29_SHA = "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6"
W_ZIP = ROOT / "train" / "models" / "v28-worker-leg1" / "model_final.zip"
# 2026-09-23: re-saved with neutral local paths (content otherwise unchanged); was 2f7bc9dd810956c3
W_SHA = "0c6f014da19c3bf27b208d55adc76be13fcad8bc5f744b95f4f743be97426434"
W_NPZ = ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz"
W_NPZ_SHA = "976b6c05edaa0a32bb30bd372782e1201c72b029cedcbb3a5bf2361d34f27f8a"
START_NT = 3_497_984            # v28-leg1 zip num_timesteps (=7x499,712, asserted at launch)
SCI_ANCHOR = EVAL / "v29-mfresh-full32.json"    # 140.3 (science anchor: same manager, single variable)
SCI_SHA = "08633101c010a2975b9001a71660bb50d43e681842e3fe1befdfdfd48f99ce63"
LAUNCH_ANCHOR = EVAL / "v28-G3-leg1.json"       # 112.4 (launch anchor: the incumbent assembled agent)
LAUNCH_SHA = "6fc6a44c7862424ab5f71ff3a5031adfd34a3e33f9f4f2f8aee781a07711e59d"
SCREEN_BASE_JSON = EVAL / "v29-mfresh-s16.json"  # trip-line baseline 147.0 (16 seeds, M29 assembly)
BC_SD = RUNS / "bc-worker" / "policy_sd.pt"
KING_SD = V30 / "king_anchor_sd.pt"             # stable path (panel major: no timestamps)

LEG = 244 * 2048                # 499,712 (quantum 2048 = n_steps x num_envs, v28 definition)
QUANTUM = 2048
LEGS = 2
BUDGET_STEPS = LEGS * LEG       # hard budget of new steps, independent per arm
BETA = 0.015625
HARD_LINE = 62.8                # historical disaster floor (more conservative on the 140 scale, kept)
TRIP_LINE = 124.95              # = 0.85x147.0 (v29-mfresh-s16, 16 seeds, M29 definition)
SCREEN_BASE = 147.0
FLOOR = 129.1                   # = 0.92x140.3 (lineage-ratio precedent v25/v29; an independent gate)
PD112_LINE, WINS112_LINE, DEATHS_MAX = 4.0, 18, 6
PRE140_LINE = 2.0               # anti-free-riding prerequisite (panel blocker: raised to the same line as scientific "valid")
D2DEATH_LAUNCH_MAX = 3          # launch conjunction: deep-level casualties no worse than baseline 3 (guards against narrative merging)
R4 = {"descend": 0.0204, "override_sentinel": 0.03, "override_void": 0.08, "cap": 0.05}
CALIBRATED_PROTOCOL_VERSION = 2
ARMS = {"king": {"base_seed": 301_000, "override": str(KING_SD)},
        "bc": {"base_seed": 305_000, "override": None}}


class OperationalFailure(RuntimeError):
    """Infrastructure/wiring failure; distinct from a normal scientific trip line, the process must exit non-zero."""


def budgeted_leg_steps(spent_steps: int, cap: int = LEG) -> int:
    remaining = max(0, BUDGET_STEPS - spent_steps)
    return min(cap, (remaining // QUANTUM) * QUANTUM)


def ensure_retry_budget(spent_steps: int, cap: int = LEG) -> None:
    if budgeted_leg_steps(spent_steps, cap) == 0:
        raise OperationalFailure("failed attempts exhausted the hard budget; the current leg cannot be completed")


def observed_attempt_steps(base: int, global_steps: int, status_steps: int,
                           allocated: int) -> int:
    observed = max(int(global_steps), int(status_steps))
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
    """Only an explicit PASS can be an authorization input; the SKIPPED of an empty probe must fail closed."""
    return probes_ok is True


def missing_authorization_arms(models: dict) -> list[str]:
    """A two-arm experiment missing either arm must never degrade into a single-arm authorization."""
    return sorted(set(ARMS) - set(models))


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    with open(V30 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def sha256(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def sha16(p) -> str:
    return sha256(p)[:16]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_calibrated_protocol() -> None:
    if PROTOCOL_VERSION != CALIBRATED_PROTOCOL_VERSION:
        raise OperationalFailure(
            "v30 relay HARD_LINE/eligibility/R4/science and launch static thresholds are calibrated only under pre-v3 environment semantics; "
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


def require_sha256(path: pathlib.Path, expected_sha: str, name: str) -> None:
    require(path.is_file() and sha256(path) == expected_sha,
            f"{name} sha drift/missing: {path}")


def read_comparable_reference(path: pathlib.Path, *, tag: str, worker,
                              manager, seeds: range, name: str) -> dict:
    """Active science/launch anchors accept only schema-v2 archives of the current semantics and a fixed assembly."""
    try:
        snapshot = freeze_eval_identity(ROOT, worker, manager)
        expected = expected_eval_identity(snapshot, tag=tag, seeds=seeds)
        document = read_eval_archive(path, **expected)
        verify_eval_identity(snapshot, ROOT)
        return document
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise OperationalFailure(
            f"{name} does not satisfy the current schema-v2 comparability contract; "
            "after an environment-semantics change, re-run the baseline with the fixed worker/manager"
        ) from exc


def read_comparable_science() -> dict:
    return read_comparable_reference(
        SCI_ANCHOR, tag="v29-mfresh-full32", worker=W_ZIP,
        manager=M29_NPZ, seeds=range(7000, 7032), name="science anchor")


def read_comparable_launch() -> dict:
    return read_comparable_reference(
        LAUNCH_ANCHOR, tag="v28-G3-leg1", worker=W_ZIP,
        manager=None, seeds=range(7000, 7032), name="launch anchor")


def read_comparable_screen() -> dict:
    return read_comparable_reference(
        SCREEN_BASE_JSON, tag="v29-mfresh-s16", worker=W_ZIP,
        manager=M29_NPZ, seeds=range(7000, 7016), name="trip-line baseline")


def zip_steps(p: pathlib.Path) -> int:
    try:
        with zipfile.ZipFile(p) as z:
            return int(json.loads(z.read("data"))["num_timesteps"])
    except Exception:
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


def run(cmd, logfile, timeout) -> int:
    with open(V30 / logfile, "w") as lf:
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


def exam(model_path, tag, seeds):
    require_sha256(M29_NPZ, M29_SHA, "M29 npz")
    out = EVAL / f"{tag}.json"
    require(not out.exists(), f"archive immutability: {out} already exists, refusing to overwrite")
    try:
        seed_values = parse_seed_range(seeds)
        snapshot = freeze_eval_identity(ROOT, model_path, M29_NPZ)
        expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
        command = [PY, "train/eval_assembled.py",
                   "--worker", snapshot["worker"]["path"],
                   "--manager-npz", snapshot["manager"]["path"],
                   "--seeds", seeds, "--tag", tag]
        rc = run(command, f"exam-{tag}.{time.time_ns()}.log", timeout=1_800)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    if rc != 0:
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    try:
        d = read_eval_archive(out, **expected)
        verify_eval_identity(snapshot, ROOT)
        d["agg"]["_sha"] = sha16(out)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    return d


def exam_retry(model_path, tag, seeds):
    d = exam(model_path, tag, seeds)
    if d is None:
        log({"event": "exam_crash", "tag": tag, "note": "evaluation failed; retaking once"})
        d = exam(model_path, tag, seeds)
    return d


def by_seed(rows) -> dict:
    m = {r["seed"]: r for r in rows}
    require(len(rows) == len(m), "seed set is malformed (contains a duplicate seed)")
    require(set(m) == set(range(7000, 7032)), "seed set is malformed (must be 7000-7031)")
    return m


def metrics(d) -> dict:
    rows, a = d["rows"], d["agg"]
    by_seed(rows)  # seal off missing and duplicate rows before any mean/death count enters model selection
    hist = a.get("worker_action_hist") or {}
    counts = ({int(k): int(v) for k, v in hist.items()} if isinstance(hist, dict)
              else {i: int(v) for i, v in enumerate(hist)})   # JSON dict keys are strings (found in an autopsy)
    calls = sum(counts.values())
    return {"mean": a["ret_mean"], "died": a["died"],
            "dive_per_ep": round(sum(r["mode_seq"].count("D") for r in rows) / len(rows), 2),
            "depth2_seeds": sum(1 for r in rows if r["depth"] >= 2),
            "d2_deaths": sum(1 for r in rows if r["died"] and r["depth"] >= 2),
            "bonus_per_ep": round(sum(8 * sum(range(1, r["depth"])) for r in rows) / len(rows), 2),
            "a13_share": round(counts.get(13, 0) / calls, 4) if calls else None,
            "tau": a["farm_tau_mean"], "override": a["override_rate"],
            "descend": a["farm_descend_rate"], "cap": a["cap_rate"],
            "depth_median": a.get("depth_median"), "sha": a["_sha"]}


def qual_of(d) -> dict:
    a = d["agg"]
    dpe = sum(r["mode_seq"].count("D") for r in d["rows"]) / len(d["rows"])
    void = a["override_rate"] >= R4["override_void"] or (dpe > 1 and a["died"] > 6)
    hard_ok = a["farm_descend_rate"] <= R4["descend"] and a["cap_rate"] < R4["cap"]
    override_ok = a["override_rate"] < R4["override_sentinel"]
    dual = dpe > 1 and hard_ok and not override_ok
    ok = a["died"] <= DEATHS_MAX and not void and ((hard_ok and override_ok) or dual)
    return {"qual_ok": ok, "void": void, "dual_attr": dual}


def preflight():
    require_calibrated_protocol()
    for p, s, name in ((M29_NPZ, M29_SHA, "M29 npz"), (W_ZIP, W_SHA, "worker zip"),
                       (W_NPZ, W_NPZ_SHA, "worker npz")):
        require_sha256(p, s, name)
    require(zip_steps(W_ZIP) == START_NT, "START assertion failed (worker zip num_timesteps)")
    require(BC_SD.exists(), "BC teacher sd missing (the bc arm's teacher carries this absolute path inside the zip)")
    read_comparable_science()
    read_comparable_launch()
    read_comparable_screen()
    tags = ["v30-GA0W", "v30-golden"] + [f"v30-{a}-leg{k}" for a in ARMS for k in (1, 2)] \
        + [f"v30-{a}-full32" for a in ARMS]
    for t in tags:
        require(not (EVAL / f"{t}.json").exists(), f"target archive already exists: {t} (restart protocol: .void it first)")
    for a in ARMS:
        for k in (1, 2):
            require(not (RUNS / f"v30-{a}-leg{k}").exists(), f"leftover run directory: v30-{a}-leg{k}")
    log({"event": "preflight_ok", "bc_sd_sha": sha16(BC_SD), "start_nt": START_NT})


def main():
    try:
        with exclusive_lock(DRIVER_LOCK, DRIVER_LOCK_PURPOSE):
            _main()
    except (OperationalFailure, OutputReservationError) as e:
        log({"event": "OPERATIONAL_FAILURE", "why": str(e)})
        attention("operational failure:\n" + str(e))
        raise SystemExit(2) from e
    except Exception as e:
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("driver died with an exception:\n" + traceback.format_exc())
        raise


def _main():
    preflight()
    science_doc = read_comparable_science()
    science_base = science_doc["agg"]["ret_mean"]
    floor = round(0.92 * science_base, 1)
    d2death_launch_max = sum(
        r["died"] and r["depth"] >= 2 for r in science_doc["rows"])
    screen_base = read_comparable_screen()["agg"]["ret_mean"]
    trip_line = round(0.85 * screen_base, 2)
    log({"event": "start", "prereg": "docs/prereg/PREREG-v30.md", "leg": LEG, "legs_per_arm": LEGS,
         "trip": trip_line, "screen_base": screen_base,
         "floor": floor, "science_base": science_base, "pre140": PRE140_LINE,
         "launch": [PD112_LINE, WINS112_LINE, DEATHS_MAX, d2death_launch_max]})

    # ---- king anchor sd export + G-KL-W fidelity gate (one of the three identity-chain latches) ----
    require_sha256(W_ZIP, W_SHA, "worker zip")
    if run([PY, "train/export_manager_sd.py", str(W_ZIP), str(KING_SD)],
           "export-king-sd.log", timeout=600) != 0 or not KING_SD.exists():
        why = "king anchor sd export failed"
        log({"event": "STOP", "why": why})
        attention(why)
        raise OperationalFailure(why)
    king_sha = sha256(KING_SD)
    require_sha256(W_NPZ, W_NPZ_SHA, "worker npz")
    if run([PY, "train/check_teacher_parity.py", str(KING_SD), str(W_NPZ)],
           "gklw.log", timeout=600) != 0:
        why = "G-KL-W fidelity gate mismatch: the self-anchor teacher is not faithful to the king"
        log({"event": "STOP", "why": why})
        attention("G-KL-W mismatch")
        raise OperationalFailure(why)
    require(sha256(KING_SD) == king_sha, "G-KL-W identity chain broken (file modified after the gate)")
    log({"event": "g_kl_w", "king_sd_sha": king_sha[:16], "parity": "0/1000"})

    # ---- G-A0W: assembled regression (v28-leg1 npz x M29 npz full 32 == the 140.3 archive) ----
    require_sha256(W_NPZ, W_NPZ_SHA, "worker npz")
    ga0 = exam_retry(str(W_NPZ), "v30-GA0W", "7000-7031")
    if ga0 is None:
        why = "G-A0W exam failed repeatedly"
        log({"event": "STOP", "why": why})
        attention(why)
        raise OperationalFailure(why)
    ref_sci = by_seed(read_comparable_science()["rows"])
    bad = [s for s, r in by_seed(ga0["rows"]).items()
           if (abs(r["ret"] - ref_sci[s]["ret"]) > 0.01 or r["died"] != ref_sci[s]["died"]
               or r["depth"] != ref_sci[s]["depth"]
               or r["mode_seq"] != ref_sci[s]["mode_seq"])]
    log({"event": "g_a0w", "mismatch_seeds": bad, "n_ok": 32 - len(bad)})
    if bad:
        why = "G-A0W bit-level regression mismatch: re-anchor by hand"
        log({"event": "STOP", "why": why})
        attention(f"G-A0W mismatch: {bad}")
        raise OperationalFailure(why)

    # ---- two arms in series: each a 2-leg resume chain ----
    arm_models = {}
    for arm, cfg in ARMS.items():
        require_sha256(W_ZIP, W_SHA, "worker zip")
        nt_chain = START_NT
        prev = str(W_ZIP)
        burned = 0
        spent_steps = 0
        attempts = {}
        halted = None
        k = 1
        while k <= LEGS:
            leg_steps = budgeted_leg_steps(spent_steps)
            if leg_steps < LEG:
                log({"event": "leg_budget_shrunk", "arm": arm, "leg": k,
                     "steps": leg_steps, "spent": spent_steps,
                     "remaining": BUDGET_STEPS - spent_steps,
                     "note": "all completed new steps and failed burned steps are deducted from each arm's hard budget, one by one"})
            if leg_steps == 0:
                log({"event": "budget_exhausted", "arm": arm, "leg": k,
                     "spent": spent_steps})
                break
            seed_k = cfg["base_seed"] + 1_000 * (k - 1)
            probes = reachable_probes(nt_chain, leg_steps)
            if len(probes) < 2:
                log({"event": "calib_trimmed", "arm": arm, "leg": k,
                     "probes": probes, "note": "probes unreachable after burned steps shortened the leg were trimmed"})
            attempts[k] = attempts.get(k, 0) + 1
            run_name = f"v30-{arm}-leg{k}"
            run_dir = RUNS / run_name
            for fn in ("status.json", "calib.jsonl", "sentinel.jsonl"):
                p = run_dir / fn
                if p.exists():
                    p.rename(p.with_suffix(f".pre{attempts[k]}.{time.strftime('%H%M%S')}.void")) \
                        if fn != "status.json" else p.unlink()
            cmd = [PY, "train/train_ppo.py", "--worker", "--algo", "mppo", "--gamma", "1.0",
                   "--max-steps", "3000", "--num-envs", "4", "--n-steps", "512",
                   "--lr", "3e-4", "--ent-coef", "0.005", "--seed", str(seed_k),
                   "--total-steps", str(leg_steps), "--run-name", run_name,
                   "--distill-beta", str(BETA), "--teacher-sd", str(BC_SD), "--skip-dry",
                   "--manager-npz", str(M29_NPZ), "--resume-from", prev,
                   "--allow-manager-change",
                   "--calib-probes", ",".join(str(p) for p in probes),
                   "--calib-record-only"]
            require_sha256(M29_NPZ, M29_SHA, "M29 npz")
            if cfg["override"]:
                require(sha256(KING_SD) == king_sha, "king anchor sd drift (per-leg identity chain)")
                cmd += ["--teacher-override", cfg["override"]]
            log({"event": "leg_start", "arm": arm, "leg": k, "attempt": attempts[k],
                 "seed": seed_k, "steps": leg_steps, "resume_from": prev,
                 "probes": probes, "teacher": cfg["override"] or "BC (carried in the zip)"})
            model_path = run_dir / "model_final.zip"
            old_mtime = model_path.stat().st_mtime_ns if model_path.exists() else None
            t0 = time.time()
            rc = run(cmd, f"{run_name}.try{attempts[k]}.log", timeout=16_200)  # 4.5h
            dt = round(time.time() - t0)
            fresh_model = model_path.exists() and model_path.stat().st_mtime_ns != old_mtime
            nt = zip_steps(model_path) if fresh_model else 0
            try:
                st = json.loads((run_dir / "status.json").read_text())["total_steps"]
            except Exception:
                st = 0
            expected = nt_chain + leg_steps
            clean = rc == 0 and fresh_model and nt == expected
            sampled = observed_attempt_steps(nt_chain, nt, st, leg_steps)
            if not clean:
                partial = sampled
                charged = failed_attempt_charge(partial, leg_steps)
                spent_steps += charged
                burned += charged
                log({"event": "leg_crash", "arm": arm, "leg": k, "attempt": attempts[k],
                     "rc": rc, "nt_zip": nt, "expected": expected,
                     "burned_observed": partial, "burned_charged": charged,
                     "burned_total": burned,
                     "spent_total": spent_steps})
                ensure_retry_budget(spent_steps)
                if attempts[k] >= 4:
                    halted = f"leg {k} crashed 4 times in a row (operational self-protection, not a registered gate)"
                    log({"event": "crash_halt", "arm": arm, "why": halted})
                    attention(f"{arm} {halted}")
                    raise OperationalFailure(f"{arm} {halted}")
                continue
            require(sampled == leg_steps, "observed steps of a clean close disagree with the allocation")
            spent_steps += sampled
            nt_chain = expected
            if dt > 7_200:
                log({"event": "SLOW_MACHINE", "arm": arm, "leg": k, "dt_sec": dt,
                     "note": "slow period, recorded only, not judged (sanity close line 4h)"})
                attention(f"{arm} leg {k} wall clock {dt}s (slow machine, line not hit)")
            # G-oasis (leg 1 only; panel major: ff_dry downgraded to a record)
            if k == 1:
                sent = run_dir / "sentinel.jsonl"
                lines = []
                if sent.exists():
                    for l in sent.read_text().splitlines():
                        if '"sentinel": "v23"' in l:
                            try:
                                lines.append(json.loads(l))
                            except Exception:
                                pass
                if not lines:
                    halted = "G-oasis has no sentinel row"
                    log({"event": "STOP_ARM", "arm": arm, "why": halted})
                    attention(f"{arm}:{halted}")
                    raise OperationalFailure(f"{arm}:{halted}")
                last = lines[-1]
                if last.get("dry", 1) != 0:
                    halted = "G-oasis failed: the learning window contains dry (the true skip_dry invariant was broken)"
                    log({"event": "STOP_ARM", "arm": arm, "why": halted,
                         "dry": last.get("dry")})
                    attention(f"{arm}:{halted}")
                    raise OperationalFailure(f"{arm}:{halted}")
                if last.get("ff_dry", 0) == 0:
                    log({"event": "g_oasis_note", "arm": arm, "ff_dry": 0,
                         "note": "under the M29 distribution, ff_dry=0 at leg start is a legal form (it originates from the v22-H calibration); recorded, not judged"})
                    attention(f"{arm} leg1 ff_dry=0 (recorded; to be judged by manual inspection)")
                log({"event": "g_oasis", "arm": arm, "dry": last.get("dry"),
                     "ff_dry": last.get("ff_dry"), "fresh": last.get("fresh"), "ok": True})
            # G-CAL wiring gate (recorded only, not judged; probes_ok per leg)
            calib_p = run_dir / "calib.jsonl"
            recs = ([json.loads(l) for l in calib_p.read_text().splitlines()]
                    if calib_p.exists() else [])
            probes_ok = (all(any(p <= r["step"] < p + QUANTUM and r["g_ce"] > 0
                                 and r["distill_ce"] > 0 for r in recs)
                             for p in probes) if probes else None)
            probe_status = ("SKIPPED" if probes_ok is None
                            else "PASS" if probes_ok else "FAIL")
            log({"event": "g_cal", "arm": arm, "leg": k, "probes_ok": probes_ok,
                 "probe_status": probe_status,
                 "records": [{kk: r.get(kk) for kk in
                              ("step", "g_ce", "teacher_diverge", "tripped")} for r in recs]})
            if probes_ok is False:
                halted = f"G-CAL wiring failed (leg {k})"
                log({"event": "STOP_ARM", "arm": arm, "why": halted})
                attention(f"{arm}:{halted}")
                raise OperationalFailure(f"{arm}:{halted}")
            # leg exam (M29 assembly, 16 seeds)
            r = exam_retry(run_dir / "model_final.zip", f"v30-{arm}-leg{k}", "7000-7015")
            if r is None:
                halted = f"leg {k} exam failed repeatedly"
                log({"event": "STOP_ARM", "arm": arm, "why": halted})
                attention(f"{arm}:{halted}")
                raise OperationalFailure(f"{arm}:{halted}")
            score = round(r["agg"]["ret_mean"], 1)
            log({"event": "leg_exam", "arm": arm, "leg": k, "score": score,
                 "died": r["agg"]["died"], "diverge": r["agg"].get("script_divergence_rate"),
                 "sha": r["agg"]["_sha"], "model_sha": sha16(run_dir / "model_final.zip"),
                 "nt_chain": nt_chain, "dt_sec": dt})
            prev = str(run_dir / "model_final.zip")
            if candidate_probe_eligible(probes_ok):
                arm_models[arm] = prev        # the last clean-close leg whose wiring PASSed
            else:
                log({"event": "candidate_ineligible", "arm": arm, "leg": k,
                     "why": "G-CAL probes SKIPPED; this model must not form an authorizable arm"})
            if score < HARD_LINE:
                log({"event": "HARD_TRIP", "arm": arm, "leg": k, "score": score})
                attention(f"{arm} hard trip: leg {k} = {score}")
                break
            # If runtime/content change during long training, the old screen v2 fails closed here.
            screen_base = read_comparable_screen()["agg"]["ret_mean"]
            trip_line = round(0.85 * screen_base, 2)
            if score < trip_line:
                log({"event": "trip_halt", "arm": arm, "leg": k, "score": score,
                     "line": trip_line,
                     "note": f"single leg <0.85x{screen_base}, this arm stops training (arms are independent)"})
                attention(f"{arm} trip-line stop: leg {k} = {score}")
                break
            if dt > 14_400:
                log({"event": "sanity_finish", "arm": arm, "leg": k, "dt_sec": dt,
                     "note": "wall clock >4h, sanity close (recalibrated clock)"})
                attention(f"{arm} sanity close: leg {k} {dt}s")
                break
            k += 1
        if arm not in arm_models:
            log({"event": "arm_dead", "arm": arm, "why": halted or "no clean leg"})
            attention(f"{arm} arm has no examinable model")

    missing_arms = missing_authorization_arms(arm_models)
    if missing_arms:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": "an arm with a wiring PASS is missing; authorization forbidden",
             "missing_arms": missing_arms})
        why = f"authorizable arms missing: {missing_arms}"
        attention(why)
        raise OperationalFailure(why)

    # ---- both arms full 32 (M29 assembly) ----
    full, mets = {}, {}
    for arm, mp in arm_models.items():
        d = exam_retry(pathlib.Path(mp), f"v30-{arm}-full32", "7000-7031")
        if d is None:
            why = f"{arm} full 32 failed repeatedly: manual autopsy"
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
        full[arm] = d
        mets[arm] = metrics(d)
        log({"event": "full32", "arm": arm, **mets[arm]})
    if len(full) == 2:
        fk, fb = by_seed(full["king"]["rows"]), by_seed(full["bc"]["rows"])
        r30_6 = sum(fk[s]["ret"] - fb[s]["ret"] for s in fk) / 32
        r30_6w = sum(fk[s]["ret"] > fb[s]["ret"] for s in fk)
        log({"event": "r30_6", "king_minus_bc_mean": round(r30_6, 2), "king_wins": r30_6w,
             "note": "|mean diff|<2 means direction undetermined (pre-registered reading rule)"})

    # ---- eligibility/winner/substitution (v29 D3-2) ----
    quals = {a: qual_of(full[a]) for a in full}
    log({"event": "quals", **{a: quals[a] for a in full}})
    pool = [a for a in full if quals[a]["qual_ok"]]
    if not pool:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": "both arms ineligible: no winner, the relay proposition unanswered (outside statistical power)",
             "arms": {a: mets[a] for a in full}})
        attention("verdict: both arms ineligible")
        return
    prelim = max(full, key=lambda a: full[a]["agg"]["ret_mean"])
    if prelim not in pool:
        log({"event": "substitution", "blocked": prelim, "why": quals[prelim]})
    ms = {a: full[a]["agg"]["ret_mean"] for a in pool}
    band = [a for a in pool if max(ms.values()) - ms[a] <= 0.05]
    if len(band) > 1:
        dmin = min(full[a]["agg"]["died"] for a in band)
        band = [a for a in band if full[a]["agg"]["died"] == dmin]
        winner = "king" if "king" in band else band[0]
    else:
        winner = band[0]
    W, wm = full[winner], mets[winner]
    wrows = by_seed(W["rows"])
    log({"event": "winner", "arm": winner, "mean": wm["mean"], "died": wm["died"],
         "substituted": winner != prelim})

    # ---- dual-anchor pairing ----
    science_doc = read_comparable_science()
    launch_doc = read_comparable_launch()
    ref_sci = by_seed(science_doc["rows"])
    ref_launch = by_seed(launch_doc["rows"])
    science_base = science_doc["agg"]["ret_mean"]
    launch_base = launch_doc["agg"]["ret_mean"]
    floor = round(0.92 * science_base, 1)
    d2death_launch_max = sum(
        r["died"] and r["depth"] >= 2 for r in science_doc["rows"])
    d112 = [wrows[s]["ret"] - ref_launch[s]["ret"] for s in sorted(ref_launch)]
    pd112, wins112 = sum(d112) / 32, sum(x > 0 for x in d112)
    d140 = [wrows[s]["ret"] - ref_sci[s]["ret"] for s in sorted(ref_sci)]
    pd140, wins140 = sum(d140) / 32, sum(x > 0 for x in d140)
    log({"event": "paired", "vs112_mean": round(pd112, 2), "vs112_wins": wins112,
         "vs140_mean": round(pd140, 2), "vs140_wins": wins140})
    log({"event": "draw_ledger", "note": "4th challenger draw on the same-pool 18/32 line (11->16->17->this case);"
         " the ledger records only, does not judge; P(wins>=18|p=.5)~43% note; new-evidence conjunction added to the launch line (prerequisite >=+2 and d2death<=3)"})

    # ---- scientific main verdict (four tiers in order; panel: the valid tier is bound to the exposure guard) ----
    d2, d2d = wm["depth2_seeds"], wm["d2_deaths"]
    if d2 >= 12 and d2d <= 1 and pd140 >= 2.0:
        sci = f"relay valid (exposure>=12 and deep-level casualties<=1 and vs {science_base} >=+2)"
    elif d2d >= 3 and pd140 < 0:
        sci = "relay invalid (the weak spot is not fixable by this prescription: a lead for courses 3/4)"
    elif d2d in (2, 3) and abs(pd140) < 2.0:
        sci = "signal diluted / can't-learn tier (proposition undetermined)"
    elif d2 < 12:
        sci = (f"out of band (exposure collapse: depth2={d2}<12; d2 deaths={d2d},"
               f" vs {science_base}={pd140:+.2f}) recorded, no narrative")
    else:
        sci = f"out of band (d2 deaths={d2d}, vs {science_base}={pd140:+.2f}) recorded, no narrative"
    log({"event": "science_verdict", "verdict": sci, "depth2": d2, "d2_deaths": d2d,
         "pd140": round(pd140, 2), "a13": wm["a13_share"], "dive": wm["dive_per_ep"],
         "note": "the scientific main verdict and launch/throne/Mark-I never rewrite each other (footnote against narrative merging)"})

    # ---- floor (independent gate; the scientific main verdict is already out, both directions count as answers) ----
    if wm["mean"] < floor:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": f"winner {wm['mean']} < floor {floor} (=0.92x{science_base}): retraining did not reproduce"
                        f" the starting level, the relay launch flow ends; scientific main verdict: {sci}"})
        attention(f"verdict: below the floor; scientific main verdict: {sci}")
        return

    # ---- launch (in order: prerequisite -> deep-level casualties -> the four-way conjunction) ----
    launch = (pd140 >= PRE140_LINE and d2d <= d2death_launch_max
              and pd112 >= PD112_LINE and wins112 >= WINS112_LINE)
    if launch:
        golden_cmd = (f"{PY} {ROOT / 'train' / 'eval_assembled.py'} --worker "
                      f"{pathlib.Path(arm_models[winner]).with_suffix('')} --manager-npz {M29_NPZ} "
                      f"--seeds 9000-9031 --tag v30-golden --board")
        dual_note = ("[dual attribution undecided; decide first, then spend: manual launch only after writing back dual_attr_ruling]"
                     if quals[winner]["dual_attr"] else "")
        sci_note = ("" if sci.startswith("relay valid")
                    else f"[scientific main verdict is not 'valid' ({sci}); the verdict must not use relay-success wording,"
                         f" d2 deaths={d2d} vs baseline 3]")
        log({"event": "GOLDEN_AUTHORIZED", "arm": winner, "probe32_mean": wm["mean"],
             "died": wm["died"], "vs112": [round(pd112, 2), wins112],
             "vs140": [round(pd140, 2), wins140], "model": arm_models[winner],
             "model_sha": sha16(pathlib.Path(arm_models[winner])),
             "full32_sha": wm["sha"], "golden_cmd": golden_cmd,
             "p_line": "in order: deaths>6 revert; >=101.2 and deaths<=4 take the throne; (97.2,101.2) and deaths<=4 point estimate;"
                       " >97.2 and deaths 5-6 tie (safety); [93.9,97.2] tie; <93.9 revert",
             "mark1": "= scientific main verdict valid and the v29 three-condition side verdict re-judged as learned on the winner's full 32 and P30 takes the throne",
             "note": dual_note + sci_note + "4th actual opening in the gold pool's history; single arm, once; write back golden_result after the opening"})
        attention(dual_note + f"gold-standard evaluation awaiting manual launch: {winner}; scientific main verdict: {sci}")
        return

    # ---- no launch: exhaustive dispatch (in order) ----
    quad = (f" (vs {launch_base} {pd112:+.2f}/won {wins112},"
            f" vs {science_base} {pd140:+.2f}, d2 deaths {d2d})")
    wins_note = f" (width-shift note: won {wins112}/32 >=14, the tier does not change)" if wins112 >= 14 else ""
    if pd140 < PRE140_LINE:
        verdict = (f"insufficient new evidence tier: vs {science_base} {pd140:+.2f} < +2"
                   f": padding with the existing stock blocked, does not spend the gold run{quad}")
    elif d2d > d2death_launch_max:
        verdict = (f"deep-level casualties not improved, blocked tier: d2 deaths {d2d} > baseline "
                   f"{d2death_launch_max}: does not spend the gold run{quad}")
    elif pd112 >= PD112_LINE and wins112 < WINS112_LINE:
        verdict = f"mean gain but width not reached: point-estimate gain, does not spend the gold run{quad}{wins_note}"
    elif pd112 >= 2.0:
        verdict = f"probe-level improvement, does not spend the gold run{quad}{wins_note}"
    else:
        verdict = f"the incumbent assembled agent stays, the relay has no launch-level gain (power-limited){quad}"
    log({"event": "VERDICT_PATH", "golden_authorized": False, "verdict": verdict,
         "science_verdict": sci, "winner": winner, "winner_mean": wm["mean"]})
    attention(f"verdict (no launch): {verdict}; scientific main verdict: {sci}")


if __name__ == "__main__":
    main()
