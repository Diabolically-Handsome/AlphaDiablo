"""v31 "new-world re-education" driver (sole executor of the clauses of docs/prereg/PREREG-v31-new-world-reeducation.md;
a targeted rework of run_v29_relection.py).

Clone difference table (maps clause by clause to PREREG-v31; includes the panel's two BLOCK corrections):
- Era: CALIBRATED_PROTOCOL_VERSION 2->3; the four R2 anchors with explicit paths + full-text sha +
  full comparability re-verification against the current runtime (freeze->expected->read->verify, the full v30 precedent)
- train_ppo.py addition: a manager-side options resume port (freeze prerequisite RESUME_SMOKE +
  G-R31 on the ledger); cont arm --total-steps 160000 (incremental semantics), nt gate 320000
- Instrument definition: every evaluation in the case uses the npz worker; the G-A0 bit-level regression is withdrawn -> D3-0 reference sanity gate
  (died<=8 and R in [100,170]) + CASE_RUNTIME case-level runtime reconciliation (a prerequisite of every exam)
- Reference terminal clause: once validly archived, refs are frozen at case level; resumed runs accept them and never re-burn
- Verdict lines derived on site: ABANDON=(75/112.4)xR, FLOOR=(85/92)xR (the fractions are canonical)
- arm_done carries a steps_status diagnostic field (nt_zip stays the single step-count source)
The gold-standard evaluation is not launched here (started manually, single arm, once). Ledger: train/runs/v31/gate_ledger.jsonl.
Usage: .venv/bin/python train/run_v31_neweducation.py
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

import numpy as np

from eval_contract import (PROTOCOL_VERSION, OperationalFailure, OutputReservationError,
                           exclusive_lock, expected_eval_identity,
                           freeze_eval_identity, read_eval_archive,
                           verify_eval_identity)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
V31 = RUNS / "v31"
V31.mkdir(parents=True, exist_ok=True)
LEDGER = V31 / "gate_ledger.jsonl"
EVAL = RUNS / "eval-assembled"

W_NPZ = ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz"
W_NPZ_SHA = "976b6c05edaa0a32bb30bd372782e1201c72b029cedcbb3a5bf2361d34f27f8a"
W24_NPZ = ROOT / "train" / "models" / "v24-worker-leg7" / "policy.npz"
W24_NPZ_SHA = "a31fa7c6b18b5c3593f4e1753d97aac9386689aa6ad8b158c526b673c57fbc2a"
M29_NPZ = ROOT / "train" / "models" / "v29-manager-mfresh" / "policy.npz"
M29_NPZ_SHA = "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6"
M29_ZIP = RUNS / "v29-mfresh" / "model_final.zip"
M29_ZIP_SHA = "9d5820bfb951f6ba122b98ebad707f75a5471b4ce35dd09c3e8105775a3097ee"

# R2 ANCHOR_GRANT consumption (explicit paths + full-text sha + full runtime comparability re-verification)
R2_ANCHORS = {
    # 2026-09-23: redacted with neutral local paths (content otherwise unchanged); was 8ab6b51065105a67
    "r2-launch": ("bf9a543759bde261eed92e0beccc3b9422c21c10876c07e7da81ec441ec5a47b",
                  "W", None),
    # 2026-09-23: redacted with neutral local paths (content otherwise unchanged); was 8d6a05d3517a6481
    "r2-science": ("37fab311e0a07a1ea22b35fd13d23448291b1d461d2f3b81f2dbe0e9d9391511",
                   "W", "M29"),
    # 2026-09-23: redacted with neutral local paths (content otherwise unchanged); was 2324648a416cbc0c
    "r2-throne": ("1390a72db6a527a90238d37b2935adc0872c13bc0cc609e356bd72a86752fc41",
                  "W24", None),
    # 2026-09-23: redacted with neutral local paths (content otherwise unchanged); was 71c298e05b6bf19e
    "r2-script": ("ede6c0507e4010649f426de43748b27c2117faff373341f4b39b827b5e4645f5",
                  "script", None),
}

ABANDON_FRAC = (75.0, 112.4)     # canonical = the fraction; 0.6673 is a display rounding
FLOOR_FRAC = (85.0, 92.0)
PAIRED_DIFF = 4.0
PAIRED_WINS = 18
DEATHS_MAX = 6
REF_DIED_MAX = 8                 # D3-0 reference sanity gate
R_CORRIDOR = (100.0, 170.0)      # D3-0 (endpoints registered by the panel)
R4 = {"descend": 0.0204, "override_sentinel": 0.03, "override_void": 0.08, "cap": 0.05}
CALIBRATED_PROTOCOL_VERSION = 3

ARMS = {
    # cli_steps is --total-steps (incremental semantics for resumed legs); nt_target is the target gate
    "v31-mfresh": {
        "cli_steps": 160_000, "nt_target": 160_000,
        "extra": ["--ent-coef", "0.02", "--lr", "3e-4", "--seed", "22"]},
    "v31-mcont": {
        "cli_steps": 160_000, "nt_target": 320_000,
        "extra": ["--ent-coef", "0.02", "--lr", "3e-4", "--seed", "26",
                  "--resume-from", str(M29_ZIP), "--allow-legacy-resume"]},
}

CASE_RT: dict | None = None      # CASE_RUNTIME case-level runtime five shas (settled in preflight)


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    with open(V31 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def sha16(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def sha256(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    """Single pre-check channel (panel minor): every failure is an OperationalFailure -> exit code 2."""
    if not condition:
        raise OperationalFailure(message)


def require_calibrated_protocol() -> None:
    if PROTOCOL_VERSION != CALIBRATED_PROTOCOL_VERSION:
        raise OperationalFailure(
            "v31 verdict lines are calibrated in the protocol-v3 world (R2 anchors); another protocol version bump requires re-anchoring")


def runtime_five(snapshot) -> dict:
    rt = snapshot["runtime"]
    return {"bridge": rt["bridge"]["sha256"], "engine": rt["engine"]["sha256"],
            "game_data": rt["content"]["game_data"]["sha256"],
            "assets": rt["content"]["assets"]["sha256"],
            "protocol": rt["python_protocol"]["sha256"]}


def assert_case_runtime(snapshot, where: str):
    """Case-level runtime reconciliation (the in-case version of R2 W10): the prerequisite for references and arm exams to be comparable over time."""
    require(CASE_RT is not None, "CASE_RUNTIME not settled")
    current = runtime_five(snapshot)
    if current != CASE_RT:
        log({"event": "CASE_HALT_RUNTIME_DRIFT", "where": where,
             "case": CASE_RT, "current": current})
        attention(f"case-level runtime drift ({where}); references and arm exams are not comparable; stopping")
        raise OperationalFailure(f"case-level runtime drift ({where})")


def anchor_spec(code):
    if code == "W":
        return str(W_NPZ)
    if code == "W24":
        return str(W24_NPZ)
    if code == "M29":
        return str(M29_NPZ)
    return code            # "script"


def read_r2_anchor(tag: str) -> dict:
    """Full R2 anchor consumption: full-text sha + freeze->expected->read->verify (v30 precedent)."""
    path = EVAL / f"{tag}.json"
    expected_sha, wcode, mcode = R2_ANCHORS[tag]
    require(path.is_file() and sha256(path) == expected_sha,
            f"R2 anchor sha drift/missing: {tag}")
    manager = anchor_spec(mcode) if mcode else None
    try:
        snapshot = freeze_eval_identity(ROOT, anchor_spec(wcode), manager)
        expected = expected_eval_identity(snapshot, tag=tag,
                                          seeds=range(9000, 9032))
        document = read_eval_archive(path, **expected)
        verify_eval_identity(snapshot, ROOT)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise OperationalFailure(
            f"R2 anchor {tag} is not comparable with the current runtime; a separate re-anchoring case is required (scientific reading is final,"
            f" re-burning forbidden): {exc}") from exc
    return document


def run(cmd, logfile, timeout) -> int:
    with open(V31 / logfile, "w") as lf:
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
    """Real SB3 chain reading (the single step-count source; the throttled status count always lags)."""
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
    assert_case_runtime(snapshot, f"exam:{tag}")      # case-level reconciliation is a prerequisite; no crash retake for it
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


def validate_ref_archive(tag, worker, manager_npz):
    """Re-verification of a reference archive accepted on resume (reference terminal clause)."""
    out = EVAL / f"{tag}.json"
    snapshot = freeze_eval_identity(ROOT, worker, manager_npz)
    assert_case_runtime(snapshot, f"ref-adopt:{tag}")
    expected = expected_eval_identity(snapshot, tag=tag,
                                      seeds=range(7000, 7032))
    d = read_eval_archive(out, **expected)
    verify_eval_identity(snapshot, ROOT)
    d["agg"]["_sha"] = sha16(out)
    return d


def ref_exam_or_adopt(events, tag, worker, manager_npz=None):
    """Reference run: once validly archived it is frozen at case level; resumed runs accept it; .void re-burning forbidden."""
    out = EVAL / f"{tag}.json"
    on_ledger = [e for e in events
                 if e.get("event") == "ref_archive" and e.get("tag") == tag]
    if out.exists():
        require(bool(on_ledger),
                f"{tag} leftover pre-case archive present but the ledger has no ref_archive record; stopping")
        try:
            d = validate_ref_archive(tag, worker, manager_npz)
        except OperationalFailure:
            raise
        except Exception as exc:
            log({"event": "REF_INVALID", "tag": tag,
                 "why": (str(exc).splitlines() or ["?"])[0]})
            attention(f"{tag} reference archive failed re-verification (REF_INVALID); the whole case stops")
            raise OperationalFailure(f"{tag} reference archive failed re-verification") from exc
        require(d["agg"]["_sha"] == on_ledger[-1]["sha16"],
                f"{tag} archive does not match the ledger ref_archive sha")
        log({"event": "ref_adopted", "tag": tag, "sha16": d["agg"]["_sha"]})
        return d
    require(not on_ledger, f"{tag} on the ledger but the archive is missing (REF_INVALID); stopping")
    d = exam_retry(worker, tag, "7000-7031", manager_npz)
    if d is None:
        raise OperationalFailure(f"{tag} reference exam failed repeatedly (leftover archive sealed as .void;"
                                 " substituting the gold-pool r2-launch across pools as the baseline is forbidden)")
    log({"event": "ref_archive", "tag": tag, "sha16": d["agg"]["_sha"]})
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


def depth_gauge(d) -> dict:
    a = d["agg"]
    return {"died": a["died"], "depth2_seeds": depth2_count(d["rows"]),
            "dive_per_ep": round(dive_per_ep(d["rows"]), 2),
            "bonus_per_ep": round(bonus_per_ep(d["rows"]), 2),
            "override": a["override_rate"], "descend": a["farm_descend_rate"],
            "cap": a["cap_rate"]}


def preflight(events):
    global CASE_RT
    require_calibrated_protocol()
    require(W_NPZ.exists() and sha256(W_NPZ) == W_NPZ_SHA,
            "worker npz missing or sha drift")
    require(W24_NPZ.exists() and sha256(W24_NPZ) == W24_NPZ_SHA,
            "v24-leg7 npz (throne anchor identity item) missing or sha drift")
    require(M29_NPZ.exists() and sha256(M29_NPZ) == M29_NPZ_SHA,
            "M29 manager npz (archived item) missing or sha drift")
    require(M29_ZIP.exists() and sha256(M29_ZIP) == M29_ZIP_SHA,
            "M29 checkpoint (cont arm initialization) missing or sha drift")
    require(zip_steps(M29_ZIP) == 160_000, "M29 checkpoint step count is malformed")
    anchors = {tag: read_r2_anchor(tag) for tag in R2_ANCHORS}   # full re-verification
    # G-R31: zero-training load-and-export == archived npz (loading yields the original; re-proven at every launch)
    g31 = V31 / "g_r31.npz"
    if g31.exists():
        g31.unlink()
    require(run([PY, "train/export_manager_npz.py", str(M29_ZIP), str(g31)],
                f"g-r31.{time.time_ns()}.log", timeout=600) == 0 and g31.exists(),
            "G-R31 export failed")
    a, b = np.load(g31), np.load(M29_NPZ)
    require(set(a.files) == set(b.files)
            and all(np.array_equal(a[k], b[k]) for k in a.files),
            "G-R31 bit-level mismatch: M29 zip load-and-export != archived npz")
    log({"event": "G_R31", "bitwise": "6/6", "source_zip_sha16": sha16(M29_ZIP)})
    # Assert the freeze prerequisites are on record (panel BLOCK: smoke test fails -> this case is not frozen)
    require(any(e.get("event") == "RESUME_SMOKE" and e.get("rc") == 0
                for e in events), "RESUME_SMOKE event missing (panel freeze prerequisite)")
    for t in ["v31-golden"] + [f"{a_}-{s}" for a_ in ARMS for s in ("s16", "full32")]:
        require(not (EVAL / f"{t}.json").exists(),
                f"target archive already exists: {t} (restart protocol: .void it first)")
    for a_ in ARMS:
        require(not (RUNS / a_).exists(), f"leftover run directory: {a_} (restart protocol: archive it first)")
    # CASE_RUNTIME case-level runtime settle/reconcile
    snapshot = freeze_eval_identity(ROOT, str(W_NPZ), None)
    five = runtime_five(snapshot)
    prior_rt = [e for e in events if e.get("event") == "CASE_RUNTIME"]
    if prior_rt:
        require({k: prior_rt[0][k] for k in five} == five,
                "CASE_RUNTIME resumed-run reconciliation mismatch; stopping")
    else:
        log({"event": "CASE_RUNTIME", **five})
    CASE_RT = five
    # Incumbent identity pinned: freeze_eval_identity (default manager) has a built-in DEFAULT_MANAGER_SHA256
    # assertion, and the full r2-launch re-verification also uses the default manager, so reference-run manager == incumbent manager holds.
    log({"event": "preflight_ok", "worker_npz_sha": sha16(W_NPZ),
         "m29_npz_sha": sha16(M29_NPZ), "m29_zip_sha": sha16(M29_ZIP),
         "default_manager_sha16": snapshot["manager"]["sha256"][:16],
         "r2_anchor_sha16": {t: R2_ANCHORS[t][0][:16] for t in R2_ANCHORS}})
    return anchors


def main():
    try:
        with exclusive_lock(V31 / ".driver.lock", "v31 driver"):
            _main()
    except (OperationalFailure, OutputReservationError) as e:
        log({"event": "OPERATIONAL_FAILURE", "why": str(e)})
        attention("operational failure:\n" + str(e))
        raise SystemExit(2) from e
    except Exception as e:   # catch-all clause: every unexpected exception must be recorded; no silent death
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("driver died with an exception:\n" + traceback.format_exc())
        raise


def _read_ledger() -> list[dict]:
    if not LEDGER.is_file():
        return []
    events = []
    for i, line in enumerate(LEDGER.read_text().splitlines(), 1):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise OperationalFailure(f"ledger line {i} cannot be parsed; stopping: {exc}") from exc
    return events


def _main():
    events = _read_ledger()
    anchors = preflight(events)
    launch_gold = anchors["r2-launch"]["agg"]["ret_mean"]
    throne = anchors["r2-throne"]["agg"]["ret_mean"]
    script_ref = anchors["r2-script"]["agg"]["ret_mean"]

    # ---- the two reference runs (terminal clause: archived = frozen; resumed runs accept them) ----
    ref_l = ref_exam_or_adopt(events, "v31-ref-launch", str(W_NPZ))
    ref_s = ref_exam_or_adopt(events, "v31-ref-science", str(W_NPZ),
                              manager_npz=str(M29_NPZ))
    R = ref_l["agg"]["ret_mean"]
    # D3-0 reference sanity gate (the era-switch equivalent of the G-A0 instrument gate; an operational guard, not a scientific verdict)
    ref_gauge = depth_gauge(ref_l)
    log({"event": "ref_sanity", "R": R, "corridor": R_CORRIDOR,
         "died": ref_l["agg"]["died"], "died_max": REF_DIED_MAX,
         "ref_launch_gauge": ref_gauge,
         "ref_science_gauge": depth_gauge(ref_s),
         "cross_pool_note": {"r2_launch_golden": launch_gold,
                             "definition": "probe pool vs gold pool, recorded only, not judged; 133.9 is only a circuit-breaker input here"}})
    if not (R_CORRIDOR[0] <= R <= R_CORRIDOR[1]) or ref_l["agg"]["died"] > REF_DIED_MAX:
        log({"event": "CASE_HALT_REF_CORRIDOR", "R": R,
             "died": ref_l["agg"]["died"]})
        attention("probe-pool reference collapsed/instrument anomaly: the on-site verdict-line derivation loses its meaning; stopping,"
                  " no arm training, no gold seeds spent; a re-measurement needs a separate case and a new freeze")
        raise OperationalFailure("D3-0 reference sanity gate not passed")
    abandon = round(R * ABANDON_FRAC[0] / ABANDON_FRAC[1], 1)
    floor_repro = round(R * FLOOR_FRAC[0] / FLOOR_FRAC[1], 1)
    log({"event": "refs", "ref_launch_mean": R,
         "ref_science_mean": ref_s["agg"]["ret_mean"],
         "abandon_line": f"{abandon} =(75/112.4)×{R}",
         "floor_repro": f"{floor_repro} =(85/92)×{R}",
         "ref_launch_sha": ref_l["agg"]["_sha"],
         "ref_science_sha": ref_s["agg"]["_sha"]})
    ref_rows = by_seed(ref_l["rows"])
    refsci_rows = by_seed(ref_s["rows"])

    # ---- train the two arms in series ----
    npz = {}
    for name, spec in ARMS.items():
        cmd = [PY, "train/train_ppo.py", "--options", "--algo", "mppo", "--gamma", "1.0",
               "--max-steps", "3000", "--n-steps", "64", "--num-envs", "4",
               "--total-steps", str(spec["cli_steps"]), "--worker-npz", str(W_NPZ),
               "--run-name", name] + spec["extra"]
        log({"event": "arm_start", "arm": name, "cli_steps": spec["cli_steps"],
             "nt_target": spec["nt_target"], "cmd_extra": spec["extra"]})
        t0 = time.time()
        rc = run(cmd, f"train-{name}.log", timeout=14_400)   # 4h hang guard
        sp = RUNS / name / "status.json"
        try:
            steps = json.loads(sp.read_text())["total_steps"] if sp.exists() else 0
        except Exception:
            steps = 0
        nt = zip_steps(RUNS / name / "model_final.zip")   # the single step-count source for the target gate
        log({"event": "arm_done", "arm": name, "rc": rc, "nt_zip": nt,
             "steps_status": steps, "dt_min": round((time.time() - t0) / 60, 1)})
        if rc != 0 or nt != spec["nt_target"]:
            why = (f"{name} training did not reach its target (rc={rc}, nt_zip={nt}, "
                   f"nt_target={spec['nt_target']}): proposition not examined;"
                   " this version adds no retraining (v25 clause)")
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
        log({"event": "npz_exported", "arm": name, "npz_sha": sha16(out)})

    # ---- early abandonment gate (16 seeds) ----
    s16 = {}
    for name in ARMS:
        d = exam_retry(str(W_NPZ), f"{name}-s16", "7000-7015", manager_npz=npz[name])
        if d is None:
            why = f"{name} screen exam failed repeatedly"
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
        s16[name] = d["agg"]["ret_mean"]
        log({"event": "screen16", "arm": name, "score": d["agg"]["ret_mean"],
             "died": d["agg"]["died"]})
    if all(v < abandon for v in s16.values()):
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"both arms' screens < {abandon} (=(75/112.4)x{R}): training failed,"
                    " the succession proposition was not examined (full 32 skipped)",
             "s16": s16, "R": R,
             "science_note": "on the abandonment path the D3-8 continued-training net-change observation is honestly absent"})
        attention("verdict: training failed, proposition not examined")
        return

    # ---- both arms full 32 (with depth instrumentation) ----
    full = {}
    for name in ARMS:
        d = exam_retry(str(W_NPZ), f"{name}-full32", "7000-7031", manager_npz=npz[name])
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
    fr = by_seed(full["v31-mfresh"]["rows"])
    co = by_seed(full["v31-mcont"]["rows"])
    r31_2 = sum(co[s]["ret"] - fr[s]["ret"] for s in fr) / 32
    means = {n: full[n]["agg"]["ret_mean"] for n in ARMS}
    log({"event": "r31_2", "paired_cont_minus_fresh_mean": round(r31_2, 2),
         "arms_full32": means, "R": R,
         "definition": "v3 from scratch vs continued training from the v2 stock: a whole-path comparison, single-factor attribution forbidden;"
                " a single paired descriptive statistic, not to be read as significance; the step ledger must state both definitions:"
                " cont total steps 2x (320k cumulative) and equal new-world steps (160k each)"})

    # ---- scientific observation (D3-8): always compute cont vs ref-science (net change of continued training relative to its predecessor) ----
    cont_sci = [co[s]["ret"] - refsci_rows[s]["ret"] for s in sorted(refsci_rows)]
    log({"event": "science_observation",
         "cont_vs_m29ref_paired_mean": round(sum(cont_sci) / 32, 2),
         "wins": sum(x > 0 for x in cont_sci),
         "definition": "recorded only, not judged; net change of continued training relative to its predecessor (M29 v3 probe reading)"})

    # ---- per-arm eligibility ----
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
    log({"event": "quals", **{n: quals[n] for n in ARMS},
         "line_note": "the death gate/void line is systematically too tight for deep-diving behavior in v3 (lines not recalibrated,"
                      " a known bias); any verdict decided by them must also cite the depth distribution and per-seed died"})
    pool = [n for n in ARMS if quals[n]["qual_ok"]]
    if not pool:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": "both arms ineligible (deaths/sentinel/void): no winner, the succession proposition unanswered (outside statistical power)",
             "arms": {n: {"mean": means[n], "died": full[n]["agg"]["died"],
                          **quals[n]} for n in ARMS},
             "r31_2": round(r31_2, 2), "R": R})
        attention("verdict: both arms ineligible, no winner (depth instrumentation recorded with the full32 events)")
        return
    prelim = max(ARMS, key=lambda n: means[n])
    if prelim not in pool:
        log({"event": "substitution", "blocked": prelim, "why": quals[prelim],
             "note": "the mean winner was blocked by eligibility; per D3-2 the eligible arm takes its place"})
    ms = {n: means[n] for n in pool}
    band = [n for n in pool if max(ms.values()) - ms[n] <= 0.05]
    if len(band) > 1:
        dmin = min(full[n]["agg"]["died"] for n in band)
        band = [n for n in band if full[n]["agg"]["died"] == dmin]
        winner = "v31-mfresh" if "v31-mfresh" in band else band[0]
    else:
        winner = band[0]
    W = full[winner]
    wa = W["agg"]
    wrows = by_seed(W["rows"])
    dual_attr = quals[winner]["dual_attr"]
    log({"event": "winner", "arm": winner, "mean": wa["ret_mean"], "died": wa["died"],
         "substituted": winner != prelim,
         "residual_note": "the winner is a max-of-2 order statistic; launch type-I error ~x2 (residual #1)"})

    # ---- depth side verdict (v29 D3-7 three tiers + v31 on-site baseline clause) ----
    d2, dpe = depth2_count(W["rows"]), dive_per_ep(W["rows"])
    field_d2 = ref_gauge["depth2_seeds"]
    if d2 >= 12 and 0.5 <= dpe <= 3 and wa["died"] <= DEATHS_MAX:
        if field_d2 >= 12:
            depth_verdict = (f"reaches the line inherited from v2 (>=12), but the on-site baseline is already above the line"
                             f" (ref-launch depth2={field_d2}): no discriminating power")
        else:
            depth_verdict = "depth economy learned (v3 calibration note: discrimination weakened, no added narrative)"
    elif d2 <= 7:
        depth_verdict = f"depth not unlocked (<= baseline 7; on-site baseline ref depth2={field_d2})"
    else:
        depth_verdict = (f"out of band (depth2={d2}, dive={dpe:.2f}, died={wa['died']};"
                         f" on-site baseline ref depth2={field_d2}): recorded, no narrative")
    log({"event": "depth_verdict", "depth2_seeds": d2, "dive_per_ep": round(dpe, 2),
         "bonus_per_ep": round(bonus_per_ep(W["rows"]), 2),
         "field_baseline_depth2": field_d2, "verdict": depth_verdict,
         "note": "side verdict; the throne and Mark-I determinations follow D3-6 and the roadmap clause (docs/design/ROADMAP-course-plan.md) (guarding against over-narration);"
                 " the three-tier boundaries are numbers inherited from v2"})

    # ---- reproduction floor ----
    if wa["ret_mean"] < floor_repro:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"winner {wa['ret_mean']} < {floor_repro} (=(85/92)x{R}):"
                    f" retraining did not reproduce the reference level, proposition not examined; depth side verdict: {depth_verdict}",
             "arms_full32": means, "r31_2": round(r31_2, 2), "R": R})
        attention("verdict: reference level not reproduced")
        return

    # ---- launch criterion (paired vs v31-ref-launch, keyed by seed) ----
    diffs = [wrows[s]["ret"] - ref_rows[s]["ret"] for s in sorted(ref_rows)]
    pd_mean = sum(diffs) / 32
    pd_wins = sum(x > 0 for x in diffs)
    launch = pd_mean >= PAIRED_DIFF and pd_wins >= PAIRED_WINS
    log({"event": "launch_check", "paired_mean": round(pd_mean, 2), "paired_wins": pd_wins,
         "died": wa["died"], "dive_per_ep": round(dpe, 2),
         "dual_attribution": dual_attr, "tau_note": wa["farm_tau_mean"],
         "multiple_comparison_note": "5th challenger draw on the same-pool 18/32 line"
                                     " (11->16->17->v30 not reached->this case);"
                                     " the P(wins>=18|p=.5)~43% note goes with the verdict;"
                                     " this case's baseline is the on-site derived R, not a historical archive, noted as well",
         "line_note": "+4/18/deaths 6 are lines inherited from v2; v3 error rates not recalibrated; a verdict near a line must carry the note"})

    p_hi = round(throne + 4.0, 1)
    P_LINE = (f"P line quick reference (decided in order; the throne's incumbent anchor value {throne} / script reference {script_ref},"
              f" R2 gold-pool archives read on site, re-measuring does not change the title): deaths>6 -> revert;"
              f" gold>={p_hi} and deaths<=4 -> P31 takes the throne; in ({throne},{p_hi}) and deaths<=4 -> point-estimate gain, throne unchanged;"
              f" >{throne} and deaths 5-6 -> tie (safety); in [{script_ref},{throne}] -> tie;"
              f" <{script_ref} -> revert. Gold-pool opening history: the gold standard was actually opened three times (v22/v23/v24),"
              " the five R2 anchoring runs were measurement exposures; if this case launches, it is the 4th actual gold opening and the 9th cumulative gold-pool exposure,"
              " the fixed-pool bias goes with the verdict. Title succession follows the D3-6 clause: the launch line moves the incumbent assembled agent,"
              " the P line moves the throne, and the two titles are declared on separate lines")
    if launch:
        golden_cmd = (f"{PY} {ROOT / 'train' / 'eval_assembled.py'} --worker {W_NPZ} "
                      f"--manager-npz {npz[winner]} --seeds 9000-9031 "
                      f"--tag v31-golden --board")
        dual_note = ("[dual attribution undecided] override crossed its line and was let through on the dual-attribution path; before spending the gold run a human must decide"
                     " mix drift vs real degradation and write back a dual_attr_ruling event; decide first, then spend; "
                     if dual_attr else "")
        log({"event": "GOLDEN_AUTHORIZED", "arm": winner, "probe32_mean": wa["ret_mean"],
             "died": wa["died"], "wins": pd_wins, "mean_diff": round(pd_mean, 2),
             "arms_full32": means, "r31_2": round(r31_2, 2), "R": R,
             "manager_npz": npz[winner], "manager_npz_sha": sha16(npz[winner]),
             "full32_sha": wa["_sha"], "golden_cmd": golden_cmd, "p_line": P_LINE,
             "authorization_source": "precedent of the campaign-level gold-evaluation discipline (v24-v30; the discretion is recorded in the R2/v31 case rationale)",
             "note": dual_note + "the gold-standard evaluation is started manually, single arm, once; losing/non-launched arms never see"
                     " the 9000 range; write back a golden_result event after the opening"})
        attention(dual_note + f"gold-standard evaluation awaiting manual launch: {winner} (command and P line quick reference in the ledger);"
                  f" depth side verdict: {depth_verdict}")
        return

    # ---- no launch: exhaustive dispatch ----
    wins_note = (f" (width-shift note: won {pd_wins}/32 >=14, the tier does not change)"
                 if pd_wins >= 14 else "")
    cont_note = ("cont loss qualified: hyperparameters were not adapted for continued training (no lr schedule, optimizer state inherited,"
                 " entropy follows the fresh recipe); the collapse is a path-level result and must not be read as old knowledge necessarily being a burden"
                 " (v25 bare fine-tuning precedent); " if winner != "v31-mcont" else "")
    if pd_mean >= PAIRED_DIFF and pd_wins < PAIRED_WINS:
        verdict = (f"mean gain +{pd_mean:.2f} but width not reached (won {pd_wins}/32 < 18)"
                   ": a point-estimate gain, does not spend the gold run")
    elif pd_mean >= 2.0:
        verdict = (f"paired mean diff {pd_mean:.2f} in [+2,+4): a probe-level improvement, does not spend the gold run"
                   f" (won {pd_wins}/32){wins_note}")
    else:
        verdict = (f"paired mean diff {pd_mean:.2f} <+2: the incumbent stays, new-world re-education brings no gain"
                   f" (power-limited) (won {pd_wins}/32){wins_note}")
    log({"event": "VERDICT_PATH", "golden_authorized": False,
         "verdict": verdict, "depth_verdict": depth_verdict,
         "winner": winner, "winner_mean": wa["ret_mean"],
         "paired_mean": round(pd_mean, 2), "paired_wins": pd_wins,
         "arms_full32": means, "r31_2": round(r31_2, 2), "R": R,
         "cont_note": cont_note})
    attention(f"verdict (no launch): {verdict}; depth side verdict: {depth_verdict}")


if __name__ == "__main__":
    main()
