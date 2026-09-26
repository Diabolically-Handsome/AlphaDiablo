"""v32 "potion autonomy" driver (sole executor of the clauses of docs/prereg/PREREG-v32-potion-autonomy.md;
a targeted merge of the run_v31_neweducation.py skeleton x the run_v30_relay.py leg recipe).

Clone difference table (PREREG-v32, clause by clause):
- Pre-check sequence: the two G0 forms (identity replay + ghost) -> bc_worker regeneration (R1 gate) ->
  KING_SD regeneration + parity -> two in-case refs + per-seed bit-level reconciliation against the v31 references
  (REF_BITEQ, replacing the R-corridor sanity gate) -> two legs
- Two legs, same-seed control (single variable = --no-drink-sovereignty), v30 leg recipe verbatim
  (lr 3e-4 / ent .005 / under M29 / beta=0.015625 / skip-dry / self-anchor override),
  both resume the king zip directly (the second registered use of the legacy port); nt gate 3,997,696
- R2 anchor consumption: full-text sha + anchor-bridge clause only (the protocol has changed, so a full re-verification is bound to mismatch;
  throne/script numbers are matched via the bridge proofs, read as strict json)
- R32 main verdict (sov x M29 vs ref-science, three tiers) + R32 split (sov-ctrl) +
  a12 instrumentation (worker_action_hist["12"]/episode)
The gold-standard evaluation is not launched here (it is started manually). Ledger: train/runs/v32/gate_ledger.jsonl.
Usage: .venv/bin/python train/run_v32_sovereign.py
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
                           strict_json_loads, verify_eval_identity)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
V32 = RUNS / "v32"
V32.mkdir(parents=True, exist_ok=True)
LEDGER = V32 / "gate_ledger.jsonl"
EVAL = RUNS / "eval-assembled"

KING_ZIP = ROOT / "train" / "models" / "v28-worker-leg1" / "model_final.zip"
# 2026-09-23: re-saved with neutral local paths (content otherwise unchanged); was 2f7bc9dd810956c3
KING_ZIP_SHA = "0c6f014da19c3bf27b208d55adc76be13fcad8bc5f744b95f4f743be97426434"
KING_NPZ = ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz"
KING_NPZ_SHA = "976b6c05edaa0a32bb30bd372782e1201c72b029cedcbb3a5bf2361d34f27f8a"
H_NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
M29_NPZ = ROOT / "train" / "models" / "v29-manager-mfresh" / "policy.npz"
M29_NPZ_SHA = "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6"
BC_SD = RUNS / "bc-worker" / "policy_sd.pt"
KING_SD = V32 / "king_anchor_sd.pt"
TRAJ_BASELINE = ROOT / "docs" / "assets" / "g0v32_traj_baseline.json"
TRAJ_BASELINE_SHA = "2f567d7abfc5b4714cac4ff3236b09ab67e22485d2b05b52d9dc956692a26d6f"
THRONE_BASELINE = ROOT / "docs" / "assets" / "g0v32_traj_baseline_throne.json"
THRONE_BASELINE_SHA = "e235e96972f8f2eeaaebec2bc8970c3e3f77b715f26c364250c42d8d37131cfc"
ZEROFLIP_REPORT = ROOT / "train" / "runs" / "probe-zeroflip" / "report.json"
ZEROFLIP_SHA = "e1a8266a70d63ffbe602e5e806209b15b6ad45688390fcfbbe0b435d7038a077"  # English keys/notes; the pre-translation digest was b2a2062304b08162

# v31 reference archives (REF_BITEQ bit-level reconciliation targets; full-text sha pinned)
V31_REF = {
    "launch": (EVAL / "v31-ref-launch.json",
               "df17db995661c3994215c614ac4beeb6e4670bd1527d51a1198d69d960e01541"),
    "science": (EVAL / "v31-ref-science.json",
                "d842a8fa75c3b9234f4197a3f9fdc41458e8efeacf1d878934c349cf533c2489"),
}
# R2 anchors (consumed by the bridge clause: strict json + full-text sha; the protocol has changed, so full re-verification is forbidden)
R2_BRIDGE = {
    "throne": (EVAL / "r2-throne.json",
               # 2026-09-23: redacted with neutral local paths (content otherwise unchanged); was 2324648a416cbc0c
               "1390a72db6a527a90238d37b2935adc0872c13bc0cc609e356bd72a86752fc41"),
    "script": (EVAL / "r2-script.json",
               # 2026-09-23: redacted with neutral local paths (content otherwise unchanged); was 71c298e05b6bf19e
               "ede6c0507e4010649f426de43748b27c2117faff373341f4b39b827b5e4645f5"),
}

LEG_STEPS = 499_712
NT_TARGET = 3_997_696          # = king 3,497,984 + 499,712 (incremental semantics)
BETA = 0.015625
SEED = 303_000
ABANDON_FRAC = (75.0, 112.4)
FLOOR_FRAC = (85.0, 92.0)
PAIRED_DIFF = 4.0
PAIRED_WINS = 18
DEATHS_MAX = 6
SURV_DIED_LINE = 3             # R32 main verdict: ref-science died baseline (measured in v31)
SURV_MEAN_BAND = -2.0          # mean-not-lower definition: paired mean difference >= -2
A12_USE_LINE = 0.1             # "autonomy given but not used" tier: a12/episode < 0.1
R4 = {"descend": 0.0204, "override_sentinel": 0.03, "override_void": 0.08, "cap": 0.05}
CALIBRATED_PROTOCOL_VERSION = 3

LEGS = {"v32-sov": [], "v32-ctrl": ["--no-drink-sovereignty"]}
CASE_RT: dict | None = None


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    with open(V32 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def sha16(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def sha256(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OperationalFailure(message)


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
        attention(f"case-level runtime drift ({where}); stopping")
        raise OperationalFailure(f"case-level runtime drift ({where})")


def run(cmd, logfile, timeout) -> int:
    with open(V32 / logfile, "w") as lf:
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


def read_bridge(name: str) -> float:
    path, expected = R2_BRIDGE[name]
    require(path.is_file() and sha256(path) == expected,
            f"R2 bridge archive sha drift/missing: {name}")
    return float(strict_json_loads(path.read_bytes())["agg"]["ret_mean"])


def exam(worker, tag, seeds, manager_npz=None):
    out = EVAL / f"{tag}.json"
    require(not out.exists(), f"archive immutability: {out} already exists, refusing to overwrite")
    lo, hi = (int(x) for x in seeds.split("-", 1))
    seed_values = list(range(lo, hi + 1))
    snapshot = freeze_eval_identity(ROOT, worker, manager_npz)
    assert_case_runtime(snapshot, f"exam:{tag}")
    expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
    worker_arg = (worker if snapshot["worker"]["kind"] in {"script", "bc"}
                  else snapshot["worker"]["path"])
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


def exam_or_adopt(events, worker, tag, seeds, manager_npz=None):
    """Exam terminal clause (generalized from the v31 reference terminal): archived + ledger = frozen; resumed runs accept it; no re-burning."""
    out = EVAL / f"{tag}.json"
    prior = [e for e in events
             if e.get("event") == "exam_ok" and e.get("tag") == tag]
    if out.exists():
        require(bool(prior), f"{tag} leftover archive present but the ledger has no exam_ok; stopping")
        d = validate_adopted(tag, worker, seeds, manager_npz)
        require(d["agg"]["_sha"] == prior[-1]["sha"],
                f"{tag} archive does not match the ledger exam_ok sha")
        log({"event": "exam_adopted", "tag": tag, "sha": d["agg"]["_sha"]})
        return d
    require(not prior, f"{tag} on the ledger but the archive is missing (REF_INVALID type); stopping")
    return exam_retry(worker, tag, seeds, manager_npz)


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


def by_seed(rows, lo=7000, hi=7032) -> dict:
    m = {r["seed"]: r for r in rows}
    require(len(rows) == len(m), "seed set is malformed (contains a duplicate seed)")
    require(set(m) == set(range(lo, hi)), f"seed set is malformed (must be {lo}-{hi - 1})")
    return m


def a12_per_ep(agg) -> float:
    hist = agg.get("worker_action_hist", {}) or {}
    return round(int(hist.get("12", 0)) / max(1, int(agg.get("n", 32))), 2)


def biteq(new_doc, ref_name: str):
    """REF_BITEQ: bit-level, per-seed, per-field reconciliation of the new reference against the v31 reference of the same name."""
    path, expected = V31_REF[ref_name]
    require(path.is_file() and sha256(path) == expected,
            f"v31 reference archive sha drift: {ref_name}")
    old = strict_json_loads(path.read_bytes())
    old_rows, new_rows = old["rows"], new_doc["rows"]
    require(len(old_rows) == len(new_rows) == 32, "reference row count is malformed")
    diffs = []
    for o, n in zip(sorted(old_rows, key=lambda r: r["seed"]),
                    sorted(new_rows, key=lambda r: r["seed"])):
        if o != n:
            diffs.append(o["seed"])
    core = ("ret_mean", "ret_median", "died", "depth_median", "kills_mean",
            "farm_tau_mean", "override_rate", "cap_rate")
    agg_diff = [k for k in core if old["agg"].get(k) != new_doc["agg"].get(k)]
    if diffs or agg_diff:
        log({"event": "CASE_HALT_G0", "via": "REF_BITEQ", "ref": ref_name,
             "row_diff_seeds": diffs, "agg_diff": agg_diff})
        attention(f"REF_BITEQ FAIL ({ref_name}): E1 is not inert on the incumbent stack, so G0 loses its meaning")
        raise SystemExit(7)
    log({"event": "REF_BITEQ", "ref": ref_name, "rows": 32, "agg_core": "equal"})


def leg_starts(events, leg) -> int:
    return sum(1 for e in events
               if e.get("event") == "leg_start" and e.get("leg") == leg)


def read_ledger() -> list[dict]:
    if not LEDGER.is_file():
        return []
    out = []
    for i, line in enumerate(LEDGER.read_text().splitlines(), 1):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise OperationalFailure(f"ledger line {i} cannot be parsed: {exc}") from exc
    return out


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def preflight(events):
    global CASE_RT
    require(PROTOCOL_VERSION == CALIBRATED_PROTOCOL_VERSION, "contract version drift")
    dirty = [l for l in git("status", "--porcelain").splitlines()
             if l != "?? train/leaderboard-assembled-v3.md"]
    require(not dirty, f"W1: working tree not clean {dirty}")
    head = git("rev-parse", "HEAD")
    for path in ("docs/prereg/PREREG-v32-potion-autonomy.md", "train/run_v32_sovereign.py"):
        touch = git("log", "-1", "--format=%H", "--", path)
        require(touch == head, f"W1: {path} last touched != HEAD")
    freezes = [e for e in events if e.get("event") == "FREEZE_SHA"]
    if freezes:
        require(freezes[-1]["sha"] == head, "W1: HEAD != last FREEZE_SHA on the ledger")
    require(KING_ZIP.is_file() and sha256(KING_ZIP) == KING_ZIP_SHA, "king zip drift")
    require(sha256(KING_NPZ) == KING_NPZ_SHA, "king npz drift")
    require(sha256(M29_NPZ) == M29_NPZ_SHA, "M29 npz drift")
    require(zip_steps(KING_ZIP) == 3_497_984, "king zip step count is malformed")
    for name, (path, expected) in {**V31_REF, **R2_BRIDGE}.items():
        require(path.is_file() and sha256(path) == expected,
                f"archive pin mismatch: {name}")
    require(ZEROFLIP_REPORT.is_file()
            and sha256(ZEROFLIP_REPORT) == ZEROFLIP_SHA,
            "zero-flip probe report pin mismatch (anchor-bridge proof 1, archived item)")
    require(THRONE_BASELINE.is_file()
            and sha256(THRONE_BASELINE) == THRONE_BASELINE_SHA,
            "throne trajectory closure baseline pin mismatch (anchor-bridge proof 1, promoted item)")
    tags = (["v32-ref-launch", "v32-ref-science", "v32-golden"]
            + [f"{leg}-{s}" for leg in LEGS for s in ("s16", "full32", "m29")])
    for t in tags:
        if not any(e.get("event") == "exam_ok" and e.get("tag") == t
                   for e in events):
            require(not (EVAL / f"{t}.json").exists(),
                    f"W9: target archive already exists: {t} (restart protocol: .void it first)")
    for leg in LEGS:
        if leg_starts(events, leg) == 0:
            require(not (RUNS / leg).exists(), f"leftover run directory: {leg}")
    snapshot = freeze_eval_identity(ROOT, str(KING_NPZ), None)
    CASE_RT = runtime_five(snapshot)
    prior_rt = [e for e in events if e.get("event") == "CASE_RUNTIME"]
    if prior_rt:
        require(prior_rt[0]["five"] == CASE_RT, "W10: resumed-run runtime reconciliation mismatch")
    else:
        log({"event": "CASE_RUNTIME", "five": CASE_RT})
    if not freezes:
        log({"event": "FREEZE_SHA", "sha": head})
    log({"event": "preflight_ok", "king_zip": KING_ZIP_SHA[:16],
         "bc_sd": sha16(BC_SD) if BC_SD.exists() else None,
         "throne_bridge": read_bridge("throne"),
         "script_bridge": read_bridge("script")})
    return head


def stage_done(events, name) -> bool:
    return any(e.get("event") == name for e in events)


def _main():
    events = read_ledger()
    if any(e.get("event") in ("VERDICT_PATH", "GOLDEN_AUTHORIZED")
           for e in events):
        print("case closed / gold-standard evaluation awaiting manual launch: idempotent exit", flush=True)
        return
    preflight(events)

    # ---- S1 the two G0 forms (re-verified at every launch: a guard rail, not a one-off ritual) ----
    require(TRAJ_BASELINE.is_file()
            and sha256(TRAJ_BASELINE) == TRAJ_BASELINE_SHA,
            "G0 baseline archive pin mismatch (must be the file checked in with the frozen commit)")
    if not stage_done(events, "G0_BASELINE"):
        log({"event": "G0_BASELINE", "sha": TRAJ_BASELINE_SHA,
             "seeds": "7000-7015",
             "captured": "before the change (before the E1 work, clean code from stash)"})
    if not stage_done(events, "E1_SCOPE_NOTE"):
        log({"event": "E1_SCOPE_NOTE",
             "note": "the belt precondition was an implementation choice beyond the pre-registered text (in the spirit of 4C + precedent 14 + the ghost measurement showing it"
                     " necessary); no case-specific response; listed separately in the report"})
    require(run([PY, "train/probe_g0_traj.py", "replay"],
                f"g0-replay.{time.time_ns()}.log", 1_800) == 0,
            "G0-identity replay failed (king)")
    require(run([PY, "train/probe_g0_traj.py", "replay", "throne"],
                f"g0-replay-throne.{time.time_ns()}.log", 1_800) == 0,
            "G0-identity replay failed (throne, anchor-bridge proof 1 promoted item)")
    log({"event": "G0_REPLAY", "verdict": "PASS", "seeds": "7000-7015",
         "workers": ["king", "throne"]})
    require(run([PY, "train/probe_g0_ghost.py"],
                f"g0-ghost.{time.time_ns()}.log", 1_800) == 0,
            "G0-ghost failed")
    log({"event": "G0_GHOST", "verdict": "PASS"})

    # ---- S2 bc_worker regeneration (R1 gate; the teacher receipt of the new impl world) ----
    if stage_done(events, "BC_REGEN"):
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "train"))
        from train_ppo import _validate_bc_report
        rec = _validate_bc_report(BC_SD, "data_gate", verify_replay=False)
        prior_bc = [e for e in events if e.get("event") == "BC_REGEN"][-1]
        require(rec["policy_sha256"] == prior_bc.get(
                    "policy_sha256", rec["policy_sha256"]),
                "BC_SD identity chain broken across launches")
    if not stage_done(events, "BC_REGEN"):
        require(run([PY, "train/bc_worker.py"],
                    f"bc-regen.{time.time_ns()}.log", 3_600) == 0,
                "bc_worker regeneration failed (R-W FAIL type; the whole case stops)")
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "train"))
        from train_ppo import _validate_bc_report
        rec = _validate_bc_report(BC_SD, "data_gate", verify_replay=False)
        log({"event": "BC_REGEN", "held_out_top1": rec["held_out_top1"],
             "pairs": rec["pairs"], "policy_sha256": rec["policy_sha256"]})

    # ---- S3 KING_SD regeneration + teacher attestation ----
    if stage_done(events, "KING_SD_OK"):
        prior_ks = [e for e in events if e.get("event") == "KING_SD_OK"][-1]
        require(KING_SD.exists()
                and sha256(KING_SD) == prior_ks.get("sha256", ""),
                "KING_SD identity chain broken across launches")
    if not stage_done(events, "KING_SD_OK"):
        require(run([PY, "train/export_manager_sd.py", str(KING_ZIP),
                     str(KING_SD)], "export-king-sd.log", 600) == 0
                and KING_SD.exists(), "KING_SD export failed")
        require(run([PY, "train/check_teacher_parity.py", str(KING_SD),
                     str(KING_NPZ)], "parity-king.log", 600) == 0,
                "G-KL-W: king anchor sd vs npz attestation failed")
        log({"event": "KING_SD_OK", "sha256": sha256(KING_SD)})

    # ---- S4 in-case refs + REF_BITEQ (case-level bit-level G0) ----
    refs = {}
    for name, manager in (("launch", None), ("science", str(M29_NPZ))):
        tag = f"v32-ref-{name}"
        already = any(e.get("event") == "exam_ok" and e.get("tag") == tag
                      for e in events)
        d = exam_or_adopt(events, str(KING_NPZ), tag, "7000-7031",
                          manager_npz=manager)
        require(d is not None, f"{tag} reference exam failed repeatedly")
        biteq(d, name)
        if not already:
            log({"event": "exam_ok", "tag": tag, "mean": d["agg"]["ret_mean"],
                 "died": d["agg"]["died"], "sha": d["agg"]["_sha"]})
            events.append({"event": "exam_ok", "tag": tag,
                           "sha": d["agg"]["_sha"]})
        refs[name] = d
    R = refs["launch"]["agg"]["ret_mean"]
    abandon = round(R * ABANDON_FRAC[0] / ABANDON_FRAC[1], 1)
    floor_repro = round(R * FLOOR_FRAC[0] / FLOOR_FRAC[1], 1)
    log({"event": "refs", "R": R, "abandon": abandon, "floor": floor_repro,
         "science_ref": refs["science"]["agg"]["ret_mean"],
         "science_died": refs["science"]["agg"]["died"]})
    ref_rows = by_seed(refs["launch"]["rows"])
    sci_rows = by_seed(refs["science"]["rows"])

    # ---- S5 two legs in series (same seeds; single variable = the autonomy knob) ----
    npz = {}
    for leg, extra in LEGS.items():
        model_path = RUNS / leg / "model_final.zip"
        out = RUNS / leg / "policy.npz"
        if model_path.exists() and zip_steps(model_path) == NT_TARGET:
            if not out.exists():
                # Re-export channel: training reached its target and only the npz is missing; do not record leg_start and do not use up quota
                require(run([PY, "train/export_worker_npz.py", str(model_path),
                             str(out)], f"export-{leg}.retry.log", 600) == 0
                        and out.exists(), f"{leg} re-export failed; stopping")
                log({"event": "npz_exported", "leg": leg,
                     "sha256": sha256(out), "note": "re-export (not a relaunch)"})
            log({"event": "leg_skip_complete", "leg": leg})
            npz[leg] = str(out)
            continue
        require(leg_starts(events, leg) < 2, f"{leg} launch quota exhausted (ledger-based)")
        cmd = [PY, "train/train_ppo.py", "--worker", "--algo", "mppo",
               "--gamma", "1.0", "--max-steps", "3000", "--n-steps", "512",
               "--num-envs", "4", "--lr", "3e-4", "--ent-coef", "0.005",
               "--seed", str(SEED), "--total-steps", str(LEG_STEPS),
               "--run-name", leg, "--distill-beta", str(BETA),
               "--teacher-sd", str(BC_SD), "--teacher-override", str(KING_SD),
               "--skip-dry", "--manager-npz", str(M29_NPZ),
               "--resume-from", str(KING_ZIP), "--allow-legacy-resume",
               "--calib-probes", "3747984,3947984",
               "--calib-record-only"] + extra
        ev = {"event": "leg_start", "leg": leg, "seed": SEED,
              "sovereignty": not extra}
        log(ev)
        events.append(ev)
        t0 = time.time()
        rc = run(cmd, f"train-{leg}.log", timeout=21_600)
        nt = zip_steps(model_path)
        log({"event": "leg_done", "leg": leg, "rc": rc, "nt_zip": nt,
             "dt_min": round((time.time() - t0) / 60, 1)})
        require(rc == 0 and nt == NT_TARGET,
                f"{leg} training did not reach its target (rc={rc}, nt={nt}, target={NT_TARGET})"
                ": proposition not examined; this version adds no retraining")
        require(run([PY, "train/export_worker_npz.py", str(model_path),
                     str(out)], f"export-{leg}.log", 600) == 0 and out.exists(),
                f"{leg} npz export failed")
        npz[leg] = str(out)
        log({"event": "npz_exported", "leg": leg, "sha256": sha256(out)})

    # ---- S6 exams: s16 -> full32 (H) -> full32 (M29) ----
    s16 = {}
    for leg in LEGS:
        tag = f"{leg}-s16"
        already = any(e.get("event") == "exam_ok" and e.get("tag") == tag
                      for e in events)
        d = exam_or_adopt(events, npz[leg], tag, "7000-7015")
        require(d is not None, f"{leg} screen failed repeatedly")
        s16[leg] = d["agg"]["ret_mean"]
        if not already:
            log({"event": "exam_ok", "tag": tag, "mean": d["agg"]["ret_mean"],
                 "died": d["agg"]["died"], "sha": d["agg"]["_sha"]})
            events.append({"event": "exam_ok", "tag": tag,
                           "sha": d["agg"]["_sha"]})
    if all(v < abandon for v in s16.values()):
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"both legs' screens < {abandon}: training failed or failed to transfer across managers"
                    " (trained under M29, screened under H); the autonomy proposition was not examined",
             "s16": s16, "R": R})
        attention("verdict: training failed, proposition not examined")
        return
    full, m29 = {}, {}
    for leg in LEGS:
        tag = f"{leg}-full32"
        already = any(e.get("event") == "exam_ok" and e.get("tag") == tag
                      for e in events)
        d = exam_or_adopt(events, npz[leg], tag, "7000-7031")
        require(d is not None, f"{leg} full-32 failed repeatedly")
        full[leg] = d
        a = d["agg"]
        if not already:
            events.append({"event": "exam_ok", "tag": tag, "sha": a["_sha"]})
            log({"event": "exam_ok", "tag": tag, "mean": a["ret_mean"],
                 "died": a["died"], "a12_per_ep": a12_per_ep(a),
                 "a13_per_ep": round(int(a.get("worker_action_hist", {})
                                         .get("13", 0)) / 32, 2),
                 "rwin_per_ep": round(sum(r["mode_seq"].count("R")
                                          for r in d["rows"]) / 32, 2),
                 "depth2": depth2_count(d["rows"]),
                 "dive": round(dive_per_ep(d["rows"]), 2),
                 "override": a["override_rate"], "sha": a["_sha"]})
        tag2 = f"{leg}-m29"
        already2 = any(e.get("event") == "exam_ok" and e.get("tag") == tag2
                       for e in events)
        d2 = exam_or_adopt(events, npz[leg], tag2, "7000-7031",
                           manager_npz=str(M29_NPZ))
        require(d2 is not None, f"{leg} M29 deep-dive exam failed repeatedly")
        m29[leg] = d2
        a2 = d2["agg"]
        if not already2:
            events.append({"event": "exam_ok", "tag": tag2, "sha": a2["_sha"]})
            log({"event": "exam_ok", "tag": tag2, "mean": a2["ret_mean"],
                 "died": a2["died"], "a12_per_ep": a12_per_ep(a2),
                 "a13_per_ep": round(int(a2.get("worker_action_hist", {})
                                         .get("13", 0)) / 32, 2),
                 "rwin_per_ep": round(sum(r["mode_seq"].count("R")
                                          for r in d2["rows"]) / 32, 2),
                 "depth2": depth2_count(d2["rows"]),
                 "sha": a2["_sha"]})

    # ---- R32 split (sov-ctrl, same seeds pair by pair, recorded only, not judged) ----
    sov_rows = by_seed(full["v32-sov"]["rows"])
    ctrl_rows = by_seed(full["v32-ctrl"]["rows"])
    split = [sov_rows[s]["ret"] - ctrl_rows[s]["ret"] for s in sorted(sov_rows)]
    tri_ctrl = [ctrl_rows[s]["ret"] - ref_rows[s]["ret"] for s in sorted(ref_rows)]
    tri_sov = [sov_rows[s]["ret"] - ref_rows[s]["ret"] for s in sorted(ref_rows)]
    sp_mean = sum(split) / 32
    log({"event": "R32_SPLIT",
         "paired_sov_minus_ctrl": round(sp_mean, 2),
         "wins_sov": sum(x > 0 for x in split),
         "direction": ("direction undetermined (|mean diff|<2, following the R30.6 reading precedent)"
                       if abs(sp_mean) < 2.0 else "direction reading"),
         "triangle": {"ctrl_minus_ref": round(sum(tri_ctrl) / 32, 2),
                      "sov_minus_ref": round(sum(tri_sov) / 32, 2),
                      "note": "triangle reconciliation; the single sov-ctrl reading must not stand for the autonomy effect"},
         "died": {"sov": full["v32-sov"]["agg"]["died"],
                  "ctrl": full["v32-ctrl"]["agg"]["died"]},
         "a12": {"sov": a12_per_ep(full["v32-sov"]["agg"]),
                 "ctrl": a12_per_ep(full["v32-ctrl"]["agg"])},
         "definition": "same seeds only guarantee equal initial weights and env stream; the autonomy mask changes sampling from the first renormalized"
                 " distribution on, so trajectories diverge at once: 'single variable' is a configuration-level statement;"
                 " the two legs' step ledgers are identical (+499712 each), so there is no second definition"})

    # ---- R32 main verdict (sov x M29 vs ref-science; three exhaustive tiers, independent of launch) ----
    sm = m29["v32-sov"]
    sm_rows = by_seed(sm["rows"])
    surv_diffs = [sm_rows[s]["ret"] - sci_rows[s]["ret"] for s in sorted(sci_rows)]
    surv_mean = sum(surv_diffs) / 32
    sm_died = sm["agg"]["died"]
    sm_a12 = a12_per_ep(sm["agg"])
    ctrl_m29_died = m29["v32-ctrl"]["agg"]["died"]
    if (sm_a12 >= A12_USE_LINE and sm_died < SURV_DIED_LINE
            and surv_mean >= SURV_MEAN_BAND):
        if ctrl_m29_died < SURV_DIED_LINE:
            main_verdict = (f"survival improvement co-occurs with autonomy (retraining side-effect candidate, ctrl x M29 died "
                            f"{ctrl_m29_died} also <{SURV_DIED_LINE}; causal verbs forbidden): "
                            f"died {sm_died} and mean diff {surv_mean:.2f} and "
                            f"a12/episode {sm_a12}>={A12_USE_LINE}")
        else:
            main_verdict = (f"autonomy converts into survival: died {sm_died}<{SURV_DIED_LINE} and "
                            f"paired mean diff {surv_mean:.2f}>={SURV_MEAN_BAND} and "
                            f"a12/episode {sm_a12}>={A12_USE_LINE} (ctrl x M29 died "
                            f"{ctrl_m29_died} does not co-occur): the pre-registered success line is met")
    elif (sm_a12 < A12_USE_LINE and sm_died <= SURV_DIED_LINE
          and surv_mean >= SURV_MEAN_BAND):
        note2 = (" (low usage co-occurs with improved survival; attribution unproven; the death-seed set difference goes with the verdict)"
                 if sm_a12 > 0 and sm_died < SURV_DIED_LINE else "")
        main_verdict = (f"autonomy given but not used: a12/episode {sm_a12}<{A12_USE_LINE} and "
                        f"died {sm_died}<={SURV_DIED_LINE} and mean diff "
                        f"{surv_mean:.2f}>={SURV_MEAN_BAND}: 4C harmless and ineffective, "
                        f"move to the 4B agenda{note2}")
    else:
        if sm_a12 >= A12_USE_LINE and sm_died <= SURV_DIED_LINE \
                and surv_mean >= SURV_MEAN_BAND:
            sub = "used, but survival not converted and no degradation ('harmful' forbidden)" + (
                "; return improved but survival unproven" if surv_mean > 0 else "")
        elif sm_a12 >= A12_USE_LINE:
            sub = "used and degraded"
        else:
            sub = f"low usage and degraded (retraining/protocol side-effect candidate, a12={sm_a12})"
        main_verdict = (f"tier 3 ({sub}): a12/episode {sm_a12}, died {sm_died}"
                        f" (line {SURV_DIED_LINE}), mean diff {surv_mean:.2f}"
                        f" (band {SURV_MEAN_BAND}): triggers the decoupling clause, "
                        "move to the bundled evaluation case (the pre-registered 'if it does not work' branch)")
    ref_dead = {r["seed"] for r in refs["science"]["rows"] if r["died"]}
    sov_dead = {r["seed"] for r in sm["rows"] if r["died"]}
    log({"event": "R32_MAIN", "verdict": main_verdict, "died": sm_died,
         "a12_per_ep": sm_a12, "paired_mean_vs_sciref": round(surv_mean, 2),
         "wins": sum(x > 0 for x in surv_diffs),
         "binom_note": "P(died<=2|n=32,p=3/32)~0.41: the died tier is a direction reading, "
                       "not evidence of significance (same discipline as the 18/32 line note)",
         "ctrl_m29": {"mean": m29["v32-ctrl"]["agg"]["ret_mean"],
                      "died": ctrl_m29_died,
                      "a12": a12_per_ep(m29["v32-ctrl"]["agg"])},
         "death_seed_sets": {"rescued": sorted(ref_dead - sov_dead),
                             "new_deaths": sorted(sov_dead - ref_dead)},
         "depth_vectors": {
             "sov_m29": [r["depth"] for r in sorted(sm["rows"],
                                                    key=lambda r: r["seed"])],
             "sciref": [r["depth"] for r in sorted(refs["science"]["rows"],
                                                   key=lambda r: r["seed"])]},
         "belt_econ": {"sov_m29_a13": round(int(sm["agg"].get(
                           "worker_action_hist", {}).get("13", 0)) / 32, 2),
                       "sciref_a13": round(int(refs["science"]["agg"].get(
                           "worker_action_hist", {}).get("13", 0)) / 32, 2)}})

    # ---- eligibility/winner/floor/launch (v31 ladder verbatim) ----
    def qual_of(d):
        a = d["agg"]
        dpe_ = dive_per_ep(d["rows"])
        void_ = (a["override_rate"] >= R4["override_void"]
                 or (dpe_ > 1 and a["died"] > 6))
        hard_ok_ = a["farm_descend_rate"] <= R4["descend"] and a["cap_rate"] < R4["cap"]
        override_ok_ = a["override_rate"] < R4["override_sentinel"]
        dual_ = dpe_ > 1 and hard_ok_ and not override_ok_
        ok_ = (a["died"] <= DEATHS_MAX and not void_
               and ((hard_ok_ and override_ok_) or dual_))
        return {"qual_ok": ok_, "void": void_, "dual_attr": dual_}

    quals = {n: qual_of(full[n]) for n in LEGS}
    cap_note = {}
    for n in LEGS:
        a = full[n]["agg"]
        if (not quals[n]["qual_ok"] and a["died"] <= DEATHS_MAX
                and a["cap_rate"] >= R4["cap"]):
            cap_note[n] = ("disqualified only by cap_rate: gate-semantics drift candidate (autonomous drinking"
                           " prolongs survival and pushes up cap hits; the line was not recalibrated, and adjusting it is forbidden in this case)")
    log({"event": "quals", **quals,
         **({"cap_drift_note": cap_note} if cap_note else {})})
    pool = [n for n in LEGS if quals[n]["qual_ok"]]
    if not pool:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": "both legs ineligible: no winner, the succession proposition unanswered (outside statistical power)",
             "arms_full32": {n: full[n]["agg"]["ret_mean"] for n in LEGS},
             "R": R})
        attention("verdict: both legs ineligible (R32 main verdict/split are on record)")
        return
    prelim = max(LEGS, key=lambda n: full[n]["agg"]["ret_mean"])
    if prelim not in pool:
        log({"event": "substitution", "blocked": prelim, "why": quals[prelim]})
    ms = {n: full[n]["agg"]["ret_mean"] for n in pool}
    band = [n for n in pool if max(ms.values()) - ms[n] <= 0.05]
    if len(band) > 1:
        dmin = min(full[n]["agg"]["died"] for n in band)
        band = [n for n in band if full[n]["agg"]["died"] == dmin]
        winner = "v32-sov" if "v32-sov" in band else band[0]
    else:
        winner = band[0]
    W = full[winner]
    wa = W["agg"]
    wrows = by_seed(W["rows"])
    log({"event": "winner", "leg": winner, "mean": wa["ret_mean"],
         "died": wa["died"], "substituted": winner != prelim})
    if wa["ret_mean"] < floor_repro:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"winner {wa['ret_mean']} < {floor_repro} (=(85/92)x{R})"
                    ": reference level not reproduced, the succession proposition not examined",
             "arms_full32": {n: full[n]["agg"]["ret_mean"] for n in LEGS},
             "R": R, "main_verdict": main_verdict})
        attention("verdict: reference level not reproduced (R32 main verdict/split are on record)")
        return
    diffs = [wrows[s]["ret"] - ref_rows[s]["ret"] for s in sorted(ref_rows)]
    pd_mean = sum(diffs) / 32
    pd_wins = sum(d > 0 for d in diffs)
    launch = pd_mean >= PAIRED_DIFF and pd_wins >= PAIRED_WINS
    log({"event": "launch_check", "paired_mean": round(pd_mean, 2),
         "paired_wins": pd_wins, "died": wa["died"],
         "multiple_comparison": "5th draw on the 18/32 line; P(>=18|p=.5)~30%"
                                " (exact 0.2983; the ~43% in v29-v31 was a slip for"
                                " P(>=17), corrected in a note with this case's verdict)"})
    throne = read_bridge("throne")
    script_ref = read_bridge("script")
    p_hi = round(throne + 4.0, 1)
    P_LINE = (f"P line (anchor-bridge matching: the throne's incumbent anchor value {throne} is the R2 re-measurement of r2-throne; "
              f"re-measuring does not change the title, the throne title still belongs to the incumbent assembled agent; script reference {script_ref}): deaths>6 -> revert; "
              f"gold>={p_hi} and deaths<=4 -> P32 takes the throne; in ({throne},{p_hi}) and deaths<=4 -> point estimate; "
              f">{throne} and deaths 5-6 -> tie (safety); in [{script_ref},{throne}] -> tie; "
              f"<{script_ref} -> revert. Title succession follows the written version of the v31 appendix; the two gates are declared on separate lines. "
              "Gold-pool opening history: the gold standard was actually opened three times (v22/v23/v24); the five R2 anchoring runs were measurement"
              " exposures; v31 did not open it; if this case launches, it is the 4th actual gold opening and the 9th cumulative gold-pool"
              " exposure; the fixed-pool bias goes with the verdict")
    if launch:
        golden_cmd = (f"{PY} train/eval_assembled.py --worker {npz[winner]} "
                      f"--manager-npz {H_NPZ} --seeds 9000-9031 "
                      f"--tag v32-golden --board")
        log({"event": "GOLDEN_AUTHORIZED", "leg": winner,
             "probe32_mean": wa["ret_mean"], "died": wa["died"],
             "wins": pd_wins, "mean_diff": round(pd_mean, 2),
             "worker_npz_sha": sha16(npz[winner]), "full32_sha": wa["_sha"],
             "golden_cmd": golden_cmd, "p_line": P_LINE,
             "main_verdict": main_verdict,
             "note": "gold-standard evaluation started manually, single arm, once; governed by the campaign-level gold-evaluation"
                     " discipline (v31 clause); the three anchor-bridge proofs (zero-flip archive/REF_BITEQ x2/"
                     "ghost assertion E) are on record; the throne anchor bridge is an out-of-distribution extrapolation (qualified in the verdict)"})
        attention(f"gold-standard evaluation awaiting manual launch: {winner}; R32 main verdict: {main_verdict}")
        return
    wins_note = (f" (width note: won {pd_wins}/32 >=14)" if pd_wins >= 14 else "")
    if pd_mean >= PAIRED_DIFF:
        verdict = f"mean gain +{pd_mean:.2f} but width not reached (won {pd_wins}/32): a point-estimate gain does not spend the gold run"
    elif pd_mean >= 2.0:
        verdict = f"paired mean diff {pd_mean:.2f} in [+2,+4): a probe-level improvement does not spend the gold run{wins_note}"
    else:
        verdict = f"paired mean diff {pd_mean:.2f} <+2: the incumbent stays{wins_note}"
    log({"event": "VERDICT_PATH", "golden_authorized": False,
         "verdict": verdict, "main_verdict": main_verdict,
         "winner": winner, "arms_full32": {n: full[n]["agg"]["ret_mean"]
                                           for n in LEGS},
         "paired_mean": round(pd_mean, 2), "paired_wins": pd_wins, "R": R})
    attention(f"verdict (no launch): {verdict}; R32 main verdict: {main_verdict}")


def main():
    try:
        with exclusive_lock(V32 / ".driver.lock", "v32 driver"):
            _main()
    except (OperationalFailure, OutputReservationError) as e:
        log({"event": "OPERATIONAL_FAILURE", "why": str(e)})
        attention("operational failure:\n" + str(e))
        raise SystemExit(2) from e
    except SystemExit:
        raise
    except Exception as e:
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("driver died with an exception:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
