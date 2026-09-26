"""v30 verdict resume runner (restart-protocol executor; PREREG-v30 residual/restart clauses + the crash autopsy).

Background: the main driver died in the full-32 instrumentation stage from a JSON dict-key bug in metrics()
(worker_action_hist keys are strings, so sum raised TypeError). All training is clean and intact
(the 4-leg nt chain is bit-exact; the bc leg-2 trip-line stop is the clause working normally), and the king full-32 archive was
written legitimately (v30-king-full32.json, all 32 rows). This runner reuses all clean assets and executes only the remaining
verdict clauses: bc full 32 -> r30_6 -> eligibility/substitution -> scientific main verdict -> floor -> dual-anchor launch ->
verdict text. The logic is verbatim the same source as the main driver (it imports the fixed library functions).
Usage: .venv/bin/python train/run_v30_verdict.py
"""
import pathlib
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_contract import (PROTOCOL_VERSION, OperationalFailure,  # noqa: E402
                           OutputReservationError,
                           exclusive_lock, expected_eval_identity,
                           freeze_eval_identity, read_eval_archive,
                           sha256_file, verify_eval_identity)
from run_v30_relay import (D2DEATH_LAUNCH_MAX, EVAL, FLOOR, LAUNCH_ANCHOR,  # noqa: E402
                           LAUNCH_SHA, M29_NPZ, M29_SHA, PD112_LINE, PRE140_LINE,
                           ROOT, RUNS, SCI_ANCHOR, SCI_SHA, PY, WINS112_LINE,
                           attention, by_seed, log, metrics, qual_of,
                           read_comparable_launch, read_comparable_science,
                           require, run, sha16)
from run_v30_relay import OperationalFailure as RelayOperationalFailure  # noqa: E402

ARM_MODELS = {"king": str(RUNS / "v30-king-leg2" / "model_final.zip"),
              "bc": str(RUNS / "v30-bc-leg2" / "model_final.zip")}   # final leg = the last clean-close leg (clause)
ARM_MODEL_SHA256 = {
    "king": "179b5ede444601279b79790be477246ecb6546eb1d4e6de7d923702f8634e24f",
    "bc": "37acee85b3450168deed267bd8e0797522e4af19bb7b8c83858ccde97aad09f7",
}
KING_ARCHIVE_SHA256 = "6ce157a36703aca65554489c5742a96c75a4585d7212bf7247492403db4bfac6"
CALIBRATED_PROTOCOL_VERSION = 2


def require_calibrated_protocol() -> None:
    if PROTOCOL_VERSION != CALIBRATED_PROTOCOL_VERSION:
        raise OperationalFailure(
            "v30 verdict eligibility/R4/science and launch static thresholds are calibrated only under pre-v3 environment semantics; "
            "re-run the protocol-v3 baseline and update the pre-registration by hand first; mixing in the old thresholds is forbidden"
        )


def read_comparable_king_archive() -> dict:
    """The king result reused by the resumed verdict is also an active model-selection input; no legacy amnesty."""
    path = EVAL / "v30-king-full32.json"
    try:
        snapshot = freeze_eval_identity(ROOT, ARM_MODELS["king"], M29_NPZ)
        expected = expected_eval_identity(
            snapshot, tag="v30-king-full32", seeds=range(7000, 7032))
        document = read_eval_archive(path, **expected)
        verify_eval_identity(snapshot, ROOT)
        return document
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise OperationalFailure(
            "v30-king-full32 does not satisfy the current schema-v2 comparability contract; "
            "after an environment-semantics change, re-run that arm with the fixed king leg2 worker + M29 manager"
        ) from exc


def exam(model_path, tag, seeds):
    """Evaluation for the resumed verdict only: the full input identity is locked both before launch and after collection."""
    out = EVAL / f"{tag}.json"
    require(not out.exists(), f"archive immutability: {out} already exists, refusing to overwrite")
    lo, hi = (int(x) for x in seeds.split("-", 1))
    seed_values = list(range(lo, hi + 1))
    require(seed_values and lo >= 0, f"illegal seed range: {seeds}")
    snapshot = freeze_eval_identity(ROOT, model_path, M29_NPZ)
    expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
    cmd = [PY, "train/eval_assembled.py",
           "--worker", snapshot["worker"]["path"],
           "--manager-npz", snapshot["manager"]["path"],
           "--seeds", seeds, "--tag", tag]
    if run(cmd, f"exam-{tag}.{time.time_ns()}.log", timeout=1_800) != 0:
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    try:
        document = read_eval_archive(out, **expected)
        verify_eval_identity(snapshot, ROOT)
        document["agg"]["_sha"] = sha16(out)
        return document
    except (OSError, KeyError, TypeError, ValueError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None


def exam_retry(model_path, tag, seeds):
    document = exam(model_path, tag, seeds)
    if document is None:
        log({"event": "exam_crash", "tag": tag,
             "note": "evaluation failed; retaking once under the crash clause"})
        document = exam(model_path, tag, seeds)
    return document


def main():
    try:
        with exclusive_lock(RUNS / "v30" / ".verdict.lock", "v30 verdict resume driver"):
            _main()
    except (OperationalFailure, RelayOperationalFailure,
            OutputReservationError) as exc:
        log({"event": "OPERATIONAL_FAILURE", "why": str(exc)})
        attention("verdict resume operational failure:\n" + str(exc))
        raise SystemExit(2) from exc
    except Exception as exc:
        log({"event": "VERDICT_EXCEPTION", "why": repr(exc),
             "traceback": traceback.format_exc()})
        attention("verdict resume died with an exception:\n" + traceback.format_exc())
        raise


def _main():
    require_calibrated_protocol()
    king_archive = EVAL / "v30-king-full32.json"
    require(M29_NPZ.is_file() and sha256_file(M29_NPZ) == M29_SHA,
            "M29 manager npz drift/missing")
    for arm, model in ARM_MODELS.items():
        path = pathlib.Path(model)
        require(path.is_file() and sha256_file(path) == ARM_MODEL_SHA256[arm],
                f"{arm} resumed-verdict model missing or sha drift: {model}")
    read_comparable_science()
    read_comparable_launch()
    read_comparable_king_archive()
    log({"event": "restart", "why": "main driver DRIVER_EXCEPTION (metrics dict-key bug,"
         " fixed in run_v30_relay.metrics); training assets all clean, king full-32 archive legitimately on disk",
         "reused": {"v30-king-full32.json": sha16(EVAL / "v30-king-full32.json"),
                    "king_leg2": sha16(ARM_MODELS["king"]), "bc_leg2": sha16(ARM_MODELS["bc"])},
         "note": "restart protocol: resume only the verdict stage, touching neither training nor existing archives"})
    science_doc = read_comparable_science()
    ref_sci = by_seed(science_doc["rows"])

    full, mets = {}, {}
    d = read_comparable_king_archive()
    d["agg"]["_sha"] = sha16(king_archive)
    full["king"] = d
    mets["king"] = metrics(d)
    log({"event": "full32", "arm": "king", "reused_archive": True, **mets["king"]})
    d = exam_retry(pathlib.Path(ARM_MODELS["bc"]), "v30-bc-full32", "7000-7031")
    if d is None:
        why = "bc full 32 failed repeatedly: manual autopsy"
        log({"event": "STOP", "why": why})
        attention(why)
        raise OperationalFailure(why)
    full["bc"] = d
    mets["bc"] = metrics(d)
    log({"event": "full32", "arm": "bc", **mets["bc"]})

    fk, fb = by_seed(full["king"]["rows"]), by_seed(full["bc"]["rows"])
    r30_6 = sum(fk[s]["ret"] - fb[s]["ret"] for s in fk) / 32
    r30_6w = sum(fk[s]["ret"] > fb[s]["ret"] for s in fk)
    log({"event": "r30_6", "king_minus_bc_mean": round(r30_6, 2), "king_wins": r30_6w,
         "note": "|mean diff|<2 means direction undetermined (pre-registered reading rule)"})

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
         " the ledger records only, does not judge; P(wins>=18|p=.5)~43% note; new-evidence conjunction added to the launch line"})

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
         "note": "the scientific main verdict and launch/throne/Mark-I never rewrite each other; the verdict carries both arms' instrumentation (quals/full32 events)"})

    if wm["mean"] < floor:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": f"winner {wm['mean']} < floor {floor} (=0.92x{science_base}): retraining did not reproduce"
                        f" the starting level, the launch flow ends; scientific main verdict: {sci}"})
        attention(f"verdict: below the floor; scientific main verdict: {sci}")
        return

    launch = (pd140 >= PRE140_LINE and d2d <= d2death_launch_max
              and pd112 >= PD112_LINE and wins112 >= WINS112_LINE)
    if launch:
        golden_cmd = (f"{PY} {ROOT / 'train' / 'eval_assembled.py'} --worker "
                      f"{pathlib.Path(ARM_MODELS[winner]).with_suffix('')} --manager-npz {M29_NPZ} "
                      f"--seeds 9000-9031 --tag v30-golden --board")
        dual_note = ("[dual attribution undecided; decide first, then spend: manual launch only after writing back dual_attr_ruling]"
                     if quals[winner]["dual_attr"] else "")
        sci_note = ("" if sci.startswith("relay valid")
                    else f"[scientific main verdict is not 'valid' ({sci}); the verdict must not use relay-success wording]")
        log({"event": "GOLDEN_AUTHORIZED", "arm": winner, "probe32_mean": wm["mean"],
             "died": wm["died"], "vs112": [round(pd112, 2), wins112],
             "vs140": [round(pd140, 2), wins140], "model": ARM_MODELS[winner],
             "model_sha": sha16(pathlib.Path(ARM_MODELS[winner])), "full32_sha": wm["sha"],
             "golden_cmd": golden_cmd,
             "p_line": "in order: deaths>6 revert; >=101.2 and deaths<=4 take the throne; (97.2,101.2) and deaths<=4 point estimate;"
                       " >97.2 and deaths 5-6 tie (safety); [93.9,97.2] tie; <93.9 revert",
             "note": dual_note + sci_note
                     + "4th actual opening in the gold pool's history; single arm, once; write back golden_result after the opening"})
        attention(dual_note + f"gold-standard evaluation awaiting manual launch: {winner}; scientific main verdict: {sci}")
        return

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
