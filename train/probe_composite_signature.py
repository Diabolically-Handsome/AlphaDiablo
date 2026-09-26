"""B1-E5: F-lock composite-signature script + manager-invariant discriminator v2 (PREREG-B1 E5/D3).

Composite signature (constants verbatim from PREREG-B1 D3, frozen for the case): seed s is an "F-lock candidate" iff
  (i)   s is in the depth>=2 seed set of the same-pool king x H reference;
  (ii)  the leg x H archive has 0 D windows for that seed (D windows := count of "D" in mode_seq);
  (iii) the seed's median F-window tau is in the floor band [25, 40] (closed; the tau median comes from the B1-E4
        replay report per_seed.farm_tau_median -- archives do not store per-window tau, replay is the only source).
(ii) AND (iii) are both required -- a healthy throne also reaches tau==25 in 71.4% (P5 (2)), so the tau floor alone is not a criterion.

Post-hit triage: AND discriminator v2 says "trajectory variable" -> F-lock type; "invariant (incl. near miss)" ->
worker-damage candidate, carrying the E5 semantic-narrowing caveat.

Discriminator v2 (same definition as docs/assets/manager_invariant_registry.json v2):
  - strict_invariant: all 13 fields (incl. mode_seq character by character) equal (the v1 criterion);
  - near_miss (P4: correction for the 7017 type): outcome fields {ret, depth, died, kills} equal
    AND at least one trajectory field {farm_n, farm_tau_mean, farm_tau_sum, farm_descend, windows,
    beats, overrides, cap, mode_seq} differs;
  - variable: the outcome fields already differ (trajectory variable).

Mandatory caveats (attached to every verdict, PREREG-B1 D3):
  - signature specificity is uncalibrated (no healthy-leg control in the case, false-positive rate unknown; conversion clause in D3);
  - narrowed discriminator semantics: "manager invariant" only shows "swapping the examiner does not help", not "the loss is unrecoverable";
  - horizon caveat: all signature conditions hold under the max-steps 3000 evaluation protocol; this case may not change the horizon.

Usage:
  .venv/bin/python train/probe_composite_signature.py \
      --ref-archive <king x H same-pool reference.json> --leg-h-archive <leg x H.json> \
      --leg-m29-archive <leg x M29.json> --leg-replay <B1-E4 report.json> \
      --out <report.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))

# ---- D3 constants (pre-registered, frozen for the case) ----
TAU_FLOOR_LO = 25.0
TAU_FLOOR_HI = 40.0            # closed interval [25, 40]
OUTCOME_FIELDS = ("ret", "depth", "died", "kills")
TRAJECTORY_FIELDS = ("farm_n", "farm_tau_mean", "farm_tau_sum",
                     "farm_descend", "windows", "beats", "overrides",
                     "cap", "mode_seq")
INVARIANT_FIELDS_V1 = OUTCOME_FIELDS + TRAJECTORY_FIELDS   # 13 fields
MANDATORY_CAVEATS = (
    "signature specificity uncalibrated: no healthy-leg control in the case, false-positive rate unknown (PREREG-B1 D3 residual 13; "
    "conversion clause: if RB.4 lands on 'not reproduced', the P8 leg hit rate becomes the first measured false-positive rate)",
    "narrowed discriminator semantics (from P7 ruling 3): 'manager invariant' only shows 'swapping the examiner does not help', "
    "not 'the loss is unrecoverable'",
    "horizon caveat: signature conditions hold under the evaluation-protocol horizon (max-steps 3000); "
    "'D windows = 0' is a statement within the horizon (precedent: sov_7029 was only 8 windows from crossing zero)",
)


def d_window_count(row: dict) -> int:
    """D windows := count of 'D' in mode_seq (the death mark does not affect the 'D' character itself)."""
    return str(row["mode_seq"]).count("D")


def invariant_class_v2(row_h: dict, row_m29: dict) -> str:
    """Two rows, same worker, different manager -> strict_invariant / near_miss / variable."""
    if all(row_h[f] == row_m29[f] for f in INVARIANT_FIELDS_V1):
        return "strict_invariant"
    if (all(row_h[f] == row_m29[f] for f in OUTCOME_FIELDS)
            and any(row_h[f] != row_m29[f] for f in TRAJECTORY_FIELDS)):
        return "near_miss"
    return "variable"


def composite_signature(ref_rows: dict[int, dict], leg_rows: dict[int, dict],
                        tau_median_by_seed: dict[int, float | None]) -> dict:
    """Per-seed composite-signature verdict. Inputs are {seed: row}; tau median from the E4 replay report."""
    ref_d2 = sorted(s for s, r in ref_rows.items() if r["depth"] >= 2)
    per_seed = {}
    for s in ref_d2:
        if s not in leg_rows:
            raise ValueError(f"leg archive lacks reference depth>=2 seed {s}")
        leg_d = d_window_count(leg_rows[s])
        tau_med = tau_median_by_seed.get(s)
        cond_ii = leg_d == 0
        cond_iii = (tau_med is not None
                    and TAU_FLOOR_LO <= float(tau_med) <= TAU_FLOOR_HI)
        per_seed[s] = {
            "cond_i_ref_depth2": True,
            "cond_ii_leg_d_windows_zero": cond_ii,
            "leg_d_windows": leg_d,
            "cond_iii_tau_floor": cond_iii,
            "farm_tau_median": tau_med,
            "signature_hit": bool(cond_ii and cond_iii),
        }
    hits = sorted(s for s, v in per_seed.items() if v["signature_hit"])
    return {"ref_depth2_seeds": ref_d2, "n_ref_depth2": len(ref_d2),
            "per_seed": per_seed, "hits": hits, "n_hits": len(hits)}


def triage(signature: dict, leg_h_rows: dict[int, dict],
           leg_m29_rows: dict[int, dict]) -> dict:
    """Post-hit triage: trajectory variable -> F-lock type; invariant (incl. near miss) -> worker-damage candidate."""
    out = {}
    for s in signature["hits"]:
        if s not in leg_m29_rows:
            raise ValueError(f"M29-side archive lacks seed {s}; triage undecidable")
        cls = invariant_class_v2(leg_h_rows[s], leg_m29_rows[s])
        out[s] = {
            "invariant_class_v2": cls,
            "verdict": ("F-lock type" if cls == "variable"
                        else "worker-damage candidate (with the E5 semantic-narrowing caveat)"),
        }
    n_flock = sum(1 for v in out.values() if v["verdict"] == "F-lock type")
    return {"per_seed": out, "n_flock_type": n_flock,
            "n_worker_damage_candidate": len(out) - n_flock}


def rows_by_seed(doc: dict) -> dict[int, dict]:
    rows = {int(r["seed"]): r for r in doc["rows"]}
    if len(rows) != len(doc["rows"]):
        raise ValueError("archive contains duplicate seed")
    return rows


def main() -> int:
    from eval_contract import strict_json_loads

    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-archive", required=True,
                    help="same-pool king x H reference archive (source of the depth>=2 seed set)")
    ap.add_argument("--leg-h-archive", required=True, help="leg x H archive")
    ap.add_argument("--leg-m29-archive", required=True, help="leg x M29 archive")
    ap.add_argument("--leg-replay", required=True,
                    help="B1-E4 replay report for leg x H (source of farm_tau_median)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = {k: pathlib.Path(getattr(args, k)) for k in
             ("ref_archive", "leg_h_archive", "leg_m29_archive", "leg_replay")}
    payloads = {k: p.read_bytes() for k, p in paths.items()}
    ref = rows_by_seed(strict_json_loads(payloads["ref_archive"]))
    leg_h = rows_by_seed(strict_json_loads(payloads["leg_h_archive"]))
    leg_m29 = rows_by_seed(strict_json_loads(payloads["leg_m29_archive"]))
    replay = strict_json_loads(payloads["leg_replay"])
    if not replay.get("fidelity_ok", False):
        raise ValueError("E4 replay report failed fidelity; tau median not usable")
    tau = {int(s): v.get("farm_tau_median")
           for s, v in replay.get("per_seed", {}).items()}

    sig = composite_signature(ref, leg_h, tau)
    tri = triage(sig, leg_h, leg_m29)
    report = {
        "probe": "b1-composite-signature",
        "constants": {"tau_floor_band": [TAU_FLOOR_LO, TAU_FLOOR_HI],
                      "outcome_fields": list(OUTCOME_FIELDS),
                      "trajectory_fields": list(TRAJECTORY_FIELDS)},
        "inputs_sha256": {k: hashlib.sha256(v).hexdigest()
                          for k, v in payloads.items()},
        "signature": sig,
        "triage": tri,
        "mandatory_caveats": list(MANDATORY_CAVEATS),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"composite signature: hits {sig['n_hits']}/{sig['n_ref_depth2']}; "
          f"F-lock type {tri['n_flock_type']}, worker-damage candidates "
          f"{tri['n_worker_damage_candidate']}; report saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
