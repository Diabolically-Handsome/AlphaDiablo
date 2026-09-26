"""B1-E4 offline replay channel for manager observation drift (PREREG-B1 E4; modelled on probe_f2_replay.py).

Evaluation archives do not store observations; the replay probe is the only channel. For a given (worker npz x manager npz x
evaluation archive) it re-runs the deterministic protocol of eval_assembled.evaluate window by window and records
the manager observations without side effects; replay fidelity is self-checked by reconciling archive rows (ROW_FIELDS)
field by field -- the fidelity anchor order is fixed: write the evaluation archive first, then replay and reconcile (engineering m-8).

OBS_DRIFT statistics (schema pinned, statistics M-5; closed enumeration, no rewording on the reading day):
  - the 8 appended dims (manager observation indices 295-302), each with {mean, std, min, max, P5, P50, P95},
    over all window-end observations (the observation returned by each window's step(), incl. the terminal one);
  - window-end walkable shape (local-map indices 44-164): window-end mean of walkSum/121 + south-west band mean;
  - D-window decision shape (RB.10 throne D-window reading): at decision points that choose DIVE, the
    walkSum/121 mean and the south-west band mean (the decision observation is the previous window's end state; the first window uses the start state).

South-west band definition (pinned during implementation; the case file has no machine-executable precedent -- if it
disagrees with the original P6 formation definition it must be corrected before freeze; this definition is listed among the deviations
in the freeze report): 11x11 local map index=(dy+5)*11+(dx+5) (bridge local_map: outer dy, inner dx); engine displacement.hpp sets
Direction::SouthWest = {0, +1}, so the south-west band is the 15-cell strip with dy in [1,5] and |dx|<=1.

Safety: recording is pure numpy forward passes and array copies; it consumes no RNG and changes no env state; no evaluation
archive is written. Output = a single report JSON (--out) carrying the five runtime sha values and all weight sha values.

Usage:
  .venv/bin/python train/probe_b1_obsdrift.py --worker <npz> --archive <json> \
      --out <report.json> [--manager <npz>] [--seeds 7001,7004] [--smoke]
Exit codes: 0 = every seed replayed with fidelity; 1 = any mismatch (the report is still written and records it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))

from eval_contract import (bridge_binary_path,            # noqa: E402
                           runtime_identity, strict_json_loads)

H_NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
ROW_FIELDS = ("ret", "depth", "died", "kills", "farm_n", "farm_tau_sum",
              "farm_descend", "windows", "beats", "overrides", "cap",
              "mode_seq")
APPENDED8 = ("time_remaining", "stagnation_clock", "layer_kills",
             "layer_time", "last_opt_farm", "last_opt_dive",
             "last_opt_resupply", "last_opt_tau")   # manager observation 295..302
WALK_LO, WALK_HI = 44, 165        # walkable local-map indices [44, 164] in the base observation
SW_BAND_CELLS = tuple((dy + 5) * 11 + (dx + 5)
                      for dy in (1, 2, 3, 4, 5) for dx in (-1, 0, 1))
_PCTS = (5, 50, 95)


def sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _stats(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    p5, p50, p95 = (float(x) for x in np.percentile(v, _PCTS))
    return {"mean": round(float(v.mean()), 6), "std": round(float(v.std()), 6),
            "min": round(float(v.min()), 6), "max": round(float(v.max()), 6),
            "P5": round(p5, 6), "P50": round(p50, 6), "P95": round(p95, 6)}


def _walk_summary(obs_rows: np.ndarray) -> dict:
    if len(obs_rows) == 0:
        return {"n": 0, "walk_sum_over_121_mean": None, "sw_band_mean": None}
    walk = obs_rows[:, WALK_LO:WALK_HI]
    sw = walk[:, list(SW_BAND_CELLS)]
    return {"n": int(len(obs_rows)),
            "walk_sum_over_121_mean": round(float(walk.sum(axis=1).mean()), 3),
            "sw_band_mean": round(float(sw.mean()), 4)}


def replay_archive(worker_npz: str | pathlib.Path,
                   manager_npz: str | pathlib.Path,
                   archive_path: str | pathlib.Path,
                   seeds: list[int] | None = None,
                   limit: int | None = None) -> dict:
    """Replay archive seeds and return the OBS_DRIFT report dict (pure, writes nothing)."""
    from diablogym import NumpyManager, OptionsEnv
    from diablogym.options_env import DIVE, FARM

    archive_payload = pathlib.Path(archive_path).read_bytes()
    doc = strict_json_loads(archive_payload)
    ref_rows = {int(r["seed"]): r for r in doc["rows"]}
    worker_sha = sha(worker_npz)
    manager_sha = sha(manager_npz)
    meta = doc.get("meta", {})
    if meta.get("worker", {}).get("sha256") != worker_sha:
        raise ValueError("archive worker sha differs from the replay worker npz; fidelity premise void")
    if meta.get("manager", {}).get("sha256") != manager_sha:
        raise ValueError("archive manager sha differs from the replay manager npz; fidelity premise void")
    replay_seeds = sorted(ref_rows) if seeds is None else list(seeds)
    if limit is not None:
        replay_seeds = replay_seeds[:limit]
    missing = [s for s in replay_seeds if s not in ref_rows]
    if missing:
        raise ValueError(f"requested replay seeds are not in the archive: {missing}")

    rt = runtime_identity(ROOT, bridge_binary_path(ROOT))
    runtime_five = {
        "bridge": rt["bridge"]["sha256"], "engine": rt["engine"]["sha256"],
        "game_data": rt["content"]["game_data"]["sha256"],
        "assets": rt["content"]["assets"]["sha256"],
        "protocol": rt["python_protocol"]["sha256"]}

    mgr = NumpyManager(str(manager_npz))
    net = NumpyManager(str(worker_npz))
    net.require_worker_contract()
    workers = {FARM: net.worker_callback()}
    env = OptionsEnv(max_steps=3000, workers=workers)
    fidelity: dict[str, object] = {}
    per_seed: dict[str, dict] = {}
    end_obs_all: list[np.ndarray] = []       # window-end observations (incl. terminal)
    d_decision_obs: list[np.ndarray] = []    # D-window decision observations (RB.10 D-window side)
    try:
        for seed in replay_seeds:
            obs, _ = env.reset(seed=seed)
            done = trunc = False
            R = 0.0
            farm = {"n": 0, "tau": 0, "descend": 0}
            allw = {"n": 0, "beats": 0, "overrides": 0, "cap": 0}
            seq = ""
            farm_taus: list[int] = []
            d_windows = 0
            end_obs_seed: list[np.ndarray] = []
            while not (done or trunc):
                m = env.action_masks()
                decision_obs = np.asarray(obs, dtype=np.float32).copy()
                opt = mgr.choose(obs, m)
                obs, r, done, trunc, info = env.step(opt)
                R += float(r)
                oe = info["option_extra"]
                end_obs_seed.append(np.asarray(obs, dtype=np.float32).copy())
                allw["n"] += 1
                allw["beats"] += oe["beats"]
                allw["overrides"] += oe["overrides"]
                allw["cap"] += oe["reason"] == "cap"
                if oe["opt"] == FARM:
                    farm["n"] += 1
                    farm["tau"] += oe["tau"]
                    farm["descend"] += oe["reason"] == "descend"
                    farm_taus.append(int(oe["tau"]))
                if oe["opt"] == DIVE:
                    d_windows += 1
                    d_decision_obs.append(decision_obs)
                seq = oe["mode_seq"]
            raw = env.env._raw
            row = {"seed": seed, "ret": round(R, 2),
                   "depth": raw["dungeon_level"],
                   "died": bool(raw.get("dead")),
                   "kills": env.env._ep_kills, "farm_n": farm["n"],
                   "farm_tau_mean": round(farm["tau"] / max(1, farm["n"]), 1),
                   "farm_tau_sum": farm["tau"],
                   "farm_descend": farm["descend"], "windows": allw["n"],
                   "beats": allw["beats"], "overrides": allw["overrides"],
                   "cap": allw["cap"], "mode_seq": seq}
            ok = all(row[f] == ref_rows[seed][f] for f in ROW_FIELDS)
            fidelity[str(seed)] = "bit-identical" if ok else {
                "MISMATCH": {f: [row[f], ref_rows[seed][f]]
                             for f in ROW_FIELDS
                             if row[f] != ref_rows[seed][f]}}
            end_obs_all.extend(end_obs_seed)
            tau_median = (float(np.median(farm_taus)) if farm_taus else None)
            per_seed[str(seed)] = {
                "ret": row["ret"], "depth": row["depth"], "died": row["died"],
                "d_windows": d_windows, "farm_taus": farm_taus,
                "farm_tau_median": tau_median,
                "tau_floor_distance": (None if tau_median is None
                                       else round(tau_median - 25.0, 1)),
                "walk_end": _walk_summary(np.asarray(end_obs_seed)
                                          if end_obs_seed else np.empty((0, 303))),
            }
            print(f"  seed {seed}: ret {R:.1f} D-windows {d_windows} "
                  f"reconcile {'OK' if ok else 'MISMATCH!'}", flush=True)
    finally:
        env.close()
        workers.clear()

    end_arr = (np.stack(end_obs_all) if end_obs_all
               else np.empty((0, 295 + 8), dtype=np.float32))
    d_arr = (np.stack(d_decision_obs) if d_decision_obs
             else np.empty((0, 295 + 8), dtype=np.float32))
    appended8 = {name: _stats(end_arr[:, 295 + i]) if len(end_arr) else None
                 for i, name in enumerate(APPENDED8)}
    report = {
        "probe": "b1-obsdrift",
        "prereg": "PREREG-B1 E4 (schema pinned; closed set of statistics)",
        "archive": str(archive_path),
        "archive_sha256": hashlib.sha256(archive_payload).hexdigest(),
        "worker_sha256": worker_sha,
        "manager_sha256": manager_sha,
        "runtime_five": runtime_five,
        "seeds": replay_seeds,
        "fidelity": fidelity,
        "fidelity_ok": all(v == "bit-identical" for v in fidelity.values()),
        "sw_band_cells": list(SW_BAND_CELLS),
        "appended8_window_end": appended8,
        "walk_window_end": _walk_summary(end_arr),
        "walk_d_decision": _walk_summary(d_arr),
        "per_seed": per_seed,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True, help="worker npz path")
    ap.add_argument("--manager", default=str(H_NPZ), help="manager npz (default H)")
    ap.add_argument("--archive", required=True, help="evaluation archive JSON (archive first, then replay)")
    ap.add_argument("--out", required=True, help="report JSON output path")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seed subset (default = all archive seeds)")
    ap.add_argument("--smoke", action="store_true", help="replay only the first seed")
    args = ap.parse_args()
    seeds = ([int(x) for x in args.seeds.split(",") if x.strip()]
             if args.seeds else None)
    report = replay_archive(args.worker, args.manager, args.archive, seeds,
                            limit=1 if args.smoke else None)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    bad = [k for k, v in report["fidelity"].items() if v != "bit-identical"]
    print(f"replay fidelity: {len(report['fidelity']) - len(bad)}/{len(report['fidelity'])}"
          f" bit-identical; mismatches {bad if bad else 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
