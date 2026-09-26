"""B1-E8: machine extractor for the W-C canary scoring seed set (PREREG-B1 W-C/E8).

Extracts every depth>=2 seed from the given evaluation archive (the extraction rule follows the run_v32_sovereign
depth2_count precedent verbatim: row["depth"] >= 2), unioned with the healthy controls {7003, 7011} (P5 seeds without
spikes, negative controls). The driver pins the result with a CANARY_SET event (carrying n_D and
the per-seed list); this extractor is committed with the freeze commit -- otherwise "the archive rows decide" could not be re-verified.

Usage:
  .venv/bin/python train/extract_canary_set.py \
      [train/runs/eval-assembled/v32-ref-launch.json] [--controls 7003,7011]
Output: JSON to stdout (archive/archive_sha256/depth2_seeds/n_D/controls/C).
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

DEFAULT_ARCHIVE = ROOT / "train" / "runs" / "eval-assembled" / "v32-ref-launch.json"
DEFAULT_CONTROLS = (7003, 7011)   # PREREG-B1 W-C: P5 spike-free healthy controls, negative controls


def depth2_seeds(rows: list[dict]) -> list[int]:
    """depth>=2 extraction (per the run_v32_sovereign.depth2_count precedent, same row rule)."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("archive rows missing/empty")
    seeds = sorted(int(r["seed"]) for r in rows if r["depth"] >= 2)
    if len(seeds) != len(set(seeds)):
        raise ValueError("archive rows contain duplicate seed")
    return seeds


def extract(archive_path: str | pathlib.Path,
            controls: tuple[int, ...] = DEFAULT_CONTROLS) -> dict:
    from eval_contract import strict_json_loads

    p = pathlib.Path(archive_path)
    payload = p.read_bytes()
    doc = strict_json_loads(payload)
    d2 = depth2_seeds(doc["rows"])
    archive_seed_set = {int(r["seed"]) for r in doc["rows"]}
    bad_controls = [s for s in controls if s not in archive_seed_set]
    if bad_controls:
        raise ValueError(f"healthy control seeds not in the archive seed set: {bad_controls}")
    overlap = sorted(set(d2) & set(controls))
    if overlap:
        raise ValueError(f"healthy controls overlap the depth>=2 set, negative control void: {overlap}")
    return {
        "archive": str(p),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "depth2_seeds": d2,
        "n_D": len(d2),
        "controls": sorted(controls),
        "C": sorted(set(d2) | set(controls)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", nargs="?", default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--controls", default=",".join(map(str, DEFAULT_CONTROLS)),
                    help="healthy control seeds (comma-separated; empty string = no controls unioned)")
    args = ap.parse_args()
    controls = tuple(int(x) for x in args.controls.split(",") if x.strip())
    print(json.dumps(extract(args.archive, controls), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
