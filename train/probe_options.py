"""v22 pre-launch probe: G1b mechanical fidelity / G2 vocabulary sufficiency / G3 evaluation-range reference rows / G4 budget calibration.

Usage: .venv/bin/python train/probe_options.py --oracle /path/to/oracle_mountain.json
By default only the diagnostic JSON is written; the leaderboard is updated only with an explicit --write-board after every gate passes.

Gates (pre-registered):
  G1b wrapper-rush (always DIVE) vs the oracle rush arm, per seed |delta| <= max(5%, 1.0)
  G2  teacher (drained flag or clvl>=dlvl+2 -> DIVE) mean >=36 and paired wins over wrapper-retire >=24/32
  G3  the three reference arms' 9000-9031 scores are written to the hierarchy leaderboard of the current protocol version
  G4  measured tau-bar and throughput -> fix the numbers of the dual-currency stopping rule
"""
import argparse
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))

from eval_contract import (OutputReservationError, reserve_output, sha256_file,
                           strict_json_loads)
from evaluate import (atomic_write_text, contract_sha256,
                      ensure_leaderboard_compatible, scripted_leaderboard_row,
                      require_fresh_native_runtime,
                      upsert_leaderboard_rows, validated_episode_extra,
                      verify_loaded_native_runtime, verify_standalone_contract,
                      versioned_row_key)
from evaluate_options import (LB, LB_LOCK, LEADERBOARD_HEADER,
                              hierarchy_contract)

PROBE_SEEDS = list(range(7000, 7032))
EVAL_SEEDS = list(range(9000, 9032))
# diablogym must not be imported before the runtime contract is frozen: once the native extension/engine is mapped,
# hashing only the on-disk path afterwards could record "old bytes already loaded" as "new bytes on disk". The numbers are
# part of the evaluation protocol; after the real import the public constants of options_env are checked one by one.
FARM, DIVE = 0, 1
OptionsEnv = None


def _masked_option_or_first_legal(requested, mask) -> int:
    """Keep a legal proposal or deterministically choose the first legal option."""
    valid = np.asarray(mask, dtype=bool)
    if valid.shape != (3,):
        raise ValueError(
            f"probe option action mask shape is malformed: {valid.shape} != (3,)")
    legal = np.flatnonzero(valid)
    if len(legal) == 0:
        raise ValueError("probe option action mask is all False")
    if (
        isinstance(requested, (int, np.integer))
        and not isinstance(requested, (bool, np.bool_))
    ):
        candidate = int(requested)
        if 0 <= candidate < 3 and bool(valid[candidate]):
            return candidate
    return int(legal[0])


def _options_env_class():
    if OptionsEnv is not None:  # Injected explicitly by unit tests; None in production.
        return OptionsEnv
    from diablogym import OptionsEnv as env_class
    from diablogym.options_env import DIVE as actual_dive, FARM as actual_farm

    if (actual_farm, actual_dive) != (FARM, DIVE):
        raise RuntimeError(
            "OptionsEnv option indices disagree with the probe protocol: "
            f"actual={(actual_farm, actual_dive)}, expected={(FARM, DIVE)}")
    return env_class


def run_policy(env, choose, seed):
    obs, _ = env.reset(seed=seed)
    done = trunc = False
    R, taus, info = 0.0, [], {}
    while not (done or trunc):
        m = env.action_masks()
        opt = _masked_option_or_first_legal(choose(env, m), m)
        obs, r, done, trunc, info = env.step(opt)
        R += r
        taus.append(info["option_extra"]["tau"])
    ex = validated_episode_extra(info, seed)
    if not math.isfinite(float(R)):
        raise RuntimeError(f"seed {seed} cumulative return contains NaN/Inf")
    oe = info.get("option_extra")
    if not isinstance(oe, dict) or "mode_seq" not in oe:
        raise RuntimeError(f"seed {seed} lacks a complete option_extra")
    return {"ret": round(R, 2), "depth": ex["depth"],
            "died": ex["died"], "kills": ex["kills"],
            "decisions": len(taus), "tau_mean": round(sum(taus) / max(1, len(taus)), 1),
            "tau_sum": sum(taus),
            "mode_seq": oe["mode_seq"]}


POLICIES = {
    "wrapper-retire": lambda env, m: FARM,
    "wrapper-rush": lambda env, m: DIVE,
    "teacher": lambda env, m: DIVE if (env.exhausted or
                                       env.env._raw["char_level"] >= env.env._raw["dungeon_level"] + 2)
                              else FARM,
}


def validate_oracle_rush(oracle) -> dict[int, float]:
    """Freeze the exact seed->return mapping G1b needs; duplicate/extra/non-finite values are all rejected."""
    try:
        entries = oracle["arms"]["rush"]
    except (KeyError, TypeError) as exc:
        raise ValueError("oracle lacks arms.rush") from exc
    if not isinstance(entries, list):
        raise ValueError("oracle arms.rush must be a list")
    by_seed: dict[int, float] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "error" in entry:
            continue
        try:
            seed = entry["seed"]
            value = entry["snaps"]["3000"]["ret"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"oracle rush[{index}] structure is malformed") from exc
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"oracle rush[{index}].seed is not an integer")
        if seed in by_seed:
            raise ValueError(f"oracle rush contains a duplicate seed: {seed}")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"oracle seed {seed} ret is not numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"oracle seed {seed} ret is not finite")
        by_seed[seed] = value
    expected = set(PROBE_SEEDS)
    if set(by_seed) != expected:
        missing = sorted(expected - set(by_seed))
        extra = sorted(set(by_seed) - expected)
        raise ValueError(f"oracle rush seed set is malformed: missing={missing}, extra={extra}")
    return by_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", type=pathlib.Path, required=True,
                    help="path to oracle_mountain.json (no longer assumes a fixed /private/tmp location)")
    ap.add_argument("--output", type=pathlib.Path,
                    default=ROOT / "train" / "runs" / "probes" / "probe_v22.json")
    ap.add_argument("--write-board", action="store_true",
                    help="update the 9000-range reference rows after every gate PASSes; by default only the diagnostic JSON is produced")
    args = ap.parse_args()
    require_fresh_native_runtime("probe_options.py")
    output_path = args.output.resolve()
    if output_path == args.oracle.resolve():
        ap.error("--output cannot overwrite --oracle")
    if output_path in {LB.resolve(), LB_LOCK.resolve()}:
        ap.error("--output cannot overwrite the current protocol leaderboard or its lock file")
    try:
        oracle_payload = args.oracle.read_bytes()
        oracle = strict_json_loads(oracle_payload)
        oracle_rush = validate_oracle_rush(oracle)
        oracle_sha256 = hashlib.sha256(oracle_payload).hexdigest()
    except (OSError, ValueError) as e:
        ap.error(f"cannot read oracle: {e}")

    contract = hierarchy_contract()
    if args.write_board:
        ensure_leaderboard_compatible(
            LB, contract, initial_text=LEADERBOARD_HEADER)
    try:
        # The probe detail is also publication evidence: each target may be created only once, so an old archive cannot be silently
        # replaced; to re-run, explicitly choose a new --output or archive the old file first.
        with reserve_output(args.output):
            env = _options_env_class()(max_steps=3000)
            verify_loaded_native_runtime(contract)
            try:
                out, g1b, g2 = _collect_probe(
                    oracle_rush, env, args.oracle.resolve(), oracle_sha256,
                    contract)
            finally:
                # A close failure must also happen before any official JSON/leaderboard publication.
                env.close()
            if (out.get("meta", {}).get("contract") != contract
                    or out.get("meta", {}).get("contract_sha256")
                    != contract_sha256(contract)):
                raise RuntimeError("probe result is not bound to the pre-launch standalone contract")
            verify_standalone_contract(contract)
            return _publish_probe(args, out, g1b, g2)
    except OutputReservationError as exc:
        ap.error(str(exc))


def _collect_probe(oracle_rush: dict[int, float], env, oracle_path: pathlib.Path,
                   oracle_sha256: str, contract: dict) -> tuple[dict, bool, bool]:
    out = {
        "meta": {
            "oracle_path": str(oracle_path), "oracle_sha256": oracle_sha256,
            "contract": contract, "contract_sha256": contract_sha256(contract),
        },
        "probe": {}, "eval_refs": {},
    }
    t_wall0 = time.time()

    # ---- probe range 7000-7031 ----
    for name, pol in POLICIES.items():
        eps = []
        for seed in PROBE_SEEDS:
            eps.append({"seed": seed, **run_policy(env, pol, seed)})
        out["probe"][name] = eps
        rs = [e["ret"] for e in eps]
        print(f"[probe] {name}: mean {sum(rs)/32:.1f} med {statistics.median(rs):.1f} "
              f"died {sum(e['died'] for e in eps)}/32 "
              f"depth_med {statistics.median(e['depth'] for e in eps)} "
              f"decisions_med {statistics.median(e['decisions'] for e in eps)}", flush=True)

    # G1b: wrapper-rush vs the oracle rush (3000 snapshot)
    fails = []
    for e in out["probe"]["wrapper-rush"]:
        ref = oracle_rush[e["seed"]]
        if abs(e["ret"] - ref) > max(0.05 * abs(ref), 1.0):
            fails.append((e["seed"], e["ret"], ref))
    g1b = len(fails) == 0
    print(f"G1b {'PASS' if g1b else 'FAIL'}: wrapper-rush vs oracle rush per-seed deviations over the limit {len(fails)}/32 "
          + (f"first case {fails[0]}" if fails else ""), flush=True)

    # G2: teacher sufficiency
    t_rets = [e["ret"] for e in out["probe"]["teacher"]]
    r_rets = [e["ret"] for e in out["probe"]["wrapper-retire"]]
    t_mean = sum(t_rets) / 32
    retire_by_seed = {e["seed"]: e["ret"] for e in out["probe"]["wrapper-retire"]}
    wins = sum(e["ret"] > retire_by_seed[e["seed"]] for e in out["probe"]["teacher"])
    g2 = t_mean >= 36 and wins >= 24
    grey = 34 <= t_mean < 36
    print(f"G2 {'PASS' if g2 else ('GREY' if grey else 'FAIL')}: teacher mean {t_mean:.1f} (line 36, grey band [34,36)) "
          f"paired wins over retire {wins}/32 (line 24)", flush=True)

    # ---- G3: evaluation-range reference rows 9000-9031 ----
    for name, pol in POLICIES.items():
        eps = [{"seed": s, **run_policy(env, pol, s)} for s in EVAL_SEEDS]
        out["eval_refs"][name] = eps
        rs = [e["ret"] for e in eps]
        print(f"[eval] {name}: mean {sum(rs)/32:.1f} died {sum(e['died'] for e in eps)}/32 "
              f"depth_med {statistics.median(e['depth'] for e in eps)}", flush=True)

    # G4: budget calibration
    wall = time.time() - t_wall0
    all_probe = [e for eps in out["probe"].values() for e in eps]
    all_eps = all_probe + [e for eps in out["eval_refs"].values() for e in eps]
    tau_bar = sum(e["tau_sum"] for e in all_probe) / max(1, sum(e["decisions"] for e in all_probe))
    total_micro = sum(e["tau_sum"] for e in all_eps)
    micro_per_s = total_micro / wall
    print(f"G4: tau-bar ~{tau_bar:.0f} micro-beats/option, throughput ~{micro_per_s:.0f} micro/s (single env), "
          f"40k manager steps ~ {40_000 * tau_bar / 1e6:.1f}M micro-steps; "
          f"4-env estimated wall clock ~ {40_000 * tau_bar / (micro_per_s * 2.5) / 60:.0f} minutes", flush=True)

    return out, g1b, g2


def _publish_probe(args, out: dict, g1b: bool, g2: bool) -> int:
    try:
        contract = out["meta"]["contract"]
        oracle_path = out["meta"]["oracle_path"]
        expected_oracle_sha = out["meta"]["oracle_sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError("probe output lacks complete provenance") from exc
    verify_standalone_contract(contract)
    try:
        current_oracle_sha = sha256_file(oracle_path)
    except OSError as exc:
        raise RuntimeError("probe oracle unreadable before publishing") from exc
    if current_oracle_sha != expected_oracle_sha:
        raise RuntimeError("probe oracle changed before the diagnostic JSON was published")
    payload = json.dumps(out, default=float, allow_nan=False, sort_keys=True)
    if args.output.exists():
        raise OutputReservationError(f"probe archive already exists, refusing to overwrite: {args.output}")
    atomic_write_text(args.output, payload)
    print(f"probe detail saved to {args.output}", flush=True)

    if not (g1b and g2):
        print("not every gate PASSed: refusing to write the leaderboard", flush=True)
        return 1

    # Evaluation-range reference rows go into the new table
    if not args.write_board:
        print("G3: --write-board not given, leaderboard unchanged", flush=True)
        print("GATES: G1b=PASS G2=PASS", flush=True)
        return 0
    contract = out["meta"]["contract"]
    oracle_sha256 = out["meta"]["oracle_sha256"]
    rows = {}
    for name in POLICIES:
        eps = out["eval_refs"][name]
        rs = sorted(e["ret"] for e in eps)
        label = f"{name} (scripted ref)"
        result_sha256 = hashlib.sha256(json.dumps(
            eps, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
        scripted_identity = hashlib.sha256(
            f"{oracle_sha256}\0{result_sha256}".encode("ascii")).hexdigest()
        key = versioned_row_key(label, scripted_identity)
        visible = (f"| {key} | {sum(rs)/32:.1f} | "
                   f"{statistics.median(rs):.1f} | "
                   f"{sum(e['died'] for e in eps)}/32 | "
                   f"{statistics.median(e['depth'] for e in eps)} | "
                   "G3 reference |")
        rows[key] = scripted_leaderboard_row(
            visible, row_key=key, contract=contract, policy=name,
            oracle_path=out["meta"]["oracle_path"],
            oracle_sha256=oracle_sha256, result_sha256=result_sha256)
    upsert_leaderboard_rows(
        LB, rows, contract=contract, initial_text=LEADERBOARD_HEADER,
        lock_path=LB_LOCK)
    print(f"G3: reference rows written to {LB.name}", flush=True)
    print("GATES: G1b=PASS G2=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
