"""G0-identity bit-level trajectory probe, skip_dry=True endpoint (v33 content case E0; for both baseline and replay).

The rig has exactly the same shape as train/probe_g0_traj.py (full-stack worker path: WorkerWindowEnv +
king/throne npz argmax beat by beat); **the only difference = the WorkerWindowEnv constructor explicitly passes
skip_dry=True** (the endpoint where the script's inner loop runs dry-level revisit windows, the p==1.0 side of G0-1).
The skip_dry=False endpoint is still covered by the original script + the original baseline files (g0v32_traj_baseline*.json);
this script touches the originals not at all (E0 hard rule: baseline before the change, never modify existing files).

Constants (SEEDS/H_NPZ/KING_NPZ/THRONE_NPZ/f2hex) are imported directly from the original script to prevent mirror drift;
episode_digest mirrors the original text, adding only skip_dry=True to the env constructor line.

Usage:
  baseline (before the E work): .venv/bin/python train/probe_g0_traj_skip_dry.py baseline [throne]
  replay (after the E work): .venv/bin/python train/probe_g0_traj_skip_dry.py replay [throne]
Replay mode compares the baseline file seed by seed; any bit that differs -> exit code 1 (G0 fails).
"""
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from probe_g0_traj import (  # noqa: E402  single source of truth for the original script's constants
    H_NPZ, KING_NPZ, SEEDS, THRONE_NPZ, f2hex)
from diablogym.worker_env import NumpyManager, WorkerWindowEnv  # noqa: E402

OUT = ROOT / "docs" / "assets" / "g0v32_traj_baseline_skip_dry_true.json"
OUT_THRONE = ROOT / "docs" / "assets" / "g0v32_traj_baseline_throne_skip_dry_true.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def episode_digest(seed: int, npz=None) -> dict:
    net = NumpyManager(str(npz or KING_NPZ))
    net.require_worker_contract()
    env = WorkerWindowEnv(str(H_NPZ), max_steps=3000, rng_seed=0,
                          seed_scope="replay",
                          log_windows=True, skip_dry=True)   # <- the only difference from the original script
    h = hashlib.sha256()
    wages = 0.0
    steps = 0
    obs, _ = env.reset(seed=seed)
    while obs is not None:
        masks = env.oe._worker_masks()
        policy_obs = env.oe._worker_policy_observation(
            net.worker_observation_view)
        a = net.choose_worker(
            policy_obs,
            masks,
            observation_view=net.worker_observation_view,
        )
        obs2, w, term, trunc, info = env.step(a)
        h.update(f"{a},{f2hex(w)},{int(term)},{int(trunc)};".encode())
        wages += float(w)
        steps += 1
        if term or trunc:
            ex = info.get("option_extra")
            if ex is not None:
                assert abs(ex["W"] - (ex["R"] - ex["bonus"])) < 1e-6, (
                    f"wage identity broken: seed {seed} W={ex['W']} "
                    f"R={ex['R']} bonus={ex['bonus']}")
            h.update(b"|WIN|")
            obs = env.next_window()
        else:
            obs = obs2
    tot = env.window_log
    assert abs(sum(w["W"] for w in tot)
               - (sum(w["R"] for w in tot) - sum(w["bonus"] for w in tot))) < 1e-5, \
        f"per-episode wage identity broken: seed {seed}"
    raw = env.oe.env._raw
    final = (f"d{raw['dungeon_level']},dead{int(raw['dead'])},"
             f"hp{raw['hp']},xp{raw.get('experience', raw.get('xp', 0))}")
    h.update(final.encode())
    stats = dict(env.stats)
    env.close()
    return {"seed": seed, "sha": h.hexdigest(), "steps": steps,
            "wages": round(wages, 6), "final": final, "stats": stats}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    throne = len(sys.argv) > 2 and sys.argv[2] == "throne"
    npz = THRONE_NPZ if throne else KING_NPZ
    out = OUT_THRONE if throne else OUT
    rows = [episode_digest(s, npz) for s in SEEDS]
    for r in rows:
        print(f"  seed {r['seed']}: {r['sha'][:16]} steps {r['steps']} "
              f"wages {r['wages']}", flush=True)
    if mode == "baseline":
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"baseline saved to {out}")
        return 0
    base = {r["seed"]: r for r in json.loads(out.read_text())}
    bad = [r["seed"] for r in rows
           if r["sha"] != base[r["seed"]]["sha"]
           or r["steps"] != base[r["seed"]]["steps"]]
    if bad:
        print(f"G0-identity skip_dry=True FAIL ({'throne' if throne else 'king'}): mismatched seeds {bad}")
        return 1
    print(f"G0-identity skip_dry=True PASS ({'throne' if throne else 'king'}): {len(rows)} seeds bit-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
