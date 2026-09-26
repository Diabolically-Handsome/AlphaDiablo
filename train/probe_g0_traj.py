"""G0-identity bit-level trajectory probe (PREREG-v32 W-G0; for both baseline and replay).

Rig = the full-stack worker path: WorkerWindowEnv (H manager npz) drives the windows, and the king npz worker
runs argmax (via _worker_masks) beat by beat: exactly the mask path E1 touches. seeds
7000-7015, one episode each; hash = sha256(per beat [action, wage hex, term, trunc] +
per-window boundaries + terminal summary); the wage identity W == R - bonus is asserted per window, plus a per-episode ledger assertion
(fast-forward windows are booked too; log_windows keeps the full window order). The baseline file is checked in with the frozen commit
(docs/assets, outside gitignore).

Usage:
  baseline (before the E1 work): .venv/bin/python train/probe_g0_traj.py baseline
  replay (after the E1 work): .venv/bin/python train/probe_g0_traj.py replay
Replay mode compares baseline.json seed by seed; any bit that differs -> exit code 1 (G0 fails).
"""
import hashlib
import json
import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym.worker_env import NumpyManager, WorkerWindowEnv  # noqa: E402

SEEDS = list(range(7000, 7016))
H_NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
KING_NPZ = ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz"
THRONE_NPZ = ROOT / "train" / "models" / "v24-worker-leg7" / "policy.npz"
OUT = ROOT / "docs" / "assets" / "g0v32_traj_baseline.json"
OUT_THRONE = ROOT / "docs" / "assets" / "g0v32_traj_baseline_throne.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
DESCEND_UNIT = 8.0


def f2hex(x: float) -> str:
    return struct.pack("<d", float(x)).hex()


def episode_digest(seed: int, npz=None) -> dict:
    net = NumpyManager(str(npz or KING_NPZ))
    net.require_worker_contract()
    env = WorkerWindowEnv(str(H_NPZ), max_steps=3000, rng_seed=0,
                          seed_scope="replay",
                          log_windows=True)
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
        print(f"G0-identity FAIL ({'throne' if throne else 'king'}): mismatched seeds {bad}")
        return 1
    print(f"G0-identity PASS ({'throne' if throne else 'king'}): {len(rows)} seeds bit-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
