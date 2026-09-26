"""Dungeon descent + seed divergence test.

Flow: reset(seed) → path through town to the cathedral entrance (25,29) → trigger WM_DIABNEXTLVL →
arrive on dungeon level 1 → snapshot (player entry position + all monsters).
Assertions: two snapshots with the same seed are identical; snapshots with different seeds differ.

Usage (repository root):  .venv/bin/python tests/descend_seed_test.py
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import numpy as np

import diablogym
from diablogym import DiabloGymEnv, bridge


def walk_to_target(tx, ty, max_ticks=6000, hop=8, ticks_per_hop=16, trace=False):
    """Walk toward the target tile in segments (one path search reaches about 25 steps; far targets need relays).

    Returns (final observation, ticks used). Returns early once the dungeon is entered.
    """
    rng = np.random.default_rng(7)  # perturbation when stuck; a fixed seed keeps the whole run deterministic
    obs = bridge.observe()
    used = 0
    last_pos = (obs["player_x"], obs["player_y"])
    stuck = 0
    while used < max_ticks:
        if obs["dungeon_level"] != 0:
            return obs, used  # already down
        px, py = obs["player_x"], obs["player_y"]
        dx, dy = tx - px, ty - py
        dist = max(abs(dx), abs(dy))
        if dist == 0:
            # Standing on the target tile: the trigger fires only at PM_STAND, and the level change needs another event-pump round
            for i in range(30):
                obs = bridge.step(ticks=8)
                used += 8
                if i < 3:
                    print(f"    [debug] tick+{(i+1)*8} after stepping on the stairs: mode={obs['player_mode']} "
                          f"pos=({obs['player_x']},{obs['player_y']}) level={obs['dungeon_level']} "
                          f"triggers={obs['triggers']}")
                if obs["dungeon_level"] != 0:
                    break
            return obs, used
        frac = min(hop, dist) / dist
        wx, wy = px + round(dx * frac), py + round(dy * frac)
        bridge.act_walk(wx, wy)
        obs = bridge.step(ticks=ticks_per_hop)
        used += ticks_per_hop
        if trace and used <= 20 * ticks_per_hop:
            print(f"    [trace] command→({wx},{wy})  actual ({px},{py})→({obs['player_x']},{obs['player_y']}) "
                  f"mode={obs['player_mode']} path0={obs['walkpath0']}")
        pos = (obs["player_x"], obs["player_y"])
        if pos == last_pos:
            stuck += 1
            # Stuck in place (wall/path failure): perturb randomly, then continue
            jx, jy = int(rng.integers(-6, 7)), int(rng.integers(-6, 7))
            bridge.act_walk(px + jx, py + jy)
            obs = bridge.step(ticks=ticks_per_hop)
            used += ticks_per_hop
        else:
            stuck = 0
        last_pos = (obs["player_x"], obs["player_y"])
        if stuck > 20:
            break
    return obs, used


def descend(seed, trace=False):
    """Start a new game and walk down to dungeon level 1; returns (L1 snapshot, seconds, ticks)."""
    t0 = time.time()
    obs = bridge.reset(seed=seed)
    assert obs["dungeon_level"] == 0, "the game should start in town"
    if trace:
        dump_grid("town after reset", obs["player_x"], obs["player_y"])

    stairs = [t for t in obs["triggers"] if t["msg"] == diablogym.bridge.WM_DIABNEXTLVL]
    assert stairs, f"no down stairs found in the town observation: {obs['triggers']}"
    sx, sy = stairs[0]["x"], stairs[0]["y"]

    obs, used = walk_to_target(sx, sy, trace=trace)
    assert obs["dungeon_level"] == 1, (
        f"failed to enter dungeon level 1 (final position ({obs['player_x']},{obs['player_y']}), "
        f"level {obs['dungeon_level']}, used {used} ticks)"
    )
    snapshot = {
        "entry": (obs["player_x"], obs["player_y"]),
        "level_type": obs["level_type"],
        "monsters": tuple(sorted((m["type"], m["x"], m["y"], m["max_hp"]) for m in obs["monsters"])),
    }
    return snapshot, time.time() - t0, used


def dump_grid(tag, cx, cy, r=7):
    """Print the passability map centred on (cx,cy): . walkable  # wall  M monster  P player  O object"""
    print(f"    [grid] {tag} centred on ({cx},{cy}):")
    for y in range(cy - r, cy + r + 1):
        row = ""
        for x in range(cx - r, cx + r + 1):
            t = bridge.probe_tile(x, y)
            if t["monster"] != 0:
                row += "M"
            elif t["player"] != 0:
                row += "P"
            elif t["object"] != 0:
                row += "O"
            elif not t["walkable"]:
                row += "#"
            else:
                row += "."
        print(f"      {row}")


def brief(snap):
    mons = snap["monsters"]
    return (f"entry {snap['entry']}, {len(mons)} monsters"
            f" (first 3: {[m[:3] for m in mons[:3]]})")


def main():
    print("== Dungeon descent + seed divergence test ==")
    print(f"The town stairs position is read from the observation triggers (expected (25,29))\n")
    DiabloGymEnv()  # one-time engine initialization (with the default asset/data/save paths)

    snap_a1, dt1, ticks1 = descend(seed=1001)
    print(f"seed=1001 run 1: {brief(snap_a1)}  [{dt1:.2f}s, {ticks1} tick]")
    snap_a2, dt2, _ = descend(seed=1001)
    print(f"seed=1001 run 2: {brief(snap_a2)}  [{dt2:.2f}s]")
    snap_b, dt3, _ = descend(seed=2002)
    print(f"seed=2002        : {brief(snap_b)}  [{dt3:.2f}s]")

    assert snap_a1 == snap_a2, (
        "two dungeon snapshots with the same seed differ!\n"
        f"A1: {snap_a1}\nA2: {snap_a2}"
    )
    print("\nPASS: same seed (1001) gives identical dungeon level 1 twice; determinism holds")

    assert snap_a1 != snap_b, "different seeds gave the same dungeon level 1; seed injection is broken!"
    diff_parts = []
    if snap_a1["entry"] != snap_b["entry"]:
        diff_parts.append("entry position")
    if snap_a1["monsters"] != snap_b["monsters"]:
        diff_parts.append(f"monster layout ({len(snap_a1['monsters'])} vs {len(snap_b['monsters'])} monsters)")
    print(f"PASS: different seeds (1001 vs 2002) diverge in the dungeon in: {', '.join(diff_parts)}")

    print("\n== All passed: pathing, level change, seed divergence OK ==")


if __name__ == "__main__":
    main()
