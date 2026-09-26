"""Compare dPiece and walkable of the same town tiles in ep1/ep2: is it the tile map or the property table that broke?"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from diablogym import DiabloGymEnv, bridge

DiabloGymEnv()

TILES = [(75, 68), (73, 66), (71, 64), (70, 63), (69, 62), (74, 61), (77, 71), (72, 69)]


def sample():
    return {(x, y): (bridge.probe_tile(x, y)["piece"], bridge.probe_tile(x, y)["walkable"]) for x, y in TILES}


from descend_seed_test import walk_to_target

obs = bridge.reset(seed=1001)
ep1 = sample()
# Go down to dungeon level 1 and come back to restart, reproducing the contamination path
stairs = [t for t in obs["triggers"] if t["msg"] == bridge.WM_DIABNEXTLVL][0]
obs, _ = walk_to_target(stairs["x"], stairs["y"])
assert obs["dungeon_level"] == 1, "precondition: ep1 must reach L1"
print(f"(ep1 reached L1 with {len(obs['monsters'])} monsters; now restarting ep2)\n")
bridge.reset(seed=1001)
ep2 = sample()

print(f"{'tile':>10} | {'ep1 piece/walk':>15} | {'ep2 piece/walk':>15} | same?")
for t in TILES:
    p1, w1 = ep1[t]
    p2, w2 = ep2[t]
    mark = "✓" if (p1, w1) == (p2, w2) else "✗✗✗"
    print(f"{str(t):>10} | {p1:>10}/{str(w1):>4} | {p2:>10}/{str(w2):>4} | {mark}")

same_piece = all(ep1[t][0] == ep2[t][0] for t in TILES)
same_walk = all(ep1[t][1] == ep2[t][1] for t in TILES)
print(f"\ndPiece all equal: {same_piece}   walkable all equal: {same_walk}")
if same_piece and not same_walk:
    print("→ same pieces but different passability: the broken part is the tile property table (SOLData / TileProperties)")
elif not same_piece:
    print("→ the pieces themselves differ: the broken part is town map creation (dPiece/dungeon data)")
if not (same_piece and same_walk):
    raise AssertionError(f"town pieces/passability drifted on reset after entering the dungeon: {ep1=}, {ep2=}")
