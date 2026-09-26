"""Compare asset reachability in ep1 / ep2: does the MPQ mount break after EndGame teardown?"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from diablogym import DiabloGymEnv, bridge

DiabloGymEnv()

ASSETS = [
    "levels\\towndata\\town.sol",
    "levels\\towndata\\town.cel",
    "nlevels\\towndata\\town.cel",
    "levels\\l1data\\l1.sol",
]


def probe(tag):
    print(f"  {tag}:")
    result = {}
    for a in ASSETS:
        r = bridge.probe_asset(a)
        result[a] = (bool(r["ok"]), int(r["size"]))
        print(f"    {'OK ' if r['ok'] else 'FAIL'} {a}  ({r['size']} bytes)")
    return result


bridge.reset(seed=1001)
ep1 = probe("ep1 after town reset")
bridge.end_game()
ended = probe("ep1 after end_game")
bridge.reset(seed=1001)
ep2 = probe("ep2 after town reset")

if not (ep1 == ended == ep2):
    raise AssertionError(f"asset reachability drifted across end_game/reset: {ep1=}, {ended=}, {ep2=}")
for required in ("levels\\towndata\\town.sol", "levels\\towndata\\town.cel",
                 "levels\\l1data\\l1.sol"):
    if not ep1[required][0] or ep1[required][1] <= 0:
        raise AssertionError(f"required asset unreadable: {required} -> {ep1[required]}")
