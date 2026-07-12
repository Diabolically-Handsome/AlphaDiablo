"""对比 ep1 / ep2 的资产可达性:MPQ 挂载是否在 EndGame 拆解后失效。"""
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
        print(f"    {'OK ' if r['ok'] else 'FAIL'} {a}  ({r['size']} 字节)")
    return result


bridge.reset(seed=1001)
ep1 = probe("ep1 城镇 reset 后")
bridge.end_game()
ended = probe("ep1 end_game 后")
bridge.reset(seed=1001)
ep2 = probe("ep2 城镇 reset 后")

if not (ep1 == ended == ep2):
    raise AssertionError(f"资产可达性跨 end_game/reset 漂移: {ep1=}, {ended=}, {ep2=}")
for required in ("levels\\towndata\\town.sol", "levels\\towndata\\town.cel",
                 "levels\\l1data\\l1.sol"):
    if not ep1[required][0] or ep1[required][1] <= 0:
        raise AssertionError(f"必需资产不可读: {required} -> {ep1[required]}")
