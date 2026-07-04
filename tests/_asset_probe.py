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
    for a in ASSETS:
        r = bridge.probe_asset(a)
        print(f"    {'OK ' if r['ok'] else 'FAIL'} {a}  ({r['size']} 字节)")


bridge.reset(seed=1001)
probe("ep1 城镇 reset 后")
bridge.end_game()
probe("ep1 end_game 后")
bridge.reset(seed=1001)
probe("ep2 城镇 reset 后")
