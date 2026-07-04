"""对比 ep1/ep2 城镇同一批格子的 dPiece 与 walkable:定位是地块还是属性表坏了。"""
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
# 下到地牢 1 层再回来重开,复现污染路径
stairs = [t for t in obs["triggers"] if t["msg"] == 0][0]
obs, _ = walk_to_target(stairs["x"], stairs["y"])
assert obs["dungeon_level"] == 1, "先决条件:ep1 要成功下到 L1"
print(f"(ep1 已下到 L1,怪物 {len(obs['monsters'])} 只;现在重开 ep2)\n")
bridge.reset(seed=1001)
ep2 = sample()

print(f"{'tile':>10} | {'ep1 piece/walk':>15} | {'ep2 piece/walk':>15} | 一致?")
for t in TILES:
    p1, w1 = ep1[t]
    p2, w2 = ep2[t]
    mark = "✓" if (p1, w1) == (p2, w2) else "✗✗✗"
    print(f"{str(t):>10} | {p1:>10}/{str(w1):>4} | {p2:>10}/{str(w2):>4} | {mark}")

same_piece = all(ep1[t][0] == ep2[t][0] for t in TILES)
same_walk = all(ep1[t][1] == ep2[t][1] for t in TILES)
print(f"\ndPiece 全部一致: {same_piece}   walkable 全部一致: {same_walk}")
if same_piece and not same_walk:
    print("→ 地块相同但通行性不同:坏的是『地块属性表』(SOLData / TileProperties)")
elif not same_piece:
    print("→ 地块本身不同:坏的是『城镇地图创建』(dPiece/dungeon 数据)")
