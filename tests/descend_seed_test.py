"""下地牢 + 种子分化测试。

流程:reset(seed) → 城镇寻路走到教堂入口 (25,29) → 触发 WM_DIABNEXTLVL →
到达地牢 1 层 → 快照(玩家入口位置 + 全部怪物)。
断言:同种子两次快照完全一致;不同种子快照不同。

用法(仓库根目录):  .venv/bin/python tests/descend_seed_test.py
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import numpy as np

import diablogym
from diablogym import DiabloGymEnv, bridge


def walk_to_target(tx, ty, max_ticks=6000, hop=8, ticks_per_hop=16, trace=False):
    """分段寻路走向目标格(单次寻路上限约 25 步,远目标需要接力)。

    返回 (最终观测, 用掉的 tick 数)。中途进入地牢即提前返回。
    """
    rng = np.random.default_rng(7)  # 卡住时的扰动;固定种子保证全程确定性
    obs = bridge.observe()
    used = 0
    last_pos = (obs["player_x"], obs["player_y"])
    stuck = 0
    while used < max_ticks:
        if obs["dungeon_level"] != 0:
            return obs, used  # 已经下去了
        px, py = obs["player_x"], obs["player_y"]
        dx, dy = tx - px, ty - py
        dist = max(abs(dx), abs(dy))
        if dist == 0:
            # 已站上目标格:触发器要等 PM_STAND 才点火,关卡切换再走一轮事件泵
            for i in range(30):
                obs = bridge.step(ticks=8)
                used += 8
                if i < 3:
                    print(f"    [debug] 站上楼梯后 tick+{(i+1)*8}: mode={obs['player_mode']} "
                          f"pos=({obs['player_x']},{obs['player_y']}) 层={obs['dungeon_level']} "
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
            print(f"    [trace] 命令→({wx},{wy})  实际 ({px},{py})→({obs['player_x']},{obs['player_y']}) "
                  f"mode={obs['player_mode']} path0={obs['walkpath0']}")
        pos = (obs["player_x"], obs["player_y"])
        if pos == last_pos:
            stuck += 1
            # 原地卡住(撞墙/路径失败):随机扰动一下再继续
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
    """新开一局并走下地牢 1 层,返回 (L1 快照, 用时秒, tick 数)。"""
    t0 = time.time()
    obs = bridge.reset(seed=seed)
    assert obs["dungeon_level"] == 0, "开局应在城镇"
    if trace:
        dump_grid("reset 后城镇", obs["player_x"], obs["player_y"])

    stairs = [t for t in obs["triggers"] if t["msg"] == diablogym.bridge.WM_DIABNEXTLVL]
    assert stairs, f"城镇观测里没找到下行楼梯: {obs['triggers']}"
    sx, sy = stairs[0]["x"], stairs[0]["y"]

    obs, used = walk_to_target(sx, sy, trace=trace)
    assert obs["dungeon_level"] == 1, (
        f"没能进入地牢 1 层(最终位置 ({obs['player_x']},{obs['player_y']}),"
        f"层 {obs['dungeon_level']},用了 {used} tick)"
    )
    snapshot = {
        "entry": (obs["player_x"], obs["player_y"]),
        "level_type": obs["level_type"],
        "monsters": tuple(sorted((m["type"], m["x"], m["y"], m["max_hp"]) for m in obs["monsters"])),
    }
    return snapshot, time.time() - t0, used


def dump_grid(tag, cx, cy, r=7):
    """打印以 (cx,cy) 为中心的通行地图:. 可走  #墙  M怪  P玩家  O物体"""
    print(f"    [grid] {tag} 以 ({cx},{cy}) 为中心:")
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
    return (f"入口 {snap['entry']} 怪物 {len(mons)} 只"
            f"(前3: {[m[:3] for m in mons[:3]]})")


def main():
    print("== 下地牢 + 种子分化测试 ==")
    print(f"城镇楼梯位置将从观测 triggers 中读取(预期 (25,29))\n")
    DiabloGymEnv()  # 完成一次性引擎初始化(带默认资产/数据/存档路径)

    snap_a1, dt1, ticks1 = descend(seed=1001)
    print(f"seed=1001 第 1 次: {brief(snap_a1)}  [{dt1:.2f}s, {ticks1} tick]")
    snap_a2, dt2, _ = descend(seed=1001)
    print(f"seed=1001 第 2 次: {brief(snap_a2)}  [{dt2:.2f}s]")
    snap_b, dt3, _ = descend(seed=2002)
    print(f"seed=2002        : {brief(snap_b)}  [{dt3:.2f}s]")

    assert snap_a1 == snap_a2, (
        "同种子两次下地牢快照不一致!\n"
        f"A1: {snap_a1}\nA2: {snap_a2}"
    )
    print("\nPASS: 同种子(1001)两次地牢 1 层完全一致 —— 确定性成立")

    assert snap_a1 != snap_b, "不同种子的地牢 1 层竟然相同 —— 种子注入无效!"
    diff_parts = []
    if snap_a1["entry"] != snap_b["entry"]:
        diff_parts.append("入口位置")
    if snap_a1["monsters"] != snap_b["monsters"]:
        diff_parts.append(f"怪物布局({len(snap_a1['monsters'])} vs {len(snap_b['monsters'])} 只)")
    print(f"PASS: 不同种子(1001 vs 2002)地牢分化于: {', '.join(diff_parts)}")

    print("\n== 全部通过:寻路、关卡切换、种子分化 OK ==")


if __name__ == "__main__":
    main()
