"""寻路助手:接力 hop 走位(单次寻路上限约 25 步)+ 卡住扰动 + 脚本化下地牢。"""

from __future__ import annotations

import numpy as np


def _scene_identity(obs):
    is_set = bool(obs.get("is_set_level", False))
    return (int(obs["dungeon_level"]), is_set,
            int(obs.get("set_level_id", 0)) if is_set else 0)


def walk_to(bridge, tx, ty, max_ticks=6000, hop=8, ticks_per_hop=16, jitter_seed=7):
    """分段走向目标格;中途换层立即返回。返回 (最终观测, 用掉 tick 数)。"""
    for name, value in (("max_ticks", max_ticks), ("hop", hop),
                        ("ticks_per_hop", ticks_per_hop)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} 必须是正整数，收到 {value!r}")
    tx, ty = int(tx), int(ty)
    max_ticks, hop, ticks_per_hop = int(max_ticks), int(hop), int(ticks_per_hop)
    rng = np.random.default_rng(jitter_seed)
    obs = bridge.observe()
    start_scene = _scene_identity(obs)
    used = 0
    last_pos = (obs["player_x"], obs["player_y"])
    stuck = 0
    while used < max_ticks:
        if _scene_identity(obs) != start_scene:
            return obs, used
        px, py = obs["player_x"], obs["player_y"]
        dx, dy = tx - px, ty - py
        dist = max(abs(dx), abs(dy))
        if dist == 0:
            # 站上目标格:触发器要等 PM_STAND,关卡切换还要一轮事件泵
            for _ in range(30):
                if used >= max_ticks:
                    break
                ticks = min(8, max_ticks - used)
                obs = bridge.step(ticks=ticks)
                used += ticks
                if _scene_identity(obs) != start_scene:
                    break
            return obs, used
        frac = min(hop, dist) / dist
        bridge.act_walk(px + round(dx * frac), py + round(dy * frac))
        ticks = min(ticks_per_hop, max_ticks - used)
        obs = bridge.step(ticks=ticks)
        used += ticks
        if (_scene_identity(obs) != start_scene
                or obs.get("dead") or obs.get("game_over")):
            return obs, used
        pos = (obs["player_x"], obs["player_y"])
        if pos == last_pos:
            stuck += 1
            if used >= max_ticks:
                break
            jx, jy = int(rng.integers(-6, 7)), int(rng.integers(-6, 7))
            bridge.act_walk(px + jx, py + jy)
            ticks = min(ticks_per_hop, max_ticks - used)
            obs = bridge.step(ticks=ticks)
            used += ticks
            jitter_pos = (obs["player_x"], obs["player_y"])
            if jitter_pos != pos:
                stuck = 0
            if (_scene_identity(obs) != start_scene
                    or obs.get("dead") or obs.get("game_over")):
                return obs, used
        else:
            stuck = 0
        last_pos = (obs["player_x"], obs["player_y"])
        if stuck > 20:
            break
    return obs, used


def descend_to_dungeon(bridge):
    """从城镇出生点走到教堂楼梯并进入地牢 1 层。返回 L1 首帧观测。

    城镇布局固定(楼梯恒在 (25,29)),对任意种子都适用。
    """
    obs = bridge.observe()
    if obs["dungeon_level"] != 0:
        raise ValueError("descend_to_dungeon 需要从城镇出发")
    stairs = [t for t in obs["triggers"] if t["msg"] == bridge.WM_DIABNEXTLVL]
    if not stairs:
        raise RuntimeError(f"城镇观测里没有下行楼梯: {obs['triggers']}")
    obs, used = walk_to(bridge, stairs[0]["x"], stairs[0]["y"])
    if obs["dungeon_level"] != 1:
        raise RuntimeError(
            f"脚本化下地牢失败:{used} tick 后仍在层 {obs['dungeon_level']},"
            f"位置 ({obs['player_x']},{obs['player_y']})"
        )
    return obs
