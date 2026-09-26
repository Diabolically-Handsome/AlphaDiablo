"""Pathing helpers: relayed hop walking (one path search reaches about 25 steps) + stuck perturbation + scripted dungeon entry."""

from __future__ import annotations

import numpy as np


def _scene_identity(obs):
    is_set = bool(obs.get("is_set_level", False))
    return (int(obs["dungeon_level"]), is_set,
            int(obs.get("set_level_id", 0)) if is_set else 0)


def walk_to(bridge, tx, ty, max_ticks=6000, hop=8, ticks_per_hop=16, jitter_seed=7,
            *, raw_validator=None):
    """Walk toward the target tile in segments; return immediately on a level change. Returns (final observation, ticks used)."""
    for name, value in (("max_ticks", max_ticks), ("hop", hop),
                        ("ticks_per_hop", ticks_per_hop)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    tx, ty = int(tx), int(ty)
    max_ticks, hop, ticks_per_hop = int(max_ticks), int(hop), int(ticks_per_hop)
    rng = np.random.default_rng(jitter_seed)
    obs = bridge.observe()
    if raw_validator is not None:
        raw_validator(obs)
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
            # Standing on the target tile: triggers wait for PM_STAND, and a level change needs one more event-pump round
            for _ in range(30):
                if used >= max_ticks:
                    break
                ticks = min(8, max_ticks - used)
                obs = bridge.step(ticks=ticks)
                if raw_validator is not None:
                    raw_validator(obs)
                used += ticks
                if _scene_identity(obs) != start_scene:
                    break
            return obs, used
        frac = min(hop, dist) / dist
        bridge.act_walk(px + round(dx * frac), py + round(dy * frac))
        ticks = min(ticks_per_hop, max_ticks - used)
        obs = bridge.step(ticks=ticks)
        if raw_validator is not None:
            raw_validator(obs)
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
            if raw_validator is not None:
                raw_validator(obs)
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


def descend_to_dungeon(bridge, *, raw_validator=None):
    """Walk from the town spawn point to the cathedral stairs and enter dungeon level 1. Returns the first L1 observation.

    The town layout is fixed (the stairs are always at (25,29)), so this works for any seed.
    """
    obs = bridge.observe()
    if raw_validator is not None:
        raw_validator(obs)
    if obs["dungeon_level"] != 0:
        raise ValueError("descend_to_dungeon must start from town")
    stairs = [t for t in obs["triggers"] if t["msg"] == bridge.WM_DIABNEXTLVL]
    if not stairs:
        raise RuntimeError(f"no down stairs in the town observation: {obs['triggers']}")
    if raw_validator is None:
        obs, used = walk_to(bridge, stairs[0]["x"], stairs[0]["y"])
    else:
        obs, used = walk_to(bridge, stairs[0]["x"], stairs[0]["y"], raw_validator=raw_validator)
    if obs["dungeon_level"] != 1:
        raise RuntimeError(
            f"scripted dungeon entry failed: still on level {obs['dungeon_level']} after {used} ticks, "
            f"position ({obs['player_x']},{obs['player_y']})"
        )
    return obs
