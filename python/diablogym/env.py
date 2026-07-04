"""DiabloGymEnv —— Gymnasium 包装(v0:结构化向量观测 + 离散动作)。

观测向量(float32,长度 12 + K*4 + 2*(2R+1)²,R=5 时共 286):
  [hp/maxhp, mana/maxmana, xp(log1p/10), gold/1000, char_level/50,
   dungeon_level/16, player_x/112, player_y/112,
   存活怪数/50, 最近怪距离/30(无怪=1),
   最近下行楼梯方向 dx/56, dy/56(本层无则 0,0)]
  + K 个最近怪物的 (dx/20, dy/20, hp/max_hp, 1存在标志)
  + 11×11 局部地图两通道(可走性、怪物占位)——run4 教训:没有空间感知,
    奖励再好也是"盲人拿完美账本"(隔墙锁定、穿墙塑形、找不到房门)

动作(Discrete(11)):
  0      原地不动
  1-8    朝八方向走一格(寻路)
  9      交战宏:锁定最近怪物持续追击,直到它死/自己死/换层/超时(≤10 拍)
         (v2 教训:单拍攻击会被下一个走位动作打断,策略学不会"坚持进攻")
  10     探索宏:走向 25×25 视野内最近的"可走且未踏足"边疆点;发现猎物
         (最近怪 ≤6 格)立即交还控制权;无边疆点时朝下行楼梯走
         (run5 教训:出生区无可达怪时,反应式策略不会"换个房间找")

奖励(v2,逐刀致密化,Lawrence 提案 + 防磨刀修正):
  +0.5 * (本刀伤害/目标最大血) * 残血系数     每刀即时到账;系数 1.0→1.5,
        残血系数 = 1 + 0.5*(1 - 击后血量比)   越残血越值钱(补刀激励),
                                              挂在伤害占比上→无磨刀/秒杀漏洞
  +1.0 * 击杀                                  收头奖励
  +0.01 * ΔXP                                  真实目标(升级)
  +8.0  * Δ地牢层                               ≈4 只怪的价值,清完才值得下楼
  +0.005 * 自己走近最近怪的格数(远离同额扣)
  -0.002  原地不动(含撞墙)
  -2.0 死亡   +10.0 通关
  历史教训:v0 的掉血惩罚→面壁塌缩;v1 的"怪贴脸也计分"→站桩钓鱼。
"""

from __future__ import annotations

import math
import pathlib
import tempfile

import gymnasium as gym
import numpy as np

from . import bridge, nav

# 八方向(等距地牢的 tile 坐标系)
_DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
_K_MONSTERS = 8
_MAP_RADIUS = 5  # 11×11 局部地图

_DEFAULT_ASSETS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "build" / "engine" / "devilutionx.app" / "Contents" / "Resources"
)


class DiabloGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        assets_dir: str | None = None,
        save_dir: str | None = None,
        data_dir: str | None = None,
        ticks_per_step: int = 4,
        max_steps: int = 5000,
        start_in_dungeon: bool = False,
        include_raw: bool = True,
    ):
        super().__init__()
        assets = str(assets_dir or _DEFAULT_ASSETS)
        saves = save_dir or tempfile.mkdtemp(prefix="diablogym-saves-")
        data = str(
            data_dir
            or pathlib.Path.home() / "Library/Application Support/diasurgical/devilution"
        )
        bridge.init(assets_dir=assets, save_dir=saves, data_dir=data, hero_class=0)

        self.ticks_per_step = ticks_per_step
        self.max_steps = max_steps
        self.start_in_dungeon = start_in_dungeon
        self.include_raw = include_raw
        side = 2 * _MAP_RADIUS + 1
        self.action_space = gym.spaces.Discrete(11)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(12 + _K_MONSTERS * 4 + 2 * side * side,), dtype=np.float32,
        )
        self._raw = None
        self._steps = 0
        self._ep_kills = 0
        self._ep_start_xp = 0
        self._visited: set[tuple[int, int]] = set()

    # ---------- gymnasium API ----------

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        actual_seed = seed if seed is not None else int(self.np_random.integers(2**31))
        self._raw = bridge.reset(seed=actual_seed)
        if self.start_in_dungeon:
            # 城镇布局固定,脚本化走到教堂楼梯(约 500-900 tick,~0.05s)
            self._raw = nav.descend_to_dungeon(bridge)
        self._steps = 0
        self._ep_kills = 0
        self._ep_start_xp = int(self._raw["xp"])
        self._visited = {(self._raw["player_x"], self._raw["player_y"])}
        return self._vectorize(self._raw), self._info(self._raw)

    def step(self, action: int):
        prev = self._raw
        action = int(action)
        if action == 9:
            self._raw, micro = self._macro_engage()
        elif action == 10:
            self._raw, micro = self._macro_explore()
        else:
            self._apply_action(action)
            self._raw = bridge.step(ticks=self.ticks_per_step)
            micro = 1
        self._steps += micro
        self._visited.add((self._raw["player_x"], self._raw["player_y"]))

        # 击杀统计:同层内 id 消失即击杀(换层时基线失效,跳过)
        if self._raw["dungeon_level"] == prev["dungeon_level"]:
            cur_ids = {m["id"] for m in self._raw["monsters"]}
            self._ep_kills += sum(1 for m in prev["monsters"] if m["id"] not in cur_ids)

        reward = self._reward(prev, self._raw)
        terminated = bool(self._raw["dead"] or self._raw["game_over"] or self._raw["victory"])
        truncated = self._steps >= self.max_steps

        info = self._info(self._raw)
        if terminated or truncated:
            info["episode_extra"] = {
                "xp": int(self._raw["xp"]) - self._ep_start_xp,
                "kills": self._ep_kills,
                "char_level": self._raw["char_level"],
                "depth": self._raw["dungeon_level"],
                "died": bool(self._raw["dead"]),
                "gold": self._raw["gold"],
            }
        return self._vectorize(self._raw), reward, terminated, truncated, info

    def _info(self, raw):
        return {"raw": raw} if self.include_raw else {}

    # ---------- 内部 ----------

    def _apply_action(self, action: int) -> None:
        obs = self._raw
        px, py = obs["player_x"], obs["player_y"]
        if 1 <= action <= 8:
            dx, dy = _DIRS[action - 1]
            bridge.act_walk(px + dx, py + dy)

    def _macro_engage(self, max_beats: int = 10):
        """交战宏:锁定最近怪物,持续下追击指令直到分出结果或超时。"""
        target = self._nearest_monster(self._raw)
        if target is None:
            return bridge.step(ticks=self.ticks_per_step), 1
        tid = target["id"]
        start_level = self._raw["dungeon_level"]
        raw = prev = self._raw
        beats = 0
        for beats in range(1, max_beats + 1):
            bridge.act_attack_monster(tid)
            raw = bridge.step(ticks=self.ticks_per_step)
            cur_target = next((m for m in raw["monsters"] if m["id"] == tid), None)
            if cur_target is None or raw["dead"] or raw["dungeon_level"] != start_level:
                break
            # 止损:连续 2 拍既没接近目标也没造成伤害(多半隔墙不可达)→ 提前放弃,
            # 把决策权还给策略,避免 run3 式"对着墙白烧 10 拍"
            if beats >= 2:
                prev_target = next((m for m in prev["monsters"] if m["id"] == tid), None)
                if prev_target is not None and cur_target["hp"] >= prev_target["hp"]:
                    d_prev = max(abs(prev_target["x"] - prev["player_x"]), abs(prev_target["y"] - prev["player_y"]))
                    d_cur = max(abs(cur_target["x"] - raw["player_x"]), abs(cur_target["y"] - raw["player_y"]))
                    if d_cur >= d_prev:
                        break
            prev = raw
        return raw, beats

    _EXPLORE_RADIUS = 12  # 25×25 搜索窗

    def _macro_explore(self, max_beats: int = 12):
        """探索宏:走向最近的未踏足可走边疆点;发现猎物立即交还控制权。"""
        raw = self._raw
        px, py = raw["player_x"], raw["player_y"]
        r = self._EXPLORE_RADIUS
        side = 2 * r + 1
        lm = bridge.local_map(radius=r)
        walk = lm["walkable"]

        # 候选:可走、离玩家 ≥5 格、且不在足迹邻域(±1)内的边疆点
        near_visited = self._visited | {
            (x + dx, y + dy) for x, y in self._visited for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        }
        candidates = []
        for i, w in enumerate(walk):
            if not w:
                continue
            tx, ty = px + (i % side) - r, py + (i // side) - r
            d_player = max(abs(tx - px), abs(ty - py))
            if d_player >= 5 and (tx, ty) not in near_visited:
                candidates.append((d_player, tx, ty))
        if candidates:
            _, tx, ty = min(candidates)  # 最近的边疆点(便宜且稳)
        else:
            # 本窗内已探明:朝下行楼梯推进(层级目标),没有就原地一拍
            stairs = [t for t in raw.get("triggers", []) if t["msg"] == 0]
            if not stairs:
                return bridge.step(ticks=self.ticks_per_step), 1
            tx, ty = stairs[0]["x"], stairs[0]["y"]

        start_level = raw["dungeon_level"]
        last_pos = (px, py)
        stall = 0
        beats = 0
        for beats in range(1, max_beats + 1):
            bridge.act_walk(tx, ty)
            raw = bridge.step(ticks=self.ticks_per_step)
            pos = (raw["player_x"], raw["player_y"])
            self._visited.add(pos)
            nd = self._nearest_dist(raw)
            if (raw["dead"] or raw["dungeon_level"] != start_level
                    or (nd is not None and nd <= 6)          # 发现猎物,交还控制权
                    or max(abs(pos[0] - tx), abs(pos[1] - ty)) <= 1):  # 到达
                break
            stall = stall + 1 if pos == last_pos else 0
            if stall >= 2:  # 目标不可达,止损
                break
            last_pos = pos
        return raw, beats

    @staticmethod
    def _nearest_monster(obs):
        px, py = obs["player_x"], obs["player_y"]
        best, best_d = None, None
        for m in obs["monsters"]:
            d = abs(m["x"] - px) + abs(m["y"] - py)
            if best_d is None or d < best_d:
                best, best_d = m, d
        return best

    @staticmethod
    def _nearest_dist(raw):
        px, py = raw["player_x"], raw["player_y"]
        dists = [max(abs(m["x"] - px), abs(m["y"] - py)) for m in raw["monsters"]]
        return min(dists) if dists else None

    @staticmethod
    def _combat_reward(prev, cur) -> float:
        """逐刀伤害奖励 + 击杀奖励(id 匹配的血量差分;1 层无怪物互殴,归因安全)。"""
        cur_hp = {m["id"]: m["hp"] for m in cur["monsters"]}
        r = 0.0
        for m in prev["monsters"]:
            hp_after = cur_hp.get(m["id"], 0)  # id 消失 = 已死,击后血量按 0 计
            damage = m["hp"] - hp_after
            if damage <= 0:
                continue
            hp_after_frac = hp_after / max(1, m["max_hp"])
            finish_mult = 1.0 + 0.5 * (1.0 - hp_after_frac)  # 残血系数 1.0→1.5
            r += 0.5 * (damage / max(1, m["max_hp"])) * finish_mult
            if m["id"] not in cur_hp:
                r += 1.0  # 击杀收头
        return r

    @classmethod
    def _reward(cls, prev, cur) -> float:
        r = 0.01 * (cur["xp"] - prev["xp"])
        r += 8.0 * (cur["dungeon_level"] - prev["dungeon_level"])
        if cur["dungeon_level"] == prev["dungeon_level"]:
            r += cls._combat_reward(prev, cur)
        # 接近塑形:仅当是"自己走近"才有奖励(v2 教训:怪主动贴脸也计分,
        # 会训出"站桩钓鱼却不开打"的白嫖策略)
        if cur["dungeon_level"] == prev["dungeon_level"]:
            moved = (cur["player_x"], cur["player_y"]) != (prev["player_x"], prev["player_y"])
            d0, d1 = cls._nearest_dist(prev), cls._nearest_dist(cur)
            if moved and d0 is not None and d1 is not None:
                r += 0.005 * (d0 - d1)
            if not moved:
                r -= 0.002  # 反面壁/站桩
        if cur["dead"]:
            r -= 2.0
        if cur["victory"]:
            r += 10.0
        return float(r)

    @classmethod
    def _vectorize(cls, obs) -> np.ndarray:
        px, py = obs["player_x"], obs["player_y"]
        nearest = cls._nearest_dist(obs)
        stairs = [t for t in obs.get("triggers", []) if t["msg"] == 0]  # WM_DIABNEXTLVL
        if stairs:
            st = min(stairs, key=lambda t: max(abs(t["x"] - px), abs(t["y"] - py)))
            stair_dx, stair_dy = (st["x"] - px) / 56.0, (st["y"] - py) / 56.0
        else:
            stair_dx = stair_dy = 0.0
        vec = [
            obs["hp"] / max(1, obs["max_hp"]),
            obs["mana"] / max(1, obs["max_mana"]),
            math.log1p(obs["xp"]) / 10.0,
            obs["gold"] / 1000.0,
            obs["char_level"] / 50.0,
            obs["dungeon_level"] / 16.0,
            px / 112.0,
            py / 112.0,
            min(1.0, len(obs["monsters"]) / 50.0),
            min(1.0, nearest / 30.0) if nearest is not None else 1.0,
            stair_dx,
            stair_dy,
        ]
        monsters = sorted(
            obs["monsters"], key=lambda m: abs(m["x"] - px) + abs(m["y"] - py)
        )[:_K_MONSTERS]
        for m in monsters:
            vec += [(m["x"] - px) / 20.0, (m["y"] - py) / 20.0, m["hp"] / max(1, m["max_hp"]), 1.0]
        vec += [0.0, 0.0, 0.0, 0.0] * (_K_MONSTERS - len(monsters))
        lm = bridge.local_map(radius=_MAP_RADIUS)
        vec += [float(v) for v in lm["walkable"]]
        vec += [float(v) for v in lm["monster"]]
        return np.asarray(vec, dtype=np.float32)
