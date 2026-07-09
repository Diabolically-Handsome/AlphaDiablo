"""v22 策略脑/操作脑:OptionsEnv——冻结宏之上的 SMDP 包装器。

设计稿:docs/DESIGN.md v22 章(评审团 wf_66e41e30 合成,Lawrence 批准)。
核心承诺:
  - 操作脑 = 神谕 oracle_mountain 的内环逐字移植并冻结(平稳 SMDP);
  - "榨干→下潜"不写死在脚本里,升格为策略脑的决策(本章唯一考题);
  - 换层必归还控制权(显式不变量:每层至少一次新决策,16 层扩展性的地基);
  - γ_mgr=1.0 由训练侧保证,选项内奖励不折现累加——策略脑优化的量
    逐字等于神谕账本(3000 步不折现回报);
  - 喝药是脑干反射(hp<0.5∧belt>0 → 12),刻意不是选项(v12 幽灵防复活)。

选项词表 Discrete(3):
  0 FARM     清剿本层:有怪交战,无怪捡药/捡装/探索。终止:升级/停滞 140/换层。
  1 DIVE     战斗下潜一层:挡路者(<3 格)打穿,否则下楼宏。终止:降层/停滞 140。
  2 RESUPPLY 潜前补给:连续捡药。掩码:无地面药或腰带满。终止:满/无/零进展。
通用终止:死亡/截断/TAU_CAP=600 微拍(命中率入 info,>5% 报警)。
反藏身处:FARM 永不掩码(保底);榨干旗置位后复选 FARM 强制 ≥25 拍地板。
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .env import DiabloGymEnv

FARM, DIVE, RESUPPLY = 0, 1, 2
KILL_PATIENCE = 140   # 微拍:本层无新杀即"榨干"(与神谕同价;DIVE 的下降停滞同价)
TAU_CAP = 600         # 选项最长占用(straggler 税封顶)
REVISIT_FLOOR = 25    # 榨干旗下复选 FARM 的最小占用(堵秒终止搅拌键)
RESUPPLY_CAP = 60


def _floor_heals(raw) -> bool:
    return any(it.get("heal") for it in raw.get("floor_items", []))


def dispatch(mode: str, raw, gear_available: bool) -> int:
    """神谕内环逐字移植(纯函数,冻结)。mode ∈ {farm, dive, resupply}。
    注意:神谕农期的 stagnant>=140→11 子分支被刻意剔除——归策略脑管。"""
    hp = raw["hp"] / max(1, raw["max_hp"])
    belt = raw.get("belt_heals", 0)
    if hp < 0.5 and belt > 0:          # 脑干反射,嵌在一切模式里
        return 12
    if mode == "resupply":
        return 13
    if mode == "dive":
        near = _nearest(raw)
        if belt <= 2 and _floor_heals(raw):
            return 13
        if near is not None and near < 3:
            return 9
        return 11
    # farm
    if len(raw.get("monsters", [])) > 0:
        return 9
    if belt < 8 and _floor_heals(raw):
        return 13
    if gear_available:
        return 14
    return 10


def _nearest(raw):
    px, py = raw["player_x"], raw["player_y"]
    ds = [max(abs(m["x"] - px), abs(m["y"] - py)) for m in raw.get("monsters", [])]
    return min(ds) if ds else None


class OptionsEnv(gym.Env):
    """step(option) 把选项跑到终止,返回(303 维观测, 不折现累计奖励, ...)。"""

    N_EXTRA_MGR = 8  # time_remaining/停滞钟/本层杀/本层耗时/上选项 one-hot(3)/上选项 τ

    def __init__(self, max_steps: int = 3000, **env_kwargs):
        super().__init__()
        env_kwargs.setdefault("descend_ladder", True)
        env_kwargs.setdefault("death_ladder", True)
        env_kwargs.setdefault("start_in_dungeon", True)
        env_kwargs.setdefault("include_raw", False)
        self.env = DiabloGymEnv(max_steps=max_steps, **env_kwargs)
        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(3)
        base = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(base + self.N_EXTRA_MGR,), dtype=np.float32)
        self._reset_wrapper_state()

    # ---- wrapper 状态(跨选项持续)----
    def _reset_wrapper_state(self):
        self.layer_clock = 0          # 本层无新杀微拍数(新杀/换层清零)
        self.exhausted = False        # 榨干旗(新杀/换层清除)
        self._fuse_sig = None         # B4 保险丝签名(跨选项边界持续)
        self._fuse = 0
        self._layer_kills0 = 0
        self._layer_steps0 = 0
        self._last_opt = -1
        self._last_tau = 0
        self._cap_hits = 0
        self._decisions = 0
        self.mode_seq = []

    def _sig(self, a, raw):
        return (a, raw["player_x"], raw["player_y"], raw.get("belt_heals", 0),
                raw["char_level"], self.env._ep_kills, raw["dungeon_level"])

    def _tick_layer_clock(self, kills_before, lvl_before, steps_delta):
        raw = self.env._raw
        if raw["dungeon_level"] != lvl_before:
            self.layer_clock = 0
            self.exhausted = False
            self._layer_kills0 = self.env._ep_kills
            self._layer_steps0 = self.env._steps
        elif self.env._ep_kills > kills_before:
            self.layer_clock = 0
            self.exhausted = False
        else:
            self.layer_clock += steps_delta

    # ---- gym 接口 ----
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self._reset_wrapper_state()
        return self._mgr_obs(obs), info

    def action_masks(self) -> np.ndarray:
        raw = self.env._raw
        m = np.ones(3, dtype=bool)
        # DIVE:本层无下行楼梯才掩(引擎级 triggers,几乎恒可选)
        m[DIVE] = any(t.get("msg") == 0 for t in raw.get("triggers", []))
        m[RESUPPLY] = _floor_heals(raw) and raw.get("belt_heals", 0) < 8
        m[FARM] = True  # 保底位,永不掩码
        return m

    def step(self, option: int):
        option = int(option)
        assert self.action_masks()[option], f"选项 {option} 被掩码却被选择"
        raw = self.env._raw
        R = 0.0
        t0 = self.env._steps
        clvl0 = raw["char_level"]
        dlvl0 = raw["dungeon_level"]
        floor = REVISIT_FLOOR if (option == FARM and self.exhausted) else 0
        reason = "cap"
        obs, done, trunc, info = None, False, False, {}
        mode = ("farm", "dive", "resupply")[option]
        resupply_stall = 0
        while True:
            raw = self.env._raw
            a = dispatch(mode, raw, bool(self.env.action_masks()[14]))
            # B4 保险丝(签名含动作;跨选项持续)
            sig = self._sig(a, raw)
            if sig == self._fuse_sig:
                self._fuse += 1
                if self._fuse >= 25:
                    a, self._fuse = 10, 0
            else:
                self._fuse = 0
            self._fuse_sig = self._sig(a, raw)
            kills_b = self.env._ep_kills
            lvl_b = raw["dungeon_level"]
            steps_b = self.env._steps
            belt_b = raw.get("belt_heals", 0)
            obs, r, done, trunc, info = self.env.step(a)
            R += float(r)
            tau = self.env._steps - t0
            self._tick_layer_clock(kills_b, lvl_b, self.env._steps - steps_b)
            if done or trunc:
                reason = "death" if self.env._raw.get("dead") else "end"
                break
            if self.env._raw["dungeon_level"] != dlvl0:
                reason = "descend"      # 换层必归还(显式不变量)
                break
            if tau < floor:
                continue
            if option == FARM and self.env._raw["char_level"] > clvl0:
                reason = "levelup"
                break
            if option == FARM and self.layer_clock >= KILL_PATIENCE:
                reason = "exhausted"
                self.exhausted = True
                break
            if option == DIVE and tau >= KILL_PATIENCE:
                reason = "stall"
                break
            if option == RESUPPLY:
                if self.env._raw.get("belt_heals", 0) <= belt_b:
                    resupply_stall += 1
                else:
                    resupply_stall = 0
                if (self.env._raw.get("belt_heals", 0) >= 8
                        or not _floor_heals(self.env._raw)
                        or resupply_stall >= 2 or tau >= RESUPPLY_CAP):
                    reason = "done"
                    break
            if tau >= TAU_CAP:
                self._cap_hits += 1
                reason = "cap"
                break
        tau = self.env._steps - t0
        self._decisions += 1
        self._last_opt, self._last_tau = option, tau
        self.mode_seq.append("FDR"[option] + ("†" if reason == "death" else ""))
        info = dict(info)
        info["option_extra"] = {
            "opt": option, "tau": tau, "reason": reason,
            "micro_steps": self.env._steps, "decisions": self._decisions,
            "cap_hits": self._cap_hits, "mode_seq": "".join(self.mode_seq),
        }
        return self._mgr_obs(obs), R, done, trunc, info

    def _mgr_obs(self, base_obs) -> np.ndarray:
        raw = self.env._raw
        one_hot = [0.0, 0.0, 0.0]
        if self._last_opt >= 0:
            one_hot[self._last_opt] = 1.0
        extra = np.asarray([
            1.0 - self.env._steps / max(1, self.max_steps),          # 余时
            min(1.0, self.layer_clock / KILL_PATIENCE),              # 停滞钟(核心决策变量)
            min(1.0, (self.env._ep_kills - self._layer_kills0) / 50.0),
            min(1.0, (self.env._steps - self._layer_steps0) / 1500.0),
            *one_hot,
            min(1.0, self._last_tau / TAU_CAP),
        ], dtype=np.float32)
        return np.concatenate([np.asarray(base_obs, dtype=np.float32), extra])


class StagnationClockWrapper(gym.Wrapper):
    """恶魔臂 F 专用:295+1=296 维平面包装——停滞钟与 OptionsEnv 同一块表。"""

    def __init__(self, env: DiabloGymEnv):
        super().__init__(env)
        base = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(base + 1,), dtype=np.float32)
        self._clock = 0
        self._kills = 0
        self._lvl = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self._clock, self._kills, self._lvl = 0, 0, self.env._raw["dungeon_level"]
        return self._obs(obs), info

    def action_masks(self):
        return self.env.action_masks()

    def step(self, action):
        steps_b = self.env._steps
        obs, r, done, trunc, info = self.env.step(action)
        raw = self.env._raw
        if raw["dungeon_level"] != self._lvl or self.env._ep_kills > self._kills:
            self._clock = 0
            self._lvl = raw["dungeon_level"]
            self._kills = self.env._ep_kills
        else:
            self._clock += self.env._steps - steps_b
        return self._obs(obs), r, done, trunc, info

    def _obs(self, base_obs):
        return np.concatenate([np.asarray(base_obs, dtype=np.float32),
                               np.asarray([min(1.0, self._clock / KILL_PATIENCE)],
                                          dtype=np.float32)])
