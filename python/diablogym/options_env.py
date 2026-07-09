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

v23(docs/PREREG-v23.md):窗口循环的逐拍簿记(保险丝/反射/终止阶梯)抽成
共享方法——OptionsEnv(组装/评测)与 WorkerWindowEnv(在位训练)跑同一段
代码,消灭"第三份实现"。支持 workers={选项: 策略} 把某选项的脚本内环换成
可学习工人:
  - 反射所有权上提:工人永不观测"反射待发"态,窗口开始与每个工人动作之后
    由包装器排水(逐拍过保险丝/时钟/终止阶梯)。脚本路径为恒等变换
    (dispatch 首分支即同款检查,提前判定不改变动作序列)。
  - 工人工资 w_t = r_t − 换层奖金(唯一剥除项;下潜套利修复,教训16)。
    账本恒等式:Σw ≡ 窗口R − DESCEND_UNIT×ΣΔdlvl⁺。经理账本一字不动。
  - 工人观测 298 维 = 基础 295 + [τ/TAU_CAP, 停滞钟/140, 榨干旗]。
  - 工人动作恒掩 11(下楼归经理 DIVE 职权)与 12(喝药归脑干)。
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .env import DESCEND_UNIT, DiabloGymEnv

FARM, DIVE, RESUPPLY = 0, 1, 2
KILL_PATIENCE = 140   # 微拍:本层无新杀即"榨干"(与神谕同价;DIVE 的下降停滞同价)
TAU_CAP = 600         # 选项最长占用(straggler 税封顶)
REVISIT_FLOOR = 25    # 榨干旗下复选 FARM 的最小占用(堵秒终止搅拌键)
RESUPPLY_CAP = 60
N_EXTRA_WORKER = 3    # 工人观测追加维:τ 钟 / 停滞钟 / 榨干旗


def _floor_heals(raw) -> bool:
    return any(it.get("heal") for it in raw.get("floor_items", []))


def _reflex(raw) -> bool:
    """脑干反射条件(与 dispatch 首分支逐字同款)。"""
    return raw["hp"] / max(1, raw["max_hp"]) < 0.5 and raw.get("belt_heals", 0) > 0


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

    def __init__(self, max_steps: int = 3000, workers: dict | None = None, **env_kwargs):
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
        self._workers = workers or {}
        self._last_base_obs = None
        self._win = None
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
        self._win = None

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
        self._last_base_obs = obs
        return self._mgr_obs(obs), info

    def action_masks(self) -> np.ndarray:
        raw = self.env._raw
        m = np.ones(3, dtype=bool)
        # DIVE:本层无下行楼梯才掩(引擎级 triggers,几乎恒可选)
        m[DIVE] = any(t.get("msg") == 0 for t in raw.get("triggers", []))
        m[RESUPPLY] = _floor_heals(raw) and raw.get("belt_heals", 0) < 8
        m[FARM] = True  # 保底位,永不掩码
        return m

    # ---- 共享窗口核(v23:OptionsEnv 与 WorkerWindowEnv 唯一实现)----
    def _win_begin(self, option: int):
        option = int(option)
        assert self.action_masks()[option], f"选项 {option} 被掩码却被选择"
        raw = self.env._raw
        self._win = {
            "opt": option,
            "mode": ("farm", "dive", "resupply")[option],
            "t0": self.env._steps,
            "clvl0": raw["char_level"],
            "dlvl0": raw["dungeon_level"],
            "floor": REVISIT_FLOOR if (option == FARM and self.exhausted) else 0,
            "resupply_stall": 0,
            "R": 0.0, "W": 0.0, "bonus": 0.0,
            "beats": 0, "overrides": 0, "drains": 0,
            "done": False, "trunc": False, "last_info": {},
        }

    def _beat(self, a: int):
        """一拍:保险丝 → env.step → 观测缓存 → 停滞钟。不含终止判定。"""
        raw = self.env._raw
        requested = a
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
        self._last_base_obs = obs
        self._tick_layer_clock(kills_b, lvl_b, self.env._steps - steps_b)
        return float(r), done, trunc, info, a != requested, lvl_b, belt_b

    def _win_term(self, done, trunc, belt_b):
        """七级终止阶梯(顺序与 v22 逐行同构)。返回 reason 或 None。"""
        w = self._win
        raw = self.env._raw
        tau = self.env._steps - w["t0"]
        if done or trunc:
            return "death" if raw.get("dead") else "end"
        if raw["dungeon_level"] != w["dlvl0"]:
            return "descend"      # 换层必归还(显式不变量)
        if tau < w["floor"]:
            return None
        opt = w["opt"]
        if opt == FARM and raw["char_level"] > w["clvl0"]:
            return "levelup"
        if opt == FARM and self.layer_clock >= KILL_PATIENCE:
            self.exhausted = True
            return "exhausted"
        if opt == DIVE and tau >= KILL_PATIENCE:
            return "stall"
        if opt == RESUPPLY:
            if raw.get("belt_heals", 0) <= belt_b:
                w["resupply_stall"] += 1
            else:
                w["resupply_stall"] = 0
            if (raw.get("belt_heals", 0) >= 8
                    or not _floor_heals(raw)
                    or w["resupply_stall"] >= 2 or tau >= RESUPPLY_CAP):
                return "done"
        if tau >= TAU_CAP:
            self._cap_hits += 1
            return "cap"
        return None

    def _win_beat(self, a: int):
        """一拍 + 账本(经理 R / 工人 W)+ 终止判定。返回 reason 或 None。"""
        w = self._win
        r, done, trunc, info, overridden, lvl_b, belt_b = self._beat(a)
        w["beats"] += 1
        w["overrides"] += int(overridden)
        w["R"] += r
        cur_lvl = self.env._raw["dungeon_level"]
        bonus = DESCEND_UNIT * sum(range(lvl_b, cur_lvl)) if cur_lvl > lvl_b else 0.0
        w["bonus"] += bonus
        w["W"] += r - bonus
        w["done"], w["trunc"], w["last_info"] = done, trunc, info
        return self._win_term(done, trunc, belt_b)

    def _drain(self):
        """反射排水:hp<0.5∧belt>0 时由包装器逐拍喝药(工人路径专用;
        每一拍照常过保险丝/时钟/终止阶梯——喝药拍可跨榨干/CAP/死亡)。"""
        w = self._win
        while _reflex(self.env._raw):
            w["drains"] += 1
            reason = self._win_beat(12)
            if reason is not None:
                return reason
        return None

    def _win_step_worker(self, a: int):
        """工人一步 = 工人动作一拍 + 反射尾部排水。返回 reason 或 None。"""
        if not self._worker_masks()[a]:
            raise ValueError(f"工人动作 {a} 被掩码却被执行")
        reason = self._win_beat(a)
        if reason is None:
            reason = self._drain()
        return reason

    def _win_end(self, reason: str):
        """收窗:经理状态机推进 + option_extra。返回 (extra, base_info, done, trunc)。"""
        w = self._win
        tau = self.env._steps - w["t0"]
        self._decisions += 1
        self._last_opt, self._last_tau = w["opt"], tau
        self.mode_seq.append("FDR"[w["opt"]] + ("†" if reason == "death" else ""))
        extra = {
            "opt": w["opt"], "tau": tau, "reason": reason,
            "micro_steps": self.env._steps, "decisions": self._decisions,
            "cap_hits": self._cap_hits, "mode_seq": "".join(self.mode_seq),
            "R": w["R"], "W": w["W"], "bonus": w["bonus"],
            "beats": w["beats"], "overrides": w["overrides"], "drains": w["drains"],
            "dlvl0": w["dlvl0"], "dlvl_end": self.env._raw["dungeon_level"],
            "dry": w["floor"] > 0,      # 开窗时榨干旗在位(干层复访窗)
            "base_done": w["done"] or w["trunc"],
            "base_trunc": w["trunc"] and not w["done"],
        }
        base_info, done, trunc = w["last_info"], w["done"], w["trunc"]
        self._win = None
        return extra, base_info, done, trunc

    # ---- 工人视角(v23)----
    def _worker_obs(self) -> np.ndarray:
        w = self._win
        tau = self.env._steps - w["t0"] if w is not None else self._last_tau
        extra = np.asarray([
            min(1.0, tau / TAU_CAP),
            min(1.0, self.layer_clock / KILL_PATIENCE),
            1.0 if self.exhausted else 0.0,
        ], dtype=np.float32)
        return np.concatenate([np.asarray(self._last_base_obs, dtype=np.float32), extra])

    def _worker_masks(self) -> np.ndarray:
        m = np.array(self.env.action_masks(), dtype=bool)
        m[11] = False   # 下楼归经理(DIVE 职权;走格踩楼梯由剥薪封死激励)
        m[12] = False   # 喝药归脑干
        return m

    def step(self, option: int):
        self._win_begin(option)
        mode = self._win["mode"]
        worker = self._workers.get(int(option))
        if worker is not None:
            reason = self._drain()      # 开窗排水:工人首个观测必须是无反射态
            while reason is None:
                a = int(worker(self._worker_obs(), self._worker_masks()))
                reason = self._win_step_worker(a)
        else:
            while True:
                raw = self.env._raw
                a = 12 if _reflex(raw) else dispatch(
                    mode, raw, bool(self.env.action_masks()[14]))
                reason = self._win_beat(a)
                if reason is not None:
                    break
        extra, base_info, done, trunc = self._win_end(reason)
        info = dict(base_info)
        info["option_extra"] = extra
        return self._mgr_obs(self._last_base_obs), extra["R"], done, trunc, info

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
