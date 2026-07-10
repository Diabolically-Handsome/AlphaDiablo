"""v23 WorkerWindowEnv:FARM 操作脑的在位训练环境(docs/PREREG-v23.md)。

一个 episode = 冻结 v22-H 经理选中的一个 FARM 窗口。
  - reset():同局快进——经理(numpy 前向,argmax+经理掩码)逐窗决策,
    DIVE/RESUPPLY 窗口由脚本内环跑完(OptionsEnv.step 原路径,簿记全同源);
    遇 FARM 即开窗、排水反射,把首个无反射观测交给工人。基础局死/截断则
    滚入新局(含出生快进的天然代价)。跨窗口 wrapper 状态(停滞钟/榨干旗/
    保险丝/上选项)绝不清零——经理 303 维观测的状态机与 OptionsEnv 同一段代码。
  - step(a):工人一拍 + 反射尾部排水(共享窗口核 _win_step_worker)。
    奖励 = 工资 w(原始奖励 − 换层奖金)。自然收窗 → terminated;
    仅基础局 3000 步截断 → truncated(SB3 bootstrap 语义,预注册拍板#1)。
  - 训练种子:显式采样器,拒采探针 [7000,7032) 与金种子 [9000,9032)。
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .options_env import FARM, N_EXTRA_WORKER, OptionsEnv

_FORBIDDEN = ((7000, 7032), (9000, 9032))


class NumpyManager:
    """冻结 v22-H 的 numpy 前向(MlpPolicy(64,64) 策略侧;G0' 与 SB3 逐位对账)。"""

    def __init__(self, npz_path: str):
        z = np.load(npz_path)
        self.w0, self.b0 = z["w0"].astype(np.float32), z["b0"].astype(np.float32)
        self.w1, self.b1 = z["w1"].astype(np.float32), z["b1"].astype(np.float32)
        self.wa, self.ba = z["wa"].astype(np.float32), z["ba"].astype(np.float32)

    def logits(self, obs: np.ndarray) -> np.ndarray:
        h = np.tanh(self.w0 @ obs.astype(np.float32) + self.b0)
        h = np.tanh(self.w1 @ h + self.b1)
        return self.wa @ h + self.ba

    def choose(self, obs: np.ndarray, mask: np.ndarray) -> int:
        lg = self.logits(obs)
        lg = np.where(np.asarray(mask, dtype=bool), lg, -np.inf)
        return int(np.argmax(lg))


def sample_train_seed(rng: np.random.Generator) -> int:
    """训练种子采样器:拒采探针池与金种子段(预注册 D2)。"""
    while True:
        s = int(rng.integers(0, 2**31))
        if not any(lo <= s < hi for lo, hi in _FORBIDDEN):
            return s


class WorkerWindowEnv(gym.Env):
    """SB3 视角:obs 298 维,Discrete(15) 恒掩 11/12,episode = 一个 FARM 窗口。"""

    metadata = {"render_modes": []}

    def __init__(self, manager_npz: str, max_steps: int = 3000,
                 rng_seed: int | None = None, log_windows: bool = False,
                 skip_dry: bool = False, **env_kwargs):
        super().__init__()
        self.oe = OptionsEnv(max_steps=max_steps, **env_kwargs)
        self.mgr = NumpyManager(manager_npz)
        base = self.oe.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(base + N_EXTRA_WORKER,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(15)
        self._rng = np.random.default_rng(rng_seed)
        self._alive = False
        self.log_windows = log_windows
        self.skip_dry = skip_dry   # v26 绿洲:干层复访窗由脚本代跑,不进学习分布
        self.window_log = []      # log_windows=True 时:全部窗口(含快进窗)按序入册
        self.stats = {"windows": 0, "dry": 0, "fresh": 0, "ff_windows": 0,
                      "ff_dry": 0, "episodes": 0, "reseeds": 0, "reasons": {}}

    # ---- 内务 ----
    def _new_episode(self, seed=None):
        s = int(seed) if seed is not None else sample_train_seed(self._rng)
        self.oe.reset(seed=s)
        self._alive = True
        self.stats["episodes"] += 1

    def _log(self, extra, fast_forward: bool):
        if self.log_windows:
            self.window_log.append(dict(extra, ff=fast_forward))
        if fast_forward:
            self.stats["ff_windows"] += 1

    def _mgr_choose(self) -> int:
        mobs = self.oe._mgr_obs(self.oe._last_base_obs)
        return self.mgr.choose(mobs, self.oe.action_masks())

    def next_window(self):
        """推进到本局下一个 FARM 窗口的首个无反射观测;局尽返回 None,**绝不滚新局**
        (BC 示范/测试的种子纪律依赖这一点——滚局只许发生在 reset())。"""
        if not self._alive:
            return None
        while True:
            opt = self._mgr_choose()
            if opt == FARM:
                dry = self.oe.exhausted
                if dry and self.skip_dry:
                    # v26 绿洲:干层复访窗(榨干旗在位)由脚本内环代跑——
                    # 与 DIVE/RESUPPLY 同路,簿记同源,不成为学习 episode
                    _, _, done, trunc, info = self.oe.step(FARM)
                    self._log(info["option_extra"], fast_forward=True)
                    self.stats["ff_dry"] += 1
                    if done or trunc:
                        self._alive = False
                        return None
                    continue
                self.oe._win_begin(FARM)
                reason = self.oe._drain()   # 开窗排水:工人首观测无反射态
                if reason is None:
                    self.stats["windows"] += 1
                    self.stats["dry" if dry else "fresh"] += 1
                    return self.oe._worker_obs()
                # 排水拍直接终结了窗口(死亡/榨干/CAP……):按快进窗入册,继续找
                extra, _, done, trunc = self.oe._win_end(reason)
                self._log(extra, fast_forward=True)
                if done or trunc:
                    self._alive = False
                    return None
            else:
                _, _, done, trunc, info = self.oe.step(opt)   # 脚本内环,同源簿记
                self._log(info["option_extra"], fast_forward=True)
                if done or trunc:
                    self._alive = False
                    return None

    # ---- gym 接口 ----
    def reset(self, *, seed=None, options=None):
        if seed is not None or not self._alive:
            self._new_episode(seed)
        while True:
            obs = self.next_window()
            if obs is not None:
                return obs, {}
            self.stats["reseeds"] += 1    # 兜底滚局(显式种子局零 FARM 窗时也会走到——
            self._new_episode()           # BC 侧用 stats 断言封死示范池逃逸口)

    def step(self, action):
        win = self.oe._win
        w_before, o_before = win["W"], win["overrides"]
        reason = self.oe._win_step_worker(int(action))
        wage = win["W"] - w_before
        overridden = win["overrides"] > o_before   # 本步含保险丝强制拍(BC 剔除用)
        obs = self.oe._worker_obs()          # 收窗前取终观测(窗口仍持有 τ)
        if reason is None:
            return obs, wage, False, False, {"overridden": overridden}
        extra, _, done, trunc = self.oe._win_end(reason)
        self._log(extra, fast_forward=False)
        self.stats["reasons"][reason] = self.stats["reasons"].get(reason, 0) + 1
        if done or trunc:
            self._alive = False
        truncated = bool(extra["base_trunc"])
        return obs, wage, not truncated, truncated, {"option_extra": extra,
                                                     "overridden": overridden}

    def action_masks(self) -> np.ndarray:
        return self.oe._worker_masks()
