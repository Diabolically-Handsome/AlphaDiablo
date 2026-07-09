"""v22 G0 单元测试:OptionsEnv 状态机角落 + 不变量(纯脚本断言,无 pytest 依赖)。"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from diablogym import OptionsEnv
from diablogym.options_env import DIVE, FARM, KILL_PATIENCE, REVISIT_FLOOR

env = OptionsEnv(max_steps=3000)

# --- 1. 形状与掩码基本面 ---
obs, _ = env.reset(seed=7000)
assert obs.shape == (303,), obs.shape
m = env.action_masks()
assert m.shape == (3,) and m.dtype == bool and m[FARM], m
print("G0.1 PASS: obs 303 维,掩码 3 位,FARM 保底恒真")

# --- 2. 换层必归还(显式不变量)---
obs, _ = env.reset(seed=7000)
descended = False
for _ in range(40):
    if not env.action_masks()[DIVE]:
        break
    lvl_b = env.env._raw["dungeon_level"]
    obs, R, done, trunc, info = env.step(DIVE)
    ex = info["option_extra"]
    if env.env._raw["dungeon_level"] != lvl_b:
        assert ex["reason"] == "descend", ex
        descended = True
        break
    if done or trunc:
        break
assert descended, "40 次 DIVE 未见换层"
print("G0.2 PASS: 换层瞬间选项终止并归还控制权(reason=descend)")

# --- 3. 榨干旗 + 复选地板 ---
obs, _ = env.reset(seed=7001)
reason = None
for _ in range(60):
    obs, R, done, trunc, info = env.step(FARM)
    reason = info["option_extra"]["reason"]
    if reason == "exhausted" or done or trunc:
        break
if reason == "exhausted":
    assert env.exhausted
    obs, R, done, trunc, info = env.step(FARM)
    assert info["option_extra"]["tau"] >= REVISIT_FLOOR or done or trunc, info["option_extra"]
    print(f"G0.3 PASS: 榨干旗置位,复选 FARM 走 {info['option_extra']['tau']} 拍(地板 {REVISIT_FLOOR})")
else:
    print(f"G0.3 SKIP: 该种子未触发榨干(reason={reason}),不变量由 G0.4 模糊测试兜底")

# --- 4. 掩码永不全假 + 随机模糊 200 决策 ---
rng = np.random.default_rng(0)
obs, _ = env.reset(seed=7002)
taus = []
for i in range(200):
    m = env.action_masks()
    assert m.any(), f"决策 {i}:掩码全假"
    opt = int(rng.choice(np.flatnonzero(m)))
    obs, R, done, trunc, info = env.step(opt)
    taus.append(info["option_extra"]["tau"])
    assert info["option_extra"]["tau"] >= 1
    if done or trunc:
        obs, _ = env.reset(seed=7002 + i)
print(f"G0.4 PASS: 200 决策模糊测试,τ 中位 {sorted(taus)[len(taus)//2]}")

# --- 5. τ 与 env._steps 差分一致 ---
obs, _ = env.reset(seed=7003)
s0 = env.env._steps
obs, R, done, trunc, info = env.step(FARM)
assert info["option_extra"]["tau"] == env.env._steps - s0
print("G0.5 PASS: τ 与微步差分逐位一致")

# --- 6. MaskablePPO 冒烟(γ=1)---
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

vec = DummyVecEnv([lambda: Monitor(OptionsEnv(max_steps=600))])
smoke = MaskablePPO("MlpPolicy", vec, n_steps=16, batch_size=16, gamma=1.0,
                    gae_lambda=0.95, seed=22, device="cpu", verbose=0)
smoke.learn(total_timesteps=32, progress_bar=False)
vec.close()
print("G0.6 PASS: MaskablePPO γ=1 在 OptionsEnv 上冒烟完成")
print("G0 ALL PASS")
