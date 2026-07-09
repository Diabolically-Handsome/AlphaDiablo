"""v23 G0' 单元测试:WorkerWindowEnv 机械保真(纯脚本断言,同 G0 惯例)。

前置:train/models/v22-h-manager/policy.npz(export_manager_npz.py 产出)。
(a) 脚本工人驱动 WorkerWindowEnv ≡ OptionsEnv+冻结H 直跑(种子 7000-7007 逐位);
(b) 工人掩码(恒掩 11/12,14 透传);
(c) 工资恒等式 Σw ≡ R − bonus,逐窗断言;override/drains 入册;
(d) numpy 经理 ≡ SB3 predict(1000 obs);
(e) 自然收窗 terminated / 基础局截断 truncated,且 VecEnv 注入 TimeLimit.truncated。
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import NumpyManager, OptionsEnv, WorkerWindowEnv
from diablogym.options_env import FARM, dispatch

NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
assert NPZ.exists(), f"缺 {NPZ} —— 先跑 train/export_manager_npz.py"

SEEDS = list(range(7000, 7008))


def win_sig(extra):
    return (extra["opt"], extra["tau"], extra["reason"], round(extra["R"], 3))


# --- (d) numpy 经理 ≡ SB3 predict ---
from sb3_contrib import MaskablePPO

sb3_mgr = MaskablePPO.load(str(ROOT / "train" / "models" / "v22-h-manager" / "model_final"),
                           device="cpu")
np_mgr = NumpyManager(str(NPZ))
rng = np.random.default_rng(1)
mask3 = np.ones(3, dtype=bool)
for i in range(1000):
    o = rng.standard_normal(303).astype(np.float32)
    a_np = np_mgr.choose(o, mask3)
    a_sb, _ = sb3_mgr.predict(o, action_masks=mask3, deterministic=True)
    assert a_np == int(a_sb), f"obs {i}: numpy {a_np} != sb3 {int(a_sb)}"
print("G0'.d PASS: numpy 经理与 SB3 predict 1000 obs 逐位一致")

# --- (a)+(c) 等价性 + 工资恒等式 ---
oe = OptionsEnv(max_steps=3000)
runA = {}
for seed in SEEDS:
    obs, _ = oe.reset(seed=seed)
    done = trunc = False
    wins = []
    while not (done or trunc):
        opt = np_mgr.choose(obs, oe.action_masks())
        obs, r, done, trunc, info = oe.step(opt)
        wins.append(win_sig(info["option_extra"]))
    runA[seed] = {"wins": wins, "steps": oe.env._steps,
                  "seq": info["option_extra"]["mode_seq"]}

wwe = WorkerWindowEnv(str(NPZ), max_steps=3000, rng_seed=0, log_windows=True)
runB = {}
for seed in SEEDS:
    n0 = len(wwe.window_log)
    obs, _ = wwe.reset(seed=seed)
    while obs is not None:
        a = dispatch("farm", wwe.oe.env._raw, bool(wwe.oe.env.action_masks()[14]))
        obs2, w, term, trunc, info = wwe.step(a)
        obs = wwe.next_window() if (term or trunc) else obs2
    entries = wwe.window_log[n0:]
    from diablogym.env import DESCEND_UNIT
    for e in entries:
        assert abs(e["W"] - (e["R"] - e["bonus"])) < 1e-6, e   # (c) 账本自洽
        # (c') 剥薪公式独立对账:bonus ≡ DESCEND_UNIT×Σrange(dlvl0, dlvl_end)
        # (七级阶梯保证换层即收窗,故逐窗 ΣΔdlvl⁺ 塌缩为端点差)
        expect = (DESCEND_UNIT * sum(range(e["dlvl0"], e["dlvl_end"]))
                  if e["dlvl_end"] > e["dlvl0"] else 0.0)
        assert abs(e["bonus"] - expect) < 1e-6, e
        if e["reason"] == "descend":
            assert e["bonus"] > 0.0, e
        elif e["opt"] == FARM and not e["base_done"]:
            assert e["bonus"] == 0.0, e   # 换层拍与局终同拍时豁免(reason=death/end)
    runB[seed] = {"wins": [win_sig(e) for e in entries],
                  "steps": wwe.oe.env._steps,
                  "seq": entries[-1]["mode_seq"]}

for seed in SEEDS:
    A, B = runA[seed], runB[seed]
    assert A["wins"] == B["wins"], (
        f"seed {seed} 窗口序列失配:\nA={A['wins'][:8]}...\nB={B['wins'][:8]}...\n"
        f"len A {len(A['wins'])} B {len(B['wins'])}")
    assert A["steps"] == B["steps"] and A["seq"] == B["seq"], (seed, A["steps"], B["steps"])
print(f"G0'.a PASS: {len(SEEDS)} 种子窗口序列/τ/逐窗R/mode_seq/微步终点逐位一致 "
      f"(共 {sum(len(runA[s]['wins']) for s in SEEDS)} 窗)")
print("G0'.c PASS: 工资恒等式 Σw ≡ R − bonus 全窗成立")

# --- (b) 工人掩码 ---
obs, _ = wwe.reset(seed=7008)
m = wwe.action_masks()
base = wwe.oe.env.action_masks()
assert m.shape == (15,) and m.dtype == bool
assert not m[11] and not m[12], m
assert m[14] == base[14]
assert obs.shape == (298,), obs.shape
print("G0'.b PASS: 掩码恒掩 11/12、14 透传;工人观测 298 维")

# --- (e) terminated / truncated 语义 + VecEnv TimeLimit.truncated ---
seen_term = seen_trunc = False
wwe_s = WorkerWindowEnv(str(NPZ), max_steps=150, rng_seed=3, log_windows=False)
obs, _ = wwe_s.reset(seed=101)
for _ in range(2000):
    a = dispatch("farm", wwe_s.oe.env._raw, bool(wwe_s.oe.env.action_masks()[14]))
    obs, w, term, trunc, info = wwe_s.step(a)
    if term or trunc:
        assert term != trunc, (term, trunc)
        if trunc:
            assert info["option_extra"]["base_trunc"], info["option_extra"]
            seen_trunc = True
        else:
            seen_term = True
        if seen_term and seen_trunc:
            break
        obs, _ = wwe_s.reset()
assert seen_trunc, "150 步短局未观察到 truncated 路径"
assert seen_term, "未观察到 terminated 路径"

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

vec = DummyVecEnv([lambda: Monitor(WorkerWindowEnv(str(NPZ), max_steps=150, rng_seed=5))])
vobs = vec.reset()
tl_seen = False
for _ in range(3000):
    vobs, vr, vdone, vinfos = vec.step(np.asarray([10]))
    if vdone[0]:
        if vinfos[0].get("TimeLimit.truncated"):
            assert "terminal_observation" in vinfos[0]
            tl_seen = True
            break
vec.close()
assert tl_seen, "VecEnv 未注入 TimeLimit.truncated(SB3 bootstrap 分支依赖)"
print("G0'.e PASS: 自然收窗 terminated;基础局截断 truncated 且 VecEnv 注入 TimeLimit.truncated")

print("G0' ALL PASS")
