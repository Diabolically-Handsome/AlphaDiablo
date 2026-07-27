"""v24 G-KL 长程人工闸门:A 焊死端 / B 零化端 / C 教师保真。

前置是按当前协议重建的 bc-worker/policy_sd.pt。无需训练产物的 beta、mask、
G-CAL 即时停止和零 beta 回归都在 test_training_core.py 中由 CI 独立覆盖。
"""
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import torch as th

if __name__ != "__main__":
    raise unittest.SkipTest(
        "test_leash.py 是需显式执行的长程人工闸门，不在 unittest discovery 中运行")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))

from leashed_ppo import HUGE_NEG, LeashedMaskablePPO, build_teacher
from train_ppo import (_GEAR_PRESENT_INDEX, _select_batch_size,
                       _validate_bc_report)
from bc_worker import split_by_episode

SD = ROOT / "train" / "runs" / "bc-worker" / "policy_sd.pt"
NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
if not SD.is_file() or not NPZ.is_file():
    print("SKIP: test_leash.py 是 BC 重建后的长程人工闸门；当前缺少训练产物")
    raise SystemExit(0)

# ---------- 训练入口边界回归 ----------
assert _select_batch_size(512, 4) == 256
tail_safe = _select_batch_size(257, 1)
assert 257 % tail_safe != 1, tail_safe
assert _GEAR_PRESENT_INDEX == 293
groups = np.repeat(np.arange(20), 3)
tr, ho, _ = split_by_episode(groups)
assert not set(groups[tr]).intersection(groups[ho])
try:
    _validate_bc_report(SD, "data_gate")
except ValueError as exc:
    print(f"SKIP: BC 产物不满足当前协议，先重跑 train/bc_worker.py: {exc}")
    raise SystemExit(0) from None
print("训练入口 PASS:batch 无 singleton 尾批;装备位 293;BC held-out 整局隔离")

# ---------- G-KL-C:教师保真(torch ≡ numpy 载荷) ----------
from eval_assembled import np_policy_from_sd

teacher = build_teacher(str(SD))
np_net = np_policy_from_sd(str(SD))
pairs = [(teacher[0], np_net.w0, np_net.b0), (teacher[2], np_net.w1, np_net.b1),
         (teacher[4], np_net.wa, np_net.ba)]
for lin, w, b in pairs:
    assert np.allclose(lin.weight.detach().numpy(), w, atol=1e-6)
    assert np.allclose(lin.bias.detach().numpy(), b, atol=1e-6)
rng = np.random.default_rng(3)
mism, maxd = 0, 0.0
for _ in range(1000):
    o = rng.standard_normal(298).astype(np.float32)
    lt = teacher(th.from_numpy(o)).detach().numpy()
    ln = np_net.forensic_worker_logits(o)
    maxd = max(maxd, float(np.abs(lt - ln).max()))
    mism += int(lt.argmax() != ln.argmax())
assert maxd < 1e-4 and mism == 0, (maxd, mism)
print(f"G-KL-C PASS: 教师 6 张量 allclose;1000 obs logits 最大差 {maxd:.2e},argmax 失配 0")

# ---------- 掩位贡献恰为 0(钉死 HUGE_NEG 语义) ----------
obs_b = th.from_numpy(rng.standard_normal((64, 298)).astype(np.float32))
mask_b = th.ones(64, 15, dtype=th.bool)
mask_b[:, 11] = False
mask_b[:, 12] = False
t_logits = teacher(obs_b)
t_logits = th.where(mask_b, t_logits, th.full_like(t_logits, HUGE_NEG))
t_probs = th.softmax(t_logits, dim=-1)
assert (t_probs[:, 11] == 0).all() and (t_probs[:, 12] == 0).all()
fake_logp = th.full((64, 15), HUGE_NEG)
contrib = t_probs * fake_logp
assert (contrib[:, 11] == 0).all() and (contrib[:, 12] == 0).all()
assert th.isfinite((-(t_probs * fake_logp).sum(-1))).all()
print("G-KL-A.掩位 PASS: 教师掩位概率精确 0,0×(-1e8)=0,CE 有限")

# ---------- 共用小环境(一进程一引擎:DummyVecEnv 单 env) ----------
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from diablogym import WorkerWindowEnv

def mk():
    return Monitor(WorkerWindowEnv(
        str(NPZ), max_steps=3000, rng_seed=42, seed_scope="replay"))

venv = DummyVecEnv([mk])

# ---------- G-KL-B:零化端(β=0 单次 train() 与原版逐位等价) ----------
from sb3_contrib import MaskablePPO

def fill_buffer(model, seed):
    r = np.random.default_rng(seed)
    buf = model.rollout_buffer
    buf.reset()
    obs = r.standard_normal((model.n_steps, 1, 298)).astype(np.float32)
    for i in range(model.n_steps):
        mask = np.ones((1, 15), dtype=bool)
        mask[:, 11] = mask[:, 12] = False
        a = np.asarray([int(r.choice([9, 10, 13, 14]))])
        buf.add(obs[i], a, np.asarray([float(r.standard_normal())]),
                np.asarray([i % 50 == 0]), th.zeros(1), th.zeros(1),
                action_masks=mask.reshape(1, -1))
    buf.compute_returns_and_advantage(last_values=th.zeros(1), dones=np.zeros(1))

kw = dict(n_steps=64, batch_size=64, gamma=1.0, ent_coef=0.005, seed=7,
          device="cpu", verbose=0)
m_leash = LeashedMaskablePPO("MlpPolicy", venv, distill_beta=0.0, **kw)
m_plain = MaskablePPO("MlpPolicy", venv, **kw)
m_plain.policy.load_state_dict(m_leash.policy.state_dict())  # 权重逐位对齐
for m in (m_leash, m_plain):
    m._setup_learn(total_timesteps=64)
    fill_buffer(m, seed=11)
th.manual_seed(99); np.random.seed(99); m_leash.train()
th.manual_seed(99); np.random.seed(99); m_plain.train()
for (k1, p1), (k2, p2) in zip(m_leash.policy.state_dict().items(),
                              m_plain.policy.state_dict().items()):
    assert k1 == k2 and th.allclose(p1, p2, atol=1e-7), f"β=0 不等价: {k1}"
print("G-KL-B PASS: β=0 单次 train() 更新后全参数与原版 MaskablePPO 逐位一致")

# ---------- fail-loud:β>0 无教师必须炸 ----------
m_bad = LeashedMaskablePPO("MlpPolicy", venv, distill_beta=1.0, teacher_path=None, **kw)
m_bad._setup_learn(total_timesteps=64)
fill_buffer(m_bad, seed=12)
try:
    m_bad.train()
    raise SystemExit("fail-loud FAIL: β>0 无教师竟未拒绝")
except RuntimeError:
    print("G-KL.fail-loud PASS: β>0 无教师显式拒绝")

# ---------- 非法 β / 全无效 mask / 冻结期空梯度探针 ----------
try:
    LeashedMaskablePPO("MlpPolicy", venv, distill_beta=-0.1, **kw)
    raise SystemExit("negative-beta FAIL:负 β 竟未拒绝")
except ValueError:
    pass
m_guard = LeashedMaskablePPO("MlpPolicy", venv, distill_beta=1.0,
                             teacher_path=str(SD), **kw)
try:
    m_guard._teacher_probs(obs_b[:2], th.zeros(2, 15, dtype=th.bool))
    raise SystemExit("all-false-mask FAIL:全 False mask 竟未拒绝")
except ValueError:
    pass
m_guard._calib_probe(th.tensor(0.0), th.tensor(0.0), 0.0)
print("Leash 防护 PASS:负 β/全 False mask 拒绝;空梯度探针记 0 不崩溃")

# ---------- 主动 G-CAL 裁决后不得再更新当前 minibatch ----------
with tempfile.TemporaryDirectory() as td:
    m_trip = LeashedMaskablePPO(
        "MlpPolicy", venv, distill_beta=1.0, teacher_path=str(SD),
        calib_probes=[0], calib_out=str(pathlib.Path(td) / "calib.jsonl"), **kw)
    m_trip._setup_learn(total_timesteps=64)
    fill_buffer(m_trip, seed=14)
    before = {k: v.detach().clone() for k, v in m_trip.policy.state_dict().items()}
    m_trip.train()
    assert m_trip._calib_tripped
    assert all(th.equal(before[k], v) for k, v in m_trip.policy.state_dict().items())
print("G-CAL immediate-stop PASS:裁决 minibatch 未产生权重更新")

# ---------- G-KL-A:焊死端(随机初始化 + β=100,30k 步收敛到教师) ----------
mA = LeashedMaskablePPO("MlpPolicy", venv, distill_beta=100.0, teacher_path=str(SD),
                        n_steps=512, batch_size=256, gamma=1.0, ent_coef=0.005,
                        seed=13, device="cpu", verbose=0)
mA.learn(total_timesteps=30_000, progress_bar=False)
assert mA._last_distill_ce < 0.05, f"CE 未焊死: {mA._last_distill_ce}"
# 2000 个真实 rollout 态上 argmax 一致率
env1 = WorkerWindowEnv(
    str(NPZ), max_steps=3000, rng_seed=77, seed_scope="replay")
obs, _ = env1.reset()
agree = tot = 0
while tot < 2000:
    m = env1.action_masks()
    a_s, _ = mA.predict(obs, action_masks=m, deterministic=True)
    with th.no_grad():
        tl = teacher(th.from_numpy(np.asarray(obs, dtype=np.float32)))
        tl = th.where(th.from_numpy(m), tl, th.full_like(tl, HUGE_NEG))
    agree += int(int(a_s) == int(tl.argmax()))
    tot += 1
    obs, w, term, trunc, _ = env1.step(int(a_s))
    if term or trunc:
        obs, _ = env1.reset()
rate = agree / tot
assert rate >= 0.99, f"焊死端一致率 {rate:.4f} < 0.99"
print(f"G-KL-A PASS: β=100 随机起跑 30k 步,CE={mA._last_distill_ce:.4f},"
      f"2000 态 argmax 一致率 {rate:.4f}")

venv.close()
print("G-KL ALL PASS")
