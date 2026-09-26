"""v24 G-KL long-running manual gate: A welded end / B zeroed end / C teacher fidelity.

The precondition is bc-worker/policy_sd.pt rebuilt under the current protocol. beta, mask, G-CAL immediate stop and the
zero-beta regression, which need no training output, are covered independently in CI by test_training_core.py.
"""
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import torch as th

if __name__ != "__main__":
    raise unittest.SkipTest(
        "test_leash.py is a long-running manual gate that must be run explicitly; it does not run under unittest discovery")

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
    print("SKIP: test_leash.py is a long-running manual gate for after the BC rebuild; the training outputs are missing")
    raise SystemExit(0)

# ---------- training entry-point boundary regressions ----------
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
    print(f"SKIP: the BC outputs do not satisfy the current protocol; rerun train/bc_worker.py first: {exc}")
    raise SystemExit(0) from None
print("training entry PASS: no singleton tail batch; gear slot 293; BC held-out split by whole episode")

# ---------- G-KL-C: teacher fidelity (torch == numpy payload) ----------
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
print(f"G-KL-C PASS: 6 teacher tensors allclose; max logit difference over 1000 obs {maxd:.2e}, argmax mismatches 0")

# ---------- masked positions contribute exactly 0 (pins the HUGE_NEG semantics) ----------
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
print("G-KL-A.mask PASS: teacher probability at masked positions is exactly 0, 0x(-1e8)=0, CE finite")

# ---------- shared small environment (one engine per process: DummyVecEnv with a single env) ----------
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from diablogym import WorkerWindowEnv

def mk():
    return Monitor(WorkerWindowEnv(
        str(NPZ), max_steps=3000, rng_seed=42, seed_scope="replay"))

venv = DummyVecEnv([mk])

# ---------- G-KL-B: zeroed end (a single beta=0 train() is bit-identical to the original) ----------
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
m_plain.policy.load_state_dict(m_leash.policy.state_dict())  # weights aligned bit for bit
for m in (m_leash, m_plain):
    m._setup_learn(total_timesteps=64)
    fill_buffer(m, seed=11)
th.manual_seed(99); np.random.seed(99); m_leash.train()
th.manual_seed(99); np.random.seed(99); m_plain.train()
for (k1, p1), (k2, p2) in zip(m_leash.policy.state_dict().items(),
                              m_plain.policy.state_dict().items()):
    assert k1 == k2 and th.allclose(p1, p2, atol=1e-7), f"beta=0 not equivalent: {k1}"
print("G-KL-B PASS: after one beta=0 train() update all parameters match the original MaskablePPO bit for bit")

# ---------- fail-loud: beta>0 without a teacher must raise ----------
m_bad = LeashedMaskablePPO("MlpPolicy", venv, distill_beta=1.0, teacher_path=None, **kw)
m_bad._setup_learn(total_timesteps=64)
fill_buffer(m_bad, seed=12)
try:
    m_bad.train()
    raise SystemExit("fail-loud FAIL: beta>0 without a teacher was not rejected")
except RuntimeError:
    print("G-KL.fail-loud PASS: beta>0 without a teacher is rejected explicitly")

# ---------- illegal beta / all-invalid mask / empty-gradient probe during the freeze ----------
try:
    LeashedMaskablePPO("MlpPolicy", venv, distill_beta=-0.1, **kw)
    raise SystemExit("negative-beta FAIL: a negative beta was not rejected")
except ValueError:
    pass
m_guard = LeashedMaskablePPO("MlpPolicy", venv, distill_beta=1.0,
                             teacher_path=str(SD), **kw)
try:
    m_guard._teacher_probs(obs_b[:2], th.zeros(2, 15, dtype=th.bool))
    raise SystemExit("all-false-mask FAIL: an all-False mask was not rejected")
except ValueError:
    pass
m_guard._calib_probe(th.tensor(0.0), th.tensor(0.0), 0.0)
print("Leash guards PASS: negative beta / all-False mask rejected; the empty-gradient probe records 0 without crashing")

# ---------- after an active G-CAL verdict the current minibatch must not update ----------
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
print("G-CAL immediate-stop PASS: the verdict minibatch produced no weight update")

# ---------- G-KL-A: welded end (random init + beta=100, converges to the teacher in 30k steps) ----------
mA = LeashedMaskablePPO("MlpPolicy", venv, distill_beta=100.0, teacher_path=str(SD),
                        n_steps=512, batch_size=256, gamma=1.0, ent_coef=0.005,
                        seed=13, device="cpu", verbose=0)
mA.learn(total_timesteps=30_000, progress_bar=False)
assert mA._last_distill_ce < 0.05, f"CE not welded: {mA._last_distill_ce}"
# argmax agreement rate on 2000 real rollout states
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
assert rate >= 0.99, f"welded-end agreement rate {rate:.4f} < 0.99"
print(f"G-KL-A PASS: beta=100 from random init, 30k steps, CE={mA._last_distill_ce:.4f}, "
      f"argmax agreement rate over 2000 states {rate:.4f}")

venv.close()
print("G-KL ALL PASS")
