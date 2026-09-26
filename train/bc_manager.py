"""v22 insurance arm H-BC (launched when P6 triggers): option-level teacher demos -> BC warm start of the policy brain.

Teacher = the probe_options teacher (drained flag or clvl >= dlvl+2 -> DIVE, otherwise FARM).
Demo seeds 100-227 (disjoint from the probe/evaluation pools). Writes policy_sd.pt for
train_ppo --options --bc-init; includes a replay check (303-dim memoryless hypothesis, option level).
"""
import json
import hashlib
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import OptionsEnv
from diablogym.options_env import DIVE, FARM
from eval_contract import PROTOCOL_VERSION, exclusive_lock
from train_ppo import (
    _BC_REPORT_SCHEMA_VERSION,
    _implementation_bundle_sha256,
    _masked_action_or_first_legal,
)

OUT = ROOT / "train" / "runs" / "bc-manager"
OUT.mkdir(parents=True, exist_ok=True)
DEMO_SEEDS = list(range(100, 228))
REPLAY_SEEDS = list(range(7000, 7032))


def artifact_provenance():
    return {
        "schema_version": _BC_REPORT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "implementation_sha256": _implementation_bundle_sha256(),
        "generator_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()).hexdigest(),
    }


def begin_output_attempt():
    old = [OUT / name for name in ("policy_sd.pt", "bc_report.json")
           if (OUT / name).exists()]
    if old:
        archive = OUT / "_previous" / str(time.time_ns())
        archive.mkdir(parents=True)
        for path in old:
            path.replace(archive / path.name)
    (OUT / "bc_report.json").write_text(json.dumps({"hypothesis": "RUNNING"}))


def write_report(record):
    tmp = OUT / "bc_report.tmp.json"
    tmp.write_text(json.dumps(record))
    tmp.replace(OUT / "bc_report.json")


def teacher(env):
    raw = env.env._raw
    return DIVE if (env.exhausted or raw["char_level"] >= raw["dungeon_level"] + 2) else FARM


def rollout(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    done = trunc = False
    R, pairs = 0.0, []
    while not (done or trunc):
        m = env.action_masks()
        opt = _masked_action_or_first_legal(
            policy(env, obs, m),
            m,
            n_actions=3,
            label="BC manager rollout",
        )
        pairs.append((np.asarray(obs, dtype=np.float32), opt))
        obs, r, done, trunc, _ = env.step(opt)
        R += r
    return R, pairs


class MgrHead(nn.Module):
    def __init__(self, obs_dim=303, n_act=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh())
        self.head = nn.Linear(64, n_act)

    def forward(self, x):
        return self.head(self.net(x))


def main():
    provenance = artifact_provenance()
    begin_output_attempt()
    env = OptionsEnv(max_steps=3000)
    X, Y, rets = [], [], []
    for seed in DEMO_SEEDS:
        R, pairs = rollout(env, lambda e, o, m: teacher(e), seed)
        rets.append(R)
        for o, a in pairs:
            X.append(o); Y.append(a)
    t_mean = sum(rets) / len(rets)
    print(f"demos: {len(Y)} decision pairs, teacher mean return {t_mean:.1f} (demo pool)", flush=True)

    torch.manual_seed(22)
    X = torch.from_numpy(np.stack(X)); Y = torch.from_numpy(np.asarray(Y, dtype=np.int64))
    model = MgrHead(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y),
                                     batch_size=256, shuffle=True,
                                     generator=torch.Generator().manual_seed(22))
    for ep in range(10):
        tot = n = corr = 0
        for xb, yb in dl:
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(yb); n += len(yb)
            corr += int((logits.argmax(1) == yb).sum())
        print(f"BC ep{ep}: loss {tot/n:.4f} acc {corr/n:.3f}", flush=True)

    # replay (7000 pool, option-level memoryless hypothesis)
    model.eval()
    def bc_policy(e, o, m):
        with torch.no_grad():
            lg = model(torch.from_numpy(np.asarray(o, dtype=np.float32)).unsqueeze(0))[0]
            lg[~torch.from_numpy(m)] = -1e9
            return int(lg.argmax())
    replay = [rollout(env, bc_policy, s)[0] for s in REPLAY_SEEDS]
    # same-pool teacher baseline (fair ratio)
    t7000 = [rollout(env, lambda e, o, m: teacher(e), s)[0] for s in REPLAY_SEEDS]
    bc_mean, t7_mean = sum(replay) / 32, sum(t7000) / 32
    if t7_mean <= 0:
        raise RuntimeError(f"same-pool teacher mean return {t7_mean:.3f} <= 0; ratio gate undefined")
    ratio = bc_mean / t7_mean
    print(f"replay: BC {bc_mean:.1f} vs same-pool teacher {t7_mean:.1f} = {ratio:.2f}x (line 0.85)", flush=True)

    ok = ratio >= 0.85
    report = {
        "pairs": len(Y), "teacher_demo_mean": t_mean,
        "bc_replay_7000": bc_mean, "teacher_7000": t7_mean, "ratio": ratio,
        "hypothesis": "PASS" if ok else "FAIL", **provenance}
    if not ok:
        write_report(report)
        raise RuntimeError(
            f"memoryless-function gate FAIL (ratio={ratio:.3f}); refusing to overwrite policy_sd.pt")
    sd = {"mlp_extractor.policy_net.0.weight": model.net[0].weight,
          "mlp_extractor.policy_net.0.bias": model.net[0].bias,
          "mlp_extractor.policy_net.2.weight": model.net[2].weight,
          "mlp_extractor.policy_net.2.bias": model.net[2].bias,
          "action_net.weight": model.head.weight,
          "action_net.bias": model.head.bias}
    if artifact_provenance() != provenance:
        raise RuntimeError("implementation/engine/content drifted while BC manager was running")
    policy_tmp = OUT / "policy_sd.tmp.pt"
    torch.save({k: v.detach().clone() for k, v in sd.items()}, policy_tmp)
    policy_tmp.replace(OUT / "policy_sd.pt")
    report["policy_sha256"] = hashlib.sha256(
        (OUT / "policy_sd.pt").read_bytes()).hexdigest()
    write_report(report)
    env.close()
    print(f"saved {OUT}/policy_sd.pt", flush=True)


if __name__ == "__main__":
    with exclusive_lock(OUT / ".bc.lock", "BC manager artifacts"):
        main()
