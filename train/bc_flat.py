"""v22 devil arm F: behaviour cloning of spiral2 demonstrations on the flat 296-dim observation.

Usage:
  collect + train + replay check: .venv/bin/python train/bc_flat.py
Output: train/runs/bc-flat/policy_sd.pt (for --bc-init) + bc_report.json
Demo seeds 100-227 (disjoint from the 7000 probe block and the 9000 evaluation block).
Teacher = spiral2 flat logic (oracle verbatim + stall-clock-driven drain-then-descend).
Replay check = a direct test of "spiral2 is a memoryless function of the 296-dim observation" (>= 0.85 x teacher mean).
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

from diablogym import DiabloGymEnv, StagnationClockWrapper
from diablogym.options_env import KILL_PATIENCE, dispatch
from eval_contract import PROTOCOL_VERSION, exclusive_lock
from train_ppo import _BC_REPORT_SCHEMA_VERSION, _implementation_bundle_sha256

OUT = ROOT / "train" / "runs" / "bc-flat"
OUT.mkdir(parents=True, exist_ok=True)
DEMO_SEEDS = list(range(100, 228))       # 128 episodes
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
    (OUT / "bc_report.json").write_text(json.dumps({
        "memoryless_hypothesis": "RUNNING"}))


def write_report(record):
    tmp = OUT / "bc_report.tmp.json"
    tmp.write_text(json.dumps(record))
    tmp.replace(OUT / "bc_report.json")


def teacher_action(env_flat):
    """spiral2 flat teacher: stall clock >= 140 -> 11 descend; otherwise the oracle farm/dive inner loop."""
    raw = env_flat.env._raw
    clvl, dlvl = raw["char_level"], raw["dungeon_level"]
    if env_flat._clock >= KILL_PATIENCE:
        return 11
    mode = "dive" if clvl >= dlvl + 2 else "farm"
    masks, nearest = env_flat.env.controller_action_context()
    return dispatch(
        mode, raw, bool(masks[14]), action_mask=masks,
        nearest_engageable_distance=nearest)


def collect():
    env = StagnationClockWrapper(DiabloGymEnv(
        ticks_per_step=4, max_steps=3000, start_in_dungeon=True,
        include_raw=False, descend_ladder=True, death_ladder=True))
    X, Y, rets = [], [], []
    for seed in DEMO_SEEDS:
        obs, _ = env.reset(seed=seed)
        done = trunc = False
        R = 0.0
        while not (done or trunc):
            a = teacher_action(env)
            X.append(np.asarray(obs, dtype=np.float32))
            Y.append(a)
            obs, r, done, trunc, _ = env.step(a)
            R += r
        rets.append(R)
    print(f"demos: {len(X)} pairs, teacher mean return {sum(rets)/len(rets):.1f}", flush=True)
    env.close()
    return np.stack(X), np.asarray(Y, dtype=np.int64), sum(rets) / len(rets)


class PiHead(nn.Module):
    """Isomorphic to the policy side of SB3 MlpPolicy(64,64): mlp_extractor.policy_net + action_net."""

    def __init__(self, obs_dim=296, n_act=15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh())
        self.head = nn.Linear(64, n_act)

    def forward(self, x):
        return self.head(self.net(x))


def _masked_replay_action(model, observation, action_mask) -> int:
    """Apply the same complete 15-action mask used by deployed MaskablePPO."""
    valid = np.asarray(action_mask, dtype=bool)
    if valid.shape != (15,):
        raise RuntimeError(
            f"BC flat replay action mask shape invalid: {valid.shape} != (15,)")
    if not bool(valid.any()):
        raise RuntimeError("BC flat replay action mask is all False")
    logits = model(
        torch.from_numpy(
            np.asarray(observation, dtype=np.float32)).unsqueeze(0))[0]
    if tuple(logits.shape) != (15,):
        raise RuntimeError(
            f"BC flat replay policy output shape invalid: {tuple(logits.shape)} != (15,)")
    masked_logits = logits.masked_fill(
        ~torch.as_tensor(valid, dtype=torch.bool, device=logits.device),
        -torch.inf,
    )
    return int(masked_logits.argmax().item())


def train_bc(X, Y):
    torch.manual_seed(22)
    model = PiHead(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    dl = torch.utils.data.DataLoader(
        ds, batch_size=512, shuffle=True,
        generator=torch.Generator().manual_seed(22))
    for epoch in range(8):
        tot = n = correct = 0
        for xb, yb in dl:
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(yb); n += len(yb)
            correct += int((logits.argmax(1) == yb).sum())
        print(f"BC epoch {epoch}: loss {tot/n:.4f} acc {correct/n:.3f}", flush=True)
    return model


def replay(model):
    env = StagnationClockWrapper(DiabloGymEnv(
        ticks_per_step=4, max_steps=3000, start_in_dungeon=True,
        include_raw=False, descend_ladder=True, death_ladder=True))
    rets, teacher_rets = [], []
    model.eval()
    with torch.no_grad():
        for seed in REPLAY_SEEDS:
            obs, _ = env.reset(seed=seed)
            done = trunc = False
            R = 0.0
            while not (done or trunc):
                a = _masked_replay_action(
                    model, obs, env.env.action_masks())
                obs, r, done, trunc, _ = env.step(a)
                R += r
            rets.append(R)
    # The original code divided the teacher mean on the 100-227 demo pool by the BC mean on 7000-7031,
    # treating the seed-difficulty gap as policy loss. Now uses a same-pool, same-environment baseline.
    for seed in REPLAY_SEEDS:
        obs, _ = env.reset(seed=seed)
        done = trunc = False
        R = 0.0
        while not (done or trunc):
            obs, r, done, trunc, _ = env.step(teacher_action(env))
            R += r
        teacher_rets.append(R)
    mean = sum(rets) / len(rets)
    teacher_mean = sum(teacher_rets) / len(teacher_rets)
    if teacher_mean <= 0:
        raise RuntimeError(f"same-pool teacher mean return {teacher_mean:.3f} <= 0; ratio gate undefined")
    ratio = mean / teacher_mean
    print(f"replay: BC {mean:.1f} vs same-pool teacher {teacher_mean:.1f} "
          f"= {ratio:.2f}x (line 0.85)", flush=True)
    env.close()
    return mean, teacher_mean, ratio


def export_sb3_sd(model):
    """Map to the (policy-side) state_dict key names of SB3 MaskablePPO('MlpPolicy')."""
    sd = {
        "mlp_extractor.policy_net.0.weight": model.net[0].weight,
        "mlp_extractor.policy_net.0.bias": model.net[0].bias,
        "mlp_extractor.policy_net.2.weight": model.net[2].weight,
        "mlp_extractor.policy_net.2.bias": model.net[2].bias,
        "action_net.weight": model.head.weight,
        "action_net.bias": model.head.bias,
    }
    return {k: v.detach().clone() for k, v in sd.items()}


def main():
    provenance = artifact_provenance()
    begin_output_attempt()
    X, Y, teacher_mean = collect()
    model = train_bc(X, Y)
    bc_mean, teacher_replay_mean, ratio = replay(model)
    ok = ratio >= 0.85
    report = {
        "pairs": len(Y), "teacher_mean_demo": teacher_mean,
        "bc_replay_mean_7000s": bc_mean, "teacher_replay_mean_7000s": teacher_replay_mean,
        "ratio": ratio, "memoryless_hypothesis": "PASS" if ok else "FAIL",
        **provenance,
    }
    if not ok:
        write_report(report)
        raise RuntimeError(
            f"memoryless-function gate FAIL (ratio={ratio:.3f}); refusing to overwrite policy_sd.pt")
    policy_tmp = OUT / "policy_sd.tmp.pt"
    if artifact_provenance() != provenance:
        raise RuntimeError("implementation/engine/content drifted while BC flat was running")
    torch.save(export_sb3_sd(model), policy_tmp)
    policy_tmp.replace(OUT / "policy_sd.pt")
    report["policy_sha256"] = hashlib.sha256(
        (OUT / "policy_sd.pt").read_bytes()).hexdigest()
    write_report(report)
    print(f"saved {OUT}/policy_sd.pt; memoryless-function hypothesis: PASS", flush=True)


if __name__ == "__main__":
    with exclusive_lock(OUT / ".bc.lock", "BC flat artifacts"):
        main()
