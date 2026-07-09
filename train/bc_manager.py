"""v22 保险臂 H-BC(P6 触发时发车):选项级教师示范 → 策略脑 BC 热启动。

教师 = probe_options 的 teacher(榨干旗或 clvl≥dlvl+2 → DIVE,否则 FARM)。
示范种子 100-227(与探针/评估池零交叉)。产出 policy_sd.pt 供
train_ppo --options --bc-init 使用;附重放检查(303 维无记忆假设,选项级)。
"""
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import OptionsEnv
from diablogym.options_env import DIVE, FARM

OUT = ROOT / "train" / "runs" / "bc-manager"
OUT.mkdir(parents=True, exist_ok=True)
DEMO_SEEDS = list(range(100, 228))
REPLAY_SEEDS = list(range(7000, 7032))


def teacher(env):
    raw = env.env._raw
    return DIVE if (env.exhausted or raw["char_level"] >= raw["dungeon_level"] + 2) else FARM


def rollout(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    done = trunc = False
    R, pairs = 0.0, []
    while not (done or trunc):
        m = env.action_masks()
        opt = policy(env, obs, m)
        if not m[opt]:
            opt = FARM
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
    env = OptionsEnv(max_steps=3000)
    X, Y, rets = [], [], []
    for seed in DEMO_SEEDS:
        R, pairs = rollout(env, lambda e, o, m: teacher(e), seed)
        rets.append(R)
        for o, a in pairs:
            X.append(o); Y.append(a)
    t_mean = sum(rets) / len(rets)
    print(f"示范:{len(Y)} 决策对,教师均回报 {t_mean:.1f}(示范池)", flush=True)

    torch.manual_seed(22)
    X = torch.from_numpy(np.stack(X)); Y = torch.from_numpy(np.asarray(Y, dtype=np.int64))
    model = MgrHead(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y),
                                     batch_size=256, shuffle=True)
    for ep in range(10):
        tot = n = corr = 0
        for xb, yb in dl:
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(yb); n += len(yb)
            corr += int((logits.argmax(1) == yb).sum())
        print(f"BC ep{ep}: loss {tot/n:.4f} acc {corr/n:.3f}", flush=True)

    # 重放(7000 池,选项级无记忆假设)
    def bc_policy(e, o, m):
        with torch.no_grad():
            lg = model(torch.from_numpy(np.asarray(o, dtype=np.float32)).unsqueeze(0))[0]
            lg[~torch.from_numpy(m)] = -1e9
            return int(lg.argmax())
    replay = [rollout(env, bc_policy, s)[0] for s in REPLAY_SEEDS]
    # 教师同池基准(公平比)
    t7000 = [rollout(env, lambda e, o, m: teacher(e), s)[0] for s in REPLAY_SEEDS]
    bc_mean, t7_mean = sum(replay) / 32, sum(t7000) / 32
    ratio = bc_mean / t7_mean if t7_mean else 0
    print(f"重放:BC {bc_mean:.1f} vs 同池教师 {t7_mean:.1f} = {ratio:.2f} 倍(线 0.85)", flush=True)

    sd = {"mlp_extractor.policy_net.0.weight": model.net[0].weight,
          "mlp_extractor.policy_net.0.bias": model.net[0].bias,
          "mlp_extractor.policy_net.2.weight": model.net[2].weight,
          "mlp_extractor.policy_net.2.bias": model.net[2].bias,
          "action_net.weight": model.head.weight,
          "action_net.bias": model.head.bias}
    torch.save({k: v.detach().clone() for k, v in sd.items()}, OUT / "policy_sd.pt")
    (OUT / "bc_report.json").write_text(json.dumps({
        "pairs": len(Y), "teacher_demo_mean": t_mean,
        "bc_replay_7000": bc_mean, "teacher_7000": t7_mean, "ratio": ratio,
        "hypothesis": "PASS" if ratio >= 0.85 else "FAIL"}))
    print(f"已存 {OUT}/policy_sd.pt", flush=True)


if __name__ == "__main__":
    main()
