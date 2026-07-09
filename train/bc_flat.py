"""v22 恶魔臂 F:平面 296 维 + spiral2 示范行为克隆。

用法:
  采集+训练+重放检查:.venv/bin/python train/bc_flat.py
产出:train/runs/bc-flat/policy_sd.pt(--bc-init 用)+ bc_report.json
示范种子 100-355(与探针 7000 段、评估 9000 段零交叉)。
教师 = spiral2 平面逻辑(神谕逐字 + 停滞钟驱动的榨干下楼)。
重放检查 = "spiral2 是否为 296 维观测的无记忆函数"的直接裁决(≥0.85×教师均值)。
"""
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import DiabloGymEnv, StagnationClockWrapper
from diablogym.options_env import KILL_PATIENCE, dispatch

OUT = ROOT / "train" / "runs" / "bc-flat"
OUT.mkdir(parents=True, exist_ok=True)
DEMO_SEEDS = list(range(100, 228))       # 128 局
REPLAY_SEEDS = list(range(7000, 7032))


def teacher_action(env_flat):
    """spiral2 平面教师:停滞钟≥140 → 11 下楼;否则神谕农/潜内环。"""
    raw = env_flat.env._raw
    clvl, dlvl = raw["char_level"], raw["dungeon_level"]
    if env_flat._clock >= KILL_PATIENCE:
        return 11
    mode = "dive" if clvl >= dlvl + 2 else "farm"
    return dispatch(mode, raw, bool(env_flat.env.action_masks()[14]))


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
    print(f"示范:{len(X)} 对,教师均回报 {sum(rets)/len(rets):.1f}", flush=True)
    return np.stack(X), np.asarray(Y, dtype=np.int64), sum(rets) / len(rets)


class PiHead(nn.Module):
    """与 SB3 MlpPolicy(64,64) 策略侧同构:mlp_extractor.policy_net + action_net。"""

    def __init__(self, obs_dim=296, n_act=15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh())
        self.head = nn.Linear(64, n_act)

    def forward(self, x):
        return self.head(self.net(x))


def train_bc(X, Y):
    torch.manual_seed(22)
    model = PiHead(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True)
    for epoch in range(8):
        tot = n = correct = 0
        for xb, yb in dl:
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(yb); n += len(yb)
            correct += int((logits.argmax(1) == yb).sum())
        print(f"BC epoch {epoch}: loss {tot/n:.4f} acc {correct/n:.3f}", flush=True)
    return model


def replay(model, teacher_mean):
    env = StagnationClockWrapper(DiabloGymEnv(
        ticks_per_step=4, max_steps=3000, start_in_dungeon=True,
        include_raw=False, descend_ladder=True, death_ladder=True))
    rets = []
    with torch.no_grad():
        for seed in REPLAY_SEEDS:
            obs, _ = env.reset(seed=seed)
            done = trunc = False
            R = 0.0
            while not (done or trunc):
                logits = model(torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0))[0]
                # 装备键掩码(与训练环境同规则:14 号仅在有可穿装备时合法)
                if not env.env.action_masks()[14]:
                    logits[14] = -1e9
                a = int(logits.argmax())
                obs, r, done, trunc, _ = env.step(a)
                R += r
            rets.append(R)
    mean = sum(rets) / len(rets)
    ratio = mean / teacher_mean if teacher_mean else 0
    print(f"重放:BC 均回报 {mean:.1f} = 教师的 {ratio:.2f} 倍(线 0.85)", flush=True)
    return mean, ratio


def export_sb3_sd(model):
    """映射到 SB3 MaskablePPO('MlpPolicy') 的 state_dict 键名(策略侧)。"""
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
    X, Y, teacher_mean = collect()
    model = train_bc(X, Y)
    bc_mean, ratio = replay(model, teacher_mean)
    torch.save(export_sb3_sd(model), OUT / "policy_sd.pt")
    (OUT / "bc_report.json").write_text(json.dumps({
        "pairs": len(Y), "teacher_mean_demo": teacher_mean,
        "bc_replay_mean_7000s": bc_mean, "ratio": ratio,
        "memoryless_hypothesis": "PASS" if ratio >= 0.85 else "FAIL",
    }))
    print(f"已存 {OUT}/policy_sd.pt;无记忆函数假设:{'PASS' if ratio >= 0.85 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
