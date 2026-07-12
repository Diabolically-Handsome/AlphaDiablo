"""v22 恶魔臂 F:平面 296 维 + spiral2 示范行为克隆。

用法:
  采集+训练+重放检查:.venv/bin/python train/bc_flat.py
产出:train/runs/bc-flat/policy_sd.pt(--bc-init 用)+ bc_report.json
示范种子 100-227(与探针 7000 段、评估 9000 段零交叉)。
教师 = spiral2 平面逻辑(神谕逐字 + 停滞钟驱动的榨干下楼)。
重放检查 = "spiral2 是否为 296 维观测的无记忆函数"的直接裁决(≥0.85×教师均值)。
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
DEMO_SEEDS = list(range(100, 228))       # 128 局
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
    env.close()
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
                logits = model(torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0))[0]
                # 装备键掩码(与训练环境同规则:14 号仅在有可穿装备时合法)
                if not env.env.action_masks()[14]:
                    logits[14] = -1e9
                a = int(logits.argmax())
                obs, r, done, trunc, _ = env.step(a)
                R += r
            rets.append(R)
    # 原实现用 100-227 示范池教师均值除 7000-7031 BC 均值，
    # 把种子难度差当成策略损失。改为同池、同环境基准。
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
        raise RuntimeError(f"同池教师均回报 {teacher_mean:.3f} <= 0，比值闸无定义")
    ratio = mean / teacher_mean
    print(f"重放:BC {mean:.1f} vs 同池教师 {teacher_mean:.1f} "
          f"= {ratio:.2f} 倍(线 0.85)", flush=True)
    env.close()
    return mean, teacher_mean, ratio


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
            f"无记忆函数闸 FAIL(ratio={ratio:.3f});拒绝覆写 policy_sd.pt")
    policy_tmp = OUT / "policy_sd.tmp.pt"
    if artifact_provenance() != provenance:
        raise RuntimeError("BC flat 运行期间实现/引擎/内容发生漂移")
    torch.save(export_sb3_sd(model), policy_tmp)
    policy_tmp.replace(OUT / "policy_sd.pt")
    report["policy_sha256"] = hashlib.sha256(
        (OUT / "policy_sd.pt").read_bytes()).hexdigest()
    write_report(report)
    print(f"已存 {OUT}/policy_sd.pt;无记忆函数假设:PASS", flush=True)


if __name__ == "__main__":
    with exclusive_lock(OUT / ".bc.lock", "BC flat 产物"):
        main()
