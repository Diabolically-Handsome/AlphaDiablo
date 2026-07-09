"""v23:FARM 操作脑 BC 热启动(docs/PREREG-v23.md D4)。

在位采集:冻结 H 经理 + 脚本教师(dispatch farm 分支;反射拍是包装器所有,
天然不入集;保险丝强制拍整拍剔除)。示范种子 100-227,只录 FARM 窗口。
产出 train/runs/bc-worker/policy_sd.pt(SB3 键名,--bc-init 用)+ bc_report.json。
闸门 G1(数据侧):held-out top-1 ≥0.95;样本 ≥300 的类召回 ≥0.85
(不达标 → 类加权 CE 重训一次,BC 唯一重试)。
"""
import json
import pathlib
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import WorkerWindowEnv
from diablogym.options_env import dispatch

OUT = ROOT / "train" / "runs" / "bc-worker"
OUT.mkdir(parents=True, exist_ok=True)
NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
DEMO_SEEDS = list(range(100, 228))


def teacher_action(env: WorkerWindowEnv) -> int:
    raw = env.oe.env._raw
    return dispatch("farm", raw, bool(env.oe.env.action_masks()[14]))


def collect():
    env = WorkerWindowEnv(str(NPZ), max_steps=3000, rng_seed=0)
    X, Y = [], []
    dropped = 0
    for i, seed in enumerate(DEMO_SEEDS):
        obs, _ = env.reset(seed=seed)
        while obs is not None:
            a = teacher_action(env)
            pair = (np.asarray(obs, dtype=np.float32), a)
            obs2, w, term, trunc, info = env.step(a)
            if info.get("overridden"):
                dropped += 1          # 保险丝改写过的步整拍剔除
            else:
                X.append(pair[0]); Y.append(pair[1])
            # 局尽 next_window 返回 None(绝不滚新局——示范池纪律)
            obs = env.next_window() if (term or trunc) else obs2
        if (i + 1) % 16 == 0:
            print(f"  采集 {i+1}/{len(DEMO_SEEDS)} 局,{len(Y)} 对(剔除 {dropped})",
                  flush=True)
    # 示范池纪律断言:每个示范种子恰好一局,零兜底滚局(否则数据混入未知种子)
    assert env.stats["episodes"] == len(DEMO_SEEDS), env.stats
    assert env.stats["reseeds"] == 0, env.stats
    print(f"示范:{len(Y)} 决策对,剔除保险丝拍 {dropped},"
          f"类分布 {dict(sorted(Counter(Y).items()))}", flush=True)
    return np.stack(X), np.asarray(Y, dtype=np.int64)


class PiHead(nn.Module):
    """与 SB3 MlpPolicy(64,64) 策略侧同构。"""

    def __init__(self, obs_dim=298, n_act=15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh())
        self.head = nn.Linear(64, n_act)

    def forward(self, x):
        return self.head(self.net(x))


def train_bc(X, Y, class_weights=None):
    torch.manual_seed(23)
    n = len(Y)
    idx = np.random.default_rng(23).permutation(n)
    cut = int(n * 0.9)
    tr, ho = idx[:cut], idx[cut:]
    model = PiHead(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    wt = None
    if class_weights is not None:
        wt = torch.as_tensor(class_weights, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(Y[tr]))
    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True)
    for epoch in range(8):
        tot = cnt = correct = 0
        for xb, yb in dl:
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb, weight=wt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(yb); cnt += len(yb)
            correct += int((logits.argmax(1) == yb).sum())
        print(f"BC epoch {epoch}: loss {tot/cnt:.4f} acc {correct/cnt:.3f}", flush=True)
    # held-out 评分 + 逐类召回(门槛类 = 全集样本 ≥300 的类,召回在 held-out 上量——
    # 审查团修正:若按 held-out 内 ≥300 筛类,门槛被 10% 切片稀释十倍)
    with torch.no_grad():
        pred = model(torch.from_numpy(X[ho])).argmax(1).numpy()
    yh = Y[ho]
    top1 = float((pred == yh).mean())
    full_counts = Counter(Y.tolist())
    recalls = {}
    for c in sorted(k for k, v in full_counts.items() if v >= 300):
        m = yh == c
        if m.sum() > 0:
            recalls[int(c)] = round(float((pred[m] == c).mean()), 3)
    return model, top1, recalls


def export_sb3_sd(model):
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
    X, Y = collect()
    np.savez_compressed(OUT / "demos.npz", X=X, Y=Y)
    model, top1, recalls = train_bc(X, Y)
    retrained = False
    if top1 < 0.95 or any(r < 0.85 for r in recalls.values()):
        print(f"首训未达标(top1 {top1:.3f} 召回 {recalls})→ 类加权重训(唯一重试)",
              flush=True)
        counts = np.bincount(Y, minlength=15).astype(np.float64)
        weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
        weights = weights / weights[weights > 0].mean()
        model, top1, recalls = train_bc(X, Y, class_weights=weights)
        retrained = True
    ok = top1 >= 0.95 and all(r >= 0.85 for r in recalls.values())
    torch.save(export_sb3_sd(model), OUT / "policy_sd.pt")
    (OUT / "bc_report.json").write_text(json.dumps({
        "pairs": len(Y), "held_out_top1": round(top1, 4),
        "class_recalls": recalls, "class_weighted_retry": retrained,
        "data_gate": "PASS" if ok else "FAIL"}, ensure_ascii=False))
    print(f"held-out top-1 {top1:.3f} 召回 {recalls} retry={retrained} "
          f"→ 数据闸 {'PASS' if ok else 'FAIL'};已存 {OUT}/policy_sd.pt", flush=True)


if __name__ == "__main__":
    main()
