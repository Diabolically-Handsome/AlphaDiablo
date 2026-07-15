"""v23:FARM 操作脑 BC 热启动(docs/PREREG-v23.md D4)。

在位采集:冻结 H 经理 + 脚本教师(dispatch farm 分支;反射拍是包装器所有,
天然不入集;保险丝强制拍整拍剔除)。示范种子 100-227,只录 FARM 窗口。
产出 train/runs/bc-worker/policy_sd.pt(SB3 键名,--bc-init 用)+ bc_report.json。
闸门 G1(数据侧):held-out top-1 ≥0.95;样本 ≥300 的类召回 ≥0.85
(不达标 → 类加权 CE 重训一次,BC 唯一重试)。
"""
import json
import hashlib
import pathlib
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import WorkerWindowEnv
from diablogym.options_env import dispatch
from eval_contract import PROTOCOL_VERSION, exclusive_lock
from train_ppo import (_BC_REPORT_SCHEMA_VERSION, _WORKER_BC_DEMO_SEEDS,
                       _WORKER_BC_FORBIDDEN_ACTIONS,
                       _implementation_bundle_sha256)

OUT = ROOT / "train" / "runs" / "bc-worker"
OUT.mkdir(parents=True, exist_ok=True)
NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
DEMO_SEEDS = list(_WORKER_BC_DEMO_SEEDS)


def artifact_provenance():
    return {
        "schema_version": _BC_REPORT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "implementation_sha256": _implementation_bundle_sha256(),
        "generator_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()).hexdigest(),
        "manager_npz_sha256": hashlib.sha256(NPZ.read_bytes()).hexdigest(),
    }


def begin_output_attempt():
    """去掉 canonical 旧产物，防止本次 FAIL 后下游误吃上次权重。"""
    old = [OUT / name for name in ("policy_sd.pt", "bc_report.json", "demos.npz")
           if (OUT / name).exists()]
    if old:
        archive = OUT / "_previous" / str(time.time_ns())
        archive.mkdir(parents=True)
        for path in old:
            path.replace(archive / path.name)
    (OUT / "bc_report.json").write_text(json.dumps({"data_gate": "RUNNING"}))


def write_report(record):
    tmp = OUT / "bc_report.tmp.json"
    tmp.write_text(json.dumps(record, ensure_ascii=False))
    tmp.replace(OUT / "bc_report.json")


def teacher_action(env: WorkerWindowEnv) -> int:
    raw = env.oe.env._raw
    return dispatch("farm", raw, bool(env.oe.env.action_masks()[14]))


def collect():
    env = WorkerWindowEnv(str(NPZ), max_steps=3000, rng_seed=0)
    X, Y, groups = [], [], []
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
                X.append(pair[0]); Y.append(pair[1]); groups.append(seed)
            # 局尽 next_window 返回 None(绝不滚新局——示范池纪律)
            obs = env.next_window() if (term or trunc) else obs2
        if (i + 1) % 16 == 0:
            print(f"  采集 {i+1}/{len(DEMO_SEEDS)} 局,{len(Y)} 对(剔除 {dropped})",
                  flush=True)
    # 示范池纪律断言:每个示范种子恰好一局,零兜底滚局(否则数据混入未知种子)
    if env.stats["episodes"] != len(DEMO_SEEDS) or env.stats["reseeds"] != 0:
        raise RuntimeError(f"示范池种子纪律破坏: {env.stats}")
    print(f"示范:{len(Y)} 决策对,剔除保险丝拍 {dropped},"
          f"类分布 {dict(sorted(Counter(Y).items()))}", flush=True)
    env.close()
    groups_array = np.asarray(groups, dtype=np.int64)
    labels = np.asarray(Y, dtype=np.int64)
    if not np.array_equal(np.unique(groups_array), np.asarray(DEMO_SEEDS)):
        raise RuntimeError("示范集没有精确覆盖固定种子 100..227")
    if np.isin(labels, _WORKER_BC_FORBIDDEN_ACTIONS).any():
        raise RuntimeError("示范集含禁采动作 11/12(11 恒掩归经理;12 系脚本教师"
                           "排水后不采——主权世代示范池纪律)")
    return np.stack(X), labels, groups_array


class PiHead(nn.Module):
    """与 SB3 MlpPolicy(64,64) 策略侧同构。"""

    def __init__(self, obs_dim=298, n_act=15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh())
        self.head = nn.Linear(64, n_act)

    def forward(self, x):
        return self.head(self.net(x))


def split_by_episode(groups):
    """整局切分，避免同一条确定性轨迹的相邻帧泄漏到 held-out。"""
    episodes = np.unique(groups)
    if len(episodes) < 2:
        raise ValueError("BC held-out 至少需要 2 个独立 episode")
    order = np.random.default_rng(23).permutation(episodes)
    n_holdout = max(1, int(round(len(order) * 0.1)))
    holdout_episodes = order[:n_holdout]
    ho_mask = np.isin(groups, holdout_episodes)
    tr, ho = np.flatnonzero(~ho_mask), np.flatnonzero(ho_mask)
    if len(tr) == 0 or len(ho) == 0:
        raise ValueError("BC episode split 产生了空训练集或空 held-out")
    return tr, ho, holdout_episodes


def train_bc(X, Y, groups, class_weights=None):
    torch.manual_seed(23)
    tr, ho, _ = split_by_episode(groups)
    model = PiHead(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    wt = None
    if class_weights is not None:
        wt = torch.as_tensor(class_weights, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(Y[tr]))
    dl = torch.utils.data.DataLoader(
        ds, batch_size=512, shuffle=True,
        generator=torch.Generator().manual_seed(23))
    for epoch in range(8):
        tot = cnt = correct = 0
        for xb, yb in dl:
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb, weight=wt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(yb); cnt += len(yb)
            correct += int((logits.argmax(1) == yb).sum())
        print(f"BC epoch {epoch}: loss {tot/cnt:.4f} acc {correct/cnt:.3f}", flush=True)
    # held-out 评分 + 逐类召回(门槛类 = 全集样本 ≥300 的类,召回在 held-out 上量——
    # 审查团修正:若按 held-out 内 ≥300 筛类,门槛被 10% 切片稀释十倍)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X[ho])).argmax(1).numpy()
    yh = Y[ho]
    top1 = float((pred == yh).mean())
    full_counts = Counter(Y.tolist())
    recalls = {}
    for c in sorted(k for k, v in full_counts.items() if v >= 300):
        m = yh == c
        # 门槛类若在 held-out 中零覆盖，必须 fail closed，不能从
        # recalls 字典中消失后被 all([]) 当成 PASS。
        recalls[int(c)] = (round(float((pred[m] == c).mean()), 3)
                           if m.sum() > 0 else 0.0)
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
    provenance = artifact_provenance()
    begin_output_attempt()
    X, Y, groups = collect()
    demos_tmp = OUT / "demos.tmp.npz"
    np.savez_compressed(demos_tmp, X=X, Y=Y, episode_id=groups)
    demos_tmp.replace(OUT / "demos.npz")
    tr, ho, holdout_episodes = split_by_episode(groups)
    model, top1, recalls = train_bc(X, Y, groups)
    retrained = False
    if top1 < 0.95 or any(r < 0.85 for r in recalls.values()):
        print(f"首训未达标(top1 {top1:.3f} 召回 {recalls})→ 类加权重训(唯一重试)",
              flush=True)
        counts = np.bincount(Y[tr], minlength=15).astype(np.float64)
        weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
        weights = weights / weights[weights > 0].mean()
        model, top1, recalls = train_bc(X, Y, groups, class_weights=weights)
        retrained = True
    ok = top1 >= 0.95 and all(r >= 0.85 for r in recalls.values())
    report = {
        "pairs": len(Y), "held_out_top1": round(top1, 4),
        "held_out_pairs": int(len(ho)),
        "held_out_episodes": [int(x) for x in sorted(holdout_episodes)],
        "class_recalls": recalls, "class_weighted_retry": retrained,
        "data_gate": "PASS" if ok else "FAIL", **provenance}
    if not ok:
        write_report(report)
        raise RuntimeError(
            f"BC 数据闸 FAIL(top1={top1:.3f}, recalls={recalls});"
            "拒绝覆写 policy_sd.pt")
    policy_tmp = OUT / "policy_sd.tmp.pt"
    if artifact_provenance() != provenance:
        raise RuntimeError("BC worker 运行期间实现/引擎/内容/经理发生漂移")
    torch.save(export_sb3_sd(model), policy_tmp)
    policy_tmp.replace(OUT / "policy_sd.pt")
    report["policy_sha256"] = hashlib.sha256(
        (OUT / "policy_sd.pt").read_bytes()).hexdigest()
    report["demos_sha256"] = hashlib.sha256(
        (OUT / "demos.npz").read_bytes()).hexdigest()
    write_report(report)
    print(f"held-out top-1 {top1:.3f} 召回 {recalls} retry={retrained} "
          f"→ 数据闸 PASS;已存 {OUT}/policy_sd.pt", flush=True)


if __name__ == "__main__":
    with exclusive_lock(OUT / ".bc.lock", "BC worker 产物"):
        main()
