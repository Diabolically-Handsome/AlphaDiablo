"""G0-恒等 轨迹位级探针(PREREG-v32 W-G0;基线/重放两用)。

Rig = 全栈工人路径:WorkerWindowEnv(H 经理 npz)驱动窗口,king npz 工人
argmax(经 _worker_masks)逐拍执行——恰是 E1 触碰的掩码路径。seeds
7000-7015 各一局;哈希 = sha256(逐拍 [动作, 工资 hex, term, trunc] +
逐窗边界 + 终局摘要);工资恒等式 W ≡ R − bonus 逐窗断言 + 逐局合账断言
(快进窗一并入账,log_windows 全窗序)。基线档随冻结 commit 入库
(docs/assets,gitignore 之外)。

用法:
  基线(E1 施工前):.venv/bin/python train/probe_g0_traj.py baseline
  重放(E1 施工后):.venv/bin/python train/probe_g0_traj.py replay
重放模式逐种子对比 baseline.json,任何一位不同 → 退出码 1(G0 失败)。
"""
import hashlib
import json
import pathlib
import struct
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym.worker_env import NumpyManager, WorkerWindowEnv  # noqa: E402

SEEDS = list(range(7000, 7016))
H_NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
KING_NPZ = ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz"
THRONE_NPZ = ROOT / "train" / "models" / "v24-worker-leg7" / "policy.npz"
OUT = ROOT / "docs" / "assets" / "g0v32_traj_baseline.json"
OUT_THRONE = ROOT / "docs" / "assets" / "g0v32_traj_baseline_throne.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
DESCEND_UNIT = 8.0


def f2hex(x: float) -> str:
    return struct.pack("<d", float(x)).hex()


def episode_digest(seed: int, npz=None) -> dict:
    net = NumpyManager(str(npz or KING_NPZ))
    env = WorkerWindowEnv(str(H_NPZ), max_steps=3000, rng_seed=0,
                          log_windows=True)
    h = hashlib.sha256()
    wages = 0.0
    steps = 0
    obs, _ = env.reset(seed=seed)
    while obs is not None:
        masks = env.oe._worker_masks()
        logits = np.where(masks, net.logits(np.asarray(obs, np.float32)), -1e9)
        a = int(np.argmax(logits))
        obs2, w, term, trunc, info = env.step(a)
        h.update(f"{a},{f2hex(w)},{int(term)},{int(trunc)};".encode())
        wages += float(w)
        steps += 1
        if term or trunc:
            ex = info.get("option_extra")
            if ex is not None:
                assert abs(ex["W"] - (ex["R"] - ex["bonus"])) < 1e-6, (
                    f"工资恒等式破裂:seed {seed} W={ex['W']} "
                    f"R={ex['R']} bonus={ex['bonus']}")
            h.update(b"|WIN|")
            obs = env.next_window()
        else:
            obs = obs2
    tot = env.window_log
    assert abs(sum(w["W"] for w in tot)
               - (sum(w["R"] for w in tot) - sum(w["bonus"] for w in tot))) < 1e-5, \
        f"逐局工资恒等式破裂:seed {seed}"
    raw = env.oe.env._raw
    final = (f"d{raw['dungeon_level']},dead{int(raw['dead'])},"
             f"hp{raw['hp']},xp{raw.get('experience', raw.get('xp', 0))}")
    h.update(final.encode())
    stats = dict(env.stats)
    env.close()
    return {"seed": seed, "sha": h.hexdigest(), "steps": steps,
            "wages": round(wages, 6), "final": final, "stats": stats}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    throne = len(sys.argv) > 2 and sys.argv[2] == "throne"
    npz = THRONE_NPZ if throne else KING_NPZ
    out = OUT_THRONE if throne else OUT
    rows = [episode_digest(s, npz) for s in SEEDS]
    for r in rows:
        print(f"  seed {r['seed']}: {r['sha'][:16]} steps {r['steps']} "
              f"wages {r['wages']}", flush=True)
    if mode == "baseline":
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"基线已存 {out}")
        return 0
    base = {r["seed"]: r for r in json.loads(out.read_text())}
    bad = [r["seed"] for r in rows
           if r["sha"] != base[r["seed"]]["sha"]
           or r["steps"] != base[r["seed"]]["steps"]]
    if bad:
        print(f"G0-恒等 失败({'throne' if throne else 'king'}):失配种子 {bad}")
        return 1
    print(f"G0-恒等 PASS({'throne' if throne else 'king'}):{len(rows)} 种子逐位相同")
    return 0


if __name__ == "__main__":
    sys.exit(main())
