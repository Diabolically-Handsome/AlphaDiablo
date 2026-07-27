"""G0-恒等 轨迹位级探针 · skip_dry=True 端点(内容案课⑤x④乙 E0;基线/重放两用)。

Rig 与 train/probe_g0_traj.py 逐字同型(全栈工人路径:WorkerWindowEnv +
king/throne npz argmax 逐拍),**唯一差异 = WorkerWindowEnv 构造显式传
skip_dry=True**(干层复访窗脚本内环代跑端点,G0-1 之 p≡1.0 对应端)。
skip_dry=False 端点仍由原脚本 + 原基线档(g0v32_traj_baseline*.json)承担;
本脚本零触碰原件(E0 铁律:变更前基线,禁改现存文件)。

常量(SEEDS/H_NPZ/KING_NPZ/THRONE_NPZ/f2hex)直接 import 原脚本,防镜像漂移;
episode_digest 系原文镜像,仅 env 构造行加 skip_dry=True。

用法:
  基线(E 施工前):.venv/bin/python train/probe_g0_traj_skip_dry.py baseline [throne]
  重放(E 施工后):.venv/bin/python train/probe_g0_traj_skip_dry.py replay [throne]
重放模式逐种子对比基线档,任何一位不同 → 退出码 1(G0 失败)。
"""
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from probe_g0_traj import (  # noqa: E402  原脚本常量单一真源
    H_NPZ, KING_NPZ, SEEDS, THRONE_NPZ, f2hex)
from diablogym.worker_env import NumpyManager, WorkerWindowEnv  # noqa: E402

OUT = ROOT / "docs" / "assets" / "g0v32_traj_baseline_skip_dry_true.json"
OUT_THRONE = ROOT / "docs" / "assets" / "g0v32_traj_baseline_throne_skip_dry_true.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def episode_digest(seed: int, npz=None) -> dict:
    net = NumpyManager(str(npz or KING_NPZ))
    net.require_worker_contract()
    env = WorkerWindowEnv(str(H_NPZ), max_steps=3000, rng_seed=0,
                          seed_scope="replay",
                          log_windows=True, skip_dry=True)   # ← 与原脚本唯一差异
    h = hashlib.sha256()
    wages = 0.0
    steps = 0
    obs, _ = env.reset(seed=seed)
    while obs is not None:
        masks = env.oe._worker_masks()
        policy_obs = env.oe._worker_policy_observation(
            net.worker_observation_view)
        a = net.choose_worker(
            policy_obs,
            masks,
            observation_view=net.worker_observation_view,
        )
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
        print(f"G0-恒等 skip_dry=True 失败({'throne' if throne else 'king'}):失配种子 {bad}")
        return 1
    print(f"G0-恒等 skip_dry=True PASS({'throne' if throne else 'king'}):{len(rows)} 种子逐位相同")
    return 0


if __name__ == "__main__":
    sys.exit(main())
