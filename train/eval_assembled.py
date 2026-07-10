"""v23 组装体评测:冻结 H 经理 + {脚本|BC|PPO} FARM 工人(docs/PREREG-v23.md)。

用法:
  H7 基线:  .venv/bin/python train/eval_assembled.py --worker script --seeds 7000-7031
  G0'' 回归:… --worker script --seeds 7000-7031 --check-probe docs/assets/window_econ_v23_probe.json
  G1 BC 重放:… --worker bc --seeds 7000-7031
  G3 初筛:  … --worker train/runs/<run>/ckpt/model_XXX_steps --seeds 7000-7015
  金评(唯一一次):… --worker <胜者> --seeds 9000-9031 --board
协议:argmax(经理 numpy 前向 = G0' 位级对账过的同一段代码)、3000 微步、
回报 = 经理不折现账本。R4 哨兵(换层率/override/cap/τ̄)一并产出。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import NumpyManager, OptionsEnv
from diablogym.options_env import FARM

NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
LB = ROOT / "train" / "leaderboard-hier.md"
OUTDIR = ROOT / "train" / "runs" / "eval-assembled"


def np_policy_from_sd(sd_path: str) -> NumpyManager:
    import torch
    sd = torch.load(sd_path, map_location="cpu")
    m = NumpyManager.__new__(NumpyManager)
    m.w0 = sd["mlp_extractor.policy_net.0.weight"].numpy().astype(np.float32)
    m.b0 = sd["mlp_extractor.policy_net.0.bias"].numpy().astype(np.float32)
    m.w1 = sd["mlp_extractor.policy_net.2.weight"].numpy().astype(np.float32)
    m.b1 = sd["mlp_extractor.policy_net.2.bias"].numpy().astype(np.float32)
    m.wa = sd["action_net.weight"].numpy().astype(np.float32)
    m.ba = sd["action_net.bias"].numpy().astype(np.float32)
    return m


def load_worker(spec: str):
    """返回 (workers dict 或 None, 标签)。spec ∈ script | bc | *.npz | SB3 zip。"""
    if spec == "script":
        return None, "script"
    if spec == "bc":
        net = np_policy_from_sd(str(ROOT / "train" / "runs" / "bc-worker" / "policy_sd.pt"))
        return {FARM: lambda obs, mask: net.choose(obs, mask)}, "bc"
    if spec.endswith(".npz"):
        net = NumpyManager(spec)      # 通用 MLP 前向,298→15 同构可用(v25 G-A0)
        return {FARM: lambda obs, mask: net.choose(obs, mask)}, pathlib.Path(spec).parent.name
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(spec, device="cpu")

    def w(obs, mask):
        a, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(a)
    return {FARM: w}, pathlib.Path(spec).stem


def parse_seeds(s: str):
    lo, hi = s.split("-")
    return list(range(int(lo), int(hi) + 1))


def evaluate(workers, seeds, manager_npz=None):
    mgr = NumpyManager(str(manager_npz or NPZ))
    env = OptionsEnv(max_steps=3000, workers=workers)
    engage = None
    if workers:
        # 参与度取证(2026-07-10 法证会审的后续):调用数/动作直方/与脚本分歧率,
        # 让评测文件自带"worker 真的在开车"的证据,顺带量出 PPO 漂离教师的距离
        from diablogym.options_env import dispatch
        inner = workers[FARM]
        engage = {"calls": 0, "hist": Counter(), "diverge": 0}

        def instrumented(obs, mask, _inner=inner):
            a = int(_inner(obs, mask))
            engage["calls"] += 1
            engage["hist"][int(a)] += 1
            s = dispatch("farm", env.env._raw, bool(env.env.action_masks()[14]))
            if a != s:
                engage["diverge"] += 1
            return a

        workers[FARM] = instrumented   # env 持同一 dict 引用,原地替换生效
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = trunc = False
        R = 0.0
        farm = Counter()
        allw = Counter()
        seq = ""
        while not (done or trunc):
            opt = mgr.choose(obs, env.action_masks())
            obs, r, done, trunc, info = env.step(opt)
            R += float(r)
            oe = info["option_extra"]
            allw["n"] += 1
            allw["beats"] += oe["beats"]
            allw["overrides"] += oe["overrides"]
            allw["cap"] += oe["reason"] == "cap"
            if oe["opt"] == FARM:
                farm["n"] += 1
                farm["tau"] += oe["tau"]
                farm["descend"] += oe["reason"] == "descend"
            seq = oe["mode_seq"]
        raw = env.env._raw
        rows.append({
            "seed": seed, "ret": round(R, 2), "depth": raw["dungeon_level"],
            "died": bool(raw.get("dead")), "kills": env.env._ep_kills,
            "farm_n": farm["n"], "farm_tau_mean": round(farm["tau"] / max(1, farm["n"]), 1),
            "farm_descend": farm["descend"], "windows": allw["n"],
            "beats": allw["beats"], "overrides": allw["overrides"], "cap": allw["cap"],
            "mode_seq": seq,
        })
        print(f"  seed {seed}: ret {R:.1f} depth {raw['dungeon_level']} "
              f"died {bool(raw.get('dead'))} farmτ̄ {rows[-1]['farm_tau_mean']}", flush=True)
    return rows, engage


def digest(rows):
    n = len(rows)
    rets = sorted(r["ret"] for r in rows)
    farm_n = sum(r["farm_n"] for r in rows)
    return {
        "n": n,
        "ret_mean": round(sum(rets) / n, 1),
        "ret_median": round(statistics.median(rets), 2),
        "died": sum(r["died"] for r in rows),
        "depth_median": statistics.median(r["depth"] for r in rows),
        "l3": sum(r["depth"] >= 3 for r in rows),
        "kills_mean": round(sum(r["kills"] for r in rows) / n, 1),
        "farm_tau_mean": round(sum(r["farm_n"] * r["farm_tau_mean"] for r in rows)
                               / max(1, farm_n), 1),
        "farm_descend_rate": round(sum(r["farm_descend"] for r in rows) / max(1, farm_n), 4),
        "override_rate": round(sum(r["overrides"] for r in rows)
                               / max(1, sum(r["beats"] for r in rows)), 4),
        "cap_rate": round(sum(r["cap"] for r in rows) / max(1, sum(r["windows"] for r in rows)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True, help="script | bc | *.npz | ckpt 路径")
    ap.add_argument("--manager-npz", default=None,
                    help="v25:经理 npz(默认 v22-h——旧档回归口径不变)")
    ap.add_argument("--seeds", default="7000-7031")
    ap.add_argument("--board", action="store_true", help="金评后写 leaderboard-hier.md")
    ap.add_argument("--check-probe", default=None,
                    help="G0'':对 v23 前探针存档逐种子回归(ret±0.6/depth/died 全等)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    workers, label = load_worker(args.worker)
    seeds = parse_seeds(args.seeds)
    rows, engage = evaluate(workers, seeds, manager_npz=args.manager_npz)
    agg = digest(rows)
    if engage:
        agg["worker_calls"] = engage["calls"]
        agg["worker_action_hist"] = dict(sorted(engage["hist"].items()))
        agg["script_divergence_rate"] = round(engage["diverge"] / max(1, engage["calls"]), 4)
        print(f"  参与度:worker 调用 {engage['calls']},动作直方 {agg['worker_action_hist']},"
              f"与脚本分歧率 {agg['script_divergence_rate']}")
    tag = args.tag or f"{label}-{args.seeds}"
    print(f"{tag}: ret {agg['ret_mean']} (med {agg['ret_median']}) died {agg['died']}/{agg['n']} "
          f"depth_med {agg['depth_median']} | R4: 换层率 {agg['farm_descend_rate']} "
          f"override {agg['override_rate']} cap {agg['cap_rate']} farmτ̄ {agg['farm_tau_mean']}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"{tag}.json").write_text(json.dumps(
        {"agg": agg, "rows": rows}, ensure_ascii=False, indent=1))
    print(f"已存 {OUTDIR}/{tag}.json")

    if args.check_probe:
        ref = json.loads(pathlib.Path(args.check_probe).read_text())["argmax_episodes"]
        ref = {e["seed"]: e for e in ref}
        bad = []
        for r in rows:
            e = ref.get(r["seed"])
            if e is None:
                continue
            if (abs(r["ret"] - e["ep_R"]) > 0.6 or r["depth"] != e["depth"]
                    or r["died"] != e["died"]):
                bad.append((r["seed"], r["ret"], e["ep_R"], r["depth"], e["depth"]))
        print(f"G0'' 回归:{len(rows) - len(bad)}/{len(rows)} 一致"
              + (f";失配 {bad}" if bad else " —— PASS"))
        if bad:
            raise SystemExit(1)

    if args.board:
        lines = LB.read_text().splitlines(keepends=True)
        last_row = max(i for i, l in enumerate(lines) if l.startswith("|"))
        lines.insert(last_row + 1,
                     f"| {tag} | {agg['ret_mean']} | {agg['ret_median']} | "
                     f"{agg['died']}/{agg['n']} | {agg['depth_median']} | "
                     f"hier+learned-FARM; L3+ {agg['l3']}; kills {agg['kills_mean']}; "
                     f"换层率 {agg['farm_descend_rate']} override {agg['override_rate']} |\n")
        LB.write_text("".join(lines))
        print(f"已写入 {LB.name}")


if __name__ == "__main__":
    main()
