"""R10 G0 探针 rev3:审时度势下潜(SMART)vs 恒 FARM,v1/v2 对照。

修正 rev2 的门槛定义错误:恒 DIVE 是自杀稻草人(基座555 死31/32),
把它抬成冠军会立"冲楼梯自杀"套利。真实命题 = 经济 v2 下,仓库既有的
教师式启发(榨干 或 char_level>=dungeon_level+2 时 DIVE,否则 FARM)
应显著跑赢恒 FARM;v1 下应≈0或为负(塌缩根因复现)。

用法: probe_r10_smart_dive.py <out.json> <seed_base> <n_seeds>
与 wage-discriminability-g0-rev2b.json 同种子可直接配对。
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path.home() / "AlphaDiablo" / "diablogym"
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))

OUT = pathlib.Path(sys.argv[1])
SEED_BASE = int(sys.argv[2])
N_SEEDS = int(sys.argv[3])
LEVEL_MARGIN = int(sys.argv[4]) if len(sys.argv) > 4 else 2

import leashed_ppo  # noqa: F401
from sb3_contrib import MaskablePPO
from diablogym import OptionsEnv
from diablogym.options_env import FARM, DIVE, RESUPPLY
from diablogym.worker_env import is_reserved_train_seed

WORKER_ZIP = ROOT / "train" / "runs" / "r9-reeducation" / "staging" / "worker.zip"

seeds = [SEED_BASE + i for i in range(N_SEEDS)]
bad = [s for s in seeds if is_reserved_train_seed(s)]
if bad:
    raise SystemExit(f"种子纪律:{bad} 撞已登记池,换基座")

model = MaskablePPO.load(str(WORKER_ZIP), device="cpu")
contract = model.diablogym_contract
view = contract["worker_policy_observation_view"]
sovereignty = bool(contract["drink_sovereignty"])


def w(policy_obs, mask):
    policy_mask = mask
    if not sovereignty:
        policy_mask = np.asarray(mask, dtype=bool).copy()
        policy_mask[12] = False
    a, _ = model.predict(policy_obs, action_masks=policy_mask,
                         deterministic=True)
    return int(a)


w.diablogym_worker_observation_view = view
w.diablogym_worker_action12_mode = (
    "environment-mask" if sovereignty else "permanently-masked")

FALLBACK_ORDER = (FARM, DIVE, RESUPPLY)


def smart_choice(env, mask):
    raw = env.env._raw
    want = (DIVE if (env.exhausted
                     or raw["char_level"]
                     >= raw["dungeon_level"] + LEVEL_MARGIN)
            else FARM)
    if mask[want]:
        return want
    return next(x for x in FALLBACK_ORDER if mask[x])


results = {}
for economy in ("v1", "v2"):
    env = OptionsEnv(
        max_steps=3000,
        workers={0: w},
        drink_sovereignty=sovereignty,
        worker_observation_view=view,
        manager_observation_view="raw-v4",
        reward_economy=economy,
    )
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = trunc = False
        ret = 0.0
        dives = 0
        windows = 0
        while not (done or trunc):
            mask = np.asarray(env.action_masks(), dtype=bool)
            a = smart_choice(env, mask)
            dives += int(a == DIVE)
            windows += 1
            obs, r, done, trunc, _info = env.step(a)
            ret += float(r)
        raw = env.env._raw
        rows.append({"seed": seed, "ret": round(ret, 3),
                     "depth": int(raw["dungeon_level"]),
                     "died": bool(raw.get("dead")),
                     "windows": windows, "dive_windows": dives})
    rets = [r["ret"] for r in rows]
    depths = [r["depth"] for r in rows]
    results[f"{economy}:SMART"] = {
        "mean_ret": round(float(np.mean(rets)), 3),
        "mean_depth": round(float(np.mean(depths)), 3),
        "max_depth": int(max(depths)),
        "died": int(sum(r["died"] for r in rows)),
        "rows": rows,
    }
    print(f"{economy}:SMART mean_ret={results[f'{economy}:SMART']['mean_ret']} "
          f"mean_depth={results[f'{economy}:SMART']['mean_depth']} "
          f"max_depth={results[f'{economy}:SMART']['max_depth']} "
          f"died={results[f'{economy}:SMART']['died']}/{N_SEEDS}", flush=True)
    env.close()

ref_path = ROOT / "train" / "runs" / "r10-staging" / (
    "wage-discriminability-g0-rev2b.json")
if ref_path.is_file():
    ref = json.load(open(ref_path))
    for econ in ("v1", "v2"):
        farm = {r["seed"]: r["ret"] for r in ref[f"{econ}:FARM"]["rows"]}
        smart = {r["seed"]: r["ret"]
                 for r in results[f"{econ}:SMART"]["rows"]}
        common = sorted(set(farm) & set(smart))
        if common:
            diffs = [smart[s] - farm[s] for s in common]
            wins = sum(x > 0 for x in diffs)
            results[f"{econ}:smart_minus_farm"] = {
                "mean": round(float(np.mean(diffs)), 3),
                "median": round(float(np.median(diffs)), 3),
                "wins": f"{wins}/{len(common)}",
            }
            print(f"{econ}: SMART-FARM 配对 mean={np.mean(diffs):+.2f} "
                  f"median={np.median(diffs):+.2f} wins={wins}/{len(common)}",
                  flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
print("WROTE", OUT, flush=True)
