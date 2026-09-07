"""R10 G0 探针:指令工资可辨性(经济 v1 vs v2 对照)。

R9 行为塌缩取证的直接对策——三个常数经理(恒 FARM / 恒 DIVE / 恒 RESUPPLY,
被掩时回退到合法项 argmax 固定序)在同一批工程种子上分别在 v1/v2 经济下
跑完整局。判据:v2 下 DIVE 常数策略的平均回报必须显著高于 FARM 常数策略
(v1 下两者近似相等正是塌缩根因)。只读探针:不写 repo 状态,输出 JSON。

用法: probe_r10_wage_discriminability.py <out.json> <seed_base> <n_seeds>
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

import leashed_ppo  # noqa: F401  policy class 注册
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

MODES = {"FARM": FARM, "DIVE": DIVE, "RESUPPLY": RESUPPLY}
FALLBACK_ORDER = (FARM, DIVE, RESUPPLY)

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
    for mode_name, mode in MODES.items():
        rows = []
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            done = trunc = False
            ret = 0.0
            forced = 0
            total = 0
            while not (done or trunc):
                mask = np.asarray(env.action_masks(), dtype=bool)
                if mask[mode]:
                    a = mode
                else:
                    forced += 1
                    a = next(x for x in FALLBACK_ORDER if mask[x])
                total += 1
                obs, r, done, trunc, _info = env.step(a)
                ret += float(r)
            raw = env.env._raw
            rows.append({
                "seed": seed, "ret": round(ret, 3),
                "depth": int(raw["dungeon_level"]),
                "died": bool(raw.get("dead")),
                "windows": total, "forced_fallbacks": forced,
            })
        rets = [r["ret"] for r in rows]
        depths = [r["depth"] for r in rows]
        results[f"{economy}:{mode_name}"] = {
            "mean_ret": round(float(np.mean(rets)), 3),
            "std_ret": round(float(np.std(rets)), 3),
            "mean_depth": round(float(np.mean(depths)), 3),
            "max_depth": int(max(depths)),
            "died": int(sum(r["died"] for r in rows)),
            "rows": rows,
        }
        print(f"{economy}:{mode_name}: mean_ret="
              f"{results[f'{economy}:{mode_name}']['mean_ret']} "
              f"mean_depth={results[f'{economy}:{mode_name}']['mean_depth']} "
              f"died={results[f'{economy}:{mode_name}']['died']}/{N_SEEDS}",
              flush=True)
    env.close()

for econ in ("v1", "v2"):
    gap = (results[f"{econ}:DIVE"]["mean_ret"]
           - results[f"{econ}:FARM"]["mean_ret"])
    results[f"{econ}:dive_minus_farm"] = round(gap, 3)
    print(f"{econ}: DIVE-FARM 回报差 = {gap:+.3f}", flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
print("WROTE", OUT, flush=True)
