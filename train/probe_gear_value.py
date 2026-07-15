"""③A 装备价值诊断探针(DESIGN-课③④ 批件;研究探针,零协议触碰)。

配对反事实:H 经理 × 脚本工人,变体 A = 原装 dispatch,变体 B = 剥夺
gear_available(monkeypatch,强制 False)。7000-7031 逐种子配对,测
甲的因果价值曲线与 a14 机会频次。产出 JSON + 汇总表;结论进设计文书
附录/法证,不进任何评测档案(eval-assembled 零写入)。
用法:.venv/bin/python train/probe_gear_value.py
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import OptionsEnv                    # noqa: E402
from diablogym import options_env as oe_mod         # noqa: E402
from diablogym.worker_env import NumpyManager       # noqa: E402

SEEDS = list(range(7000, 7032))
H_NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
OUT = ROOT / "train" / "runs" / "probe-gear-value"
OUT.mkdir(parents=True, exist_ok=True)

_orig_dispatch = oe_mod.dispatch
_counter = {"a14": 0, "opps": 0}


def _spy_dispatch(mode, raw, gear_available):
    if gear_available:
        _counter["opps"] += 1
    a = _orig_dispatch(mode, raw, gear_available)
    if a == 14:
        _counter["a14"] += 1
    return a


def _deprived_dispatch(mode, raw, gear_available):
    if gear_available:
        _counter["opps"] += 1
    return _orig_dispatch(mode, raw, False)


def run_variant(name, dispatch_fn):
    oe_mod.dispatch = dispatch_fn
    mgr = NumpyManager(str(H_NPZ))
    env = OptionsEnv(max_steps=3000)
    rows = []
    try:
        for seed in SEEDS:
            _counter["a14"] = _counter["opps"] = 0
            obs, _ = env.reset(seed=seed)
            done = trunc = False
            ret = 0.0
            while not (done or trunc):
                m = env.action_masks()
                logits = mgr.logits(np.asarray(obs, dtype=np.float32))
                logits = np.where(m, logits, -1e9)
                obs, r, done, trunc, _info = env.step(int(np.argmax(logits)))
                ret += r
            raw = env.env._raw
            rows.append({
                "seed": seed, "ret": round(ret, 2),
                "depth": int(raw["dungeon_level"]), "died": bool(raw["dead"]),
                "a14": _counter["a14"], "opps": _counter["opps"],
                "ac": int(raw.get("armor_class", -1)),
            })
            print(f"  [{name}] seed {seed}: ret {ret:7.2f} depth "
                  f"{raw['dungeon_level']} died {raw['dead']} "
                  f"a14 {_counter['a14']}/{_counter['opps']}", flush=True)
    finally:
        env.close()
        oe_mod.dispatch = _orig_dispatch
    return rows


def main():
    armed = run_variant("armed", _spy_dispatch)
    deprived = run_variant("deprived", _deprived_dispatch)
    by = {r["seed"]: r for r in armed}
    diffs = [(by[d["seed"]]["ret"] - d["ret"]) for d in deprived]
    per_depth = {}
    for d in deprived:
        a = by[d["seed"]]
        key = max(a["depth"], d["depth"])
        per_depth.setdefault(key, []).append(a["ret"] - d["ret"])
    summary = {
        "paired_mean_armed_minus_deprived": round(float(np.mean(diffs)), 2),
        "wins_armed": int(sum(x > 0 for x in diffs)),
        "per_depth_paired_mean": {k: [round(float(np.mean(v)), 2), len(v)]
                                  for k, v in sorted(per_depth.items())},
        "a14_per_ep_armed": round(float(np.mean([r["a14"] for r in armed])), 2),
        "opps_per_ep": round(float(np.mean([r["opps"] for r in armed])), 2),
        "died": {"armed": sum(r["died"] for r in armed),
                 "deprived": sum(r["died"] for r in deprived)},
        "ac_mean": {"armed": round(float(np.mean([r["ac"] for r in armed])), 1),
                    "deprived": round(float(np.mean([r["ac"] for r in deprived])), 1)},
    }
    (OUT / "probe_report.json").write_text(json.dumps(
        {"summary": summary, "armed": armed, "deprived": deprived},
        ensure_ascii=False, indent=1))
    print("\n=== ③A 汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
