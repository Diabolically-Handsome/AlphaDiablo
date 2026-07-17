"""P5 重放探针(FORENSICS-F2 取证队列;总设计师 2026-07-17 批文排产)。

对灾难种子(ctrl 视角 CAT11 ∪ sov 视角增补 ∪ 健康对照)× 三工人
(king/v32-sov/v32-ctrl)× H 经理,复刻 eval_assembled.evaluate 的
确定性协议逐窗重放,并在 mgr.choose 之前无副作用记录:经理 303 维
观测、选项掩码、全选项 logits、所选选项;步后记录该窗 reason/tau/opt。

安全性:记录仅为 numpy 纯前向与数组拷贝,不消耗 RNG、不改环境状态;
重放保真度以档案行(ret/depth/died/kills/mode_seq)逐字段对账自证。
零评测档案写入——产物落 train/runs/probe-f2-replay/(gitignored),
report.json 携运行时五 sha 与全部权重 sha。

用法:.venv/bin/python train/probe_f2_replay.py [--smoke]
"""
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))

from eval_contract import (bridge_binary_path,            # noqa: E402
                           runtime_identity)

CAT11 = [7001, 7004, 7005, 7013, 7017, 7020, 7021, 7024, 7025, 7029, 7031]
SOV_EXTRA = [7007, 7014, 7019, 7030]
CONTROLS = [7003, 7011]
SEEDS = sorted(set(CAT11 + SOV_EXTRA + CONTROLS))

H_NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
WORKERS = {
    "king": ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz",
    "sov":  ROOT / "train" / "runs" / "v32-sov" / "policy.npz",
    "ctrl": ROOT / "train" / "runs" / "v32-ctrl" / "policy.npz",
}
ARCHIVE = {
    "king": ROOT / "train" / "runs" / "eval-assembled" / "v32-ref-launch.json",
    "sov":  ROOT / "train" / "runs" / "eval-assembled" / "v32-sov-full32.json",
    "ctrl": ROOT / "train" / "runs" / "eval-assembled" / "v32-ctrl-full32.json",
}
OUT = ROOT / "train" / "runs" / "probe-f2-replay"
OUT.mkdir(parents=True, exist_ok=True)
ROW_FIELDS = ("ret", "depth", "died", "kills", "farm_n", "farm_tau_sum",
              "farm_descend", "windows", "beats", "overrides", "cap",
              "mode_seq")


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    smoke = "--smoke" in sys.argv
    seeds = SEEDS[:1] if smoke else SEEDS
    from diablogym import NumpyManager, OptionsEnv
    from diablogym.options_env import FARM
    mgr = NumpyManager(str(H_NPZ))
    fidelity = {}
    report = {
        "authorization": "总设计师 2026-07-17 批文:P5 排产",
        "seeds": seeds, "manager_sha256": sha(H_NPZ),
        "workers_sha256": {k: sha(v) for k, v in WORKERS.items()},
        "runtime_five": None, "fidelity": fidelity,
    }
    rt = runtime_identity(ROOT, bridge_binary_path(ROOT))
    report["runtime_five"] = {
        "bridge": rt["bridge"]["sha256"], "engine": rt["engine"]["sha256"],
        "game_data": rt["content"]["game_data"]["sha256"],
        "assets": rt["content"]["assets"]["sha256"],
        "protocol": rt["python_protocol"]["sha256"]}
    for wname, wpath in WORKERS.items():
        ref_rows = {r["seed"]: r for r in
                    json.load(open(ARCHIVE[wname]))["rows"]}
        net = NumpyManager(str(wpath))
        workers = {FARM: lambda obs, mask, _n=net: int(_n.choose(obs, mask))}
        env = OptionsEnv(max_steps=3000, workers=workers)
        try:
            for seed in seeds:
                obs_log, mask_log, logit_log, chosen, extras = [], [], [], [], []
                obs, _ = env.reset(seed=seed)
                done = trunc = False
                R = 0.0
                farm = {"n": 0, "tau": 0, "descend": 0}
                allw = {"n": 0, "beats": 0, "overrides": 0, "cap": 0}
                seq = ""
                while not (done or trunc):
                    m = env.action_masks()
                    # ---- 无副作用记录(纯前向;不消耗 RNG,不改状态)----
                    obs_log.append(np.asarray(obs, dtype=np.float32).copy())
                    mask_log.append(np.asarray(m, dtype=bool).copy())
                    logit_log.append(
                        np.asarray(mgr.logits(obs), dtype=np.float32).copy())
                    opt = mgr.choose(obs, m)
                    chosen.append(int(opt))
                    obs, r, done, trunc, info = env.step(opt)
                    R += float(r)
                    oe = info["option_extra"]
                    extras.append({"opt": int(oe["opt"]),
                                   "reason": str(oe["reason"]),
                                   "tau": int(oe["tau"]),
                                   "beats": int(oe["beats"]),
                                   "overrides": int(oe["overrides"])})
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
                row = {"seed": seed, "ret": round(R, 2),
                       "depth": raw["dungeon_level"],
                       "died": bool(raw.get("dead")),
                       "kills": env.env._ep_kills, "farm_n": farm["n"],
                       "farm_tau_mean": round(farm["tau"] / max(1, farm["n"]), 1),
                       "farm_tau_sum": farm["tau"],
                       "farm_descend": farm["descend"], "windows": allw["n"],
                       "beats": allw["beats"], "overrides": allw["overrides"],
                       "cap": allw["cap"], "mode_seq": seq}
                ok = all(row[f] == ref_rows[seed][f] for f in ROW_FIELDS)
                fidelity[f"{wname}-{seed}"] = "位级同一" if ok else {
                    "MISMATCH": {f: [row[f], ref_rows[seed][f]]
                                 for f in ROW_FIELDS
                                 if row[f] != ref_rows[seed][f]}}
                np.savez_compressed(
                    OUT / f"{wname}_{seed}.npz",
                    obs=np.stack(obs_log), masks=np.stack(mask_log),
                    logits=np.stack(logit_log),
                    chosen=np.asarray(chosen, dtype=np.int16))
                (OUT / f"{wname}_{seed}.windows.json").write_text(
                    json.dumps({"row": row, "extras": extras},
                               ensure_ascii=False))
                print(f"  {wname} seed {seed}: ret {R:.1f} windows {allw['n']}"
                      f" 对账 {'OK' if ok else 'MISMATCH!'}", flush=True)
        finally:
            env.close()
            workers.clear()
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    bad = [k for k, v in fidelity.items() if v != "位级同一"]
    print(f"重放保真:{len(fidelity) - len(bad)}/{len(fidelity)} 位级同一;"
          f"失配 {bad if bad else '无'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
