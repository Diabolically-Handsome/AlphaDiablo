"""B1-E4 经理观测漂移离线重放通道(PREREG-B1 E4;probe_f2_replay.py 范式)。

评测档案不存观测,重放探针系唯一通道。对指定 (worker npz × manager npz ×
评测档案) 复刻 eval_assembled.evaluate 的确定性协议逐窗重放,并无副作用
记录经理观测;重放保真度以档案行(ROW_FIELDS)逐字段对账自证——保真锚
顺序写死:先落评测档案、后重放对账(承工程 m-8)。

OBS_DRIFT 统计量(schema 钉死,承统计 M-5;封闭枚举,禁判读日另择表述):
  - 8 追加维(经理观测下标 295-302)逐维 {mean, std, min, max, P5, P50, P95},
    统计面 = 全部窗末观测(每窗 step() 返回之观测,含终局观测);
  - 窗末 walkable 体态(局部图下标 44-164):walkΣ/121 窗末均值 + 西南带均值;
  - D 窗决策体态(RB.10 王座 D 窗侧读数):选中 DIVE 之决策点观测上的
    walkΣ/121 均值与西南带均值(决策点观测即上一窗窗末态;首窗为开局态)。

西南带定义(施工钉死;卷内无机器可执行先例,若与 P6 编队原口径不符须于
冻结前勘正——本定义已单列入呈报偏离清单):11×11 局部图 index=(dy+5)*11+
(dx+5)(bridge local_map 外层 dy、内层 dx);引擎 displacement.hpp 定
Direction::SouthWest = {0, +1},故西南带取 dy∈[1,5]、|dx|≤1 之 15 格带。

安全性:记录仅为 numpy 纯前向与数组拷贝,不消耗 RNG、不改环境状态;零评测
档案写入。产物 = 单个 report JSON(--out),携运行时五 sha 与全部权重 sha。

用法:
  .venv/bin/python train/probe_b1_obsdrift.py --worker <npz> --archive <json> \
      --out <report.json> [--manager <npz>] [--seeds 7001,7004] [--smoke]
退出码:0 = 全部种子重放保真;1 = 任一失配(报告仍落盘,如实登记)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))

from eval_contract import (bridge_binary_path,            # noqa: E402
                           runtime_identity, strict_json_loads)

H_NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
ROW_FIELDS = ("ret", "depth", "died", "kills", "farm_n", "farm_tau_sum",
              "farm_descend", "windows", "beats", "overrides", "cap",
              "mode_seq")
APPENDED8 = ("time_remaining", "stagnation_clock", "layer_kills",
             "layer_time", "last_opt_farm", "last_opt_dive",
             "last_opt_resupply", "last_opt_tau")   # 经理观测 295..302
WALK_LO, WALK_HI = 44, 165        # 基础观测 walkable 局部图下标 [44, 164]
SW_BAND_CELLS = tuple((dy + 5) * 11 + (dx + 5)
                      for dy in (1, 2, 3, 4, 5) for dx in (-1, 0, 1))
_PCTS = (5, 50, 95)


def sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _stats(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    p5, p50, p95 = (float(x) for x in np.percentile(v, _PCTS))
    return {"mean": round(float(v.mean()), 6), "std": round(float(v.std()), 6),
            "min": round(float(v.min()), 6), "max": round(float(v.max()), 6),
            "P5": round(p5, 6), "P50": round(p50, 6), "P95": round(p95, 6)}


def _walk_summary(obs_rows: np.ndarray) -> dict:
    if len(obs_rows) == 0:
        return {"n": 0, "walk_sum_over_121_mean": None, "sw_band_mean": None}
    walk = obs_rows[:, WALK_LO:WALK_HI]
    sw = walk[:, list(SW_BAND_CELLS)]
    return {"n": int(len(obs_rows)),
            "walk_sum_over_121_mean": round(float(walk.sum(axis=1).mean()), 3),
            "sw_band_mean": round(float(sw.mean()), 4)}


def replay_archive(worker_npz: str | pathlib.Path,
                   manager_npz: str | pathlib.Path,
                   archive_path: str | pathlib.Path,
                   seeds: list[int] | None = None,
                   limit: int | None = None) -> dict:
    """重放档案种子并产出 OBS_DRIFT 报告 dict(纯函数式,不落盘)。"""
    from diablogym import NumpyManager, OptionsEnv
    from diablogym.options_env import DIVE, FARM

    archive_payload = pathlib.Path(archive_path).read_bytes()
    doc = strict_json_loads(archive_payload)
    ref_rows = {int(r["seed"]): r for r in doc["rows"]}
    worker_sha = sha(worker_npz)
    manager_sha = sha(manager_npz)
    meta = doc.get("meta", {})
    if meta.get("worker", {}).get("sha256") != worker_sha:
        raise ValueError("档案 worker sha 与重放 worker npz 不一致,保真前提失义")
    if meta.get("manager", {}).get("sha256") != manager_sha:
        raise ValueError("档案 manager sha 与重放 manager npz 不一致,保真前提失义")
    replay_seeds = sorted(ref_rows) if seeds is None else list(seeds)
    if limit is not None:
        replay_seeds = replay_seeds[:limit]
    missing = [s for s in replay_seeds if s not in ref_rows]
    if missing:
        raise ValueError(f"请求重放的种子不在档案内: {missing}")

    rt = runtime_identity(ROOT, bridge_binary_path(ROOT))
    runtime_five = {
        "bridge": rt["bridge"]["sha256"], "engine": rt["engine"]["sha256"],
        "game_data": rt["content"]["game_data"]["sha256"],
        "assets": rt["content"]["assets"]["sha256"],
        "protocol": rt["python_protocol"]["sha256"]}

    mgr = NumpyManager(str(manager_npz))
    net = NumpyManager(str(worker_npz))
    net.require_worker_contract()
    workers = {FARM: net.worker_callback()}
    env = OptionsEnv(max_steps=3000, workers=workers)
    fidelity: dict[str, object] = {}
    per_seed: dict[str, dict] = {}
    end_obs_all: list[np.ndarray] = []       # 窗末观测(含终局观测)
    d_decision_obs: list[np.ndarray] = []    # D 窗决策点观测(RB.10 D 窗侧)
    try:
        for seed in replay_seeds:
            obs, _ = env.reset(seed=seed)
            done = trunc = False
            R = 0.0
            farm = {"n": 0, "tau": 0, "descend": 0}
            allw = {"n": 0, "beats": 0, "overrides": 0, "cap": 0}
            seq = ""
            farm_taus: list[int] = []
            d_windows = 0
            end_obs_seed: list[np.ndarray] = []
            while not (done or trunc):
                m = env.action_masks()
                decision_obs = np.asarray(obs, dtype=np.float32).copy()
                opt = mgr.choose(obs, m)
                obs, r, done, trunc, info = env.step(opt)
                R += float(r)
                oe = info["option_extra"]
                end_obs_seed.append(np.asarray(obs, dtype=np.float32).copy())
                allw["n"] += 1
                allw["beats"] += oe["beats"]
                allw["overrides"] += oe["overrides"]
                allw["cap"] += oe["reason"] == "cap"
                if oe["opt"] == FARM:
                    farm["n"] += 1
                    farm["tau"] += oe["tau"]
                    farm["descend"] += oe["reason"] == "descend"
                    farm_taus.append(int(oe["tau"]))
                if oe["opt"] == DIVE:
                    d_windows += 1
                    d_decision_obs.append(decision_obs)
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
            fidelity[str(seed)] = "位级同一" if ok else {
                "MISMATCH": {f: [row[f], ref_rows[seed][f]]
                             for f in ROW_FIELDS
                             if row[f] != ref_rows[seed][f]}}
            end_obs_all.extend(end_obs_seed)
            tau_median = (float(np.median(farm_taus)) if farm_taus else None)
            per_seed[str(seed)] = {
                "ret": row["ret"], "depth": row["depth"], "died": row["died"],
                "d_windows": d_windows, "farm_taus": farm_taus,
                "farm_tau_median": tau_median,
                "tau_floor_distance": (None if tau_median is None
                                       else round(tau_median - 25.0, 1)),
                "walk_end": _walk_summary(np.asarray(end_obs_seed)
                                          if end_obs_seed else np.empty((0, 303))),
            }
            print(f"  seed {seed}: ret {R:.1f} D窗 {d_windows} "
                  f"对账 {'OK' if ok else 'MISMATCH!'}", flush=True)
    finally:
        env.close()
        workers.clear()

    end_arr = (np.stack(end_obs_all) if end_obs_all
               else np.empty((0, 295 + 8), dtype=np.float32))
    d_arr = (np.stack(d_decision_obs) if d_decision_obs
             else np.empty((0, 295 + 8), dtype=np.float32))
    appended8 = {name: _stats(end_arr[:, 295 + i]) if len(end_arr) else None
                 for i, name in enumerate(APPENDED8)}
    report = {
        "probe": "b1-obsdrift",
        "prereg": "PREREG-B1 E4(schema 钉死;统计量封闭枚举)",
        "archive": str(archive_path),
        "archive_sha256": hashlib.sha256(archive_payload).hexdigest(),
        "worker_sha256": worker_sha,
        "manager_sha256": manager_sha,
        "runtime_five": runtime_five,
        "seeds": replay_seeds,
        "fidelity": fidelity,
        "fidelity_ok": all(v == "位级同一" for v in fidelity.values()),
        "sw_band_cells": list(SW_BAND_CELLS),
        "appended8_window_end": appended8,
        "walk_window_end": _walk_summary(end_arr),
        "walk_d_decision": _walk_summary(d_arr),
        "per_seed": per_seed,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True, help="工人 npz 路径")
    ap.add_argument("--manager", default=str(H_NPZ), help="经理 npz(默认 H)")
    ap.add_argument("--archive", required=True, help="评测档案 JSON(先落档后重放)")
    ap.add_argument("--out", required=True, help="报告 JSON 输出路径")
    ap.add_argument("--seeds", default=None,
                    help="逗号分隔种子子集(默认 = 档案全部种子)")
    ap.add_argument("--smoke", action="store_true", help="只重放首个种子")
    args = ap.parse_args()
    seeds = ([int(x) for x in args.seeds.split(",") if x.strip()]
             if args.seeds else None)
    report = replay_archive(args.worker, args.manager, args.archive, seeds,
                            limit=1 if args.smoke else None)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    bad = [k for k, v in report["fidelity"].items() if v != "位级同一"]
    print(f"重放保真:{len(report['fidelity']) - len(bad)}/{len(report['fidelity'])}"
          f" 位级同一;失配 {bad if bad else '无'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
