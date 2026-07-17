"""P7 剧本经理强制 DIVE 探针(FORENSICS-F2 取证队列;总设计师批:引擎重放、零训练)。

靶点(FORENSICS-F2 卷 P7 靶单②③):
  ① 剧本经理强制 D:在灾难种子上以"剧本经理"替代 v22-h 经理选窗——前 K 窗
     照抄 H 的真实选择(自 P5 存档 train/runs/probe-f2-replay/{worker}_{seed}.npz
     之 chosen 序列回放),到达强制点强制选 DIVE 一次(DIVE 掩码恒真已由 P5
     4337/4337 实证;若违反如实报告),之后交还 H argmax。观察:工人是否潜成
     (窗 reason == "descend")、下潜后的回报;对比 king/sov/ctrl 三工人。
     终审 "sov 被选 D 时 0/3 潜成 vs ctrl 2/3(n=3)"之下潜执行分化。
  ② 7017/7013/7021 残差终审:7013/7021 王座无 D 窗、7017 四配置全失守,
     同用强制 D 剧本看三工人下潜执行与下潜后表现。

强制点方案(--plan 打印全量建议矩阵):
  A = 首个 exhausted 收窗之后的下一窗(按本工人自己的回放在线触发;
      与 P5 存档预期窗序对账);
  B = 对有王座 D 窗的种子,用王座(king 回放)首次按 D 的同一窗序。

自校验:确定性协议下强制点之前的前缀应与 P5 位级相同——逐窗对账
obs/masks/H-logits(位级)与收窗五字段(opt/reason/tau/beats/overrides);
任何失配即停并如实入册(prefix_divergence)。

安全性:与 P5 同款——记录仅为 numpy 纯前向与数组拷贝,不消耗 RNG、不改
环境状态;零评测档案写入,产物落 train/runs/probe-f2-p7/(gitignored),
report.json 携运行时五 sha 与全部权重 sha。

用法:
  .venv/bin/python train/probe_f2_p7.py --smoke          # ctrl:7024:A + king:7004:B
  .venv/bin/python train/probe_f2_p7.py --plan           # 打印全量建议清单(不跑)
  .venv/bin/python train/probe_f2_p7.py sov:7017:B ...   # 显式排产(值守)
"""
import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))

from eval_contract import (bridge_binary_path,            # noqa: E402
                           runtime_identity)

H_NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
WORKERS = {
    "king": ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz",
    "sov":  ROOT / "train" / "runs" / "v32-sov" / "policy.npz",
    "ctrl": ROOT / "train" / "runs" / "v32-ctrl" / "policy.npz",
}
P5_DIR = ROOT / "train" / "runs" / "probe-f2-replay"
OUT = ROOT / "train" / "runs" / "probe-f2-p7"
ROW_FIELDS = ("ret", "depth", "died", "kills", "farm_n", "farm_tau_sum",
              "farm_descend", "windows", "beats", "overrides", "cap",
              "mode_seq")
PREFIX_FIELDS = ("opt", "reason", "tau", "beats", "overrides")  # P5 extras 存档字段
SMOKE_JOBS = ["ctrl:7024:A", "king:7004:B"]


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def load_p5(worker, seed):
    """P5 存档:npz(obs/masks/logits/chosen)+ windows.json(row/extras)。"""
    npz_path = P5_DIR / f"{worker}_{seed}.npz"
    js_path = P5_DIR / f"{worker}_{seed}.windows.json"
    if not npz_path.exists() or not js_path.exists():
        raise FileNotFoundError(f"缺 P5 存档:{npz_path.name} / {js_path.name}")
    with np.load(npz_path) as z:
        arch = {k: z[k].copy() for k in ("obs", "masks", "logits", "chosen")}
    meta = json.loads(js_path.read_text())
    arch["extras"], arch["row"] = meta["extras"], meta["row"]
    return arch


def throne_d_ordinals(seed):
    """王座(king)回放中 H 选 D 的窗序列表(方案 B 的强制点来源)。"""
    from diablogym.options_env import DIVE
    with np.load(P5_DIR / f"king_{seed}.npz") as z:
        return [int(i) for i in np.nonzero(z["chosen"] == DIVE)[0]]


def first_exhausted(arch):
    for i, e in enumerate(arch["extras"]):
        if e["reason"] == "exhausted":
            return i
    return None


def planned_force_idx(worker, seed, scheme, arch):
    """存档预期强制点(A=首 exhausted+1;B=王座首 D 窗序)。None=不可排。"""
    if scheme == "A":
        fe = first_exhausted(arch)
        if fe is None or fe + 1 >= len(arch["chosen"]):
            return None  # 无 exhausted 窗,或其后无窗(回放局同窗结束)
        return fe + 1
    ords = throne_d_ordinals(seed)
    if not ords or ords[0] >= len(arch["chosen"]):
        return None      # 王座无 D 窗,或窗序超出本工人回放窗数
    return ords[0]


class ScriptedManager:
    """剧本经理:前 K 窗照抄 P5 存档 chosen → 强制点选 DIVE 一次 → 之后透传 H。

    choose() 在前缀段内做位级自校验(obs/masks/H-logits 对 P5 存档);
    window_closed() 于每窗收束后喂入 extra,驱动方案 A 的 exhausted 触发。
    """

    def __init__(self, h_mgr, arch, scheme, force_idx):
        from diablogym.options_env import DIVE
        self._DIVE = DIVE
        self.h = h_mgr
        self.arch = arch
        self.scheme = scheme
        self.force_idx = force_idx    # 方案 B 用;方案 A 由 exhausted 在线触发
        self.t = 0                    # 当前窗序(0 起)
        self.forced_at = None
        self.arm_next = False         # 方案 A:首个 exhausted 已收,下一窗强制
        self.divergence = None

    def _prefix_check(self, o, m, hlg):
        t, a, bad = self.t, self.arch, {}
        if not np.array_equal(o, a["obs"][t]):
            d = np.nonzero(o != a["obs"][t])[0]
            bad["obs_dims"] = [int(i) for i in d[:8]]
        if not np.array_equal(m, a["masks"][t]):
            bad["masks"] = [m.tolist(), a["masks"][t].tolist()]
        if not np.array_equal(hlg, a["logits"][t]):
            bad["logits"] = [hlg.tolist(), a["logits"][t].tolist()]
        if bad:
            self.divergence = {"t": t, **bad}
        return not bad

    def choose(self, o, m, hlg):
        """返回 (opt|None, src)。src ∈ prefix/forced/H/force_blocked/
        prefix_divergence/prefix_overrun;opt=None 表示本局须终止并入册。"""
        t = self.t
        if self.forced_at is not None:
            return int(self.h.choose(o, m)), "H"
        if (t == self.force_idx) if self.scheme == "B" else self.arm_next:
            if not bool(m[self._DIVE]):
                return None, "force_blocked"   # 违反"DIVE 恒亮"实证,如实报告
            self.forced_at = t
            return self._DIVE, "forced"
        if t >= len(self.arch["chosen"]):
            return None, "prefix_overrun"      # 防御:回放局应与存档同窗结束
        if not self._prefix_check(o, m, hlg):
            return None, "prefix_divergence"
        return int(self.arch["chosen"][t]), "prefix"

    def window_closed(self, extra):
        if (self.scheme == "A" and self.forced_at is None
                and extra["reason"] == "exhausted"):
            self.arm_next = True
        self.t += 1


def run_job(worker, seed, scheme, h_mgr):
    from diablogym import NumpyManager, OptionsEnv
    from diablogym.options_env import FARM
    arch = load_p5(worker, seed)
    planned = planned_force_idx(worker, seed, scheme, arch)
    if planned is None:
        return {"job": f"{worker}:{seed}:{scheme}", "status": "unschedulable",
                "note": ("方案A:存档无 exhausted 窗或其后无窗" if scheme == "A"
                         else "方案B:王座无 D 窗或窗序超出本工人回放窗数")}
    script = ScriptedManager(h_mgr, arch, scheme,
                             planned if scheme == "B" else None)
    net = NumpyManager(str(WORKERS[worker]))
    workers = {FARM: lambda obs, mask, _n=net: int(_n.choose(obs, mask))}
    env = OptionsEnv(max_steps=3000, workers=workers)
    obs_log, mask_log, logit_log, chosen, windows = [], [], [], [], []
    status = "completed"
    t0 = time.time()
    try:
        obs, _ = env.reset(seed=seed)
        done = trunc = False
        R = 0.0
        farm = {"n": 0, "tau": 0, "descend": 0}
        allw = {"n": 0, "beats": 0, "overrides": 0, "cap": 0}
        seq = ""
        while not (done or trunc):
            m = env.action_masks()
            # ---- 无副作用记录(纯前向;不消耗 RNG,不改状态)----
            o = np.asarray(obs, dtype=np.float32).copy()
            mb = np.asarray(m, dtype=bool).copy()
            hlg = np.asarray(h_mgr.logits(o), dtype=np.float32).copy()
            opt, src = script.choose(o, mb, hlg)
            if opt is None:            # 强制点被掩/前缀失配/超窗:终止入册
                status = src
                break
            obs_log.append(o)
            mask_log.append(mb)
            logit_log.append(hlg)
            chosen.append(int(opt))
            obs, r, done, trunc, info = env.step(opt)
            R += float(r)
            oe = info["option_extra"]
            windows.append({
                "t": script.t, "src": src, "opt": int(oe["opt"]),
                "reason": str(oe["reason"]), "tau": int(oe["tau"]),
                "beats": int(oe["beats"]), "overrides": int(oe["overrides"]),
                "drains": int(oe["drains"]), "R": round(float(oe["R"]), 3),
                "dlvl0": int(oe["dlvl0"]), "dlvl_end": int(oe["dlvl_end"])})
            if src == "prefix":        # 前缀收窗五字段对账(P5 extras 存档)
                ref = arch["extras"][script.t]
                mm = {f: [windows[-1][f], ref[f]] for f in PREFIX_FIELDS
                      if windows[-1][f] != ref[f]}
                if mm:
                    script.divergence = {"t": script.t, "fields": mm}
                    status = "prefix_divergence"
                    script.window_closed(oe)
                    break
            allw["n"] += 1
            allw["beats"] += oe["beats"]
            allw["overrides"] += oe["overrides"]
            allw["cap"] += oe["reason"] == "cap"
            if oe["opt"] == FARM:
                farm["n"] += 1
                farm["tau"] += oe["tau"]
                farm["descend"] += oe["reason"] == "descend"
            seq = oe["mode_seq"]
            script.window_closed(oe)
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
    finally:
        env.close()
        workers.clear()
    if status == "completed" and script.forced_at is None:
        status = "force_not_reached"
    forced_w = next((w for w in windows if w["src"] == "forced"), None)
    n_pre = sum(1 for w in windows if w["src"] == "prefix")
    ret_pre = round(sum(w["R"] for w in windows if w["src"] == "prefix"), 2)
    result = {
        "job": f"{worker}:{seed}:{scheme}", "worker": worker, "seed": seed,
        "scheme": scheme, "status": status,
        "force_idx_planned": planned, "forced_at": script.forced_at,
        "prefix_windows": n_pre,
        "prefix_fidelity": ("位级同一" if script.divergence is None
                            else script.divergence),
        "forced_window": forced_w,
        "descend_success": (forced_w["reason"] == "descend"
                            if forced_w else None),
        "ret_pre_force": ret_pre,
        "ret_from_force": (round(row["ret"] - ret_pre, 2)
                           if forced_w else None),
        "post_force_windows": [w for w in windows
                               if script.forced_at is not None
                               and w["t"] > script.forced_at],
        "row": row, "p5_row": arch["row"],
        "delta_vs_p5": {f: round(row[f] - arch["row"][f], 2)
                        for f in ("ret", "depth", "kills")},
        "row_equals_p5": all(row[f] == arch["row"][f] for f in ROW_FIELDS),
        "elapsed_s": round(time.time() - t0, 1),
        "windows": windows,
    }
    stem = f"{worker}_{seed}_{scheme}"
    np.savez_compressed(
        OUT / f"{stem}.npz",
        obs=np.stack(obs_log) if obs_log else np.zeros((0, 303), np.float32),
        masks=(np.stack(mask_log) if mask_log
               else np.zeros((0, 3), bool)),
        logits=(np.stack(logit_log) if logit_log
                else np.zeros((0, 3), np.float32)),
        chosen=np.asarray(chosen, dtype=np.int16),
        forced_at=np.int64(-1 if script.forced_at is None
                           else script.forced_at))
    (OUT / f"{stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1))
    return result


def print_plan():
    """全量建议清单:worker × P5 种子 × 方案,按存档可排性过滤。"""
    seeds = sorted({int(p.stem.split("_")[1])
                    for p in P5_DIR.glob("king_*.npz")})
    jobs = []
    for scheme in ("A", "B"):
        for seed in seeds:
            for worker in ("king", "sov", "ctrl"):
                arch = load_p5(worker, seed)
                pf = planned_force_idx(worker, seed, scheme, arch)
                if pf is not None:
                    jobs.append((f"{worker}:{seed}:{scheme}", pf,
                                 len(arch["chosen"])))
    print(f"# 全量建议清单({len(jobs)} 局;强制点=存档预期窗序/总窗数)")
    for j, pf, n in jobs:
        print(f"  {j:<16} 强制点 w{pf:<4} / {n} 窗")
    print("# 排产命令示例:.venv/bin/python train/probe_f2_p7.py "
          + " ".join(j for j, _, _ in jobs[:3]) + " ...")
    return jobs


def parse_job(s):
    parts = s.split(":")
    if len(parts) != 3 or parts[0] not in WORKERS or parts[2] not in ("A", "B"):
        raise SystemExit(f"任务格式须为 worker:seed:方案(A|B),收到 {s!r}")
    return parts[0], int(parts[1]), parts[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("jobs", nargs="*", help="worker:seed:scheme,如 ctrl:7024:A")
    ap.add_argument("--smoke", action="store_true",
                    help=f"冒烟预设:{' '.join(SMOKE_JOBS)}")
    ap.add_argument("--plan", action="store_true", help="打印全量建议清单后退出")
    args = ap.parse_args()
    if args.plan:
        print_plan()
        return 0
    job_strs = SMOKE_JOBS if args.smoke else args.jobs
    if not job_strs:
        ap.error("须给出任务列表,或 --smoke / --plan")
    jobs = [parse_job(s) for s in job_strs]
    from diablogym import NumpyManager
    OUT.mkdir(parents=True, exist_ok=True)
    h_mgr = NumpyManager(str(H_NPZ))
    report = {
        "authorization": "总设计师批文:F2 案 P7 排产(引擎重放、零训练)",
        "jobs": job_strs, "manager_sha256": sha(H_NPZ),
        "workers_sha256": {k: sha(v) for k, v in WORKERS.items()},
        "runtime_five": None, "results": {},
    }
    rt = runtime_identity(ROOT, bridge_binary_path(ROOT))
    report["runtime_five"] = {
        "bridge": rt["bridge"]["sha256"], "engine": rt["engine"]["sha256"],
        "game_data": rt["content"]["game_data"]["sha256"],
        "assets": rt["content"]["assets"]["sha256"],
        "protocol": rt["python_protocol"]["sha256"]}
    failures = []
    for worker, seed, scheme in jobs:
        res = run_job(worker, seed, scheme, h_mgr)
        key = res["job"]
        report["results"][key] = {
            k: res.get(k) for k in
            ("status", "force_idx_planned", "forced_at", "prefix_windows",
             "prefix_fidelity", "descend_success", "ret_pre_force",
             "ret_from_force", "row_equals_p5", "delta_vs_p5", "elapsed_s")}
        if res["status"] in ("prefix_divergence", "force_blocked",
                             "prefix_overrun"):
            failures.append(key)
        fw = res.get("forced_window")
        print(f"  {key}: {res['status']}"
              f" 前缀 {res.get('prefix_windows')} 窗"
              f"({res.get('prefix_fidelity') if isinstance(res.get('prefix_fidelity'), str) else '失配!'})"
              f" 强制@{res.get('forced_at')}"
              + (f" DIVE→{fw['reason']}(τ{fw['tau']},"
                 f"dlvl {fw['dlvl0']}→{fw['dlvl_end']})"
                 f" 潜成={res['descend_success']}"
                 f" 潜后ret {res['ret_from_force']}"
                 f" 终局 ret {res['row']['ret']}/depth {res['row']['depth']}"
                 f"/kills {res['row']['kills']}"
                 f"(原案 {res['p5_row']['ret']}/{res['p5_row']['depth']}"
                 f"/{res['p5_row']['kills']})" if fw else ""),
              flush=True)
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    print(f"P7 收卷:{len(jobs) - len(failures)}/{len(jobs)} 无失配;"
          f"失配 {failures if failures else '无'};产物 {OUT}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
