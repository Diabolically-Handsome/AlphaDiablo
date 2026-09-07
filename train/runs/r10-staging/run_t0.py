"""R17 Gate T0(缩短版,主席 2026-09-06 口谕):零训练因子探针,新池 2_133。

4 组(预注册 r17-PREREG-FROZEN-20260906.md §八 B):
  a0-arm-v3   A0′:7e31dc54,协议关,readiness-v3 教练(R16 部署形态)
  a0-cert-v3  A0′:认证工人,同上
  t0-none     (a) 7e31dc54,l2-town-v1 / none / legacy-v1 / coach-v03(只拾金不回城)
  t0-full     (d) 7e31dc54,l2-town-v1 / full / sustain-v6 / coach-v03
每组 48 种子 2133000-2133047,6000 拍,sample 解码,R16 形态旗;仪表 = probe_r17_deployment.py。
判据(预注册 §三,缩短版只判两条):(d) 对 A0′-arm 成对 saved−lost ≥ +6/48 且死亡率差 UCB95 < 0;
L2 到达 ≥ 0.8 × A0′;L2 每千拍风险率 ≤ 0.7 × A0′;(d) − (a) 存活 ≥ +4/48;(d) forced-unready 份额 ≤ 25%。
用法:run_t0.py [--concurrency N]
"""
import json
import math
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path.home() / "AlphaDiablo" / "diablogym"
RUNS = ROOT / "train" / "runs"
STAGING = RUNS / "r10-staging"
LEDGER = STAGING / "r13_ledger.jsonl"
PROBE = STAGING / "probe_r17_deployment.py"
OUT = STAGING / "t0"
PY = str(ROOT / ".venv" / "bin" / "python")
WORKERS = {
    "arm": RUNS / "r16-arm-a-constitution" / "model_candidate.zip",
    "cert": RUNS / "r9-reeducation" / "staging" / "worker.zip",
}
R16_FORM = {"explore_global_hunt": True, "farm_scene_cap": 3600,
            "reset_layer_clock_on_window": True, "reward_economy": "v4",
            "drink_sovereignty": True}
SEEDS = "2133000-2133047"
MAX_STEPS = 6000
DECODING = "sample"
JOBS = (
    ("a0-arm-v3", "arm", {"manager": "readiness-v3"}),
    ("a0-cert-v3", "cert", {"manager": "readiness-v3"}),
    ("t0-none", "arm", {"manager": "resource", "resource_protocol": "l2-town-v1",
                        "resource_purchase_mode": "none",
                        "resource_service_policy": "legacy-v1",
                        "resource_readiness_law": "coach-v03"}),
    ("t0-full", "arm", {"manager": "resource", "resource_protocol": "l2-town-v1",
                        "resource_purchase_mode": "full",
                        "resource_service_policy": "sustain-v6",
                        "resource_readiness_law": "coach-v03"}),
)


def log(event):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    print("[ledger]", json.dumps(event, ensure_ascii=False)[:400], flush=True)


def launch(name, worker, overrides):
    out = OUT / f"{name}.json"
    cmd = [PY, str(PROBE), str(WORKERS[worker]), SEEDS, str(out), str(MAX_STEPS),
           DECODING, json.dumps({**R16_FORM, **overrides}, ensure_ascii=False)]
    logf = open(OUT / f"{name}.log", "w")
    return name, out, subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)


def paired(rows_x, rows_ref):
    """x vs ref:saved = ref 死 / x 活;lost = ref 活 / x 死;UCB95(单侧)= 死亡率差 x−ref。"""
    ref = {r["seed"]: bool(r["died"]) for r in rows_ref}
    diffs, saved, lost = [], 0, 0
    for r in rows_x:
        s = r["seed"]
        if s not in ref:
            continue
        dx, dr = bool(r["died"]), ref[s]
        diffs.append(int(dx) - int(dr))
        saved += int(dr and not dx)
        lost += int(dx and not dr)
    n = len(diffs)
    mean = sum(diffs) / n if n else float("nan")
    var = (sum((d - mean) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    ucb = mean + 1.645 * math.sqrt(var / n) if n else float("nan")
    return {"n": n, "saved": saved, "lost": lost, "net": saved - lost,
            "death_rate_diff": round(mean, 4), "ucb95_one_sided": round(ucb, 4)}


def summarize(docs):
    a0 = docs["a0-arm-v3"]; full = docs["t0-full"]; none = docs["t0-none"]
    def agg(d, k, default=None):
        return d["agg"].get(k, default)
    def resource_stats(d):
        rows = [r.get("resource") for r in d["rows"] if r.get("resource")]
        desc = sum(r["receipts"] for r in rows)
        forced = sum(r["descents_forced_unready"] for r in rows)
        trips = sum(1 for r in rows if r.get("service_attempted"))
        gold = [r["gold_final"] for r in rows]
        return {"descent_receipts": desc, "forced_unready": forced,
                "forced_unready_share": round(forced / desc, 3) if desc else None,
                "episodes_with_service": trips,
                "gold_final_mean": round(sum(gold) / len(gold), 1) if gold else None}
    table = {}
    for name, d in docs.items():
        table[name] = {"alive": agg(d, "alive_n"), "died": agg(d, "died"),
                       "l2_reach": agg(d, "l2_reach"), "l2_hazard_per_1k": agg(d, "l2_hazard_per_1k"),
                       "clvl_mean": agg(d, "clvl_mean"), "kills_mean": agg(d, "kills_mean"),
                       "fd_plus_1800_observed_n": agg(d, "fd_plus_1800_observed_n"),
                       "mask_forced_descent_share": agg(d, "mask_forced_descent_share"),
                       "resource": resource_stats(d) if name.startswith("t0-") else None}
    p_full_vs_a0 = paired(full["rows"], a0["rows"])
    p_full_vs_none = paired(full["rows"], none["rows"])
    l2_a0 = agg(a0, "l2_reach") or 0
    haz_a0 = agg(a0, "l2_hazard_per_1k")
    haz_full = agg(full, "l2_hazard_per_1k")
    full_res = table["t0-full"]["resource"]
    criteria = {
        "1_full_vs_a0_net_ge_6_and_ucb_lt_0": bool(p_full_vs_a0["net"] >= 6 and p_full_vs_a0["ucb95_one_sided"] < 0),
        "2_l2_reach_ge_0.8_a0": bool((agg(full, "l2_reach") or 0) >= 0.8 * l2_a0),
        "3_l2_hazard_le_0.7_a0": (bool(haz_full <= 0.7 * haz_a0) if (haz_full is not None and haz_a0) else None),
        "4_full_minus_none_alive_ge_4": bool((agg(full, "alive_n") or 0) - (agg(none, "alive_n") or 0) >= 4),
        "5_forced_unready_share_le_0.25": (bool(full_res["forced_unready_share"] <= 0.25)
                                            if full_res and full_res["forced_unready_share"] is not None else None),
    }
    passed = all(v is True for v in criteria.values())
    return {"table": table, "paired_full_vs_a0_arm": p_full_vs_a0, "paired_full_vs_none": p_full_vs_none,
            "criteria": criteria, "verdict": "T0_PASS" if passed else "T0_FAIL",
            "seeds": SEEDS, "max_steps": MAX_STEPS, "decoding": DECODING, "form": R16_FORM,
            "prereg": "train/runs/r17-PREREG-FROZEN-20260906.md sha256 96f5ba391e379c823b343804bca1bcca479ff8b72aaa27193af85e6ebcfd79ae",
            "bridge": str((ROOT / "build").resolve())}


def main():
    args = sys.argv[1:]
    concurrency = int(args[args.index("--concurrency") + 1]) if "--concurrency" in args else 4
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log({"event": "T0_START", "seeds": SEEDS, "jobs": [j[0] for j in JOBS], "max_steps": MAX_STEPS,
         "decoding": DECODING, "concurrency": concurrency, "bridge": str((ROOT / "build").resolve())})
    pending = list(JOBS); running = []; done = {}
    while pending or running:
        while pending and len(running) < concurrency:
            running.append(launch(*pending.pop(0)))
        time.sleep(5)
        still = []
        for name, out, proc in running:
            rc = proc.poll()
            if rc is None:
                still.append((name, out, proc)); continue
            if rc != 0 or not out.exists():
                log({"event": "T0_JOB_FAILED", "job": name, "rc": rc}); raise SystemExit(1)
            done[name] = json.load(open(out))
            log({"event": "T0_JOB_DONE", "job": name, "alive": done[name]["agg"].get("alive_n"),
                 "l2_reach": done[name]["agg"].get("l2_reach"),
                 "l2_hazard_per_1k": done[name]["agg"].get("l2_hazard_per_1k")})
        running = still
    summary = summarize(done)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(summary, open(OUT / "T0-SUMMARY.json", "w"), ensure_ascii=False, indent=1)
    log({"event": "T0_VERDICT", **{k: v for k, v in summary.items() if k in ("verdict", "criteria", "paired_full_vs_a0_arm", "paired_full_vs_none", "elapsed_s")}})
    print(json.dumps(summary["table"], ensure_ascii=False, indent=1))
    print("VERDICT:", summary["verdict"])


if __name__ == "__main__":
    main()
