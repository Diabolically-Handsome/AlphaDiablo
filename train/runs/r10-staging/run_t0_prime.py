"""R17 Gate T0′(2026-09-06 晚,主席「现在就来」):卖装备经济(R17.1-D)合并后的配对复测,同池 2_133。

两组同钟(sustain-loot-v1 硬性要求 completion-l2-v1:首次 L2 截止 12000 实际拍、之后 1800 观察、观测分母 6000):
  t0p-full-loot   (d′) 7e31dc54,l2-town-v1 / full / sustain-loot-v1 / coach-v03 / completion-l2-v1
  t0p-full-noloot (d″) 7e31dc54,l2-town-v1 / full / sustain-v6      / coach-v03 / completion-l2-v1
配对比较 (d′) vs (d″) 分离"捡+卖"的净效应;另对 T0 的 6000 拍 A0′-arm 与 (d) 作跨钟参考(只报不判)。
判据(沿用预注册 §三 对 (d) 的两条未达项,改以同钟对照):
  L2 每千拍风险率 (d′) ≤ 0.7 × A0′-arm(跨钟,信息量)且 (d′) ≤ (d″);forced-unready 份额 (d′) ≤ 25%;
  成对存活 (d′) − (d″) saved−lost ≥ +4/48。全部满足 → T0′_PASS(允许起草训练臂发射令);否则 FAIL。
用法:run_t0_prime.py [--concurrency N]
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_t0 as base  # noqa: E402

OUT = base.STAGING / "t0-prime"
T0_OUT = base.STAGING / "t0"
COMMON = {"manager": "resource", "resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
          "resource_readiness_law": "coach-v03", "worker_time_protocol": "completion-l2-v1"}
JOBS = (
    ("t0p-full-loot", "arm", {**COMMON, "resource_service_policy": "sustain-loot-v1"}),
    ("t0p-full-noloot", "arm", {**COMMON, "resource_service_policy": "sustain-v6"}),
)


def launch(name, worker, overrides):
    import subprocess
    out = OUT / f"{name}.json"
    cmd = [base.PY, str(base.PROBE), str(base.WORKERS[worker]), base.SEEDS, str(out), str(base.MAX_STEPS),
           base.DECODING, json.dumps({**base.R16_FORM, **overrides}, ensure_ascii=False)]
    logf = open(OUT / f"{name}.log", "w")
    return name, out, subprocess.Popen(cmd, cwd=base.ROOT, stdout=logf, stderr=subprocess.STDOUT)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base.log({"event": "T0_PRIME_START", "seeds": base.SEEDS, "jobs": [j[0] for j in JOBS],
              "clock": "completion-l2-v1 (12000 physical / 6000 obs denominator / 1800 follow-up)",
              "bridge": str((base.ROOT / "build").resolve())})
    running = [launch(*j) for j in JOBS]
    done = {}
    while running:
        time.sleep(5)
        still = []
        for name, out, proc in running:
            rc = proc.poll()
            if rc is None:
                still.append((name, out, proc)); continue
            if rc != 0 or not out.exists():
                base.log({"event": "T0_PRIME_JOB_FAILED", "job": name, "rc": rc}); raise SystemExit(1)
            done[name] = json.load(open(out))
            base.log({"event": "T0_PRIME_JOB_DONE", "job": name, "alive": done[name]["agg"].get("alive_n"),
                      "l2_reach": done[name]["agg"].get("l2_reach"),
                      "l2_hazard_per_1k": done[name]["agg"].get("l2_hazard_per_1k")})
        running = still
    ref = {n: json.load(open(T0_OUT / f"{n}.json")) for n in ("a0-arm-v3", "t0-full")}
    docs = {**done, **ref}
    loot, noloot, a0, d6000 = docs["t0p-full-loot"], docs["t0p-full-noloot"], docs["a0-arm-v3"], docs["t0-full"]

    def res(d):
        rows = [r.get("resource") or {} for r in d["rows"] if r.get("resource")]
        desc = sum(r.get("receipts", 0) for r in rows)
        forced = sum(r.get("descents_forced_unready", 0) for r in rows)
        gold = [r.get("gold_final", 0) for r in rows]
        return {"descent_receipts": desc, "forced_unready": forced,
                "forced_unready_share": round(forced / desc, 3) if desc else None,
                "episodes_with_service": sum(1 for r in rows if r.get("service_attempted")),
                "gold_final_mean": round(sum(gold) / len(gold), 1) if gold else None}

    table = {}
    for name, d in docs.items():
        a = d["agg"]
        table[name] = {"alive": a.get("alive_n"), "died": a.get("died"), "l2_reach": a.get("l2_reach"),
                       "l2_hazard_per_1k": a.get("l2_hazard_per_1k"), "clvl_mean": a.get("clvl_mean"),
                       "kills_mean": a.get("kills_mean"), "micro_steps_mean": a.get("micro_steps_mean"),
                       "fd_plus_1800_observed_n": a.get("fd_plus_1800_observed_n"),
                       "resource": res(d) if name.startswith("t0") else None}
    p_loot_vs_noloot = base.paired(loot["rows"], noloot["rows"])
    p_loot_vs_a0 = base.paired(loot["rows"], a0["rows"])
    haz_l, haz_n, haz_a0 = (loot["agg"].get("l2_hazard_per_1k"), noloot["agg"].get("l2_hazard_per_1k"),
                            a0["agg"].get("l2_hazard_per_1k"))
    fr = table["t0p-full-loot"]["resource"]["forced_unready_share"]
    criteria = {
        "1_loot_minus_noloot_alive_net_ge_4": bool(p_loot_vs_noloot["net"] >= 4),
        "2_loot_hazard_le_noloot": bool(haz_l is not None and haz_n is not None and haz_l <= haz_n),
        "3_loot_hazard_le_0.7_a0_crossclock_info": bool(haz_l is not None and haz_a0 and haz_l <= 0.7 * haz_a0),
        "4_loot_forced_unready_share_le_0.25": bool(fr is not None and fr <= 0.25),
    }
    passed = all(criteria[k] for k in ("1_loot_minus_noloot_alive_net_ge_4", "2_loot_hazard_le_noloot",
                                        "4_loot_forced_unready_share_le_0.25"))
    summary = {"table": table, "paired_loot_vs_noloot": p_loot_vs_noloot, "paired_loot_vs_a0_arm_crossclock": p_loot_vs_a0,
               "criteria": criteria, "verdict": "T0_PRIME_PASS" if passed else "T0_PRIME_FAIL",
               "seeds": base.SEEDS, "clock": "completion-l2-v1", "elapsed_s": round(time.time() - t0, 1),
               "bridge": str((base.ROOT / "build").resolve())}
    json.dump(summary, open(OUT / "T0-PRIME-SUMMARY.json", "w"), ensure_ascii=False, indent=1)
    base.log({"event": "T0_PRIME_VERDICT", "verdict": summary["verdict"], "criteria": criteria,
              "paired_loot_vs_noloot": p_loot_vs_noloot, "paired_loot_vs_a0_arm_crossclock": p_loot_vs_a0,
              "elapsed_s": summary["elapsed_s"]})
    print(json.dumps(table, ensure_ascii=False, indent=1))
    print("VERDICT:", summary["verdict"])


if __name__ == "__main__":
    main()
