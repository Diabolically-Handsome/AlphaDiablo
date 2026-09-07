"""R18-A Gate T0″(2026-09-06 深夜,主席令「接口不全,再做一个回城的接口,现在就开干」):
撤退接口 retreat-v1 的零训练机理探针,同池 2_133,同钟 completion-l2-v1,同工人 7e31dc54。

两组配对:
  t0pp-loot-retreat   (e)  (d′) + resource_retreat=retreat-v1(L2 上 HP ≤ 50% 或药空且 HP ≤ 75% → 撤到 L1;每次撤退多换一次回城)
  t0pp-loot-noretreat (d′) 与 T0′ 的 t0p-full-loot 同配置(在 R18-A 字节上重跑,兼作"关=逐位不变"的部署级回归)
判据(预注册 r18-A-PREREG):
  1. 成对存活 (e) − (d′) saved−lost ≥ +6/48 且死亡率差 UCB95 < 0;
  2. (e) L2 死亡数 ≤ 0.7 × (d′);
  3. 机理:撤退尝试中到达上层的份额 ≥ 0.6,且撤退途中死亡份额 ≤ 0.25;
  4. 回归:(d′) 重跑行 与 T0′ t0p-full-loot 行逐位相同(否则整个探针作废)。
  1、2、4 全部满足 → T0″_PASS;4 不满足 → VOID;其余 → FAIL(机理项只报不判)。
用法:run_t0_double_prime.py
"""
import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_t0 as base  # noqa: E402

OUT = base.STAGING / "t0-double-prime"
T0P_OUT = base.STAGING / "t0-prime"
COMMON = {"manager": "resource", "resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
          "resource_readiness_law": "coach-v03", "worker_time_protocol": "completion-l2-v1",
          "resource_service_policy": "sustain-loot-v1"}
JOBS = (
    ("t0pp-loot-retreat", "arm", {**COMMON, "resource_retreat": "retreat-v1"}),
    ("t0pp-loot-noretreat", "arm", {**COMMON}),
)


def launch(name, worker, overrides):
    import subprocess
    out = OUT / f"{name}.json"
    cmd = [base.PY, str(base.PROBE), str(base.WORKERS[worker]), base.SEEDS, str(out), str(base.MAX_STEPS),
           base.DECODING, json.dumps({**base.R16_FORM, **overrides}, ensure_ascii=False)]
    logf = open(OUT / f"{name}.log", "w")
    return name, out, subprocess.Popen(cmd, cwd=base.ROOT, stdout=logf, stderr=subprocess.STDOUT)


def rows_sha(doc):
    return hashlib.sha256(json.dumps(doc["rows"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def strip_retreat_keys(rows):
    """The retreat-off rows carry two extra None-valued keys under `resource`
    (retreats_started, retreat) that T0′ rows predate; compare without them."""
    out = []
    for r in rows:
        r = json.loads(json.dumps(r))
        res = r.get("resource")
        if isinstance(res, dict):
            res.pop("retreats_started", None)
            res.pop("retreat", None)
        out.append(r)
    return out


def retreat_stats(doc):
    attempts = []
    for r in doc["rows"]:
        ret = (r.get("resource") or {}).get("retreat") or {}
        attempts.extend(ret.get("attempts", []))
    n = len(attempts)
    ascended = sum(1 for a in attempts if a.get("outcome") == "ascended")
    died = sum(1 for a in attempts if a.get("reason") == "retreat_died")
    failed = sum(1 for a in attempts if a.get("outcome") == "failed")
    reasons = {}
    for a in attempts:
        reasons[str(a.get("reason"))] = reasons.get(str(a.get("reason")), 0) + 1
    triggers = {}
    for a in attempts:
        triggers[str(a.get("trigger"))] = triggers.get(str(a.get("trigger")), 0) + 1
    hp0 = [a["hp0"] / a["max_hp"] for a in attempts if a.get("max_hp")]
    dur = [a["beat1"] - a["beat0"] for a in attempts if a.get("beat1") is not None]
    episodes = sum(1 for r in doc["rows"] if ((r.get("resource") or {}).get("retreat") or {}).get("attempted"))
    trips = [((r.get("resource") or {}).get("retreat") or {}).get("ascended", 0) for r in doc["rows"]]
    return {"attempts": n, "ascended": ascended, "failed": failed, "died_during_retreat": died,
            "ascended_share": round(ascended / n, 3) if n else None,
            "died_share": round(died / n, 3) if n else None,
            "reasons": reasons, "triggers": triggers,
            "hp0_frac_mean": round(sum(hp0) / len(hp0), 3) if hp0 else None,
            "duration_beats_median": sorted(dur)[len(dur) // 2] if dur else None,
            "episodes_with_retreat": episodes,
            "episodes_with_2plus_ascents": sum(1 for t in trips if t >= 2)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base.log({"event": "T0_DOUBLE_PRIME_START", "seeds": base.SEEDS, "jobs": [j[0] for j in JOBS],
              "clock": "completion-l2-v1", "bridge": str((base.ROOT / "build").resolve())})
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
                base.log({"event": "T0_DOUBLE_PRIME_JOB_FAILED", "job": name, "rc": rc}); raise SystemExit(1)
            done[name] = json.load(open(out))
            base.log({"event": "T0_DOUBLE_PRIME_JOB_DONE", "job": name, "alive": done[name]["agg"].get("alive_n"),
                      "l2_reach": done[name]["agg"].get("l2_reach"),
                      "l2_deaths": done[name]["agg"].get("l2_deaths"),
                      "l2_hazard_per_1k": done[name]["agg"].get("l2_hazard_per_1k")})
        running = still
    ret, ctl = done["t0pp-loot-retreat"], done["t0pp-loot-noretreat"]
    t0p = json.load(open(T0P_OUT / "t0p-full-loot.json"))
    ctl_sha = hashlib.sha256(json.dumps(strip_retreat_keys(ctl["rows"]), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    t0p_sha = hashlib.sha256(json.dumps(strip_retreat_keys(t0p["rows"]), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    regression_equal = ctl_sha == t0p_sha

    table = {}
    for name, d in (("t0pp-loot-retreat", ret), ("t0pp-loot-noretreat", ctl), ("t0p-full-loot (T0′)", t0p)):
        a = d["agg"]
        table[name] = {"alive": a.get("alive_n"), "died": a.get("died"), "l2_reach": a.get("l2_reach"),
                       "l2_deaths": a.get("l2_deaths"), "l2_beats_total": a.get("l2_beats_total"),
                       "l2_hazard_per_1k": a.get("l2_hazard_per_1k"), "clvl_mean": a.get("clvl_mean"),
                       "kills_mean": a.get("kills_mean"), "death_dlvl_hist": a.get("death_dlvl_hist"),
                       "fd_plus_1800_observed_n": a.get("fd_plus_1800_observed_n")}
    paired = base.paired(ret["rows"], ctl["rows"])
    l2d_ret, l2d_ctl = ret["agg"].get("l2_deaths"), ctl["agg"].get("l2_deaths")
    mech = retreat_stats(ret)
    criteria = {
        "1_retreat_minus_control_alive_net_ge_6_and_ucb_lt_0": bool(
            paired["net"] >= 6 and paired.get("ucb95_one_sided", 1) < 0),
        "2_l2_deaths_le_0.7_control": bool(l2d_ret is not None and l2d_ctl is not None and l2d_ret <= 0.7 * l2d_ctl),
        "3_mechanism_ascended_ge_0.6_and_died_le_0.25_report_only": bool(
            mech["ascended_share"] is not None and mech["ascended_share"] >= 0.6
            and mech["died_share"] is not None and mech["died_share"] <= 0.25),
        "4_control_rows_identical_to_t0_prime": regression_equal,
    }
    if not regression_equal:
        verdict = "T0_DOUBLE_PRIME_VOID"
    elif criteria["1_retreat_minus_control_alive_net_ge_6_and_ucb_lt_0"] and criteria["2_l2_deaths_le_0.7_control"]:
        verdict = "T0_DOUBLE_PRIME_PASS"
    else:
        verdict = "T0_DOUBLE_PRIME_FAIL"
    summary = {"table": table, "paired_retreat_vs_control": paired, "retreat_mechanism": mech,
               "criteria": criteria, "verdict": verdict, "seeds": base.SEEDS, "clock": "completion-l2-v1",
               "control_rows_sha": ctl_sha, "t0_prime_rows_sha": t0p_sha,
               "elapsed_s": round(time.time() - t0, 1), "bridge": str((base.ROOT / "build").resolve())}
    json.dump(summary, open(OUT / "T0-DOUBLE-PRIME-SUMMARY.json", "w"), ensure_ascii=False, indent=1)
    base.log({"event": "T0_DOUBLE_PRIME_VERDICT", "verdict": verdict, "criteria": criteria,
              "paired_retreat_vs_control": paired, "retreat_mechanism": mech,
              "control_rows_sha16": ctl_sha[:16], "t0_prime_rows_sha16": t0p_sha[:16],
              "elapsed_s": summary["elapsed_s"]})
    print(json.dumps(table, ensure_ascii=False, indent=1))
    print(json.dumps({"paired": paired, "mechanism": mech, "criteria": criteria}, ensure_ascii=False, indent=1))
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
