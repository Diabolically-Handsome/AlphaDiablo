"""R18-D/E 零训练探针(2026-09-07 凌晨,主席令「四点全部修复」之一、二):
仇恨上限 hold-v1 与择敌 threat-v1,在撤退 + 长观察窗(completion-l2-r18c)之上,四臂配对(同池 2_133,同工人 7e31dc54)。
判据见 r18-DE-PREREG-20260907.md(回归硬判;H1–H5 只报不判)。用法:run_r18de_probe.py
"""
import hashlib
import json
import pathlib
import statistics as st
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_t0 as base  # noqa: E402
import run_r18c_probe as r18c  # noqa: E402

OUT = base.STAGING / "r18de-probe"
R18C_OUT = base.STAGING / "r18c-probe"
COMMON = {"manager": "resource", "resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
          "resource_readiness_law": "coach-v03", "worker_time_protocol": "completion-l2-r18c",
          "resource_service_policy": "sustain-loot-v1", "resource_retreat": "retreat-v1"}
JOBS = (
    ("r18de-retreat", "arm", {**COMMON}),
    ("r18de-cap", "arm", {**COMMON, "aggro_cap": "hold-v1"}),
    ("r18de-threat", "arm", {**COMMON, "engagement_priority": "threat-v1"}),
    ("r18de-cap-threat", "arm", {**COMMON, "aggro_cap": "hold-v1", "engagement_priority": "threat-v1"}),
)
NEW_KEYS = ("aggro_cap", "aggro_cap_mask_hits", "engagement_priority", "engagement_decisions", "engagement_reordered")


def launch(name, worker, overrides):
    import subprocess
    out = OUT / f"{name}.json"
    cmd = [base.PY, str(base.PROBE), str(base.WORKERS[worker]), base.SEEDS, str(out), str(base.MAX_STEPS),
           base.DECODING, json.dumps({**base.R16_FORM, **overrides}, ensure_ascii=False)]
    logf = open(OUT / f"{name}.log", "w")
    return name, out, subprocess.Popen(cmd, cwd=base.ROOT, stdout=logf, stderr=subprocess.STDOUT)


def strip_new_keys(rows):
    out = []
    for r in rows:
        r = json.loads(json.dumps(r))
        res = r.get("resource")
        if isinstance(res, dict):
            for k in NEW_KEYS:
                res.pop(k, None)
        out.append(r)
    return out


def rows_sha(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def arm_stats(doc):
    rows = doc["rows"]
    a = doc["agg"]
    attempts = [x for r in rows for x in (r18c.retreat_of(r).get("attempts") or [])]
    near = [x.get("monsters_near0", 0) for x in attempts]
    l2_beats = a.get("l2_beats_total") or 0
    l2_kills = sum(f["kills"] for r in rows for f in (r.get("floors") or []) if f["dlvl"] == 2)
    died_mid = sum(1 for r in rows for x in (r18c.retreat_of(r).get("attempts") or []) if x.get("outcome") is None and r["died"])
    return {"alive": a.get("alive_n"), "died": a.get("died"), "l2_reach": a.get("l2_reach"), "l2_deaths": a.get("l2_deaths"),
            "l2_beats_total": l2_beats, "l2_hazard_per_1k": a.get("l2_hazard_per_1k"),
            "l2_kills_per_1k_beats": round(1000.0 * l2_kills / l2_beats, 2) if l2_beats else None,
            "depth_hist": a.get("depth_hist"), "l3": a.get("l3"), "clvl_mean": a.get("clvl_mean"),
            "death_dlvl_hist": a.get("death_dlvl_hist"), "survivor_end_depth_hist": r18c.survivor_end_depth(doc),
            "l2_arrival_plus_9000": r18c.l2_plus(doc, 9000),
            "retreat_attempts": len(attempts), "retreat_ascended": sum(1 for x in attempts if x.get("outcome") == "ascended"),
            "retreat_died_mid": died_mid,
            "monsters_near0_at_trigger_mean": round(st.mean(near), 2) if near else None,
            "aggro_cap_mask_hits_total": sum(int((r.get("resource") or {}).get("aggro_cap_mask_hits") or 0) for r in rows),
            "engagement_decisions_total": sum(int((r.get("resource") or {}).get("engagement_decisions") or 0) for r in rows),
            "engagement_reordered_total": sum(int((r.get("resource") or {}).get("engagement_reordered") or 0) for r in rows),
            "loops": r18c.loop_stats(doc), "occupancy_after_first_ascent": r18c.occupancy_after_first_ascent(doc)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ref = json.load(open(R18C_OUT / "r18c-loot-retreat.json"))
    base.log({"event": "R18DE_PROBE_START", "seeds": base.SEEDS, "jobs": [j[0] for j in JOBS],
              "clock": "completion-l2-r18c", "bridge": str((base.ROOT / "build").resolve())})
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
                base.log({"event": "R18DE_PROBE_JOB_FAILED", "job": name, "rc": rc}); raise SystemExit(1)
            done[name] = json.load(open(out))
            base.log({"event": "R18DE_PROBE_JOB_DONE", "job": name, "alive": done[name]["agg"].get("alive_n"),
                      "l2_deaths": done[name]["agg"].get("l2_deaths"), "l2_hazard_per_1k": done[name]["agg"].get("l2_hazard_per_1k")})
        running = still
    ctl = done["r18de-retreat"]
    # regression: control rows identical to the R18-C retreat arm (new None keys stripped)
    ctl_sha, ref_sha = rows_sha(strip_new_keys(ctl["rows"])), rows_sha(strip_new_keys(ref["rows"]))
    regression_equal = ctl_sha == ref_sha
    # prefix identity through first L2 arrival for every arm vs the control
    ctl_rows = {r["seed"]: r for r in ctl["rows"]}
    prefix_diffs = {}
    for name, d in done.items():
        rows = {r["seed"]: r for r in d["rows"]}
        prefix_diffs[name] = [s for s in rows if r18c.prefix_through_first_l2(rows[s]) != r18c.prefix_through_first_l2(ctl_rows.get(s, {}))]
    prefix_ok = all(not v for v in prefix_diffs.values())
    table = {name: arm_stats(d) for name, d in done.items()}
    paired = {name: base.paired(d["rows"], ctl["rows"]) for name, d in done.items() if name != "r18de-retreat"}
    h = table
    hyps = {
        "H1_cap_hazard_lt_ctl": bool((h["r18de-cap"]["l2_hazard_per_1k"] or 9) < (h["r18de-retreat"]["l2_hazard_per_1k"] or 0)),
        "H1_capthreat_hazard_lt_ctl": bool((h["r18de-cap-threat"]["l2_hazard_per_1k"] or 9) < (h["r18de-retreat"]["l2_hazard_per_1k"] or 0)),
        "H2_capthreat_paired_net_gt_0_and_ucb_lt_0": bool(paired["r18de-cap-threat"]["net"] > 0 and paired["r18de-cap-threat"]["ucb95_one_sided"] < 0),
        "H3_cap_monsters_near0_lt_ctl": bool((h["r18de-cap"]["monsters_near0_at_trigger_mean"] or 99) < (h["r18de-retreat"]["monsters_near0_at_trigger_mean"] or 0)),
        "H4_threat_l2_kills_per_1k_ge_ctl": bool((h["r18de-threat"]["l2_kills_per_1k_beats"] or 0) >= (h["r18de-retreat"]["l2_kills_per_1k_beats"] or 0)),
        "H5_l3_plus": {name: h[name]["l3"] for name in h},
    }
    verdict = ("R18DE_REPORT_ONLY" if (regression_equal and prefix_ok)
               else "R18DE_VOID_" + ("REGRESSION" if not regression_equal else "PREFIX"))
    summary = {"table": table, "paired_vs_retreat": paired, "hypotheses": hyps,
               "regression_control_equal_to_r18c": regression_equal, "control_rows_sha16": ctl_sha[:16], "r18c_rows_sha16": ref_sha[:16],
               "prefix_diff_seeds": prefix_diffs, "verdict": verdict, "seeds": base.SEEDS, "clock": "completion-l2-r18c",
               "elapsed_s": round(time.time() - t0, 1), "bridge": str((base.ROOT / "build").resolve())}
    json.dump(summary, open(OUT / "R18DE-SUMMARY.json", "w"), ensure_ascii=False, indent=1)
    base.log({"event": "R18DE_PROBE_DONE", "verdict": verdict, "hypotheses": hyps, "paired_vs_retreat": paired,
              "regression_control_equal_to_r18c": regression_equal, "elapsed_s": summary["elapsed_s"]})
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("loops", "occupancy_after_first_ascent")} for k, v in table.items()}, ensure_ascii=False, indent=1))
    print(json.dumps({"paired": paired, "hypotheses": hyps, "regression": regression_equal, "prefix_diffs": prefix_diffs}, ensure_ascii=False, indent=1))
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
