"""R18-F 零训练探针:回城卷轴 portal-v1(撤退 + 长观察窗之上),两臂配对。判据见 r18-F-PREREG-20260907.md。"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_t0 as base  # noqa: E402
import run_r18c_probe as r18c  # noqa: E402
import run_r18de_probe as de  # noqa: E402

OUT = base.STAGING / "r18f-probe"
G_OUT = base.STAGING / "r18g-probe"
COMMON = dict(de.COMMON)
JOBS = (
    ("r18f-retreat", "arm", {**COMMON}),
    ("r18f-retreat-portal", "arm", {**COMMON, "resource_portal": "portal-v1"}),
)
NEW_KEYS = de.NEW_KEYS + ("hunt_scope", "portals_started", "portal", "service_reason", "terminal_reason")


def launch(name, worker, overrides):
    import subprocess
    out = OUT / f"{name}.json"
    cmd = [base.PY, str(base.PROBE), str(base.WORKERS[worker]), base.SEEDS, str(out), str(base.MAX_STEPS),
           base.DECODING, json.dumps({**base.R16_FORM, **overrides}, ensure_ascii=False)]
    logf = open(OUT / f"{name}.log", "w")
    return name, out, subprocess.Popen(cmd, cwd=base.ROOT, stdout=logf, stderr=subprocess.STDOUT)


def strip(rows):
    out = []
    for r in rows:
        r = json.loads(json.dumps(r))
        res = r.get("resource")
        if isinstance(res, dict):
            for k in NEW_KEYS:
                res.pop(k, None)
        out.append(r)
    return out


def portal_stats(doc):
    rows = doc["rows"]
    legs, reasons, bought, cast = [], {}, 0, 0
    for r in rows:
        p = (r.get("resource") or {}).get("portal") or {}
        for a in p.get("attempts", []) if isinstance(p.get("attempts"), list) else []:
            legs.append(a)
            reasons[str(a.get("reason"))] = reasons.get(str(a.get("reason")), 0) + 1
        bought += int(p.get("scrolls_bought", 0) or 0)
        cast += int(p.get("casts", 0) or 0)
    ps = [int((r.get("resource") or {}).get("portals_started") or 0) for r in rows]
    return {"episodes_with_portal_activity": sum(1 for r in rows if ((r.get("resource") or {}).get("portal") or {}).get("attempted")),
            "legs": len(legs), "leg_reasons": reasons, "scrolls_bought": bought, "casts": cast,
            "portals_completed_hist": {str(v): ps.count(v) for v in sorted(set(ps))},
            "gold_final_mean": round(sum((r.get("resource") or {}).get("gold_final", 0) for r in rows) / len(rows), 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ref = json.load(open(G_OUT / "r18g-retreat.json"))
    base.log({"event": "R18F_PROBE_START", "seeds": base.SEEDS, "jobs": [j[0] for j in JOBS], "clock": "completion-l2-r18c",
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
                base.log({"event": "R18F_PROBE_JOB_FAILED", "job": name, "rc": rc}); raise SystemExit(1)
            done[name] = json.load(open(out))
            base.log({"event": "R18F_PROBE_JOB_DONE", "job": name, "alive": done[name]["agg"].get("alive_n"),
                      "l2_deaths": done[name]["agg"].get("l2_deaths"), "l2_hazard_per_1k": done[name]["agg"].get("l2_hazard_per_1k")})
        running = still
    ctl, p = done["r18f-retreat"], done["r18f-retreat-portal"]
    regression_equal = de.rows_sha(strip(ctl["rows"])) == de.rows_sha(strip(ref["rows"]))
    ctl_rows = {r["seed"]: r for r in ctl["rows"]}
    p_rows = {r["seed"]: r for r in p["rows"]}
    prefix_diffs = [s for s in p_rows if r18c.prefix_through_first_l2(p_rows[s]) != r18c.prefix_through_first_l2(ctl_rows.get(s, {}))]
    table = {name: de.arm_stats(d) for name, d in done.items()}
    table["r18f-retreat-portal"]["portal"] = portal_stats(p)
    paired = base.paired(p["rows"], ctl["rows"])
    tp, tc = table["r18f-retreat-portal"], table["r18f-retreat"]
    died_share = lambda t: (t["retreat_died_mid"] / t["retreat_attempts"]) if t["retreat_attempts"] else None
    hyps = {"H0_l2_reach_not_lower": bool((tp["l2_reach"] or 0) >= (tc["l2_reach"] or 0) - 3),
            "H1_hazard_lt_ctl": bool((tp["l2_hazard_per_1k"] or 9) < (tc["l2_hazard_per_1k"] or 0)),
            "H2_paired_net_gt_0_and_ucb_lt_0": bool(paired["net"] > 0 and paired["ucb95_one_sided"] < 0),
            "H3_retreat_died_share_lt_ctl": bool((died_share(tp) or 9) < (died_share(tc) or 0)),
            "H4_redescents_and_occupancy_ge_ctl": bool(tp["loops"]["episodes_redescended_after_ascent"] >= tc["loops"]["episodes_redescended_after_ascent"]
                                                       and (tp["occupancy_after_first_ascent"]["l2plus_share_median"] or 0) >= (tc["occupancy_after_first_ascent"]["l2plus_share_median"] or 0)),
            "H5_portal_economy": tp["portal"]}
    # The witch leg lengthens the first town trip, so the portal arm pre-L2 trajectory
    # legitimately differs (stochastic worker, per-episode RNG stream shifted): the
    # prefix comparison is REPORTED, not gated; the control regression is the gate.
    verdict = "R18F_REPORT_ONLY" if regression_equal else "R18F_VOID"
    summary = {"table": table, "paired_portal_vs_retreat": paired, "hypotheses": hyps,
               "regression_control_equal_to_r18g": regression_equal, "prefix_diff_seeds": prefix_diffs, "verdict": verdict,
               "seeds": base.SEEDS, "clock": "completion-l2-r18c", "elapsed_s": round(time.time() - t0, 1),
               "bridge": str((base.ROOT / "build").resolve())}
    json.dump(summary, open(OUT / "R18F-SUMMARY.json", "w"), ensure_ascii=False, indent=1)
    base.log({"event": "R18F_PROBE_DONE", "verdict": verdict, "hypotheses": hyps, "paired": paired,
              "regression_control_equal_to_r18g": regression_equal, "elapsed_s": summary["elapsed_s"]})
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("loops", "occupancy_after_first_ascent")} for k, v in table.items()}, ensure_ascii=False, indent=1))
    print(json.dumps({"paired": paired, "hypotheses": hyps, "regression": regression_equal, "prefix_diffs": prefix_diffs}, ensure_ascii=False, indent=1))
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
