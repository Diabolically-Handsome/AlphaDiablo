"""R18-G 零训练探针:主线 L2+ 关闭 a10 全图求战(拉怪机制本身),撤退 + 长观察窗之上,两臂配对。判据见 r18-G-PREREG-20260907.md。"""
import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_t0 as base  # noqa: E402
import run_r18c_probe as r18c  # noqa: E402
import run_r18de_probe as de  # noqa: E402

OUT = base.STAGING / "r18g-probe"
DE_OUT = base.STAGING / "r18de-probe"
COMMON = dict(de.COMMON)
JOBS = (
    ("r18g-retreat", "arm", {**COMMON}),
    ("r18g-retreat-hunt-l1", "arm", {**COMMON, "hunt_scope": "l1-only"}),
)
NEW_KEYS = de.NEW_KEYS + ("hunt_scope",)


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ref = json.load(open(DE_OUT / "r18de-retreat.json"))
    base.log({"event": "R18G_PROBE_START", "seeds": base.SEEDS, "jobs": [j[0] for j in JOBS], "clock": "completion-l2-r18c"})
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
                base.log({"event": "R18G_PROBE_JOB_FAILED", "job": name, "rc": rc}); raise SystemExit(1)
            done[name] = json.load(open(out))
            base.log({"event": "R18G_PROBE_JOB_DONE", "job": name, "alive": done[name]["agg"].get("alive_n"),
                      "l2_deaths": done[name]["agg"].get("l2_deaths"), "l2_hazard_per_1k": done[name]["agg"].get("l2_hazard_per_1k")})
        running = still
    ctl, g = done["r18g-retreat"], done["r18g-retreat-hunt-l1"]
    regression_equal = de.rows_sha(strip(ctl["rows"])) == de.rows_sha(strip(ref["rows"]))
    ctl_rows = {r["seed"]: r for r in ctl["rows"]}
    g_rows = {r["seed"]: r for r in g["rows"]}
    prefix_diffs = [s for s in g_rows if r18c.prefix_through_first_l2(g_rows[s]) != r18c.prefix_through_first_l2(ctl_rows.get(s, {}))]
    table = {name: de.arm_stats(d) for name, d in done.items()}
    paired = base.paired(g["rows"], ctl["rows"])
    hyps = {"H1_hazard_lt_ctl": bool((table["r18g-retreat-hunt-l1"]["l2_hazard_per_1k"] or 9) < (table["r18g-retreat"]["l2_hazard_per_1k"] or 0)),
            "H2_paired_net_gt_0_and_ucb_lt_0": bool(paired["net"] > 0 and paired["ucb95_one_sided"] < 0),
            "H3_monsters_near0_lt_ctl": bool((table["r18g-retreat-hunt-l1"]["monsters_near0_at_trigger_mean"] or 99) < (table["r18g-retreat"]["monsters_near0_at_trigger_mean"] or 0)),
            "H4_kills_per_1k_le_ctl_expected_cost": bool((table["r18g-retreat-hunt-l1"]["l2_kills_per_1k_beats"] or 0) <= (table["r18g-retreat"]["l2_kills_per_1k_beats"] or 0)),
            "H5_l3_plus": {n: table[n]["l3"] for n in table}}
    verdict = "R18G_REPORT_ONLY" if (regression_equal and not prefix_diffs) else "R18G_VOID"
    summary = {"table": table, "paired_hunt_l1_vs_retreat": paired, "hypotheses": hyps,
               "regression_control_equal_to_r18de": regression_equal, "prefix_diff_seeds": prefix_diffs, "verdict": verdict,
               "seeds": base.SEEDS, "clock": "completion-l2-r18c", "elapsed_s": round(time.time() - t0, 1)}
    json.dump(summary, open(OUT / "R18G-SUMMARY.json", "w"), ensure_ascii=False, indent=1)
    base.log({"event": "R18G_PROBE_DONE", "verdict": verdict, "hypotheses": hyps, "paired": paired,
              "regression_control_equal_to_r18de": regression_equal, "elapsed_s": summary["elapsed_s"]})
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("loops", "occupancy_after_first_ascent")} for k, v in table.items()}, ensure_ascii=False, indent=1))
    print(json.dumps({"paired": paired, "hypotheses": hyps, "regression": regression_equal, "prefix_diffs": prefix_diffs}, ensure_ascii=False, indent=1))
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
