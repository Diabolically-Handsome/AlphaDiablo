"""R18 P0 零训练探针:撤退效应是否依赖卖装备经济(训练合同只允许 sustain-v6)。
两臂配对,同池 2_133,同工人 7e31dc54,时钟 completion-l2-r18c,经济 sustain-v6(不卖装备):
  p0-v6-retreat   l2-town-v1/full/sustain-v6/coach-v03 + retreat-v1
  p0-v6           同上,无撤退
只报不判:配对 saved-lost 与 UCB95、L2 死亡、风险率、循环统计;两臂首次 L2 到达前缀必须逐局相同(否则 VOID)。
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_t0 as base  # noqa: E402
import run_r18c_probe as r18c  # noqa: E402
import run_r18de_probe as de  # noqa: E402

OUT = base.STAGING / "r18p0-probe"
COMMON = {"manager": "resource", "resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
          "resource_readiness_law": "coach-v03", "worker_time_protocol": "completion-l2-r18c",
          "resource_service_policy": "sustain-v6"}
JOBS = (
    ("p0-v6-retreat", "arm", {**COMMON, "resource_retreat": "retreat-v1"}),
    ("p0-v6", "arm", {**COMMON}),
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
    base.log({"event": "R18P0_PROBE_START", "seeds": base.SEEDS, "jobs": [j[0] for j in JOBS], "clock": "completion-l2-r18c",
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
                base.log({"event": "R18P0_PROBE_JOB_FAILED", "job": name, "rc": rc}); raise SystemExit(1)
            done[name] = json.load(open(out))
            base.log({"event": "R18P0_PROBE_JOB_DONE", "job": name, "alive": done[name]["agg"].get("alive_n"),
                      "l2_deaths": done[name]["agg"].get("l2_deaths"), "l2_hazard_per_1k": done[name]["agg"].get("l2_hazard_per_1k")})
        running = still
    ret, ctl = done["p0-v6-retreat"], done["p0-v6"]
    ctl_rows = {r["seed"]: r for r in ctl["rows"]}
    ret_rows = {r["seed"]: r for r in ret["rows"]}
    prefix_diffs = [s for s in ret_rows if r18c.prefix_through_first_l2(ret_rows[s]) != r18c.prefix_through_first_l2(ctl_rows.get(s, {}))]
    table = {name: de.arm_stats(d) for name, d in done.items()}
    paired = base.paired(ret["rows"], ctl["rows"])
    verdict = "R18P0_REPORT_ONLY" if not prefix_diffs else "R18P0_VOID_PREFIX"
    summary = {"table": table, "paired_retreat_vs_none": paired, "prefix_diff_seeds": prefix_diffs, "verdict": verdict,
               "seeds": base.SEEDS, "clock": "completion-l2-r18c", "economy": "sustain-v6", "elapsed_s": round(time.time() - t0, 1),
               "bridge": str((base.ROOT / "build").resolve())}
    json.dump(summary, open(OUT / "R18P0-SUMMARY.json", "w"), ensure_ascii=False, indent=1)
    base.log({"event": "R18P0_PROBE_DONE", "verdict": verdict, "paired": paired,
              "alive": {n: t["alive"] for n, t in table.items()}, "l2_deaths": {n: t["l2_deaths"] for n, t in table.items()},
              "l2_hazard": {n: t["l2_hazard_per_1k"] for n, t in table.items()}, "elapsed_s": summary["elapsed_s"]})
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("loops", "occupancy_after_first_ascent")} for k, v in table.items()}, ensure_ascii=False, indent=1))
    print(json.dumps({"paired": paired, "prefix_diffs": prefix_diffs}, ensure_ascii=False, indent=1))
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
