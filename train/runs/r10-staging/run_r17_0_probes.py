# 复核修复(2026-09-06):~/r17_work/review/R17-0-WP-ADVERSARIAL-REVIEW-20260906.md H1/H2/H4/M2/M5
"""R17.0 仪表跑批驱动(零训练、零 C++;panel memo §三 R17.0(a))。

顺序:
  1. 逐位回归(矩阵之前):probe_r17_deployment(readiness-v3, arm 工人,
     2114000-2114015, 6000 拍, sample, R16 部署形态)的 v3 列行 sha 必须等于
     r16-deploy-arm.json 的行 sha(v3 agg 键亦须相等);不等则台账
     R17_0_REGRESSION_FAIL 并退出,不跑矩阵。
  2. 矩阵:{arm=7e31dc54, cert=r9 认证工人} × {readiness-v3, readiness-v3-strict,
     const-FARM, const-DIVE} × 2114000-2114047(48)× 6000 拍 × sample × R16 形态;
     最多 3 个探针进程并行(各自 torch 单线程);输出
     r17-0/<worker>-<manager>.json + 同名 .log。
  3. 汇总 r17-0/r17-0-summary.json;台账 R17_0_PROBE_DELIVERED(回归通过后)
     与 R17_0_BASELINE_A0(汇总后)。
用法:run_r17_0_probes.py [--skip-regression] [--summary-only] [--concurrency N]
种子:仅 2114000-2114047(已消耗池 2_114);不触碰任何处女池。
"""
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path.home() / "AlphaDiablo" / "diablogym"
RUNS = ROOT / "train" / "runs"
STAGING = RUNS / "r10-staging"
OUT = STAGING / "r17-0"
LEDGER = STAGING / "r13_ledger.jsonl"
PY = str(ROOT / ".venv" / "bin" / "python")
PROBE = STAGING / "probe_r17_deployment.py"
R16_ARM_ARCHIVE = STAGING / "r16-deploy-arm.json"

sys.path.insert(0, str(STAGING))
from probe_r17_deployment import V3_ROW_KEYS, rows_sha_v3  # noqa: E402

WORKERS = {
    "arm": RUNS / "r16-arm-a-constitution" / "model_candidate.zip",
    "cert": RUNS / "r9-reeducation" / "staging" / "worker.zip",
}
MANAGERS = ("readiness-v3", "readiness-v3-strict", "const-FARM", "const-DIVE")
# 配对比较的参照臂(M5:v3 vs 其余三经理,共享种子)
REFERENCE_MANAGER = "readiness-v3"
R16_FORM = {"explore_global_hunt": True, "farm_scene_cap": 3600,
            "reset_layer_clock_on_window": True, "reward_economy": "v4",
            "drink_sovereignty": True}
MATRIX_SEEDS = "2114000-2114047"
REGRESSION_SEEDS = "2114000-2114015"
MAX_STEPS = 6000
DECODING = "sample"
DEFAULT_CONCURRENCY = 3
# v3 agg 键(回归口径:与 r16-deploy-arm.json 的 agg 逐项相等)
V3_AGG_KEYS = (
    "n", "died", "victories", "depth_hist", "l3", "l5", "depth_mean",
    "clvl_mean", "ac_mean", "kills_mean", "gold_mean", "hit_damage_mean",
    "xp_mean", "xp_per_1k_mean", "micro_steps_mean", "alive_n",
    "alive_clvl_mean", "alive_ac_mean", "alive_kills_mean", "alive_gold_mean",
    "alive_xp_per_1k_mean", "median_steps_alive", "dead_n", "dead_clvl_mean",
    "dead_ac_mean", "dead_kills_mean", "dead_gold_mean", "dead_xp_per_1k_mean",
    "median_steps_dead", "dead_post_death_ac_mean", "dead_post_death_gold_mean",
    "dead_post_death_hit_damage_mean")
SUMMARY_AGG_KEYS = (
    "n", "died", "alive_n", "l2_reach", "depth_hist", "depth_mean",
    "descents_total", "forced_descents", "farm_masked_at_descent_share",
    "mask_forced_descent_share",
    "ready_descents", "ready_descent_share", "descent_trigger_hist",
    "descent_forced_reason_hist", "cleared_true_decisions",
    "episodes_with_descent",
    "mean_beat_first_descent", "median_beat_first_descent",
    "alive_at_first_descent_plus_1800", "alive_at_first_descent_plus_1800_n",
    "fd_plus_1800_observed_n", "fd_plus_1800_censored_n",
    "l2_beats_total", "l2_deaths",
    "l2_hazard_per_1k", "l1_beats_total", "l1_kills_total",
    "l1_kills_per_1k_beats", "clvl_mean", "ac_mean", "kills_mean",
    "hit_damage_mean", "xp_per_1k_mean", "alive_clvl_mean", "alive_ac_mean",
    "alive_kills_mean", "dead_clvl_mean", "dead_ac_mean", "dead_kills_mean",
    "median_steps_dead", "descent_ac_hist", "descent_ac_mean",
    "descent_belt_hist", "descent_belt_mean", "descent_clvl_hist",
    "descent_hp_frac_mean", "descent_ratio_v2_mean",
    "descent_floor_beats_median", "descent_monsters_left_prev_floor_median",
    "dive_windows_total", "dive_windows_descended",
    "dive_window_end_reason_hist", "dive_window_tau_median",
    "windows_total", "mask_forced_decisions_total",
    "forced_dive_windows_total", "death_dlvl_hist",
    "death_beats_on_floor_median", "death_belt_last_live_hist",
    "death_floor_potions_visible_mean", "death_floor_potions_reachable_mean",
    "death_floor_potions_total_mean")


def now():
    return time.strftime("%H:%M:%S")


def say(msg):
    print(f"[{now()}] {msg}", flush=True)


def log(event):
    event = {"t": now(), **event}
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    say("[ledger] " + json.dumps(event, ensure_ascii=False)[:400])


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def form_json(manager):
    return json.dumps({**R16_FORM, "manager": manager}, ensure_ascii=False,
                      separators=(",", ":"))


def probe_cmd(worker_zip, seeds, out_json, manager):
    return [PY, str(PROBE), str(worker_zip), seeds, str(out_json),
            str(MAX_STEPS), DECODING, form_json(manager)]


def launch(name, worker_zip, seeds, out_json, manager):
    log_path = OUT / f"{name}.log"
    fh = open(log_path, "w")
    cmd = probe_cmd(worker_zip, seeds, out_json, manager)
    fh.write("$ " + " ".join(cmd) + "\n")
    fh.flush()
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=ROOT)
    say(f"launch {name}: {' '.join(cmd[1:])}")
    return {"name": name, "proc": proc, "fh": fh, "t0": time.time(),
            "out": out_json}


def finish(job):
    rc = job["proc"].wait()
    job["fh"].close()
    dt = time.time() - job["t0"]
    say(f"done {job['name']} rc={rc} in {dt:.0f}s")
    return rc, dt


def regression():
    """回归:新探针 readiness-v3 × arm × 16 种子 vs r16-deploy-arm.json 逐位。"""
    out = OUT / "regress-arm-readiness-v3-16.json"
    job = launch("regress-arm-readiness-v3-16", WORKERS["arm"],
                 REGRESSION_SEEDS, out, "readiness-v3")
    rc, dt = finish(job)
    if rc != 0:
        log({"event": "R17_0_REGRESSION_FAIL", "reason": f"probe rc={rc}",
             "log": str(OUT / f"{job['name']}.log")})
        raise SystemExit(1)
    arch = json.loads(R16_ARM_ARCHIVE.read_text())
    new = json.loads(out.read_text())
    sha_arch = rows_sha_v3(arch["rows"])
    sha_new = rows_sha_v3(new["rows"])
    diff_seeds = []
    for ra, rn in zip(arch["rows"], new["rows"]):
        sub = {k: rn[k] for k in V3_ROW_KEYS}
        if sub != ra:
            diff_seeds.append({
                "seed": ra["seed"],
                "diff_keys": [k for k in V3_ROW_KEYS if sub[k] != ra[k]]})
    agg_eq = all(arch["agg"][k] == new["agg"][k] for k in V3_AGG_KEYS)
    result = {
        "archive": str(R16_ARM_ARCHIVE),
        "archive_probe": arch.get("probe"),
        "archive_rows_sha_v3": sha_arch,
        "probe_out": str(out),
        "probe_rows_sha_v3": sha_new,
        "rows_sha_equal": sha_arch == sha_new,
        "n_rows": [len(arch["rows"]), len(new["rows"])],
        "diff_seeds": diff_seeds,
        "agg_v3_equal": agg_eq,
        "elapsed_s": round(dt, 1),
        "source_changed_during_run": new.get("source_changed_during_run"),
    }
    (OUT / "regression.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1))
    say("REGRESSION " + json.dumps(result, ensure_ascii=False))
    if not (result["rows_sha_equal"] and agg_eq and not diff_seeds):
        log({"event": "R17_0_REGRESSION_FAIL", **result})
        raise SystemExit(1)
    return result


def run_matrix(concurrency):
    jobs = [(f"{w}-{m}", WORKERS[w], m) for w in WORKERS for m in MANAGERS]
    pending = list(jobs)
    running = []
    results = {}
    log({"event": "R17_0_MATRIX_LAUNCH", "jobs": [j[0] for j in jobs],
         "seeds": MATRIX_SEEDS, "max_steps": MAX_STEPS, "decoding": DECODING,
         "form": R16_FORM, "concurrency": concurrency,
         "probe_sha256": sha256(PROBE)})
    while pending or running:
        while pending and len(running) < concurrency:
            name, zip_path, manager = pending.pop(0)
            running.append(launch(name, zip_path, MATRIX_SEEDS,
                                  OUT / f"{name}.json", manager))
        time.sleep(5)
        still = []
        for job in running:
            if job["proc"].poll() is None:
                still.append(job)
                continue
            rc, dt = finish(job)
            results[job["name"]] = {"rc": rc, "elapsed_s": round(dt, 1)}
            if rc != 0:
                log({"event": "R17_0_PROBE_JOB_FAILURE", "job": job["name"],
                     "rc": rc, "log": str(OUT / f"{job['name']}.log")})
        running = still
    return results


def wilson95(k, n):
    """存活率的 Wilson 95% 区间(纯 Python,无 scipy;z=1.959964)。"""
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return [round(max(0.0, centre - half), 4),
            round(min(1.0, centre + half), 4)]


def mcnemar_exact_p(b, c):
    """McNemar 精确双侧 p(不一致数 b+c 上的 Binomial(0.5);b+c=0 记 1.0)。"""
    m = b + c
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(0, min(b, c) + 1))
    return round(min(1.0, 2.0 * tail / (2.0 ** m)), 4)


def paired_manager_comparison(alive_by_combo):
    """M5:同一工人下 readiness-v3 vs 其余三经理的配对比较(共享种子)。

    b = 仅参照臂存活,c = 仅对照臂存活(McNemar 不一致数);另给每臂存活率
    的 Wilson 95% 区间。四臂共用 MATRIX_SEEDS,故逐种子可配对。"""
    out = {}
    for w in WORKERS:
        ref = alive_by_combo.get(f"{w}-{REFERENCE_MANAGER}")
        if not ref:
            continue
        arms = {}
        for m in MANAGERS:
            other = alive_by_combo.get(f"{w}-{m}")
            if not other:
                continue
            shared = sorted(set(ref) & set(other))
            k = sum(1 for s in shared if other[s])
            entry = {
                "n_shared_seeds": len(shared),
                "alive": k,
                "alive_rate": (round(k / len(shared), 4) if shared else None),
                "wilson95": wilson95(k, len(shared)),
            }
            if m != REFERENCE_MANAGER:
                b = sum(1 for s in shared if ref[s] and not other[s])
                c = sum(1 for s in shared if other[s] and not ref[s])
                entry["mcnemar_vs_reference"] = {
                    "b_alive_reference_only": b,
                    "c_alive_other_only": c,
                    "discordant_n": b + c,
                    "exact_two_sided_p": mcnemar_exact_p(b, c)}
            arms[m] = entry
        out[w] = {"reference": REFERENCE_MANAGER, "arms": arms}
    return out


def dirty_tree_gate(summary):
    """H1 硬门:任一组合在代码树变动中跑出即整表作废(复核报告 H1)。"""
    dirty = {name: c["source_changed_during_run"]
             for name, c in summary["combos"].items()
             if c.get("source_changed_during_run")}
    if not dirty:
        return
    log({"event": "R17_0_MATRIX_DIRTY_TREE",
         "summary": str(OUT / "r17-0-summary.json"),
         "reason": "探针运行期间源码树变动,矩阵不可采信(须在最终字节上重跑)",
         "dirty_combos": dirty})
    raise SystemExit(1)


def summarize(job_results=None):
    combos = {}
    alive_by_combo = {}
    for w in WORKERS:
        for m in MANAGERS:
            name = f"{w}-{m}"
            path = OUT / f"{name}.json"
            if not path.exists():
                combos[name] = {"missing": True}
                continue
            doc = json.loads(path.read_text())
            agg = doc["agg"]
            alive_by_combo[name] = {
                int(r["seed"]): (not r["died"]) for r in doc["rows"]}
            entry = {"worker": w, "manager": m, "worker_zip": doc["worker_zip"],
                     "worker_sha16": doc["source_identity"]["worker_zip"][:16],
                     "alive": agg["n"] - agg["died"],
                     "ready_over_descents": (
                         f"{agg['ready_descents']}/{agg['descents_total']}"),
                     "rows_sha_v3": doc["v3_compat"]["rows_sha_v3"],
                     "source_changed_during_run": doc.get(
                         "source_changed_during_run")}
            for k in SUMMARY_AGG_KEYS:
                entry[k] = agg.get(k)
            if job_results and name in job_results:
                entry["job"] = job_results[name]
            combos[name] = entry
    regression_path = OUT / "regression.json"
    summary = {
        "doc": "r17-0-summary", "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": ("R17.0 A0' 基线 + 强制下楼份额测量(零训练,零 C++;"
                    "panel memo §三 R17.0(a))"),
        "seeds": MATRIX_SEEDS, "max_steps": MAX_STEPS, "decoding": DECODING,
        "form": R16_FORM, "managers": list(MANAGERS),
        "workers": {w: str(p) for w, p in WORKERS.items()},
        "probe": str(PROBE), "probe_sha256": sha256(PROBE),
        "regression": (json.loads(regression_path.read_text())
                       if regression_path.exists() else None),
        "metric_semantics": {
            "farm_masked_at_descent_share": (
                "下楼时 ¬mask[FARM] 的份额(旧名 forced_descent_share);"
                "只说 FARM 当时被封,与教练意愿无关——教练自愿 DIVE 时 FARM "
                "恰被封也计入,故它 ≥ 真正的「经理被推翻」份额"),
            "mask_forced_descent_share": (
                "trigger=mask_forced 的下楼份额,即教练不想 DIVE、靠掩码回退"
                "阶梯落到 DIVE 的份额;这才是「经理被推翻」的口径"),
            "cleared_true_decisions": (
                "教练 cleared 子句为真的决策数;为 0 即 readiness-v3 与 "
                "readiness-v3-strict 逐位相同是经验巧合而非结构必然"),
            "l2_hazard_per_1k": "L2 死亡数 / ΣL2 停留拍 × 1000",
            "alive_at_first_descent_plus_1800": (
                "首降拍+1800 时仍存活的份额;分母 = fd_plus_1800_observed_n "
                "(有下楼且随访窗被观测到的局)。存活但局末 < 首降拍+1800 者"
                "为右删失,**不计存活、不进分母**,只在 fd_plus_1800_censored_n "
                "计数;None=无可观测样本。注:MAX_STEPS=6000 且首降均值约 4500,"
                "该指标在当前预算下结构性近乎不可观测(复核报告 H2)"),
            "fd_plus_1800_observed_n": (
                "alive_at_first_descent_plus_1800 的观测分母(未删失的有下楼局数)"),
            "fd_plus_1800_censored_n": "有下楼但随访窗被局末截断(右删失)的局数",
            "l1_kills_per_1k_beats": "L1 击杀 / ΣL1 停留拍 × 1000(榨取率)",
            "descent_ac_hist/descent_belt_hist": "每次下楼时 AC / 腰带药数分布",
            "paired_manager_comparison": (
                "同一工人、共享种子上 readiness-v3 vs 其余三经理的配对比较:"
                "McNemar 不一致数(b=仅 v3 存活 / c=仅对照存活)+ 精确双侧 p,"
                "以及各臂存活率的 Wilson 95% 区间(复核报告 M5)")},
        "paired_manager_comparison": paired_manager_comparison(alive_by_combo),
        "combos": combos,
    }
    (OUT / "r17-0-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def main():
    args = sys.argv[1:]
    concurrency = DEFAULT_CONCURRENCY
    if "--concurrency" in args:
        concurrency = int(args[args.index("--concurrency") + 1])
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if "--summary-only" in args:
        summary = summarize()
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        dirty_tree_gate(summary)
        return
    if "--skip-regression" not in args:
        reg = regression()
        log({"event": "R17_0_PROBE_DELIVERED",
             "probe": str(PROBE), "probe_sha256": sha256(PROBE),
             "driver": str(STAGING / "run_r17_0_probes.py"),
             "managers": ["readiness-v1", *MANAGERS],
             "regression": {k: reg[k] for k in (
                 "archive_rows_sha_v3", "probe_rows_sha_v3", "rows_sha_equal",
                 "agg_v3_equal", "n_rows", "elapsed_s")},
             "telemetry": ["descents[] (beat/from/to/belt/hp/AC/clvl/dmg/"
                           "ratio_v2/ready/forced/trigger/forced_reason/"
                           "kills/floor_kills)",
                           "death (belt last-live+post, floor potions "
                           "total/visible/reachable, dlvl, beats_on_floor)",
                           "floors[] (beats/kills/monsters_left/exit_reason)",
                           "dive_windows[] (end_reason/tau/descended)",
                           "agg: farm_masked_at_descent_share/"
                           "mask_forced_descent_share/descents_total/"
                           "ready_descents/cleared_true_decisions/"
                           "mean_beat_first_descent/"
                           "alive_at_first_descent_plus_1800/l2_hazard_per_1k"]})
    results = run_matrix(concurrency)
    summary = summarize(results)
    # H1:任一组合在代码树变动中跑出即整表作废,不写 A0 基线
    dirty_tree_gate(summary)
    failed = [n for n, r in results.items() if r["rc"] != 0]
    brief = {}
    for name, c in summary["combos"].items():
        if c.get("missing"):
            brief[name] = "MISSING"
            continue
        brief[name] = {
            "alive": c["alive"], "l2_reach": c["l2_reach"],
            "farm_masked_at_descent_share": c["farm_masked_at_descent_share"],
            "mask_forced_descent_share": c["mask_forced_descent_share"],
            "ready_over_descents": c["ready_over_descents"],
            "l2_hazard_per_1k": c["l2_hazard_per_1k"],
            "alive_at_fd_plus_1800": c["alive_at_first_descent_plus_1800"],
            "alive_at_fd_plus_1800_n": c["alive_at_first_descent_plus_1800_n"],
            "mean_beat_first_descent": c["mean_beat_first_descent"],
            "clvl": c["clvl_mean"], "ac": c["ac_mean"],
            "kills": c["kills_mean"],
            "l1_kills_per_1k": c["l1_kills_per_1k_beats"],
            "descent_ac_hist": c["descent_ac_hist"],
            "descent_belt_hist": c["descent_belt_hist"],
            "descent_trigger_hist": c["descent_trigger_hist"]}
    log({"event": "R17_0_BASELINE_A0",
         "summary": str(OUT / "r17-0-summary.json"),
         "seeds": MATRIX_SEEDS, "max_steps": MAX_STEPS, "decoding": DECODING,
         "form": R16_FORM, "elapsed_s": round(time.time() - t0, 1),
         "failed_jobs": failed, "combos": brief})
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
