"""R18-C 零训练探针(2026-09-07 凌晨,主席「彻夜跑一个,明早看看有没有进展」):
把首降后的观察窗从 1800 拍放宽到 9000 拍(新配方 completion-l2-r18c,到达截止与观测分母不变),
看撤退接口在有时间的情况下能不能把"撤 → 恢复 → 再下"这个循环转起来。

两组配对(同池 2_133,同工人 7e31dc54,同教练 coach-v03,同经济 sustain-loot-v1):
  r18c-loot-retreat    (e-long)  retreat-v1 + completion-l2-r18c
  r18c-loot-noretreat  (d′-long) 对照,completion-l2-r18c
再与 T0″ 的两臂(1800 拍窗)作跨钟参考(只报不判)。

预注册判据(全部只报不判——这是机理探针,不是发射闸门):
  P1 再下:撤退臂中"到达上层后再次下到 L2"的局数,及其中活到钟响的局数;
  P2 深度:两臂最大主线深度分布(L3 及以上到达数)与幸存者钟响时所在层;
  P3 存活:配对 saved−lost 与 UCB95(与 T0″ 同法);L2 到达 + N 拍存活(显式报删失数,只对 r18c 两臂算 N=9000);
  P4 暴露:L2 每千拍风险率;首次上楼之后在 L2+ 的占用份额(Opus 审阅指出的关键指标:区分"在推进"和"躲在上层");
  P5 循环:每局成功撤退次数直方图,二次尝试次数与其成功率(未结的尝试单列);
  P6 前缀同一性:(d′-long) 每局到首次 L2 到达为止的 descents 前缀(拍、层、面板)与 T0″ 对照臂逐局相同;不同则 VOID。
用法:run_r18c_probe.py
"""
import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_t0 as base  # noqa: E402

OUT = base.STAGING / "r18c-probe"
T0PP_OUT = base.STAGING / "t0-double-prime"
COMMON = {"manager": "resource", "resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
          "resource_readiness_law": "coach-v03", "worker_time_protocol": "completion-l2-r18c",
          "resource_service_policy": "sustain-loot-v1"}
JOBS = (
    ("r18c-loot-retreat", "arm", {**COMMON, "resource_retreat": "retreat-v1"}),
    ("r18c-loot-noretreat", "arm", {**COMMON}),
)
PREFIX_KEYS = ("beat", "from_dlvl", "to_dlvl", "belt_heals", "hp", "max_hp", "armor_class", "char_level", "hit_damage")


def launch(name, worker, overrides):
    import subprocess
    out = OUT / f"{name}.json"
    cmd = [base.PY, str(base.PROBE), str(base.WORKERS[worker]), base.SEEDS, str(out), str(base.MAX_STEPS),
           base.DECODING, json.dumps({**base.R16_FORM, **overrides}, ensure_ascii=False)]
    logf = open(OUT / f"{name}.log", "w")
    return name, out, subprocess.Popen(cmd, cwd=base.ROOT, stdout=logf, stderr=subprocess.STDOUT)


def retreat_of(row):
    return ((row.get("resource") or {}).get("retreat") or {})


def first_l2_arrival(row):
    for d in row.get("descents") or []:
        if d.get("to_dlvl") == 2:
            return int(d["beat"])
    return None


def l2_plus(doc, n):
    """Alive at (first L2 arrival + n), anchored on the L2 arrival where the follow-up starts.
    Explicit censoring: survivors whose episode ended before the horizon are reported, never dropped."""
    reached = observed = alive = censored = 0
    for r in doc["rows"]:
        a = first_l2_arrival(r)
        if a is None:
            continue
        reached += 1
        end = int(r["micro_steps"])
        if r["died"]:
            death_beat = int((r.get("death") or {}).get("beat", end))
            observed += 1
            alive += int(death_beat > a + n)
        elif end >= a + n:
            observed += 1; alive += 1
        else:
            censored += 1
    return {"n": n, "reached_l2": reached, "observed": observed, "alive": alive, "censored_survivors": censored}


def occupancy_after_first_ascent(doc):
    """Share of beats spent on L2+ after the first successful ascent (retreat arm); survivor end depth."""
    shares, end_depth_hist, n = [], {}, 0
    for r in doc["rows"]:
        asc = [a for a in retreat_of(r).get("attempts", []) if a.get("outcome") == "ascended"]
        if not asc:
            continue
        t0 = int(asc[0]["beat1"])
        deep = total = 0
        for f in r.get("floors") or []:
            s, e = int(f["entry_beat"]), int(f["exit_beat"])
            if e <= t0:
                continue
            seg = e - max(s, t0)
            total += seg
            if int(f["dlvl"]) >= 2:
                deep += seg
        if total > 0:
            shares.append(deep / total); n += 1
        if not r["died"] and r.get("floors"):
            k = str(r["floors"][-1]["dlvl"])
            end_depth_hist[k] = end_depth_hist.get(k, 0) + 1
    shares.sort()
    return {"episodes": n, "l2plus_share_mean": round(sum(shares) / len(shares), 3) if shares else None,
            "l2plus_share_median": round(shares[len(shares) // 2], 3) if shares else None,
            "episodes_share_ge_0.5": sum(1 for s in shares if s >= 0.5),
            "survivor_end_depth_hist_after_ascent": end_depth_hist}


def survivor_end_depth(doc):
    hist = {}
    for r in doc["rows"]:
        if not r["died"] and r.get("floors"):
            k = str(r["floors"][-1]["dlvl"]); hist[k] = hist.get(k, 0) + 1
    return hist


def loop_stats(doc):
    rows = doc["rows"]
    redescended = redescended_alive = second_attempts = second_ok = second_unresolved = 0
    ascents_hist = {}
    for r in rows:
        attempts = retreat_of(r).get("attempts", [])
        asc = [a for a in attempts if a.get("outcome") == "ascended"]
        ascents_hist[str(len(asc))] = ascents_hist.get(str(len(asc)), 0) + 1
        if asc:
            first = asc[0]["beat1"]
            later_l2 = [d for d in (r.get("descents") or []) if d.get("to_dlvl") == 2 and d.get("beat", 0) > first]
            if later_l2:
                redescended += 1
                redescended_alive += int(not r["died"])
        if len(attempts) >= 2:
            second_attempts += 1
            second_ok += int(attempts[1].get("outcome") == "ascended")
            second_unresolved += int(attempts[1].get("outcome") is None)
    depth_hist = {}
    for r in rows:
        depth_hist[str(r["depth"])] = depth_hist.get(str(r["depth"]), 0) + 1
    unresolved = sum(1 for r in rows for a in retreat_of(r).get("attempts", []) if a.get("outcome") is None)
    unresolved_died = sum(1 for r in rows for a in retreat_of(r).get("attempts", []) if a.get("outcome") is None and r["died"])
    return {"episodes_redescended_after_ascent": redescended, "of_which_alive_at_end": redescended_alive,
            "ascents_per_episode_hist": ascents_hist, "episodes_with_second_attempt": second_attempts,
            "second_attempt_ascended": second_ok, "second_attempt_unresolved": second_unresolved,
            "attempts_unresolved_total": unresolved, "attempts_unresolved_in_died_episodes": unresolved_died,
            "max_depth_hist": depth_hist, "l3_plus": sum(1 for r in rows if int(r["depth"]) >= 3),
            "survivor_end_depth_hist": survivor_end_depth(doc)}


def prefix_through_first_l2(row):
    out = []
    for d in row.get("descents") or []:
        out.append({k: d.get(k) for k in PREFIX_KEYS})
        if d.get("to_dlvl") == 2:
            break
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    t0pp = {n: json.load(open(T0PP_OUT / f"{n}.json")) for n in ("t0pp-loot-retreat", "t0pp-loot-noretreat")}
    base.log({"event": "R18C_PROBE_START", "seeds": base.SEEDS, "jobs": [j[0] for j in JOBS],
              "clock": "completion-l2-r18c (12000 arrival / 6000 obs / 9000 follow-up)",
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
                base.log({"event": "R18C_PROBE_JOB_FAILED", "job": name, "rc": rc}); raise SystemExit(1)
            done[name] = json.load(open(out))
            base.log({"event": "R18C_PROBE_JOB_DONE", "job": name, "alive": done[name]["agg"].get("alive_n"),
                      "l2_reach": done[name]["agg"].get("l2_reach"), "l2_deaths": done[name]["agg"].get("l2_deaths"),
                      "l3": done[name]["agg"].get("l3"), "depth_hist": done[name]["agg"].get("depth_hist")})
        running = still
    ret, ctl = done["r18c-loot-retreat"], done["r18c-loot-noretreat"]
    # protocol receipts: the arms must have run under r18c, the references under v1
    protocols = {name: (d.get("r16_environment") or {}).get("worker_time_protocol")
                 for name, d in (("r18c-loot-retreat", ret), ("r18c-loot-noretreat", ctl), ("t0pp-loot-retreat", t0pp["t0pp-loot-retreat"]))}
    # P6: control descents prefix through the first L2 arrival must equal T0'' control, seed by seed
    long_rows = {r["seed"]: r for r in ctl["rows"]}
    short_rows = {r["seed"]: r for r in t0pp["t0pp-loot-noretreat"]["rows"]}
    diff_seeds = [s for s in long_rows if prefix_through_first_l2(long_rows[s]) != prefix_through_first_l2(short_rows.get(s, {}))]
    fd_equal = not diff_seeds and set(long_rows) == set(short_rows)
    table = {}
    for name, d, long_clock in (("r18c-loot-retreat", ret, True), ("r18c-loot-noretreat", ctl, True),
                                ("t0pp-loot-retreat (1800)", t0pp["t0pp-loot-retreat"], False),
                                ("t0pp-loot-noretreat (1800)", t0pp["t0pp-loot-noretreat"], False)):
        a = d["agg"]
        row = {"alive": a.get("alive_n"), "died": a.get("died"), "l2_reach": a.get("l2_reach"),
               "l2_deaths": a.get("l2_deaths"), "l2_beats_total": a.get("l2_beats_total"),
               "l2_hazard_per_1k": a.get("l2_hazard_per_1k"), "depth_hist": a.get("depth_hist"),
               "l3": a.get("l3"), "clvl_mean": a.get("clvl_mean"), "kills_mean": a.get("kills_mean"),
               "micro_steps_mean": a.get("micro_steps_mean"), "death_dlvl_hist": a.get("death_dlvl_hist"),
               "survivor_end_depth_hist": survivor_end_depth(d),
               "l2_arrival_plus_1800": l2_plus(d, 1800)}
        if long_clock:
            row["l2_arrival_plus_9000"] = l2_plus(d, 9000)
        table[name] = row
    paired = base.paired(ret["rows"], ctl["rows"])
    summary = {"table": table, "paired_retreat_vs_control": paired,
               "loops_retreat_arm": loop_stats(ret), "loops_control_arm": loop_stats(ctl),
               "occupancy_after_first_ascent_retreat_arm": occupancy_after_first_ascent(ret),
               "P6_control_prefix_equal_to_t0pp": fd_equal, "P6_diff_seeds": diff_seeds,
               "protocol_receipts": protocols,
               "verdict": "R18C_REPORT_ONLY" if fd_equal else "R18C_VOID_CONTROL_PREFIX_DIFFERS",
               "seeds": base.SEEDS, "clock": "completion-l2-r18c", "elapsed_s": round(time.time() - t0, 1),
               "bridge": str((base.ROOT / "build").resolve()),
               "rows_sha_retreat": hashlib.sha256(json.dumps(ret["rows"], sort_keys=True).encode()).hexdigest()[:16],
               "rows_sha_control": hashlib.sha256(json.dumps(ctl["rows"], sort_keys=True).encode()).hexdigest()[:16]}
    json.dump(summary, open(OUT / "R18C-SUMMARY.json", "w"), ensure_ascii=False, indent=1)
    base.log({"event": "R18C_PROBE_DONE", "verdict": summary["verdict"], "paired": paired,
              "loops_retreat_arm": summary["loops_retreat_arm"], "loops_control_arm": summary["loops_control_arm"],
              "occupancy": summary["occupancy_after_first_ascent_retreat_arm"], "protocol_receipts": protocols,
              "elapsed_s": summary["elapsed_s"]})
    print(json.dumps(table, ensure_ascii=False, indent=1))
    print(json.dumps({"paired": paired, "loops_retreat": summary["loops_retreat_arm"],
                      "loops_control": summary["loops_control_arm"],
                      "occupancy": summary["occupancy_after_first_ascent_retreat_arm"],
                      "P6": fd_equal, "P6_diff_seeds": diff_seeds, "protocols": protocols}, ensure_ascii=False, indent=1))
    print("VERDICT:", summary["verdict"])


if __name__ == "__main__":
    main()
