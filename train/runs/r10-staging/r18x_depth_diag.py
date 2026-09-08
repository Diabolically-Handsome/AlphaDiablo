"""Why does the pair not descend from L2?  Read-only diagnosis over probe rows.

usage: python depth_diag.py label=path.json [label=path.json ...]
"""
import json, sys, statistics as st
from collections import Counter


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 1) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def hist(xs):
    return dict(sorted(Counter(xs).items(), key=lambda kv: str(kv[0])))


def floor_at(row, beat, main_only=True):
    for f in row["floors"]:
        if f["entry_beat"] <= beat < (f["exit_beat"] if f["exit_beat"] is not None else 10**9):
            return f
    return None


def analyse(label, path):
    d = json.load(open(path))
    rows = d["rows"]
    n = len(rows)
    out = {"label": label, "n": n, "manager": d.get("manager"), "max_steps": d.get("max_steps")}
    alive = [r for r in rows if not r["died"]]
    out["alive"] = len(alive)
    out["final_depth_hist"] = hist(r["depth"] for r in rows)
    out["max_main_depth_hist"] = hist(r["resource"]["max_main_depth_reached"] for r in rows)
    out["l2_reach"] = sum(1 for r in rows if r["resource"]["max_main_depth_reached"] >= 2)
    out["l3_reach"] = sum(1 for r in rows if r["resource"]["max_main_depth_reached"] >= 3)

    # ---- L2 floor visits
    l2 = [(r, f) for r in rows for f in r["floors"] if f["dlvl"] == 2]
    out["l2_visits"] = len(l2)
    out["l2_visit_exit_hist"] = hist(f["exit_reason"] for _, f in l2)
    out["l2_visit_beats_med"] = med(f["beats"] for _, f in l2)
    out["l2_visit_kills_med"] = med(f["kills"] for _, f in l2)
    out["l2_visit_kills_total"] = sum(f["kills"] for _, f in l2)
    out["l2_beats_total"] = sum(f["beats"] for _, f in l2 if f["beats"] is not None)
    out["l2_kills_per_1k"] = round(1000 * out["l2_visit_kills_total"] / max(1, out["l2_beats_total"]), 2)
    out["l2_monsters_left_at_exit_med"] = med(f["monsters_left_at_exit"] for _, f in l2)
    first_l2 = {}
    for r in rows:
        for f in r["floors"]:
            if f["dlvl"] == 2:
                first_l2[r["seed"]] = f
                break
    out["l2_first_visit_monsters_at_entry_est_med"] = med(
        (f["kills"] + f["monsters_left_at_exit"]) for f in first_l2.values())
    l1 = [f for r in rows for f in r["floors"] if f["dlvl"] == 1]
    out["l1_beats_total"] = sum(f["beats"] for f in l1 if f["beats"] is not None)
    out["l1_kills_per_1k"] = round(1000 * sum(f["kills"] for f in l1) / max(1, out["l1_beats_total"]), 2)

    # ---- DIVE windows by floor
    dw_by_floor = Counter()
    dw_l2 = []
    dw_l1 = []
    for r in rows:
        for w in r["dive_windows"]:
            f = floor_at(r, w["beat0"])
            dl = f["dlvl"] if f else None
            dw_by_floor[dl] += 1
            if dl == 2:
                dw_l2.append((r, w))
            elif dl == 1:
                dw_l1.append((r, w))
    out["dive_windows_by_floor"] = dict(dw_by_floor)
    out["dive_l2_n"] = len(dw_l2)
    out["dive_l2_end_hist"] = hist(w["end_reason"] for _, w in dw_l2)
    out["dive_l2_trigger_hist"] = hist(w["trigger"] for _, w in dw_l2)
    out["dive_l2_forced_hist"] = hist(w.get("forced_reason") for _, w in dw_l2)
    out["dive_l2_tau_med"] = med(w["tau"] for _, w in dw_l2)
    out["dive_l2_tau_sum"] = sum(w["tau"] for _, w in dw_l2)
    out["dive_l2_share_of_l2_beats"] = round(out["dive_l2_tau_sum"] / max(1, out["l2_beats_total"]), 3)
    out["dive_l2_descended"] = sum(1 for _, w in dw_l2 if w["descended"])
    out["dive_l1_n"] = len(dw_l1)
    out["dive_l1_end_hist"] = hist(w["end_reason"] for _, w in dw_l1)
    out["dive_l1_tau_med"] = med(w["tau"] for _, w in dw_l1)
    # episodes reaching L2 that never opened a DIVE window on L2
    seeds_l2 = {r["seed"] for r, _ in l2}
    seeds_dive_l2 = {r["seed"] for r, _ in dw_l2}
    out["l2_episodes_without_any_l2_dive_window"] = len(seeds_l2 - seeds_dive_l2)
    # first DIVE window on L2: how long after first L2 arrival
    lat = []
    for r in rows:
        f2 = first_l2.get(r["seed"])
        if not f2:
            continue
        ws = [w for w in r["dive_windows"] if f2["entry_beat"] <= w["beat0"] < (f2["exit_beat"] or 10**9)]
        if ws:
            lat.append(min(w["beat0"] for w in ws) - f2["entry_beat"])
    out["first_l2_dive_latency_med"] = med(lat)
    out["first_l2_dive_latency_n"] = len(lat)

    # ---- descents
    desc = [(r, x) for r in rows for x in r["descents"]]
    out["descent_pairs_hist"] = hist(f'{x["from_dlvl"]}->{x["to_dlvl"]}' for _, x in desc)
    d12 = [x for _, x in desc if x["from_dlvl"] == 1 and x["to_dlvl"] == 2]
    d23 = [x for _, x in desc if x["from_dlvl"] == 2 and x["to_dlvl"] == 3]
    out["d12_n"] = len(d12)
    out["d12_clvl_hist"] = hist(x["char_level"] for x in d12)
    out["d12_ac_hist"] = hist(x["armor_class"] for x in d12)
    out["d12_hp_frac_mean"] = mean(x["hp"] / max(1, x["max_hp"]) for x in d12)
    out["d12_trigger_hist"] = hist(x["trigger"] for x in d12)
    out["d12_forced_reason_hist"] = hist(x["forced_reason"] for x in d12)
    out["d12_floor_beats_before_med"] = med(x["floor_beats_before_descent"] for x in d12)
    out["d12_monsters_left_prev_med"] = med(x["monsters_left_prev_floor"] for x in d12)
    out["d23_n"] = len(d23)
    out["d23_detail"] = [{k: x[k] for k in ("beat", "char_level", "armor_class", "hp", "max_hp", "trigger",
                                              "forced_reason", "floor_beats_before_descent",
                                              "floor_kills_before_descent", "monsters_left_prev_floor")}
                         for x in d23]

    # ---- deaths on L2
    dl2 = [r for r in rows if r["died"] and r["death"]["dlvl"] == 2]
    out["l2_deaths"] = len(dl2)
    out["l2_death_window_opt_hist"] = hist(r["death"]["window_opt"] for r in dl2)
    out["l2_death_beats_on_floor_med"] = med(r["death"]["beats_on_current_floor"] for r in dl2)
    out["l2_death_belt_hist"] = hist(r["death"]["belt_heals_last_live"] for r in dl2)
    out["l2_death_potions_reachable_mean"] = mean(r["death"]["floor_potions_reachable"] for r in dl2)
    out["l2_death_clvl_hist"] = hist(r["char_level"] for r in dl2)
    out["l2_death_ac_hist"] = hist(r["armor_class"] for r in dl2)
    out["l2_death_during_dive_window"] = sum(
        1 for r in dl2 if any(w["beat0"] <= r["death"]["beat"] <= (w["beat1"] or 10**9)
                              for w in r["dive_windows"]))
    # retreat context at death
    ret_active = 0
    for r in dl2:
        rt = (r["resource"].get("retreat") or {})
        if rt.get("active"):
            ret_active += 1
    out["l2_death_retreat_active"] = ret_active

    # ---- worker level at L2 arrival vs end
    out["clvl_final_hist"] = hist(r["char_level"] for r in rows)
    out["clvl_final_alive_hist"] = hist(r["char_level"] for r in alive)
    out["ac_final_hist"] = hist(r["armor_class"] for r in rows)
    out["gold_final_med"] = med(r["gold"] for r in rows)
    out["l2_beats_per_episode_med"] = med(r["l2_beats"] for r in rows if r["l2_beats"])
    out["episode_micro_steps_med"] = med(r["micro_steps"] for r in rows)
    out["alive_final_dlvl_hist"] = hist(r["depth"] for r in alive)
    # windows aggregate
    out["windows_sum"] = {k: sum(r["windows"].get(k, 0) or 0 for r in rows)
                          for k in ("total", "farm", "dive", "resupply", "forced_dive", "mask_forced",
                                    "fallback", "coach_dive_wants", "coach_cleared_true")}
    out["retreats"] = {
        "attempts": sum(len((r["resource"].get("retreat") or {}).get("attempts", [])) for r in rows),
        "ascended": sum((r["resource"].get("retreat") or {}).get("ascended", 0) or 0 for r in rows),
        "failed": sum((r["resource"].get("retreat") or {}).get("failed", 0) or 0 for r in rows),
    }
    return out


def main():
    res = []
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        res.append(analyse(label, path))
    for o in res:
        print("=" * 100)
        print(o["label"])
        for k, v in o.items():
            if k == "label":
                continue
            print(f"  {k}: {json.dumps(v, ensure_ascii=False)}")
    json.dump(res, open("/tmp/depth_diag.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
