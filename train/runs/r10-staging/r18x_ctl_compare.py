"""Paired comparison: full-interface control (retreat+portal+hunt-l1+loot) vs last night's arms, per seed."""
import json, glob, statistics as st
from collections import Counter

base = "/home/laure/AlphaDiablo/diablogym/train/runs/r10-staging/"


def load(paths):
    rows = []
    for p in paths:
        rows += json.load(open(p))["rows"]
    return {r["seed"]: r for r in rows}


arms = {
    "retreat": load([base + "r18f-probe/r18f-retreat.json"]),
    "retreat+portal": load([base + "r18f-probe/r18f-retreat-portal.json"]),
    "retreat+hunt_l1": load([base + "r18g-probe/r18g-retreat-hunt-l1.json"]),
    "FULL(ret+portal+hunt_l1)": load(sorted(glob.glob(base + "r18-arm-control/ctl-full-*.json"))),
}


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 1) if xs else None


for name, R in arms.items():
    rows = list(R.values())
    alive = [r for r in rows if not r["died"]]
    l2 = [r for r in rows if r["resource"]["max_main_depth_reached"] >= 2]
    l2b = sum(r["l2_beats"] or 0 for r in rows)
    l2d = sum(1 for r in rows if r["died"] and r["death"]["dlvl"] == 2)
    port = [r["resource"].get("portal") for r in rows if r["resource"].get("portal")]
    ret = [r["resource"].get("retreat") for r in rows if r["resource"].get("retreat")]
    print("== %-26s n %d alive %d L2 %d L3 %d | L2 deaths %d hazard/1k %.3f | gold_final med %s | clvl>=4 %d | retreats %d/%d asc | portals started %s" % (
        name, len(rows), len(alive), len(l2), sum(1 for r in rows if r["resource"]["max_main_depth_reached"] >= 3),
        l2d, 1000 * l2d / max(1, l2b), med(r["resource"]["gold_final"] for r in rows),
        sum(1 for r in rows if r["char_level"] >= 4),
        sum(len(x.get("attempts", [])) for x in ret), sum(x.get("ascended", 0) or 0 for x in ret),
        sum(r["resource"].get("portals_started") or 0 for r in rows)))
    # death context
    dl2 = [r for r in rows if r["died"] and r["death"]["dlvl"] == 2]
    print("   L2 death: window_opt %s; belt %s; beats_on_floor med %s; retreat active at death %d; portal active %d" % (
        dict(Counter(r["death"]["window_opt"] for r in dl2)), dict(Counter(r["death"]["belt_heals_last_live"] for r in dl2)),
        med(r["death"]["beats_on_current_floor"] for r in dl2),
        sum(1 for r in dl2 if (r["resource"].get("retreat") or {}).get("active")),
        sum(1 for r in dl2 if (r["resource"].get("portal") or {}).get("active"))))

F = arms["FULL(ret+portal+hunt_l1)"]
for other in ("retreat", "retreat+portal", "retreat+hunt_l1"):
    O = arms[other]
    common = sorted(set(F) & set(O))
    saved = sum(1 for s in common if O[s]["died"] and not F[s]["died"])
    lost = sum(1 for s in common if not O[s]["died"] and F[s]["died"])
    print("FULL vs %-18s: n %d saved %d lost %d net %+d" % (other, len(common), saved, lost, saved - lost))
print("seed | retreat | +portal | +hunt_l1 | FULL   (alive/died, depth, gold_final, portals, retreats)")
for s in sorted(F):
    def cell(r):
        if r is None:
            return "   -   "
        p = r["resource"].get("portals_started") or 0
        rt = len(((r["resource"].get("retreat") or {}).get("attempts", [])))
        return "%s d%d g%3d p%d r%d" % ("A" if not r["died"] else "x", r["resource"]["max_main_depth_reached"], r["resource"]["gold_final"], p, rt)
    print(s, "|", " | ".join(cell(arms[k].get(s)) for k in arms))
