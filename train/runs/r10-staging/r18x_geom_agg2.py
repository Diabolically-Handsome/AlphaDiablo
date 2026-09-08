"""Compare the natural replay (shard-*) with the forced-a11 counterfactual (force-*)."""
import json, glob, statistics as st, sys
from collections import Counter

base = "/home/laure/AlphaDiablo/diablogym/train/runs/r10-staging/r18x-l2geom/"


def load(prefix):
    rows = []
    for f in sorted(glob.glob(base + prefix + "-*.json")):
        rows += json.load(open(f))["rows"]
    return {r["seed"]: r for r in rows}


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 1) if xs else None


A = load("shard")
B = load("force")
print("natural n", len(A), " forced n", len(B))
for name, R in (("natural", A), ("forced-a11", B)):
    rows = list(R.values())
    alive = [r for r in rows if not r["died"]]
    print("== %s: alive %d/%d; max_main_depth hist %s; death-floor hist %s; final clvl hist %s" % (
        name, len(alive), len(rows), dict(sorted(Counter(r["max_main_depth"] for r in rows).items())),
        dict(sorted(Counter(r["depth"] for r in rows if r["died"]).items())),
        dict(sorted(Counter(r["char_level"] for r in rows).items()))))
    V = [v for r in rows for v in r["visits"]]
    print("   L2 visits %d exit %s; L2 beats total %d; kills on L2 %d; dive decisions %d a11 %d forced %d" % (
        len(V), dict(Counter(v["exit_reason"] for v in V)), sum(v["beats"] for v in V), sum(v["kills"] for v in V),
        sum(v["dive_beats"] for v in V), sum(v["dive_actions"].get("11", 0) for v in V), sum(v.get("forced_a11", 0) for v in V)))
    fe = [r["first_entry"] for r in rows if r["first_entry"]]
    l3 = [r for r in rows if r["max_main_depth"] >= 3]
    print("   L2 reach %d; L3 reach %d; L4+ reach %d; alive among L3-reachers %d; median steps alive %s dead %s" % (
        len(fe), len(l3), sum(1 for r in rows if r["max_main_depth"] >= 4), sum(1 for r in l3 if not r["died"]),
        med(r["micro_steps"] for r in alive), med(r["micro_steps"] for r in rows if r["died"])))
    # time from first L2 entry to first descent-exit visit
    lat = []
    for r in rows:
        if not r["first_entry"]:
            continue
        dv = [v for v in r["visits"] if v["exit_reason"] == "descend"]
        if dv:
            lat.append(dv[0]["exit_beat"] - r["first_entry"]["beat"])
    print("   episodes descending from L2: %d; beats from first L2 entry to first 2->3 descent: med %s" % (len(lat), med(lat)))

print("== paired by seed (forced vs natural): depth gained %d, lost %d; survival saved %d lost %d" % (
    sum(1 for s in B if s in A and B[s]["max_main_depth"] > A[s]["max_main_depth"]),
    sum(1 for s in B if s in A and B[s]["max_main_depth"] < A[s]["max_main_depth"]),
    sum(1 for s in B if s in A and A[s]["died"] and not B[s]["died"]),
    sum(1 for s in B if s in A and not A[s]["died"] and B[s]["died"])))
for s in sorted(B):
    if s in A:
        a, b = A[s], B[s]
        print("  seed %d: natural depth %d %s clvl %d steps %d | forced depth %d %s clvl %d steps %d" % (
            s, a["max_main_depth"], "died" if a["died"] else "alive", a["char_level"], a["micro_steps"],
            b["max_main_depth"], "died" if b["died"] else "alive", b["char_level"], b["micro_steps"]))
