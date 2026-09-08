"""Aggregate the R18-X L2 stairs-geometry replay shards and check identity vs the recorded arm."""
import json, glob, statistics as st, sys
from collections import Counter

base = "/home/laure/AlphaDiablo/diablogym/train/runs/r10-staging/"
rows = []
changed = set()
for f in sorted(glob.glob(base + "r18x-l2geom/shard-*.json")):
    d = json.load(open(f))
    rows += d["rows"]
    changed |= set(d["source_changed_during_run"])
print("rows", len(rows), "source_changed", sorted(changed))

ref = {r["seed"]: r for r in json.load(open(base + "r18f-probe/r18f-retreat.json"))["rows"]}
mism = []
for r in rows:
    q = ref.get(r["seed"])
    if q is None:
        continue
    if (q["micro_steps"], q["depth"], q["died"], q["kills"], q["char_level"]) != (
            r["micro_steps"], r["depth"], r["died"], r["kills"], r["char_level"]):
        mism.append((r["seed"], (q["micro_steps"], q["depth"], q["died"], q["kills"]),
                     (r["micro_steps"], r["depth"], r["died"], r["kills"])))
print("identity vs r18f-retreat rows: mismatches", len(mism), mism[:5])


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 1) if xs else None


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


fe = [r["first_entry"] for r in rows if r["first_entry"]]
print("== first L2 entry (n=%d)" % len(fe))
print("  d0 (cheb to down-stairs at arrival): med", med(f["d0"] for f in fe), "q25", q([f["d0"] for f in fe], .25),
      "q75", q([f["d0"] for f in fe], .75), "hist", dict(sorted(Counter(min(f["d0"], 40) // 5 * 5 for f in fe).items())))
print("  up_to_down stairs distance: med", med(f["up_to_down"] for f in fe), "q25", q([f["up_to_down"] for f in fe], .25), "q75", q([f["up_to_down"] for f in fe], .75))
print("  roster: med", med(f["roster"] for f in fe), " monsters in the corridor box: med", med(f["mons_between"] for f in fe),
      " near stairs(<=8): med", med(f["mons_near_stairs"] for f in fe))
print("  clvl hist", Counter(f["clvl"] for f in fe), "ac hist", dict(sorted(Counter(f["ac"] for f in fe).items())))

V = [v for r in rows for v in r["visits"]]
print("== L2 visits n=%d exit hist %s" % (len(V), dict(Counter(v["exit_reason"] for v in V))))
print("  d0 med", med(v["d0"] for v in V), " dmin med", med(v["dmin"] for v in V),
      " visits with dmin<=2:", sum(1 for v in V if v["dmin"] is not None and v["dmin"] <= 2),
      " dmin<=5:", sum(1 for v in V if v["dmin"] is not None and v["dmin"] <= 5))
desc = [v for v in V if v["exit_reason"] == "descend"]
print("  descend visits:", [(v["seed"], v["d0"], v["dmin"], v["beats"], v["kills"], v["dive_beats"], dict(v["dive_actions"])) for v in desc])
near = [v for v in V if v["d0"] is not None and v["d0"] <= 6]
print("  visits arriving within 6 tiles of the stairs: %d; of those descended %d, died %d, ascended %d" % (
    len(near), sum(1 for v in near if v["exit_reason"] == "descend"), sum(1 for v in near if v["exit_reason"] == "death"),
    sum(1 for v in near if v["exit_reason"] == "ascend")))
tot_dive = sum(v["dive_beats"] for v in V)
a11 = sum(v["dive_actions"].get("11", 0) for v in V)
a9 = sum(v["dive_actions"].get("9", 0) for v in V)
a10 = sum(v["dive_actions"].get("10", 0) for v in V)
avail = sum(v["a11_avail_dive_beats"] for v in V)
print("== DIVE-window worker decisions on L2: %d; a11 chosen %d; a11 available %d; a9 %d; a10 %d; action hist %s" % (
    tot_dive, a11, avail, a9, a10, dict(sorted(sum((Counter(v["dive_actions"]) for v in V), Counter()).items(), key=lambda kv: int(kv[0])))))
d5 = sum(v["dive_d_le5_beats"] for v in V)
print("  DIVE decisions taken within 5 tiles of the stairs: %d; a11 available then %d; a11 chosen then %d" % (
    d5, sum(v["dive_d_le5_a11_avail"] for v in V), sum(v["dive_d_le5_a11"] for v in V)))
alld = sum(v["beats_obs"] for v in V)
print("  all worker decisions on L2: %d; a11 available %d; a11 chosen %d; a11 available at d<=2: %d" % (
    alld, sum(v["a11_avail_beats"] for v in V), sum(v["actions"].get("11", 0) for v in V), sum(v["a11_avail_d_le2_beats"] for v in V)))
pc = Counter()
for v in V:
    for k, x in v["path_checks"].items():
        if isinstance(x, int):
            pc[k] += x
print("== path checks (every 10th decision): n %d; monster-avoiding path reaches stairs %d; free path reaches %d; avoid None %d; mean end-distance avoid %.1f free %.1f" % (
    pc["n"], pc["avoid_reach"], pc["free_reach"], pc["avoid_none"], pc["avoid_end_d_sum"] / max(1, pc["n"]), pc["free_end_d_sum"] / max(1, pc["n"])))
print("  hp_min med", med(v["hp_min"] for v in V), " near monsters mean per decision", round(sum(v["near_sum"] for v in V) / max(1, alld), 2))
json.dump({"rows": rows}, open(base + "r18x-l2geom/R18X-ALL.json", "w"), ensure_ascii=False)
