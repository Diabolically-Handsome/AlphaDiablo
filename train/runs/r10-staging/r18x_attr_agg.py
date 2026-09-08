"""R18-X death bestiary v0: damage taken and deaths by monster type per floor (heuristic attribution)."""
import csv, glob, json
from collections import Counter, defaultdict

base = "/home/laure/AlphaDiablo/diablogym/train/runs/r10-staging/r18x-l2geom/"
names = {}
with open("/home/laure/alphadiablo-dev/devilutionX/assets/txtdata/monsters/monstdat.tsv", encoding="utf-8") as fh:
    rd = csv.reader(fh, delimiter="\t")
    header = next(rd)
    idx_name = header.index("name"); idx_ai = header.index("ai"); idx_lvl = header.index("level")
    idx_hp = header.index("hitPointsMaximum"); idx_dmg = header.index("maxDamage")
    for i, row in enumerate(rd):
        if row:
            names[i] = (row[idx_name], row[idx_ai], row[idx_lvl], row[idx_hp], row[idx_dmg])

rows = []
for f in sorted(glob.glob(base + "attr-[abc].json")):
    rows += json.load(open(f))["rows"]
print("rows", len(rows), "died", sum(1 for r in rows if r["died"]))


def label(key):
    if key == "unknown":
        return "unknown"
    t, ai = key.split(":")
    n = names.get(int(t))
    return "%s(mlvl %s, hp<=%s, dmg<=%s)" % (n[0], n[2], n[3], n[4]) if n else key


for floor in ("1", "2", "3"):
    dmg = Counter(); hits = Counter()
    for r in rows:
        for k, v in (r["dmg_by_floor_type"].get(floor) or {}).items():
            dmg[label(k)] += v
        for k, v in (r["hits_by_floor_type"].get(floor) or {}).items():
            hits[label(k)] += v
    total = sum(dmg.values())
    print("== floor %s: damage taken %d over %d hits" % (floor, total, sum(hits.values())))
    for k, v in dmg.most_common(12):
        print("   %-42s dmg %5d (%4.1f%%)  hits %4d  per-hit %.1f" % (k, v, 100 * v / max(1, total), hits[k], v / max(1, hits[k])))
    killers = Counter(label(r["killer"]["key"]) for r in rows if r["died"] and r["killer"] and str(r["killer"]["dlvl"]) == floor)
    near = [r["killer"]["near_count"] for r in rows if r["died"] and r["killer"] and str(r["killer"]["dlvl"]) == floor]
    print("   deaths on floor %s: %d; killer (last hit) %s; monsters within 6 at the last hit: %s" % (
        floor, sum(killers.values()), dict(killers.most_common()), sorted(near)))
