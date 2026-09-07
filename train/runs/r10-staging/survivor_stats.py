"""部署探针存活行重算(死亡行为尸检值:单机死亡剥装+金币减半,不可用于成长统计)。"""
import json
import pathlib

BASE = pathlib.Path.home() / "AlphaDiablo" / "diablogym" / "train" / "runs" / "r10-staging"
SETS = {
    "R15a v1 new (idle-exploit)": ["r15-deploy-new-a.json", "r15-deploy-new-b.json"],
    "R15c gate new": ["r15-deploy3-new-a.json", "r15-deploy3-new-b.json"],
    "base 乙 (v2 probe)": ["r15-deploy3-base-a.json", "r15-deploy3-base-b.json"],
    "base 乙 (v1 probe)": ["r15-deploy-base-a.json", "r15-deploy-base-b.json"],
}


def mean(rows, key):
    return round(sum(r.get(key, 0) for r in rows) / max(1, len(rows)), 2)


for name, files in SETS.items():
    rows = []
    for f in files:
        p = BASE / f
        if p.exists():
            rows += json.load(open(p))["rows"]
    if not rows:
        print(name, "missing")
        continue
    alive = [r for r in rows if not r["died"]]
    dead = [r for r in rows if r["died"]]
    dead_steps = sorted(r["micro_steps"] for r in dead)
    med = dead_steps[len(dead_steps) // 2] if dead_steps else None
    print(f"{name}: n={len(rows)} alive={len(alive)} | SURVIVORS "
          f"clvl={mean(alive, 'char_level')} ac={mean(alive, 'armor_class')} "
          f"dmg={mean(alive, 'hit_damage')} kills={mean(alive, 'kills')} "
          f"depth={mean(alive, 'depth')} "
          f"maxAC={max((r.get('armor_class', 0) for r in alive), default=0)} "
          f"| DEAD median_steps={med} kills={mean(dead, 'kills')}")
