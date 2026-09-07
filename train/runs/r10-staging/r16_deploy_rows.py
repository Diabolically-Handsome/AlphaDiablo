"""R16 部署探针逐种子对照(臂 vs 认证工人),附 L2 战备比。"""
import json
import sys
import pathlib

ROOT = pathlib.Path.home() / "AlphaDiablo" / "diablogym"
sys.path.insert(0, str(ROOT / "python"))
from diablogym.worker_env import readiness_power_ratio_v2  # noqa: E402

STAGING = ROOT / "train" / "runs" / "r10-staging"
for who in ("arm", "anchor"):
    d = json.load(open(STAGING / f"r16-deploy-{who}.json"))
    rows = d["rows"]
    print("==", who, "probe", d["probe"], "manager", d.get("manager"),
          "decoding", d["decoding"], "max_steps", d["max_steps"])
    for r in sorted(rows, key=lambda r: r["seed"]):
        raw = {"char_level": r["char_level"], "armor_class": r["armor_class"],
               "item_max_damage": r["hit_damage"], "damage_mod": 0,
               "max_hp": r["max_hp"], "fire_resist": 0}
        try:
            rr = round(readiness_power_ratio_v2(raw, 2), 2)
        except Exception as e:  # noqa: BLE001
            rr = str(e)[:40]
        print("seed %d depth %d died %d steps %5d clvl %d AC %2d dmg %2d hp %3d "
              "kills %3d left_here %3d ready_L2 %s" % (
                  r["seed"], r["depth"], int(r["died"]), r["micro_steps"],
                  r["char_level"], r["armor_class"], r["hit_damage"],
                  r["max_hp"], r["kills"], r["monsters_left_here"], rr))
    by = {}
    for r in rows:
        k = "L%d-%s" % (r["depth"], "died" if r["died"] else "alive")
        by[k] = by.get(k, 0) + 1
    print("depth x died:", dict(sorted(by.items())))
