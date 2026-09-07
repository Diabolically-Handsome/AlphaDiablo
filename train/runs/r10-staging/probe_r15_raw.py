"""R15 战备门前置探针:live raw 字段清单 + 战士初始面板 + damage_mod 语义。"""
import sys

from diablogym.env import DiabloGymEnv


def main():
    env = DiabloGymEnv(max_steps=3000, descend_ladder=True,
                       death_ladder=True, start_in_dungeon=True)
    try:
        env.reset(seed=999983)
        raw = env._raw
        keys = sorted(raw.keys())
        print("RAW KEYS:", len(keys))
        scalars = {}
        for k in keys:
            v = raw[k]
            if isinstance(v, (int, float, bool, str)):
                scalars[k] = v
                print("  %s = %r" % (k, v))
            else:
                try:
                    n = len(v)
                except TypeError:
                    n = "?"
                print("  %s : %s len=%s" % (k, type(v).__name__, n))
        gear = raw.get("gear") or raw.get("equipment") or raw.get("items")
        if isinstance(gear, (list, tuple)) and gear:
            print("GEAR[0] sample:", gear[0])
    finally:
        env.close()


if __name__ == "__main__":
    main()
