"""危险度重算:今天的逐层 hazard、R14 几何拟合的外推极限、通关所需的目标值。"""
import json
import math
import pathlib

ST = pathlib.Path.home() / "AlphaDiablo" / "diablogym" / "train" / "runs" / "r10-staging"


def per_floor(path, label):
    d = json.load(open(path))
    rows = d["rows"]
    # 到达 L_k 的局数 / 其中死在 L_k 的局数(最深层即死亡层)
    reach = {}
    died_at = {}
    for r in rows:
        dep = int(r["depth"])
        for k in range(1, dep + 1):
            reach[k] = reach.get(k, 0) + 1
        if r["died"]:
            died_at[dep] = died_at.get(dep, 0) + 1
    print(f"\n{label}(n={len(rows)}, 6000 拍)")
    for k in sorted(reach):
        n, dd = reach[k], died_at.get(k, 0)
        print(f"  L{k}: 到达 {n:3d}  死在此层 {dd:3d}  条件死亡率 h({k}) = "
              f"{dd / n:.3f}")
    return reach, died_at


for f, lab in (("r17-0/cert-readiness-v3.json", "认证工人(旧)"),
               ("r17-0/arm-readiness-v3.json", "R16 修宪工人")):
    p = ST / f
    if p.is_file():
        per_floor(p, lab)

print("\n=== R14 几何拟合 h(d) = 0.23 × 1.53^(d-1) 的外推 ===")
for d in (1, 2, 3, 4, 5, 6, 8, 16):
    h = 0.23 * 1.53 ** (d - 1)
    print(f"  d={d:2d}: 形式值 {h:8.2f}" + ("  ← 概率上限 1.0,公式已失效" if h > 1 else ""))
d_star = 1 + math.log(1 / 0.23) / math.log(1.53)
print(f"  形式值触及 1.0 的层数 d* = {d_star:.2f}(即该拟合断言:当时的体制在 L4-5 必死)")

print("\n=== 打穿 16 层所需的逐层存活率(几何均值)===")
for target in (0.5, 0.2, 0.1, 0.02):
    s = target ** (1 / 16)
    print(f"  整轮成功率 {target:5.0%} → 每层存活 {s:.4f},即每层危险度 ≤ {1 - s:.3%}")
print("\n对照:今日 h(2) 见上表;R14 旧体制 h(2)=0.55、h(3)=0.67。")
