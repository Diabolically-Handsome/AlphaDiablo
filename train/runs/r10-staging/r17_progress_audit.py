"""进度审计:跨轮次的深度前沿与成长指标(只读)。"""
import json
import pathlib

ROOT = pathlib.Path.home() / "AlphaDiablo" / "diablogym"
EV = ROOT / "train" / "runs" / "eval-assembled"
ST = ROOT / "train" / "runs" / "r10-staging"

print("=== 考卷:深度直方 + 死亡(魔鬼卷 a 池,128 种子)===")
for tag in ("r10-cand-a", "r13-arm-a-smart-xdevil-a", "r13-arm-a2-shaping-xdevil-a",
            "r14-arm-d-escrow-power-xdevil-a", "r15-arm-c-readiness3-xdevil-a",
            "r16-anchor-xdevil-a", "r16-arm-a-constitution-xdevil-a"):
    p = EV / f"{tag}.json"
    if not p.is_file():
        print(f"{tag:44s} (缺)")
        continue
    d = json.load(open(p))
    a = d["agg"]
    rows = d["rows"]
    deep_alive = [r["seed"] for r in rows
                  if r.get("depth", 0) >= 3 and not r.get("died")]
    print(f"{tag:44s} died {a['died']:3d}/128  l3+ {a['l3']:3d}  "
          f"hist {a.get('depth_hist')}  深层存活 {len(deep_alive)}")

print()
print("=== 部署探针(readiness 形态,采样解码)===")
for label, p in (("R15 arm-c 16 种子/3000 拍", ST / "r15-deploy-armc.json"),
                 ("R16 认证工人 16/6000", ST / "r16-deploy-anchor.json"),
                 ("R16 修宪工人 16/6000", ST / "r16-deploy-arm.json"),
                 ("R17.0 认证 48/6000", ST / "r17-0" / "cert-readiness-v3.json"),
                 ("R17.0 修宪臂 48/6000", ST / "r17-0" / "arm-readiness-v3.json")):
    if not p.is_file():
        print(f"{label:28s} (缺 {p.name})")
        continue
    d = json.load(open(p))
    a = d["agg"]
    rows = d["rows"]
    n = a["n"]
    deep_alive = sum(1 for r in rows if r.get("depth", 0) >= 3 and not r["died"])
    maxdepth = max(r.get("depth", 0) for r in rows)
    print(f"{label:28s} n {n:2d} 存活 {n - a['died']:2d} "
          f"hist {a.get('depth_hist')} 最深 L{maxdepth} 深层存活 {deep_alive} "
          f"clvl {a.get('clvl_mean')} kills {a.get('kills_mean')} "
          f"xp/1k {a.get('xp_per_1k_mean')}")
