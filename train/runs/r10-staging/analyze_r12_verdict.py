"""R12 终判分析:四道门 x 两臂,配对基线 r10-cand(魔鬼x旧工) / r10-anchor(M29x旧工)。"""
import hashlib, json, math, pathlib

EV = pathlib.Path.home() / "AlphaDiablo" / "diablogym" / "train" / "runs" / "eval-assembled"

def load(tag):
    d = json.load(open(EV / f"{tag}.json"))
    rows = sorted(d["rows"], key=lambda r: r["seed"])
    sha = hashlib.sha256(json.dumps(d["rows"], sort_keys=True).encode()).hexdigest()[:16]
    return rows, d["agg"], sha

def lcb(deltas, z=1.645):
    n = len(deltas); m = sum(deltas)/n
    sd = math.sqrt(sum((x-m)**2 for x in deltas)/(n-1)) if n > 1 else 0.0
    return m, m - z*sd/math.sqrt(n), m + z*sd/math.sqrt(n)

BASE = {"devil": {"a": "r10-cand-a", "b": "r10-cand-b"},
        "m29":   {"a": "r10-anchor-a-rep1", "b": "r10-anchor-b-rep1"}}

# 锚位一致性(rep1 vs rep2)
for p in "ab":
    s1 = load(f"r10-anchor-{p}-rep1")[2]; s2 = load(f"r10-anchor-{p}-rep2")[2]
    print(f"anchor-{p} rep1/rep2 sha: {s1} / {s2} identical={s1==s2}")

out = {}
for arm in ("r12-arm1-m29", "r12-arm2-smart"):
    res = {}
    # ---- 魔鬼卷:门A/门B/门D ----
    for p in "ab":
        nrows, nagg, nsha = load(f"{arm}-xdevil-{p}")
        orows, oagg, osha = load(BASE["devil"][p])
        assert [r["seed"] for r in nrows] == [r["seed"] for r in orows]
        b = sum(1 for x, y in zip(orows, nrows) if x["died"] and not y["died"])
        c = sum(1 for x, y in zip(orows, nrows) if not x["died"] and y["died"])
        dd = [y["depth"] - x["depth"] for x, y in zip(orows, nrows)]
        m, lo, hi = lcb(dd)
        res[f"devil-{p}"] = {
            "sha_new": nsha, "sha_old": osha, "bit_identical": nsha == osha,
            "died_new": sum(1 for r in nrows if r["died"]), "died_old": sum(1 for r in orows if r["died"]),
            "mcnemar_b_saved": b, "mcnemar_c_lost": c,
            "depth_delta_mean": round(m, 4), "depth_delta_lcb": round(lo, 4)}
    # ---- M29卷:门C ----
    for p in "ab":
        nrows, nagg, nsha = load(f"{arm}-xm29-{p}")
        orows, oagg, osha = load(BASE["m29"][p])
        assert [r["seed"] for r in nrows] == [r["seed"] for r in orows]
        dret = [y["ret"] - x["ret"] for x, y in zip(orows, nrows)]
        dkil = [y["kills"] - x["kills"] for x, y in zip(orows, nrows)]
        dl3 = sum(1 for r in nrows if r["depth"] >= 3)
        ol3 = sum(1 for r in orows if r["depth"] >= 3)
        res[f"m29-{p}"] = {
            "sha_new": nsha, "bit_identical": nsha == osha,
            "ret_new": round(nagg["ret_mean"], 2), "ret_old": round(oagg["ret_mean"], 2),
            "ret_ratio": round(nagg["ret_mean"]/oagg["ret_mean"], 4),
            "kills_new": round(nagg["kills_mean"], 2), "kills_old": round(oagg["kills_mean"], 2),
            "kills_ratio": round(nagg["kills_mean"]/oagg["kills_mean"], 4),
            "ret_delta_lcb": round(lcb(dret)[1], 2), "kills_delta_lcb": round(lcb(dkil)[1], 2),
            "died_new": sum(1 for r in nrows if r["died"]), "died_old": sum(1 for r in orows if r["died"]),
            "l3_new": dl3, "l3_old": ol3}
    out[arm] = res

# ---- 臂间对比(M29 同池同种子配对)----
for p in "ab":
    r1 = load(f"r12-arm1-m29-xm29-{p}")[0]; r2 = load(f"r12-arm2-smart-xm29-{p}")[0]
    dret = [y["ret"] - x["ret"] for x, y in zip(r1, r2)]  # arm2 - arm1
    dkil = [y["kills"] - x["kills"] for x, y in zip(r1, r2)]
    m1, lo1, hi1 = lcb(dret); m2, lo2, hi2 = lcb(dkil)
    out[f"arm2_minus_arm1_m29_{p}"] = {"ret_delta_mean": round(m1,2), "ret_ci90": [round(lo1,2), round(hi1,2)],
                                        "kills_delta_mean": round(m2,2), "kills_ci90": [round(lo2,2), round(hi2,2)]}

print(json.dumps(out, indent=1, ensure_ascii=False))
