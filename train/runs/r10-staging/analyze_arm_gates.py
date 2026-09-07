"""R13+ 战役通用四门配对对判器。用法: analyze_arm_gates.py <arm-tag-prefix>"""
import json
import math
import pathlib
import sys

EV = pathlib.Path.home() / "AlphaDiablo" / "diablogym" / "train" / "runs" / "eval-assembled"


def load(tag):
    d = json.load(open(EV / f"{tag}.json"))
    return sorted(d["rows"], key=lambda r: r["seed"]), d["agg"]


def lcb(deltas, z=1.645):
    n = len(deltas)
    m = sum(deltas) / n
    sd = (math.sqrt(sum((x - m) ** 2 for x in deltas) / (n - 1))
          if n > 1 else 0.0)
    return m, m - z * sd / math.sqrt(n)


def mcnemar(old_rows, new_rows):
    b = sum(1 for x, y in zip(old_rows, new_rows)
            if x["died"] and not y["died"])
    c = sum(1 for x, y in zip(old_rows, new_rows)
            if not x["died"] and y["died"])
    return b, c, (c - b) + 1.645 * math.sqrt(max(1, b + c))


def main():
    arm = sys.argv[1]
    # R16 新世界:第 2 参数为新法锚前缀(如 r16-anchor),锚档案
    # {prefix}-xdevil-{p} / {prefix}-xm29-{p};缺省 = 旧法锚(R13-R15 口径)。
    anchor_prefix = sys.argv[2] if len(sys.argv) > 2 else None
    if anchor_prefix:
        pairs = tuple((p, f"{anchor_prefix}-xdevil-{p}",
                       f"{anchor_prefix}-xm29-{p}") for p in ("a", "b"))
    else:
        pairs = (("a", "r10-cand-a", "r10-anchor-a-rep1"),
                 ("b", "r10-cand-b", "r10-anchor-b-rep1"))
    out = {"anchor_prefix": anchor_prefix or "old-law(r10-cand/r10-anchor)"}
    for p, devil_anchor, m29_anchor in pairs:
        dn, dna = load(f"{arm}-xdevil-{p}")
        do, doa = load(devil_anchor)
        assert [r["seed"] for r in dn] == [r["seed"] for r in do]
        b, c, ucb = mcnemar(do, dn)
        m, lo = lcb([y["depth"] - x["depth"] for x, y in zip(do, dn)])
        out[f"devil-{p}"] = {
            "died": (sum(1 for r in dn if r["died"]),
                     sum(1 for r in do if r["died"])),
            "mcnemar_saved_lost": (b, c), "ucb95": round(ucb, 2),
            "gateA": bool(ucb < 0
                          and sum(1 for r in dn if r["died"]) <= 110),
            "depth_delta_mean": round(m, 3), "depth_lcb": round(lo, 3),
            "gateB": bool(lo > -0.10),
            "l3": (dna["l3"], doa["l3"]),
            "depth_med": (dna.get("depth_median"),
                          doa.get("depth_median"))}
        mn, mna = load(f"{arm}-xm29-{p}")
        mo, moa = load(m29_anchor)
        out[f"m29-{p}"] = {
            "ret_ratio": round(mna["ret_mean"] / moa["ret_mean"], 4),
            "kills_ratio": round(
                mna["kills_mean"] / moa["kills_mean"], 4),
            "gateC": bool(
                mna["ret_mean"] >= 0.95 * moa["ret_mean"]
                and mna["kills_mean"] >= 0.95 * moa["kills_mean"]),
            "l3": (mna["l3"], moa["l3"])}
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
