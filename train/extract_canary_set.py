"""B1-E8:W-C 金丝雀记分种子集机器提取器(PREREG-B1 W-C/E8)。

自指定评测档案提取 depth≥2 之全部种子(提取口径逐字承 run_v32_sovereign
depth2_count 先例:row["depth"] >= 2),并集健康对照 {7003, 7011}(P5 无
尖峰种子,阴性对照)。提取结果由驱动器以 CANARY_SET 事件写死(携 n_D 与
逐种子名单);本提取器随冻结 commit 入库——否则"以档案行为准"不可复验。

用法:
  .venv/bin/python train/extract_canary_set.py \
      [train/runs/eval-assembled/v32-ref-launch.json] [--controls 7003,7011]
输出:JSON 到 stdout(archive/archive_sha256/depth2_seeds/n_D/controls/C)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))

DEFAULT_ARCHIVE = ROOT / "train" / "runs" / "eval-assembled" / "v32-ref-launch.json"
DEFAULT_CONTROLS = (7003, 7011)   # PREREG-B1 W-C:P5 无尖峰健康对照,阴性对照


def depth2_seeds(rows: list[dict]) -> list[int]:
    """depth≥2 提取(照 run_v32_sovereign.depth2_count 先例,行口径同一)。"""
    if not isinstance(rows, list) or not rows:
        raise ValueError("档案 rows 缺失/为空")
    seeds = sorted(int(r["seed"]) for r in rows if r["depth"] >= 2)
    if len(seeds) != len(set(seeds)):
        raise ValueError("档案 rows 含重复 seed")
    return seeds


def extract(archive_path: str | pathlib.Path,
            controls: tuple[int, ...] = DEFAULT_CONTROLS) -> dict:
    from eval_contract import strict_json_loads

    p = pathlib.Path(archive_path)
    payload = p.read_bytes()
    doc = strict_json_loads(payload)
    d2 = depth2_seeds(doc["rows"])
    archive_seed_set = {int(r["seed"]) for r in doc["rows"]}
    bad_controls = [s for s in controls if s not in archive_seed_set]
    if bad_controls:
        raise ValueError(f"健康对照种子不在档案种子面内: {bad_controls}")
    overlap = sorted(set(d2) & set(controls))
    if overlap:
        raise ValueError(f"健康对照与 depth≥2 集相交,阴性对照失义: {overlap}")
    return {
        "archive": str(p),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "depth2_seeds": d2,
        "n_D": len(d2),
        "controls": sorted(controls),
        "C": sorted(set(d2) | set(controls)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", nargs="?", default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--controls", default=",".join(map(str, DEFAULT_CONTROLS)),
                    help="健康对照种子(逗号分隔;传空串 = 不并集对照)")
    args = ap.parse_args()
    controls = tuple(int(x) for x in args.controls.split(",") if x.strip())
    print(json.dumps(extract(args.archive, controls), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
