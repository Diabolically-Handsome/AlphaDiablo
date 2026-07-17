"""B1-E5:F-lock 复合签名判别脚本 + 经理不变式判别器 v2(PREREG-B1 E5/D3)。

复合签名(常量照 PREREG-B1 D3 逐字,案中禁调):种子 s 判"F-lock 候选" iff
  (i)   s ∈ 同池 king×H 参照 depth≥2 种子集;
  (ii)  腿×H 档案该种子 D 窗数 = 0(D 窗数 := mode_seq 中 "D" 计数);
  (iii) 该种子 F 窗 τ 中位 ∈ [25, 40] 地板区(闭区间;τ 中位来自 B1-E4
        重放报告 per_seed.farm_tau_median——档案不存逐窗 τ,重放系唯一通道)。
(ii)∧(iii) 合取强制——健康王座 τ==25 亦达 71.4%(P5 ②),τ 地板单用禁作判据。

分诊后件:∧ 判别器 v2 判"轨迹可变" → F-lock 型;判"不变(含近失)" →
工人损伤候选,且携 E5 语义收窄限定。

判别器 v2(docs/assets/manager_invariant_registry.json v2 同口径):
  - strict_invariant:13 字段(含 mode_seq 逐字符)精确相等(v1 判据);
  - near_miss(P4:7017 型勘正):结局字段 {ret, depth, died, kills} 相等
    ∧ 轨迹字段 {farm_n, farm_tau_mean, farm_tau_sum, farm_descend, windows,
    beats, overrides, cap, mode_seq} 至少一项可变;
  - variable:结局字段即不相等(轨迹可变)。

强制限定(随一切判词,PREREG-B1 D3):
  - 签名特异性未标定(案内无健康腿对照,假阳性率未知;转化条款见 D3);
  - 判别器语义收窄:"经理不变式"只证"换考官无效",不证"损失不可回收";
  - horizon 限定:一切签名条件于 max-steps 3000 评测协议,本案禁调 horizon。

用法:
  .venv/bin/python train/probe_composite_signature.py \
      --ref-archive <king×H 同池参照.json> --leg-h-archive <腿×H.json> \
      --leg-m29-archive <腿×M29.json> --leg-replay <B1-E4 报告.json> \
      --out <report.json>
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

# ---- D3 常量(预登记,案中禁调) ----
TAU_FLOOR_LO = 25.0
TAU_FLOOR_HI = 40.0            # 闭区间 [25, 40]
OUTCOME_FIELDS = ("ret", "depth", "died", "kills")
TRAJECTORY_FIELDS = ("farm_n", "farm_tau_mean", "farm_tau_sum",
                     "farm_descend", "windows", "beats", "overrides",
                     "cap", "mode_seq")
INVARIANT_FIELDS_V1 = OUTCOME_FIELDS + TRAJECTORY_FIELDS   # 13 字段
MANDATORY_CAVEATS = (
    "签名特异性未标定:案内无健康腿对照,假阳性率未知(PREREG-B1 D3 残余⑬;"
    "转化条款:RB.4 落'不复现'时 P8 腿命中率升格为假阳性率首实测)",
    "判别器语义收窄(承 P7 裁定三):'经理不变式'只证'换考官无效',"
    "不证'损失不可回收'",
    "horizon 限定:签名条件于评测协议 horizon(max-steps 3000);"
    "'D 窗 = 0'系 horizon 内陈述(sov_7029 距过零仅 8 窗先例)",
)


def d_window_count(row: dict) -> int:
    """D 窗数 := mode_seq 中 'D' 计数(†死亡符不影响 'D' 字符本身)。"""
    return str(row["mode_seq"]).count("D")


def invariant_class_v2(row_h: dict, row_m29: dict) -> str:
    """同工人异经理两行 → strict_invariant / near_miss / variable。"""
    if all(row_h[f] == row_m29[f] for f in INVARIANT_FIELDS_V1):
        return "strict_invariant"
    if (all(row_h[f] == row_m29[f] for f in OUTCOME_FIELDS)
            and any(row_h[f] != row_m29[f] for f in TRAJECTORY_FIELDS)):
        return "near_miss"
    return "variable"


def composite_signature(ref_rows: dict[int, dict], leg_rows: dict[int, dict],
                        tau_median_by_seed: dict[int, float | None]) -> dict:
    """逐种子复合签名判定。输入均为 {seed: row};τ 中位来自 E4 重放报告。"""
    ref_d2 = sorted(s for s, r in ref_rows.items() if r["depth"] >= 2)
    per_seed = {}
    for s in ref_d2:
        if s not in leg_rows:
            raise ValueError(f"腿档案缺参照 depth≥2 种子 {s}")
        leg_d = d_window_count(leg_rows[s])
        tau_med = tau_median_by_seed.get(s)
        cond_ii = leg_d == 0
        cond_iii = (tau_med is not None
                    and TAU_FLOOR_LO <= float(tau_med) <= TAU_FLOOR_HI)
        per_seed[s] = {
            "cond_i_ref_depth2": True,
            "cond_ii_leg_d_windows_zero": cond_ii,
            "leg_d_windows": leg_d,
            "cond_iii_tau_floor": cond_iii,
            "farm_tau_median": tau_med,
            "signature_hit": bool(cond_ii and cond_iii),
        }
    hits = sorted(s for s, v in per_seed.items() if v["signature_hit"])
    return {"ref_depth2_seeds": ref_d2, "n_ref_depth2": len(ref_d2),
            "per_seed": per_seed, "hits": hits, "n_hits": len(hits)}


def triage(signature: dict, leg_h_rows: dict[int, dict],
           leg_m29_rows: dict[int, dict]) -> dict:
    """签名命中后件分诊:轨迹可变 → F-lock 型;不变(含近失)→ 工人损伤候选。"""
    out = {}
    for s in signature["hits"]:
        if s not in leg_m29_rows:
            raise ValueError(f"M29 侧档案缺种子 {s},分诊不可判")
        cls = invariant_class_v2(leg_h_rows[s], leg_m29_rows[s])
        out[s] = {
            "invariant_class_v2": cls,
            "verdict": ("F-lock 型" if cls == "variable"
                        else "工人损伤候选(携 E5 语义收窄限定)"),
        }
    n_flock = sum(1 for v in out.values() if v["verdict"] == "F-lock 型")
    return {"per_seed": out, "n_flock_type": n_flock,
            "n_worker_damage_candidate": len(out) - n_flock}


def rows_by_seed(doc: dict) -> dict[int, dict]:
    rows = {int(r["seed"]): r for r in doc["rows"]}
    if len(rows) != len(doc["rows"]):
        raise ValueError("档案含重复 seed")
    return rows


def main() -> int:
    from eval_contract import strict_json_loads

    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-archive", required=True,
                    help="同池 king×H 参照档案(depth≥2 种子集来源)")
    ap.add_argument("--leg-h-archive", required=True, help="腿×H 档案")
    ap.add_argument("--leg-m29-archive", required=True, help="腿×M29 档案")
    ap.add_argument("--leg-replay", required=True,
                    help="腿×H 之 B1-E4 重放报告(farm_tau_median 来源)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = {k: pathlib.Path(getattr(args, k)) for k in
             ("ref_archive", "leg_h_archive", "leg_m29_archive", "leg_replay")}
    payloads = {k: p.read_bytes() for k, p in paths.items()}
    ref = rows_by_seed(strict_json_loads(payloads["ref_archive"]))
    leg_h = rows_by_seed(strict_json_loads(payloads["leg_h_archive"]))
    leg_m29 = rows_by_seed(strict_json_loads(payloads["leg_m29_archive"]))
    replay = strict_json_loads(payloads["leg_replay"])
    if not replay.get("fidelity_ok", False):
        raise ValueError("E4 重放报告保真未过,τ 中位不可采信")
    tau = {int(s): v.get("farm_tau_median")
           for s, v in replay.get("per_seed", {}).items()}

    sig = composite_signature(ref, leg_h, tau)
    tri = triage(sig, leg_h, leg_m29)
    report = {
        "probe": "b1-composite-signature",
        "constants": {"tau_floor_band": [TAU_FLOOR_LO, TAU_FLOOR_HI],
                      "outcome_fields": list(OUTCOME_FIELDS),
                      "trajectory_fields": list(TRAJECTORY_FIELDS)},
        "inputs_sha256": {k: hashlib.sha256(v).hexdigest()
                          for k, v in payloads.items()},
        "signature": sig,
        "triage": tri,
        "mandatory_caveats": list(MANDATORY_CAVEATS),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"复合签名:命中 {sig['n_hits']}/{sig['n_ref_depth2']};"
          f"F-lock 型 {tri['n_flock_type']},工人损伤候选 "
          f"{tri['n_worker_damage_candidate']};报告已存 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
