#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G1 闸线重标定——核心定线计算库(PREREG-G1 rev2 · W-SCRIPT 之二)。

纯计算库:输入系 extract_g1_history.extract() 之提取字典,输出系确定性结果字典;
无文件写副作用、零训练零评测、不 import 任何训练/评测入口。

覆盖条款:
  * L1a 发射宽度肢:精确二项 k*(α_width=0.05,per-candidate 单肢边际语义);
  * L1b 发射均差肢:分族置换 Q95(逐种子符号翻转,10^5 次,种子 20260717;
    平局计法照卷:Δ=0 恒贡献 0、不计赢;去杠杆敏感度);
  * L2 死亡线:H 治下健康参照集跨池死亡上包络 + M=2;STRAT_TABLE(池×治下);
    LOO 单档依赖性;M∈{0,1,2,3} 与语境限定敏感度;L2-0 适用域(同值替换 (a)(b),
    明文不及于 (c) 金评 P 线);
  * L3 复现地板:双约束可行域 + 不变更条件先判(正典 85/92 系分数);
  * D2-U 簇指派:纯数值泛函 d̄<−15 损伤 / d̄>−10 健康 / 中间带不入簇;
  * OC 表:k∈[16,24] × m∈{2,4,…,28},边际 α(MC + 精确二项 n_eff)、联合 α、
    max-of-2 修正、合成备择(+5/+10/+15)功效;L2 之 M 行全表;
  * D5 全史对照表:新线 vs 现行线逐候选逐肢通过/拦截(L2-0 适用域口径),
    翻转集与强制注记;D5-LOO 单档依赖性子报告;
  * D3 双向检验矩阵(BIDIR)与 D6 would-trip 描述性统计。

置换零分布注册口径(NULL_DIST;整合官正文基准 17.32/23.40 之复现口径,试算复核命中):
  stdlib random.Random(seed);逐抽逐种子行主序生成翻转符号
  (r.random() < 0.5 → +1,否则 −1);两健康向量共用同一符号矩阵
  (等价于逐族同种子重播);Q95 取 numpy 线性插值分位(np.quantile, 0.95)。
"""

from __future__ import annotations

import json
import random
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 注册常数(方法冻结总则 1;案中禁改)
# ---------------------------------------------------------------------------

SEED = 20260717
N_DRAWS = 100_000
N_SEEDS = 32
ALPHA_WIDTH = 0.05          # L1a:per-candidate、单肢、边际
M_MARGIN = 2                # L2:注册边距
DELTA_FLOOR = 0.02          # L3:δ
CLUSTER_LO = -15.0          # D2-U:损伤簇阈
CLUSTER_HI = -10.0          # D2-U:健康簇阈
FLOOR_CANON = Fraction(85, 92)   # L3 正典系分数;0.9239 系展示舍入不作计算口径
K_GRID = list(range(16, 25))     # OC:k ∈ [16,24]
M_GRID = list(range(2, 29, 2))   # OC:m ∈ {2,4,…,28}(rev2 扩域)
POWER_SHIFTS = (5.0, 10.0, 15.0)  # D3-1 注册合成备择
M_SENS = (0, 1, 2, 3)            # L2:M 敏感度(rev2 补 M=0 行)

CURRENT_LINES = {
    "width_k": 18,        # 现行 赢 ≥18/32
    "mean_m": 4.0,        # 现行 均差 ≥+4
    "died_max": 6,        # 现行 died ≤6(资格闸 + 发射死亡肢;L2-0 适用域)
    "floor": FLOOR_CANON, # 现行 (85/92)×R
}

#: D2-U 封闭档案枚举(CLASS_UNIVERSE;案中禁增删)——L3 簇指派与 L1b 向量论域
UNIVERSE_TAGS = [
    "p8-full32", "v31-mfresh-full32", "v31-mcont-full32",
    "v32-sov-full32", "v32-ctrl-full32",
]
UNIVERSE_BASELINE = "v31-ref-launch"
UNIVERSE_EXCLUSIONS = [
    {"which": "v2 世代档(v29/v30 系等)", "reason": "分母异世界,比值跨世界无意义"},
    {"which": "M29 治下档(*-m29)", "reason": "治下异语境(D3-2 两语境分列禁混引)"},
    {"which": "s16 档", "reason": "宽度不足(n=16 < 32)"},
]

#: L1b 族论域(数据来源冻结):worker-leg = {P8 Δ};manager-arm = {fresh Δ}(每族 N=1)
FAMILY_VECTORS = {
    "worker-leg": ["p8-full32"],
    "manager-arm": ["v31-mfresh-full32"],
}

#: L2 P_H 封闭枚举(H 治下健康参照集;跨池上包络)
P_H_TAGS = ["v31-ref-launch", "b1-ref8k-launch", "r2-launch", "r2-throne",
            "r2-script", "r2-bcworker"]
#: L2 治下过滤降出主推导之档(敏感度并列)
P_M29_TAGS = ["v31-ref-science", "b1-ref8k-science", "r2-science"]

L2_SCOPE = ("L2-0 适用域:导出线同值替换 (a) 资格闸 与 (b) 发射线死亡肢;"
            "明文不及于 (c) 金评 P 线与名分流转条款(死>6 回退/死≤4 登基/金评死≤6,"
            "零触碰清单)。D5 反事实按此口径计算。")


# ---------------------------------------------------------------------------
# 确定性序列化(W-DET 用)
# ---------------------------------------------------------------------------

def canon_bytes(obj: Any) -> bytes:
    """规范化 JSON 字节串:键排序、紧凑分隔、float 最短往返表示(确定性)。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _f(x) -> float:
    return float(x)


# ---------------------------------------------------------------------------
# L1a 精确二项
# ---------------------------------------------------------------------------

def binom_tail_exact(n: int, k: int) -> Fraction:
    """P(X ≥ k | n, p=1/2) 精确值。"""
    if k <= 0:
        return Fraction(1)
    if k > n:
        return Fraction(0)
    from math import comb
    total = sum(comb(n, j) for j in range(k, n + 1))
    return Fraction(total, 2 ** n)


def l1a_width_line(n: int = N_SEEDS, alpha: float = ALPHA_WIDTH) -> dict:
    """k* := min{k : P(X≥k | n, .5) ≤ α},精确二项;含现行线 α 与全表。"""
    table = {k: binom_tail_exact(n, k) for k in range(0, n + 1)}
    k_star = min(k for k in range(0, n + 1) if table[k] <= Fraction(alpha).limit_denominator(10**6))
    return {
        "principle": "零改进零假设(逐种子符号翻转,确定性协议无测量噪声)下,"
                     "宽度肢单肢错误加冕率 α_width=0.05;k*=min{k:P(X≥k|n=32,p=.5)≤0.05},精确二项",
        "alpha": alpha,
        "n": n,
        "k_star": k_star,
        "tail_at": {str(k): _f(table[k]) for k in range(16, n + 1)},
        "tail_exact_at_kstar": [str(table[k_star].numerator), str(table[k_star].denominator)],
        "current_k": CURRENT_LINES["width_k"],
        "current_alpha": _f(table[CURRENT_LINES["width_k"]]),
        "nature": "复核",
        "band": None,
        "note": "点值,无带(纯算术零不确定性;草案伪带已废——rev2 统计席 m-1);"
                "方向注记:现行 18 单肢 α=%.4f;开奖史 11→16→17 从未触线,"
                "收紧不翻转任何历史发射结局(D5 复核)" % _f(table[18]),
    }


# ---------------------------------------------------------------------------
# 置换零分布(注册口径)
# ---------------------------------------------------------------------------

def flip_matrix(n_draws: int = N_DRAWS, width: int = N_SEEDS, seed: int = SEED) -> np.ndarray:
    """注册符号翻转矩阵:stdlib random.Random(seed),行主序,<0.5 → +1 否则 −1。"""
    r = random.Random(seed)
    n = n_draws * width
    flat = np.fromiter((1 if r.random() < 0.5 else -1 for _ in range(n)),
                       dtype=np.int8, count=n)
    return flat.reshape(n_draws, width)


def perm_null(delta: Sequence[float], F: np.ndarray) -> dict:
    """符号翻转零分布之均值统计量;Q95 取 np.quantile 线性插值。

    平局计法(rev2 统计席 m-2):Δ=0 系翻转不变量,恒贡献 0,如实计入(不剔行)。
    """
    d = np.asarray(delta, dtype=np.float64)
    means = (F * d).mean(axis=1)
    return {
        "n_draws": int(F.shape[0]),
        "q95": _f(np.quantile(means, 0.95)),
        "q50": _f(np.quantile(means, 0.50)),
        "se_flip": _f(np.sqrt((d ** 2).sum()) / d.size),
        "n_zero_deltas": int((d == 0).sum()),
        "_means": means,  # 内部复用(OC/联合 α);出结果前剥除
    }


def l1b_family_lines(deltas_by_family: Dict[str, Dict[str, List[float]]],
                     F: np.ndarray, seed: int = SEED) -> dict:
    """分族均差线:每族 m* := max(该族各健康向量 Q95);去杠杆敏感度;魔种注记。"""
    out: Dict[str, Any] = {"families": {}}
    for fam in sorted(deltas_by_family):
        vecs = deltas_by_family[fam]
        fam_out = {}
        for tag in sorted(vecs):
            d = np.asarray(vecs[tag], dtype=np.float64)
            stats = perm_null(d, F)
            stats.pop("_means")
            # 去杠杆(留一去最大杠杆种子;敏感度报告不裁决)
            i_max = int(np.argmax(np.abs(d)))
            d_lo = np.delete(d, i_max)
            F_lo = flip_matrix(F.shape[0], d_lo.size, seed)
            s_lo = perm_null(d_lo, F_lo)
            s_lo.pop("_means")
            fam_out[tag] = {
                "dbar": _f(d.mean()),
                "q95": stats["q95"],
                "q95_2dp": round(stats["q95"], 2),
                "se_flip": stats["se_flip"],
                "n_zero_deltas": stats["n_zero_deltas"],
                "max_leverage_index": i_max,
                "max_leverage_delta": _f(d[i_max]),
                "deleverage_q95": s_lo["q95"],
            }
        m_star = max(v["q95"] for v in fam_out.values())
        out["families"][fam] = {
            "vectors": fam_out,
            "m_star": _f(m_star),
            "m_star_2dp": round(m_star, 2),
            "n_vectors": len(fam_out),
        }
    ms = [v["m_star"] for v in out["families"].values()]
    out["cross_family_max"] = _f(max(ms))  # 旧形制,降为 OC 并列行(记不裁)
    out["principle"] = ("均差肢=效应量下限,由零改进零分布(逐种子符号翻转,10^5 次,"
                       "种子 20260717)Q95 推出;族限定:worker-leg 与 manager-arm 分立,"
                       "每族 m*=max(族内健康向量 Q95);后案按候选族引用对应 m*")
    out["nature"] = "复核"  # rev3 核认⑤:输入钉死+W-DET 下带系死码,改复核点值;MC 逼近误差降注记
    out["circularity_note"] = ("manager-arm 族唯一向量来自被资格闸拦截候选(fresh);"
                              "高方差推高均差线,系保守向非放行向;呈面板单项核认——"
                              "否决则该族判『无健康向量,不可定线』,维持现行 +4 并呈报")
    return out


# ---------------------------------------------------------------------------
# L2 死亡线
# ---------------------------------------------------------------------------

def l2_death_line(died_by_tag: Dict[str, int], m_margin: int = M_MARGIN) -> dict:
    p_h = {t: died_by_tag[t] for t in P_H_TAGS}
    p_m29 = {t: died_by_tag.get(t) for t in P_M29_TAGS}
    env = max(p_h.values())
    support = sorted(t for t, v in p_h.items() if v == env)
    line = env + m_margin
    # 治下过滤如实注记【复核】:并入 M29 档后 max 是否变
    env_mixed = max([v for v in p_h.values()] + [v for v in p_m29.values() if v is not None])
    # LOO(D5-LOO 协议;单档剔除重算线)
    loo = {}
    for t in P_H_TAGS:
        rest = {k: v for k, v in p_h.items() if k != t}
        env_t = max(rest.values())
        loo[t] = {"envelope": env_t, "line": env_t + m_margin,
                  "line_changed": (env_t + m_margin) != line}
    single_dep = sorted(t for t, v in loo.items() if v["line_changed"])
    # M 敏感度 + 语境限定行
    sens = {}
    for m in M_SENS:
        sens["M=%d" % m] = {"line": env + m, "in_main": m == m_margin}
    sens["H×7000 语境限定"] = {"line": p_h["v31-ref-launch"] + m_margin,
                              "note": "严格限考池之线=0+M=2,连健康候选 P8(died 1)都临线,荒谬紧"
                                      "(混池裁量论证,呈面板单项核认)"}
    sens["剔 K1"] = {"line": loo["b1-ref8k-launch"]["line"],
                    "note": "维持拦截 fresh 之组合,不在主推导内(L2-B2)"}
    return {
        "principle": "资格闸参照系 = 同治下(H)健康参照集之跨池死亡上包络 + 边距 M=2(注册常数);"
                     "M29 治下档降出主推导入敏感度(D3-2 两语境分列禁混引)",
        "scope": L2_SCOPE,
        "P_H": p_h,
        "P_M29_sensitivity": p_m29,
        "envelope": env,
        "envelope_support": support,
        "M": m_margin,
        "line": line,
        "regime_filter_note": "过滤后 max 不变(仍 %d),治下过滤在本数据上不改线值,"
                              "改的是支撑档身份——r2-science/K2 出,K1 独撑【复核】" % env_mixed,
        "loo": loo,
        "single_archive_dependence": single_dep,
        "single_archive_label": ("单档依赖线(对 %s)——标签强制随 D7 引用进入一切后案"
                                 % ",".join(single_dep)) if single_dep else None,
        "M_sensitivity": sens,
        "nature": "复核",
        "unlooseable_note": "过松面不可查声明(统计席 M-3(b)):H×7000 资格语境下不存在任何"
                            "高死亡损伤候选档——6→8 之放宽在过松面零损伤侧样本可检;"
                            "此面不是『查了没查出』,是『不可查』",
        "univariate_note": "深度-死亡耦合闸明文不采(唯一支撑样本即 fresh 之形制视同定制,D4-4);"
                           "俟健康深潜样本 N≥3 另案",
    }


def l2_strat_table(rows: Dict[str, dict]) -> dict:
    """STRAT_TABLE:池 × 治下 分层全表(D3-3;参照档主行 + 候选注记)。"""
    ref_tags = set(P_H_TAGS) | set(P_M29_TAGS)
    cells: Dict[str, dict] = {}
    for tag in sorted(rows):
        r = rows[tag]
        if r.get("pool") is None or r.get("died") is None:
            continue
        key = "%s×%d" % (r["regime"], r["pool"])
        cell = cells.setdefault(key, {"ref_died": {}, "candidate_died": {}})
        if tag in ref_tags:
            cell["ref_died"][tag] = r["died"]
        elif r["row_class"] in ("pinned_candidate", "candidate_m29_context") and r.get("world") == "v3":
            cell["candidate_died"][tag] = r["died"]
    for key, cell in cells.items():
        cell["ref_max"] = max(cell["ref_died"].values()) if cell["ref_died"] else None
    return {"cells": cells,
            "note": "参照主行【复核】:H×7000 max 0/候选 P8 1;H×8000: 6;H×9000: 4;"
                    "M29×7000: 3;M29×8000: 6;M29×9000: 6"}


# ---------------------------------------------------------------------------
# L3 复现地板
# ---------------------------------------------------------------------------

def l3_floor(ratios_damaged: Dict[str, float], ratios_healthy: Dict[str, float],
             delta: float = DELTA_FLOOR) -> dict:
    lo = max(ratios_damaged.values()) + delta
    hi = min(ratios_healthy.values()) - delta
    feasible = lo <= hi
    cur = _f(FLOOR_CANON)
    in_region = feasible and (lo <= cur <= hi)
    if not feasible:
        outcome = "不可定线(可行域空)——现行维持 + 呈报(合法结局)"
        floor_out = ["fraction", 85, 92]
    elif in_region:
        outcome = "维持现行 85/92(不变更条件触发:重标之义系『线错了就改』,不是『能松就松』)"
        floor_out = ["fraction", 85, 92]
    else:
        mid = (lo + hi) / 2.0
        outcome = "现行出域 → floor* := 可行域中点(其自身即正典,无舍入义)"
        floor_out = ["midpoint", mid]
    return {
        "principle": "双约束可行域:(i) floor ≥ 损伤簇比值 max + δ;(ii) floor ≤ 健康簇比值 min − δ;"
                     "δ=0.02;不变更条件先判",
        "ratios_damaged": {k: _f(v) for k, v in sorted(ratios_damaged.items())},
        "ratios_healthy": {k: _f(v) for k, v in sorted(ratios_healthy.items())},
        "delta": delta,
        "feasible_lo": _f(lo),
        "feasible_hi": _f(hi),
        "feasible": feasible,
        "current_value": cur,
        "current_in_region": in_region,
        "outcome": outcome,
        "floor": floor_out,
        "nature": "复核",
        "verdict_wording_duty": "结论限『地板非病灶』级,病灶归属留判决附录,禁预写他线结论(合规席 m-5)",
    }


# ---------------------------------------------------------------------------
# D2-U 簇指派(纯数值泛函)
# ---------------------------------------------------------------------------

def d2u_assign(dbar_by_tag: Dict[str, float]) -> dict:
    clusters = {"damaged": {}, "healthy": {}, "unassigned": {}}
    for tag in sorted(dbar_by_tag):
        d = dbar_by_tag[tag]
        if d < CLUSTER_LO:
            clusters["damaged"][tag] = _f(d)
        elif d > CLUSTER_HI:
            clusters["healthy"][tag] = _f(d)
        else:
            clusters["unassigned"][tag] = _f(d)
    return {
        "rule": "d̄ < −15 → 损伤簇;d̄ > −10 → 健康簇;d̄ ∈ [−15,−10] → 不入簇,如实列名"
                "(纯数值泛函,散文谓词废除)",
        "thresholds": [CLUSTER_LO, CLUSTER_HI],
        "clusters": clusters,
        "constant_note": "−15/−10 系对已知样本间隙(−20.7 与 −5.37 之间)之安放,"
                         "当前论域内 (−20,−6) 内任取同果;外推未测(检察官席 m-2,随 CLASS_ASSIGN 落账)",
        "middle_band_empty": not clusters["unassigned"],
    }


# ---------------------------------------------------------------------------
# OC 表(含联合 α、max-of-2、合成备择功效;L2 之 M 行)
# ---------------------------------------------------------------------------

def oc_tables(deltas_by_family: Dict[str, Dict[str, List[float]]],
              F: np.ndarray, m_star_by_family: Dict[str, float]) -> dict:
    out: Dict[str, Any] = {"grid": {"k": K_GRID, "m": M_GRID},
                           "alpha_semantics": "per-candidate、单肢、边际(注册);联合与选择效应"
                                              "由本表列报受控:联合 α 列 + max-of-2 修正列 + 跨案开奖族错误注记",
                           "families": {}}
    for fam in sorted(deltas_by_family):
        for tag in sorted(deltas_by_family[fam]):
            d = np.asarray(deltas_by_family[fam][tag], dtype=np.float64)
            flipped = F * d
            means0 = flipped.mean(axis=1)
            wins0 = (flipped > 0).sum(axis=1)
            n_eff = int(d.size - (d == 0).sum())  # 平局计法:Δ=0 恒不计赢
            shift_stats = {s: (((flipped + s) > 0).sum(axis=1), means0 + s)
                           for s in POWER_SHIFTS}
            fam_out = {"tag": tag, "n_eff_binom": n_eff, "rows": []}
            for k in K_GRID:
                a_w_mc = _f((wins0 >= k).mean())
                a_w_exact = _f(binom_tail_exact(n_eff, k))
                for m in M_GRID:
                    a_m = _f((means0 >= m).mean())
                    a_joint = _f(((wins0 >= k) & (means0 >= m)).mean())
                    row = {
                        "k": k, "m": m,
                        "alpha_width_mc": a_w_mc,
                        "alpha_width_exact_binom_neff": a_w_exact,
                        "alpha_mean": a_m,
                        "alpha_joint": a_joint,
                        "alpha_maxof2": _f(1.0 - (1.0 - a_joint) ** 2),
                        "power": {},
                    }
                    for s in POWER_SHIFTS:
                        wins_s, means_s = shift_stats[s]
                        row["power"]["+%g" % s] = _f(((wins_s >= k) & (means_s >= m)).mean())
                    fam_out["rows"].append(row)
            out["families"][fam] = fam_out
    # 新旧线操作特性摘要行(D3-2 三列并列)
    summary = {}
    for fam in sorted(deltas_by_family):
        tag = sorted(deltas_by_family[fam])[0]
        d = np.asarray(deltas_by_family[fam][tag], dtype=np.float64)
        flipped = F * d
        means0 = flipped.mean(axis=1)
        wins0 = (flipped > 0).sum(axis=1)
        m_star = m_star_by_family[fam]
        def cell(k, m):
            j = _f(((wins0 >= k) & (means0 >= m)).mean())
            return {"alpha_width": _f((wins0 >= k).mean()),
                    "alpha_mean": _f((means0 >= m).mean()),
                    "alpha_joint": j,
                    "alpha_maxof2": _f(1.0 - (1.0 - j) ** 2)}
        summary[fam] = {
            "old(k=18,m=+4)": cell(18, 4.0),
            "new(k=22,m=m*)": cell(22, m_star),
        }
    out["old_vs_new"] = summary
    out["cross_family_max_row"] = {"note": "跨族取 max 之旧形制降为并列行(记不裁)",
                                   "m": _f(max(m_star_by_family.values()))}
    out["cross_case_family_note"] = ("跨案开奖族错误注记:开奖史四次(11→16→17→v30),"
                                    "逐案边际 α 之族错误未由单常数控制,由本表列报受控")
    return out


# ---------------------------------------------------------------------------
# D5 全史对照表 + LOO
# ---------------------------------------------------------------------------

D5_BANNER = ("本表系反事实报告。历届判词、名分、锚值、金池史全部维持原状;本表不构成任何"
             "复议、改判或补发依据。复审素材 = 测量素材,非复议程序,不构成 cb1e0b9f 之任何再裁。")


def _eval_row_lips(row: dict, new_lines: dict, died_line_new: int) -> Optional[dict]:
    """单候选行逐肢判(现行 vs 新线)。返回 None 表示不适用逐肢反事实。"""
    if not row.get("lip_eval") or not row.get("paired"):
        return None
    base_tag = row["baseline"]
    ps = row["paired"].get(base_tag)
    if ps is None:
        return None
    died = row["died"]
    dbar = ps["dbar"]
    wins = ps["wins"]
    fam = row.get("family") or "worker-leg"
    m_new = new_lines["mean_m_by_family"][fam]
    # 地板:era R = 基线档 agg mean;v30 并列地板参照
    floor_cur = None
    floor_new = None
    floor_notes = []
    r_era = new_lines["baseline_means"].get(base_tag)
    if r_era is not None:
        fl = _f(FLOOR_CANON) * r_era
        floor_cur = row["mean"] >= fl
        floor_new = floor_cur  # L3 维持现行(不变更条件)
        floor_notes.append("era R=%s(%.4g);地板线 %.2f" % (base_tag, r_era, fl))
    fb = row["paired"].get(new_lines["floor_baseline_by_case"].get(row["case"], ""), None)
    if fb is not None:
        r2 = new_lines["baseline_means"].get(new_lines["floor_baseline_by_case"][row["case"]])
        if r2:
            fl2 = _f(FLOOR_CANON) * r2
            floor_notes.append("并列地板参照 %.4g → 线 %.2f(判 %s;v30 史用 0.92 舍入=%.1f 注记)" % (
                r2, fl2, "过" if row["mean"] >= fl2 else "不过", 0.92 * r2))
    lips = {
        "qual_died": {"cur": died <= CURRENT_LINES["died_max"], "new": died <= died_line_new,
                      "value": died},
        "launch_width": {"cur": wins >= CURRENT_LINES["width_k"], "new": wins >= new_lines["width_k"],
                         "value": wins},
        "launch_mean": {"cur": dbar >= CURRENT_LINES["mean_m"], "new": dbar >= m_new,
                        "value": _f(dbar), "family": fam, "m_new": _f(m_new)},
        "launch_died": {"cur": died <= CURRENT_LINES["died_max"], "new": died <= died_line_new,
                        "value": died, "note": "L2-0 同值替换 (b)"},
        "floor": {"cur": floor_cur, "new": floor_new, "value": row["mean"],
                  "notes": floor_notes},
        "sentinel": {"cur": None, "new": None,
                     "note": "哨兵肢零触碰:取台账在册值不重判;资格未过者未及检"},
    }
    launch_numeric_cur = lips["qual_died"]["cur"] and lips["launch_width"]["cur"] and \
        lips["launch_mean"]["cur"] and lips["launch_died"]["cur"]
    launch_numeric_new = lips["qual_died"]["new"] and lips["launch_width"]["new"] and \
        lips["launch_mean"]["new"] and lips["launch_died"]["new"]
    return {
        "lips": lips,
        "launch_numeric_cur": launch_numeric_cur,
        "launch_numeric_new": launch_numeric_new,
        "qual_flip": lips["qual_died"]["cur"] != lips["qual_died"]["new"],
        "launch_numeric_flip": launch_numeric_cur != launch_numeric_new,
    }


def d5_table(extraction: dict, new_lines: dict) -> dict:
    rows = extraction["rows"]
    died_line_new = new_lines["died_max"]
    table = []
    for tag in sorted(rows):
        row = rows[tag]
        if row["row_class"] in ("baseline", "gold_anchor"):
            continue
        ev = _eval_row_lips(row, new_lines, died_line_new)
        entry = {
            "tag": tag,
            "case": row["case"],
            "sha16": row["sha16"],
            "world": row["world"],
            "pool": row["pool"],
            "regime": row["regime"],
            "row_class": row["row_class"],
            "mean": row["mean"],
            "died": row["died"],
            "notes": list(row["notes"]),
            "ledger_qual": row.get("ledger_qual"),
        }
        if ev is None:
            entry["lips"] = None
            entry["flip"] = None
        else:
            entry.update(ev)
            entry["flip"] = ev["qual_flip"] or ev["launch_numeric_flip"]
            if row["world"] == "v2":
                entry["notes"].append("跨世界注记强制:v2 世界档,新线对 v2 型分布形态外推未测(残余⑨)")
            if ev["lips"]["launch_width"]["value"] == new_lines["width_k"]:
                entry["notes"].append("临线注记:wins 恰在新宽度线上")
            if row["died"] == died_line_new:
                entry["notes"].append("临线注记:died 恰在新死亡线上")
        table.append(entry)

    flips_qual = sorted(e["tag"] for e in table if e.get("qual_flip"))
    flips_launch = sorted(e["tag"] for e in table if e.get("launch_numeric_flip"))
    flips_any = sorted(e["tag"] for e in table if e.get("flip"))
    # 最小注册行集之翻转(RG1.5 点值口径候选)
    minimal_rows = {"v31-mfresh-full32", "v29-mexplore-full32"}
    flips_minimal = sorted(t for t in flips_any if t in minimal_rows)
    return {
        "banner": D5_BANNER,
        "scope": L2_SCOPE,
        "draw_history": extraction["draw_history"],
        "rows": table,
        "flip_sets": {
            "qual_flips": flips_qual,
            "launch_numeric_flips": flips_launch,
            "any_flip": flips_any,
            "minimal_registered_rows_flips": flips_minimal,
            "n_any": len(flips_any),
            "n_qual": len(flips_qual),
        },
        "d4_triggers": {
            "flip_count_out_of_band(≥4)": len(flips_any) >= 4,
            "flip_set_single_candidate": len(flips_any) == 1,
            "note": "任一触发 → 面板专项审查该线是否隐性定制(D4-3);"
                    "凡翻转集非空,LINE_DERIVED 强制携 D5-LOO",
        },
    }


def d5_loo(extraction: dict, new_lines: dict, l2: dict) -> dict:
    """D5-LOO:逐档剔除 L2 定线输入后重算翻转集;有变 → 单档依赖线。"""
    base_flips = None
    reports = {}
    for drop in [None] + P_H_TAGS:
        if drop is None:
            line = l2["line"]
        else:
            line = l2["loo"][drop]["line"]
        nl = dict(new_lines)
        nl["died_max"] = line
        t = d5_table(extraction, nl)
        fl = t["flip_sets"]["any_flip"]
        if drop is None:
            base_flips = fl
        else:
            reports[drop] = {"line": line, "flip_set": fl,
                             "flip_set_changed": fl != base_flips}
    changed = sorted(k for k, v in reports.items() if v["flip_set_changed"])
    return {
        "base_flip_set": base_flips,
        "per_drop": reports,
        "single_archive_dependent_on": changed,
        "verdict": ("单档依赖线(对 %s);标签随 D7 引用,后案面板按升格审查处理"
                    % ",".join(changed)) if changed else "无单档依赖",
    }


# ---------------------------------------------------------------------------
# D3 双向检验矩阵 / D6 would-trip 描述性
# ---------------------------------------------------------------------------

def bidir_report(d5: dict, oc: dict, l2: dict, extraction: dict) -> dict:
    rows = {e["tag"]: e for e in d5["rows"]}
    healthy = ["p8-full32", "v31-mfresh-full32"]
    damaged = ["v31-mcont-full32", "v32-sov-full32", "v32-ctrl-full32"]
    strict = {}
    for tag in healthy:
        e = rows.get(tag) or {}
        lips = e.get("lips") or {}
        strict[tag] = {lip: {"blocked_cur": (v.get("cur") is False),
                             "blocked_new": (v.get("new") is False)}
                       for lip, v in lips.items() if isinstance(v, dict) and "cur" in v}
    loose = {}
    for tag in damaged:
        e = rows.get(tag) or {}
        lips = e.get("lips") or {}
        loose[tag] = {lip: {"passed_cur": (v.get("cur") is True),
                            "passed_new": (v.get("new") is True)}
                      for lip, v in lips.items() if isinstance(v, dict) and "cur" in v}
    # 语境分列(D3-2 禁混引):v32 双腿 H vs M29 died
    ctx_split = {}
    for tag in ("v32-sov", "v32-ctrl"):
        h = extraction["rows"].get(tag + "-full32") or {}
        m = extraction["rows"].get(tag + "-m29") or {}
        ctx_split[tag] = {"H_full32_died": h.get("died"), "M29_died": m.get("died"),
                          "note": "死亡闸对 F2 型劣化在资格语境(H full32)本就不咬合,如实入表"}
    matrix = {
        "L1a×过严": "健康向量 wins:P8 16、fresh 15,均 < k*=22——健康候选在宽度肢反事实被拦"
                    "(健康复现 N=2,如实报,禁聚合掩埋)",
        "L1a×过松": "开奖史 11→16→17 从未触旧线 18;新线 22 之下开奖史全不过——无损伤放行样本",
        "L1b×过严": "健康向量 d̄:P8 −5.37、fresh +2.39,均 < 各族 m*——均差肢反事实拦截健康候选;"
                    "功效曲线见 OC(+5/+10/+15 合成备择)",
        "L1b×过松": "现行 +4 远低于两族线:均差肢系四线中最松一肢(方向结论候选,判决附录落实)",
        "L2×过严": "严格同语境(H×7000)读法:健康参照 died=0、健康候选 P8 died=1——"
                   "『系统性偏紧』判词仅在跨池上包络读法下获支持;两种读法并列,禁只引其一(L2-B2)",
        "L2×过松": l2["unlooseable_note"],
        "L3×过严": "健康簇比值 min 0.9531 − δ = 0.9331 ≥ 现行 0.9239——健康侧不被地板拦",
        "L3×过松": "损伤簇比值 max 0.8168 + δ = 0.8368 ≤ 现行 0.9239——损伤侧全数被地板拦;地板非病灶",
    }
    return {
        "strict_side_healthy_blocked": strict,
        "loose_side_damaged_passed": loose,
        "context_split": ctx_split,
        "lip_direction_matrix": matrix,
        "symmetry_note": "两面同一批档案、同一提取脚本;判词按肢×方向矩阵逐格,禁单方向统称『线已修好』",
    }


def wouldtrip_ruling(extraction: dict) -> dict:
    inst = extraction["instruments"]
    da = inst["dry_anchor_seq"]
    incs = [e["increment_pp"] for e in da]
    ratios = sorted(e["grad_ratio"] for e in inst["calib_would_trip"])
    med = ratios[len(ratios) // 2] if len(ratios) % 2 == 1 else \
        (ratios[len(ratios) // 2 - 1] + ratios[len(ratios) // 2]) / 2.0
    dmg_ratios = {leg: [r["grad_ratio"] for r in inst.get(leg + "/calib", [])]
                  for leg in ("v32-sov", "v32-ctrl")}
    return {
        "structural_outcome": "不升格系结构性结局(正文,非残余):升格判据样本底线要求健康腿 ≥2;"
                              "带 would-trip 仪表之健康腿全史仅 P8 一条(fresh 系经理臂,无仪表),"
                              "本案零发不可增样——非裁量、非预测,不入 R 线",
        "dry_anchor": {
            "healthy_seq_pp": incs,
            "healthy_final_pp": incs[-1] if incs else None,
            "would_trip_count": sum(1 for e in da if e["would_trip"]),
            "n_points": len(da),
            "damaged_pp_note": "损伤之 v32 双腿仅 +1~2pp(B1 判决在册值;v32 仪表档无同名增量场,"
                               "本案零发不可重测,如实转录)",
            "ruling": "门设计前提『干层漂移 ⇒ 劣化』方向反转:健康 +8.35pp 击穿预测带 [0,4] 两倍,"
                      "若当时挂闸将冤杀健康腿、放行损伤腿,比无门更坏;原带从未具规范义(B1 RB.9 只记不裁),"
                      "本案确认不升格,降格为双参考描述记录(血统注记 0.6305 承继),"
                      "GATE_WOULD_TRIP schema 不变、判读义变更入册",
        },
        "grad_ratio": {
            "healthy_ratios": ratios,
            "healthy_median": med,
            "damaged_ratios_v32_calib": dmg_ratios,
            "ruling": "梯度比损伤 F2 血统 20-90× 与健康腿中位重叠(锚绳失效系配方性而非种子性)——维持记录仪",
        },
        "ce_gate": {"ruling": "ce 门全员饱和(重校目标线非区分器)——维持记录仪"},
        "future_case_prerequisite": "任何后案重提挂闸:先决 = 健康腿 ≥2 且损伤腿 ≥2 之判别力实证 + "
                                    "S_g 判据重算;dry-anchor 反例强制列为必读证据",
    }


# ---------------------------------------------------------------------------
# 总装
# ---------------------------------------------------------------------------

def compute_all(extraction: dict, seed: int = SEED, n_draws: int = N_DRAWS) -> dict:
    """全套定线计算(确定性;W-DET 由驱动器双跑本函数并逐字节比对)。"""
    rows = extraction["rows"]

    # --- D2-U 论域向量(封闭枚举) ---
    deltas = {}
    dbars = {}
    ratios = {}
    for tag in UNIVERSE_TAGS:
        ps = rows[tag]["paired"][UNIVERSE_BASELINE]
        deltas[tag] = ps["deltas"]
        dbars[tag] = ps["dbar"]
        ratios[tag] = rows[tag]["mean"] / rows[UNIVERSE_BASELINE]["mean"]

    class_universe = {
        "universe_tags": list(UNIVERSE_TAGS),
        "baseline": UNIVERSE_BASELINE,
        "exclusions": UNIVERSE_EXCLUSIONS,
        "d4_2_note": "封闭输入枚举 ≠ 定制条件分支:枚举系论域冻结(防驱动器选择自由度),"
                     "附逐档纳入/排除理由;定制条件分支仍禁",
    }
    class_assign = d2u_assign(dbars)

    # --- 置换零分布 ---
    F = flip_matrix(n_draws, N_SEEDS, seed)
    null_dist = {"seed": seed, "n_draws": n_draws,
                 "convention": "stdlib random.Random(seed) 行主序;r.random()<0.5→+1 否则 −1;"
                               "两族共用同一符号矩阵;Q95=np.quantile 线性插值",
                 "tie_rule": "Δ=0 恒贡献 0(翻转不变量),不计赢,如实计入"}

    fam_vecs = {fam: {t: deltas[t] for t in tags} for fam, tags in FAMILY_VECTORS.items()}
    l1a = l1a_width_line()
    l1b = l1b_family_lines(fam_vecs, F, seed)

    died_by_tag = {t: rows[t]["died"] for t in P_H_TAGS + P_M29_TAGS}
    l2 = l2_death_line(died_by_tag)
    strat = l2_strat_table(rows)

    healthy_r = {t: ratios[t] for t in class_assign["clusters"]["healthy"]}
    damaged_r = {t: ratios[t] for t in class_assign["clusters"]["damaged"]}
    l3 = l3_floor(damaged_r, healthy_r)

    m_star_by_family = {fam: v["m_star_2dp"] for fam, v in l1b["families"].items()}
    new_lines = {
        "width_k": l1a["k_star"],
        "mean_m_by_family": m_star_by_family,
        "died_max": l2["line"],
        "floor": "维持 85/92" if l3["current_in_region"] or not l3["feasible"] else l3["floor"],
        "baseline_means": {t: rows[t]["mean"] for t in sorted(rows)
                           if rows[t]["row_class"] == "baseline"},
        "floor_baseline_by_case": {"v30": "v30-GA0W"},
    }

    oc = oc_tables(fam_vecs, F, {f: v["m_star"] for f, v in l1b["families"].items()})
    d5 = d5_table(extraction, new_lines)
    loo5 = d5_loo(extraction, new_lines, l2)
    bidir = bidir_report(d5, oc, l2, extraction)
    d6 = wouldtrip_ruling(extraction)

    return {
        "null_dist": null_dist,
        "class_universe": class_universe,
        "class_assign": class_assign,
        "l1a": l1a,
        "l1b": l1b,
        "l2": l2,
        "strat_table": strat,
        "l3": l3,
        "new_lines": {k: v for k, v in new_lines.items() if k != "baseline_means"},
        "oc": oc,
        "d5": d5,
        "d5_loo": loo5,
        "bidir": bidir,
        "wouldtrip": d6,
    }
