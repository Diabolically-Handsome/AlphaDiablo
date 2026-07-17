#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G1 闸线重标定——D5 对照表行集之机器枚举(PREREG-G1 rev2 · W-SCRIPT 之一)。

职能(照卷 D5/D2-U):
  * 从六本在册台账(v29/v30/v31/v32/reanchor-r2/infra-b1)机器枚举全史 full32 候选档案
    事件(full32 / exam_ok / quals / substitution / winner / launch_check / paired /
    draw_ledger / r29_2 / r31_2 / VERDICT_PATH);
  * 扫描 eval-assembled 目录之全部 n=32 档案(D5 系报告不系定线输入,提取超集合法——
    D2-U「D5 对照表行集」款);
  * 对每档提取:tag / sha256 / sha16 / agg(mean, died, depth)/ 逐种子 (seed, ret, died,
    depth) 向量 / 池(7000/8000/9000)/ 治下(H/M29)/ 行类别 / 世代语境;
  * 同种子配对统计(候选 − era 基线):n / d̄ / wins(Δ>0,平局计法照卷 L1b:Δ=0 不计赢)/
    ties;
  * 18/32 开奖史逐案映射(以台账为准:v29 launch_check、v30 draw_ledger + paired 系钉死
    台账;11→16 之案归属由同种子配对复算对号,超集提取如实注记);
  * v29-mexplore 专行(75abd205,died 8,mean 149.0,彼时作废、fresh 递补加冕史注记,
    v2 世界语境注记强制)与 v31-fresh 专行(cb1e0b9f,首宗复审素材);
  * B1 仪表事件摘录(GATE_WOULD_TRIP dry-anchor/calib 序列、MS_REPORT 计数)供 D6 描述性
    报告;v32 双腿同名仪表档 calib.jsonl 摘录。

本脚本纯只读提取,零训练、零评测、零环境改动;不 import 任何训练/评测入口。
输出 JSON(--out;缺省打印摘要)。字典键序、行序全部排序钉死(W-DET 前置)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 路径与论域常量(D2-U;路径相对仓库根,由驱动器 W-PIN 以 sha 钉死)
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join("train", "runs", "eval-assembled")

#: 六本在册台账(W-PIN「台账」款)
LEDGERS = {
    "v29": os.path.join("train", "runs", "v29", "gate_ledger.jsonl"),
    "v30": os.path.join("train", "runs", "v30", "gate_ledger.jsonl"),
    "v31": os.path.join("train", "runs", "v31", "gate_ledger.jsonl"),
    "v32": os.path.join("train", "runs", "v32", "gate_ledger.jsonl"),
    "reanchor-r2": os.path.join("train", "runs", "reanchor-r2", "gate_ledger.jsonl"),
    "infra-b1": os.path.join("train", "runs", "infra-b1", "gate_ledger.jsonl"),
}

#: 仪表档(W-PIN「仪表档」款;哨兵档在零触碰清单,pin 不重判)
INSTRUMENT_FILES = {
    "b1-p8/calib": os.path.join("train", "runs", "b1-p8", "calib.jsonl"),
    "b1-p8/sentinel": os.path.join("train", "runs", "b1-p8", "sentinel.jsonl"),
    "v32-sov/calib": os.path.join("train", "runs", "v32-sov", "calib.jsonl"),
    "v32-sov/sentinel": os.path.join("train", "runs", "v32-sov", "sentinel.jsonl"),
    "v32-ctrl/calib": os.path.join("train", "runs", "v32-ctrl", "calib.jsonl"),
    "v32-ctrl/sentinel": os.path.join("train", "runs", "v32-ctrl", "sentinel.jsonl"),
}

#: era 基线映射(机器规则,逐案注记归属依据;lip_eval 限有台账/复算锚之案)
#:   支撑注记:v29 launch_check、v30 draw_ledger/paired、v31 refs、v32 REF_BITEQ 系
#:   钉死台账;v25-v28 之基线 v25-GA0(92.0)由 v30 draw_ledger 序列数字 11/16 与
#:   同种子配对复算对号(超集提取,如实注记)。
BASELINE_MAP: Dict[str, Dict[str, Any]] = {
    "v25": {"baseline": "v25-GA0", "basis": "超集:v25 台账外;draw 序列外注记", "lip_eval": True},
    "v26": {"baseline": "v25-GA0", "basis": "超集:v30 draw_ledger 序列#1=11 复算对号", "lip_eval": True},
    "v27": {"baseline": "v25-GA0", "basis": "超集:v27 台账外;draw 序列外注记", "lip_eval": True},
    "v28": {"baseline": "v25-GA0", "basis": "超集:v30 draw_ledger 序列#2=16 复算对号", "lip_eval": True},
    "v29": {"baseline": "v29-GA0", "basis": "台账:v29 launch_check paired_mean=27.86 复算命中", "lip_eval": True},
    "v30": {"baseline": "v29-GA0", "basis": "台账:v30 paired vs112 复算命中;地板参照 v30-GA0W 并列", "lip_eval": True,
            "floor_baseline": "v30-GA0W"},
    "v31": {"baseline": "v31-ref-launch", "basis": "台账:v31 refs 事件;R=113.0", "lip_eval": True},
    "v32": {"baseline": "v31-ref-launch", "basis": "台账:v32 REF_BITEQ(v32-ref-launch 位级同一,取 v31 原档唯一真源)", "lip_eval": True},
    "p8": {"baseline": "v31-ref-launch", "basis": "台账:infra-b1 定标数据;PRIORS 钉死", "lip_eval": True},
}

#: 基线/参照/金池角色 tag(不作候选行逐肢反事实)
BASELINE_TAGS = {
    "v23-golden", "v24-golden", "v25-GA0", "v29-GA0", "v30-GA0W",
    "v31-ref-launch", "v31-ref-science", "v32-ref-launch", "v32-ref-science",
    "b1-ref8k-launch", "b1-ref8k-science", "b1-varprobe-launch",
}
GOLD_ANCHOR_TAGS = {"r2-science", "r2-launch", "r2-throne", "r2-script", "r2-bcworker"}

#: 族指派(机器规则,按案型;L1b 分族制之 D5 反事实引用;判读注记:后案按候选族引用)
FAMILY_BY_CASE = {
    "v24": "worker-leg", "v26": "worker-leg", "v27": "worker-leg", "v28": "worker-leg",
    "v30": "worker-leg", "p8": "worker-leg",
    "v25": "manager-arm", "v29": "manager-arm", "v31": "manager-arm", "v32": "manager-arm",
}

#: 世代语境(v3 时代 := 以 v31-ref-launch 为分母/基线之档;其余系 v2 世界,跨世界注记强制)
V3_CASES = {"v31", "v32", "p8"}

#: 专行注记(D5 rev2)
SPECIAL_ROW_NOTES = {
    "v31-mfresh-full32": [
        "专行:首宗复审素材(cb1e0b9f);复审素材 = 测量素材,非复议程序,不构成任何再裁",
        "强制注记:资格判翻转、发射结局不翻转(均差 +2.39 < 两族均差线,赢 15/32 < 宽度线)",
    ],
    "v29-mexplore-full32": [
        "专行:v29-mexplore(75abd205,died 8,mean 149.0);彼时 died 8 作废、fresh 递补加冕史注记",
        "v2 世界语境注记强制(基线 v29-GA0=112.4,分母异世界,比值跨世界无意义)",
    ],
    "v30-king-full32": ["v30 未及发射检之腿(资格闸拦截,未及发射语境如实标注)"],
    "v30-bc-full32": ["v30 未及发射检之腿(递补臂,地板未过,发射流程终止)"],
}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_ledger(path: str) -> List[dict]:
    events = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # 台账坏行如实登记,不静默
                events.append({"event": "__UNPARSED__", "raw": line[:200]})
    return events


def load_archive(repo_root: str, tag: str) -> dict:
    """读取 eval-assembled 档案;逐行 seed 缺失时(schema 2)以 protocol.seeds 顺序补齐。"""
    path = os.path.join(repo_root, EVAL_DIR, tag + ".json")
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    rows = doc.get("rows") or []
    seeds = [r.get("seed") for r in rows]
    if rows and seeds[0] is None:
        proto_seeds = ((doc.get("meta") or {}).get("protocol") or {}).get("seeds") or []
        if len(proto_seeds) == len(rows):
            seeds = list(proto_seeds)
        else:
            seeds = list(range(len(rows)))  # 无种子档:序号占位,pairable=False
    doc["__seeds__"] = seeds
    doc["__path__"] = path
    return doc


def per_seed_vector(doc: dict) -> Dict[int, dict]:
    out = {}
    for s, r in zip(doc["__seeds__"], doc.get("rows") or []):
        out[int(s)] = {
            "ret": float(r.get("ret", 0.0)),
            "died": bool(r.get("died", False)),
            "depth": int(r.get("depth", 0)),
        }
    return out


def pool_of(seeds: List[int]) -> Optional[int]:
    if not seeds:
        return None
    lo = min(seeds)
    for base in (7000, 8000, 9000):
        if base <= lo < base + 1000:
            return base
    return None


def regime_of(tag: str) -> str:
    """治下判定(机器规则):-m29 后缀或 science 参照 = M29 治下;其余 H 治下。"""
    if tag.endswith("-m29") or "science" in tag:
        return "M29"
    return "H"


def case_of(tag: str) -> Optional[str]:
    if tag.startswith("p8-"):
        return "p8"
    if tag.startswith("b1-"):
        return "p8"  # B1 案之参照档,归 b1/p8 案簇
    if tag.startswith("r2-"):
        return "r2"
    for i in range(23, 33):
        if tag.startswith("v%d-" % i):
            return "v%d" % i
    return None


def paired_stats(cand: Dict[int, dict], base: Dict[int, dict]) -> Optional[dict]:
    """同种子配对:d̄、wins(Δ>0;Δ=0 不计赢,照卷 L1b 平局计法)、ties、逐种子 Δ。"""
    common = sorted(set(cand) & set(base))
    if len(common) < 32:
        return None
    deltas = [round(cand[s]["ret"] - base[s]["ret"], 10) for s in common]
    return {
        "n": len(common),
        "seeds": common,
        "dbar": sum(deltas) / len(deltas),
        "wins": sum(1 for d in deltas if d > 0),
        "ties": sum(1 for d in deltas if d == 0),
        "deltas": deltas,
    }


# ---------------------------------------------------------------------------
# 台账候选枚举
# ---------------------------------------------------------------------------

def enumerate_ledger_candidates(events_by_case: Dict[str, List[dict]]) -> Dict[str, dict]:
    """六本台账之 full32 候选档案机器枚举(候选史档款「台账机器枚举」)。"""
    cands: Dict[str, dict] = {}

    def note(tag, case, ev):
        rec = cands.setdefault(tag, {"tag": tag, "case": case, "ledger_events": []})
        rec["ledger_events"].append({k: ev[k] for k in sorted(ev) if k not in ("mode_seq",)})

    for case in ("v29", "v30", "v31"):
        for ev in events_by_case[case]:
            if ev.get("event") == "full32":
                arm = ev.get("arm", "")
                # 台账 arm 名 → 档案 tag 规则:v29/v31 系 <arm>-full32;v30 系 v30-<arm>-full32
                if case == "v30":
                    tag = "v30-%s-full32" % arm
                else:
                    tag = arm + "-full32" if not arm.endswith("-full32") else arm
                note(tag, case, ev)
    for ev in events_by_case["v32"]:
        if ev.get("event") == "exam_ok" and ev.get("tag", "").endswith("-full32"):
            note(ev["tag"], "v32", ev)
    for ev in events_by_case["infra-b1"]:
        # 仅 H 治下 full32(-m29 语境行另类,D3-2 两语境分列禁混引)
        if ev.get("event") == "exam_ok" and ev.get("tag", "").endswith("-full32"):
            note(ev["tag"], "p8", ev)
    return cands


def enumerate_context(events_by_case: Dict[str, List[dict]]) -> Dict[str, Any]:
    """资格/递补/胜者/发射检/开奖 台账语境事件(逐案映射以台账为准)。"""
    ctx: Dict[str, Any] = {}
    keep = {"quals", "substitution", "winner", "launch_check", "paired",
            "draw_ledger", "r29_2", "r31_2", "R32_SPLIT", "R32_MAIN", "VERDICT_PATH"}
    for case in ("v29", "v30", "v31", "v32"):
        ctx[case] = [e for e in events_by_case[case] if e.get("event") in keep]
    return ctx


def build_draw_history(ctx: Dict[str, Any], rows_by_tag: Dict[str, dict]) -> List[dict]:
    """18/32 开奖史逐案映射。

    钉死台账:v30 draw_ledger「第 4 次挑战者开奖(11→16→17→本案)」;v29 launch_check
    paired_wins=17;v30 paired vs112_wins=13 / vs140_wins=11。
    11/16 之案归属:由超集档案同种子配对复算对号(v26-G3-leg6 → 11,v28-G3-leg1 → 16),
    归属依据入注记(v26/v28 台账不在六本钉死台账内,系超集提取)。
    """
    def wins_of(tag):
        r = rows_by_tag.get(tag) or {}
        p = (r.get("paired") or {}).get(r.get("baseline") or "", {})
        return p.get("wins")

    hist = [
        {"draw_no": 1, "case": "v26", "tag": "v26-G3-leg6", "wins": wins_of("v26-G3-leg6"),
         "ledger_wins": 11, "basis": "v30 draw_ledger 序列#1;同种子配对复算对号(超集提取)"},
        {"draw_no": 2, "case": "v28", "tag": "v28-G3-leg1", "wins": wins_of("v28-G3-leg1"),
         "ledger_wins": 16, "basis": "v30 draw_ledger 序列#2;同种子配对复算对号(超集提取)"},
        {"draw_no": 3, "case": "v29", "tag": "v29-mfresh-full32", "wins": wins_of("v29-mfresh-full32"),
         "ledger_wins": 17, "basis": "v29 台账 launch_check(钉死)"},
        {"draw_no": 4, "case": "v30", "tag": "v30-bc-full32", "wins": wins_of("v30-bc-full32"),
         "ledger_wins": 13, "basis": "v30 台账 draw_ledger + paired vs112(钉死;胜者系递补臂 bc;"
                                     "均值胜者 king 资格闸拦截未及发射检)"},
    ]
    for h in hist:
        h["reconciled"] = (h["wins"] == h["ledger_wins"])
    return hist


# ---------------------------------------------------------------------------
# B1/v32 仪表摘录(D6 描述性输入)
# ---------------------------------------------------------------------------

def extract_instruments(repo_root: str, events_by_case: Dict[str, List[dict]]) -> dict:
    b1 = events_by_case["infra-b1"]
    dry_anchor = [e for e in b1 if e.get("event") == "GATE_WOULD_TRIP" and e.get("gate") == "dry-anchor"]
    calib_wt = [e for e in b1 if e.get("event") == "GATE_WOULD_TRIP" and e.get("gate") == "calib"]
    inst = {
        "dry_anchor_seq": [
            {"step": e["step"], "mismatch": e["mismatch"], "increment_pp": e["increment_pp"],
             "would_trip": e["would_trip"]} for e in dry_anchor
        ],
        "calib_would_trip": [
            {"step": e["step"], "grad_ratio": e["grad_ratio"], "distill_ce": e["distill_ce"],
             "would_trip_ratio": e["would_trip_ratio"], "would_trip_ce": e["would_trip_ce"]}
            for e in calib_wt
        ],
        "n_obs_drift": sum(1 for e in b1 if e.get("event") == "OBS_DRIFT"),
        "n_canary_eval": sum(1 for e in b1 if e.get("event") == "CANARY_EVAL"),
        "n_gate_would_trip": sum(1 for e in b1 if e.get("event") == "GATE_WOULD_TRIP"),
    }
    # v32 双腿同名仪表档(损伤侧梯度比;dry-anchor 增量场无同名字段,+1~2pp 系 B1 判决在册值)
    for leg in ("v32-sov", "v32-ctrl"):
        rows = []
        with open(os.path.join(repo_root, "train", "runs", leg, "calib.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        inst[leg + "/calib"] = [
            {"step": r["step"], "grad_ratio": round(r["g_pg"] / r["g_ce"], 2),
             "distill_ce": r["distill_ce"]} for r in rows
        ]
    return inst


# ---------------------------------------------------------------------------
# 主提取
# ---------------------------------------------------------------------------

def extract(repo_root: str = REPO_ROOT) -> dict:
    eval_dir = os.path.join(repo_root, EVAL_DIR)

    events_by_case = {}
    ledger_shas = {}
    for case, rel in sorted(LEDGERS.items()):
        path = os.path.join(repo_root, rel)
        events_by_case[case] = read_ledger(path)
        ledger_shas[case] = sha256_file(path)

    ledger_cands = enumerate_ledger_candidates(events_by_case)
    ctx = enumerate_context(events_by_case)

    # eval-assembled 全目录扫描:n=32 档案全收(超集合法,D5 报告面)
    all_files = sorted(f for f in os.listdir(eval_dir) if f.endswith(".json"))
    docs: Dict[str, dict] = {}
    excluded: List[dict] = []
    for fname in all_files:
        tag = fname[:-5]
        doc = load_archive(repo_root, tag)
        n = (doc.get("agg") or {}).get("n")
        if n != 32:
            excluded.append({"tag": tag, "reason": "n=%s ≠ 32(full32 行集外)" % n})
            continue
        docs[tag] = doc

    vectors = {tag: per_seed_vector(doc) for tag, doc in docs.items()}
    sha_by_tag = {tag: sha256_file(doc["__path__"]) for tag, doc in docs.items()}

    # 重复档案组(位级同一之基线别名,如 v29-GA0 == v28-G3-leg1)
    by_sha: Dict[str, List[str]] = {}
    for tag, sha in sorted(sha_by_tag.items()):
        by_sha.setdefault(sha, []).append(tag)
    dup_of = {}
    for sha, tags in by_sha.items():
        if len(tags) > 1:
            for t in tags[1:]:
                dup_of[t] = tags[0]
            dup_of[tags[0]] = None if tags[0] not in dup_of else dup_of[tags[0]]

    rows: Dict[str, dict] = {}
    for tag in sorted(docs):
        doc = docs[tag]
        agg = doc.get("agg") or {}
        seeds = sorted(vectors[tag])
        case = case_of(tag)
        cls = "unmapped"
        if tag in GOLD_ANCHOR_TAGS:
            cls = "gold_anchor"
        elif tag in BASELINE_TAGS:
            cls = "baseline"
        elif tag.endswith("-m29"):
            cls = "candidate_m29_context"
        elif tag in ledger_cands:
            cls = "pinned_candidate"
        elif case in BASELINE_MAP:
            cls = "superset_challenger"
        else:
            cls = "ancient_diagnostic"

        row = {
            "tag": tag,
            "case": case,
            "sha256": sha_by_tag[tag],
            "sha16": sha_by_tag[tag][:16],
            "n": agg.get("n"),
            "mean": agg.get("ret_mean"),
            "died": agg.get("died"),
            "depth_median": agg.get("depth_median"),
            "pool": pool_of(seeds),
            "regime": regime_of(tag),
            "world": ("v3" if case in V3_CASES else ("r2" if case == "r2" else "v2")),
            "row_class": cls,
            "family": FAMILY_BY_CASE.get(case),
            "dup_of": dup_of.get(tag),
            "notes": list(SPECIAL_ROW_NOTES.get(tag, [])),
            "paired": {},
            "baseline": None,
            "baseline_basis": None,
            "lip_eval": False,
        }

        # 配对(候选类行 + 有注册基线映射之案;M29 语境行照 D3-2 分列禁混引:只配 science 侧不入定线)
        if cls in ("pinned_candidate", "superset_challenger") and case in BASELINE_MAP:
            bm = BASELINE_MAP[case]
            row["baseline"] = bm["baseline"]
            row["baseline_basis"] = bm["basis"]
            row["lip_eval"] = bool(bm["lip_eval"])
            for btag in [bm["baseline"]] + ([bm.get("floor_baseline")] if bm.get("floor_baseline") else []):
                if btag in vectors:
                    ps = paired_stats(vectors[tag], vectors[btag])
                    if ps is not None:
                        row["paired"][btag] = ps
            if not row["paired"]:
                row["lip_eval"] = False
                row["notes"].append("基线同种子配对不可得,逐肢反事实不适用")
        elif cls == "candidate_m29_context":
            row["notes"].append("M29 治下语境行(D3-2 两语境分列禁混引;不入 H 治下资格/地板反事实)")
        elif cls == "ancient_diagnostic":
            row["notes"].append("远古/诊断档(基线归属无台账锚,仅报 agg,不作逐肢反事实)")

        if tag in ledger_cands:
            row["ledger_events"] = ledger_cands[tag]["ledger_events"]
        rows[tag] = row

    draw_history = build_draw_history(ctx, rows)
    instruments = extract_instruments(repo_root, events_by_case)

    # 台账语境回填:资格/递补/胜者判在案值
    qual_map = {}
    for case in ("v29", "v30", "v31", "v32"):
        for ev in ctx[case]:
            if ev.get("event") == "quals":
                for k, v in ev.items():
                    if isinstance(v, dict) and "qual_ok" in v:
                        qual_map[(case, k)] = v
    for tag, row in rows.items():
        case = row["case"]
        arm_names = {tag, tag.replace("-full32", ""),
                     tag.replace("v30-", "").replace("-full32", "")}
        for (c, arm), v in sorted(qual_map.items()):
            if c == case and arm in arm_names:
                row["ledger_qual"] = v

    return {
        "schema": "g1-extract-v1",
        "repo_root": repo_root,
        "eval_dir": EVAL_DIR,
        "ledger_shas": ledger_shas,
        "instrument_files": {k: sha256_file(os.path.join(repo_root, v))
                             for k, v in sorted(INSTRUMENT_FILES.items())},
        "n_archives_scanned": len(all_files),
        "excluded_non_full32": excluded,
        "rows": rows,
        "draw_history": draw_history,
        "ledger_context": ctx,
        "instruments": instruments,
        "self_note": "D5 行集系机器枚举超集;定线输入论域(D2-U)另由分析库按封闭枚举取行,"
                     "超集仅入报告面(D4-2 区分条款)。",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G1 D5 行集机器枚举(只读提取)")
    ap.add_argument("--out", help="输出 JSON 路径(缺省仅打印摘要)")
    ap.add_argument("--repo-root", default=REPO_ROOT)
    args = ap.parse_args(argv)

    data = extract(args.repo_root)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, sort_keys=True, indent=1)
        print("已写出:%s" % args.out)
    n_by_class: Dict[str, int] = {}
    for r in data["rows"].values():
        n_by_class[r["row_class"]] = n_by_class.get(r["row_class"], 0) + 1
    print("档案 n=32 计 %d;分类:%s" % (len(data["rows"]), json.dumps(n_by_class, ensure_ascii=False, sort_keys=True)))
    for h in data["draw_history"]:
        print("开奖#%d %s wins=%s(台账 %s,对号 %s)" % (
            h["draw_no"], h["tag"], h["wins"], h["ledger_wins"], "✓" if h["reconciled"] else "✗"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
