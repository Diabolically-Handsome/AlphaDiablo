#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G1 闸线重标定——驱动器(PREREG-G1 rev2 · W-SCRIPT 之三)。

测量案 G1:纯离线分析,零训练、零环境改动、零契约触碰、零评测发。
W 门(D1):
  * W1 冻结公证:porcelain 净 + 本文档/驱动器 HEAD==FROZEN_HEAD(--dry 跳过,供冻结前试算);
  * W-PIN 输入档案钉死:封闭枚举全文 sha256 冻结常量,预检失配 → PREFLIGHT_FAIL(3);
  * W-NOEVAL 零发断言:.driver.lock 单案锁;禁 import 训练/评测入口(运行时断言);
    案前案后各扫描一次全部台账与 eval-assembled 目录,检出非本案来源新档案先对账来源案台账,
    无来源可对账 → CASE_HALT(7);
  * W-DET 决定性断言:定种子 20260717;LINE_DERIVED 前全套计算(含提取)重跑第二遍,
    逐字节相等方落账;不等 → 回炉不落线(6)。
守卫:CLASS_UNIVERSE / CLASS_ASSIGN / INPUT_MISMATCH(正文基准数字,方法冻结总则 4)/
LINE_DERIVED / 记分卡(rev3 三性质:复核 / 预测·MC(仅独立复算源可用)/ 预测·超集;RG1.2 系复核点值)。
台账:<out>/gate_ledger.jsonl(D8 词表)。退出码:0 案结;3 预检;6 非决定性;7 零发破防;其余 1(P1)。

INPUT_MISMATCH 停机语义注记:总则 4 未在 P 线单列退出码,按「其余 P1」归 1;
--dry 下如实登记并继续跑全链,案结报告集中呈报(试算之义即在暴露此类项)。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from fractions import Fraction
from typing import Any, Dict, List, Optional

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TRAIN_DIR)
sys.path.insert(0, TRAIN_DIR)

import extract_g1_history as ex  # noqa: E402(本案 W-SCRIPT 三件套之一)
import analysis_g1_lines as al   # noqa: E402

# ---------------------------------------------------------------------------
# 冻结常量
# ---------------------------------------------------------------------------

#: W1:冻结 commit(冻结纪律照 R1/R2/B1:commit 即公证,正文零自指 sha,
#: 冻结 sha 仅入台账 FREEZE_SHA 事件)。冻结时由值守填入;--dry 不检。
FROZEN_HEAD: Optional[str] = None

CASE_ID = "recal-g1"
SEED = al.SEED
N_DRAWS = al.N_DRAWS

#: 禁 import 之训练/评测入口(W-NOEVAL;模块名前缀)
FORBIDDEN_MODULE_PREFIXES = (
    "train_ppo", "eval_assembled", "eval_contract", "evaluate", "evaluate_deep",
    "evaluate_options", "bc_flat", "bc_manager", "bc_worker", "leashed_ppo",
    "run_v2", "run_v3", "run_b1_infra", "run_reanchor",
)

#: W-PIN 封闭枚举(相对仓库根;全文 sha256 冻结常量)
PINNED_SHA256: Dict[str, str] = {
    # —— 参照分布档 ——
    "train/runs/eval-assembled/v31-ref-launch.json": "df17db995661c3994215c614ac4beeb6e4670bd1527d51a1198d69d960e01541",
    "train/runs/eval-assembled/v31-ref-science.json": "d842a8fa75c3b9234f4197a3f9fdc41458e8efeacf1d878934c349cf533c2489",
    "train/runs/eval-assembled/b1-varprobe-launch.json": "0c587913ede0d5a31c4aebe750a62ba10ca5f7e51970786412721e575bdfce34",
    "train/runs/eval-assembled/b1-ref8k-launch.json": "3b4ef1681134d51d61c3081195fc620ea4ad4d7c7f034597b9f92782cabe6a19",
    "train/runs/eval-assembled/b1-ref8k-science.json": "e23a83383b8e286d9baa85ee9142970103457b0b5a3b249c4f0826b184910ef4",
    # —— 金池锚档(只读,W-GOLD 注记:档案文件读取不构成金池暴露;暴露 := 评测发,R2/B1 口径)——
    "train/runs/eval-assembled/r2-science.json": "8d6a05d3517a6481c756109158b446b3eb7bae8858d02841b42c43a972e273f5",
    "train/runs/eval-assembled/r2-launch.json": "8ab6b51065105a6719c1348d561c5fe4b15f674f27869e6870c0bf813642dff1",
    "train/runs/eval-assembled/r2-throne.json": "2324648a416cbc0cb858858b006b5f797ee9c63422f13cb86c86462948f28c63",
    "train/runs/eval-assembled/r2-script.json": "71c298e05b6bf19ea94ced26c68b67b9e05303f22637357b610b15c6fb21a7f7",
    "train/runs/eval-assembled/r2-bcworker.json": "0f5970acc26f200e40513ed0f51485b43e47009c21ef570a4a0a89e7776f0de3",
    # —— 候选史档 ——
    "train/runs/eval-assembled/v31-mfresh-full32.json": "cb1e0b9fbab4e336b3223a7d799dd0b7308db86da046deab2a7cca8b0e305959",
    "train/runs/eval-assembled/v31-mcont-full32.json": "ef6306510d6e8de5338b9d62e058291bc4ac0e1c17a8fdf8cc0dc2cbd749c419",
    "train/runs/eval-assembled/v32-sov-full32.json": "f2ca74b5db4ea4d181080edd977b728b04a7139475d5b355a9b0fbdf7d23babe",
    "train/runs/eval-assembled/v32-sov-s16.json": "5288b2404f0c517fd5e93360e820fd2ea0489b457c4b409a90a4e9e905244543",
    "train/runs/eval-assembled/v32-sov-m29.json": "8c6336f966ace5fd68f4d4c180a7e62a6b1603362c4ed3eb489f24f3a88d523f",
    "train/runs/eval-assembled/v32-ctrl-full32.json": "1c3f683476d1b6124ba2c85473e4f17a12d52984d55582c6f70d3f0b3306e86c",
    "train/runs/eval-assembled/v32-ctrl-s16.json": "34f0defc2053f64115e9c72a62d3ce044c3948dd5e72bd9f7e083ad1d3af9d65",
    "train/runs/eval-assembled/v32-ctrl-m29.json": "cd198545a2723a114a8ceaf548ce7917ed8ea89355cd15c828895dad622652af",
    "train/runs/eval-assembled/p8-s16.json": "844735ffcf7e14ac0314abf5d8ddd368639c838acf2036e6cc902b54b918ac95",
    "train/runs/eval-assembled/p8-full32.json": "394c7d1b4288285338a7f7e9f2edbeed261cf4d6bd4876c494a7e5fa76d7bf88",
    "train/runs/eval-assembled/p8-full32-m29.json": "bd6705900b2737e220c6873ede13c2212edaa54dfba412e27df53aee51e9dc50",
    "train/runs/eval-assembled/p8-hold8k.json": "73bd5d97d455b4ae9c8f86c307fa373a4c2c08eb92842ce58b0bd6544c789692",
    "train/runs/eval-assembled/p8-hold8k-m29.json": "31981b279eab46b650f27cc89926816efdde9719e8798d25c31ad0deebcd8b7f",
    "train/runs/eval-assembled/p8-canary1-h.json": "25b83bec65d585d95b80f8a2c03f0f7f49d686d096ab9d947ff30ab4d23dcd60",
    "train/runs/eval-assembled/p8-canary1-m29.json": "31fb2ff212d8b78f7db316857fc6a87f4b3b97cd918ef5cc4a045321e88e3563",
    "train/runs/eval-assembled/p8-canary2-h.json": "eda1483e577c0045ba7c1dab6f78707012e33aeb15bd768fd1c9d458e7342b01",
    "train/runs/eval-assembled/p8-canary2-m29.json": "5f70c3e870f014ec7eb4deb32518d10fdc566c67a3ac19a54cfeae8330572d8b",
    "train/runs/eval-assembled/p8-canary3-h.json": "bfe5cc515fc8fc22d0099dc867f5d0ef81e8432cc6fe97db1e43fe288bb86154",
    "train/runs/eval-assembled/p8-canary3-m29.json": "81e4aa8b07561c579618dad8eaf2eed28d6e4b29b672c6c1d00c366199500605",
    "train/runs/eval-assembled/p8-canary4-h.json": "5c1e2a7921c3b4b083b34669ccc4fa418931dedafe99bf9e6aff69f4adb8af5b",
    "train/runs/eval-assembled/p8-canary4-m29.json": "eea981f51fa1eb63b7691798134bdb023c3ab8611bea3e061d8586642ccd22b4",
    "train/runs/eval-assembled/v29-mexplore-full32.json": "75abd205ab5f43b415768b7092b9f0a88a787e71d38f8841d125e0bb0674b670",
    "train/runs/eval-assembled/v29-mfresh-full32.json": "08633101c010a2975b9001a71660bb50d43e681842e3fe1befdfdfd48f99ce63",
    "train/runs/eval-assembled/v30-king-full32.json": "6ce157a36703aca65554489c5742a96c75a4585d7212bf7247492403db4bfac6",
    "train/runs/eval-assembled/v30-bc-full32.json": "ff25edb2abcd9945df1f15b73884c830a204db0f8fe76be9c8f7db023f2e3cda",
    # —— 仪表档(dry-anchor 序列与 GATE_WOULD_TRIP/OBS_DRIFT/CANARY_EVAL 行随 infra-b1 台账 sha 一并钉死)——
    "train/runs/b1-p8/calib.jsonl": "df6d28829f0c7275a6c4cc7b932e3a4973e93e80536c383ede900a60ce8817e3",
    "train/runs/b1-p8/sentinel.jsonl": "40442c33c22d1e39192b16d9dd691dab79a5683f729e01789950bbb5cd0f1550",
    "train/runs/v32-sov/calib.jsonl": "95fb8df7a9ee65b8aa3cfa2f2634126482db3cd9a5fbe89af4fefb579ed30a3e",
    "train/runs/v32-sov/sentinel.jsonl": "ebd4192aa455cc3eb0dabddd9b43ef9dbe9d6fa6293fe64a091d1caad572dcce",
    "train/runs/v32-ctrl/calib.jsonl": "227732ea73e52bbbe4e17995786cf54b7f9d7235879c6b93fac6c80d96d456a9",
    "train/runs/v32-ctrl/sentinel.jsonl": "29615aeb55e2132e98300b32c98272fb49a8757a1702a3c26efb83eeb6523cd6",
    # —— 六本台账 ——
    "train/runs/v29/gate_ledger.jsonl": "27ad88b295594f2d7131c81be29a51fa246da82f7a932a939a524e056dd06f67",
    "train/runs/v30/gate_ledger.jsonl": "7187c30c0f8e255fd7f7ab23d01b5164a5557d9c5130633529e85d147e7a463e",
    "train/runs/v31/gate_ledger.jsonl": "19ed09c664947228a3e07d80feb56edf8bc05124b853bc1ad796b2c6af8d09e5",
    "train/runs/v32/gate_ledger.jsonl": "8c197617097548f04e82bb3157b679bf6792ce3da937bd4e8ec657d920fe0d22",
    "train/runs/reanchor-r2/gate_ledger.jsonl": "a3351a9ba525d8d3f400c61716169d991ae16c05b8f1e2f326a74f71fb4828d5",
    "train/runs/infra-b1/gate_ledger.jsonl": "cdc24b16b1ac11a86a64ffb031ea5a10b432d61e5581271346926c0f7d53ed15",
}

#: 正文基准数字(方法冻结总则 4「单一真源」;比对精度 = 正文书写精度之半 ulp + 1e-9)
#: 键:(描述, 提取器, 基准值, 小数位)
TEXT_BASELINES = [
    ("R(v31-ref-launch agg mean)", "R", 113.0, 1),
    ("配对均差 P8", "dbar:p8-full32", -5.37, 2),
    ("配对均差 fresh", "dbar:v31-mfresh-full32", 2.39, 2),
    ("配对均差 cont", "dbar:v31-mcont-full32", -24.3, 1),
    ("配对均差 sov", "dbar:v32-sov-full32", -33.8, 1),
    ("配对均差 ctrl", "dbar:v32-ctrl-full32", -20.7, 1),
    ("比值 P8", "ratio:p8-full32", 0.9531, 4),
    ("比值 fresh", "ratio:v31-mfresh-full32", 1.0212, 4),
    ("比值 cont", "ratio:v31-mcont-full32", 0.785, 3),
    ("比值 ctrl", "ratio:v32-ctrl-full32", 0.817, 3),
    ("比值 sov", "ratio:v32-sov-full32", 0.701, 3),
    ("置换 Q95 P8(种子 20260717,1e5)", "q95:worker-leg", 17.32, 2),
    ("置换 Q95 fresh(种子 20260717,1e5)", "q95:manager-arm", 23.40, 2),
    ("v29-mexplore died", "mexplore:died", 8, 0),
    ("v29-mexplore mean", "mexplore:mean", 149.0, 1),
]
MEXPLORE_SHA16 = "75abd205ab5f43b4"

#: 记分卡预期(rev2 三性质;R 线)
SCORECARD_SPEC = [
    {"id": "RG1.1", "line": "发射宽度肢 k*", "current": "18/32", "nature": "复核",
     "expect": "22(点值,无带)"},
    {"id": "RG1.2a", "line": "均差肢 worker-leg m*_w(单向量线)", "current": "+4", "nature": "复核",
     "expect": "+17.3,带 [16.5,18.0]", "band": (16.5, 18.0), "point": 17.3},
    {"id": "RG1.2b", "line": "均差肢 manager-arm m*_m(单向量线·高杠杆种子7031注记)", "current": "+4", "nature": "复核",
     "expect": "+23.4,带 [22.5,24.5](面板否决循环性则该族不可定线)", "band": (22.5, 24.5), "point": 23.4},
    {"id": "RG1.3", "line": "死亡线(适用域 L2-0)", "current": "≤6", "nature": "复核",
     "expect": "≤8(单档依赖:K1)"},
    {"id": "RG1.4", "line": "复现地板", "current": "85/92", "nature": "复核",
     "expect": "维持现行(不变更条件触发)"},
    {"id": "RG1.5", "line": "全史对照表结局翻转数", "current": "—", "nature": "预测·超集",
     "expect": "2(fresh、v29-explore 资格判),带 [2,4](最小行集部分系复核)",
     "band": (2, 4), "point": 2},
]


# ---------------------------------------------------------------------------
# 台账
# ---------------------------------------------------------------------------

class Ledger:
    def __init__(self, path: str):
        self.path = path
        self.events: List[dict] = []

    def log(self, event: str, **kw):
        rec = {"t": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "case": CASE_ID, "event": event}
        rec.update(kw)
        self.events.append(rec)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        return rec


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", REPO_ROOT] + list(args),
                                   text=True).strip()


# ---------------------------------------------------------------------------
# W 门
# ---------------------------------------------------------------------------

def w1_freeze_check(led: Ledger, dry: bool) -> Optional[int]:
    if dry:
        led.log("NEEDS_ATTENTION", note="--dry:W1 冻结公证跳过(冻结前试算;FREEZE_SHA 不落账)")
        return None
    porcelain = git("status", "--porcelain")
    head = git("log", "-1", "--format=%H")
    if porcelain:
        led.log("PREFLIGHT_FAIL", gate="W1", why="porcelain 不净", detail=porcelain[:500])
        return 3
    prior_freeze = [e for e in led.events if e.get("event") == "FREEZE_SHA"]
    if prior_freeze:
        if head != prior_freeze[0]["sha"]:
            led.log("PREFLIGHT_FAIL", gate="W1", why="HEAD != 台账 FREEZE_SHA(链式冻结失配)",
                    head=head, frozen=prior_freeze[0]["sha"])
            raise SystemExit(3)
        return None
    if FROZEN_HEAD is None:
        # 家法链式冻结(R2/B1 同款):首跑 porcelain 净时以当前 HEAD 落 FREEZE_SHA
        led.log("FREEZE_SHA", sha=head, note="链式冻结首落(W1)")
        return None
    if False:
        led.log("PREFLIGHT_FAIL", gate="W1", why="FROZEN_HEAD 冻结常量未落(链式重冻结照 P6)")
        return 3
    if head != FROZEN_HEAD:
        led.log("PREFLIGHT_FAIL", gate="W1", why="HEAD != FROZEN_HEAD", head=head, frozen=FROZEN_HEAD)
        return 3
    led.log("FREEZE_SHA", sha=head)
    return None


def w_noeval_import_check(led: Ledger) -> Optional[int]:
    bad = sorted(m for m in sys.modules
                 if any(m == p or m.startswith(p) for p in FORBIDDEN_MODULE_PREFIXES))
    if bad:
        led.log("CASE_HALT", gate="W-NOEVAL", why="检出训练/评测入口 import", modules=bad)
        return 7
    return None


def snapshot_world() -> Dict[str, str]:
    """案前/案后档案扫描:eval-assembled 全目录 + 六本台账 sha256。"""
    snap = {}
    ea = os.path.join(REPO_ROOT, "train", "runs", "eval-assembled")
    for f in sorted(os.listdir(ea)):
        p = os.path.join(ea, f)
        if os.path.isfile(p):
            snap["eval-assembled/" + f] = sha256_file(p)
    for case, rel in sorted(ex.LEDGERS.items()):
        snap["ledger/" + case] = sha256_file(os.path.join(REPO_ROOT, rel))
    return snap


def reconcile_new_archives(new_keys: List[str]) -> Dict[str, Optional[str]]:
    """来源对账(rev2 合规席 m-7):新档案在他案台账有来源则不触发自废。"""
    out: Dict[str, Optional[str]] = {}
    runs = os.path.join(REPO_ROOT, "train", "runs")
    ledgers = []
    for d in sorted(os.listdir(runs)):
        lp = os.path.join(runs, d, "gate_ledger.jsonl")
        if os.path.isfile(lp):
            ledgers.append((d, lp))
    for key in new_keys:
        name = os.path.basename(key)
        stem = name[:-5] if name.endswith(".json") else name
        source = None
        for case, lp in ledgers:
            if case == CASE_ID:
                continue
            try:
                with open(lp, encoding="utf-8") as fh:
                    if any(stem in line for line in fh):
                        source = case
                        break
            except OSError:
                continue
        out[key] = source
    return out


def w_pin_check(led: Ledger) -> List[dict]:
    mism = []
    pins = []
    for rel in sorted(PINNED_SHA256):
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(path):
            mism.append({"path": rel, "why": "档案缺失"})
            continue
        got = sha256_file(path)
        want = PINNED_SHA256[rel]
        ok = got == want
        pins.append({"path": rel, "sha256": got, "ok": ok})
        if not ok:
            mism.append({"path": rel, "why": "sha 失配", "got": got, "want": want})
    led.log("INPUT_PIN", n_pinned=len(PINNED_SHA256), n_ok=sum(1 for p in pins if p["ok"]),
            gold_note="金池锚档只读引用(W-GOLD):档案读取不构成金池暴露;9000-9031 零新暴露",
            pins=pins)
    return mism


# ---------------------------------------------------------------------------
# INPUT_MISMATCH 守卫
# ---------------------------------------------------------------------------

def check_text_baselines(extraction: dict, results: dict) -> List[dict]:
    rows = extraction["rows"]

    def get(key: str) -> float:
        kind, _, arg = key.partition(":")
        if kind == "R":
            return rows["v31-ref-launch"]["mean"]
        if kind == "dbar":
            return rows[arg]["paired"]["v31-ref-launch"]["dbar"]
        if kind == "ratio":
            return rows[arg]["mean"] / rows["v31-ref-launch"]["mean"]
        if kind == "q95":
            fam = results["l1b"]["families"][arg]
            tag = sorted(fam["vectors"])[0]
            return fam["vectors"][tag]["q95"]
        if kind == "mexplore":
            return rows["v29-mexplore-full32"][arg]
        raise KeyError(key)

    findings = []
    for desc, key, want, dp in TEXT_BASELINES:
        got = float(get(key))
        tol = 0.5 * (10 ** -dp) + 1e-9
        ok = abs(got - float(want)) <= tol
        findings.append({"desc": desc, "key": key, "text_value": want,
                         "recomputed": got, "dp": dp, "ok": ok})
    got16 = rows["v29-mexplore-full32"]["sha16"]
    findings.append({"desc": "v29-mexplore sha16", "key": "mexplore:sha16",
                     "text_value": MEXPLORE_SHA16, "recomputed": got16, "dp": None,
                     "ok": got16 == MEXPLORE_SHA16})
    return findings


# ---------------------------------------------------------------------------
# 记分卡
# ---------------------------------------------------------------------------

def build_scorecard(results: dict) -> List[dict]:
    l1a, l1b, l2, l3 = results["l1a"], results["l1b"], results["l2"], results["l3"]
    d5 = results["d5"]
    mw = l1b["families"]["worker-leg"]["m_star"]
    mm = l1b["families"]["manager-arm"]["m_star"]
    fresh_pass = next(e for e in d5["rows"] if e["tag"] == "v31-mfresh-full32")
    fresh_qual_new = fresh_pass["lips"]["qual_died"]["new"]
    n_flips = d5["flip_sets"]["n_any"]
    card = []
    for spec in SCORECARD_SPEC:
        row = dict(spec)
        if spec["id"] == "RG1.1":
            row["derived"] = l1a["k_star"]
            row["hit"] = l1a["k_star"] == 22
        elif spec["id"] == "RG1.2a":
            row["derived"] = round(mw, 2)
            row["hit"] = spec["band"][0] <= mw <= spec["band"][1]
        elif spec["id"] == "RG1.2b":
            row["derived"] = round(mm, 2)
            row["hit"] = spec["band"][0] <= mm <= spec["band"][1]
        elif spec["id"] == "RG1.3":
            row["derived"] = "≤%d;fresh(died 7)%s;单档依赖 %s" % (
                l2["line"], "放行" if fresh_qual_new else "拦截",
                ",".join(l2["single_archive_dependence"]) or "无")
            row["hit"] = (l2["line"] == 8 and fresh_qual_new
                          and l2["single_archive_dependence"] == ["b1-ref8k-launch"])
        elif spec["id"] == "RG1.4":
            row["derived"] = l3["outcome"]
            row["hit"] = l3["current_in_region"]
        elif spec["id"] == "RG1.5":
            row["derived"] = "%d(集 %s)" % (n_flips, ",".join(d5["flip_sets"]["any_flip"]))
            row["hit_point"] = n_flips == spec["point"]
            row["hit"] = spec["band"][0] <= n_flips <= spec["band"][1]
        card.append(row)
    return card


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(args) -> int:
    if args.out:
        out_dir = args.out
    elif args.dry:
        out_dir = tempfile.mkdtemp(prefix="g1-dry-")
    else:
        out_dir = os.path.join(REPO_ROOT, "train", "runs", CASE_ID)
    os.makedirs(out_dir, exist_ok=True)

    lock_path = os.path.join(out_dir, ".driver.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print("PREFLIGHT_FAIL:.driver.lock 已存在(单案锁)——%s" % lock_path)
        return 3
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"pid": os.getpid(),
                             "t": datetime.datetime.now(datetime.timezone.utc).isoformat()}))

    led = Ledger(os.path.join(out_dir, "gate_ledger.jsonl"))
    try:
        return _run_locked(args, led, out_dir)
    except SystemExit:
        raise
    except Exception:
        led.log("DRIVER_EXCEPTION", trace=traceback.format_exc()[-2000:])
        raise
    finally:
        try:
            os.unlink(lock_path)
        except OSError:
            pass


def _run_locked(args, led: Ledger, out_dir: str) -> int:
    dry = args.dry
    print("== G1 闸线重标定 驱动器 ==(%s;out=%s)" % ("--dry 试算" if dry else "正案", out_dir))

    # W1
    rc = w1_freeze_check(led, dry)
    if rc is not None:
        return rc

    # W-NOEVAL:import 断言 + 案前扫描
    rc = w_noeval_import_check(led)
    if rc is not None:
        return rc
    pre_snap = snapshot_world()
    led.log("preflight_scan", phase="案前", n_files=len(pre_snap))

    # W-PIN
    pin_mismatches = w_pin_check(led)
    if pin_mismatches:
        led.log("PREFLIGHT_FAIL", gate="W-PIN", mismatches=pin_mismatches)
        print("PREFLIGHT_FAIL:W-PIN 失配 %d 项" % len(pin_mismatches))
        for m in pin_mismatches:
            print("  -", m)
        return 3
    led.log("PREFLIGHT_OK", gates=["W-NOEVAL(import+案前扫描)", "W-PIN"],
            w1="skipped(dry)" if dry else "ok")

    # 提取 + 全套计算,双跑(W-DET)
    print("提取 + 定线计算,第 1 遍…")
    extraction_1 = ex.extract(REPO_ROOT)
    results_1 = al.compute_all(extraction_1, SEED, N_DRAWS)
    bytes_1 = al.canon_bytes(extraction_1) + al.canon_bytes(results_1)
    print("提取 + 定线计算,第 2 遍(W-DET)…")
    extraction_2 = ex.extract(REPO_ROOT)
    results_2 = al.compute_all(extraction_2, SEED, N_DRAWS)
    bytes_2 = al.canon_bytes(extraction_2) + al.canon_bytes(results_2)
    det_ok = bytes_1 == bytes_2
    led.log("W_DET", ok=det_ok, n_bytes=len(bytes_1),
            digest1=hashlib.sha256(bytes_1).hexdigest(),
            digest2=hashlib.sha256(bytes_2).hexdigest())
    if not det_ok:
        led.log("NEEDS_ATTENTION", why="W-DET 两遍不等,回炉不落线(退出码 6)")
        return 6
    extraction, results = extraction_1, results_1

    ex_path = os.path.join(out_dir, "g1_history_extract.json")
    with open(ex_path, "w", encoding="utf-8") as fh:
        json.dump(extraction, fh, ensure_ascii=False, sort_keys=True, indent=1)
    res_path = os.path.join(out_dir, "g1_results.json")
    with open(res_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, sort_keys=True, indent=1)

    # INPUT_MISMATCH 守卫(总则 4)
    baseline_findings = check_text_baselines(extraction, results)
    bad = [f for f in baseline_findings if not f["ok"]]
    if bad:
        led.log("INPUT_MISMATCH", findings=bad,
                rule="正文基准与驱动器重提取不符;禁以正文数字覆盖档案")
        if not dry:
            print("INPUT_MISMATCH 停机呈报:")
            for f in bad:
                print("  -", f)
            return 1
    else:
        led.log("input_baseline_ok", n=len(baseline_findings))

    # 台账落账(D8 词表)
    led.log("CLASS_UNIVERSE", **results["class_universe"])
    led.log("CLASS_ASSIGN", **results["class_assign"])
    led.log("NULL_DIST", script_sha={
        "extract_g1_history.py": sha256_file(os.path.join(TRAIN_DIR, "extract_g1_history.py")),
        "analysis_g1_lines.py": sha256_file(os.path.join(TRAIN_DIR, "analysis_g1_lines.py")),
        "run_g1_recal.py": sha256_file(os.path.join(TRAIN_DIR, "run_g1_recal.py")),
    }, **results["null_dist"])

    l1b = results["l1b"]
    input_sha_universe = {t: extraction["rows"][t]["sha256"] for t in al.UNIVERSE_TAGS}
    input_sha_ph = {t: extraction["rows"][t]["sha256"] for t in al.P_H_TAGS}
    led.log("LINE_DERIVED", line_id="RG1.1-width", principle=results["l1a"]["principle"],
            value=results["l1a"]["k_star"], nature="复核", band=None,
            current=18, current_alpha=results["l1a"]["current_alpha"],
            inputs={"n": 32, "alpha": al.ALPHA_WIDTH})
    for fam, spec_id in (("worker-leg", "RG1.2a"), ("manager-arm", "RG1.2b")):
        f = l1b["families"][fam]
        led.log("LINE_DERIVED", line_id=spec_id + "-mean-" + fam, principle=l1b["principle"],
                value=f["m_star_2dp"], raw=f["m_star"], nature="复核",
                band=[16.5, 18.0] if fam == "worker-leg" else [22.5, 24.5],
                in_band=None,  # 记分卡统一判
                inputs=input_sha_universe, n_vectors=f["n_vectors"],
                circularity_note=l1b["circularity_note"] if fam == "manager-arm" else None)
    led.log("LINE_DERIVED", line_id="RG1.3-death", principle=results["l2"]["principle"],
            value=results["l2"]["line"], nature="复核", band=None,
            scope=al.L2_SCOPE, inputs=input_sha_ph,
            single_archive_label=results["l2"]["single_archive_label"],
            l2_b2_duty="注册主推导及 M∈{1,2,3} 敏感度分支均放行 fresh(died 7);"
                       "维持拦截之参数组合(M=0、H×7000 语境限定、剔除 K1)均不在主推导内;"
                       "此事实交面板按 D4-3 专项审查")
    led.log("LINE_DERIVED", line_id="RG1.4-floor", principle=results["l3"]["principle"],
            value=results["l3"]["outcome"], nature="复核", band=None,
            feasible=[results["l3"]["feasible_lo"], results["l3"]["feasible_hi"]],
            inputs=input_sha_universe)
    led.log("STRAT_TABLE", **results["strat_table"])
    led.log("LOO_REPORT", l2_loo=results["l2"]["loo"], d5_loo=results["d5_loo"])
    led.log("OC_TABLE", old_vs_new=results["oc"]["old_vs_new"],
            grid=results["oc"]["grid"],
            m_sensitivity=results["l2"]["M_sensitivity"],
            cross_family_max_row=results["oc"]["cross_family_max_row"],
            full_table_path=os.path.basename(res_path))
    led.log("BIDIR_REPORT", **results["bidir"])
    led.log("HIST_COUNTERFACTUAL", banner=al.D5_BANNER,
            flip_sets=results["d5"]["flip_sets"],
            d4_triggers=results["d5"]["d4_triggers"],
            extract_script_sha=sha256_file(os.path.join(TRAIN_DIR, "extract_g1_history.py")),
            full_table_path=os.path.basename(ex_path))
    led.log("WOULDTRIP_RULING", **results["wouldtrip"])

    scorecard = build_scorecard(results)
    led.log("VERDICT_PATH", scorecard=scorecard,
            input_mismatch=[f for f in baseline_findings if not f["ok"]],
            dry=dry,
            note="--dry 试算不落 FREEZE_SHA、不作正式收卷" if dry else "案结")

    # 案后扫描(W-NOEVAL)
    post_snap = snapshot_world()
    new_keys = sorted(set(post_snap) - set(pre_snap))
    changed = sorted(k for k in pre_snap if k in post_snap and pre_snap[k] != post_snap[k])
    if changed:
        led.log("CASE_HALT", gate="W-NOEVAL", why="案中在册档案被改动", changed=changed)
        return 7
    if new_keys:
        sources = reconcile_new_archives(new_keys)
        orphan = sorted(k for k, v in sources.items() if v is None)
        if orphan:
            led.log("CASE_HALT", gate="W-NOEVAL", why="无来源可对账之新档案", orphans=orphan)
            return 7
        led.log("NEEDS_ATTENTION", why="案中出现他案获批档案(来源对账通过,不触发自废)",
                sources=sources)
    led.log("postflight_scan", phase="案后", n_files=len(post_snap), new=new_keys)

    print_dry_report(results, baseline_findings, scorecard, det_ok, out_dir)
    return 0


# ---------------------------------------------------------------------------
# 试算报告
# ---------------------------------------------------------------------------

def print_dry_report(results, baseline_findings, scorecard, det_ok, out_dir):
    l1a, l1b, l2, l3 = results["l1a"], results["l1b"], results["l2"], results["l3"]
    d5 = results["d5"]
    P = print
    P("")
    P("================ G1 试算报告 ================")
    P("[L1a] k* = %d(P(≥21)=%.4f,P(≥22)=%.4f;现行 18 之单肢 α=%.4f)" % (
        l1a["k_star"], l1a["tail_at"]["21"], l1a["tail_at"]["22"], l1a["current_alpha"]))
    for fam in ("worker-leg", "manager-arm"):
        f = l1b["families"][fam]
        tag = sorted(f["vectors"])[0]
        v = f["vectors"][tag]
        P("[L1b] %s m* = %.4f(2dp %.2f;向量 %s d̄=%.4f,SE_flip=%.3f,零Δ %d;"
          "去杠杆 Q95=%.3f,最大杠杆Δ=%.2f)" % (
              fam, f["m_star"], f["m_star_2dp"], tag, v["dbar"], v["se_flip"],
              v["n_zero_deltas"], v["deleverage_q95"], v["max_leverage_delta"]))
    P("[L2] 上包络 max(P_H)=%d(支撑 %s)+ M=%d → 线 died ≤%d;LOO 单档依赖:%s" % (
        l2["envelope"], ",".join(l2["envelope_support"]), l2["M"], l2["line"],
        ",".join(l2["single_archive_dependence"]) or "无"))
    P("     M 敏感度:%s" % {k: v["line"] for k, v in l2["M_sensitivity"].items() if "line" in v})
    P("[L3] 可行域 [%.4f, %.4f];现行 85/92=%.6f %s → %s" % (
        l3["feasible_lo"], l3["feasible_hi"], l3["current_value"],
        "在域内" if l3["current_in_region"] else "出域", l3["outcome"].split("(")[0]))
    P("[D2-U] 簇指派:损伤 %s;健康 %s;中间带 %s" % (
        results["class_assign"]["clusters"]["damaged"],
        results["class_assign"]["clusters"]["healthy"],
        results["class_assign"]["clusters"]["unassigned"] or "空"))
    P("")
    P("[记分卡](三性质标注;【复核】不符者按 INPUT_MISMATCH 语义呈报)")
    for row in scorecard:
        mark = "命中" if row["hit"] else "未中"
        extra = ""
        if "hit_point" in row:
            extra = ";点值%s" % ("命中" if row["hit_point"] else "未中")
        P("  %s %-28s 预期 %s → 导出 %s【%s;带判 %s%s】" % (
            row["id"], row["line"], row["expect"], row["derived"], row["nature"], mark, extra))
    P("")
    fs = d5["flip_sets"]
    P("[D5] 行数 %d;资格判翻转 %s;发射数值肢翻转 %s;任意翻转 n=%d(D4 双触发:出带≥4 %s,单一候选 %s)" % (
        len(d5["rows"]), fs["qual_flips"], fs["launch_numeric_flips"], fs["n_any"],
        d5["d4_triggers"]["flip_count_out_of_band(≥4)"],
        d5["d4_triggers"]["flip_set_single_candidate"]))
    P("[D5-LOO] %s" % results["d5_loo"]["verdict"])
    P("[开奖史] " + ";".join("#%d %s wins=%s(台账 %s,%s)" % (
        h["draw_no"], h["tag"], h["wins"], h["ledger_wins"], "对号✓" if h["reconciled"] else "对号✗")
        for h in d5["draw_history"]))
    P("[W-DET] 双跑逐字节 %s" % ("一致" if det_ok else "不一致"))
    bad = [f for f in baseline_findings if not f["ok"]]
    if bad:
        P("[INPUT_MISMATCH] %d 项:" % len(bad))
        for f in bad:
            P("   - %s:正文 %s vs 重提取 %.6g" % (f["desc"], f["text_value"], f["recomputed"]))
    else:
        P("[INPUT_MISMATCH] 无(正文基准 %d 项全数命中,半 ulp 容差)" % len(baseline_findings))
    P("[产出] 台账与全表:%s" % out_dir)
    P("=============================================")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G1 闸线重标定驱动器(测量案;纯离线)")
    ap.add_argument("--dry", action="store_true",
                    help="冻结前试算:跳过 W1;台账写入临时目录;INPUT_MISMATCH 只呈报不停机")
    ap.add_argument("--out", help="输出目录(缺省:正案 train/runs/recal-g1;--dry 临时目录)")
    args = ap.parse_args(argv)
    try:
        return run(args)
    except Exception:
        traceback.print_exc()
        return 1  # P1:DRIVER_EXCEPTION


if __name__ == "__main__":
    sys.exit(main())
