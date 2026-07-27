"""v33「内容案:课⑤干窗课程 × ④乙正样本」驱动
(docs/PREREG-内容案-课⑤x④乙.md rev4 条款唯一执行者;骨架 = run_b1_infra.py 式
(冻结常量块 / stage_done 幂等 / pre() 预检 / --smoke 独立入口)+
run_v32_sovereign.py 之 exam_or_adopt/biteq/腿循环/胜者裁定段)。

授号注记:v33 系面板过审后合规授予(编号注记,首事件前授号);台账目录
train/runs/v33-content/ 自首事件起以实名建立。

阶段序(D2/附:驱动器结构;严格串行防后见,由 STAGE_SEQUENCE 强制):
  S0  pre() 预检(W1/W7/W-PIN 逐件 sha/事件行钉死/304000 对账/308000 处女/
      W-H8/W9/W-C 承继/CASE_RUNTIME 捕获)
  S1  G0_BASELINE 转录(E0 双基线四件 sha 核验;捕获在驱动器之外、施工之前)
  S2  S-bc1 BC-v1 重生成分列断言(demos 字节 ≡ 冻结常量 / policy torch.equal
      对 _previous 归档件)→ BC_REGEN
  S3  S-bc2 BC-v2 采集+训练 → N12_GATE(携分母定义/逐局分解/OC 消耗位;
      P-N12:唯一 OC 阈 0.65→0.70 重采一次)
  S4  G0 六件套:G0-1 双端点位级恒等 → G0-2a 零侵入对(张量级,种子 308000)
      → G0-2b 功能烟测(实测 p 序列 ≡ 注册表前缀)→ G0-幽灵 → G0-示范池
      → G0-6 REF_BITEQ 两发(≡113.0 / ≡140.9 全表,冻结先决)
  S5  FREEZE_SHA / CASE_RUNTIME / NEWLINE_ADOPT(升格审查卷两条件载荷位)
  S6  【W-LAUNCH 发车令闸】无 LAUNCH_ORDER 事件 → sys.exit(9) 停机待命
      (AWAITING_LAUNCH 非失败态,不落 NEEDS_ATTENTION)
  S7  三腿串行(L-base → L-cur → L-full;每腿:点火(经 train/launch_case.sh
      点火之驱动器子进程)→ 腿终 dry_curriculum.jsonl ≡ 注册表全表复核 →
      金丝雀离线序列(补评截止 = 该腿 s16 FIRING_START 前,驱动器内断言)
      → s16 → full32 → full32-m29)
  S8  课⑤主判 / ④乙主判(L-full 无论胜负必判)→ 胜者裁定(决胜序 D4-5)
      → 发射判据(G1 新线首引,合取式全录)
  S9  H1/H2 留出捆绑对(施于胜者,1 对/2 发)→ R 线记分卡 → VERDICT_PATH

**发车纪律:冻结 ≠ 发车;面板过审 ≠ 发车;本驱动器任何路径不得被读作发车授权。**

用法:
  .venv/bin/python train/run_v33_content.py          # 全案(幂等续跑)
  .venv/bin/python train/run_v33_content.py --smoke  # 只跑 G0-2a/2b 烟测;
                                                     # 不落正账,走独立 smoke 台账
点火一律经 train/launch_case.sh(E8):
  train/launch_case.sh train/runs/v33-content train/run_v33_content.py

退出码(D5,集中常量化于 EXIT_CODES):
  0 案结/幂等;2 额度耗尽;3 预检(PREFLIGHT_FAIL);4 不空闲/锁冲突;
  5 W5 发车前漂移;6 runtime 案中漂移;7 CASE_HALT_G0;8 REF_DIVERGENCE;
  9 AWAITING_LAUNCH(冻结待发车,非失败态);其余 P1。
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
import traceback
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from eval_contract import (PROTOCOL_VERSION, EvalContractError,
                           OperationalFailure, OutputReservationError,
                           exclusive_lock, expected_eval_identity,
                           freeze_eval_identity, read_eval_archive,
                           strict_json_loads, verify_eval_identity)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
V33 = RUNS / "v33-content"                     # 授号实名台账目录(编号注记)
LEDGER = V33 / "gate_ledger.jsonl"
SMOKE_LEDGER = V33 / "smoke_ledger.jsonl"      # --smoke 独立台账(不落正账)
EVAL = RUNS / "eval-assembled"

# ======================================================================
# 退出码表(D5 逐字,集中常量化)
# ======================================================================
EXIT_CODES = {
    0: "案结/幂等",
    2: "额度耗尽(P2 评测发 / P3 腿点火 / P5 必做发)",
    3: "预检不过(P4 PREFLIGHT_FAIL)",
    4: "不空闲/锁冲突(W8)",
    5: "W5 发车前漂移",
    6: "runtime 案中漂移",
    7: "CASE_HALT_G0",
    8: "REF_DIVERGENCE",
    9: "AWAITING_LAUNCH(FREEZE_SHA 已落、LAUNCH_ORDER 未入册之待命态,"
       "非失败,不落 NEEDS_ATTENTION)",
}

# ======================================================================
# W7 工件钉死(全文 sha 驱动器冻结常量;失配即 P4 不发车)
# ======================================================================
KING_ZIP = ROOT / "train" / "models" / "v28-worker-leg1" / "model_final.zip"
KING_ZIP_SHA = "2f7bc9dd810956c3feeb330575c9a03ddff0b476333ac429a411935985b04f42"
KING_NPZ = ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz"
KING_NPZ_SHA = "976b6c05edaa0a32bb30bd372782e1201c72b029cedcbb3a5bf2361d34f27f8a"
KING_STEPS = 3_497_984
H_NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
H_NPZ_SHA = "0f2264860b0960e7951efd424836b90c09c002cebca7bf8109fd669b13be63d7"   # == DEFAULT_MANAGER_SHA256
M29_NPZ = ROOT / "train" / "models" / "v29-manager-mfresh" / "policy.npz"
M29_NPZ_SHA = "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6"
KING_SD = RUNS / "v32" / "king_anchor_sd.pt"
KING_SD_SHA = "009aaad29d2653cde3f4e8ed2fafd8861a0f1f572a140c64118df9e3fa3df35d"  # v32 落定值(案内 parity 复验)

# BC-v1 件(W7 分列式断言;重生成非可选项,E1 改 worker_env 后旧回执必被预检拒)
BC_V1_DIR = RUNS / "bc-worker"
BC_SD = BC_V1_DIR / "policy_sd.pt"
BC_V1_DEMOS = BC_V1_DIR / "demos.npz"
BC_V1_DEMOS_SHA = "3bf892d611e41853eca8fce0cb146753af41ad2c3a21b6c581df1041fb1d9363"
# f052067a… 系 policy_sd.pt **旧落定值**(v32 BC_REGEN),仅作 torch.equal
# 对照件谱系引用——torch.save 字节不可复现,字节等断言原理性不可过(废止);
# 新 sha 以本案 BC_REGEN 事件落账为新落定值。
BC_SD_PREV_LINEAGE_SHA = "f052067a589cfcdedaf1754ae6d241d736bb97f6fc798683f395c76cb0ff98e6"

# BC-v2 件(E2 独立产物目录;仅经 --bc-aux-demos 进辅助损失,canonical 路径不动)
BC_V2_DIR = RUNS / "bc-worker-v2"
BC_V2_DEMOS = BC_V2_DIR / "demos.npz"
BC_V2_SD = BC_V2_DIR / "policy_sd.pt"
BC_V2_REPORT = BC_V2_DIR / "bc_report_v2.json"
N12_GATE_MIN = 122            # D7:≥122(=审计点估 244 之半)
RECALL12_GATE_MIN = 0.5       # D7:held-out 12 类 recall ≥0.5
PREVENTIVE_MAIN = 0.65        # D7 预防阈主案
PREVENTIVE_OC = 0.70          # 唯一注册 OC(P-N12 重采旋钮)
N12_DENOMINATOR_DEF = ("分母 = 现切分下教师 v2 迟滞模拟之实标 a12 态(窗首触发态);"
                       "度量 = held-out argmax 命中;fail-closed"
                       "(承统计 B-1 钉死;1,975 态口径废止)")
N12_CLUSTER_NOTE = ("分母簇集中注记(核认勘正口径):审计所证系 held-out 触发态口径 "
                    "74.7% 集中于局 178/187 两簇(880/596/1,975);实标窗(窗首)"
                    "口径之逐局集中度以本 N12_GATE 实测逐局分解为准;recall 读数噪声大")

# ======================================================================
# W-PIN 输入档案钉死(G1 形制,全文 sha 冻结常量;预检失配 → PREFLIGHT_FAIL 3)
# 304000 对账之"全台账"枚举面即此封闭台账集(D1;免"全"字开口)。
# ======================================================================
W_PIN = {
    "v32-ref-launch": (EVAL / "v32-ref-launch.json",
                       "48033577f8f124ae81fc5436eb44e5aa0bf541a4437d7fec99d8f5b1209c71fa"),
    "v32-ref-science": (EVAL / "v32-ref-science.json",
                        "1736185e286c1f2a98f6d4f503c72b40e322dcc31151e0e83068714c1b29f5f0"),
    "b1-ref8k-launch": (EVAL / "b1-ref8k-launch.json",
                        "3b4ef1681134d51d61c3081195fc620ea4ad4d7c7f034597b9f92782cabe6a19"),
    "b1-ref8k-science": (EVAL / "b1-ref8k-science.json",
                         "e23a83383b8e286d9baa85ee9142970103457b0b5a3b249c4f0826b184910ef4"),
    "AUDIT-内容案档案审计": (ROOT / "docs" / "AUDIT-内容案档案审计.md",
                     "433f3a970d799b786815feadc9bd96eae839d1e08de2b75724a04684b571166d"),
    # 升格审查卷(未决 E 履行件)+ g1_results.json:NEWLINE_ADOPT 两条件载荷之
    # 真源档;g1_results sha 补入 W-PIN 系该卷肢一成立条件之明文选项。
    "AUDIT-G1双标签肢升格审查": (ROOT / "docs" / "AUDIT-内容案-G1双标签肢升格审查.md",
                        "8e4be0148aa3760a5bb1a7de0514685e3f032d8c05fa77112329d30f2897c7e0"),
    "g1_results.json": (RUNS / "recal-g1" / "g1_results.json",
                        "aa03addea7003e2f6f5b2e9fb595dd75179d8a5cddb14b478c0073c6a48e74b0"),
    # E0 双基线档四件(king+throne × 双端点;captured=变更前,施工前捕获)
    "E0-king-p0": (ROOT / "docs" / "assets" / "g0v32_traj_baseline.json",
                   "2f567d7abfc5b4714cac4ff3236b09ab67e22485d2b05b52d9dc956692a26d6f"),
    "E0-king-p1": (ROOT / "docs" / "assets" / "g0v32_traj_baseline_skip_dry_true.json",
                   "bdb43707064d24b5f985c33f211755c8331d215bd41d1630ebdc4019fefff977"),
    "E0-throne-p0": (ROOT / "docs" / "assets" / "g0v32_traj_baseline_throne.json",
                     "e235e96972f8f2eeaaebec2bc8970c3e3f77b715f26c364250c42d8d37131cfc"),
    "E0-throne-p1": (ROOT / "docs" / "assets" / "g0v32_traj_baseline_throne_skip_dry_true.json",
                     "8d541b56d10c18b3421de3dd32be4f17051a03488d531916c24177814a0b28ba"),
}
# W-PIN 封闭台账集(v29-v32 / reanchor-r2 / infra-b1 / recal-g1 各台账全文)
W_PIN_LEDGERS = {
    "v29": (RUNS / "v29" / "gate_ledger.jsonl",
            "27ad88b295594f2d7131c81be29a51fa246da82f7a932a939a524e056dd06f67"),
    "v30": (RUNS / "v30" / "gate_ledger.jsonl",
            "7187c30c0f8e255fd7f7ab23d01b5164a5557d9c5130633529e85d147e7a463e"),
    "v31": (RUNS / "v31" / "gate_ledger.jsonl",
            "19ed09c664947228a3e07d80feb56edf8bc05124b853bc1ad796b2c6af8d09e5"),
    "v32": (RUNS / "v32" / "gate_ledger.jsonl",
            "8c197617097548f04e82bb3157b679bf6792ce3da937bd4e8ec657d920fe0d22"),
    "reanchor-r2": (RUNS / "reanchor-r2" / "gate_ledger.jsonl",
                    "a3351a9ba525d8d3f400c61716169d991ae16c05b8f1e2f326a74f71fb4828d5"),
    "infra-b1": (RUNS / "infra-b1" / "gate_ledger.jsonl",
                 "cdc24b16b1ac11a86a64ffb031ea5a10b432d61e5581271346926c0f7d53ed15"),
    "recal-g1": (RUNS / "recal-g1" / "gate_ledger.jsonl",
                 "f0b7874edd8a778c7fb3d0be6ff88fd676737ebe8b3fed77b6a498821d547839"),
}
# 事件行钉死(档案 sha + 行号引用 + 行字节 sha;rev3 blocker 勘正:G1 系
# LINE_DERIVED **五行**(:10-14)+ WOULDTRIP_RULING 行(:20),"六事件行"废止)
# 形制:名 → (台账键, 行号(1 起), 事件名, line_id 或 None, 行字节 sha256)
W_PIN_EVENT_LINES = {
    "B1-CANARY_SET": ("infra-b1", 19, "CANARY_SET", None,
                      "173ce61ad03db883e82ec40b7e40228a7ab496b0bcf464cfbf850679eb68e612"),
    "G1-RG1.1-width": ("recal-g1", 10, "LINE_DERIVED", "RG1.1-width",
                       "edf605cd7e3735037e89a4dee56a35f24f7695fdee0c0c2a60a77ae5384bb241"),
    "G1-RG1.2a-mean-worker-leg": ("recal-g1", 11, "LINE_DERIVED",
                                  "RG1.2a-mean-worker-leg",
                                  "099e060db3604d73db72e72b341720f96a28c0639ebc3d76849b7d2287f73d94"),
    "G1-RG1.2b-mean-manager-arm": ("recal-g1", 12, "LINE_DERIVED",
                                   "RG1.2b-mean-manager-arm",
                                   "92490e2c81883f02084554b5487a388f2fe4eaeaef61ff236aef2c7161071146"),
    "G1-RG1.3-death": ("recal-g1", 13, "LINE_DERIVED", "RG1.3-death",
                       "323fb1b7904f75d7a7ec1c39de62e85c63281be96fb09b6f173bad46a0f5fee2"),
    "G1-RG1.4-floor": ("recal-g1", 14, "LINE_DERIVED", "RG1.4-floor",
                       "63d819976044228014af33b8b2e1a54545d222af821464a0eedceaaf361ffacc"),
    "G1-WOULDTRIP_RULING": ("recal-g1", 20, "WOULDTRIP_RULING", None,
                            "95443a26551cb477d10b25c8cbc36e544a2415466cbb1256cf7ff31f4b99a2cb"),
}
# K1 出处(升格审查卷肢二④:infra-b1/gate_ledger.jsonl:14 exam_ok b1-ref8k-launch)
K1_PROVENANCE = {"ledger": "infra-b1/gate_ledger.jsonl", "line": 14,
                 "archive_sha16": "3b4ef1681134d51d"}

# FREEZE_SHA / CASE_RUNTIME 结构化占位(冻结时落账,禁伪值;正文零自指 sha:
# 冻结 sha 仅入台账 FREEZE_SHA 事件,案级五 sha 以首个 CASE_RUNTIME 事件落定)
FREEZE_SHA_PLACEHOLDER: str | None = None
CASE_RUNTIME_PLACEHOLDER: dict | None = None

# ======================================================================
# 腿配方常量(D3;共用命令形 = v32-sov 逐字 + B1 仪表旋钮;冻结钉死)
# ======================================================================
LEG_STEPS = 499_712
NT_TARGET = 3_997_696          # = king 3,497,984 + 499,712(增量语义;各腿 nt 闸)
QUANTUM = 2_048                # 512 n-steps × 4 envs
SEED = 304_000                 # 三腿同种子(P8 复用;豁免成文见 304000 对账断言)
BETA = 0.015625                # 蒸馏锚 β 冻结原值(圈 5,锚公式零触碰)
BC_AUX_LAMBDA = 0.015625       # λ_bc(D7 注册裁量常数,与 β 同量级)
MAIN_TABLE = "linear:1.0:0.5:147,hold:0.5:97"   # 退火主表(圈 2 附裁批文即定)
DRY_CURRICULUM_LEG_START = 3_497_984            # 腿相对 rollout 锚(E1②)
CALIB_PROBES = tuple(KING_STEPS + 49_152 * k for k in range(1, 11))
CKPT_EVERY = 98_304
SENTINEL_EVERY = 49_152
DRY_ANCHOR_EVERY = 49_152
# rev4 D3 勘正增列(十二附二①,待追认):E5①② record-only 仪表旋钮入共用
# 命令形——圈 5 亲批件之义务面(干/鲜 distill_ce 供课②定标数据)+ E5②
# "基线自本案首点火起建"/RC.14;零侵入由 G0-2a 旋钮全在位臂实弹担保。
DISTILL_CE_PROBE_EVERY = 49_152
DRYWIN_METRICS_EVERY = 49_152
CANARY_STEPS = tuple(KING_STEPS + 98_304 * k for k in range(1, 5))   # 中期四点
EXTRA_CKPT_STEP = KING_STEPS + 491_520          # 机械另落 ckpt:归档不入序列
LEG_TIMEOUT = 21_600           # 6h/腿

# LEGS 表 = D3 逐腿附加项逐字(权威,冻结钉死;共用形自身不产生 p≡1.0,
# L-base 必须显式携 --skip-dry;--bc-aux-demos 取 D3 字面路径之仓内定位)
LEGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("v33-base", ("--skip-dry",)),
    ("v33-cur", ("--dry-curriculum-schedule", MAIN_TABLE)),
    ("v33-full", ("--dry-curriculum-schedule", MAIN_TABLE,
                  "--bc-aux-lambda", "0.015625",
                  "--bc-aux-demos", str(BC_V2_DEMOS),
                  "--bc-aux-liveness-preflight")),
)
LEG_NAMES = tuple(name for name, _ in LEGS)
LEG_SHORT = {"v33-base": "base", "v33-cur": "cur", "v33-full": "full"}
CURRICULUM_LEGS = ("v33-cur", "v33-full")

POOL_PROBE = "7000-7031"
POOL_S16 = "7000-7015"
POOL_HOLD = "8000-8031"
REF_TAGS = {"launch": "v33-ref-launch", "science": "v33-ref-science"}
REF_MEANS = {"launch": 113.0, "science": 140.9}   # REF_BITEQ 全表恒等之 agg 锚
HOLDOUT_TAGS = ("v33-win-hold8k", "v33-win-hold8k-m29")
CANARY_TAGS = tuple(f"v33-{LEG_SHORT[leg]}-canary{k}-{m}"
                    for leg in LEG_NAMES for k in range(1, 5)
                    for m in ("h", "m29"))
LEG_EXAM_TAGS = tuple(f"v33-{LEG_SHORT[leg]}-{suffix}"
                      for leg in LEG_NAMES
                      for suffix in ("s16", "full32", "full32-m29"))
ALL_EXAM_TAGS = (tuple(REF_TAGS.values()) + LEG_EXAM_TAGS + HOLDOUT_TAGS
                 + CANARY_TAGS)

# ======================================================================
# G0 烟测常量(D2-2a;处女种规则三肢钉死:①W-PIN 封闭台账集全事件 + 评测池
# 段守卫;②307000 已非处女(infra-b1:3 G0_NULLINTRUSION 在案),预登记 =
# 308000;③G0-2b 各短跑同规则同种子)
# ======================================================================
SMOKE_SEED = 308_000
SMOKE_STEPS = 102_400          # 50 rollout,整除 2048
SMOKE_END = KING_STEPS + SMOKE_STEPS    # 3,600,384
SMOKE_CALIB = (KING_STEPS + 49_152, KING_STEPS + 98_304)
SMOKE_HOLD_TABLE = "hold:1.0:50"        # G0-2a 调度钉 p≡1.0(50 rollout 全覆盖)
# E5①② 旋钮自 rev4 起随共用命令形(with_b1_knobs=True)入烟测各臂,
# 取值 49,152(DISTILL_CE_PROBE_EVERY/DRYWIN_METRICS_EVERY;窗内各 ≥1 次);
# 裸臂 with_b1_knobs=False 不携(v32-sov 逐字不许动)。
SMOKE_RUNS = {"bare": "v33-smoke-bare", "knobs": "v33-smoke-knobs",
              "func-p": "v33-smoke-funcp", "func-aux": "v33-smoke-funcaux"}
SMOKE_TIMEOUT = 3_600
V32_SOV_CALIB = (3_747_984, 3_947_984)  # v32-sov 命令形逐字之 calib 步表(裸臂)

# ======================================================================
# 判据常量(D4/D7;全部预登记案中禁调)
# ======================================================================
# G1 新线四常数(引用件;LINE_DERIVED 行号引用,见 W_PIN_EVENT_LINES;
# 预检对台账行实值恒等复核,防"常量抄错"):
G1_PAIRED_DIFF = 17.32     # RG1.2a(worker-leg 族;"单向量线"标签强制携带)
G1_PAIRED_WINS = 22        # RG1.1(k* = min{k: P(X≥k|32,.5) ≤ 0.05})
G1_DIED_MAX = 8            # RG1.3("K1 单档依赖"标签强制携带)
FLOOR_FRAC = (85.0, 92.0)  # RG1.4 维持现行 85/92
ABANDON_FRAC = (75.0, 112.4)   # 放弃闸,适用于本案(D4-1 钉死;仅阻断发射与
                               # 名分语境,不阻断判据 2/3 科学主判)
A12_USE_LINE = 0.1         # 承 v32 注册常量(离散义 32 局 ≥4 实饮)
SURV_MEAN_BAND = -2.0      # ④乙档位均差肢 ≥−2(vs L-cur×M29 配对,重注册)
R4 = {"descend": 0.0204, "override_sentinel": 0.03,
      "override_void": 0.08, "cap": 0.05}   # 哨兵肢零触碰清单(G1 L1)
COURSE5_ANCHOR_PRIOR = 5   # 先验点(承 P8,降为先验并列不作锚;锚 = L-base 实测)
DEEPWATER_SEED = 7017      # 深水魔种常备对照(7017 结构席注记)
MULTIPLE_COMPARISON = {    # D4-1 列报义务(G1 L1a 形制;列报非裁线)
    "per_candidate_width_alpha": 0.0251,
    "effective_alpha_3arms": 0.073,        # ≈ 1 − 0.9749³
    "joint_alpha_3arms": 0.013,            # G1 联合 0.0045 基之 3 臂推算
    "unit_note": "α 单位系 per-candidate 单肢边际(G1 L1a);开奖臂 3",
}
DELEVERAGE_Q95 = 15.99     # 线侧去 7017 重算 Q95(升格审查肢一成立条件)
DELEVERAGE_Q95_RAW = 15.987129
OC_ALPHA_COLUMNS = {"worker_leg_old_alpha_mean": 0.357,
                    "worker_leg_new_alpha_mean": 0.050,
                    "note": "OC old_vs_new 四列随呈(升格审查肢一限定 6)"}

# G1 核认①② 文本逐字(升格审查卷两条件;NEWLINE_ADOPT 事件载荷位真源 =
# docs/PREREG-G1-闸线重标定.md rev3 核认修正章,经亲证逐字转录)
G1_VERIFY_NOTE_1 = ("若无后案跨池施考之在册义务,跨池覆盖系前瞻裁量而非既存义务;"
                    "当前资格语境(H×7000)下本线之全部操作性余量(1→8)系域外档"
                    "(K1)独供")
G1_VERIFY_NOTE_2 = ("否决分支措辞改\"族内唯一健康向量因循环性被否,该族不可定线\";"
                    "RG1.2a/2b 均挂\"单向量线\"标签(2b 另加\"高杠杆种子 7017"
                    "(Δ−149.86)驱动\"注记),随 D7 引用强制携带,去杠杆敏感度行"
                    "同呈;拦截向诚实注记:\"以 fresh 向量定线系系统性抬高 fresh 型"
                    "(高方差)候选之门槛\",D4-7 复活审计援引本线时强制在场。")

# R 线带常量(D8;闭区间 [lo, hi];点估凡未注出处者系裁量非估计)
RC2_BAND = {"point": -12.0, "band": (-45.0, 5.0)}
RC3_BAND = {"point": 4.0, "band": (-12.0, 25.0)}
RC4_BAND = {"point": 2.0, "band": (-12.0, 18.0)}
RC6_BAND = {"point": 0.8, "band": (0.0, 8.0)}
RC8_BAND_H = {"point": 2, "band": (0, 8)}
RC8_BAND_M29 = {"point": 5, "band": (0, 10)}
RC9_BAND = {"point": 30.0, "band": (8.0, 55.0)}
RC9_MS = {"healthy": {"max": (20.0, (5.0, 45.0)), "over": (2, (0, 8))},
          "damaged": {"max": (60.0, (45.0, 130.0)), "over": (12, (8, 17))}}
RC9_BRANCH_LINE = -20.0        # MS 双分支由 RC.2 实测落点机械触发(禁事后择带)
MS_FLAG_LINE = 20.0
RC10_BAND = {"healthy": (0.5, 1.0), "damaged": (0.0, 0.5)}
RC12_BAND = {"point": 4.0, "band": (0.0, 10.0)}   # pp;零点 0.7515
DRY_REF_THRONE = 0.7515
DRY_REF_LINEAGE = 0.6305
RC13_N12_BAND = {"point": 244, "band": (122, 700)}
RC13_RECALL_BAND = {"point": 0.8, "band": (0.5, 1.0)}
RC15_BAND = {"point": -10.0, "band": (-50.0, 10.0),
             "d2_point": 5, "d2_band": (2, 14)}

CASE_RT: dict | None = None
_LEDGER_PATH = LEDGER          # --smoke 切至 SMOKE_LEDGER(不落正账)

# 阶段序(D2 严格串行防后见;后阶段在前阶段未 done 时拒跑)
STAGE_SEQUENCE = ("G0_BASELINE", "BC_REGEN", "N12_GATE", "G0_ENDPOINT",
                  "G0_NULLINTRUSION", "G0_SMOKE", "G0_GHOST", "DEMO_LEDGER",
                  "REF_BITEQ", "FREEZE_SHA", "LAUNCH_ORDER", "LEGS")


# ======================================================================
# 台账与通用工具(run_b1_infra / run_v32_sovereign 逐字承继 + 案级扩展)
# ======================================================================

def log(event: dict):
    V33.mkdir(parents=True, exist_ok=True)
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(_LEDGER_PATH, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    V33.mkdir(parents=True, exist_ok=True)
    with open(V33 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def sha16(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def sha256(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OperationalFailure(message)


class PreflightFailure(RuntimeError):
    """P4:预检不过 → 不发车呈报(退出码 3)。"""


def pre(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightFailure(message)


def runtime_five(snapshot) -> dict:
    rt = snapshot["runtime"]
    return {"bridge": rt["bridge"]["sha256"], "engine": rt["engine"]["sha256"],
            "game_data": rt["content"]["game_data"]["sha256"],
            "assets": rt["content"]["assets"]["sha256"],
            "protocol": rt["python_protocol"]["sha256"]}


def assert_case_runtime(snapshot, where: str):
    require(CASE_RT is not None, "CASE_RUNTIME 未落定")
    current = runtime_five(snapshot)
    if current != CASE_RT:
        log({"event": "CASE_HALT_RUNTIME_DRIFT", "where": where,
             "case": CASE_RT, "current": current})
        attention(f"案级运行时漂移({where}),停机呈报")
        raise SystemExit(6)


def run(cmd, logfile, timeout) -> int:
    V33.mkdir(parents=True, exist_ok=True)
    with open(V33 / logfile, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return 124


def _archive_residue(run_dir: pathlib.Path) -> None:
    """点火前残留归档(承 B1 _prepare_run_dir 形制):目录在即整体改名
    <name>-prev<ns>,保证 jsonl 证据面自零起算(2026-07-19 勘正,
    G0-2a 16:55:51 FAIL 根因 = 烟测残留追加堆积)。"""
    if run_dir.exists():
        run_dir.rename(run_dir.with_name(
            f"{run_dir.name}-prev{time.time_ns()}"))


def zip_steps(p: pathlib.Path) -> int:
    try:
        with zipfile.ZipFile(p) as z:
            return int(json.loads(z.read("data"))["num_timesteps"])
    except Exception:
        return 0


def zip_data_field(p: pathlib.Path, key: str):
    with zipfile.ZipFile(p) as z:
        return json.loads(z.read("data")).get(key)


def read_ledger() -> list[dict]:
    if not _LEDGER_PATH.is_file():
        return []
    out = []
    for i, line in enumerate(_LEDGER_PATH.read_text().splitlines(), 1):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise OperationalFailure(f"台账第 {i} 行不可解析: {exc}") from exc
    return out


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def stage_done(events, name) -> bool:
    return any(e.get("event") == name for e in events)


def leg_starts(events, leg) -> int:
    return sum(1 for e in events
               if e.get("event") == "leg_start" and e.get("leg") == leg)


def firing_count(events, tag) -> int:
    return sum(1 for e in events
               if e.get("event") == "FIRING_START" and e.get("tag") == tag)


def s16_fired(events, leg) -> bool:
    """补评截止谓词(P-canary/合规 M-4):该腿 s16 FIRING_START 已在册。"""
    return firing_count(events, f"v33-{LEG_SHORT[leg]}-s16") > 0


def by_seed(rows, lo, hi) -> dict:
    m = {r["seed"]: r for r in rows}
    require(len(rows) == len(m), "种子集合异常(含重复 seed)")
    require(set(m) == set(range(lo, hi + 1)), f"种子集合异常(须为 {lo}-{hi})")
    return m


def depth2_count(rows) -> int:
    return sum(1 for r in rows if r["depth"] >= 2)


def d_windows(row) -> int:
    return str(row["mode_seq"]).count("D")


def dive_per_ep(rows) -> float:
    return sum(str(r["mode_seq"]).count("D") for r in rows) / max(1, len(rows))


def a12_per_ep(agg) -> float:
    hist = agg.get("worker_action_hist", {}) or {}
    return round(int(hist.get("12", 0)) / max(1, int(agg.get("n", 32))), 2)


def a13_per_ep(agg) -> float:
    hist = agg.get("worker_action_hist", {}) or {}
    return round(int(hist.get("13", 0)) / max(1, int(agg.get("n", 32))), 2)


def impl_bundle_sha16() -> str:
    import train_ppo
    return train_ppo._implementation_bundle_sha256()[:16]


def dry_curriculum_table() -> tuple[float, ...]:
    """退火主表全表(序号→p;单一真源 = train_ppo._parse_dry_curriculum_schedule;
    端点含内插语义:linear 段第 k 项 = p0+(p1−p0)·k/(n−1),第 147 量子末项恰达
    0.5——严格下降实占 299,008 步、p=0.5 实占 200,704 步,以此注册语义为真源)。"""
    import train_ppo
    table = train_ppo._parse_dry_curriculum_schedule(MAIN_TABLE)
    require(len(table) * QUANTUM == LEG_STEPS,
            f"退火主表量子账失衡:{len(table)}×{QUANTUM} != {LEG_STEPS}")
    return table


def assert_stage_prereqs(events, stage: str):
    """阶段序防后见(D2):stage 之前的全部阶段须已在册(严格串行)。"""
    idx = STAGE_SEQUENCE.index(stage)
    predicates = {
        "N12_GATE": lambda ev: any(e.get("event") == "N12_GATE"
                                   and e.get("gate") == "PASS" for e in ev),
        "G0_NULLINTRUSION": lambda ev: any(
            e.get("event") == "G0_NULLINTRUSION" and e.get("verdict") == "PASS"
            for e in ev),
        "G0_SMOKE": lambda ev: any(e.get("event") == "G0_SMOKE"
                                   and e.get("verdict") == "PASS" for e in ev),
        "REF_BITEQ": lambda ev: {e.get("ref") for e in ev
                                 if e.get("event") == "REF_BITEQ"} >= {"launch",
                                                                       "science"},
    }
    missing = []
    for name in STAGE_SEQUENCE[:idx]:
        done = predicates.get(name, lambda ev, _n=name: stage_done(ev, _n))(events)
        if not done:
            missing.append(name)
    pre(not missing, f"阶段序防后见(D2 严格串行):{stage} 之前缺 {missing}")


# ======================================================================
# 种子对账断言(D1 改写件 + D2-2a 处女种规则;纯函数面供单测正反例)
# ======================================================================

def w_pin_ledger_lines() -> dict[str, list[str]]:
    """W-PIN 封闭台账集全文行(304000/308000 对账之唯一枚举面)。"""
    return {name: path.read_text().splitlines()
            for name, (path, _sha) in W_PIN_LEDGERS.items()}


def assert_seed_304000_provenance(lines_by_ledger: dict[str, list[str]]):
    """304000 对账断言(rev3 D1 逐字):W-PIN 封闭台账集全事件扫描,历史出现
    恰为且仅为 infra-b1 P8 leg_start;复用豁免成文(同种子承 P8 三角对账之
    跨案注记:L-base 对 P8 之差 = 主权旋钮);禁种子搜索,重点火禁换种子。"""
    hits = [(name, i, raw)
            for name, lines in lines_by_ledger.items()
            for i, raw in enumerate(lines, 1) if "304000" in raw]
    pre(len(hits) == 1,
        f"304000 对账断言失败:封闭台账集内出现 {len(hits)} 处(须恰为 1 处 "
        f"infra-b1 P8 leg_start): {[(n, i) for n, i, _ in hits]}")
    name, lineno, raw = hits[0]
    pre(name == "infra-b1", f"304000 唯一出现不在 infra-b1(实在 {name}:{lineno})")
    try:
        e = json.loads(raw)
    except json.JSONDecodeError:
        pre(False, f"304000 命中行不可解析:{name}:{lineno}")
    pre(e.get("event") == "leg_start" and e.get("leg") == "b1-p8"
        and e.get("seed") == 304_000,
        f"304000 唯一出现非 P8 leg_start 事件:{name}:{lineno} {e.get('event')}")


def assert_seed_virgin(lines_by_ledger: dict[str, list[str]], seed: int):
    """处女断言(全事件语义,与 304000 条款同语义):封闭台账集零出现。"""
    token = str(seed)
    hits = [(name, i) for name, lines in lines_by_ledger.items()
            for i, raw in enumerate(lines, 1) if token in raw]
    pre(not hits, f"种子 {seed} 处女断言失败(封闭台账集全事件扫描命中): {hits}")


def assert_pool_guard(seed: int, what: str):
    """B1 评测池核验肢原封(7000/8000/9000 段撞段守卫,rank 0-3)。"""
    pre(not any(lo <= seed + rank <= hi
                for rank in range(4)
                for lo, hi in ((7000, 7031), (8000, 8031), (9000, 9031))),
        f"{what} 种子 {seed} 撞评测池段(7000/8000/9000 守卫)")


# ======================================================================
# 评测机械(exam_or_adopt 骨架承继 + P2 台账制额度 + 捆绑通道)
# ======================================================================

def exam(worker, tag, seeds, manager_npz=None):
    out = EVAL / f"{tag}.json"
    require(not out.exists(), f"档案不可变性:{out} 已存在,拒绝覆写")
    lo, hi = (int(x) for x in seeds.split("-", 1))
    seed_values = list(range(lo, hi + 1))
    snapshot = freeze_eval_identity(ROOT, worker, manager_npz)
    assert_case_runtime(snapshot, f"exam:{tag}")          # W11 每发身份重申
    expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
    worker_arg = (worker if snapshot["worker"]["kind"] in {"script", "bc"}
                  else snapshot["worker"]["path"])
    # 纪律:--manager-npz 逐次显式,禁默认回落;本案一切发无 --board
    cmd = [PY, "train/eval_assembled.py", "--worker", str(worker_arg),
           "--manager-npz", snapshot["manager"]["path"],
           "--seeds", seeds, "--tag", tag]
    if run(cmd, f"exam-{tag}.{time.time_ns()}.log", timeout=1_800) != 0:
        if out.exists():
            sealed = out.with_suffix(f".{time.time_ns()}.void")
            out.rename(sealed)
            log({"event": "RESIDUE_SEALED", "tag": tag, "void": sealed.name,
                 "why": "评测进程非零退出之残档封存(P2)"})
        return None
    try:
        d = read_eval_archive(out, **expected)
        verify_eval_identity(snapshot, ROOT)
    except (OSError, KeyError, TypeError, ValueError):
        if out.exists():
            sealed = out.with_suffix(f".{time.time_ns()}.void")
            out.rename(sealed)
            log({"event": "RESIDUE_SEALED", "tag": tag, "void": sealed.name,
                 "why": "档案身份/schema 验证失败之残档封存(P2)"})
        return None
    d["agg"]["_sha"] = sha16(out)
    return d


def validate_adopted(tag, worker, seeds, manager_npz=None):
    out = EVAL / f"{tag}.json"
    lo, hi = (int(x) for x in seeds.split("-", 1))
    snapshot = freeze_eval_identity(ROOT, worker, manager_npz)
    assert_case_runtime(snapshot, f"adopt:{tag}")
    expected = expected_eval_identity(snapshot, tag=tag,
                                      seeds=list(range(lo, hi + 1)))
    d = read_eval_archive(out, **expected)
    verify_eval_identity(snapshot, ROOT)
    d["agg"]["_sha"] = sha16(out)
    return d


def exam_case(events, worker, tag, seeds, manager_npz=None,
              bundled_with=None, extra: dict | None = None):
    """考发终局条款(v32 逐字,exam_ok/exam_adopted,禁 .void 重烧)+
    P2 台账制额度(每发 FIRING_START 计 2 次;耗尽 → P5 甲案停机)。"""
    out = EVAL / f"{tag}.json"
    prior = [e for e in events
             if e.get("event") == "exam_ok" and e.get("tag") == tag]
    if out.exists():
        require(bool(prior), f"{tag} 残档在位而台账无 exam_ok,停机呈报")
        d = validate_adopted(tag, worker, seeds, manager_npz)
        require(d["agg"]["_sha"] == prior[-1]["sha"],
                f"{tag} 档案与台账 exam_ok sha 失配")
        log({"event": "exam_adopted", "tag": tag, "sha": d["agg"]["_sha"]})
        return d
    require(not prior, f"{tag} 台账在册而档案缺失(REF_INVALID 型),停机呈报")
    while True:
        fired = firing_count(events, tag)
        require(fired < 2, f"{tag} 评测发额度耗尽(台账制 2 次)——P5 甲案停机")
        ev = {"event": "FIRING_START", "tag": tag, "attempt": fired + 1}
        log(ev)
        events.append(ev)
        d = exam(worker, tag, seeds, manager_npz)
        if d is not None:
            a = d["agg"]
            ok = {"event": "exam_ok", "tag": tag, "mean": a["ret_mean"],
                  "died": a["died"], "sha": a["_sha"]}
            if bundled_with:
                ok["bundled_with"] = bundled_with
            if extra:
                ok.update(extra)
            log(ok)
            events.append({"event": "exam_ok", "tag": tag, "sha": a["_sha"]})
            return d
        log({"event": "exam_crash", "tag": tag,
             "note": "评测失败,按 P2 额度重考(残档已 RESIDUE_SEALED)"})


def holdout_account(events, tag):
    """HOLDOUT_EXPOSURE 按发计(W-H8;本案 1 对/2 发,施于胜者——B1
    "每案至多一对"法之首个受法案)。"""
    if any(e.get("event") == "HOLDOUT_EXPOSURE" and e.get("tag") == tag
           for e in events):
        return
    n = sum(1 for e in events if e.get("event") == "HOLDOUT_EXPOSURE") + 1
    ev = {"event": "HOLDOUT_EXPOSURE", "tag": tag, "cumulative_shots": n,
          "note": "按发计;本案 1 对/2 发施于胜者(B1『每案至多一对』法首个受法案);"
                  "登记面处女性之限定承 B1 残余⑭"}
    log(ev)
    events.append(ev)


# ======================================================================
# REF_BITEQ(G0-6 冻结先决)与 P-refdiv
# ======================================================================

def biteq_diffs(new_doc, ref_key: str) -> dict:
    """REF_BITEQ 纯比对:新参照对 W-PIN 钉死之 v32 同名参照逐种子逐字段位级
    + agg 核心字段 + ret_mean 恒等(≡113.0 / ≡140.9 全表)。返回差异(空=过)。"""
    path, expected = W_PIN[f"v32-ref-{ref_key}"]
    require(path.is_file() and sha256(path) == expected,
            f"W-PIN 参照档 sha 漂移:v32-ref-{ref_key}")
    old = strict_json_loads(path.read_bytes())
    old_rows, new_rows = old["rows"], new_doc["rows"]
    require(len(old_rows) == len(new_rows) == 32, "参照行数异常")
    row_diff = [o["seed"] for o, n in zip(sorted(old_rows, key=lambda r: r["seed"]),
                                          sorted(new_rows, key=lambda r: r["seed"]))
                if o != n]
    core = ("ret_mean", "ret_median", "died", "depth_median", "kills_mean",
            "farm_tau_mean", "override_rate", "cap_rate")
    agg_diff = [k for k in core if old["agg"].get(k) != new_doc["agg"].get(k)]
    if new_doc["agg"]["ret_mean"] != REF_MEANS[ref_key]:
        agg_diff.append(f"ret_mean!={REF_MEANS[ref_key]}")
    return {"row_diff_seeds": row_diff, "agg_diff": agg_diff} \
        if (row_diff or agg_diff) else {}


def p_refdiv(events, ref_key: str, diffs: dict):
    """P-refdiv:漂移分诊(CASE_RUNTIME 复验)→ CASE_HALT_G0(7)或
    REF_DIVERGENCE(8);失配发生于冻结前,禁降级,禁"新读数也合理"。"""
    snapshot = freeze_eval_identity(ROOT, str(KING_NPZ), None)
    if CASE_RT is not None and runtime_five(snapshot) != CASE_RT:
        log({"event": "CASE_HALT_G0", "via": "REF_BITEQ+runtime-drift",
             "ref": ref_key, **diffs, "current": runtime_five(snapshot)})
        attention(f"REF_BITEQ 失配({ref_key})且运行时漂移:CASE_HALT_G0")
        raise SystemExit(7)
    log({"event": "REF_DIVERGENCE", "ref": ref_key, **diffs,
         "triage": "runtime_five 全绿而全表失配——停机呈报(P-refdiv,禁降级)"})
    attention(f"REF_BITEQ 失配({ref_key}):REF_DIVERGENCE 停机呈报")
    raise SystemExit(8)


# ======================================================================
# 预检(W 线)
# ======================================================================

def preflight(events, smoke: bool = False):
    """S0:W1/W7/W-PIN/事件行钉死/种子对账/W-H8/W9/CASE_RUNTIME。
    smoke=True 时(冻结前置,准许脏树实弹)跳过 W1/W9/W-H8。"""
    global CASE_RT
    pre(PROTOCOL_VERSION == 3, "契约版本漂移(评测协议束 != 3)")
    # 契约修订号与主表字面之跨件恒等(单一真源交叉钉)
    import train_ppo
    pre(train_ppo._CONTRACT_REVISION == 5, "train_ppo 契约修订号 != 5(E4)")
    pre(train_ppo._DRY_CURRICULUM_MAIN_TABLE == MAIN_TABLE,
        "退火主表字面与 train_ppo 单一真源失配")
    pre(train_ppo._BC_V1_DEMOS_SHA256 == BC_V1_DEMOS_SHA,
        "BC-v1 demos 冻结常量与 train_ppo(E6)失配")
    # ---- W-PIN 逐件 sha 对账(失配 → PREFLIGHT_FAIL 3) ----
    for name, (path, expected) in {**W_PIN, **W_PIN_LEDGERS}.items():
        pre(path.is_file() and sha256(path) == expected,
            f"W-PIN 档案钉死失配:{name}")
    # ---- 事件行钉死(档案 sha + 行号 + 行字节 sha + 事件名/line_id) ----
    ledger_lines = w_pin_ledger_lines()
    for name, (lkey, lineno, event_name, line_id, line_sha) in \
            W_PIN_EVENT_LINES.items():
        lines = ledger_lines[lkey]
        pre(len(lines) >= lineno, f"W-PIN 事件行缺失:{name}({lkey}:{lineno})")
        raw = lines[lineno - 1]
        pre(hashlib.sha256(raw.encode()).hexdigest() == line_sha,
            f"W-PIN 事件行字节漂移:{name}({lkey}:{lineno})")
        e = json.loads(raw)
        pre(e.get("event") == event_name,
            f"W-PIN 事件行事件名失配:{name} != {event_name}")
        if line_id is not None:
            pre(e.get("line_id") == line_id,
                f"W-PIN 事件行 line_id 失配:{name} != {line_id}")
    # G1 新线四常数对台账行实值恒等复核(引用件防抄错)
    g1 = {json.loads(ledger_lines["recal-g1"][i - 1]).get("line_id"):
          json.loads(ledger_lines["recal-g1"][i - 1]) for i in range(10, 15)}
    pre(round(float(g1["RG1.2a-mean-worker-leg"]["value"]), 2) == G1_PAIRED_DIFF,
        "G1 均差肢常量 != RG1.2a 台账实值")
    pre(int(g1["RG1.1-width"]["value"]) == G1_PAIRED_WINS,
        "G1 宽度肢常量 != RG1.1 台账实值")
    pre(int(g1["RG1.3-death"]["value"]) == G1_DIED_MAX,
        "G1 死亡肢常量 != RG1.3 台账实值")
    pre("85/92" in str(g1["RG1.4-floor"]["value"]),
        "G1 地板肢 != RG1.4 台账实值(维持现行 85/92)")
    # ---- W7 工件钉死 ----
    pre(KING_ZIP.is_file() and sha256(KING_ZIP) == KING_ZIP_SHA, "王 zip 漂移")
    pre(sha256(KING_NPZ) == KING_NPZ_SHA, "王 npz 漂移")
    pre(sha256(H_NPZ) == H_NPZ_SHA, "H npz 漂移(!= DEFAULT_MANAGER_SHA256)")
    pre(sha256(M29_NPZ) == M29_NPZ_SHA, "M29 npz 漂移")
    pre(zip_steps(KING_ZIP) == KING_STEPS, "王 zip 步数账异常")
    pre(KING_SD.is_file() and sha256(KING_SD) == KING_SD_SHA,
        "KING_SD 漂移(沿用 v32 件)")
    # ---- 304000 对账 + 308000 处女 + 评测池段守卫(D1/D2-2a) ----
    assert_seed_304000_provenance(ledger_lines)
    assert_seed_virgin(ledger_lines, SMOKE_SEED)
    assert_pool_guard(SEED, "训练")
    assert_pool_guard(SMOKE_SEED, "烟测")
    if smoke:
        log({"event": "smoke_preflight_ok", "impl_sha16": impl_bundle_sha16()})
        return None
    # ---- W1 冻结公证(smoke 免;正案要求树净 + 关键件已入库)----
    # 勘正(2026-07-19):原"最后触碰 == HEAD"系过严代理——树净已保证盘上
    # 内容 ≡ HEAD 公证态,无关件的后续提交不构成漂移;FREEZE_SHA 落账值
    # 恒取实际 HEAD,公证语义完整。此处只须关键件确已入库。
    dirty = [l for l in git("status", "--porcelain").splitlines()
             if l != "?? train/leaderboard-assembled-v3.md"]
    pre(not dirty, f"W1: 工作树不净 {dirty}")
    head = git("rev-parse", "HEAD")
    for path in ("docs/PREREG-内容案-课⑤x④乙.md", "train/run_v33_content.py"):
        touch = git("log", "-1", "--format=%H", "--", path)
        pre(bool(touch), f"W1: {path} 未入库(无公证 commit)")
    freezes = [e for e in events if e.get("event") == "FREEZE_SHA"]
    if freezes and freezes[-1]["sha"] != head:
        # P6 链式重冻结:REFREEZE_REASON 文件在位方可续链
        reason_file = V33 / "REFREEZE_REASON"
        pre(reason_file.is_file(),
            "W1: HEAD != 台账最后 FREEZE_SHA(P6 链式重冻结须 REFREEZE_REASON)")
        ev = {"event": "FREEZE_SHA", "sha": head,
              "prev_sha": freezes[-1]["sha"],
              "reason": reason_file.read_text().strip()}
        log(ev)
        events.append(ev)
    # ---- 案内 KING_SD parity 复验(W7) ----
    pre(run([PY, "train/check_teacher_parity.py", str(KING_SD),
             str(KING_NPZ)], "parity-king.log", 600) == 0,
        "G-KL-W:王锚 sd 与 npz 宣誓失败(案内 0/1000 复验)")
    # ---- W-H8 留出池纪律(登记面) ----
    holdout_virgin_scan(events)
    # ---- W9 目标档案先决 ----
    for t in ALL_EXAM_TAGS:
        has_ledger = any(e.get("event") in ("exam_ok", "CANARY_EVAL")
                         and e.get("tag") == t for e in events)
        if not has_ledger and t not in CANARY_TAGS:
            pre(not (EVAL / f"{t}.json").exists(),
                f"W9: 目标档案已存在:{t}(重启协议:先 .void)")
    for leg in LEG_NAMES:
        if leg_starts(events, leg) == 0:
            pre(not (RUNS / leg).exists(), f"运行目录残留:{leg}")
    # ---- CASE_RUNTIME 捕获与 W10 续跑对账(注:B1 型 W-E0 位级断言不可用——
    # 本案 E1 改 worker_env.py,protocol 束 sha 对 v32/B1 落定值必变,禁以
    # "sha 相同"话术呈报;零漂移替代证明 = REF_BITEQ 两发 + G0-1 双端点) ----
    snapshot = freeze_eval_identity(ROOT, str(KING_NPZ), None)
    CASE_RT = runtime_five(snapshot)
    prior_rt = [e for e in events if e.get("event") == "CASE_RUNTIME"]
    if prior_rt and prior_rt[0]["five"] != CASE_RT:
        log({"event": "CASE_HALT_ENV_DRIFT", "where": "preflight-W10",
             "expected": prior_rt[0]["five"], "current": CASE_RT})
        attention("W5/W10 发车前运行时对账失配,停机呈报")
        raise SystemExit(5)
    # ---- W-C 金丝雀记分集承继(自 B1 CANARY_SET 事件行钉死,不重提取) ----
    if not stage_done(events, "CANARY_SET"):
        b1_line = json.loads(
            ledger_lines["infra-b1"][W_PIN_EVENT_LINES["B1-CANARY_SET"][1] - 1])
        ev = {"event": "CANARY_SET", "inherited_from": "infra-b1:19(行 sha 钉死)",
              "n_D": b1_line["n_D"], "depth2_seeds": b1_line["depth2_seeds"],
              "controls": b1_line["controls"], "C": b1_line["C"],
              "note": "C 集与 n_D 承继钉死(W-C,不重提取);记分种子集仅约束 "
                      "R 线读数,不约束考发面(金丝雀考发全池 7000-7031)"}
        log(ev)
        events.append(ev)
    log({"event": "preflight_ok", "king_zip": KING_ZIP_SHA[:16],
         "king_sd": KING_SD_SHA[:16], "impl_sha16": impl_bundle_sha16(),
         "seed": SEED, "smoke_seed": SMOKE_SEED})
    return None


def holdout_virgin_scan(events):
    """W-H8 留出池纪律(登记面;B1 逐字承继,豁免集扩至 B1 四发 + 本案一对)。"""
    sanctioned = {f"{t}.json" for t in
                  ("b1-ref8k-launch", "b1-ref8k-science",
                   "p8-hold8k", "p8-hold8k-m29") + HOLDOUT_TAGS}
    pattern = re.compile(r"\b80(?:[0-2][0-9]|3[01])\b")
    offenders = []
    for arch in sorted(EVAL.glob("*.json")):
        if arch.name in sanctioned:
            continue
        try:
            doc = json.loads(arch.read_text())
            seeds = doc.get("meta", {}).get("seeds", [])
        except (json.JSONDecodeError, UnicodeDecodeError):
            offenders.append(f"{arch.name}: 不可解析")
            continue
        if any(8000 <= int(s) <= 8031 for s in seeds):
            offenders.append(arch.name)
    for board in sorted((ROOT / "train").glob("leaderboard*.md")):
        if pattern.search(board.read_text()):
            offenders.append(board.name)
    pre(not offenders, f"W-H8 留出池登记面断言失败,停机呈报: {offenders}")


# ======================================================================
# S1 G0_BASELINE 转录(E0 四件;捕获在驱动器之外、冻结 commit 之前)
# ======================================================================

def g0_baseline_stage(events):
    for key in ("E0-king-p0", "E0-king-p1", "E0-throne-p0", "E0-throne-p1"):
        path, expected = W_PIN[key]
        pre(path.is_file() and sha256(path) == expected,
            f"E0 双基线档钉死失配:{key}")
    if stage_done(events, "G0_BASELINE"):
        return
    ev = {"event": "G0_BASELINE",
          "baselines": {k: W_PIN[k][1] for k in
                        ("E0-king-p0", "E0-king-p1",
                         "E0-throne-p0", "E0-throne-p1")},
          "captured": "变更前(E0 施工前,圈外 stash 纯净码经 probe_g0_traj rig "
                      "捕获;king+throne × 双端点;驱动器外执行,驱动器只做 sha "
                      "常量核验 + 本转录)",
          "note": "G0-1 之『变更前』证明负担由 E0 阶段独立承担,禁新码对新码自证"}
    log(ev)
    events.append(ev)


# ======================================================================
# S2 S-bc1:BC-v1 重生成分列断言(W7 分列式)→ BC_REGEN
# ======================================================================

def _torch_state_equal(a_path: pathlib.Path, b_path: pathlib.Path) -> bool:
    import torch
    a = torch.load(a_path, map_location="cpu", weights_only=True)
    b = torch.load(b_path, map_location="cpu", weights_only=True)
    if set(a) != set(b):
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def bc1_stage(events):
    assert_stage_prereqs(events, "BC_REGEN")
    if stage_done(events, "BC_REGEN"):
        prior = [e for e in events if e.get("event") == "BC_REGEN"][-1]
        require(BC_SD.is_file() and sha256(BC_SD) == prior["policy_sha256"],
                "BC-v1 policy 跨发车身份链断裂(!= BC_REGEN 落定值)")
        require(sha256(BC_V1_DEMOS) == BC_V1_DEMOS_SHA,
                "BC-v1 demos 字节 != 冻结常量(跨发车身份链断裂)")
        return
    require(run([PY, "train/bc_worker.py"],
                f"bc-regen-v1.{time.time_ns()}.log", 3_600) == 0,
            "BC-v1 重生成失败(R-W FAIL 型,全案停机)")
    # 分列断言①:demos.npz 字节 sha ≡ 冻结常量(np.savez_compressed 字节确定)
    demos_sha = sha256(BC_V1_DEMOS)
    require(demos_sha == BC_V1_DEMOS_SHA,
            f"CASE_HALT:BC-v1 demos 字节 {demos_sha[:16]} != 冻结常量 "
            f"{BC_V1_DEMOS_SHA[:16]}(W7 分列断言①)")
    # 分列断言②:policy_sd.pt 对 _previous 归档旧件张量级 torch.equal
    # (torch.save 字节不可复现——字节等断言原理性不可过,废止)
    prev_dirs = sorted((BC_V1_DIR / "_previous").iterdir(),
                       key=lambda p: int(p.name)) \
        if (BC_V1_DIR / "_previous").is_dir() else []
    require(bool(prev_dirs), "CASE_HALT:BC-v1 _previous 归档缺失,无对照件")
    prev_policy = prev_dirs[-1] / "policy_sd.pt"
    require(prev_policy.is_file(), "CASE_HALT:_previous 归档内无 policy_sd.pt")
    require(_torch_state_equal(BC_SD, prev_policy),
            "CASE_HALT:BC-v1 policy 对 _previous 归档件 torch.equal 失败"
            "(W7 分列断言②)")
    import train_ppo
    rec = train_ppo._validate_bc_report(BC_SD, "data_gate", verify_replay=False)
    ev = {"event": "BC_REGEN", "held_out_top1": rec["held_out_top1"],
          "pairs": rec["pairs"], "policy_sha256": sha256(BC_SD),
          "demos_sha256": demos_sha,
          "prev_archive": prev_dirs[-1].name,
          "prev_policy_torch_equal": True,
          "prev_lineage_sha256": BC_SD_PREV_LINEAGE_SHA,
          "note": "新 policy sha 以本事件落账为新落定值;f052067a… 系旧落定值"
                  "仅作 torch.equal 对照件谱系引用;provenance 字段刷新合法"
                  "(承 v32 _previous 快照先例)"}
    log(ev)
    events.append(ev)


# ======================================================================
# S3 S-bc2:BC-v2 采集+训练 → N12_GATE(P-N12:唯一 OC 0.65→0.70 重采一次)
# ======================================================================

def _n12_event(events, verdict: str, threshold: float, oc_consumed: bool,
               report: dict | None):
    ev = {"event": "N12_GATE", "gate": verdict,
          "preventive_threshold": threshold, "oc_consumed": oc_consumed,
          "n12_gate_min": N12_GATE_MIN, "recall_12_gate_min": RECALL12_GATE_MIN,
          "denominator_definition": N12_DENOMINATOR_DEF,
          "cluster_note": N12_CLUSTER_NOTE}
    if report is not None:
        ev.update({"n12": report.get("n12"),
                   "n12_by_episode": report.get("n12_by_episode"),
                   "recall_12": report.get("recall_12"),
                   "recall_12_denominator": report.get("recall_12_denominator"),
                   "held_out_top1": report.get("held_out_top1"),
                   "pairs": report.get("pairs"),
                   "demos_sha256": report.get("demos_sha256"),
                   "policy_sha256": report.get("policy_sha256")})
    log(ev)
    events.append(ev)
    return ev


def _read_v2_report_if_any() -> dict | None:
    if not BC_V2_REPORT.is_file():
        return None
    try:
        return strict_json_loads(BC_V2_REPORT.read_bytes())
    except Exception:
        return None


def n12_gate_demos_sha(gate_event: dict) -> str:
    """N12_GATE 落定值取键(rev4 十二附二④:vacuous 兜底废止改硬失败——
    台账 N12_GATE 行缺 demos_sha256 键即身份链无锚,require 失败)。"""
    gate_sha = gate_event.get("demos_sha256")
    require(isinstance(gate_sha, str) and len(gate_sha) == 64,
            "N12_GATE 落账行缺 demos_sha256 键(身份链无锚,硬失败;"
            "rev4 十二附二④)")
    return gate_sha


def assert_lfull_demos_chain(events):
    """修①c(rev4 十二附二④):L-full 腿点火前断言 BC_V2_DEMOS 实测字节
    sha256 ≡ N12_GATE 落定值(跨发车身份链,--bc-aux-demos 之消费面)。"""
    passed = [e for e in events
              if e.get("event") == "N12_GATE" and e.get("gate") == "PASS"]
    require(bool(passed), "L-full 点火前置:N12_GATE PASS 不在册")
    gate_sha = n12_gate_demos_sha(passed[-1])
    require(BC_V2_DEMOS.is_file(),
            f"L-full 点火前置:BC-v2 demos 缺失:{BC_V2_DEMOS}")
    actual = sha256(BC_V2_DEMOS)
    require(actual == gate_sha,
            f"L-full 点火前置:BC-v2 demos 字节 {actual[:16]} != N12_GATE "
            f"落定值 {gate_sha[:16]}(跨发车身份链断裂)")


def bc2_stage(events):
    assert_stage_prereqs(events, "N12_GATE")
    passed = [e for e in events
              if e.get("event") == "N12_GATE" and e.get("gate") == "PASS"]
    if passed:
        import bc_worker
        rec = bc_worker._validate_bc_v2_report(
            BC_V2_SD, expected_manager_sha256=sha256(M29_NPZ))
        gate_sha = n12_gate_demos_sha(passed[-1])
        require(rec["demos_sha256"] == gate_sha,
                "BC-v2 demos 跨发车身份链断裂(!= N12_GATE 落定值)")
        return
    def _failed_at(threshold: float) -> bool:
        return any(e.get("event") == "N12_GATE" and e.get("gate") == "FAIL"
                   and e.get("preventive_threshold") == threshold
                   for e in events)

    if _failed_at(PREVENTIVE_OC):
        # 唯一 OC 已烧且 FAIL 在册:预案穷尽,禁再采(重启不得续命)
        attention("BC-v2 N12 闸:唯一 OC(0.70)已消耗且 FAIL 在册——预案穷尽,"
                  "不冻结呈报(P-N12);类加权须另行亲批,待批期间不得代行")
        print("P-N12:预案穷尽(台账在册),停机待裁(非失败态)", flush=True)
        raise SystemExit(0)
    oc_path = stage_done(events, "OC_CONSUMED") or _failed_at(PREVENTIVE_MAIN)
    attempts = ([(PREVENTIVE_OC, True)] if oc_path
                else [(PREVENTIVE_MAIN, False), (PREVENTIVE_OC, True)])
    for threshold, is_oc in attempts:
        if is_oc and not stage_done(events, "OC_CONSUMED"):
            # OC 先行消耗时序条款(核认新增):收束规则待追认(未决 C)注记强制
            ev = {"event": "OC_CONSUMED", "knob": "preventive_threshold",
                  "from": PREVENTIVE_MAIN, "to": PREVENTIVE_OC,
                  "note": "收束规则待追认(未决 C):类加权自动预案已除名、须另行"
                          "亲批,待批期间不得代行;发车审阅 2(f) 若拒绝追认收束,"
                          "已冻结案自动转 P6 重冻结待裁——先行消耗不构成既成事实"}
            log(ev)
            events.append(ev)
        cmd = [PY, "train/bc_worker.py", "--v2",
               "--manager-npz", str(M29_NPZ)]
        if is_oc:
            cmd += ["--preventive-threshold", str(PREVENTIVE_OC)]
        rc = run(cmd, f"bc-v2-{threshold}.{time.time_ns()}.log", 7_200)
        report = _read_v2_report_if_any()
        if rc == 0:
            import bc_worker
            rec = bc_worker._validate_bc_v2_report(
                BC_V2_SD, expected_manager_sha256=sha256(M29_NPZ))
            _n12_event(events, "PASS", threshold, is_oc, rec)
            return
        require(report is not None and report.get("data_gate") == "FAIL",
                f"BC-v2 采集/训练运维失败(rc={rc} 且无 FAIL 回执),停机呈报")
        _n12_event(events, "FAIL", threshold, is_oc, report)
    # 预案穷尽:不冻结呈报(P-N12 非失败态,NEEDS_ATTENTION 停机待裁)
    attention("BC-v2 N12/recall 闸预案穷尽(主案 0.65 + 唯一 OC 0.70 各一次):"
              "不冻结呈报,禁带闸伤发车;类加权须另行亲批,待批期间不得代行(P-N12)")
    print("P-N12:预案穷尽,不冻结呈报——停机待裁(非失败态)", flush=True)
    raise SystemExit(0)


# ======================================================================
# S4 G0 六件套
# ======================================================================

def g0_endpoint_stage(events):
    """G0-1 双端点位级恒等:p≡0.0 ≡ skip_dry=False / p≡1.0 ≡ skip_dry=True,
    对 E0 施工前双基线档(king+throne)逐种子逐拍位级对账(重放探针自带
    工资恒等式 W≡R−bonus 逐窗断言)。任一失败 → CASE_HALT_G0(7)。"""
    assert_stage_prereqs(events, "G0_ENDPOINT")
    if stage_done(events, "G0_ENDPOINT"):
        return
    replays = (
        ("king-p0", [PY, "train/probe_g0_traj.py", "replay"]),
        ("throne-p0", [PY, "train/probe_g0_traj.py", "replay", "throne"]),
        ("king-p1", [PY, "train/probe_g0_traj_skip_dry.py", "replay"]),
        ("throne-p1", [PY, "train/probe_g0_traj_skip_dry.py", "replay", "throne"]),
    )
    for name, cmd in replays:
        rc = run(cmd, f"g0-endpoint-{name}.{time.time_ns()}.log", 1_800)
        if rc != 0:
            log({"event": "CASE_HALT_G0", "via": "G0_ENDPOINT", "which": name,
                 "rc": rc})
            attention(f"G0-1 双端点位级恒等失败({name}),不冻结不发车")
            raise SystemExit(7)
    ev = {"event": "G0_ENDPOINT", "verdict": "PASS",
          "endpoints": {"p0": "skip_dry=False ≡ p≡0.0", "p1": "skip_dry=True ≡ p≡1.0"},
          "workers": ["king", "throne"], "seeds": "7000-7015",
          "baselines": {k: W_PIN[k][1][:16] for k in
                        ("E0-king-p0", "E0-king-p1",
                         "E0-throne-p0", "E0-throne-p1")},
          "bridge_note": "桥证降级条款(D2-1 注册):锚桥任一桥证失效 → 金评只出数"
                         "不对表、名分判词悬置呈报;D4-1 名分流转以本件过闸为先决,"
                         "旧桥证不自动承继"}
    log(ev)
    events.append(ev)


def _leg_shared_cmd(run_name: str, seed: int, total_steps: int,
                    calib_probes: tuple[int, ...],
                    with_b1_knobs: bool = True,
                    calib_record_only: bool = False) -> list[str]:
    """共用命令形(D3 逐字:v32-sov 逐字 + B1 仪表旋钮 + rev4 勘正增列之
    E5①② record-only 旋钮末两枚(十二附二①);主权开系默认,
    无 --no-drink-sovereignty;共用形自身不含 --skip-dry;
    with_b1_knobs=False 系 G0-2a 裸臂专用 = v32-sov 逐字,不携任何新旋钮)。"""
    cmd = [PY, "train/train_ppo.py", "--worker", "--algo", "mppo",
           "--gamma", "1.0", "--max-steps", "3000", "--n-steps", "512",
           "--num-envs", "4", "--lr", "3e-4", "--ent-coef", "0.005",
           "--seed", str(seed), "--total-steps", str(total_steps),
           "--run-name", run_name, "--distill-beta", str(BETA),
           "--teacher-sd", str(BC_SD), "--teacher-override", str(KING_SD),
           "--manager-npz", str(M29_NPZ),
           "--resume-from", str(KING_ZIP), "--allow-legacy-resume",
           "--calib-probes", ",".join(str(x) for x in calib_probes)]
    if calib_record_only:
        cmd.append("--calib-record-only")
    if with_b1_knobs:
        cmd += ["--ckpt-every-steps", str(CKPT_EVERY),
                "--sentinel-every", str(SENTINEL_EVERY),
                "--dry-anchor-every", str(DRY_ANCHOR_EVERY),
                "--distill-ce-probe-every", str(DISTILL_CE_PROBE_EVERY),
                "--drywin-metrics-every", str(DRYWIN_METRICS_EVERY)]
    return cmd


def leg_cmd(leg: str) -> list[str]:
    """三腿完整 CLI = 共用命令形 + D3 逐腿附加项逐字(LEGS 表权威)。"""
    extras = dict(LEGS)[leg]
    return _leg_shared_cmd(leg, SEED, LEG_STEPS, CALIB_PROBES) + list(extras)


def _smoke_cmd(variant: str) -> list[str]:
    """G0-2a/2b 烟测命令(种子 308000,102,400 步 = 50 rollout)。
    bare = v32-sov 命令形逐字含 --skip-dry(calib 步表 3747984,3947984 逐字,
    窗外不燃,B1 仪表旋钮不携——裸臂 ≡ HEAD 行为面);
    knobs = 全套新旋钮在位:调度钉 p≡1.0(hold:1.0:50)+ λ_bc=0(bc_aux 旗
    在位、谓词不活)+ E5①② 探针 + B1 仪表(E5①② 与 B1 旋钮自 rev4 起均随
    共用形 with_b1_knobs=True 注入;calib 步表换窗内两点,record-only
    零侵入 B1 已证;施工裁量注记);
    func-p / func-aux = G0-2b 功能短跑(主表整表,腿相对前 50 rollout 前缀)。"""
    if variant == "bare":
        return (_leg_shared_cmd(SMOKE_RUNS["bare"], SMOKE_SEED, SMOKE_STEPS,
                                V32_SOV_CALIB, with_b1_knobs=False,
                                calib_record_only=True)
                + ["--skip-dry"])
    if variant == "knobs":
        return (_leg_shared_cmd(SMOKE_RUNS["knobs"], SMOKE_SEED, SMOKE_STEPS,
                                SMOKE_CALIB, calib_record_only=True)
                + ["--dry-curriculum-schedule", SMOKE_HOLD_TABLE,
                   "--bc-aux-lambda", "0.0",
                   "--bc-aux-demos", str(BC_V2_DEMOS)])
    if variant == "func-p":
        return (_leg_shared_cmd(SMOKE_RUNS["func-p"], SMOKE_SEED, SMOKE_STEPS,
                                SMOKE_CALIB)
                + ["--dry-curriculum-schedule", MAIN_TABLE])
    if variant == "func-aux":
        return (_leg_shared_cmd(SMOKE_RUNS["func-aux"], SMOKE_SEED, SMOKE_STEPS,
                                SMOKE_CALIB)
                + ["--dry-curriculum-schedule", MAIN_TABLE,
                   "--bc-aux-lambda", str(BC_AUX_LAMBDA),
                   "--bc-aux-demos", str(BC_V2_DEMOS)])
    raise ValueError(f"未知烟测臂: {variant}")


def _load_zip_states(path: pathlib.Path):
    import torch
    with zipfile.ZipFile(path) as z:
        policy = torch.load(io.BytesIO(z.read("policy.pth")),
                            map_location="cpu", weights_only=True)
        optim = torch.load(io.BytesIO(z.read("policy.optimizer.pth")),
                           map_location="cpu", weights_only=True)
    return policy, optim


def _tree_equal(a, b, path, diffs):
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.shape == b.shape and a.dtype == b.dtype
                and torch.equal(a, b)):
            diffs.append(path)
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append(f"{path}:keys")
            return
        for k in sorted(a, key=str):
            _tree_equal(a[k], b[k], f"{path}.{k}", diffs)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}:len")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _tree_equal(x, y, f"{path}[{i}]", diffs)
        return
    if a != b:
        diffs.append(path)


def _progress_lines(run_dir: pathlib.Path) -> list[dict]:
    lines = []
    for raw in (run_dir / "progress.jsonl").read_text().splitlines():
        rec = json.loads(raw)
        rec.pop("t", None)                    # 墙钟字段非 RNG 相关,剔除
        lines.append(rec)
    return lines


def sentinel_line_counts(sentinel_path: pathlib.Path) -> dict[str, int]:
    """修②(rev4 十二附二⑤):sentinel.jsonl 双行型分计——v23 哨兵行与
    dry-anchor 干层锚行同文件(train_ppo 两回调同名落盘),分肢计数供
    G0-2a『仪表确实跑了』四件全查(RC.12 消费面 :2147 同款字面判别)。"""
    rows = (sentinel_path.read_text().splitlines()
            if sentinel_path.is_file() else [])
    return {"sentinel": sum(1 for x in rows if '"sentinel": "v23"' in x),
            "dry_anchor": sum(1 for x in rows if '"dry-anchor"' in x)}


def instrument_jsonl_digest(path: pathlib.Path) -> dict:
    """修⑤b(rev4 十二附二①/D8):仪表 jsonl 之腿终转录载荷——文件字节
    sha256 + 行数 + final 段聚合载荷(final=True 行;无则如实取末行,
    fall_back_last_line 注记随行);空档/缺档 fail-loud。"""
    require(path.is_file(), f"仪表档缺失:{path}(E5 旋钮在位而无产物)")
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    require(bool(rows), f"仪表档为空:{path}(E5 旋钮在位而零行)")
    finals = [r for r in rows if r.get("final")]
    digest = {"sha256": sha256(path), "lines": len(rows),
              "final": finals[-1] if finals else rows[-1]}
    if not finals:
        digest["fall_back_last_line"] = True
    return digest


# 修④(rev4 十二附二③)episode 种子序列恒等之可见面等价物下界:公共前缀
# 累计 len 须 ≥ 单 env 单 rollout 步数(n-steps 512)——首 rollout 内两跑权重
# 逐位同一(λ_bc 仅经首次 train() 进权重),其完窗行必逐字相等;初始 episode
# 种子流被污染则首行即分歧。
SEED_EQUIV_MIN_PREFIX_STEPS = 512


def progress_common_prefix(lines_a: list[dict], lines_b: list[dict]) -> dict:
    """修④(rev4 十二附二③):episode 种子序列恒等断言之可见面最强等价物。

    实况呈报(禁静默弱化条款之如实执行):progress.jsonl worker 行仅
    ep/t/reward/len 四字段——episode 种子唯于 WorkerWindowEnv.reset info
    一现,不入行,**episode 种子字段序列在运行产物可见面不存在**(禁伪造)。
    等价物 = 两跑 (ep, reward, len) 序列首分歧点前公共前缀恒等:轨迹分歧
    仅能自首次 train()(λ_bc 消费处)之后发生,故公共前缀须非空且累计 len
    覆盖首 rollout 下界(SEED_EQUIV_MIN_PREFIX_STEPS);种子流若被 aux rng
    在位污染,分歧将现于首行。全长种子序列之直接证明面缺失入 limitations。"""
    n = 0
    for a, b in zip(lines_a, lines_b):
        if a != b:
            break
        n += 1
    diverged = n < min(len(lines_a), len(lines_b))
    return {"prefix_lines": n,
            "prefix_len_steps": sum(int(x.get("len", 0)) for x in lines_a[:n]),
            "first_divergence_index": (n if diverged else None),
            "lines": [len(lines_a), len(lines_b)],
            "field": "(ep, reward, len) 公共前缀(episode 种子字段不在可见面,"
                     "等价物口径)"}


def g0_nullintrusion_stage(events):
    """G0-2a 零侵入对(张量级):全套新旋钮在位(p≡1.0、λ_bc=0)vs 裸配方臂
    (v32-sov 命令形逐字含 --skip-dry),2×102,400 步;policy state_dict +
    optimizer state 逐张量 torch.equal + RNG 相关遥测逐字段;
    此件即 L-base ≡ v32-sov 配方等价之担保(E4)。FAIL → CASE_HALT_G0(7)。"""
    impl16 = impl_bundle_sha16()
    for e in events:
        if (e.get("event") == "G0_NULLINTRUSION" and e.get("verdict") == "PASS"
                and e.get("impl_sha16") == impl16):
            print(f"G0-2a 已过(impl {impl16}),幂等跳过", flush=True)
            return
    pre(BC_V2_DEMOS.is_file(), "G0-2a 前置:BC-v2 demos 缺失(先过 S-bc2)")
    results = {}
    for variant in ("bare", "knobs"):
        run_dir = RUNS / SMOKE_RUNS[variant]
        _archive_residue(run_dir)   # 勘正:残留归档,防 jsonl 追加堆积(承 B1 _prepare_run_dir;16:55:51 FAIL 在册)
        t0 = time.time()
        rc = run(_smoke_cmd(variant), f"smoke-{variant}.log", SMOKE_TIMEOUT)
        nt = zip_steps(run_dir / "model_final.zip")
        results[variant] = {"rc": rc, "nt": nt,
                            "dt_min": round((time.time() - t0) / 60, 1)}
        if rc != 0 or nt != SMOKE_END:
            log({"event": "G0_NULLINTRUSION", "verdict": "FAIL",
                 "impl_sha16": impl16, "runs": results,
                 "why": f"烟测 {variant} 未达标(rc={rc}, nt={nt}, "
                        f"目标={SMOKE_END})"})
            attention("G0-2a 烟测运行失败,不冻结不发车")
            raise SystemExit(7)
    knobs_dir = RUNS / SMOKE_RUNS["knobs"]
    bare_dir = RUNS / SMOKE_RUNS["bare"]
    # 仪表确实跑了("被证零侵入"不得实为"没跑";rev4 十二附二⑤ 四件全查:
    # sentinel/dry-anchor 两肢补铸——sentinel.jsonl 同文件双行型分计)
    knobs_sentinel = sentinel_line_counts(knobs_dir / "sentinel.jsonl")
    evidence = {
        "knobs_ckpt": sorted(p.name for p in (knobs_dir / "ckpt").glob("*.zip"))
        if (knobs_dir / "ckpt").is_dir() else [],
        "knobs_calib_lines": (len((knobs_dir / "calib.jsonl").read_text()
                                  .splitlines())
                              if (knobs_dir / "calib.jsonl").is_file() else 0),
        "knobs_sentinel_lines": knobs_sentinel["sentinel"],
        "knobs_dry_anchor_lines": knobs_sentinel["dry_anchor"],
        "knobs_distill_probe_lines": (
            len((knobs_dir / "distill_ce_probe.jsonl").read_text().splitlines())
            if (knobs_dir / "distill_ce_probe.jsonl").is_file() else 0),
        "knobs_drywin_lines": (
            len((knobs_dir / "drywin_metrics.jsonl").read_text().splitlines())
            if (knobs_dir / "drywin_metrics.jsonl").is_file() else 0),
        "knobs_curriculum_lines": (
            len((knobs_dir / "dry_curriculum.jsonl").read_text().splitlines())
            if (knobs_dir / "dry_curriculum.jsonl").is_file() else 0),
        "bare_calib_lines": (len((bare_dir / "calib.jsonl").read_text()
                                 .splitlines())
                             if (bare_dir / "calib.jsonl").is_file() else 0),
    }
    instrument_ok = (evidence["knobs_calib_lines"] >= 1
                     and evidence["knobs_sentinel_lines"] >= 1
                     and evidence["knobs_dry_anchor_lines"] >= 1
                     and evidence["knobs_distill_probe_lines"] >= 1
                     and evidence["knobs_drywin_lines"] >= 1
                     and evidence["knobs_curriculum_lines"] == SMOKE_STEPS // QUANTUM
                     and f"model_{KING_STEPS + CKPT_EVERY}_steps.zip"
                     in evidence["knobs_ckpt"])
    # knobs 臂调度钉 p≡1.0 恒等复核(hold 表 50 项全 1.0)
    hold_ok = True
    if (knobs_dir / "dry_curriculum.jsonl").is_file():
        pushed = [json.loads(x) for x in (knobs_dir / "dry_curriculum.jsonl")
                  .read_text().splitlines()]
        hold_ok = ([(p["rollout_index"], p["p"]) for p in pushed]
                   == [(i, 1.0) for i in range(SMOKE_STEPS // QUANTUM)])
    # 张量级判据
    pol_b, opt_b = _load_zip_states(bare_dir / "model_final.zip")
    pol_k, opt_k = _load_zip_states(knobs_dir / "model_final.zip")
    tensor_diffs: list[str] = []
    _tree_equal(pol_b, pol_k, "policy", tensor_diffs)
    _tree_equal(opt_b, opt_k, "optim", tensor_diffs)
    # RNG 相关遥测逐字段(B1 先例;墙钟剔除)
    telemetry_diffs: list[str] = []
    prog_b, prog_k = _progress_lines(bare_dir), _progress_lines(knobs_dir)
    if len(prog_b) != len(prog_k):
        telemetry_diffs.append(f"progress 行数 {len(prog_b)} != {len(prog_k)}")
    else:
        for i, (a, b) in enumerate(zip(prog_b, prog_k)):
            if a != b:
                telemetry_diffs.append(f"progress[{i}]: {a} != {b}")
                if len(telemetry_diffs) >= 20:
                    break
    verdict = ("PASS" if not tensor_diffs and not telemetry_diffs
               and instrument_ok and hold_ok else "FAIL")
    ev = {"event": "G0_NULLINTRUSION", "verdict": verdict,
          "impl_sha16": impl16, "seed": SMOKE_SEED, "steps": SMOKE_STEPS,
          "window": [KING_STEPS, SMOKE_END], "runs": results,
          "bare_arm": "v32-sov 命令形逐字含 --skip-dry(依赖钉死,承工程 B1)",
          "knobs_arm": "dry_curriculum(hold p≡1.0)+ bc_aux(λ=0 在位不活)+ "
                       "E5①②探针 + B1 仪表(calib 换窗内两点,record-only)",
          "tensor_verdict": ("逐张量 torch.equal 全等" if not tensor_diffs
                             else tensor_diffs[:20]),
          "telemetry_verdict": ("RNG 相关字段逐字段相等" if not telemetry_diffs
                                else telemetry_diffs[:20]),
          "instruments_actually_fired": instrument_ok,
          "knobs_hold_table_identity": hold_ok,
          "instrument_evidence": evidence,
          "criteria_note": "文件字节级比对废除(SB3 zip 时间戳/pickle 不可复现);"
                           "烟测遥测除位级判据外禁作任何 go/no-go 输入"}
    log(ev)
    events.append(ev)
    if verdict != "PASS":
        attention("G0-2a 零侵入失败:仪表回炉重设计,不冻结不发车")
        raise SystemExit(7)


def _read_curriculum_jsonl(run_dir: pathlib.Path) -> list[tuple[int, float]]:
    p = run_dir / "dry_curriculum.jsonl"
    require(p.is_file(), f"dry_curriculum.jsonl 缺失:{run_dir.name}")
    return [(int(rec["rollout_index"]), float(rec["p"]))
            for rec in (json.loads(x) for x in p.read_text().splitlines())]


def verify_curriculum_prefix(run_dir: pathlib.Path, table, n_rollouts: int
                             ) -> list[str]:
    """实测 p 序列 ≡ 注册表前缀断言(核认新增,堵 G0 三件皆盲之缝)。
    返回差异清单(空 = 恒等)。"""
    actual = _read_curriculum_jsonl(run_dir)
    expected = [(i, float(table[i])) for i in range(n_rollouts)]
    diffs = []
    if len(actual) != len(expected):
        diffs.append(f"行数 {len(actual)} != {len(expected)}")
    for (ai, ap), (ei, ep) in zip(actual, expected):
        if ai != ei or ap != ep:
            diffs.append(f"[{ei}] 实测 ({ai},{ap}) != 注册 ({ei},{ep})")
            if len(diffs) >= 10:
                break
    return diffs


def _bc_aux_grad12_report(zip_path: pathlib.Path, demos_npz: pathlib.Path) -> dict:
    """G0-2b：按 rev3 training-only 正/负校准目标复算 a12 头梯度。

    旧烟测只证明正例 CE 能给第 12 行梯度，恰好会把 90% 坍缩判 PASS。本件
    必须同时消费 hard negatives 与 KING 起点锚，且两组梯度均在同一目标中。
    """
    import numpy as np
    import torch as th
    import train_ppo

    with zipfile.ZipFile(zip_path) as z:
        sd = th.load(io.BytesIO(z.read("policy.pth")),
                     map_location="cpu", weights_only=True)
    anchor_sd, _ = _load_zip_states(KING_ZIP)
    x, y, episode_id, masks, _ = train_ppo._load_bc_aux_demos_v2(
        demos_npz, expected_manager_sha256=sha256(M29_NPZ))
    bx, by, bm = train_ppo._build_bc_aux_training_bank(
        x, y, episode_id, masks)
    positive = np.flatnonzero(by == 12)
    negative = np.flatnonzero(by != 12)
    require(len(positive) > 0 and len(negative) > 0,
            "G0-2b:rev3 training-only bank 须同时含正例/hard negatives")
    size = min(256, len(by))
    pos_n = min(len(positive), max(1, int(round(
        size * train_ppo._BC_AUX_POSITIVE_FRACTION))))
    neg_n = min(len(negative), max(1, size - pos_n))
    index = np.concatenate([positive[:pos_n], negative[:neg_n]])
    obs = th.tensor(bx[index], dtype=th.float32)
    mask_t = th.tensor(bm[index])
    h = th.tanh(obs @ sd["mlp_extractor.policy_net.0.weight"].T
                + sd["mlp_extractor.policy_net.0.bias"])
    h = th.tanh(h @ sd["mlp_extractor.policy_net.2.weight"].T
                + sd["mlp_extractor.policy_net.2.bias"])
    wa = sd["action_net.weight"].clone().requires_grad_(True)
    logits = h @ wa.T + sd["action_net.bias"]
    logp = th.log_softmax(
        th.where(mask_t, logits, th.full_like(logits, -1e8)), dim=-1)
    p12 = logp[:, 12].exp().clamp(1e-7, 1.0 - 1e-7)
    pos_loss = th.nn.functional.binary_cross_entropy(
        p12[:pos_n], th.full_like(
            p12[:pos_n], train_ppo._BC_AUX_POSITIVE_TARGET))
    neg_loss = th.nn.functional.binary_cross_entropy(
        p12[pos_n:], th.full_like(
            p12[pos_n:], train_ppo._BC_AUX_NEGATIVE_TARGET))
    anchor_logits = train_ppo._policy_logits_from_sb3_state_dict(
        anchor_sd, bx[index])
    anchor_logp = th.log_softmax(
        th.where(mask_t, anchor_logits,
                 th.full_like(anchor_logits, -1e8)), dim=-1)
    anchor_probs = anchor_logp.exp()[pos_n:]
    anchor_kl = (anchor_probs * (
        anchor_logp[pos_n:] - logp[pos_n:]
    )).sum(dim=-1).mean()
    objective = (
        train_ppo._BC_AUX_POSITIVE_FRACTION * pos_loss
        + (1.0 - train_ppo._BC_AUX_POSITIVE_FRACTION) * neg_loss
        + train_ppo._BC_AUX_ANCHOR_KL_COEF * anchor_kl)
    (g,) = th.autograd.grad(objective, [wa])
    gtotal = float(th.linalg.vector_norm(g))
    g12 = float(g[12].abs().sum())
    require(gtotal > 0.0 and g12 > 0.0,
            "G0-2b:rev2 total/a12 头梯度为零")
    return {
        "objective_revision": train_ppo._BC_AUX_OBJECTIVE_REVISION,
        "n_pairs_12": int(len(positive)),
        "n_hard_negatives": int(len(negative)),
        "batch_positive": int(pos_n),
        "batch_negative": int(neg_n),
        "objective": float(objective.detach()),
        "positive_bce": float(pos_loss.detach()),
        "negative_bce": float(neg_loss.detach()),
        "anchor_kl": float(anchor_kl.detach()),
        "grad_total_l2": gtotal,
        "grad12_abs_sum": g12,
        "all_m12_true": bool(bm[:, 12].all()),
    }


def _bc_aux_behavior_report(zip_path: pathlib.Path,
                            demos_npz: pathlib.Path) -> dict:
    """现有/新 smoke 通用的 E5 held-out 行为硬门（纯离线、真实 masks）。"""
    import train_ppo

    sd, _ = _load_zip_states(zip_path)
    anchor_sd, _ = _load_zip_states(KING_ZIP)
    x, y, episode_id, masks, _ = train_ppo._load_bc_aux_demos_v2(
        demos_npz, expected_manager_sha256=sha256(M29_NPZ))
    metrics = train_ppo.bc_aux_behavior_metrics(
        sd, x, y, episode_id, masks, anchor_sd=anchor_sd,
        heldout_only=True)
    return {"metrics": metrics,
            "gate": train_ppo.bc_aux_behavior_gate(
                metrics, require_root_anchor=True)}


def g0_funcsmoke_stage(events):
    """G0-2b 功能烟测(不携相等断言):p<1 短跑(主表前 50 rollout 前缀)+
    λ_bc>0 短跑;实测 p 序列 ≡ 注册表前缀、干窗入学习分布实发、辅助 CE 实消费
    + 12 头梯度非零、12 类示范对 m[12]=True 全量。FAIL → CASE_HALT_G0(7)。
    施工裁量呈报:『训练 RNG 状态轨迹逐点相等』之全轨迹证明超出运行产物可见面,
    以两短跑 dry_curriculum.jsonl 恒等(p 抽签流不受 λ 在位扰动)+ 前缀恒等
    近似关缝,limitations 位如实入册。"""
    impl16 = impl_bundle_sha16()
    for e in events:
        if (e.get("event") == "G0_SMOKE" and e.get("verdict") == "PASS"
                and e.get("impl_sha16") == impl16):
            print(f"G0-2b 已过(impl {impl16}),幂等跳过", flush=True)
            return
    pre(BC_V2_DEMOS.is_file(), "G0-2b 前置:BC-v2 demos 缺失(先过 S-bc2)")
    table = dry_curriculum_table()
    n_roll = SMOKE_STEPS // QUANTUM
    results = {}
    for variant in ("func-p", "func-aux"):
        run_dir = RUNS / SMOKE_RUNS[variant]
        _archive_residue(run_dir)   # 勘正:同 G0-2a,残留归档
        rc = run(_smoke_cmd(variant), f"smoke-{variant}.log", SMOKE_TIMEOUT)
        nt = zip_steps(run_dir / "model_final.zip")
        results[variant] = {"rc": rc, "nt": nt}
        if rc != 0 or nt != SMOKE_END:
            log({"event": "G0_SMOKE", "verdict": "FAIL", "impl_sha16": impl16,
                 "runs": results, "why": f"{variant} 未达标(rc={rc}, nt={nt})"})
            attention("G0-2b 功能烟测运行失败,不冻结不发车")
            raise SystemExit(7)
    funcp_dir = RUNS / SMOKE_RUNS["func-p"]
    funcaux_dir = RUNS / SMOKE_RUNS["func-aux"]
    prefix_diffs = {v: verify_curriculum_prefix(RUNS / SMOKE_RUNS[v], table,
                                                n_roll)
                    for v in ("func-p", "func-aux")}
    p_stream_equal = (_read_curriculum_jsonl(funcp_dir)
                      == _read_curriculum_jsonl(funcaux_dir))
    # 修④(rev4 十二附二③):episode 种子序列恒等——可见面最强等价物断言
    # (progress.jsonl 无种子字段,口径与理由见 progress_common_prefix)
    seed_seq = progress_common_prefix(_progress_lines(funcp_dir),
                                      _progress_lines(funcaux_dir))
    seed_seq_ok = (seed_seq["prefix_lines"] >= 1
                   and seed_seq["prefix_len_steps"] >= SEED_EQUIV_MIN_PREFIX_STEPS)
    # 干窗入学习分布分支实发(E5② drywin 干组 n>0;鲜窗照常 n>0)
    dry_seen = fresh_seen = False
    drywin_path = funcp_dir / "drywin_metrics.jsonl"
    if drywin_path.is_file():
        for raw in drywin_path.read_text().splitlines():
            rec = json.loads(raw)
            windows = rec.get("windows", {})
            dry_n = (windows.get("dry") or {}).get("n", 0)
            fresh_n = (windows.get("fresh") or {}).get("n", 0)
            dry_seen = dry_seen or dry_n > 0
            fresh_seen = fresh_seen or fresh_n > 0
    # λ_bc 消费证据:挂载打印 + zip 持久化标量 + 离线 autograd 12 头梯度
    aux_log = sorted(V33.glob("smoke-func-aux.log"))
    mounted = bool(aux_log) and "[④乙]" in aux_log[-1].read_text()
    aux_lambda_in_zip = zip_data_field(funcaux_dir / "model_final.zip",
                                       "bc_aux_lambda")
    grad_report = _bc_aux_grad12_report(funcaux_dir / "model_final.zip",
                                        BC_V2_DEMOS)
    behavior_report = _bc_aux_behavior_report(
        funcaux_dir / "model_final.zip", BC_V2_DEMOS)
    verdict = ("PASS" if not prefix_diffs["func-p"]
               and not prefix_diffs["func-aux"] and p_stream_equal
               and seed_seq_ok
               and dry_seen and fresh_seen and mounted
               and aux_lambda_in_zip == BC_AUX_LAMBDA
               and behavior_report["gate"]["verdict"] == "PASS"
               else "FAIL")
    ev = {"event": "G0_SMOKE", "verdict": verdict, "impl_sha16": impl16,
          "seed": SMOKE_SEED, "steps": SMOKE_STEPS, "runs": results,
          "p_prefix_identity": {v: (d or "≡ 注册表前缀(恒等)")
                                for v, d in prefix_diffs.items()},
          "p_stream_equal_across_aux_onoff": p_stream_equal,
          "episode_seed_seq_identity": {**seed_seq, "ok": seed_seq_ok,
                                        "min_prefix_steps":
                                            SEED_EQUIV_MIN_PREFIX_STEPS},
          "dry_branch_entered_learning": dry_seen,
          "fresh_branch_present": fresh_seen,
          "bc_aux_mount_marker": mounted,
          "bc_aux_lambda_in_zip": aux_lambda_in_zip,
          "grad12": grad_report,
          "a12_behavior": behavior_report,
          "limitations": "episode 种子序列恒等肢已做(rev4 十二附二③),但系"
                         "可见面最强等价物口径:progress.jsonl 无 episode 种子"
                         "字段(worker 行仅 ep/t/reward/len),以两跑 (ep, "
                         "reward, len) 公共前缀恒等(≥首 rollout 覆盖下界)+ "
                         "p 抽签流恒等 + 派生函数单测族 + G0-2a torch.equal "
                         "兜底近似关缝——全长种子序列与『训练 RNG 状态轨迹逐点"
                         "相等』之直接证明面缺失如实在册(交接单单列呈报);"
                         "干窗跳过分支(脚本代跑)之直接产物证据不在 SB3 infos "
                         "流,以 p<1 与干组实发合取近似"}
    log(ev)
    events.append(ev)
    if verdict != "PASS":
        attention("G0-2b 功能烟测失败,不冻结不发车")
        raise SystemExit(7)


def g0_ghost_stage(events):
    """G0-幽灵(v12 病灶回归,v32 探针复用):belt=0 时 12 仍非法;_drain 兜底
    不因课程/辅助损失失效;死亡拍终止阶梯不变;死后不可饮。"""
    assert_stage_prereqs(events, "G0_GHOST")
    if stage_done(events, "G0_GHOST"):
        return
    rc = run([PY, "train/probe_g0_ghost.py"],
             f"g0-ghost.{time.time_ns()}.log", 1_800)
    if rc != 0:
        log({"event": "CASE_HALT_G0", "via": "G0_GHOST", "rc": rc})
        attention("G0-幽灵失败,不冻结不发车")
        raise SystemExit(7)
    ev = {"event": "G0_GHOST", "verdict": "PASS"}
    log(ev)
    events.append(ev)


def g0_demo_ledger_stage(events):
    """G0-示范池与世代分账(DEMO_LEDGER):v1 字节钉死 / v1 路径禁 (11,12)
    原封、v2 路径禁 11 允 12 / v2 掩码 on-manifold / schema 隔离互斥。"""
    assert_stage_prereqs(events, "DEMO_LEDGER")
    if stage_done(events, "DEMO_LEDGER"):
        return
    import bc_worker
    require(bc_worker.forbidden_actions_for_generation(1) == (11, 12),
            "世代禁采断言破缺:v1 须禁 (11,12) 原封")
    require(bc_worker.forbidden_actions_for_generation(2) == (11,),
            "世代禁采断言破缺:v2 须禁 11 允 12")
    require(sha256(BC_V1_DEMOS) == BC_V1_DEMOS_SHA,
            "DEMO_LEDGER:BC-v1 demos 字节 != 冻结常量")
    v2_rec = bc_worker._validate_bc_v2_report(
        BC_V2_SD, expected_manager_sha256=sha256(M29_NPZ))
    require(v2_rec["teacher_generation"] == 2, "v2 回执世代账异常")
    # v1 验证器对 v2 件 fail-loud(schema 隔离互斥,E2)
    import train_ppo
    v1_rejects_v2 = False
    try:
        train_ppo._validate_bc_report(BC_V2_SD, "data_gate", verify_replay=False)
    except Exception:
        v1_rejects_v2 = True
    require(v1_rejects_v2, "schema 隔离破缺:v1 验证器未拒 v2 件")
    import numpy as np
    d = np.load(BC_V2_DEMOS)
    require({"X", "Y", "episode_id", "masks"} <= set(d.files),
            "v2 demos schema 缺逐样本 masks(E2)")
    sel = d["Y"] == 12
    require(bool(sel.any()) and bool(d["masks"][sel][:, 12].all()),
            "v2 demos 12 类示范对 m[12] 断言失败")
    require(not d["masks"][:, 11].any(), "v2 demos m[11] 恒 False 断言失败")
    ev = {"event": "DEMO_LEDGER",
          "v1": {"demos_sha256": BC_V1_DEMOS_SHA,
                 "forbidden": [11, 12], "path": "canonical(train_ppo 四处一字不动)"},
          "v2": {"demos_sha256": v2_rec["demos_sha256"],
                 "policy_sha256": v2_rec["policy_sha256"],
                 "forbidden": [11], "n12": v2_rec["n12"],
                 "preventive_threshold": v2_rec["preventive_threshold"],
                 "masks": "逐样本 env.action_masks() 现场捕获(唯一 on-manifold "
                          "真源;反推口径系第二真源禁用)"},
          "schema_isolation": "v1/v2 验证器互斥实测(v1 拒 v2 件)",
          "note": "BC-v2 12 类样本全部系主权开环境真实执行拍(禁反事实标签;"
                  "overridden 整拍剔除断言原封,由 bc_worker 采集面承载)"}
    log(ev)
    events.append(ev)


def refs_stage(events):
    """G0-6 in-case refs 两发(REF_BITEQ):R1 ≡113.0 / R2 ≡140.9 全表逐种子
    逐字段;过闸系 FREEZE_SHA 先决——冻结待发车态不携未证漂移风险;失配走
    P-refdiv,禁降级。refs 系参照复核非本案候选考发,先于亲发执行合法
    (发车条款 2(g) 之如实登记项)。"""
    assert_stage_prereqs(events, "REF_BITEQ")
    refs = {}
    for ref_key, manager in (("launch", None), ("science", str(M29_NPZ))):
        tag = REF_TAGS[ref_key]
        d = exam_case(events, str(KING_NPZ), tag, POOL_PROBE,
                      manager_npz=manager,
                      extra={"note": "G0-6 REF_BITEQ 参照复核发(冻结先决,"
                                     "先于 FREEZE_SHA 与亲发;设计文书第九节)"})
        diffs = biteq_diffs(d, ref_key)
        if diffs:
            p_refdiv(events, ref_key, diffs)
        if not any(e.get("event") == "REF_BITEQ" and e.get("ref") == ref_key
                   for e in events):
            ev = {"event": "REF_BITEQ", "ref": ref_key, "tag": tag, "rows": 32,
                  "agg_core": "equal", "ret_mean": d["agg"]["ret_mean"]}
            log(ev)
            events.append(ev)
        refs[ref_key] = d
    return refs


# ======================================================================
# S5 FREEZE_SHA / CASE_RUNTIME / NEWLINE_ADOPT
# ======================================================================

def freeze_stage(events):
    assert_stage_prereqs(events, "FREEZE_SHA")
    head = git("rev-parse", "HEAD")
    if not stage_done(events, "FREEZE_SHA"):
        ev = {"event": "FREEZE_SHA", "sha": head,
              "note": "commit 即公证(冻结纪律循 R1/R2/B1/G1;正文零自指 sha);"
                      "冻结 ≠ 发车"}
        log(ev)
        events.append(ev)
    if not stage_done(events, "CASE_RUNTIME"):
        require(CASE_RT is not None, "CASE_RUNTIME 捕获缺失")
        ev = {"event": "CASE_RUNTIME", "five": CASE_RT,
              "note": "案级五 sha 落定(W-E0 位级断言不可用之诚实表述在案:"
                      "protocol 束因 E1 必变;零漂移替代证明 = REF_BITEQ 两发 + "
                      "G0-1 双端点,均已在册)"}
        log(ev)
        events.append(ev)
    if not stage_done(events, "NEWLINE_ADOPT"):
        ev = {"event": "NEWLINE_ADOPT",
              "case": "v33-content(G1 新线首个引用案;冻结呈报单列『新线采用』"
                      "项之台账对应件,G1 D7-6 首引条款履行处)",
              "lines": {"paired_diff": G1_PAIRED_DIFF,
                        "paired_diff_raw": 17.318781,
                        "wins": G1_PAIRED_WINS, "died_max": G1_DIED_MAX,
                        "floor": "85/92 维持现行(RG1.4)"},
              "labels": {"RG1.2a": "单向量线(n_vectors=1;工程免疫非分布论证明)",
                         "RG1.3": "K1 单档依赖线(对 b1-ref8k-launch)"},
              # 升格审查卷两条件载荷位(AUDIT-内容案-G1双标签肢升格审查 8e4be014…)
              "upgrade_review": {
                  "doc_sha256": W_PIN["AUDIT-G1双标签肢升格审查"][1],
                  "verdict": "两肢附条件成立",
                  "g1_verify_note_1": G1_VERIFY_NOTE_1,
                  "g1_verify_note_2": G1_VERIFY_NOTE_2,
                  "deleverage_q95": DELEVERAGE_Q95,
                  "deleverage_q95_raw": DELEVERAGE_Q95_RAW,
                  "g1_results_sha256": W_PIN["g1_results.json"][1],
                  "oc_alpha_columns": OC_ALPHA_COLUMNS,
                  "k1_provenance": K1_PROVENANCE,
                  "loo": "剔 K1 → 线 6,反事实翻转集清空(recal-g1:16)"},
              "alpha_listing": MULTIPLE_COMPARISON,
              "note": "两标签肢经升格审查单列意见(未决 E 已履行);其 8+7 条"
                      "强制随行限定与首考操作特性回填义务随判词生效"}
        log(ev)
        events.append(ev)


# ======================================================================
# S6 W-LAUNCH 发车令闸
# ======================================================================

def launch_gate(events):
    """FREEZE_SHA 之后、三腿首个 leg_start 之前:台账须有 LAUNCH_ORDER 事件
    (总设计师亲发原文与时间戳之值守转录)。无此事件 → 停机待命,退出码 9
    (AWAITING_LAUNCH 非失败态,不落 NEEDS_ATTENTION)。
    **面板过审 ≠ 发车;本函数任何路径不构成发车授权。**"""
    if not stage_done(events, "LAUNCH_ORDER"):
        print("AWAITING_LAUNCH:FREEZE_SHA 已落、LAUNCH_ORDER 未入册——"
              "驱动器于三腿首个 leg_start 之前停机待命(退出码 9,非失败态);"
              "发车令唯一形式 = 总设计师亲发之明示应答经值守逐字转录入台账",
              flush=True)
        sys.exit(9)


# ======================================================================
# S7 三腿串行 + 金丝雀离线序列 + 腿考
# ======================================================================

def dry_curriculum_table_stage(events):
    """DRY_CURRICULUM_TABLE 事件于(课程)腿点火前落全表(E1②)。"""
    if stage_done(events, "DRY_CURRICULUM_TABLE"):
        return
    table = dry_curriculum_table()
    ev = {"event": "DRY_CURRICULUM_TABLE", "schedule": MAIN_TABLE,
          "leg_start": DRY_CURRICULUM_LEG_START, "quantum": QUANTUM,
          "n_rollouts": len(table), "legs": list(CURRICULUM_LEGS),
          # rev4 十二附二⑥:全精度 float 落账(json.dumps(float) 往返无损),
          # 与 verify_curriculum_prefix 复核真源同一;round(p,10) 口径废止。
          "table": [float(p) for p in table],
          "endpoint_note": "端点含内插语义(施工注册):linear 段第 k 项 = "
                           "p0+(p1−p0)·k/(n−1),第 147 量子末项恰达 0.5——"
                           "严格下降实占 299,008 步、p=0.5 实占 200,704 步;"
                           "本注册语义为真源,全表落账即封"}
    log(ev)
    events.append(ev)


def assert_leg_lock_free(leg: str) -> None:
    """B 修(发射夜审计 major):点火前对 RUNS/<leg>/.run.lock 非阻塞 flock
    探测——锁被持即孤儿腿在位(上一发 train_ppo 进程仍活),OperationalFailure
    停机呈报且不落 leg_start(不烧额度)。探测可靠性承 train_ppo._RunLock
    语义(flock 系内核锁,进程崩溃/SIGKILL 自动释放,残留锁文件不构成持锁);
    探测得锁即刻释放,不写不删锁文件。"""
    lock_path = RUNS / leg / ".run.lock"
    if not lock_path.exists():
        return
    with open(lock_path, "r", encoding="utf-8") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            owner = f.read().strip() or "unknown"
            raise OperationalFailure(
                f"{leg} 点火前锁探测:.run.lock 被持({owner})——上一发 "
                "train_ppo 进程仍在位(孤儿腿),取证并 kill 腿进程组后重跑;"
                "本发不落 leg_start(不烧额度)") from exc
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def leg_train_log_name(leg: str, attempt: int) -> str:
    """D 修(发射夜审计 minor):腿训练日志文件名携发次(train-<leg>-a<N>.log,
    N = 本发 leg_start 序号)——run() 系 "w" 截断写,固定名第二发必毁第一发
    崩溃日志取证面。"""
    return f"train-{leg}-a{attempt}.log"


def leg_stage(events, leg: str) -> str:
    """单腿点火(P3 台账制额度 2/腿;nt 闸 3,997,696;resume 恒自王 zip;
    重点火禁换种子)+ 腿终 dry_curriculum.jsonl ≡ 注册表全表复核(课程腿)。"""
    assert_stage_prereqs(events, "LEGS")
    if leg == "v33-full":
        # 修①c(rev4 十二附二④):点火前 demos 字节 ≡ N12_GATE 落定值
        assert_lfull_demos_chain(events)
    model_path = RUNS / leg / "model_final.zip"
    out = RUNS / leg / "policy.npz"
    if not (model_path.exists() and zip_steps(model_path) == NT_TARGET):
        require(leg_starts(events, leg) < 2,
                f"{leg} 腿点火额度耗尽(台账制 2/腿,三腿共 6)——"
                "OPERATIONAL_FAILURE 全案停机,禁手改台账续命")
        # B 修:每次点火前锁探测(孤儿腿在位即停机,先于 leg_start 落账)
        assert_leg_lock_free(leg)
        # D 修:发次序号先于 leg_start 落账计得(N = 已在册 leg_starts + 1)
        attempt = leg_starts(events, leg) + 1
        ev = {"event": "leg_start", "leg": leg, "seed": SEED,
              "extras": list(dict(LEGS)[leg]),
              "recipe": "共用命令形 = v32-sov 逐字 + B1 仪表旋钮 + rev4 勘正"
                        "增列 E5①② record-only 旋钮(D3/十二附二①);"
                        "逐腿附加项 = LEGS 表逐字",
              "launcher": "train/launch_case.sh(E8;驱动器本体经其点火,"
                          "腿系驱动器子进程,PID 簿记口径见 launcher 头注)",
              "resume_from": "king zip(P3 复点火恒自王 zip;种子恒 304000)"}
        log(ev)
        events.append(ev)
        t0 = time.time()
        rc = run(leg_cmd(leg), leg_train_log_name(leg, attempt),
                 timeout=LEG_TIMEOUT)
        nt = zip_steps(model_path)
        log({"event": "leg_done", "leg": leg, "rc": rc, "nt_zip": nt,
             "dt_min": round((time.time() - t0) / 60, 1)})
        require(rc == 0 and nt == NT_TARGET,
                f"{leg} 未达标(rc={rc}, nt={nt}, 目标={NT_TARGET})——"
                "运维失败停机呈报,重启允许第二发(P3)")
    else:
        log({"event": "leg_skip_complete", "leg": leg})
    # 腿终复核:课程腿 dry_curriculum.jsonl ≡ DRY_CURRICULUM_TABLE 全表
    # (『表落了账、跑的是另一张』之缝由此关死;失配 → CASE_HALT_G0 7)
    if leg in CURRICULUM_LEGS:
        diffs = verify_curriculum_prefix(RUNS / leg, dry_curriculum_table(),
                                         LEG_STEPS // QUANTUM)
        if diffs:
            log({"event": "CASE_HALT_G0", "via": "DRY_CURRICULUM_VERIFY",
                 "leg": leg, "diffs": diffs[:10]})
            attention(f"{leg} 腿终 dry_curriculum.jsonl 对注册表全表失配")
            raise SystemExit(7)
        if not any(e.get("event") == "DRY_CURRICULUM_VERIFIED"
                   and e.get("leg") == leg for e in events):
            ev = {"event": "DRY_CURRICULUM_VERIFIED", "leg": leg,
                  "n_rollouts": LEG_STEPS // QUANTUM, "verdict": "≡ 注册表全表"}
            log(ev)
            events.append(ev)
    # 修⑤b(rev4 十二附二①/D8):腿终 DRYWIN_METRICS 事件转录——E5①②
    # 旋钮随共用命令形在位,drywin_metrics.jsonl 文件 sha256 + 行数 + final
    # 段聚合载荷落账;distill_ce_probe.jsonl 同型并入本事件(单事件双档,
    # 承 DEMO_LEDGER v1/v2 双载荷形制);D8 词表之 DRYWIN_METRICS 落笔点。
    if not any(e.get("event") == "DRYWIN_METRICS" and e.get("leg") == leg
               for e in events):
        ev = {"event": "DRYWIN_METRICS", "leg": leg,
              "every": {"drywin_metrics": DRYWIN_METRICS_EVERY,
                        "distill_ce_probe": DISTILL_CE_PROBE_EVERY},
              "drywin_metrics": instrument_jsonl_digest(
                  RUNS / leg / "drywin_metrics.jsonl"),
              "distill_ce_probe": instrument_jsonl_digest(
                  RUNS / leg / "distill_ce_probe.jsonl"),
              "note": "只记不裁(E5①②;RC.14 基线自本案首建,无先验带;"
                      "干/鲜 distill_ce 分列系课②定标数据供给义务(圈 12)"
                      "之腿侧素材;rev4 十二附二① 勘正增列旋钮之落账件)"}
        log(ev)
        events.append(ev)
    if not out.exists():
        require(run([PY, "train/export_worker_npz.py", str(model_path),
                     str(out)], f"export-{leg}.log", 600) == 0 and out.exists(),
                f"{leg} npz 导出失败(补导出通道不烧额度,重启后走 skip 分支)")
        log({"event": "npz_exported", "leg": leg, "sha256": sha256(out)})
    return str(out)


def canary_eval_done(events, tag) -> dict | None:
    hits = [e for e in events
            if e.get("event") == "CANARY_EVAL" and e.get("tag") == tag]
    return hits[-1] if hits else None


def canary_exam(events, worker_npz, tag, manager_npz, l1_fired: bool):
    """P-canary(B1 逐字):不占评测发额度;同 ckpt×manager 落档至多一条即终局;
    重试仅限无档案运维失败;补评截止 = 该腿 s16 FIRING_START 之前(驱动器内
    断言);失败只记不停腿。"""
    out = EVAL / f"{tag}.json"
    prior = canary_eval_done(events, tag)
    if prior is not None:
        require(out.exists(), f"金丝雀 {tag} 台账在册而档案缺失,停机呈报")
        d = validate_adopted(tag, worker_npz, POOL_PROBE, manager_npz)
        require(d["agg"]["_sha"] == prior["sha"],
                f"金丝雀 {tag} 档案与台账 sha 失配")
        return d, False
    if out.exists():
        # E 修(发射夜审计 minor):残档采信恢复路径——validate_adopted 抛
        # EvalContractError 时禁直穿 P1;按 P-canary"失败只记不停腿"处置:
        # 残档 .void 封存 + 事件在册 + 继续走本函数正常序(截止内可重考)。
        try:
            d = validate_adopted(tag, worker_npz, POOL_PROBE, manager_npz)
            return d, True
        except EvalContractError as exc:
            sealed = out.with_suffix(f".{time.time_ns()}.void")
            out.rename(sealed)
            log({"event": "OPERATIONAL-canary", "tag": tag,
                 "void": sealed.name,
                 "why": f"残档采信验证失败({exc}),按 P-canary 只记不停腿:"
                        "残档已 .void,续走截止内重考/截止外不可判"})
    if l1_fired:
        log({"event": "OPERATIONAL-canary", "tag": tag,
             "why": "补评截止已过(该腿 s16 FIRING_START 之前,承合规 M-4),"
                    "禁补——该点 R 线判『不可判,如实登记』"})
        return None, False
    retries = 0
    while retries < 2:
        d = exam(worker_npz, tag, POOL_PROBE, manager_npz)
        if d is not None:
            return d, True
        retries += 1
        log({"event": "OPERATIONAL-canary", "tag": tag, "retry": retries,
             "why": "无档案之运维失败,重试(计数入册)"})
    log({"event": "OPERATIONAL-canary", "tag": tag,
         "why": "重试后仍失败——只记不停腿,该点 R 线判『不可判』"})
    return None, False


def canary_stage_leg(events, leg: str) -> dict:
    """逐腿金丝雀离线序列(B1 照搬;中期四点 × {H, M29};只记不裁;
    L-full 附 A12_CANARY 中期仪表事件)。"""
    short = LEG_SHORT[leg]
    l1_fired = s16_fired(events, leg)
    canary_docs = {}
    for k, step in enumerate(CANARY_STEPS, 1):
        ckpt = RUNS / leg / "ckpt" / f"model_{step}_steps.zip"
        npz = V33 / "canary" / leg / f"policy_{step}.npz"
        if not npz.exists():
            require(ckpt.is_file(), f"金丝雀 ckpt 缺失:{ckpt}")
            npz.parent.mkdir(parents=True, exist_ok=True)
            rc = run([PY, "train/export_worker_npz.py", str(ckpt), str(npz)],
                     f"canary-export-{leg}-{step}.log", 600)
            if rc != 0 or not npz.exists():
                log({"event": "OPERATIONAL-canary", "leg": leg,
                     "ckpt_step": step,
                     "why": f"npz 导出失败(rc={rc}),该点全部读数『不可判』"})
                continue
        for mtag, manager in (("h", None), ("m29", str(M29_NPZ))):
            tag = f"v33-{short}-canary{k}-{mtag}"
            d, fresh = canary_exam(events, str(npz), tag, manager, l1_fired)
            if d is None:
                continue
            canary_docs[tag] = d
            if fresh or canary_eval_done(events, tag) is None:
                rows = sorted(d["rows"], key=lambda r: r["seed"])
                ev = {"event": "CANARY_EVAL", "tag": tag, "leg": leg, "k": k,
                      "ckpt_step": step,
                      "manager": "H" if mtag == "h" else "M29",
                      "sha": d["agg"]["_sha"],
                      "mean": d["agg"]["ret_mean"], "died": d["agg"]["died"],
                      "seeds": [r["seed"] for r in rows],
                      "ret": [r["ret"] for r in rows],
                      "died_vec": [int(r["died"]) for r in rows],
                      "depth": [r["depth"] for r in rows],
                      "d_windows": [d_windows(r) for r in rows],
                      "farm_tau_mean": [r["farm_tau_mean"] for r in rows],
                      "a12_per_ep": a12_per_ep(d["agg"]),
                      "discipline": "只记不裁;禁作任何腿内干预依据;终点检查点"
                                    "唯一 = nt 3,997,696;一切金丝雀读数一律"
                                    "描述性,裁决面 = 终考档案与发射线(面板 M3)"}
                log(ev)
                events.append({"event": "CANARY_EVAL", "tag": tag,
                               "sha": d["agg"]["_sha"]})
            if leg == "v33-full" and not any(
                    e.get("event") == "A12_CANARY" and e.get("tag") == tag
                    for e in events):
                # E5③ 中期仪表(RC.11 再灭绝动力学序列;只记不裁)。施工裁量
                # 呈报:评测档案无逐局动作直方,E5③ 逐局口径(episodes_with_
                # a12/a12_max)不可得,record_a12_canary 未调用(禁伪值)。
                hist = d["agg"].get("worker_action_hist", {}) or {}
                ev = {"event": "A12_CANARY", "tag": tag, "leg": leg,
                      "checkpoint_step": step,
                      "manager": "H" if mtag == "h" else "M29",
                      "a12_total": int(hist.get("12", 0)),
                      "episodes": int(d["agg"].get("n", 32)),
                      "a12_per_episode": a12_per_ep(d["agg"]),
                      "per_episode_note": "评测档案无逐局动作直方——逐局分解"
                                          "(episodes_with_a12/a12_max)不可得,"
                                          "如实呈报(禁伪值;E5③ 逐局口径待"
                                          "上游供数)"}
                log(ev)
                events.append(ev)
    return canary_docs


def leg_exams(events, leg: str, npz_path: str) -> dict:
    """腿考三发:s16(单 H,E1 豁免枚举①半池合规)→ full32(H)→ full32-m29。"""
    short = LEG_SHORT[leg]
    docs = {}
    docs["s16"] = exam_case(events, npz_path, f"v33-{short}-s16", POOL_S16,
                            extra={"note": "快筛(B1 E1 豁免枚举①,半池单 H "
                                           "合规)"})
    for suffix, manager, bundle in (
            ("full32", None, f"v33-{short}-full32-m29"),
            ("full32-m29", str(M29_NPZ), f"v33-{short}-full32")):
        tag = f"v33-{short}-{suffix}"
        d = exam_case(events, npz_path, tag,
                      POOL_PROBE, manager_npz=manager, bundled_with=bundle)
        a = d["agg"]
        if not any(e.get("event") == "exam_detail" and e.get("tag") == tag
                   for e in events):
            ev = {"event": "exam_detail", "tag": tag, "leg": leg,
                  "mean": a["ret_mean"], "died": a["died"],
                  "a12_per_ep": a12_per_ep(a), "a13_per_ep": a13_per_ep(a),
                  "rwin_per_ep": round(sum(str(r["mode_seq"]).count("R")
                                           for r in d["rows"]) / 32, 2),
                  "depth2": depth2_count(d["rows"]),
                  "dive": round(dive_per_ep(d["rows"]), 2),
                  "override": a["override_rate"], "cap": a["cap_rate"],
                  "sha": a["_sha"]}
            log(ev)
            events.append(ev)
        docs[suffix] = d
    return docs


# ======================================================================
# S8 判据(课⑤主判 / ④乙主判 / 资格 / 胜者 / 发射)——纯函数面供单测
# ======================================================================

def qual_of(doc) -> dict:
    """逐腿资格显式成文(D4-1/合规 m-4):died≤8(G1 L2-0 (a) 同值替换)∧
    哨兵肢零触碰(换层/cap/override,G1 L1 零触碰清单)∧ 非作废。"""
    a = doc["agg"]
    dive = dive_per_ep(doc["rows"])
    void = (a["override_rate"] >= R4["override_void"]
            or (dive > 1 and a["died"] > 6))
    sentinel_ok = (a["farm_descend_rate"] <= R4["descend"]
                   and a["cap_rate"] < R4["cap"]
                   and a["override_rate"] < R4["override_sentinel"])
    out = {"died_ok": a["died"] <= G1_DIED_MAX,
           "sentinel_zero_touch": sentinel_ok, "void": void}
    out["qual_ok"] = out["died_ok"] and sentinel_ok and not void
    if (not out["qual_ok"] and out["died_ok"] and not void
            and a["cap_rate"] >= R4["cap"]
            and a["farm_descend_rate"] <= R4["descend"]
            and a["override_rate"] < R4["override_sentinel"]):
        out["cap_drift_note"] = ("失格仅因 cap_rate——闸门语义漂移候选注记 + "
                                 "τ̄ 对 ref 差与逐窗分布(R5)随判词")
    return out


def decide_winner(means: dict, died: dict, quals: dict) -> dict:
    """胜者条款(D4-5):过资格腿中 full32(H) 均值最高;差 ≤0.05 → died 少者;
    再平 → 取 L-full(处方腿优先);均值最高者被资格拦截 → 递补(substitution
    入册)。"""
    pool = [n for n in means if quals[n]["qual_ok"]]
    if not pool:
        return {"winner": None, "pool": [], "substituted": False}
    prelim = max(means, key=lambda n: means[n])
    band = [n for n in pool if max(means[n2] for n2 in pool) - means[n] <= 0.05]
    if len(band) > 1:
        dmin = min(died[n] for n in band)
        band = [n for n in band if died[n] == dmin]
        winner = "v33-full" if "v33-full" in band else band[0]
    else:
        winner = band[0]
    return {"winner": winner, "pool": pool, "prelim": prelim,
            "substituted": winner != prelim and prelim not in pool}


def course5_ruling(anchor_hits: int, cur_hits: int) -> dict:
    """课⑤主判(D4-2 重立):唯一裁决量 = F-lock v1 命中数;裁决腿 = L-cur×H;
    锚 = L-base 案内实测(先验点 5 承 P8 并列不作锚)。裁决线全数预登记。"""
    out = {"anchor_hits_l_base": anchor_hits, "cur_hits": cur_hits,
           "anchor_prior_note": f"先验点 {COURSE5_ANCHOR_PRIOR} 承 P8"
                                "(隔两级变量,降为先验并列不作锚)"}
    if anchor_hits <= 3:
        out["verdict"] = ("地板效应,不可判——课⑤主判不出成/败判词,转干窗行为"
                          "仪表(E5②)、D 窗保持率与 τ 分布如实登记待续案")
        out["branch"] = "floor"
        return out
    if cur_hits <= anchor_hits - 2:
        out["verdict"] = f"方向成功(L-cur {cur_hits} ≤ 锚−2 = {anchor_hits - 2})"
        out["branch"] = "success"
    elif cur_hits == anchor_hits - 1:
        out["verdict"] = "噪声带内降(描述性,= 锚−1)"
        out["branch"] = "noise"
    else:
        out["verdict"] = f"未改善(L-cur {cur_hits} ≥ 锚 {anchor_hits})"
        out["branch"] = "no_improvement"
    if anchor_hits == 4:
        out["note_low_anchor"] = "基线偏低判别力降(锚=4,成功线 ≤2)"
    return out


def a12_tier_ruling(a12_full: float, died_full: int, paired_mean: float,
                    died_cur: int, a12_cur: float) -> dict:
    """④乙主判档位表(D4-3 重登记;判定顺序档1→档2→档3,穷尽由补集档保证;
    ctrl = L-cur×M29;常数按本案语境重注册,呈总设计师过目(未决 B))。"""
    out = {"a12_full": a12_full, "died_full": died_full,
           "paired_mean_vs_cur_m29": round(paired_mean, 2),
           "died_cur_m29": died_cur, "a12_cur": a12_cur,
           "main_line": a12_full >= A12_USE_LINE,
           "binom_note": "died 档位系方向读数非显著性证据(二项注记随行)"}
    if (a12_full >= A12_USE_LINE and died_full <= died_cur
            and paired_mean >= SURV_MEAN_BAND):
        out["tier"] = 1
        out["verdict"] = (f"档1(注入兑换存活):a12/局 {a12_full}≥{A12_USE_LINE}"
                          f" ∧ died {died_full}≤ctrl {died_cur} ∧ 配对均差 "
                          f"{paired_mean:.2f}≥{SURV_MEAN_BAND}")
    elif (a12_full < A12_USE_LINE and paired_mean >= SURV_MEAN_BAND
          and died_full <= died_cur + 1):
        out["tier"] = 2
        out["verdict"] = (f"档2(给了示范不用):a12/局 {a12_full}<{A12_USE_LINE}"
                          f" ∧ 均差 {paired_mean:.2f}≥{SURV_MEAN_BAND} ∧ died "
                          f"{died_full}≤ctrl+1(v32 F1 明文预期与 R11/残余④ "
                          "首要风险情形)")
        if 0 < a12_full < A12_USE_LINE and died_full <= died_cur:
            out["low_use_note"] = "低用量与存活改善并现,归因未证(v32 逐字)"
    else:
        out["tier"] = 3
        if a12_full >= A12_USE_LINE:
            sub = "(a) 用而未兑现(a12≥0.1 但存活/均值肢不满足)"
        else:
            sub = "(b) 不用且劣化"
        out["verdict"] = f"档3(补集档,穷尽由此保证):{sub}"
        out["tier3_subcase"] = sub
    if a12_cur >= A12_USE_LINE:
        out["control_crossline"] = ("对照越线分支(承统计 M-1 预登记):L-cur "
                                    f"a12/局 {a12_cur}≥{A12_USE_LINE} → 判词固定携"
                                    "『课程独立复活候选(存活复利通路,v32 残余②"
                                    "同族),④乙包归因降级为「未证增益」』,档位"
                                    "照裁但归因语句按此改述")
    if out["tier"] in (2, 3):
        out["circle12_exit"] = ("圈 12 阶梯条款出口语法(D4-4):呈总设计师开庭"
                                "再裁(换机制再试或诚实宣告灭绝并解除),禁预判、"
                                "禁自动转案")
    out["conjunction_note"] = "④乙包 = λ_bc>0 + v2 件之合取限定强制随行(R7 教师世代混杂注记随行)"
    return out


def death_seed_sets(full_m29_rows: list, ctrl_m29_rows: list) -> dict:
    """G 修(发射夜审计 minor,语境勘正注记):④乙判词"死亡种子集合差"之
    对照改取 D4-3 本案语境 ctrl = L-cur×M29(full32-m29)rows——替换原用
    v32-ref-science(跨案参照非本案 ctrl);rescued/new_deaths 语义随之
    (相对 ctrl 的救活/新死)。"""
    ctrl_dead = {r["seed"] for r in ctrl_m29_rows if r["died"]}
    full_dead = {r["seed"] for r in full_m29_rows if r["died"]}
    return {"ctrl": "L-cur×M29(full32-m29;D4-3 本案语境勘正,"
                    "替换 v32-ref-science)",
            "rescued": sorted(ctrl_dead - full_dead),
            "new_deaths": sorted(full_dead - ctrl_dead)}


# ======================================================================
# 统计纪律工具(B1 承继;R 线用)
# ======================================================================

def median(xs) -> float:
    s = sorted(xs)
    n = len(s)
    require(n > 0, "median 输入为空")
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def binom_tail_ge(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def sign_test(diffs) -> dict:
    neg = sum(1 for d in diffs if d < 0)
    pos = sum(1 for d in diffs if d > 0)
    ties = len(diffs) - neg - pos
    n = neg + pos
    p = binom_tail_ge(max(neg, pos), n) if n else 1.0
    return {"neg": neg, "pos": pos, "ties": ties, "p_one_sided": round(p, 4)}


def deleveraged_mean(diff_by_seed: dict) -> dict:
    require(len(diff_by_seed) >= 2, "去杠杆均值需要 ≥2 种子")
    lever = max(diff_by_seed, key=lambda s: abs(diff_by_seed[s]))
    rest = [v for s, v in diff_by_seed.items() if s != lever]
    return {"dropped_seed": lever,
            "dropped_value": round(diff_by_seed[lever], 2),
            "mean": round(sum(rest) / len(rest), 2)}


def loo_7017(diff_by_seed: dict) -> dict:
    """7017 留一去读数(强制并列;深水魔种常备对照,−150 级杠杆)。"""
    rest = [v for s, v in diff_by_seed.items() if s != DEEPWATER_SEED]
    return {"dropped_seed": DEEPWATER_SEED,
            "dropped_value": round(diff_by_seed.get(DEEPWATER_SEED, float("nan")), 2)
            if DEEPWATER_SEED in diff_by_seed else None,
            "mean": round(sum(rest) / len(rest), 2) if rest else None}


def band_judge(x: float, lo: float, hi: float, integer: bool = False,
               upper_open: bool = False) -> dict:
    """H 修(发射夜审计 minor):upper_open=True 时带系 [lo, hi) 半开——
    仅 RC.10 损伤分支带 [0.0, 0.5) 消费(retention 恰 0.5 须落健康分支);
    默认闭区间语义全局不动(禁全局改语义)。"""
    in_band = (lo <= x < hi) if upper_open else (lo <= x <= hi)
    if integer:
        borderline = min(abs(x - lo), abs(x - hi)) <= 1
    else:
        borderline = min(abs(x - lo), abs(x - hi)) <= 0.05 * (hi - lo)
    out = {"x": round(float(x), 4), "band": [lo, hi], "in_band": in_band}
    if upper_open:
        out["upper_open"] = True
    if borderline:
        out["borderline_note"] = "临线 + 线未重标(临线条款强制注记)"
    return out


def paired_diff_stats(leg_rows: dict, ref_rows: dict) -> dict:
    """配对均差统计块(D4 统计纪律:均值必并列中位+符号检验+去杠杆+7017 留一)。
    C 修(发射夜审计 major):增 mean_raw 全精度均值——发射合取均差肢与
    RC.9 MS 分支触发一律消费 raw 裁决(v32 先例系 raw 比较;2dp 圆整值仅作
    落账显示,0.005 窗内圆整可静默翻转判决)。"""
    diffs = {s: leg_rows[s]["ret"] - ref_rows[s]["ret"] for s in sorted(ref_rows)}
    vals = list(diffs.values())
    mean_raw = sum(vals) / len(vals)
    return {"mean": round(mean_raw, 2),
            "mean_raw": mean_raw,
            "median": round(median(vals), 2),
            "sign": sign_test(vals),
            "deleveraged": deleveraged_mean(diffs),
            "loo_7017": loo_7017(diffs),
            "wins": sum(1 for v in vals if v > 0),
            "by_seed": {str(s): round(v, 2) for s, v in diffs.items()}}


def ms_vectors(leg_h: dict, leg_m29: dict, ref_h: dict, ref_m29: dict) -> dict:
    seeds = sorted(leg_h)
    require(set(seeds) == set(leg_m29) == set(ref_h) == set(ref_m29),
            "MS 四档种子面不一致")
    signed = {}
    for s in seeds:
        dh = leg_h[s]["ret"] - ref_h[s]["ret"]
        dm = leg_m29[s]["ret"] - ref_m29[s]["ret"]
        signed[s] = round(dh - dm, 2)
    ms = {s: abs(v) for s, v in signed.items()}
    over = sorted(s for s, v in ms.items() if v > MS_FLAG_LINE)
    return {"signed_dh_minus_dm": signed,
            "ms_max": round(max(ms.values()), 2),
            "ms_median": round(median(list(ms.values())), 2),
            "over_line_seeds": over, "n_over_line": len(over),
            "flag_line": MS_FLAG_LINE}


# ======================================================================
# 重放通道(F-lock τ 供数;B1 E4 承继)
# ======================================================================

def run_obsdrift(worker_npz, archive: pathlib.Path, out: pathlib.Path,
                 manager=None) -> dict:
    if out.exists():
        report = strict_json_loads(out.read_bytes())
        if (report.get("archive_sha256") == sha256(archive)
                and report.get("fidelity_ok")):
            return report
        out.rename(out.with_suffix(f".{time.time_ns()}.stale"))
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "train/probe_b1_obsdrift.py", "--worker", str(worker_npz),
           "--archive", str(archive), "--out", str(out)]
    if manager:
        cmd += ["--manager", str(manager)]
    rc = run(cmd, f"obsdrift-{out.stem}.{time.time_ns()}.log", 3_600)
    require(rc == 0 and out.exists(),
            f"重放失败/保真失配:{archive.name}(保真锚条款,停机呈报)")
    return strict_json_loads(out.read_bytes())


def flock_hits(leg_full32_doc, ref_launch_rows: dict, leg_npz: str,
               leg: str) -> dict:
    """F-lock 复合签名 v1 命中(防火墙:v1 现行构造常数裁决,v2 回炉系仪表
    永不入本案裁决线;特异性未标定限定 + horizon 限定强制随行)。"""
    from probe_composite_signature import composite_signature
    replay = run_obsdrift(leg_npz, EVAL / f"v33-{LEG_SHORT[leg]}-full32.json",
                          V33 / "replay" / f"v33-{LEG_SHORT[leg]}-full32.json")
    tau = {int(s): v.get("farm_tau_median")
           for s, v in replay.get("per_seed", {}).items()}
    sig = composite_signature(ref_launch_rows,
                              by_seed(leg_full32_doc["rows"], 7000, 7031), tau)
    return {"n_hits": sig["n_hits"], "hits": sig["hits"],
            "hit_7017": DEEPWATER_SEED in sig["hits"],
            "caveats": ["签名特异性未标定(强制随判词)",
                        "horizon 限定(max-steps 3000,禁调)"]}


# ======================================================================
# S9 R 线记分卡 + VERDICT_PATH
# ======================================================================

def scorecard_stage(events, refs, leg_docs, canary_docs, verdicts,
                    hold_docs):
    if stage_done(events, "VERDICT_PATH"):
        return
    ref_launch = by_seed(refs["launch"]["rows"], 7000, 7031)
    ref_science = by_seed(refs["science"]["rows"], 7000, 7031)
    rows = {leg: {"h": by_seed(leg_docs[leg]["full32"]["rows"], 7000, 7031),
                  "m29": by_seed(leg_docs[leg]["full32-m29"]["rows"],
                                 7000, 7031)}
            for leg in LEG_NAMES}
    card = {}
    band_outcomes = []

    def add(name, judged):
        card[name] = judged
        if isinstance(judged, dict) and "in_band" in judged:
            band_outcomes.append((name, judged["in_band"]))

    card["RC1_ref_biteq"] = "REF_BITEQ 两发在册(≡113.0/≡140.9 全表,冻结先决已履行)"
    # RC.2 L-base full32(H) 配对均差
    st2 = paired_diff_stats(rows["v33-base"]["h"], ref_launch)
    rc2 = band_judge(st2["mean"], *RC2_BAND["band"])
    rc2.update({"point": RC2_BAND["point"], **{k: st2[k] for k in
                ("median", "sign", "deleveraged", "loo_7017")}})
    add("RC2_base_paired_mean", rc2)
    # C 修:MS 双分支机械触发消费原始均值(圆整读数贴线可静默翻转择带)
    branch = "healthy" if st2["mean_raw"] >= RC9_BRANCH_LINE else "damaged"
    # RC.3 课⑤效应(L-cur − L-base 同种子配对)
    d3 = {s: rows["v33-cur"]["h"][s]["ret"] - rows["v33-base"]["h"][s]["ret"]
          for s in sorted(ref_launch)}
    rc3 = band_judge(sum(d3.values()) / 32, *RC3_BAND["band"])
    rc3.update({"point": RC3_BAND["point"],
                "note": "带外下 → 课⑤伤害候选,判词按格改述禁『成/败』统称"})
    add("RC3_course5_effect", rc3)
    # RC.4 ④乙效应(L-full − L-cur)
    d4 = {s: rows["v33-full"]["h"][s]["ret"] - rows["v33-cur"]["h"][s]["ret"]
          for s in sorted(ref_launch)}
    rc4 = band_judge(sum(d4.values()) / 32, *RC4_BAND["band"])
    rc4.update({"point": RC4_BAND["point"],
                "note": "RC.3 与 RC.4 共享 L-cur、误差构造性反相关(两者之和 ≡ "
                        "L-full−L-base),禁作独立证据叠加(承统计 minor-4);"
                        "合取限定强制随行"})
    add("RC4_yi_effect", rc4)
    card["RC5_course5_main"] = verdicts["course5"]
    card["RC6_a12_main"] = {**verdicts["a12"],
                            "point": RC6_BAND["point"],
                            "band": list(RC6_BAND["band"]),
                            "prior_note": "v32 R32.3 预登记点 1.5、实测 0.0"
                                          "(F1 主权零行使)——本案系注入后首个"
                                          "非零预期,无实测先例(rev3 勘正)"}
    card["RC7_launch_limbs"] = verdicts["launch_limbs"]
    # RC.8 逐腿 died
    for leg in LEG_NAMES:
        rc8h = band_judge(leg_docs[leg]["full32"]["agg"]["died"],
                          *RC8_BAND_H["band"], integer=True)
        rc8h.update({"point": RC8_BAND_H["point"],
                     "died_vec": [int(r["died"]) for r in sorted(
                         leg_docs[leg]["full32"]["rows"],
                         key=lambda r: r["seed"])]})
        add(f"RC8_died_h_{LEG_SHORT[leg]}", rc8h)
        rc8m = band_judge(leg_docs[leg]["full32-m29"]["agg"]["died"],
                          *RC8_BAND_M29["band"], integer=True)
        rc8m["point"] = RC8_BAND_M29["point"]
        add(f"RC8_died_m29_{LEG_SHORT[leg]}", rc8m)
    # RC.9 双经理读数差 + MS 双分支(由 RC.2 实测落点机械触发,禁事后择带)
    for leg in LEG_NAMES:
        diffs9 = [rows[leg]["m29"][s]["ret"] - rows[leg]["h"][s]["ret"]
                  for s in sorted(ref_launch)]
        rc9 = band_judge(sum(diffs9) / 32, *RC9_BAND["band"])
        rc9["point"] = RC9_BAND["point"]
        add(f"RC9_dual_manager_{LEG_SHORT[leg]}", rc9)
        ms = ms_vectors(rows[leg]["h"], rows[leg]["m29"],
                        ref_launch, ref_science)
        msb = RC9_MS[branch]
        rc9max = band_judge(ms["ms_max"], *msb["max"][1])
        rc9max.update({"point": msb["max"][0], "branch": branch,
                       "ms_limitation": "MS≈0 不证工人无损(经理敏感损伤探测器,"
                                        "非全损伤探测器;限定强制随判词)"})
        add(f"RC9_MS_max_{LEG_SHORT[leg]}_{branch}", rc9max)
        rc9over = band_judge(ms["n_over_line"], *msb["over"][1], integer=True)
        rc9over["point"] = msb["over"][0]
        add(f"RC9_MS_overline_{LEG_SHORT[leg]}_{branch}", rc9over)
    # RC.10 金丝雀 D 窗保持率(canary4×H,C 集 depth2;分支同 RC.9)
    cs = [e for e in events if e.get("event") == "CANARY_SET"][0]
    d_c, n_d = cs["depth2_seeds"], cs["n_D"]
    for leg in LEG_NAMES:
        c4 = canary_docs.get(leg, {}).get(f"v33-{LEG_SHORT[leg]}-canary4-h")
        if c4 is None:
            card[f"RC10_retention_{LEG_SHORT[leg]}"] = \
                "不可判(金丝雀缺数,P-canary),如实登记"
            continue
        c4_rows = by_seed(c4["rows"], 7000, 7031)
        kept = [s for s in d_c if d_windows(c4_rows[s]) >= 1]
        # H 修:损伤带 [0.0, 0.5) 半开(retention 恰 0.5 落健康分支);仅此带
        rc10 = band_judge(len(kept) / n_d, *RC10_BAND[branch],
                          upper_open=(branch == "damaged"))
        rc10.update({"kept_seeds": sorted(kept), "n_D": n_d, "branch": branch,
                     "formula": "保持率 := |{s∈D_C: canary4×H 该种子 D 窗数 ≥1}|"
                                " / n_D(RB.8 公式逐字;一律描述性)"})
        add(f"RC10_retention_{LEG_SHORT[leg]}", rc10)
    # RC.11 金丝雀 a12/局 序列(L-full;无先验带逐点登记)
    a12_seq = [{"tag": e["tag"], "step": e["checkpoint_step"],
                "manager": e["manager"], "a12_per_episode": e["a12_per_episode"]}
               for e in events if e.get("event") == "A12_CANARY"]
    card["RC11_a12_canary_seq"] = {
        "seq": a12_seq,
        "note": "预期形态 = 注入后非零、中后段衰减风险(R11 v14/v15 镜像);"
                "无先验带,逐点如实登记;金丝雀只见不救,终考判据兜底"}
    # RC.12 dry-anchor 增量(腿终;零点 0.7515;从未具规范义)
    for leg in LEG_NAMES:
        sent = RUNS / leg / "sentinel.jsonl"
        dry_lines = ([json.loads(x) for x in sent.read_text().splitlines()
                      if '"dry-anchor"' in x] if sent.is_file() else [])
        finals = [r for r in dry_lines if r.get("final")] or dry_lines[-1:]
        if not finals:
            card[f"RC12_dry_anchor_{LEG_SHORT[leg]}"] = "不可判(哨兵缺数)"
            continue
        inc = (finals[-1]["mismatch"] - DRY_REF_THRONE) * 100
        rc12 = band_judge(inc, *RC12_BAND["band"])
        rc12.update({"point": RC12_BAND["point"],
                     "refs": {"throne": DRY_REF_THRONE,
                              "lineage": DRY_REF_LINEAGE},
                     "note": "描述性,不作判据肢(G1 D6 方向反转在案,+8.35pp "
                             "健康腿反例强制注记);探针集三腿统一 BC-v1 demos "
                             "字节(E6 担保链)"})
        add(f"RC12_dry_anchor_{LEG_SHORT[leg]}", rc12)
    # RC.13 BC-v2 回执(冻结先决已过闸)
    n12_ev = [e for e in events
              if e.get("event") == "N12_GATE" and e.get("gate") == "PASS"][-1]
    rc13 = band_judge(n12_ev["n12"], *RC13_N12_BAND["band"], integer=True)
    rc13.update({"point": RC13_N12_BAND["point"],
                 "recall_12": band_judge(n12_ev["recall_12"],
                                         *RC13_RECALL_BAND["band"]),
                 "oc_consumed": n12_ev.get("oc_consumed"),
                 "cluster_note": N12_CLUSTER_NOTE})
    add("RC13_n12", rc13)
    # RC.14 干窗行为仪表(基线自本案首建,无先验带;审计缺口 i 闭合起点)
    rc14 = {}
    for leg in LEG_NAMES:
        rc14[leg] = {
            "drywin_metrics": (str(RUNS / leg / "drywin_metrics.jsonl")
                               if (RUNS / leg / "drywin_metrics.jsonl").is_file()
                               else "不可得(E5② 旋钮随 rev4 D3 共用命令形"
                                    "在位而无产物——运维缺件,如实登记;"
                                    "腿终 DRYWIN_METRICS 事件应已先行拦截)"),
            "distill_ce_probe": (str(RUNS / leg / "distill_ce_probe.jsonl")
                                 if (RUNS / leg
                                     / "distill_ce_probe.jsonl").is_file()
                                 else "不可得(同上;课②定标数据供给义务之"
                                      "腿侧素材缺口如实登记)")}
    card["RC14_drywin_instruments"] = {
        **rc14, "note": "无先验带,如实登记;本案内只能自证一致性不能证常模"}
    # RC.15 8000 池胜者对
    if hold_docs:
        k1 = by_seed(strict_json_loads(
            W_PIN["b1-ref8k-launch"][0].read_bytes())["rows"], 8000, 8031)
        h1_rows = by_seed(hold_docs["h"]["rows"], 8000, 8031)
        st15 = paired_diff_stats(h1_rows, k1)
        rc15 = band_judge(st15["mean"], *RC15_BAND["band"])
        rc15.update({"point": RC15_BAND["point"], "median": st15["median"],
                     "sign": st15["sign"]})
        add("RC15_holdout_paired", rc15)
        rc15d = band_judge(depth2_count(hold_docs["h"]["rows"]),
                           RC15_BAND["d2_band"][0], RC15_BAND["d2_band"][1],
                           integer=True)
        rc15d["point"] = RC15_BAND["d2_point"]
        add("RC15_holdout_depth2", rc15d)
    else:
        card["RC15_holdout"] = "不可判(无胜者或留出对未发),如实登记"
    # 族级解读条款(本卡带判读 14 项;散发带外 ≤3 常态,禁摘樱桃)
    n_out = sum(1 for _, ok in band_outcomes if not ok)
    family = {"n_band_readings": len(band_outcomes), "n_out_of_band": n_out,
              "out_names": [n for n, ok in band_outcomes if not ok],
              "clause": "散发带外 ≤3 常态(与 B1 约 17 带 ≤3 比例相当),禁摘"
                        "樱桃;聚簇带外才升级 NEEDS_ATTENTION"}
    if n_out >= 4:
        attention(f"R 线聚簇带外({n_out} 条):{family['out_names']}")
    ev = {"event": "VERDICT_PATH", "case": "v33-content",
          "golden_authorized": verdicts.get("golden_authorized", False),
          "scorecard": card, "family_ledger": family,
          "winner": verdicts.get("winner"),
          "course5": verdicts["course5"], "a12_tier": verdicts["a12"],
          "attribution_grammar": {
              "course5": "课⑤效应 = L-cur − L-base",
              "yi": "④乙效应 = L-full − L-cur(④乙包 = λ_bc>0 + v2 件之合取)",
              "triangle": "同种子仅保初始等价,『唯一变量』系配置层陈述非轨迹配对"
                          "(承 v32 R32-拆分口径逐字)"},
          "mandatory_notes": [
              "horizon 声明:max-steps 3000,禁调,强制随判词",
              "训练 reward 对本病失明(−6% vs −34),禁作任何判据",
              "H 轨默认与不溯及既往条款承 B1 D3",
              "M29 侧评测确定性系推定(B1 残余⑮ 未清偿),推定注记随一切 M29 判词",
              "点估归属总注:R 线各点估凡未注出处者一律系裁量非估计",
              "残余①-⑰ 见预注册十二节,随判决附录承继(判决附录由值守记档)"]}
    log(ev)
    events.append(ev)
    attention(f"v33 案结记分卡在册:课⑤={verdicts['course5'].get('branch')};"
              f"④乙档={verdicts['a12'].get('tier')};"
              f"胜者={verdicts.get('winner')};判决附录由值守记档")


def log_launch_check_no_winner(events, limbs: dict, quals: dict) -> None:
    """F 修(发射夜审计 minor):无胜者分支之 launch_check 落账补 stage_done
    幂等护栏(与有胜者分支对称)——重入(续跑)不重复落账。"""
    if stage_done(events, "launch_check"):
        return
    log({"event": "launch_check", **limbs, "quals": quals})
    events.append({"event": "launch_check"})


# ======================================================================
# 主流程
# ======================================================================

def _main():
    events = read_ledger()
    if stage_done(events, "VERDICT_PATH"):
        print("案已结:幂等退出", flush=True)
        return
    preflight(events)                                   # S0
    g0_baseline_stage(events)                           # S1 E0 转录
    bc1_stage(events)                                   # S2 S-bc1
    bc2_stage(events)                                   # S3 S-bc2 + N12
    g0_endpoint_stage(events)                           # S4 G0-1
    g0_nullintrusion_stage(events)                      # S4 G0-2a
    g0_funcsmoke_stage(events)                          # S4 G0-2b
    g0_ghost_stage(events)                              # S4 G0-幽灵
    g0_demo_ledger_stage(events)                        # S4 G0-示范池
    refs = refs_stage(events)                           # S4 G0-6 REF_BITEQ
    freeze_stage(events)                                # S5 FREEZE/RT/NEWLINE
    launch_gate(events)                                 # S6 W-LAUNCH(exit 9)

    # ---- S7 三腿串行:点火 → 金丝雀 → s16 → full32 → full32-m29 ----
    dry_curriculum_table_stage(events)
    npz = {}
    leg_docs = {}
    canary_docs = {}
    for leg in LEG_NAMES:
        npz[leg] = leg_stage(events, leg)
        canary_docs[leg] = canary_stage_leg(events, leg)
        leg_docs[leg] = leg_exams(events, leg, npz[leg])

    # ---- S8 判据(科学主判先行,与发射解耦) ----
    ref_launch = by_seed(refs["launch"]["rows"], 7000, 7031)
    # sci_rows 死赋值已除(审计呈报①:D4-3 语境勘正后 science 对照退场,G 修落 death_seed_sets)
    R = refs["launch"]["agg"]["ret_mean"]
    abandon = round(R * ABANDON_FRAC[0] / ABANDON_FRAC[1], 1)
    floor_line = round(R * FLOOR_FRAC[0] / FLOOR_FRAC[1], 1)
    # 课⑤主判(RC.5):F-lock v1 命中;锚 = L-base 案内实测
    sig = {leg: flock_hits(leg_docs[leg]["full32"], ref_launch,
                           npz[leg], leg) for leg in LEG_NAMES}
    course5 = course5_ruling(sig["v33-base"]["n_hits"],
                             sig["v33-cur"]["n_hits"])
    course5.update({
        "hits_by_leg": {leg: sig[leg]["n_hits"] for leg in LEG_NAMES},
        "l_full_note": "L-full 命中数强制并列但系课⑤+④乙复合读数,永不入课⑤"
                       "裁决;与 L-cur 分歧时如实分列禁聚合",
        "seat_7017_note": {leg: sig[leg]["hit_7017"] for leg in LEG_NAMES},
        "firewall": "F-lock 以 v1 现行构造常数裁决;v2 回炉系附属交付仪表身份"
                    "永不入本案裁决线",
        "caveats": sig["v33-base"]["caveats"]})
    if not stage_done(events, "COURSE5_MAIN"):
        log({"event": "COURSE5_MAIN", **course5})
        events.append({"event": "COURSE5_MAIN"})
    # ④乙主判(D4-3):L-full 无论胜负必判;ctrl = L-cur×M29
    full_m29 = by_seed(leg_docs["v33-full"]["full32-m29"]["rows"], 7000, 7031)
    cur_m29 = by_seed(leg_docs["v33-cur"]["full32-m29"]["rows"], 7000, 7031)
    paired_fc = sum(full_m29[s]["ret"] - cur_m29[s]["ret"]
                    for s in sorted(cur_m29)) / 32
    a12_verdict = a12_tier_ruling(
        a12_per_ep(leg_docs["v33-full"]["full32-m29"]["agg"]),
        leg_docs["v33-full"]["full32-m29"]["agg"]["died"], paired_fc,
        leg_docs["v33-cur"]["full32-m29"]["agg"]["died"],
        a12_per_ep(leg_docs["v33-cur"]["full32-m29"]["agg"]))
    base_m29_agg = leg_docs["v33-base"]["full32-m29"]["agg"]
    a12_verdict["base_m29_parallel"] = {
        "mean": base_m29_agg["ret_mean"], "died": base_m29_agg["died"],
        "a12": a12_per_ep(base_m29_agg),
        "note": "重训连带防线对应件:L-cur 与 L-base 同等存活时『兑换』因果"
                "限定收紧"}
    # G 修:集合差对照 = L-cur×M29(D4-3 本案语境;勘正注记入 death_seed_sets)
    a12_verdict["death_seed_sets"] = death_seed_sets(
        leg_docs["v33-full"]["full32-m29"]["rows"],
        leg_docs["v33-cur"]["full32-m29"]["rows"])
    a12_verdict["h_side_parallel"] = {
        "a12_full_h": a12_per_ep(leg_docs["v33-full"]["full32"]["agg"]),
        "note": "H 治下读数并列、携『结构性偏紧』限定(ref depth2 7 vs 15),"
                "不单独成线;与发射判据解耦"}
    if not stage_done(events, "A12_MAIN"):
        log({"event": "A12_MAIN", **a12_verdict})
        events.append({"event": "A12_MAIN"})

    # ---- 资格 / 胜者 / 发射(G1 新线首引,合取式全录) ----
    means = {leg: leg_docs[leg]["full32"]["agg"]["ret_mean"]
             for leg in LEG_NAMES}
    died = {leg: leg_docs[leg]["full32"]["agg"]["died"] for leg in LEG_NAMES}
    quals = {leg: qual_of(leg_docs[leg]["full32"]) for leg in LEG_NAMES}
    if not stage_done(events, "quals"):
        log({"event": "quals", "quals": quals, "means": means, "died": died,
             "qualification": "资格 = died≤8(K1 单档依赖标签)∧ 哨兵肢零触碰"
                              "(换层/cap/override)∧ 非作废(D4-1 显式成文)"})
        events.append({"event": "quals"})
    win = decide_winner(means, died, quals)
    winner = win["winner"]
    verdicts = {"course5": course5, "a12": a12_verdict, "winner": winner,
                "golden_authorized": False}
    hold_docs = None
    if winner is None:
        verdicts["launch_limbs"] = {"verdict": "三腿资格全失——无胜者,发射不可考"
                                               "(科学主判已入册,与发射解耦)"}
        log_launch_check_no_winner(events, verdicts["launch_limbs"], quals)
    else:
        if win.get("substituted") and not stage_done(events, "substitution"):
            log({"event": "substitution", "blocked": win["prelim"],
                 "why": quals[win["prelim"]]})
            events.append({"event": "substitution"})
        wrows = by_seed(leg_docs[winner]["full32"]["rows"], 7000, 7031)
        st = paired_diff_stats(wrows, ref_launch)
        w_died = died[winner]
        abandon_pass = means[winner] >= abandon
        limbs = {
            # C 修:发射均差肢消费 mean_raw(raw 语义,v32 先例;17.315 型
            # 贴线读数圆整后 17.32 假过线);落账 paired_mean 仍显示 2dp。
            "paired_mean_ge_17.32": st["mean_raw"] >= G1_PAIRED_DIFF,
            "wins_ge_22of32": st["wins"] >= G1_PAIRED_WINS,
            "died_le_8": w_died <= G1_DIED_MAX,
            "floor_85_92": means[winner] >= floor_line,
            "sentinel_zero_touch": quals[winner]["sentinel_zero_touch"],
            "non_void": not quals[winner]["void"],
        }
        launch = all(limbs.values()) and abandon_pass
        deleverage_note = None
        # 注记口径与裁决肢同源(raw;审计呈报② 值守授权同步——判决与注记禁两口径)
        if DELEVERAGE_Q95 < st["mean_raw"] < G1_PAIRED_DIFF:
            deleverage_note = ("杠杆敏感带:拦截由 7017 单种子噪声贡献决定"
                               "(升格审查限定 3)")
        elif st["mean_raw"] <= DELEVERAGE_Q95:
            deleverage_note = "去杠杆线下同拦,拦截对 7017 杠杆稳健"
        verdicts["launch_limbs"] = {
            "winner": winner, "paired_mean": st["mean"], "wins": st["wins"],
            "died": w_died, "mean": means[winner],
            "lines": {"paired_diff": G1_PAIRED_DIFF,
                      "wins": G1_PAIRED_WINS, "died_max": G1_DIED_MAX,
                      "floor": floor_line, "abandon": abandon},
            "limbs": limbs, "abandon_pass": abandon_pass, "launch": launch,
            "labels": ["单向量线(RG1.2a)", "K1 单档依赖(RG1.3)"],
            "alpha_listing": MULTIPLE_COMPARISON,
            "deleverage_note": deleverage_note,
            "asymmetry_note": "died∈{7,8} 之胜者过发射线而金评结构性落回退档"
                              "(『换届未遂』)系圈 9 预先知会之可预期情形;"
                              "金评死 ≤6 维持(G1 L2-0,发射线不及于金评 P 线)"}
        if not stage_done(events, "launch_check"):
            log({"event": "launch_check", **verdicts["launch_limbs"]})
            events.append({"event": "launch_check"})
        if launch:
            verdicts["golden_authorized"] = True
        if launch and not stage_done(events, "GOLDEN_AUTHORIZED"):
            golden_cmd = (f"{PY} train/eval_assembled.py --worker {npz[winner]} "
                          f"--manager-npz {H_NPZ} --seeds 9000-9031 "
                          f"--tag v33-golden --board")
            log({"event": "GOLDEN_AUTHORIZED", "leg": winner,
                 "mean": means[winner], "died": w_died,
                 "wins": st["wins"], "mean_diff": st["mean"],
                 "worker_npz_sha": sha16(npz[winner]),
                 "golden_cmd": golden_cmd,
                 "handover_expectation": ("died∈{7,8} 换届未遂预期位:"
                                          + ("在位(died="
                                             f"{w_died})" if w_died >= 7
                                             else "不在位")),
                 "notice": "金池烧一次系战役级知会纪律(圈 9 收讫);金牌值守"
                           "手启单臂一次;名分流转以 G0-1 双基线过闸为先决"
                           "(D2-1 降级条款)"})
            attention(f"金牌待手启:{winner}(发射线过;金评死线 ≤6 不对称在案)")
        # H1/H2 留出捆绑对(施于胜者)
        d_h = exam_case(events, npz[winner], HOLDOUT_TAGS[0], POOL_HOLD,
                        bundled_with=HOLDOUT_TAGS[1])
        holdout_account(events, HOLDOUT_TAGS[0])
        d_m = exam_case(events, npz[winner], HOLDOUT_TAGS[1], POOL_HOLD,
                        manager_npz=str(M29_NPZ), bundled_with=HOLDOUT_TAGS[0])
        holdout_account(events, HOLDOUT_TAGS[1])
        hold_docs = {"h": d_h, "m29": d_m}

    # ---- S9 R 线记分卡 ----
    scorecard_stage(events, refs, leg_docs, canary_docs, verdicts, hold_docs)


def _smoke_main():
    """--smoke:只跑 G0-2a/2b 实弹烟测;不落正账,走独立 smoke 台账
    (承 run_b1_infra --smoke 形制;冻结前置,准许脏树)。"""
    global _LEDGER_PATH
    _LEDGER_PATH = SMOKE_LEDGER
    events = read_ledger()
    preflight(events, smoke=True)
    g0_nullintrusion_stage(events)
    g0_funcsmoke_stage(events)
    print("烟测阶段完成(判词见 smoke 台账 G0_NULLINTRUSION/G0_SMOKE;"
          "正账零触碰)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="只跑 G0-2a/2b 烟测(独立 smoke 台账,不落正账)")
    args = ap.parse_args()
    try:
        with exclusive_lock(V33 / ".driver.lock", "v33 驱动"):
            if args.smoke:
                _smoke_main()
            else:
                _main()
    except OutputReservationError as e:
        log({"event": "OPERATIONAL_FAILURE", "why": f"W8 锁冲突: {e}"})
        attention("W8 不空闲/锁冲突:\n" + str(e))
        raise SystemExit(4) from e
    except PreflightFailure as e:
        log({"event": "PREFLIGHT_FAIL", "why": str(e)})
        attention("P4 预检不过,不发车呈报:\n" + str(e))
        raise SystemExit(3) from e
    except OperationalFailure as e:
        log({"event": "OPERATIONAL_FAILURE", "why": str(e)})
        attention("运维失败:\n" + str(e))
        raise SystemExit(2) from e
    except SystemExit:
        raise
    except Exception as e:
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("驱动异常死亡(P1):\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
