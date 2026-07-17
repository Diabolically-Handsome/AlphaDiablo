"""B1「捆绑评测基建」驱动(docs/PREREG-B1-捆绑评测基建.md 条款唯一执行者;
克隆 run_v32_sovereign.py 骨架,常量区承继 v32 全部 W7 常量 + PRIORS 四档
sha + v32 CASE_RUNTIME 五 sha)。

阶段(D2,防后见条款由阶段顺序强制):
  S0 预检(W1/W7/PRIOR_REFERENCE/BC_SD_WAIVER/W-E0/W-H8/304000+307000 台账扫描)
  S1 W-G0-零侵入实弹(种子 307000,2×102,400 步,张量级判据)
  S2 V1 方差发(兼 RB.1 位级承接;失配循 RB.2 带外分诊)
  S3 K1/K2 留出参照两发(8000 池首曝,HOLDOUT_EXPOSURE 按发计)
  S4 CANARY_SET + CRITERION_REGISTER(先登记后开箱)
  S5 P8 腿点火(ctrl 配方逐字,偏离封闭枚举五处;台账制 2 次)
  S6 金丝雀离线序列(全部先于 L1)+ 死门 would-trip 读数(一律只记不裁)
  S7 腿考五发 L1-L5(双经理捆绑通道 + MS_REPORT)
  S8 CRITERION_VALIDATE(8000 池灾难判据验证,schema 钉死)
  S9 R 线记分卡(RB.1-RB.10;判决附录由值守记档)

**本案系测量与仪表化案:零金种子、零名分变动、零发射、零环境侵入;
一切死门读数只记不裁;M29 轨永不作裁决线。**

用法:
  .venv/bin/python train/run_b1_infra.py            # 全案(S0-S9,幂等续跑)
  .venv/bin/python train/run_b1_infra.py --plan     # 只打印施工/发车计划,零副作用
  .venv/bin/python train/run_b1_infra.py --smoke    # 只跑 S1 W-G0(冻结前置,可在
                                                    # 冻结 commit 前实弹;台账入册)
退出码:0 案结/幂等;2 额度耗尽;3 预检;4 锁冲突;5 W-E0 发车前漂移;
6 runtime 案中漂移;7 CASE_HALT_G0;8 REF_DIVERGENCE;其余 P1。
"""
from __future__ import annotations

import argparse
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

from eval_contract import (PROTOCOL_VERSION, OperationalFailure, OutputReservationError,
                           exclusive_lock, expected_eval_identity,
                           freeze_eval_identity, read_eval_archive,
                           strict_json_loads, verify_eval_identity)
from extract_canary_set import extract as extract_depth2

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
B1 = RUNS / "infra-b1"
LEDGER = B1 / "gate_ledger.jsonl"
EVAL = RUNS / "eval-assembled"

# ---- W7 工件钉死(全文 sha 驱动器冻结常量;失配即 P4 不发车) ----
KING_ZIP = ROOT / "train" / "models" / "v28-worker-leg1" / "model_final.zip"
KING_ZIP_SHA = "2f7bc9dd810956c3feeb330575c9a03ddff0b476333ac429a411935985b04f42"
KING_NPZ = ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz"
KING_NPZ_SHA = "976b6c05edaa0a32bb30bd372782e1201c72b029cedcbb3a5bf2361d34f27f8a"
KING_STEPS = 3_497_984
H_NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
H_NPZ_SHA = "0f2264860b0960e7951efd424836b90c09c002cebca7bf8109fd669b13be63d7"  # == DEFAULT_MANAGER_SHA256
M29_NPZ = ROOT / "train" / "models" / "v29-manager-mfresh" / "policy.npz"
M29_NPZ_SHA = "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6"
BC_SD = RUNS / "bc-worker" / "policy_sd.pt"
BC_SD_SHA = "f052067a589cfcdedaf1754ae6d241d736bb97f6fc798683f395c76cb0ff98e6"   # v32 BC_REGEN 落定值(BC_SD_WAIVER 条款,D0-④)
KING_SD = RUNS / "v32" / "king_anchor_sd.pt"
KING_SD_SHA = "009aaad29d2653cde3f4e8ed2fafd8861a0f1f572a140c64118df9e3fa3df35d"  # v32 KING_SD_OK 落定值

# PRIORS 前科档案钉死(指纹对照基准,先封存后引用;PRIOR_REFERENCE 事件入台账)
PRIORS = {
    "v32-ref-launch": (EVAL / "v32-ref-launch.json",
                       "48033577f8f124ae81fc5436eb44e5aa0bf541a4437d7fec99d8f5b1209c71fa",
                       113.0),
    "v32-ref-science": (EVAL / "v32-ref-science.json",
                        "1736185e286c1f2a98f6d4f503c72b40e322dcc31151e0e83068714c1b29f5f0",
                        140.9),
    "v32-ctrl-full32": (EVAL / "v32-ctrl-full32.json",
                        "1c3f683476d1b6124ba2c85473e4f17a12d52984d55582c6f70d3f0b3306e86c",
                        92.3),
    "v32-ctrl-m29": (EVAL / "v32-ctrl-m29.json",
                     "cd198545a2723a114a8ceaf548ce7917ed8ea89355cd15c828895dad622652af",
                     125.2),
}

# W-E0 环境零漂移断言:v32 台账 CASE_RUNTIME 落定值(钉为驱动器冻结常量)
V32_CASE_RT = {
    "bridge": "8c45da3ea4121eab13da2b1a62ba52a189a25e6ea36833fa98d58d0b46140ba1",
    "engine": "be59473ea7db0a122350b179d1509454bd05ebf0e7d07946f24b2f57ff1890ce",
    "game_data": "63fb47d9c76484024c7640d90ab6b7ec5e13f567a7e1a917b6c03a6631d3f2b0",
    "assets": "b344b24f7743a88b5bf3dcc9635d096e41ee14172f687383f46a9002882b954d",
    "protocol": "91beb61e08198f5e4e9f0d13cf01b8025867cd8a474601c9cbd23321edcfcba2",
}
CALIBRATED_PROTOCOL_VERSION = 3

# ---- P8 腿配方常量(D2;ctrl 配方逐字,偏离封闭枚举五处) ----
LEG_STEPS = 499_712
NT_TARGET = 3_997_696
BETA = 0.015625
QUANTUM = 2048                        # 512 n-steps × 4 envs
SEED = 304_000                        # 偏离①:原谱系种子 303000 + 1000(常量,禁种子搜索)
CALIB_PROBES = tuple(KING_STEPS + 49_152 * k for k in range(1, 11))   # 偏离②:十点步表
CKPT_EVERY = 98_304                   # 偏离③(=48×2048)
SENTINEL_EVERY = 49_152               # 偏离④
DRY_ANCHOR_EVERY = 49_152             # 偏离⑤
RUN_NAME = "b1-p8"
CANARY_STEPS = tuple(KING_STEPS + 98_304 * k for k in range(1, 5))
# 机械使然另落之 +491,520 ckpt:照常归档但不入金丝雀序列(点表预登记即封闭)
EXTRA_CKPT_STEP = KING_STEPS + 491_520
CANARY_CONTROLS = (7003, 7011)        # W-C 健康对照(P5 无尖峰,阴性对照)

# ---- W-G0 烟测常量(专用步表写死于案内) ----
SMOKE_SEED = 307_000                  # 非案腿种子(弃 304000,防偷看腿前段遥测)
SMOKE_STEPS = 102_400                 # 50 rollout,整除 2048
SMOKE_END = KING_STEPS + SMOKE_STEPS  # 3,600,384
SMOKE_CALIB = (KING_STEPS + 49_152, KING_STEPS + 98_304)   # 窗内 2 个 calib 点
SMOKE_RUNS = {"bare": "b1-smoke-bare", "knobs": "b1-smoke-knobs"}
SMOKE_TIMEOUT = 3_600

# ---- 死门 would-trip 常量(E3;一律只记不裁,阈值系定标输入) ----
CE_LINE = 0.2                         # 绝对门(饱和预期,系重校目标线非区分器)
RATIO_LINE = 30.0                     # 梯度比门(g_pg/g_ce,β=0.015625 已折入口径)
DRY_REF_THRONE = 0.7515               # 王座级绝对参考(增量门零点)
DRY_REF_LINEAGE = 0.6305              # 谱系级血统参考(v28-leg1 起点,F2-P9 在册)
DRY_TRIP_PP = 2.0                     # would-trip:增量 > +2pp

# ---- 判据常量(D3,全部预登记案中禁调) ----
MS_FLAG_LINE = 20.0
REC_H_LINE = -20.0                    # REC 首肢:Δ_H(s) ≤ −20
REC_RECOVERY = 20.0                   # REC 次肢:Δ_M29(s) ≥ Δ_H(s) + 20
MIN_JUDGEABLE_HITS = 3
TAU_FLOOR_BAND = (25.0, 40.0)

# ---- R 线带常量(D3/D5;闭区间 [lo, hi]) ----
RB3_BAND = {"point": 25.0, "band": (5.0, 50.0)}
RB4_BAND = {"point": -20.0, "band": (-45.0, -2.0),
            "median_line": -5.0, "neg_line": 21}
RB6_MEAN = {"point": 30.0, "band": (8.0, 55.0)}
RB6_MS = {"R": {"max": (60.0, (20.0, 130.0)), "over": (6, (2, 14))},
          "N": {"max": (10.0, (0.0, 30.0)), "over": (1, (0, 4))}}
RB7_BAND = {"point": -18.0, "band": (-50.0, 5.0),
            "d2_point": 7, "d2_band": (2, 14)}
RB8_BAND = {"R": (0.3, (0.0, 0.7)), "N": (0.9, (0.7, 1.0))}
RB9_BAND = {"ratio_median": (40.0, (10.0, 120.0)),
            "ratio_trips": (7, (3, 10)),
            "ce_over_share": (0.8, (0.3, 1.0)),
            "dry_increment_pp": (1.5, (0.0, 4.0))}
RB10_BAND = {"throne_d": (54.0, (40.0, 65.0)),
             "leg": {"R": (79.0, (65.0, 95.0)), "N": (55.0, (40.0, 70.0))}}
REC_PRECISION = {"point": 0.7, "band": (0.4, 1.0)}

POOL_PROBE = "7000-7031"
POOL_S16 = "7000-7015"
POOL_HOLD = "8000-8031"
HOLDOUT_TAGS = ("b1-ref8k-launch", "b1-ref8k-science", "p8-hold8k", "p8-hold8k-m29")
CANARY_TAGS = tuple(f"p8-canary{k}-{m}" for k in range(1, 5) for m in ("h", "m29"))
ALL_EXAM_TAGS = (("b1-varprobe-launch",) + HOLDOUT_TAGS
                 + ("p8-s16", "p8-full32", "p8-full32-m29") + CANARY_TAGS)

CASE_RT: dict | None = None


# ======================================================================
# 台账与通用工具(run_v32_sovereign 逐字承继 + 案级扩展)
# ======================================================================

def log(event: dict):
    B1.mkdir(parents=True, exist_ok=True)
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    B1.mkdir(parents=True, exist_ok=True)
    with open(B1 / "NEEDS_ATTENTION", "a") as f:
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
    B1.mkdir(parents=True, exist_ok=True)
    with open(B1 / logfile, "w") as lf:
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


def zip_steps(p: pathlib.Path) -> int:
    try:
        with zipfile.ZipFile(p) as z:
            return int(json.loads(z.read("data"))["num_timesteps"])
    except Exception:
        return 0


def read_ledger() -> list[dict]:
    if not LEDGER.is_file():
        return []
    out = []
    for i, line in enumerate(LEDGER.read_text().splitlines(), 1):
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


def by_seed(rows, lo, hi) -> dict:
    m = {r["seed"]: r for r in rows}
    require(len(rows) == len(m), "种子集合异常(含重复 seed)")
    require(set(m) == set(range(lo, hi + 1)), f"种子集合异常(须为 {lo}-{hi})")
    return m


def depth2_count(rows) -> int:
    return sum(1 for r in rows if r["depth"] >= 2)


def d_windows(row) -> int:
    return str(row["mode_seq"]).count("D")


def impl_bundle_sha16() -> str:
    import train_ppo
    return train_ppo._implementation_bundle_sha256()[:16]


# ======================================================================
# 统计纪律工具(D3 硬约束;纯函数,单测覆盖)
# ======================================================================

def median(xs) -> float:
    s = sorted(xs)
    n = len(s)
    require(n > 0, "median 输入为空")
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def binom_tail_ge(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k),精确二项。"""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def sign_test(diffs) -> dict:
    neg = sum(1 for d in diffs if d < 0)
    pos = sum(1 for d in diffs if d > 0)
    ties = len(diffs) - neg - pos
    n = neg + pos
    p = binom_tail_ge(max(neg, pos), n) if n else 1.0
    return {"neg": neg, "pos": pos, "ties": ties,
            "p_one_sided": round(p, 4)}


def deleveraged_mean(diff_by_seed: dict) -> dict:
    """留一去最大杠杆种子后重算均值,并列名该种子(F2 头条勘正②)。"""
    require(len(diff_by_seed) >= 2, "去杠杆均值需要 ≥2 种子")
    lever = max(diff_by_seed, key=lambda s: abs(diff_by_seed[s]))
    rest = [v for s, v in diff_by_seed.items() if s != lever]
    return {"dropped_seed": lever,
            "dropped_value": round(diff_by_seed[lever], 2),
            "mean": round(sum(rest) / len(rest), 2)}


def band_judge(x: float, lo: float, hi: float, integer: bool = False) -> dict:
    """闭区间带判读 + 临线条款(D3:连续量 |x−边界|≤0.05×带宽;计数量距边界≤1)。"""
    in_band = lo <= x <= hi
    if integer:
        borderline = min(abs(x - lo), abs(x - hi)) <= 1
    else:
        borderline = min(abs(x - lo), abs(x - hi)) <= 0.05 * (hi - lo)
    out = {"x": round(float(x), 4), "band": [lo, hi], "in_band": in_band}
    if borderline:
        out["borderline_note"] = "临线 + 线未重标(D3 临线条款强制注记)"
    return out


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """精确二项 95% 区间(D3:precision 判词强制携带,禁裸点数)。
    下界:cdf(k−1, p) = 1 − α/2 之解;上界:cdf(k, p) = α/2 之解;
    cdf(kk, p) 对 p 单调递减,二分即得。"""
    require(0 <= k <= n and n > 0, "clopper_pearson 输入非法")

    def cdf(kk: int, p: float) -> float:      # P(X <= kk | n, p)
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                   for i in range(0, kk + 1))

    def solve(kk: int, target: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if cdf(kk, mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(k - 1, 1 - alpha / 2)
    upper = 1.0 if k == n else solve(k, alpha / 2)
    return (round(lower, 4), round(upper, 4))


def ms_vectors(leg_h: dict, leg_m29: dict, ref_h: dict, ref_m29: dict) -> dict:
    """MS(s) := |Δ_H(s) − Δ_M29(s)|,Δ_m(s) = ret_leg,m − ret_ref,m(同经理配对)。
    并列带符号量 Δ_H−Δ_M29(承统计 m-7,防'M29 侧更差'型翻转被绝对值吞没)。"""
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
# 评测机械(exam_or_adopt 骨架承继 + P2 台账制额度 + E1 捆绑通道)
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
    # E1 纪律:--manager-npz 逐次显式,禁默认回落;本案一切发无 --board
    cmd = [PY, "train/eval_assembled.py", "--worker", str(worker_arg),
           "--manager-npz", snapshot["manager"]["path"],
           "--seeds", seeds, "--tag", tag]
    if run(cmd, f"exam-{tag}.{time.time_ns()}.log", timeout=1_800) != 0:
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    try:
        d = read_eval_archive(out, **expected)
        verify_eval_identity(snapshot, ROOT)
    except (OSError, KeyError, TypeError, ValueError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
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
    """考发终局条款(v32 逐字)+ P2 台账制额度(每发 FIRING_START 计 2 次)。"""
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
                ok["bundled_with"] = bundled_with     # E1:exam_ok 强制加携对侧 tag
            if extra:
                ok.update(extra)
            log(ok)
            events.append({"event": "exam_ok", "tag": tag, "sha": a["_sha"]})
            return d
        log({"event": "exam_crash", "tag": tag,
             "note": "评测失败,按 P2 额度重考(发车前断言失败一律停机不重试)"})


def holdout_account(events, tag):
    """HOLDOUT_EXPOSURE 按发计(W-H8 立法;暴露预算不可再生)。"""
    if any(e.get("event") == "HOLDOUT_EXPOSURE" and e.get("tag") == tag
           for e in events):
        return
    n = sum(1 for e in events if e.get("event") == "HOLDOUT_EXPOSURE") + 1
    ev = {"event": "HOLDOUT_EXPOSURE", "tag": tag, "cumulative_shots": n,
          "note": "按发计;本案首曝 2 对/4 发(D0-② 亲批);"
                  "'每案至多一对捆绑发'自本案案结后生效"}
    log(ev)
    events.append(ev)


def bundled_exam(events, worker, tag_h, tag_m29, seeds, holdout=False):
    """E1 双经理捆绑评测通道(驱动器级):每次工人考发自动 H 与 M29 成对。"""
    d_h = exam_case(events, worker, tag_h, seeds, manager_npz=None,
                    bundled_with=tag_m29)
    if holdout:
        holdout_account(events, tag_h)
    d_m = exam_case(events, worker, tag_m29, seeds, manager_npz=str(M29_NPZ),
                    bundled_with=tag_h)
    if holdout:
        holdout_account(events, tag_m29)
    return d_h, d_m


def ms_report(events, pool, tag_h, tag_m29, leg_h, leg_m29, ref_h, ref_m29,
              ref_names):
    if any(e.get("event") == "MS_REPORT" and e.get("pool") == pool
           for e in events):
        return
    v = ms_vectors(leg_h, leg_m29, ref_h, ref_m29)
    var_zero = stage_done(events, "VARPROBE")
    ev = {"event": "MS_REPORT", "pool": pool, "tags": [tag_h, tag_m29],
          "refs": ref_names, **v,
          "family_note": "MS 逐种子挂旗系 32 次比较之族(族账注记,承统计 M-6)"
                         + (";RB.2 判方差=0 → 旗无测量噪声假阳" if var_zero
                            else ""),
          "ms_limitation": "MS 限定(强制随判词):M29 与 H 共享 exhausted 盲区"
                           "与 7017 型选窗盲点,MS≈0 不证工人无损——MS 系"
                           "'经理敏感损伤'探测器,非全损伤探测器"}
    log(ev)
    events.append(ev)


# ======================================================================
# E4 重放通道(OBS_DRIFT)与 E2 金丝雀
# ======================================================================

def run_obsdrift(worker_npz, archive: pathlib.Path, out: pathlib.Path,
                 manager=None) -> dict:
    """先落档案、后重放对账(保真锚顺序写死,承工程 m-8);报告文件幂等采信。"""
    if out.exists():
        report = strict_json_loads(out.read_bytes())
        if (report.get("archive_sha256") == sha256(archive)
                and report.get("fidelity_ok")):
            return report
        out.rename(out.with_suffix(f".{time.time_ns()}.stale"))
    cmd = [PY, "train/probe_b1_obsdrift.py", "--worker", str(worker_npz),
           "--archive", str(archive), "--out", str(out)]
    if manager:
        cmd += ["--manager", str(manager)]
    rc = run(cmd, f"obsdrift-{out.stem}.{time.time_ns()}.log", 3_600)
    require(rc == 0 and out.exists(),
            f"E4 重放失败/保真失配:{archive.name}(保真锚条款,停机呈报)")
    return strict_json_loads(out.read_bytes())


def throne_replay() -> dict:
    return run_obsdrift(KING_NPZ, PRIORS["v32-ref-launch"][0],
                        B1 / "replay" / "throne-ref-launch.json")


def walk_subset_mean(report: dict, seeds) -> float | None:
    entries = [report["per_seed"][str(s)]["walk_end"] for s in seeds
               if str(s) in report["per_seed"]]
    total = sum(e["n"] for e in entries)
    if not total:
        return None
    return round(sum(e["walk_sum_over_121_mean"] * e["n"]
                     for e in entries if e["n"]) / total, 3)


def obs_drift_event(events, at: str, leg_report: dict, throne_report: dict,
                    d_c_seeds):
    if any(e.get("event") == "OBS_DRIFT" and e.get("at") == at
           for e in events):
        return
    diffs = {}
    for dim, leg_stats in leg_report["appended8_window_end"].items():
        th = throne_report["appended8_window_end"][dim]
        diffs[dim] = (None if leg_stats is None or th is None
                      else round(leg_stats["P50"] - th["P50"], 6))
    ev = {"event": "OBS_DRIFT", "at": at,
          "schema": "8 追加维逐维 {mean,std,min,max,P5,P50,P95} + 窗末 walkable"
                    " 体态 {walkΣ/121 窗末均值, 西南带均值}(局部图下标 44-164);"
                    "漂移读数 := 腿侧对王座侧逐维 P50 差(封闭枚举)",
          "appended8_leg": leg_report["appended8_window_end"],
          "appended8_throne": throne_report["appended8_window_end"],
          "p50_diff_leg_minus_throne": diffs,
          "walk": {"throne_d_decision": throne_report["walk_d_decision"],
                   "leg_window_end_all": leg_report["walk_window_end"],
                   "leg_window_end_dc_seeds": {
                       "seeds": sorted(d_c_seeds),
                       "walk_sum_over_121_mean": walk_subset_mean(
                           leg_report, d_c_seeds)}},
          "reports": {"leg_sha16": hashlib.sha256(json.dumps(
                          leg_report, sort_keys=True).encode()).hexdigest()[:16],
                      "throne_sha16": hashlib.sha256(json.dumps(
                          throne_report, sort_keys=True).encode()).hexdigest()[:16]},
          "note": "P1 改判限定:仪表维(8 追加维)仅翻边际窗,体态维系主嫌;"
                  "F-lock 机制在基础 295 维体态,监控面双覆盖;只记不裁"}
    log(ev)
    events.append(ev)


def canary_eval_done(events, tag) -> dict | None:
    hits = [e for e in events
            if e.get("event") == "CANARY_EVAL" and e.get("tag") == tag]
    return hits[-1] if hits else None


def canary_exam(events, worker_npz, tag, manager_npz, l1_fired: bool):
    """E2 金丝雀发:不占评测发额度;同 ckpt×manager 落档至多一条即终局;
    重试仅限无档案运维失败(计数入 OPERATIONAL-canary);补评截止 = 首个 L1
    FIRING_START 之前;失败只记不停腿(P-canary)。"""
    out = EVAL / f"{tag}.json"
    prior = canary_eval_done(events, tag)
    if prior is not None:
        require(out.exists(), f"金丝雀 {tag} 台账在册而档案缺失,停机呈报")
        d = validate_adopted(tag, worker_npz, POOL_PROBE, manager_npz)
        require(d["agg"]["_sha"] == prior["sha"],
                f"金丝雀 {tag} 档案与台账 sha 失配")
        return d, False
    if out.exists():
        # 同一驱动进程在落档与记账之间崩溃的残档:身份复验后采信(终局条款)
        d = validate_adopted(tag, worker_npz, POOL_PROBE, manager_npz)
        return d, True
    if l1_fired:
        log({"event": "OPERATIONAL-canary", "tag": tag,
             "why": "补评截止已过(首个 L1 FIRING_START 之前),禁补——"
                    "该点 R 线判'不可判,如实登记'"})
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
         "why": "重试后仍失败——只记不停腿,该点 R 线判'不可判'"})
    return None, False


def canary_stage(events, d_c_seeds):
    """S6:金丝雀离线序列(检查点离线通道;只记不裁;全部先于 L1)。"""
    l1_fired = firing_count(events, "p8-s16") > 0
    throne_rep = throne_replay()
    canary_docs = {}
    for k, step in enumerate(CANARY_STEPS, 1):
        ckpt = RUNS / RUN_NAME / "ckpt" / f"model_{step}_steps.zip"
        npz = B1 / "canary" / f"policy_{step}.npz"
        if not npz.exists():
            require(ckpt.is_file(), f"金丝雀 ckpt 缺失:{ckpt}")
            # 案外独立进程逐 ckpt 离线导出(现成 export_worker_npz,自带 parity)
            rc = run([PY, "train/export_worker_npz.py", str(ckpt), str(npz)],
                     f"canary-export-{step}.log", 600)
            if rc != 0 or not npz.exists():
                log({"event": "OPERATIONAL-canary", "ckpt_step": step,
                     "why": f"npz 导出失败(rc={rc}),该点全部读数'不可判'"})
                continue
        replay_rep = None
        for mtag, manager in (("h", None), ("m29", str(M29_NPZ))):
            tag = f"p8-canary{k}-{mtag}"
            d, fresh = canary_exam(events, str(npz), tag, manager, l1_fired)
            if d is None:
                continue
            canary_docs[tag] = d
            if mtag == "h":
                replay_rep = run_obsdrift(
                    npz, EVAL / f"{tag}.json",
                    B1 / "replay" / f"{tag}.json")
                obs_drift_event(events, tag, replay_rep, throne_rep, d_c_seeds)
            if fresh or canary_eval_done(events, tag) is None:
                rows = sorted(d["rows"], key=lambda r: r["seed"])
                tau_note = None
                if mtag == "h" and replay_rep is not None:
                    tau_note = {str(r["seed"]): replay_rep["per_seed"]
                                .get(str(r["seed"]), {}).get("tau_floor_distance")
                                for r in rows}
                ev = {"event": "CANARY_EVAL", "tag": tag, "k": k,
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
                      "tau_floor_distance_footnote": tau_note,
                      "footnote": "τ 中位距 25 地板之近失距离(承统计 m-9,"
                                  "只记不裁;P5 型重放取证明示出界不做;"
                                  "τ 中位系 H 侧 E4 重放口径)",
                      "discipline": "只记不裁;禁作任何腿内干预依据;"
                                    "终点检查点唯一 = nt 3,997,696"}
                log(ev)
                events.append({"event": "CANARY_EVAL", "tag": tag,
                               "sha": d["agg"]["_sha"]})
    return canary_docs


def deadgate_stage(events):
    """S6b:死门 would-trip 双读数(E3;一律只记不裁,阈值系定标输入)。"""
    leg_dir = RUNS / RUN_NAME
    calib_path = leg_dir / "calib.jsonl"
    require(calib_path.is_file(), "calib.jsonl 缺失(E3 十点步表未产出)")
    done_steps = {e.get("step") for e in events
                  if e.get("event") == "GATE_WOULD_TRIP"
                  and e.get("gate") == "calib"}
    ratios = []
    ce_values = []
    for line in calib_path.read_text().splitlines():
        rec = json.loads(line)
        ratio = (rec["g_pg"] / rec["g_ce"]) if rec["g_ce"] else None
        ratios.append(ratio)
        ce_values.append(rec["distill_ce"])
        if rec["step"] in done_steps:
            continue
        ev = {"event": "GATE_WOULD_TRIP", "gate": "calib", "step": rec["step"],
              "distill_ce": rec["distill_ce"], "ce_line": CE_LINE,
              "would_trip_ce": rec["distill_ce"] > CE_LINE,
              "g_pg": rec["g_pg"], "g_ce": rec["g_ce"],
              "grad_ratio": None if ratio is None else round(ratio, 2),
              "ratio_line": RATIO_LINE,
              "would_trip_ratio": (ratio is not None and ratio > RATIO_LINE),
              "口径": "比值 := calib.jsonl 之 g_pg/g_ce,g_ce = ‖∇(β·distill_ce)‖"
                     "(β=0.015625 已折入;字面裸 CE 梯度读法差 64×,禁用)",
              "note": "只记不裁;绝对门系重校目标线非区分器,饱和即预期,"
                      "判词禁作异常叙事(承工程 m-2)"}
        log(ev)
        events.append(ev)
    sentinel_path = leg_dir / "sentinel.jsonl"
    require(sentinel_path.is_file(), "sentinel.jsonl 缺失(E3 哨兵加密未产出)")
    dry_lines = [json.loads(x) for x in sentinel_path.read_text().splitlines()
                 if '"dry-anchor"' in x]
    done_dry = {e.get("step") for e in events
                if e.get("event") == "GATE_WOULD_TRIP"
                and e.get("gate") == "dry-anchor"}
    for rec in dry_lines:
        if rec["step"] in done_dry:
            continue
        inc_pp = round((rec["mismatch"] - DRY_REF_THRONE) * 100, 2)
        ev = {"event": "GATE_WOULD_TRIP", "gate": "dry-anchor",
              "step": rec["step"], "mismatch": rec["mismatch"],
              "ref_throne": DRY_REF_THRONE, "ref_lineage": DRY_REF_LINEAGE,
              "increment_pp": inc_pp, "trip_line_pp": DRY_TRIP_PP,
              "would_trip": inc_pp > DRY_TRIP_PP,
              "note": "双参考钉死(承统计 M-4/工程 m-3):增量门零点 0.7515"
                      "(王座级);血统参考 0.6305(v28-leg1 起点)——漂移早于"
                      " v32 之 +12pp 血统事实随定标数据入册,防误导后案挂闸;"
                      "回调跨界后首发,首点非入场步,如实登记;只记不裁"}
        log(ev)
        events.append(ev)
    return {"ratios": ratios, "ce_values": ce_values, "dry_lines": dry_lines}


# ======================================================================
# S1 W-G0-零侵入实弹(冻结先决;张量级判据)
# ======================================================================

def _smoke_cmd(variant: str) -> list[str]:
    """烟测命令(ctrl 配方逐字,仅换种子/步数;knobs 侧加全套仪表旋钮)。"""
    cmd = [PY, "train/train_ppo.py", "--worker", "--algo", "mppo",
           "--gamma", "1.0", "--max-steps", "3000", "--n-steps", "512",
           "--num-envs", "4", "--lr", "3e-4", "--ent-coef", "0.005",
           "--seed", str(SMOKE_SEED), "--total-steps", str(SMOKE_STEPS),
           "--run-name", SMOKE_RUNS[variant],
           "--distill-beta", str(BETA),
           "--teacher-sd", str(BC_SD), "--teacher-override", str(KING_SD),
           "--skip-dry", "--manager-npz", str(M29_NPZ),
           "--resume-from", str(KING_ZIP), "--allow-legacy-resume",
           "--no-drink-sovereignty", "--calib-record-only"]
    if variant == "knobs":
        cmd += ["--calib-probes", ",".join(str(x) for x in SMOKE_CALIB),
                "--ckpt-every-steps", str(CKPT_EVERY),
                "--sentinel-every", str(SENTINEL_EVERY),
                "--dry-anchor-every", str(DRY_ANCHOR_EVERY)]
    return cmd


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


def _state_digest(policy, optim) -> str:
    import torch
    h = hashlib.sha256()

    def eat(node, path):
        if isinstance(node, torch.Tensor):
            h.update(path.encode())
            h.update(str(node.dtype).encode())
            h.update(str(tuple(node.shape)).encode())
            h.update(node.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(node, dict):
            for k in sorted(node, key=str):
                eat(node[k], f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, x in enumerate(node):
                eat(x, f"{path}[{i}]")
        else:
            h.update(f"{path}={node!r}".encode())

    eat(policy, "policy")
    eat(optim, "optim")
    return h.hexdigest()


def _progress_lines(run_dir: pathlib.Path) -> list[dict]:
    lines = []
    for raw in (run_dir / "progress.jsonl").read_text().splitlines():
        rec = json.loads(raw)
        rec.pop("t", None)                    # 墙钟字段非 RNG 相关,剔除
        lines.append(rec)
    return lines


def _final_sentinel_lines(run_dir: pathlib.Path) -> dict:
    out = {}
    p = run_dir / "sentinel.jsonl"
    if p.is_file():
        for raw in p.read_text().splitlines():
            rec = json.loads(raw)
            if rec.get("final"):
                out[rec["sentinel"]] = rec
    return out


def smoke_stage(events) -> None:
    """W-G0-零侵入(E 项旋钮之冻结先决,实弹)。幂等:PASS 事件且 impl 束
    sha16 与当前一致即免跑;FAIL → CASE_HALT_G0(退出码 7)不冻结不发车。"""
    impl16 = impl_bundle_sha16()
    for e in events:
        if (e.get("event") == "G0_NULLINTRUSION" and e.get("verdict") == "PASS"
                and e.get("impl_sha16") == impl16):
            print(f"W-G0 已过(impl {impl16}),幂等跳过", flush=True)
            return
    # 烟测前置断言(工件面;不含 W1 git 断言——烟测系冻结前置,准许脏树实弹)
    pre(KING_ZIP.is_file() and sha256(KING_ZIP) == KING_ZIP_SHA, "王 zip 漂移")
    pre(zip_steps(KING_ZIP) == KING_STEPS, "王 zip 步数账异常")
    pre(sha256(M29_NPZ) == M29_NPZ_SHA, "M29 npz 漂移")
    pre(sha256(BC_SD) == BC_SD_SHA, "BC_SD 与 v32 BC_REGEN 落定值不符")
    pre(sha256(KING_SD) == KING_SD_SHA, "KING_SD 与 v32 落定值不符")
    scan_foreign_ledgers_for_seeds((SMOKE_SEED,))
    pre(not any(lo <= SMOKE_SEED + rank <= hi
                for rank in range(4)
                for lo, hi in ((7000, 7031), (8000, 8031), (9000, 9031))),
        "烟测种子撞评测池")
    snapshot = freeze_eval_identity(ROOT, str(KING_NPZ), None)
    if runtime_five(snapshot) != V32_CASE_RT:
        log({"event": "CASE_HALT_ENV_DRIFT", "where": "smoke",
             "expected": V32_CASE_RT, "current": runtime_five(snapshot)})
        attention("W-E0 环境零漂移断言失败(烟测前),不冻结不发车")
        raise SystemExit(5)

    results = {}
    for variant in ("bare", "knobs"):
        run_dir = RUNS / SMOKE_RUNS[variant]
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
            attention("W-G0 烟测运行失败,不冻结不发车")
            raise SystemExit(7)

    # 仪表确实跑了(“仪表被证零侵入”不得实为“仪表根本没跑”)
    knobs_dir = RUNS / SMOKE_RUNS["knobs"]
    bare_dir = RUNS / SMOKE_RUNS["bare"]
    evidence = {
        "knobs_ckpt": sorted(p.name for p in (knobs_dir / "ckpt").glob("*.zip"))
        if (knobs_dir / "ckpt").is_dir() else [],
        "bare_ckpt": sorted(p.name for p in (bare_dir / "ckpt").glob("*.zip"))
        if (bare_dir / "ckpt").is_dir() else [],
        "knobs_calib_lines": (len((knobs_dir / "calib.jsonl").read_text()
                                  .splitlines())
                              if (knobs_dir / "calib.jsonl").is_file() else 0),
        "bare_calib_lines": (len((bare_dir / "calib.jsonl").read_text()
                                 .splitlines())
                             if (bare_dir / "calib.jsonl").is_file() else 0),
        "knobs_sentinel_lines": len((knobs_dir / "sentinel.jsonl").read_text()
                                    .splitlines()),
        "bare_sentinel_lines": len((bare_dir / "sentinel.jsonl").read_text()
                                   .splitlines()),
    }
    instrument_ok = (
        f"model_{KING_STEPS + CKPT_EVERY}_steps.zip" in evidence["knobs_ckpt"]
        and evidence["knobs_calib_lines"] >= 1
        and not evidence["bare_ckpt"] and evidence["bare_calib_lines"] == 0
        and sum(1 for raw in (knobs_dir / "sentinel.jsonl").read_text()
                .splitlines()
                if json.loads(raw).get("sentinel") == "v23"
                and json.loads(raw)["step"] < SMOKE_END) >= 1
        and sum(1 for raw in (knobs_dir / "sentinel.jsonl").read_text()
                .splitlines()
                if json.loads(raw).get("sentinel") == "dry-anchor"
                and json.loads(raw)["step"] < SMOKE_END) >= 1)

    # 张量级判据:policy state_dict + optimizer state 逐张量 torch.equal
    pol_b, opt_b = _load_zip_states(bare_dir / "model_final.zip")
    pol_k, opt_k = _load_zip_states(knobs_dir / "model_final.zip")
    tensor_diffs: list[str] = []
    _tree_equal(pol_b, pol_k, "policy", tensor_diffs)
    _tree_equal(opt_b, opt_k, "optim", tensor_diffs)
    digest_bare = _state_digest(pol_b, opt_b)
    digest_knobs = _state_digest(pol_k, opt_k)

    # RNG 相关遥测逐字段(step/loss/ret 等;墙钟剔除)
    telemetry_diffs: list[str] = []
    prog_b, prog_k = _progress_lines(bare_dir), _progress_lines(knobs_dir)
    if len(prog_b) != len(prog_k):
        telemetry_diffs.append(
            f"progress 行数 {len(prog_b)} != {len(prog_k)}")
    else:
        for i, (a, b) in enumerate(zip(prog_b, prog_k)):
            if a != b:
                telemetry_diffs.append(f"progress[{i}]: {a} != {b}")
                if len(telemetry_diffs) >= 20:
                    break
    status_fields = ("total_steps", "leg_steps", "episodes", "target_reached")
    st_b = json.loads((bare_dir / "status.json").read_text())
    st_k = json.loads((knobs_dir / "status.json").read_text())
    for f in status_fields:
        if st_b[f] != st_k[f]:
            telemetry_diffs.append(f"status.{f}: {st_b[f]} != {st_k[f]}")
    fin_b, fin_k = _final_sentinel_lines(bare_dir), _final_sentinel_lines(knobs_dir)
    for name in ("v23", "dry-anchor"):
        if name in fin_b and name in fin_k:
            if fin_b[name] != fin_k[name]:
                telemetry_diffs.append(
                    f"final {name}: {fin_b[name]} != {fin_k[name]}")
        else:
            telemetry_diffs.append(f"final {name} 行缺席: "
                                   f"bare={name in fin_b}, knobs={name in fin_k}")

    verdict = ("PASS" if not tensor_diffs and not telemetry_diffs
               and instrument_ok else "FAIL")
    ev = {"event": "G0_NULLINTRUSION", "verdict": verdict,
          "impl_sha16": impl16, "seed": SMOKE_SEED,
          "steps": SMOKE_STEPS, "window": [KING_STEPS, SMOKE_END],
          "smoke_step_table": {
              "calib": list(SMOKE_CALIB),
              "ckpt": [KING_STEPS + CKPT_EVERY],
              "sentinel_first": ((KING_STEPS // SENTINEL_EVERY) + 1)
              * SENTINEL_EVERY,
              "dry_anchor_first": ((KING_STEPS // DRY_ANCHOR_EVERY) + 1)
              * DRY_ANCHOR_EVERY},
          "runs": results,
          "tensor_verdict": ("逐张量 torch.equal 全等" if not tensor_diffs
                             else tensor_diffs[:20]),
          "state_digest_sha": {"bare": digest_bare, "knobs": digest_knobs},
          "telemetry_verdict": ("RNG 相关字段逐字段相等" if not telemetry_diffs
                                else telemetry_diffs[:20]),
          "instruments_actually_fired": instrument_ok,
          "instrument_evidence": evidence,
          "criteria_note": "文件字节级比对废除(SB3 zip 成员时间戳与 pickle "
                           "序列化不可复现);烟测遥测除位级判据外禁作任何 "
                           "go/no-go 或回炉输入;短程外推全程之 IO/时钟理论"
                           "残余入残余⑨"}
    log(ev)
    events.append(ev)
    if verdict != "PASS":
        attention("W-G0 零侵入失败:仪表回炉重设计,不冻结不发车")
        raise SystemExit(7)


# ======================================================================
# S0 预检(W 线)
# ======================================================================

def scan_foreign_ledgers_for_seeds(seeds: tuple[int, ...]):
    """预检断言:目标种子未见于历案台账任一 leg_start(命中 → 停机呈报)。"""
    hits = []
    for ledger in sorted(RUNS.glob("*/gate_ledger.jsonl")):
        if ledger == LEDGER:
            continue
        for line in ledger.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "leg_start" and e.get("seed") in seeds:
                hits.append({"ledger": str(ledger), "seed": e["seed"]})
    pre(not hits, f"种子已见于历案台账 leg_start,停机呈报(禁顺延取值): {hits}")


def holdout_virgin_scan(events):
    """W-H8 留出池处女断言(登记面;历史未登记之手工暴露不可探测,残余⑭)。"""
    pattern = re.compile(r"\b80(?:[0-2][0-9]|3[01])\b")
    sanctioned = set()
    for tag in HOLDOUT_TAGS:
        if any(e.get("event") in ("exam_ok", "FIRING_START")
               and e.get("tag") == tag for e in events):
            sanctioned.add(f"{tag}.json")
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
    for ledger in sorted(RUNS.glob("*/gate_ledger.jsonl")):
        if ledger == LEDGER:
            continue
        for line in ledger.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") in ("leg_start", "exam_ok", "FIRING_START"):
                seed_val = e.get("seed")
                if isinstance(seed_val, int) and 8000 <= seed_val <= 8031:
                    offenders.append(f"{ledger.parent.name}: {e}")
    pre(not offenders, f"W-H8 留出池处女断言失败,停机呈报: {offenders}")
    if not stage_done(events, "HOLDOUT_FIRSTBURN"):
        ev = {"event": "HOLDOUT_FIRSTBURN",
              "pairs": 2, "shots": 4, "tags": list(HOLDOUT_TAGS),
              "authorization": "D0-② 亲批(2026-07-17『三个都准了』);"
                               "按发计入 HOLDOUT_EXPOSURE;"
                               "'每案至多一对捆绑发'自本案案结后生效",
              "evidence_grade": "登记面处女断言(承统计 m-5,残余⑭)"}
        log(ev)
        events.append(ev)


def preflight(events):
    global CASE_RT
    pre(PROTOCOL_VERSION == CALIBRATED_PROTOCOL_VERSION, "契约版本漂移")
    # ---- W1 冻结公证 ----
    dirty = [l for l in git("status", "--porcelain").splitlines()
             if l != "?? train/leaderboard-assembled-v3.md"]
    pre(not dirty, f"W1: 工作树不净 {dirty}")
    head = git("rev-parse", "HEAD")
    for path in ("docs/PREREG-B1-捆绑评测基建.md", "train/run_b1_infra.py"):
        touch = git("log", "-1", "--format=%H", "--", path)
        pre(touch == head, f"W1: {path} 最后触碰 != HEAD")
    freezes = [e for e in events if e.get("event") == "FREEZE_SHA"]
    if freezes and freezes[-1]["sha"] != head:
        reason_file = B1 / "REFREEZE_REASON"
        pre(reason_file.is_file(),
            "W1: HEAD != 台账最后 FREEZE_SHA(链式重冻结须 REFREEZE_REASON 文件)")
        ev = {"event": "FREEZE_SHA", "sha": head,
              "prev_sha": freezes[-1]["sha"],
              "reason": reason_file.read_text().strip()}
        log(ev)
        events.append(ev)
        freezes.append(ev)
    # ---- W7 工件钉死 ----
    pre(KING_ZIP.is_file() and sha256(KING_ZIP) == KING_ZIP_SHA, "王 zip 漂移")
    pre(sha256(KING_NPZ) == KING_NPZ_SHA, "王 npz 漂移")
    pre(sha256(H_NPZ) == H_NPZ_SHA, "H npz 漂移(!= DEFAULT_MANAGER_SHA256)")
    pre(sha256(M29_NPZ) == M29_NPZ_SHA, "M29 npz 漂移")
    pre(zip_steps(KING_ZIP) == KING_STEPS, "王 zip 步数账异常")
    pre(KING_SD.is_file() and sha256(KING_SD) == KING_SD_SHA,
        "KING_SD 漂移(沿用 v32 件)")
    pre(run([PY, "train/check_teacher_parity.py", str(KING_SD),
             str(KING_NPZ)], "parity-king.log", 600) == 0,
        "G-KL-W:王锚 sd 与 npz 宣誓失败(案内 0/1000 复验)")
    # BC_SD 沿用 v32 重生成件之显式裁量条款(承合规 M-2;非默认承继)
    pre(BC_SD.is_file() and sha256(BC_SD) == BC_SD_SHA,
        "BC_SD 与 v32 BC_REGEN 落定值不符")
    if not stage_done(events, "BC_SD_WAIVER"):
        ev = {"event": "BC_SD_WAIVER", "bc_sd_sha256": BC_SD_SHA,
              "basis": "BC 生成链不经训练驱动层旋钮 ∧ runtime_five 零漂移"
                       "(W-E0)——案级裁量豁免,随冻结呈报单列(D0-④)"}
        log(ev)
        events.append(ev)
    # PRIORS 前科档案钉死(先封存后引用)
    for name, (path, expected, mean) in PRIORS.items():
        pre(path.is_file() and sha256(path) == expected,
            f"PRIORS 档案钉死失配:{name}")
        doc = strict_json_loads(path.read_bytes())
        pre(doc["agg"]["ret_mean"] == mean,
            f"PRIORS {name} ret_mean != 台账落定值 {mean}")
    if not stage_done(events, "PRIOR_REFERENCE"):
        ev = {"event": "PRIOR_REFERENCE",
              "priors": {n: {"sha256": s, "ret_mean": m}
                         for n, (_, s, m) in PRIORS.items()},
              "note": "v32 四档指纹对照基准,先封存后引用"}
        log(ev)
        events.append(ev)
    # ---- W-E0 环境零漂移断言(runtime_five == v32 CASE_RUNTIME 落定值) ----
    snapshot = freeze_eval_identity(ROOT, str(KING_NPZ), None)
    five = runtime_five(snapshot)
    if five != V32_CASE_RT:
        log({"event": "CASE_HALT_ENV_DRIFT", "expected": V32_CASE_RT,
             "current": five})
        attention("W-E0 环境+评测协议束漂移,不冻结不发车,呈报")
        raise SystemExit(5)
    CASE_RT = five
    prior_rt = [e for e in events if e.get("event") == "CASE_RUNTIME"]
    if prior_rt:
        pre(prior_rt[0]["five"] == five, "W10: 续跑运行时对账失配")
    else:
        log({"event": "CASE_RUNTIME", "five": five,
             "w_e0": "与 v32 CASE_RUNTIME 逐字相等(环境与评测协议束零漂移)"})
        events.append({"event": "CASE_RUNTIME", "five": five})
    # ---- 种子纪律扫描(304000/307000)+ W-H8 ----
    scan_foreign_ledgers_for_seeds((SEED, SMOKE_SEED))
    pre(not any(lo <= SEED + rank <= hi for rank in range(4)
                for lo, hi in ((7000, 7031), (8000, 8031), (9000, 9031))),
        "W-H8: 训练种子撞评测池(驱动器侧断言,train_ppo 守卫仅含 7000/9000 段)")
    pre(SEED == 303_000 + 1_000, "种子选择规则失守(D2 ①:原谱系种子+1000)")
    holdout_virgin_scan(events)
    # ---- W9 目标档案先决 ----
    for t in ALL_EXAM_TAGS:
        has_ledger = any(e.get("event") in ("exam_ok", "CANARY_EVAL")
                         and e.get("tag") == t for e in events)
        if not has_ledger:
            adoptable = (t in CANARY_TAGS)   # 金丝雀残档走终局采信条款
            if not adoptable:
                pre(not (EVAL / f"{t}.json").exists(),
                    f"W9: 目标档案已存在:{t}(重启协议:先 .void)")
    if leg_starts(events, RUN_NAME) == 0:
        pre(not (RUNS / RUN_NAME).exists(), f"运行目录残留:{RUN_NAME}")
    if not freezes:
        log({"event": "FREEZE_SHA", "sha": head})
    log({"event": "preflight_ok", "king_zip": KING_ZIP_SHA[:16],
         "bc_sd": BC_SD_SHA[:16], "king_sd": KING_SD_SHA[:16],
         "impl_sha16": impl_bundle_sha16()})
    return head


# ======================================================================
# S2 V1 方差发(RB.1 位级承接 + RB.2)
# ======================================================================

def varprobe_stage(events):
    tag = "b1-varprobe-launch"
    d = exam_case(events, str(KING_NPZ), tag, POOL_PROBE, manager_npz=None,
                  extra={"note": "V1 方差发:独立重评授权在此落章;只记不裁,"
                                 "不改锚不改参照名分;V1 永不替代配对基准,"
                                 "v32-ref-launch 原档终局地位与 RB.4 基准身份不变"})
    ref_doc = strict_json_loads(PRIORS["v32-ref-launch"][0].read_bytes())
    old_rows = sorted(ref_doc["rows"], key=lambda r: r["seed"])
    new_rows = sorted(d["rows"], key=lambda r: r["seed"])
    require(len(old_rows) == len(new_rows) == 32, "参照行数异常")
    diff_seeds = [o["seed"] for o, n in zip(old_rows, new_rows) if o != n]
    core = ("ret_mean", "ret_median", "died", "depth_median", "kills_mean",
            "farm_tau_mean", "override_rate", "cap_rate")
    agg_diff = [k for k in core
                if ref_doc["agg"].get(k) != d["agg"].get(k)]
    if diff_seeds or agg_diff:
        # RB.2 带外:先行漂移分诊(W-E0/CASE_RUNTIME 全项复验)
        snapshot = freeze_eval_identity(ROOT, str(KING_NPZ), None)
        if runtime_five(snapshot) != V32_CASE_RT:
            log({"event": "CASE_HALT_ENV_DRIFT", "where": "varprobe-triage",
                 "current": runtime_five(snapshot)})
            attention("V1 失配且运行时漂移:CASE_HALT")
            raise SystemExit(6)
        max_dret = max(abs(n["ret"] - o["ret"])
                       for o, n in zip(old_rows, new_rows))
        log({"event": "REF_DIVERGENCE", "tag": tag,
             "row_diff_seeds": diff_seeds, "agg_diff": agg_diff,
             "max_abs_dret": round(max_dret, 2),
             "triage": "runtime_five 全绿且失配呈非系统性——停机呈报总设计师"
                       "裁量;专项方差案(≥k 次重复)另立;禁以 n=2 之 1 自由度"
                       "读数造全项目噪声带,禁'如实发布即续案'"})
        attention("V1 方差发对 v32-ref-launch 失配,REF_DIVERGENCE 停机呈报")
        raise SystemExit(8)
    if not stage_done(events, "VARPROBE"):
        ev = {"event": "VARPROBE", "tag": tag, "rows": 32,
              "max_abs_dret": 0.0, "band": [0, 0], "in_band": True,
              "verdict": "种子内测量方差 = 0(确定性协议,限本运行时世界、"
                         "限 H 侧——M29 侧系推定,注记随判词,实测另案,"
                         "承合规 m-4/残余⑮):配对 Δ 之显著性评估必须走跨"
                         "种子符号检验,禁以测量噪声名义作 t 检验;F2 统计"
                         "登记之'方差未知'缺口就此闭合",
              "rb1": "RB.1 位级承接 PASS(launch 侧逐种子逐字段 biteq;"
                     "science 侧以 PRIORS 全文 sha 钉死承接,140.9 不设新发)"}
        log(ev)
        events.append(ev)
    return d


# ======================================================================
# S4 CANARY_SET + CRITERION_REGISTER(先登记后开箱)
# ======================================================================

def canary_set_stage(events) -> dict:
    prior = [e for e in events if e.get("event") == "CANARY_SET"]
    if prior:
        return prior[0]
    info = extract_depth2(PRIORS["v32-ref-launch"][0], CANARY_CONTROLS)
    require(info["archive_sha256"] == PRIORS["v32-ref-launch"][1],
            "CANARY_SET 提取源档案 sha 漂移")
    ev = {"event": "CANARY_SET", "n_D": info["n_D"],
          "depth2_seeds": info["depth2_seeds"],
          "controls": info["controls"], "C": info["C"],
          "source_archive_sha256": info["archive_sha256"],
          "extractor": "train/extract_canary_set.py(E8,随冻结 commit 入库)",
          "note": "腿点火前冻结;记分种子集仅约束 RB.5/RB.8,不约束考发面"
                  "(金丝雀评测种子面 = 全池 7000-7031)"}
    log(ev)
    events.append(ev)
    return ev


def criterion_register_stage(events):
    if stage_done(events, "CRITERION_REGISTER"):
        return [e for e in events if e.get("event") == "CRITERION_REGISTER"][0]
    info = extract_depth2(EVAL / "b1-ref8k-launch.json", ())
    ev = {"event": "CRITERION_REGISTER",
          "criterion": "REC(s) := [Δ_H(s) ≤ −20] ∧ [Δ_M29(s) ≥ Δ_H(s) + 20]"
                        "(经理敏感真值;各对同经理参照配对 K1/K2)",
          "holdout_depth2_seeds": info["depth2_seeds"],
          "n_holdout_depth2": info["n_D"],
          "source_archive_sha256": info["archive_sha256"],
          "min_judgeable_hits": MIN_JUDGEABLE_HITS,
          "precision_point": REC_PRECISION["point"],
          "precision_band": list(REC_PRECISION["band"]),
          "note": "先登记后开箱(承统计 m-2)——名单本身在此实例化;本事件系"
                  " D3 判据之重申性落账,非案中另立判据之口子(承合规 m-7);"
                  "池级注记(承统计 m-6):REC 首肢在池级整体下移情形近于恒真"
                  "(RB.7 点估 −18),判别负担主压 M29 回收肢"}
    log(ev)
    events.append(ev)
    return ev


# ======================================================================
# S5 P8 腿(唯一训练点火;台账制 2 次;禁换种子)
# ======================================================================

def leg_cmd() -> list[str]:
    return [PY, "train/train_ppo.py", "--worker", "--algo", "mppo",
            "--gamma", "1.0", "--max-steps", "3000", "--n-steps", "512",
            "--num-envs", "4", "--lr", "3e-4", "--ent-coef", "0.005",
            "--seed", str(SEED), "--total-steps", str(LEG_STEPS),
            "--run-name", RUN_NAME, "--distill-beta", str(BETA),
            "--teacher-sd", str(BC_SD), "--teacher-override", str(KING_SD),
            "--skip-dry", "--manager-npz", str(M29_NPZ),
            "--resume-from", str(KING_ZIP), "--allow-legacy-resume",
            "--no-drink-sovereignty",
            "--calib-probes", ",".join(str(x) for x in CALIB_PROBES),
            "--calib-record-only",
            "--ckpt-every-steps", str(CKPT_EVERY),
            "--sentinel-every", str(SENTINEL_EVERY),
            "--dry-anchor-every", str(DRY_ANCHOR_EVERY)]


def leg_stage(events) -> str:
    model_path = RUNS / RUN_NAME / "model_final.zip"
    out = RUNS / RUN_NAME / "policy.npz"
    if model_path.exists() and zip_steps(model_path) == NT_TARGET:
        if not out.exists():
            require(run([PY, "train/export_worker_npz.py", str(model_path),
                         str(out)], "export-p8.retry.log", 600) == 0
                    and out.exists(), "P8 补导出失败,停机呈报")
            log({"event": "npz_exported", "leg": RUN_NAME,
                 "sha256": sha256(out), "note": "补导出(非重点火,不烧额度)"})
        log({"event": "leg_skip_complete", "leg": RUN_NAME})
        return str(out)
    require(leg_starts(events, RUN_NAME) < 2,
            "P8 腿点火额度耗尽(台账制 2 次)——OPERATIONAL_FAILURE 全案停机,"
            "禁手改台账续命")
    ev = {"event": "leg_start", "leg": RUN_NAME, "seed": SEED,
          "recipe": "v30/v32 ctrl 配方逐字;偏离封闭枚举五处"
                    "(seed/calib十点/ckpt-every/sentinel-every/dry-anchor-every)",
          "resume_from": "king zip(重点火系同命题续办;种子恒 304000,"
                         "重点火禁换种子)"}
    log(ev)
    events.append(ev)
    t0 = time.time()
    rc = run(leg_cmd(), f"train-{RUN_NAME}.log", timeout=21_600)
    nt = zip_steps(model_path)
    log({"event": "leg_done", "leg": RUN_NAME, "rc": rc, "nt_zip": nt,
         "dt_min": round((time.time() - t0) / 60, 1)})
    require(rc == 0 and nt == NT_TARGET,
            f"P8 腿未达标(rc={rc}, nt={nt}, 目标={NT_TARGET})——运维闸三式"
            "之一未过,按 P3 额度重点火(本案无放弃闸:一切分数皆读数)")
    require(run([PY, "train/export_worker_npz.py", str(model_path),
                 str(out)], "export-p8.log", 600) == 0 and out.exists(),
            "P8 npz 导出失败(补导出通道不烧额度,重启后走 leg_skip 分支)")
    log({"event": "npz_exported", "leg": RUN_NAME, "sha256": sha256(out)})
    return str(out)


# ======================================================================
# S8 CRITERION_VALIDATE
# ======================================================================

def criterion_validate_stage(events, reg_ev, k1, k2, l4, l5, l4_replay):
    if stage_done(events, "CRITERION_VALIDATE"):
        return
    d2_seeds = reg_ev["holdout_depth2_seeds"]
    k1_rows = by_seed(k1["rows"], 8000, 8031)
    k2_rows = by_seed(k2["rows"], 8000, 8031)
    l4_rows = by_seed(l4["rows"], 8000, 8031)
    l5_rows = by_seed(l5["rows"], 8000, 8031)
    tau = {int(s): v.get("farm_tau_median")
           for s, v in l4_replay.get("per_seed", {}).items()}
    from probe_composite_signature import composite_signature
    sig = composite_signature(k1_rows, l4_rows, tau)
    require(sorted(sig["ref_depth2_seeds"]) == sorted(d2_seeds),
            "CRITERION_VALIDATE 分母与 CRITERION_REGISTER 名单不一致")
    confusion = {"hit_rec": [], "hit_norec": [], "miss_rec": [], "miss_norec": []}
    for s in d2_seeds:
        dh = l4_rows[s]["ret"] - k1_rows[s]["ret"]
        dm = l5_rows[s]["ret"] - k2_rows[s]["ret"]
        rec = (dh <= REC_H_LINE) and (dm >= dh + REC_RECOVERY)
        hit = sig["per_seed"][s]["signature_hit"]
        key = f"{'hit' if hit else 'miss'}_{'rec' if rec else 'norec'}"
        confusion[key].append(s)
    tp, fp = len(confusion["hit_rec"]), len(confusion["hit_norec"])
    fn, tn = len(confusion["miss_rec"]), len(confusion["miss_norec"])
    hits = tp + fp
    if hits == 0:
        precision_verdict = ("指纹不复现于留出池(命中 = 0)——与 RB.5 联判,"
                             "不构成重烧或加烧理由")
        precision = None
        ci = None
    elif hits < MIN_JUDGEABLE_HITS:
        precision_verdict = (f"不可判(样本不足:命中 {hits} < "
                             f"{MIN_JUDGEABLE_HITS}),不入带判读,"
                             "循 RB.7 同型降级呈报")
        precision = round(tp / hits, 4)
        ci = clopper_pearson(tp, hits)
    else:
        precision = round(tp / hits, 4)
        ci = clopper_pearson(tp, hits)
        precision_verdict = band_judge(precision, *REC_PRECISION["band"])
    recall = round(tp / (tp + fn), 4) if (tp + fn) else None
    ev = {"event": "CRITERION_VALIDATE",
          "denominator": {"seeds": d2_seeds, "n": len(d2_seeds),
                          "source": "CRITERION_REGISTER 所嵌 8000 池 "
                                    "king×H depth≥2 名单"},
          "confusion_2x2": {k: sorted(v) for k, v in confusion.items()},
          "tp_fp_fn_tn": [tp, fp, fn, tn],
          "precision": precision,
          "precision_ci95_clopper_pearson": ci,
          "precision_verdict": precision_verdict,
          "recall": recall,
          "recall_note": "recall 系首曝定标读数,无先验带,如实登记",
          "pool_note": "REC 首肢在池级整体下移情形近于恒真,判别负担主压 "
                       "M29 回收肢(承统计 m-6,判词如实携此注记)",
          "caveats": ["签名特异性未标定(D3,强制随判词)",
                      "horizon 限定(评测协议 max-steps 3000)"]}
    log(ev)
    events.append(ev)
    return ev


# ======================================================================
# S9 R 线记分卡
# ======================================================================

def rb4_grid(x: float, med: float, sign: dict, delev: dict) -> dict:
    lo, hi = RB4_BAND["band"]

    def region(v):
        return "in" if lo <= v <= hi else ("below" if v < lo else "above")

    limb_s1 = med <= RB4_BAND["median_line"]
    limb_s2 = sign["neg"] >= RB4_BAND["neg_line"]
    limb_s3 = region(delev["mean"]) == region(x)
    limbs = int(limb_s1) + int(limb_s2) + int(limb_s3)
    if lo <= x <= hi:
        cell = "复现" if limbs >= 2 else "杠杆驱动候选"
    elif x < lo:
        cell = "放大" if limbs >= 2 else "杠杆驱动候选(负向变体)"
    elif med <= RB4_BAND["median_line"] or sign["neg"] >= RB4_BAND["neg_line"]:
        cell = "均值掩蔽候选"
    else:
        cell = "不复现"
    return {"x": round(x, 2), **band_judge(x, lo, hi),
            "limbs": {"S1_median": [round(med, 2), limb_s1],
                      "S2_signs": [sign["neg"], limb_s2],
                      "S3_deleveraged": [delev, limb_s3]},
            "n_limbs": limbs, "cell": cell,
            "note": "各格皆有效结论,判词按格改述,禁以'成/败'统称;"
                    "禁均值单独裁决(承统计 B-2)"}


def scorecard_stage(events, docs, canary_docs, gates, reg_ev, n_D, d_c_seeds,
                    leg_replay, throne_rep):
    if stage_done(events, "VERDICT_PATH"):
        return
    ref_launch = by_seed(strict_json_loads(
        PRIORS["v32-ref-launch"][0].read_bytes())["rows"], 7000, 7031)
    l2 = by_seed(docs["p8-full32"]["rows"], 7000, 7031)
    l3 = by_seed(docs["p8-full32-m29"]["rows"], 7000, 7031)
    k1 = by_seed(docs["b1-ref8k-launch"]["rows"], 8000, 8031)
    k2 = by_seed(docs["b1-ref8k-science"]["rows"], 8000, 8031)
    l4 = by_seed(docs["p8-hold8k"]["rows"], 8000, 8031)
    card = {}
    band_outcomes = []

    def add(name, judged):
        card[name] = judged
        if isinstance(judged, dict) and "in_band" in judged:
            band_outcomes.append((name, judged["in_band"]))

    # RB.1/RB.2 已由 VARPROBE 事件落账
    card["RB1_RB2"] = ("VARPROBE 在册:位级承接 PASS,方差带 [0,0] 带内"
                       if stage_done(events, "VARPROBE") else "缺席(缓判)")
    # RB.3 双经理读数差(8000 池王座参照)
    diffs_k = [k2[s]["ret"] - k1[s]["ret"] for s in sorted(k1)]
    rb3 = band_judge(sum(diffs_k) / 32, *RB3_BAND["band"])
    rb3.update({"point": RB3_BAND["point"], "median": round(median(diffs_k), 2),
                "sign": sign_test(diffs_k),
                "deleveraged": deleveraged_mean(
                    {s: k2[s]["ret"] - k1[s]["ret"] for s in k1})})
    add("RB3_dual_manager_8k", rb3)
    # RB.4 主判读数一(复合主判)
    diff4 = {s: l2[s]["ret"] - ref_launch[s]["ret"] for s in sorted(l2)}
    x4 = sum(diff4.values()) / 32
    rb4 = rb4_grid(x4, median(list(diff4.values())),
                   sign_test(list(diff4.values())), deleveraged_mean(diff4))
    add("RB4_f2_reproduction", rb4)
    cell = rb4["cell"]
    branch = ("R" if cell in ("复现", "放大") else
              "N" if cell == "不复现" else None)
    # RB.5 F-lock 指纹复现(以 CANARY_SET 实数 n_D 为准)
    from probe_composite_signature import composite_signature
    tau = {int(s): v.get("farm_tau_median")
           for s, v in leg_replay.get("per_seed", {}).items()}
    sig7k = composite_signature(ref_launch, l2, tau)
    point5 = round(0.7 * n_D)
    lo5 = math.ceil(0.55 * n_D)
    rb5 = band_judge(sig7k["n_hits"], lo5, n_D, integer=True)
    rb5.update({"point": point5, "hits": sig7k["hits"],
                "not_reproduced_line": lo5 - 1,
                "verdict": ("指纹不复现(与 RB.4 联判)"
                            if sig7k["n_hits"] <= lo5 - 1 else "带判读"),
                "caveat": "签名特异性未标定(强制限定,D3);下缘依据:带下缘"
                          "容许种子面近半衰减仍判复现(承统计 M-3)"})
    add("RB5_flock_fingerprint", rb5)
    # RB.6 P8 腿双经理读数差
    diffs6 = [l3[s]["ret"] - l2[s]["ret"] for s in sorted(l2)]
    rb6 = band_judge(sum(diffs6) / 32, *RB6_MEAN["band"])
    rb6["point"] = RB6_MEAN["point"]
    rb6["branch_note"] = "均差线无条件(健康王座 +27.9 与损伤腿 +29.1/+32.9 皆在带)"
    add("RB6_mean_l3_minus_l2", rb6)
    ms_ev = [e for e in events if e.get("event") == "MS_REPORT"
             and e.get("pool") == "7000"]
    if branch and ms_ev:
        msb = RB6_MS[branch]
        m = ms_ev[-1]
        rb6ms_max = band_judge(m["ms_max"], *msb["max"][1])
        rb6ms_max["point"] = msb["max"][0]
        rb6ms_over = band_judge(m["n_over_line"], *msb["over"][1], integer=True)
        rb6ms_over["point"] = msb["over"][0]
        add(f"RB6_MS_max_branch_{branch}", rb6ms_max)
        add(f"RB6_MS_overline_branch_{branch}", rb6ms_over)
    else:
        card["RB6_MS"] = "条件线不可判(RB.4 落候选格),如实登记"
    # RB.7 留出池转移
    diff7 = {s: l4[s]["ret"] - k1[s]["ret"] for s in sorted(l4)}
    rb7 = band_judge(sum(diff7.values()) / 32, *RB7_BAND["band"])
    rb7.update({"point": RB7_BAND["point"],
                "median": round(median(list(diff7.values())), 2),
                "sign": sign_test(list(diff7.values())),
                "deleveraged": deleveraged_mean(diff7)})
    add("RB7_holdout_transfer", rb7)
    d2_8k = reg_ev["n_holdout_depth2"]
    rb7d = band_judge(d2_8k, RB7_BAND["d2_band"][0], RB7_BAND["d2_band"][1],
                      integer=True)
    rb7d["point"] = RB7_BAND["d2_point"]
    if d2_8k < 2:
        rb7d["verdict"] = "签名判据在该池'不可考',CRITERION_VALIDATE 降级呈报"
    add("RB7_holdout_depth2_count", rb7d)
    # RB.8 金丝雀仪表(保持率;度量公式钉死)
    c4h = canary_docs.get("p8-canary4-h")
    if c4h is not None and branch:
        c4_rows = by_seed(c4h["rows"], 7000, 7031)
        kept = [s for s in d_c_seeds if d_windows(c4_rows[s]) >= 1]
        retention = len(kept) / n_D
        pt, band8 = RB8_BAND[branch]
        rb8 = band_judge(retention, *band8)
        rb8.update({"point": pt, "kept_seeds": sorted(kept), "n_D": n_D,
                    "formula": "保持率 := |{s∈D_C: 腿×H 金丝雀(+393,216 点)"
                               "档案该种子 D 窗数 ≥ 1}| / n_D"})
        add(f"RB8_canary_retention_branch_{branch}", rb8)
    else:
        card["RB8_canary_retention"] = ("不可判(金丝雀缺数或 RB.4 落候选格),"
                                        "如实登记(P-canary)")
    # RB.9 死门 would-trip(只记不裁,定标输入)
    ratios = [r for r in gates["ratios"] if r is not None]
    if ratios:
        rb9a = band_judge(median(ratios), *RB9_BAND["ratio_median"][1])
        rb9a["point"] = RB9_BAND["ratio_median"][0]
        add("RB9_grad_ratio_median", rb9a)
        trips = sum(1 for r in ratios if r > RATIO_LINE)
        rb9b = band_judge(trips, *RB9_BAND["ratio_trips"][1], integer=True)
        rb9b.update({"point": RB9_BAND["ratio_trips"][0],
                     "n_points": len(gates["ratios"])})
        add("RB9_ratio_would_trips", rb9b)
    ce_share = (sum(1 for c in gates["ce_values"] if c > CE_LINE)
                / max(1, len(gates["ce_values"])))
    rb9c = band_judge(ce_share, *RB9_BAND["ce_over_share"][1])
    rb9c.update({"point": RB9_BAND["ce_over_share"][0],
                 "note": "饱和即预期,该门系重校目标线非区分器(注记随判词)"})
    add("RB9_ce_over_share", rb9c)
    final_dry = [r for r in gates["dry_lines"] if r.get("final")] or \
        gates["dry_lines"][-1:]
    if final_dry:
        inc = (final_dry[-1]["mismatch"] - DRY_REF_THRONE) * 100
        rb9d = band_judge(inc, *RB9_BAND["dry_increment_pp"][1])
        rb9d.update({"point": RB9_BAND["dry_increment_pp"][0],
                     "reading": "腿终点 dry-anchor 失配对王座级零点 0.7515 之"
                                "增量(pp);血统注记:谱系起点 0.6305 随定标数据"})
        add("RB9_dry_anchor_increment_pp", rb9d)
    # RB.10 经理观测体态漂移
    th_walk = throne_rep["walk_d_decision"]["walk_sum_over_121_mean"]
    rb10t = band_judge(th_walk, *RB10_BAND["throne_d"][1])
    rb10t["point"] = RB10_BAND["throne_d"][0]
    add("RB10_throne_d_walk", rb10t)
    leg_walk = walk_subset_mean(leg_replay, d_c_seeds)
    if leg_walk is not None and branch:
        pt, band10 = RB10_BAND["leg"][branch]
        rb10l = band_judge(leg_walk, *band10)
        rb10l["point"] = pt
        add(f"RB10_leg_walk_branch_{branch}", rb10l)
    else:
        card["RB10_leg_walk"] = "不可判(RB.4 落候选格或重放缺数),如实登记"
    card["RB10_p50_diffs_note"] = ("8 追加维逐维 P50 差并列于 OBS_DRIFT 事件;"
                                   "P1 改判限定随行:仪表维仅翻边际窗,"
                                   "体态维系主嫌")

    # 族级解读条款(承统计 M-6):全表带内/带外计数,禁摘樱桃
    n_out = sum(1 for _, ok in band_outcomes if not ok)
    out_names = [n for n, ok in band_outcomes if not ok]
    family = {"n_band_readings": len(band_outcomes), "n_out_of_band": n_out,
              "out_names": out_names,
              "clause": "目标覆盖意图 ≈85%;散发带外(≤3 条且机制互异)系统计"
                        "常态,禁作系统性异常叙事;聚簇带外(≥4 条同向或同机制)"
                        "才升级 NEEDS_ATTENTION"}
    if n_out >= 4:
        attention(f"R 线聚簇带外({n_out} 条):{out_names}")

    # 转化条款(措辞按 N=2 限定收口,承统计 M-7)
    if cell == "不复现":
        transform = ("RB.4 落'不复现' → F2 全案结论限定升级为'训练种子特异"
                     "候选',死门重校阈值定标改以 P8 腿实测分布为准——该定标"
                     "只能定特异性(不误报健康),不能定灵敏度(能否报灾难),"
                     "此限定随转化条款写死;P8 腿自身即首个健康腿,其签名命中"
                     "率升格为假阳性率首实测,入定标交付")
    elif cell in ("复现", "放大"):
        transform = ("RB.4 落'复现' → 判词:『F-lock 于第二训练种子上复现"
                     "(2/2,同配方、同教师、同治下),可复制性证据 1→2』,"
                     "强制携残余①⑨⑩,禁作总体级'可复制命题'陈述;审判庭"
                     "菜单④换届案先决之一满足,呈报总设计师排产")
    else:
        transform = f"RB.4 落'{cell}'——条件线判'不可判,如实登记'"

    ev = {"event": "VERDICT_PATH", "case": "B1", "golden_authorized": False,
          "scorecard": card, "family_ledger": family,
          "rb4_cell": cell, "transform_clause": transform,
          "acceptance_three_limbs": {
              "①处方①转正": "E1 通道 + 范本义务条款落账(金评划界依 D0-① "
                            "亲批:不立法,金评维持单 H);义务范围明文排除金评",
              "②处方②定标数据": "10 点 calib / 双门 would-trip / dry-anchor "
                               "双参考 / OBS_DRIFT / 金丝雀全序列落档,"
                               "供死门重校正式挂闸案引用",
              "③P8 复现判词": f"RB.4/RB.5 联判:{cell}(各向皆合法结论,"
                             "无档位竞赛)"},
          "mandatory_notes": [
              "horizon 敏感性声明(承 P5 ④):一切签名/判据/灾难名单条件于"
              "评测协议 horizon;本案内禁调 horizon",
              "H 轨默认条款:M29 一切读数永不进入任何裁决线",
              "不溯及既往条款:历届判词、名分、锚值全部维持原状",
              "残余①-⑮ 见预注册全文,随判决附录承继"]}
    log(ev)
    events.append(ev)
    attention(f"B1 案结记分卡在册:RB.4 = {cell};判决附录由值守记档")


# ======================================================================
# --plan(零副作用)
# ======================================================================

def print_plan():
    def fmt_cmd(c):
        return " ".join(str(x).replace(str(ROOT) + "/", "") for x in c)

    plan = {
        "case": "B1 捆绑评测基建(测量与仪表化案;零金种子/零名分/零发射/零环境侵入)",
        "driver": "train/run_b1_infra.py(克隆 run_v32_sovereign.py 骨架)",
        "ledger": str(LEDGER.relative_to(ROOT)),
        "stages": [
            "S0 预检:W1/W7(king zip·npz/H/M29/KING_SD+parity/BC_SD_WAIVER)/"
            "PRIOR_REFERENCE 四档/W-E0(runtime_five == v32 CASE_RUNTIME)/"
            "W-H8 留出池处女+HOLDOUT_FIRSTBURN(2对/4发)/304000+307000 台账扫描/"
            "W9/FREEZE_SHA",
            "S1 W-G0 零侵入实弹(冻结先决)",
            "S2 V1 方差发 b1-varprobe-launch(兼 RB.1 位级承接;带 [0,0])",
            "S3 K1 b1-ref8k-launch + K2 b1-ref8k-science(8000 池首曝,按发计)",
            "S4 CANARY_SET(E8 提取 depth≥2 ∪ {7003,7011})+ CRITERION_REGISTER"
            "(嵌 K1 之 8000 池 depth≥2 名单)",
            "S5 P8 腿点火(额度 2;nt 闸 3,997,696;timeout 6h;launcher 按 E6)",
            "S6 金丝雀离线序列(4 ckpt × {H,M29} 共 8 发,全部先于 L1;"
            "OBS_DRIFT 随档;+3,989,504 ckpt 归档不入序列)+ GATE_WOULD_TRIP"
            "(calib 十点双门 + dry-anchor 双参考;一律只记不裁)",
            "S7 腿考五发:L1 p8-s16(单 H,E1 豁免①)→ L2/L3 捆绑 + MS_REPORT"
            " → L4/L5 留出捆绑 + MS_REPORT",
            "S8 CRITERION_VALIDATE(2×2 混淆表 + precision(CP 95%)+ recall;"
            "命中 <3 判不可判)",
            "S9 R 线记分卡 RB.1-RB.10 + 转化条款 → VERDICT_PATH(判决附录值守记档)",
        ],
        "leg_cmd": fmt_cmd(leg_cmd()),
        "leg_deviations_closed_enum": {
            "①seed": f"{SEED}(=303000+1000,禁种子搜索/重选)",
            "②calib_probes": list(CALIB_PROBES),
            "③ckpt_every_steps": CKPT_EVERY,
            "④sentinel_every": SENTINEL_EVERY,
            "⑤dry_anchor_every": DRY_ANCHOR_EVERY},
        "canary_points": {"mid4": list(CANARY_STEPS),
                          "terminal_npz": NT_TARGET,
                          "archived_not_in_sequence": EXTRA_CKPT_STEP},
        "smoke_w_g0": {
            "seed": SMOKE_SEED, "steps": SMOKE_STEPS,
            "window": [KING_STEPS, SMOKE_END],
            "bare_cmd": fmt_cmd(_smoke_cmd("bare")),
            "knobs_cmd": fmt_cmd(_smoke_cmd("knobs")),
            "guaranteed_in_window": {
                "calib_points": list(SMOKE_CALIB),
                "ckpt_saves": [KING_STEPS + CKPT_EVERY],
                "sentinel_first": ((KING_STEPS // SENTINEL_EVERY) + 1)
                * SENTINEL_EVERY,
                "dry_anchor_first": ((KING_STEPS // DRY_ANCHOR_EVERY) + 1)
                * DRY_ANCHOR_EVERY},
            "criteria": "policy state_dict + optimizer state 逐张量 torch.equal"
                        " + RNG 相关遥测字段逐字段;文件字节级比对废除"},
        "exam_table": [
            {"发": "V1", "tag": "b1-varprobe-launch", "worker": "king npz",
             "manager": "H", "seeds": POOL_PROBE},
            {"发": "K1", "tag": "b1-ref8k-launch", "worker": "king npz",
             "manager": "H", "seeds": POOL_HOLD},
            {"发": "K2", "tag": "b1-ref8k-science", "worker": "king npz",
             "manager": "M29", "seeds": POOL_HOLD},
            {"发": "L1", "tag": "p8-s16", "worker": "P8 npz", "manager": "H",
             "seeds": POOL_S16},
            {"发": "L2", "tag": "p8-full32", "worker": "P8 npz", "manager": "H",
             "seeds": POOL_PROBE},
            {"发": "L3", "tag": "p8-full32-m29", "worker": "P8 npz",
             "manager": "M29", "seeds": POOL_PROBE},
            {"发": "L4", "tag": "p8-hold8k", "worker": "P8 npz", "manager": "H",
             "seeds": POOL_HOLD},
            {"发": "L5", "tag": "p8-hold8k-m29", "worker": "P8 npz",
             "manager": "M29", "seeds": POOL_HOLD}],
        "would_trip_lines": {"distill_ce": CE_LINE, "grad_ratio": RATIO_LINE,
                             "dry_anchor": f"零点 {DRY_REF_THRONE},"
                                           f"+{DRY_TRIP_PP}pp;血统参考 "
                                           f"{DRY_REF_LINEAGE}",
                             "discipline": "本案内一律只记不裁"},
        "exit_codes": {0: "案结/幂等", 2: "额度耗尽", 3: "预检", 4: "锁冲突",
                       5: "W-E0 发车前漂移", 6: "runtime 案中漂移",
                       7: "CASE_HALT_G0", 8: "REF_DIVERGENCE"},
    }
    print(json.dumps(plan, ensure_ascii=False, indent=1))


# ======================================================================
# 主流程
# ======================================================================

def _main():
    events = read_ledger()
    if stage_done(events, "VERDICT_PATH"):
        print("案已结:幂等退出", flush=True)
        return
    preflight(events)
    smoke_stage(events)                                        # S1
    varprobe_stage(events)                                     # S2
    k1, k2 = bundled_exam(events, str(KING_NPZ), "b1-ref8k-launch",
                          "b1-ref8k-science", POOL_HOLD, holdout=True)  # S3
    cs = canary_set_stage(events)                              # S4
    reg_ev = criterion_register_stage(events)
    n_D, d_c_depth2 = cs["n_D"], cs["depth2_seeds"]
    leg_npz = leg_stage(events)                                # S5
    canary_docs = canary_stage(events, d_c_depth2)             # S6
    gates = deadgate_stage(events)
    docs = {"b1-ref8k-launch": k1, "b1-ref8k-science": k2}
    docs["p8-s16"] = exam_case(events, leg_npz, "p8-s16", POOL_S16,
                               manager_npz=None,
                               extra={"note": "运维健全读数(E1 豁免枚举①,"
                                              "半池单 H 合规)"})       # S7 L1
    l2, l3 = bundled_exam(events, leg_npz, "p8-full32", "p8-full32-m29",
                          POOL_PROBE)
    docs["p8-full32"], docs["p8-full32-m29"] = l2, l3
    ref_launch = by_seed(strict_json_loads(
        PRIORS["v32-ref-launch"][0].read_bytes())["rows"], 7000, 7031)
    ref_science = by_seed(strict_json_loads(
        PRIORS["v32-ref-science"][0].read_bytes())["rows"], 7000, 7031)
    ms_report(events, "7000", "p8-full32", "p8-full32-m29",
              by_seed(l2["rows"], 7000, 7031), by_seed(l3["rows"], 7000, 7031),
              ref_launch, ref_science,
              ["v32-ref-launch(PRIORS 钉死)", "v32-ref-science(PRIORS 钉死)"])
    l4, l5 = bundled_exam(events, leg_npz, "p8-hold8k", "p8-hold8k-m29",
                          POOL_HOLD, holdout=True)
    docs["p8-hold8k"], docs["p8-hold8k-m29"] = l4, l5
    ms_report(events, "8000", "p8-hold8k", "p8-hold8k-m29",
              by_seed(l4["rows"], 8000, 8031), by_seed(l5["rows"], 8000, 8031),
              by_seed(k1["rows"], 8000, 8031), by_seed(k2["rows"], 8000, 8031),
              ["b1-ref8k-launch", "b1-ref8k-science"])
    # S8:L4 重放(τ 中位;8000 池)+ CRITERION_VALIDATE
    l4_replay = run_obsdrift(leg_npz, EVAL / "p8-hold8k.json",
                             B1 / "replay" / "p8-hold8k.json")
    criterion_validate_stage(events, reg_ev, k1, k2, l4, l5, l4_replay)
    # S9:终腿重放(τ 中位 + RB.10 腿侧 + 终局 OBS_DRIFT)+ 记分卡
    throne_rep = throne_replay()
    leg_replay = run_obsdrift(leg_npz, EVAL / "p8-full32.json",
                              B1 / "replay" / "p8-full32.json")
    obs_drift_event(events, "terminal", leg_replay, throne_rep, d_c_depth2)
    scorecard_stage(events, docs, canary_docs, gates, reg_ev, n_D,
                    d_c_depth2, leg_replay, throne_rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="打印计划,零副作用")
    ap.add_argument("--smoke", action="store_true",
                    help="只跑 S1 W-G0 零侵入烟测(冻结前置,准许脏树)")
    args = ap.parse_args()
    if args.plan:
        print_plan()
        return
    try:
        with exclusive_lock(B1 / ".driver.lock", "B1 驱动"):
            if args.smoke:
                smoke_stage(read_ledger())
                print("W-G0 烟测阶段完成(判词见台账 G0_NULLINTRUSION)",
                      flush=True)
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
