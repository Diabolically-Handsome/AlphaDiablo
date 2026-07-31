"""R9「认证班底经理再教育」驱动(docs/PREREG-R9-manager-reeducation.md 条款唯一执行者;
run_v29_relection.py 定向改造克隆,非薄包装)。

克隆差异表(PREREG-R9 逐条对应):
- 班底:工人 = R8 认证发布件 model_final.zip(rev26,dual-v4-asymmetric-v3,sha 2837288d…)
  经 staging 无回执路径接入(0o444,借 run_r8_certification._stage_eval_file 模式,
  规避 eval_assembled 发布回执强制闸);基线经理 = M29 npz(sha 89441388…,legacy-v3)。
- CALIBRATED_PROTOCOL_VERSION = 4;裁决线区(ABANDON/FLOOR)不再钉常数,由新锚现场
  推导(ABANDON = 锚均值×0.66,FLOOR = 锚均值×85/92,推导式+数值入 ledger)。
- exam 评测命令补 --manager-policy-observation-view(M29 = legacy-v3;候选臂 = raw-v4)
  + --worker 指 staged zip。
- 种子集:复现池 A = 2_123_000-127,池 B = 2_124_000-127,终考 = 2_125_000-255
  (评测银行处女段,批文「锚并入R9」唯一新池消费);by_seed 随动。
- ARMS:r9-mfresh(纯 fresh)与 r9-mcurr(同 + --deep-start-curriculum p=0.5,target=2,cap=8);
  两臂皆 --worker-zip <staged> --worker-zip-sha256 <sha> --manager-policy-observation-view
  raw-v4;160k 步,4h/臂超时保。
- 序列:preflight → G0-6 旧端点全表重放(probe_efix_g0 双旋钮拨旧,重放 R8 终考
  official-r8-final-{baseline,candidate} 各 256 局位级对账)→ 新锚烧制(M29×认证工人,
  2_123/2_124 各 128 + 2_125 的 256)→ 双臂训练 → export_manager_npz+parity → 臂考
  (每臂 2_123/2_124 各 128)→ 配对判决 → 胜者 2_125 终考 256 对 + 终判。
- 判决层:import r8_statistics(不改它)。其 analyze_paired_archives 的档案联检契约把
  「经理」槽钉为 numpy_policy 且两侧内容必须等同(R8 几何:变量=工人,共享=经理);
  R9 恰为对偶(共享=认证工人 zip[sb3_checkpoint],变量=经理 npz),任何槽位改写都要
  伪造 kind/num_timesteps 身份字段。故本驱动以其注册原语(_student_t_upper_critical/
  _exact_sign_p_value/_clopper_pearson_upper)逐式复刻同一判决数学
  (phase=development/final 最小对数语义);tests/test_r9_machinery.py 对共享原语层
  (t 临界/符号检验/Clopper-Pearson)保持数值等价断言,防实现漂移。
- 闸门肢(主席回炉令 2026-07-31:R9 科学问题=深度,闸门随之;族错 α=0.05 按
  3 约束重摊):
  ① depth 优越肢(主指标,新增):逐种子配对 depth 差(候选−基线),
     mean_lcb + exact_sign 双检,minimum_effect=0;键名 'depth' 系官方 R8 档案
     rows 现场核认(无 dungeon_level 键);
  ② wage 非劣肢(由优越改非劣):mean LCB ≥ −0.10×锚池 wage 均值(逐池随锚
     现场推导,推导式+线值入 ledger;exact_sign 撤出闸门,sign 信息照记不裁)。
     经理用农层时间换下潜系本案期望行为,不以 wage 未优越判死;wage 崩塌
     (跌破锚均值一成)仍拦;
  ③ 死亡肢不动:精确条件 McNemar 非劣(边距 0.10)+ 死亡观测线按 v31-D3-10
     遗留义务由新锚现场推导(line = 锚 died 数 + 边距×对数),禁承继 6/32 绝对线;
  ④ ret/kills/worker_kills 从闸门肢降为 record-only 诊断:照算入档,
     不进 failed_checks 判决。
- 胜者拣选(末令):过门臂比 depth 配对均差,带宽 0.10 内视为并列;并列破格
  ①合并池逐种子 died 总数更低者,②仍平取 r9-mfresh(奥卡姆:课程臂须以
  可见深度优势自证)。ret 带撤出拣选(照记入 quals)。
- 判词强制携 depth 直方 + 逐种子 died + 三防呆仪表(首次强制交权中位/DIVE 潜成率/
  dlvl 停留比,奠基卷基线 1495/25%/16058:3512;仪表二/三系 probe 级读数,
  ledger 登记 probe 复核命令,机器不擅自增发探针)。
- 金池 9000 与 7000/8000/12000 留出池零接触;--board 不用;本案不烧金牌。
账本:train/runs/r9-reeducation/gate_ledger.jsonl。
用法:.venv/bin/python train/run_r9_reeducation.py(发车须主席亲批)。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import signal
import stat
import subprocess
import time
import traceback
import zipfile

import r8_statistics
from eval_contract import (PROTOCOL_VERSION, OperationalFailure, OutputReservationError,
                           exclusive_lock, expected_eval_identity,
                           freeze_eval_identity, read_eval_archive,
                           verify_eval_identity)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
R9 = RUNS / "r9-reeducation"          # 战役控制目录(惰性创建,施工/import 零写盘)
LEDGER = R9 / "gate_ledger.jsonl"
STAGING = R9 / "staging"
STAGED_WORKER = STAGING / "worker.zip"
EVAL = RUNS / "eval-assembled"

# ---- 班底常量(现场 sha256sum 取全,2026-07-31) ----
W_ZIP_SRC = RUNS / "r8-certification-published" / "model_final.zip"
W_ZIP_SHA = "2837288dad19a685925558a0d86e1cecd951d5f065d1d2f367f667c13b9cf006"
M29_NPZ = ROOT / "train" / "models" / "v29-manager-mfresh" / "policy.npz"
M29_SHA = "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6"

# ---- G0-6 遗留义务(E-fix):R8 终考旧端点全表重放素材 ----
G0_PROBE = RUNS / "efix-g0-evidence" / "probe_efix_g0.py"
R8_FINAL = {
    "official-r8-final-baseline-2122000": {
        "archive_sha256":
            "e7c97a0aef1d3abd4eee18e574938575f853ee0f4b8d02b8118bb8917fc9d37d",
        "worker_zip": RUNS / "r8-certification-control" / "eval-inputs"
                      / "official-r8-final-baseline-2122000" / "worker.zip",
        "worker_sha256":
            "2f7bc9dd810956c3feeb330575c9a03ddff0b476333ac429a411935985b04f42",
    },
    "official-r8-final-candidate-2122000": {
        "archive_sha256":
            "42a8f5e2bfdda5dcc366e524e1d25d759eba303eed58b2caf4de4e25695ba032",
        "worker_zip": RUNS / "r8-certification-control" / "eval-inputs"
                      / "official-r8-final-candidate-2122000" / "worker.zip",
        "worker_sha256": W_ZIP_SHA,
    },
}
R8_FINAL_SEEDS = (2_122_000, 2_122_255)

# ---- 池划定(PREREG-R9 §2;本案唯一新池消费) ----
POOL_A = (2_123_000, 2_123_127)
POOL_B = (2_124_000, 2_124_127)
POOL_FINAL = (2_125_000, 2_125_255)

TAG_ANCHOR = {"a": "r9-anchor-a-2123000", "b": "r9-anchor-b-2124000",
              "final": "r9-anchor-final-2125000"}
POOLS = {"a": POOL_A, "b": POOL_B, "final": POOL_FINAL}

STEPS = 160_000
CALIBRATED_PROTOCOL_VERSION = 4
# 裁决线推导系数(基数 = 新锚 A+B 合并均值;数值发车夜由锚读数代入并入 ledger):
ABANDON_FACTOR = 0.66            # v29 先例 75/112.4≈0.667 的注册化
FLOOR_NUM, FLOOR_DEN = 85.0, 92.0  # v25/v29 比例 85/92 沿用,基数换新锚
DEATH_MARGIN = 0.10              # 精确条件 McNemar 非劣边距(PREREG-R8 §1.3 同值)
FAMILYWISE_ALPHA = 0.05
# 胜者拣选带(末令):depth 配对均差,带宽 0.10 内并列→①低死②mfresh。
WINNER_DEPTH_TIE_BAND = 0.10
# 临线注记阈(v31 判词纪律承继;注册于此,判词强制携带):
NEAR_LINE_DEATH_GAP = 1          # |candidate_deaths − 推导线| ≤ 1 命
NEAR_LINE_LCB_BAND = 0.5         # 任一均值肢 LCB ∈ [0, 0.5)
NEAR_LINE_FLOOR_GAP = 1.0        # |胜者均值 − FLOOR| ≤ 1.0
NEAR_LINE_UCB_BAND = 0.01        # McNemar UCB 距边距 ≤ 0.01

# ---- 判决闸门肢(主席回炉令 2026-07-31,预注册于模块 docstring) ----
# 主指标 = depth 优越肢:键名 'depth' 系官方 R8 档案 rows 现场核认。
DEPTH_RULE = r8_statistics.MetricRule("depth")
# wage 非劣肢:线 = −WAGE_NI_FRACTION×锚池 wage 均值(逐池现场推导,含端点过)。
WAGE_KEY = "farm_worker_wage"
WAGE_NI_FRACTION = 0.10
# 闸门约束数 = depth + wage 非劣 + 死亡 = 3;族错 α=0.05 按此重摊。
GATE_CONSTRAINT_COUNT = 3
# ret/kills/worker_kills 降为 record-only 诊断(0-优越口径照算,与旧闸读数
# 可比;不占 α、不进 failed_checks)。
RECORD_ONLY_RULES = (
    r8_statistics.MetricRule("ret"),
    r8_statistics.MetricRule("kills"),
    r8_statistics.MetricRule("farm_worker_kills"),
)
R9_STATISTICS_SCHEMA = "diablogym-r9-paired-statistics/2"
R9_METHOD_REVISION = (r8_statistics.R8_METHOD_REVISION
                      + "+r9-role-dual-depth-gate/2")

# 三防呆仪表(判「深度解锁」须同时移动;基线值 = 奠基卷实测,修后按新锚重读):
GAUGE_BASELINES = {
    "first_forced_handover_median_micro_steps": 1495,   # probe 级(决策流)
    "dive_window_success_rate": 0.25,                    # 档案可算(mode_seq/depth)
    "dlvl_dwell_ratio_l1_l2": [16058, 3512],             # probe 级(逐拍 dlvl)
}

ARMS = {
    "r9-mfresh": ["--ent-coef", "0.02", "--lr", "3e-4", "--seed", "22"],
    "r9-mcurr": ["--ent-coef", "0.02", "--lr", "3e-4", "--seed", "22",
                 "--deep-start-curriculum", "p=0.5,target=2,cap=8"],
}


class CampaignError(RuntimeError):
    """R9 战役输入/文件系统不满足预注册契约。"""


def log(event: dict):
    R9.mkdir(parents=True, exist_ok=True)
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    R9.mkdir(parents=True, exist_ok=True)
    with open(R9 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha16(p) -> str:
    return sha256(p)[:16]


# ---- staging 三函数(借 run_r8_certification.py:741/1333/1360/3030 模式) ----

def _stable_read(path: pathlib.Path) -> bytes:
    """Read one regular file identity and reject symlink/replace races."""
    path = pathlib.Path(path)
    try:
        before_path = path.lstat()
    except OSError as exc:
        raise CampaignError(f"文件不可读:{path}: {exc}") from exc
    require(not stat.S_ISLNK(before_path.st_mode), f"拒绝符号链接输入:{path}")
    require(stat.S_ISREG(before_path.st_mode), f"输入不是普通文件:{path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CampaignError(f"文件不可稳定打开:{path}: {exc}") from exc
    try:
        first = os.fstat(fd)
        require(stat.S_ISREG(first.st_mode), f"已打开输入不是普通文件:{path}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        second = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise CampaignError(f"文件读取后身份消失:{path}: {exc}") from exc
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_mode, item.st_size,
        item.st_mtime_ns, item.st_ctime_ns,
    )
    require(
        identity(before_path) == identity(first)
        == identity(second) == identity(after_path),
        f"文件读取期间被替换或修改:{path}",
    )
    return b"".join(chunks)


def sha256(p) -> str:
    return hashlib.sha256(_stable_read(pathlib.Path(p))).hexdigest()


def _fsync_directory(path: pathlib.Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise CampaignError(f"目录不可打开以 fsync:{path}: {exc}") from exc
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_bytes_exclusive(
        path: pathlib.Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            mode,
        )
    except FileExistsError:
        require(_stable_read(path) == payload,
                f"不可变文件已存在但内容漂移:{path}")
        return
    with os.fdopen(fd, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _stage_eval_file(
        source: pathlib.Path, destination: pathlib.Path,
        *, expected_sha256: str | None = None) -> str:
    payload = _stable_read(source)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        require(
            digest == expected_sha256,
            f"eval staging 源 SHA 漂移:{source}:{digest} != {expected_sha256}",
        )
    _write_bytes_exclusive(destination, payload, mode=0o444)
    try:
        os.chmod(destination, 0o444)
    except OSError as exc:
        raise CampaignError(f"eval staging 无法设为只读:{destination}: {exc}") from exc
    _fsync_directory(destination.parent)
    require(sha256(destination) == digest,
            f"eval staging 副本 SHA 漂移:{destination}")
    return digest


# ---- 运维原语(v29 骨架) ----

def run(cmd, logfile, timeout) -> int:
    R9.mkdir(parents=True, exist_ok=True)
    with open(R9 / logfile, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)   # 连锅端:SubprocVecEnv 孙进程防孤儿
            except ProcessLookupError:
                pass
            proc.wait()
            return 124    # 挂死护栏:按崩溃/失败落账(运维护栏,非判决输入)


def zip_steps(p: pathlib.Path) -> int:
    """SB3 真链读数(v29 面板 blocker 修正:status 节流计数必滞后)。"""
    try:
        with zipfile.ZipFile(p) as z:
            return int(json.loads(z.read("data"))["num_timesteps"])
    except Exception:
        return 0


def seeds_arg(pool: tuple[int, int]) -> str:
    lo, hi = pool
    return f"{lo}-{hi}"


def by_seed(rows, pool: tuple[int, int]) -> dict:
    lo, hi = pool
    m = {r["seed"]: r for r in rows}
    require(len(rows) == len(m), "种子集合异常(含重复 seed)")
    require(set(m) == set(range(lo, hi + 1)),
            f"种子集合异常(须为 {lo}-{hi})")
    return m


def require_calibrated_protocol() -> None:
    if PROTOCOL_VERSION != CALIBRATED_PROTOCOL_VERSION:
        raise OperationalFailure(
            "R9 的裁决线推导式/池划定仅在 protocol-v4 环境语义下预注册;"
            f"当前 PROTOCOL_VERSION={PROTOCOL_VERSION}。协议再迁移须先重开预注册,"
            "禁止混用旧阈值")


# ---- 评测(exam) ----

def exam(tag: str, pool: tuple[int, int], manager_npz: pathlib.Path,
         manager_view: str, timeout: int):
    """staged 认证工人 × 指定经理;返回 (validated_doc, archive_sha256) 或 None。"""
    require(manager_view in ("legacy-v3", "raw-v4"), f"经理视图非法:{manager_view}")
    out = EVAL / f"{tag}.json"
    require(not out.exists(), f"档案不可变性:{out} 已存在,拒绝覆写")
    lo, hi = pool
    seed_values = list(range(lo, hi + 1))
    snapshot = freeze_eval_identity(ROOT, str(STAGED_WORKER), str(manager_npz))
    require(snapshot["worker"]["kind"] == "sb3_checkpoint"
            and snapshot["worker"]["sha256"] == W_ZIP_SHA,
            "staged 认证工人身份漂移")
    expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
    cmd = [PY, "train/eval_assembled.py",
           "--worker", snapshot["worker"]["path"],
           "--manager-npz", snapshot["manager"]["path"],
           "--manager-policy-observation-view", manager_view,
           "--seeds", seeds_arg(pool), "--tag", tag]
    if run(cmd, f"exam-{tag}.{time.time_ns()}.log", timeout=timeout) != 0:
        if out.exists():    # 半截档案轮转,给重考让路
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    try:
        d = read_eval_archive(out, **expected)
        verify_eval_identity(snapshot, ROOT)
    except (OSError, KeyError, TypeError, ValueError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    return d, sha256(out)


def exam_retry(tag, pool, manager_npz, manager_view, timeout):
    result = exam(tag, pool, manager_npz, manager_view, timeout)
    if result is None:
        log({"event": "exam_crash", "tag": tag, "note": "评测失败,按崩溃条款重考一次"})
        result = exam(tag, pool, manager_npz, manager_view, timeout)
    return result


# ---- 深度仪表(判词强制随行) ----

def depth_hist(rows) -> dict:
    hist: dict[str, int] = {}
    for r in rows:
        hist[str(int(r["depth"]))] = hist.get(str(int(r["depth"])), 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: int(kv[0])))


def l3_plus(rows) -> int:
    return sum(1 for r in rows if r["depth"] >= 3)


def died_seeds(rows) -> list:
    return sorted(int(r["seed"]) for r in rows if r["died"])


def dive_per_ep(rows) -> float:
    return sum(r["mode_seq"].count("D") for r in rows) / max(1, len(rows))


def dive_gauges(rows) -> dict:
    """DIVE 潜成率(档案口径):Σ(depth−1)/Σ(D 窗数)。起点恒 1 层,
    每次潜成恰移一层;probe 口径(逐窗 dlvl0→dlvl_end)由 probe 复核。"""
    dives = sum(r["mode_seq"].count("D") for r in rows)
    descents = sum(max(0, int(r["depth"]) - 1) for r in rows)
    return {
        "dive_windows": int(dives),
        "descents": int(descents),
        "dive_window_success_rate": (descents / dives) if dives else None,
    }


def bonus_per_ep(rows) -> float:
    # 下楼奖金兑现:depth=d 兑现 8×(1+2+…+(d−1));d≤1 为 0
    return sum(8 * sum(range(1, r["depth"])) for r in rows) / max(1, len(rows))


def depth_dashboard(rows) -> dict:
    return {
        "n": len(rows),
        "depth_hist": depth_hist(rows),
        "l3_plus": l3_plus(rows),
        "died": sum(1 for r in rows if r["died"]),
        "died_seeds": died_seeds(rows),
        "dive_per_ep": round(dive_per_ep(rows), 3),
        "bonus_per_ep": round(bonus_per_ep(rows), 3),
        **dive_gauges(rows),
    }


def gauge_report(anchor_rows, candidate_rows, anchor_tag: str) -> dict:
    """三防呆仪表:仪表 2 档案可算(锚/候选双读);仪表 1/3 系 probe 级,
    登记基线值 + 锚侧复核命令(probe_r9_dive 经理钉 M29,恰为锚组装体;
    候选侧须经理可参数化探针变体,机器不擅增发,义务入册)。"""
    probe_cmd = (f"{PY} {G0_PROBE.with_name('probe_r9_dive.py')} "
                 f"<out.json> <逗号种子表> {STAGED_WORKER} {anchor_tag}.json")
    return {
        "baselines_foundation_dossier": GAUGE_BASELINES,
        "dive_window_success_rate": {
            "anchor": dive_gauges(anchor_rows)["dive_window_success_rate"],
            "candidate": dive_gauges(candidate_rows)["dive_window_success_rate"],
        },
        "probe_level_gauges": {
            "gauges": ["first_forced_handover_median_micro_steps",
                       "dlvl_dwell_ratio_l1_l2"],
            "anchor_recheck_cmd": probe_cmd,
            "candidate_recheck_note":
                "probe_r9_dive 经理硬编码 M29;胜者侧复核须经理可参数化探针"
                "变体,属人工义务,本机器不擅自增发探针",
        },
    }


# ---- 判决核(r8_statistics 同械同式,R9 角色几何) ----

def _shared_worker_identity(meta_worker: dict, label: str) -> dict:
    require(isinstance(meta_worker, dict), f"{label} worker identity 非法")
    require(meta_worker.get("kind") == "sb3_checkpoint",
            f"{label} 共享工人 kind 必须是 sb3_checkpoint")
    require(meta_worker.get("sha256") == W_ZIP_SHA,
            f"{label} 共享工人 sha 必须等于认证发布件:{meta_worker.get('sha256')!r}")
    require(meta_worker.get("gate_report_sha256") is None,
            f"{label} 共享工人不得携发布回执(staging 无回执路径)")
    return {key: meta_worker.get(key)
            for key in ("kind", "sha256", "num_timesteps", "gate_report_sha256")}


def paired_judgment(baseline: dict, candidate: dict, *,
                    baseline_sha256: str, candidate_sha256: str,
                    phase: str) -> dict:
    """R9 配对判决:R8 注册原语逐式复刻,闸门肢按主席回炉令重定。

    baseline = 新锚档案(M29×认证工人);candidate = 候选经理档案(候选×同一工人)。
    闸门(3 约束,族错 α=0.05 均摊):① depth 优越(主指标,mean_lcb +
    exact_sign 双检,minimum_effect=0);② wage 非劣(mean LCB ≥
    −WAGE_NI_FRACTION×锚池 wage 均值,现场推导,含端点;sign 撤出闸门);
    ③ 死亡:精确条件 McNemar 非劣(边距 0.10)+ 观测线按新锚推导
    (line = 锚 died 数 + 边距×对数;禁绝对常数线)。
    ret/kills/worker_kills 系 record-only 诊断,照算入档不进判决。
    """
    require(phase in ("development", "final"), f"phase 非法:{phase}")
    floor_pairs = (r8_statistics.MIN_DEVELOPMENT_PAIRS if phase == "development"
                   else r8_statistics.MIN_FINAL_PAIRS)
    for label, document in (("baseline", baseline), ("candidate", candidate)):
        require(isinstance(document, dict)
                and set(document) == {"schema_version", "meta", "agg", "rows"},
                f"{label} 档案必须是已验 schema-v5 档案")
        require(document["schema_version"]
                == r8_statistics.SUPPORTED_EVAL_ARCHIVE_SCHEMA,
                f"{label} 档案 schema 必须为 v5")
    b_meta, c_meta = baseline["meta"], candidate["meta"]
    require(b_meta["protocol"] == c_meta["protocol"],
            "baseline/candidate 协议或种子表不一致")
    require(b_meta["runtime"] == c_meta["runtime"],
            "baseline/candidate 运行时/内容身份不一致")
    # R9 角色几何:共享件 = 认证工人 zip(内容身份必须等同),变量 = 经理 npz。
    require(_shared_worker_identity(b_meta["worker"], "baseline")
            == _shared_worker_identity(c_meta["worker"], "candidate"),
            "baseline/candidate 共享工人内容身份不一致")
    for label, meta in (("baseline", b_meta), ("candidate", c_meta)):
        require(meta["manager"].get("kind") == "numpy_policy",
                f"{label} 经理 kind 必须是 numpy_policy")
    b_mgr_sha = b_meta["manager"]["sha256"]
    c_mgr_sha = c_meta["manager"]["sha256"]
    require(b_mgr_sha == M29_SHA, f"基线经理必须是 M29:{b_mgr_sha!r}")
    require(b_mgr_sha != c_mgr_sha, "配对两侧经理内容必须不同(变量=经理)")
    require(isinstance(baseline_sha256, str) and isinstance(candidate_sha256, str)
            and baseline_sha256 != candidate_sha256,
            "档案字节 SHA 必须给全且不同")

    protocol = b_meta["protocol"]
    require(protocol.get("deterministic") is True
            and isinstance(protocol.get("seeds"), list),
            "确定性协议种子表缺失")
    seeds = protocol["seeds"]
    b_rows, c_rows = baseline["rows"], candidate["rows"]
    require([row.get("seed") for row in b_rows] == seeds
            and [row.get("seed") for row in c_rows] == seeds,
            "行序必须与协议种子表逐位一致")
    n_pairs = len(seeds)
    require(n_pairs >= floor_pairs,
            f"{phase} 判决至少需要 {floor_pairs} 对(收到 {n_pairs})")

    all_metric_keys = ([DEPTH_RULE.key, WAGE_KEY]
                       + [rule.key for rule in RECORD_ONLY_RULES])
    per_constraint_alpha = FAMILYWISE_ALPHA / GATE_CONSTRAINT_COUNT
    t_critical = r8_statistics._student_t_upper_critical(
        per_constraint_alpha, n_pairs - 1)
    checks: dict = {}
    paired_hash_rows = []
    for b_row, c_row in zip(b_rows, c_rows):
        require(b_row.get("seed") == c_row.get("seed"), "配对行 seed 不一致")
        hash_row = {"seed": b_row["seed"], "baseline": {}, "candidate": {}}
        for key in all_metric_keys:
            r8_statistics._finite_number(
                b_row.get(key), f"baseline seed {b_row['seed']} {key}")
            r8_statistics._finite_number(
                c_row.get(key), f"candidate seed {c_row['seed']} {key}")
            hash_row["baseline"][key] = b_row[key]
            hash_row["candidate"][key] = c_row[key]
        require(isinstance(b_row.get("died"), bool)
                and isinstance(c_row.get("died"), bool),
                "died 必须是逐行 bool")
        hash_row["baseline"]["died"] = b_row["died"]
        hash_row["candidate"]["died"] = c_row["died"]
        paired_hash_rows.append(hash_row)

    def limb_stats(key: str, minimum_effect: float) -> dict:
        """r8 注册式逐字复刻的配对统计(全肢 higher 方向;过门判定由闸门层定)。"""
        baseline_values = [float(row[key]) for row in b_rows]
        candidate_values = [float(row[key]) for row in c_rows]
        improvement_values = [c - b for b, c
                              in zip(baseline_values, candidate_values)]
        baseline_mean = math.fsum(baseline_values) / n_pairs
        candidate_mean = math.fsum(candidate_values) / n_pairs
        improvement_mean = math.fsum(improvement_values) / n_pairs
        squared = math.fsum((v - improvement_mean) ** 2
                            for v in improvement_values)
        sample_stddev = math.sqrt(squared / (n_pairs - 1))
        standard_error = sample_stddev / math.sqrt(n_pairs)
        lower_bound = improvement_mean - t_critical * standard_error
        for label, value in (("improvement_mean", improvement_mean),
                             ("lower_confidence_bound", lower_bound)):
            require(math.isfinite(value), f"{key}.{label} 必须有限")
        centered = [v - float(minimum_effect) for v in improvement_values]
        wins = sum(v > 0.0 for v in centered)
        losses = sum(v < 0.0 for v in centered)
        sign_p = r8_statistics._exact_sign_p_value(wins, wins + losses)
        return {
            "direction": "higher",
            "minimum_effect": float(minimum_effect),
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "raw_candidate_minus_baseline_mean": improvement_mean,
            "improvement_mean": improvement_mean,
            "sample_stddev": sample_stddev,
            "standard_error": standard_error,
            "one_sided_t_critical": t_critical,
            "lower_confidence_bound": lower_bound,
            "sign_test": {
                "wins_above_minimum_effect": wins,
                "losses_below_minimum_effect": losses,
                "ties_at_minimum_effect": n_pairs - wins - losses,
                "non_ties": wins + losses,
                "one_sided_exact_p_value": sign_p,
            },
        }

    metrics: dict = {}
    # ① depth 优越肢(主指标;mean+sign 双检共用同一 α 份额,r8 同型)。
    depth = limb_stats(DEPTH_RULE.key, float(DEPTH_RULE.minimum_effect))
    depth_mean_passed = (depth["lower_confidence_bound"]
                         > float(DEPTH_RULE.minimum_effect))
    depth_sign_passed = (depth["sign_test"]["one_sided_exact_p_value"]
                         <= per_constraint_alpha)
    checks["depth.mean_lcb"] = depth_mean_passed
    checks["depth.exact_sign"] = depth_sign_passed
    depth.update({
        "kind": "superiority",
        "mean_lcb_passed": depth_mean_passed,
        "sign_test": {**depth["sign_test"], "required": True,
                      "passed": depth_sign_passed},
        "passed": depth_mean_passed and depth_sign_passed,
    })
    metrics["depth"] = depth
    # ② wage 非劣肢:线 = −WAGE_NI_FRACTION×锚池 wage 均值(逐池随锚现场
    #    推导,含端点过);exact_sign 撤出闸门(sign 信息照记不裁)。
    anchor_wage_mean = math.fsum(
        float(row[WAGE_KEY]) for row in b_rows) / n_pairs
    wage_minimum_effect = -WAGE_NI_FRACTION * anchor_wage_mean
    wage = limb_stats(WAGE_KEY, wage_minimum_effect)
    wage_ni_passed = wage["lower_confidence_bound"] >= wage_minimum_effect
    checks["farm_worker_wage.noninferiority_lcb"] = wage_ni_passed
    wage.update({
        "kind": "noninferiority",
        "noninferiority": {
            "formula": "minimum_effect = -WAGE_NI_FRACTION × anchor_wage_mean"
                       "(逐池随锚现场推导;LCB ≥ 线 即过,含端点)",
            "margin_fraction": WAGE_NI_FRACTION,
            "anchor_wage_mean": anchor_wage_mean,
            "minimum_effect": wage_minimum_effect,
            "lcb_passed": wage_ni_passed,
        },
        "sign_test": {**wage["sign_test"], "required": False, "passed": None},
        "passed": wage_ni_passed,
    })
    metrics[WAGE_KEY] = wage
    # ③ record-only 诊断肢:照算入档,不进 checks/failed_checks(回炉令 ④)。
    record_only_metrics: dict = {}
    for rule in RECORD_ONLY_RULES:
        stats = limb_stats(rule.key, float(rule.minimum_effect))
        stats.update({
            "record_only": True,
            "note": "0-优越口径照算(与旧闸读数可比);不占 α,不进判决",
            "mean_lcb_indicative": (stats["lower_confidence_bound"]
                                    > float(rule.minimum_effect)),
            "sign_test": {**stats["sign_test"], "required": False,
                          "passed": None},
        })
        record_only_metrics[rule.key] = stats

    baseline_deaths = sum(bool(row["died"]) for row in b_rows)
    candidate_deaths = sum(bool(row["died"]) for row in c_rows)
    candidate_only = sum((not bool(b["died"])) and bool(c["died"])
                         for b, c in zip(b_rows, c_rows))
    baseline_only = sum(bool(b["died"]) and (not bool(c["died"]))
                        for b, c in zip(b_rows, c_rows))
    component_alpha = per_constraint_alpha / 2.0
    discordant = candidate_only + baseline_only
    if discordant:
        theta_upper = r8_statistics._clopper_pearson_upper(
            candidate_only, discordant, component_alpha)
        risk_upper = (2.0 * theta_upper - 1.0) * discordant / n_pairs
    else:
        theta_upper = 0.0
        risk_upper = 0.0
    death_bound_passed = risk_upper <= DEATH_MARGIN
    # v31-D3-10 落点:死亡观测线由新锚现场推导(禁 6/32 绝对线)。
    derived_line = baseline_deaths + DEATH_MARGIN * n_pairs
    observed_within_line = candidate_deaths <= derived_line
    checks["deaths.noninferiority_upper_bound"] = death_bound_passed
    checks["deaths.observed_within_derived_line"] = observed_within_line
    death_report = {
        "key": "died",
        "method": "exact-conditional-mcnemar-clopper-pearson-risk-difference",
        "noninferiority_margin": DEATH_MARGIN,
        "baseline_deaths": baseline_deaths,
        "candidate_deaths": candidate_deaths,
        "baseline_death_rate": baseline_deaths / n_pairs,
        "candidate_death_rate": candidate_deaths / n_pairs,
        "observed_candidate_minus_baseline_risk":
            (candidate_deaths - baseline_deaths) / n_pairs,
        "candidate_only_deaths": candidate_only,
        "baseline_only_deaths": baseline_only,
        "concordant_pairs": n_pairs - discordant,
        "component_alpha": component_alpha,
        "discordant_pairs": discordant,
        "candidate_only_conditional_theta_upper_bound": theta_upper,
        "candidate_minus_baseline_risk_upper_bound": risk_upper,
        "derived_observed_line": {
            "formula": "line_deaths = anchor_deaths + margin*n_pairs"
                       "(v31-D3-10:随锚现场推导,禁承继 6/32 绝对线)",
            "anchor_deaths": baseline_deaths,
            "margin": DEATH_MARGIN,
            "n_pairs": n_pairs,
            "line_deaths": derived_line,
        },
        "observed_within_derived_line": observed_within_line,
        "noninferiority_passed": death_bound_passed,
        "passed": death_bound_passed and observed_within_line,
    }
    failed_checks = sorted(name for name, ok in checks.items() if not ok)
    near_line_notes = []
    if abs(candidate_deaths - derived_line) <= NEAR_LINE_DEATH_GAP:
        near_line_notes.append(
            f"死亡观测距推导线 ≤{NEAR_LINE_DEATH_GAP} 命"
            f"(candidate={candidate_deaths},line={derived_line:.1f})")
    if abs(DEATH_MARGIN - risk_upper) <= NEAR_LINE_UCB_BAND:
        near_line_notes.append(
            f"McNemar UCB 距边距 ≤{NEAR_LINE_UCB_BAND}(UCB={risk_upper:.4f})")
    depth_lcb = metrics["depth"]["lower_confidence_bound"]
    if 0.0 <= depth_lcb < NEAR_LINE_LCB_BAND:
        near_line_notes.append(f"depth.LCB 临线({depth_lcb:.3f})")
    wage_gap = metrics[WAGE_KEY]["lower_confidence_bound"] - wage_minimum_effect
    if 0.0 <= wage_gap < NEAR_LINE_LCB_BAND:
        near_line_notes.append(f"wage 非劣 LCB 距线临线(gap={wage_gap:.3f})")
    result = {
        "schema_version": R9_STATISTICS_SCHEMA,
        "method_revision": R9_METHOD_REVISION,
        "phase": phase,
        "n_pairs": n_pairs,
        "required_pairs": floor_pairs,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "simultaneous_confidence": 1.0 - FAMILYWISE_ALPHA,
        "constraint_count": GATE_CONSTRAINT_COUNT,
        "per_constraint_alpha": per_constraint_alpha,
        "source": {
            "eval_schema_version": baseline["schema_version"],
            "baseline_archive_sha256": baseline_sha256,
            "candidate_archive_sha256": candidate_sha256,
            "baseline_manager_sha256": b_mgr_sha,
            "candidate_manager_sha256": c_mgr_sha,
            "shared_worker_sha256": W_ZIP_SHA,
            "role_geometry": "r9-dual(共享=认证工人 zip,变量=经理 npz)",
        },
        "seeds_sha256": r8_statistics._canonical_sha256(seeds),
        "paired_data_sha256": r8_statistics._canonical_sha256(paired_hash_rows),
        "rules": {
            "gating": [
                {"key": "depth", "kind": "superiority", "direction": "higher",
                 "minimum_effect": float(DEPTH_RULE.minimum_effect),
                 "checks": ["mean_lcb", "exact_sign"]},
                {"key": WAGE_KEY, "kind": "noninferiority",
                 "direction": "higher",
                 "minimum_effect": wage_minimum_effect,
                 "formula": "minimum_effect = "
                            f"-{WAGE_NI_FRACTION} × anchor_wage_mean",
                 "checks": ["noninferiority_lcb"]},
                {"key": "died",
                 "kind": "mcnemar-noninferiority+derived-observed-line",
                 "noninferiority_margin": DEATH_MARGIN,
                 "checks": ["noninferiority_upper_bound",
                            "observed_within_derived_line"]},
            ],
            "record_only": [rule.key for rule in RECORD_ONLY_RULES],
        },
        "metrics": metrics,
        "record_only_metrics": record_only_metrics,
        "death_noninferiority": death_report,
        "near_line_notes": near_line_notes,
        "verdict": {
            "status": "PASS" if not failed_checks else "FAIL",
            "checks": checks,
            "failed_checks": failed_checks,
        },
    }
    json.dumps(result, allow_nan=False)
    return result


def select_winner(quals: dict, eligible: list) -> str:
    """胜者拣选(末令):过门臂比 depth 配对均差(主指标)。

    带宽 ``WINNER_DEPTH_TIE_BAND`` 内视为并列;并列破格:①合并池逐种子
    died 总数更低者;②仍平取 r9-mfresh(奥卡姆:课程臂须以可见深度优势
    自证)。ret 带已撤出拣选(照记入 quals,仅存档不裁)。
    """
    require(bool(eligible), "胜者拣选需要至少一个过门臂")
    require(all(name in quals for name in eligible), "过门臂缺 quals 读数")
    ms = {name: quals[name]["depth_paired_mean_ab"] for name in eligible}
    band = [name for name in eligible
            if max(ms.values()) - ms[name] <= WINNER_DEPTH_TIE_BAND]
    if len(band) > 1:
        dmin = min(quals[name]["pooled_died"] for name in band)
        band = [name for name in band if quals[name]["pooled_died"] == dmin]
        return "r9-mfresh" if "r9-mfresh" in band else band[0]
    return band[0]


def analysis_digest(analysis: dict) -> dict:
    """ledger 判决摘要:主判决行首列 depth 读数(回炉令 3)。"""
    depth_m = analysis["metrics"]["depth"]
    wage_m = analysis["metrics"][WAGE_KEY]
    death = analysis["death_noninferiority"]
    return {
        "depth": {
            "paired_mean": round(depth_m["improvement_mean"], 3),
            "lcb": round(depth_m["lower_confidence_bound"], 3),
            "sign_p": depth_m["sign_test"]["one_sided_exact_p_value"],
            "wins": depth_m["sign_test"]["wins_above_minimum_effect"],
            "non_ties": depth_m["sign_test"]["non_ties"],
            "passed": depth_m["passed"],
        },
        "wage_ni": {
            "lcb": round(wage_m["lower_confidence_bound"], 3),
            "line": round(wage_m["noninferiority"]["minimum_effect"], 3),
            "anchor_wage_mean": round(
                wage_m["noninferiority"]["anchor_wage_mean"], 3),
            "passed": wage_m["passed"],
        },
        "death": {
            "baseline_deaths": death["baseline_deaths"],
            "candidate_deaths": death["candidate_deaths"],
            "mcnemar_ucb": round(
                death["candidate_minus_baseline_risk_upper_bound"], 4),
            "derived_line": death["derived_observed_line"]["line_deaths"],
            "passed": death["passed"],
        },
        "record_only": {
            key: round(value["improvement_mean"], 3)
            for key, value in analysis["record_only_metrics"].items()},
        "failed_checks": analysis["verdict"]["failed_checks"],
        "near_line_notes": analysis["near_line_notes"],
        "paired_data_sha16": analysis["paired_data_sha256"][:16],
    }


# ---- G0-6 遗留义务:旧端点全表重放(E-fix REF_BITEQ) ----

def probe_mode_seq(windows) -> str:
    return "".join("FDR"[int(w["opt"])] + ("†" if w["reason"] == "death" else "")
                   for w in windows)


def g0_6_replay():
    lo, hi = R8_FINAL_SEEDS
    seeds = list(range(lo, hi + 1))
    for name, spec in R8_FINAL.items():
        archive_path = EVAL / f"{name}.json"
        actual_archive_sha = sha256(archive_path)
        require(actual_archive_sha == spec["archive_sha256"],
                f"R8 终考档案字节漂移:{name}:{actual_archive_sha}")
        # 旧档案系 E-fix 前协议束产物,禁绑当前运行时身份;按钉死字节 sha +
        # tag/seeds/worker 期望读取(内部 agg↔rows 复核照走)。
        doc = read_eval_archive(
            archive_path, expected_tag=name, expected_seeds=seeds,
            expected_worker_sha256=spec["worker_sha256"],
            expected_manager_sha256=M29_SHA)
        ref = by_seed(doc["rows"], R8_FINAL_SEEDS)
        worker_zip = spec["worker_zip"]
        require(sha256(worker_zip) == spec["worker_sha256"],
                f"G0-6 重放工人 sha 漂移:{worker_zip}")
        out = R9 / "g0" / f"g0-{name}.{time.time_ns()}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [PY, str(G0_PROBE), str(out),
               ",".join(str(s) for s in seeds),
               str(worker_zip), f"{name}.json", "old"]
        rc = run(cmd, f"g0-{name}.{time.time_ns()}.log", timeout=5_400)
        if rc != 0 or not out.exists():
            log({"event": "g0_6_crash", "archive": name, "rc": rc,
                 "note": "重放进程失败,按崩溃条款重试一次"})
            rc = run(cmd, f"g0-{name}.{time.time_ns()}.log", timeout=5_400)
            if rc != 0 or not out.exists():
                why = f"G0-6 重放进程连败:{name}(rc={rc})"
                log({"event": "STOP", "why": why})
                attention(why)
                raise OperationalFailure(why)
        records = json.loads(out.read_text())
        require(sorted(r["seed"] for r in records) == seeds,
                f"G0-6 重放种子集不完整:{name}")
        bad = []
        for rec in records:
            row = ref[rec["seed"]]
            if (abs(rec["ret"] - row["ret"]) > 1e-9
                    or int(rec["depth"]) != int(row["depth"])
                    or bool(rec["died"]) != bool(row["died"])
                    or probe_mode_seq(rec["windows"]) != row["mode_seq"]):
                bad.append(int(rec["seed"]))
        log({"event": "g0_6_replay", "archive": name, "n": len(records),
             "n_exact": len(records) - len(bad),
             "mismatch_seeds": bad[:20],
             "mismatch_total": len(bad),
             "records_sha16": sha16(out),
             "verdict": "REF_BITEQ" if not bad else "FAIL"})
        if bad:
            why = (f"G0-6 位级对账失配:{name} 计 {len(bad)} 种子——"
                   "旧端点重放未复现 R8 终考,按 E-fix 回退条款人工审理")
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
    log({"event": "g0_6_verdict", "status": "REF_BITEQ",
         "note": "R8 终考 512 局旧端点全表位级复现;E-fix G0-6 遗留义务清偿"})


# ---- preflight ----

def preflight():
    require_calibrated_protocol()
    require(G0_PROBE.is_file(), f"G0-6 探针缺失:{G0_PROBE}")
    require(W_ZIP_SRC.is_file() and sha256(W_ZIP_SRC) == W_ZIP_SHA,
            "认证工人发布件缺失或 sha 漂移")
    require(M29_NPZ.is_file() and sha256(M29_NPZ) == M29_SHA,
            "M29 经理 npz 缺失或 sha 漂移")
    for name, spec in R8_FINAL.items():
        require((EVAL / f"{name}.json").is_file(), f"R8 终考档案缺失:{name}")
        require(spec["worker_zip"].is_file(), f"G0-6 重放工人缺失:{spec['worker_zip']}")
    staged_sha = _stage_eval_file(W_ZIP_SRC, STAGED_WORKER,
                                  expected_sha256=W_ZIP_SHA)
    tags = list(TAG_ANCHOR.values())
    tags += [f"{arm}-{pool}-{POOLS[pool][0]}" for arm in ARMS for pool in ("a", "b")]
    tags += [f"r9-final-{arm}-{POOL_FINAL[0]}" for arm in ARMS]
    for t in tags:
        require(not (EVAL / f"{t}.json").exists(),
                f"目标档案已存在:{t}(重启协议:先 .void)")
    for arm in ARMS:
        require(not (RUNS / arm).exists(), f"运行目录残留:{arm}(重启协议:先归档)")
    log({"event": "preflight_ok", "prereg": "docs/PREREG-R9-manager-reeducation.md",
         "protocol_version": PROTOCOL_VERSION,
         "worker_zip_sha16": W_ZIP_SHA[:16], "staged_sha16": staged_sha[:16],
         "m29_sha16": M29_SHA[:16],
         "r8_archives": {name: spec["archive_sha256"][:16]
                         for name, spec in R8_FINAL.items()},
         "target_tags": tags})


# ---- 主序列 ----

def main():
    try:
        R9.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(R9 / ".driver.lock", "R9 驱动"):
            _main()
    except (OperationalFailure, OutputReservationError) as e:
        log({"event": "OPERATIONAL_FAILURE", "why": str(e)})
        attention("运维失败:\n" + str(e))
        raise SystemExit(2) from e
    except Exception as e:   # 条款兜底:任何未预期异常必须入册,不许无声死亡
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("驱动异常死亡:\n" + traceback.format_exc())
        raise


def _main():
    preflight()
    log({"event": "start", "prereg": "docs/PREREG-R9-manager-reeducation.md",
         "steps": STEPS, "arms": {n: e for n, e in ARMS.items()},
         "pools": {"a": seeds_arg(POOL_A), "b": seeds_arg(POOL_B),
                   "final": seeds_arg(POOL_FINAL)},
         "death_margin": DEATH_MARGIN, "familywise_alpha": FAMILYWISE_ALPHA,
         "note": "金池 9000/留出池零接触;--board 不用;本案不烧金牌"})

    # ---- G0-6:E-fix 遗留义务(先于一切新池消费) ----
    g0_6_replay()

    # ---- 新锚烧制(唯一新池消费;候选此刻不存在,零接触锚池) ----
    anchors: dict[str, tuple[dict, str]] = {}
    for pool in ("a", "b", "final"):
        tag = TAG_ANCHOR[pool]
        timeout = 3_600 if pool == "final" else 1_800
        result = exam_retry(tag, POOLS[pool], M29_NPZ, "legacy-v3", timeout)
        if result is None:
            why = f"锚烧制连败:{tag}"
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
        doc, doc_sha = result
        anchors[pool] = (doc, doc_sha)
        log({"event": "anchor", "tag": tag, "mean": doc["agg"]["ret_mean"],
             "sha16": doc_sha[:16], **depth_dashboard(doc["rows"])})

    # ---- 裁决线现场推导(推导式+数值入册;v31-D3-10 死亡线随各池锚在判决核内推导) ----
    ab_rows = anchors["a"][0]["rows"] + anchors["b"][0]["rows"]
    anchor_ab_mean = sum(r["ret"] for r in ab_rows) / len(ab_rows)
    abandon = round(anchor_ab_mean * ABANDON_FACTOR, 1)
    floor_repro = round(anchor_ab_mean * FLOOR_NUM / FLOOR_DEN, 1)
    anchor_wage_means = {
        pool: round(math.fsum(float(r[WAGE_KEY]) for r in anchors[pool][0]["rows"])
                    / len(anchors[pool][0]["rows"]), 3)
        for pool in anchors}
    log({"event": "derived_lines",
         "anchor_ab_mean": round(anchor_ab_mean, 3),
         "abandon_formula": f"ABANDON = anchor_ab_mean × {ABANDON_FACTOR}",
         "abandon": abandon,
         "floor_formula": f"FLOOR_REPRO = anchor_ab_mean × {FLOOR_NUM}/{FLOOR_DEN}",
         "floor_repro": floor_repro,
         "death_line_formula": "逐池:line_deaths = anchor_deaths + "
                               f"{DEATH_MARGIN}×n_pairs(判决核内随锚推导)",
         "anchor_deaths": {pool: sum(1 for r in anchors[pool][0]["rows"] if r["died"])
                           for pool in anchors},
         "wage_ni_formula": f"逐池:wage 非劣线 = −{WAGE_NI_FRACTION}×锚池 "
                            f"{WAGE_KEY} 均值(判决核内随锚推导,LCB≥线 含端点过)",
         "anchor_wage_means": anchor_wage_means,
         "wage_ni_lines": {pool: round(-WAGE_NI_FRACTION * mean, 3)
                           for pool, mean in anchor_wage_means.items()},
         "gate_note": "闸门肢 = depth 优越(主指标)+ wage 非劣 + 死亡;"
                      "ret/kills/worker_kills 系 record-only(回炉令)"})

    # ---- 双臂串行训练 ----
    npz: dict[str, pathlib.Path] = {}
    for name, extra in ARMS.items():
        cmd = [PY, "train/train_ppo.py", "--options", "--algo", "mppo",
               "--gamma", "1.0", "--max-steps", "3000", "--n-steps", "64",
               "--num-envs", "4", "--total-steps", str(STEPS),
               "--worker-zip", str(STAGED_WORKER),
               "--worker-zip-sha256", W_ZIP_SHA,
               "--manager-policy-observation-view", "raw-v4",
               "--run-name", name] + extra
        log({"event": "arm_start", "arm": name, "cmd_extra": extra})
        t0 = time.time()
        rc = run(cmd, f"train-{name}.log", timeout=14_400)   # 4h 挂死护栏
        sp = RUNS / name / "status.json"
        try:
            steps = json.loads(sp.read_text())["total_steps"] if sp.exists() else 0
        except Exception:
            steps = 0
        nt = zip_steps(RUNS / name / "model_final.zip")   # 达标闸唯一计步源(SB3 真链)
        log({"event": "arm_done", "arm": name, "rc": rc, "nt_zip": nt,
             "steps_status": steps, "dt_min": round((time.time() - t0) / 60, 1)})
        if rc != 0 or nt != STEPS:
            why = (f"{name} 训练未达标(rc={rc}, nt_zip={nt}, status={steps})"
                   "——命题未考,本版不追加重训(v25 条款承继)")
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
        out = RUNS / name / "policy.npz"
        if run([PY, "train/export_manager_npz.py",
                str(RUNS / name / "model_final.zip"), str(out)],
               f"export-{name}.log", timeout=600) != 0 or not out.exists():
            why = f"{name} npz 导出/parity 失败"
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
        npz[name] = out
        log({"event": "g_parity", "arm": name, "npz_sha16": sha16(out)})

    # ---- 臂考(每臂 2_123/2_124 各 128;候选经理视图 raw-v4) ----
    arm_docs: dict[tuple[str, str], tuple[dict, str]] = {}
    for name in ARMS:
        for pool in ("a", "b"):
            tag = f"{name}-{pool}-{POOLS[pool][0]}"
            result = exam_retry(tag, POOLS[pool], npz[name], "raw-v4", 1_800)
            if result is None:
                why = f"{name} 臂考连败:{tag}"
                log({"event": "STOP", "why": why})
                attention(why)
                raise OperationalFailure(why)
            doc, doc_sha = result
            arm_docs[(name, pool)] = (doc, doc_sha)
            anchor_map = by_seed(anchors[pool][0]["rows"], POOLS[pool])
            cand_map = by_seed(doc["rows"], POOLS[pool])
            paired_mean = sum(cand_map[s]["ret"] - anchor_map[s]["ret"]
                              for s in anchor_map) / len(anchor_map)
            log({"event": "arm_exam", "arm": name, "pool": pool, "tag": tag,
                 "mean": doc["agg"]["ret_mean"],
                 "paired_mean_vs_anchor": round(paired_mean, 3),
                 "sha16": doc_sha[:16], **depth_dashboard(doc["rows"])})

    # ---- 提前放弃闸(推导线 ABANDON;四读数全线以下 → 训练失败,免统计) ----
    exam_means = {f"{name}:{pool}": arm_docs[(name, pool)][0]["agg"]["ret_mean"]
                  for name in ARMS for pool in ("a", "b")}
    tripped = all(v < abandon for v in exam_means.values())
    log({"event": "abandon_check", "abandon": abandon, "means": exam_means,
         "tripped": tripped})
    if tripped:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"双臂双池均 <{abandon}(= 锚×{ABANDON_FACTOR})——训练失败,"
                    "再教育命题未考(免终考)"})
        attention("判决:训练失败,命题未考(ABANDON 推导线)")
        return

    # ---- 配对判决(r8 同械同式;development 语义,每臂两池) ----
    analyses: dict[tuple[str, str], dict] = {}
    for name in ARMS:
        for pool in ("a", "b"):
            doc, doc_sha = arm_docs[(name, pool)]
            anchor_doc, anchor_sha = anchors[pool]
            analysis = paired_judgment(
                anchor_doc, doc, baseline_sha256=anchor_sha,
                candidate_sha256=doc_sha, phase="development")
            analyses[(name, pool)] = analysis
            log({"event": "paired_analysis", "arm": name, "pool": pool,
                 "status": analysis["verdict"]["status"],
                 **analysis_digest(analysis)})

    # ---- 复现门/资格与胜者(过门臂中取配对均差最大;带内低死,再并列取 mfresh) ----
    quals = {}
    for name in ARMS:
        both_pass = all(analyses[(name, pool)]["verdict"]["status"] == "PASS"
                        for pool in ("a", "b"))
        pooled_rows = (arm_docs[(name, "a")][0]["rows"]
                       + arm_docs[(name, "b")][0]["rows"])
        pooled_mean = sum(r["ret"] for r in pooled_rows) / len(pooled_rows)
        anchor_ab = {**by_seed(anchors["a"][0]["rows"], POOL_A),
                     **by_seed(anchors["b"][0]["rows"], POOL_B)}
        cand_ab = {**by_seed(arm_docs[(name, "a")][0]["rows"], POOL_A),
                   **by_seed(arm_docs[(name, "b")][0]["rows"], POOL_B)}
        paired_diffs = [cand_ab[s]["ret"] - anchor_ab[s]["ret"]
                        for s in sorted(anchor_ab)]
        depth_diffs = [cand_ab[s]["depth"] - anchor_ab[s]["depth"]
                       for s in sorted(anchor_ab)]
        quals[name] = {
            "both_pools_pass": both_pass,
            # 主指标读数行首列(回炉令 3);亦为胜者拣选量(末令)。
            "depth_paired_mean_ab": round(
                sum(depth_diffs) / len(depth_diffs), 3),
            "depth_paired_wins_ab": sum(d > 0 for d in depth_diffs),
            "pooled_mean": round(pooled_mean, 3),
            "floor_pass": pooled_mean >= floor_repro,
            "paired_mean_ab": round(sum(paired_diffs) / len(paired_diffs), 3),
            "paired_wins_ab": sum(d > 0 for d in paired_diffs),
            "pooled_died": sum(1 for r in pooled_rows if r["died"]),
        }
    log({"event": "quals", **{n: quals[n] for n in ARMS}})

    pool_pass = [n for n in ARMS if quals[n]["both_pools_pass"]]
    if not pool_pass:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": "双臂复现门失败(depth 优越/wage 非劣/死亡肢未同时过)"
                        "——无胜者,再教育命题未答(功效外)",
             "arms": quals})
        attention("判决:双臂复现门失败,无胜者(深度仪表已随 arm_exam 入册)")
        return
    prelim = max(ARMS, key=lambda n: quals[n]["depth_paired_mean_ab"])
    if prelim not in pool_pass:
        log({"event": "substitution", "blocked": prelim,
             "why": quals[prelim],
             "note": "depth 配对均差胜者复现门拦截,由过门臂递补(v29 D3-2 同款)"})
    winner = select_winner(quals, pool_pass)
    log({"event": "winner", "arm": winner, **quals[winner],
         "selection_rule": f"depth 配对均差最大;带 {WINNER_DEPTH_TIE_BAND} 内"
                           "并列→①低死②mfresh(末令;ret 带已撤出拣选,"
                           "照记 quals 不裁)",
         "substituted": winner != prelim})

    # ---- 复现地板(推导线 FLOOR;胜者 A+B 合并均值) ----
    floor_note = ("(临线:距 FLOOR ≤1.0)"
                  if abs(quals[winner]["pooled_mean"] - floor_repro)
                  <= NEAR_LINE_FLOOR_GAP else "")
    if not quals[winner]["floor_pass"]:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"胜者 {quals[winner]['pooled_mean']} < {floor_repro}"
                    f"(= 锚×{FLOOR_NUM}/{FLOOR_DEN}){floor_note}——重训未复现"
                    "参考水平,再教育命题未考"})
        attention("判决:未复现参考水平(FLOOR 推导线)")
        return
    log({"event": "floor_check", "winner": winner,
         "pooled_mean": quals[winner]["pooled_mean"],
         "floor": floor_repro, "passed": True, "near_line": bool(floor_note)})

    # ---- 胜者终考(2_125 一次性 256 对)+ 终判(final 语义) ----
    final_tag = f"r9-final-{winner}-{POOL_FINAL[0]}"
    result = exam_retry(final_tag, POOL_FINAL, npz[winner], "raw-v4", 3_600)
    if result is None:
        why = f"终考连败:{final_tag}"
        log({"event": "STOP", "why": why})
        attention(why)
        raise OperationalFailure(why)
    final_doc, final_sha = result
    log({"event": "final_exam", "arm": winner, "tag": final_tag,
         "mean": final_doc["agg"]["ret_mean"], "sha16": final_sha[:16],
         **depth_dashboard(final_doc["rows"])})
    anchor_final_doc, anchor_final_sha = anchors["final"]
    final_analysis = paired_judgment(
        anchor_final_doc, final_doc, baseline_sha256=anchor_final_sha,
        candidate_sha256=final_sha, phase="final")
    log({"event": "final_analysis", "arm": winner,
         "status": final_analysis["verdict"]["status"],
         **analysis_digest(final_analysis)})

    # ---- 深度副判(科学结论;判「深度解锁」须三防呆仪表同时移动) ----
    anchor_rows = anchor_final_doc["rows"]
    final_rows = final_doc["rows"]
    gauges = gauge_report(anchor_rows, final_rows, TAG_ANCHOR["final"])
    anchor_l3 = l3_plus(anchor_rows)
    cand_l3 = l3_plus(final_rows)
    anchor_rate = gauges["dive_window_success_rate"]["anchor"]
    cand_rate = gauges["dive_window_success_rate"]["candidate"]
    rate_moved = (anchor_rate is not None and cand_rate is not None
                  and cand_rate > anchor_rate)
    if cand_l3 > anchor_l3 and rate_moved:
        depth_verdict = ("深度经济已动(L3+ 上移且潜成率上移;判『深度解锁』须"
                         "防呆仪表 1/3(交权中位/停留比)probe 复核同向后方可定谳)")
    elif cand_l3 <= anchor_l3:
        depth_verdict = f"再教育未解锁深度(L3+ {cand_l3} ≤ 锚 {anchor_l3})"
    else:
        depth_verdict = (f"带外(L3+ {anchor_l3}→{cand_l3},潜成率 "
                         f"{anchor_rate}→{cand_rate}),入册不叙事")
    log({"event": "depth_verdict", "anchor_l3": anchor_l3, "candidate_l3": cand_l3,
         "anchor_depth_hist": depth_hist(anchor_rows),
         "candidate_depth_hist": depth_hist(final_rows),
         "gauges": gauges, "verdict": depth_verdict,
         "note": "副判;王座/发布认定不在本案(防过度叙事,B8:不烧金牌)"})

    # ---- 终判入册(主判决行首列 depth 读数;回炉令 3) ----
    final_status = final_analysis["verdict"]["status"]
    near = final_analysis["near_line_notes"]
    digest = analysis_digest(final_analysis)
    depth_r = digest["depth"]
    wage_r = digest["wage_ni"]
    death_r = digest["death"]
    verdict_text = (
        f"R9 终判 {final_status}:depth 配对均差 {depth_r['paired_mean']:+.3f}"
        f"(LCB {depth_r['lcb']:+.3f},sign {depth_r['wins']}/"
        f"{depth_r['non_ties']},p={depth_r['sign_p']:.3g};"
        f"L3+ {anchor_l3}→{cand_l3})"
        f";wage 非劣{'过' if wage_r['passed'] else '未过'}"
        f"(LCB {wage_r['lcb']:+.3f} vs 线 {wage_r['line']:+.3f})"
        f";死亡{'过' if death_r['passed'] else '未过'}"
        f"(候选 {death_r['candidate_deaths']} vs 锚 "
        f"{death_r['baseline_deaths']},推导线 {death_r['derived_line']:.1f},"
        f"UCB {death_r['mcnemar_ucb']})"
        + ("" if final_status == "PASS"
           else f";未过肢:{final_analysis['verdict']['failed_checks']}")
        + f";深度副判:{depth_verdict}"
        + (f";临线注记:{near}" if near else ""))
    log({"event": "VERDICT_FINAL", "status": final_status, "arm": winner,
         **digest,
         "verdict": verdict_text,
         "candidate_depth_hist": depth_hist(final_rows),
         "anchor_depth_hist": depth_hist(anchor_rows),
         "candidate_died_seeds": died_seeds(final_rows),
         "anchor_died_seeds": died_seeds(anchor_rows),
         "gauges": gauges,
         "note": "发布/王座另案由主席裁;败臂/未考臂永不见 2_125 之外新池;"
                 "金池 9000 全程零接触"})
    attention(f"R9 终判 {final_status}({winner});{verdict_text}")


if __name__ == "__main__":
    import sys
    # 发车护栏(2026-07-31 运维事故:--help 被无视直接开跑,及时掐停,
    # 池零消耗):本驱动无 CLI 参数,任何 argv 一律拒绝退出——发车必须
    # 是裸调用的明确意图,不给口误留门。
    if len(sys.argv) > 1:
        print("run_r9_reeducation 不接受任何参数;裸调用即发车(发车在主席)。",
              file=sys.stderr)
        raise SystemExit(2)
    main()
