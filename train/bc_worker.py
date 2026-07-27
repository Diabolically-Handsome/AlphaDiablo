"""v23:FARM 操作脑 BC 热启动(docs/PREREG-v23.md D4)。

在位采集:冻结 H 经理 + 脚本教师(dispatch farm 分支;反射拍是包装器所有,
天然不入集;保险丝强制拍整拍剔除)。当前示范种子 2102000-2102127,只录 FARM
窗口；已消费 2100000-2100127 永久 burned。
产出 train/runs/bc-worker/policy_sd.pt(SB3 键名,--bc-init 用)+ bc_report.json。
候选选择:既有训练局内再作整局 validation；validation top-1 <0.95 或
训练侧样本 ≥300 的类之 validation 召回 <0.85 → 类加权 CE 重训一次
(BC 唯一重试)。候选冻结后，final held-out 才首次用于同阈值数据闸与回执。

E2 乙1′(PREREG-内容案-课⑤x④乙 E2):增教师 v2(bc-worker-v2)——
dispatch("farm") 前置预防饮分支 hp∈[0.5,0.65)∧belt>0→12。该目标必须
完全由 298 维可见状态决定；旧“每窗一次”隐藏闩会把同一血线合法态的
75% 标成非 12，实测在 recall=.60 时最低 FPR 仍 3.5%，不可学习。
世代旗 teacher_generation 1/2;v2 产物独立目录 runs/bc-worker-v2/
(v1 canonical 路径一字不动,规避 _previous 归档互斥);v2 demos 增逐样本
masks(env.action_masks() 现场捕获,唯一 on-manifold 真源);v2 回执独立
文件名 bc_report_v2.json + 独立 schema 标识 + 专用验证器(v1 验证器对
v2 件天然 fail-loud);n₁₂ 闸与 recall 门读数入回执(fail-closed)。
`python train/bc_worker.py` = v1(原样);`--v2 [--preventive-threshold 0.7]` = v2。

方案甲(2026-07-19 亲批):① v2 采集局数扩为 v1 × 3(_V2_COLLECTION_EPISODE_FACTOR;
rev6 诊断已查看旧 100..483 池，rev12 又发现 1000..1383 的 final 域被
覆盖诊断提前读取；后续 2101000..2101383 也已由一次性 producer 打开，
当前 active v2 池迁至预先未使用且不相交的 2103000..2103383；② v2 主训改类平衡
加权 CE(w_c = N/(K·n_c) 标准平衡式,类集 = 实际出现类)。两者只作用于 v2;
v1 采集局数/种子纪律与 v1 训练路径(train_bc 调用不传权)一字不动。

审计修复(2026-07-25):极稀有 a12 用全类平衡 CE 会把约 0.04% 正例放大
千倍，却没有显式约束误报；旧件虽 recall_12 过门，held-out 上曾产生 867
个假饮。v2 最终模型因此增加训练集专用的教师边界校准阶段：真实 masks
决定 a12 是否可达，hp 带与 feature 297 的可见本窗饮药闩进入 SB3 六张量内
的一条从优化第 0 步起隔离的保留神经通路，偏置只由 nested-fit episode
拟合；固定 validation/final-heldout 整局仅用于选型/最终硬门，不参与调参。
"""
import json
import hashlib
import math
import os
import pathlib
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import WorkerWindowEnv
from diablogym.options_env import (
    WORKER_DRINK_LATCH_FEATURE,
    WORKER_OBSERVATION_VIEW_A12_OVERLAY,
    dispatch,
)
from eval_contract import PROTOCOL_VERSION, exclusive_lock
from train_ppo import (_A12_FPR_MAX, _A12_HIGH_HP_FALSE_DRINK_MAX,
                       _A12_CALIBRATION_DRINK_LATCH_FEATURE,
                       _A12_CALIBRATION_HP_FEATURE,
                       _A12_CALIBRATION_PREDICATE,
                       _A12_CALIBRATION_SCHEMA_VERSION,
                       _A12_CALIBRATION_TRAIN_RECALL_TARGET,
                       _A12_LEGAL_NEGATIVE_PROBABILITY_MAX,
                       _A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX,
                       _A12_VISIBLE_HP_BOUNDARY_EPS,
                       _A12_PRECISION_MIN, _A12_PREDICTED_SHARE_MAX,
                       _A12_PREDICTED_SHARE_MIN, _A12_RECALL_MIN,
                       _A13_SPILLOVER_MAX, _BC_REPORT_SCHEMA_VERSION,
                       _BC_AUX_BEHAVIOR_METRIC_KEYS,
                       _BC_FINAL_HOLDOUT_MARKER_SCHEMA,
                       _BC_FINAL_SPLIT_SEED,
                       _BC_SELECTION_SPLIT_SEED,
                       _BC_SELECTION_VALIDATION_FRACTION,
                       _BC_V2_TEACHER_RECALL_MIN,
                       _BC_V2_DEMOS_SCHEMA_VERSION,
                       _BC_V2_PASS_KEYS,
                       _BC_V2_REPORT_SCHEMA_VERSION,
                       _BC_V2_COLLECTION_EPISODES,
                       _WORKER_BC_DEMO_SEEDS,
                       _WORKER_BC_FORBIDDEN_ACTIONS,
                       _WORKER_BC_REQUIRED_RECALL_ACTIONS,
                       _WORKER_BC_MIN_ACTION14_LABELS,
                       _WORKER_BC_MIN_ACTION14_EPISODES,
                       _assert_bc_final_holdout_pool_disjoint,
                       _implementation_bundle_sha256,
                       _bc_final_holdout_marker_path,
                       _bc_final_holdout_pool_spec,
                       _bc_v2_post_drink_coverage,
                       _validate_bc_final_holdout_marker,
                       bc_aux_behavior_gate, bc_aux_behavior_metrics)

OUT = ROOT / "train" / "runs" / "bc-worker"
OUT.mkdir(parents=True, exist_ok=True)
NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
DEMO_SEEDS = list(_WORKER_BC_DEMO_SEEDS)

# ---- E2 乙1′ 教师 v2(bc-worker-v2)常量注册(PREREG-内容案 E2/D7)----
OUT_V2 = ROOT / "train" / "runs" / "bc-worker-v2"  # v2 独立产物目录;v1 canonical 原封
TEACHER_GENERATION_V1 = 1
TEACHER_GENERATION_V2 = 2
_PREVENTIVE_HP_LOW = 0.5                # 预防带下界(闭;恰在 0.5 脑干反射之上,
                                        # 反射态归排水,教师原理上不可见)
_PREVENTIVE_THRESHOLD_MAIN = 0.65       # 主案预防阈(D7"预防阈"行)
# The historical 0.70 OC reused the same final episodes after observing 0.65.
# Rev13 disables it until an independent, training-reserved pool is registered.
_REGISTERED_PREVENTIVE_THRESHOLDS = (_PREVENTIVE_THRESHOLD_MAIN,)
_V2_FORBIDDEN_ACTIONS = (11,)           # v2 路径禁 11 允 12(守卫面不弱化;
                                        # v1 路径 _WORKER_BC_FORBIDDEN_ACTIONS 原封)
_N12_GATE_MIN = 122                     # n₁₂ 闸 ≥122(=审计点估 244 之半,D7)
_RECALL12_GATE_MIN = _BC_V2_TEACHER_RECALL_MIN
_BC_V2_REPORT_NAME = "bc_report_v2.json"  # v2 回执独立文件名(schema 隔离)
# 独立 schema 标识:取非 int 字符串——v1 验证器(train_ppo._validate_bc_report)
# 之键集合精确等断言 + _is_plain_int(schema_version)==1 断言对 v2 件双重必炸。
# ---- v2 a12 训练集校准（导出后仍是标准 SB3 64x64 MLP）----
if _A12_CALIBRATION_DRINK_LATCH_FEATURE != WORKER_DRINK_LATCH_FEATURE:
    raise RuntimeError("BC-v2 校准主动饮位与 Worker 观测契约漂移")

# ---- BC 候选选择集（最终 held-out 只许在选型完成后读取）----
# 逆频率加权后的优化问题比无权主训收敛慢。旧代码仍只给重试 8 epoch；
# protocol-v4 实测 validation 尚在 .9017 / recall10=.816 时便停，而同一
# 冻结目标到 12 epoch 已达 .9982 / .997。v2 的主训/重试更曾因相同
# objective、seed 和 8 epoch 成为逐位相同的假重试。
_BC_PRIMARY_EPOCHS = 8
_BC_WEIGHTED_RETRY_EPOCHS = 12

# ---- append-only replacement collection registries ----
_V2_COLLECTION_EPISODE_FACTOR = 3   # 方案甲 a:v2 采集局数 = v1 × 3
# 当前 active registry 为 v2 2103000..2103383、v1
# 2102000..2102127；所有旧池永久留在 burned/训练拒采表。
if list(DEMO_SEEDS) != list(range(DEMO_SEEDS[0],
                                  DEMO_SEEDS[0] + len(DEMO_SEEDS))):
    raise RuntimeError("v1 示范种子非连续升序整数:v2 种子延拓规则前提破坏")
DEMO_SEEDS_V2 = list(_BC_V2_COLLECTION_EPISODES)
if len(DEMO_SEEDS_V2) != _V2_COLLECTION_EPISODE_FACTOR * len(DEMO_SEEDS):
    raise RuntimeError("rev6 v2 replacement 种子数不等于 v1×3")

def artifact_provenance():
    return {
        "schema_version": _BC_REPORT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "implementation_sha256": _implementation_bundle_sha256(),
        "generator_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()).hexdigest(),
        "manager_npz_sha256": hashlib.sha256(NPZ.read_bytes()).hexdigest(),
    }


def _final_holdout_pool_spec(
        generation: int, seeds: list[int] | tuple[int, ...]) -> dict:
    """不可变 final-pool 身份；不含实现哈希，代码变化也不能重开同一池。"""
    return _bc_final_holdout_pool_spec(generation, seeds)


def _final_holdout_marker_path(
        out_dir: pathlib.Path, generation: int,
        seeds: list[int] | tuple[int, ...]) -> tuple[pathlib.Path, dict, str]:
    return _bc_final_holdout_marker_path(out_dir, generation, seeds)


def _fsync_directory(path: pathlib.Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _assert_final_holdout_unused(
        out_dir: pathlib.Path, generation: int,
        seeds: list[int] | tuple[int, ...]) -> None:
    marker, _, pool_sha256 = _final_holdout_marker_path(
        out_dir, generation, seeds)
    if marker.exists():
        raise RuntimeError(
            "BC 注册池已被一次性消费，禁止同池再次采集/评分:"
            f"generation={generation},pool_sha256={pool_sha256},"
            f"marker={marker}")
    _assert_bc_final_holdout_pool_disjoint(
        out_dir, generation, seeds)


def _mark_final_holdout_started(
        out_dir: pathlib.Path, generation: int,
        seeds: list[int] | tuple[int, ...], provenance: dict) -> dict:
    """Before any episode reset, durably burn the whole registered pool."""
    marker, spec, pool_sha256 = _final_holdout_marker_path(
        out_dir, generation, seeds)
    marker_parent_was_missing = not marker.parent.exists()
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker_parent_was_missing:
        # fsync the runs directory as well as the registry directory: without
        # this, a crash can lose the newly-created registry entry
        # even after the marker file itself was synchronized.
        _fsync_directory(marker.parent.parent)
    # Exact-pool O_EXCL alone cannot serialize two *different* marker names
    # whose episode sets overlap.  Hold one registry-wide lock across the
    # overlap scan and durable marker creation to close that TOCTOU window.
    with exclusive_lock(
            marker.parent / ".registry.lock",
            "BC final heldout registry"):
        if marker.exists():
            raise RuntimeError(
                "BC pool one-shot marker 已存在，禁止重采/重读:"
                f"{marker}")
        _assert_bc_final_holdout_pool_disjoint(
            out_dir, generation, seeds)
        record = {
            **spec,
            "pool_sha256": pool_sha256,
            "marker_schema_version": _BC_FINAL_HOLDOUT_MARKER_SCHEMA,
            "started_at_ns": time.time_ns(),
            "provenance": dict(provenance),
            "consumption_stage": "before_pool_collection",
        }
        payload = json.dumps(
            record, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")
        try:
            fd = os.open(
                str(marker), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(
                "BC pool one-shot marker 已存在，禁止重采/重读:"
                f"{marker}") from exc
        try:
            with os.fdopen(fd, "wb", closefd=True) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(marker.parent)
        except Exception:
            # 即使写入中断也保留 marker；存在性本身就是“可能已打开”的证据。
            raise
    return {
        **record,
        "marker_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _reject_same_generator_terminal(
        out_dir: pathlib.Path, report_name: str, provenance: dict) -> None:
    """direct CLI 也不得把同生成器终态移入 _previous 后绕过 launcher。"""
    candidates = [pathlib.Path(out_dir) / report_name]
    candidates.extend(sorted(
        (pathlib.Path(out_dir) / "_previous").glob(
            f"*/{report_name}")))
    for report in candidates:
        if not report.is_file():
            continue
        try:
            record = json.loads(report.read_text())
        except (OSError, ValueError):
            continue
        if (
            isinstance(record, dict)
            and record.get("data_gate") in {"PASS", "FAIL"}
            and record.get("implementation_sha256")
            == provenance.get("implementation_sha256")
            and record.get("generator_sha256")
            == provenance.get("generator_sha256")
        ):
            raise RuntimeError(
                "当前 implementation/generator 已有 BC 科学终态，"
                f"禁止 direct CLI 重试或归档绕过:{report}")


def begin_output_attempt(provenance: dict | None = None):
    """去掉 canonical 旧产物，防止本次 FAIL 后下游误吃上次权重。"""
    provenance = artifact_provenance() if provenance is None else provenance
    _assert_final_holdout_unused(
        OUT, TEACHER_GENERATION_V1, DEMO_SEEDS)
    _reject_same_generator_terminal(
        OUT, "bc_report.json", provenance)
    old = [OUT / name for name in ("policy_sd.pt", "bc_report.json", "demos.npz")
           if (OUT / name).exists()]
    if old:
        archive = OUT / "_previous" / str(time.time_ns())
        archive.mkdir(parents=True)
        for path in old:
            path.replace(archive / path.name)
    (OUT / "bc_report.json").write_text(json.dumps({"data_gate": "RUNNING"}))


def write_report(record):
    tmp = OUT / "bc_report.tmp.json"
    tmp.write_text(json.dumps(record, ensure_ascii=False))
    tmp.replace(OUT / "bc_report.json")


def teacher_action(env: WorkerWindowEnv) -> int:
    raw = env.oe.env._raw
    context = getattr(env.oe, "_worker_masks_and_distance", None)
    if callable(context):
        masks, nearest = context()
        return dispatch(
            "farm", raw, bool(masks[14]), action_mask=masks,
            nearest_engageable_distance=nearest)
    # Narrow compatibility branch for pure collection fixtures.  Production
    # WorkerWindowEnv always exposes the exact context above.
    masks = np.asarray(env.action_masks(), dtype=np.bool_)
    return dispatch(
        "farm", raw, bool(masks[14]), action_mask=masks)


def collect():
    env = WorkerWindowEnv(
        str(NPZ), max_steps=3000, rng_seed=0, seed_scope="bc-v1",
        # BC-v1 initializes/supervises the ordinary frozen Worker actor.  Its
        # demonstrations must use the same canonical protocol-v3 view as
        # rev14 PPO and deployment.
        legacy_policy_observation_view=True)
    X, Y, groups = [], [], []
    dropped = 0
    dropped_no_effect = 0
    dropped_no_effect_by_action = Counter()
    for i, seed in enumerate(DEMO_SEEDS):
        obs, _ = env.reset(seed=seed)
        while obs is not None:
            a = teacher_action(env)
            pair = (np.asarray(obs, dtype=np.float32), a)
            obs2, w, term, trunc, info = env.step(a)
            if info.get("overridden"):
                dropped += 1          # 保险丝改写过的步整拍剔除
            elif "executed_action" not in info:
                raise RuntimeError("BC-v1 step 缺少 executed_action 回执")
            elif info["executed_action"] is None:
                # A legal proposal may still lose a dynamic native race or
                # degrade to an explicit wait.  Training that observation
                # against the unexecuted proposal recreates the action10
                # no-op attractor in supervised initialization.
                dropped_no_effect += 1
                dropped_no_effect_by_action[int(a)] += 1
            else:
                if int(info["executed_action"]) != int(a):
                    raise RuntimeError(
                        "BC-v1 executed_action 与教师提案不一致:"
                        f"requested={a},executed={info['executed_action']}")
                X.append(pair[0]); Y.append(pair[1]); groups.append(seed)
            # 局尽 next_window 返回 None(绝不滚新局——示范池纪律)
            obs = env.next_window() if (term or trunc) else obs2
        if (i + 1) % 16 == 0:
            print(f"  采集 {i+1}/{len(DEMO_SEEDS)} 局,{len(Y)} 对"
                  f"(保险丝剔除 {dropped},无效果剔除 {dropped_no_effect}"
                  f"{dict(sorted(dropped_no_effect_by_action.items()))})",
                  flush=True)
    # 示范池纪律断言:每个示范种子恰好一局,零兜底滚局(否则数据混入未知种子)
    if env.stats["episodes"] != len(DEMO_SEEDS) or env.stats["reseeds"] != 0:
        raise RuntimeError(f"示范池种子纪律破坏: {env.stats}")
    print(f"示范:{len(Y)} 决策对,剔除保险丝拍 {dropped},"
          f"剔除无效果拍 {dropped_no_effect},"
          f"按动作 {dict(sorted(dropped_no_effect_by_action.items()))},"
          f"类分布 {dict(sorted(Counter(Y).items()))}", flush=True)
    env.close()
    groups_array = np.asarray(groups, dtype=np.int64)
    labels = np.asarray(Y, dtype=np.int64)
    if not np.array_equal(np.unique(groups_array), np.asarray(DEMO_SEEDS)):
        raise RuntimeError(
            "示范集没有精确覆盖当前固定种子 2102000..2102127")
    if np.isin(labels, _WORKER_BC_FORBIDDEN_ACTIONS).any():
        raise RuntimeError("示范集含禁采动作 11/12(11 恒掩归经理;12 系脚本教师"
                           "排水后不采——主权世代示范池纪律)")
    return np.stack(X), labels, groups_array


class PiHead(nn.Module):
    """与 SB3 MlpPolicy(64,64) 策略侧同构。"""

    def __init__(self, obs_dim=298, n_act=15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh())
        self.head = nn.Linear(64, n_act)

    def forward(self, x):
        return self.head(self.net(x))


def split_by_episode(groups):
    """整局切分，避免同一条确定性轨迹的相邻帧泄漏到 held-out。"""
    episodes = np.unique(groups)
    if len(episodes) < 2:
        raise ValueError("BC held-out 至少需要 2 个独立 episode")
    order = np.random.default_rng(_BC_FINAL_SPLIT_SEED).permutation(episodes)
    n_holdout = max(1, int(round(len(order) * 0.1)))
    holdout_episodes = order[:n_holdout]
    ho_mask = np.isin(groups, holdout_episodes)
    tr, ho = np.flatnonzero(~ho_mask), np.flatnonzero(ho_mask)
    if len(tr) == 0 or len(ho) == 0:
        raise ValueError("BC episode split 产生了空训练集或空 held-out")
    return tr, ho, holdout_episodes


def _split_fit_validation_by_episode(groups):
    """在既有训练 episode 内再切 validation，最终 held-out 完全隔离。

    外层 ``split_by_episode`` 仍定义报告/最终闸所用的 canonical held-out；
    本函数只对其训练侧 episode 作第二次确定性整局切分。validation 仅用于
    首训是否触发唯一 retry，绝不进入梯度。至少需要三局，才能同时保留
    fit / validation / final-held-out 三个非空集合。
    """
    groups = np.asarray(groups)
    outer_train, outer_holdout, _ = split_by_episode(groups)
    training_episodes = np.unique(groups[outer_train])
    if len(training_episodes) < 2:
        raise ValueError(
            "BC 候选选择至少需要 3 个独立 episode"
            "(fit/validation/final-held-out 各非空)")
    order = np.random.default_rng(
        _BC_SELECTION_SPLIT_SEED).permutation(training_episodes)
    n_validation = max(
        1, int(round(len(order) * _BC_SELECTION_VALIDATION_FRACTION)))
    # round(0.1*n) 在当前池不会吞光 fit；仍显式钳住以防未来小池改动。
    n_validation = min(n_validation, len(order) - 1)
    validation_episodes = order[:n_validation]
    validation_mask = np.isin(groups, validation_episodes)
    outer_train_mask = np.zeros(len(groups), dtype=np.bool_)
    outer_train_mask[outer_train] = True
    fit = np.flatnonzero(outer_train_mask & ~validation_mask)
    validation = np.flatnonzero(outer_train_mask & validation_mask)
    if len(fit) == 0 or len(validation) == 0:
        raise ValueError("BC nested episode split 产生空 fit 或 validation")
    if (np.intersect1d(fit, validation).size
            or np.intersect1d(fit, outer_holdout).size
            or np.intersect1d(validation, outer_holdout).size):
        raise RuntimeError("BC nested episode split 集合相交")
    return fit, validation, validation_episodes


def _require_zero_exact_label_conflicts(X, Y) -> None:
    """A2 demos-validity 硬断言:同一精确观测不得携带互斥标签。"""
    import hashlib as _hashlib
    seen: dict = {}
    for i in range(len(X)):
        key = _hashlib.blake2b(X[i].tobytes(), digest_size=16).digest()
        prev = seen.get(key)
        if prev is None:
            seen[key] = int(Y[i])
        elif prev != int(Y[i]):
            raise RuntimeError(
                f"BC demos 标签冲突:同一观测行携带动作 {prev} 与 {int(Y[i])}")


def _bc_quality_gate_passes(top1, recalls) -> bool:
    """v1/v2 共用候选质量条件；输入域由调用者决定。"""
    return float(top1) >= 0.95 and all(
        float(recall) >= 0.85 for recall in recalls.values())


def _bc_v2_quality_gate_passes(top1, recalls) -> bool:
    """v2 通用分类门；a12 只走其独立的安全/召回行为门。

    a12 有单独注册的 recall/FPR/高血误饮/占比/13 溢出约束和训练集校准。
    把它同时塞进 ``>=300 类 recall>=.85`` 会在 n12 随采集扩容跨过 300
    时突然把专用 ``recall>=.5`` 门抬成 .85，且与冻结校准目标 .75 冲突。
    """
    return float(top1) >= 0.95 and all(
        int(action) == 12 or float(recall) >= 0.85
        for action, recall in recalls.items())


def _require_v1_action14_coverage(labels, groups) -> dict:
    """Fail before candidate training when the upgrade target is too sparse.

    Action 14 is a high-value but naturally rare event.  Letting it inherit the
    generic ``>=300`` eligibility rule made it disappear from both candidate
    selection and the final receipt even when the aggregate classifier looked
    excellent.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if (labels.ndim != 1 or groups.shape != labels.shape
            or labels.dtype != np.int64 or groups.dtype != np.int64):
        raise RuntimeError(
            "BC-v1 a14 覆盖检查要求同形一维 int64 labels/groups")
    selected = labels == 14
    readings = {
        "labels": int(selected.sum()),
        "episodes": int(np.unique(groups[selected]).size),
    }
    if (readings["labels"] < _WORKER_BC_MIN_ACTION14_LABELS
            or readings["episodes"] < _WORKER_BC_MIN_ACTION14_EPISODES):
        raise RuntimeError(
            "BC-v1 a14 严格升级覆盖不足:"
            f"labels={readings['labels']}"
            f"<{_WORKER_BC_MIN_ACTION14_LABELS},"
            f"episodes={readings['episodes']}"
            f"<{_WORKER_BC_MIN_ACTION14_EPISODES}")
    return readings


def _reserve_a12_calibration_path(model: PiHead) -> None:
    """从优化第 0 步起隔离 3+1 个校准神经元，避免事后删除普通类表示。"""
    if not isinstance(model, PiHead):
        raise TypeError("a12 保留通路只接受 PiHead")
    w0, b0 = model.net[0].weight, model.net[0].bias
    w1, b1 = model.net[2].weight, model.net[2].bias
    wa = model.head.weight
    with torch.no_grad():
        w0[61:64].zero_()
        b0[61:64].zero_()
        w1[:, 61:64].zero_()
        w1[63].zero_()
        b1[63].zero_()
        wa[:, 63].zero_()

    masks = []
    for parameter in (w0, b0, w1, b1, wa):
        masks.append(torch.ones_like(parameter))
    masks[0][61:64].zero_()
    masks[1][61:64].zero_()
    masks[2][:, 61:64].zero_()
    masks[2][63].zero_()
    masks[3][63].zero_()
    masks[4][:, 63].zero_()
    model._a12_reservation_hooks = [
        parameter.register_hook(
            lambda gradient, live_mask=live_mask: gradient * live_mask)
        for parameter, live_mask in zip((w0, b0, w1, b1, wa), masks)
    ]


def train_bc(X, Y, groups, class_weights=None, *,
             epochs: int = _BC_PRIMARY_EPOCHS, masks=None,
             reserve_a12_path: bool = False,
             required_recall_actions=()):
    """只在 fit 上训练，并只在 validation 上返回候选选择读数。

    最终 held-out 不在本函数的训练、逐类计数或评分路径中；调用者必须在
    所有 retry/选型完成之后，显式调用 ``_score_bc_model`` 做最终一次闸评。
    """
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("BC epochs 必须是正整数")
    if masks is not None:
        masks = np.asarray(masks)
        if masks.shape != (len(Y), 15) or masks.dtype != np.bool_:
            raise ValueError(
                f"BC validation masks 形状/dtype 非法:{masks.shape}/{masks.dtype}")
    torch.manual_seed(23)
    fit, validation, _ = _split_fit_validation_by_episode(groups)
    outer_train, _, _ = split_by_episode(groups)
    model = PiHead(X.shape[1])
    if reserve_a12_path:
        _reserve_a12_calibration_path(model)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    wt = None
    if class_weights is not None:
        wt = torch.as_tensor(class_weights, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X[fit]), torch.from_numpy(Y[fit]))
    dl = torch.utils.data.DataLoader(
        ds, batch_size=512, shuffle=True,
        generator=torch.Generator().manual_seed(23))
    for epoch in range(epochs):
        tot = cnt = correct = 0
        for xb, yb in dl:
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb, weight=wt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(yb); cnt += len(yb)
            correct += int((logits.argmax(1) == yb).sum())
        print(f"BC epoch {epoch}: loss {tot/cnt:.4f} acc {correct/cnt:.3f}", flush=True)
    # 候选选择只读 validation；门槛类计数也只看 outer training episodes，
    # 防止 final held-out 标签通过“是否达到 300”侧信道改变 retry 路径。
    top1, recalls = _score_bc_model_indices(
        model, X, Y, validation, eligibility_indices=outer_train,
        masks=masks, required_recall_actions=required_recall_actions)
    return model, top1, recalls


def export_sb3_sd(model):
    sd = {
        "mlp_extractor.policy_net.0.weight": model.net[0].weight,
        "mlp_extractor.policy_net.0.bias": model.net[0].bias,
        "mlp_extractor.policy_net.2.weight": model.net[2].weight,
        "mlp_extractor.policy_net.2.bias": model.net[2].bias,
        "action_net.weight": model.head.weight,
        "action_net.bias": model.head.bias,
    }
    return {k: v.detach().clone() for k, v in sd.items()}


def _masked_bc_predictions(model, X, masks=None) -> np.ndarray:
    """模型 argmax；v2 必须消费采集时真实 masks，v1 传 None 保持原语义。"""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(X, dtype=np.float32)))
        if masks is not None:
            masks = np.asarray(masks)
            if masks.shape != tuple(logits.shape) or masks.dtype != np.bool_:
                raise RuntimeError(
                    f"BC 评分 masks 形状/dtype 非法:{masks.shape}/{masks.dtype}")
            logits = torch.where(
                torch.from_numpy(masks), logits,
                torch.full_like(logits, -1e8))
        return logits.argmax(1).cpu().numpy()


def _score_bc_model_indices(model, X, Y, score_indices, *,
                            eligibility_indices, masks=None,
                            required_recall_actions=()):
    """在显式行集评分；逐类门槛计数只读取显式 eligibility 行集。"""
    X = np.asarray(X)
    Y = np.asarray(Y)
    score_indices = np.asarray(score_indices, dtype=np.int64)
    eligibility_indices = np.asarray(eligibility_indices, dtype=np.int64)
    if (score_indices.ndim != 1 or eligibility_indices.ndim != 1
            or len(score_indices) == 0 or len(eligibility_indices) == 0):
        raise ValueError("BC 评分集/类别资格集必须是一维非空索引")
    required = tuple(required_recall_actions)
    if (any(not isinstance(action, (int, np.integer))
            or isinstance(action, (bool, np.bool_))
            or not 0 <= int(action) < 15 for action in required)
            or len(set(map(int, required))) != len(required)):
        raise ValueError("BC required_recall_actions 必须是无重复的 0..14 整数")
    pred = _masked_bc_predictions(
        model, X[score_indices],
        None if masks is None else np.asarray(masks)[score_indices])
    yh = Y[score_indices]
    top1 = float((pred == yh).mean())
    eligible_counts = Counter(Y[eligibility_indices].tolist())
    recalls = {}
    gated_classes = {
        int(k) for k, v in eligible_counts.items() if v >= 300
    }
    gated_classes.update(map(int, required))
    for c in sorted(gated_classes):
        selected = yh == c
        recalls[int(c)] = (
            float((pred[selected] == c).mean())
            if selected.sum() > 0 else 0.0)
    return top1, recalls


def _score_bc_model(model, X, Y, groups, masks=None,
                    required_recall_actions=()):
    """选型完成后，在固定 final held-out 上重算 top-1/逐类召回。

    校准会改写最终导出的六张量，所以 v2 必须在校准后重新评分，禁止沿用
    校准前的 validation 数字。v1 调用不传 masks，报告 schema 保持原样。
    """
    _, ho, _ = split_by_episode(groups)
    return _score_bc_model_indices(
        model, X, Y, ho, eligibility_indices=np.arange(len(Y)),
        masks=masks, required_recall_actions=required_recall_actions)


def main():
    provenance = artifact_provenance()
    begin_output_attempt(provenance)
    # ``collect`` itself traverses every registered episode and performs
    # whole-pool integrity checks.  Burn the pool before the first reset so a
    # crash after those reads can never leave an apparently unused final set.
    pool_marker = _mark_final_holdout_started(
        OUT, TEACHER_GENERATION_V1, DEMO_SEEDS, provenance)
    pool_evidence = {
        "final_pool_sha256": pool_marker["pool_sha256"],
        "final_holdout_marker_sha256": pool_marker["marker_sha256"],
    }
    write_report({
        "data_gate": "FAIL",
        "failure_stage": "pool_collection_started",
        "final_heldout_consumed": True,
        **pool_evidence,
        **provenance,
    })
    X, Y, groups = collect()
    action14_coverage = _require_v1_action14_coverage(Y, groups)
    print(f"v1 a14 严格升级覆盖 {action14_coverage}", flush=True)
    demos_tmp = OUT / "demos.tmp.npz"
    np.savez_compressed(demos_tmp, X=X, Y=Y, episode_id=groups)
    demos_tmp.replace(OUT / "demos.npz")
    tr, ho, holdout_episodes = split_by_episode(groups)
    fit, _, _ = _split_fit_validation_by_episode(groups)
    # train_bc 只把 fit/validation 送入梯度或候选读数；整池虽已采集，
    # 但已由 pre-collection marker 永久记为消费。
    model, top1, recalls = train_bc(
        X, Y, groups,
        required_recall_actions=_WORKER_BC_REQUIRED_RECALL_ACTIONS)
    retrained = False
    if not _bc_quality_gate_passes(top1, recalls):
        print(f"首训 validation 未达标(top1 {top1:.3f} 召回 {recalls})"
              "→ 类加权重训(唯一重试)",
              flush=True)
        counts = np.bincount(Y[fit], minlength=15).astype(np.float64)
        weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
        weights = weights / weights[weights > 0].mean()
        model, top1, recalls = train_bc(
            X, Y, groups, class_weights=weights,
            epochs=_BC_WEIGHTED_RETRY_EPOCHS,
            required_recall_actions=_WORKER_BC_REQUIRED_RECALL_ACTIONS)
        retrained = True
    if not _bc_quality_gate_passes(top1, recalls):
        # A2 修正案(总设计师 2026-07-27 批「A2方案为主」):策略质量线降为
        # 只记不裁——0.95/0.85 系 R5/R6(BC 当教师锚)时代遗留;R7 训练命令
        # 不消费本策略(teacher=KING_SD),demos 才是被消费物(dry-anchor 锚)。
        # 实测天花板(200ep+类权,burned-2_104 demos 离线诊断):top1≈0.88-0.90、
        # 稀有键≤0.43——门在 07-25 未过滤 v3 视图下结构性不可达。
        print(f"候选质量线未达(top1 {top1:.3f} 召回 {recalls})——A2 只记不裁,"
              "继续发布流程(质量读数随报告落档)", flush=True)
    # 候选已冻结；final held-out 现在才进入模型评分路径。
    top1, recalls = _score_bc_model(
        model, X, Y, groups,
        required_recall_actions=_WORKER_BC_REQUIRED_RECALL_ACTIONS)
    # A2:发布门 = demos-validity(池覆盖/局纪律/a14 覆盖已由 collect 与
    # _require_v1_action14_coverage 硬断言;此处补零标签冲突硬断言)。
    # held_out_top1/class_recalls 保持原字段落档,身份与逐位复算链原封。
    _require_zero_exact_label_conflicts(X, Y)
    ok = True
    report = {
        "pairs": len(Y), "held_out_top1": round(top1, 4),
        "held_out_pairs": int(len(ho)),
        "held_out_episodes": [int(x) for x in sorted(holdout_episodes)],
        "class_recalls": recalls, "class_weighted_retry": retrained,
        **pool_evidence,
        "data_gate": "PASS" if ok else "FAIL", **provenance}
    if not ok:
        write_report(report)
        raise RuntimeError(
            f"BC 数据闸 FAIL(top1={top1:.3f}, recalls={recalls});"
            "拒绝覆写 policy_sd.pt")
    policy_tmp = OUT / "policy_sd.tmp.pt"
    if artifact_provenance() != provenance:
        raise RuntimeError("BC worker 运行期间实现/引擎/内容/经理发生漂移")
    torch.save(export_sb3_sd(model), policy_tmp)
    policy_tmp.replace(OUT / "policy_sd.pt")
    report["policy_sha256"] = hashlib.sha256(
        (OUT / "policy_sd.pt").read_bytes()).hexdigest()
    report["demos_sha256"] = hashlib.sha256(
        (OUT / "demos.npz").read_bytes()).hexdigest()
    write_report(report)
    print(f"held-out top-1 {top1:.3f} 召回 {recalls} retry={retrained} "
          f"→ 数据闸 PASS;已存 {OUT}/policy_sd.pt", flush=True)


# ==== E2 乙1′ v2 件(以下全部新增;v1 面(上文)一字不动)====

def forbidden_actions_for_generation(generation: int) -> tuple[int, ...]:
    """禁采断言世代条件化(E2):v1 禁 (11,12) 原封;v2 禁 11 允 12。"""
    if generation == TEACHER_GENERATION_V1:
        return tuple(_WORKER_BC_FORBIDDEN_ACTIONS)   # (11, 12) 原封
    if generation == TEACHER_GENERATION_V2:
        return _V2_FORBIDDEN_ACTIONS                 # 禁 11 允 12(守卫面不弱化)
    raise ValueError(f"未知教师世代: {generation!r}(只有 1/2)")


def teacher_v2_preventive_trigger(
        observation, masks,
        preventive_threshold: float = _PREVENTIVE_THRESHOLD_MAIN) -> bool:
    """E2 教师 v2 的纯可观测前置谓词。

    ``hp∈[0.5, 阈) ∧ live m12 ∧ 本窗尚未主动饮``。hp 下界闭、上界开；
    belt/动作合法性只读采集时现场 mask；每窗饮历史只读 feature 297 的
    已公开符号位。目标不再读取教师对象内部状态或未入 298 维的窗口历史。
    """
    observation = np.asarray(observation)
    masks = np.asarray(masks)
    if observation.shape != (298,) or not np.issubdtype(
            observation.dtype, np.floating):
        raise ValueError(
            f"TeacherV2 观测必须是 (298,) 浮点数组:{observation.shape}/"
            f"{observation.dtype}")
    if masks.shape != (15,) or masks.dtype != np.bool_:
        raise ValueError(
            f"TeacherV2 masks 必须是 (15,) bool:{masks.shape}/{masks.dtype}")
    hp = float(observation[_A12_CALIBRATION_HP_FEATURE])
    latch = float(observation[_A12_CALIBRATION_DRINK_LATCH_FEATURE])
    if not math.isfinite(hp) or not math.isfinite(latch):
        raise ValueError("TeacherV2 可见 hp/主动饮位必须有限")
    no_prior_window_drink = latch >= 0.0
    return bool(
        hp >= _PREVENTIVE_HP_LOW - _A12_VISIBLE_HP_BOUNDARY_EPS
        and hp < preventive_threshold - _A12_VISIBLE_HP_BOUNDARY_EPS
        and masks[12]
        and no_prior_window_drink)


class TeacherV2:
    """E2 乙1′ 教师 v2：可观测的预防饮分支优先于 dispatch。

    每次决策只读当前 298 维观测与现场 mask；对象本身不保存“本窗是否
    喝过”的隐藏状态。feature 297 显式公开该位，故相同观测永远同一标签。
    """

    def __init__(self, preventive_threshold: float = _PREVENTIVE_THRESHOLD_MAIN):
        if preventive_threshold not in _REGISTERED_PREVENTIVE_THRESHOLDS:
            raise ValueError(
                f"预防阈 {preventive_threshold!r} 未注册:rev13 仅允许 "
                "0.65 主案；旧 0.70 OC 没有独立 fresh pool")
        self.preventive_threshold = float(preventive_threshold)

    def begin_window(self) -> None:
        """兼容采集器的窗口通知；教师目标本身保持无状态。"""

    def action(self, env, observation=None, masks=None) -> int:
        raw = env.oe.env._raw
        if observation is None:
            builder = getattr(
                env.oe, "_worker_policy_observation", None)
            observation = (
                builder(WORKER_OBSERVATION_VIEW_A12_OVERLAY)
                if callable(builder)
                else env.oe._worker_obs()
            )
        if masks is None:
            masks = np.asarray(env.action_masks(), dtype=np.bool_)
        observed_hp = float(
            np.asarray(observation)[_A12_CALIBRATION_HP_FEATURE])
        raw_hp = raw["hp"] / max(1, raw["max_hp"])
        if not math.isclose(
                observed_hp, raw_hp, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(
                "TeacherV2 决策态 obs/raw hp 错位:"
                f"{observed_hp} != {raw_hp}")
        visible_latch = float(
            np.asarray(observation)[
                _A12_CALIBRATION_DRINK_LATCH_FEATURE])
        raw_preventive_state = (
            _PREVENTIVE_HP_LOW <= raw_hp < self.preventive_threshold
            and int(raw.get("belt_heals", 0)) > 0
            and visible_latch >= 0.0
        )
        if raw_preventive_state and not bool(np.asarray(masks)[12]):
            raise RuntimeError(
                "教师 v2 可见预防态动作 12 不在现场掩码内:"
                "on-manifold 示范纪律破坏")
        if teacher_v2_preventive_trigger(
                observation, masks, self.preventive_threshold):
            return 12
        context = getattr(env.oe, "_worker_masks_and_distance", None)
        if callable(context):
            masks, nearest = context()
            a = dispatch(
                "farm", raw, bool(masks[14]), action_mask=masks,
                nearest_engageable_distance=nearest)
        else:
            a = dispatch(
                "farm", raw, bool(masks[14]), action_mask=masks)
        if a == 12:
            # dispatch 内嵌 0.5 反射分支对教师应为死代码:开窗排水 + 反射尾部
            # 排水保证工人观测永为无反射态;走到此处即示范池纪律破坏,禁静默采。
            raise RuntimeError("教师 v2 见反射态(hp<0.5∧belt>0):排水失守,"
                               "a12 实标只许出自前置预防分支")
        return a


def collect_v2(preventive_threshold: float = _PREVENTIVE_THRESHOLD_MAIN,
               manager_npz: str | pathlib.Path = NPZ,
               manager_sha256: str | None = None):
    """E2:主权开采集环真实执行、真实入池(on-manifold,禁反事实标签)。

    与 v1 collect() 使用同一真实执行纪律:
      - 逐样本 masks 系决策态 env.action_masks() 现场捕获(唯一真源;
        a12 之 belt 位自 obs belt 维反推系第二真源,禁用);
      - overridden 与 executed_action=None 整拍剔除（拒绝/无效果拍不入池）;
      - TeacherV2 标签只依当前 298 维 hp/主动饮位与现场 mask；
        窗口通知不改变同观测标签；
      - 禁采断言世代条件化:v2 禁 11 允 12;
      - 当前采集环消费 DEMO_SEEDS_V2(v1 × 3,固定
        2103000..2103383)；v1 consume fresh 2102000..2102127。
    返回 (X, labels, groups, masks, belts);belts 系逐样本决策态腰带读数
    (腰带经济回执供源)。
    """
    teacher = TeacherV2(preventive_threshold)
    manager_path = pathlib.Path(manager_npz)
    if not manager_path.is_file():
        raise RuntimeError(f"BC-v2 经理 npz 不存在:{manager_path}")
    if manager_sha256 is None:
        manager_sha256 = hashlib.sha256(
            manager_path.read_bytes()).hexdigest()
    if (not isinstance(manager_sha256, str) or len(manager_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in manager_sha256)):
        raise RuntimeError("BC-v2 manager_sha256 非法")
    env = WorkerWindowEnv(
        str(manager_path), max_steps=3000, rng_seed=0,
        manager_sha256=manager_sha256, seed_scope="bc-v2",
        # Teacher-v2 and the A12 gate require the registered overlay: an exact
        # v3 base plus reversible free-slot/drink-latch fields.  Keep this
        # explicit so defaults cannot rewrite targets or feed the inherited
        # actor an unrecoverable filtered v4 row.
        legacy_policy_observation_view=False)
    X, Y, M, groups, belts = [], [], [], [], []
    dropped = 0
    dropped_no_effect = 0
    dropped_no_effect_by_action = Counter()
    for i, seed in enumerate(DEMO_SEEDS_V2):
        obs, _ = env.reset(seed=seed)
        teacher.begin_window()            # reset 即首窗,开闩
        while obs is not None:
            masks = np.asarray(env.action_masks(), dtype=bool)  # 现场捕获
            a = teacher.action(env, obs, masks)
            if not masks[a]:
                raise RuntimeError(
                    f"教师 v2 提案 {a} 不在现场掩码内(a12 须 m[12]=True):"
                    "on-manifold 示范纪律破坏")
            belt = int(env.oe.env._raw.get("belt_heals", 0))
            pair = (np.asarray(obs, dtype=np.float32), a, masks, belt)
            obs2, w, term, trunc, info = env.step(a)
            if info.get("overridden"):
                dropped += 1              # 保险丝改写过的步整拍剔除(v1 条款原封)
            elif "executed_action" not in info:
                raise RuntimeError("BC-v2 step 缺少 executed_action 回执")
            elif info["executed_action"] is None:
                dropped_no_effect += 1
                dropped_no_effect_by_action[int(a)] += 1
            else:
                if int(info["executed_action"]) != int(a):
                    raise RuntimeError(
                        "BC-v2 executed_action 与教师提案不一致:"
                        f"requested={a},executed={info['executed_action']}")
                X.append(pair[0]); Y.append(pair[1]); M.append(pair[2])
                groups.append(seed); belts.append(pair[3])
            farm_window_end = bool(info.get("farm_window_end", False))
            if farm_window_end:
                # 保留显式窗口通知接口，便于未来无状态教师做逐窗审计；
                # 它不得改变同一可见状态的目标动作。
                teacher.begin_window()
            if term or trunc:
                # 局尽 next_window 返回 None(绝不滚新局——示范池纪律)
                obs = env.next_window()
            else:
                obs = obs2
        if (i + 1) % 16 == 0:
            print(f"  v2 采集 {i+1}/{len(DEMO_SEEDS_V2)} 局,{len(Y)} 对"
                  f"(保险丝剔除 {dropped},"
                  f"无效果剔除 {dropped_no_effect}"
                  f"{dict(sorted(dropped_no_effect_by_action.items()))})",
                  flush=True)
    if env.stats["episodes"] != len(DEMO_SEEDS_V2) or env.stats["reseeds"] != 0:
        raise RuntimeError(f"示范池种子纪律破坏: {env.stats}")
    print(f"v2 示范:{len(Y)} 决策对,剔除保险丝拍 {dropped},"
          f"剔除无效果拍 {dropped_no_effect},"
          f"按动作 {dict(sorted(dropped_no_effect_by_action.items()))},"
          f"类分布 {dict(sorted(Counter(Y).items()))}", flush=True)
    env.close()
    groups_array = np.asarray(groups, dtype=np.int64)
    labels = np.asarray(Y, dtype=np.int64)
    if not np.array_equal(np.unique(groups_array), np.asarray(DEMO_SEEDS_V2)):
        raise RuntimeError(
            "v2 示范集没有精确覆盖当前固定种子 "
            "2103000..2103383")
    if np.isin(labels, forbidden_actions_for_generation(
            TEACHER_GENERATION_V2)).any():
        raise RuntimeError("v2 示范集含禁采动作 11(11 恒掩归经理;"
                           "12 系 v2 预防饮实标,允采)")
    masks_array = np.stack(M).astype(bool)
    belts_array = np.asarray(belts, dtype=np.int64)
    visible_targets = np.fromiter(
        (teacher_v2_preventive_trigger(
            row, mask, preventive_threshold)
         for row, mask in zip(X, masks_array)),
        dtype=np.bool_, count=len(labels))
    if not np.array_equal(labels == 12, visible_targets):
        mismatch = np.flatnonzero((labels == 12) != visible_targets)
        raise RuntimeError(
            "v2 整池标签不等于可见 TeacherV2 谓词"
            f"(hp/m12/主动饮位)，首个错位行={int(mismatch[0])}")
    return np.stack(X), labels, groups_array, masks_array, belts_array


def _save_demos_v2(out_dir: pathlib.Path, X, labels, groups, masks,
                   provenance: dict, *,
                   destination: pathlib.Path | None = None) -> pathlib.Path:
    """写自带 provenance 的 v2/3 demos；PASS report 才是 bundle 提交标记。"""
    n = len(labels)
    if masks.shape != (n, 15) or masks.dtype != np.bool_:
        raise RuntimeError(f"v2 masks 形状/dtype 非法: {masks.shape}/{masks.dtype}"
                           "(须 (pairs, 15) bool)")
    if not masks[np.arange(n), labels].all():
        raise RuntimeError("v2 demos 存在提案不在掩码内的样本"
                           "(含 a12 须 m[12]=True;on-manifold 破坏)")
    if masks[:, 11].any():
        raise RuntimeError("v2 demos m[11] 必须恒 False(11 恒掩归经理)")
    required_provenance = {
        "schema_version", "protocol_version", "implementation_sha256",
        "generator_sha256", "manager_npz_sha256", "teacher_generation",
        "preventive_threshold",
    }
    if not isinstance(provenance, dict) \
            or not required_provenance <= set(provenance):
        raise RuntimeError("v2 demos provenance 缺字段:"
                           f"{sorted(required_provenance - set(provenance or {}))}")
    final = destination or (out_dir / "demos.npz")
    tmp = final.with_name(f".{final.stem}.{time.time_ns()}.tmp.npz")
    np.savez_compressed(
        tmp, X=X, Y=labels, episode_id=groups, masks=masks,
        schema_version=np.asarray(_BC_V2_DEMOS_SCHEMA_VERSION),
        protocol_version=np.asarray(provenance["protocol_version"],
                                    dtype=np.int64),
        implementation_sha256=np.asarray(provenance["implementation_sha256"]),
        generator_sha256=np.asarray(provenance["generator_sha256"]),
        manager_npz_sha256=np.asarray(provenance["manager_npz_sha256"]),
        teacher_generation=np.asarray(provenance["teacher_generation"],
                                      dtype=np.int64),
        preventive_threshold=np.asarray(provenance["preventive_threshold"],
                                        dtype=np.float64),
    )
    tmp.replace(final)
    return final


def _balanced_class_weights(labels, n_act: int = 15) -> np.ndarray:
    """方案甲 b(2026-07-19 亲批):类平衡权重 w_c = N/(K·n_c)(标准平衡式)。

    类集 = 实际出现类(n_c > 0);K = 出现类数;N = 样本总数;未出现类记
    0.0(CE weight 向量占位,训练面不消费)。纯整数计数算术,确定性。
    只作用于 v2 主训调用;v1 训练路径零触碰(train_bc 缺省参 None,v1
    调用不传)。
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        raise RuntimeError("类平衡权重:空标签集(方案甲 b 前提破坏)")
    counts = np.bincount(labels, minlength=n_act).astype(np.float64)
    present = counts > 0
    k = int(present.sum())
    weights = np.zeros(n_act, dtype=np.float64)
    weights[present] = float(labels.size) / (k * counts[present])
    return weights


def _wire_a12_teacher_boundary(
        model: PiHead, preventive_threshold: float) -> None:
    """把稀有 a12 判别器写入标准 64x64 策略 MLP 的保留神经元。

    第一层 61..63 分别编码 hp≥0.5、hp<预防阈、本窗尚未主动饮；第二层
    63 形成软 AND。belt 可用性由现场 action mask 的 m12 精确执行。该
    边界与 TeacherV2 的可见谓词同构；其余动作不读保留神经元，a12 不再
    读旧的失校准全类 CE 行。导出仍只有 SB3 原生六张量。
    """
    if not isinstance(model, PiHead):
        raise RuntimeError("a12 校准只接受 PiHead(SB3 64x64 同构模型)")
    if model.net[0].weight.shape[0] != 64 \
            or model.net[2].weight.shape != (64, 64) \
            or model.head.weight.shape != (15, 64):
        raise RuntimeError("a12 校准所需 64x64/15 策略头形制不匹配")
    if model.net[0].weight.shape[1] <= max(
            _A12_CALIBRATION_HP_FEATURE,
            _A12_CALIBRATION_DRINK_LATCH_FEATURE):
        raise RuntimeError("a12 校准缺少 worker hp/主动饮可见位")

    # Lower edge is closed and hp<0.5 is never live-m12, so a small outward
    # pad safely makes exact hp=0.5 a positive.  The upper edge is open:
    # its center must be the exact teacher threshold.  The old ``+0.002``
    # moved that center to 0.652 and placed the real 0.651163 negatives on
    # the positive side of the circuit.  A steeper exact-center transition
    # gives margin without changing the registered predicate.
    slope = 100.0
    lower_edge_pad = 0.002
    conjunction_slope = 8.0
    conjunction_floor = 2.0
    with torch.no_grad():
        w0, b0 = model.net[0].weight, model.net[0].bias
        w1, b1 = model.net[2].weight, model.net[2].bias
        wa, ba = model.head.weight, model.head.bias

        # train_bc(reserve_a12_path=True) 必须从第 0 步就隔离这些参数；
        # 否则此处清零会删掉普通动作已经学到的表示。
        reserved_zero = (
            int(torch.count_nonzero(w0[61:64])) == 0
            and int(torch.count_nonzero(b0[61:64])) == 0
            and int(torch.count_nonzero(w1[:, 61:64])) == 0
            and int(torch.count_nonzero(w1[63])) == 0
            and int(torch.count_nonzero(b1[63])) == 0
            and int(torch.count_nonzero(wa[:, 63])) == 0
        )
        if not reserved_zero:
            raise RuntimeError(
                "a12 校准保留通路曾参与普通 CE；拒绝破坏性事后接线")

        w0[61, _A12_CALIBRATION_HP_FEATURE] = slope
        b0[61] = -slope * (_PREVENTIVE_HP_LOW - lower_edge_pad)
        w0[62, _A12_CALIBRATION_HP_FEATURE] = -slope
        b0[62] = slope * float(preventive_threshold)
        w0[63, _A12_CALIBRATION_DRINK_LATCH_FEATURE] = slope
        # 未饮域最低为 0，已饮域最高为 -1；0.05 margin 令二者饱和分离。
        b0[63] = slope * 0.05

        w1[63].zero_()
        b1[63] = -conjunction_slope * conjunction_floor
        w1[63, 61:64] = conjunction_slope

        wa[12].zero_()
        wa[12, 63] = 8.0
        ba[12].zero_()  # 随后仅由训练 episode 的 margin 分位拟合


def _calibrate_a12_policy(model: PiHead, X, labels, groups, masks,
                          preventive_threshold: float) -> dict:
    """仅在 nested-fit 上校准 a12，返回可入回执的完整拟合记录。

    validation 和 final-heldout 都不读取。可见 hp/每窗饮历史边界固定为
    教师谓词，仅以冻结的 fit 正例 75% 召回为目标拟合 a12 bias，并用
    独立 validation/final 的 0.50 门留出抽样裕量；同时以生产概率门
    约束 soft leakage，并用更严的 fit FPR/high-HP 线筛选；禁止窥看
    任一评估域后调参。
    """
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    masks = np.asarray(masks)
    if X.ndim != 2 or X.shape[1] <= max(
            _A12_CALIBRATION_HP_FEATURE,
            _A12_CALIBRATION_DRINK_LATCH_FEATURE) \
            or labels.shape != (len(X),) or groups.shape != (len(X),) \
            or masks.shape != (len(X), 15) or masks.dtype != np.bool_:
        raise RuntimeError("a12 校准输入形状/dtype 异常")
    fit, validation, validation_episodes = (
        _split_fit_validation_by_episode(groups))
    _, final_heldout, final_heldout_episodes = split_by_episode(groups)
    fit_episode_count = int(np.unique(groups[fit]).size)
    true12 = labels[fit] == 12
    negative = ~true12
    legal12 = masks[fit, 12]
    if int(true12.sum()) < 2 or not bool(legal12[true12].all()):
        raise RuntimeError("a12 校准 fit 切分正例不足或正例被真实 mask 禁止")
    legal_negative = negative & legal12
    if int(legal_negative.sum()) < 3 * int(true12.sum()):
        raise RuntimeError("a12 校准 fit 缺少至少 3:1 的真实 m12 hard negatives")

    target_count = int(math.ceil(
        _A12_CALIBRATION_TRAIN_RECALL_TARGET * int(true12.sum())))
    minimum_share = target_count / len(fit)
    if minimum_share > _A12_PREDICTED_SHARE_MAX:
        raise RuntimeError(
            "BC-v2 a12 fit 门算术不可达:"
            f"ceil({int(true12.sum())}×"
            f"{_A12_CALIBRATION_TRAIN_RECALL_TARGET})/{len(fit)}"
            f"={minimum_share:.8f} > predicted_share_max "
            f"{_A12_PREDICTED_SHARE_MAX};拒绝靠降安全门或调 bias 掩盖")
    _wire_a12_teacher_boundary(model, preventive_threshold)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X[fit]))
        mask_t = torch.from_numpy(masks[fit])
        other = torch.where(
            mask_t, logits, torch.full_like(logits, -1e8))
        other[:, 12] = -1e8
        margin = (logits[:, 12] - other.max(1).values).cpu().numpy()

    # 最保守地跨过冻结 fit recall 目标对应的 margin；不以评估域定 bias。
    positive_margin = np.sort(margin[true12])[::-1]
    cut = float(positive_margin[target_count - 1])
    bias = float(-cut + 1e-6)
    with torch.no_grad():
        model.head.bias[12] = bias
        # 参数张量是 float32；回执必须绑定“真正部署的值”，不能记录赋值前
        # 的 Python float。大幅 bias 的量化误差可到 1e-6，足以让生产者
        # 自己生成的合法件在 PPO 入口被拒。
        bias = float(model.head.bias[12].item())

    def deployed_fit_metrics() -> dict:
        # 必须走 PPO 消费端同一份六张量前向+mask+argmax。旧的
        # ``margin+bias >= 0`` 捷径在 float32 加法舍入或 logit 平局时会把
        # action 9/12 的 tie 算成 12，而真实 argmax 选择较小索引 9。
        behavior = bc_aux_behavior_metrics(
            export_sb3_sd(model), X[fit], labels[fit], groups[fit],
            masks[fit], heldout_only=False)
        return {
            "bias_12": float(model.head.bias[12].item()),
            "tp": int(behavior["tp"]),
            "fp": int(behavior["fp"]),
            "precision_12": float(behavior["precision_12"]),
            "recall_12": float(behavior["recall_12"]),
            "fpr_12": float(behavior["fpr_12"]),
            "predicted_share_12": float(
                behavior["predicted_share_12"]),
            "high_hp_false_drink_rate": float(
                behavior["high_hp_false_drink_rate"]),
            "legal_negative_probability_12_mean": float(
                behavior["legal_negative_probability_12_mean"]),
            "legal_negative_probability_12_max": float(
                behavior["legal_negative_probability_12_max"]),
            "a13_spillover": float(behavior["a13_spillover"]),
            "_high_hp_non_a12": int(behavior["high_hp_non_a12"]),
        }

    selected = deployed_fit_metrics()
    # 初始解析 bias 理论上跨过第 target_count 个 margin；float32 最终前向
    # 仍可能因舍入形成 tie。只在 fit 上逐 ULP 上推到真实 argmax 达目标，
    # 且随后所有 FPR/share 门都按同一最终六张量重算。
    for _ in range(64):
        if (selected["recall_12"]
                >= _A12_CALIBRATION_TRAIN_RECALL_TARGET):
            break
        with torch.no_grad():
            current = model.head.bias[12]
            current.copy_(torch.nextafter(
                current, torch.full_like(current, float("inf"))))
        selected = deployed_fit_metrics()
    else:
        raise RuntimeError(
            "BC-v2 a12 fit 校准在 64 个 float32 ULP 内仍无法达到目标召回")

    bias = selected["bias_12"]
    tp = selected["tp"]
    fp = selected["fp"]
    precision = selected["precision_12"]
    recall = selected["recall_12"]
    fpr = selected["fpr_12"]
    share = selected["predicted_share_12"]
    high_hp_rate = selected["high_hp_false_drink_rate"]
    legal_negative_probability_mean = selected[
        "legal_negative_probability_12_mean"]
    legal_negative_probability_max = selected[
        "legal_negative_probability_12_max"]
    a13_spillover = selected["a13_spillover"]
    high_hp_count = selected.pop("_high_hp_non_a12")
    if not (
            recall >= _A12_CALIBRATION_TRAIN_RECALL_TARGET
            and precision >= max(0.10, _A12_PRECISION_MIN)
            and fpr <= _A12_FPR_MAX * 0.5
            and _A12_PREDICTED_SHARE_MIN <= share
            <= _A12_PREDICTED_SHARE_MAX
            and high_hp_count > 0
            and high_hp_rate <= _A12_HIGH_HP_FALSE_DRINK_MAX * 0.5
            and legal_negative_probability_mean
            <= _A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX
            and legal_negative_probability_max
            <= _A12_LEGAL_NEGATIVE_PROBABILITY_MAX
            and a13_spillover <= _A13_SPILLOVER_MAX):
        compact = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in selected.items() if k != "bias_12"
        }
        raise RuntimeError(
            "BC-v2 a12 fit 校准不可达；拒绝让失校准分类器进入 validation:"
            f"{compact}")

    return {
        "schema_version": _A12_CALIBRATION_SCHEMA_VERSION,
        "fit_scope": "nested-fit-episodes-only",
        "fit_pairs": int(len(fit)),
        "fit_episodes": fit_episode_count,
        "validation_pairs_excluded": int(len(validation)),
        "validation_episodes_excluded": int(len(validation_episodes)),
        "final_heldout_pairs_excluded": int(len(final_heldout)),
        "final_heldout_episodes_excluded": int(
            len(final_heldout_episodes)),
        "hp_low": _PREVENTIVE_HP_LOW,
        "hp_high": float(preventive_threshold),
        "hp_feature": _A12_CALIBRATION_HP_FEATURE,
        "drink_latch_feature": _A12_CALIBRATION_DRINK_LATCH_FEATURE,
        "predicate": _A12_CALIBRATION_PREDICATE,
        "bias_12": selected["bias_12"],
        "target_recall_12": _A12_CALIBRATION_TRAIN_RECALL_TARGET,
        "fit_metrics": {
            key: (int(value) if key in {"tp", "fp"} else float(value))
            for key, value in selected.items()
            if key != "bias_12"
        },
    }


def _n12_readings(labels, groups, belts) -> dict:
    """n₁₂ 闸读数：全部可见预防带内 a12 执行态（overridden 拍剔除）。"""
    n = int(len(labels))
    a12 = labels == 12
    a13 = labels == 13
    n12 = int(a12.sum())
    by_episode = {str(int(seed)): int((a12 & (groups == seed)).sum())
                  for seed in np.unique(groups[a12])}   # 零计局省略,余局恒 0
    return {
        "n12": n12,
        "n12_by_episode": by_episode,
        "class_share_12": round(n12 / n, 6) if n else 0.0,
        "class_share_13": round(int(a13.sum()) / n, 6) if n else 0.0,
        "belt_economy": {
            # 裁量注记:图纸"腰带经济读数"未钉字段,注册为三项——a12 实标态
            # 腰带均值 / 全集腰带均值 / a13(拾取补给)对数;零覆盖记 0.0 不消失。
            "belt_mean_at_a12": (round(float(belts[a12].mean()), 4)
                                 if n12 > 0 else 0.0),
            "belt_mean_overall": round(float(belts.mean()), 4) if n else 0.0,
            "a13_pairs": int(a13.sum()),
        },
    }


def _a12_behavior_from_model(model, X, labels, groups, masks) -> dict:
    """以真实 v2 masks 在固定 held-out 整局上评估 a12 行为面。"""
    model.eval()
    return bc_aux_behavior_metrics(
        export_sb3_sd(model), X, labels, groups, masks,
        heldout_only=True)


def _a12_behavior_from_indices(
        model, X, labels, groups, masks, indices) -> dict:
    """用显式 selection 行集评估 a12，绝不隐式切到 final-heldout。"""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("a12 validation indices 必须是一维非空索引")
    model.eval()
    return bc_aux_behavior_metrics(
        export_sb3_sd(model),
        np.asarray(X)[indices], np.asarray(labels)[indices],
        np.asarray(groups)[indices], np.asarray(masks)[indices],
        heldout_only=False)


def _bc_v2_a12_gate_passes(metrics: dict) -> tuple[bool, dict]:
    """BC-v2 专用 a12 门 = 通用行为安全门 + 注册的 0.5 recall。"""
    gate = bc_aux_behavior_gate(
        metrics, require_teacher_recall=False)
    recall = metrics.get("recall_12") if isinstance(metrics, dict) else None
    passes = (
        gate["verdict"] == "PASS"
        and isinstance(recall, (int, float))
        and not isinstance(recall, bool)
        and math.isfinite(float(recall))
        and float(recall) >= _RECALL12_GATE_MIN)
    return passes, gate


def _recall12_from_model(model, X, labels, groups, masks) -> tuple[float, int]:
    """recall 门(E2/D7 钉死):分母 = 现切分下 held-out 之实标 a12 态
    （全部由可见 hp/belt 谓词产生的 on-manifold 执行态）；
    度量 = held-out argmax 命中;fail-closed 承 v1 类召回先例
    (train_bc 零覆盖记 0.0 不消失)。返回 (recall_12, 分母计数)。"""
    _, ho, _ = split_by_episode(groups)
    pred = _masked_bc_predictions(model, X[ho], masks[ho])
    true12 = labels[ho] == 12
    denominator = int(true12.sum())
    recall = (float((pred[true12] == 12).mean())
              if denominator else 0.0)
    return round(recall, 4), denominator


def artifact_provenance_v2(
        preventive_threshold: float,
        manager_npz: str | pathlib.Path = NPZ) -> dict:
    manager_path = pathlib.Path(manager_npz)
    try:
        manager_payload = manager_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"BC-v2 经理 npz 缺失/不可读:{manager_path}") from exc
    return {
        "schema_version": _BC_V2_REPORT_SCHEMA_VERSION,
        "teacher_generation": TEACHER_GENERATION_V2,
        "preventive_threshold": preventive_threshold,
        "protocol_version": PROTOCOL_VERSION,
        "implementation_sha256": _implementation_bundle_sha256(),
        "generator_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()).hexdigest(),
        "manager_npz_sha256": hashlib.sha256(manager_payload).hexdigest(),
    }


def begin_output_attempt_v2(provenance: dict | None = None):
    """v2 目录去旧防误吃(v1 同型;目录独立,规避 _previous 归档互斥)。"""
    provenance = (
        artifact_provenance_v2(_PREVENTIVE_THRESHOLD_MAIN, NPZ)
        if provenance is None else provenance)
    OUT_V2.mkdir(parents=True, exist_ok=True)
    _assert_final_holdout_unused(
        OUT_V2, TEACHER_GENERATION_V2, DEMO_SEEDS_V2)
    _reject_same_generator_terminal(
        OUT_V2, _BC_V2_REPORT_NAME, provenance)
    old = [OUT_V2 / name
           for name in ("policy_sd.pt", _BC_V2_REPORT_NAME, "demos.npz")
           if (OUT_V2 / name).exists()]
    if old:
        archive = OUT_V2 / "_previous" / str(time.time_ns())
        archive.mkdir(parents=True)
        for path in old:
            path.replace(archive / path.name)
    # 上次进程若死在 report 提交前，pending 数据绝不能在下一次尝试中
    # 留作貌似可用输入；canonical 三件套已经在上面统一归档。
    for residue in OUT_V2.glob(".*.tmp.npz"):
        residue.unlink(missing_ok=True)
    (OUT_V2 / "demos.pending.npz").unlink(missing_ok=True)
    (OUT_V2 / _BC_V2_REPORT_NAME).write_text(json.dumps(
        {"data_gate": "RUNNING", "teacher_generation": TEACHER_GENERATION_V2}))


def write_report_v2(record):
    tmp = OUT_V2 / "bc_report_v2.tmp.json"
    tmp.write_text(json.dumps(record, ensure_ascii=False))
    tmp.replace(OUT_V2 / _BC_V2_REPORT_NAME)


def _validate_bc_v2_report(p: pathlib.Path,
                           expected_implementation_sha256: str | None = None,
                           *, policy_payload: bytes | None = None,
                           report_payload: bytes | None = None,
                           demos_payload: bytes | None = None,
                           expected_manager_sha256: str | None = None) -> dict:
    """BC-v2 专用验证器(E2 schema 隔离,形制承 v1 验证器)。

    键集合精确等断言 + 独立 schema 标识:对 v1 件必炸(键集合不等);
    v1 验证器(train_ppo._validate_bc_report)对 v2 件亦必炸(键集合
    精确等 + schema_version 非 int 双重不相容)——v1/v2 验证器互斥。
    demos 实测字节断言(rev4 十二附二④ 补铸):demos.npz 字节 sha256 ≡
    回执 demos_sha256,镜像 policy 侧真字节断言形制。
    """
    from eval_contract import EvalContractError, strict_json_loads

    def _fail(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def _plain_int(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    def _plain_num(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    report = p.with_name(_BC_V2_REPORT_NAME)
    try:
        frozen_report = (report.read_bytes() if report_payload is None
                         else report_payload)
        rec = strict_json_loads(frozen_report)
    except (OSError, EvalContractError) as exc:
        raise ValueError(f"BC-v2 回执缺失/不可读: {report}") from exc
    _fail(isinstance(rec, dict), f"BC-v2 回执必须是 JSON 对象: {report}")
    _fail(set(rec) == set(_BC_V2_PASS_KEYS),
          f"BC-v2 回执字段/schema 不匹配(v1/v2 件互斥): {report}")
    _fail(rec["schema_version"] == _BC_V2_REPORT_SCHEMA_VERSION,
          f"BC-v2 回执 schema 标识不符: {rec['schema_version']!r}")
    _fail(rec["teacher_generation"] == TEACHER_GENERATION_V2,
          f"BC-v2 回执 teacher_generation 必须为 2: {rec['teacher_generation']!r}")
    _fail(rec["preventive_threshold"] in _REGISTERED_PREVENTIVE_THRESHOLDS,
          f"BC-v2 回执预防阈未注册: {rec['preventive_threshold']!r}")
    _fail(rec["data_gate"] == "PASS",
          f"拒绝采信未过 data_gate 闸的 BC-v2 件: {rec['data_gate']!r}")
    _fail(_plain_num(rec["held_out_top1"])
          and math.isfinite(float(rec["held_out_top1"]))
          and 0.95 <= float(rec["held_out_top1"]) <= 1.0,
          f"BC-v2 held-out top1 门不满足: {rec['held_out_top1']!r}")
    class_recalls = rec["class_recalls"]
    _fail(isinstance(class_recalls, dict) and class_recalls,
          "BC-v2 class_recalls 必须是非空对象")
    for raw_action, raw_recall in class_recalls.items():
        try:
            action = int(raw_action)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"BC-v2 class_recalls 键必须是动作编号: {raw_action!r}"
            ) from exc
        _fail(str(action) == str(raw_action) and 0 <= action < 15,
              f"BC-v2 class_recalls 键非法: {raw_action!r}")
        _fail(_plain_num(raw_recall)
              and math.isfinite(float(raw_recall))
              and 0.0 <= float(raw_recall) <= 1.0,
              f"BC-v2 class_recalls[{raw_action!r}] 非法: {raw_recall!r}")
        if action != 12:
            _fail(float(raw_recall) >= 0.85,
                  "BC-v2 非 a12 逐类召回门不满足:"
                  f" action={action}, recall={raw_recall!r}")
    _fail(_plain_int(rec["n12"]) and rec["n12"] >= _N12_GATE_MIN,
          f"BC-v2 n₁₂ 闸不满足(≥{_N12_GATE_MIN}): {rec['n12']!r}")
    _fail(_plain_num(rec["recall_12"])
          and _RECALL12_GATE_MIN <= float(rec["recall_12"]) <= 1.0,
          f"BC-v2 recall 门不满足(≥{_RECALL12_GATE_MIN}): {rec['recall_12']!r}")
    behavior = rec["a12_behavior"]
    _fail(isinstance(behavior, dict)
          and set(behavior) == set(_BC_AUX_BEHAVIOR_METRIC_KEYS),
          f"BC-v2 a12_behavior 字段/schema 不匹配: {behavior!r}")
    _fail(behavior["pairs"] == rec["held_out_pairs"]
          and behavior["true_a12"] + behavior["all_non_a12"]
          == behavior["pairs"]
          and behavior["non_a12"] <= behavior["all_non_a12"]
          and behavior["tp"] + behavior["fn"] == behavior["true_a12"]
          and behavior["fp"] + behavior["tn"] == behavior["non_a12"]
          and behavior["tp"] + behavior["fp"]
          == behavior["predicted_a12"],
          "BC-v2 a12_behavior 计数/分母不闭合")
    recomputed_gate = bc_aux_behavior_gate(
        behavior, require_teacher_recall=False)
    _fail(rec["a12_behavior_gate"] == recomputed_gate,
          "BC-v2 a12_behavior_gate 与行为读数/注册阈值不一致")
    _fail(recomputed_gate["verdict"] == "PASS",
          "BC-v2 a12 行为硬门未过:"
          f"{recomputed_gate['reasons']}")
    _fail(float(behavior["recall_12"]) >= _RECALL12_GATE_MIN,
          f"BC-v2 a12 专用 recall 门不满足(≥{_RECALL12_GATE_MIN}):"
          f" {behavior['recall_12']!r}")
    _fail(isinstance(rec["held_out_episodes"], list)
          and _plain_int(rec["collection_episodes"])
          and rec["collection_episodes"]
          >= max(1, len(rec["held_out_episodes"])),
          f"BC-v2 回执 collection_episodes 非法:"
          f"{rec['collection_episodes']!r}")
    calibration = rec["a12_calibration"]
    calibration_keys = {
        "schema_version", "fit_scope", "fit_pairs", "fit_episodes",
        "validation_pairs_excluded", "validation_episodes_excluded",
        "final_heldout_pairs_excluded",
        "final_heldout_episodes_excluded", "hp_low", "hp_high",
        "hp_feature", "drink_latch_feature", "predicate", "bias_12",
        "target_recall_12", "fit_metrics",
    }
    _fail(isinstance(calibration, dict)
          and set(calibration) == calibration_keys,
          "BC-v2 a12_calibration 字段/schema 不匹配")
    _fail(calibration["schema_version"] == _A12_CALIBRATION_SCHEMA_VERSION
          and calibration["fit_scope"] == "nested-fit-episodes-only",
          "BC-v2 a12_calibration 身份/拟合域不符")
    _fail(_plain_int(calibration["fit_pairs"])
          and _plain_int(calibration["validation_pairs_excluded"])
          and _plain_int(calibration["final_heldout_pairs_excluded"])
          and calibration["fit_pairs"] > 0
          and calibration["validation_pairs_excluded"] > 0
          and calibration["final_heldout_pairs_excluded"] > 0
          and calibration["fit_pairs"]
          + calibration["validation_pairs_excluded"]
          + calibration["final_heldout_pairs_excluded"]
          == rec["pairs"],
          "BC-v2 a12_calibration 三域 pairs 切分不闭合")
    _fail(calibration["final_heldout_pairs_excluded"]
          == rec["held_out_pairs"],
          "BC-v2 a12_calibration final-heldout pairs 与回执不一致")
    _fail(_plain_int(calibration["fit_episodes"])
          and _plain_int(calibration["validation_episodes_excluded"])
          and _plain_int(calibration["final_heldout_episodes_excluded"])
          and calibration["fit_episodes"] > 0
          and calibration["validation_episodes_excluded"] > 0
          and calibration["final_heldout_episodes_excluded"] > 0
          and calibration["fit_episodes"]
          + calibration["validation_episodes_excluded"]
          + calibration["final_heldout_episodes_excluded"]
          == rec["collection_episodes"],
          "BC-v2 a12_calibration 三域 episode 切分不闭合")
    _fail(calibration["final_heldout_episodes_excluded"]
          == len(rec["held_out_episodes"]),
          "BC-v2 a12_calibration final-heldout episodes 与回执不一致")
    _fail(calibration["hp_low"] == _PREVENTIVE_HP_LOW
          and calibration["hp_high"] == rec["preventive_threshold"]
          and calibration["hp_feature"] == _A12_CALIBRATION_HP_FEATURE
          and calibration["drink_latch_feature"]
          == _A12_CALIBRATION_DRINK_LATCH_FEATURE
          and calibration["predicate"] == _A12_CALIBRATION_PREDICATE
          and calibration["target_recall_12"]
          == _A12_CALIBRATION_TRAIN_RECALL_TARGET
          and _plain_num(calibration["bias_12"])
          and math.isfinite(float(calibration["bias_12"])),
          "BC-v2 a12_calibration 参数未注册")
    fit_metrics = calibration["fit_metrics"]
    metric_keys = {
        "tp", "fp", "precision_12", "recall_12", "fpr_12",
        "predicted_share_12", "high_hp_false_drink_rate",
        "legal_negative_probability_12_mean",
        "legal_negative_probability_12_max",
        "a13_spillover",
    }
    _fail(isinstance(fit_metrics, dict)
          and set(fit_metrics) == metric_keys
          and _plain_int(fit_metrics["tp"])
          and _plain_int(fit_metrics["fp"])
          and 0 <= fit_metrics["tp"] <= calibration["fit_pairs"]
          and 0 <= fit_metrics["fp"] <= calibration["fit_pairs"]
          and all(
              _plain_num(fit_metrics[key])
              and math.isfinite(float(fit_metrics[key]))
              and 0.0 <= float(fit_metrics[key]) <= 1.0
              for key in metric_keys - {"tp", "fp"})
          and math.isclose(
              float(fit_metrics["precision_12"]),
              fit_metrics["tp"] / max(
                  1, fit_metrics["tp"] + fit_metrics["fp"]),
              rel_tol=0.0, abs_tol=1e-15)
          and math.isclose(
              float(fit_metrics["predicted_share_12"]),
              (fit_metrics["tp"] + fit_metrics["fp"])
              / calibration["fit_pairs"],
              rel_tol=0.0, abs_tol=1e-15)
          and fit_metrics["recall_12"]
          >= _A12_CALIBRATION_TRAIN_RECALL_TARGET
          and fit_metrics["precision_12"] >= max(0.10, _A12_PRECISION_MIN)
          and fit_metrics["fpr_12"] <= _A12_FPR_MAX * 0.5
          and _A12_PREDICTED_SHARE_MIN
          <= fit_metrics["predicted_share_12"]
          <= _A12_PREDICTED_SHARE_MAX
          and fit_metrics["high_hp_false_drink_rate"]
          <= _A12_HIGH_HP_FALSE_DRINK_MAX * 0.5
          and fit_metrics["legal_negative_probability_12_mean"]
          <= _A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX
          and fit_metrics["legal_negative_probability_12_max"]
          <= _A12_LEGAL_NEGATIVE_PROBABILITY_MAX
          and fit_metrics["a13_spillover"] <= _A13_SPILLOVER_MAX,
          "BC-v2 a12_calibration fit 安全读数不满足")
    # 回执旧 recall 字段与新行为面必须同源，禁一个 PASS 一个 FAIL。
    _fail(math.isclose(float(rec["recall_12"]),
                       round(float(behavior["recall_12"]), 4),
                       rel_tol=0, abs_tol=1e-12)
          and rec["recall_12_denominator"] == behavior["true_a12"],
          "BC-v2 recall 旧字段与 a12_behavior 不一致")
    if "12" in class_recalls:
        _fail(math.isclose(
                  float(class_recalls["12"]),
                  float(behavior["recall_12"]),
                  rel_tol=0.0, abs_tol=1e-15),
              "BC-v2 class_recalls[12] 与专用行为 recall 不一致")
    # 方案甲回执新字段断言(2026-07-19 亲批)
    _fail(isinstance(rec["held_out_episodes"], list)
          and _plain_int(rec["collection_episodes"])
          and rec["collection_episodes"] >= max(1, len(rec["held_out_episodes"])),
          f"BC-v2 回执 collection_episodes 非法: {rec['collection_episodes']!r}")
    class_weights = rec["class_weights"]
    _fail(isinstance(class_weights, dict) and len(class_weights) > 0,
          f"BC-v2 回执 class_weights 必须是非空对象: {report}")
    for raw_class, raw_weight in class_weights.items():
        try:
            class_id = int(raw_class)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"BC-v2 回执 class_weights 键必须是动作编号: {raw_class!r}"
            ) from exc
        _fail(str(class_id) == str(raw_class) and 0 <= class_id < 15,
              f"BC-v2 回执 class_weights 键非法: {raw_class!r}")
        _fail(_plain_num(raw_weight) and raw_weight > 0,
              f"BC-v2 回执 class_weights[{raw_class!r}] 非法: {raw_weight!r}")
    _fail(rec["protocol_version"] == PROTOCOL_VERSION,
          f"BC-v2 回执协议过期: {rec['protocol_version']!r}")
    expected_impl = (expected_implementation_sha256
                     if expected_implementation_sha256 is not None
                     else _implementation_bundle_sha256())
    _fail(rec["implementation_sha256"] == expected_impl,
          "BC-v2 回执的实现/引擎/游戏内容身份与当前运行时不一致")
    generator_sha = hashlib.sha256(
        pathlib.Path(__file__).read_bytes()).hexdigest()
    _fail(rec["generator_sha256"] == generator_sha,
          "BC-v2 回执生成器已漂移: train/bc_worker.py")
    _validate_bc_final_holdout_marker(
        p.parent, TEACHER_GENERATION_V2, DEMO_SEEDS_V2, rec)
    manager_sha = rec["manager_npz_sha256"]
    _fail(isinstance(manager_sha, str) and len(manager_sha) == 64
          and all(ch in "0123456789abcdef" for ch in manager_sha),
          f"BC-v2 回执 manager_npz_sha256 非法: {manager_sha!r}")
    if expected_manager_sha256 is not None:
        _fail(isinstance(expected_manager_sha256, str)
              and len(expected_manager_sha256) == 64
              and all(ch in "0123456789abcdef"
                      for ch in expected_manager_sha256),
              "BC-v2 预期 manager SHA 非法")
        _fail(manager_sha == expected_manager_sha256,
              "BC-v2 回执经理身份与训练经理不一致:"
              f"{manager_sha} != {expected_manager_sha256}")
    expected_sha = rec["policy_sha256"]
    _fail(isinstance(expected_sha, str) and len(expected_sha) == 64,
          f"BC-v2 回执缺少 policy_sha256 绑定: {report}")
    try:
        frozen_policy = (p.read_bytes() if policy_payload is None
                         else policy_payload)
    except OSError as exc:
        raise ValueError(f"BC-v2 权重缺失/不可读: {p}") from exc
    _fail(hashlib.sha256(frozen_policy).hexdigest() == expected_sha,
          "BC-v2 权重与回执 SHA 不匹配")
    _fail(isinstance(rec["demos_sha256"], str) and len(rec["demos_sha256"]) == 64,
          f"BC-v2 回执缺少 demos_sha256 绑定: {report}")
    demos = p.with_name("demos.npz")
    try:
        frozen_demos = (demos.read_bytes() if demos_payload is None
                        else demos_payload)
    except OSError as exc:
        raise ValueError(f"BC-v2 demos 缺失/不可读: {demos}") from exc
    _fail(hashlib.sha256(frozen_demos).hexdigest() == rec["demos_sha256"],
          "BC-v2 demos 与回执 SHA 不匹配")
    return rec


def main_v2(preventive_threshold: float = _PREVENTIVE_THRESHOLD_MAIN,
            manager_npz: str | pathlib.Path = NPZ):
    if preventive_threshold not in _REGISTERED_PREVENTIVE_THRESHOLDS:
        raise ValueError(f"预防阈 {preventive_threshold!r} 未注册"
                         "(rev13 仅 0.65 主案；0.70 无独立 fresh pool)")
    manager_path = pathlib.Path(manager_npz)
    provenance = artifact_provenance_v2(
        preventive_threshold, manager_path)
    begin_output_attempt_v2(provenance)
    # collect_v2 validates and prints whole-pool labels/masks.  The stable
    # O_EXCL registry must therefore be committed before the first reset, not
    # merely before final scoring.
    pool_marker = _mark_final_holdout_started(
        OUT_V2, TEACHER_GENERATION_V2, DEMO_SEEDS_V2, provenance)
    pool_evidence = {
        "final_pool_sha256": pool_marker["pool_sha256"],
        "final_holdout_marker_sha256": pool_marker["marker_sha256"],
    }
    write_report_v2({
        "data_gate": "FAIL",
        "failure_stage": "pool_collection_started",
        "final_heldout_consumed": True,
        **pool_evidence,
        **provenance,
    })
    X, labels, groups, masks, belts = collect_v2(
        preventive_threshold, manager_path,
        manager_sha256=provenance["manager_npz_sha256"])
    post_drink_coverage = _bc_v2_post_drink_coverage(
        X, labels, groups, masks, scopes=("fit", "validation"))
    print(
        "v2 可见后饮 hard-negative 预选域覆盖 "
        f"{post_drink_coverage}",
        flush=True,
    )
    pending_demos = OUT_V2 / "demos.pending.npz"
    _save_demos_v2(
        OUT_V2, X, labels, groups, masks, provenance,
        destination=pending_demos)
    tr, ho, holdout_episodes = split_by_episode(groups)
    fit, _, _ = _split_fit_validation_by_episode(groups)
    # 方案甲 b(2026-07-19 亲批):v2 主训即类平衡加权 CE——权重自训练切分
    # 标签计数确定性导出(w_c = N/(K·n_c),类集 = 实际出现类);v1 主训
    # 调用 train_bc(X, Y, groups) 不传权,零触碰。类别权重只读 fit，避免
    # validation/final-held-out 标签反向塑造候选。
    class_weights_v2 = _balanced_class_weights(labels[fit])
    # train_bc 只把 fit/validation 送入梯度或候选读数；整池采集本身已由
    # pre-collection marker 永久登记。
    model, top1, recalls = train_bc(
        X, labels, groups, class_weights=class_weights_v2,
        masks=masks, reserve_a12_path=True)
    retrained = False
    if not _bc_v2_quality_gate_passes(top1, recalls):
        # v1 同款唯一重试,触发条件限 v1 面质量闸(top1/≥300 类召回)——
        # n₁₂/recall_12 永不触发类加权重训:类加权系 N12 已除名预案,
        # 须另行亲批(PREREG-内容案 D2-5/D5 P-N12/D7)。
        print(f"v2 首训 validation 未达标(top1 {top1:.3f} 召回 {recalls})"
              "→ 类加权重训(唯一重试;n₁₂/recall_12 永不触发此重试)",
              flush=True)
        counts = np.bincount(labels[fit], minlength=15).astype(np.float64)
        weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
        weights = weights / weights[weights > 0].mean()
        model, top1, recalls = train_bc(
            X, labels, groups, class_weights=weights,
            epochs=_BC_WEIGHTED_RETRY_EPOCHS, masks=masks,
            reserve_a12_path=True)
        retrained = True
    if not _bc_v2_quality_gate_passes(top1, recalls):
        pending_demos.unlink(missing_ok=True)
        write_report_v2({
            "data_gate": "FAIL",
            "failure_stage": "candidate_selection",
            "validation_top1": float(top1),
            "validation_class_recalls": recalls,
            "class_weighted_retry": retrained,
            "final_heldout_consumed": True,
            **pool_evidence,
            **provenance,
        })
        raise RuntimeError(
            f"BC-v2 candidate validation FAIL(top1={top1:.3f}, "
            f"recalls={recalls});final held-out 未进入候选指标，"
            "但整池已一次性消费，拒绝发布/同池重试")
    # 类平衡 CE 对 0.04% 稀有类只改善 recall、没有误报约束。最终导出前
    # 用训练 episode + 真实 masks 校准网络内 a12 通路，再对最终六张量
    # 重新计算 held-out top1/recalls，禁止沿用校准前读数。
    try:
        a12_calibration = _calibrate_a12_policy(
            model, X, labels, groups, masks, preventive_threshold)
    except Exception as exc:
        pending_demos.unlink(missing_ok=True)
        write_report_v2({
            "data_gate": "FAIL",
            "failure_stage": "training_calibration",
            "validation_top1": float(top1),
            "validation_class_recalls": recalls,
            "class_weighted_retry": retrained,
            "error": f"{type(exc).__name__}: {exc}",
            "final_heldout_consumed": True,
            **pool_evidence,
            **provenance,
        })
        raise
    # 校准会直接改最终导出的 action head；它不是只读后处理。最终候选须
    # 用同一 nested validation 和采集时真实 masks 再过一次选择门，未过
    # 时 final-heldout 仍保持零读取。
    outer_train, _, _ = split_by_episode(groups)
    _, validation, _ = _split_fit_validation_by_episode(groups)
    validation_top1, validation_recalls = _score_bc_model_indices(
        model, X, labels, validation,
        eligibility_indices=outer_train, masks=masks)
    validation_a12 = _a12_behavior_from_indices(
        model, X, labels, groups, masks, validation)
    validation_a12_passes, validation_a12_gate = (
        _bc_v2_a12_gate_passes(validation_a12))
    if (not _bc_v2_quality_gate_passes(
            validation_top1, validation_recalls)
            or not validation_a12_passes):
        pending_demos.unlink(missing_ok=True)
        write_report_v2({
            "data_gate": "FAIL",
            "failure_stage": "post_calibration_selection",
            "validation_top1": float(validation_top1),
            "validation_class_recalls": validation_recalls,
            "validation_a12_behavior": validation_a12,
            "validation_a12_behavior_gate": validation_a12_gate,
            "class_weighted_retry": retrained,
            "final_heldout_consumed": True,
            **pool_evidence,
            **provenance,
        })
        raise RuntimeError(
            "BC-v2 post-calibration validation FAIL"
            f"(top1={validation_top1:.3f}, recalls={validation_recalls});"
            "final held-out 未进入候选指标，但整池已一次性消费，"
            "拒绝发布/同池重试")
    # 从此点开始 final split 才进入候选模型的评分/行为统计路径。
    final_post_drink_coverage = _bc_v2_post_drink_coverage(
        X, labels, groups, masks, scopes=("final",))
    print(
        "v2 可见后饮 hard-negative final 域覆盖 "
        f"{final_post_drink_coverage}",
        flush=True,
    )
    top1, recalls = _score_bc_model(
        model, X, labels, groups, masks=masks)
    a12_behavior = _a12_behavior_from_model(
        model, X, labels, groups, masks)
    a12_behavior_gate = bc_aux_behavior_gate(
        a12_behavior, require_teacher_recall=False)
    a12_behavior_passes, _ = _bc_v2_a12_gate_passes(a12_behavior)
    recall_12 = round(float(a12_behavior["recall_12"]), 4)
    recall_12_denominator = int(a12_behavior["true_a12"])
    readings = _n12_readings(labels, groups, belts)
    ok = (_bc_v2_quality_gate_passes(top1, recalls)
          and readings["n12"] >= _N12_GATE_MIN
          and a12_behavior_passes)
    report = {
        "pairs": len(labels), "held_out_top1": float(top1),
        "held_out_pairs": int(len(ho)),
        "held_out_episodes": [int(x) for x in sorted(holdout_episodes)],
        "class_recalls": recalls, "class_weighted_retry": retrained,
        "recall_12": recall_12,
        "recall_12_denominator": recall_12_denominator,
        "recall_12_gate_min": _RECALL12_GATE_MIN,
        "a12_behavior": a12_behavior,
        "a12_behavior_gate": a12_behavior_gate,
        "a12_calibration": a12_calibration,
        "n12_gate_min": _N12_GATE_MIN,
        # 方案甲回执新字段(2026-07-19 亲批):实测采集局数(×3 纪律已由
        # collect_v2 断言钉死)+ 主训类平衡权重逐类摘要(仅实际出现类)。
        "collection_episodes": int(np.unique(groups).size),
        "class_weights": {str(int(c)): round(float(class_weights_v2[c]), 6)
                          for c in np.flatnonzero(class_weights_v2 > 0)},
        **pool_evidence,
        **readings,
        "data_gate": "PASS" if ok else "FAIL", **provenance}
    if not ok:
        pending_demos.unlink(missing_ok=True)
        write_report_v2(report)
        raise RuntimeError(
            f"BC-v2 数据闸 FAIL(top1={top1:.3f}, n12={readings['n12']}, "
            f"recall_12={recall_12},"
            f" behavior={a12_behavior_gate['reasons']});"
            "拒绝覆写 policy_sd.pt。旧 0.70 OC 与主案共用 final pool，"
            "rev13 已禁用；若 n₁₂ 不足须先注册独立 fresh pool，"
            "不得同池重采")
    policy_tmp = OUT_V2 / "policy_sd.tmp.pt"
    if artifact_provenance_v2(
            preventive_threshold, manager_path) != provenance:
        raise RuntimeError("BC-v2 运行期间实现/引擎/内容/经理发生漂移")
    torch.save(export_sb3_sd(model), policy_tmp)
    policy_tmp.replace(OUT_V2 / "policy_sd.pt")
    report["policy_sha256"] = hashlib.sha256(
        (OUT_V2 / "policy_sd.pt").read_bytes()).hexdigest()
    report["demos_sha256"] = hashlib.sha256(
        pending_demos.read_bytes()).hexdigest()
    # report 是 bundle 的最后提交标记：policy/demos 即使先发布后进程崩溃，
    # sibling 仍为 RUNNING，训练消费者会 fail-closed。只有三件字节与回执
    # 哈希全部就位后，PASS report 才原子替换。
    pending_demos.replace(OUT_V2 / "demos.npz")
    write_report_v2(report)
    print(f"v2 held-out top-1 {top1:.3f} n12 {readings['n12']} "
          f"recall_12 {recall_12} retry={retrained} → 数据闸 PASS;"
          f"已存 {OUT_V2}/policy_sd.pt", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BC worker 采集/训练(默认 v1 面原样;--v2 系 E2 乙1′ 件)")
    parser.add_argument("--v2", action="store_true",
                        help="教师 v2 采集+训练 → runs/bc-worker-v2/"
                             "(PREREG-内容案 E2)")
    parser.add_argument("--preventive-threshold", type=float, default=None,
                        choices=_REGISTERED_PREVENTIVE_THRESHOLDS,
                        help="v2 预防阈:rev13 仅允许 0.65 主案"
                             "(仅与 --v2 同用)")
    parser.add_argument(
        "--manager-npz", default=None,
        help="BC-v2 采集所用经理 policy.npz；缺省仍为 canonical v22 H。"
             "所选经理字节 SHA 会同时写入 demos/report，训练消费者必须与"
             "实际 --manager-npz 对账（仅与 --v2 同用）")
    cli = parser.parse_args()
    if cli.v2:
        threshold = (cli.preventive_threshold
                     if cli.preventive_threshold is not None
                     else _PREVENTIVE_THRESHOLD_MAIN)
        manager = pathlib.Path(cli.manager_npz) if cli.manager_npz else NPZ
        OUT_V2.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(OUT_V2 / ".bc.lock", "BC worker v2 产物"):
            main_v2(threshold, manager)
    else:
        if cli.preventive_threshold is not None:
            parser.error("--preventive-threshold 仅与 --v2 同用(v1 面一字不动)")
        if cli.manager_npz is not None:
            parser.error("--manager-npz 仅与 --v2 同用(v1 面一字不动)")
        with exclusive_lock(OUT / ".bc.lock", "BC worker 产物"):
            main()
