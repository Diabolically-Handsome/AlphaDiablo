"""DiabloGym v1 训练:PPO 学清地牢 1 层。

用法(仓库根目录):
  .venv/bin/python train/train_ppo.py --total-steps 2998272 --num-envs 4
  (指标落盘到 runs/<run>/progress.jsonl + status.json,dashboard.py 实时读取)
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import functools
import hashlib
import io
import json
import math
import os
import pathlib
import random
import shutil
import sys
import time
import types
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from eval_contract import PROTOCOL_VERSION, RUNTIME_PACKAGE_VERSIONS
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


_POLICY_HEAD_KEYS = (
    "mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.0.bias",
    "mlp_extractor.policy_net.2.weight", "mlp_extractor.policy_net.2.bias",
    "action_net.weight", "action_net.bias",
)
_RUN_ARTIFACTS = (
    "progress.jsonl", "status.json", "status.tmp.json", "sentinel.jsonl", "calib.jsonl",
    # 发射夜审计 A 修:三件新仪表档系追加写("a" 模式),不入列则同名重跑
    # 残留堆积,课程腿第二发腿终全表复核必假判 CASE_HALT_G0(G0-2a 16:55:51 同因)。
    "dry_curriculum.jsonl", "distill_ce_probe.jsonl", "drywin_metrics.jsonl",
    "bc_aux_monitor.jsonl", "bc_aux_behavior_receipt.json",
    "bc_aux_liveness_preflight.json",
    "model_final.zip", "model_candidate.zip", "model_development.zip",
    "policy.npz", "policy_sd.pt", "tb", "ckpt",
)
_ARTIFACT_SCOPE_RESULTS = {
    "development": ("model_development.zip", "DEVELOPMENT_ONLY"),
    "candidate": ("model_candidate.zip", "PRODUCTION_CANDIDATE"),
    "production": ("model_final.zip", "PUBLISHED"),
}
_GEAR_PRESENT_INDEX = 293  # base obs zero-based; 文档中的“第 294 维”
# 训练契约修订号单一真源。rev7 把此前只写进易丢运行配置的
# distill_beta / teacher_sha256 / calib_record_only 纳入 checkpoint 契约，
# 并登记 active bc_aux 是否携 liveness。否则 beta=0 或只记不裁的腿也能
# 伪装成同一正式目标。
# rev10 把有害的固定 argmax circuit 改为策略分布内的精确 5% mixture；
# rev11 修掉其“随机探索存在、确定性部署永远不可达”的致命错位：四个稳定
# raw 战斗特征 + bias 组成 5 参数 contextual gate，仍在同一精确 mixture
# 分布中训练，且发布前必须证明确定性 a12 跨多个 held-out episode 可达。
# rev12 撤销最后这条确定性动作配额：是否在 argmax 下喝药由 PPO 的战斗
# 回报自主裁决；发布仍要求原生认证的探索样本、低误饮、低漂移与资源安全，
# 实际战力则交给独立 paired efficacy 门。
# rev13 把 WorkerWindowEnv 的跨窗口终局信用与附加死亡成本纳入不可省略的
# 训练契约。旧实现把冻结 manager/script 期间发生的真实死亡只写审计账，
# PPO 收到 0；同时 FARM 内死亡与快进死亡若另加风险成本又很容易重复计罚。
# 两个旋钮必须由环境逐 transition 出具分账，默认值保持 rev12 逐位语义。
# rev14 显式绑定 Worker 策略看到 raw protocol-v4 还是 legacy protocol-v3
# 观测，但当时所谓 legacy 只解码 286/297；v4 已经在更早的怪物/地图/物品
# 列丢掉信息，旧 actor 仍然 OOD。rev15 要求环境在 lossless raw 边界重建
# 完整 v3 base。普通 actor 收 exact legacy-v3；A12 custom 收
# legacy-v3-a12-overlay（只把可逆的 packed belt/latch 覆盖回 286/297），
# 其继承 actor/value 在内部解码后也恰为完整 v3。
# rev16 closes the optimization target itself: real base-game termination and
# cross-window bootstrap are named, manager view is explicit, and main-PPO
# gradient clipping records whether actor/critic are independently bounded.
# rev18 binds the asymmetric 635-wide v2 wire contract.  v2 distinguishes an
# armed fuse at counter zero and excludes the training-only p_skip field from
# the actor context.  rev19 replaces that still-aliased wire with v3's bounded
# controller snapshot: combat candidates/ledger, potion order, exact scene
# state, and the radius-12 map consumed by a9/a10/a13/a14 are visible and
# execution uses the same frozen snapshot.  Rev20 replaces the affine context
# bias with a zero-output nonlinear context×legacy fusion.  Migrated logits
# remain bitwise V28 at ignition while conditional combat decisions become
# representable.  Rev21 binds root-only KING supervision, actor-rollout beta
# annealing, centered context fusion and context-specific reward/PPO evidence.
# Rev22 replaces both flat 9,100-wide branches with the frozen-layout shared
# block encoders, binds their layout identity/capacity, and clips legacy root,
# actor context, and critic independently.
# Rev23 removes a14 from legacy KING/V28 distillation support.  The historical
# teacher never executed a14, so supervising that logit actively opposes the
# new on-policy whole-loadout gear objective.
# Rev24 records the causal role of every policy source.  In particular, merely
# binding BC-v1 demos for PASS validation/read-only dry-anchor instrumentation
# is no longer allowed to masquerade as BC initialization or an optimizer
# objective; a14's fixed prior and native-reward PPO path are named separately.
# It also distinguishes a progressful TimeLimit truncation from the
# no-progress budget failure, which is terminal, non-bootstrap, and charged
# the current depth's death-equivalent base cost plus the configured risk cost.
# Rev25 upgrades the formal asymmetric Worker policy-gradient audit to
# diablogym-worker-onpolicy-pg/9.  That receipt seals the collection actor,
# independently closes GAE/log-probability inputs, and proves the optimizer's
# realised actor/root/context movement.  Rev24 remains an immutable historical
# evaluation contract bound to its original /8 audit; it is not silently
# reinterpreted as /9 and is not accepted for current-training continuation.
_CONTRACT_REVISION = 25
_REGISTERED_DUAL_WORKER_PG_AUDIT_SCHEMAS = types.MappingProxyType({
    24: "diablogym-worker-onpolicy-pg/8",
    25: "diablogym-worker-onpolicy-pg/9",
})
_POLICY_SOURCE_ROLES_SCHEMA = "diablogym-policy-source-roles/1"
_WORKER_EPISODE_BOUNDARY_V24 = (
    "base-game-terminal-plus-no-progress-timeout-failure"
)
_WORKER_NO_PROGRESS_TIMEOUT_CONTRACT = {
    "boundary": "terminated-no-bootstrap",
    "reward": (
        "death-ladder-base-plus-additional-terminal-death-cost"
    ),
}
_WORKER_VIEW_LEGACY_V3 = "legacy-v3"
_WORKER_VIEW_A12_OVERLAY = "legacy-v3-a12-overlay"
_WORKER_VIEW_DUAL_V4_ASYMMETRIC = "dual-v4-asymmetric-v3"
_DUAL_LEGACY_ACTOR_MIGRATION = "legacy-actor-migration"
_DUAL_ENV_RESTART_CONTINUATION = (
    "parameter-continuation-with-environment-restart-v1"
)
_RESUME_LINEAGE_SCHEMA = "diablogym-resume-lineage/1"
_ASYMMETRIC_ACTOR_INIT_SCHEMA = (
    "diablogym-asymmetric-worker-actor-init/5"
)
_ASYMMETRIC_ACTOR_INIT_METHOD = (
    "copy-v28-root-plus-zero-structured-centered-context-v3"
)
_ASYMMETRIC_CONTEXT_ARCHITECTURE = (
    "layout-v1-shared-blocks-centered-context-product-legacy-zero-output-v2"
)
_ASYMMETRIC_CRITIC_RESET_SCHEMA = (
    "diablogym-worker-critic-reset/3"
)
_ASYMMETRIC_CRITIC_RESET_METHOD = (
    "structured-layout-v1-orthogonal-value-only-v2"
)
_ASYMMETRIC_CRITIC_ARCHITECTURE = (
    "layout-v1-independent-shared-blocks-centered-value-v2"
)
_ASYMMETRIC_CANONICAL_EVIDENCE_SCHEMA = (
    "diablogym-asymmetric-worker-canonical-evidence/1"
)
_RUNTIME_VERSIONS = dict(RUNTIME_PACKAGE_VERSIONS)
_ALGORITHM_RECIPE = {
    "gae_lambda": 0.95,
    "n_epochs": 10,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,
    "target_kl": None,
    "use_sde": False,
    "sde_sample_freq": -1,
}
_SCHEDULE_PROBES = (1.0, 0.5, 0.0)
_BC_REPORT_SCHEMA_VERSION = 1
_EXPORT_MANIFEST_SCHEMA_VERSION = 1
# Every predecessor registry remains burned once collection opens it, even if
# no PASS artifact survives.  The immediately preceding 2100000/2101000 pools
# were consumed and therefore cannot service a new R7 prepare-bc.  Register
# fresh, disjoint active pools without deleting any old range from the
# ordinary-training exclusion table.
_WORKER_BC_DEMO_SEEDS = tuple(range(2_108_000, 2_108_128))
_BURNED_BC_EPISODES = frozenset(
    (*range(100, 484), *range(1000, 1384),
     *range(2000, 2128), *range(3000, 3384),
     *range(2_100_000, 2_100_128),
     *range(2_101_000, 2_101_384)))
_WORKER_BC_FORBIDDEN_ACTIONS = (11, 12)
# a14 is deliberately sparse, so the generic >=300-sample class rule used to
# omit it while aggregate top-1 still passed.  It is now a mandatory recall
# class with explicit demonstration breadth.
_WORKER_BC_REQUIRED_RECALL_ACTIONS = (14,)
_WORKER_BC_MIN_ACTION14_LABELS = 64
_WORKER_BC_MIN_ACTION14_EPISODES = 16
# E3 ④乙:禁采断言世代条件化(图纸 E2 共同真源)——v1 禁 (11,12) 原封(上行),
# v2 禁 11 允 12(守卫面不弱化)。
_WORKER_BC_V2_FORBIDDEN_ACTIONS = (11,)
# E3 ④乙:λ_bc 主案冻结常量(D7 注册裁量,与蒸馏锚 β=0.015625 同量级);
# 系 L-full CLI 显传值之文档/测试锚,非 --bc-aux-lambda 默认值(默认 0.0=不在位)。
_BC_AUX_MAIN_LAMBDA = 0.015625
# v33 修复版辅助目标。旧目标只保留 y==12 的 199 个正例，并在每个 PPO
# minibatch 重放，实测会把策略推成“m[12] 合法便几乎总喝”。rev2 同时钉死
# 数据面、损失面与消费频率，写入 training_contract，禁止旧 rev5 检查点把
# positive-only 语义静默带进新腿。
# rev3 进一步把辅助 bank 限定在 BC-v2 的 training episodes；原始 held-out
# episodes 只服务最终发布门，禁止正例/负例直接泄漏进优化器后再冒充独立验收。
# rev4 把负例 KL/rollout monitor 的锚从“每腿起点”改为首次 aux 根策略，
# 并随 checkpoint 持久化；禁止 continuation 逐腿重锚后累积漂移。
# rev5 把每 rollout 一次辅助更新从首个 PPO minibatch 的联合裁剪中拆出，
# 在 PPO epochs 后走独立 optimizer step；生产与 liveness 预检共用同一
# zero_grad/backward/clip/step 原子路径，避免纯 aux 预检系统性高估可学性。
# rev6 证明了结构路径可无损安装，却也给出了决定性的线上反证：把教师
# hp-band 目标冻结成 argmax 并不会增强战力，只会把成功的脑干反射提前，
# 16 局死亡完全不变且收益下降。rev7 因而把教师降格为训练期探索先验：
# actor 无损扩为 68，新增列只承载一个专用概率标量。rev8 修掉 rev7 最后
# 一个尾部漏洞：单一 raw-logit 权重无法在异质状态上同时做到“平均 5%”
# 和“从不成为 argmax”。现在在策略分布层精确定义
# π'(12)=ε, π'(a≠12)=(1−ε)π_non12(a)，rollout 与 evaluate_actions 共用；
# ε=.05 时 top non12≥.95/14>.05，故逐状态构造性不成 argmax。但 rev8
# 把 ε 封顶 0.25，真实 V28 eligible 状态确定性切换阈值却是 0.327..0.499；
# 训练能采到 a12、正式 deterministic eval 仍一次不会喝。rev9 以四个稳定
# raw 特征(HP、怪物密度、最近怪距离、腰带经济)+bias 学 ε(s)，概率上限
# 0.95，硬谓词外仍精确为 0；五个参数不向旧 actor 反传。rev10 不再把
# 教师式 deterministic a12 配额混入发布：PPO 可学会或拒绝该动作，最终
# 效力只由 paired 战力门裁决。
_BC_AUX_OBJECTIVE_REVISION = 11
_BC_AUX_CIRCUIT_SCHEMA = "a12-onpolicy-contextual-mixture-adapter/1"
_BC_AUX_CIRCUIT_BASE_WIDTH = 64
_BC_AUX_CIRCUIT_EXPANDED_WIDTH = 68
_BC_AUX_CIRCUIT_ACTION = 12
_BC_AUX_CIRCUIT_GATE_FEATURE_INDICES = (0, 8, 9, 286)
_BC_AUX_CIRCUIT_GATE_PARAMETER_COLUMNS = (64, 65, 66, 67)
_BC_AUX_CIRCUIT_INITIAL_PROBABILITY = 0.05
_BC_AUX_CIRCUIT_PROBABILITY_MIN = 0.001
_BC_AUX_CIRCUIT_PROBABILITY_MAX = 0.95
_BC_AUX_CIRCUIT_INITIAL_GATE_BIAS = math.log(
    (_BC_AUX_CIRCUIT_INITIAL_PROBABILITY
     - _BC_AUX_CIRCUIT_PROBABILITY_MIN)
    / (_BC_AUX_CIRCUIT_PROBABILITY_MAX
       - _BC_AUX_CIRCUIT_INITIAL_PROBABILITY)
)
_BC_AUX_CIRCUIT_GATE_PARAMETER_ABS_MAX = 8.0
_BC_AUX_CIRCUIT_KING_SUPPORT = "legal-non12-non14-renormalized"
_BC_AUX_MIN_DETERMINISTIC_A12_EPISODES = 2
_BC_AUX_MIN_DETERMINISTIC_A12_MARGIN = 1e-4
_BC_AUX_MIN_EXPECTED_A12_SAMPLES = 20.0
_BC_AUX_MIN_ACTUAL_A12_SAMPLES = 10
_BC_AUX_NEGATIVE_RATIO = 8
_BC_AUX_MIN_NEGATIVE_RATIO = 3
# feature297<0 且 m12=False 的非 12 行证明“同窗已饮”可见闩确实关闭了
# worker-owned 饮药键。rev8 mixture 在谓词外构造性给 p12=0，因此这些行
# 用作三域接口/掩码证据，不再伪装成可送进动作12 BCE 的 legal negative。
_BC_AUX_MIN_POST_DRINK_NEGATIVE_RATIO = 1
_BC_AUX_UPDATE_EVERY = 1
_BC_AUX_POSITIVE_FRACTION = 0.25
_BC_AUX_POSITIVE_TARGET = 0.65
_BC_AUX_NEGATIVE_TARGET = 0.01
_BC_AUX_ANCHOR_KL_COEF = 0.25
_BC_V2_DEMOS_SCHEMA_VERSION = "bc-worker-v2-demos/5"
_BC_V2_REPORT_SCHEMA_VERSION = "bc-worker-v2/7"
_BC_V2_TEACHER_GENERATION = 2
_BC_V2_PREVENTIVE_THRESHOLDS = (0.65,)
# Collection itself inspects whole-pool integrity, so the producer creates an
# immutable one-shot marker before the first reset.  The previous 2101000 pool
# has already been opened and stays burned; the active v2 registry below is
# disjoint from every predecessor, active v1, and the R7 eval bank.
_BC_V2_COLLECTION_EPISODES = tuple(range(2_103_000, 2_103_384))
if (
    set(_WORKER_BC_DEMO_SEEDS) & _BURNED_BC_EPISODES
    or set(_BC_V2_COLLECTION_EPISODES) & _BURNED_BC_EPISODES
    or set(_WORKER_BC_DEMO_SEEDS) & set(_BC_V2_COLLECTION_EPISODES)
):
    raise RuntimeError("active BC fresh seed registries 与已查看/彼此的池发生重叠")
_BC_V2_N12_MIN = 122
# BC-v2 producer/consumer 共同真源。bc_worker 反向从本模块导入，训练入口
# 则直接用同一常量重算三域回执，避免双份 schema/切分静默漂移。
_A12_CALIBRATION_SCHEMA_VERSION = "a12-teacher-boundary/3"
_A12_CALIBRATION_HP_FEATURE = 0
_A12_CALIBRATION_DRINK_LATCH_FEATURE = 297
# A 0.60 fit target left only ten percentage points above the 0.50 independent
# gate.  With roughly fifty a12 examples per episode-level validation split,
# ordinary sampling variation made that margin unreliable.  The fresh rev13
# pool freezes a 0.75 fit-only target before it is opened; final heldout remains
# unavailable to calibration.
_A12_CALIBRATION_TRAIN_RECALL_TARGET = 0.75
_A12_CALIBRATION_PREDICATE = (
    "visible-hp-band-live-m12-and-no-prior-window-drink")
_A12_VISIBLE_HP_BOUNDARY_EPS = 1e-6
_BC_FINAL_SPLIT_SEED = 23
_BC_FINAL_HOLDOUT_POOL_SCHEMA = "bc-final-holdout-pool/1"
_BC_FINAL_HOLDOUT_MARKER_SCHEMA = "bc-final-holdout-consumption/2"
_BC_SELECTION_VALIDATION_FRACTION = 0.10
_BC_SELECTION_SPLIT_SEED = 2301
# BC-v2 / smoke 共用的行为硬门。门限都在原始 held-out 分布上计算，不在
# 1:8 富集训练 bank 上计算；分母和真实逐样本 masks 一并入回执。
_A12_PRECISION_MIN = 0.05
_A12_RECALL_MIN = 0.50
# BC-v2 教师本身比后续 PPO 发布门更严格；bc_worker 从这里导入，
# 避免专用 0.5 门再次被通用“≥300 类 recall≥0.85”隐式覆盖。
_BC_V2_TEACHER_RECALL_MIN = 0.50
_A12_FPR_MAX = 0.002
_A12_PREDICTED_SHARE_MIN = 0.00005
_A12_PREDICTED_SHARE_MAX = 0.01
_A12_HIGH_HP_FALSE_DRINK_MAX = 0.001
_A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX = 1e-4
_A12_LEGAL_NEGATIVE_PROBABILITY_MAX = 1e-3
_A13_SPILLOVER_MAX = 0.02
# 最终发布相对“首次挂载 bc_aux 的根策略”而非每腿起点裁漂移。否则每腿
# 各退 15% 可在若干 continuation 后累积毁掉战斗/探索而逐腿全部 PASS。
_BC_AUX_ROOT_ARGMAX_DRIFT_MAX = 0.20
_BC_AUX_ROOT_TV_MAX = 0.15
_BC_AUX_ROOT_KL_MAX = 0.25
_BC_AUX_CRITICAL_RETENTION_MIN = 0.50
_BC_AUX_CRITICAL_ACTIONS = (9, 10, 13)
_BC_AUX_BEHAVIOR_METRIC_KEYS = frozenset({
    "scope", "mask_mode", "pairs", "tp", "fp", "fn", "tn",
    "true_a12", "non_a12", "all_non_a12", "predicted_a12",
    "predicted_a12_episodes", "predicted_a12_margin_min",
    "precision_12", "recall_12", "fpr_12", "predicted_share_12",
    "high_hp_non_a12", "high_hp_false_drinks",
    "high_hp_false_drink_rate", "predicted_share_13", "true_share_13",
    "eligible_probability_12_min", "eligible_probability_12_mean",
    "eligible_probability_12_max",
    "legal_negative_probability_12_mean",
    "legal_negative_probability_12_max",
    "legal_negative_probability_12_sum",
    "a13_reference", "a13_reference_share", "a13_spillover",
    "mean_probability_12", "anchor",
})
_BC_AUX_BEHAVIOR_RECEIPT_SCHEMA_VERSION = "bc-aux-behavior/8"
_BC_AUX_LIVENESS_PREFLIGHT_SCHEMA_VERSION = "bc-aux-liveness-preflight/4"
_BC_V2_PASS_KEYS = frozenset({
    "schema_version", "teacher_generation", "preventive_threshold",
    "pairs", "held_out_top1", "held_out_pairs", "held_out_episodes",
    "class_recalls", "class_weighted_retry",
    "n12", "n12_gate_min", "n12_by_episode",
    "recall_12", "recall_12_denominator", "recall_12_gate_min",
    "class_share_12", "class_share_13", "belt_economy",
    "a12_behavior", "a12_behavior_gate", "a12_calibration",
    "collection_episodes", "class_weights",
    "data_gate", "protocol_version", "implementation_sha256",
    "generator_sha256", "manager_npz_sha256", "policy_sha256",
    "demos_sha256", "final_pool_sha256",
    "final_holdout_marker_sha256",
})
# 历史 v3 driver 的兼容符号；当前训练路径绝不再把它当作真值。协议 v4
# 改变 Worker/掩码/窗口语义，BC-v1 必须重采，探针身份改由当前
# protocol+implementation 严格 PASS 回执的 demos_sha256 动态绑定。
_BC_V1_DEMOS_SHA256 = (
    "3bf892d611e41853eca8fce0cb146753af41ad2c3a21b6c581df1041fb1d9363")
# E5 探针专用 rng 种子(承 DryAnchorSentinel rng(26) 先例,孪生件同形;
# 只读探针自有流,不碰训练 RNG)。
_E5_PROBE_RNG_SEED = 26
# E5 探针示范态每组抽样上限(承 DryAnchorSentinel 固定抽 2000 先例)。
_E5_PROBE_GROUP_CAP = 2000
_BC_REPLAY_SEEDS = tuple(range(7000, 7032))
_BC_REPLAY_CACHE: dict[tuple[str, str, str, str], dict] = {}
_BC_PASS_KEYS = {
    "data_gate": {
        "schema_version", "pairs", "held_out_top1", "held_out_pairs",
        "held_out_episodes", "class_recalls", "class_weighted_retry",
        "data_gate", "protocol_version", "implementation_sha256",
        "generator_sha256", "manager_npz_sha256", "policy_sha256",
        "demos_sha256", "final_pool_sha256",
        "final_holdout_marker_sha256",
    },
    "hypothesis": {
        "schema_version", "pairs", "teacher_demo_mean", "bc_replay_7000",
        "teacher_7000", "ratio", "hypothesis", "protocol_version",
        "implementation_sha256", "generator_sha256", "policy_sha256",
    },
    "memoryless_hypothesis": {
        "schema_version", "pairs", "teacher_mean_demo",
        "bc_replay_mean_7000s", "teacher_replay_mean_7000s", "ratio",
        "memoryless_hypothesis", "protocol_version",
        "implementation_sha256", "generator_sha256", "policy_sha256",
    },
}


def _require(condition: bool, message: str) -> None:
    """训练契约不能用 assert：`python -O` 会把 assert 整段删掉。"""
    if not condition:
        raise ValueError(message)


def _masked_action_or_first_legal(
        requested, mask, *, n_actions: int, label: str) -> int:
    """Return ``requested`` when legal, otherwise the first legal action.

    Manager teachers intentionally remain simple heuristics and may propose an
    option that a newer environment contract has forced closed.  Falling back
    to a hard-coded action merely repeats the same violation when that action
    is also masked.  Validate the complete mask and derive the deterministic
    fallback from the mask itself; an all-false mask is an environment contract
    failure and must stop the run.
    """
    import numpy as np

    valid = np.asarray(mask, dtype=bool)
    _require(
        valid.shape == (n_actions,),
        f"{label} 动作掩码形状异常:{valid.shape} != {(n_actions,)}",
    )
    legal = np.flatnonzero(valid)
    _require(len(legal) > 0, f"{label} 动作掩码全假")
    if (
        isinstance(requested, (int, np.integer))
        and not isinstance(requested, (bool, np.bool_))
    ):
        candidate = int(requested)
        if 0 <= candidate < n_actions and bool(valid[candidate]):
            return candidate
    return int(legal[0])


def _is_plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _finite_number(value, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{label} 必须是数值")
    result = float(value)
    _require(math.isfinite(result), f"{label} 必须有限")
    return result


def _requested_drink_sovereignty(args) -> bool | None:
    """Return the CLI's tri-state action-12 request.

    New command lines use ``drink_sovereignty=None`` for automatic binding to
    a strict Worker NPZ contract.  Older unit fixtures still expose only the
    historical negative flag, so retain that input spelling without letting it
    leak back into the production parser.
    """
    if hasattr(args, "drink_sovereignty"):
        requested = getattr(args, "drink_sovereignty")
        _require(
            requested is None or isinstance(requested, bool),
            "drink_sovereignty 请求必须是 bool 或 None",
        )
        return requested
    legacy_disabled = getattr(args, "no_drink_sovereignty", False)
    _require(
        isinstance(legacy_disabled, bool),
        "no_drink_sovereignty 兼容字段必须是 bool",
    )
    return not legacy_disabled


def _effective_drink_sovereignty(args) -> bool:
    """Return the already-resolved action-12 semantics for contracts/config."""
    if hasattr(args, "resolved_drink_sovereignty"):
        resolved = getattr(args, "resolved_drink_sovereignty")
        _require(
            isinstance(resolved, bool),
            "resolved_drink_sovereignty 必须是 bool",
        )
        return resolved
    requested = _requested_drink_sovereignty(args)
    # Worker training and OptionsEnv without a tagged Worker historically use
    # environment-managed action 12.  ``None`` is only materially different
    # while assembling a strict Worker NPZ, where the metadata is authoritative.
    return True if requested is None else requested


def _resolve_training_drink_sovereignty(
        args, *, worker_npz_sha256: str | None) -> bool:
    """Resolve a tri-state CLI request against immutable Worker metadata."""
    requested = _requested_drink_sovereignty(args)
    worker_npz = getattr(args, "worker_npz", None)
    if getattr(args, "options", False) and worker_npz:
        from diablogym import NumpyManager
        from diablogym.worker_env import (
            WORKER_ACTION12_ENVIRONMENT_MASK,
            WORKER_ACTION12_PERMANENTLY_MASKED,
        )

        _require(
            _is_sha256(worker_npz_sha256),
            "Options Worker action12 解析必须绑定已捕获 NPZ SHA256",
        )
        net = NumpyManager(
            worker_npz, expected_sha256=worker_npz_sha256)
        net.require_io_shape(298, 15, "Options worker")
        mode = net.worker_action12_mode
        _require(
            mode in {
                WORKER_ACTION12_ENVIRONMENT_MASK,
                WORKER_ACTION12_PERMANENTLY_MASKED,
            },
            f"Worker NPZ action12 mode 非法:{mode!r}",
        )
        derived = mode == WORKER_ACTION12_ENVIRONMENT_MASK
        _require(
            requested is None or requested == derived,
            "命令行 drink_sovereignty 与 Worker NPZ action12 contract "
            f"冲突:requested={requested},contract={mode!r}",
        )
        return derived
    return True if requested is None else requested


def _bc_final_holdout_pool_spec(generation: int, seeds) -> dict:
    """Canonical one-shot final-pool identity shared by producer/consumer."""
    _require(
        _is_plain_int(generation) and generation in (1, 2),
        f"BC final heldout generation 非法:{generation!r}")
    raw_seeds = list(seeds)
    _require(
        bool(raw_seeds)
        and all(_is_plain_int(seed) for seed in raw_seeds)
        and len(raw_seeds) == len(set(raw_seeds)),
        "BC final heldout seed registry 必须是非空无重复整数表")
    normalized = [int(seed) for seed in raw_seeds]
    return {
        # Pool identity must not change when the marker file format evolves.
        # Otherwise a schema bump could make the same episodes look fresh.
        "schema_version": _BC_FINAL_HOLDOUT_POOL_SCHEMA,
        "teacher_generation": generation,
        "episode_seeds": normalized,
        "outer_split_seed": _BC_FINAL_SPLIT_SEED,
        "outer_holdout_fraction": 0.10,
        "split_unit": "whole-episode",
    }


def _bc_final_holdout_marker_identity(generation: int, seeds) -> tuple[dict, str]:
    spec = _bc_final_holdout_pool_spec(generation, seeds)
    payload = json.dumps(
        spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return spec, hashlib.sha256(payload).hexdigest()


def _bc_final_holdout_marker_path(
        out_dir: str | pathlib.Path, generation: int, seeds
) -> tuple[pathlib.Path, dict, str]:
    """Return the stable registry path for a one-shot BC pool.

    The registry is a sibling of the BC artifact directories, not a child of
    either bundle.  Renaming/archiving ``bc-worker[-v2]`` therefore cannot
    make the same seed pool look unused again.
    """
    spec, pool_sha256 = _bc_final_holdout_marker_identity(
        generation, seeds)
    registry = pathlib.Path(out_dir).parent / "_bc_final_holdout_registry"
    return registry / f"{pool_sha256}.json", spec, pool_sha256


def _assert_bc_final_holdout_pool_disjoint(
        out_dir: str | pathlib.Path, generation: int, seeds) -> None:
    """Reject any exact *or partial* reuse of a previously opened BC pool.

    The marker filename binds the exact pool, but exact-hash lookup alone does
    not prevent a later campaign from registering an overlapping superset or
    shifted range.  Every marker is therefore treated as an append-only global
    episode registry across both teacher generations.  A malformed historical
    marker is fail-closed: ignoring it could silently relabel consumed episodes
    as fresh.
    """
    marker, spec, _ = _bc_final_holdout_marker_path(
        out_dir, generation, seeds)
    requested = set(spec["episode_seeds"])
    registry = marker.parent
    if not registry.exists():
        return
    from eval_contract import EvalContractError, strict_json_loads

    _require(registry.is_dir(),
             f"BC final heldout registry 不是目录:{registry}")
    _require(not registry.is_symlink(),
             f"BC final heldout registry 不允许符号链接:{registry}")
    entries = sorted(registry.iterdir())
    unexpected = [
        entry for entry in entries
        if entry.name != ".registry.lock" and entry.suffix != ".json"
    ]
    _require(
        not unexpected,
        "BC final heldout registry 含未知残件，无法证明未消费:"
        f"{unexpected}",
    )
    for lock in (entry for entry in entries
                 if entry.name == ".registry.lock"):
        _require(lock.is_file() and not lock.is_symlink(),
                 f"BC final heldout registry lock 非普通文件:{lock}")
    marker_keys = {
        "schema_version",
        "teacher_generation",
        "episode_seeds",
        "outer_split_seed",
        "outer_holdout_fraction",
        "split_unit",
        "pool_sha256",
        "marker_schema_version",
        "started_at_ns",
        "provenance",
        "consumption_stage",
    }
    for existing in (entry for entry in entries if entry.suffix == ".json"):
        _require(existing.is_file() and not existing.is_symlink(),
                 f"BC final heldout registry marker 非普通文件:{existing}")
        try:
            record = strict_json_loads(existing.read_bytes())
        except (OSError, EvalContractError) as exc:
            raise ValueError(
                f"BC final heldout registry 旧 marker 不可解析:{existing}"
            ) from exc
        _require(
            isinstance(record, dict)
            and set(record) == marker_keys
            and _is_plain_int(record.get("teacher_generation"))
            and record["teacher_generation"] in (1, 2)
            and isinstance(record.get("episode_seeds"), list)
            and bool(record["episode_seeds"])
            and all(_is_plain_int(seed)
                    for seed in record["episode_seeds"])
            and len(record["episode_seeds"])
            == len(set(record["episode_seeds"])),
            f"BC final heldout registry 旧 marker pool 身份非法:{existing}",
        )
        existing_spec, existing_pool_sha256 = (
            _bc_final_holdout_marker_identity(
                record["teacher_generation"],
                record["episode_seeds"],
            )
        )
        _require(
            all(record.get(key) == value
                for key, value in existing_spec.items())
            and record.get("pool_sha256") == existing_pool_sha256
            and existing.stem == existing_pool_sha256
            and record.get("marker_schema_version")
            == _BC_FINAL_HOLDOUT_MARKER_SCHEMA
            and _is_plain_int(record.get("started_at_ns"))
            and record["started_at_ns"] > 0
            and isinstance(record.get("provenance"), dict)
            and bool(record["provenance"])
            and record.get("consumption_stage")
            == "before_pool_collection",
            f"BC final heldout registry 旧 marker 完整身份非法:{existing}",
        )
        overlap = requested.intersection(record["episode_seeds"])
        _require(
            not overlap,
            "BC final heldout episode 与已消费 registry 部分/全部重叠:"
            f"marker={existing},overlap={sorted(overlap)[:16]},"
            f"overlap_n={len(overlap)}",
        )


def _validate_bc_final_holdout_marker(
        out_dir: str | pathlib.Path, generation: int, seeds,
        expected_report: dict) -> dict:
    """Require the immutable pre-collection marker bound by a PASS report."""
    from eval_contract import EvalContractError, strict_json_loads

    marker, spec, pool_sha256 = _bc_final_holdout_marker_path(
        out_dir, generation, seeds)
    try:
        marker_payload = marker.read_bytes()
        record = strict_json_loads(marker_payload)
    except (OSError, EvalContractError) as exc:
        raise ValueError(
            f"BC final heldout one-shot marker 缺失/不可读:{marker}") from exc
    expected_keys = {
        *spec,
        "pool_sha256",
        "marker_schema_version",
        "started_at_ns",
        "provenance",
        "consumption_stage",
    }
    _require(
        isinstance(record, dict) and set(record) == expected_keys,
        "BC final heldout marker 字段/schema 不精确")
    _require(
        all(record[key] == value for key, value in spec.items())
        and record["pool_sha256"] == pool_sha256
        and record["marker_schema_version"]
        == _BC_FINAL_HOLDOUT_MARKER_SCHEMA
        and _is_plain_int(record["started_at_ns"])
        and record["started_at_ns"] > 0
        and record["consumption_stage"] == "before_pool_collection",
        "BC final heldout marker pool/时间/消费阶段身份不闭合")
    _require(
        isinstance(expected_report, dict)
        and expected_report.get("final_pool_sha256") == pool_sha256
        and _is_sha256(
            expected_report.get("final_holdout_marker_sha256"))
        and expected_report["final_holdout_marker_sha256"]
        == hashlib.sha256(marker_payload).hexdigest(),
        "BC PASS report 未精确绑定 final pool/marker 字节")
    provenance = record["provenance"]
    _require(isinstance(provenance, dict), "BC final marker provenance 非对象")
    provenance_keys = {
        "schema_version",
        "protocol_version",
        "implementation_sha256",
        "generator_sha256",
        "manager_npz_sha256",
    }
    if generation == 2:
        provenance_keys |= {"teacher_generation", "preventive_threshold"}
    _require(
        set(provenance) == provenance_keys
        and all(
            provenance.get(key) == expected_report.get(key)
            for key in provenance_keys
        ),
        "BC final heldout marker provenance 与 PASS report 不一致")
    return record


def _checkpoint_path(path: str | pathlib.Path) -> pathlib.Path:
    """按 SB3 规则容忍命令行省略 `.zip`。"""
    p = pathlib.Path(path)
    return p if p.exists() or p.suffix == ".zip" else pathlib.Path(f"{p}.zip")


def _capture_file_sha256(path: str | pathlib.Path, label: str) -> str:
    p = pathlib.Path(path)
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} 不可读: {p}: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


_IMPLEMENTATION_SOURCE_FILES = (
    "train/train_ppo.py",
    "train/leashed_ppo.py",
    "train/models.py",
    "train/eval_contract.py",
    "python/diablogym/__init__.py",
    "python/diablogym/controller_wire.py",
    "python/diablogym/env.py",
    "python/diablogym/nav.py",
    "python/diablogym/options_env.py",
    "python/diablogym/worker_env.py",
)


def _implementation_bundle_sha256() -> str:
    """Bind resume to code, native binaries and the actual game content."""
    import sysconfig

    from eval_contract import (content_identity, engine_binary_path,
                               runtime_versions_identity)

    root = pathlib.Path(__file__).resolve().parents[1]
    rel_paths = [
        pathlib.Path(relative)
        for relative in _IMPLEMENTATION_SOURCE_FILES
    ]
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    _require(bool(suffix), "当前 Python 没有 EXT_SUFFIX，无法绑定原生桥")
    rel_paths.append(pathlib.Path("build") / f"_diablogym{suffix}")

    digest = hashlib.sha256()
    for rel in rel_paths:
        p = root / rel
        try:
            payload = p.read_bytes()
        except OSError as exc:
            raise ValueError(f"实现绑定文件不可读: {p}: {exc}") from exc
        name = rel.as_posix().encode()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    # The bridge dynamically links the engine, so hashing only the extension
    # is insufficient. Locate the one actual engine binary fail-closed rather
    # than silently omitting it on another build layout.
    engine = engine_binary_path(root)
    try:
        engine_payload = engine.read_bytes()
    except OSError as exc:
        raise ValueError(f"实现绑定 engine 不可读: {engine}: {exc}") from exc
    engine_label = b"native-engine"
    digest.update(len(engine_label).to_bytes(4, "big"))
    digest.update(engine_label)
    digest.update(len(engine_payload).to_bytes(8, "big"))
    digest.update(engine_payload)

    # MPQ and Resources change world generation, collision, monsters and
    # rewards without changing source or binary bytes. Bind content rather
    # than host-specific absolute paths so an identical relocation is safe.
    content = content_identity(root)
    content_contract = {
        "game_data_sha256": content["game_data"]["sha256"],
        "assets_sha256": content["assets"]["sha256"],
        "assets_file_count": content["assets"]["file_count"],
    }
    encoded_content = json.dumps(
        content_contract, sort_keys=True, separators=(",", ":")).encode("ascii")
    content_label = b"runtime-content-v1"
    digest.update(len(content_label).to_bytes(4, "big"))
    digest.update(content_label)
    digest.update(len(encoded_content).to_bytes(8, "big"))
    digest.update(encoded_content)

    # BC generators call this helper too.  Binding only source/native/content
    # would let a policy trained under a different NumPy/Torch/SB3 numerical
    # stack present the same implementation identity later.
    encoded_versions = json.dumps(
        runtime_versions_identity(), sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    versions_label = b"runtime-versions-v1"
    digest.update(len(versions_label).to_bytes(4, "big"))
    digest.update(versions_label)
    digest.update(len(encoded_versions).to_bytes(8, "big"))
    digest.update(encoded_versions)
    return digest.hexdigest()


def _validate_runtime_versions() -> None:
    from importlib.metadata import version

    actual = {name: version(name) for name in _RUNTIME_VERSIONS}
    # 跨平台注记(2026-07-27 WSL2 移植):Linux 轮子带本地版本段(2.12.1+cpu),
    # 门槛按公开版本段比对;身份记录仍存完整本地版本(同 eval_contract 修订)。
    mismatches = {name: (actual[name], expected)
                  for name, expected in _RUNTIME_VERSIONS.items()
                  if actual[name] != expected
                  and actual[name].split("+", 1)[0] != expected}
    _require(not mismatches,
             f"训练运行时版本漂移（升级须重做数值回归）: {mismatches}")


def _validate_model_recipe(model, expected_target_kl=None) -> None:
    """Reject a foreign/resumed PPO whose hidden defaults changed."""
    clip_samples = tuple(float(model.clip_range(progress))
                         for progress in _SCHEDULE_PROBES)
    clip_vf_samples = (None if model.clip_range_vf is None else
                       tuple(float(model.clip_range_vf(progress))
                             for progress in _SCHEDULE_PROBES))
    actual = {
        "gae_lambda": float(model.gae_lambda),
        "n_epochs": int(model.n_epochs),
        # A schedule can equal the registered constant at progress=1 while
        # silently annealing later.  Sample the beginning, midpoint and end.
        "clip_range": clip_samples,
        "clip_range_vf": clip_vf_samples,
        "vf_coef": float(model.vf_coef),
        "max_grad_norm": float(model.max_grad_norm),
        "normalize_advantage": bool(model.normalize_advantage),
        "target_kl": model.target_kl,
        "use_sde": bool(getattr(model, "use_sde", False)),
        "sde_sample_freq": int(getattr(model, "sde_sample_freq", -1)),
    }
    expected_recipe = dict(_ALGORITHM_RECIPE)
    expected_recipe["target_kl"] = expected_target_kl
    expected_recipe["clip_range"] = (
        _ALGORITHM_RECIPE["clip_range"],) * len(_SCHEDULE_PROBES)
    if _ALGORITHM_RECIPE["clip_range_vf"] is not None:
        expected_recipe["clip_range_vf"] = (
            _ALGORITHM_RECIPE["clip_range_vf"],) * len(_SCHEDULE_PROBES)
    differences = {
        key: (actual[key], expected)
        for key, expected in expected_recipe.items()
        if (actual[key] != expected if not isinstance(expected, float)
            else not math.isclose(actual[key], expected, rel_tol=0, abs_tol=1e-12))
    }
    _require(not differences,
             f"PPO 隐含算法配方漂移（foreign/resume checkpoint）: {differences}")


def _worker_policy_observation_view(args) -> str | None:
    if not getattr(args, "worker", False):
        return None
    explicit = getattr(args, "worker_policy_observation_view", None)
    legacy_flag = bool(getattr(
        args, "legacy_worker_policy_observation_view", False))
    if explicit is not None:
        return str(explicit)
    if _bc_aux_structural_active(args):
        return _WORKER_VIEW_A12_OVERLAY
    if legacy_flag:
        return _WORKER_VIEW_LEGACY_V3
    return None


def _root_context_critic_gradient_clipping(max_grad_norm: float) -> dict:
    """Return the one registered clipping recipe from one finite bound."""
    value = float(max_grad_norm)
    _require(
        math.isfinite(value) and value > 0.0,
        "root/context/critic gradient clipping 要求有限正 max_grad_norm",
    )
    return {
        "mode": "separate-root-context-critic-v2",
        "root_max_norm": value / math.sqrt(2.0),
        "context_max_norm": value / math.sqrt(2.0),
        "combined_actor_max_norm": value,
        "critic_max_norm": value,
        "optimizer": "single",
        "trainable_shared_parameters": "forbidden",
    }


def _new_current_asymmetric_worker_policy():
    """Construct the registered policy topology without an environment."""
    import numpy as np
    import torch
    from gymnasium import spaces
    from leashed_ppo import (
        ASYMMETRIC_WORKER_OBSERVATION_DIM,
        AsymmetricWorkerMaskableActorCriticPolicy,
    )

    return AsymmetricWorkerMaskableActorCriticPolicy(
        spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(ASYMMETRIC_WORKER_OBSERVATION_DIM,),
            dtype=np.float32,
        ),
        spaces.Discrete(15),
        lambda _progress_remaining: 3e-4,
        net_arch={"pi": [64, 64], "vf": [64, 64]},
        activation_fn=torch.nn.Tanh,
    )


@functools.lru_cache(maxsize=1)
def _cached_current_asymmetric_worker_runtime_evidence() -> dict:
    """Measure current capacity/layout without consuming caller RNG streams."""
    import numpy as np
    import torch
    from leashed_ppo import asymmetric_worker_runtime_evidence

    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state().clone()
    try:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        policy = _new_current_asymmetric_worker_policy()
        return asymmetric_worker_runtime_evidence(policy)
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)


def _current_asymmetric_worker_runtime_evidence() -> dict:
    """Return an isolated copy so callers cannot corrupt the cached spec."""
    return copy.deepcopy(
        _cached_current_asymmetric_worker_runtime_evidence())


def _validate_registered_dual_worker_contract(
        contract: dict, *, expected_contract_revision: int,
        expected_worker_onpolicy_pg_audit_schema: str,
        runtime_evidence: dict | None = None) -> dict:
    """Validate a registered rev24+ dual-Worker architecture/optimizer ABI.

    Capacity is measured from the current policy topology instead of copied
    into another list of numeric constants.  The caller must supply the exact
    contract/audit pair so historical rev24 (/8) can be evaluated without
    either upgrading its claim or weakening current rev25 (/9).
    """
    from leashed_ppo import (
        ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES,
        ASYMMETRIC_WORKER_CONTEXT_INITIALIZER,
        ASYMMETRIC_WORKER_FROZEN_ACTOR_PARAMETER_COUNT,
        ASYMMETRIC_WORKER_FROZEN_ACTOR_TENSOR_COUNT,
        ASYMMETRIC_WORKER_FROZEN_CONTEXT_PARAMETER_COUNT,
        ASYMMETRIC_WORKER_FROZEN_CONTEXT_TENSOR_COUNT,
        ASYMMETRIC_WORKER_FROZEN_CRITIC_PARAMETER_COUNT,
        ASYMMETRIC_WORKER_FROZEN_CRITIC_TENSOR_COUNT,
        ASYMMETRIC_WORKER_RUNTIME_EVIDENCE_SCHEMA,
        WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
    )
    from diablogym.controller_wire import (
        DUAL_WORKER_LAYOUT,
        DUAL_WORKER_LAYOUT_SHA256,
    )

    _require(
        _REGISTERED_DUAL_WORKER_PG_AUDIT_SCHEMAS.get(
            expected_contract_revision)
        == expected_worker_onpolicy_pg_audit_schema,
        "dual-v4 contract revision/PG audit schema 未注册或错配",
    )
    _validate_policy_source_roles(contract)
    evidence = (
        _current_asymmetric_worker_runtime_evidence()
        if runtime_evidence is None else runtime_evidence
    )
    _require(
        isinstance(evidence, dict)
        and evidence.get("schema")
        == ASYMMETRIC_WORKER_RUNTIME_EVIDENCE_SCHEMA
        and isinstance(evidence.get("layout"), dict)
        and evidence["layout"].get("schema") == DUAL_WORKER_LAYOUT.schema
        and evidence["layout"].get("sha256")
        == DUAL_WORKER_LAYOUT_SHA256
        and evidence["layout"].get("observation_dim")
        == DUAL_WORKER_LAYOUT.observation_dim
        and isinstance(evidence.get("policy"), dict)
        and isinstance(evidence.get("context"), dict),
        "当前 asymmetric Worker runtime evidence 非法",
    )
    policy = evidence["policy"]
    context = evidence["context"]
    _require(
        context.get("parameter_count")
        == ASYMMETRIC_WORKER_FROZEN_CONTEXT_PARAMETER_COUNT
        and context.get("tensor_count")
        == ASYMMETRIC_WORKER_FROZEN_CONTEXT_TENSOR_COUNT
        and policy.get("actor_parameter_count")
        == ASYMMETRIC_WORKER_FROZEN_ACTOR_PARAMETER_COUNT
        and policy.get("actor_tensor_count")
        == ASYMMETRIC_WORKER_FROZEN_ACTOR_TENSOR_COUNT
        and policy.get("critic_parameter_count")
        == ASYMMETRIC_WORKER_FROZEN_CRITIC_PARAMETER_COUNT
        and policy.get("critic_tensor_count")
        == ASYMMETRIC_WORKER_FROZEN_CRITIC_TENSOR_COUNT,
        "当前 asymmetric Worker 容量/张量拓扑漂移",
    )
    actor_migration = (
        contract.get("actor_migration")
        if isinstance(contract, dict) else None)
    critic_migration = (
        contract.get("critic_migration")
        if isinstance(contract, dict) else None)
    algorithm_recipe = (
        contract.get("algorithm_recipe")
        if isinstance(contract, dict) else None)
    max_grad_norm = (
        algorithm_recipe.get("max_grad_norm")
        if isinstance(algorithm_recipe, dict) else None)
    action14_logit_bonus = (
        contract.get("worker_action14_logit_bonus")
        if isinstance(contract, dict) else None)
    _require(
        isinstance(max_grad_norm, (int, float))
        and not isinstance(max_grad_norm, bool)
        and math.isclose(
            float(max_grad_norm),
            float(_ALGORITHM_RECIPE["max_grad_norm"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "dual-v4 contract max_grad_norm 未绑定当前配方",
    )
    expected_clipping = _root_context_critic_gradient_clipping(
        float(max_grad_norm))
    _require(
        _is_plain_int(expected_contract_revision)
        and expected_contract_revision >= 24
        and isinstance(expected_worker_onpolicy_pg_audit_schema, str)
        and bool(expected_worker_onpolicy_pg_audit_schema)
        and isinstance(contract, dict)
        and contract.get("schema_version") == 2
        and contract.get("contract_revision") == expected_contract_revision
        and contract.get("mode") == "worker"
        and contract.get("worker_policy_observation_view")
        == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
        and contract.get("worker_episode_boundary")
        == _WORKER_EPISODE_BOUNDARY_V24
        and contract.get("worker_window_bootstrap")
        == "next-learning-window"
        and contract.get("worker_no_progress_timeout")
        == _WORKER_NO_PROGRESS_TIMEOUT_CONTRACT
        and isinstance(action14_logit_bonus, (int, float))
        and not isinstance(action14_logit_bonus, bool)
        and math.isfinite(float(action14_logit_bonus))
        and 0.0 <= float(action14_logit_bonus) <= 10.0
        and contract.get("observation_shape")
        == [DUAL_WORKER_LAYOUT.observation_dim]
        and contract.get("action_n") == 15
        and isinstance(actor_migration, dict)
        and actor_migration.get("method")
        == _ASYMMETRIC_ACTOR_INIT_METHOD
        and actor_migration.get("context_architecture")
        == _ASYMMETRIC_CONTEXT_ARCHITECTURE
        and actor_migration.get("controller_layout_schema")
        == DUAL_WORKER_LAYOUT.schema
        and actor_migration.get("controller_layout_sha256")
        == DUAL_WORKER_LAYOUT_SHA256
        and actor_migration.get("target_actor_parameter_tensors")
        == policy.get("actor_tensor_count")
        and actor_migration.get("target_actor_parameter_count")
        == policy.get("actor_parameter_count")
        and actor_migration.get("context_parameter_tensors")
        == context.get("tensor_count")
        and actor_migration.get("context_parameter_count")
        == context.get("parameter_count")
        and actor_migration.get("context_initialization") == {
            "hidden": ASYMMETRIC_WORKER_CONTEXT_INITIALIZER,
            "output": "exact-zero-disabled-through-critic-warmup",
        }
        and actor_migration.get(
            "actor_context_excluded_observation_features")
        == list(ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES)
        and _is_sha256(actor_migration.get(
            "source_checkpoint_sha256"))
        and _is_sha256(actor_migration.get("source_actor_sha256"))
        and _is_sha256(actor_migration.get("migrated_actor_sha256"))
        and isinstance(critic_migration, dict)
        and critic_migration.get("method")
        == _ASYMMETRIC_CRITIC_RESET_METHOD
        and critic_migration.get("critic_architecture")
        == _ASYMMETRIC_CRITIC_ARCHITECTURE
        and critic_migration.get("controller_layout_schema")
        == DUAL_WORKER_LAYOUT.schema
        and critic_migration.get("controller_layout_sha256")
        == DUAL_WORKER_LAYOUT_SHA256
        and critic_migration.get("critic_parameter_tensors")
        == policy.get("critic_tensor_count")
        and critic_migration.get("critic_parameter_count")
        == policy.get("critic_parameter_count")
        and critic_migration.get("source_checkpoint_sha256")
        == actor_migration.get("source_checkpoint_sha256")
        and critic_migration.get("source_actor_sha256")
        == actor_migration.get("source_actor_sha256")
        and _is_plain_int(critic_migration.get("warmup_steps"))
        and critic_migration["warmup_steps"] > 0
        and critic_migration.get("gradient_clip_mode")
        == expected_clipping["mode"]
        and critic_migration.get("worker_onpolicy_pg_audit_schema")
        == expected_worker_onpolicy_pg_audit_schema
        and critic_migration.get(
            "worker_onpolicy_pg_min_optimizer_steps_per_joint_rollout")
        == WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT
        and contract.get("gradient_clipping") == expected_clipping,
        f"checkpoint 不是完整注册的 rev{expected_contract_revision} "
        "dual-v4 Worker",
    )
    return evidence


def _validate_current_dual_worker_contract(
        contract: dict, *, runtime_evidence: dict | None = None) -> dict:
    """Validate the exact current dual-Worker training/resume contract."""
    from leashed_ppo import WORKER_ONPOLICY_PG_AUDIT_SCHEMA

    return _validate_registered_dual_worker_contract(
        contract,
        expected_contract_revision=_CONTRACT_REVISION,
        expected_worker_onpolicy_pg_audit_schema=(
            WORKER_ONPOLICY_PG_AUDIT_SCHEMA),
        runtime_evidence=runtime_evidence,
    )


def _classify_dual_worker_resume(args, resume_data: dict) -> str | None:
    """Classify V28 migration versus an explicit environment-restart resume.

    SB3 checkpoints do not contain the native Diablo world, wrapper/controller
    state, VecEnv observations, or every RNG stream.  A dual checkpoint can
    therefore preserve parameters, Adam and counters, but it cannot continue
    the same trajectory.  Keep that weaker operation explicit in both the
    classifier name and the required CLI acknowledgement.
    """
    if (_worker_policy_observation_view(args)
            != _WORKER_VIEW_DUAL_V4_ASYMMETRIC):
        return None
    _require(isinstance(resume_data, dict),
             "dual-v4 resume checkpoint data 不是对象")
    saved = resume_data.get("diablogym_contract")
    reset_critic = bool(getattr(args, "reset_worker_critic", False))
    allow_legacy = bool(getattr(args, "allow_legacy_resume", False))
    if saved is None:
        _require(
            reset_critic and allow_legacy,
            "从 298 维旧 Worker 点火 dual-v4 必须同时显式 "
            "--reset-worker-critic 与 --allow-legacy-resume",
        )
        return _DUAL_LEGACY_ACTOR_MIGRATION

    _require(isinstance(saved, dict),
             "dual-v4 checkpoint training_contract 不是对象")
    _require(
        not reset_critic and not allow_legacy,
        "已有 dual-v4 training_contract 的 checkpoint 必须参数/优化器续接；"
        "禁止重复 reset critic 或伪装 legacy migration",
    )
    # Validate the complete architecture/optimizer ABI before classifying the
    # weaker environment-restart continuation operation.
    _validate_current_dual_worker_contract(saved)
    _require(
        bool(getattr(args, "allow_environment_restart_resume", False)),
        "dual-v4 checkpoint 不含原生世界/包装器/RNG 状态；参数与 Adam "
        "续接会从新环境轨迹重启。必须显式传 "
        "--allow-environment-restart-resume",
    )
    return _DUAL_ENV_RESTART_CONTINUATION


def _build_resume_lineage(
        resume_data: dict, *, parent_sha256: str, operation: str,
        seed: int | None, optimizer_reset: bool,
        critic_reset: bool) -> dict:
    """Build the checkpoint-persisted receipt for a non-exact resume."""
    _require(
        isinstance(resume_data, dict)
        and _is_sha256(parent_sha256)
        and operation in {
            _DUAL_LEGACY_ACTOR_MIGRATION,
            _DUAL_ENV_RESTART_CONTINUATION,
        },
        "resume lineage 输入非法",
    )
    parent_steps = resume_data.get("num_timesteps")
    _require(
        _is_plain_int(parent_steps) and parent_steps >= 0,
        "resume lineage parent num_timesteps 非法",
    )
    _require(
        seed is None
        or (_is_plain_int(seed) and 0 <= seed < 2**32),
        "resume lineage seed 非法",
    )
    previous = resume_data.get("_resume_lineage")
    if previous is None:
        generation = 1
    else:
        _require(
            isinstance(previous, dict)
            and previous.get("schema") == _RESUME_LINEAGE_SCHEMA
            and _is_plain_int(previous.get("generation"))
            and previous["generation"] >= 1,
            "resume checkpoint 的既有 lineage 非法",
        )
        generation = previous["generation"] + 1
    return {
        "schema": _RESUME_LINEAGE_SCHEMA,
        "generation": generation,
        "operation": operation,
        "immediate_parent_sha256": parent_sha256,
        "immediate_parent_num_timesteps": parent_steps,
        "environment_state_mode":
            "reinitialized-no-native-or-wrapper-snapshot",
        "rng_state_mode": (
            "reseeded-from-explicit-cli-seed"
            if seed is not None else
            "runtime-new-streams-without-exact-restoration"
        ),
        "requested_seed": seed,
        "policy_parameter_state": (
            "v28-root-transplanted-context-canonical-init"
            if operation == _DUAL_LEGACY_ACTOR_MIGRATION
            else "checkpoint-preserved"
        ),
        "optimizer_state": "reset" if optimizer_reset else "preserved",
        "critic_state": "reset" if critic_reset else "preserved",
        "exact_trajectory_continuation": False,
    }


def _validate_worker_policy_observation_binding(args, model) -> None:
    """Close the CLI view selector against the policy class actually loaded."""
    if not args.worker:
        return
    from leashed_ppo import (
        ASYMMETRIC_WORKER_OBSERVATION_DIM,
        A12MixtureMaskableActorCriticPolicy,
        AsymmetricWorkerMaskableActorCriticPolicy,
    )

    expected_spec = _bc_aux_circuit_spec()
    custom = isinstance(
        model.policy, A12MixtureMaskableActorCriticPolicy)
    asymmetric = isinstance(
        model.policy, AsymmetricWorkerMaskableActorCriticPolicy)
    model_spec = getattr(model, "_bc_aux_circuit_spec", None)
    policy_spec = getattr(model.policy, "bc_aux_mixture_spec", None)
    if custom:
        _require(
            model_spec == expected_spec
            and policy_spec == expected_spec
            and getattr(model, "policy_class", None)
            is A12MixtureMaskableActorCriticPolicy,
            "实际 A12 custom policy class/spec 未闭合",
        )
    else:
        _require(
            model_spec is None and policy_spec is None
            and getattr(model, "policy_class", None)
            is not A12MixtureMaskableActorCriticPolicy,
            "普通 Worker 携带残留 A12 class/spec",
        )
    expected_custom = _bc_aux_structural_active(args)
    _require(
        custom is expected_custom,
        "CLI structural A12 配方与实际加载 policy class 不一致",
    )
    legacy_view = bool(getattr(
        args, "legacy_worker_policy_observation_view", False))
    selected_view = _worker_policy_observation_view(args)
    _require(
        (
            custom
            and not legacy_view
            and selected_view == _WORKER_VIEW_A12_OVERLAY
        )
        or (
            asymmetric
            and not legacy_view
            and selected_view == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
        )
        or (
            not custom
            and not asymmetric
            and selected_view == _WORKER_VIEW_LEGACY_V3
        ),
        "实际 Worker policy class 与环境 observation view 不一致",
    )
    _require(
        asymmetric
        is (selected_view == _WORKER_VIEW_DUAL_V4_ASYMMETRIC),
        "dual-v4-asymmetric-v3 观测与实际 asymmetric policy class 不一致",
    )
    expected_action14_bonus = float(getattr(
        args, "worker_action14_logit_bonus", 0.0))
    actual_action14_bonus = getattr(
        model.policy, "action14_logit_bonus", None)
    _require(
        (
            asymmetric
            and isinstance(actual_action14_bonus, float)
            and math.isclose(
                actual_action14_bonus,
                expected_action14_bonus,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )
        or (
            not asymmetric
            and expected_action14_bonus == 0.0
            and actual_action14_bonus is None
        ),
        "CLI action14 logit prior 与实际 Worker policy 不一致",
    )
    actor_migration = getattr(model, "_actor_migration_receipt", None)
    _require(
        (asymmetric and isinstance(actor_migration, dict))
        or (not asymmetric and actor_migration is None),
        "asymmetric policy class 与 actor migration receipt 不一致",
    )
    expected_observation_dim = (
        ASYMMETRIC_WORKER_OBSERVATION_DIM if asymmetric else 298)
    _require(
        tuple(model.observation_space.shape) == (expected_observation_dim,),
        "Worker policy class 与实际 observation space 形状不一致:"
        f"{model.observation_space.shape} != ({expected_observation_dim},)",
    )


def _check_finite_tree(value, label: str) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        _require(torch.isfinite(value).all().item(), f"{label} 含 NaN/Inf")
    elif isinstance(value, dict):
        for key, child in value.items():
            _check_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, float):
        _require(math.isfinite(value), f"{label} 含非有限标量")


def _validate_checkpoint_bytes(payload: bytes, label: str,
                               require_leashed: bool = False) -> dict:
    """Validate one immutable checkpoint byte string."""
    import torch

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            bad_member = archive.testzip()
            _require(bad_member is None, f"checkpoint CRC 失败: {bad_member}")
            member_names = archive.namelist()
            _require(len(member_names) == len(set(member_names)),
                     f"checkpoint 含重复 ZIP 成员: {label}")
            names = set(member_names)
            _require({"data", "policy.pth", "policy.optimizer.pth"} <= names,
                     f"checkpoint 缺关键成员: {label}")
            data = json.loads(archive.read("data"))
            saved_sb3 = archive.read("_stable_baselines3_version").decode().strip()
            _require(saved_sb3 == _RUNTIME_VERSIONS["stable-baselines3"],
                     f"checkpoint SB3 版本 {saved_sb3} 与运行时配方不符")
            for name in sorted(n for n in names if n.endswith(".pth")):
                state = torch.load(io.BytesIO(archive.read(name)), map_location="cpu",
                                   weights_only=True)
                _check_finite_tree(state, name)
    except (OSError, KeyError, ValueError, RuntimeError, zipfile.BadZipFile,
            json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("checkpoint"):
            raise
        raise ValueError(f"checkpoint 不可读/不安全: {label}: {exc}") from exc
    _require(isinstance(data, dict), f"checkpoint data 不是对象: {label}")
    try:
        steps = data["num_timesteps"]
    except KeyError as exc:
        raise ValueError("checkpoint num_timesteps 缺失/非法") from exc
    _require(_is_plain_int(steps) and steps >= 0,
             "checkpoint num_timesteps 必须是非负普通整数")
    if require_leashed:
        _require("distill_beta" in data,
                 "resume 检查点不是 LeashedMaskablePPO（缺 distill_beta 标记）")
    return data


def _validate_checkpoint_file(path: str | pathlib.Path,
                              require_leashed: bool = False) -> dict:
    """CRC + metadata + finite policy/Adam, read from the path exactly once."""
    p = _checkpoint_path(path)
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"checkpoint 不可读/不安全: {p}: {exc}") from exc
    return _validate_checkpoint_bytes(payload, str(p), require_leashed)


def _validate_leashed_metadata(data: dict) -> dict:
    _require("distill_beta" in data,
             "resume 检查点不是 LeashedMaskablePPO（缺 distill_beta 标记）")
    try:
        beta = float(data["distill_beta"])
    except (TypeError, ValueError) as exc:
        raise ValueError("resume 检查点 distill_beta 标记非法") from exc
    _require(math.isfinite(beta) and beta >= 0,
             "resume 检查点 distill_beta 必须是有限非负数")
    return data


def _validate_resumable_leashed_boundary(data: dict) -> dict:
    """Reject a contracted checkpoint whose latest rollout was not trained.

    SB3 increments ``num_timesteps`` during collection, before ``train()``
    consumes that buffer.  Saving in that interval creates a particularly
    dangerous checkpoint: a later run starts its budget/curriculum after the
    collected samples although the parameters never received their gradient.
    Registered legacy migration sources have no project contract and are
    validated by their own frozen SHA/topology gates; every contracted
    Leashed continuation must carry the stronger optimizer receipt.
    """
    contract = data.get("diablogym_contract")
    if contract is None:
        return data
    _require(isinstance(contract, dict),
             "resume training_contract 不是对象")
    steps = data.get("num_timesteps")
    completed = data.get("_last_completed_ppo_rollout_steps")
    optimizer_steps = data.get("_ppo_optimizer_steps_completed")
    _require(
        _is_plain_int(completed) and completed == steps,
        "resume checkpoint 不是已由 optimizer 消费的 rollout 边界:"
        f"completed={completed!r},num_timesteps={steps!r}",
    )
    _require(
        _is_plain_int(optimizer_steps) and optimizer_steps > 0,
        "resume checkpoint 缺已完成 PPO optimizer step 回执:"
        f"{optimizer_steps!r}",
    )
    n_steps = contract.get("n_steps")
    num_envs = contract.get("num_envs")
    _require(
        _is_plain_int(n_steps) and n_steps > 0
        and _is_plain_int(num_envs) and num_envs > 0,
        "resume training_contract 缺合法 n_steps/num_envs",
    )
    quantum = n_steps * num_envs
    _require(
        steps % quantum == 0,
        "resume checkpoint num_timesteps 未对齐训练 rollout 量子:"
        f"{steps} % {quantum}",
    )
    # These lists belong to the buffer that is about to be optimized.  Normal
    # Leashed saves deliberately exclude them; reject forged/foreign payloads
    # that try to carry an unconsumed stream across an environment restart.
    for key in (
        "_worker_onpolicy_pg_pending_receipts",
        "_bc_aux_pending_action_receipts",
    ):
        _require(
            key not in data or data[key] == [],
            f"resume checkpoint 携未消费 pending receipt:{key}",
        )
    return data


def _validate_leashed_checkpoint(path: str | pathlib.Path) -> dict:
    """用保存元数据区分 Leashed 检查点，禁止普通 MaskablePPO 冒充续训源。"""
    return _validate_resumable_leashed_boundary(
        _validate_leashed_metadata(
            _validate_checkpoint_file(path, require_leashed=True)))


def _capture_leashed_checkpoint(path: str | pathlib.Path) -> tuple[bytes, dict, str]:
    """Capture, validate and hash the exact bytes later passed to SB3.load()."""
    p = _checkpoint_path(path)
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"resume 检查点不可读: {p}: {exc}") from exc
    data = _validate_resumable_leashed_boundary(
        _validate_leashed_metadata(
            _validate_checkpoint_bytes(
                payload, str(p), require_leashed=True)))
    return payload, data, hashlib.sha256(payload).hexdigest()


def _select_batch_size(n_steps: int, n_envs: int, cap: int = 256) -> int:
    """保持 256 配方；仅在尾 minibatch 恰为 1 时下调，避免 std=NaN。"""
    rollout_size = n_steps * n_envs
    _require(rollout_size > 1, "n_steps * num_envs 必须大于 1")
    for size in range(cap, 1, -1):
        if rollout_size <= size or rollout_size % size != 1:
            return size
    return rollout_size


def _reset_policy_optimizer(model, learning_rate: float) -> None:
    """重建 policy optimizer，清除 continuation 驮带的全部 Adam moments。

    权重张量不经 state_dict 往返；只替换 optimizer 对象与 lr schedule。
    调用者须在 load 完成、训练开始前执行，并由 ``--reset-optimizer`` 显式
    授权。该助手独立可测，防“只把 step 写零但 exp_avg 仍在”的伪重置。
    """
    import torch

    _require(math.isfinite(learning_rate) and learning_rate > 0,
             "reset optimizer 学习率必须是有限正数")
    policy = model.policy
    optimizer_class = getattr(policy, "optimizer_class", None)
    optimizer_kwargs = dict(getattr(policy, "optimizer_kwargs", {}) or {})
    _require(optimizer_class is not None, "policy 缺 optimizer_class，无法安全重建")
    before = {name: value.detach().clone()
              for name, value in policy.state_dict().items()}
    model.learning_rate = float(learning_rate)
    model._setup_lr_schedule()
    optimizer_kwargs.pop("lr", None)
    policy.optimizer = optimizer_class(
        policy.parameters(), lr=float(model.lr_schedule(1.0)),
        **optimizer_kwargs)
    _require(not policy.optimizer.state,
             "reset optimizer 后仍含旧 state/Adam moments")
    for name, value in policy.state_dict().items():
        _require(torch.equal(before[name], value.detach()),
                 f"reset optimizer 意外改动 policy 权重:{name}")


def _stable_named_tensor_sha256(state: dict, keys) -> str:
    """Match leashed_ppo's versioned named-parameter digest."""
    import numpy as np
    import torch

    keys = tuple(keys)
    _require(
        isinstance(state, dict) and all(key in state for key in keys),
        "policy state_dict 缺少摘要张量",
    )
    digest = hashlib.sha256()
    for key in sorted(keys):
        tensor = state[key]
        _require(
            isinstance(tensor, torch.Tensor)
            and bool(torch.isfinite(tensor).all().item()),
            f"policy state_dict 张量非法:{key}",
        )
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _checkpoint_policy_state(checkpoint_payload: bytes) -> dict:
    import torch

    try:
        with zipfile.ZipFile(io.BytesIO(checkpoint_payload)) as archive:
            state = torch.load(
                io.BytesIO(archive.read("policy.pth")),
                map_location="cpu",
                weights_only=True,
            )
    except (OSError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ValueError("checkpoint policy.pth 无法解析") from exc
    _require(isinstance(state, dict), "checkpoint policy.pth 不是 state_dict")
    return state


def _initialize_asymmetric_worker_actor(
        model, *, source_checkpoint_payload: bytes,
        source_checkpoint_sha256: str) -> dict:
    """Copy V28's six actor tensors into the current asymmetric policy."""
    import numpy as np
    import torch
    from sb3_contrib.common.maskable.policies import (
        MaskableActorCriticPolicy,
    )
    from stable_baselines3.common.torch_layers import (
        FlattenExtractor,
        MlpExtractor,
    )
    from leashed_ppo import (
        ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES,
        ASYMMETRIC_WORKER_CONTEXT_INITIALIZER,
        ASYMMETRIC_WORKER_LEGACY_DIM,
        ASYMMETRIC_WORKER_OBSERVATION_DIM,
        AsymmetricWorkerMaskableActorCriticPolicy,
        LeashedMaskablePPO,
        actor_parameter_sha256,
        strict_actor_critic_parameter_partition,
    )
    from diablogym.controller_wire import (
        DUAL_WORKER_LAYOUT,
        DUAL_WORKER_LAYOUT_SHA256,
    )

    _require(
        isinstance(model.policy, AsymmetricWorkerMaskableActorCriticPolicy),
        "dual-v4 actor migration 要求 asymmetric policy class")
    _require(str(getattr(model, "device", "")) == "cpu",
             "dual-v4 actor bitwise migration 当前只认证 CPU")
    _require(_is_sha256(source_checkpoint_sha256),
             "dual-v4 actor migration 缺 source checkpoint SHA")
    _require(
        hashlib.sha256(source_checkpoint_payload).hexdigest()
        == source_checkpoint_sha256,
        "dual-v4 actor migration source payload/SHA 不一致",
    )
    source = _checkpoint_policy_state(source_checkpoint_payload)
    critic_keys = (
        "mlp_extractor.value_net.0.weight",
        "mlp_extractor.value_net.0.bias",
        "mlp_extractor.value_net.2.weight",
        "mlp_extractor.value_net.2.bias",
        "value_net.weight",
        "value_net.bias",
    )
    expected_source_keys = set((*_POLICY_HEAD_KEYS, *critic_keys))
    _require(
        set(source) == expected_source_keys,
        "dual-v4 actor migration source policy state_dict 字段不精确:"
        f"missing={sorted(expected_source_keys - set(source))},"
        f"extra={sorted(set(source) - expected_source_keys)}",
    )
    # ``BaseAlgorithm.load()`` reconstructs the source model and calls its
    # saved ``set_random_seed``.  This forensic load must not replace the
    # already-created target run's Python/NumPy/Torch streams with V28's fixed
    # seed: those streams drive action sampling and PPO minibatch permutations,
    # and collapsing them would invalidate independent training replications.
    # The migration is CPU-only above, so the complete Torch stream in scope is
    # the CPU generator captured here.
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state().clone()
    try:
        source_model = LeashedMaskablePPO.load(
            io.BytesIO(source_checkpoint_payload),
            env=None,
            device="cpu",
            teacher_path=None,
            teacher_sha256=None,
        )
    except Exception as exc:
        raise ValueError(
            "dual-v4 actor migration source model 无法按真实 policy 加载"
        ) from exc
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
    source_policy = source_model.policy
    _require(
        type(source_model) is LeashedMaskablePPO
        and type(source_policy) is MaskableActorCriticPolicy
        and getattr(source_model, "policy_class", None)
        is MaskableActorCriticPolicy
        and type(source_policy.features_extractor) is FlattenExtractor
        and bool(source_policy.share_features_extractor)
        and source_policy.activation_fn is torch.nn.Tanh
        and source_policy.net_arch == {
            "pi": [64, 64], "vf": [64, 64]}
        and type(source_policy.mlp_extractor) is MlpExtractor
        and tuple(source_model.observation_space.shape) == (298,)
        and getattr(source_model.action_space, "n", None) == 15,
        "dual-v4 actor migration source 不是注册的 "
        "plain Flatten/Tanh 298→64→64→15 policy",
    )
    expected_layers = (
        torch.nn.Linear,
        torch.nn.Tanh,
        torch.nn.Linear,
        torch.nn.Tanh,
    )
    _require(
        tuple(type(layer) for layer in source_policy.mlp_extractor.policy_net)
        == expected_layers
        and tuple(
            type(layer)
            for layer in source_policy.mlp_extractor.value_net
        ) == expected_layers,
        "dual-v4 actor migration source MLP 层序不精确",
    )
    loaded_source_state = source_policy.state_dict()
    _require(
        set(loaded_source_state) == expected_source_keys
        and all(
            torch.equal(loaded_source_state[key].detach().cpu(), source[key])
            for key in expected_source_keys
        ),
        "dual-v4 actor migration source 实际 policy 与 policy.pth 不一致",
    )
    source_actor_sha256 = _stable_named_tensor_sha256(
        source, _POLICY_HEAD_KEYS)
    source_critic_sha256 = _stable_named_tensor_sha256(
        source, critic_keys)
    target = model.policy.state_dict()
    with torch.no_grad():
        for key in _POLICY_HEAD_KEYS:
            _require(
                key in target
                and target[key].shape == source[key].shape
                and target[key].dtype == source[key].dtype,
                f"dual-v4 actor tensor 不可逐位移植:{key}",
            )
            target[key].copy_(source[key])
    extractor = model.policy.mlp_extractor
    adapter = extractor.context_adapter
    parameter_groups = adapter.named_parameter_groups()
    _require(
        tuple(parameter_groups) == ("encoder", "interaction", "output"),
        "dual-v4 actor context semantic parameter groups 漂移",
    )
    hidden_parameters = (
        *parameter_groups["encoder"],
        *parameter_groups["interaction"],
    )
    output_parameters = parameter_groups["output"]
    all_context_parameters = tuple(adapter.parameters())
    context_parameter_count = sum(
        int(parameter.numel()) for parameter in all_context_parameters)
    hidden_nonzero = sum(
        int(torch.count_nonzero(parameter).item())
        for parameter in hidden_parameters
    )
    output_nonzero = sum(
        int(torch.count_nonzero(parameter).item())
        for parameter in output_parameters
    )
    encoder_nonzero = sum(
        int(torch.count_nonzero(parameter).item())
        for parameter in parameter_groups["encoder"]
    )
    interaction_nonzero = sum(
        int(torch.count_nonzero(parameter).item())
        for parameter in parameter_groups["interaction"]
    )
    _require(
        not extractor.actor_context_enabled
        and len(all_context_parameters) > 0
        and context_parameter_count > 0
        and encoder_nonzero > 0
        and interaction_nonzero > 0
        and hidden_nonzero == encoder_nonzero + interaction_nonzero
        and output_nonzero == 0
        and all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in (*hidden_parameters, *output_parameters)
        ),
        "dual-v4 actor context 起点必须关闭、hidden canonical nonzero、"
        "output canonical zero",
    )
    for key in _POLICY_HEAD_KEYS:
        _require(torch.equal(target[key], source[key]),
                 f"dual-v4 actor 移植后不逐位相同:{key}")
    # Tensor equality alone does not prove semantic equality if activation or
    # preprocessing topology drifted.  Run both graphs on a deterministic
    # nontrivial batch and require exact raw-logit identity before signing the
    # migration receipt.
    probe_rows = 4
    probe_legacy = torch.linspace(
        -1.0,
        1.0,
        steps=probe_rows * 298,
        dtype=source[_POLICY_HEAD_KEYS[0]].dtype,
    ).reshape(probe_rows, 298)
    with torch.no_grad():
        source_features = source_policy.extract_features(probe_legacy)
        source_hidden = source_policy.mlp_extractor.forward_actor(
            source_features)
        source_logits = source_policy.action_net(source_hidden)
        probe_dual = torch.zeros(
            (probe_rows, ASYMMETRIC_WORKER_OBSERVATION_DIM),
            dtype=probe_legacy.dtype,
            device=model.device,
        )
        probe_dual[:, :298] = probe_legacy.to(model.device)
        target_features = model.policy.extract_features(probe_dual)
        target_latent = model.policy.mlp_extractor.forward_actor(
            target_features)
        target_logits = model.policy.action_net(target_latent).cpu()
        # p_skip is a rollout sampler control, not deployable game state.
        # Prove at the adapter's nonzero pre-output (rather than at its
        # zero-initialized output) that the semantic exclusion is structural.
        exclusion_probe = torch.linspace(
            -0.75,
            0.75,
            steps=probe_rows * ASYMMETRIC_WORKER_OBSERVATION_DIM,
            dtype=probe_legacy.dtype,
            device=model.device,
        ).reshape(probe_rows, ASYMMETRIC_WORKER_OBSERVATION_DIM)
        exclusion_probe[:, :298] = probe_legacy.to(model.device)
        exclusion_zero = exclusion_probe.clone()
        exclusion_one = exclusion_probe.clone()
        exclusion_index = DUAL_WORKER_LAYOUT.p_skip_semantic_index
        exclusion_zero[:, exclusion_index] = 0.0
        exclusion_one[:, exclusion_index] = 1.0
        exclusion_legacy = extractor.policy_net(
            exclusion_probe[:, :298])
        excluded_preoutput_equal = torch.equal(
            adapter.preoutput(exclusion_zero, exclusion_legacy),
            adapter.preoutput(exclusion_one, exclusion_legacy),
        )
    _require(
        torch.equal(source_logits, target_logits)
        and excluded_preoutput_equal,
        "dual-v4 actor 移植 logits 或 p_skip 结构排除不逐位成立",
    )
    probe_array = source_logits.contiguous().numpy()
    probe_digest = hashlib.sha256()
    probe_digest.update(str(probe_array.dtype).encode("ascii"))
    probe_digest.update(
        np.asarray(probe_array.shape, dtype=np.int64).tobytes())
    probe_digest.update(probe_array.tobytes())
    partition = strict_actor_critic_parameter_partition(
        model.policy, optimizer=model.policy.optimizer)
    context_names = {
        id(parameter):
            f"mlp_extractor.context_adapter.{name}"
        for name, parameter in adapter.named_parameters()
    }
    context_keys = tuple(context_names[id(parameter)]
                         for parameter in all_context_parameters)
    context_encoder_keys = tuple(
        context_names[id(parameter)]
        for parameter in parameter_groups["encoder"])
    context_interaction_keys = tuple(
        context_names[id(parameter)]
        for parameter in parameter_groups["interaction"])
    context_output_keys = tuple(
        context_names[id(parameter)]
        for parameter in parameter_groups["output"])
    receipt = {
        "schema": _ASYMMETRIC_ACTOR_INIT_SCHEMA,
        "method": _ASYMMETRIC_ACTOR_INIT_METHOD,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_actor_sha256": source_actor_sha256,
        "source_critic_sha256": source_critic_sha256,
        "source_policy_class": "MaskableActorCriticPolicy",
        "source_policy_observation_shape": [298],
        "migrated_actor_sha256": actor_parameter_sha256(
            model.policy, optimizer=model.policy.optimizer),
        "source_actor_parameter_tensors": len(_POLICY_HEAD_KEYS),
        "target_actor_parameter_tensors": len(partition["actor"]),
        "target_actor_parameter_count": sum(
            int(parameter.numel()) for parameter in partition["actor"]),
        "context_parameter_tensors": len(all_context_parameters),
        "context_parameter_count": context_parameter_count,
        "context_enabled": extractor.actor_context_enabled,
        "context_architecture": _ASYMMETRIC_CONTEXT_ARCHITECTURE,
        "context_initializer": ASYMMETRIC_WORKER_CONTEXT_INITIALIZER,
        "controller_layout_schema": DUAL_WORKER_LAYOUT.schema,
        "controller_layout_sha256": DUAL_WORKER_LAYOUT_SHA256,
        "context_sha256": _stable_named_tensor_sha256(
            model.policy.state_dict(), context_keys),
        "context_encoder_sha256": _stable_named_tensor_sha256(
            model.policy.state_dict(), context_encoder_keys),
        "context_interaction_sha256": _stable_named_tensor_sha256(
            model.policy.state_dict(), context_interaction_keys),
        "context_output_sha256": _stable_named_tensor_sha256(
            model.policy.state_dict(), context_output_keys),
        "context_hidden_nonzero": hidden_nonzero,
        "context_output_nonzero": output_nonzero,
        "context_excluded_preoutput_bitwise_equal":
            bool(excluded_preoutput_equal),
        "actor_context_excluded_observation_features": list(
            ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES),
        "bitwise_probe_rows": probe_rows,
        "bitwise_probe_sha256": probe_digest.hexdigest(),
    }
    model._actor_migration_receipt = dict(receipt)
    return receipt


def _reset_worker_critic(
        model, *, training_seed: int,
        source_checkpoint_sha256: str,
        source_actor_sha256: str | None = None,
        source_critic_sha256: str | None = None) -> dict:
    """Reinitialize only the value branch for the full-game Worker target.

    V28's critic learned on FARM-window pseudo terminals.  The current Worker
    bootstraps across those windows and terminates only with the underlying
    game, so retaining that value function is a target mismatch rather than a
    continuation.  Preserve the migrated actor bit-for-bit, initialize the
    independent structured critic with SB3's native orthogonal gains under an
    isolated deterministic RNG stream, and return a layout/capacity receipt.
    """
    import torch
    from leashed_ppo import (
        actor_parameter_sha256,
        critic_parameter_sha256,
        strict_actor_critic_parameter_partition,
    )
    from diablogym.controller_wire import (
        DUAL_WORKER_LAYOUT,
        DUAL_WORKER_LAYOUT_SHA256,
    )

    _require(
        isinstance(training_seed, int) and not isinstance(training_seed, bool)
        and 0 <= training_seed < 2**32,
        "fresh Worker critic 要求显式 uint32 训练 seed",
    )
    _require(_is_sha256(source_checkpoint_sha256),
             "fresh Worker critic 缺可信 source checkpoint SHA256")
    _require(str(getattr(model, "device", "")) == "cpu",
             "fresh Worker critic 迁移当前只允许 CPU，以保证可复现初始化")
    policy = model.policy
    partition = strict_actor_critic_parameter_partition(
        policy, optimizer=policy.optimizer)
    actor_before = actor_parameter_sha256(
        policy, optimizer=policy.optimizer)
    critic_before = critic_parameter_sha256(
        policy, optimizer=policy.optimizer)
    if source_actor_sha256 is None:
        source_actor_sha256 = actor_before
    if source_critic_sha256 is None:
        source_critic_sha256 = critic_before
    _require(_is_sha256(source_actor_sha256)
             and _is_sha256(source_critic_sha256),
             "fresh critic 缺 source actor/critic SHA")
    rng_before = torch.random.get_rng_state().clone()
    init_seed = int.from_bytes(
        hashlib.sha256(
            b"diablogym/full-game-worker-critic-v1\0"
            + str(training_seed).encode("ascii")
        ).digest()[:8],
        byteorder="big",
    ) & ((1 << 63) - 1)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(init_seed)
        policy.mlp_extractor.value_net.apply(
            functools.partial(policy.init_weights, gain=math.sqrt(2.0)))
        policy.value_net.apply(
            functools.partial(policy.init_weights, gain=1.0))
    _require(torch.equal(torch.random.get_rng_state(), rng_before),
             "fresh critic 初始化污染全局 Torch RNG")
    actor_after = actor_parameter_sha256(
        policy, optimizer=policy.optimizer)
    critic_after = critic_parameter_sha256(
        policy, optimizer=policy.optimizer)
    _require(actor_after == actor_before,
             "fresh critic 初始化改写冻结 V28 actor")
    _require(critic_after != critic_before,
             "fresh critic 初始化后摘要未变化")
    _require(all(
        bool(parameter.detach().isfinite().all().item())
        for parameter in partition["critic"]
    ), "fresh critic 初始化产生 NaN/Inf")
    critic_parameter_count = sum(
        int(parameter.numel()) for parameter in partition["critic"])
    _require(
        len(partition["actor"]) > 0
        and len(partition["critic"]) > 0
        and critic_parameter_count > 0,
        "fresh structured critic 容量/张量数漂移",
    )
    receipt = {
        "schema": _ASYMMETRIC_CRITIC_RESET_SCHEMA,
        "method": _ASYMMETRIC_CRITIC_RESET_METHOD,
        "critic_architecture": _ASYMMETRIC_CRITIC_ARCHITECTURE,
        "controller_layout_schema": DUAL_WORKER_LAYOUT.schema,
        "controller_layout_sha256": DUAL_WORKER_LAYOUT_SHA256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_actor_sha256": source_actor_sha256,
        "source_critic_sha256": source_critic_sha256,
        "training_seed": training_seed,
        "init_seed": init_seed,
        "actor_sha256_before": actor_before,
        "actor_sha256_after": actor_after,
        "critic_sha256_before": critic_before,
        "critic_sha256_after": critic_after,
        "actor_parameter_tensors": len(partition["actor"]),
        "critic_parameter_tensors": len(partition["critic"]),
        "critic_parameter_count": critic_parameter_count,
    }
    model._critic_reset_receipt = dict(receipt)
    return receipt


def _canonical_asymmetric_worker_migration_evidence(
        *, source_checkpoint_payload: bytes,
        source_checkpoint_sha256: str,
        training_seed: int) -> dict:
    """Reconstruct the canonical actor migration and critic reset receipts.

    R7 consumes this independently reconstructed evidence rather than trusting
    the initial sub-hashes embedded by the artifact under inspection.  The
    reconstruction uses no environment rollout and restores every global RNG
    stream touched by policy/source-model construction.
    """
    import numpy as np
    import torch
    from leashed_ppo import asymmetric_worker_runtime_evidence

    _require(
        isinstance(source_checkpoint_payload, bytes)
        and _is_sha256(source_checkpoint_sha256)
        and hashlib.sha256(source_checkpoint_payload).hexdigest()
        == source_checkpoint_sha256,
        "canonical asymmetric evidence source payload/SHA 不一致",
    )
    _require(
        _is_plain_int(training_seed) and 0 <= training_seed < 2**32,
        "canonical asymmetric evidence 要求 uint32 training_seed",
    )
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state().clone()
    try:
        # Reproduce SB3's target-policy construction seed so even the
        # pre-reset critic digest is independently reconstructible.  The
        # migrated actor and reset critic that survive this helper remain
        # defined by their dedicated canonical paths.
        random.seed(training_seed)
        np.random.seed(training_seed)
        torch.manual_seed(training_seed)
        target = types.SimpleNamespace(
            policy=_new_current_asymmetric_worker_policy())
        target.device = target.policy.device
        actor_receipt = _initialize_asymmetric_worker_actor(
            target,
            source_checkpoint_payload=source_checkpoint_payload,
            source_checkpoint_sha256=source_checkpoint_sha256,
        )
        critic_receipt = _reset_worker_critic(
            target,
            training_seed=training_seed,
            source_checkpoint_sha256=source_checkpoint_sha256,
            source_actor_sha256=actor_receipt["source_actor_sha256"],
            source_critic_sha256=actor_receipt["source_critic_sha256"],
        )
        runtime = asymmetric_worker_runtime_evidence(target.policy)
        _require(
            runtime["policy"]["actor_sha256"]
            == actor_receipt["migrated_actor_sha256"]
            and runtime["policy"]["critic_sha256"]
            == critic_receipt["critic_sha256_after"]
            and runtime["policy"]["actor_tensor_count"]
            == actor_receipt["target_actor_parameter_tensors"]
            and runtime["policy"]["actor_parameter_count"]
            == actor_receipt["target_actor_parameter_count"]
            and runtime["context"]["tensor_count"]
            == actor_receipt["context_parameter_tensors"]
            and runtime["context"]["parameter_count"]
            == actor_receipt["context_parameter_count"]
            and runtime["policy"]["critic_tensor_count"]
            == critic_receipt["critic_parameter_tensors"]
            and runtime["policy"]["critic_parameter_count"]
            == critic_receipt["critic_parameter_count"],
            "canonical asymmetric migration/reset/runtime evidence 未闭合",
        )
        return {
            "schema": _ASYMMETRIC_CANONICAL_EVIDENCE_SCHEMA,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "training_seed": training_seed,
            "actor_migration": copy.deepcopy(actor_receipt),
            "critic_reset": copy.deepcopy(critic_receipt),
            "runtime": copy.deepcopy(runtime),
        }
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)


def _policy_source_roles(args, demos_sha256: str | None) -> dict:
    """Describe which frozen artifacts can actually change policy parameters."""
    resume = bool(getattr(args, "resume_from", None))
    bc_init = bool(getattr(args, "bc_init", None))
    beta = float(getattr(args, "distill_beta", 0.0))
    teacher_override = bool(getattr(args, "teacher_override", None))
    worker = bool(getattr(args, "worker", False))
    dual = (
        worker
        and _worker_policy_observation_view(args)
        == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
    )
    action14_bonus = float(getattr(
        args, "worker_action14_logit_bonus", 0.0))

    direct_bc_uses = []
    if bc_init:
        direct_bc_uses.append("policy-initialization")
    if worker and beta > 0.0 and not teacher_override:
        direct_bc_uses.append("distillation-teacher")

    if beta == 0.0:
        distillation_teacher = "disabled"
    elif teacher_override:
        distillation_teacher = "teacher-override"
    elif worker:
        distillation_teacher = "configured-bc-v1-export"
    else:
        distillation_teacher = "configured-teacher"

    action14_sources = []
    if worker:
        if bc_init:
            action14_sources.append("bc-v1-policy-initialization")
        if dual and action14_bonus != 0.0:
            action14_sources.append("fixed-logit-prior")
        action14_sources.append("native-reward-bound-on-policy-ppo")

    roles = {
        "schema": _POLICY_SOURCE_ROLES_SCHEMA,
        "initialization": (
            "resume-checkpoint"
            if resume else
            "bc-v1-policy-initialization"
            if bc_init else
            "random-initialization"
        ),
        "bc_v1_direct_policy_uses": direct_bc_uses,
        "bc_v1_dataset_uses": (
            ["pass-gate", "read-only-dry-anchor-instrumentation"]
            if demos_sha256 is not None else []
        ),
        "distillation_teacher": distillation_teacher,
        "worker_action14_policy_sources": action14_sources,
    }
    return roles


def _validate_policy_source_roles(contract: dict) -> dict:
    """Fail closed when provenance labels claim a policy path that is absent."""
    _require(isinstance(contract, dict),
             "policy source roles 要求 training contract 对象")
    roles = contract.get("policy_source_roles")
    _require(
        isinstance(roles, dict)
        and set(roles) == {
            "schema",
            "initialization",
            "bc_v1_direct_policy_uses",
            "bc_v1_dataset_uses",
            "distillation_teacher",
            "worker_action14_policy_sources",
        }
        and roles.get("schema") == _POLICY_SOURCE_ROLES_SCHEMA,
        "policy source roles schema/字段不精确",
    )
    initialization = roles["initialization"]
    direct = roles["bc_v1_direct_policy_uses"]
    dataset = roles["bc_v1_dataset_uses"]
    teacher = roles["distillation_teacher"]
    action14 = roles["worker_action14_policy_sources"]
    _require(
        initialization in {
            "resume-checkpoint",
            "bc-v1-policy-initialization",
            "random-initialization",
        }
        and isinstance(direct, list)
        and len(direct) == len(set(direct))
        and set(direct).issubset({
            "policy-initialization", "distillation-teacher"})
        and (
            ("policy-initialization" in direct)
            is (initialization == "bc-v1-policy-initialization")
        )
        and (
            contract.get("demos_sha256") is None
            or _is_sha256(contract.get("demos_sha256"))
        )
        and dataset == (
            ["pass-gate", "read-only-dry-anchor-instrumentation"]
            if contract.get("demos_sha256") is not None else []
        ),
        "BC-v1 policy/data role 与初始化或 demos 绑定不一致",
    )
    beta = contract.get("distill_beta")
    _require(
        isinstance(beta, (int, float))
        and not isinstance(beta, bool)
        and math.isfinite(float(beta))
        and float(beta) >= 0.0,
        "policy source roles 缺有限非负 distill_beta",
    )
    if float(beta) == 0.0:
        _require(
            teacher == "disabled"
            and "distillation-teacher" not in direct,
            "β=0 却宣称存在 distillation policy path",
        )
    else:
        _require(
            teacher in {
                "teacher-override",
                "configured-bc-v1-export",
                "configured-teacher",
            }
            and (
                ("distillation-teacher" in direct)
                is (teacher == "configured-bc-v1-export")
            ),
            "distillation teacher 与 BC-v1 direct role 不闭合",
        )
        _require(
            _is_sha256(contract.get("teacher_sha256")),
            "启用 distillation 却缺少有效 teacher_sha256",
        )

    mode = contract.get("mode")
    bonus = contract.get("worker_action14_logit_bonus")
    _require(
        isinstance(action14, list)
        and len(action14) == len(set(action14)),
        "worker a14 policy sources 必须是无重复列表",
    )
    expected_action14 = []
    if mode == "worker":
        if initialization == "bc-v1-policy-initialization":
            expected_action14.append("bc-v1-policy-initialization")
        if (
            contract.get("worker_policy_observation_view")
            == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
            and isinstance(bonus, (int, float))
            and not isinstance(bonus, bool)
            and math.isfinite(float(bonus))
            and float(bonus) != 0.0
        ):
            expected_action14.append("fixed-logit-prior")
        expected_action14.append(
            "native-reward-bound-on-policy-ppo")
    _require(
        action14 == expected_action14,
        "worker a14 policy sources 与真实初始化/prior/PPO 路径不一致",
    )
    if (
        mode == "worker"
        and contract.get("worker_policy_observation_view")
        == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
    ):
        _require(
            initialization == "resume-checkpoint"
            and isinstance(contract.get("actor_migration"), dict)
            and contract["actor_migration"].get("method")
            == _ASYMMETRIC_ACTOR_INIT_METHOD,
            "dual-v4 actor migration 必须如实登记为 resume-checkpoint",
        )
    return roles


def _training_contract(args, model, batch_size: int,
                       manager_npz_sha256: str | None = None,
                       worker_npz_sha256: str | None = None,
                       demos_sha256: str | None = None,
                       implementation_sha256: str | None = None,
                       bc_aux_demos_sha256: str | None = None) -> dict:
    mode = "worker" if args.worker else "options" if args.options else \
        "flat_clock" if args.flat_clock else "flat"
    manager_policy_view = (
        "legacy-v3" if args.worker
        else getattr(args, "manager_policy_observation_view", "raw-v4")
        if args.options
        else None
    )
    gradient_clip_mode = getattr(args, "gradient_clip_mode", "global")
    gradient_clipping = (
        _root_context_critic_gradient_clipping(model.max_grad_norm)
        if gradient_clip_mode == "separate-root-context-critic-v2"
        else
        {
            "mode": "separate-actor-critic-v1",
            "actor_max_norm": float(model.max_grad_norm),
            "critic_max_norm": float(model.max_grad_norm),
            "optimizer": "single",
            "trainable_shared_parameters": "forbidden",
        }
        if gradient_clip_mode == "separate-actor-critic-v1"
        else {
            "mode": "global",
            "max_norm": float(model.max_grad_norm),
        }
    )
    critic_receipt = getattr(
        model, "_critic_migration_receipt", None)
    actor_receipt = getattr(model, "_actor_migration_receipt", None)
    asymmetric_runtime = None
    if critic_receipt is not None or actor_receipt is not None:
        from leashed_ppo import asymmetric_worker_runtime_evidence
        asymmetric_runtime = asymmetric_worker_runtime_evidence(
            model.policy)
    if critic_receipt is None:
        critic_migration = "disabled"
    else:
        from leashed_ppo import (
            WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
            WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
        )
        from diablogym.controller_wire import (
            DUAL_WORKER_LAYOUT,
            DUAL_WORKER_LAYOUT_SHA256,
        )

        _require(
            isinstance(critic_receipt, dict)
            and critic_receipt.get("schema")
            == _ASYMMETRIC_CRITIC_RESET_SCHEMA
            and critic_receipt.get("method")
            == _ASYMMETRIC_CRITIC_RESET_METHOD
            and critic_receipt.get("critic_architecture")
            == _ASYMMETRIC_CRITIC_ARCHITECTURE
            and critic_receipt.get("controller_layout_schema")
            == DUAL_WORKER_LAYOUT.schema
            and critic_receipt.get("controller_layout_sha256")
            == DUAL_WORKER_LAYOUT_SHA256
            and asymmetric_runtime is not None
            and critic_receipt.get("actor_parameter_tensors")
            == asymmetric_runtime["policy"]["actor_tensor_count"]
            and critic_receipt.get("critic_parameter_tensors")
            == asymmetric_runtime["policy"]["critic_tensor_count"]
            and critic_receipt.get("critic_parameter_count")
            == asymmetric_runtime["policy"]["critic_parameter_count"]
            and _is_sha256(
                critic_receipt.get("source_checkpoint_sha256"))
            and isinstance(critic_receipt.get("warmup_steps"), int)
            and critic_receipt["warmup_steps"] > 0
            and critic_receipt.get("gradient_clip_mode")
            in {
                "separate-actor-critic-v1",
                "separate-root-context-critic-v2",
            }
            and _is_sha256(critic_receipt.get("source_actor_sha256"))
            and critic_receipt.get(
                "worker_onpolicy_pg_audit_schema")
            == WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
            "模型 critic migration receipt 非法，拒绝写训练契约",
        )
        critic_migration = {
            "method": critic_receipt["method"],
            "critic_architecture":
                critic_receipt["critic_architecture"],
            "controller_layout_schema":
                critic_receipt["controller_layout_schema"],
            "controller_layout_sha256":
                critic_receipt["controller_layout_sha256"],
            "critic_parameter_tensors":
                critic_receipt["critic_parameter_tensors"],
            "critic_parameter_count":
                critic_receipt["critic_parameter_count"],
            "source_checkpoint_sha256":
                critic_receipt["source_checkpoint_sha256"],
            "warmup_steps": critic_receipt["warmup_steps"],
            "gradient_clip_mode":
                critic_receipt["gradient_clip_mode"],
            "source_actor_sha256":
                critic_receipt["source_actor_sha256"],
            "worker_onpolicy_pg_audit_schema":
                critic_receipt[
                    "worker_onpolicy_pg_audit_schema"],
            "worker_onpolicy_pg_min_optimizer_steps_per_joint_rollout":
                WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
        }
    if actor_receipt is None:
        actor_migration = "disabled"
    else:
        from leashed_ppo import (
            ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES,
            ASYMMETRIC_WORKER_CONTEXT_INITIALIZER,
        )
        from diablogym.controller_wire import (
            DUAL_WORKER_LAYOUT,
            DUAL_WORKER_LAYOUT_SHA256,
        )
        _require(
            isinstance(actor_receipt, dict)
            and actor_receipt.get("schema")
            == _ASYMMETRIC_ACTOR_INIT_SCHEMA
            and actor_receipt.get("method")
            == _ASYMMETRIC_ACTOR_INIT_METHOD
            and _is_sha256(
                actor_receipt.get("source_checkpoint_sha256"))
            and _is_sha256(actor_receipt.get("source_actor_sha256"))
            and _is_sha256(actor_receipt.get("migrated_actor_sha256"))
            and _is_sha256(actor_receipt.get("context_sha256"))
            and _is_sha256(
                actor_receipt.get("context_encoder_sha256"))
            and _is_sha256(
                actor_receipt.get("context_interaction_sha256"))
            and _is_sha256(
                actor_receipt.get("context_output_sha256"))
            and actor_receipt.get("source_policy_class")
            == "MaskableActorCriticPolicy"
            and actor_receipt.get("source_policy_observation_shape")
            == [298]
            and actor_receipt.get("source_actor_parameter_tensors") == 6
            and asymmetric_runtime is not None
            and actor_receipt.get("target_actor_parameter_tensors")
            == asymmetric_runtime["policy"]["actor_tensor_count"]
            and actor_receipt.get("target_actor_parameter_count")
            == asymmetric_runtime["policy"]["actor_parameter_count"]
            and actor_receipt.get("context_parameter_tensors")
            == asymmetric_runtime["context"]["tensor_count"]
            and actor_receipt.get("context_parameter_count")
            == asymmetric_runtime["context"]["parameter_count"]
            and actor_receipt.get("context_enabled") is False
            and actor_receipt.get("context_architecture")
            == _ASYMMETRIC_CONTEXT_ARCHITECTURE
            and actor_receipt.get("context_initializer")
            == ASYMMETRIC_WORKER_CONTEXT_INITIALIZER
            and actor_receipt.get("controller_layout_schema")
            == DUAL_WORKER_LAYOUT.schema
            and actor_receipt.get("controller_layout_sha256")
            == DUAL_WORKER_LAYOUT_SHA256
            and _is_plain_int(
                actor_receipt.get("context_hidden_nonzero"))
            and actor_receipt["context_hidden_nonzero"] > 0
            and actor_receipt.get("context_output_nonzero") == 0
            and actor_receipt.get(
                "context_excluded_preoutput_bitwise_equal") is True
            and actor_receipt.get(
                "actor_context_excluded_observation_features")
            == list(ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES)
            and actor_receipt.get("bitwise_probe_rows") == 4
            and _is_sha256(
                actor_receipt.get("bitwise_probe_sha256")),
            "模型 asymmetric actor migration receipt 非法，拒绝写训练契约",
        )
        actor_migration = {
            "method": actor_receipt["method"],
            "source_checkpoint_sha256":
                actor_receipt["source_checkpoint_sha256"],
            "source_actor_sha256":
                actor_receipt["source_actor_sha256"],
            "migrated_actor_sha256":
                actor_receipt["migrated_actor_sha256"],
            "context_architecture":
                actor_receipt["context_architecture"],
            "controller_layout_schema":
                actor_receipt["controller_layout_schema"],
            "controller_layout_sha256":
                actor_receipt["controller_layout_sha256"],
            "target_actor_parameter_tensors":
                actor_receipt["target_actor_parameter_tensors"],
            "target_actor_parameter_count":
                actor_receipt["target_actor_parameter_count"],
            "context_parameter_tensors":
                actor_receipt["context_parameter_tensors"],
            "context_parameter_count":
                actor_receipt["context_parameter_count"],
            "context_initialization": {
                "hidden": actor_receipt["context_initializer"],
                "output": "exact-zero-disabled-through-critic-warmup",
            },
            "actor_context_excluded_observation_features":
                actor_receipt[
                    "actor_context_excluded_observation_features"],
        }
    action_count = getattr(model.action_space, "n", None)
    from leashed_ppo import LEGACY_DISTILLATION_EXCLUDED_ACTIONS
    contract = {
        "schema_version": 2,
        "contract_revision": _CONTRACT_REVISION,   # v32:+drink_sovereignty(④丙 环境语义入契约)
        "implementation_sha256": implementation_sha256,
        "mode": mode,
        "arch": args.arch,
        "max_steps": args.max_steps,
        "num_envs": args.num_envs,
        "n_steps": args.n_steps,
        "batch_size": batch_size,
        "gamma": args.gamma,
        "learning_rate": args.lr,
        "ent_coef": args.ent_coef,
        "distill_beta": float(args.distill_beta),
        "distillation": {
            "initial_beta": float(args.distill_beta),
            "scope": (
                "legacy-root-logits"
                if (
                    args.worker
                    and _worker_policy_observation_view(args)
                    == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
                    and float(args.distill_beta) > 0.0
                )
                else "full-policy-logits"
            ),
            "excluded_actions": (
                list(LEGACY_DISTILLATION_EXCLUDED_ACTIONS)
                if float(args.distill_beta) > 0.0 else []
            ),
            "anneal_actor_rollouts": int(getattr(
                args, "distill_anneal_actor_rollouts", 0)),
            "schedule": (
                "linear-inclusive-zero-v1"
                if int(getattr(
                    args,
                    "distill_anneal_actor_rollouts",
                    0)) > 0
                else "constant"
            ),
        },
        "teacher_sha256": getattr(model, "teacher_sha256", None),
        "calib_record_only": bool(args.calib_record_only),
        "device": str(model.device),
        "skip_dry": bool(args.skip_dry),
        "drink_sovereignty": _effective_drink_sovereignty(args),
        "worker_fast_forward_reward_credit": getattr(
            args, "worker_fast_forward_reward_credit", "none"),
        "worker_additional_terminal_death_cost":
            float(getattr(
                args, "worker_additional_terminal_death_cost", 0.0)),
        "legacy_policy_observation_view": (
            _worker_policy_observation_view(args)
            == _WORKER_VIEW_LEGACY_V3),
        "worker_policy_observation_view":
            _worker_policy_observation_view(args),
        "worker_action14_logit_bonus": float(getattr(
            args, "worker_action14_logit_bonus", 0.0)),
        "manager_policy_observation_view": manager_policy_view,
        "worker_episode_boundary": (
            _WORKER_EPISODE_BOUNDARY_V24 if args.worker else None),
        "worker_window_bootstrap": (
            "next-learning-window" if args.worker else None),
        "worker_no_progress_timeout": (
            dict(_WORKER_NO_PROGRESS_TIMEOUT_CONTRACT)
            if args.worker else None
        ),
        "gradient_clipping": gradient_clipping,
        "actor_migration": actor_migration,
        "critic_migration": critic_migration,
        "artifact_scope": getattr(args, "artifact_scope", "production"),
        # E4 rev5 双键(圈 7,三腿统一):disabled 或实况载荷;skip_dry 键
        # 保持 CLI 旗字面值不受此二键影响(rev3 勘正,契约与回执同构)。
        "dry_curriculum": _contract_dry_curriculum(args),
        "bc_aux": _contract_bc_aux(args, bc_aux_demos_sha256),
        "manager_npz_sha256": manager_npz_sha256,
        "worker_npz_sha256": worker_npz_sha256,
        "demos_sha256": demos_sha256,
        "policy_source_roles":
            _policy_source_roles(args, demos_sha256),
        # Gymnasium exposes Discrete.n (and on some versions shape entries) as
        # NumPy integer scalars.  Normalize before this contract is embedded in
        # status.json/SB3 data; stdlib json intentionally cannot encode np.int64.
        "observation_shape": [int(value) for value in model.observation_space.shape],
        "action_n": None if action_count is None else int(action_count),
        "runtime_versions": dict(_RUNTIME_VERSIONS),
        "algorithm_recipe": {
            **_ALGORITHM_RECIPE,
            "target_kl": getattr(args, "target_kl", None),
        },
    }
    _validate_policy_source_roles(contract)
    if (
        args.worker
        and _worker_policy_observation_view(args)
        == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
    ):
        _validate_current_dual_worker_contract(
            contract, runtime_evidence=asymmetric_runtime)
    return contract


def _validate_resume_contract(saved: dict | None, current: dict,
                              allow_manager_change: bool = False,
                              allow_legacy_resume: bool = False,
                              allow_optimizer_reset: bool = False,
                              allow_target_kl_change: bool = False) -> None:
    if saved is None:
        _require(allow_legacy_resume,
                 "resume checkpoint 无 training_contract，无法证明原训练环境/资源；"
                 "如确需一次性迁移，请显式传 --allow-legacy-resume")
        print("   [legacy migration] 已显式允许无契约 checkpoint；"
              f"本腿将写入 contract_revision {_CONTRACT_REVISION} 契约")
        return
    _require(isinstance(saved, dict), "checkpoint training_contract 不是对象")
    allowed = {"manager_npz_sha256"} if allow_manager_change else set()
    if allow_optimizer_reset:
        # reset 明确切断旧 Adam moments；学习率与 reset 回执因此属于本腿
        # 新 optimizer 身份，不应拿上一腿契约阻止该显式迁移。reset 本身是
        # 本腿事件，只进 config/receipt，不进持久 resume equality。
        allowed.add("learning_rate")
    saved_recipe = saved.get("algorithm_recipe")
    current_recipe = current.get("algorithm_recipe")
    if allow_target_kl_change and isinstance(saved_recipe, dict) \
            and isinstance(current_recipe, dict):
        saved_without = {**saved_recipe, "target_kl": current_recipe.get("target_kl")}
        if saved_without == current_recipe:
            allowed.add("algorithm_recipe")
    differences = {key: (saved.get(key), current.get(key))
                   for key in sorted(set(saved) | set(current))
                   if key not in allowed and saved.get(key) != current.get(key)}
    _require(not differences, f"resume 训练/环境契约漂移: {differences}")


# ---- E1 ⑤A 干窗课程(PREREG-内容案 E1,rev3 核认勘正) ----

# 腿相对锚定常量:腿起点恒 = 王 zip 终点 3,497,984(P3 复点火恒自王 zip);
# 全局步锚定禁用——p 表序号 = (num_timesteps − 3,497,984) / 2048。
_DRY_CURRICULUM_LEG_START = 3_497_984
# 主表(批文即定,圈 2 附裁):前 147×2048=301,056 步线性 1.0→0.5,
# 后 97×2048=198,656 步持 0.5;147+97=244 量子恰等腿长 499,712。
_DRY_CURRICULUM_MAIN_TABLE = "linear:1.0:0.5:147,hold:0.5:97"


def _dry_window_mechanism_active(args) -> bool:
    """E1 波及面谓词(rev3 勘正,四门统一):干窗机制在位 = skip_dry ∨ schedule。"""
    return bool(args.skip_dry) or bool(args.dry_curriculum_schedule)


def _mount_dry_anchor_sentinel(args) -> bool:
    """E1 四门之 dry_cb 挂载门(原 :2101-2104 谓词):worker ∧ 干窗机制在位。"""
    return bool(args.worker) and _dry_window_mechanism_active(args)


def _precheck_dry_window_demos(args) -> None:
    """E1 四门之 demos/BC 预检门(原 :499-505):谓词改写为机制在位,断言原封。"""
    if not _dry_window_mechanism_active(args):
        return
    demos = pathlib.Path(__file__).resolve().parent / "runs" / "bc-worker" / "demos.npz"
    _require(demos.is_file(),
             f"干窗机制(--skip-dry/--dry-curriculum-schedule)所需示范集不存在: {demos}")
    # v4:探针集绑定当前严格 PASS 回执；旧 v3 protocol/implementation
    # 即使字节仍等历史常量也必须拒绝。
    _assert_bc_v1_demos_frozen(demos)
    policy = demos.with_name("policy_sd.pt")
    _require(policy.is_file(),
             f"干窗机制(--skip-dry/--dry-curriculum-schedule)所需 BC 权重不存在: {policy}")
    report = _validate_bc_report(policy, "data_gate")
    _load_dry_anchor_demos(demos, report.get("demos_sha256"))


def _capture_dry_window_demos_sha256(args) -> str | None:
    """E1 四门之 demos_sha256 捕获门(原 :1766-1771):谓词改写,路径与断言原封。"""
    if not _dry_window_mechanism_active(args):
        return None
    demos = pathlib.Path(__file__).resolve().parent / "runs" / "bc-worker" / "demos.npz"
    # v4:从当前严格 PASS 回执捕获，不信历史 v3 冻结常量。
    _assert_bc_v1_demos_frozen(demos)
    report = _validate_bc_report(demos.with_name("policy_sd.pt"), "data_gate")
    _, _, demos_sha256 = _load_dry_anchor_demos(demos, report.get("demos_sha256"))
    return demos_sha256


def _parse_dry_curriculum_schedule(spec: str) -> tuple[float, ...]:
    """解析 --dry-curriculum-schedule 为逐 rollout"序号→p"全表。

    语法(逗号分隔段,段内冒号分隔):
      linear:<p0>:<p1>:<n> —— n(≥2)个 rollout 端点含线性 p0→p1,
                              第 k 项 = p0 + (p1−p0)·k/(n−1),k=0..n−1;
      hold:<p>:<n>         —— n(≥1)个 rollout 恒 p。
    全部 p 须为 [0, 1] 内有限数。主表 = linear:1.0:0.5:147,hold:0.5:97。
    """
    _require(isinstance(spec, str) and bool(spec.strip()),
             "--dry-curriculum-schedule 不能为空")
    table: list[float] = []
    for raw_segment in spec.split(","):
        segment = raw_segment.strip()
        fields = segment.split(":")
        if fields[0] == "linear":
            _require(len(fields) == 4,
                     f"--dry-curriculum-schedule 段格式应为 linear:<p0>:<p1>:<n>: {segment!r}")
            try:
                p0, p1, n = float(fields[1]), float(fields[2]), int(fields[3])
            except ValueError as exc:
                raise ValueError(
                    f"--dry-curriculum-schedule 段数值不可解析: {segment!r}") from exc
            _require(n >= 2, f"linear 段须 n≥2(单点请用 hold): {segment!r}")
            values = [p0 + (p1 - p0) * k / (n - 1) for k in range(n)]
        elif fields[0] == "hold":
            _require(len(fields) == 3,
                     f"--dry-curriculum-schedule 段格式应为 hold:<p>:<n>: {segment!r}")
            try:
                p, n = float(fields[1]), int(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"--dry-curriculum-schedule 段数值不可解析: {segment!r}") from exc
            _require(n >= 1, f"hold 段须 n≥1: {segment!r}")
            values = [p] * n
        else:
            raise ValueError(
                f"--dry-curriculum-schedule 未知段类型(只允许 linear/hold): {segment!r}")
        _require(all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values),
                 f"--dry-curriculum-schedule p 值必须在 [0, 1] 内: {segment!r}")
        table.extend(values)
    return tuple(table)


def _resolve_dry_curriculum_start(
        schedule_table,
        *,
        start_steps: int,
        rollout_quantum: int,
        total_steps: int,
        leg_start: int = _DRY_CURRICULUM_LEG_START,
) -> tuple[int, float]:
    """Resolve the probability that must be live before SB3's first reset.

    ``BaseAlgorithm._setup_learn()`` resets a newly attached VecEnv before
    callbacks receive ``on_training_start``.  A native continuation therefore
    cannot initialize every environment with table[0] and repair it later:
    that reset may already select/skip the first FARM window under the wrong
    probability.  Bind the initial environment value to the immutable
    checkpoint step and reject a table that cannot cover the remaining run.
    """
    table = tuple(float(value) for value in schedule_table)
    _require(table and all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in table
    ), "dry-curriculum 起点解析需要非空 [0,1] 有限概率表")
    for label, value in (
            ("start_steps", start_steps),
            ("rollout_quantum", rollout_quantum),
            ("total_steps", total_steps),
            ("leg_start", leg_start)):
        _require(
            _is_plain_int(value),
            f"dry-curriculum {label} 必须是普通整数",
        )
    _require(rollout_quantum > 0 and total_steps > 0,
             "dry-curriculum rollout 量子/训练步数必须为正")
    _require(total_steps % rollout_quantum == 0,
             "dry-curriculum 训练步数必须按 rollout 量子闭合")
    offset = start_steps - leg_start
    _require(
        offset >= 0 and offset % rollout_quantum == 0,
        "dry-curriculum checkpoint 起点不在腿相对 rollout 边界:"
        f"start={start_steps},leg_start={leg_start},quantum={rollout_quantum}",
    )
    index = offset // rollout_quantum
    rollout_count = total_steps // rollout_quantum
    _require(
        index < len(table)
        and index + rollout_count <= len(table),
        "dry-curriculum 表不足以覆盖 continuation:"
        f"start_index={index},rollouts={rollout_count},table={len(table)}",
    )
    return int(index), float(table[index])


# ---- E3 ④乙 辅助示范通路(PREREG-内容案 E3;两旗互不强制,零侵入条款) ----


def _bc_aux_active(args) -> bool:
    """Auxiliary path is active only with demos and an explicit mechanism.

    ``--bc-aux-graft`` is the rev9 circuit used by the rev10 objective.  The
    λ predicate is retained here only so old contracts fail with a precise
    migration error in ``_validate_args`` rather than silently becoming
    zero-intrusion.
    """
    return bool(getattr(args, "bc_aux_demos", None)) and (
        float(getattr(args, "bc_aux_lambda", 0.0)) > 0
        or bool(getattr(args, "bc_aux_graft", False)))


def _bc_aux_structural_active(args) -> bool:
    return bool(getattr(args, "bc_aux_graft", False)
                and getattr(args, "bc_aux_demos", None))


def _parse_bc_aux_demos_v2(path: str | pathlib.Path):
    """E3 ④乙:bc-worker-v2 示范集专用验证器(镜像断言按世代分别成文)。

    v1 面(_BC_REPORT_SCHEMA_VERSION=1/_validate_bc_report/_load_dry_anchor_demos/
    canonical bc-worker 路径)原封零触碰;本验证器单列,v2 demos schema 承图纸
    E2 共同真源 = v1 键(X/Y/episode_id)+ 逐样本 masks 数组(采集时
    env.action_masks() 现场捕获系唯一 on-manifold 真源,obs 反推口径禁用)。
    世代条件化禁采镜像:v2 禁 11 允 12。v1 之干态双通道饱和断言
    系 dry-anchor 探针专属,不随镜(施工注记)。返回
    (X, Y, episode_id, masks, sha256)；episode_id 保留给真实 held-out 行为门，
    禁把整池训练态冒充独立验证态。
    """
    import numpy as np

    p = pathlib.Path(path)
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"④乙 v2 示范集不可读: {p}: {exc}") from exc
    sha256 = hashlib.sha256(payload).hexdigest()
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            required = {
                "X", "Y", "episode_id", "masks",
                "schema_version", "protocol_version",
                "implementation_sha256", "generator_sha256",
                "manager_npz_sha256", "teacher_generation",
                "preventive_threshold",
            }
            _require(set(data.files) == required,
                     "④乙 v2 demos.npz schema/provenance 字段不精确匹配:"
                     f"缺={sorted(required - set(data.files))},"
                     f"多={sorted(set(data.files) - required)}")
            x, y = data["X"].copy(), data["Y"].copy()
            episode_id, masks = data["episode_id"].copy(), data["masks"].copy()
            meta = {}
            for key in required - {"X", "Y", "episode_id", "masks"}:
                value = np.asarray(data[key])
                _require(value.shape == (),
                         f"④乙 v2 provenance {key} 必须是 0-d 标量")
                meta[key] = value.item()
    except (OSError, ValueError) as exc:
        raise ValueError(f"④乙 v2 示范集不可读: {p}: {exc}") from exc
    _require(meta["schema_version"] == _BC_V2_DEMOS_SCHEMA_VERSION,
             f"④乙 v2 demos schema 过期:{meta['schema_version']!r}")
    _require(isinstance(meta["protocol_version"], (int, np.integer))
             and int(meta["protocol_version"]) == PROTOCOL_VERSION,
             "④乙 v2 demos 评测/环境协议过期:"
             f"{meta['protocol_version']!r} != {PROTOCOL_VERSION}")
    _require(meta["implementation_sha256"] == _implementation_bundle_sha256(),
             "④乙 v2 demos implementation_sha256 与当前训练实现不一致")
    generator = pathlib.Path(__file__).with_name("bc_worker.py")
    _require(meta["generator_sha256"]
             == hashlib.sha256(generator.read_bytes()).hexdigest(),
             "④乙 v2 demos generator_sha256 漂移:train/bc_worker.py")
    _require(_is_sha256(meta["manager_npz_sha256"]),
             "④乙 v2 demos manager_npz_sha256 非法")
    _require(isinstance(meta["teacher_generation"], (int, np.integer))
             and int(meta["teacher_generation"]) == _BC_V2_TEACHER_GENERATION,
             f"④乙 v2 demos teacher_generation 非 2:{meta['teacher_generation']!r}")
    _require(isinstance(meta["preventive_threshold"], (int, float,
                                                        np.integer, np.floating))
             and float(meta["preventive_threshold"])
             in _BC_V2_PREVENTIVE_THRESHOLDS,
             "④乙 v2 demos preventive_threshold 未注册:"
             f"{meta['preventive_threshold']!r}")
    _require(x.ndim == 2 and x.shape[1] == 298
             and y.ndim == 1 and len(x) == len(y),
             f"④乙 v2 数组形状异常:X={x.shape},Y={y.shape}")
    _require(x.dtype == np.float32 and np.issubdtype(y.dtype, np.integer),
             f"④乙 v2 dtype 异常:X={x.dtype},Y={y.dtype}")
    _require(episode_id.ndim == 1 and len(episode_id) == len(x)
             and np.issubdtype(episode_id.dtype, np.integer)
             and len(np.unique(episode_id)) >= 2,
             "④乙 v2 episode_id 形状/类型/独立局数异常")
    _require(masks.ndim == 2 and masks.shape == (len(x), 15)
             and masks.dtype == np.bool_,
             f"④乙 v2 masks 形状/dtype 异常:{getattr(masks, 'shape', None)},"
             f"{getattr(masks, 'dtype', None)}")
    _require(bool(((y >= 0) & (y < 15)).all()), "④乙 v2 标签越界")
    _require(not np.isin(y, _WORKER_BC_V2_FORBIDDEN_ACTIONS).any(),
             "④乙 v2 示范集含世代禁采动作 11(v2 禁 11 允 12,守卫面不弱化)")
    # 裁量强化注记:逐样本标签须为自身掩码合法位(on-manifold 真实执行拍之必然,
    # 掩位标签将使辅助 CE 取 -1e8 位);其对 12 类对蕴含图纸 m[12]=True 断言。
    _require(bool(masks[np.arange(len(y)), y].all()),
             "④乙 v2 存在标签被自身掩码禁止的示范对(on-manifold 破缺,fail-loud)")
    latch = x[:, _A12_CALIBRATION_DRINK_LATCH_FEATURE]
    _require(bool(np.isfinite(latch).all())
             and bool((((0.0 <= latch) & (latch <= 1.0))
                       | ((-2.0 <= latch) & (latch <= -1.0))).all()),
             "④乙 v2 feature297 主动饮位不在未饮[0,1]/已饮[-2,-1]编码域")
    hp = x[:, _A12_CALIBRATION_HP_FEATURE]
    threshold = float(meta["preventive_threshold"])
    visible_target = (
        (hp >= 0.5 - _A12_VISIBLE_HP_BOUNDARY_EPS)
        & (hp < threshold - _A12_VISIBLE_HP_BOUNDARY_EPS)
        & masks[:, 12]
        & (latch >= 0.0)
    )
    _require(np.array_equal(y == 12, visible_target),
             "④乙 v2 标签不等于现场可见 TeacherV2 谓词"
             "(hp/m12/主动饮位)，拒绝隐藏状态或 obs/raw 错位")
    return x, y, episode_id, masks, sha256, meta


def _load_bc_aux_demos_v2(
        path: str | pathlib.Path, *,
        expected_manager_sha256: str):
    """读取且验证已提交的 BC-v2 PASS bundle。

    demos 内嵌 metadata 只能证明“自报身份”；生成器在训练/校准失败前就
    可能写出数据。sibling PASS report 是三件套的提交标记，必须绑定当前
    demos/policy 字节、固定 384 episode 与 n12 数据门，RUNNING/FAIL/缺件
    一律拒绝进入 PPO 优化器。
    """
    from eval_contract import EvalContractError, strict_json_loads
    import numpy as np

    p = pathlib.Path(path)
    _require(_is_sha256(expected_manager_sha256),
             "④乙 v2 loader 缺本次训练 manager_npz_sha256")
    x, y, episode_id, masks, demos_sha256, meta = (
        _parse_bc_aux_demos_v2(p))
    _require(meta["manager_npz_sha256"] == expected_manager_sha256,
             "④乙 v2 demos 经理分布与本次训练 --manager-npz 不一致:"
             f"{meta['manager_npz_sha256']} != {expected_manager_sha256}")
    report_path = p.with_name("bc_report_v2.json")
    policy_path = p.with_name("policy_sd.pt")
    try:
        report = strict_json_loads(report_path.read_bytes())
    except (OSError, EvalContractError) as exc:
        raise ValueError(
            f"④乙 v2 PASS 回执缺失/不可读:{report_path}") from exc
    _require(isinstance(report, dict), "④乙 v2 PASS 回执必须是 JSON 对象")
    _require(set(report) == set(_BC_V2_PASS_KEYS),
             "④乙 v2 PASS 回执字段/schema 不精确:"
             f"missing={sorted(set(_BC_V2_PASS_KEYS) - set(report))},"
             f"extra={sorted(set(report) - set(_BC_V2_PASS_KEYS))}")
    _require(report["schema_version"] == _BC_V2_REPORT_SCHEMA_VERSION
             and report["data_gate"] == "PASS",
             "④乙 v2 sibling report 未通过当前 schema/data_gate")
    for key in (
            "protocol_version", "implementation_sha256", "generator_sha256",
            "manager_npz_sha256", "teacher_generation",
            "preventive_threshold"):
        _require(report[key] == meta[key],
                 f"④乙 v2 report/demos provenance 不一致:{key}")
    _validate_bc_final_holdout_marker(
        p.parent, 2, _BC_V2_COLLECTION_EPISODES, report)
    _require(report["demos_sha256"] == demos_sha256,
             "④乙 v2 PASS 回执未绑定现场 demos 字节")
    _require(report["pairs"] == len(y)
             and report["collection_episodes"]
             == len(_BC_V2_COLLECTION_EPISODES),
             "④乙 v2 PASS 回执 pairs/collection_episodes 不闭合")
    episodes = np.unique(episode_id)
    _require(np.array_equal(
        episodes, np.asarray(_BC_V2_COLLECTION_EPISODES, dtype=episodes.dtype)),
        "④乙 v2 demos 必须精确覆盖当前固定 episode "
        "2103000..2103383")
    _bc_v2_post_drink_coverage(x, y, episode_id, masks)
    n12 = int((y == 12).sum())
    _require(report["n12"] == n12
             and report["n12_gate_min"] == _BC_V2_N12_MIN
             and n12 >= _BC_V2_N12_MIN,
             "④乙 v2 n12 数据门与现场标签不闭合")
    heldout = _bc_v2_holdout_indices(episode_id)
    heldout_episodes = sorted(
        int(v) for v in np.unique(episode_id[heldout]))
    _require(report["held_out_pairs"] == len(heldout)
             and report["held_out_episodes"] == heldout_episodes,
             "④乙 v2 held-out episode split 与现场数据不闭合")
    gate = report["a12_behavior_gate"]
    _require(isinstance(gate, dict) and gate.get("verdict") == "PASS",
             "④乙 v2 sibling report 的 a12 行为门未 PASS")
    try:
        policy_payload = policy_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"④乙 v2 PASS 回执绑定权重缺失/不可读:{policy_path}") from exc
    _require(hashlib.sha256(policy_payload).hexdigest()
             == report["policy_sha256"],
             "④乙 v2 PASS 回执未绑定现场 policy_sd.pt 字节")

    # 哈希只能证明“这些字节被回执点名”，不能证明它们是可执行策略，更不能
    # 证明回执中的 held-out 行为读数确由这些权重产生。只使用 weights_only
    # 反序列化，并把 BC-v2 的标准六张量形制/有限性逐项钉死。
    import torch as th
    try:
        policy_sd = th.load(
            io.BytesIO(policy_payload), map_location="cpu",
            weights_only=True)
    except Exception as exc:
        raise ValueError(
            "④乙 v2 policy_sd.pt 不是安全可解析的 weights-only 权重"
        ) from exc
    expected_shapes = {
        "mlp_extractor.policy_net.0.weight": (64, x.shape[1]),
        "mlp_extractor.policy_net.0.bias": (64,),
        "mlp_extractor.policy_net.2.weight": (64, 64),
        "mlp_extractor.policy_net.2.bias": (64,),
        "action_net.weight": (15, 64),
        "action_net.bias": (15,),
    }
    _require(isinstance(policy_sd, dict)
             and set(policy_sd) == set(_POLICY_HEAD_KEYS),
             "④乙 v2 policy_sd.pt 策略头六张量键集合不精确")
    for key, shape in expected_shapes.items():
        tensor = policy_sd[key]
        _require(isinstance(tensor, th.Tensor)
                 and tensor.dtype == th.float32
                 and tuple(tensor.shape) == shape
                 and bool(th.isfinite(tensor).all()),
                 f"④乙 v2 policy_sd.pt 张量形制/有限性异常:"
                 f"{key}={getattr(tensor, 'shape', None)}/"
                 f"{getattr(tensor, 'dtype', None)}")

    _validate_bc_v2_calibration_receipt(
        report, policy_sd, x, y, episode_id, masks)

    # 哈希和 a12 安全门都不能证明其余战斗类仍在。用现场 held-out/masks
    # 重算 top1 与全部有足够覆盖的逐类召回；a12 走下方专用 0.5+安全门，
    # 其余类仍严格要求 0.85。
    with th.no_grad():
        heldout_logits = _policy_logits_from_sb3_state_dict(
            policy_sd, x[heldout])
        heldout_mask = th.as_tensor(masks[heldout], dtype=th.bool)
        heldout_pred = th.where(
            heldout_mask, heldout_logits,
            th.full_like(heldout_logits, -1e8)
        ).argmax(dim=-1).cpu().numpy()
    heldout_y = y[heldout]
    observed_top1 = float((heldout_pred == heldout_y).mean())
    _require(
        isinstance(report["held_out_top1"], (int, float))
        and not isinstance(report["held_out_top1"], bool)
        and math.isclose(
            float(report["held_out_top1"]), observed_top1,
            rel_tol=0.0, abs_tol=1e-15)
        and observed_top1 >= 0.95,
        "④乙 v2 held_out_top1 与现场权重重算不一致或未过门")
    full_counts = np.bincount(y, minlength=15)
    gated_actions = np.flatnonzero(full_counts >= 300)
    reported_recalls = report["class_recalls"]
    _require(
        isinstance(reported_recalls, dict)
        and set(reported_recalls)
        == {str(int(action)) for action in gated_actions},
        "④乙 v2 class_recalls 类集合与现场 demos 不一致")
    for action in gated_actions:
        selected = heldout_y == action
        recall = (
            float((heldout_pred[selected] == action).mean())
            if selected.any() else 0.0)
        reported = reported_recalls[str(int(action))]
        _require(
            isinstance(reported, (int, float))
            and not isinstance(reported, bool)
            and math.isclose(
                float(reported), recall,
                rel_tol=0.0, abs_tol=1e-15),
            "④乙 v2 class_recalls 与现场权重重算不一致:"
            f" action={int(action)}")
        if int(action) != 12:
            _require(
                recall >= 0.85,
                "④乙 v2 非 a12 逐类召回未过 0.85:"
                f" action={int(action)}, recall={recall}")

    observed_behavior = bc_aux_behavior_metrics(
        policy_sd, x, y, episode_id, masks, heldout_only=True)
    reported_behavior = report["a12_behavior"]

    def _metrics_match(expected, observed) -> bool:
        # JSON 中计数/字符串/None 必须逐字等；浮点容许 CPU 前向与 JSON
        # round-trip 的最后几位差异，但拒绝非有限值。
        if isinstance(expected, dict) or isinstance(observed, dict):
            return (isinstance(expected, dict)
                    and isinstance(observed, dict)
                    and set(expected) == set(observed)
                    and all(_metrics_match(expected[key], observed[key])
                            for key in expected))
        if (isinstance(expected, (int, float))
                and not isinstance(expected, bool)
                and isinstance(observed, (int, float))
                and not isinstance(observed, bool)):
            if isinstance(expected, int) and isinstance(observed, int):
                return expected == observed
            return (math.isfinite(float(expected))
                    and math.isfinite(float(observed))
                    and math.isclose(float(expected), float(observed),
                                     rel_tol=1e-7, abs_tol=1e-9))
        return type(expected) is type(observed) and expected == observed

    _require(_metrics_match(reported_behavior, observed_behavior),
             "④乙 v2 回执 a12_behavior 与现场权重重算不一致")
    # BC-v2 的教师召回由下方独立的、同源 0.5 门负责；通用安全门在
    # rev8 起与部署侧 exact-mixture 使用同一 schema，不能再暗含
    # “a12 必须成为 argmax”的旧目标，否则 producer/loader/evaluator
    # 会对同一份回执算出不同 gate。
    observed_gate = bc_aux_behavior_gate(
        observed_behavior, require_teacher_recall=False)
    _require(observed_gate["verdict"] == "PASS",
             "④乙 v2 现场权重重算 a12 行为门未 PASS:"
             f"{observed_gate['reasons']}")
    _require(
        observed_behavior["recall_12"] >= _BC_V2_TEACHER_RECALL_MIN
        and report["recall_12_denominator"]
        == observed_behavior["true_a12"]
        and math.isclose(
            float(report["recall_12"]),
            round(float(observed_behavior["recall_12"]), 4),
            rel_tol=0.0, abs_tol=1e-12),
        "④乙 v2 现场权重未过教师专用 a12 recall 0.5 门"
        "或 legacy recall 字段不同源")
    _require(report["a12_behavior_gate"] == observed_gate,
             "④乙 v2 回执 a12_behavior_gate 与现场重算不一致")
    return x, y, episode_id, masks, demos_sha256


def _filter_bc_aux_demo_pairs(x, y, masks):
    """构造 legacy 校准 bank：全部正例 + 有真实 m[12] 的 hard negatives。

    旧实现 ``keep = y == 12`` 会在没有任何反例的情况下反复最大化 π(12)，
    102,400 步 smoke 已实测坍缩为 91%“有药就喝”。修复版只消费采集时现场
    masks，负例按 8:1 封顶并同时覆盖：
      * hp>=0.65 的明确非触发态（旧模型假阳性主体）；
      * 非 a9 动作，防负例侧退化为“全部学 a9”。
    feature297<0 的后饮行现在必须同时 m12=False；它们只证明可见闩/掩码
    接口闭合，不能送进要求 m12=True 的 legacy BCE bank。
    选择为确定性等距子样，不消费训练或辅助 RNG。
    """
    import numpy as np

    x = np.asarray(x)
    y = np.asarray(y)
    masks = np.asarray(masks)
    _require(x.ndim == 2 and x.shape[1] == 298
             and y.shape == (len(x),)
             and masks.shape == (len(x), 15),
             "a12 校准 bank 输入形状异常")
    positive = np.flatnonzero(y == 12)
    _require(len(positive) > 0,
             "--bc-aux-demos 中没有 12 类示范对(fail-loud)")
    _require(bool(masks[positive, 12].all()),
             "12 类示范对存在 m[12]=False:采集面 on-manifold 破缺")
    negative = np.flatnonzero((y != 12) & masks[:, 12])
    _require(len(negative) > 0,
             "--bc-aux-demos 没有 m[12]=True 的非 12 hard negative；"
             "拒绝退回 positive-only 目标")
    _require(len(negative) >= _BC_AUX_MIN_NEGATIVE_RATIO * len(positive),
             "a12 校准 bank 合法 hard negatives 不足:"
             f"{len(negative)} < {_BC_AUX_MIN_NEGATIVE_RATIO}×"
             f"{len(positive)}；疑似主动饮可见位/安全负例覆盖不足")
    post_drink_closed = np.flatnonzero(
        (y != 12)
        & ~masks[:, 12]
        & (x[:, _A12_CALIBRATION_DRINK_LATCH_FEATURE] < 0.0))
    _require(
        len(post_drink_closed)
        >= _BC_AUX_MIN_POST_DRINK_NEGATIVE_RATIO * len(positive),
        "a12 demos 的可见后饮关闩证据不足:"
        f"{len(post_drink_closed)} < "
        f"{_BC_AUX_MIN_POST_DRINK_NEGATIVE_RATIO}×{len(positive)}；"
        "无法证明同窗重复饮已由可见状态+掩码关闭")

    limit = min(len(negative), _BC_AUX_NEGATIVE_RATIO * len(positive))
    chosen: list[int] = []
    seen: set[int] = set()

    def take(pool, quota):
        pool = np.asarray([int(i) for i in pool if int(i) not in seen],
                          dtype=np.int64)
        count = min(int(quota), len(pool), limit - len(chosen))
        if count <= 0:
            return
        # floor(k*N/n) 在 n<=N 时严格递增，故无重复且跨原序列均匀覆盖。
        pick = pool[np.floor(
            np.arange(count, dtype=np.float64) * len(pool) / count
        ).astype(np.int64)]
        for raw in pick:
            index = int(raw)
            if index not in seen:
                chosen.append(index)
                seen.add(index)

    hp = x[:, 0]
    take(negative[hp[negative] >= 0.65], max(1, limit // 2))
    take(negative[(hp[negative] >= 0.5) & (hp[negative] < 0.65)],
         max(1, limit // 4))
    take(negative[y[negative] != 9], max(1, limit // 8))
    take(negative, limit - len(chosen))
    _require(len(chosen) == limit,
             f"a12 hard-negative 选择数异常:{len(chosen)} != {limit}")
    keep = np.concatenate([positive, np.asarray(chosen, dtype=np.int64)])
    return x[keep], y[keep], masks[keep]


def _bc_v2_holdout_indices(episode_id):
    """镜像 bc_worker.split_by_episode 的 rng(23)/10% 整局切分。

    train_ppo 不能反向 import bc_worker（后者已 import 本模块），故在此保留
    小而封闭的纯函数，并由测试逐数组对齐。
    """
    import numpy as np

    groups = np.asarray(episode_id)
    _require(groups.ndim == 1 and len(groups) > 0
             and np.issubdtype(groups.dtype, np.integer),
             "BC-v2 held-out episode_id 非法")
    episodes = np.unique(groups)
    _require(len(episodes) >= 2, "BC-v2 held-out 至少需要 2 个独立 episode")
    order = np.random.default_rng(_BC_FINAL_SPLIT_SEED).permutation(episodes)
    held = order[:max(1, int(round(len(order) * 0.1)))]
    indices = np.flatnonzero(np.isin(groups, held))
    _require(0 < len(indices) < len(groups),
             "BC-v2 episode split 产生空训练集或空 held-out")
    return indices


def _bc_v2_training_indices(episode_id):
    """与固定 held-out 整局互斥且完备的训练行索引。"""
    import numpy as np

    groups = np.asarray(episode_id)
    heldout = _bc_v2_holdout_indices(groups)
    selected = np.ones(len(groups), dtype=np.bool_)
    selected[heldout] = False
    training = np.flatnonzero(selected)
    _require(len(training) + len(heldout) == len(groups)
             and not np.intersect1d(training, heldout).size,
             "BC-v2 training/held-out 行切分不互斥完备")
    train_episodes = np.unique(groups[training])
    heldout_episodes = np.unique(groups[heldout])
    _require(not np.intersect1d(train_episodes, heldout_episodes).size,
             "BC-v2 training/held-out episode 发生泄漏")
    return training


def _bc_v2_fit_validation_indices(episode_id):
    """镜像 bc_worker 的 nested rng(2301)/10% 整局候选选择切分。"""
    import numpy as np

    groups = np.asarray(episode_id)
    training = _bc_v2_training_indices(groups)
    training_episodes = np.unique(groups[training])
    _require(len(training_episodes) >= 2,
             "BC-v2 nested split 至少需要两个 training episodes")
    order = np.random.default_rng(
        _BC_SELECTION_SPLIT_SEED).permutation(training_episodes)
    n_validation = max(
        1, int(round(
            len(order) * _BC_SELECTION_VALIDATION_FRACTION)))
    n_validation = min(n_validation, len(order) - 1)
    validation_episodes = order[:n_validation]
    training_mask = np.zeros(len(groups), dtype=np.bool_)
    training_mask[training] = True
    validation_mask = np.isin(groups, validation_episodes)
    fit = np.flatnonzero(training_mask & ~validation_mask)
    validation = np.flatnonzero(training_mask & validation_mask)
    heldout = _bc_v2_holdout_indices(groups)
    _require(len(fit) > 0 and len(validation) > 0
             and len(fit) + len(validation) + len(heldout) == len(groups)
             and not np.intersect1d(fit, validation).size
             and not np.intersect1d(fit, heldout).size
             and not np.intersect1d(validation, heldout).size,
             "BC-v2 nested fit/validation/final 三域不互斥完备")
    return fit, validation


def _bc_v2_post_drink_coverage(
        x, y, episode_id, masks, *,
        scopes: tuple[str, ...] = ("fit", "validation", "final")) -> dict:
    """在调用者获准的域证明可见后饮闩与 m12 掩码同源。

    Producer 在候选冻结前只能传 ``("fit", "validation")``。默认三域
    仅供已经提交的一次性 bundle 消费端复验；这样覆盖诊断本身不能再偷看
    final heldout。
    """
    import numpy as np

    x = np.asarray(x)
    y = np.asarray(y)
    groups = np.asarray(episode_id)
    masks = np.asarray(masks)
    _require(
        x.ndim == 2 and x.shape[1] == 298
        and y.shape == groups.shape == (len(x),)
        and masks.shape == (len(x), 15)
        and masks.dtype == np.bool_,
        "BC-v2 后饮覆盖输入形状/dtype 异常",
    )
    fit, validation = _bc_v2_fit_validation_indices(groups)
    heldout = _bc_v2_holdout_indices(groups)
    allowed = {
        "fit": fit,
        "validation": validation,
        "final": heldout,
    }
    _require(
        isinstance(scopes, tuple)
        and bool(scopes)
        and len(scopes) == len(set(scopes))
        and all(scope in allowed for scope in scopes),
        f"BC-v2 后饮覆盖 scopes 非法:{scopes!r}",
    )
    result = {}
    for scope in scopes:
        indices = allowed[scope]
        positive = int((y[indices] == 12).sum())
        post_drink_negative = int((
            (y[indices] != 12)
            & ~masks[indices, 12]
            & (x[indices, _A12_CALIBRATION_DRINK_LATCH_FEATURE] < 0.0)
        ).sum())
        _require(
            positive > 0
            and post_drink_negative
            >= _BC_AUX_MIN_POST_DRINK_NEGATIVE_RATIO * positive,
            f"BC-v2 {scope} 域缺少可见后饮关闩覆盖:"
            f"positive={positive},post_drink_negative="
            f"{post_drink_negative},要求≥"
            f"{_BC_AUX_MIN_POST_DRINK_NEGATIVE_RATIO}:1",
        )
        result[scope] = {
            "positive_a12": positive,
            "post_drink_masked_negatives": post_drink_negative,
        }
    return result


def _build_bc_aux_training_bank(x, y, episode_id, masks):
    """只从固定 training episodes 构造优化 bank；held-out 永不入图。"""
    training = _bc_v2_training_indices(episode_id)
    return _filter_bc_aux_demo_pairs(
        x[training], y[training], masks[training])


def _policy_logits_from_sb3_state_dict(
        policy_sd, obs, *, legacy_scene_clock: bool = False,
        action_masks=None, circuit_spec=None,
        return_raw_actor_logits: bool = False):
    """以 SB3 六张策略头张量离线前向；供 BC-v2/E5/G0 共用。

    ``legacy_scene_clock`` 是保留给既有调用方的参数名；为真时实际应用
    完整 protocol-v3 worker 兼容视图（286 恢复 heals 主刻度，297 恢复
    exhausted 位）。当前学生的 contextual gate 仍必须读取 raw signed
    观测，不能在这个通用前向里默认转换。
    """
    import numpy as np
    import torch as th

    required = (
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.policy_net.2.weight",
        "mlp_extractor.policy_net.2.bias",
        "action_net.weight", "action_net.bias",
    )
    _require(isinstance(return_raw_actor_logits, bool),
             "return_raw_actor_logits 必须为 bool")
    _require(isinstance(policy_sd, dict)
             and all(key in policy_sd for key in required),
             "离线行为门所需 SB3 策略头六张量不全")
    raw_x = th.as_tensor(np.asarray(obs, dtype=np.float32))
    x = raw_x
    if legacy_scene_clock:
        from leashed_ppo import _legacy_worker_observation_view

        x = _legacy_worker_observation_view(x)
    w0, b0, w1, b1, wa, ba = (policy_sd[key] for key in required)
    tensors = tuple(th.as_tensor(t, dtype=th.float32)
                    for t in (w0, b0, w1, b1, wa, ba))
    w0, b0, w1, b1, wa, ba = tensors
    _require(w0.ndim == 2 and w0.shape[1] == x.shape[1]
             and wa.ndim == 2 and wa.shape[0] == 15,
             "离线行为门策略头形状异常")
    with th.no_grad():
        h = th.tanh(x @ w0.T + b0)
        h = th.tanh(h @ w1.T + b1)
        logits = h @ wa.T + ba
        raw_actor_logits = logits
        expanded = int(w0.shape[0]) == _BC_AUX_CIRCUIT_EXPANDED_WIDTH
        _require(
            (expanded and circuit_spec == _bc_aux_circuit_spec())
            or (not expanded and circuit_spec is None),
            "离线行为门不得仅凭 68 宽猜测 adapter 语义；"
            "必须显式绑定当前 contextual circuit spec",
        )
        if expanded:
            _require(action_masks is not None,
                     "contextual mixture 离线前向必须提供逐样本 masks")
            from leashed_ppo import (
                _a12_mixture_logits,
                _legacy_worker_observation_view,
            )
            # The live adapter feeds the complete protocol-v3 view to the old
            # actor while eligibility reads the untouched signed row.
            legacy_x = _legacy_worker_observation_view(raw_x)
            h = th.tanh(legacy_x @ w0.T + b0)
            h = th.tanh(h @ w1.T + b1)
            raw_logits = h @ wa.T + ba
            raw_actor_logits = raw_logits
            spec = circuit_spec
            feature_indices = tuple(
                int(value) for value in spec["gate_feature_indices"])
            parameter_columns = tuple(
                int(value) for value in spec["gate_parameter_columns"])
            gate_logits = (
                raw_x[:, feature_indices]
                @ wa[
                    int(spec["action_index"]),
                    list(parameter_columns),
                ]
                + ba[int(spec["action_index"])]
            )
            logits = _a12_mixture_logits(
                raw_logits,
                raw_x,
                action_masks,
                gate_logits,
                action=int(spec["action_index"]),
                hp_low=float(spec["hp_low"]),
                hp_high=float(spec["hp_high"]),
                boundary_epsilon=float(spec["boundary_epsilon"]),
                probability_min=float(spec["probability_min"]),
                probability_max=float(spec["probability_max"]),
            )
    return raw_actor_logits if return_raw_actor_logits else logits


def _validate_bc_v2_calibration_receipt(
        report, policy_sd, x, y, episode_id, masks) -> None:
    """用 canonical demos + 最终六张量重算校准 fit 回执与三域边界。"""
    import numpy as np
    import torch as th

    calibration = report.get("a12_calibration")
    expected_keys = {
        "schema_version", "fit_scope", "fit_pairs", "fit_episodes",
        "validation_pairs_excluded", "validation_episodes_excluded",
        "final_heldout_pairs_excluded",
        "final_heldout_episodes_excluded", "hp_low", "hp_high",
        "hp_feature", "drink_latch_feature", "predicate", "bias_12",
        "target_recall_12", "fit_metrics",
    }
    _require(isinstance(calibration, dict)
             and set(calibration) == expected_keys,
             "④乙 v2 a12_calibration 字段/schema 不精确")
    _require(
        calibration["schema_version"] == _A12_CALIBRATION_SCHEMA_VERSION
        and calibration["fit_scope"] == "nested-fit-episodes-only"
        and calibration["hp_low"] == 0.5
        and calibration["hp_high"] == report["preventive_threshold"]
        and calibration["hp_feature"] == _A12_CALIBRATION_HP_FEATURE
        and calibration["drink_latch_feature"]
        == _A12_CALIBRATION_DRINK_LATCH_FEATURE
        and calibration["predicate"] == _A12_CALIBRATION_PREDICATE
        and calibration["target_recall_12"]
        == _A12_CALIBRATION_TRAIN_RECALL_TARGET,
        "④乙 v2 a12_calibration 身份/可见谓词不符")

    fit, validation = _bc_v2_fit_validation_indices(episode_id)
    final_heldout = _bc_v2_holdout_indices(episode_id)
    expected_counts = {
        "fit_pairs": len(fit),
        "fit_episodes": len(np.unique(np.asarray(episode_id)[fit])),
        "validation_pairs_excluded": len(validation),
        "validation_episodes_excluded":
            len(np.unique(np.asarray(episode_id)[validation])),
        "final_heldout_pairs_excluded": len(final_heldout),
        "final_heldout_episodes_excluded":
            len(np.unique(np.asarray(episode_id)[final_heldout])),
    }
    for key, expected in expected_counts.items():
        value = calibration[key]
        _require(_is_plain_int(value) and value > 0 and value == expected,
                 "④乙 v2 a12_calibration 三域确定性切分不一致:"
                 f"{key}={value!r} != {expected}")

    bias = _finite_number(
        calibration["bias_12"], "④乙 v2 a12_calibration bias_12")
    policy_bias = float(
        th.as_tensor(policy_sd["action_net.bias"])[12].detach().cpu())
    # 生产者从最终 float32 参数回读后写 JSON；JSON 数字能无损往返该值，
    # 因而这里要求精确相等。近似容差会重新打开“回执不是部署权重”的缝。
    _require(bias == policy_bias,
        "④乙 v2 a12_calibration bias_12 未绑定最终 policy")

    fx = np.asarray(x)[fit]
    fy = np.asarray(y)[fit]
    fm = np.asarray(masks)[fit]
    logits = _policy_logits_from_sb3_state_dict(policy_sd, fx)
    masked = th.where(
        th.as_tensor(fm, dtype=th.bool), logits,
        th.full_like(logits, -1e8))
    pred = masked.argmax(dim=-1).cpu().numpy()
    probabilities = th.softmax(masked, dim=-1)[:, 12].cpu().numpy()
    pred12 = pred == 12
    true12 = fy == 12
    negative = ~true12
    legal12 = fm[:, 12]
    legal_negative = negative & legal12
    high_hp = (
        negative & legal12
        & (fx[:, _A12_CALIBRATION_HP_FEATURE] >= 0.65))
    tp = int((pred12 & true12).sum())
    fp = int((pred12 & negative).sum())
    high_hp_fp = int((pred12 & high_hp).sum())
    legal_negative_probabilities = probabilities[legal_negative]
    legal_negative_probabilities_f64 = legal_negative_probabilities.astype(
        np.float64, copy=False)
    legal_negative_probability_mean = (
        float(legal_negative_probabilities_f64.mean())
        if len(legal_negative_probabilities) else 0.0)
    legal_negative_probability_max = (
        float(legal_negative_probabilities.max())
        if len(legal_negative_probabilities) else 0.0)
    pred13_share = float((pred == 13).mean())
    true13_share = float((fy == 13).mean())
    observed = {
        "tp": tp,
        "fp": fp,
        "precision_12": tp / max(1, tp + fp),
        "recall_12": tp / max(1, int(true12.sum())),
        "fpr_12": fp / max(1, int(legal_negative.sum())),
        "predicted_share_12": float(pred12.mean()),
        "high_hp_false_drink_rate":
            high_hp_fp / max(1, int(high_hp.sum())),
        "legal_negative_probability_12_mean":
            legal_negative_probability_mean,
        "legal_negative_probability_12_max":
            legal_negative_probability_max,
        "a13_spillover": max(0.0, pred13_share - true13_share),
    }
    metrics = calibration["fit_metrics"]
    _require(isinstance(metrics, dict)
             and set(metrics) == set(observed),
             "④乙 v2 a12_calibration fit_metrics 字段不精确")
    for key, expected in observed.items():
        value = metrics[key]
        if key in {"tp", "fp"}:
            _require(_is_plain_int(value) and value >= 0
                     and value == expected,
                     "④乙 v2 a12_calibration fit 计数未绑定现场策略:"
                     f"{key}={value!r} != {expected}")
        else:
            numeric = _finite_number(
                value, f"④乙 v2 a12_calibration fit_metrics.{key}")
            _require(0.0 <= numeric <= 1.0
                     and math.isclose(
                         numeric, expected, rel_tol=0.0, abs_tol=1e-15),
                     "④乙 v2 a12_calibration fit 指标未绑定现场策略:"
                     f"{key}={numeric!r} != {expected!r}")
    _require(
        observed["recall_12"] >= _A12_CALIBRATION_TRAIN_RECALL_TARGET
        and observed["precision_12"] >= max(0.10, _A12_PRECISION_MIN)
        and observed["fpr_12"] <= _A12_FPR_MAX * 0.5
        and _A12_PREDICTED_SHARE_MIN
        <= observed["predicted_share_12"]
        <= _A12_PREDICTED_SHARE_MAX
        and int(high_hp.sum()) > 0
        and observed["high_hp_false_drink_rate"]
        <= _A12_HIGH_HP_FALSE_DRINK_MAX * 0.5
        and observed["legal_negative_probability_12_mean"]
        <= _A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX
        and observed["legal_negative_probability_12_max"]
        <= _A12_LEGAL_NEGATIVE_PROBABILITY_MAX
        and observed["a13_spillover"] <= _A13_SPILLOVER_MAX,
        "④乙 v2 a12_calibration 现场 fit 安全门未过")


def bc_aux_behavior_metrics(
        policy_sd, x, y, episode_id, masks, *,
        anchor_sd=None, heldout_only: bool = True, circuit_spec=None) -> dict:
    """E5/a12 行为面：真实 v2 masks 上的 held-out 分类与起点漂移。

    这不是训练 loss 的重述。precision/FPR/predicted share/high-HP 误饮可抓
    “recall 很高但到处喝”的坍缩；a13 spillover 和 anchor TV/KL 则抓把
    非触发态原有补给/交战分布挤走的副作用。
    """
    import numpy as np
    import torch as th

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    episode_id = np.asarray(episode_id)
    masks = np.asarray(masks)
    _require(x.ndim == 2 and y.shape == (len(x),)
             and episode_id.shape == (len(x),)
             and masks.shape == (len(x), 15)
             and masks.dtype == np.bool_,
             "E5 BC-v2 行为门输入形状/dtype 异常")
    indices = (_bc_v2_holdout_indices(episode_id) if heldout_only
               else np.arange(len(x), dtype=np.int64))
    hx, hy, hm = x[indices], y[indices], masks[indices]
    logits = _policy_logits_from_sb3_state_dict(
        policy_sd, hx, action_masks=hm, circuit_spec=circuit_spec)
    mask_t = th.as_tensor(hm, dtype=th.bool)
    masked = th.where(mask_t, logits, th.full_like(logits, -1e8))
    probs = th.softmax(masked, dim=-1)
    pred = probs.argmax(dim=-1).cpu().numpy()

    true12 = hy == 12
    pred12 = pred == 12
    all_negative = ~true12
    # FPR 的可行动分母只能是现场 m12=True 的非 12 态。把 quota 已耗尽、
    # 空腰带等 m12=False 结构性不可能态混入，会随掩码强化而把 FPR 人为
    # 稀释至近零，恰好掩盖“在每个仍可喝状态都喝”的坍缩。
    legal_negative = all_negative & hm[:, 12]
    tp = int((pred12 & true12).sum())
    fp = int((pred12 & legal_negative).sum())
    fn = int((~pred12 & true12).sum())
    tn = int((~pred12 & legal_negative).sum())
    positive_denom = int(true12.sum())
    all_negative_denom = int(all_negative.sum())
    negative_denom = int(legal_negative.sum())
    predicted_positive = int(pred12.sum())
    high_hp_negative = legal_negative & (hx[:, 0] >= 0.65)
    high_hp_denom = int(high_hp_negative.sum())
    high_hp_fp = int((pred12 & high_hp_negative).sum())
    legal_negative_p12 = probs[:, 12].cpu().numpy()[legal_negative]
    eligible_p12 = probs[:, 12].cpu().numpy()[true12]
    # NumPy preserves float32 for mean/sum by default.  On a constant vector
    # its accumulated mean can round *above* the elementwise maximum, which
    # made the fail-closed order check reject a mathematically valid policy.
    # Accumulate audit scalars in float64 while preserving deployed float32
    # probabilities for min/max and argmax.
    legal_negative_p12_f64 = legal_negative_p12.astype(
        np.float64, copy=False)
    eligible_p12_f64 = eligible_p12.astype(np.float64, copy=False)
    all_p12_f64 = probs[:, 12].cpu().numpy().astype(
        np.float64, copy=False)
    non12_support = mask_t.clone()
    non12_support[:, _BC_AUX_CIRCUIT_ACTION] = False
    top_non12_probability = th.where(
        non12_support, probs, th.zeros_like(probs)
    ).max(dim=-1).values.cpu().numpy()
    a12_margins = (
        probs[:, _BC_AUX_CIRCUIT_ACTION].cpu().numpy()
        - top_non12_probability
    )
    predicted_episode_count = int(np.unique(
        episode_id[indices][pred12]).size)
    predicted_margin_min = (
        float(a12_margins[pred12].min()) if predicted_positive else 0.0)
    n = len(indices)

    pred13_share = float((pred == 13).mean()) if n else 0.0
    true13_share = float((hy == 13).mean()) if n else 0.0
    anchor = None
    if anchor_sd is not None:
        # 根锚来自 protocol-v3 worker 语义。只转换 anchor 输入；上面的
        # 当前候选 contextual gate 仍消费 packed belt 与 signed 闩位。
        anchor_logits = _policy_logits_from_sb3_state_dict(
            anchor_sd, hx, legacy_scene_clock=True)
        if circuit_spec == _bc_aux_circuit_spec():
            from leashed_ppo import (
                _legacy_distillation_masks,
                _masked_log_softmax_from_raw,
            )

            root_support = _legacy_distillation_masks(mask_t)
            anchor_logp = _masked_log_softmax_from_raw(
                anchor_logits, root_support)
            anchor_probs = anchor_logp.exp()
            current_root_logits = _policy_logits_from_sb3_state_dict(
                policy_sd,
                hx,
                action_masks=hm,
                circuit_spec=circuit_spec,
                return_raw_actor_logits=True,
            )
            current_root_logp = _masked_log_softmax_from_raw(
                current_root_logits, root_support)
            current_root_probs = current_root_logp.exp()
            current_root_pred = (
                current_root_probs.argmax(dim=-1).cpu().numpy())
        else:
            anchor_support = mask_t.clone()
            anchor_support[:, _BC_AUX_CIRCUIT_ACTION] = False
            anchor_masked = th.where(
                anchor_support, anchor_logits,
                th.full_like(anchor_logits, -1e8))
            anchor_probs = th.softmax(anchor_masked, dim=-1)
            anchor_logp = th.log(
                anchor_probs.clamp_min(th.finfo(anchor_probs.dtype).eps))
            current_root_probs = probs
            current_root_logp = th.log(
                probs.clamp_min(th.finfo(probs.dtype).eps))
            current_root_pred = pred
        anchor_pred = anchor_probs.argmax(dim=-1).cpu().numpy()
        critical_retention = {}
        for action in _BC_AUX_CRITICAL_ACTIONS:
            selected = anchor_pred == action
            support = int(selected.sum())
            retained = int(
                (selected & (current_root_pred == action)).sum())
            critical_retention[str(action)] = {
                "support": support,
                "retained": retained,
                "retention": (retained / support if support else None),
            }
        anchor = {
            "argmax_drift": float(
                (anchor_pred != current_root_pred).mean()),
            "tv_mean": float(
                (0.5 * (current_root_probs - anchor_probs)
                 .abs().sum(dim=-1)).mean()),
            "kl_anchor_to_policy": float(
                (anchor_probs * (
                    anchor_logp
                    - current_root_logp
                )).sum(dim=-1).mean().clamp_min(0.0)),
            "a12_probability_delta": float(
                (probs[:, 12] - anchor_probs[:, 12]).mean()),
            "a13_predicted_share": float((anchor_pred == 13).mean()),
            "critical_action_retention": critical_retention,
        }
        a13_reference = anchor["a13_predicted_share"]
        a13_reference_name = "anchor_argmax"
    else:
        a13_reference = true13_share
        a13_reference_name = "heldout_label"

    return {
        "scope": "heldout" if heldout_only else "full",
        "mask_mode": "bc-v2-recorded",
        "pairs": int(n),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "true_a12": positive_denom,
        "non_a12": negative_denom,
        "all_non_a12": all_negative_denom,
        "predicted_a12": predicted_positive,
        "predicted_a12_episodes": predicted_episode_count,
        "predicted_a12_margin_min": predicted_margin_min,
        "precision_12": (tp / predicted_positive
                         if predicted_positive else 0.0),
        "recall_12": tp / positive_denom if positive_denom else 0.0,
        "fpr_12": fp / negative_denom if negative_denom else 1.0,
        "predicted_share_12": predicted_positive / n if n else 0.0,
        "high_hp_non_a12": high_hp_denom,
        "high_hp_false_drinks": high_hp_fp,
        "high_hp_false_drink_rate": (
            high_hp_fp / high_hp_denom if high_hp_denom else 1.0),
        "eligible_probability_12_min": (
            float(eligible_p12.min()) if positive_denom else 0.0),
        "eligible_probability_12_mean": (
            float(eligible_p12_f64.mean()) if positive_denom else 0.0),
        "eligible_probability_12_max": (
            float(eligible_p12.max()) if positive_denom else 0.0),
        "legal_negative_probability_12_mean": (
            float(legal_negative_p12_f64.mean())
            if negative_denom else 1.0),
        "legal_negative_probability_12_max": (
            float(legal_negative_p12.max())
            if negative_denom else 1.0),
        "legal_negative_probability_12_sum": (
            float(legal_negative_p12_f64.sum())
            if negative_denom else float("inf")),
        "predicted_share_13": pred13_share,
        "true_share_13": true13_share,
        "a13_reference": a13_reference_name,
        "a13_reference_share": a13_reference,
        "a13_spillover": max(0.0, pred13_share - a13_reference),
        "mean_probability_12": float(all_p12_f64.mean()) if n else 0.0,
        "anchor": anchor,
    }


def bc_aux_behavior_gate(
        metrics: dict, *, require_root_anchor: bool = False,
        require_teacher_recall: bool = True,
        require_deployable_a12: bool = False) -> dict:
    """生产硬门；返回结构化原因，调用者不得仅消费一个 PASS 字符串。"""
    reasons = []
    if not isinstance(metrics, dict) \
            or set(metrics) != set(_BC_AUX_BEHAVIOR_METRIC_KEYS):
        reasons.append("metric_schema_mismatch")
        metrics = metrics if isinstance(metrics, dict) else {}

    def plain_int(key):
        value = metrics.get(key)
        if not _is_plain_int(value) or value < 0:
            reasons.append(f"{key}_invalid")
            return None
        return int(value)

    def finite_number(key):
        value = metrics.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            reasons.append(f"{key}_invalid")
            return None
        return float(value)

    def at_least(key, threshold):
        value = finite_number(key)
        if value is None or value < threshold:
            reasons.append(f"{key}<{threshold}")

    def at_most(key, threshold):
        value = finite_number(key)
        if value is None or value > threshold:
            reasons.append(f"{key}>{threshold}")

    if metrics.get("scope") not in {"heldout", "full"}:
        reasons.append("scope_invalid")
    if metrics.get("mask_mode") != "bc-v2-recorded":
        reasons.append("mask_mode_invalid")
    count_keys = (
        "pairs", "tp", "fp", "fn", "tn", "true_a12", "non_a12",
        "all_non_a12", "predicted_a12", "predicted_a12_episodes",
        "high_hp_non_a12", "high_hp_false_drinks",
    )
    counts = {key: plain_int(key) for key in count_keys}
    if all(value is not None for value in counts.values()):
        if not (
            counts["pairs"] > 0
            and counts["tp"] + counts["fn"] == counts["true_a12"]
            and counts["fp"] + counts["tn"] == counts["non_a12"]
            and counts["true_a12"] + counts["all_non_a12"]
            == counts["pairs"]
            and counts["non_a12"] <= counts["all_non_a12"]
            and counts["predicted_a12"] == counts["tp"] + counts["fp"]
            and counts["predicted_a12_episodes"]
            <= counts["predicted_a12"]
            and counts["high_hp_false_drinks"]
            <= counts["high_hp_non_a12"]
            and (
                (counts["predicted_a12"] == 0
                 and counts["predicted_a12_episodes"] == 0)
                or (
                    counts["predicted_a12"] > 0
                    and 1 <= counts["predicted_a12_episodes"]
                    <= counts["predicted_a12"]
                )
            )
        ):
            reasons.append("count_closure_invalid")

    unit_interval_keys = (
        "precision_12", "recall_12", "fpr_12", "predicted_share_12",
        "high_hp_false_drink_rate", "predicted_share_13",
        "true_share_13", "eligible_probability_12_min",
        "eligible_probability_12_mean", "eligible_probability_12_max",
        "legal_negative_probability_12_mean",
        "legal_negative_probability_12_max", "a13_reference_share",
        "a13_spillover", "mean_probability_12",
    )
    numbers = {
        key: finite_number(key)
        for key in (
            *unit_interval_keys,
            "predicted_a12_margin_min",
            "legal_negative_probability_12_sum",
        )
    }
    for key in unit_interval_keys:
        value = numbers[key]
        if value is not None and not 0.0 <= value <= 1.0:
            reasons.append(f"{key}_out_of_range")
    margin = numbers["predicted_a12_margin_min"]
    if margin is not None and not -1.0 <= margin <= 1.0:
        reasons.append("predicted_a12_margin_min_out_of_range")
    if (
        margin is not None
        and counts.get("predicted_a12") is not None
        and (
            (counts["predicted_a12"] == 0 and margin != 0.0)
            or (counts["predicted_a12"] > 0 and margin < 0.0)
        )
    ):
        reasons.append("predicted_a12_margin_count_mismatch")
    probability_sum = numbers["legal_negative_probability_12_sum"]
    if probability_sum is not None and probability_sum < 0.0:
        reasons.append("legal_negative_probability_12_sum_out_of_range")
    if all(numbers.get(key) is not None for key in (
            "eligible_probability_12_min",
            "eligible_probability_12_mean",
            "eligible_probability_12_max")) and not (
        numbers["eligible_probability_12_min"]
        <= numbers["eligible_probability_12_mean"]
        <= numbers["eligible_probability_12_max"]
    ):
        reasons.append("eligible_probability_order_invalid")
    if all(numbers.get(key) is not None for key in (
            "legal_negative_probability_12_mean",
            "legal_negative_probability_12_max",
            "legal_negative_probability_12_sum")) and not (
        numbers["legal_negative_probability_12_mean"]
        <= numbers["legal_negative_probability_12_max"] + 1e-12
        and (
            counts.get("non_a12") is None
            or math.isclose(
                numbers["legal_negative_probability_12_sum"],
                numbers["legal_negative_probability_12_mean"]
                * counts["non_a12"],
                rel_tol=1e-5,
                abs_tol=1e-6,
            )
        )
    ):
        reasons.append("legal_negative_probability_closure_invalid")
    if all(counts.get(key) is not None for key in (
            "pairs", "tp", "fp", "true_a12", "non_a12",
            "predicted_a12", "high_hp_non_a12",
            "high_hp_false_drinks")):
        expected_rates = {
            "precision_12": (
                counts["tp"] / counts["predicted_a12"]
                if counts["predicted_a12"] else 0.0),
            "recall_12": (
                counts["tp"] / counts["true_a12"]
                if counts["true_a12"] else 0.0),
            "fpr_12": (
                counts["fp"] / counts["non_a12"]
                if counts["non_a12"] else 1.0),
            "predicted_share_12":
                counts["predicted_a12"] / counts["pairs"],
            "high_hp_false_drink_rate": (
                counts["high_hp_false_drinks"]
                / counts["high_hp_non_a12"]
                if counts["high_hp_non_a12"] else 1.0),
        }
        for key, expected in expected_rates.items():
            value = numbers.get(key)
            if value is not None and not math.isclose(
                    value, expected, rel_tol=0.0, abs_tol=1e-12):
                reasons.append(f"{key}_count_mismatch")
    if all(
        counts.get(key) is not None
        for key in ("pairs", "true_a12", "non_a12")
    ) and all(
        numbers.get(key) is not None
        for key in (
            "eligible_probability_12_mean",
            "legal_negative_probability_12_sum",
            "mean_probability_12",
        )
    ) and counts["pairs"] > 0:
        # The recorded mask makes p12 exactly zero on the remaining
        # ``all_non_a12 - non_a12`` rows.  Bind the global probability mean
        # to the two non-zero domains instead of accepting an unrelated
        # self-reported scalar.
        expected_mean_probability = (
            numbers["eligible_probability_12_mean"]
            * counts["true_a12"]
            + numbers["legal_negative_probability_12_sum"]
        ) / counts["pairs"]
        if not math.isclose(
                numbers["mean_probability_12"],
                expected_mean_probability,
                rel_tol=1e-5, abs_tol=1e-7):
            reasons.append("mean_probability_12_closure_invalid")

    reference = metrics.get("a13_reference")
    reference_share = numbers.get("a13_reference_share")
    predicted_share_13 = numbers.get("predicted_share_13")
    spillover = numbers.get("a13_spillover")
    anchor = metrics.get("anchor")
    if anchor is None:
        if reference != "heldout_label":
            reasons.append("a13_reference_invalid")
        true_share = numbers.get("true_share_13")
        if (
            reference_share is not None
            and true_share is not None
            and not math.isclose(
                reference_share, true_share,
                rel_tol=0.0, abs_tol=1e-12)
        ):
            reasons.append("a13_reference_share_mismatch")
    else:
        if reference != "anchor_argmax":
            reasons.append("a13_reference_invalid")
        expected_anchor_keys = {
            "argmax_drift", "tv_mean", "kl_anchor_to_policy",
            "a12_probability_delta", "a13_predicted_share",
            "critical_action_retention",
        }
        if not isinstance(anchor, dict) or set(anchor) != expected_anchor_keys:
            reasons.append("root_anchor_schema_mismatch")
        else:
            anchor_a13 = anchor.get("a13_predicted_share")
            if (
                not isinstance(anchor_a13, (int, float))
                or isinstance(anchor_a13, bool)
                or not math.isfinite(float(anchor_a13))
                or not 0.0 <= float(anchor_a13) <= 1.0
            ):
                reasons.append("root_anchor.a13_predicted_share_invalid")
            elif (
                reference_share is not None
                and not math.isclose(
                    reference_share, float(anchor_a13),
                    rel_tol=0.0, abs_tol=1e-12)
            ):
                reasons.append("a13_reference_share_mismatch")
            delta = anchor.get("a12_probability_delta")
            if (
                not isinstance(delta, (int, float))
                or isinstance(delta, bool)
                or not math.isfinite(float(delta))
                or not -1.0 <= float(delta) <= 1.0
            ):
                reasons.append("root_anchor.a12_probability_delta_invalid")
    if (
        predicted_share_13 is not None
        and reference_share is not None
        and spillover is not None
        and not math.isclose(
            spillover,
            max(0.0, predicted_share_13 - reference_share),
            rel_tol=0.0, abs_tol=1e-12)
    ):
        reasons.append("a13_spillover_closure_invalid")

    at_least("true_a12", 1)
    at_least("non_a12", 1)
    at_least("all_non_a12", 1)
    at_least("high_hp_non_a12", 1)
    if require_teacher_recall:
        at_least("precision_12", _A12_PRECISION_MIN)
        at_least("recall_12", _A12_RECALL_MIN)
        at_least("predicted_share_12", _A12_PREDICTED_SHARE_MIN)
    elif isinstance(metrics.get("predicted_a12"), int) \
            and not isinstance(metrics.get("predicted_a12"), bool) \
            and metrics["predicted_a12"] > 0:
        # A learned argmax is allowed, but if it exists it must still be
        # concentrated on teacher-eligible states rather than global spill.
        at_least("precision_12", _A12_PRECISION_MIN)
    if require_deployable_a12:
        at_least(
            "predicted_a12_episodes",
            _BC_AUX_MIN_DETERMINISTIC_A12_EPISODES)
        at_least(
            "predicted_a12_margin_min",
            _BC_AUX_MIN_DETERMINISTIC_A12_MARGIN)
    at_most("fpr_12", _A12_FPR_MAX)
    at_most("predicted_share_12", _A12_PREDICTED_SHARE_MAX)
    at_most("high_hp_false_drink_rate", _A12_HIGH_HP_FALSE_DRINK_MAX)
    at_most(
        "legal_negative_probability_12_mean",
        _A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX)
    at_most(
        "legal_negative_probability_12_max",
        _A12_LEGAL_NEGATIVE_PROBABILITY_MAX)
    at_most("a13_spillover", _A13_SPILLOVER_MAX)
    if require_root_anchor:
        if not isinstance(anchor, dict):
            reasons.append("root_anchor_missing")
        else:
            for key, threshold in (
                ("argmax_drift", _BC_AUX_ROOT_ARGMAX_DRIFT_MAX),
                ("tv_mean", _BC_AUX_ROOT_TV_MAX),
                ("kl_anchor_to_policy", _BC_AUX_ROOT_KL_MAX),
            ):
                value = anchor.get(key)
                if (not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                        or float(value) < 0.0
                        or float(value) > threshold):
                    reasons.append(f"root_anchor.{key}>{threshold}")
            retention = anchor.get("critical_action_retention")
            if not isinstance(retention, dict):
                reasons.append("root_anchor.critical_action_retention_missing")
            else:
                for action in _BC_AUX_CRITICAL_ACTIONS:
                    row = retention.get(str(action))
                    # 某动作在根策略的 held-out argmax 中零覆盖时没有可保留
                    # 行，跳过该动作；有覆盖则至少保住一半，防 9/10/13
                    # 被 a12 校准或 PPO 更新整类抹除。
                    if not isinstance(row, dict):
                        reasons.append(
                            f"root_anchor.action_{action}_retention_missing")
                        continue
                    support = row.get("support")
                    retained = row.get("retained")
                    value = row.get("retention")
                    if (not _is_plain_int(support) or support < 0
                            or not _is_plain_int(retained)
                            or not 0 <= retained <= support):
                        reasons.append(
                            f"root_anchor.action_{action}_support_invalid")
                    elif support > 0 and (
                            not isinstance(value, (int, float))
                            or isinstance(value, bool)
                            or not math.isfinite(float(value))
                            or not math.isclose(
                                float(value), retained / support,
                                rel_tol=0.0, abs_tol=1e-12)
                            or float(value)
                            < _BC_AUX_CRITICAL_RETENTION_MIN):
                        reasons.append(
                            f"root_anchor.action_{action}_retention"
                            f"<{_BC_AUX_CRITICAL_RETENTION_MIN}")
                    elif support == 0 and value is not None:
                        reasons.append(
                            f"root_anchor.action_{action}_zero_support_retention")
    # Keep structured receipts deterministic even when several closure checks
    # discover the same bad scalar through different paths.
    reasons = list(dict.fromkeys(reasons))
    return {
        "verdict": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "thresholds": {
            "precision_12_min": _A12_PRECISION_MIN,
            "recall_12_min": (
                _A12_RECALL_MIN if require_teacher_recall else None),
            "teacher_recall_required": bool(require_teacher_recall),
            "deployable_a12_required": bool(require_deployable_a12),
            "deterministic_a12_episode_min": (
                _BC_AUX_MIN_DETERMINISTIC_A12_EPISODES
                if require_deployable_a12 else None),
            "deterministic_a12_margin_min": (
                _BC_AUX_MIN_DETERMINISTIC_A12_MARGIN
                if require_deployable_a12 else None),
            "fpr_12_max": _A12_FPR_MAX,
            "predicted_share_12_min": (
                _A12_PREDICTED_SHARE_MIN
                if require_teacher_recall else None),
            "predicted_share_12_max": _A12_PREDICTED_SHARE_MAX,
            "high_hp_false_drink_rate_max":
                _A12_HIGH_HP_FALSE_DRINK_MAX,
            "legal_negative_probability_12_mean_max":
                _A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX,
            "legal_negative_probability_12_max":
                _A12_LEGAL_NEGATIVE_PROBABILITY_MAX,
            "a13_spillover_max": _A13_SPILLOVER_MAX,
            "root_anchor_required": bool(require_root_anchor),
            "root_argmax_drift_max": _BC_AUX_ROOT_ARGMAX_DRIFT_MAX,
            "root_tv_mean_max": _BC_AUX_ROOT_TV_MAX,
            "root_kl_anchor_to_policy_max": _BC_AUX_ROOT_KL_MAX,
            "root_critical_action_retention_min":
                _BC_AUX_CRITICAL_RETENTION_MIN,
        },
    }


def _policy_head_snapshot(policy) -> dict:
    """冻结当前 SB3 策略头六张量；用于起点锚与最终发布行为门。"""
    state = policy.state_dict()
    _require(all(key in state for key in _POLICY_HEAD_KEYS),
             "策略 state_dict 缺少行为门所需六张量")
    return {
        key: state[key].detach().cpu().clone()
        for key in _POLICY_HEAD_KEYS
    }


def _policy_head_sha256(state: dict) -> str:
    """跨 torch.save 版本稳定的六张量内容摘要。"""
    import numpy as np

    _require(isinstance(state, dict)
             and set(state) == set(_POLICY_HEAD_KEYS),
             "策略头摘要输入键集合不精确")
    digest = hashlib.sha256()
    for key in _POLICY_HEAD_KEYS:
        tensor = state[key].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _persistent_bc_aux_root_anchor(model) -> dict:
    """首次 aux 腿冻结根锚；后续 checkpoint continuation 继承同一锚。"""
    current = _policy_head_snapshot(model.policy)
    root = getattr(model, "bc_aux_root_anchor_sd", None)
    if root is None:
        root = current
    _require(isinstance(root, dict)
             and set(root) == set(_POLICY_HEAD_KEYS),
             "checkpoint 中 bc_aux 根策略锚字段异常")
    frozen = {}
    for key in _POLICY_HEAD_KEYS:
        value = root[key]
        _require(hasattr(value, "detach")
                 and bool(value.detach().isfinite().all().item()),
                 f"bc_aux 根策略锚张量异常:{key}")
        frozen[key] = value.detach().cpu().clone()
    # rev9 widens only the live actor.  Its persistent root intentionally
    # remains the exact 64-wide V28 head and therefore must not be forced to
    # match the candidate's 68-wide shapes.
    w0, b0, w1, b1, wa, ba = (
        frozen[key] for key in _POLICY_HEAD_KEYS)
    # The root anchor is self-describing.  Do not require a fully constructed
    # SB3 model here: publication tests and offline gate tooling deliberately
    # use a minimal policy carrier with no observation/action spaces.
    obs_dim = int(w0.shape[1]) if w0.ndim == 2 else -1
    action_dim = int(wa.shape[0]) if wa.ndim == 2 else -1
    _require(
        w0.ndim == 2 and w0.shape[1] == obs_dim
        and w0.shape[0] == _BC_AUX_CIRCUIT_BASE_WIDTH
        and b0.shape == (w0.shape[0],)
        and w1.shape == (w0.shape[0], w0.shape[0])
        and b1.shape == (w0.shape[0],)
        and wa.shape == (action_dim, w0.shape[0])
        and ba.shape == (action_dim,),
        "bc_aux 根策略锚必须是 298→64→64→15 的 pre-adapter actor；"
        "existing adapter 缺 root 时禁止从 68 宽当前头静默重锚",
    )
    current_w0 = current["mlp_extractor.policy_net.0.weight"]
    current_wa = current["action_net.weight"]
    _require(
        current_w0.ndim == 2 and int(current_w0.shape[1]) == obs_dim
        and current_wa.ndim == 2 and int(current_wa.shape[0]) == action_dim,
        "bc_aux 当前策略与根锚输入/动作维度不一致",
    )
    model.bc_aux_root_anchor_sd = {
        key: value.clone() for key, value in frozen.items()
    }
    return frozen


def _bc_aux_circuit_spec() -> dict:
    """Canonical identity of the rev9 contextual exact-mixture adapter."""
    return {
        "schema_version": _BC_AUX_CIRCUIT_SCHEMA,
        "base_width": _BC_AUX_CIRCUIT_BASE_WIDTH,
        "expanded_width": _BC_AUX_CIRCUIT_EXPANDED_WIDTH,
        "action_index": _BC_AUX_CIRCUIT_ACTION,
        "gate_feature_indices":
            list(_BC_AUX_CIRCUIT_GATE_FEATURE_INDICES),
        "gate_parameter_columns":
            list(_BC_AUX_CIRCUIT_GATE_PARAMETER_COLUMNS),
        "hp_low": 0.5,
        "hp_high": 0.65,
        "boundary_epsilon": _A12_VISIBLE_HP_BOUNDARY_EPS,
        "initial_probability": _BC_AUX_CIRCUIT_INITIAL_PROBABILITY,
        "initial_gate_bias": _BC_AUX_CIRCUIT_INITIAL_GATE_BIAS,
        "probability_min": _BC_AUX_CIRCUIT_PROBABILITY_MIN,
        "probability_max": _BC_AUX_CIRCUIT_PROBABILITY_MAX,
        "gate_parameter_abs_max":
            _BC_AUX_CIRCUIT_GATE_PARAMETER_ABS_MAX,
    }


def _expand_policy_with_bc_aux_circuit(model) -> dict:
    """Losslessly widen V28 and install the exact distribution adapter.

    The value branch is untouched.  The old 64x64 actor block and all fourteen
    non-a12 output rows are copied bitwise; every new/cross block stays zero.
    Four action-head cells store coefficients over stable raw features and
    action12's bias stores their intercept.  The policy distribution maps that
    five-parameter score to ε(s); no a12 gradient enters the legacy actor.
    The caller must reset the optimizer after this topology/class change.
    """
    import torch as th

    policy = model.policy
    net = policy.mlp_extractor.policy_net
    _require(len(net) >= 4
             and isinstance(net[0], th.nn.Linear)
             and isinstance(net[2], th.nn.Linear)
             and isinstance(policy.action_net, th.nn.Linear),
             "a12 circuit 只支持标准两层 MlpPolicy actor")
    expected_spec = _bc_aux_circuit_spec()
    existing_spec = getattr(model, "_bc_aux_circuit_spec", None)
    if existing_spec is not None:
        from leashed_ppo import A12MixtureMaskableActorCriticPolicy
        _require(existing_spec == expected_spec,
                 "checkpoint a12 adapter spec 与 rev9 不一致")
        _require(
            isinstance(policy, A12MixtureMaskableActorCriticPolicy)
            and model.policy_class is A12MixtureMaskableActorCriticPolicy
            and policy.bc_aux_mixture_spec == expected_spec
            and model.policy_kwargs.get("bc_aux_mixture_spec")
            == expected_spec
            and
            net[0].weight.shape
            == (_BC_AUX_CIRCUIT_EXPANDED_WIDTH,
                int(model.observation_space.shape[0]))
            and net[2].weight.shape
            == (_BC_AUX_CIRCUIT_EXPANDED_WIDTH,
                _BC_AUX_CIRCUIT_EXPANDED_WIDTH)
            and policy.action_net.weight.shape
            == (int(model.action_space.n),
                _BC_AUX_CIRCUIT_EXPANDED_WIDTH),
            "checkpoint a12 circuit 拓扑与 spec 不一致")
        return expected_spec

    old0, old1, olda = net[0], net[2], policy.action_net
    base = _BC_AUX_CIRCUIT_BASE_WIDTH
    width = _BC_AUX_CIRCUIT_EXPANDED_WIDTH
    obs_dim = int(model.observation_space.shape[0])
    action_dim = int(model.action_space.n)
    _require(
        old0.weight.shape == (base, obs_dim)
        and old0.bias.shape == (base,)
        and old1.weight.shape == (base, base)
        and old1.bias.shape == (base,)
        and olda.weight.shape == (action_dim, base)
        and olda.bias.shape == (action_dim,)
        and action_dim > _BC_AUX_CIRCUIT_ACTION
        and obs_dim > _A12_CALIBRATION_DRINK_LATCH_FEATURE,
        "a12 circuit 只允许从冻结 298→64→64→15 V28 actor 迁移")
    root = _persistent_bc_aux_root_anchor(model)
    _require(_policy_head_sha256(root)
             == _policy_head_sha256(_policy_head_snapshot(policy)),
             "首次 a12 circuit 迁移前 root 必须逐位等于当前 V28 actor")

    device, dtype = old0.weight.device, old0.weight.dtype
    new0 = th.nn.Linear(
        obs_dim, width, bias=True, device=device, dtype=dtype)
    new1 = th.nn.Linear(
        width, width, bias=True, device=device, dtype=dtype)
    newa = th.nn.Linear(
        width, action_dim, bias=True, device=device, dtype=dtype)
    with th.no_grad():
        new0.weight.zero_()
        new0.bias.zero_()
        new1.weight.zero_()
        new1.bias.zero_()
        newa.weight.zero_()
        newa.bias.copy_(olda.bias)
        new0.weight[:base].copy_(old0.weight)
        new0.bias[:base].copy_(old0.bias)
        new1.weight[:base, :base].copy_(old1.weight)
        new1.bias[:base].copy_(old1.bias)
        newa.weight[:, :base].copy_(olda.weight)

        # a12 was permanently masked during V28/KING training.  Its old row is
        # untrained and is ignored by the contextual gate; zero it so no
        # alternate latent path can masquerade as a registered raw feature.
        newa.weight[_BC_AUX_CIRCUIT_ACTION].zero_()
        # Zero coefficients + this affine-sigmoid intercept produce exactly
        # 5% in every eligible state.  The four coefficients occupy the
        # otherwise-unused expanded columns and are consumed directly from raw
        # observations by the custom distribution, not via zero latent units.
        newa.bias[_BC_AUX_CIRCUIT_ACTION] = (
            _BC_AUX_CIRCUIT_INITIAL_GATE_BIAS)

    new0.train(old0.training)
    new1.train(old1.training)
    newa.train(olda.training)
    net[0] = new0
    net[2] = new1
    policy.action_net = newa
    policy.mlp_extractor.latent_dim_pi = width
    net_arch = {"pi": [width, width], "vf": [64, 64]}
    policy.net_arch = net_arch
    model.policy_kwargs = dict(getattr(model, "policy_kwargs", {}) or {})
    model.policy_kwargs["net_arch"] = net_arch
    model.policy_kwargs["bc_aux_mixture_spec"] = expected_spec
    from leashed_ppo import A12MixtureMaskableActorCriticPolicy
    # Both are pure-Python nn.Module classes with identical storage layout;
    # changing the behavior class preserves the live tensors just copied.
    policy.__class__ = A12MixtureMaskableActorCriticPolicy
    policy.bc_aux_mixture_spec = expected_spec
    model.policy_class = A12MixtureMaskableActorCriticPolicy
    model._bc_aux_circuit_spec = expected_spec
    return expected_spec


def _calibrate_bc_aux_adapter_weight(
        model, x, y, episode_id, masks) -> dict:
    """Initialize the exact on-policy mixture and verify nested safety.

    No data-dependent parameter fitting remains: ε is the preregistered 5% in
    every eligible state.  Fit/validation data are consumed only to prove that
    the visible predicate, masks, old-policy argmax and distribution identity
    agree.  The final held-out split is not read here.
    """
    import numpy as np
    import torch as th

    spec = _bc_aux_circuit_spec()
    _require(getattr(model, "_bc_aux_circuit_spec", None) == spec,
             "a12 adapter 校准前 spec 缺失/漂移")
    from leashed_ppo import A12MixtureMaskableActorCriticPolicy
    _require(
        isinstance(model.policy, A12MixtureMaskableActorCriticPolicy)
        and model.policy.bc_aux_mixture_spec == spec
        and model.policy_class is A12MixtureMaskableActorCriticPolicy,
        "a12 mixture policy class/spec 未原子安装")
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(episode_id)
    masks = np.asarray(masks)
    _require(
        x.ndim == 2 and x.shape[1] == 298
        and y.shape == groups.shape == (len(x),)
        and masks.shape == (len(x), 15)
        and masks.dtype == np.bool_,
        "a12 adapter 校准输入形状/dtype 异常")
    fit, validation = _bc_v2_fit_validation_indices(groups)
    visible_eligible = (
        (x[:, 0] >= spec["hp_low"] - spec["boundary_epsilon"])
        & (x[:, 0] < spec["hp_high"] - spec["boundary_epsilon"])
        & (x[:, _A12_CALIBRATION_DRINK_LATCH_FEATURE] >= 0.0)
        & masks[:, _BC_AUX_CIRCUIT_ACTION]
    )
    _require(
        np.array_equal(y == _BC_AUX_CIRCUIT_ACTION, visible_eligible),
        "a12 mixture 数据标签不等于可见 hp/latch/m12 谓词；"
        "拒绝用旧语义 demos 初始化")
    positive_fit = fit[y[fit] == _BC_AUX_CIRCUIT_ACTION]
    _require(len(positive_fit) >= 2
             and bool(masks[positive_fit, _BC_AUX_CIRCUIT_ACTION].all()),
             "a12 adapter nested-fit 正例不足/不合法")
    action = _BC_AUX_CIRCUIT_ACTION
    parameter_columns = tuple(
        int(value) for value in spec["gate_parameter_columns"])
    action_bias = model.policy.action_net.bias
    adapter_weight = model.policy.action_net.weight
    _require(
        bool((adapter_weight[
            action, list(parameter_columns)] == 0).all().item())
        and math.isclose(
            float(action_bias[action].detach().cpu()),
            float(spec["initial_gate_bias"]),
            rel_tol=0.0,
            abs_tol=2e-7,
        ),
        "a12 contextual gate 初始系数/bias 已漂移")

    device = model.device

    def positive_probability_summary(indices) -> dict:
        with th.no_grad():
            dist = model.policy.get_distribution(
                th.as_tensor(x[indices], device=device),
                action_masks=th.as_tensor(masks[indices], device=device))
            probabilities = dist.distribution.logits[:, action].exp()
            return {
                "min": float(probabilities.min().cpu()),
                "mean": float(probabilities.mean().cpu()),
                "max": float(probabilities.max().cpu()),
                "predicted_a12": int(
                    (dist.distribution.logits.argmax(dim=-1) == action)
                    .sum().cpu()),
            }

    target_probability = float(spec["initial_probability"])

    fit_positive_summary = positive_probability_summary(positive_fit)
    fit_positive_probability = fit_positive_summary["mean"]
    positive_validation = validation[
        y[validation] == _BC_AUX_CIRCUIT_ACTION]
    _require(len(positive_validation) >= 1,
             "a12 adapter nested-validation 无正例")
    validation_positive_summary = positive_probability_summary(
        positive_validation)
    validation_positive_probability = validation_positive_summary["mean"]
    candidate = _policy_head_snapshot(model.policy)
    fit_metrics = bc_aux_behavior_metrics(
        candidate, x[fit], y[fit], groups[fit], masks[fit],
        anchor_sd=model.bc_aux_root_anchor_sd, heldout_only=False,
        circuit_spec=spec)

    _require(
        math.isclose(
            fit_positive_probability, target_probability,
            rel_tol=0.0, abs_tol=5e-7)
        and math.isclose(
            fit_positive_summary["min"], target_probability,
            rel_tol=0.0, abs_tol=5e-7)
        and math.isclose(
            fit_positive_summary["max"], target_probability,
            rel_tol=0.0, abs_tol=5e-7)
        and fit_positive_summary["predicted_a12"] == 0
        and fit_metrics["predicted_a12"] == 0
        and fit_metrics["anchor"]["argmax_drift"] == 0.0
        and fit_metrics["fpr_12"] == 0.0
        and fit_metrics["high_hp_false_drink_rate"]
        == 0.0
        and fit_metrics["legal_negative_probability_12_mean"]
        <= _A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX
        and fit_metrics["legal_negative_probability_12_max"]
        <= _A12_LEGAL_NEGATIVE_PROBABILITY_MAX,
        "a12 adapter nested-fit 初始探索/安全门未过")
    validation_metrics = bc_aux_behavior_metrics(
        candidate, x[validation], y[validation],
        groups[validation], masks[validation],
        anchor_sd=model.bc_aux_root_anchor_sd, heldout_only=False,
        circuit_spec=spec)
    validation_gate = bc_aux_behavior_gate(
        validation_metrics, require_root_anchor=True,
        require_teacher_recall=False)
    _require(validation_gate["verdict"] == "PASS",
             "a12 adapter nested-validation 安全门未过:"
             f"{validation_gate['reasons']}")
    _require(
        math.isclose(
            validation_positive_probability, target_probability,
            rel_tol=0.0, abs_tol=5e-7)
        and math.isclose(
            validation_positive_summary["min"], target_probability,
            rel_tol=0.0, abs_tol=5e-7)
        and math.isclose(
            validation_positive_summary["max"], target_probability,
            rel_tol=0.0, abs_tol=5e-7)
        and validation_positive_summary["predicted_a12"] == 0
        and validation_metrics["predicted_a12"] == 0
        and validation_metrics["anchor"]["argmax_drift"] == 0.0,
        "a12 adapter nested-validation 探索概率/argmax 异常")
    return {
        "fit_pairs": int(len(fit)),
        "validation_pairs": int(len(validation)),
        "fit_positive_a12": int(len(positive_fit)),
        "validation_positive_a12": int(len(positive_validation)),
        "initializer": "exact-contextual-legal-support-mixture",
        "gate_feature_indices": list(spec["gate_feature_indices"]),
        "gate_parameter_columns":
            list(spec["gate_parameter_columns"]),
        "target_probability_12": target_probability,
        "fit_positive_probability_min_12":
            fit_positive_summary["min"],
        "fit_positive_probability_12": fit_positive_probability,
        "fit_positive_probability_max_12":
            fit_positive_summary["max"],
        "validation_positive_probability_min_12":
            validation_positive_summary["min"],
        "validation_positive_probability_12":
            validation_positive_probability,
        "validation_positive_probability_max_12":
            validation_positive_summary["max"],
        "initial_argmax_lower_bound": (
            (1.0 - target_probability) / 14.0
            - target_probability),
        "initial_gate_bias": float(
            action_bias[action].detach().cpu()),
        "gate_coefficients": [
            float(value)
            for value in adapter_weight[
                action, list(parameter_columns)].detach().cpu()
        ],
        "probability_min": float(spec["probability_min"]),
        "probability_max": float(spec["probability_max"]),
        "fit_metrics": fit_metrics,
        "validation_metrics": validation_metrics,
        "validation_gate": validation_gate,
        "candidate_policy_head_sha256":
            _policy_head_sha256(candidate),
    }


def _bc_aux_liveness_call_plan(total_steps: int, n_steps: int,
                               num_envs: int) -> dict:
    """把生产腿预算换算成 ``train()``/aux 调用数，禁止手写 244 漂移。"""
    quantum = int(n_steps) * int(num_envs)
    _require(quantum > 0 and int(total_steps) > 0
             and int(total_steps) % quantum == 0,
             "bc_aux liveness 预算必须按 rollout 量子整除")
    train_calls = int(total_steps) // quantum
    _require(_BC_AUX_UPDATE_EVERY >= 1,
             "bc_aux update_every 必须为正整数")
    aux_calls = len(range(0, train_calls, _BC_AUX_UPDATE_EVERY))
    return {
        "rollout_quantum": quantum,
        "train_calls": train_calls,
        "aux_optimizer_calls": aux_calls,
        "update_every": _BC_AUX_UPDATE_EVERY,
    }


def _simulate_bc_aux_liveness(
        model, *, bank, x, y, episode_id, masks,
        bc_aux_lambda: float, seed: int | None,
        call_plan: dict) -> dict:
    """在隔离 clone 上跑生产同数 aux 调用，只以 training episodes 裁门。

    这是环境点火前的必要条件探针：保留 checkpoint 中真实 policy、Adam
    moments、lr、batch size 与 persistent root；唯一拿掉的是 PPO/蒸馏
    梯度。探针绝不把模拟后的权重带回生产模型，也绝不读取 held-out 行。
    """
    import numpy as np
    from leashed_ppo import derive_bc_aux_rng

    _require(math.isfinite(bc_aux_lambda) and bc_aux_lambda > 0,
             "bc_aux liveness λ 必须是有限正数")
    _require(isinstance(call_plan, dict)
             and call_plan.get("update_every") == _BC_AUX_UPDATE_EVERY,
             "bc_aux liveness 调用计划与当前 objective 不一致")
    train_calls = int(call_plan.get("train_calls", -1))
    expected_aux_calls = int(call_plan.get("aux_optimizer_calls", -1))
    _require(train_calls > 0 and expected_aux_calls > 0,
             "bc_aux liveness 调用计划为空")

    bx, by, bm = bank
    root = _persistent_bc_aux_root_anchor(model)
    start = _policy_head_snapshot(model.policy)
    model.bc_aux_lambda = float(bc_aux_lambda)
    model.mount_bc_aux_demos(
        bx, by, bm, rng=derive_bc_aux_rng(seed))
    optimizer = model.policy.optimizer
    optimizer_state_entries = len(optimizer.state)
    optimizer_lrs = [
        float(group["lr"]) for group in optimizer.param_groups]
    _require(all(math.isfinite(value) and value > 0
                 for value in optimizer_lrs),
             "bc_aux liveness optimizer lr 非法")

    applied = 0
    last_loss = None
    for call_index in range(train_calls):
        # 镜像 LeashedMaskablePPO.train() 的调用计数与 due 判定；当前 rev5
        # update_every=1，生产 499,712/2,048 因而严格得到 244 次。
        model._bc_aux_train_calls += 1
        if call_index % _BC_AUX_UPDATE_EVERY != 0:
            continue
        aux_loss = model._apply_bc_aux_step()
        applied += 1
        last_loss = float(aux_loss.detach().cpu())
    _require(applied == expected_aux_calls,
             "bc_aux liveness 实际 aux 调用数与生产计划不一致:"
             f"{applied} != {expected_aux_calls}")

    groups = np.asarray(episode_id)
    training = _bc_v2_training_indices(groups)
    heldout = _bc_v2_holdout_indices(groups)
    train_episodes = np.unique(groups[training])
    heldout_episodes = np.unique(groups[heldout])
    _require(not np.intersect1d(
        train_episodes, heldout_episodes).size,
        "bc_aux liveness training/heldout episode 泄漏")
    candidate = _policy_head_snapshot(model.policy)
    metrics = bc_aux_behavior_metrics(
        candidate, np.asarray(x)[training], np.asarray(y)[training],
        groups[training], np.asarray(masks)[training],
        anchor_sd=root, heldout_only=False)
    gate = bc_aux_behavior_gate(
        metrics, require_root_anchor=True)
    episode_digest = hashlib.sha256(
        np.asarray(train_episodes, dtype=np.int64).tobytes()).hexdigest()
    return {
        "status": "PASS" if gate["verdict"] == "PASS" else "FAIL",
        "simulation": "isolated-aux-only-necessary-condition",
        "evaluation_scope": "bc-v2-training-episodes-only",
        "heldout_rows_consumed": 0,
        "split": {
            "training_pairs": int(len(training)),
            "training_episodes": int(len(train_episodes)),
            "training_episode_ids_sha256": episode_digest,
            "heldout_pairs_excluded": int(len(heldout)),
            "heldout_episodes_excluded": int(len(heldout_episodes)),
            "episode_disjoint": True,
        },
        "bank": {
            "pairs": int(len(by)),
            "true_a12": int((np.asarray(by) == 12).sum()),
            "hard_negatives": int((np.asarray(by) != 12).sum()),
        },
        "optimizer": {
            "class": (f"{type(optimizer).__module__}."
                      f"{type(optimizer).__qualname__}"),
            "state_entries_at_start": int(optimizer_state_entries),
            "learning_rates_at_start": optimizer_lrs,
            "max_grad_norm": float(model.max_grad_norm),
        },
        "policy": {
            "start_head_sha256": _policy_head_sha256(start),
            "root_head_sha256": _policy_head_sha256(root),
            "simulated_end_head_sha256": _policy_head_sha256(candidate),
        },
        "calls": {**call_plan, "actual_aux_optimizer_calls": applied},
        "last_unscaled_aux_loss": last_loss,
        "metrics": metrics,
        "gate": gate,
    }


def _run_bc_aux_policy_gradient_canary(
        model, *, x, y, episode_id, masks, spec: dict) -> dict:
    """Prove the real distribution/optimizer path can raise eligible p(a12).

    The canary uses nested-validation positives only, applies one genuine
    optimizer step through ``get_distribution().log_prob()``, and then restores
    both policy and optimizer bit-for-bit.  It is a wiring/learnability
    necessary condition, not evidence that the live environment will assign a
    positive advantage to drinking.
    """
    import numpy as np
    import torch as th

    groups = np.asarray(episode_id)
    labels = np.asarray(y)
    observations = np.asarray(x, dtype=np.float32)
    action_masks = np.asarray(masks)
    _, validation = _bc_v2_fit_validation_indices(groups)
    action = int(spec["action_index"])
    positive = validation[labels[validation] == action]
    _require(
        len(positive) > 0
        and bool(action_masks[positive, action].all()),
        "a12 policy-gradient canary 缺 nested-validation 合法正例")

    # Keep the probe compact and deterministic while spanning more than one
    # episode whenever the validation split permits it.
    positive = positive[:min(256, len(positive))]
    obs_t = th.as_tensor(observations[positive], device=model.device)
    masks_t = th.as_tensor(
        action_masks[positive], dtype=th.bool, device=model.device)
    actions_t = th.full(
        (len(positive),), action, dtype=th.long, device=model.device)
    policy_state = {
        key: value.detach().clone()
        for key, value in model.policy.state_dict().items()
    }
    optimizer = model.policy.optimizer
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    was_training = bool(model.policy.training)
    start_head = _policy_head_snapshot(model.policy)
    gate_bias_parameter = model.policy.action_net.bias
    restored = False
    try:
        model.policy.set_training_mode(True)
        with th.no_grad():
            before_distribution = model.policy.get_distribution(
                obs_t, action_masks=masks_t)
            before_probability = float(
                before_distribution.distribution.logits[:, action]
                .exp().mean().cpu())
            bias_before = float(
                gate_bias_parameter[action].detach().cpu())

        optimizer.zero_grad()
        distribution = model.policy.get_distribution(
            obs_t, action_masks=masks_t)
        loss = -distribution.log_prob(actions_t).mean()
        _require(bool(th.isfinite(loss).item()),
                 "a12 policy-gradient canary loss 非有限")
        loss.backward()
        gradient = gate_bias_parameter.grad
        _require(
            gradient is not None
            and gradient.shape == gate_bias_parameter.shape
            and bool(th.isfinite(gradient).all().item()),
            "a12 policy-gradient canary gate bias 梯度缺失/非有限")
        bias_gradient = float(gradient[action].detach().cpu())
        _require(
            bias_gradient < 0.0,
            "a12 policy-gradient canary 有利优势未产生提升 gate 的梯度")
        circuit_snapshot = model._protect_bc_aux_circuit_before_step()
        gradient_norm = float(th.nn.utils.clip_grad_norm_(
            model.policy.parameters(), model.max_grad_norm).detach().cpu())
        _require(math.isfinite(gradient_norm) and gradient_norm > 0.0,
                 "a12 policy-gradient canary 梯度范数非法")
        optimizer.step()
        model._project_bc_aux_adapter_weight()
        model._assert_bc_aux_circuit_unchanged(circuit_snapshot)

        with th.no_grad():
            after_distribution = model.policy.get_distribution(
                obs_t, action_masks=masks_t)
            after_probability = float(
                after_distribution.distribution.logits[:, action]
                .exp().mean().cpu())
            bias_after = float(
                gate_bias_parameter[action].detach().cpu())
        probability_delta = after_probability - before_probability
        bias_delta = bias_after - bias_before
        movement_required = (
            before_probability
            < float(spec["probability_max"]) - 1e-6)
        _require(
            after_probability >= before_probability
            and (not movement_required or probability_delta > 0.0)
            and (not movement_required or bias_delta > 0.0),
            "a12 policy-gradient canary optimizer step 未提升 eligible p(a12)")
        end_head = _policy_head_snapshot(model.policy)
        return {
            "schema_version": "a12-policy-gradient-canary/1",
            "scope": "bc-v2-nested-validation-positive-only",
            "pairs": int(len(positive)),
            "heldout_rows_consumed": 0,
            "objective": "negative-mean-log-probability-action12",
            "optimizer_steps": 1,
            "movement_required": movement_required,
            "probability_12_before": before_probability,
            "probability_12_after": after_probability,
            "probability_12_delta": probability_delta,
            "gate_bias_before": bias_before,
            "gate_bias_after": bias_after,
            "gate_bias_delta": bias_delta,
            "gate_bias_gradient": bias_gradient,
            "gradient_norm_before_clip": gradient_norm,
            "start_policy_head_sha256": _policy_head_sha256(start_head),
            "stepped_policy_head_sha256": _policy_head_sha256(end_head),
            "state_restored": True,
        }
    finally:
        model.policy.load_state_dict(policy_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        model.policy.set_training_mode(was_training)
        restored = all(
            th.equal(model.policy.state_dict()[key], value)
            for key, value in policy_state.items())
        if not restored:
            raise RuntimeError(
                "a12 policy-gradient canary 未恢复隔离 policy 状态")


def _simulate_bc_aux_circuit_liveness(
        model, *, x, y, episode_id, masks,
        learning_rate: float, call_plan: dict) -> dict:
    """Install/gate the exact rev9 contextual on-policy mixture adapter."""
    start = _policy_head_snapshot(model.policy)
    root = _persistent_bc_aux_root_anchor(model)
    existing = getattr(model, "_bc_aux_circuit_spec", None) is not None
    if not existing:
        _require(_policy_head_sha256(start) == _policy_head_sha256(root),
                 "rev9 首次安装起点不是未改写 V28 root")
    spec = _expand_policy_with_bc_aux_circuit(model)
    if existing:
        _require(not callable(model.learning_rate)
                 and math.isclose(
                     float(model.learning_rate), learning_rate,
                     rel_tol=0.0, abs_tol=1e-12),
                 "rev9 continuation 学习率漂移")
        frozen_values = []
        for parameter, protected in (
                model._bc_aux_circuit_protected_tensors()):
            frozen_values.append(parameter.detach()[protected])
            state = model.policy.optimizer.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                moment = state.get(key)
                if moment is not None:
                    _require(
                        moment.shape == parameter.shape
                        and bool((moment[protected] == 0).all().item()),
                        f"rev9 continuation 受保护 Adam {key} 非零/错形")
        _require(
            frozen_values
            and all(bool((value == 0).all().item())
                    for value in frozen_values),
            "rev9 continuation 固定 adapter 张量不再是 canonical zero")
        action = int(spec["action_index"])
        columns = [
            int(value) for value in spec["gate_parameter_columns"]]
        gate_coefficients = model.policy.action_net.weight[
            action, columns].detach()
        gate_bias = model.policy.action_net.bias[action].detach()
        limit = float(spec["gate_parameter_abs_max"])
        _require(
            bool(gate_coefficients.isfinite().all().item())
            and bool(gate_bias.isfinite().item())
            and bool((gate_coefficients.abs() <= limit).all().item())
            and abs(float(gate_bias.cpu())) <= limit,
            "rev9 continuation contextual gate 非有限/越界")
        fit, validation = _bc_v2_fit_validation_indices(episode_id)
        candidate = _policy_head_snapshot(model.policy)
        metrics = bc_aux_behavior_metrics(
            candidate,
            x[validation], y[validation], episode_id[validation],
            masks[validation],
            anchor_sd=model.bc_aux_root_anchor_sd,
            heldout_only=False,
            circuit_spec=spec,
        )
        gate = bc_aux_behavior_gate(
            metrics, require_root_anchor=True,
            require_teacher_recall=False)
        calibration = {
            "initializer": "preserved-continuation",
            "gate_coefficients": [
                float(value) for value in gate_coefficients.cpu()],
            "gate_bias": float(gate_bias.cpu()),
            "fit_pairs_excluded_from_retuning": int(len(fit)),
            "validation_pairs": int(len(validation)),
            "validation_metrics": metrics,
            "validation_gate": gate,
            "candidate_policy_head_sha256":
                _policy_head_sha256(candidate),
        }
    else:
        _reset_policy_optimizer(model, learning_rate)
        calibration = _calibrate_bc_aux_adapter_weight(
            model, x, y, episode_id, masks)
    policy_gradient_canary = _run_bc_aux_policy_gradient_canary(
        model, x=x, y=y, episode_id=episode_id, masks=masks, spec=spec)
    candidate = _policy_head_snapshot(model.policy)
    # Calibration already gates nested validation.  This duplicate explicit
    # verdict keeps the preflight receipt easy to verify without executing
    # code from an untrusted JSON field.
    gate = calibration["validation_gate"]
    return {
        "status": "PASS" if gate["verdict"] == "PASS" else "FAIL",
        "simulation":
            "isolated-exact-mixture-with-policy-gradient-canary",
        "installation": (
            "preserved-continuation" if existing else "first-install"),
        "evaluation_scope": "bc-v2-nested-validation-only",
        "heldout_rows_consumed": 0,
        "circuit": {
            **spec,
            "king_support": _BC_AUX_CIRCUIT_KING_SUPPORT,
        },
        "optimizer": {
            "class": (
                f"{type(model.policy.optimizer).__module__}."
                f"{type(model.policy.optimizer).__qualname__}"),
            "state_entries_at_start": int(
                len(model.policy.optimizer.state)),
            "learning_rates_at_start": [
                float(group["lr"])
                for group in model.policy.optimizer.param_groups
            ],
            "reset_after_topology_change": not existing,
        },
        "policy": {
            "start_head_sha256": _policy_head_sha256(start),
            "root_head_sha256": _policy_head_sha256(root),
            "grafted_head_sha256": _policy_head_sha256(candidate),
            "actor_width_before": (
                _BC_AUX_CIRCUIT_EXPANDED_WIDTH if existing
                else _BC_AUX_CIRCUIT_BASE_WIDTH),
            "actor_width_after": _BC_AUX_CIRCUIT_EXPANDED_WIDTH,
        },
        "calls": {
            "planned_train_calls": int(call_plan["train_calls"]),
            "aux_optimizer_calls": 0,
            "policy_gradient_canary_calls": 1,
            "initial_adapter_calibrations": 0 if existing else 1,
            "trainable_adapter_parameters": 5,
        },
        "policy_gradient_canary": policy_gradient_canary,
        "calibration": calibration,
        "metrics": calibration["validation_metrics"],
        "gate": gate,
    }


def _write_bc_aux_behavior_receipt(path: pathlib.Path, record: dict) -> None:
    """FAIL 亦原子落回执；canonical 模型只在 PASS 后发布。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        payload = json.dumps(
            record, ensure_ascii=False, sort_keys=True,
            allow_nan=False).encode("utf-8")
        with open(tmp, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_PUBLICATION_PROVENANCE_KEYS = frozenset({
    "protocol_version", "implementation_sha256", "manager_npz_sha256",
    "resume_checkpoint_sha256", "teacher_sha256",
    "bc_aux_demos_sha256", "bc_aux_liveness_preflight_sha256",
    "training_contract_sha256", "start_steps", "target_global_steps",
    "seed", "optimizer_reset", "target_kl", "distill_beta",
    "bc_aux_lambda", "bc_aux_mode", "calib_record_only",
})


def _validate_publication_provenance(
        provenance: dict, *, demos_sha256: str, final_step: int) -> dict:
    """严格冻结最终模型的训练谱系，供评测入口现场复验。"""
    _require(isinstance(provenance, dict)
             and set(provenance) == _PUBLICATION_PROVENANCE_KEYS,
             "bc_aux 最终发布谱系字段不完整")
    _require(provenance["protocol_version"] == PROTOCOL_VERSION,
             "bc_aux 最终发布谱系协议不匹配")
    for key in (
            "implementation_sha256", "manager_npz_sha256",
            "resume_checkpoint_sha256", "teacher_sha256",
            "bc_aux_demos_sha256", "bc_aux_liveness_preflight_sha256",
            "training_contract_sha256"):
        _require(_is_sha256(provenance[key]),
                 f"bc_aux 最终发布谱系 {key} 非法")
    _require(provenance["bc_aux_demos_sha256"] == demos_sha256,
             "bc_aux 最终发布谱系 demos 与行为门不一致")
    start = provenance["start_steps"]
    target = provenance["target_global_steps"]
    _require(_is_plain_int(start) and _is_plain_int(target)
             and 0 <= start < target == int(final_step),
             "bc_aux 最终发布谱系步数不闭合")
    seed = provenance["seed"]
    _require(seed is None or (_is_plain_int(seed) and 0 <= seed < 2**32),
             "bc_aux 最终发布谱系 seed 非法")
    _require(isinstance(provenance["optimizer_reset"], bool),
             "bc_aux 最终发布谱系 optimizer_reset 必须是 bool")
    target_kl = provenance["target_kl"]
    _require(target_kl is None or (
        isinstance(target_kl, (int, float))
        and not isinstance(target_kl, bool)
        and math.isfinite(float(target_kl))
        and float(target_kl) > 0),
        "bc_aux 最终发布谱系 target_kl 非法")
    value = provenance["distill_beta"]
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0,
        "bc_aux 最终发布谱系 distill_beta 必须是有限正数")
    _require(
        provenance["bc_aux_mode"]
        == "expanded-trainable-a12-contextual-mixture"
        and isinstance(provenance["bc_aux_lambda"], (int, float))
        and not isinstance(provenance["bc_aux_lambda"], bool)
        and float(provenance["bc_aux_lambda"]) == 0.0,
        "bc_aux 最终发布谱系必须绑定 rev10 contextual mixture/λ=0")
    _require(provenance["calib_record_only"] is False,
             "bc_aux 正式发布禁止 calib_record_only 绕过硬门")
    return dict(provenance)


def _run_bc_aux_liveness_preflight(
        *, run_dir: pathlib.Path, args,
        resume_checkpoint_bytes: bytes,
        resume_checkpoint_sha256: str,
        bank, x, y, episode_id, masks,
        demos_sha256: str, manager_npz_sha256: str,
        implementation_sha256: str, batch_size: int) -> tuple[dict, str]:
    """环境点火前在隔离 checkpoint clone 上执行 aux 必要条件预检。

    FAIL/ERROR 只发布 sibling receipt，主动移除 canonical model_final 并抛错；
    PASS clone 也立即丢弃，绝不把预检权重或 RNG 状态渗入真实训练。
    """
    import numpy as np
    import torch as th
    from leashed_ppo import LeashedMaskablePPO

    for label, value in (
            ("resume_checkpoint_sha256", resume_checkpoint_sha256),
            ("demos_sha256", demos_sha256),
            ("manager_npz_sha256", manager_npz_sha256),
            ("implementation_sha256", implementation_sha256)):
        _require(_is_sha256(value),
                 f"bc_aux liveness 输入 {label} 非法")
    _require(isinstance(resume_checkpoint_bytes, bytes)
             and len(resume_checkpoint_bytes) > 0,
             "bc_aux liveness 缺已冻结 resume checkpoint 字节")
    structural = _bc_aux_structural_active(args)
    call_plan = (
        {
            "rollout_quantum": int(args.n_steps * args.num_envs),
            "train_calls": int(
                args.total_steps // (args.n_steps * args.num_envs)),
            "aux_optimizer_calls": 0,
            "policy_gradient_canary_calls": 1,
            "initial_adapter_calibrations": 1,
            "trainable_adapter_parameters": 5,
        }
        if structural else _bc_aux_liveness_call_plan(
            args.total_steps, args.n_steps, args.num_envs)
    )
    receipt_path = pathlib.Path(run_dir) / "bc_aux_liveness_preflight.json"
    base = {
        "schema_version": _BC_AUX_LIVENESS_PREFLIGHT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "objective_revision": _BC_AUX_OBJECTIVE_REVISION,
        "inputs": {
            "resume_checkpoint_sha256": resume_checkpoint_sha256,
            "demos_sha256": demos_sha256,
            "manager_npz_sha256": manager_npz_sha256,
            "implementation_sha256": implementation_sha256,
        },
        "config": {
            "bc_aux_lambda": float(args.bc_aux_lambda),
            "seed": args.seed,
            "device": str(args.device),
            "learning_rate": float(args.lr),
            "distill_beta": float(args.distill_beta),
            "target_kl": args.target_kl,
            "reset_optimizer": bool(args.reset_optimizer),
            "n_steps": int(args.n_steps),
            "num_envs": int(args.num_envs),
            "batch_size": int(batch_size),
            "total_steps": int(args.total_steps),
            "mechanism": (
                "expanded-trainable-a12-contextual-mixture"
                if structural else "legacy-gradient-aux"),
            **({
                "circuit": {
                    **_bc_aux_circuit_spec(),
                    "king_support":
                        _BC_AUX_CIRCUIT_KING_SUPPORT,
                },
            } if structural else {
                "positive_fraction": _BC_AUX_POSITIVE_FRACTION,
                "positive_target": _BC_AUX_POSITIVE_TARGET,
                "negative_target": _BC_AUX_NEGATIVE_TARGET,
                "anchor_kl_coef": _BC_AUX_ANCHOR_KL_COEF,
            }),
            **call_plan,
        },
    }

    # SB3 load(seed=...) 会重播全局 Python/NumPy/Torch 种子。预检是隔离探针，
    # 必须在真实 model/env 建立前逐流恢复，不能改变随后生产轨迹。
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = th.get_rng_state().clone()
    cuda_states = (
        th.cuda.get_rng_state_all()
        if th.cuda.is_available() else None)
    record = None
    failure = None
    try:
        load_kw = {
            "teacher_path": None,
            "teacher_sha256": None,
        }
        if args.seed is not None:
            load_kw["seed"] = args.seed
        model = LeashedMaskablePPO.load(
            io.BytesIO(resume_checkpoint_bytes), env=None,
            device=args.device, **load_kw)
        if structural:
            # Whether the isolated clone is a first install or a genuine
            # continuation is checkpoint state, not a launcher assumption.
            # Keep the preregistered config and the executed ``calls`` receipt
            # identical so a preserved adapter cannot be falsely described as
            # having been recalibrated.
            initial_calibrations = int(
                getattr(model, "_bc_aux_circuit_spec", None) is None)
            call_plan["initial_adapter_calibrations"] = (
                initial_calibrations)
            base["config"]["initial_adapter_calibrations"] = (
                initial_calibrations)
        saved_lr = model.learning_rate
        if structural:
            # Topology changes inside the exact structural simulation; its
            # helper resets the optimizer immediately afterwards.
            pass
        elif args.reset_optimizer:
            _reset_policy_optimizer(model, args.lr)
        else:
            _require(not callable(saved_lr)
                     and math.isclose(float(saved_lr), args.lr,
                                      rel_tol=0, abs_tol=1e-12),
                     "bc_aux liveness resume 学习率与生产 CLI 不一致:"
                     f"{saved_lr} != {args.lr}")
            _require(all(math.isclose(
                float(group["lr"]), args.lr,
                rel_tol=0, abs_tol=1e-12)
                for group in model.policy.optimizer.param_groups),
                "bc_aux liveness checkpoint optimizer 当前 lr 与生产不一致")
        model.target_kl = args.target_kl
        _require(math.isclose(float(model.ent_coef), args.ent_coef,
                              rel_tol=0, abs_tol=1e-12)
                 and math.isclose(float(model.gamma), args.gamma,
                                  rel_tol=0, abs_tol=1e-12)
                 and model.n_steps == args.n_steps
                 and model.batch_size == batch_size,
                 "bc_aux liveness clone 与生产冻结配方不一致")
        _validate_model_recipe(
            model, expected_target_kl=args.target_kl)
        result = (
            _simulate_bc_aux_circuit_liveness(
                model, x=x, y=y, episode_id=episode_id,
                masks=masks, learning_rate=args.lr,
                call_plan=call_plan)
            if structural else
            _simulate_bc_aux_liveness(
                model, bank=bank, x=x, y=y, episode_id=episode_id,
                masks=masks, bc_aux_lambda=args.bc_aux_lambda,
                seed=args.seed, call_plan=call_plan)
        )
        record = {**base, **result}
        del model
    except Exception as exc:
        failure = exc
        record = {
            **base,
            "status": "ERROR",
            "simulation": (
                "isolated-exact-mixture-with-policy-gradient-canary"
                if structural else
                "isolated-aux-only-necessary-condition"),
            "evaluation_scope": (
                "bc-v2-nested-validation-only"
                if structural else "bc-v2-training-episodes-only"),
            "heldout_rows_consumed": 0,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        th.set_rng_state(torch_state)
        if cuda_states is not None:
            th.cuda.set_rng_state_all(cuda_states)

    _write_bc_aux_behavior_receipt(receipt_path, record)
    receipt_sha256 = hashlib.sha256(
        receipt_path.read_bytes()).hexdigest()
    if record["status"] != "PASS":
        (pathlib.Path(run_dir) / "model_final.zip").unlink(
            missing_ok=True)
        if failure is not None:
            raise ValueError(
                "bc_aux liveness preflight ERROR，环境未点火、拒绝训练；"
                f"详见 {receipt_path}") from failure
        _require(
            False,
            "bc_aux liveness preflight FAIL，环境未点火、拒绝训练:"
            f"{record['gate']['reasons']}；详见 {receipt_path}")
    return record, receipt_sha256


def _publish_model_final_with_bc_aux_gate(
        model, destination: str | pathlib.Path, *,
        x, y, episode_id, masks, demos_sha256: str,
        anchor_sd: dict, publication_provenance: dict) -> dict:
    """Publish only after the adapter's held-out *safety* gate passes.

    Teacher recall and deterministic a12 deployment are intentionally not
    objectives in rev10: PPO decides from combat return whether a12 belongs
    in its greedy policy.  This transaction gates spill, drift and resource
    safety, records whether PPO actually sampled enough native-certified a12
    transitions, and leaves efficacy to paired native evaluation.
    """
    _require(_is_sha256(demos_sha256),
             "bc_aux 最终行为门缺 demos_sha256 绑定")
    _require(isinstance(anchor_sd, dict)
             and set(anchor_sd) == set(_POLICY_HEAD_KEYS),
             "bc_aux 最终行为门缺挂载起点策略锚")
    final = pathlib.Path(destination)
    if final.suffix.lower() != ".zip":
        final = pathlib.Path(f"{final}.zip")
    current_sd = _policy_head_snapshot(model.policy)
    metrics = bc_aux_behavior_metrics(
        current_sd, x, y, episode_id, masks,
        anchor_sd=anchor_sd, heldout_only=True,
        circuit_spec=_bc_aux_circuit_spec())
    gate = bc_aux_behavior_gate(
        metrics, require_root_anchor=True,
        require_teacher_recall=False,
        require_deployable_a12=False)
    provenance = _validate_publication_provenance(
        publication_provenance, demos_sha256=demos_sha256,
        final_step=int(model.num_timesteps))
    eligible_states = getattr(model, "_bc_aux_eligible_states", 0)
    requested_a12 = getattr(model, "_bc_aux_requested_a12", 0)
    sampled_a12 = getattr(model, "_bc_aux_sampled_a12", 0)
    rejected_a12 = getattr(model, "_bc_aux_rejected_a12", 0)
    unexpected_sampled_a12 = getattr(
        model, "_bc_aux_unexpected_sampled_a12", 0)
    for name, value in (
        ("eligible_states", eligible_states),
        ("requested_a12", requested_a12),
        ("sampled_a12", sampled_a12),
        ("rejected_a12", rejected_a12),
        ("unexpected_sampled_a12", unexpected_sampled_a12),
    ):
        _require(
            _is_plain_int(value) and value >= 0,
            f"bc_aux rollout 计数 {name} 非普通非负整数",
        )
    _require(
        requested_a12 == sampled_a12 + rejected_a12
        and requested_a12 <= eligible_states,
        "bc_aux rollout 请求/执行/拒绝计数不闭合",
    )
    expected_a12_mass = getattr(
        model, "_bc_aux_expected_a12_mass", 0.0)
    _require(
        isinstance(expected_a12_mass, (int, float))
        and not isinstance(expected_a12_mass, bool)
        and math.isfinite(float(expected_a12_mass))
        and 0.0 <= float(expected_a12_mass) <= float(eligible_states),
        "bc_aux rollout expected_a12_mass 非有限数或超出 eligible 闭包",
    )
    exploration_reasons = []
    if not math.isfinite(expected_a12_mass) \
            or expected_a12_mass < _BC_AUX_MIN_EXPECTED_A12_SAMPLES:
        exploration_reasons.append(
            f"expected_a12_mass<{_BC_AUX_MIN_EXPECTED_A12_SAMPLES}")
    if sampled_a12 < _BC_AUX_MIN_ACTUAL_A12_SAMPLES:
        exploration_reasons.append(
            f"sampled_a12<{_BC_AUX_MIN_ACTUAL_A12_SAMPLES}")
    if unexpected_sampled_a12 != 0:
        exploration_reasons.append("unexpected_sampled_a12!=0")
    exploration_status = (
        "INFORMATIVE" if not exploration_reasons
        else "INSUFFICIENT_OR_INVALID")
    record = {
        "schema_version": _BC_AUX_BEHAVIOR_RECEIPT_SCHEMA_VERSION,
        "step": int(model.num_timesteps),
        "demos_sha256": demos_sha256,
        "objective_revision": _BC_AUX_OBJECTIVE_REVISION,
        "evaluation_scope": "original-bc-v2-heldout-episodes",
        "mask_mode": "bc-v2-recorded",
        "anchor": {
            "identity": "bc-aux-root-policy",
            "policy_head_sha256": _policy_head_sha256(anchor_sd),
        },
        "candidate_policy_head_sha256": _policy_head_sha256(current_sd),
        "provenance": provenance,
        "metrics": metrics,
        "gate": gate,
        "exploration_evidence": {
            "eligible_states": eligible_states,
            "expected_a12_mass": float(expected_a12_mass),
            "requested_a12": requested_a12,
            "sampled_a12": sampled_a12,
            "rejected_a12": rejected_a12,
            "unexpected_sampled_a12": unexpected_sampled_a12,
            "minimum_expected_a12_mass":
                _BC_AUX_MIN_EXPECTED_A12_SAMPLES,
            "minimum_actual_a12_samples":
                _BC_AUX_MIN_ACTUAL_A12_SAMPLES,
            "information_status": exploration_status,
            "reasons": exploration_reasons,
        },
        "publication": (
            "GATE_PASS_PENDING"
            if gate["verdict"] == "PASS" and not exploration_reasons
            else "REFUSED"),
        "model_sha256": None,
        "save_error": None,
    }
    receipt = final.parent / "bc_aux_behavior_receipt.json"
    if gate["verdict"] != "PASS" or exploration_reasons:
        _write_bc_aux_behavior_receipt(receipt, record)
        _require(False,
                 "bc_aux 最终 held-out/探索证据硬门 FAIL，拒绝发布"
                 f" model_final:{gate['reasons'] + exploration_reasons}；"
                 f"详见 {receipt}")

    # 不得先把 gate PASS 写成“已发布”：磁盘满/CRC/finite 校验失败时那会
    # 留下一张成功回执却没有模型。canonical zip 成功且重新哈希后，才把
    # PUBLISHED+model_sha256 原子落盘。失败回执明确为 SAVE_FAILED。
    try:
        _atomic_save_model(model, final)
        model_sha256 = hashlib.sha256(final.read_bytes()).hexdigest()
        _require(_is_sha256(model_sha256),
                 "model_final 发布后 SHA256 计算异常")
        record["model_sha256"] = model_sha256
        record["publication"] = "PUBLISHED"
        _write_bc_aux_behavior_receipt(receipt, record)
    except Exception as exc:
        # 若模型已写而最终回执未能提交，宁可撤掉无证 canonical；checkpoint
        # 恢复件仍在 ckpt/，不会因这里 fail-closed 丢掉训练进度。
        final.unlink(missing_ok=True)
        record["publication"] = "SAVE_FAILED"
        record["model_sha256"] = None
        record["save_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        try:
            _write_bc_aux_behavior_receipt(receipt, record)
        except Exception:
            pass
        raise
    return record


# ---- E4 契约 rev5 双键 + E6 探针集钉死(PREREG-内容案 E4/E6) ----


def _contract_dry_curriculum(args):
    """E4 rev5 键:课⑤不在位 = "disabled";在位 = {"schedule": CLI 字面表述}。

    载荷取 --dry-curriculum-schedule 之 CLI 字面串(与 D3 逐腿附加项列同源,
    L-cur/L-full 携主表;L-base 无 schedule → disabled)。schedule 字面即
    全表语义真源(_parse_dry_curriculum_schedule 确定性展开),resume 对账
    由字符串相等承载。
    """
    return ({"schedule": str(args.dry_curriculum_schedule)}
            if args.dry_curriculum_schedule else "disabled")


def _contract_bc_aux(args, bc_aux_demos_sha256):
    """rev12 key: bind the contextual mixture and its immutable data.

    The rejected rev5 gradient mechanism is deliberately not representable in
    a new production contract.
    """
    if not _bc_aux_active(args):
        return "disabled"
    _require(_bc_aux_structural_active(args)
             and float(args.bc_aux_lambda) == 0.0,
             "rev12 contract 只接受 structural graft")
    _require(_is_sha256(bc_aux_demos_sha256),
             "④乙在位但缺 bc-worker-v2 示范集 sha256,拒绝写入 rev5 契约")
    return {"mode": "expanded-trainable-a12-contextual-mixture",
            "lambda": float(args.bc_aux_lambda),
            "demos_sha256": bc_aux_demos_sha256,
            "objective_revision": _BC_AUX_OBJECTIVE_REVISION,
            "circuit": {
                **_bc_aux_circuit_spec(),
            },
            "king_support": _BC_AUX_CIRCUIT_KING_SUPPORT,
            "aux_optimizer_calls_per_rollout": 0,
            "initial_calibration":
                "exact-five-percent-contextual-legal-support-mixture",
            "trainable_adapter_parameters": 5,
            "post_step_projection": {
                "gate_parameter_abs_max":
                    _BC_AUX_CIRCUIT_GATE_PARAMETER_ABS_MAX,
                "probability_min":
                    _BC_AUX_CIRCUIT_PROBABILITY_MIN,
                "probability_max":
                    _BC_AUX_CIRCUIT_PROBABILITY_MAX,
            },
            "liveness_preflight": bool(args.bc_aux_liveness_preflight)}


def _assert_bc_v1_demos_frozen(path: str | pathlib.Path) -> str:
    """兼容函数名：把 BC-v1 demos 绑定到当前严格 PASS 回执。

    历史常量只服务 protocol-v3 driver 导入兼容，不能证明 v4 语义。这里先
    验 policy/report 的 protocol、implementation、generator 与重放证据，
    再要求报告 demos_sha256 等于现场字节；因此旧报告自然 fail-closed，
    新 v4 重采件也不会被旧 3bf8… 永久挡住。
    """
    demos = pathlib.Path(path)
    actual = _capture_file_sha256(demos, "BC-v1 demos(当前 PASS 回执绑定)")
    policy = demos.with_name("policy_sd.pt")
    _require(policy.is_file(), f"BC-v1 当前权重缺失/不可读: {policy}")
    report = _validate_bc_report(policy, "data_gate")
    expected = report.get("demos_sha256")
    _require(_is_sha256(expected),
             "BC-v1 当前 PASS 回执缺 demos_sha256")
    _require(actual == expected,
             "BC-v1 demos 与当前 PASS 回执字节漂移:"
             f"{actual} != {expected}")
    return actual


def _validate_args(args) -> None:
    _require(args.total_steps > 0, "--total-steps 必须 > 0")
    _require(args.num_envs > 0, "--num-envs 必须 > 0")
    _require(args.n_steps > 0, "--n-steps 必须 > 0")
    rollout_quantum = args.n_steps * args.num_envs
    _require(args.total_steps % rollout_quantum == 0,
             "--total-steps 必须能被 n_steps * num_envs 整除，"
             f"否则 SB3 会静默向上多采样（当前量子 {rollout_quantum}）")
    _require(args.max_steps > 0, "--max-steps 必须 > 0")
    _require(math.isfinite(args.lr) and args.lr > 0, "--lr 必须是有限正数")
    _require(math.isfinite(args.gamma) and 0 <= args.gamma <= 1,
             "--gamma 必须在 [0, 1] 内")
    _require(math.isfinite(args.ent_coef) and args.ent_coef >= 0,
             "--ent-coef 必须是有限非负数")
    _require(math.isfinite(args.distill_beta) and args.distill_beta >= 0,
             "--distill-beta 必须是有限非负数")
    action14_logit_bonus = float(getattr(
        args, "worker_action14_logit_bonus", 0.0))
    _require(
        math.isfinite(action14_logit_bonus)
        and 0.0 <= action14_logit_bonus <= 10.0,
        "--worker-action14-logit-bonus 必须是 [0,10] 内有限数",
    )
    _require(
        action14_logit_bonus == 0.0
        or (
            args.worker
            and args.algo == "mppo"
            and _worker_policy_observation_view(args)
            == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
        ),
        "--worker-action14-logit-bonus 非零只适用于 "
        "dual-v4 asymmetric Worker MaskablePPO",
    )
    distill_anneal_actor_rollouts = int(getattr(
        args, "distill_anneal_actor_rollouts", 0))
    _require(
        distill_anneal_actor_rollouts == 0
        or distill_anneal_actor_rollouts >= 2,
        "--distill-anneal-actor-rollouts 必须为 0 或 >=2",
    )
    _require(
        distill_anneal_actor_rollouts == 0
        or (
            args.distill_beta > 0
            and args.worker
            and args.algo == "mppo"
        ),
        "--distill-anneal-actor-rollouts 只适用于 "
        "β>0 的 Worker MaskablePPO",
    )
    _require(args.target_kl is None
             or (math.isfinite(args.target_kl) and args.target_kl > 0),
             "--target-kl 给定时必须是有限正数")
    _require(not args.reset_optimizer or bool(args.resume_from),
             "--reset-optimizer 只适用于 --resume-from continuation")
    reset_worker_critic = bool(getattr(
        args, "reset_worker_critic", False))
    critic_warmup_steps = int(getattr(
        args, "critic_warmup_steps", 0))
    gradient_clip_mode = getattr(
        args, "gradient_clip_mode", "global")
    _require(
        gradient_clip_mode in {
            "global",
            "separate-actor-critic-v1",
            "separate-root-context-critic-v2",
        },
        "--gradient-clip-mode 不在注册模式内",
    )
    _require(critic_warmup_steps >= 0,
             "--critic-warmup-steps 不能为负")
    if reset_worker_critic:
        _require(
            args.worker and args.algo == "mppo" and bool(args.resume_from),
            "--reset-worker-critic 只适用于 "
            "--worker --algo mppo --resume-from",
        )
        _require(args.reset_optimizer,
                 "--reset-worker-critic 必须同时 --reset-optimizer")
        _require(args.seed is not None,
                 "--reset-worker-critic 必须显式提供 --seed")
        _require(str(args.device) == "cpu",
                 "--reset-worker-critic 当前只允许 --device cpu")
        _require(
            critic_warmup_steps > 0
            and critic_warmup_steps % rollout_quantum == 0
            and critic_warmup_steps < args.total_steps,
            "--critic-warmup-steps 必须为正、整除 rollout 量子且小于"
            " --total-steps，以保证 warmup 后 actor 真正更新",
        )
        _require(
            gradient_clip_mode in {
                "separate-actor-critic-v1",
                "separate-root-context-critic-v2",
            },
            "--reset-worker-critic 必须使用"
            " --gradient-clip-mode 分组裁剪",
        )
    else:
        _require(
            critic_warmup_steps == 0,
            "--critic-warmup-steps 只随 --reset-worker-critic 使用",
        )
    _require(
        gradient_clip_mode == "global"
        or (args.worker and args.algo == "mppo" and bool(args.resume_from)),
        "分组裁剪只适用于 Worker MaskablePPO continuation",
    )
    _require(args.freeze_policy_steps >= 0, "--freeze-policy-steps 不能为负")
    _require(args.ckpt_every_steps > 0, "--ckpt-every-steps 必须 > 0")
    _require(args.sentinel_every > 0, "--sentinel-every 必须 > 0")
    _require(args.dry_anchor_every > 0, "--dry-anchor-every 必须 > 0")
    # E5 仪表旋钮(只记不裁;0 = 不在位 = 零侵入,G0-2a 先决)
    _require(args.distill_ce_probe_every >= 0, "--distill-ce-probe-every 不能为负")
    _require(args.drywin_metrics_every >= 0, "--drywin-metrics-every 不能为负")
    _require(
        getattr(args, "worker_fast_forward_reward_credit", "none") in {
            "none", "terminal-death-only"
        },
        "--worker-fast-forward-reward-credit 只允许 "
        "none/terminal-death-only",
    )
    _require(
        math.isfinite(getattr(
            args, "worker_additional_terminal_death_cost", 0.0))
        and getattr(args, "worker_additional_terminal_death_cost", 0.0)
        >= 0.0,
        "--worker-additional-terminal-death-cost 必须是有限非负数",
    )
    _require(
        args.worker
        or (
            getattr(
                args, "worker_fast_forward_reward_credit", "none") == "none"
            and getattr(
                args, "worker_additional_terminal_death_cost", 0.0) == 0.0
        ),
        "Worker 奖励契约旋钮只适用于 --worker",
    )
    legacy_policy_view = bool(getattr(
        args, "legacy_worker_policy_observation_view", False))
    explicit_worker_view = getattr(
        args, "worker_policy_observation_view", None)
    _require(
        explicit_worker_view in {
            None,
            _WORKER_VIEW_LEGACY_V3,
            _WORKER_VIEW_DUAL_V4_ASYMMETRIC,
        },
        "--worker-policy-observation-view 只允许 "
        "legacy-v3/dual-v4-asymmetric-v3",
    )
    _require(
        args.worker or not legacy_policy_view,
        "--legacy-worker-policy-observation-view 只适用于 --worker",
    )
    _require(
        args.worker or explicit_worker_view is None,
        "--worker-policy-observation-view 只适用于 --worker",
    )
    _require(
        not (legacy_policy_view and explicit_worker_view is not None),
        "旧 legacy Worker 旗与显式 worker observation view 互斥",
    )
    manager_policy_view = getattr(
        args, "manager_policy_observation_view", "raw-v4")
    _require(
        manager_policy_view in {"raw-v4", "legacy-v3"},
        "--manager-policy-observation-view 只允许 raw-v4/legacy-v3",
    )
    _require(
        args.options or manager_policy_view == "raw-v4",
        "--manager-policy-observation-view=legacy-v3 只适用于 --options；"
        "--worker 的冻结 M29 视图由 WorkerWindowEnv 强制绑定",
    )
    if args.worker:
        if _bc_aux_structural_active(args):
            _require(
                not legacy_policy_view
                and explicit_worker_view is None,
                "A12 custom policy 必须读取 legacy-v3-a12-overlay；"
                "旧 actor/value 由 policy 内部解码为完整 v3",
            )
            _require(
                _requested_drink_sovereignty(args) is not False,
                "A12 custom policy 必须开启 drink sovereignty；"
                "--no-drink-sovereignty 会让 m[12] 永久关闭并切断在线学习",
            )
        elif explicit_worker_view == _WORKER_VIEW_DUAL_V4_ASYMMETRIC:
            _require(
                bool(args.resume_from),
                "dual-v4-asymmetric-v3 必须从已登记 Worker checkpoint "
                "点火或显式环境重启式参数续接",
            )
        else:
            _require(
                legacy_policy_view
                or explicit_worker_view == _WORKER_VIEW_LEGACY_V3,
                "普通 Worker policy 必须显式携"
                " legacy-v3 observation view，"
                "保证训练与部署同用 protocol-v3 actor/value 输入",
            )
    _require(
        getattr(args, "artifact_scope", "production")
        in _ARTIFACT_SCOPE_RESULTS,
        "--artifact-scope 只允许 development/candidate/production",
    )
    _require(args.distill_ce_probe_every == 0
             or (args.worker and args.algo == "mppo"),
             "--distill-ce-probe-every 只适用于 --worker --algo mppo"
             "(探针需 Leashed 教师)")
    _require(args.drywin_metrics_every == 0 or args.worker,
             "--drywin-metrics-every 只适用于 --worker")
    if args.run_name is not None:
        _require(bool(args.run_name) and pathlib.Path(args.run_name).name == args.run_name
                 and args.run_name not in (".", ".."),
                 "--run-name 必须是单个目录名，不能含路径分隔符")
    if args.seed is not None:
        _require(0 <= args.seed < 2**32, "--seed 必须在 [0, 2**32) 内")
        _require(args.seed + args.num_envs - 1 < 2**32,
                 "--seed + num_envs - 1 必须小于 2**32")

    modes = int(args.worker) + int(args.options) + int(args.flat_clock)
    _require(modes <= 1, "--worker/--options/--flat-clock 互斥")
    # E1 两旗互斥断言(承工程 B1)+ 四门之互斥/模式门:谓词 = skip_dry ∨ schedule。
    _require(not (args.skip_dry and args.dry_curriculum_schedule),
             "--skip-dry 与 --dry-curriculum-schedule 互斥")
    _require(not _dry_window_mechanism_active(args) or args.worker,
             "--skip-dry/--dry-curriculum-schedule 只能与 --worker 同用")
    if args.dry_curriculum_schedule:
        curriculum_table = _parse_dry_curriculum_schedule(args.dry_curriculum_schedule)
        _require(len(curriculum_table) * rollout_quantum >= args.total_steps,
                 f"--dry-curriculum-schedule p 表 {len(curriculum_table)} 项"
                 f"不足以覆盖本腿 {args.total_steps // rollout_quantum} 个 rollout"
                 "(腿相对锚定禁越界钳位)")
    _require(not args.worker_npz or args.options, "--worker-npz 只能与 --options 同用")
    _require(not args.teacher_override or (args.resume_from and args.worker),
             "--teacher-override 只能与 worker 侧 --resume-from 同用")
    _require(not args.allow_manager_change or (args.resume_from and args.worker),
             "--allow-manager-change 只允许 worker resume 显式换经理")
    _require(not args.allow_legacy_resume or args.resume_from,
             "--allow-legacy-resume 只能与 --resume-from 同用")
    allow_environment_restart_resume = bool(getattr(
        args, "allow_environment_restart_resume", False))
    _require(
        not allow_environment_restart_resume
        or (args.worker and args.algo == "mppo" and bool(args.resume_from)),
        "--allow-environment-restart-resume 只适用于 "
        "--worker --algo mppo --resume-from",
    )
    _require(
        not (args.worker and args.allow_legacy_resume)
        or reset_worker_critic,
        "旧 Worker checkpoint 的一次性迁移必须同时"
        " --reset-worker-critic；仅清 optimizer 不能修复窗口终止 critic",
    )
    _require(args.freeze_policy_steps == 0 or args.bc_init,
             "--freeze-policy-steps > 0 时必须提供 --bc-init")
    _require(args.bc_init or args.init_source == "bc",
             "--init-source checkpoint 必须与 --bc-init 同用")
    _require(args.distill_beta == 0 or (args.worker and args.algo == "mppo"),
             "--distill-beta > 0 只适用于 --worker --algo mppo")
    _require(not (args.calib_probes or args.calib_record_only)
             or (args.worker and args.algo == "mppo"),
             "G-CAL 参数只适用于 --worker --algo mppo")
    # E3 ④乙:两旗互不强制(单独给任一旗不报错);在位 = λ_bc>0 ∧ demos 给定。
    _require(math.isfinite(args.bc_aux_lambda) and args.bc_aux_lambda >= 0,
             "--bc-aux-lambda 必须是有限非负数")
    _require(not args.bc_aux_graft or bool(args.bc_aux_demos),
             "--bc-aux-graft 必须同时提供 --bc-aux-demos")
    _require(
        not _bc_aux_structural_active(args)
        or math.isclose(float(args.bc_aux_lambda), 0.0,
                        rel_tol=0.0, abs_tol=0.0),
        "rev10 contextual mixture adapter 要求 --bc-aux-lambda=0；"
        "禁止恢复不可达的梯度辅助拔河")
    _require(
        not (args.bc_aux_lambda > 0 and bool(args.bc_aux_demos)),
        "rev5 gradient bc_aux 已被实测否决；请改用"
        " --bc-aux-graft --bc-aux-lambda 0")
    _require(not _bc_aux_active(args) or (args.worker and args.algo == "mppo"),
             "④乙辅助通路"
             "只适用于 --worker --algo mppo")   # 承 --distill-beta 同型门(裁量注记)
    _require(not args.bc_aux_liveness_preflight
             or (_bc_aux_active(args) and bool(args.resume_from)),
             "--bc-aux-liveness-preflight 只适用于带 --resume-from 的"
             "在位 bc_aux 生产腿")
    _require(
        getattr(args, "artifact_scope", "production") == "production"
        or not _bc_aux_active(args),
        "非 production 工件不得消费 bc_aux final-heldout 发布门；"
        "请关闭 bc_aux，或使用独立 nested-validation 开发路径",
    )
    if _bc_aux_active(args):
        _require(bool(args.resume_from),
                 "在位 bc_aux 正式腿必须从已冻结 worker checkpoint resume")
        _require(args.bc_aux_liveness_preflight,
                 "在位 bc_aux 正式腿必须携 --bc-aux-liveness-preflight")
        _require(args.distill_beta > 0,
                 "在位 bc_aux 正式腿必须保留非零 KING/BC 蒸馏皮筋")
        _require(_bc_aux_structural_active(args),
                 "正式 bc_aux 只接受 rev10 contextual mixture adapter")
        resume_metadata = _validate_leashed_checkpoint(args.resume_from)
        saved_adapter = resume_metadata.get("_bc_aux_circuit_spec")
        if saved_adapter is None:
            _require(args.reset_optimizer,
                     "首次 a12 actor 扩宽必须 --reset-optimizer，"
                     "禁止旧 Adam moments 错绑新拓扑")
        else:
            _require(saved_adapter == _bc_aux_circuit_spec(),
                     "continuation checkpoint 的 a12 mixture spec 漂移")
            _require(not args.reset_optimizer,
                     "已有 a12 mixture continuation 禁止 --reset-optimizer；"
                     "必须保留 PPO 已学 ε、Adam moments 与探索计数")
        _require(not args.calib_record_only,
                 "在位 bc_aux 禁止 --calib-record-only 绕过 G-CAL 硬门")
        _require(pathlib.Path(args.bc_aux_demos).is_file(),
                 f"④乙 v2 示范集(bc-worker-v2)不存在: {args.bc_aux_demos}")
        # v2 专用验证器 + 正/负校准 bank fail-loud(镜像 _precheck 先例,
        # 加载即弃;不在位时零侵入——连文件存在性都不查)
        expected_manager_sha256 = _capture_file_sha256(
            args.manager_npz, "manager_npz")
        _x, _y, _, _masks, _ = _load_bc_aux_demos_v2(
            args.bc_aux_demos,
            expected_manager_sha256=expected_manager_sha256)
    _require(args.arch != "attn" or not (args.worker or args.options or args.flat_clock),
             "EntityAttention 只支持 295 维平面观测")

    if args.resume_from:
        _require((args.worker or args.options) and args.algo == "mppo",
                 "--resume-from 只支持 worker/options 的 mppo 检查点")
        _require(not args.bc_init and args.freeze_policy_steps == 0,
                 "--resume-from 禁与 --bc-init/--freeze-policy-steps 同用")
        _require(_checkpoint_path(args.resume_from).is_file(),
                 f"resume 检查点不存在: {args.resume_from}")
        if args.worker:
            resume_metadata = _validate_leashed_checkpoint(args.resume_from)
            contracted = resume_metadata.get("diablogym_contract") is not None
            _require(
                contracted is allow_environment_restart_resume,
                "带 training_contract 的 Worker checkpoint 只能以显式 "
                "--allow-environment-restart-resume 做参数/Adam 续接；"
                "旧点火 checkpoint 则必须走独立 legacy migration 门",
            )
        else:
            # v31 经理续训口:通用检查点闸(CRC/关键成员/步数/权重有限性),
            # distill_beta 系工人 Leashed 专属标记,经理检查点不作此断言。
            _validate_checkpoint_file(args.resume_from)
    if args.bc_init:
        _require(pathlib.Path(args.bc_init).is_file(), f"BC 权重不存在: {args.bc_init}")
        if args.init_source == "bc":
            gate = ("data_gate" if args.worker else "hypothesis" if args.options
                    else "memoryless_hypothesis")
            _validate_bc_report(pathlib.Path(args.bc_init), gate)
        else:
            _validate_export_manifest(pathlib.Path(args.bc_init))
    if args.teacher_override:
        _require(pathlib.Path(args.teacher_override).is_file(),
                 f"教师覆写文件不存在: {args.teacher_override}")
        _validate_export_manifest(pathlib.Path(args.teacher_override))
    if args.worker:
        _require(args.algo == "mppo" and args.gamma == 1.0 and args.max_steps == 3000,
                 "PREREG-v23:--worker 须配 --algo mppo --gamma 1.0 --max-steps 3000")
        _require(pathlib.Path(args.manager_npz).is_file(),
                 f"经理 npz 不存在: {args.manager_npz}")
        if args.distill_beta > 0 and not args.resume_from:
            _require(pathlib.Path(args.teacher_sd).is_file(),
                     f"教师 state_dict 不存在: {args.teacher_sd}")
            _validate_bc_report(pathlib.Path(args.teacher_sd), "data_gate")
        # E1 四门之 demos/BC 预检门:skip_dry ∨ schedule(谓词在助手内,断言原封)
        _precheck_dry_window_demos(args)
    if args.options:
        _require(args.algo == "mppo" and args.gamma == 1.0 and args.max_steps == 3000,
                 "PREREG-v25:--options 须配 --algo mppo --gamma 1.0 --max-steps 3000")
        if args.worker_npz:
            _require(args.n_steps == 64 and args.seed is not None,
                     "PREREG-v25 D2:换届选举须 --n-steps 64 且显式 --seed")
            _require(pathlib.Path(args.worker_npz).is_file(),
                     f"工人 npz 不存在: {args.worker_npz}")
    if args.seed is not None:
        from diablogym.worker_env import is_reserved_train_seed

        _require(not any(
            is_reserved_train_seed(args.seed + rank)
            for rank in range(args.num_envs)),
                 "种子纪律:--seed + 实际 env rank 撞已登记 BC/评测池")

    try:
        probes = [int(x) for x in args.calib_probes.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError("--calib-probes 必须是逗号分隔的整数") from exc
    _require(all(p >= 0 for p in probes), "--calib-probes 不能包含负数")
    _select_batch_size(args.n_steps, args.num_envs)


def _prepare_run_dir(run_dir: pathlib.Path, resume_from: str | None,
                     protected_inputs=()) -> None:
    """同名重跑时保全旧产物，同时避免 progress/tb/ckpt 混入新尝试。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    run_root = run_dir.resolve()
    protected = [pathlib.Path(p).resolve() for p in protected_inputs if p]
    in_output_tree = [p for p in protected
                      if p == run_root or p.is_relative_to(run_root)]
    _require(not in_output_tree,
             "训练输入不能位于本次 run 输出目录内；请先复制到 train/models "
             f"或独立 inputs 目录: {in_output_tree}")
    existing = [run_dir / name for name in _RUN_ARTIFACTS if (run_dir / name).exists()]
    if not existing:
        return
    if resume_from:
        source = _checkpoint_path(resume_from).resolve()
        _require(all(p.resolve() != source
                     and not (p.is_dir() and source.is_relative_to(p.resolve()))
                     for p in existing),
                 "不能从同一 run_dir 的 model_final 原地 resume；请换 run-name")
    archive = run_dir / "_attempts" / f"pre-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}"
    archive.mkdir(parents=True)
    for path in existing:
        shutil.move(str(path), str(archive / path.name))
    print(f"   同名 run 旧产物已归档: {archive}")


class _RunLock:
    """进程级独占锁；内核会在崩溃/SIGKILL 时自动释放 flock。"""

    def __init__(self, run_dir: pathlib.Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / ".run.lock"
        self._file = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._file.seek(0)
            owner = self._file.read().strip() or "unknown"
            self._file.close()
            raise RuntimeError(f"run_dir 正被另一训练进程占用: {run_dir} ({owner})") from exc
        self._file.seek(0)
        self._file.truncate()
        self._file.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        if getattr(self, "_file", None) is None or self._file.closed:
            return
        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()

    def __del__(self):
        self.close()


class _TrainingResources:
    """Own the run lock and VecEnv from acquisition through every failure path."""

    def __init__(self):
        self.run_lock = None
        self.vec_env = None

    @staticmethod
    def _close_vec_env(vec_env) -> None:
        import signal
        import threading

        # SIGALRM is process-global and can only be installed from the main
        # thread.  Keep embedding/tests safe by falling back to an ordinary
        # close outside it, and restore any host handler/timer afterwards.
        armed = threading.current_thread() is threading.main_thread()
        previous_handler = previous_alarm = None

        def _close_timeout(*_):
            raise TimeoutError("vec_env.close() 超时(疑似 worker 已死)")

        if armed:
            previous_handler = signal.getsignal(signal.SIGALRM)
            previous_alarm = signal.alarm(0)
            try:
                signal.signal(signal.SIGALRM, _close_timeout)
                signal.alarm(20)
            except Exception:
                # Do not leave the caller's timer cancelled if signal setup is
                # unavailable in an unusual embedding environment.
                signal.signal(signal.SIGALRM, previous_handler)
                if previous_alarm:
                    signal.alarm(previous_alarm)
                armed = False
        try:
            vec_env.close()
        except Exception as exc:
            print(f"vec_env.close 异常(忽略,不影响已保存的模型): {exc}")
        finally:
            if armed:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)
                if previous_alarm:
                    signal.alarm(previous_alarm)

    def close(self) -> None:
        # Clear ownership before calling user/library cleanup so a recursive
        # or repeated close remains idempotent even when close itself raises.
        vec_env, self.vec_env = self.vec_env, None
        run_lock, self.run_lock = self.run_lock, None
        try:
            if vec_env is not None:
                self._close_vec_env(vec_env)
        finally:
            if run_lock is not None:
                run_lock.close()


def _atomic_save_model(model, destination: str | pathlib.Path) -> pathlib.Path:
    """先写唯一临时 zip、完整验 CRC/finite，再原子发布 canonical。"""
    final = pathlib.Path(destination)
    if final.suffix.lower() != ".zip":
        final = pathlib.Path(f"{final}.zip")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(
        f".{final.stem}.{os.getpid()}.{time.time_ns()}.tmp.zip")
    try:
        model.save(str(tmp))
        # SB3 close() 只把字节交给内核；在校验/原子替换前强制落盘，
        # 避免断电后留下一个名字已发布但数据未持久化的 checkpoint。
        with open(tmp, "rb") as stream:
            os.fsync(stream.fileno())
        _validate_checkpoint_file(tmp, require_leashed=hasattr(model, "distill_beta"))
        os.replace(tmp, final)
        try:
            directory_fd = os.open(final.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # 部分非 POSIX/网络文件系统不支持目录 fsync；文件本身
            # 仍已持久化，且 replace 的原子性不受影响。
            pass
    finally:
        tmp.unlink(missing_ok=True)
    return final


def _validate_worker_bc_evidence(rec: dict, demos_payload: bytes,
                                 policy_payload: bytes) -> None:
    """Recompute the worker holdout gate from its immutable evidence bytes."""
    import numpy as np
    import torch

    try:
        with np.load(io.BytesIO(demos_payload), allow_pickle=False) as archive:
            _require(set(archive.files) == {"X", "Y", "episode_id"},
                     "BC worker demos.npz 字段必须精确为 X/Y/episode_id")
            x = archive["X"]
            y = archive["Y"]
            episode_id = archive["episode_id"]
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("BC worker"):
            raise
        raise ValueError("BC worker demos.npz 不可解析") from exc

    pairs = rec["pairs"]
    _require(x.ndim == 2 and x.shape == (pairs, 298)
             and x.dtype == np.float32,
             f"BC worker X 形状/dtype 与报告不一致: {x.shape}/{x.dtype}")
    _require(y.shape == (pairs,) and y.dtype == np.int64,
             f"BC worker Y 形状/dtype 与报告不一致: {y.shape}/{y.dtype}")
    _require(episode_id.shape == (pairs,) and episode_id.dtype == np.int64,
             "BC worker episode_id 形状/dtype 与报告不一致")
    _require(np.isfinite(x).all(), "BC worker demos X 含 NaN/Inf")
    from diablogym.worker_env import legacy_worker_policy_observation_view

    _require(
        np.array_equal(x, legacy_worker_policy_observation_view(x)),
        "BC worker demos X 不是 canonical protocol-v3 policy view；"
        "疑似把 raw packed-belt/signed-latch 示范用于 legacy actor",
    )
    _require(bool(((0 <= y) & (y < 15)).all()),
             "BC worker demos Y 含越界动作")
    _require(not np.isin(y, _WORKER_BC_FORBIDDEN_ACTIONS).any(),
             "BC worker demos Y 含禁采动作 11/12(11 恒掩;12 系教师排水后不采)")
    _require(bool((episode_id >= 0).all()),
             "BC worker demos episode_id 不能为负")
    episodes = np.unique(episode_id)
    expected_demo_seeds = np.asarray(_WORKER_BC_DEMO_SEEDS, dtype=np.int64)
    _require(np.array_equal(episodes, expected_demo_seeds),
             "BC worker demos 必须精确覆盖固定示范种子 "
             "2108000..2108127 各至少一对")
    action14 = y == 14
    action14_episodes = int(np.unique(episode_id[action14]).size)
    _require(
        int(action14.sum()) >= _WORKER_BC_MIN_ACTION14_LABELS
        and action14_episodes >= _WORKER_BC_MIN_ACTION14_EPISODES,
        "BC worker demos a14 严格升级覆盖不足:"
        f"labels={int(action14.sum())},episodes={action14_episodes}",
    )
    order = np.random.default_rng(_BC_FINAL_SPLIT_SEED).permutation(episodes)
    n_holdout = max(1, int(round(len(order) * 0.1)))
    expected_episodes = np.sort(order[:n_holdout])
    reported_episodes = np.asarray(rec["held_out_episodes"], dtype=np.int64)
    _require(np.array_equal(reported_episodes, expected_episodes),
             "BC worker held_out_episodes 与确定性整局切分不一致")
    holdout_indices = np.flatnonzero(np.isin(episode_id, expected_episodes))
    _require(len(holdout_indices) == rec["held_out_pairs"],
             "BC worker held_out_pairs 与 demos episode_id 重算不一致")
    _require(0 < len(holdout_indices) < pairs,
             "BC worker 确定性整局切分产生空训练集或空 held-out")

    try:
        state = torch.load(io.BytesIO(policy_payload), map_location="cpu",
                           weights_only=True)
    except Exception as exc:
        raise ValueError("BC worker policy state_dict 不可解析") from exc
    required = _POLICY_HEAD_KEYS
    _require(isinstance(state, dict) and set(state) == set(required),
             "BC worker policy state_dict 字段必须精确匹配策略头")
    tensors = [state[key] for key in required]
    _require(all(isinstance(value, torch.Tensor) for value in tensors),
             "BC worker policy state_dict 含非 Tensor 值")
    w0, b0, w1, b1, wa, ba = tensors
    _require(w0.shape == (64, 298) and b0.shape == (64,)
             and w1.shape == (64, 64) and b1.shape == (64,)
             and wa.shape == (15, 64) and ba.shape == (15,),
             "BC worker policy 必须是 298→64→64→15")
    _require(all(value.dtype == torch.float32 for value in tensors),
             "BC worker policy 策略头 dtype 必须是 float32")
    _require(all(torch.isfinite(value).all().item() for value in tensors),
             "BC worker policy 策略头含 NaN/Inf")

    # Mirror bc_worker.train_bc exactly: one CPU batch over X[holdout].  Using
    # different chunk sizes can select a different GEMM kernel and flip an
    # argmax at a near-tie even though the policy bytes are identical.
    with torch.no_grad():
        obs = torch.from_numpy(np.ascontiguousarray(x[holdout_indices]))
        hidden = torch.tanh(torch.nn.functional.linear(obs, w0, b0))
        hidden = torch.tanh(torch.nn.functional.linear(hidden, w1, b1))
        logits = torch.nn.functional.linear(hidden, wa, ba)
        pred = logits.argmax(1).cpu().numpy()
    heldout_y = y[holdout_indices]
    top1 = round(float((pred == heldout_y).mean()), 4)
    _require(rec["held_out_top1"] == top1,
             "BC worker held_out_top1 与 demos/policy 重算不一致: "
             f"{rec['held_out_top1']} != {top1}")

    full_counts = np.bincount(y, minlength=15)
    gated_classes = np.asarray(sorted({
        *map(int, np.flatnonzero(full_counts >= 300)),
        *_WORKER_BC_REQUIRED_RECALL_ACTIONS,
    }), dtype=np.int64)
    reported_recalls = {int(key): value
                        for key, value in rec["class_recalls"].items()}
    _require(set(reported_recalls) == set(map(int, gated_classes)),
             "BC worker class_recalls 类集合与 demos 全集计数不一致")
    for class_id in gated_classes:
        mask = heldout_y == class_id
        recall = (float((pred[mask] == class_id).mean())
                  if mask.any() else 0.0)
        _require(math.isclose(
                     float(reported_recalls[int(class_id)]), recall,
                     rel_tol=0.0, abs_tol=1e-15),
                 "BC worker class_recalls 与 demos/policy 重算不一致: "
                 f"class={class_id}, {reported_recalls[int(class_id)]} != {recall}")


def _recompute_replay_bc_evidence(required_gate: str,
                                  policy_payload: bytes) -> dict:
    """Execute the deterministic BC demo/replay pools from frozen weights.

    Aggregate JSON is not evidence: a random policy plus edited means and SHA
    fields used to pass.  These gates are infrequent, pre-training operations,
    so correctness wins over the roughly minute-scale deterministic replay.
    """
    import numpy as np
    import torch

    dimensions = {
        "hypothesis": (303, 3),
        "memoryless_hypothesis": (296, 15),
    }
    _require(required_gate in dimensions, f"未知 replay BC gate: {required_gate}")
    obs_dim, action_dim = dimensions[required_gate]
    try:
        state = torch.load(io.BytesIO(policy_payload), map_location="cpu",
                           weights_only=True)
    except Exception as exc:
        raise ValueError("BC replay policy state_dict 不可解析") from exc
    _require(isinstance(state, dict) and set(state) == set(_POLICY_HEAD_KEYS),
             "BC replay policy state_dict 字段必须精确匹配策略头")
    tensors = [state[key] for key in _POLICY_HEAD_KEYS]
    _require(all(isinstance(value, torch.Tensor) for value in tensors),
             "BC replay policy state_dict 含非 Tensor 值")
    w0, b0, w1, b1, wa, ba = tensors
    _require(w0.shape == (64, obs_dim) and b0.shape == (64,)
             and w1.shape == (64, 64) and b1.shape == (64,)
             and wa.shape == (action_dim, 64) and ba.shape == (action_dim,),
             f"BC replay policy 必须是 {obs_dim}→64→64→{action_dim}")
    _require(all(value.dtype == torch.float32 for value in tensors),
             "BC replay policy 策略头 dtype 必须是 float32")
    _require(all(torch.isfinite(value).all().item() for value in tensors),
             "BC replay policy 策略头含 NaN/Inf")

    def policy_action(obs, mask) -> int:
        vector = np.asarray(obs, dtype=np.float32)
        _require(vector.shape == (obs_dim,),
                 f"BC replay 观测维度异常: {vector.shape} != {(obs_dim,)}")
        valid = np.asarray(mask, dtype=bool)
        _require(valid.shape == (action_dim,) and bool(valid.any()),
                 "BC replay 动作掩码维度异常或全假")
        with torch.no_grad():
            x = torch.from_numpy(vector)
            hidden = torch.tanh(torch.nn.functional.linear(x, w0, b0))
            hidden = torch.tanh(torch.nn.functional.linear(hidden, w1, b1))
            logits = torch.nn.functional.linear(hidden, wa, ba)
            logits = logits.masked_fill(~torch.from_numpy(valid), -torch.inf)
            return int(logits.argmax().item())

    if required_gate == "hypothesis":
        from diablogym import OptionsEnv
        from diablogym.options_env import DIVE, FARM

        env = OptionsEnv(max_steps=3000)

        def teacher_action(manager_env, _obs, _mask) -> int:
            raw = manager_env.env._raw
            return (DIVE if (manager_env.exhausted
                             or raw["char_level"] >= raw["dungeon_level"] + 2)
                    else FARM)

        def rollout(chooser, seed: int) -> tuple[float, int]:
            obs, _ = env.reset(seed=seed)
            done = trunc = False
            total = 0.0
            pairs = 0
            while not (done or trunc):
                mask = env.action_masks()
                action = _masked_action_or_first_legal(
                    chooser(env, obs, mask),
                    mask,
                    n_actions=3,
                    label="BC manager 重放",
                )
                obs, reward, done, trunc, _ = env.step(action)
                total += float(reward)
                pairs += 1
            return total, pairs

        try:
            demo = [rollout(teacher_action, seed)
                    for seed in _WORKER_BC_DEMO_SEEDS]
            replay = [rollout(
                lambda _env, obs, mask: policy_action(obs, mask), seed)[0]
                for seed in _BC_REPLAY_SEEDS]
            teacher_replay = [rollout(teacher_action, seed)[0]
                              for seed in _BC_REPLAY_SEEDS]
        finally:
            env.close()
        teacher_demo_mean = sum(value for value, _ in demo) / len(demo)
        bc_mean = sum(replay) / len(replay)
        teacher_mean = sum(teacher_replay) / len(teacher_replay)
        _require(teacher_mean > 0, "BC manager 重算 teacher replay 非正")
        return {
            "pairs": sum(count for _, count in demo),
            "teacher_demo_mean": teacher_demo_mean,
            "bc_replay_7000": bc_mean,
            "teacher_7000": teacher_mean,
            "ratio": bc_mean / teacher_mean,
        }

    from diablogym import DiabloGymEnv, StagnationClockWrapper
    from diablogym.options_env import KILL_PATIENCE, dispatch

    env = StagnationClockWrapper(DiabloGymEnv(
        ticks_per_step=4, max_steps=3000, start_in_dungeon=True,
        include_raw=False, descend_ladder=True, death_ladder=True))

    def teacher_action(flat_env, _obs) -> int:
        raw = flat_env.env._raw
        if flat_env._clock >= KILL_PATIENCE:
            return 11
        mode = ("dive" if raw["char_level"] >= raw["dungeon_level"] + 2
                else "farm")
        masks, nearest = flat_env.env.controller_action_context()
        return dispatch(
            mode, raw, bool(masks[14]), action_mask=masks,
            nearest_engageable_distance=nearest)

    def rollout(chooser, seed: int) -> tuple[float, int]:
        obs, _ = env.reset(seed=seed)
        done = trunc = False
        total = 0.0
        pairs = 0
        while not (done or trunc):
            action = int(chooser(env, obs))
            obs, reward, done, trunc, _ = env.step(action)
            total += float(reward)
            pairs += 1
        return total, pairs

    try:
        demo = [rollout(teacher_action, seed)
                for seed in _WORKER_BC_DEMO_SEEDS]
        replay = [rollout(
            lambda flat_env, obs: policy_action(
                obs, flat_env.env.action_masks()), seed)[0]
            for seed in _BC_REPLAY_SEEDS]
        teacher_replay = [rollout(teacher_action, seed)[0]
                          for seed in _BC_REPLAY_SEEDS]
    finally:
        env.close()
    teacher_demo_mean = sum(value for value, _ in demo) / len(demo)
    bc_mean = sum(replay) / len(replay)
    teacher_mean = sum(teacher_replay) / len(teacher_replay)
    _require(teacher_mean > 0, "BC flat 重算 teacher replay 非正")
    return {
        "pairs": sum(count for _, count in demo),
        "teacher_mean_demo": teacher_demo_mean,
        "bc_replay_mean_7000s": bc_mean,
        "teacher_replay_mean_7000s": teacher_mean,
        "ratio": bc_mean / teacher_mean,
    }


def _validate_bc_report(p: pathlib.Path, required_gate: str,
                        expected_implementation_sha256: str | None = None,
                        *, policy_payload: bytes | None = None,
                        report_payload: bytes | None = None,
                        verify_replay: bool = True) -> dict:
    """验证 BC 闸门，并绑定权重、生成器及完整训练运行时。

    调用方若已为 TOCTOU 安全冻结了 policy/report 字节，必须通过 payload
    传入；验证器不会再读路径。训练加载与组装评测因此共享完全相同的
    schema/指标/来源闸门，而不是各维护一份逐渐漂移的子集。
    """
    from eval_contract import EvalContractError, PROTOCOL_VERSION, strict_json_loads

    _require(required_gate in {"data_gate", "hypothesis", "memoryless_hypothesis"},
             f"未知 BC gate: {required_gate}")
    report = p.with_name("bc_report.json")
    try:
        frozen_report = (report.read_bytes() if report_payload is None
                         else report_payload)
        rec = strict_json_loads(frozen_report)
    except (OSError, EvalContractError) as exc:
        raise ValueError(f"BC 闸门报告缺失/不可读: {report}") from exc
    _require(isinstance(rec, dict), f"BC 闸门报告必须是 JSON 对象: {report}")
    _require(set(rec) == _BC_PASS_KEYS[required_gate],
             f"BC 闸门报告字段/schema 不匹配: {report}")
    _require(_is_plain_int(rec.get("schema_version"))
             and rec["schema_version"] == _BC_REPORT_SCHEMA_VERSION,
             f"BC 闸门报告 schema 过期: {rec.get('schema_version')!r}")
    _require(rec[required_gate] == "PASS",
             f"拒绝加载未过 {required_gate} 闸的 BC 权重: {rec[required_gate]!r}")
    expected_sha = rec.get("policy_sha256")
    _require(_is_sha256(expected_sha),
             f"BC 闸门报告缺少 policy_sha256 绑定: {report}")
    try:
        frozen_policy = p.read_bytes() if policy_payload is None else policy_payload
    except OSError as exc:
        raise ValueError(f"BC 权重缺失/不可读: {p}") from exc
    actual_sha = hashlib.sha256(frozen_policy).hexdigest()
    _require(actual_sha == expected_sha,
             f"BC 权重与闸门报告 SHA 不匹配: {actual_sha} != {expected_sha}")

    # A policy hash proves which bytes were loaded, but not which world made
    # their demonstrations.  In particular, pre-v3 worker demos may contain
    # trajectories that returned to town.  Bind every PASS report to the
    # current environment/native/content bundle and the exact BC generator.
    _require(rec.get("protocol_version") == PROTOCOL_VERSION,
             f"BC 报告协议过期: {rec.get('protocol_version')!r} != {PROTOCOL_VERSION}")
    expected_impl = (expected_implementation_sha256
                     if expected_implementation_sha256 is not None
                     else _implementation_bundle_sha256())
    _require(rec.get("implementation_sha256") == expected_impl,
             "BC 报告的实现/引擎/游戏内容身份与当前运行时不一致")
    root = pathlib.Path(__file__).resolve().parents[1]
    generator_name = {
        "data_gate": "bc_worker.py",
        "hypothesis": "bc_manager.py",
        "memoryless_hypothesis": "bc_flat.py",
    }[required_gate]
    generator_sha = hashlib.sha256(
        (root / "train" / generator_name).read_bytes()).hexdigest()
    _require(rec.get("generator_sha256") == generator_sha,
             f"BC 报告生成器已漂移: train/{generator_name}")
    if required_gate == "data_gate":
        _validate_bc_final_holdout_marker(
            p.parent, 1, _WORKER_BC_DEMO_SEEDS, rec)
        pairs = rec["pairs"]
        held_out_pairs = rec["held_out_pairs"]
        top1 = _finite_number(rec["held_out_top1"], "BC held_out_top1")
        _require(_is_plain_int(pairs) and pairs > 0
                 and _is_plain_int(held_out_pairs) and 0 < held_out_pairs < pairs,
                 "BC worker 报告样本计数非法")
        # A2(2026-07-27 批):质量线只记不裁,此处仅约束读数范围;
        # 逐位复算一致性由 _validate_worker_bc_evidence 保证。
        _require(top1 >= 0.0 and top1 <= 1.0,
                 "BC worker held-out top-1 读数越界")
        episodes = rec["held_out_episodes"]
        _require(isinstance(episodes, list) and episodes
                 and all(_is_plain_int(value) and value >= 0 for value in episodes)
                 and episodes == sorted(set(episodes)),
                 "BC worker held_out_episodes 非规范")
        _require(isinstance(rec["class_weighted_retry"], bool),
                 "BC worker class_weighted_retry 必须是 bool")
        recalls = rec["class_recalls"]
        _require(isinstance(recalls, dict), "BC worker class_recalls 必须是对象")
        for raw_class, raw_recall in recalls.items():
            try:
                class_id = int(raw_class)
            except (TypeError, ValueError) as exc:
                raise ValueError("BC worker class_recalls 键必须是动作编号") from exc
            _require(str(class_id) == str(raw_class) and 0 <= class_id < 15,
                     f"BC worker class_recalls 键非法: {raw_class!r}")
            recall = _finite_number(raw_recall,
                                    f"BC worker class_recalls[{raw_class!r}]")
            # A2:同上,召回为只记不裁读数。
            _require(0.0 <= recall <= 1.0,
                     "BC worker class_recalls 读数越界")
        _require(_is_sha256(rec.get("demos_sha256")),
                 "BC worker 报告缺少 demos_sha256")
        demos_path = p.with_name("demos.npz")
        try:
            demos_payload = demos_path.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"BC worker 示范集缺失/不可读: {demos_path}") from exc
        actual_demos_sha = hashlib.sha256(demos_payload).hexdigest()
        _require(actual_demos_sha == rec["demos_sha256"],
                 "BC worker demos.npz 与闸门报告 SHA 不匹配: "
                 f"{actual_demos_sha} != {rec['demos_sha256']}")
        _validate_worker_bc_evidence(rec, demos_payload, frozen_policy)
        manager = root / "train" / "models" / "v22-h-manager" / "policy.npz"
        manager_sha = hashlib.sha256(manager.read_bytes()).hexdigest()
        _require(rec.get("manager_npz_sha256") == manager_sha,
                 "BC worker 报告未绑定当前冻结 manager NPZ")
    else:
        pairs = rec["pairs"]
        ratio = _finite_number(rec["ratio"], "BC replay ratio")
        _require(_is_plain_int(pairs) and pairs > 0, "BC 报告样本计数非法")
        if required_gate == "hypothesis":
            demo_key, bc_key, teacher_key = (
                "teacher_demo_mean", "bc_replay_7000", "teacher_7000")
        else:
            demo_key, bc_key, teacher_key = (
                "teacher_mean_demo", "bc_replay_mean_7000s",
                "teacher_replay_mean_7000s")
        _finite_number(rec[demo_key], f"BC {demo_key}")
        bc_replay = _finite_number(rec[bc_key], f"BC {bc_key}")
        teacher_replay = _finite_number(rec[teacher_key], f"BC {teacher_key}")
        _require(teacher_replay > 0,
                 f"BC {teacher_key} 必须为正，replay ratio 才有定义")
        recomputed_ratio = bc_replay / teacher_replay
        _require(math.isclose(ratio, recomputed_ratio,
                              rel_tol=1e-12, abs_tol=1e-12),
                 "BC replay ratio 与报告中的 BC/teacher 指标不一致: "
                 f"{ratio} != {recomputed_ratio}")
        _require(ratio >= 0.85, "BC 报告 PASS 与 replay ratio 不一致")
        if verify_replay:
            cache_key = (required_gate, actual_sha, expected_impl, generator_sha)
            evidence = _BC_REPLAY_CACHE.get(cache_key)
            cache_miss = evidence is None
            if evidence is None:
                print(f"   BC {required_gate}: 重放固定 demo/replay 种子复核报告证据")
                evidence = _recompute_replay_bc_evidence(
                    required_gate, frozen_policy)
            for key, actual_value in evidence.items():
                reported_value = rec[key]
                if key == "pairs":
                    matches = (_is_plain_int(reported_value)
                               and reported_value == actual_value)
                else:
                    matches = (isinstance(reported_value, (int, float))
                               and not isinstance(reported_value, bool)
                               and math.isclose(float(reported_value),
                                                float(actual_value),
                                                rel_tol=1e-12,
                                                abs_tol=1e-9))
                _require(matches,
                         "BC replay 报告与冻结 policy/当前 runtime 重算不一致: "
                         f"{key}={reported_value!r} != {actual_value!r}")
            if cache_miss:
                _BC_REPLAY_CACHE[cache_key] = dict(evidence)
    return rec


def _dry_anchor_partition(x):
    """按 worker v4 双通道语义返回互补的 (pre-dry, fresh) 行掩码。

    worker transition 的观测发生在动作执行前；真正把 140/1800 计数推到
    cap 的动作随后立即收窗，因此合法示范池通常只能看到 cap-1，而不会
    看到恰好 1.0。col297 的负域还承载可见饮药闩，须先解码其 scene clock，
    不能把负数误当 fresh。阈值直接取环境两个 cap 的单步前沿，避免探针因
    一个不可能出现的终态观测而静默零覆盖。
    """
    import numpy as np
    from diablogym.options_env import FARM_SCENE_CAP, KILL_PATIENCE

    values = np.asarray(x)
    _require(values.ndim == 2 and values.shape[1] == 298,
             f"dry-anchor 分组输入形状异常:{values.shape}")
    short_threshold = np.float32(
        (KILL_PATIENCE - 1) / KILL_PATIENCE)
    scene_threshold = np.float32(
        (FARM_SCENE_CAP - 1) / FARM_SCENE_CAP)
    encoded_scene = values[:, 297]
    scene_clock = np.where(
        encoded_scene < 0.0, -encoded_scene - 1.0, encoded_scene)
    dry = ((values[:, 296] >= short_threshold)
           | (scene_clock >= scene_threshold))
    return dry, ~dry


def _load_dry_anchor_demos(path: str | pathlib.Path,
                           expected_sha256: str | None) -> tuple[object, object, str]:
    """Hash and parse the same demos.npz bytes, then enforce the BC binding."""
    import numpy as np

    p = pathlib.Path(path)
    _require(isinstance(expected_sha256, str) and len(expected_sha256) == 64,
             f"BC 闸门报告缺少 demos_sha256 绑定: {p.with_name('bc_report.json')}")
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"dry-anchor 示范集不可读: {p}: {exc}") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    _require(actual_sha256 == expected_sha256,
             f"dry-anchor demos SHA 不匹配: {actual_sha256} != {expected_sha256}")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            _require(all(key in data for key in ("X", "Y", "episode_id")),
                     "dry-anchor demos.npz 缺少 X/Y/episode_id")
            x, y = data["X"].copy(), data["Y"].copy()
            episode_id = data["episode_id"].copy()
    except (OSError, ValueError) as exc:
        raise ValueError(f"dry-anchor 示范集不可读: {p}: {exc}") from exc
    _require(x.ndim == 2 and x.shape[1] == 298
             and y.ndim == 1 and len(x) == len(y),
             f"dry-anchor 数组形状异常:X={x.shape},Y={y.shape}")
    _require(x.dtype == np.float32 and np.issubdtype(y.dtype, np.integer),
             f"dry-anchor dtype 异常:X={x.dtype},Y={y.dtype}")
    _require(episode_id.ndim == 1 and len(episode_id) == len(x)
             and np.issubdtype(episode_id.dtype, np.integer)
             and len(np.unique(episode_id)) >= 2,
             "dry-anchor episode_id 形状/类型/独立局数异常")
    dry, _ = _dry_anchor_partition(x)
    _require(bool(dry.any()),
             "dry-anchor 示范集没有双通道 cap-1 前沿态"
             "(col296 short clock / col297 signed farm-scene clock)")
    return x, y, actual_sha256


def _export_manifest_path(p: pathlib.Path) -> pathlib.Path:
    return p.with_name(f"{p.name}.manifest.json")


def _validate_export_manifest(p: pathlib.Path) -> dict:
    import torch

    from eval_contract import EvalContractError, strict_json_loads

    manifest_path = _export_manifest_path(p)
    try:
        rec = strict_json_loads(manifest_path.read_bytes())
    except (OSError, EvalContractError) as exc:
        raise ValueError(f"checkpoint 导出清单缺失/不可读: {manifest_path}") from exc
    _require(isinstance(rec, dict), f"checkpoint 导出清单必须是 JSON 对象: {manifest_path}")
    _require(set(rec) == {
        "schema_version", "artifact_type", "artifact_sha256",
        "source_checkpoint", "source_checkpoint_sha256", "tensor_count"},
        f"checkpoint 导出清单字段异常: {manifest_path}")
    _require(_is_plain_int(rec["schema_version"])
             and rec["schema_version"] == _EXPORT_MANIFEST_SCHEMA_VERSION,
             f"checkpoint 导出清单 schema 非法: {rec['schema_version']!r}")
    _require(rec.get("artifact_type") == "checkpoint_policy_state",
             f"导出清单 artifact_type 异常: {rec.get('artifact_type')!r}")
    expected = rec.get("artifact_sha256")
    try:
        artifact_payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"checkpoint 导出件不可读: {p}: {exc}") from exc
    actual = hashlib.sha256(artifact_payload).hexdigest()
    _require(_is_sha256(expected) and expected == actual,
             f"checkpoint 导出件与清单 SHA 不匹配: {actual} != {expected!r}")
    source_sha = rec.get("source_checkpoint_sha256")
    _require(_is_sha256(source_sha),
             "checkpoint 导出清单缺少 source_checkpoint_sha256")
    source_checkpoint = rec.get("source_checkpoint")
    _require(isinstance(source_checkpoint, str) and source_checkpoint
             and pathlib.Path(source_checkpoint).is_absolute(),
             "checkpoint 导出清单 source_checkpoint 必须是绝对路径")
    source_path = pathlib.Path(source_checkpoint)
    _require(str(source_path.resolve()) == source_checkpoint,
             "checkpoint 导出清单 source_checkpoint 必须是规范绝对路径")
    _require(source_path.resolve() != p.resolve(),
             "checkpoint 导出件不能把自身声明为源 checkpoint")
    try:
        source_payload = source_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"checkpoint 导出清单声明的源 checkpoint 不可读: {source_path}") from exc
    actual_source_sha = hashlib.sha256(source_payload).hexdigest()
    _require(actual_source_sha == source_sha,
             "checkpoint 导出清单的源 checkpoint SHA 不匹配: "
             f"{actual_source_sha} != {source_sha}")
    _validate_checkpoint_bytes(source_payload, str(source_path))

    tensor_count = rec.get("tensor_count")
    _require(_is_plain_int(tensor_count) and tensor_count > 0,
             "checkpoint 导出清单 tensor_count 必须是正整数")
    try:
        artifact_state = torch.load(
            io.BytesIO(artifact_payload), map_location="cpu", weights_only=True)
        with zipfile.ZipFile(io.BytesIO(source_payload)) as source_archive:
            source_state = torch.load(
                io.BytesIO(source_archive.read("policy.pth")),
                map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("checkpoint 导出件或源 policy state_dict 不可解析") from exc
    _require(isinstance(artifact_state, dict) and isinstance(source_state, dict),
             "checkpoint 导出件与源 policy 必须都是 state_dict")
    _require(tensor_count == len(artifact_state) == len(source_state),
             "checkpoint 导出清单 tensor_count 与导出件/源 policy 不一致")
    _require(set(artifact_state) == set(source_state),
             "checkpoint 导出件字段与源 checkpoint policy 不一致")
    for key in source_state:
        source_value = source_state[key]
        artifact_value = artifact_state[key]
        _require(isinstance(source_value, torch.Tensor)
                 and isinstance(artifact_value, torch.Tensor),
                 f"checkpoint policy 字段不是 Tensor: {key}")
        _require(torch.isfinite(artifact_value).all().item(),
                 f"checkpoint 导出件含 NaN/Inf: {key}")
        _require(artifact_value.shape == source_value.shape
                 and artifact_value.dtype == source_value.dtype
                 and torch.equal(artifact_value, source_value),
                 f"checkpoint 导出件张量与源 checkpoint policy 不一致: {key}")
    return rec


def _load_bc_state_dict(path: str, policy, required_gate: str,
                        source_kind: str = "bc") -> dict:
    """校验 BC 闸门与关键张量，禁止“0 键命中但 strict=False”静默起跑。"""
    import torch

    p = pathlib.Path(path)
    manifest = None
    if source_kind == "bc":
        expected_sha256 = _validate_bc_report(p, required_gate)["policy_sha256"]
    elif source_kind == "checkpoint":
        manifest = _validate_export_manifest(p)
        expected_sha256 = manifest["artifact_sha256"]
    else:
        raise ValueError(f"未知 init source kind: {source_kind}")

    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"--bc-init 权重不可读: {p}: {exc}") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    _require(actual_sha256 == expected_sha256,
             f"BC/init 权重在闸门校验后发生漂移: "
             f"{actual_sha256} != {expected_sha256}")
    # Integrity check and torch deserialization consume the same bytes.
    sd = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    _require(isinstance(sd, dict), "--bc-init 必须是 policy state_dict")
    bad_types = [k for k, value in sd.items() if not isinstance(value, torch.Tensor)]
    _require(not bad_types, f"BC state_dict 含非 Tensor 值: {bad_types}")
    nonfinite = [k for k, value in sd.items()
                 if not torch.isfinite(value).all().item()]
    _require(not nonfinite, f"BC state_dict 含 NaN/Inf: {nonfinite}")
    target = policy.state_dict()
    if source_kind == "checkpoint":
        missing_all = sorted(set(target) - set(sd))
        unexpected_all = sorted(set(sd) - set(target))
        mismatched_all = sorted(
            key for key in set(target) & set(sd)
            if target[key].shape != sd[key].shape)
        dtype_mismatched = sorted(
            key for key in set(target) & set(sd)
            if target[key].dtype != sd[key].dtype)
        _require(not missing_all and not unexpected_all and not mismatched_all
                 and not dtype_mismatched,
                 "checkpoint 全量 policy state_dict 与目标不精确一致: "
                 f"missing={missing_all}, unexpected={unexpected_all}, "
                 f"shape={mismatched_all}, dtype={dtype_mismatched}")
        _require(manifest is not None and manifest["tensor_count"] == len(sd),
                 "checkpoint 导出清单 tensor_count 与 state_dict 不一致")
    missing = [k for k in _POLICY_HEAD_KEYS if k not in sd]
    mismatched = [k for k in _POLICY_HEAD_KEYS if k in sd and
                  (k not in target or target[k].shape != sd[k].shape)]
    _require(not missing, f"BC state_dict 缺少策略头键: {missing}")
    _require(not mismatched, f"BC state_dict 策略头形状不匹配: {mismatched}")
    return sd


def make_env(max_steps: int = 1500, deep: bool = False, death_ladder: bool = False,
             options: bool = False, flat_clock: bool = False,
             worker: bool = False, manager_npz: str | None = None,
             worker_npz: str | None = None, skip_dry: float | bool = False,
             drink_sovereignty: bool | None = None,
             legacy_policy_observation_view: bool = False,
             worker_policy_observation_view: str | None = None,
             manager_policy_observation_view: str = "raw-v4",
             worker_fast_forward_reward_credit: str = "none",
             worker_additional_terminal_death_cost: float = 0.0,
             manager_npz_sha256: str | None = None,
             worker_npz_sha256: str | None = None,
             implementation_sha256: str | None = None):
    if implementation_sha256 is not None:
        actual_implementation = _implementation_bundle_sha256()
        _require(actual_implementation == implementation_sha256,
                 "训练实现 bundle 在 VecEnv 创建前发生漂移: "
                 f"{actual_implementation} != {implementation_sha256}")
    from diablogym import DiabloGymEnv

    def with_seed_discipline(env):
        """平面/策略脑通用的逐局种子包装；worker 自身有跨窗口局状态，不套。"""
        import gymnasium as gym
        import numpy as np
        from diablogym.worker_env import (
            is_reserved_train_seed,
            sample_train_seed,
        )

        class _SeedDiscipline(gym.Wrapper):
            def __init__(self, wrapped):
                super().__init__(wrapped)
                self._seed_rng = np.random.default_rng()

            def reset(self, *, seed=None, options=None):
                if seed is not None:
                    if is_reserved_train_seed(seed):
                        raise ValueError(f"训练 reset 拒绝保留种子 {seed}")
                    self._seed_rng = np.random.default_rng(seed)
                else:
                    seed = sample_train_seed(self._seed_rng)
                obs, info = self.env.reset(seed=seed, options=options)
                info["episode_seed"] = seed
                return obs, info

            def action_masks(self):
                return self.env.action_masks()

        return Monitor(_SeedDiscipline(env))

    if worker:
        # v4 窗口线：episode = 一整局底层游戏；自然 FARM 窗边界为
        # nonterminal，并经 info["farm_window_end"] 显式上报。
        # (rng_seed=None → 各子进程独立熵源,种子采样器拒采全部登记评测池)
        # v26:skip_dry=True 时干层复访窗由脚本代跑,不进学习分布(绿洲处方)
        from diablogym import WorkerWindowEnv
        worker_drink_sovereignty = (
            True if drink_sovereignty is None else drink_sovereignty)
        return Monitor(WorkerWindowEnv(manager_npz=manager_npz, max_steps=max_steps,
                                       skip_dry=skip_dry,
                                       drink_sovereignty=worker_drink_sovereignty,
                                       legacy_policy_observation_view=(
                                           legacy_policy_observation_view),
                                       policy_observation_view=(
                                           worker_policy_observation_view),
                                       fast_forward_reward_credit=(
                                           worker_fast_forward_reward_credit),
                                       additional_terminal_death_cost=(
                                           worker_additional_terminal_death_cost),
                                       seed_scope="train",
                                       manager_sha256=manager_npz_sha256))
    if options:
        # v22:策略脑/操作脑——OptionsEnv 自带 deep+death_ladder 默认
        # v25:worker_npz 非空时挂 npz 工人(NumpyManager 在本函数体内构造——
        # spawn 子进程免 torch,PREREG-v25 D1 条款),并套种子纪律薄包装
        from diablogym import NumpyManager, OptionsEnv

        if worker_npz:
            # 条款要点:工人以 npz+numpy 前向进子进程(不 pickle 网络、不 load SB3
            # 模型、不逐拍 torch 前向)。torch 模块本身随 train_ppo 顶层 import 进入
            # 子进程(v23 先例同),"无 torch"断言不可实现,预注册已如实修正。
            net = NumpyManager(worker_npz, expected_sha256=worker_npz_sha256)
            net.require_io_shape(298, 15, "Options worker")
            net.require_worker_contract()
            env = OptionsEnv(max_steps=max_steps,
                             drink_sovereignty=drink_sovereignty,
                             manager_observation_view=(
                                 manager_policy_observation_view),
                             worker_observation_view=(
                                 net.worker_observation_view),
                             workers={0: net.worker_callback()})
        else:
            env = OptionsEnv(max_steps=max_steps,
                             drink_sovereignty=drink_sovereignty,
                             manager_observation_view=(
                                 manager_policy_observation_view))

        # 无论是否挂 npz 工人，经理训练都必须遵守同一种子纪律。
        return with_seed_discipline(env)
    if flat_clock:
        # v22 恶魔臂 F:296 维平面(停滞钟与策略脑同一块表)
        from diablogym import StagnationClockWrapper
        return with_seed_discipline(StagnationClockWrapper(DiabloGymEnv(
            ticks_per_step=4, max_steps=max_steps, start_in_dungeon=True,
            include_raw=False, descend_ladder=True, death_ladder=True)))
    env = DiabloGymEnv(
        ticks_per_step=4,      # 每个决策 = 0.2 秒游戏时间
        max_steps=max_steps,   # 1500 = 冠军(v6)配方;3000 = v10 长局实验 + v17 深水区。
                               # 32 种子排行榜评估固定 1500 步(可比性);深水区章另立新表
        start_in_dungeon=True, # 跳过城镇,直接站在地牢 1 层入口
        include_raw=False,     # 训练不传 raw 大字典(多进程 IPC 减负)
        descend_ladder=deep,   # v17:下楼奖金层数递进(8×N),给"往下活着"一个未来
        death_ladder=death_ladder,  # v18:死在 N 层罚 8×N——"活着抵达"要赢过"摸到深度"
    )
    return with_seed_discipline(env)


def _is_publishable_rollout_boundary(
        model, *, rollout_buffer_may_be_reset: bool = False) -> bool:
    """Require collection plus an explicit optimizer-consumption receipt.

    ``MaskablePPO.collect_rollouts()`` resets its buffer immediately before
    ``callback.on_rollout_start()``.  Periodic checkpointing runs at that hook
    because the preceding ``train()`` has just returned, so callers may
    explicitly rely on the persisted Leashed receipt after that reset.  Final
    publication keeps the stricter default and still requires ``buffer.full``.
    """
    rollout_full = bool(getattr(getattr(model, "rollout_buffer", None),
                                "full", False))
    if ((not rollout_full and not rollout_buffer_may_be_reset)
            or bool(getattr(model, "_calib_tripped", False))):
        return False
    if not hasattr(model, "_last_completed_ppo_rollout_steps"):
        # Non-Leashed legacy algorithms do not yet emit the stronger receipt.
        return True
    completed_at = getattr(
        model, "_last_completed_ppo_rollout_steps", None)
    optimizer_steps = getattr(
        model, "_ppo_optimizer_steps_completed", None)
    return (
        _is_plain_int(completed_at)
        and completed_at == int(getattr(model, "num_timesteps", -1))
        and _is_plain_int(optimizer_steps)
        and optimizer_steps > 0
    )


def _asymmetric_worker_deployment_evidence_complete(model) -> bool:
    """Close the structured actor/critic deployment evidence without key lore."""
    from leashed_ppo import (
        ASYMMETRIC_WORKER_RUNTIME_EVIDENCE_SCHEMA,
        AsymmetricWorkerMaskableActorCriticPolicy,
        asymmetric_worker_runtime_evidence,
    )
    from diablogym.controller_wire import (
        DUAL_WORKER_LAYOUT,
        DUAL_WORKER_LAYOUT_SHA256,
    )

    if not isinstance(
            getattr(model, "policy", None),
            AsymmetricWorkerMaskableActorCriticPolicy):
        return True
    try:
        evidence = asymmetric_worker_runtime_evidence(model.policy)
        layout = evidence["layout"]
        policy = evidence["policy"]
        context = evidence["context"]
        groups = context["parameter_groups"]
        probes = evidence["probes"]
        focused_effects = probes["actor_focused_effects"]
        actor_receipt = getattr(
            model, "_actor_migration_receipt", None)
        critic_receipt = getattr(
            model, "_critic_migration_receipt", None)
        anneal_rollouts = getattr(
            model, "distill_anneal_actor_rollouts", None)
        anneal_completed = getattr(
            model, "_distill_actor_rollouts_completed", None)
        last_effective_beta = getattr(
            model, "_last_effective_distill_beta", None)
        distillation_complete = (
            (
                float(getattr(model, "distill_beta", 0.0)) == 0.0
                and anneal_rollouts == 0
            )
            or (
                _is_plain_int(anneal_rollouts)
                and anneal_rollouts >= 2
                and _is_plain_int(anneal_completed)
                and anneal_completed >= anneal_rollouts
                and isinstance(last_effective_beta, (int, float))
                and not isinstance(last_effective_beta, bool)
                and math.isfinite(float(last_effective_beta))
                and float(last_effective_beta) == 0.0
            )
        )
        action_probe_keys = (
            (
                "forced_context_action_effect",
                "forced_context_action_logit_changed_elements",
                "forced_context_action_logit_max_abs_delta",
            ),
            (
                "nonzero_context_action_effect",
                "nonzero_context_action_logit_changed_elements",
                "nonzero_context_action_logit_max_abs_delta",
            ),
        )
        action_effects_closed = all(
            probes.get(enabled_key) is True
            and _is_plain_int(probes.get(count_key))
            and probes[count_key] > 0
            and isinstance(probes.get(delta_key), (int, float))
            and not isinstance(probes[delta_key], bool)
            and math.isfinite(float(probes[delta_key]))
            and float(probes[delta_key]) > 0.0
            for enabled_key, count_key, delta_key in action_probe_keys
        )
        focused_effects_closed = (
            isinstance(focused_effects, dict)
            and set(focused_effects) == {
                "current_v4_base",
                "wrapper_scalars",
                "controller_combat",
            }
            and all(
                isinstance(record, dict)
                and record.get("registered_actor_input") is True
                and _is_plain_int(record.get("feature_index"))
                and record.get("preoutput_effect") is True
                and _is_plain_int(
                    record.get("preoutput_changed_elements"))
                and record["preoutput_changed_elements"] > 0
                and isinstance(
                    record.get("preoutput_max_abs_delta"),
                    (int, float),
                )
                and not isinstance(
                    record["preoutput_max_abs_delta"], bool)
                and math.isfinite(float(
                    record["preoutput_max_abs_delta"]))
                and float(record["preoutput_max_abs_delta"]) > 0.0
                and record.get("context_action_effect") is True
                and _is_plain_int(record.get(
                    "context_action_logit_changed_elements"))
                and record[
                    "context_action_logit_changed_elements"] > 0
                and isinstance(record.get(
                    "context_action_logit_max_abs_delta"), (int, float))
                and not isinstance(
                    record["context_action_logit_max_abs_delta"], bool)
                and math.isfinite(float(
                    record["context_action_logit_max_abs_delta"]))
                and float(
                    record["context_action_logit_max_abs_delta"]) > 0.0
                for record in focused_effects.values()
            )
        )
        group_receipt_keys = {
            "encoder": "context_encoder_sha256",
            "interaction": "context_interaction_sha256",
            "output": "context_output_sha256",
        }
        groups_closed = (
            isinstance(groups, dict)
            and set(groups) == set(group_receipt_keys)
            and all(
                isinstance(groups[name], dict)
                and _is_plain_int(groups[name].get("tensor_count"))
                and groups[name]["tensor_count"] > 0
                and _is_plain_int(groups[name].get("parameter_count"))
                and groups[name]["parameter_count"] > 0
                and _is_plain_int(groups[name].get("nonzero_count"))
                and groups[name]["nonzero_count"] > 0
                and _is_sha256(groups[name].get("sha256"))
                for name in group_receipt_keys
            )
        )
        actor_receipt_closed = (
            isinstance(actor_receipt, dict)
            and actor_receipt.get("schema")
            == _ASYMMETRIC_ACTOR_INIT_SCHEMA
            and actor_receipt.get("method")
            == _ASYMMETRIC_ACTOR_INIT_METHOD
            and actor_receipt.get("context_architecture")
            == _ASYMMETRIC_CONTEXT_ARCHITECTURE
            and actor_receipt.get("controller_layout_schema")
            == DUAL_WORKER_LAYOUT.schema
            and actor_receipt.get("controller_layout_sha256")
            == DUAL_WORKER_LAYOUT_SHA256
            and actor_receipt.get("target_actor_parameter_tensors")
            == policy["actor_tensor_count"]
            and actor_receipt.get("target_actor_parameter_count")
            == policy["actor_parameter_count"]
            and actor_receipt.get("context_parameter_tensors")
            == context["tensor_count"]
            and actor_receipt.get("context_parameter_count")
            == context["parameter_count"]
            and _is_sha256(
                actor_receipt.get("migrated_actor_sha256"))
            and policy["actor_sha256"]
            != actor_receipt["migrated_actor_sha256"]
            and groups_closed
            and all(
                _is_sha256(actor_receipt.get(receipt_key))
                and groups[name]["sha256"]
                != actor_receipt[receipt_key]
                for name, receipt_key in group_receipt_keys.items()
            )
        )
        critic_receipt_closed = (
            isinstance(critic_receipt, dict)
            and critic_receipt.get("schema")
            == _ASYMMETRIC_CRITIC_RESET_SCHEMA
            and critic_receipt.get("method")
            == _ASYMMETRIC_CRITIC_RESET_METHOD
            and critic_receipt.get("critic_architecture")
            == _ASYMMETRIC_CRITIC_ARCHITECTURE
            and critic_receipt.get("controller_layout_schema")
            == DUAL_WORKER_LAYOUT.schema
            and critic_receipt.get("controller_layout_sha256")
            == DUAL_WORKER_LAYOUT_SHA256
            and critic_receipt.get("actor_parameter_tensors")
            == policy["actor_tensor_count"]
            and critic_receipt.get("critic_parameter_tensors")
            == policy["critic_tensor_count"]
            and critic_receipt.get("critic_parameter_count")
            == policy["critic_parameter_count"]
            and _is_sha256(
                critic_receipt.get("critic_sha256_after"))
            and policy["critic_sha256"]
            != critic_receipt["critic_sha256_after"]
            and critic_receipt.get("source_checkpoint_sha256")
            == actor_receipt.get("source_checkpoint_sha256")
            and critic_receipt.get("source_actor_sha256")
            == actor_receipt.get("source_actor_sha256")
        )
        return bool(
            evidence.get("schema")
            == ASYMMETRIC_WORKER_RUNTIME_EVIDENCE_SCHEMA
            and layout.get("schema") == DUAL_WORKER_LAYOUT.schema
            and layout.get("sha256") == DUAL_WORKER_LAYOUT_SHA256
            and layout.get("observation_dim")
            == DUAL_WORKER_LAYOUT.observation_dim
            and context.get("enabled") is True
            and _is_sha256(policy.get("actor_sha256"))
            and _is_sha256(policy.get("critic_sha256"))
            and probes.get("p_skip_preoutput_invariant") is True
            and probes.get("p_skip_preoutput_max_abs_delta") == 0.0
            and probes.get("p_skip_preoutput_zero_sha256")
            == probes.get("p_skip_preoutput_one_sha256")
            and probes.get("p_skip_action_logits_invariant") is True
            and probes.get("p_skip_action_logits_max_abs_delta") == 0.0
            and probes.get("p_skip_action_logits_zero_sha256")
            == probes.get("p_skip_action_logits_one_sha256")
            and action_effects_closed
            and focused_effects_closed
            and actor_receipt_closed
            and critic_receipt_closed
            and distillation_complete
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return False


def _is_exact_training_completion(model, target_global_steps: int) -> bool:
    """只有完整更新边界且全局步精确到达冻结目标才允许发布。

    Callback 返回 ``False`` 时 SB3 的 ``learn`` 会正常返回；仅检查 full
    buffer 会把任意较早 rollout 边界误认成完整腿。严格相等也同时拦住
    静默 overshoot。
    """
    migration_start = getattr(
        model, "_critic_warmup_start_timesteps", None)
    worker_pg_complete = True
    if migration_start is not None:
        try:
            from leashed_ppo import worker_onpolicy_pg_audit_complete
            worker_pg_complete = worker_onpolicy_pg_audit_complete(model)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            worker_pg_complete = False
    migration_complete = (
        migration_start is None
        or (
            bool(getattr(model, "_critic_warmup_completed", False))
            and int(getattr(
                model, "_critic_warmup_rollouts_completed", -1))
            == int(getattr(
                model, "_critic_warmup_expected_rollouts", -2))
            and int(getattr(
                model, "_critic_warmup_optimizer_steps_completed", 0)) > 0
            and int(getattr(
                model, "_actor_optimizer_steps_completed", 0)) > 0
            and int(getattr(model, "num_timesteps", -1))
            > int(getattr(
                model, "_critic_warmup_until_timesteps", 2**63 - 1))
        )
    )
    asymmetric_complete = (
        _asymmetric_worker_deployment_evidence_complete(model))
    return (
        _is_publishable_rollout_boundary(model)
        and int(getattr(model, "num_timesteps", -1))
        == int(target_global_steps)
        and migration_complete
        and worker_pg_complete
        and asymmetric_complete
    )


def _require_exact_training_completion(model, target_global_steps: int) -> None:
    """把 SB3 的“正常提前返回”提升为进程失败，禁止调度器假成功。"""
    if _is_exact_training_completion(model, target_global_steps):
        return
    raise RuntimeError(
        "训练未精确停在已完成更新的冻结目标:"
        f" num_timesteps={int(getattr(model, 'num_timesteps', -1))},"
        f" target_global_steps={int(target_global_steps)},"
        f" rollout_full={bool(getattr(getattr(model, 'rollout_buffer', None), 'full', False))},"
        f" calib_tripped={bool(getattr(model, '_calib_tripped', False))},"
        " last_completed_ppo_rollout_steps="
        f"{getattr(model, '_last_completed_ppo_rollout_steps', None)!r},"
        " ppo_optimizer_steps_completed="
        f"{getattr(model, '_ppo_optimizer_steps_completed', None)!r},"
        " critic_warmup_complete="
        f"{getattr(model, '_critic_warmup_completed', None)!r},"
        " critic_warmup_optimizer_steps="
        f"{getattr(model, '_critic_warmup_optimizer_steps_completed', None)!r},"
        " actor_optimizer_steps="
        f"{getattr(model, '_actor_optimizer_steps_completed', None)!r},"
        " worker_onpolicy_pg_joint_rollouts="
        f"{getattr(model, '_worker_onpolicy_pg_joint_rollouts', None)!r},"
        " worker_onpolicy_pg_qualifying_rollouts="
        f"{getattr(model, '_worker_onpolicy_pg_qualifying_rollouts', None)!r},"
        " actor_context_enabled="
        f"{getattr(getattr(getattr(model, 'policy', None), 'mlp_extractor', None), 'actor_context_enabled', None)!r}")


class AtomicRolloutCheckpointCallback(BaseCallback):
    """只在上一 rollout 已完成更新的边界保存，杜绝“步数已记、梯度未吃”。"""

    def __init__(self, run_dir: pathlib.Path, every_steps: int = 250_000,
                 implementation_sha256: str | None = None):
        super().__init__()
        self.run_dir = run_dir
        self.every_steps = every_steps
        self.implementation_sha256 = implementation_sha256
        self.period = None
        self.next_at = None
        self.last_saved = None

    def _on_training_start(self) -> None:
        quantum = int(self.model.n_steps * self.model.get_env().num_envs)
        self.period = max(quantum, (self.every_steps // quantum) * quantum)
        self.next_at = int(self.num_timesteps + self.period)

    def _save_due(self, *, rollout_buffer_may_be_reset: bool = False) -> None:
        if self.next_at is None or self.num_timesteps < self.next_at:
            return
        # ``_on_rollout_start`` normally follows a completed ``train()``, but a
        # calibration trip or any other zero-update/stale-receipt path can leave
        # num_timesteps one rollout ahead of the weights.  Reuse the same strong
        # predicate as final publication instead of inferring completion merely
        # from callback timing.
        if not _is_publishable_rollout_boundary(
                self.model,
                rollout_buffer_may_be_reset=rollout_buffer_may_be_reset):
            if bool(getattr(self.model, "_calib_tripped", False)):
                print("   [G-CAL] 当前 rollout 未完成更新，拒绝发布 checkpoint")
            else:
                print("   当前 rollout 缺完整 PPO 更新回执，拒绝发布 checkpoint")
            return
        step = int(self.num_timesteps)
        if self.last_saved != step:
            if self.implementation_sha256 is not None:
                actual = _implementation_bundle_sha256()
                _require(actual == self.implementation_sha256,
                         "训练期间实现/引擎/游戏内容发生漂移，拒绝发布 checkpoint: "
                         f"{actual} != {self.implementation_sha256}")
            path = self.run_dir / "ckpt" / f"model_{step}_steps.zip"
            _atomic_save_model(self.model, path)
            self.last_saved = step
            print(f"   rollout-boundary checkpoint: {path}")
        while self.next_at <= step:
            self.next_at += self.period

    def _on_rollout_start(self) -> None:
        # 首次调用 num_timesteps=起点；后续调用发生在上一 rollout train() 之后。
        self._save_due(rollout_buffer_may_be_reset=True)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        # 正常收官不会进入下一次 _on_rollout_start；只有 buffer.full 才表示
        # 最后一批确已 train()。回调中途终止的半 rollout 不得冒充 checkpoint。
        if _is_publishable_rollout_boundary(self.model):
            self._save_due()


class WorkerSentinelCallback(BaseCallback):
    """v23 哨兵(PREREG 附录A/C):每 500k 步汇总子进程 WorkerWindowEnv.stats
    (干/鲜层窗配比、终止原因谱、兜底滚局数)+ 累计动作份额 → sentinel.jsonl。
    塌缩裁决本身走 2M/4M 检查点组装重放(附录C),此处只供遥测与验尸。"""

    def __init__(self, run_dir: pathlib.Path, every: int = 500_000):
        super().__init__()
        self.run_dir = run_dir
        self.every = every
        self.next_at = every
        self.action_counts = None
        self._last_emit_step = None

    def _on_training_start(self) -> None:
        # v24 修正:resume 腿的全局步不从 0 起——对齐到下一个 500k 边界,防空喷
        self.next_at = ((self.num_timesteps // self.every) + 1) * self.every

    def _on_step(self) -> bool:
        import numpy as np
        # v24 G-CAL:标定探针置旗即终止本腿(驱动裁决重标定,预注册条款)
        if getattr(self.model, "_calib_tripped", False):
            print("   [G-CAL] 生产校准硬门触发 —— 终止本腿,交驱动裁决")
            return False
        acts = self.locals.get("actions")
        if acts is not None:
            if self.action_counts is None:
                self.action_counts = np.zeros(15, dtype=np.int64)
            for a in np.asarray(acts).ravel():
                self.action_counts[int(a)] += 1
        if self.num_timesteps >= self.next_at:
            while self.num_timesteps >= self.next_at:
                self.next_at += self.every
            self._emit(final=False)
        return True

    def _emit(self, final: bool) -> None:
        import numpy as np

        step = int(self.num_timesteps)
        if self._last_emit_step == step:
            return
        per_env = self.model.get_env().get_attr("stats")   # 经 Monitor.__getattr__ 透传
        count_keys = (
            "windows", "dry", "fresh", "ff_windows", "ff_dry",
            "ff_terminals", "episodes", "reseeds",
            "interrupted_resets", "manual_ff_calls",
            "direct_terminal_deaths",
            "transition_ff_terminal_deaths",
            "reset_ff_terminal_deaths",
            "manual_ff_terminal_deaths",
            "direct_no_progress_timeouts",
            "transition_ff_no_progress_timeouts",
            "reset_ff_no_progress_timeouts",
            "manual_ff_no_progress_timeouts",
        )
        reward_keys = (
            "transition_ff_reward", "reset_ff_reward", "manual_ff_reward",
            "direct_existing_terminal_death_reward",
            "direct_additional_terminal_death_reward",
            "transition_ff_terminal_death_reward",
            "transition_ff_additional_terminal_death_reward",
            "credited_ff_terminal_death_reward",
            "reset_ff_terminal_death_reward",
            "reset_ff_additional_terminal_death_reward",
            "additional_terminal_death_reward",
            "direct_no_progress_timeout_failure_reward",
            "transition_ff_no_progress_timeout_failure_reward",
            "reset_ff_no_progress_timeout_failure_reward",
            "manual_ff_no_progress_timeout_failure_reward",
            "credited_no_progress_timeout_failure_reward",
        )
        env = self.model.get_env()
        credit_modes = env.get_attr("fast_forward_reward_credit")
        configured_costs = env.get_attr("additional_terminal_death_cost")
        _require(
            isinstance(per_env, (list, tuple)) and len(per_env) > 0
            and len(credit_modes) == len(per_env)
            and len(configured_costs) == len(per_env),
            "Worker sentinel 子环境 stats/奖励配置数量不闭合",
        )
        _require(
            all(isinstance(mode, str)
                and mode in {"none", "terminal-death-only"}
                for mode in credit_modes)
            and len(set(credit_modes)) == 1,
            "Worker sentinel 子环境 fast-forward reward credit 漂移",
        )
        normalized_costs = [
            _finite_number(value, "Worker sentinel additional death cost")
            for value in configured_costs
        ]
        _require(
            all(value >= 0.0 for value in normalized_costs)
            and all(value == normalized_costs[0]
                    for value in normalized_costs),
            "Worker sentinel 子环境 additional death cost 漂移/非法",
        )
        agg = {key: 0 for key in count_keys}
        agg.update({key: 0.0 for key in reward_keys})
        reasons = {}
        ff_reasons = {}
        for env_index, stats in enumerate(per_env):
            _require(
                isinstance(stats, dict)
                and all(key in stats for key in count_keys)
                and all(key in stats for key in reward_keys)
                and isinstance(stats.get("reasons"), dict)
                and isinstance(stats.get("ff_reasons"), dict),
                f"Worker sentinel env[{env_index}] stats 字段不完整",
            )
            for key in count_keys:
                value = stats[key]
                _require(
                    _is_plain_int(value) and value >= 0,
                    f"Worker sentinel env[{env_index}].{key} "
                    "必须是非负普通整数",
                )
                agg[key] += value
            for key in reward_keys:
                agg[key] += _finite_number(
                    stats[key],
                    f"Worker sentinel env[{env_index}].{key}",
                )
            for key, value in stats["reasons"].items():
                _require(
                    isinstance(key, str) and bool(key)
                    and _is_plain_int(value) and value >= 0,
                    f"Worker sentinel env[{env_index}] reasons 非法",
                )
                reasons[key] = reasons.get(key, 0) + value
            for key, value in stats["ff_reasons"].items():
                _require(
                    isinstance(key, str) and bool(key)
                    and _is_plain_int(value) and value >= 0,
                    f"Worker sentinel env[{env_index}] ff_reasons 非法",
                )
                ff_reasons[key] = ff_reasons.get(key, 0) + value
        for key in reward_keys:
            agg[key] = round(agg[key], 6)
        top1 = int(self.action_counts.argmax()) if self.action_counts is not None else -1
        share = (float(self.action_counts[top1] / max(1, self.action_counts.sum()))
                 if top1 >= 0 else 0.0)
        line = {"sentinel": "v23", "step": step, **agg,
                "dry_share": round(agg["dry"] / max(1, agg["dry"] + agg["fresh"]), 4),
                "reasons": reasons, "ff_reasons": ff_reasons,
                "fast_forward_reward_credit_mode": credit_modes[0],
                "configured_additional_terminal_death_cost":
                    normalized_costs[0],
                "top1_action": top1, "top1_share": round(share, 4),
                # v24 皮筋读数(与 gate_ledger 双簿对账)
                "beta_initial":
                    getattr(self.model, "distill_beta", None),
                "beta": getattr(
                    self.model,
                    "_last_effective_distill_beta",
                    None),
                "distill_actor_rollouts_completed": getattr(
                    self.model,
                    "_distill_actor_rollouts_completed",
                    None),
                "distill_ce": getattr(self.model, "_last_distill_ce", None),
                "teacher_entropy": getattr(
                    self.model, "_last_teacher_entropy", None),
                "distill_kl": getattr(
                    self.model, "_last_distill_kl", None),
                "distill_tv": getattr(
                    self.model, "_last_distill_tv", None),
                "teacher_diverge": getattr(self.model, "_last_diverge", None)}
        if final:
            line["final"] = True
        with open(self.run_dir / "sentinel.jsonl", "a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._last_emit_step = step
        print(f"   [哨兵] {line}")

    def _on_training_end(self) -> None:
        # 腿长常取 499,712(<500k)；若只按间隔写，完整腿反而零哨兵记录。
        if self.num_timesteps > 0 and self._last_emit_step != int(self.num_timesteps):
            self._emit(final=True)


class DryAnchorSentinel(BaseCallback):
    """v26 干层锚哨兵(只记不裁):demos.npz 中双通道 cap-1 前沿教师样本
    固定抽 2000,每 500k 步测学生 argmax 对教师标签的失配率——skip_dry 下干层行为
    无锚裸奔,这只表是它唯一的观察者。"""

    def __init__(self, run_dir: pathlib.Path, demos_npz: str,
                 expected_sha256: str, every: int = 500_000):
        super().__init__()
        import numpy as np
        self.run_dir = run_dir
        self.every = every
        self.next_at = every
        self._last_emit_step = None
        X, Y, self.demos_sha256 = _load_dry_anchor_demos(
            demos_npz, expected_sha256)
        m, _ = _dry_anchor_partition(X)
        idx = np.random.default_rng(26).choice(
            np.flatnonzero(m), size=min(2000, int(m.sum())), replace=False)
        self.X, self.Y = X[idx], Y[idx]
        if (not np.isfinite(self.X).all() or np.any(self.Y < 0)
                or np.any(self.Y >= 15)):
            raise ValueError("dry-anchor 样本含非有限观测或越界标签")

    def _on_training_start(self) -> None:
        self.next_at = ((self.num_timesteps // self.every) + 1) * self.every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_at:
            while self.num_timesteps >= self.next_at:
                self.next_at += self.every
            self._emit(final=False)
        return True

    def _emit(self, final: bool) -> None:
        import numpy as np
        import torch as th

        step = int(self.num_timesteps)
        if self._last_emit_step == step:
            return
        with th.no_grad():
            raw_obs = th.as_tensor(self.X, device=self.model.device)
            # 干层锚系只记不裁遥测:掩码保留 v28-v30 旧口径(11/12 恒掩)
            # 以维持跨腿失配曲线同尺——v32 主权后部署掩码含 12,但教师
            # 标签(demos)无 12,主权行为由评测 a12 仪表另量(PREREG-v32
            # 口径注);改此口径会断代历史遥测,故如实登记不改。
            masks = th.ones((len(self.X), 15), dtype=th.bool,
                            device=self.model.device)
            masks[:, 11] = masks[:, 12] = False
            masks[:, 14] = raw_obs[:, _GEAR_PRESENT_INDEX] > 0.5
            policy_obs = _probe_policy_observation_view(
                self.model, raw_obs)
            dist = self.model.policy.get_distribution(
                policy_obs, action_masks=masks)
            pred = dist.distribution.logits.argmax(-1).cpu().numpy()
        mis = float((pred != self.Y).mean())
        line = {"sentinel": "dry-anchor", "step": step,
                "mismatch": round(mis, 4), "n": int(len(self.Y))}
        if final:
            line["final"] = True
        with open(self.run_dir / "sentinel.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        self._last_emit_step = step
        print(f"   [干层锚] {line}")

    def _on_training_end(self) -> None:
        if self.num_timesteps > 0 and self._last_emit_step != int(self.num_timesteps):
            self._emit(final=True)


class DryCurriculumCallback(BaseCallback):
    """E1 ⑤A:register the next p_skip before collecting each rollout.

    - p 表锚定腿相对 rollout 序号 (num_timesteps − 3,497,984)/2048,
      全局步锚定禁用;腿前/失准/越界一律抛(禁钳位——对抗席"越界钳至表尾
      恒 0.5"构造由此关死);
    - schedule_table 属性暴露"序号→p"全表(供驱动器落 DRY_CURRICULUM_TABLE);
    - 首值在训练启动时直接对齐；此后 rollout-start 经
      ``schedule_skip_dry_p(next_p, n_steps)`` 给每个 Worker env 登记独立
      倒计时。第 n 个 Worker action 完成后，环境在构造 next_obs、跨窗或
      VecEnv auto-reset 前原子提交 next_p；
    - rollout-tail 回调只核验提交回执及 feature 601，不再事后首次切换；
      逐 rollout 账与 env 读回值仍须精确等于注册表，任一失配即抛。
    """

    def __init__(self, schedule_table, run_dir: pathlib.Path | None = None,
                 leg_start: int = _DRY_CURRICULUM_LEG_START):
        super().__init__()
        table = tuple(float(p) for p in schedule_table)
        _require(len(table) > 0, "dry-curriculum p 表不能为空")
        _require(all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in table),
                 "dry-curriculum p 表必须全部在 [0, 1] 内")
        self.schedule_table = table
        self.leg_start = int(leg_start)
        self.run_dir = pathlib.Path(run_dir) if run_dir is not None else None
        self.pushed: list[dict] = []   # 逐 rollout 实际推送账(序号→p)
        self.quantum = None
        self._active_index = None
        self._active_p = None
        self._boundary_preapplied_index = None
        self._scheduled_next_index = None
        self._scheduled_next_p = None

    def _on_training_start(self) -> None:
        self.quantum = int(self.model.n_steps * self.model.get_env().num_envs)
        index = self._rollout_index()
        self._activate(
            index,
            getattr(self.model, "_last_obs", None),
            boundary_preapply=False,
        )

    def _rollout_index(self) -> int:
        offset = int(self.num_timesteps) - self.leg_start
        _require(offset >= 0,
                 f"dry-curriculum 腿相对锚定失义: num_timesteps={self.num_timesteps} "
                 f"在腿起点 {self.leg_start} 之前(全局步锚定禁用,腿恒自王 zip 复点火)")
        _require(offset % self.quantum == 0,
                 f"dry-curriculum rollout 边界失准: 腿内偏移 {offset} "
                 f"不是量子 {self.quantum} 的整数倍")
        index = offset // self.quantum
        _require(index < len(self.schedule_table),
                 f"dry-curriculum 腿相对 rollout 序号 {index} 越界"
                 f"(p 表长 {len(self.schedule_table)},禁钳位)")
        return index

    def _refresh_dual_observation(
            self, observation, p: float, label: str) -> bool:
        """Refresh the cached Markov state when only curriculum p changes."""
        if observation is None:
            return False
        import numpy as np
        from diablogym.options_env import (
            DUAL_WORKER_OBSERVATION_DIM,
            DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE,
        )

        _require(
            isinstance(observation, np.ndarray)
            and observation.ndim == 2
            and observation.shape[0]
            == int(self.model.get_env().num_envs)
            and observation.shape[1] in {
                298, DUAL_WORKER_OBSERVATION_DIM},
            f"dry-curriculum {label} cached observation 形状异常:"
            f"{getattr(observation, 'shape', None)!r}",
        )
        if observation.shape[1] != DUAL_WORKER_OBSERVATION_DIM:
            return False
        _require(
            bool(observation.flags.writeable)
            and np.issubdtype(observation.dtype, np.floating)
            and bool(np.isfinite(observation).all()),
            f"dry-curriculum {label} dual cached observation "
            "必须可写、浮点且有限",
        )
        observation[
            :, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE] = p
        _require(
            bool(np.all(
                observation[
                    :, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]
                == np.asarray(p, dtype=observation.dtype)
            )),
            f"dry-curriculum {label} dual p_skip 缓存刷新失败",
        )
        return True

    def _verify_active_probability(self, index: int, p: float) -> None:
        env = self.model.get_env()
        # rev3 E1② 恒等断言:实际在位 p(逐 env 读回)≡ 注册表对应项,失配即抛。
        for rank, actual in enumerate(env.get_attr("skip_dry")):
            _require(float(actual) == p,
                     f"dry-curriculum 恒等断言失配: env[{rank}] 在位 p={actual} "
                     f"!= 注册表[{index}]={p}")

    def _schedule_states(self) -> list[dict]:
        states = self.model.get_env().env_method(
            "skip_dry_schedule_state")
        _require(
            isinstance(states, (list, tuple))
            and len(states) == int(self.model.get_env().num_envs)
            and all(isinstance(state, dict) for state in states),
            "dry-curriculum per-env schedule state 形状异常",
        )
        return list(states)

    def _verify_schedule_states(
            self, *, current_p: float,
            pending_p: float | None, remaining: int) -> None:
        expected_keys = {
            "current_probability",
            "pending_probability",
            "remaining_env_steps",
        }
        for rank, state in enumerate(self._schedule_states()):
            _require(
                set(state) == expected_keys
                and float(state["current_probability"]) == current_p
                and (
                    state["pending_probability"] is None
                    if pending_p is None
                    else float(state["pending_probability"]) == pending_p
                )
                and state["remaining_env_steps"] == remaining,
                "dry-curriculum per-env schedule state 失配:"
                f"env[{rank}]={state!r},"
                f"expected=({current_p},{pending_p},{remaining})",
            )

    def _register_next_probability(self, index: int, p: float) -> None:
        _require(
            self._scheduled_next_index is None
            and self._scheduled_next_p is None,
            "dry-curriculum 上一 rollout 倒计时尚未闭合",
        )
        target = getattr(self.model, "_total_timesteps", None)
        _require(
            _is_plain_int(target)
            and int(target) >= int(self.num_timesteps) + int(self.quantum),
            "dry-curriculum 缺有效全局训练终点",
        )
        has_next_rollout = (
            int(self.num_timesteps) + int(self.quantum) < int(target)
        )
        self._verify_schedule_states(
            current_p=p, pending_p=None, remaining=0)
        if not has_next_rollout:
            return
        next_index = int(index) + 1
        _require(
            next_index < len(self.schedule_table),
            "dry-curriculum 下一 rollout 超出注册表:"
            f"{next_index} >= {len(self.schedule_table)}",
        )
        next_p = float(self.schedule_table[next_index])
        remaining = int(self.model.n_steps)
        receipts = self.model.get_env().env_method(
            "schedule_skip_dry_p", next_p, remaining)
        _require(
            isinstance(receipts, (list, tuple))
            and len(receipts) == int(self.model.get_env().num_envs),
            "dry-curriculum 倒计时登记回执形状异常",
        )
        self._verify_schedule_states(
            current_p=p, pending_p=next_p, remaining=remaining)
        self._scheduled_next_index = next_index
        self._scheduled_next_p = next_p

    def _verify_dual_observation_probability(
            self, observation, p: float, label: str) -> bool:
        if observation is None:
            return False
        import numpy as np
        from diablogym.options_env import (
            DUAL_WORKER_OBSERVATION_DIM,
            DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE,
        )

        _require(
            isinstance(observation, np.ndarray)
            and observation.ndim == 2
            and observation.shape[0]
            == int(self.model.get_env().num_envs)
            and observation.shape[1] in {
                298, DUAL_WORKER_OBSERVATION_DIM},
            f"dry-curriculum {label} cached observation 形状异常:"
            f"{getattr(observation, 'shape', None)!r}",
        )
        if observation.shape[1] != DUAL_WORKER_OBSERVATION_DIM:
            return False
        expected = np.asarray(p, dtype=observation.dtype)
        _require(
            bool(np.all(
                observation[
                    :, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]
                == expected
            )),
            f"dry-curriculum {label} dual p_skip 尚未由环境原子切换",
        )
        return True

    def _activate(
            self, index: int, observation,
            *, boundary_preapply: bool) -> bool:
        p = self.schedule_table[index]
        env = self.model.get_env()
        env.env_method("set_skip_dry_p", p)   # 经 Monitor __getattr__ 透传
        self._verify_active_probability(index, p)
        refreshed = self._refresh_dual_observation(
            observation, p,
            "rollout-tail" if boundary_preapply else "rollout-start",
        )
        self._active_index = int(index)
        self._active_p = float(p)
        self._boundary_preapplied_index = (
            int(index) if boundary_preapply else None)
        return refreshed

    def _on_rollout_start(self) -> None:
        index = self._rollout_index()
        p = self.schedule_table[index]
        preapplied = self._boundary_preapplied_index == index
        _require(
            self._active_index == index and self._active_p == p,
            "dry-curriculum rollout 边界未由环境内倒计时原子切换:"
            f"active=({self._active_index},{self._active_p}),"
            f"expected=({index},{p})",
        )
        self._verify_active_probability(index, p)
        refreshed = self._refresh_dual_observation(
            getattr(self.model, "_last_obs", None),
            p,
            "rollout-start",
        )
        entry = {"rollout_index": int(index), "p": float(p),
                 "num_timesteps": int(self.num_timesteps),
                 "boundary_preapplied": bool(preapplied),
                 "cached_dual_observation_refreshed": bool(refreshed)}
        self.pushed.append(entry)
        if self.run_dir is not None:
            with open(self.run_dir / "dry_curriculum.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
        self._boundary_preapplied_index = None
        self._register_next_probability(index, p)

    def _on_step(self) -> bool:
        # WorkerWindowEnv owns the actual boundary switch: every environment
        # counts successful Worker actions and commits the pending probability
        # immediately after action N, before it constructs new_obs or returns a
        # terminal that VecEnv will auto-reset.  This callback only verifies
        # that atomic commit and refreshes the exact ndarray later used for the
        # rollout-tail value and next rollout cache.
        n_steps = self.locals.get("n_steps")
        n_rollout_steps = self.locals.get("n_rollout_steps")
        if not (
            _is_plain_int(n_steps)
            and _is_plain_int(n_rollout_steps)
            and n_steps == n_rollout_steps - 1
        ):
            return True
        if self._scheduled_next_index is None:
            _require(
                self._scheduled_next_p is None,
                "dry-curriculum next index/p 部分登记",
            )
            self._verify_schedule_states(
                current_p=float(self._active_p),
                pending_p=None,
                remaining=0,
            )
            return True

        next_index = self._rollout_index()
        next_p = float(self.schedule_table[next_index])
        _require(
            self._active_index is not None
            and next_index == self._active_index + 1
            and next_index == self._scheduled_next_index
            and next_p == self._scheduled_next_p,
            "dry-curriculum rollout-tail 序号/概率未连续:"
            f"active={self._active_index},"
            f"scheduled=({self._scheduled_next_index},"
            f"{self._scheduled_next_p}),next=({next_index},{next_p})",
        )
        self._verify_active_probability(next_index, next_p)
        self._verify_schedule_states(
            current_p=next_p, pending_p=None, remaining=0)
        new_obs = self.locals.get("new_obs")
        self._verify_dual_observation_probability(
            new_obs, next_p, "rollout-tail")
        self._refresh_dual_observation(
            new_obs, next_p, "rollout-tail")
        self._active_index = next_index
        self._active_p = next_p
        self._boundary_preapplied_index = next_index
        self._scheduled_next_index = None
        self._scheduled_next_p = None
        return True


# ---- E5 新增仪表(PREREG-内容案 E5;全数只记不裁,默认零侵入,纳入 W-G0
# 证明范围;探针示范集绑定当前协议严格 PASS 的 BC-v1 回执) ----


def _probe_legacy_masks(obs):
    """E5 探针共用旧口径掩码(承 DryAnchorSentinel v28-v30 口径逐字:11/12 恒掩,
    14 依 gear 位)——示范集系 v1 世代无逐样本掩码,完整部署掩码不可自 obs 全量
    重构(E2 注记,反推口径系第二真源禁用);取跨腿同尺之旧口径并在输出行注记
    mask_mode,只记不裁(施工裁量,交接单单列)。"""
    import torch as th

    masks = th.ones((len(obs), 15), dtype=th.bool, device=obs.device)
    masks[:, 11] = masks[:, 12] = False
    masks[:, 14] = obs[:, _GEAR_PRESENT_INDEX] > 0.5
    return masks


def _probe_policy_observation_view(model, raw_obs):
    """Feed probe rows through the same policy-view boundary as deployment.

    A12 custom policies need the untouched signed latch for their contextual
    gate and decode only the inherited actor/value path internally.  Every
    ordinary Worker policy consumes the canonical protocol-v3 view.
    """
    from leashed_ppo import (
        A12MixtureMaskableActorCriticPolicy,
        ASYMMETRIC_WORKER_LEGACY_DIM,
        ASYMMETRIC_WORKER_OBSERVATION_DIM,
        AsymmetricWorkerMaskableActorCriticPolicy,
        _legacy_worker_observation_view,
    )

    if isinstance(model.policy, A12MixtureMaskableActorCriticPolicy):
        return raw_obs
    if isinstance(model.policy, AsymmetricWorkerMaskableActorCriticPolicy):
        if raw_obs.shape[-1] == ASYMMETRIC_WORKER_OBSERVATION_DIM:
            return raw_obs
        _require(
            raw_obs.shape[-1] == ASYMMETRIC_WORKER_LEGACY_DIM,
            "asymmetric Worker 离线探针只接受 298 或当前 dual 维观测",
        )
        # Historical dry-anchor rows contain no trustworthy v4/controller
        # context.  Preserve their exact v3 root and explicitly mark every
        # appended field unknown-as-zero; never reinterpret old columns as
        # current state merely to satisfy the wider policy shape.
        padded = raw_obs.new_zeros(
            (*raw_obs.shape[:-1], ASYMMETRIC_WORKER_OBSERVATION_DIM))
        padded[..., :ASYMMETRIC_WORKER_LEGACY_DIM] = (
            _legacy_worker_observation_view(raw_obs))
        return padded
    return _legacy_worker_observation_view(raw_obs)


class DistillCeProbe(BaseCallback):
    """E5① 干/鲜 distill_ce 分列离线探针(只记不裁;DryAnchorSentinel 之孪生件)。

    固定示范态集(当前协议 BC-v1 PASS 回执绑定)上按 col296 short clock 或
    col297 解码后的 farm_scene_fraction 是否进入 cap-1 前沿分干/鲜两组,
    算教师-学生 distill CE(公式镜像 leashed_ppo train() 皮筋段:
    ce = −Σ t_probs·logp_all 之均值;教师/学生同喂旧口径掩码);专用 rng
    (承 dry-anchor rng(26) 先例),零触训练路径(纯读+IO,不碰训练 RNG/梯度/
    env 流)。训练内 buffer 分列为守 G0-2a 零侵入证明面而废止(工程 M2),
    本件即其注册替代形制;课②定标数据供给义务(圈 12 改写)同由此满足。
    输出 run_dir/distill_ce_probe.jsonl。
    """

    def __init__(self, run_dir: pathlib.Path, demos_npz: str, every: int):
        super().__init__()
        import numpy as np

        self.run_dir = run_dir
        self.every = int(every)
        _require(self.every > 0, "distill-ce 探针间隔必须 > 0")
        self.next_at = self.every
        self._last_emit_step = None
        expected_sha = _assert_bc_v1_demos_frozen(demos_npz)
        X, _, self.demos_sha256 = _load_dry_anchor_demos(
            demos_npz, expected_sha)
        dry, fresh = _dry_anchor_partition(X)
        dry_rows = np.flatnonzero(dry)
        fresh_rows = np.flatnonzero(fresh)
        _require(len(dry_rows) > 0 and len(fresh_rows) > 0,
                 "distill-ce 探针需干/鲜两组示范态均非空(fail-loud)")
        rng = np.random.default_rng(_E5_PROBE_RNG_SEED)
        dry_idx = rng.choice(dry_rows,
                             size=min(_E5_PROBE_GROUP_CAP, len(dry_rows)),
                             replace=False)
        fresh_idx = rng.choice(fresh_rows,
                               size=min(_E5_PROBE_GROUP_CAP, len(fresh_rows)),
                               replace=False)
        self.X_dry, self.X_fresh = X[dry_idx], X[fresh_idx]
        if not (np.isfinite(self.X_dry).all() and np.isfinite(self.X_fresh).all()):
            raise ValueError("distill-ce 探针样本含非有限观测")

    def _on_training_start(self) -> None:
        self.next_at = ((self.num_timesteps // self.every) + 1) * self.every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_at:
            while self.num_timesteps >= self.next_at:
                self.next_at += self.every
            self._emit(final=False)
        return True

    def _group_metrics(self, x) -> dict:
        import torch as th

        from leashed_ppo import (
            _legacy_distillation_masks,
            _masked_log_softmax_from_raw,
        )

        with th.no_grad():
            raw_obs = th.as_tensor(x, device=self.model.device)
            masks = _probe_legacy_masks(raw_obs)
            policy_obs = _probe_policy_observation_view(
                self.model, raw_obs)
            distill_masks = _legacy_distillation_masks(
                masks)
            t_probs = self.model._teacher_probs(
                policy_obs, distill_masks)
            student_logits = self.model._student_distillation_logits(
                policy_obs)
            logp_all = _masked_log_softmax_from_raw(
                student_logits, distill_masks)
            teacher_logp = th.where(
                t_probs > 0.0,
                th.log(t_probs),
                th.zeros_like(t_probs),
            )
            ce = -(t_probs * logp_all).sum(dim=-1).mean()
            entropy = -(
                t_probs * teacher_logp).sum(dim=-1).mean()
            kl = (
                t_probs * (teacher_logp - logp_all)
            ).sum(dim=-1).mean()
            tv = 0.5 * th.abs(
                t_probs - logp_all.exp()).sum(dim=-1).mean()
            return {
                "ce": float(ce),
                "teacher_entropy": float(entropy),
                "kl": float(kl),
                "tv": float(tv),
            }

    def _group_ce(self, x) -> float:
        """Compatibility accessor for the historical CE-only probe."""
        return self._group_metrics(x)["ce"]

    def _emit(self, final: bool) -> None:
        step = int(self.num_timesteps)
        if self._last_emit_step == step:
            return
        _require(getattr(self.model, "teacher", None) is not None,
                 "distill-ce 探针需教师在位(Leashed teacher;fail-loud)")
        dry = self._group_metrics(self.X_dry)
        fresh = self._group_metrics(self.X_fresh)
        line = {"probe": "distill-ce", "step": step,
                "dry_ce": round(dry["ce"], 6),
                "dry_teacher_entropy":
                    round(dry["teacher_entropy"], 6),
                "dry_kl": round(dry["kl"], 6),
                "dry_tv": round(dry["tv"], 6),
                "dry_n": int(len(self.X_dry)),
                "fresh_ce": round(fresh["ce"], 6),
                "fresh_teacher_entropy":
                    round(fresh["teacher_entropy"], 6),
                "fresh_kl": round(fresh["kl"], 6),
                "fresh_tv": round(fresh["tv"], 6),
                "fresh_n": int(len(self.X_fresh)),
                "beta_initial":
                    getattr(self.model, "distill_beta", None),
                "beta": getattr(
                    self.model,
                    "_last_effective_distill_beta",
                    None),
                "distill_actor_rollouts_completed": getattr(
                    self.model,
                    "_distill_actor_rollouts_completed",
                    None),
                "mask_mode": "legacy-root-exclude-a12-a14",
                "demos_sha16": self.demos_sha256[:16]}
        if final:
            line["final"] = True
        with open(self.run_dir / "distill_ce_probe.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        self._last_emit_step = step
        print(f"   [干/鲜蒸馏探针] {line}")

    def _on_training_end(self) -> None:
        if self.num_timesteps > 0 and self._last_emit_step != int(self.num_timesteps):
            self._emit(final=True)


class DryWindowMetricsCallback(BaseCallback):
    """E5② 干窗行为仪表(只记不裁;审计缺口 i 之闭合起点,基线自本案首建)。

    两读数面,均挂现有采样面、零新增训练侧接触:
    ① 干态动作分布——固定干态示范集(当前协议 BC-v1 PASS 回执绑定,
       dry-anchor 同款采样面)上学生策略之分布熵与 argmax 直方图(旧口径掩码,
       mask_mode 注记随行);
    ② 窗口经济——SB3 rollout infos 流之窗末 option_extra(学习窗,快进窗
       不经此流)按干/鲜分组聚合工资 W 与宽度(τ̄/depth=dlvl_end);逐 emit
       区间清零(区间局部均值);n=0 组记 n:0、均值 null 不消失(fail-closed)。
    输出 run_dir/drywin_metrics.jsonl(台账词 DRYWIN_METRICS 之进程侧原料)。
    """

    _WINDOW_KEYS = ("n", "wage_sum", "tau_sum", "depth_sum")

    def __init__(self, run_dir: pathlib.Path, demos_npz: str, every: int):
        super().__init__()
        import numpy as np

        self.run_dir = run_dir
        self.every = int(every)
        _require(self.every > 0, "drywin 仪表间隔必须 > 0")
        self.next_at = self.every
        self._last_emit_step = None
        expected_sha = _assert_bc_v1_demos_frozen(demos_npz)
        X, _, self.demos_sha256 = _load_dry_anchor_demos(
            demos_npz, expected_sha)
        dry, _ = _dry_anchor_partition(X)
        dry_rows = np.flatnonzero(dry)
        _require(len(dry_rows) > 0, "drywin 仪表需干态示范集非空(fail-loud)")
        rng = np.random.default_rng(_E5_PROBE_RNG_SEED)
        idx = rng.choice(dry_rows,
                         size=min(_E5_PROBE_GROUP_CAP, len(dry_rows)),
                         replace=False)
        self.X_dry = X[idx]
        if not np.isfinite(self.X_dry).all():
            raise ValueError("drywin 仪表干态样本含非有限观测")
        self._acc = self._fresh_acc()

    @classmethod
    def _fresh_acc(cls) -> dict:
        return {group: dict.fromkeys(cls._WINDOW_KEYS, 0)
                for group in ("dry", "fresh")}

    def _on_training_start(self) -> None:
        self.next_at = ((self.num_timesteps // self.every) + 1) * self.every

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            extra = info.get("option_extra") if isinstance(info, dict) else None
            if extra is None:
                continue
            acc = self._acc["dry" if extra.get("dry") else "fresh"]
            acc["n"] += 1
            acc["wage_sum"] += float(extra["W"])
            acc["tau_sum"] += float(extra["tau"])
            acc["depth_sum"] += float(extra["dlvl_end"])
        if self.num_timesteps >= self.next_at:
            while self.num_timesteps >= self.next_at:
                self.next_at += self.every
            self._emit(final=False)
        return True

    def _dry_state_readout(self) -> tuple[float, list[int]]:
        import numpy as np
        import torch as th

        with th.no_grad():
            raw_obs = th.as_tensor(self.X_dry, device=self.model.device)
            masks = _probe_legacy_masks(raw_obs)
            policy_obs = _probe_policy_observation_view(
                self.model, raw_obs)
            dist = self.model.policy.get_distribution(
                policy_obs, action_masks=masks)
            entropy = float(dist.distribution.entropy().mean())
            pred = dist.distribution.logits.argmax(-1).cpu().numpy()
        hist = np.bincount(pred, minlength=15)
        return entropy, [int(count) for count in hist]

    @staticmethod
    def _window_summary(acc: dict) -> dict:
        n = acc["n"]
        mean = (lambda total: round(total / n, 4) if n else None)
        return {"n": int(n), "wage_mean": mean(acc["wage_sum"]),
                "tau_mean": mean(acc["tau_sum"]),
                "depth_mean": mean(acc["depth_sum"])}

    def _emit(self, final: bool) -> None:
        step = int(self.num_timesteps)
        if self._last_emit_step == step:
            return
        entropy, hist = self._dry_state_readout()
        line = {"metrics": "drywin", "step": step,
                "dry_state_entropy": round(entropy, 6),
                "dry_state_n": int(len(self.X_dry)),
                "dry_state_argmax_hist": hist,
                "windows": {group: self._window_summary(self._acc[group])
                            for group in ("dry", "fresh")},
                "mask_mode": "dry-anchor-legacy",
                "demos_sha16": self.demos_sha256[:16]}
        if final:
            line["final"] = True
        with open(self.run_dir / "drywin_metrics.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        self._acc = self._fresh_acc()
        self._last_emit_step = step
        print(f"   [干窗行为] {line}")

    def _on_training_end(self) -> None:
        if self.num_timesteps > 0 and self._last_emit_step != int(self.num_timesteps):
            self._emit(final=True)


# E5③ 金丝雀 a12/局 中期仪表(检查点离线序列用;可独立调用的统计函数+记录器,
# 不挂训练回调——训练路径零接触,RC.11 逐点如实登记,只记不裁)。
_A12_CANARY_SCHEMA_VERSION = "a12-canary/1"
_A12_CANARY_STATS_KEYS = frozenset({
    "episodes", "a12_total", "a12_per_episode", "episodes_with_a12", "a12_max"})


def a12_canary_stats(a12_counts) -> dict:
    """E5③ 统计件:逐局 a12 实饮计数序列 → a12/局 读数(驱动器自检查点评测
    档案逐局提取后调用)。空序列/负数/非整数 fail-loud(零局之 a12/局 无定义,
    禁静默记 0 冒充实测)。"""
    counts = list(a12_counts)
    _require(len(counts) > 0, "a12 金丝雀统计需 ≥1 局(空序列 fail-loud)")
    _require(all(_is_plain_int(count) for count in counts),
             "a12 逐局计数必须全为整数")
    _require(all(count >= 0 for count in counts), "a12 逐局计数不能为负")
    total = sum(counts)
    return {"episodes": len(counts),
            "a12_total": int(total),
            "a12_per_episode": round(total / len(counts), 6),
            "episodes_with_a12": sum(1 for count in counts if count > 0),
            "a12_max": int(max(counts))}


def record_a12_canary(out_path: str | pathlib.Path, *, checkpoint_step: int,
                      manager: str, stats: dict, tag: str | None = None) -> dict:
    """E5③ 记录器:a12 金丝雀读数落 jsonl 一行(台账词 A12_CANARY 之进程侧
    原料;schema 封闭,键集合精确等断言,fail-loud)。返回落笔行。"""
    _require(_is_plain_int(checkpoint_step) and checkpoint_step >= 0,
             "a12 金丝雀 checkpoint_step 必须是非负整数")
    _require(isinstance(manager, str) and bool(manager),
             "a12 金丝雀 manager 必须是非空字符串")
    _require(isinstance(stats, dict) and set(stats) == set(_A12_CANARY_STATS_KEYS),
             f"a12 金丝雀 stats 键集合必须精确等于 {sorted(_A12_CANARY_STATS_KEYS)}")
    line = {"canary": "a12", "schema_version": _A12_CANARY_SCHEMA_VERSION,
            "checkpoint_step": int(checkpoint_step), "manager": manager}
    if tag is not None:
        _require(isinstance(tag, str) and bool(tag),
                 "a12 金丝雀 tag 给定时必须是非空字符串")
        line["tag"] = tag
    line.update(stats)
    with open(out_path, "a") as f:
        f.write(json.dumps(line) + "\n")
    return line


class EpisodeJsonlCallback(BaseCallback):
    """逐局把战绩写进 progress.jsonl;周期性刷新 status.json(供 dashboard 轮询)。"""

    def __init__(self, run_dir: pathlib.Path, config: dict):
        super().__init__()
        self.run_dir = run_dir
        self.config = config
        self.ep_count = 0
        self.t0 = time.time()
        self._progress = open(run_dir / "progress.jsonl", "a", buffering=1)
        self._last_status = 0.0
        self._steps0 = 0

    def _on_training_start(self) -> None:
        # v24 修正:sps 按本腿增量计(resume 腿否则虚高几十倍,降档闸门失明)
        self._steps0 = self.num_timesteps
        self.t0 = time.time()

    def _write_status(self, now: float, training_ended: bool = False) -> None:
        elapsed = now - self.t0
        target_steps = self.config.get("target_global_steps",
                                       self.config["total_steps"])
        rollout_full = bool(getattr(
            getattr(getattr(self, "model", None), "rollout_buffer", None),
            "full", False))
        status = {
            "run": self.run_dir.name,
            "total_steps": int(self.num_timesteps),
            "target_steps": target_steps,
            "start_steps": self.config.get("start_steps", 0),
            "leg_steps": int(self.num_timesteps
                             - self.config.get("start_steps", 0)),
            "leg_target_steps": self.config["total_steps"],
            "episodes": self.ep_count,
            "sps": round((self.num_timesteps - self._steps0) / max(1e-9, elapsed)),
            "elapsed_sec": round(elapsed),
            "updated_at": now,
            "training_ended": training_ended,
            "rollout_full": rollout_full,
            "target_reached": int(self.num_timesteps) >= int(target_steps),
            "config": self.config,
        }
        # dashboard 轮询不应读到半截 JSON。
        tmp = self.run_dir / "status.tmp.json"
        tmp.write_text(json.dumps(status, ensure_ascii=False))
        tmp.replace(self.run_dir / "status.json")

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep is None:
                continue
            self.ep_count += 1
            extra = info.get("episode_extra", {})
            line = {
                "ep": self.ep_count,
                "t": round(time.time() - self.t0, 1),
                "reward": round(float(ep["r"]), 3),
                "len": int(ep["l"]),
                **extra,
            }
            self._progress.write(json.dumps(line, ensure_ascii=False) + "\n")

        now = time.time()
        if now - self._last_status > 1.0:
            self._last_status = now
            self._write_status(now)
        return True

    def _on_training_end(self) -> None:
        # 短训练或早停可在 1s 刷新窗内结束；若不强制落盘，
        # 驱动会把完整腿误判为少训了数步。
        self._write_status(time.time(), training_ended=True)
        self.close()

    def close(self) -> None:
        """异常路径也能幂等关闭逐局日志文件。"""
        if not self._progress.closed:
            self._progress.close()


def _record_run_publication_status(
        run_dir: pathlib.Path, state: str, *,
        model_sha256: str | None = None,
        detail: str | None = None) -> None:
    """Atomically disambiguate "training ended" from "model published".

    ``EpisodeJsonlCallback`` necessarily finishes before the final behavior and
    provenance gates run.  Without a second, terminal status write, a scheduler
    that only watches ``status.json`` can mistake a refused publication for a
    successful run even though the process exits non-zero.
    """
    allowed = {
        "PUBLISHED", "PRODUCTION_CANDIDATE", "DEVELOPMENT_ONLY",
        "TRAINING_ERROR", "PUBLICATION_REFUSED",
    }
    _require(state in allowed, f"未知发布状态: {state}")
    if state in {"PUBLISHED", "PRODUCTION_CANDIDATE", "DEVELOPMENT_ONLY"}:
        _require(_is_sha256(model_sha256),
                 f"{state} 状态必须绑定对应模型 SHA-256")
    else:
        _require(model_sha256 is None,
                 f"{state} 状态不得登记已发布模型 SHA-256")

    status_path = run_dir / "status.json"
    try:
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取末态 status.json: {exc}") from exc
    _require(isinstance(status, dict), "status.json 顶层必须是对象")
    status.update({
        "training_ended": True,
        "publication_status": state,
        "model_published": state == "PUBLISHED",
        "model_production_candidate": state == "PRODUCTION_CANDIDATE",
        "model_development_only": state == "DEVELOPMENT_ONLY",
        "model_sha256": model_sha256,
        "publication_detail": detail,
        "updated_at": time.time(),
    })
    tmp = run_dir / "status.tmp.json"
    tmp.write_text(json.dumps(status, ensure_ascii=False))
    tmp.replace(status_path)


def _main(resources: _TrainingResources):
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=1_998_848,
                    help="新增样本数；必须整除 n_steps×num_envs，禁止 SB3 静默超采")
    ap.add_argument(
        "--artifact-scope",
        choices=tuple(_ARTIFACT_SCOPE_RESULTS),
        default="production",
        help="development 只生成 model_development.zip/DEVELOPMENT_ONLY，"
             "candidate 只生成 model_candidate.zip/PRODUCTION_CANDIDATE；"
             "二者均不得读取 final-heldout 或冒充发布件；production 才生成"
             " model_final.zip/PUBLISHED",
    )
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--device", default="cpu", help="cpu / mps(小 MLP 通常 cpu 更快)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=512, help="每个 env 每轮采样步数")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "rppo", "mppo"],
                    help="rppo = RecurrentPPO/LSTM(B 计划:学习记忆替代手写宏状态机);"
                         "mppo = MaskablePPO(v16:无效动作掩码,env.action_masks)")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "attn"],
                    help="attn = 实体注意力感知(v9:AlphaStar 式 entity encoder + 地图 CNN)")
    ap.add_argument("--max-steps", type=int, default=1500,
                    help="episode 步数上限;1500 = 冠军(v6)配方,3000 = v10 长局实验")
    ap.add_argument("--seed", type=int, default=None,
                    help="训练种子(SB3 全局种子 + 环境 reset 种子;多进程采样时序仍会引入少量不确定性,只保证近似复现)")
    ap.add_argument("--deep", action="store_true",
                    help="v17 深水区:下楼奖金层数递进(N→N+1 付 8×N);配合 --max-steps 3000")
    ap.add_argument("--death-ladder", action="store_true",
                    help="v18:死亡成本随层数定价(死在 N 层罚 8×N,替代恒 -2)")
    ap.add_argument("--options", action="store_true",
                    help="v22:策略脑/操作脑(OptionsEnv,Discrete(3);须配 --algo mppo --gamma 1.0)")
    ap.add_argument("--flat-clock", action="store_true",
                    help="v22 恶魔臂:296 维平面(停滞钟入观测),配 --bc-init 用")
    ap.add_argument("--worker", action="store_true",
                    help="v23:FARM 操作脑在位训练(WorkerWindowEnv,Discrete(15) 掩 11/12;"
                         "须配 --algo mppo --gamma 1.0,见 docs/PREREG-v23.md)")
    ap.add_argument("--manager-npz",
                    default=str(pathlib.Path(__file__).resolve().parent
                                / "models" / "v22-h-manager" / "policy.npz"),
                    help="冻结经理权重 npz(export_manager_npz.py 产出)")
    ap.add_argument("--worker-npz", default=None,
                    help="v25:经理训练时挂 npz 工人(OptionsEnv workers 组装口)")
    ap.add_argument(
        "--manager-policy-observation-view",
        choices=["raw-v4", "legacy-v3"],
        default="raw-v4",
        help="Options 经理策略输入契约：新经理默认 raw-v4；仅续训/复现"
             "旧 M29 时显式选 legacy-v3。WorkerWindow 内的冻结 M29 "
             "不读取此旗，而由环境固定为 legacy-v3",
    )
    ap.add_argument("--skip-dry", action="store_true",
                    help="v26 绿洲:干层复访窗脚本代跑,工人只在鲜层窗上课")
    ap.add_argument("--dry-curriculum-schedule", default=None,
                    help="E1 ⑤A 干窗课程退火表:逗号分隔段,"
                         "'linear:<p0>:<p1>:<n>'(n≥2,端点含线性)或 'hold:<p>:<n>';"
                         "p∈[0,1] 为干层复访窗的脚本代跑概率,按腿相对 rollout 序号"
                         " (num_timesteps−3497984)/2048 取表,逐 rollout 于采集前推送;"
                         "与 --skip-dry 互斥,仅 --worker。主表 = "
                         "linear:1.0:0.5:147,hold:0.5:97")
    action12_group = ap.add_mutually_exclusive_group()
    action12_group.add_argument(
        "--drink-sovereignty",
        dest="drink_sovereignty",
        action="store_const",
        const=True,
        help="显式要求环境管理 Worker action12；挂严格 Worker NPZ 时"
             "必须与其 metadata 一致",
    )
    action12_group.add_argument(
        "--no-drink-sovereignty",
        dest="drink_sovereignty",
        action="store_const",
        const=False,
        help="关闭工人喝药主权(m[12] 恢复恒掩)；挂严格 Worker NPZ 时"
             "必须与其 metadata 一致",
    )
    ap.set_defaults(drink_sovereignty=None)
    ap.add_argument(
        "--legacy-worker-policy-observation-view",
        action="store_true",
        help="rev15 普通 Worker actor/value 必选：在环境边界把 feature286/297 "
             "连同怪物/地图/物品通道重建为完整 protocol-v3；"
             "A12 custom policy 禁用此旗并接收 legacy-v3-a12-overlay，"
             "由其内部解码旧网络输入",
    )
    ap.add_argument(
        "--worker-policy-observation-view",
        choices=[
            _WORKER_VIEW_LEGACY_V3,
            _WORKER_VIEW_DUAL_V4_ASYMMETRIC,
        ],
        default=None,
        help="显式 Worker 输入契约。dual-v4-asymmetric-v3 保留前298列"
             "为 V28/KING 根输入，并追加完整 v4/controller/mask 上下文；"
             "可从旧 Worker 以 fresh critic 一次性点火；后续 checkpoint "
             "只允许显式环境重启式参数续接",
    )
    ap.add_argument(
        "--worker-fast-forward-reward-credit",
        choices=["none", "terminal-death-only"],
        default="none",
        help="rev13 Worker 奖励契约：默认 none 保持旧语义；"
             "terminal-death-only 只把同一 transition 内冻结经理/脚本"
             "阶段发生的真实底层死亡分量传给 PPO，不领取其正收益",
    )
    ap.add_argument(
        "--worker-additional-terminal-death-cost",
        type=float,
        default=0.0,
        help="rev13 Worker 生存风险成本；仅真实 death 恰好计一次，"
             "FARM 内/快进终局同构，默认 0 保持旧语义",
    )
    ap.add_argument(
        "--worker-action14-logit-bonus",
        type=float,
        default=0.0,
        help="仅在 exact a14 mask 为真时施加的可训练装备 logit 先验；"
             "0=关闭，dual-v4 Worker 专用",
    )
    ap.add_argument("--ent-coef", type=float, default=0.02,
                    help="熵系数(v22 恶魔臂微调用 0.005 防 BC 漂移)")
    ap.add_argument("--bc-init", default=None,
                    help="行为克隆热启动:载入策略头 state_dict 路径")
    ap.add_argument("--init-source", choices=["bc", "checkpoint"], default="bc",
                    help="--bc-init 的来源类型；checkpoint 必须带 export_manager_sd 清单")
    ap.add_argument("--freeze-policy-steps", type=int, default=0,
                    help="BC 热启动后冻结策略头只训价值头的步数")
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="折扣因子。0.99 半衰期 69 步(1500 步旧章口径);"
                         "v20 深水区用 0.997;--options(v22)应为 1.0")
    ap.add_argument("--distill-beta", type=float, default=0.0,
                    help="v24 皮筋系数 β(CE 对冻结 BC 教师;0=纯 v23 配方,G-KL-B 证逐位等价)")
    ap.add_argument(
        "--distill-anneal-actor-rollouts",
        type=int,
        default=0,
        help="仅计 actor 解冻后的完整 rollout，将 β 从初值线性退火"
             "至 0；0=恒定，非零必须 >=2",
    )
    ap.add_argument("--teacher-sd",
                    default=str(pathlib.Path(__file__).resolve().parent
                                / "runs" / "bc-worker" / "policy_sd.pt"),
                    help="v24 教师 state_dict(SB3 键名)")
    ap.add_argument("--bc-aux-lambda", type=float, default=0.0,
                    help="E3 ④乙:辅助校准系数 λ_bc(a12 正例 + 合法非 a12"
                         " hard negatives + 持久根策略 KL；只消费 training"
                         " episodes；主案冻结常量 0.015625,D7)。"
                         "须与 --bc-aux-demos 同在方在位;两旗互不强制,"
                         "任一不在 → 零侵入(不加载不采样不进损失图)")
    ap.add_argument("--bc-aux-demos", default=None,
                    help="E3 ④乙:bc-worker-v2 示范集 demos.npz 路径(v2 schema="
                         "X/Y/episode_id+逐样本 masks,专用验证器;v1 canonical"
                         " 路径 runs/bc-worker 分毫不动)")
    ap.add_argument(
        "--bc-aux-graft", action="store_true",
        help="rev10:把 V28 actor 无损扩为 68 宽，并在策略分布内安装"
             "逐 eligible 状态初始精确 5%% 的 contextual a12 mixture；"
             "rollout/log-prob 同源，五个稳定 raw-feature gate 参数由 PPO"
             "按战斗回报自主裁决；正式路径要求"
             " --bc-aux-lambda=0")
    ap.add_argument(
        "--bc-aux-liveness-preflight", action="store_true",
        help="在位 aux 正式腿的必选门：环境点火前，以 resume worker/Adam 的隔离 clone "
             "运行本腿同数 aux 调用；仅 training episodes 裁安全可学门，"
             "FAIL 即拒绝训练")
    ap.add_argument("--resume-from", default=None,
                    help="v24 分腿续训:上一腿 model_final.zip 路径(禁与 --bc-init/--freeze 同用)")
    ap.add_argument("--reset-optimizer", action="store_true",
                    help="continuation 安全旋钮:load 后重建 optimizer，清空全部 "
                         "Adam step/exp_avg/exp_avg_sq；默认关闭以兼容旧配方。"
                         "启用时允许显式降低 --lr，policy 权重保持逐位不变")
    ap.add_argument(
        "--reset-worker-critic",
        action="store_true",
        help="把旧窗口终止目标的 Worker value MLP/head 按 SB3 原生"
             "正交初始化重建；actor 位级保留。须配 reset-optimizer、"
             "显式 seed、critic warmup 与分组裁剪",
    )
    ap.add_argument(
        "--critic-warmup-steps",
        type=int,
        default=0,
        help="fresh critic 的 critic-only 样本数；必须整除完整 rollout "
             "量子且小于 total-steps。warmup 后 actor 才开始联合更新",
    )
    ap.add_argument(
        "--gradient-clip-mode",
        choices=[
            "global",
            "separate-actor-critic-v1",
            "separate-root-context-critic-v2",
        ],
        default="global",
        help="主 PPO 梯度裁剪；v2 把 actor root/context 各裁至"
             " max_grad_norm/sqrt(2)，critic 独立裁至 max_grad_norm，"
             "仍使用同一个 Adam",
    )
    ap.add_argument("--target-kl", type=float, default=None,
                    help="PPO 近似 KL 早停阈值；默认 None 保持旧配方。"
                         "continuation 建议配合更低 --lr 与 --reset-optimizer "
                         "显式启用（如 0.02），值须为有限正数")
    ap.add_argument("--calib-probes", default="",
                    help="v24 G-CAL 探针全局步(逗号分隔,只在腿 1 传 300000,600000)")
    ap.add_argument("--calib-record-only", action="store_true",
                    help="v28:G-CAL 只记不裁——tripped 位照写 calib.jsonl,旗不武装"
                         "(续航起点分歧 41.5%%,20%% 阈值对定居点失义;面板修正)")
    ap.add_argument("--teacher-override", default=None,
                    help="v30 锚随王走:resume 时以此 sd 覆写 zip 驮带的 teacher_path"
                         "(经 load kwargs 注入,_setup_model 一次建对;仅 resume 分支有效)")
    ap.add_argument("--allow-manager-change", action="store_true",
                    help="显式允许 worker resume 更换 manager_npz；默认契约禁止")
    ap.add_argument("--allow-legacy-resume", action="store_true",
                    help="一次性迁移无 training_contract 的旧 checkpoint；默认拒绝")
    ap.add_argument(
        "--allow-environment-restart-resume",
        action="store_true",
        help="明确承认带契约 Worker checkpoint 只续接 policy/Adam/全局计数，"
             "不会恢复原生世界、wrapper/controller 或完整 RNG/轨迹状态；"
             "缺此旗时 fail-closed",
    )
    # B1-E0 仪表旋钮(封闭枚举三枚,PREREG-B1;皆纯读+IO,不触 RNG/梯度/env 流/
    # 掩码/契约字段;默认值逐字承继原写死常量,缺省行为零漂移,W-G0 实弹钉死)
    ap.add_argument("--ckpt-every-steps", type=int, default=250_000,
                    help="B1-E0:暴露 AtomicRolloutCheckpointCallback.every_steps"
                         "(全局步;量子对齐与拒发半更新 ckpt 由回调原逻辑保证)")
    ap.add_argument("--sentinel-every", type=int, default=500_000,
                    help="B1-E0:WorkerSentinelCallback 汇总间隔(全局步,纯读+IO)")
    ap.add_argument("--dry-anchor-every", type=int, default=500_000,
                    help="B1-E0:DryAnchorSentinel 间隔(全局步;干态按 col296 "
                         "exhausted 或 col297 farm_scene_fraction 饱和判定;"
                         "自有 rng(26),不碰训练 RNG)")
    # E5 仪表旋钮(PREREG-内容案 E5,封闭枚举两枚;皆纯读+IO,只记不裁,
    # 默认 0 = 不在位 = 代码路径与 HEAD 等价,G0-2a 先决;探针示范集钉
    # BC-v1 demos 字节,E6)
    ap.add_argument("--distill-ce-probe-every", type=int, default=0,
                    help="E5①:干/鲜 distill_ce 分列离线探针间隔(全局步;"
                         "0=不在位;固定示范态集按 col296 exhausted 或 "
                         "col297 farm_scene_fraction 饱和分组,专用 rng,"
                         "零触训练路径;输出 distill_ce_probe.jsonl)")
    ap.add_argument("--drywin-metrics-every", type=int, default=0,
                    help="E5②:干窗行为仪表间隔(全局步;0=不在位;干态动作"
                         "分布熵/a 分布 + 干/鲜窗工资与宽度 τ̄/depth,只记 "
                         "drywin_metrics.jsonl)")
    invocation_argv = list(sys.argv[1:])
    args = ap.parse_args()

    try:
        _validate_runtime_versions()
        _validate_args(args)
    except ValueError as exc:
        ap.error(str(exc))

    run_name = args.run_name or (
        time.strftime("ppo-l1-%m%d-%H%M%S")
        + f"-{os.getpid()}-{time.time_ns() % 1_000_000_000:09d}")
    run_dir = pathlib.Path(__file__).resolve().parent / "runs" / run_name
    try:
        run_lock = _RunLock(run_dir)
    except RuntimeError as exc:
        ap.error(str(exc))
    resources.run_lock = run_lock
    protected_inputs = [args.bc_init, args.teacher_override]
    if args.worker:
        protected_inputs.append(args.manager_npz)
        if args.distill_beta > 0 and not args.resume_from:
            protected_inputs.append(args.teacher_sd)
    if args.options and args.worker_npz:
        protected_inputs.append(args.worker_npz)
    if _bc_aux_active(args):
        protected_inputs.append(args.bc_aux_demos)   # E3:v2 示范集同受保护
    _prepare_run_dir(run_dir, args.resume_from, protected_inputs)

    # Capture all externally supplied brains before any VecEnv/subprocess can
    # load them.  Children receive these exact expectations and parse from a
    # single read, so an atomic path replacement is fail-loud rather than a
    # silent mixed-policy run.
    manager_npz_sha256 = (_capture_file_sha256(args.manager_npz, "manager_npz")
                          if args.worker else None)
    worker_npz_sha256 = (_capture_file_sha256(args.worker_npz, "worker_npz")
                         if args.worker_npz else None)
    args.resolved_drink_sovereignty = (
        _resolve_training_drink_sovereignty(
            args, worker_npz_sha256=worker_npz_sha256))
    implementation_sha256 = _implementation_bundle_sha256()

    fresh_teacher_sha256 = None
    if args.worker and args.distill_beta > 0 and not args.resume_from:
        fresh_teacher_sha256 = _validate_bc_report(
            pathlib.Path(args.teacher_sd), "data_gate")["policy_sha256"]

    teacher_override_sha256 = None
    if args.teacher_override:
        teacher_override_sha256 = _validate_export_manifest(
            pathlib.Path(args.teacher_override))["artifact_sha256"]

    # E1 四门之 demos_sha256 捕获门:skip_dry ∨ schedule(谓词在助手内,断言原封)
    demos_sha256 = _capture_dry_window_demos_sha256(args)

    # E1 ⑤A:课程表在此解析一次,供 env 初值与课程回调共用(_validate_args 已验)。
    dry_curriculum_table = (
        _parse_dry_curriculum_schedule(args.dry_curriculum_schedule)
        if args.dry_curriculum_schedule else None)

    # E3 ④乙:在位方加载 v2 示范集；辅助优化 bank 只能消费固定 training
    # episodes，原始 held-out episodes 完整留给最终发布硬门。随后在训练切分
    # 内过 12 类/hard-negative 过滤。不在位 → 零侵入,不加载(图纸字面)。
    bc_aux_bank = None
    bc_aux_fit = None
    bc_aux_validation = None
    bc_aux_demos_sha256 = None
    _aux_x = _aux_y = _aux_episode_id = _aux_masks = None
    if _bc_aux_active(args):
        (_aux_x, _aux_y, _aux_episode_id, _aux_masks,
         bc_aux_demos_sha256) = (
            _load_bc_aux_demos_v2(
                args.bc_aux_demos,
                expected_manager_sha256=manager_npz_sha256))
        if _bc_aux_structural_active(args):
            _fit, _validation = _bc_v2_fit_validation_indices(
                _aux_episode_id)
            bc_aux_fit = (
                _aux_x[_fit], _aux_y[_fit], _aux_masks[_fit])
            # Keep a non-None marker for the shared active-path setup below;
            # structural code never consumes this as a replay minibatch.
            bc_aux_bank = bc_aux_fit
            bc_aux_validation = (
                _aux_x[_validation], _aux_y[_validation],
                _aux_masks[_validation])
        else:
            bc_aux_bank = _build_bc_aux_training_bank(
                _aux_x, _aux_y, _aux_episode_id, _aux_masks)
    elif args.bc_aux_lambda > 0:
        # 图纸字面:未给 --bc-aux-demos 即零侵入(两旗互不强制);
        # 如实打印防误配静默(施工裁量注记)。
        print("   [④乙] --bc-aux-lambda>0 但未给 --bc-aux-demos:"
              "辅助通路按 E3 零侵入条款不在位")

    resume_checkpoint_bytes = None
    resume_data = None
    resume_checkpoint_sha256 = None
    if args.resume_from and args.worker:
        (resume_checkpoint_bytes, resume_data,
         resume_checkpoint_sha256) = _capture_leashed_checkpoint(args.resume_from)
    elif args.resume_from:
        # v31 经理续训口:通用捕获(字节冻结 + 通用闸 + sha),类保真交由加载段
        _resume_path = _checkpoint_path(args.resume_from)
        try:
            resume_checkpoint_bytes = _resume_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"resume 检查点不可读: {_resume_path}: {exc}") from exc
        resume_data = _validate_checkpoint_bytes(
            resume_checkpoint_bytes, str(_resume_path), False)
        resume_checkpoint_sha256 = hashlib.sha256(
            resume_checkpoint_bytes).hexdigest()

    dry_curriculum_start_index = None
    dry_curriculum_start_probability = None
    if dry_curriculum_table:
        dry_curriculum_start_index, dry_curriculum_start_probability = (
            _resolve_dry_curriculum_start(
                dry_curriculum_table,
                start_steps=(
                    int(resume_data["num_timesteps"])
                    if resume_data is not None else 0
                ),
                rollout_quantum=int(args.n_steps * args.num_envs),
                total_steps=int(args.total_steps),
            )
        )

    batch_size = _select_batch_size(args.n_steps, args.num_envs)
    bc_aux_preflight = None
    bc_aux_preflight_sha256 = None
    if args.bc_aux_liveness_preflight:
        # 生产 aux 是 continuation 配方；只有 resume zip 才同时携真实起始
        # worker 与 Adam moments。没有它就无法在环境点火前做同构 liveness
        # 沙盒，宁可 fail-closed，不以随机新模型冒充预检。
        _require(resume_checkpoint_bytes is not None
                 and resume_checkpoint_sha256 is not None,
                 "bc_aux 在位训练须提供 --resume-from，"
                 "以便环境点火前执行真实 worker/optimizer liveness preflight")
        (bc_aux_preflight,
         bc_aux_preflight_sha256) = _run_bc_aux_liveness_preflight(
            run_dir=run_dir, args=args,
            resume_checkpoint_bytes=resume_checkpoint_bytes,
            resume_checkpoint_sha256=resume_checkpoint_sha256,
            bank=bc_aux_bank,
            x=_aux_x, y=_aux_y, episode_id=_aux_episode_id,
            masks=_aux_masks,
            demos_sha256=bc_aux_demos_sha256,
            manager_npz_sha256=manager_npz_sha256,
            implementation_sha256=implementation_sha256,
            batch_size=batch_size)

    hierarchical = args.worker or args.options or args.flat_clock
    effective_deep = True if hierarchical else args.deep
    effective_death_ladder = True if hierarchical else args.death_ladder
    config = {
        # 审计回执：精确保留本进程实际交给 argparse 的 argv；不参与训练契约。
        "invocation_argv": invocation_argv,
        "total_steps": args.total_steps,
        "num_envs": args.num_envs,
        "device": args.device,
        "lr": args.lr,
        "n_steps": args.n_steps,
        "batch_size": batch_size,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "algo": ({"rppo": "RecurrentPPO/MlpLstmPolicy",
                  "mppo": "MaskablePPO/MlpPolicy(gear-key mask)"}.get(args.algo, "PPO/MlpPolicy")
                 + ("+EntityAttention" if args.arch == "attn" else "")),
        "goal": ("深水区:层数递进奖金,活着往下潜(L3/L4)" if effective_deep
                 else "地牢 1 层:杀怪拿 XP,找楼梯下 2 层"),
        "deep": effective_deep,
        "death_ladder": effective_death_ladder,
        "gamma": args.gamma,
        "options": args.options,      # v22:True 时 Monitor ep_len 口径=策略脑决策数
        "flat_clock": args.flat_clock,
        "worker": args.worker,        # v4:True 时 ep=底层整局；FARM 窗边界 nonterminal
        "skip_dry": args.skip_dry,
        "drink_sovereignty":
            _effective_drink_sovereignty(args),   # resolved action12 contract
        "legacy_policy_observation_view":
            args.legacy_worker_policy_observation_view,
        "worker_policy_observation_view":
            _worker_policy_observation_view(args),
        "worker_action14_logit_bonus": float(getattr(
            args, "worker_action14_logit_bonus", 0.0)),
        "manager_policy_observation_view": (
            "legacy-v3" if args.worker
            else args.manager_policy_observation_view
            if args.options else None),
        "worker_fast_forward_reward_credit":
            args.worker_fast_forward_reward_credit,
        "worker_additional_terminal_death_cost":
            float(args.worker_additional_terminal_death_cost),
        "artifact_scope": args.artifact_scope,
        # E4 rev5 双键(契约与 config 回执同构增键;skip_dry 键仍 CLI 旗
        # 字面值,机制在位状态由此二键承载,rev3 勘正)
        "dry_curriculum": _contract_dry_curriculum(args),
        "dry_curriculum_start_index": dry_curriculum_start_index,
        "dry_curriculum_start_probability":
            dry_curriculum_start_probability,
        "bc_aux": _contract_bc_aux(args, bc_aux_demos_sha256),
        "bc_aux_liveness_preflight": (
            {
                "status": bc_aux_preflight["status"],
                "receipt_sha256": bc_aux_preflight_sha256,
                "schema_version":
                    _BC_AUX_LIVENESS_PREFLIGHT_SCHEMA_VERSION,
            } if bc_aux_preflight is not None else "disabled"),
        "calib_probes_requested": args.calib_probes,

        "bc_init": args.bc_init,
        "init_source": args.init_source,
        "ent_coef": args.ent_coef,
        "target_kl": args.target_kl,
        "reset_optimizer": args.reset_optimizer,
        "reset_worker_critic": args.reset_worker_critic,
        "critic_warmup_steps": args.critic_warmup_steps,
        "gradient_clip_mode": args.gradient_clip_mode,
        "freeze_policy_steps": args.freeze_policy_steps,
        "distill_beta": args.distill_beta,    # v24 皮筋
        "distill_anneal_actor_rollouts": int(getattr(
            args, "distill_anneal_actor_rollouts", 0)),
        "resume_from": args.resume_from,
        "resume_checkpoint_sha256": resume_checkpoint_sha256,
        "worker_npz": args.worker_npz,        # v25 换届:经理训练挂 npz 工人
        # v30 接力:自证据链——本腿在谁治下、拴谁的锚,进程侧留回执(面板 minor)
        "manager_npz": args.manager_npz,
        "manager_npz_sha16": (manager_npz_sha256[:16]
                               if manager_npz_sha256 else None),
        "teacher_override": args.teacher_override,
        "teacher_override_sha16": (teacher_override_sha256[:16]
                                    if teacher_override_sha256 else None),
        "demos_sha16": demos_sha256[:16] if demos_sha256 else None,
        "implementation_sha16": implementation_sha256[:16],
        "allow_manager_change": args.allow_manager_change,
        "allow_legacy_resume": args.allow_legacy_resume,
        "allow_environment_restart_resume":
            args.allow_environment_restart_resume,
        # B1-E0 仪表旋钮回执(只读遥测,不入 training_contract,契约零触碰)
        "ckpt_every_steps": args.ckpt_every_steps,
        "sentinel_every": args.sentinel_every,
        "dry_anchor_every": args.dry_anchor_every,
        # E5 仪表旋钮回执(同上:只读遥测,不入 training_contract)
        "distill_ce_probe_every": args.distill_ce_probe_every,
        "drywin_metrics_every": args.drywin_metrics_every,
    }
    print(f"== DiabloGym PPO 训练 == run={run_name}")
    print(f"   {config}")

    env_fn = functools.partial(
        make_env,
        max_steps=args.max_steps,
        deep=args.deep,
        death_ladder=args.death_ladder,
        options=args.options,
        flat_clock=args.flat_clock,
        worker=args.worker,
        manager_npz=args.manager_npz,
        worker_npz=args.worker_npz,
        # SB3 _setup_learn 的 env.reset() 先于 callback training_start。
        # 首次迁移从表首起跑；native continuation 必须从 checkpoint 对应项
        # 起跑，不能先按 table[0] 选择窗口再只改观测中的 p。
        skip_dry=(dry_curriculum_start_probability
                  if dry_curriculum_table else args.skip_dry),
        drink_sovereignty=_effective_drink_sovereignty(args),
        legacy_policy_observation_view=(
            args.legacy_worker_policy_observation_view),
        worker_policy_observation_view=(
            _worker_policy_observation_view(args)),
        manager_policy_observation_view=(
            args.manager_policy_observation_view),
        worker_fast_forward_reward_credit=(
            args.worker_fast_forward_reward_credit),
        worker_additional_terminal_death_cost=(
            args.worker_additional_terminal_death_cost),
        manager_npz_sha256=manager_npz_sha256,
        worker_npz_sha256=worker_npz_sha256,
        implementation_sha256=implementation_sha256,
    )
    if args.num_envs == 1:
        vec_env = DummyVecEnv([env_fn])
    else:
        vec_env = SubprocVecEnv([env_fn] * args.num_envs, start_method="spawn")
    resources.vec_env = vec_env

    common = dict(
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,  # 默认 0.02(首训 0.01 曾面壁塌缩);v22 恶魔臂 0.005
        gae_lambda=_ALGORITHM_RECIPE["gae_lambda"],
        n_epochs=_ALGORITHM_RECIPE["n_epochs"],
        clip_range=_ALGORITHM_RECIPE["clip_range"],
        clip_range_vf=_ALGORITHM_RECIPE["clip_range_vf"],
        vf_coef=_ALGORITHM_RECIPE["vf_coef"],
        max_grad_norm=_ALGORITHM_RECIPE["max_grad_norm"],
        normalize_advantage=_ALGORITHM_RECIPE["normalize_advantage"],
        target_kl=args.target_kl,
        device=args.device,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        seed=args.seed,
    )
    bc_aux_circuit_calibration = None
    actor_migration_receipt = None
    critic_migration_receipt = None
    dual_resume_kind = None
    resume_lineage = None
    policy_kwargs = {}
    if (
        args.worker
        and _worker_policy_observation_view(args)
        == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
    ):
        policy_kwargs["action14_logit_bonus"] = float(
            args.worker_action14_logit_bonus)
    if args.arch == "attn":
        from models import EntityAttentionExtractor
        policy_kwargs = dict(
            features_extractor_class=EntityAttentionExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128], vf=[128]),
        )
    if args.algo == "rppo":
        model = RecurrentPPO(
            "MlpLstmPolicy", vec_env,
            n_steps=args.n_steps, batch_size=batch_size,
            policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1, **policy_kwargs),
            use_sde=_ALGORITHM_RECIPE["use_sde"],
            sde_sample_freq=_ALGORITHM_RECIPE["sde_sample_freq"],
            **common,
        )
    elif args.algo == "mppo":
        # v16:掩码采样与掩码更新都由 MaskablePPO 处理;掩码本身来自
        # env.action_masks()(经 VecEnv.env_method 收集)。注意这是算法实现的
        # 整体更换,开牌异常时首要嫌疑人(诚实账本已记)。
        # v24:worker 路一律走 LeashedMaskablePPO(β=0 时 G-KL-B 证与原版逐位等价)
        calib = [int(x) for x in args.calib_probes.split(",") if x.strip()]
        if args.resume_from and args.options:
            # v31 经理续训口:类保真(存什么类续什么类,M29 系平 MaskablePPO;
            # 不涉教师/β,无 G-KL-B 义务);封条断言照 v24 原封。
            from sb3_contrib import MaskablePPO
            _require(resume_checkpoint_bytes is not None and resume_data is not None,
                     "resume checkpoint 捕获状态缺失")
            _load_kw = {"seed": args.seed} if args.seed is not None else {}
            model = MaskablePPO.load(io.BytesIO(resume_checkpoint_bytes), env=vec_env,
                                     device=args.device, **_load_kw)
            model.tensorboard_log = str(run_dir / "tb")
            saved_lr = model.learning_rate
            if args.reset_optimizer:
                _reset_policy_optimizer(model, args.lr)
            else:
                _require(not callable(saved_lr)
                         and math.isclose(float(saved_lr), args.lr,
                                          rel_tol=0, abs_tol=1e-12),
                         f"resume 学习率不符: checkpoint={saved_lr}, CLI={args.lr};"
                         "如需安全降 lr 请显式 --reset-optimizer")
            model.target_kl = args.target_kl
            _require(math.isclose(float(model.ent_coef), args.ent_coef,
                                  rel_tol=0, abs_tol=1e-12)
                     and math.isclose(float(model.gamma), args.gamma,
                                      rel_tol=0, abs_tol=1e-12)
                     and model.n_steps == args.n_steps
                     and model.batch_size == batch_size,
                     "PREREG-v24 封-5:resume 腿超参与冻结配方不符")
            _require(model.target_kl == args.target_kl,
                     "resume target_kl 显式覆盖失败")
            if args.seed is not None:
                model.set_random_seed(args.seed)
                model.seed = args.seed  # set_random_seed 不更新持久化属性,防 zip 续写旧 seed
            print(f"   [v31] options resume @ {model.num_timesteps} 步(经理续训口)")
        elif args.resume_from:
            from leashed_ppo import (
                AsymmetricWorkerMaskableActorCriticPolicy,
                LeashedMaskablePPO,
            )
            _require(resume_checkpoint_bytes is not None and resume_data is not None,
                     "resume checkpoint 捕获状态缺失")
            _load_kw = {"seed": args.seed} if args.seed is not None else {}
            if args.distill_beta == 0:
                # β=0 不消费教师；旧 zip 中的绝对路径不应让可用检查点因搬家而失效。
                _load_kw.update(teacher_path=None, teacher_sha256=None)
            elif args.teacher_override:
                # v30 锚随王走:kwargs 在 zip data 之后、_setup_model 之前生效,
                # 教师一次建对(post-load 重建系次优解,面板 major 裁定弃用)
                _load_kw["teacher_path"] = args.teacher_override
                _load_kw["teacher_sha256"] = teacher_override_sha256
            else:
                saved_teacher = resume_data.get("teacher_path")
                _require(isinstance(saved_teacher, str) and saved_teacher,
                         "β>0 resume 检查点没有 teacher_path；请显式 --teacher-override")
                teacher_report = _validate_bc_report(
                    pathlib.Path(saved_teacher), "data_gate")
                current_teacher_sha = teacher_report["policy_sha256"]
                saved_teacher_sha = resume_data.get("teacher_sha256")
                if saved_teacher_sha is not None:
                    _require(saved_teacher_sha == current_teacher_sha,
                             "检查点教师 SHA 与当前 BC 报告不一致；"
                             "换锚必须显式 --teacher-override")
                    _load_kw["teacher_sha256"] = saved_teacher_sha
                else:
                    # 旧检查点没有 SHA 字段：只允许以当前 PASS+绑定报告做一次 TOFU 迁移。
                    _load_kw["teacher_sha256"] = current_teacher_sha
                _load_kw["teacher_path"] = saved_teacher
            dual_actor_migration = (
                _worker_policy_observation_view(args)
                == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
            )
            dual_resume_kind = (
                _classify_dual_worker_resume(args, resume_data)
                if dual_actor_migration else None
            )
            if dual_resume_kind == _DUAL_LEGACY_ACTOR_MIGRATION:
                # A 298-wide V28 checkpoint cannot be loaded against a
                # wider environment.  Construct the registered asymmetric
                # topology from scratch, then transplant only the six actor
                # tensors.  Critic, optimizer and all algorithm-side counters
                # deliberately start fresh while the absolute lineage step is
                # inherited from the immutable source checkpoint.
                source_steps = resume_data.get("num_timesteps")
                _require(
                    _is_plain_int(source_steps) and source_steps >= 0,
                    "dual-v4 actor migration 的 source num_timesteps 非法",
                )
                model = LeashedMaskablePPO(
                    AsymmetricWorkerMaskableActorCriticPolicy,
                    vec_env,
                    n_steps=args.n_steps,
                    batch_size=batch_size,
                    policy_kwargs=policy_kwargs or None,
                    distill_beta=args.distill_beta,
                    distill_anneal_actor_rollouts=int(getattr(
                        args, "distill_anneal_actor_rollouts", 0)),
                    teacher_path=_load_kw.get("teacher_path"),
                    teacher_sha256=_load_kw.get("teacher_sha256"),
                    calib_probes=calib,
                    calib_out=(
                        str(run_dir / "calib.jsonl") if calib else None),
                    **common,
                )
                model.num_timesteps = int(source_steps)
                actor_migration_receipt = (
                    _initialize_asymmetric_worker_actor(
                        model,
                        source_checkpoint_payload=resume_checkpoint_bytes,
                        source_checkpoint_sha256=resume_checkpoint_sha256,
                    )
                )
            else:
                _require(
                    not dual_actor_migration
                    or dual_resume_kind == _DUAL_ENV_RESTART_CONTINUATION,
                    "dual-v4 resume 分类未闭合",
                )
                model = LeashedMaskablePPO.load(
                    io.BytesIO(resume_checkpoint_bytes),
                    env=vec_env,
                    device=args.device,
                    **_load_kw,
                )
                if dual_resume_kind == _DUAL_ENV_RESTART_CONTINUATION:
                    actor_migration_receipt = getattr(
                        model, "_actor_migration_receipt", None)
                    critic_migration_receipt = getattr(
                        model, "_critic_migration_receipt", None)
                    _require(
                        isinstance(actor_migration_receipt, dict)
                        and isinstance(critic_migration_receipt, dict),
                        "dual-v4 continuation checkpoint "
                        "缺完整 migration receipts",
                    )
            if getattr(model, "teacher_path", None) and not args.teacher_override:
                _validate_bc_report(pathlib.Path(model.teacher_path), "data_gate")
            # PREREG-v24 D4:β 显式覆盖(load 直写 __dict__ 无校验,不许静默续命);
            # tb 路径同理(否则腿 2-8 曲线全写进腿 1 目录);旋钮封条断言。
            _require(hasattr(model, "distill_beta"),
                     "LeashedMaskablePPO.load 后缺少 distill_beta 内部属性")
            model.distill_beta = args.distill_beta
            model.distill_anneal_actor_rollouts = int(getattr(
                args, "distill_anneal_actor_rollouts", 0))
            _require(
                isinstance(
                    getattr(
                        model,
                        "_distill_actor_rollouts_completed",
                        None),
                    int,
                )
                and not isinstance(
                    model._distill_actor_rollouts_completed, bool)
                and model._distill_actor_rollouts_completed >= 0,
                "resume checkpoint distill actor rollout 计数非法",
            )
            model.calib_probes, model.calib_out = calib, (
                str(run_dir / "calib.jsonl") if calib else None)
            model.calib_record_only = args.calib_record_only
            model.tensorboard_log = str(run_dir / "tb")
            bc_aux_adapter_existing = False
            if _bc_aux_structural_active(args):
                bc_aux_adapter_existing = (
                    getattr(model, "_bc_aux_circuit_spec", None) is not None)
                _expand_policy_with_bc_aux_circuit(model)
            saved_lr = model.learning_rate
            if args.reset_worker_critic:
                _require(
                    resume_data.get("diablogym_contract") is None,
                    "已有 full-game training_contract 的 Worker "
                    "禁止重复重建 critic",
                )
                _require(
                    not bc_aux_adapter_existing
                    and not _bc_aux_structural_active(args),
                    "fresh critic 迁移不得与 A12 circuit 同时改写拓扑",
                )
                source_actor_sha256 = (
                    actor_migration_receipt["source_actor_sha256"]
                    if actor_migration_receipt is not None else None
                )
                source_critic_sha256 = (
                    actor_migration_receipt["source_critic_sha256"]
                    if actor_migration_receipt is not None else None
                )
                critic_migration_receipt = _reset_worker_critic(
                    model,
                    training_seed=args.seed,
                    source_checkpoint_sha256=resume_checkpoint_sha256,
                    source_actor_sha256=source_actor_sha256,
                    source_critic_sha256=source_critic_sha256,
                )
            if args.reset_optimizer:
                _require(not bc_aux_adapter_existing,
                         "已有 a12 mixture 不得清空 optimizer")
                _reset_policy_optimizer(model, args.lr)
            else:
                _require(not callable(saved_lr)
                         and math.isclose(float(saved_lr), args.lr,
                                          rel_tol=0, abs_tol=1e-12),
                         f"resume 学习率不符: checkpoint={saved_lr}, CLI={args.lr};"
                         "如需安全降 lr 请显式 --reset-optimizer")
            if args.reset_worker_critic:
                migration = model.configure_critic_migration(
                    gradient_clip_mode=args.gradient_clip_mode,
                    critic_warmup_steps=args.critic_warmup_steps,
                )
                critic_migration_receipt.update({
                    **migration,
                    "optimizer_reset": True,
                })
                if actor_migration_receipt is not None:
                    _require(
                        critic_migration_receipt["source_actor_sha256"]
                        == actor_migration_receipt["source_actor_sha256"]
                        and critic_migration_receipt[
                            "source_critic_sha256"]
                        == actor_migration_receipt["source_critic_sha256"]
                        and critic_migration_receipt[
                            "actor_sha256_before"]
                        == actor_migration_receipt[
                            "migrated_actor_sha256"]
                        and critic_migration_receipt[
                            "actor_sha256_after"]
                        == actor_migration_receipt[
                            "migrated_actor_sha256"]
                        and critic_migration_receipt["actor_sha256"]
                        == actor_migration_receipt[
                            "migrated_actor_sha256"],
                        "asymmetric actor 与 critic migration 回执未闭合",
                    )
                model._critic_migration_receipt = dict(
                    critic_migration_receipt)
            model.target_kl = args.target_kl
            _require(math.isclose(float(model.ent_coef), args.ent_coef,
                                  rel_tol=0, abs_tol=1e-12)
                     and math.isclose(float(model.gamma), args.gamma,
                                      rel_tol=0, abs_tol=1e-12)
                     and model.n_steps == args.n_steps
                     and model.batch_size == batch_size,
                     "PREREG-v24 封-5:resume 腿超参与冻结配方不符")
            _require(model.target_kl == args.target_kl,
                     "resume target_kl 显式覆盖失败")
            if args.teacher_override:
                # v30 身份链断言(面板 blocker:闸过的文件与训练吃进的文件必须同一)
                _require(model.teacher_path == args.teacher_override, "教师覆写未生效")
                _require(model.teacher[0].in_features == 298
                         and model.teacher[-1].out_features == 15,
                         "自锚教师形状异常(须 298→15 工人网)")
            if args.distill_beta > 0:
                _require(model.teacher is not None, "β>0 但教师未随 teacher_path 重建")
            if _bc_aux_structural_active(args):
                if bc_aux_adapter_existing:
                    columns = list(
                        _BC_AUX_CIRCUIT_GATE_PARAMETER_COLUMNS)
                    bc_aux_circuit_calibration = {
                        "initializer": "preserved-continuation",
                        "gate_coefficients": [
                            float(value) for value in
                            model.policy.action_net.weight[
                                _BC_AUX_CIRCUIT_ACTION,
                                columns,
                            ].detach().cpu()
                        ],
                        "gate_bias": float(
                            model.policy.action_net.bias[
                                _BC_AUX_CIRCUIT_ACTION].detach().cpu()),
                        "candidate_policy_head_sha256":
                            _policy_head_sha256(
                                _policy_head_snapshot(model.policy)),
                    }
                else:
                    bc_aux_circuit_calibration = (
                        _calibrate_bc_aux_adapter_weight(
                            model, _aux_x, _aux_y,
                            _aux_episode_id, _aux_masks))
                _require(
                    bc_aux_preflight is not None
                    and bc_aux_circuit_calibration[
                        "candidate_policy_head_sha256"]
                    == bc_aux_preflight["policy"][
                        "grafted_head_sha256"],
                    "正式 a12 circuit 与隔离 preflight 权重不逐位相同")
            if args.seed is not None:
                model.set_random_seed(args.seed)
                model.seed = args.seed  # set_random_seed 不会更新持久化属性，防 zip 继续写腿1 seed
            print(f"   [v24] resume @ {model.num_timesteps} 步,β={model.distill_beta}")
        elif args.worker:
            from leashed_ppo import LeashedMaskablePPO
            model = LeashedMaskablePPO(
                "MlpPolicy", vec_env, n_steps=args.n_steps, batch_size=batch_size,
                policy_kwargs=policy_kwargs or None,
                distill_beta=args.distill_beta,
                distill_anneal_actor_rollouts=int(getattr(
                    args, "distill_anneal_actor_rollouts", 0)),
                teacher_path=args.teacher_sd if args.distill_beta > 0 else None,
                teacher_sha256=fresh_teacher_sha256,
                calib_probes=calib,
                calib_out=str(run_dir / "calib.jsonl") if calib else None,
                **common)
            model.calib_record_only = args.calib_record_only
        else:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO("MlpPolicy", vec_env, n_steps=args.n_steps,
                                batch_size=batch_size,
                                policy_kwargs=policy_kwargs or None, **common)
    else:
        model = PPO("MlpPolicy", vec_env, n_steps=args.n_steps, batch_size=batch_size,
                    policy_kwargs=policy_kwargs or None,
                    use_sde=_ALGORITHM_RECIPE["use_sde"],
                    sde_sample_freq=_ALGORITHM_RECIPE["sde_sample_freq"],
                    **common)

    if dual_resume_kind is not None:
        _require(
            resume_data is not None
            and resume_checkpoint_sha256 is not None,
            "dual Worker resume lineage 缺 parent payload identity",
        )
        resume_lineage = _build_resume_lineage(
            resume_data,
            parent_sha256=resume_checkpoint_sha256,
            operation=dual_resume_kind,
            seed=args.seed,
            optimizer_reset=bool(args.reset_optimizer),
            critic_reset=bool(args.reset_worker_critic),
        )
        model._resume_lineage = dict(resume_lineage)
    config["resume_lineage"] = (
        dict(resume_lineage) if resume_lineage is not None else None)

    if args.worker and args.algo == "mppo":
        actual_clip_mode = getattr(model, "gradient_clip_mode", "global")
        _require(
            actual_clip_mode == args.gradient_clip_mode,
            "CLI gradient clip mode 与实际 Leashed 模型不一致:"
            f"{args.gradient_clip_mode!r} != {actual_clip_mode!r}",
        )
    config["critic_migration_receipt"] = (
        dict(critic_migration_receipt)
        if critic_migration_receipt is not None else None)
    config["actor_migration_receipt"] = (
        dict(actor_migration_receipt)
        if actor_migration_receipt is not None else None)

    # E3 ④乙:这里只显式覆盖 λ；示范 bank 必须等 BC init/continuation 全部
    # 落位后再挂，才能把“真实训练起点策略”冻结为锚。旧顺序在 fresh
    # --bc-init 时错误地锚住随机初始化。
    if bc_aux_bank is not None:
        _require(hasattr(model, "bc_aux_lambda"),
                 "④乙辅助通路要求 LeashedMaskablePPO(--worker --algo mppo)")
        model.bc_aux_lambda = args.bc_aux_lambda
    elif hasattr(model, "bc_aux_lambda"):
        model.bc_aux_lambda = 0.0

    _validate_model_recipe(model, expected_target_kl=args.target_kl)
    _validate_worker_policy_observation_binding(args, model)
    current_contract = _training_contract(
        args, model, batch_size,
        manager_npz_sha256=manager_npz_sha256,
        worker_npz_sha256=worker_npz_sha256,
        demos_sha256=demos_sha256,
        implementation_sha256=implementation_sha256,
        bc_aux_demos_sha256=bc_aux_demos_sha256,   # E4 rev5:④乙在位载荷
    )
    if args.resume_from:
        _validate_resume_contract(
            getattr(model, "diablogym_contract", None), current_contract,
            allow_manager_change=args.allow_manager_change,
            allow_legacy_resume=args.allow_legacy_resume,
            allow_optimizer_reset=args.reset_optimizer,
            allow_target_kl_change=args.target_kl is not None)
    model.diablogym_contract = current_contract
    config["training_contract"] = current_contract
    config["teacher_sha256"] = getattr(model, "teacher_sha256", None)

    # status 的 total_steps 是 SB3 全局步；target_steps 也必须同口径。
    config["start_steps"] = int(model.num_timesteps)
    target_global_steps = int(model.num_timesteps + args.total_steps)
    config["target_global_steps"] = target_global_steps
    publication_provenance = None
    if bc_aux_bank is not None:
        publication_provenance = {
            "protocol_version": PROTOCOL_VERSION,
            "implementation_sha256": implementation_sha256,
            "manager_npz_sha256": manager_npz_sha256,
            "resume_checkpoint_sha256": resume_checkpoint_sha256,
            "teacher_sha256": getattr(model, "teacher_sha256", None),
            "bc_aux_demos_sha256": bc_aux_demos_sha256,
            "bc_aux_liveness_preflight_sha256":
                bc_aux_preflight_sha256,
            "training_contract_sha256":
                _canonical_json_sha256(current_contract),
            "start_steps": int(model.num_timesteps),
            "target_global_steps": target_global_steps,
            "seed": args.seed,
            "optimizer_reset": bool(args.reset_optimizer),
            "target_kl": args.target_kl,
            "distill_beta": float(args.distill_beta),
            "bc_aux_lambda": float(args.bc_aux_lambda),
            "bc_aux_mode":
                "expanded-trainable-a12-contextual-mixture",
            "calib_record_only": bool(args.calib_record_only),
        }
        # 在任何真实 rollout 前先验证完整谱系；最终发布时再以实际终点
        # 复验一次，防训练路径中途篡改字段。
        _validate_publication_provenance(
            publication_provenance,
            demos_sha256=bc_aux_demos_sha256,
            final_step=target_global_steps)
        config["publication_provenance"] = publication_provenance

    if args.bc_init:
        # v22 恶魔臂:BC 热启动策略头;冻结期只训价值头(经典雷:新价值头的
        # 首次 PPO 更新会摧毁 BC 策略,先冻结抗住)
        gate = ("data_gate" if args.worker else "hypothesis" if args.options
                else "memoryless_hypothesis")
        sd = _load_bc_state_dict(args.bc_init, model.policy, gate, args.init_source)
        missing, unexpected = model.policy.load_state_dict(
            sd, strict=args.init_source == "checkpoint")
        _require(not unexpected, f"BC state_dict 含未知键: {unexpected}")
        _require(all(k not in missing for k in _POLICY_HEAD_KEYS),
                 f"BC 策略头未完整加载: {missing}")
        print(f"   BC 热启动:loaded(missing={len(missing)}, unexpected={len(unexpected)})")
        if args.freeze_policy_steps > 0:
            from stable_baselines3.common.callbacks import BaseCallback

            pi_params = (list(model.policy.mlp_extractor.policy_net.parameters())
                         + list(model.policy.action_net.parameters()))
            if getattr(model.policy, "share_features_extractor", False):
                pi_params += list(model.policy.features_extractor.parameters())
            # 共享特征提取器若仍被 value loss 更新，即使头部 requires_grad=False，
            # BC 策略输出也会在“冻结”期漂移。去重后一并冻结。
            pi_params = list({id(p): p for p in pi_params}.values())
            for p in pi_params:
                p.requires_grad = False

            class _Unfreeze(BaseCallback):
                def __init__(self, when):
                    super().__init__()
                    self.when, self.done_ = when, False

                def _on_rollout_start(self):
                    # PPO 在 rollout 收完后才统一更新。若在跨过阈值的
                    # _on_step 中解冻，该整批(包含阈值前样本)都会更新
                    # 策略。只在下一个 rollout 起点解冻，硬保证前
                    # freeze_policy_steps 个样本只训价值头。
                    if not self.done_ and self.num_timesteps >= self.when:
                        for p in pi_params:
                            p.requires_grad = True
                        self.done_ = True
                        print(f"   策略头解冻 @ {self.num_timesteps}")

                def _on_step(self):
                    return True

            unfreeze_cb = _Unfreeze(args.freeze_policy_steps)
        else:
            unfreeze_cb = None
    else:
        unfreeze_cb = None

    bc_aux_anchor_sd = None
    if bc_aux_bank is not None:
        from leashed_ppo import derive_bc_aux_rng

        # 此时 resume/BC-init 均已完成。首次 aux 腿建立 persistent root；
        # continuation 从 checkpoint 复用同一根锚。必须先恢复/建立根锚再 mount，
        # 让负例 KL 与 rollout monitor 都不发生逐腿重锚。原始 held-out
        # X/Y/episode/masks 另存内存供最终发布门。
        bc_aux_anchor_sd = _persistent_bc_aux_root_anchor(model)
        if _bc_aux_structural_active(args):
            _require(bc_aux_fit is not None
                     and bc_aux_validation is not None,
                     "a12 circuit fit/validation 未构造")
            model.mount_bc_aux_circuit_fit(*bc_aux_fit)
            model.mount_bc_aux_circuit_validation(
                *bc_aux_validation)
            initial_monitor = model._bc_aux_rollout_monitor()
            _require(
                initial_monitor is not None
                and not initial_monitor["tripped"],
                "a12 circuit 点火前在线门未 PASS")
        else:
            model.mount_bc_aux_demos(
                *bc_aux_bank, rng=derive_bc_aux_rng(args.seed))
        model.bc_aux_monitor_out = str(run_dir / "bc_aux_monitor.jsonl")
        # 第一个真实 rollout 就读取 PPO/value/entropy/KING/aux 的联合梯度
        # 与固定 bank；不能等到 +49,152 才发现方向冲突。这个探针只窥视
        # 示范流，rev5 的真实独立 aux step 仍在 epochs 后严格消费一批。
        first_real_probe = int(model.num_timesteps) + int(
            model.n_steps * model.get_env().num_envs)
        model.calib_probes = sorted(set(
            list(getattr(model, "calib_probes", ())) + [first_real_probe]))
        model.calib_out = str(run_dir / "calib.jsonl")
        config["calib_probes_effective"] = list(model.calib_probes)
        print(f"   [④乙] 辅助示范通路在位: λ_bc={model.bc_aux_lambda},"
              f" bank n={len(bc_aux_bank[1])}"
              f"(a12={int((bc_aux_bank[1] == 12).sum())},"
              f" hard-neg={int((bc_aux_bank[1] != 12).sum())}),"
              f" objective_rev={_BC_AUX_OBJECTIVE_REVISION},"
              f" mode={'circuit' if _bc_aux_structural_active(args) else 'legacy'},"
              f" demos_sha16={bc_aux_demos_sha256[:16]},"
              " anchor=first-aux-root(persistent across continuations))")

    # 每 ~25 万个已完成更新的样本存一次原子检查点；499,712 步腿至少有中点保护。
    # B1-E0:三处间隔改由 CLI 旋钮供值(默认逐字承旧常量,缺省行为零漂移)。
    ckpt = AtomicRolloutCheckpointCallback(
        run_dir, every_steps=args.ckpt_every_steps,
        implementation_sha256=implementation_sha256)
    sentinel_cb = (WorkerSentinelCallback(run_dir, every=args.sentinel_every)
                   if args.worker else None)
    # E1 四门之 dry_cb 挂载门:worker ∧ (skip_dry ∨ schedule)(谓词在助手内)
    dry_cb = (DryAnchorSentinel(run_dir, str(pathlib.Path(__file__).resolve().parent
                                             / "runs" / "bc-worker" / "demos.npz"),
                                  demos_sha256, every=args.dry_anchor_every)
              if _mount_dry_anchor_sentinel(args) else None)
    # E1 ⑤A 课程回调(schedule 仅 --worker,_validate_args 已断言)
    curriculum_cb = (DryCurriculumCallback(dry_curriculum_table, run_dir=run_dir)
                     if (args.worker and dry_curriculum_table) else None)
    # E5 仪表挂载(只记不裁;旋钮 0 = 不挂载 = 回调列与 HEAD 等价,G0-2a
    # 先决;探针示范集一律钉 canonical BC-v1 demos 字节,E6 构造内断言)
    _probe_demos = str(pathlib.Path(__file__).resolve().parent
                       / "runs" / "bc-worker" / "demos.npz")
    distill_ce_cb = (DistillCeProbe(run_dir, _probe_demos,
                                    every=args.distill_ce_probe_every)
                     if (args.worker and args.distill_ce_probe_every > 0)
                     else None)
    if distill_ce_cb is not None:
        _require(getattr(model, "teacher", None) is not None,
                 "--distill-ce-probe-every>0 需 Leashed 教师在位"
                 "(β>0 或 teacher_path;fail-loud 于点火前)")
    drywin_cb = (DryWindowMetricsCallback(run_dir, _probe_demos,
                                          every=args.drywin_metrics_every)
                 if (args.worker and args.drywin_metrics_every > 0) else None)
    # 让唯一持有文件句柄的 callback 最后构造；其后的 setup 不再有可失败 I/O。
    callback = EpisodeJsonlCallback(run_dir, config)
    learn_completed = False
    try:
        # E1 回调序钉死:课程回调居列首——rollout-start 先登记下一边界的
        # per-env 倒计时，rollout-tail 再先于其他回调核验环境内原子提交。
        cbs = (([curriculum_cb] if curriculum_cb else [])
               + [callback, ckpt] + ([unfreeze_cb] if unfreeze_cb else [])
               + ([sentinel_cb] if sentinel_cb else [])
               + ([dry_cb] if dry_cb else [])
               # E5 仪表居列尾(纯读+IO;不在位时本两项为空,列与 HEAD 等价)
               + ([distill_ce_cb] if distill_ce_cb else [])
               + ([drywin_cb] if drywin_cb else []))
        # v24:resume 腿 reset_num_timesteps=False(False 语义 = 再训 N 步,全局步连续
        # → ckpt 文件名全局唯一、β 日程与预算记账不断;审计 BLOCKER 2)
        model.learn(total_timesteps=args.total_steps, callback=cbs,
                    reset_num_timesteps=not args.resume_from)
        # collect_rollouts 被 callback 中途终止时 learn() 也会正常返回；此外
        # G-CAL 可在 full buffer 的首个 minibatch 拒绝整个更新。两者都不能
        # 把 num_timesteps 已记、梯度未吃的权重发布成正式终点。
        _require_exact_training_completion(model, target_global_steps)
        learn_completed = True
    finally:
        active_exception = sys.exc_info()[1]
        active_error = active_exception is not None
        callback.close()
        save_error = None
        model_saved = False
        output_name, successful_publication_state = (
            _ARTIFACT_SCOPE_RESULTS[args.artifact_scope])
        output_path = run_dir / output_name
        if not active_error and learn_completed:
            try:
                _require(
                    int(model.num_timesteps) == target_global_steps,
                    "最终发布步数不等于冻结目标:"
                    f"{int(model.num_timesteps)} != {target_global_steps}")
                final_implementation = _implementation_bundle_sha256()
                _require(final_implementation == implementation_sha256,
                         "训练期间实现/引擎/游戏内容发生漂移，"
                         f"拒绝生成 {output_name}: "
                         f"{final_implementation} != {implementation_sha256}")
                if args.artifact_scope != "production":
                    _require(
                        bc_aux_bank is None,
                        "非 production 工件不得进入 bc_aux "
                        "final-heldout 发布门",
                    )
                    _atomic_save_model(model, output_path)
                elif bc_aux_bank is not None:
                    _require(
                        all(value is not None for value in (
                            _aux_x, _aux_y, _aux_episode_id, _aux_masks,
                            bc_aux_demos_sha256, bc_aux_anchor_sd)),
                        "bc_aux 最终发布门证据/起点锚缺失")
                    _publish_model_final_with_bc_aux_gate(
                        model, run_dir / "model_final.zip",
                        x=_aux_x, y=_aux_y,
                        episode_id=_aux_episode_id, masks=_aux_masks,
                        demos_sha256=bc_aux_demos_sha256,
                        anchor_sd=bc_aux_anchor_sd,
                        publication_provenance=publication_provenance)
                else:
                    _atomic_save_model(model, output_path)
                model_saved = True
            except Exception as exc:
                # close 必须执行；原实现在 save 失败时会直接跳过子进程清理。
                save_error = exc
                print(f"模型保存失败: {exc}")
        else:
            # 异常或半 rollout 早停时，num_timesteps 已可能包含尚未更新的样本；
            # 这类权重不能冒充正式终点。最近的 rollout-boundary ckpt 仍可恢复。
            print(f"训练未停在完整更新边界，拒绝生成 {output_name}")
        if model_saved:
            publication_state = successful_publication_state
            publication_sha256 = _capture_file_sha256(
                output_path, f"已冻结 {output_name}")
            publication_detail = None
        elif active_error:
            publication_state = "TRAINING_ERROR"
            publication_sha256 = None
            publication_detail = (
                f"{type(active_exception).__name__}: {active_exception}")
        else:
            publication_state = "PUBLICATION_REFUSED"
            publication_sha256 = None
            publication_detail = (
                f"{type(save_error).__name__}: {save_error}"
                if save_error is not None
                else "training did not complete at a publishable update boundary")
        status_error = None
        try:
            _record_run_publication_status(
                run_dir, publication_state,
                model_sha256=publication_sha256,
                detail=publication_detail)
        except Exception as status_exc:
            # 先保留异常并完成资源回收。若训练/保存本身没有更早异常，
            # terminal status 是正式工件事务的一部分，失败必须令子进程
            # 非零退出；否则 R7 会看到已生成模型却永远缺完成回执。
            status_error = status_exc
            print(f"最终状态写入失败: {status_exc}")
        # 工作进程崩溃时 close 可能在断管上阻塞；资源所有者
        # 统一做超时清理，并恢复宿主的 SIGALRM handler/timer。
        resources.close()
        if model_saved:
            print(f"模型已保存: {output_path}")
        if save_error is not None and not active_error:
            raise save_error
        if status_error is not None and not active_error:
            raise RuntimeError(
                "模型/训练终态无法事务化写入 status.json"
            ) from status_error


def main():
    resources = _TrainingResources()
    try:
        return _main(resources)
    finally:
        # Covers every pre-learn failure too: model/load, contract, BC init,
        # and callback construction.  The normal learn-finally path is
        # idempotent and clears these handles before returning here.
        resources.close()


if __name__ == "__main__":
    main()
