"""v24-KL「皮筋」:LeashedMaskablePPO(docs/PREREG-v24.md D1/D4)。

总损失 = PPO 原三项 + β · CE(π_T, π_θ),其中:
  - 教师 π_T = 冻结 BC 网(train/runs/bc-worker/policy_sd.pt,G1 证与脚本零分歧);
  - 教师 logits 先按逐样本 rollout 掩码置 -1e8 再 softmax(掩位精确下溢为 0,
    0×(-1e8)=0,无 NaN——G-KL-A 断言钉死此性质);
  - ∂CE/∂z = π_θ − π_T,逐分量有界 [−1,1]:弹簧,不是焊点(审计实测教师
    top-1 中位 0.99971 但 logits 有界,反向 KL 因与熵奖励互殴被判死,见预注册)。
  - β=0 时整段被 if 跳过,train() 与原版逐位等价(G-KL-B 受控对照钉死)。

train() 系 sb3_contrib 2.x ppo_mask.py 的诚实复写(上游无损失 hook),
插入点仅一处;上游若升版须重新比对(预注册 D4 入册警示)。
E3(内容案 ④乙,PREREG-内容案-课⑤x④乙):第二插入段——辅助示范 CE
λ_bc·(−log π_θ(y_demo|s_demo)),正样本注入之字面实现(无教师全分布项,
不与 KING 锚在其余动作上对拉);锚教师/β=0.015625/锚公式(上段皮筋)零触碰
(圈 3/圈 5 双引)。λ_bc=0 或示范池未挂载时整段不进图(不加载不采样,
G0-2a 张量级恒等先决);demo minibatch 用专用 rng 流(播种同 E1③ 形制)。
标定探针(G-CAL):到达 --calib-probes 指定全局步时,对首个 minibatch 用
autograd.grad 测 g_ce/g_pg 与 teacher_diverge,写 calib.jsonl;
diverge>20% 置 _calib_tripped,由哨兵回调终止本腿(驱动裁决重标定)。
v28:calib_record_only=True 时探针只记不裁(tripped 位照记入 jsonl,旗不武装
——续航起点分歧 41.5%,20% 阈值对定居点失义;面板 blocker 修正)。
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import pathlib
from importlib.metadata import version

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.distributions import (
    MaskableCategoricalDistribution,
)
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor, MlpExtractor
from stable_baselines3.common.utils import explained_variance

from diablogym.controller_wire import (
    CONTROLLER_SNAPSHOT_COMBAT_FIELDS,
    CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS,
    CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS,
    CONTROLLER_SNAPSHOT_EQUIPPED_FIELDS,
    CONTROLLER_SNAPSHOT_EQUIPPED_ROW_FIELDS,
    CONTROLLER_SNAPSHOT_EQUIPPED_SLOTS,
    CONTROLLER_SNAPSHOT_MAP_CHANNELS,
    CONTROLLER_SNAPSHOT_MISSILE_FIELDS,
    CONTROLLER_SNAPSHOT_MISSILE_LIMIT,
    CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_DIM,
    CONTROLLER_SNAPSHOT_MISSILE_ROW_FIELDS,
    CONTROLLER_SNAPSHOT_MONSTER_FIELDS,
    CONTROLLER_SNAPSHOT_MONSTER_LIMIT,
    CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_DIM,
    CONTROLLER_SNAPSHOT_MONSTER_ROW_FIELDS,
    DUAL_WORKER_LAYOUT,
    DUAL_WORKER_LAYOUT_SHA256,
    DUAL_WORKER_OBSERVATION_DIM,
)

HUGE_NEG = -1e8
GRADIENT_CLIP_GLOBAL = "global"
GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1 = "separate-actor-critic-v1"
GRADIENT_CLIP_ROOT_CONTEXT_CRITIC_V2 = (
    "separate-root-context-critic-v2"
)
WORKER_ONPOLICY_PG_AUDIT_SCHEMA = "diablogym-worker-onpolicy-pg/10"
ASYMMETRIC_WORKER_RUNTIME_EVIDENCE_SCHEMA = (
    "diablogym-asymmetric-worker-runtime-evidence/1"
)
# Deprecated source-level alias for the short-lived audit prototype.  The
# receipt itself is deliberately Worker-policy-reward scoped: transition_reward
# includes all registered Worker wage/death-credit terms, not combat alone.
COMBAT_ONPOLICY_PG_AUDIT_SCHEMA = WORKER_ONPOLICY_PG_AUDIT_SCHEMA
_WORKER_ONPOLICY_PG_MIN_NORM = 1e-12
_WORKER_ONPOLICY_PG_MIN_ALIGNMENT_COSINE = 1e-12
WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT = 8
_WORKER_COMBAT_EFFECT_REASONS = frozenset({
    "target_damage",
    "target_removed",
    "kill",
})
_WORKER_ACTION_EFFECT_AUDIT_KEYS = frozenset({
    "requested_action",
    "native_attempts",
    "native_accepts",
    "request_executed",
    "material_effect",
    "effect_reasons",
    "same_scene",
    "stall_cost_applied",
})


class _AuditedMaskableRolloutBuffer(MaskableRolloutBuffer):
    """Maskable buffer that seals the exact inputs and outputs of SB3 GAE.

    ``callback.on_rollout_end`` runs after GAE and before ``train()``.  Keeping
    an immutable copy here lets the formal Worker receipt independently
    recompute the recursion and reject callbacks or later code that mutate the
    rollout arrays before PPO consumes them.
    """

    _AUDIT_ARRAY_NAMES = (
        "observations",
        "actions",
        "rewards",
        "episode_starts",
        "values",
        "log_probs",
        "advantages",
        "returns",
        "action_masks",
    )

    def reset(self) -> None:
        super().reset()
        self._formal_gae_snapshot = None

    def compute_returns_and_advantage(
            self, last_values: th.Tensor, dones: np.ndarray) -> None:
        last_values_np = (
            last_values.detach().clone().cpu().numpy().flatten())
        dones_np = np.asarray(dones).copy()
        super().compute_returns_and_advantage(last_values, dones)
        self._formal_gae_snapshot = {
            "last_values": last_values_np,
            "dones": dones_np,
            **{
                name: np.asarray(getattr(self, name)).copy()
                for name in self._AUDIT_ARRAY_NAMES
            },
        }
_GRADIENT_CLIP_MODES = frozenset({
    GRADIENT_CLIP_GLOBAL,
    GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
    GRADIENT_CLIP_ROOT_CONTEXT_CRITIC_V2,
})
# Frozen V28/KING were trained on the protocol-v3 worker contract:
#   286 = belt_heals / 8
#   296 = no-kill clock / KILL_PATIENCE
#   297 = exhausted (0/1)
# Rev15's A12 overlay retains the 298-wide shape but changes 286 to a
# reversible legacy-heals/free-slot packing and 297 to a signed drink-latch
# encoding.  Every other base feature is already rebuilt from native raw as
# exact protocol-v3 before it reaches this tensor helper.  Shape equality alone
# is not semantic compatibility.
_LEGACY_BELT_FEATURE = 286
_LEGACY_LAYER_CLOCK_FEATURE = 296
_LEGACY_EXHAUSTED_FEATURE = 297
_LEGACY_DISTILL_EXCLUDED_ACTION = 12
# KING/V28 predates both Worker-only protocol extensions.  It never observed
# a12 (drink sovereignty) as legal and its historical deployment produced no
# a14 gear decisions.  Neither logit is a meaningful teacher target.
LEGACY_DISTILLATION_EXCLUDED_ACTIONS = (12, 14)
ASYMMETRIC_WORKER_OBSERVATION_DIM = DUAL_WORKER_OBSERVATION_DIM
ASYMMETRIC_WORKER_LEGACY_DIM = 298
# p_skip is a training sampler control, not a property of the deployed game
# state.  The critic may condition on it to model the curriculum's rollout
# distribution.  R7 now contains an explicit p=0 tail, but exposing the
# scheduler knob to the actor would still create a non-game-state shortcut
# that cannot exist in deployment.  Keep it for value learning and remove it
# only from the actor residual.
ASYMMETRIC_WORKER_SKIP_DRY_FEATURE = (
    DUAL_WORKER_LAYOUT.p_skip_semantic_index
)
ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES = (
    ASYMMETRIC_WORKER_SKIP_DRY_FEATURE,
)
ASYMMETRIC_WORKER_ACTOR_EXCLUDED_SEGMENTS = tuple(
    segment.name
    for segment in DUAL_WORKER_LAYOUT.segments
    if (
        "critic" in segment.semantic_tags
        and "actor" not in segment.semantic_tags
    )
)
ASYMMETRIC_WORKER_CONTEXT_HIDDEN_DIM = 48
ASYMMETRIC_WORKER_CONTEXT_INTERACTION_DIM = 32
ASYMMETRIC_WORKER_FROZEN_CONTEXT_PARAMETER_COUNT = 61_058
ASYMMETRIC_WORKER_FROZEN_CONTEXT_TENSOR_COUNT = 43
ASYMMETRIC_WORKER_FROZEN_ACTOR_PARAMETER_COUNT = 85_329
ASYMMETRIC_WORKER_FROZEN_ACTOR_TENSOR_COUNT = 49
ASYMMETRIC_WORKER_FROZEN_CRITIC_PARAMETER_COUNT = 67_139
ASYMMETRIC_WORKER_FROZEN_CRITIC_TENSOR_COUNT = 44
ASYMMETRIC_WORKER_CONTEXT_INIT_SEED = 0xA17C0DE
ASYMMETRIC_WORKER_CONTEXT_INITIALIZER = (
    "structured-centered-local-uniform-fanin-v2-seed-169328862"
)


def _unique_module_parameters(module: th.nn.Module, label: str) -> tuple:
    if not isinstance(module, th.nn.Module):
        raise RuntimeError(f"actor/critic partition 缺少 {label} module")
    parameters = tuple(module.parameters())
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError(f"actor/critic partition 的 {label} 含重复参数")
    return parameters


def strict_actor_critic_parameter_partition(policy, optimizer=None) -> dict:
    """Return a complete, disjoint Worker actor/critic parameter partition.

    PPO's policy and value losses are separable only when every optimized
    parameter belongs to exactly one branch.  A trainable shared feature
    extractor would make per-branch clipping ambiguous, so the v1 contract
    rejects it instead of silently assigning its gradient to either side.
    """
    required = ("mlp_extractor", "action_net", "value_net")
    missing = [name for name in required if not hasattr(policy, name)]
    if missing:
        raise RuntimeError(
            f"actor/critic partition 缺少 policy modules:{missing}")
    mlp = policy.mlp_extractor
    if not hasattr(mlp, "policy_net") or not hasattr(mlp, "value_net"):
        raise RuntimeError("actor/critic partition 要求独立 policy/value MLP")

    actor_modules = [mlp.policy_net, policy.action_net]
    context_adapter = getattr(mlp, "context_adapter", None)
    if context_adapter is not None:
        actor_modules.append(context_adapter)
    critic_modules = [mlp.value_net, policy.value_net]
    if bool(getattr(policy, "share_features_extractor", False)):
        extractor = getattr(policy, "features_extractor", None)
        shared = _unique_module_parameters(
            extractor, "shared features_extractor")
        if shared:
            raise RuntimeError(
                "separate actor/critic clip 禁止 trainable shared "
                "features_extractor")
    else:
        actor_modules.append(getattr(policy, "pi_features_extractor", None))
        critic_modules.append(getattr(policy, "vf_features_extractor", None))

    def collect(modules, role):
        ordered = []
        seen = set()
        for index, module in enumerate(modules):
            for parameter in _unique_module_parameters(
                    module, f"{role}[{index}]"):
                identity = id(parameter)
                if identity in seen:
                    continue
                seen.add(identity)
                ordered.append(parameter)
        if not ordered:
            raise RuntimeError(f"actor/critic partition 的 {role} 为空")
        return tuple(ordered), seen

    actor, actor_ids = collect(actor_modules, "actor")
    critic, critic_ids = collect(critic_modules, "critic")
    overlap = actor_ids.intersection(critic_ids)
    if overlap:
        raise RuntimeError(
            "actor/critic partition 存在跨分支共享参数")

    policy_parameters = tuple(policy.parameters())
    policy_ids = [id(parameter) for parameter in policy_parameters]
    if len(set(policy_ids)) != len(policy_ids):
        raise RuntimeError("policy.parameters() 含重复参数")
    classified = actor_ids.union(critic_ids)
    if classified != set(policy_ids):
        named = {
            id(parameter): name
            for name, parameter in policy.named_parameters()
        }
        missing_names = sorted(
            named.get(identity, f"<id:{identity}>")
            for identity in set(policy_ids).difference(classified)
        )
        foreign = sorted(
            identity for identity in classified.difference(policy_ids)
        )
        raise RuntimeError(
            "actor/critic partition 未精确覆盖 policy 参数:"
            f"missing={missing_names},foreign={foreign}")

    if optimizer is not None:
        optimized = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        optimized_ids = [id(parameter) for parameter in optimized]
        if len(set(optimized_ids)) != len(optimized_ids):
            raise RuntimeError("policy optimizer 含重复参数")
        if set(optimized_ids) != set(policy_ids):
            raise RuntimeError(
                "policy optimizer 参数集合与 policy.parameters() 不一致")
    return {"actor": actor, "critic": critic}


def _parameter_group_sha256(policy, parameters) -> str:
    selected = {id(parameter) for parameter in parameters}
    named = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if id(parameter) in selected
    ]
    if {id(parameter) for _, parameter in named} != selected:
        raise RuntimeError("参数摘要无法解析完整 named_parameters")
    digest = hashlib.sha256()
    for name, parameter in sorted(named):
        tensor = parameter.detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(
            np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def actor_parameter_sha256(policy, optimizer=None) -> str:
    partition = strict_actor_critic_parameter_partition(
        policy, optimizer=optimizer)
    return _parameter_group_sha256(policy, partition["actor"])


def critic_parameter_sha256(policy, optimizer=None) -> str:
    partition = strict_actor_critic_parameter_partition(
        policy, optimizer=optimizer)
    return _parameter_group_sha256(policy, partition["critic"])


def _finite_gradient_norm(parameters) -> float:
    live = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not live:
        return 0.0
    if not all(bool(gradient.isfinite().all().item())
               for gradient in live):
        raise RuntimeError("actor/critic gradient 含 NaN/Inf")
    norm = float(th.sqrt(th.stack([
        gradient.float().pow(2).sum() for gradient in live
    ]).sum()).detach().cpu())
    if not math.isfinite(norm):
        raise RuntimeError("actor/critic gradient norm 非有限")
    return norm


def _finite_autograd_gradient_norm(
        scalar: th.Tensor, parameters, *, retain_graph: bool) -> float:
    """Measure one loss term without populating ``parameter.grad``.

    The formal Worker receipt uses this on the PPO clipped surrogate before
    entropy, distillation, BC and value losses are added.  Consequently an
    actor update caused only by those other terms cannot masquerade as an
    on-policy policy gradient.
    """
    parameters = tuple(parameters)
    if not parameters or not scalar.requires_grad:
        return 0.0
    gradients = th.autograd.grad(
        scalar,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    live = [gradient for gradient in gradients if gradient is not None]
    if not live:
        return 0.0
    if not all(bool(th.isfinite(gradient).all().item())
               for gradient in live):
        raise RuntimeError("formal policy-gradient audit 含 NaN/Inf")
    norm = float(th.sqrt(th.stack([
        gradient.detach().double().pow(2).sum()
        for gradient in live
    ]).sum()).cpu())
    if not math.isfinite(norm):
        raise RuntimeError("formal policy-gradient audit norm 非有限")
    return norm


def _finite_autograd_gradients(
        scalar: th.Tensor, parameters, *,
        retain_graph: bool) -> tuple:
    """Return finite detached gradients in parameter order (``None`` kept)."""
    parameters = tuple(parameters)
    if not parameters or not scalar.requires_grad:
        return tuple(None for _ in parameters)
    gradients = th.autograd.grad(
        scalar,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    if not all(
        gradient is None
        or bool(th.isfinite(gradient).all().item())
        for gradient in gradients
    ):
        raise RuntimeError("formal context-gradient audit 含 NaN/Inf")
    return tuple(
        None if gradient is None else gradient.detach()
        for gradient in gradients
    )


def _finite_gradient_tuple_norm(gradients) -> float:
    live = tuple(
        gradient for gradient in gradients if gradient is not None)
    if not live:
        return 0.0
    value = float(th.sqrt(th.stack([
        gradient.double().pow(2).sum()
        for gradient in live
    ]).sum()).cpu())
    if not math.isfinite(value):
        raise RuntimeError("formal context-gradient norm 非有限")
    return value


def _finite_gradient_tuple_dot(left, right) -> float:
    left = tuple(left)
    right = tuple(right)
    if len(left) != len(right):
        raise RuntimeError("formal context-gradient dot 形状不闭合")
    value = 0.0
    for lhs, rhs in zip(left, right):
        if lhs is None or rhs is None:
            continue
        if lhs.shape != rhs.shape:
            raise RuntimeError(
                "formal context-gradient dot 张量形状漂移")
        value += float((lhs.double() * rhs.double()).sum().cpu())
    if not math.isfinite(value):
        raise RuntimeError("formal context-gradient dot 非有限")
    return value


def _select_parameter_gradients(
        parameters, gradients, selected_parameters) -> tuple:
    """Select an ordered semantic subgroup without tuple-position coupling."""
    parameters = tuple(parameters)
    gradients = tuple(gradients)
    selected_parameters = tuple(selected_parameters)
    if len(parameters) != len(gradients):
        raise RuntimeError("formal context parameter/gradient 数量不闭合")
    by_id = {
        id(parameter): gradient
        for parameter, gradient in zip(parameters, gradients, strict=True)
    }
    if (
        len(by_id) != len(parameters)
        or any(id(parameter) not in by_id
               for parameter in selected_parameters)
    ):
        raise RuntimeError("formal context semantic group 未闭合")
    return tuple(by_id[id(parameter)] for parameter in selected_parameters)


_WORKER_ONPOLICY_PG_RECEIPT_KEYS = frozenset({
    "schema",
    "rollout_end_timesteps",
    "collection_actor_sha256",
    "onpolicy_log_prob_source",
    "onpolicy_log_prob_max_abs_delta",
    "transition_reward_source",
    "transition_reward_samples",
    "transition_reward_nonzero_samples",
    "transition_reward_positive_samples",
    "transition_reward_negative_samples",
    "transition_reward_sum",
    "transition_reward_abs_sum",
    "transition_reward_mean",
    "transition_reward_variance",
    "no_progress_timeout_samples",
    "no_progress_timeout_base_failure_reward_sum",
    "no_progress_timeout_additional_failure_reward_sum",
    "no_progress_timeout_failure_reward_sum",
    "requested_action_counts",
    "executed_action_counts",
    "combat_effect_samples",
    "combat_transition_reward_nonzero_samples",
    "combat_transition_reward_positive_samples",
    "combat_transition_reward_negative_samples",
    "combat_transition_reward_sum",
    "combat_positive_advantage_samples",
    "combat_transition_reward_abs_sum",
    "combat_reward_centered_l2",
    "combat_reward_centered_actor_grad_norm",
    "combat_reward_centered_root_grad_norm",
    "combat_reward_centered_context_grad_norm",
    "combat_reward_centered_context_output_grad_norm",
    "combat_reward_centered_context_encoder_grad_norm",
    "combat_reward_centered_context_interaction_grad_norm",
    "reward_centered_l2",
    "reward_centered_actor_grad_norm",
    "reward_centered_context_grad_norm",
    "reward_centered_context_output_grad_norm",
    "reward_centered_context_encoder_grad_norm",
    "reward_centered_context_interaction_grad_norm",
    "advantage_source",
    "gae_recomputed_max_abs_delta",
    "return_recomputed_max_abs_delta",
    "gae_advantage_samples",
    "gae_advantage_nonzero_samples",
    "gae_advantage_mean",
    "gae_advantage_variance",
    "pure_ppo_gradient_source",
    "pure_ppo_actor_grad_measurements",
    "pure_ppo_actor_grad_norm_max",
    "pure_ppo_actor_grad_norm_mean",
    "pure_ppo_root_grad_measurements",
    "pure_ppo_root_grad_norm_max",
    "pure_ppo_root_grad_norm_mean",
    "pure_ppo_context_grad_measurements",
    "pure_ppo_context_grad_norm_max",
    "pure_ppo_context_grad_norm_mean",
    "pure_ppo_context_output_grad_measurements",
    "pure_ppo_context_output_grad_norm_max",
    "pure_ppo_context_output_grad_norm_mean",
    "pure_ppo_context_encoder_grad_measurements",
    "pure_ppo_context_encoder_grad_norm_max",
    "pure_ppo_context_encoder_grad_norm_mean",
    "pure_ppo_context_interaction_grad_measurements",
    "pure_ppo_context_interaction_grad_norm_max",
    "pure_ppo_context_interaction_grad_norm_mean",
    "distill_context_grad_measurements",
    "distill_context_grad_norm_max",
    "combined_root_dot_pure_ppo_sum",
    "pure_ppo_root_grad_sq_sum",
    "combined_root_on_pure_ppo_projection",
    "combined_context_dot_pure_ppo_sum",
    "pure_ppo_context_grad_sq_sum",
    "combined_context_on_pure_ppo_projection",
    "pure_ppo_actor_dot_combat_reward_sum",
    "pure_ppo_actor_on_combat_reward_projection",
    "pure_ppo_root_dot_combat_reward_sum",
    "pure_ppo_root_on_combat_reward_projection",
    "pure_ppo_context_dot_combat_reward_sum",
    "pure_ppo_context_on_combat_reward_projection",
    "optimizer_delta_source",
    "optimizer_delta_actor_l2",
    "optimizer_delta_actor_dot_combat_reward_descent",
    "optimizer_delta_actor_on_combat_reward_descent_projection",
    "optimizer_delta_actor_on_combat_reward_descent_cosine",
    "optimizer_delta_root_l2",
    "optimizer_delta_root_dot_combat_reward_descent",
    "optimizer_delta_root_on_combat_reward_descent_projection",
    "optimizer_delta_root_on_combat_reward_descent_cosine",
    "optimizer_delta_context_l2",
    "optimizer_delta_context_dot_combat_reward_descent",
    "optimizer_delta_context_on_combat_reward_descent_projection",
    "optimizer_delta_context_on_combat_reward_descent_cosine",
    "optimizer_steps",
    "qualifies",
    "kl_early_stopped",
})


def _worker_onpolicy_pg_receipt_qualifies(receipt: dict) -> bool:
    """Return the single canonical liveness verdict for a closed receipt."""
    optimizer_steps = receipt["optimizer_steps"]
    return bool(
        receipt["transition_reward_samples"] >= 2
        and receipt["transition_reward_nonzero_samples"] >= 2
        and receipt["transition_reward_variance"] > 0.0
        and receipt["combat_effect_samples"] >= 1
        and receipt[
            "combat_transition_reward_nonzero_samples"
        ] >= 1
        and receipt[
            "combat_transition_reward_positive_samples"
        ] >= 1
        and receipt["combat_transition_reward_sum"] > 0.0
        and receipt["combat_positive_advantage_samples"] >= 1
        and receipt["combat_transition_reward_abs_sum"] > 0.0
        and receipt["combat_reward_centered_l2"] > 0.0
        and receipt["combat_reward_centered_actor_grad_norm"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["combat_reward_centered_root_grad_norm"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["combat_reward_centered_context_grad_norm"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt[
            "combat_reward_centered_context_output_grad_norm"
        ] > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt[
            "combat_reward_centered_context_encoder_grad_norm"
        ] > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt[
            "combat_reward_centered_context_interaction_grad_norm"
        ] > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["reward_centered_l2"] > 0.0
        and receipt["reward_centered_actor_grad_norm"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["reward_centered_context_grad_norm"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["reward_centered_context_output_grad_norm"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["reward_centered_context_encoder_grad_norm"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["reward_centered_context_interaction_grad_norm"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["pure_ppo_actor_grad_measurements"] == optimizer_steps
        and receipt["pure_ppo_root_grad_measurements"] == optimizer_steps
        and receipt["pure_ppo_context_grad_measurements"] == optimizer_steps
        and receipt[
            "pure_ppo_context_output_grad_measurements"
        ] == optimizer_steps
        and receipt[
            "pure_ppo_context_encoder_grad_measurements"
        ] == optimizer_steps
        and receipt[
            "pure_ppo_context_interaction_grad_measurements"
        ] == optimizer_steps
        and receipt["distill_context_grad_measurements"] == optimizer_steps
        and optimizer_steps
        >= WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT
        and receipt["pure_ppo_actor_grad_norm_max"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["pure_ppo_root_grad_norm_max"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["pure_ppo_context_grad_norm_max"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["pure_ppo_context_output_grad_norm_max"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["pure_ppo_context_encoder_grad_norm_max"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["pure_ppo_context_interaction_grad_norm_max"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["distill_context_grad_norm_max"] == 0.0
        and receipt["pure_ppo_root_grad_sq_sum"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["combined_root_on_pure_ppo_projection"] > 0.0
        and receipt["pure_ppo_context_grad_sq_sum"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["combined_context_on_pure_ppo_projection"] > 0.0
        and receipt[
            "pure_ppo_actor_on_combat_reward_projection"
        ] > 0.0
        and receipt[
            "pure_ppo_root_on_combat_reward_projection"
        ] > 0.0
        and receipt[
            "pure_ppo_context_on_combat_reward_projection"
        ] > 0.0
        # The gradients above are pre-step graph evidence.  Adam moments,
        # clipping, entropy and distillation can still make the *real*
        # parameter movement point elsewhere.  Close that last gap against
        # the rollout-start, reward-causal combat descent direction.
        and receipt["optimizer_delta_actor_l2"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["optimizer_delta_root_l2"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt["optimizer_delta_context_l2"]
        > _WORKER_ONPOLICY_PG_MIN_NORM
        and receipt[
            "optimizer_delta_actor_on_combat_reward_descent_projection"
        ] > 0.0
        and receipt[
            "optimizer_delta_actor_on_combat_reward_descent_cosine"
        ] > _WORKER_ONPOLICY_PG_MIN_ALIGNMENT_COSINE
        and receipt[
            "optimizer_delta_root_on_combat_reward_descent_projection"
        ] > 0.0
        and receipt[
            "optimizer_delta_root_on_combat_reward_descent_cosine"
        ] > _WORKER_ONPOLICY_PG_MIN_ALIGNMENT_COSINE
        and receipt[
            "optimizer_delta_context_on_combat_reward_descent_projection"
        ] > 0.0
        and receipt[
            "optimizer_delta_context_on_combat_reward_descent_cosine"
        ] > _WORKER_ONPOLICY_PG_MIN_ALIGNMENT_COSINE
    )


def validate_worker_onpolicy_pg_receipt(
        receipt, *, expected_samples: int | None = None) -> bool:
    """Validate one JSON-compatible, reward-bound actor-gradient receipt."""
    if not isinstance(receipt, dict) \
            or set(receipt) != _WORKER_ONPOLICY_PG_RECEIPT_KEYS:
        return False
    integer_keys = (
        "rollout_end_timesteps",
        "transition_reward_samples",
        "transition_reward_nonzero_samples",
        "transition_reward_positive_samples",
        "transition_reward_negative_samples",
        "no_progress_timeout_samples",
        "combat_effect_samples",
        "combat_transition_reward_nonzero_samples",
        "combat_transition_reward_positive_samples",
        "combat_transition_reward_negative_samples",
        "combat_positive_advantage_samples",
        "gae_advantage_samples",
        "gae_advantage_nonzero_samples",
        "pure_ppo_actor_grad_measurements",
        "pure_ppo_root_grad_measurements",
        "pure_ppo_context_grad_measurements",
        "pure_ppo_context_output_grad_measurements",
        "pure_ppo_context_encoder_grad_measurements",
        "pure_ppo_context_interaction_grad_measurements",
        "distill_context_grad_measurements",
        "optimizer_steps",
    )
    if any(type(receipt.get(key)) is not int for key in integer_keys):
        return False
    float_keys = (
        "onpolicy_log_prob_max_abs_delta",
        "transition_reward_sum",
        "transition_reward_abs_sum",
        "transition_reward_mean",
        "transition_reward_variance",
        "no_progress_timeout_base_failure_reward_sum",
        "no_progress_timeout_additional_failure_reward_sum",
        "no_progress_timeout_failure_reward_sum",
        "combat_transition_reward_sum",
        "combat_transition_reward_abs_sum",
        "combat_reward_centered_l2",
        "combat_reward_centered_actor_grad_norm",
        "combat_reward_centered_root_grad_norm",
        "combat_reward_centered_context_grad_norm",
        "combat_reward_centered_context_output_grad_norm",
        "combat_reward_centered_context_encoder_grad_norm",
        "combat_reward_centered_context_interaction_grad_norm",
        "reward_centered_l2",
        "reward_centered_actor_grad_norm",
        "reward_centered_context_grad_norm",
        "reward_centered_context_output_grad_norm",
        "reward_centered_context_encoder_grad_norm",
        "reward_centered_context_interaction_grad_norm",
        "gae_recomputed_max_abs_delta",
        "return_recomputed_max_abs_delta",
        "gae_advantage_mean",
        "gae_advantage_variance",
        "pure_ppo_actor_grad_norm_max",
        "pure_ppo_actor_grad_norm_mean",
        "pure_ppo_root_grad_norm_max",
        "pure_ppo_root_grad_norm_mean",
        "pure_ppo_context_grad_norm_max",
        "pure_ppo_context_grad_norm_mean",
        "pure_ppo_context_output_grad_norm_max",
        "pure_ppo_context_output_grad_norm_mean",
        "pure_ppo_context_encoder_grad_norm_max",
        "pure_ppo_context_encoder_grad_norm_mean",
        "pure_ppo_context_interaction_grad_norm_max",
        "pure_ppo_context_interaction_grad_norm_mean",
        "distill_context_grad_norm_max",
        "combined_root_dot_pure_ppo_sum",
        "pure_ppo_root_grad_sq_sum",
        "combined_root_on_pure_ppo_projection",
        "combined_context_dot_pure_ppo_sum",
        "pure_ppo_context_grad_sq_sum",
        "combined_context_on_pure_ppo_projection",
        "pure_ppo_actor_dot_combat_reward_sum",
        "pure_ppo_actor_on_combat_reward_projection",
        "pure_ppo_root_dot_combat_reward_sum",
        "pure_ppo_root_on_combat_reward_projection",
        "pure_ppo_context_dot_combat_reward_sum",
        "pure_ppo_context_on_combat_reward_projection",
        "optimizer_delta_actor_l2",
        "optimizer_delta_actor_dot_combat_reward_descent",
        "optimizer_delta_actor_on_combat_reward_descent_projection",
        "optimizer_delta_actor_on_combat_reward_descent_cosine",
        "optimizer_delta_root_l2",
        "optimizer_delta_root_dot_combat_reward_descent",
        "optimizer_delta_root_on_combat_reward_descent_projection",
        "optimizer_delta_root_on_combat_reward_descent_cosine",
        "optimizer_delta_context_l2",
        "optimizer_delta_context_dot_combat_reward_descent",
        "optimizer_delta_context_on_combat_reward_descent_projection",
        "optimizer_delta_context_on_combat_reward_descent_cosine",
    )
    if any(type(receipt.get(key)) is not float
           or not math.isfinite(receipt[key])
           for key in float_keys):
        return False
    requested_counts = receipt.get("requested_action_counts")
    executed_counts = receipt.get("executed_action_counts")
    if (
        not isinstance(requested_counts, list)
        or not isinstance(executed_counts, list)
        or len(requested_counts) != 15
        or len(executed_counts) != 15
        or any(type(value) is not int or value < 0
               for value in requested_counts)
        or any(type(value) is not int or value < 0
               for value in executed_counts)
        or any(executed > requested
               for requested, executed in zip(
                   requested_counts, executed_counts, strict=True))
    ):
        return False
    samples = receipt["transition_reward_samples"]
    if samples <= 0:
        return False
    nonzero = receipt["transition_reward_nonzero_samples"]
    positive = receipt["transition_reward_positive_samples"]
    negative = receipt["transition_reward_negative_samples"]
    combat_samples = receipt["combat_effect_samples"]
    combat_nonzero = receipt[
        "combat_transition_reward_nonzero_samples"]
    combat_positive = receipt[
        "combat_transition_reward_positive_samples"]
    combat_negative = receipt[
        "combat_transition_reward_negative_samples"]
    combat_positive_advantages = receipt[
        "combat_positive_advantage_samples"]
    measurements = receipt["pure_ppo_actor_grad_measurements"]
    root_measurements = receipt[
        "pure_ppo_root_grad_measurements"]
    context_measurements = receipt[
        "pure_ppo_context_grad_measurements"]
    output_measurements = receipt[
        "pure_ppo_context_output_grad_measurements"]
    encoder_measurements = receipt[
        "pure_ppo_context_encoder_grad_measurements"]
    interaction_measurements = receipt[
        "pure_ppo_context_interaction_grad_measurements"]
    distill_measurements = receipt[
        "distill_context_grad_measurements"]
    optimizer_steps = receipt["optimizer_steps"]
    timeout_samples = receipt["no_progress_timeout_samples"]
    timeout_base_sum = receipt[
        "no_progress_timeout_base_failure_reward_sum"]
    timeout_additional_sum = receipt[
        "no_progress_timeout_additional_failure_reward_sum"]
    timeout_total_sum = receipt[
        "no_progress_timeout_failure_reward_sum"]
    qualifies = _worker_onpolicy_pg_receipt_qualifies(receipt)
    pure_root_sq = receipt["pure_ppo_root_grad_sq_sum"]
    expected_root_projection = (
        receipt["combined_root_dot_pure_ppo_sum"]
        / pure_root_sq
        if pure_root_sq > 0.0 else 0.0
    )
    root_projection_closed = math.isclose(
        receipt["combined_root_on_pure_ppo_projection"],
        expected_root_projection,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    pure_context_sq = receipt["pure_ppo_context_grad_sq_sum"]
    expected_projection = (
        receipt["combined_context_dot_pure_ppo_sum"]
        / pure_context_sq
        if pure_context_sq > 0.0 else 0.0
    )
    projection_closed = math.isclose(
        receipt["combined_context_on_pure_ppo_projection"],
        expected_projection,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    reward_mean_closed = math.isclose(
        receipt["transition_reward_mean"],
        receipt["transition_reward_sum"] / samples,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    reward_second_moment_closed = math.isclose(
        receipt["reward_centered_l2"] ** 2,
        receipt["transition_reward_variance"] * samples,
        rel_tol=1e-10,
        abs_tol=1e-12,
    )
    combat_actor_sq = (
        receipt["combat_reward_centered_actor_grad_norm"] ** 2)
    combat_root_sq = (
        receipt["combat_reward_centered_root_grad_norm"] ** 2)
    combat_context_sq = (
        receipt["combat_reward_centered_context_grad_norm"] ** 2)
    combat_partition_closed = math.isclose(
        combat_actor_sq,
        combat_root_sq + combat_context_sq,
        rel_tol=1e-10,
        abs_tol=1e-24,
    )

    def _closed_delta_projection(dot_key, projection_key, denominator):
        expected = (
            receipt[dot_key] / denominator
            if denominator > 0.0 else 0.0
        )
        return math.isclose(
            receipt[projection_key],
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

    delta_actor_closed = _closed_delta_projection(
        "optimizer_delta_actor_dot_combat_reward_descent",
        "optimizer_delta_actor_on_combat_reward_descent_projection",
        combat_actor_sq,
    )
    delta_root_closed = _closed_delta_projection(
        "optimizer_delta_root_dot_combat_reward_descent",
        "optimizer_delta_root_on_combat_reward_descent_projection",
        combat_root_sq,
    )
    delta_context_closed = _closed_delta_projection(
        "optimizer_delta_context_dot_combat_reward_descent",
        "optimizer_delta_context_on_combat_reward_descent_projection",
        combat_context_sq,
    )
    repeated_combat_actor_sq = optimizer_steps * combat_actor_sq
    repeated_combat_root_sq = optimizer_steps * combat_root_sq
    repeated_combat_context_sq = optimizer_steps * combat_context_sq
    pure_combat_actor_closed = _closed_delta_projection(
        "pure_ppo_actor_dot_combat_reward_sum",
        "pure_ppo_actor_on_combat_reward_projection",
        repeated_combat_actor_sq,
    )
    pure_combat_root_closed = _closed_delta_projection(
        "pure_ppo_root_dot_combat_reward_sum",
        "pure_ppo_root_on_combat_reward_projection",
        repeated_combat_root_sq,
    )
    pure_combat_context_closed = _closed_delta_projection(
        "pure_ppo_context_dot_combat_reward_sum",
        "pure_ppo_context_on_combat_reward_projection",
        repeated_combat_context_sq,
    )
    pure_combat_partition_closed = (
        receipt["pure_ppo_actor_dot_combat_reward_sum"]
        == receipt["pure_ppo_root_dot_combat_reward_sum"]
        + receipt["pure_ppo_context_dot_combat_reward_sum"]
    )
    delta_partition_closed = (
        receipt[
            "optimizer_delta_actor_dot_combat_reward_descent"
        ]
        == receipt[
            "optimizer_delta_root_dot_combat_reward_descent"
        ] + receipt[
            "optimizer_delta_context_dot_combat_reward_descent"
        ]
    )

    def _closed_leq(left, right, *, abs_tol=1e-18):
        return left <= right or math.isclose(
            left,
            right,
            rel_tol=1e-10,
            abs_tol=abs_tol,
        )

    def _delta_geometry_closed(partition, combat_grad_norm):
        delta_norm = receipt[f"optimizer_delta_{partition}_l2"]
        dot = receipt[
            f"optimizer_delta_{partition}"
            "_dot_combat_reward_descent"
        ]
        cosine = receipt[
            f"optimizer_delta_{partition}"
            "_on_combat_reward_descent_cosine"
        ]
        denominator = delta_norm * combat_grad_norm
        expected_cosine = (
            dot / denominator if denominator > 0.0 else 0.0)
        return (
            delta_norm >= 0.0
            and _closed_leq(abs(dot), denominator, abs_tol=1e-24)
            and _closed_leq(abs(cosine), 1.0, abs_tol=1e-12)
            and math.isclose(
                cosine,
                expected_cosine,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )

    delta_actor_geometry_closed = _delta_geometry_closed(
        "actor",
        receipt["combat_reward_centered_actor_grad_norm"],
    )
    delta_root_geometry_closed = _delta_geometry_closed(
        "root",
        receipt["combat_reward_centered_root_grad_norm"],
    )
    delta_context_geometry_closed = _delta_geometry_closed(
        "context",
        receipt["combat_reward_centered_context_grad_norm"],
    )
    delta_l2_partition_closed = math.isclose(
        receipt["optimizer_delta_actor_l2"] ** 2,
        receipt["optimizer_delta_root_l2"] ** 2
        + receipt["optimizer_delta_context_l2"] ** 2,
        rel_tol=1e-10,
        abs_tol=1e-24,
    )

    norm_pairs = (
        ("pure_ppo_actor_grad_norm_mean",
         "pure_ppo_actor_grad_norm_max"),
        ("pure_ppo_root_grad_norm_mean",
         "pure_ppo_root_grad_norm_max"),
        ("pure_ppo_context_grad_norm_mean",
         "pure_ppo_context_grad_norm_max"),
        ("pure_ppo_context_output_grad_norm_mean",
         "pure_ppo_context_output_grad_norm_max"),
        ("pure_ppo_context_encoder_grad_norm_mean",
         "pure_ppo_context_encoder_grad_norm_max"),
        ("pure_ppo_context_interaction_grad_norm_mean",
         "pure_ppo_context_interaction_grad_norm_max"),
    )
    norm_summaries_closed = all(
        0.0 <= receipt[mean_key]
        and _closed_leq(
            receipt[mean_key], receipt[max_key])
        for mean_key, max_key in norm_pairs
    )
    root_square_bounds_closed = (
        _closed_leq(
            receipt["pure_ppo_root_grad_norm_max"] ** 2,
            receipt["pure_ppo_root_grad_sq_sum"],
        )
        and _closed_leq(
            receipt["pure_ppo_root_grad_sq_sum"],
            optimizer_steps
            * receipt["pure_ppo_root_grad_norm_max"] ** 2,
        )
        and _closed_leq(
            optimizer_steps
            * receipt["pure_ppo_root_grad_norm_mean"] ** 2,
            receipt["pure_ppo_root_grad_sq_sum"],
        )
    )
    context_square_bounds_closed = (
        _closed_leq(
            receipt["pure_ppo_context_grad_norm_max"] ** 2,
            receipt["pure_ppo_context_grad_sq_sum"],
        )
        and _closed_leq(
            receipt["pure_ppo_context_grad_sq_sum"],
            optimizer_steps
            * receipt["pure_ppo_context_grad_norm_max"] ** 2,
        )
        and _closed_leq(
            optimizer_steps
            * receipt["pure_ppo_context_grad_norm_mean"] ** 2,
            receipt["pure_ppo_context_grad_sq_sum"],
        )
    )
    return (
        receipt["schema"] == WORKER_ONPOLICY_PG_AUDIT_SCHEMA
        and isinstance(receipt["collection_actor_sha256"], str)
        and len(receipt["collection_actor_sha256"]) == 64
        and all(
            character in "0123456789abcdef"
            for character in receipt["collection_actor_sha256"]
        )
        and receipt["onpolicy_log_prob_source"]
        == "collection-policy-vs-MaskableRolloutBuffer.log_probs"
        and receipt["onpolicy_log_prob_max_abs_delta"] <= 1e-6
        and receipt["transition_reward_source"]
        == "WorkerWindowEnv.info.transition_reward"
        and receipt["advantage_source"]
        == "independently-recomputed-SB3-GAE"
        and receipt["gae_recomputed_max_abs_delta"] == 0.0
        and receipt["return_recomputed_max_abs_delta"] == 0.0
        and receipt["pure_ppo_gradient_source"]
        == "clipped-PPO-surrogate-only"
        and receipt["rollout_end_timesteps"] >= 0
        and (expected_samples is None or samples == expected_samples)
        and sum(requested_counts) == samples
        and sum(executed_counts) <= samples
        and 0 <= combat_nonzero <= combat_samples
        <= executed_counts[9] <= requested_counts[9]
        and combat_positive + combat_negative == combat_nonzero
        and 0 <= combat_positive_advantages <= combat_samples
        and combat_nonzero <= nonzero
        and _closed_leq(
            receipt["combat_transition_reward_abs_sum"],
            receipt["transition_reward_abs_sum"],
            abs_tol=1e-12,
        )
        and (
            receipt["combat_transition_reward_abs_sum"] == 0.0
        ) == (combat_nonzero == 0)
        and receipt["gae_advantage_samples"] == samples
        and 0 <= nonzero <= samples
        and positive + negative == nonzero
        and (
            receipt["transition_reward_abs_sum"] == 0.0
        ) == (nonzero == 0)
        and _closed_leq(
            abs(receipt["transition_reward_sum"]),
            receipt["transition_reward_abs_sum"],
            abs_tol=1e-12,
        )
        and _closed_leq(
            abs(receipt["combat_transition_reward_sum"]),
            receipt["combat_transition_reward_abs_sum"],
            abs_tol=1e-12,
        )
        and (positive > 0
             or receipt["transition_reward_sum"] <= 0.0)
        and (negative > 0
             or receipt["transition_reward_sum"] >= 0.0)
        and (combat_positive > 0
             or receipt["combat_transition_reward_sum"] <= 0.0)
        and (combat_negative > 0
             or receipt["combat_transition_reward_sum"] >= 0.0)
        and reward_mean_closed
        and reward_second_moment_closed
        and 0 <= timeout_samples <= samples
        and math.isclose(
            timeout_total_sum,
            timeout_base_sum + timeout_additional_sum,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and (
            (
                timeout_base_sum == 0.0
                and timeout_additional_sum == 0.0
                and timeout_total_sum == 0.0
            )
            if timeout_samples == 0
            else (
                timeout_base_sum < 0.0
                and timeout_additional_sum <= 0.0
                and timeout_total_sum < 0.0
            )
        )
        and 0 <= receipt["gae_advantage_nonzero_samples"] <= samples
        and receipt["transition_reward_abs_sum"] >= 0.0
        and receipt["transition_reward_variance"] >= 0.0
        and receipt["combat_transition_reward_abs_sum"] >= 0.0
        and receipt["combat_reward_centered_l2"] >= 0.0
        and receipt["combat_reward_centered_actor_grad_norm"] >= 0.0
        and receipt["combat_reward_centered_root_grad_norm"] >= 0.0
        and receipt["combat_reward_centered_context_grad_norm"] >= 0.0
        and receipt[
            "combat_reward_centered_context_output_grad_norm"
        ] >= 0.0
        and receipt[
            "combat_reward_centered_context_encoder_grad_norm"
        ] >= 0.0
        and receipt[
            "combat_reward_centered_context_interaction_grad_norm"
        ] >= 0.0
        and receipt["reward_centered_l2"] >= 0.0
        and receipt["reward_centered_actor_grad_norm"] >= 0.0
        and receipt["reward_centered_context_grad_norm"] >= 0.0
        and receipt["reward_centered_context_output_grad_norm"] >= 0.0
        and receipt["reward_centered_context_encoder_grad_norm"] >= 0.0
        and receipt["reward_centered_context_interaction_grad_norm"]
        >= 0.0
        and receipt["gae_advantage_variance"] >= 0.0
        and measurements == optimizer_steps
        and root_measurements == optimizer_steps
        and context_measurements == optimizer_steps
        and output_measurements == optimizer_steps
        and encoder_measurements == optimizer_steps
        and interaction_measurements == optimizer_steps
        and distill_measurements == optimizer_steps
        # A4 修正案(2026-07-27 批系):target_kl 早停的 rollout 以记录旗
        # 豁免至 ≥1(活性);未早停仍要求满一 epoch(≥8)。
        and receipt.get("kl_early_stopped") in (True, False)
        and optimizer_steps >= (
            1 if receipt["kl_early_stopped"] is True
            else WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT
        )
        and receipt["pure_ppo_actor_grad_norm_max"] >= 0.0
        and receipt["pure_ppo_actor_grad_norm_mean"] >= 0.0
        and receipt["pure_ppo_root_grad_norm_max"] >= 0.0
        and receipt["pure_ppo_root_grad_norm_mean"] >= 0.0
        and receipt["pure_ppo_context_grad_norm_max"] >= 0.0
        and receipt["pure_ppo_context_grad_norm_mean"] >= 0.0
        and receipt["pure_ppo_context_output_grad_norm_max"] >= 0.0
        and receipt["pure_ppo_context_output_grad_norm_mean"] >= 0.0
        and receipt["pure_ppo_context_encoder_grad_norm_max"] >= 0.0
        and receipt["pure_ppo_context_encoder_grad_norm_mean"] >= 0.0
        and receipt["pure_ppo_context_interaction_grad_norm_max"] >= 0.0
        and receipt["pure_ppo_context_interaction_grad_norm_mean"] >= 0.0
        and norm_summaries_closed
        and receipt["pure_ppo_root_grad_norm_max"]
        <= receipt["pure_ppo_actor_grad_norm_max"]
        and receipt["pure_ppo_context_grad_norm_max"]
        <= receipt["pure_ppo_actor_grad_norm_max"]
        and receipt["pure_ppo_context_output_grad_norm_max"]
        <= receipt["pure_ppo_context_grad_norm_max"]
        and receipt["pure_ppo_context_encoder_grad_norm_max"]
        <= receipt["pure_ppo_context_grad_norm_max"]
        and receipt["pure_ppo_context_interaction_grad_norm_max"]
        <= receipt["pure_ppo_context_grad_norm_max"]
        and receipt["distill_context_grad_norm_max"] == 0.0
        and receipt["pure_ppo_root_grad_sq_sum"] >= 0.0
        and root_square_bounds_closed
        and root_projection_closed
        and receipt["pure_ppo_context_grad_sq_sum"] >= 0.0
        and context_square_bounds_closed
        and projection_closed
        and pure_combat_actor_closed
        and pure_combat_root_closed
        and pure_combat_context_closed
        and pure_combat_partition_closed
        and receipt["optimizer_delta_source"]
        == "rollout-start-to-post-main-PPO-parameters"
        and combat_partition_closed
        and delta_actor_closed
        and delta_root_closed
        and delta_context_closed
        and delta_partition_closed
        and delta_actor_geometry_closed
        and delta_root_geometry_closed
        and delta_context_geometry_closed
        and delta_l2_partition_closed
        and receipt["qualifies"] is qualifies
    )


def worker_onpolicy_pg_audit_complete(model) -> bool:
    """Return whether a migration run has formal reward-driven PG evidence."""
    receipts = getattr(
        model, "_worker_onpolicy_pg_rollout_receipts", None)
    joint_rollouts = getattr(
        model, "_worker_onpolicy_pg_joint_rollouts", None)
    qualifying_rollouts = getattr(
        model, "_worker_onpolicy_pg_qualifying_rollouts", None)
    expected_samples = (
        int(getattr(model, "n_steps", 0))
        * int(getattr(model, "n_envs", 0))
    )
    if (
        getattr(model, "_worker_onpolicy_pg_audit_required", None) is not True
        or not isinstance(receipts, list)
        or type(joint_rollouts) is not int
        or type(qualifying_rollouts) is not int
        or joint_rollouts != len(receipts)
        or expected_samples <= 0
        or not receipts
    ):
        return False
    valid = [
        validate_worker_onpolicy_pg_receipt(
            receipt, expected_samples=expected_samples)
        for receipt in receipts
    ]
    return (
        all(valid)
        and qualifying_rollouts
        == sum(receipt["qualifies"] is True for receipt in receipts)
        and 0 < qualifying_rollouts <= joint_rollouts
    )


# Source compatibility only; newly persisted schemas/fields use ``worker``.
validate_combat_onpolicy_pg_receipt = (
    validate_worker_onpolicy_pg_receipt)
combat_onpolicy_pg_audit_complete = worker_onpolicy_pg_audit_complete


def clip_actor_critic_gradients(
        policy, optimizer, max_grad_norm: float, *,
        actor_frozen: bool,
        separate_root_context: bool = False,
) -> dict:
    """Apply independent main-PPO clipping while retaining one optimizer.

    During critic-only warmup the actor graph is intentionally still built so
    calibration can inspect its counterfactual gradient.  Clearing gradients
    to ``None`` after backward makes Adam skip those parameters exactly,
    including weight decay and any existing moments.
    """
    if (not isinstance(max_grad_norm, (int, float))
            or not math.isfinite(float(max_grad_norm))
            or float(max_grad_norm) <= 0.0):
        raise ValueError("max_grad_norm 必须是有限正数")
    partition = strict_actor_critic_parameter_partition(
        policy, optimizer=optimizer)
    actor = partition["actor"]
    critic = partition["critic"]
    actor_counterfactual_norm = _finite_gradient_norm(actor)
    adapter = getattr(
        getattr(policy, "mlp_extractor", None),
        "context_adapter",
        None,
    )
    context = (
        tuple(adapter.parameters())
        if isinstance(adapter, th.nn.Module) else ()
    )
    context_ids = {id(parameter) for parameter in context}
    root = tuple(
        parameter for parameter in actor
        if id(parameter) not in context_ids
    )
    if context and (
        not root
        or len(root) + len(context) != len(actor)
    ):
        raise RuntimeError(
            "actor root/context gradient 分组不闭合")
    root_counterfactual_norm = _finite_gradient_norm(root)
    context_counterfactual_norm = _finite_gradient_norm(context)
    if separate_root_context and not context:
        raise RuntimeError(
            "root/context 分组裁剪要求 nonlinear context adapter")
    if actor_frozen:
        for parameter in actor:
            parameter.grad = None
    if separate_root_context:
        branch_limit = float(max_grad_norm) / math.sqrt(2.0)
        root_norm = float(th.nn.utils.clip_grad_norm_(
            root, branch_limit,
            error_if_nonfinite=True).detach().cpu())
        context_norm = float(th.nn.utils.clip_grad_norm_(
            context, branch_limit,
            error_if_nonfinite=True).detach().cpu())
        actor_norm = math.hypot(root_norm, context_norm)
        root_clip_scale = (
            min(1.0, branch_limit / max(root_norm, branch_limit))
            if not actor_frozen else 0.0
        )
        context_clip_scale = (
            min(
                1.0,
                branch_limit / max(context_norm, branch_limit),
            )
            if not actor_frozen else 0.0
        )
        root_clipped = bool(root_norm > branch_limit)
        context_clipped = bool(context_norm > branch_limit)
    else:
        branch_limit = None
        actor_norm = float(th.nn.utils.clip_grad_norm_(
            actor, float(max_grad_norm),
            error_if_nonfinite=True).detach().cpu())
        root_norm = root_counterfactual_norm
        context_norm = context_counterfactual_norm
        common_scale = (
            min(
                1.0,
                float(max_grad_norm)
                / max(actor_norm, float(max_grad_norm)),
            )
            if not actor_frozen else 0.0
        )
        root_clip_scale = common_scale
        context_clip_scale = common_scale
        root_clipped = bool(actor_norm > float(max_grad_norm))
        context_clipped = bool(
            context and actor_norm > float(max_grad_norm))
    critic_norm = float(th.nn.utils.clip_grad_norm_(
        critic, float(max_grad_norm), error_if_nonfinite=True).detach().cpu())
    if not all(math.isfinite(value) for value in (
            actor_counterfactual_norm, root_counterfactual_norm,
            context_counterfactual_norm, root_norm, context_norm,
            actor_norm, critic_norm, root_clip_scale,
            context_clip_scale)):
        raise RuntimeError("actor/critic pre-clip norm 非有限")
    actor_clip_scale = (
        min(root_clip_scale, context_clip_scale)
        if context else root_clip_scale
    )
    return {
        "actor_counterfactual_norm": actor_counterfactual_norm,
        "root_counterfactual_norm": root_counterfactual_norm,
        "context_counterfactual_norm": context_counterfactual_norm,
        "actor_preclip_norm": actor_norm,
        "root_preclip_norm": root_norm,
        "context_preclip_norm": context_norm,
        "critic_preclip_norm": critic_norm,
        "actor_clipped": bool(root_clipped or context_clipped),
        "root_clipped": root_clipped,
        "context_clipped": context_clipped,
        "actor_clip_scale": actor_clip_scale,
        "root_clip_scale": root_clip_scale,
        "context_clip_scale": context_clip_scale,
        "root_context_max_norm": branch_limit,
        "critic_clipped": bool(critic_norm > float(max_grad_norm)),
        "actor_frozen": bool(actor_frozen),
    }


def _legacy_worker_observation_view(obs: th.Tensor) -> th.Tensor:
    """Return the canonical, non-mutating protocol-v3 worker view.

    Observation spaces smaller than 298 dimensions predate/do not use this
    worker contract (notably self-contained unit-test environments), so they
    are returned unchanged.

    The overlay belt scalar is ``legacy_heals/8 + free_slots/128`` with both
    counts integral in ``[0, 8]``.  Flooring its eightfold value exactly removes
    the reversible free-slot sub-tick.  Current ``_worker_obs`` separately
    maintains the exact old no-kill clock/exhausted state and encodes a drink
    latch as ``-(1 + exhausted)``.  Decoding that sign preserves the historical
    distinction between a saturated no-kill clock and the exhausted flag.
    """
    if obs.ndim == 0 or obs.shape[-1] <= _LEGACY_EXHAUSTED_FEATURE:
        return obs
    legacy = obs.clone()
    packed_belt = legacy[..., _LEGACY_BELT_FEATURE]
    legacy[..., _LEGACY_BELT_FEATURE] = (
        th.floor(th.clamp(packed_belt * 8.0, min=0.0, max=8.0))
        / 8.0
    )
    encoded_exhausted = legacy[..., _LEGACY_EXHAUSTED_FEATURE]
    legacy[..., _LEGACY_EXHAUSTED_FEATURE] = th.where(
        encoded_exhausted < 0.0,
        -encoded_exhausted - 1.0,
        encoded_exhausted,
    )
    return legacy


# Kept as a narrow source-compatibility alias for out-of-tree forensic
# notebooks.  New production code and tests must use the semantic name above.
_legacy_scene_clock_view = _legacy_worker_observation_view


def _a12_mixture_logits(
        raw_logits: th.Tensor,
        observations: th.Tensor,
        action_masks,
        gate_logits: th.Tensor,
        *,
        action: int = 12,
        hp_low: float = 0.5,
        hp_high: float = 0.65,
        boundary_epsilon: float = 1e-6,
        probability_min: float = 0.001,
        probability_max: float = 0.95,
) -> th.Tensor:
    """Build an exact, state-conditioned a12 mixture on legal support.

    For an eligible row,

    ``ε(s)=p_min+(p_max-p_min)·sigmoid(g(s))``

    ``π'(12)=ε(s)`` and ``π'(a≠12)=(1-ε(s))·π_non12(a)``.

    This is done at the policy distribution layer (not as an environment-side
    action override), so rollout sampling and PPO log-prob evaluation use the
    same distribution.  The gate starts at ε=0.05, where a12 cannot be
    argmax, but unlike rev8 it can learn a contextual ε(s)>0.5 and therefore
    become reachable under deterministic deployment.  The visible hp/latch
    predicate remains a hard outer safety envelope: outside it p12 is exactly
    zero regardless of the learned gate.
    """
    if raw_logits.ndim != 2 or observations.ndim != 2:
        raise ValueError("a12 mixture 只接受二维 batch logits/observations")
    if raw_logits.shape[0] != observations.shape[0] \
            or not 0 <= int(action) < raw_logits.shape[1]:
        raise ValueError("a12 mixture batch/action 形状异常")
    if observations.shape[1] <= _LEGACY_EXHAUSTED_FEATURE:
        raise ValueError("a12 mixture 要求 298 维 worker observation")
    if action_masks is None:
        masks = th.ones_like(raw_logits, dtype=th.bool)
    else:
        masks = th.as_tensor(action_masks, device=raw_logits.device)
        masks = masks.reshape(raw_logits.shape).bool()
    non12 = masks.clone()
    non12[:, int(action)] = False
    if not bool(non12.any(dim=-1).all().item()):
        raise ValueError("a12 mixture 每行至少需要一个合法非12动作")

    hp = observations[:, 0]
    latch = observations[:, _LEGACY_EXHAUSTED_FEATURE]
    eligible = (
        (hp >= hp_low - boundary_epsilon)
        & (hp < hp_high - boundary_epsilon)
        & (latch >= 0.0)
        & masks[:, int(action)]
    )
    if (
        not isinstance(probability_min, (int, float))
        or not isinstance(probability_max, (int, float))
        or not np.isfinite(float(probability_min))
        or not np.isfinite(float(probability_max))
        or not 0.0 < float(probability_min) < float(probability_max) < 1.0
    ):
        raise ValueError("a12 mixture 概率上下界非法")
    gate = gate_logits.to(
        device=raw_logits.device, dtype=raw_logits.dtype)
    if gate.ndim == 0:
        gate = gate.expand(raw_logits.shape[0])
    else:
        gate = gate.reshape(-1)
    if gate.shape != (raw_logits.shape[0],) \
            or not bool(th.isfinite(gate).all().item()):
        raise ValueError("a12 mixture gate logits 形状/有限性异常")
    probability = (
        float(probability_min)
        + (float(probability_max) - float(probability_min))
        * th.sigmoid(gate)
    )
    row_probability = th.where(
        eligible, probability, th.zeros_like(hp))
    non12_lse = th.logsumexp(
        th.where(non12, raw_logits, th.full_like(raw_logits, HUGE_NEG)),
        dim=-1,
    )
    mixed = (
        raw_logits
        - non12_lse.unsqueeze(-1)
        + th.log1p(-row_probability).unsqueeze(-1)
    )
    a12_log_probability = th.where(
        eligible,
        th.log(row_probability.clamp_min(th.finfo(raw_logits.dtype).tiny)),
        th.full_like(row_probability, HUGE_NEG),
    )
    mixed[:, int(action)] = a12_log_probability
    return th.where(masks, mixed, th.full_like(mixed, HUGE_NEG))


class _ExactMixtureCategoricalDistribution(
        MaskableCategoricalDistribution):
    """Categorical sampler with analytic mixture log-probabilities.

    PyTorch's ``Categorical(logits=...)`` always runs another log-sum-exp
    normalization.  The logits above are already exact log probabilities, so
    that second normalization is mathematically redundant.  In float32 its
    residual derivative is nevertheless around 1e-8; Adam can turn that into
    a persistent update of the supposedly isolated legacy actor on an a12
    sample.  Sampling/mode still use the standard masked categorical, while
    PPO log-prob and entropy consume the analytic mixture values directly.
    """

    def __init__(self, action_dim: int):
        super().__init__(action_dim)
        self._original_exact_log_probs: th.Tensor | None = None
        self._exact_log_probs: th.Tensor | None = None
        self._exact_probs: th.Tensor | None = None

    def _refresh_exact_values(self, exact: th.Tensor) -> None:
        """Match sampler values while retaining the analytic derivatives."""
        normalized_logs = self.distribution.logits
        normalized_probs = self.distribution.probs
        analytic_probs = exact.exp()
        # Straight-through numerical correction: forward values are bitwise
        # those used by the categorical sampler, while backward follows the
        # already-normalized mixture identity.  This avoids both the legacy
        # actor gradient leak and an O(ulp) sample/log-prob mismatch.
        self._exact_log_probs = (
            exact + (normalized_logs - exact).detach())
        self._exact_probs = (
            analytic_probs
            + (normalized_probs - analytic_probs).detach())

    def proba_distribution(
            self, action_logits: th.Tensor, *,
            exact_log_probs: th.Tensor | None = None):
        super().proba_distribution(action_logits)
        if exact_log_probs is None:
            self._original_exact_log_probs = None
            self._exact_log_probs = None
            self._exact_probs = None
        else:
            exact = exact_log_probs.reshape(-1, self.action_dim)
            if exact.shape != self.distribution.logits.shape:
                raise ValueError("exact mixture log-prob 形状异常")
            self._original_exact_log_probs = exact
            self._refresh_exact_values(exact)
        return self

    def apply_masking(self, masks) -> None:
        super().apply_masking(masks)
        exact = self._original_exact_log_probs
        if exact is None:
            self._exact_log_probs = None
            self._exact_probs = None
            return
        if masks is None:
            self._refresh_exact_values(exact)
            return
        mask = th.as_tensor(
            masks, dtype=th.bool, device=exact.device).reshape(exact.shape)
        masked_exact = th.where(
            mask, exact, th.full_like(exact, HUGE_NEG))
        self._refresh_exact_values(masked_exact)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        exact = self._exact_log_probs
        if exact is None:
            return super().log_prob(actions)
        indices = actions.long().reshape(-1, 1)
        if indices.shape[0] != exact.shape[0]:
            raise ValueError("exact mixture action batch 形状异常")
        return exact.gather(1, indices).reshape(-1)

    def entropy(self) -> th.Tensor:
        exact = self._exact_log_probs
        probabilities = self._exact_probs
        if exact is None or probabilities is None:
            return super().entropy()
        terms = th.where(
            probabilities > 0.0,
            probabilities * exact,
            th.zeros_like(exact),
        )
        return -terms.sum(dim=-1)


_DUAL_WORKER_SEGMENTS = {
    segment.name: segment for segment in DUAL_WORKER_LAYOUT.segments
}
_STRUCTURED_CONTEXT_SUMMARY_DIM = (
    32  # scalar blocks
    + 24  # map
    + 50 + CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_DIM
    + 48 + CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_DIM
    + 8  # belt
    + 8  # exact player/scene
    + 16 + CONTROLLER_SNAPSHOT_EQUIPPED_SLOTS * 12
    + 3  # sticky target
    + 16  # item targets
)


def _wire_segment(name: str):
    try:
        return _DUAL_WORKER_SEGMENTS[name]
    except KeyError as exc:
        raise RuntimeError(
            f"structured Worker encoder 缺 wire segment:{name}") from exc


class _CenteredTanhLinear(th.nn.Module):
    """A trainable thresholded projection whose exact zero input maps to zero."""

    def __init__(
            self, in_features: int, out_features: int, *,
            bias: bool = True, device=None):
        super().__init__()
        self.linear = th.nn.Linear(
            in_features, out_features, bias=bias, device=device)

    @property
    def in_features(self) -> int:
        return int(self.linear.in_features)

    @property
    def out_features(self) -> int:
        return int(self.linear.out_features)

    @property
    def weight(self) -> th.nn.Parameter:
        return self.linear.weight

    @property
    def bias(self) -> th.nn.Parameter | None:
        return self.linear.bias

    def forward(self, values: th.Tensor) -> th.Tensor:
        activated = th.tanh(self.linear(values))
        if self.linear.bias is None:
            return activated
        shape = (1,) * (values.ndim - 1) + (-1,)
        return activated - th.tanh(self.linear.bias).view(shape)


class _CenteredTanhConv2d(th.nn.Module):
    """Centered convolution used by the zero-anchored spatial encoder."""

    def __init__(self, *args, device=None, **kwargs):
        super().__init__()
        self.conv = th.nn.Conv2d(*args, device=device, **kwargs)

    def forward(self, values: th.Tensor) -> th.Tensor:
        activated = th.tanh(self.conv(values))
        if self.conv.bias is None:
            return activated
        return activated - th.tanh(self.conv.bias).view(1, -1, 1, 1)


def _masked_row_summaries(
        encoded: th.Tensor,
        present: th.Tensor,
) -> tuple[th.Tensor, th.Tensor]:
    """Return presence-aware mean/max without allowing padding to dominate."""
    if (
        encoded.ndim != 3
        or present.shape != encoded.shape[:2]
        or present.dtype != th.bool
    ):
        raise ValueError("structured row summary 形状异常")
    count = present.sum(dim=1, keepdim=True)
    weights = present.to(encoded.dtype).unsqueeze(-1)
    mean = (encoded * weights).sum(dim=1) / count.clamp_min(1).to(
        encoded.dtype)
    masked = encoded.masked_fill(~present.unsqueeze(-1), -th.inf)
    maximum = masked.max(dim=1).values
    maximum = th.where(count > 0, maximum, th.zeros_like(maximum))
    return mean, maximum


class _StructuredWorkerContextEncoder(th.nn.Module):
    """Layout-bound shared encoder for the dynamic non-legacy wire.

    Repeated map cells, monsters, missiles and equipment rows share weights;
    combat/resource blocks receive their own capacity instead of competing
    against thousands of map columns in one fan-in matrix.  Every learned
    activation is centered, so an all-zero wire maps to exact zero even after
    biases train.
    """

    def __init__(self, *, include_p_skip: bool, device):
        super().__init__()
        if (
            not DUAL_WORKER_LAYOUT.schema.startswith(
                "diablogym-dual-worker-layout/")
            or DUAL_WORKER_LAYOUT.banned_rng_tag_violations
        ):
            raise RuntimeError("structured Worker encoder 未认证当前 wire layout")
        self.include_p_skip = bool(include_p_skip)
        scalar_candidates = (
            "current_v4_base",
            "wrapper_scalars",
            "fuse_streak",
            "action_mask",
            "manager_mask",
        )
        role = "critic" if self.include_p_skip else "actor"
        self._scalar_segment_names = tuple(
            name for name in scalar_candidates
            if role in _wire_segment(name).semantic_tags
        )
        scalar_dim = sum(
            _wire_segment(name).width
            for name in self._scalar_segment_names
        )
        if (
            not self.include_p_skip
            and "wrapper_scalars" in self._scalar_segment_names
        ):
            scalar_dim -= 1
        if scalar_dim <= 0:
            raise RuntimeError("structured scalar wire 无可用字段")
        self.scalar_encoder = _CenteredTanhLinear(
            scalar_dim, 32, device=device)

        # The wire packs softwall/closed-door/explosive-softwall into one
        # exact 0..7 plane.  Decode it before a spatially shared projection.
        self.map_depthwise_1 = _CenteredTanhConv2d(
            9, 9, kernel_size=3, padding=1, groups=9, device=device)
        self.map_pointwise_1 = _CenteredTanhConv2d(
            9, 8, kernel_size=1, device=device)
        self.map_depthwise_2 = _CenteredTanhConv2d(
            8, 8, kernel_size=3, stride=2, padding=1,
            groups=8, device=device)
        self.map_pointwise_2 = _CenteredTanhConv2d(
            8, 12, kernel_size=1, device=device)
        self.map_depthwise_3 = _CenteredTanhConv2d(
            12, 12, kernel_size=3, stride=2, padding=1,
            groups=12, device=device)
        self.map_pointwise_3 = _CenteredTanhConv2d(
            12, 12, kernel_size=1, device=device)
        self.map_projection = _CenteredTanhLinear(
            12 * 7 * 7, 24, device=device)

        self.monster_projection_1 = _CenteredTanhLinear(
            CONTROLLER_SNAPSHOT_MONSTER_FIELDS, 24, device=device)
        self.monster_projection_2 = _CenteredTanhLinear(
            24, 16, device=device)
        self.missile_projection_1 = _CenteredTanhLinear(
            CONTROLLER_SNAPSHOT_MISSILE_FIELDS, 20, device=device)
        self.missile_projection_2 = _CenteredTanhLinear(
            20, 12, device=device)
        self.belt_projection = _CenteredTanhLinear(
            _wire_segment("controller_belt").width, 8, device=device)
        self.exact_projection = _CenteredTanhLinear(
            _wire_segment("controller_exact").width, 8, device=device)

        combat_global_dim = (
            len(CONTROLLER_SNAPSHOT_COMBAT_FIELDS)
            + CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS
            + CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS
        )
        if (
            combat_global_dim
            + CONTROLLER_SNAPSHOT_EQUIPPED_SLOTS
            * CONTROLLER_SNAPSHOT_EQUIPPED_FIELDS
            != _wire_segment("controller_combat").width
        ):
            raise RuntimeError("structured combat wire 维度未闭合")
        self._combat_global_dim = combat_global_dim
        self.combat_projection = _CenteredTanhLinear(
            combat_global_dim, 16, device=device)
        self.gear_projection = _CenteredTanhLinear(
            CONTROLLER_SNAPSHOT_EQUIPPED_FIELDS, 12, device=device)
        self.item_projection = _CenteredTanhLinear(
            _wire_segment("controller_item_targets").width,
            16,
            device=device,
        )

        try:
            self._monster_present_index = (
                CONTROLLER_SNAPSHOT_MONSTER_ROW_FIELDS.index("present"))
            self._monster_blocked_index = (
                CONTROLLER_SNAPSHOT_MONSTER_ROW_FIELDS.index("blocked"))
            self._missile_present_index = (
                CONTROLLER_SNAPSHOT_MISSILE_ROW_FIELDS.index("present"))
            self._gear_present_index = (
                CONTROLLER_SNAPSHOT_EQUIPPED_ROW_FIELDS.index("present"))
        except ValueError as exc:
            raise RuntimeError(
                "structured entity wire 缺 presence/blocked 字段") from exc
        self._p_skip_index = int(
            DUAL_WORKER_LAYOUT.p_skip_semantic_index)
        wrapper = _wire_segment("wrapper_scalars")
        if not wrapper.start <= self._p_skip_index < wrapper.stop:
            raise RuntimeError("p_skip 未位于 wrapper scalar segment")
        if not self.include_p_skip:
            for name in (
                "controller_map",
                "controller_monsters",
                "controller_missiles",
                "controller_belt",
                "controller_exact",
                "controller_combat",
                "controller_sticky",
                "controller_item_targets",
            ):
                if "actor" not in _wire_segment(name).semantic_tags:
                    raise RuntimeError(
                        "structured actor encoder 误接 critic-only segment:"
                        f"{name}")

    @staticmethod
    def _values(features: th.Tensor, name: str) -> th.Tensor:
        segment = _wire_segment(name)
        return features[:, segment.start:segment.stop]

    def _scalar_values(self, features: th.Tensor) -> th.Tensor:
        blocks = []
        for name in self._scalar_segment_names:
            values = self._values(features, name)
            if not self.include_p_skip and name == "wrapper_scalars":
                relative = (
                    self._p_skip_index
                    - _wire_segment(name).start
                )
                values = th.cat(
                    (values[:, :relative], values[:, relative + 1:]),
                    dim=1,
                )
            blocks.append(values)
        return th.cat(blocks, dim=1)

    @staticmethod
    def decode_map_wire(wire: th.Tensor) -> th.Tensor:
        """Decode the normalized three-bit softwall plane losslessly."""
        if (
            wire.ndim != 4
            or tuple(wire.shape[1:]) != (
                len(CONTROLLER_SNAPSHOT_MAP_CHANNELS), 25, 25)
        ):
            raise ValueError("structured map wire 形状异常")
        softwall_index = CONTROLLER_SNAPSHOT_MAP_CHANNELS.index(
            "softwall_kind")
        # The wire keeps the exact 3-bit code in float32 as code/7 so every
        # map plane remains on the common [0, 1] scale.
        packed = th.round(
            wire[:, softwall_index:softwall_index + 1] * 7.0
        ).clamp_(0.0, 7.0)
        softwall = th.remainder(packed, 2.0)
        closed_door = th.remainder(th.floor(packed / 2.0), 2.0)
        explosive = th.remainder(th.floor(packed / 4.0), 2.0)
        decoded = th.cat((
            wire[:, :softwall_index],
            softwall,
            closed_door,
            explosive,
            wire[:, softwall_index + 1:],
        ), dim=1)
        if decoded.shape[1:] != (9, 25, 25):
            raise RuntimeError("structured map decode 形状漂移")
        return decoded

    def _map_summary(self, features: th.Tensor) -> th.Tensor:
        segment = _wire_segment("controller_map")
        wire = self._values(features, segment.name).reshape(
            -1, *segment.shape)
        decoded = self.decode_map_wire(wire)
        hidden = self.map_pointwise_1(self.map_depthwise_1(decoded))
        hidden = self.map_pointwise_2(self.map_depthwise_2(hidden))
        hidden = self.map_pointwise_3(self.map_depthwise_3(hidden))
        return self.map_projection(hidden.flatten(start_dim=1))

    def _monster_summary(self, features: th.Tensor) -> th.Tensor:
        segment = _wire_segment("controller_monsters")
        values = self._values(features, segment.name)
        row_width = (
            CONTROLLER_SNAPSHOT_MONSTER_LIMIT
            * CONTROLLER_SNAPSHOT_MONSTER_FIELDS
        )
        rows = values[:, :row_width].reshape(
            -1,
            CONTROLLER_SNAPSHOT_MONSTER_LIMIT,
            CONTROLLER_SNAPSHOT_MONSTER_FIELDS,
        )
        overflow = values[:, row_width:]
        if overflow.shape[1] != CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_DIM:
            raise RuntimeError("structured monster overflow 形状漂移")
        encoded = self.monster_projection_2(
            self.monster_projection_1(rows))
        present = rows[:, :, self._monster_present_index] > 0.5
        encoded = encoded * present.to(encoded.dtype).unsqueeze(-1)
        mean, maximum = _masked_row_summaries(encoded, present)

        unblocked = (
            present
            & (rows[:, :, self._monster_blocked_index] < 0.5)
        )
        has_unblocked = unblocked.any(dim=1)
        selected_index = th.where(
            has_unblocked,
            unblocked.to(th.int64).argmax(dim=1),
            present.to(th.int64).argmax(dim=1),
        )
        batch = th.arange(
            rows.shape[0], device=rows.device, dtype=th.long)
        selected = encoded[batch, selected_index]
        selected = selected * present.any(
            dim=1, keepdim=True).to(encoded.dtype)
        counts = th.stack((
            present.sum(dim=1).to(encoded.dtype)
            / float(CONTROLLER_SNAPSHOT_MONSTER_LIMIT),
            (present & ~unblocked).sum(dim=1).to(encoded.dtype)
            / float(CONTROLLER_SNAPSHOT_MONSTER_LIMIT),
        ), dim=1)
        return th.cat(
            (selected, mean, maximum, counts, overflow), dim=1)

    def _missile_summary(self, features: th.Tensor) -> th.Tensor:
        segment = _wire_segment("controller_missiles")
        values = self._values(features, segment.name)
        row_width = (
            CONTROLLER_SNAPSHOT_MISSILE_LIMIT
            * CONTROLLER_SNAPSHOT_MISSILE_FIELDS
        )
        rows = values[:, :row_width].reshape(
            -1,
            CONTROLLER_SNAPSHOT_MISSILE_LIMIT,
            CONTROLLER_SNAPSHOT_MISSILE_FIELDS,
        )
        overflow = values[:, row_width:]
        if overflow.shape[1] != CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_DIM:
            raise RuntimeError("structured missile overflow 形状漂移")
        encoded = self.missile_projection_2(
            self.missile_projection_1(rows))
        present = rows[:, :, self._missile_present_index] > 0.5
        encoded = encoded * present.to(encoded.dtype).unsqueeze(-1)
        mean, maximum = _masked_row_summaries(encoded, present)
        first_two = encoded[:, :2].flatten(start_dim=1)
        return th.cat((first_two, mean, maximum, overflow), dim=1)

    def _combat_summary(self, features: th.Tensor) -> th.Tensor:
        values = self._values(features, "controller_combat")
        global_values = values[:, :self._combat_global_dim]
        gear = values[:, self._combat_global_dim:].reshape(
            -1,
            CONTROLLER_SNAPSHOT_EQUIPPED_SLOTS,
            CONTROLLER_SNAPSHOT_EQUIPPED_FIELDS,
        )
        gear_encoded = self.gear_projection(gear)
        gear_encoded = (
            gear_encoded
            * (
                gear[:, :, self._gear_present_index:
                          self._gear_present_index + 1] > 0.5
            ).to(gear_encoded.dtype)
        )
        return th.cat((
            self.combat_projection(global_values),
            gear_encoded.flatten(start_dim=1),
        ), dim=1)

    def forward(self, features: th.Tensor) -> th.Tensor:
        if (
            features.ndim != 2
            or features.shape[1] != DUAL_WORKER_LAYOUT.observation_dim
        ):
            raise ValueError("structured Worker encoder 输入形状异常")
        blocks = (
            self.scalar_encoder(self._scalar_values(features)),
            self._map_summary(features),
            self._monster_summary(features),
            self._missile_summary(features),
            self.belt_projection(
                self._values(features, "controller_belt")),
            self.exact_projection(
                self._values(features, "controller_exact")),
            self._combat_summary(features),
            self._values(features, "controller_sticky"),
            self.item_projection(
                self._values(features, "controller_item_targets")),
        )
        summary = th.cat(blocks, dim=1)
        if summary.shape[1] != _STRUCTURED_CONTEXT_SUMMARY_DIM:
            raise RuntimeError(
                "structured Worker context summary 维度漂移:"
                f"{summary.shape[1]}")
        return summary


class NonlinearFusedContextAdapter(th.nn.Module):
    """Structured, exact-zero actor residual with true context interaction."""

    def __init__(
            self, observation_dim: int, latent_dim: int, *,
            excluded_observation_features: tuple[int, ...], device):
        super().__init__()
        if (
            observation_dim != DUAL_WORKER_LAYOUT.observation_dim
            or latent_dim <= 0
        ):
            raise ValueError("structured context adapter 维度非法")
        excluded = tuple(
            int(index) for index in excluded_observation_features)
        if excluded != (
                int(DUAL_WORKER_LAYOUT.p_skip_semantic_index),):
            raise ValueError(
                "structured actor 只认证按 wire semantic 排除 p_skip")
        self._excluded_observation_features = excluded
        self.encoder = _StructuredWorkerContextEncoder(
            include_p_skip=False, device=device)
        self.context_projection = _CenteredTanhLinear(
            _STRUCTURED_CONTEXT_SUMMARY_DIM,
            ASYMMETRIC_WORKER_CONTEXT_HIDDEN_DIM,
            device=device,
        )
        self.context_gate = _CenteredTanhLinear(
            ASYMMETRIC_WORKER_CONTEXT_HIDDEN_DIM,
            ASYMMETRIC_WORKER_CONTEXT_INTERACTION_DIM,
            device=device,
        )
        self.legacy_gate = th.nn.Linear(
            latent_dim,
            ASYMMETRIC_WORKER_CONTEXT_INTERACTION_DIM,
            bias=True,
            device=device,
        )
        self.fusion = _CenteredTanhLinear(
            ASYMMETRIC_WORKER_CONTEXT_HIDDEN_DIM
            + ASYMMETRIC_WORKER_CONTEXT_INTERACTION_DIM,
            latent_dim,
            device=device,
        )
        self.output = th.nn.Linear(
            latent_dim, latent_dim, bias=False, device=device)
        self._validate_parameter_groups()

    @property
    def excluded_context_features(self) -> tuple[int, ...]:
        """Compatibility alias; values are absolute wire indices."""
        return self._excluded_observation_features

    @property
    def excluded_observation_features(self) -> tuple[int, ...]:
        return self._excluded_observation_features

    def named_parameter_groups(self) -> dict[str, tuple]:
        """Stable semantic groups for clipping and formal PG evidence."""
        return {
            "encoder": (
                *tuple(self.encoder.parameters()),
                *tuple(self.context_projection.parameters()),
            ),
            "interaction": (
                *tuple(self.context_gate.parameters()),
                *tuple(self.legacy_gate.parameters()),
                *tuple(self.fusion.parameters()),
            ),
            "output": tuple(self.output.parameters()),
        }

    def evidence_parameter_groups(self) -> dict[str, tuple]:
        """Projection/fusion aliases for older evidence consumers."""
        groups = self.named_parameter_groups()
        return {
            "projection": groups["encoder"],
            "fusion": groups["interaction"],
            "output": groups["output"],
        }

    def _validate_parameter_groups(self) -> None:
        groups = self.named_parameter_groups()
        if tuple(groups) != ("encoder", "interaction", "output"):
            raise RuntimeError("structured context parameter group 顺序漂移")
        flattened = tuple(
            parameter
            for values in groups.values()
            for parameter in values
        )
        parameters = tuple(self.parameters())
        if (
            not all(groups.values())
            or len({id(parameter) for parameter in flattened})
            != len(flattened)
            or {id(parameter) for parameter in flattened}
            != {id(parameter) for parameter in parameters}
        ):
            raise RuntimeError(
                "structured context parameter groups 未精确覆盖")

    def initialize_canonical(self) -> None:
        """Initialize hidden modules from a private RNG and output at zero."""
        generator = th.Generator(device=self.output.weight.device)
        generator.manual_seed(ASYMMETRIC_WORKER_CONTEXT_INIT_SEED)
        with th.no_grad():
            for name, module in self.named_modules():
                if name == "output":
                    continue
                linear_or_conv = (
                    isinstance(module, th.nn.Linear)
                    or isinstance(module, th.nn.Conv2d)
                )
                if not linear_or_conv:
                    continue
                if isinstance(module, th.nn.Linear):
                    fan_in = int(module.in_features)
                else:
                    fan_in = (
                        int(module.in_channels)
                        // int(module.groups)
                        * int(module.kernel_size[0])
                        * int(module.kernel_size[1])
                    )
                bound = 1.0 / math.sqrt(float(fan_in))
                module.weight.uniform_(
                    -bound, bound, generator=generator)
                if module.bias is not None:
                    module.bias.zero_()
            self.output.weight.zero_()

    def preoutput(
            self, features: th.Tensor,
            legacy_latent: th.Tensor) -> th.Tensor:
        if (
            features.ndim != 2
            or legacy_latent.ndim != 2
            or features.shape[0] != legacy_latent.shape[0]
            or features.shape[1] != DUAL_WORKER_LAYOUT.observation_dim
            or legacy_latent.shape[1] != self.legacy_gate.in_features
        ):
            raise ValueError("structured nonlinear adapter 输入形状异常")
        context = self.context_projection(self.encoder(features))
        context_gate = self.context_gate(context)
        legacy_gate = th.tanh(
            self.legacy_gate(legacy_latent.detach()))
        interaction = context_gate * legacy_gate
        return self.fusion(th.cat((context, interaction), dim=1))

    def forward(
            self, features: th.Tensor,
            legacy_latent: th.Tensor) -> th.Tensor:
        return self.output(self.preoutput(features, legacy_latent))


class _StructuredWorkerCritic(th.nn.Module):
    """Independent small critic; no actor parameter is shared or warmed."""

    def __init__(self, *, device):
        super().__init__()
        self.encoder = _StructuredWorkerContextEncoder(
            include_p_skip=True, device=device)
        self.context_projection = _CenteredTanhLinear(
            _STRUCTURED_CONTEXT_SUMMARY_DIM,
            ASYMMETRIC_WORKER_CONTEXT_HIDDEN_DIM,
            device=device,
        )
        self.legacy_projection = _CenteredTanhLinear(
            ASYMMETRIC_WORKER_LEGACY_DIM, 32, device=device)
        self.fusion = _CenteredTanhLinear(
            ASYMMETRIC_WORKER_CONTEXT_HIDDEN_DIM + 32,
            64,
            device=device,
        )
        self.post = _CenteredTanhLinear(64, 64, device=device)

    def forward(self, features: th.Tensor) -> th.Tensor:
        if (
            features.ndim != 2
            or features.shape[1] != DUAL_WORKER_LAYOUT.observation_dim
        ):
            raise ValueError("structured critic features 形状异常")
        context = self.context_projection(self.encoder(features))
        legacy = self.legacy_projection(
            features[:, :ASYMMETRIC_WORKER_LEGACY_DIM])
        return self.post(self.fusion(th.cat((legacy, context), dim=1)))


class AsymmetricWorkerMlpExtractor(th.nn.Module):
    """Bitwise legacy actor plus a learnable v4 context residual.

    The old actor branch performs the exact original 298→64→64 sequence;
    it is never widened into a different GEMM.  A separate zero-initialized
    context adapter is disabled throughout critic warmup and later adds a
    zero-output nonlinear residual at the final policy latent.  Training-only
    sampler controls are masked out of that actor residual.  The fresh critic
    consumes the complete bounded controller snapshot used by the executed
    macros, rather than a lossy legacy-only value input.
    """

    def __init__(
            self, feature_dim: int, net_arch,
            activation_fn, device):
        super().__init__()
        if int(feature_dim) != ASYMMETRIC_WORKER_OBSERVATION_DIM:
            raise ValueError(
                "asymmetric Worker extractor 要求 "
                f"{ASYMMETRIC_WORKER_OBSERVATION_DIM} 维观测")
        if activation_fn is not th.nn.Tanh:
            raise ValueError(
                "asymmetric Worker actor migration 只认证 Tanh 激活")
        if isinstance(net_arch, dict):
            pi_arch = list(net_arch.get("pi", ()))
            vf_arch = list(net_arch.get("vf", ()))
        else:
            pi_arch = vf_arch = list(net_arch)
        if pi_arch != [64, 64] or vf_arch != [64, 64]:
            raise ValueError(
                "asymmetric Worker 只认证 pi/vf=[64,64] 拓扑")
        legacy = MlpExtractor(
            ASYMMETRIC_WORKER_LEGACY_DIM,
            net_arch={"pi": pi_arch, "vf": []},
            activation_fn=activation_fn,
            device=device,
        )
        self.policy_net = legacy.policy_net
        self.value_net = _StructuredWorkerCritic(device=device)
        self.latent_dim_pi = legacy.latent_dim_pi
        self.latent_dim_vf = 64
        context_dim = (
            ASYMMETRIC_WORKER_OBSERVATION_DIM
            - ASYMMETRIC_WORKER_LEGACY_DIM
        )
        self.context_adapter = NonlinearFusedContextAdapter(
            ASYMMETRIC_WORKER_OBSERVATION_DIM,
            self.latent_dim_pi,
            excluded_observation_features=(
                ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES),
            device=device,
        )
        actor_context_mask = th.ones(
            context_dim,
            dtype=self.context_adapter.context_projection.weight.dtype,
            device=device,
        )
        for segment in DUAL_WORKER_LAYOUT.segments:
            if (
                "critic" in segment.semantic_tags
                and "actor" not in segment.semantic_tags
                and segment.stop > ASYMMETRIC_WORKER_LEGACY_DIM
            ):
                start = max(
                    segment.start, ASYMMETRIC_WORKER_LEGACY_DIM)
                actor_context_mask[
                    start - ASYMMETRIC_WORKER_LEGACY_DIM:
                    segment.stop - ASYMMETRIC_WORKER_LEGACY_DIM
                ] = 0.0
        for absolute_index in ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES:
            if not (
                ASYMMETRIC_WORKER_LEGACY_DIM
                <= absolute_index
                < ASYMMETRIC_WORKER_OBSERVATION_DIM
            ):
                raise ValueError(
                    "asymmetric actor excluded feature 越界:"
                    f"{absolute_index}")
            actor_context_mask[
                absolute_index - ASYMMETRIC_WORKER_LEGACY_DIM] = 0.0
        self.register_buffer(
            "_actor_context_feature_mask",
            actor_context_mask,
            persistent=False,
        )
        self.register_buffer(
            "_context_enabled",
            th.zeros((), dtype=th.bool, device=device),
        )

    @property
    def actor_context_enabled(self) -> bool:
        return bool(self._context_enabled.item())

    def enable_actor_context(self) -> None:
        self._context_enabled.fill_(True)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        if (features.ndim != 2
                or features.shape[1]
                != ASYMMETRIC_WORKER_OBSERVATION_DIM):
            raise ValueError("asymmetric actor features 形状异常")
        legacy = self.policy_net(
            features[:, :ASYMMETRIC_WORKER_LEGACY_DIM])
        if not self.actor_context_enabled:
            return legacy
        # The structured encoder physically omits p_skip by semantic field
        # name.  The compatibility mask remains audit-only; it is not relied
        # upon as the exclusion mechanism.
        context = self.context_adapter(features, legacy)
        return legacy + context

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        if (features.ndim != 2
                or features.shape[1]
                != ASYMMETRIC_WORKER_OBSERVATION_DIM):
            raise ValueError("asymmetric critic features 形状异常")
        return self.value_net(features)

    def forward(self, features: th.Tensor):
        return self.forward_actor(features), self.forward_critic(features)


class AsymmetricWorkerMaskableActorCriticPolicy(
        MaskableActorCriticPolicy):
    """Protocol-v3-root actor with full protocol-v4 critic/context input."""

    def __init__(
            self, *args, action14_logit_bonus: float = 0.0, **kwargs):
        if (
            not isinstance(action14_logit_bonus, (int, float))
            or isinstance(action14_logit_bonus, bool)
            or not np.isfinite(float(action14_logit_bonus))
            or not 0.0 <= float(action14_logit_bonus) <= 10.0
        ):
            raise ValueError(
                "action14_logit_bonus 必须是 [0,10] 内有限数")
        self.action14_logit_bonus = float(action14_logit_bonus)
        super().__init__(*args, **kwargs)
        if int(getattr(self.action_space, "n", 0)) != 15:
            raise ValueError(
                "asymmetric Worker action14 prior 要求 15 动作离散空间")
        # MaskableActorCriticPolicy orthogonally initializes every Linear in
        # the extractor.  Replace the context branch with its private,
        # run-seed-independent hidden initialization and exact-zero output.
        self.mlp_extractor.context_adapter.initialize_canonical()

    def _build_mlp_extractor(self) -> None:
        if (
            type(self.features_extractor) is not FlattenExtractor
            or not bool(self.share_features_extractor)
        ):
            raise ValueError(
                "asymmetric Worker actor migration 只认证共享 FlattenExtractor")
        self.mlp_extractor = AsymmetricWorkerMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _worker_distribution(
            self, latent_pi: th.Tensor, action_masks):
        """Apply the registered gear prior only on exact legal a14 rows.

        V28 assigns roughly 0.5% aggregate probability to a14 even on current
        scripted a14-positive states, so ordinary entropy exploration produces
        about one gear sample per two hundred opportunities.  A fixed logit
        offset is an on-policy prior rather than an action override: rollout
        sampling and PPO evaluation use the same distribution, and gradients
        still flow through the original a14 logit.  The environment's exact
        whole-loadout mask remains the sole eligibility authority.
        """
        raw_logits = self.action_net(latent_pi)
        adjusted_logits = raw_logits
        if self.action14_logit_bonus > 0.0 and action_masks is not None:
            masks = th.as_tensor(
                action_masks, dtype=th.bool, device=raw_logits.device,
            ).reshape(raw_logits.shape)
            legal_a14 = masks[:, 14]
            if bool(legal_a14.any().item()):
                adjusted_logits = raw_logits.clone()
                adjusted_logits[:, 14] = (
                    adjusted_logits[:, 14]
                    + legal_a14.to(raw_logits.dtype)
                    * self.action14_logit_bonus
                )
        distribution = self.action_dist.proba_distribution(
            action_logits=adjusted_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def forward(
            self, obs: th.Tensor, deterministic: bool = False,
            action_masks=None):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._worker_distribution(
            latent_pi, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def get_distribution(self, obs, action_masks=None):
        features = super().extract_features(
            obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._worker_distribution(latent_pi, action_masks)

    def evaluate_actions(
            self, obs: th.Tensor, actions: th.Tensor,
            action_masks=None):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._worker_distribution(
            latent_pi, action_masks)
        return (
            self.value_net(latent_vf),
            distribution.log_prob(actions),
            distribution.entropy(),
        )


def _runtime_tensor_bundle_sha256(*tensors: th.Tensor) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(tensors):
        tensor = value.detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(str(index).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _asymmetric_worker_behavior_probe(
        policy: AsymmetricWorkerMaskableActorCriticPolicy,
) -> th.Tensor:
    """Build deterministic, nonzero, RNG-free rows for runtime evidence."""
    try:
        reference = next(policy.parameters())
    except StopIteration as exc:
        raise RuntimeError("asymmetric Worker policy 无参数") from exc
    columns = th.arange(
        DUAL_WORKER_LAYOUT.observation_dim,
        dtype=reference.dtype,
        device=reference.device,
    ).unsqueeze(0)
    rows = th.arange(
        4,
        dtype=reference.dtype,
        device=reference.device,
    ).unsqueeze(1)
    probe = th.remainder(columns * 37.0 + rows * 17.0, 97.0) / 96.0

    # Keep the normalized bit-pack semantically valid and cover codes whose
    # individual bits distinguish all three decoded planes.
    map_segment = _wire_segment("controller_map")
    map_values = probe[:, map_segment.start:map_segment.stop].reshape(
        -1, *map_segment.shape)
    softwall_index = CONTROLLER_SNAPSHOT_MAP_CHANNELS.index(
        "softwall_kind")
    codes = th.tensor(
        (1.0, 2.0, 4.0, 7.0),
        dtype=probe.dtype,
        device=probe.device,
    ).view(-1, 1, 1)
    map_values[:, softwall_index].copy_(codes / 7.0)
    probe[:, DUAL_WORKER_LAYOUT.p_skip_semantic_index] = 0.0
    return probe


def asymmetric_worker_runtime_evidence(policy) -> dict:
    """Return the canonical, state-key-free asymmetric Worker evidence.

    Completion gates, R7 receipts and evaluators should consume this helper
    instead of knowing private module paths or relying on tuple positions.
    The probes are deterministic and do not touch Torch's global RNG.
    """
    if not isinstance(
            policy, AsymmetricWorkerMaskableActorCriticPolicy):
        raise TypeError(
            "runtime evidence 要求 AsymmetricWorkerMaskableActorCriticPolicy")
    extractor = policy.mlp_extractor
    if not isinstance(extractor, AsymmetricWorkerMlpExtractor):
        raise RuntimeError("runtime evidence 的 asymmetric extractor 缺失")
    adapter = extractor.context_adapter
    groups = adapter.named_parameter_groups()
    if tuple(groups) != ("encoder", "interaction", "output"):
        raise RuntimeError("runtime evidence 的 context groups 漂移")
    partition = strict_actor_critic_parameter_partition(
        policy, optimizer=getattr(policy, "optimizer", None))
    policy_parameters = (
        *partition["actor"],
        *partition["critic"],
    )
    if not all(bool(parameter.detach().isfinite().all().item())
               for parameter in policy_parameters):
        raise RuntimeError("runtime evidence 检出 policy NaN/Inf")
    actor_ids = {id(parameter) for parameter in partition["actor"]}
    grouped = tuple(
        parameter for values in groups.values() for parameter in values)
    if not all(id(parameter) in actor_ids for parameter in grouped):
        raise RuntimeError("runtime evidence context 参数不属于 actor")

    group_evidence = {}
    for name, parameters in groups.items():
        group_evidence[name] = {
            "parameter_count": int(sum(
                parameter.numel() for parameter in parameters)),
            "tensor_count": int(len(parameters)),
            "nonzero_count": int(sum(
                th.count_nonzero(parameter.detach()).item()
                for parameter in parameters
            )),
            "sha256": _parameter_group_sha256(policy, parameters),
        }

    context_enabled = extractor.actor_context_enabled
    probe = _asymmetric_worker_behavior_probe(policy)
    p_skip_zero = probe.clone()
    p_skip_one = probe.clone()
    p_skip_one[:, DUAL_WORKER_LAYOUT.p_skip_semantic_index] = 1.0
    with th.no_grad():
        legacy = extractor.policy_net(
            probe[:, :ASYMMETRIC_WORKER_LEGACY_DIM])
        p_skip_preoutput_zero = adapter.preoutput(p_skip_zero, legacy)
        p_skip_preoutput_one = adapter.preoutput(p_skip_one, legacy)
        p_skip_logits_zero = policy.action_net(
            legacy + adapter(p_skip_zero, legacy))
        p_skip_logits_one = policy.action_net(
            legacy + adapter(p_skip_one, legacy))
        legacy_logits = policy.action_net(legacy)
        forced_context_latent = legacy + adapter(probe, legacy)
        forced_context_logits = policy.action_net(forced_context_latent)
        deployed_logits = policy.action_net(extractor.forward_actor(probe))
        focused_probe_outputs = {}
        for segment_name in (
            "current_v4_base",
            "wrapper_scalars",
            "controller_combat",
        ):
            segment = _wire_segment(segment_name)
            registered = "actor" in segment.semantic_tags
            record = {
                "registered_actor_input": bool(registered),
                "feature_index": None,
                "preoutput_effect": False,
                "preoutput_changed_elements": 0,
                "preoutput_max_abs_delta": 0.0,
                "context_action_effect": False,
                "context_action_logit_changed_elements": 0,
                "context_action_logit_max_abs_delta": 0.0,
            }
            if registered:
                feature_index = int(segment.start)
                if feature_index == DUAL_WORKER_LAYOUT.p_skip_semantic_index:
                    feature_index += 1
                if not (
                    segment.start <= feature_index < segment.stop
                    and feature_index
                    != DUAL_WORKER_LAYOUT.p_skip_semantic_index
                ):
                    raise RuntimeError(
                        "runtime evidence 无合法 actor scalar probe 字段")
                scalar_zero = probe.clone()
                scalar_one = probe.clone()
                scalar_zero[:, feature_index] = 0.0
                scalar_one[:, feature_index] = 1.0
                preoutput_zero = adapter.preoutput(scalar_zero, legacy)
                preoutput_one = adapter.preoutput(scalar_one, legacy)
                scalar_delta = (
                    preoutput_one - preoutput_zero).abs()
                focused_logits_zero = policy.action_net(
                    legacy + adapter(scalar_zero, legacy))
                focused_logits_one = policy.action_net(
                    legacy + adapter(scalar_one, legacy))
                focused_action_delta = (
                    focused_logits_one - focused_logits_zero).abs()
                focused_action_changed = int(
                    th.count_nonzero(focused_action_delta).item())
                record.update({
                    "feature_index": feature_index,
                    "preoutput_effect": bool(
                        th.count_nonzero(scalar_delta).item() > 0),
                    "preoutput_changed_elements": int(
                        th.count_nonzero(scalar_delta).item()),
                    "preoutput_max_abs_delta": float(
                        scalar_delta.max().item()),
                    "context_action_effect": bool(
                        context_enabled and focused_action_changed > 0),
                    "context_action_logit_changed_elements":
                        focused_action_changed,
                    "context_action_logit_max_abs_delta": float(
                        focused_action_delta.max().item()),
                })
                focused_probe_outputs[segment_name] = (
                    record,
                    scalar_zero,
                    scalar_one,
                    preoutput_zero,
                    preoutput_one,
                    scalar_delta,
                    focused_logits_zero,
                    focused_logits_one,
                    focused_action_delta,
                )
            else:
                focused_probe_outputs[segment_name] = (
                    record,
                )

    p_skip_delta = (
        p_skip_preoutput_one - p_skip_preoutput_zero).abs()
    p_skip_action_delta = (
        p_skip_logits_one - p_skip_logits_zero).abs()
    forced_delta = (forced_context_logits - legacy_logits).abs()
    deployed_delta = (deployed_logits - legacy_logits).abs()
    probe_tensors = (
        probe,
        legacy,
        p_skip_preoutput_zero,
        p_skip_preoutput_one,
        p_skip_logits_zero,
        p_skip_logits_one,
        legacy_logits,
        forced_context_latent,
        forced_context_logits,
        deployed_logits,
        p_skip_delta,
        p_skip_action_delta,
        forced_delta,
        deployed_delta,
        *(
            value
            for outputs in focused_probe_outputs.values()
            for value in outputs[1:]
        ),
    )
    if not all(bool(value.isfinite().all().item())
               for value in probe_tensors):
        raise RuntimeError("runtime evidence 行为探针产生 NaN/Inf")
    p_skip_invariant = bool(th.equal(
        p_skip_preoutput_zero, p_skip_preoutput_one))
    p_skip_action_invariant = bool(th.equal(
        p_skip_logits_zero, p_skip_logits_one))
    forced_changed = int(th.count_nonzero(forced_delta).item())
    deployed_changed = int(th.count_nonzero(deployed_delta).item())
    return {
        "schema": ASYMMETRIC_WORKER_RUNTIME_EVIDENCE_SCHEMA,
        "layout": {
            "schema": DUAL_WORKER_LAYOUT.schema,
            "sha256": DUAL_WORKER_LAYOUT_SHA256,
            "observation_dim": int(DUAL_WORKER_LAYOUT.observation_dim),
            "p_skip_semantic_index": int(
                DUAL_WORKER_LAYOUT.p_skip_semantic_index),
        },
        "policy": {
            "actor_parameter_count": int(sum(
                parameter.numel()
                for parameter in partition["actor"])),
            "actor_tensor_count": int(len(partition["actor"])),
            "actor_sha256": _parameter_group_sha256(
                policy, partition["actor"]),
            "critic_parameter_count": int(sum(
                parameter.numel()
                for parameter in partition["critic"])),
            "critic_tensor_count": int(len(partition["critic"])),
            "critic_sha256": _parameter_group_sha256(
                policy, partition["critic"]),
        },
        "context": {
            "enabled": bool(context_enabled),
            "initializer": ASYMMETRIC_WORKER_CONTEXT_INITIALIZER,
            "excluded_observation_features": [
                int(index)
                for index in adapter.excluded_observation_features
            ],
            "actor_excluded_segments": [
                {
                    "name": segment.name,
                    "start": int(segment.start),
                    "stop": int(segment.stop),
                }
                for segment in DUAL_WORKER_LAYOUT.segments
                if segment.name
                in ASYMMETRIC_WORKER_ACTOR_EXCLUDED_SEGMENTS
            ],
            "parameter_count": int(sum(
                parameter.numel() for parameter in grouped)),
            "tensor_count": int(len(grouped)),
            "parameter_groups": group_evidence,
        },
        "probes": {
            "input_sha256": _runtime_tensor_bundle_sha256(probe),
            "p_skip_preoutput_invariant": p_skip_invariant,
            "p_skip_preoutput_max_abs_delta": float(
                p_skip_delta.max().item()),
            "p_skip_preoutput_zero_sha256":
                _runtime_tensor_bundle_sha256(p_skip_preoutput_zero),
            "p_skip_preoutput_one_sha256":
                _runtime_tensor_bundle_sha256(p_skip_preoutput_one),
            "p_skip_action_logits_invariant":
                p_skip_action_invariant,
            "p_skip_action_logits_max_abs_delta": float(
                p_skip_action_delta.max().item()),
            "p_skip_action_logits_zero_sha256":
                _runtime_tensor_bundle_sha256(p_skip_logits_zero),
            "p_skip_action_logits_one_sha256":
                _runtime_tensor_bundle_sha256(p_skip_logits_one),
            "forced_context_action_effect": bool(forced_changed > 0),
            "forced_context_action_logit_changed_elements":
                forced_changed,
            "forced_context_action_logit_max_abs_delta": float(
                forced_delta.max().item()),
            "nonzero_context_action_effect": bool(
                context_enabled and deployed_changed > 0),
            "nonzero_context_action_logit_changed_elements":
                deployed_changed,
            "nonzero_context_action_logit_max_abs_delta": float(
                deployed_delta.max().item()),
            "actor_scalar_preoutput_effects": {
                name: outputs[0]
                for name, outputs in focused_probe_outputs.items()
                if name in {"current_v4_base", "wrapper_scalars"}
            },
            "actor_focused_effects": {
                name: outputs[0]
                for name, outputs in focused_probe_outputs.items()
            },
        },
    }


class A12MixtureMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """Maskable policy whose dedicated scalar is an exact a12 mixture mass."""

    def __init__(self, *args, bc_aux_mixture_spec=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bc_aux_mixture_spec = bc_aux_mixture_spec

    def _mixture_distribution(
            self, raw_observations: th.Tensor, latent_pi: th.Tensor,
            action_masks):
        spec = self.bc_aux_mixture_spec
        raw_logits = self.action_net(latent_pi)
        if spec is None:
            distribution = self.action_dist.proba_distribution(
                action_logits=raw_logits)
        else:
            action = int(spec["action_index"])
            feature_indices = tuple(int(value) for value in
                                    spec["gate_feature_indices"])
            parameter_columns = tuple(int(value) for value in
                                      spec["gate_parameter_columns"])
            if (
                len(feature_indices) != len(parameter_columns)
                or not feature_indices
                or max(feature_indices) >= raw_observations.shape[1]
                or max(parameter_columns) >= self.action_net.weight.shape[1]
            ):
                raise ValueError("a12 contextual gate feature/parameter 映射异常")
            # The gate consumes four stable, directly observable combat
            # features.  Its coefficients live in otherwise-unused expanded
            # action-head columns, but do not consume the permanently-zero
            # latent marker neurons.  This gives PPO exactly five adapter
            # parameters and prevents a12 gradients from flowing into the
            # legacy actor representation.
            gate_features = raw_observations[:, feature_indices]
            gate_coefficients = self.action_net.weight[
                action, list(parameter_columns)]
            gate_logits = (
                gate_features @ gate_coefficients
                + self.action_net.bias[action]
            )
            logits = _a12_mixture_logits(
                raw_logits, raw_observations, action_masks, gate_logits,
                action=action,
                hp_low=float(spec["hp_low"]),
                hp_high=float(spec["hp_high"]),
                boundary_epsilon=float(spec["boundary_epsilon"]),
                probability_min=float(spec["probability_min"]),
                probability_max=float(spec["probability_max"]),
            )
            distribution = _ExactMixtureCategoricalDistribution(
                int(raw_logits.shape[-1])).proba_distribution(
                    action_logits=logits, exact_log_probs=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def _adapter_latents(self, obs: th.Tensor):
        # V28 actor/value were trained on protocol-v3 worker semantics.
        # Convert only their network input; eligibility above still reads the
        # original packed-belt/signed-latch observation.  Thus installing the
        # adapter is behavior-preserving on all fourteen non-a12 actions,
        # including post-drink states.
        network_obs = (
            _legacy_worker_observation_view(obs)
            if self.bc_aux_mixture_spec is not None else obs)
        features = self.extract_features(network_obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        pi_features, vf_features = features
        return (
            self.mlp_extractor.forward_actor(pi_features),
            self.mlp_extractor.forward_critic(vf_features),
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False,
                action_masks=None):
        latent_pi, latent_vf = self._adapter_latents(obs)
        values = self.value_net(latent_vf)
        distribution = self._mixture_distribution(
            obs, latent_pi, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def get_distribution(self, obs, action_masks=None):
        latent_pi, _ = self._adapter_latents(obs)
        return self._mixture_distribution(obs, latent_pi, action_masks)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor,
                         action_masks=None):
        latent_pi, latent_vf = self._adapter_latents(obs)
        distribution = self._mixture_distribution(
            obs, latent_pi, action_masks)
        return (
            self.value_net(latent_vf),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        # collect_rollouts uses this separate method for the rollout tail and
        # TimeLimit bootstrap.  It must share the exact decoded critic view
        # used by forward/evaluate_actions or negative latch rows would feed
        # two incompatible V(s) definitions into one GAE buffer.
        _, latent_vf = self._adapter_latents(obs)
        return self.value_net(latent_vf)


def _legacy_distillation_masks(
        action_masks: th.Tensor, *,
        excluded_actions: tuple[int, ...]
        = LEGACY_DISTILLATION_EXCLUDED_ACTIONS) -> th.Tensor:
    """Return the exact support on which the legacy KING may supervise.

    KING/V28 were trained before the current Worker-only a12/a14 semantics.
    Including either action in the distillation softmax invents a target for a
    logit the teacher never learned.  Exclude unsupported actions from *both*
    distributions and renormalize the remaining legal actions.  PPO still
    consumes the original rollout mask, so the student can learn those actions
    from on-policy reward without fabricated KING supervision.
    """
    if action_masks.ndim < 1 or action_masks.shape[-1] <= 0:
        raise ValueError("legacy distillation masks 必须有非空动作维")
    if (
        not isinstance(excluded_actions, tuple)
        or any(
            not isinstance(action, int)
            or isinstance(action, bool)
            or action < 0
            for action in excluded_actions
        )
        or len(set(excluded_actions)) != len(excluded_actions)
    ):
        raise ValueError("legacy distillation excluded_actions 非法")

    mask = action_masks.bool()
    in_range = tuple(
        action for action in excluded_actions
        if action < mask.shape[-1]
    )
    if not in_range:
        return mask

    mask = mask.clone()
    mask[..., list(in_range)] = False
    if not bool(mask.reshape(-1, mask.shape[-1]).any(dim=-1).all().item()):
        actions = ",".join(f"a{action}" for action in in_range)
        raise ValueError(
            f"排除 {actions} 后存在全 False 行，无法定义 KING 条件分布")
    return mask


def _masked_log_softmax_from_raw(
        raw_logits: th.Tensor, support: th.Tensor) -> th.Tensor:
    """Normalize exactly on ``support`` without pre-normalization leakage.

    ``MaskableCategorical`` first constructs a Categorical from all raw logits
    and only then applies its mask to the already-normalized logits.  That is
    mathematically equivalent in exact arithmetic, but not in float32 when an
    excluded action is very large: its common normalization constant can
    perturb the retained actions before the second normalization.  KING must
    be invariant to unsupported actions once they are outside its support, so
    mask the *raw* scores and normalize once.
    """
    mask = support.reshape(raw_logits.shape).bool()
    if not bool(mask.any(dim=-1).all().item()):
        raise ValueError("动作支持集存在全 False 行，无法归一化")
    masked = th.where(
        mask, raw_logits, th.full_like(raw_logits, HUGE_NEG))
    return F.log_softmax(masked, dim=-1)


# E3③ demo minibatch 专用 rng 流之固定偏移(播种规则同 E1③ 形制:自训练种子
# 以固定偏移确定性派生,零染训练 RNG;承 worker_env._P_SKIP_SEED_OFFSET=2**33+26
# 先例)。偏移值系施工裁量注记:取 2**34+26,与 p_skip 流族 (seed+rank)+2**33+26
# (rank<num_envs,远小于 2**33)及 [0,2**32) 训练种子域构造性不相交。
_BC_AUX_SEED_OFFSET = 2**34 + 26
# 必须与 train_ppo rev11 契约载荷逐值一致。常量留在算法模块，避免训练热路
# 反向 import CLI 模块；tests 以双边断言防漂移。
_BC_AUX_OBJECTIVE_REVISION = 11
_BC_AUX_UPDATE_EVERY = 1
_BC_AUX_POSITIVE_FRACTION = 0.25
_BC_AUX_POSITIVE_TARGET = 0.65
_BC_AUX_NEGATIVE_TARGET = 0.01
_BC_AUX_ANCHOR_KL_COEF = 0.25
_BC_AUX_POLICY_HEAD_KEYS = (
    "mlp_extractor.policy_net.0.weight",
    "mlp_extractor.policy_net.0.bias",
    "mlp_extractor.policy_net.2.weight",
    "mlp_extractor.policy_net.2.bias",
    "action_net.weight",
    "action_net.bias",
)
_BC_AUX_CIRCUIT_SCHEMA = "a12-onpolicy-contextual-mixture-adapter/1"
# G-CAL 固定 bank 漂移只负责截断破坏性更新；最终 recall/precision 等完整
# E5 行为门在原始 held-out 分布上裁决。
_BC_AUX_CALIB_TV_MAX = 0.15
_BC_AUX_CALIB_ARGMAX_DRIFT_MAX = 0.20
_BC_AUX_CALIB_FPR_MAX = 0.05
_BC_AUX_CALIB_PREDICTED_SHARE_MAX = 0.25
# 每个 rollout 更新后的比例监控；只裁“过度饮药/动作挤出”上界，不要求
# recall 下界（起点 recall 低正是辅助学习要解决的问题）。
_BC_AUX_ROLLOUT_FPR_MAX = 0.01
_BC_AUX_ROLLOUT_PREDICTED_SHARE_MAX = 0.20
_BC_AUX_ROLLOUT_HIGH_HP_FPR_MAX = 0.01
_BC_AUX_ROLLOUT_A13_SPILLOVER_MAX = 0.05
_BC_AUX_CIRCUIT_FPR_MAX = 0.002
_BC_AUX_CIRCUIT_SHARE_MAX = 0.02
_BC_AUX_CIRCUIT_HIGH_HP_FPR_MAX = 0.001
_BC_AUX_CIRCUIT_TV_MAX = 0.15
_BC_AUX_CIRCUIT_ARGMAX_DRIFT_MAX = 0.20
_BC_AUX_CIRCUIT_LEGAL_NEG_P12_MEAN_MAX = 1e-4
_BC_AUX_CIRCUIT_LEGAL_NEG_P12_MAX = 1e-3
_BC_AUX_CIRCUIT_PROBABILITY_MIN = 0.001
_BC_AUX_CIRCUIT_PROBABILITY_MAX = 0.95
_BC_AUX_CIRCUIT_GATE_PARAMETER_ABS_MAX = 8.0
_COPIED_SB3_CONTRIB_VERSION = "2.9.0"
if version("sb3-contrib") != _COPIED_SB3_CONTRIB_VERSION:
    raise RuntimeError(
        "leashed_ppo.py 只审计过 sb3-contrib "
        f"{_COPIED_SB3_CONTRIB_VERSION}；当前为 {version('sb3-contrib')}，"
        "请锁回该版本或重新完成 G-KL-B 上游等价审计")


def derive_bc_aux_rng(seed: int | None) -> np.random.Generator:
    """E3③:demo minibatch 专用 rng 流(固定偏移确定性派生,复现主张由此成立)。

    seed=None 时取随机源——镜像 worker_env rng_seed=None 分支形制;
    案腿一律显式 --seed,该分支仅覆盖非案手跑。
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed) + _BC_AUX_SEED_OFFSET)


def build_teacher(sd_path: str | pathlib.Path | bytes) -> th.nn.Module:
    """从 SB3 键名 state_dict 组装冻结教师(PiHead 298→64→64→15 同构)。"""
    source = io.BytesIO(sd_path) if isinstance(sd_path, bytes) else sd_path
    sd = th.load(source, map_location="cpu", weights_only=True)
    required = (
        "mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.policy_net.2.weight", "mlp_extractor.policy_net.2.bias",
        "action_net.weight", "action_net.bias",
    )
    if not isinstance(sd, dict):
        raise ValueError("教师文件必须是 policy state_dict")
    missing = [k for k in required if k not in sd]
    if missing:
        raise ValueError(f"教师 state_dict 缺键: {missing}")

    w0, b0 = sd[required[0]], sd[required[1]]
    w1, b1 = sd[required[2]], sd[required[3]]
    wa, ba = sd[required[4]], sd[required[5]]
    if not all(isinstance(t, th.Tensor) for t in (w0, b0, w1, b1, wa, ba)):
        raise ValueError("教师 state_dict 的策略头值必须都是 Tensor")
    if (w0.ndim != 2 or w1.ndim != 2 or wa.ndim != 2
            or b0.shape != (w0.shape[0],) or b1.shape != (w1.shape[0],)
            or ba.shape != (wa.shape[0],) or w1.shape[1] != w0.shape[0]
            or wa.shape[1] != w1.shape[0]):
        raise ValueError("教师 state_dict 的 MLP 形状不自洽")
    if not all(th.isfinite(t).all().item() for t in (w0, b0, w1, b1, wa, ba)):
        raise ValueError("教师 state_dict 含 NaN/Inf")
    net = th.nn.Sequential(
        th.nn.Linear(w0.shape[1], w0.shape[0]), th.nn.Tanh(),
        th.nn.Linear(w1.shape[1], w1.shape[0]), th.nn.Tanh(),
        th.nn.Linear(wa.shape[1], wa.shape[0]))
    with th.no_grad():
        net[0].weight.copy_(w0)
        net[0].bias.copy_(b0)
        net[2].weight.copy_(w1)
        net[2].bias.copy_(b1)
        net[4].weight.copy_(wa)
        net[4].bias.copy_(ba)
    net.eval()
    net.requires_grad_(False)
    return net


class LeashedMaskablePPO(MaskablePPO):
    def __init__(self, *args, distill_beta: float = 0.0, teacher_path: str | None = None,
                 teacher_sha256: str | None = None,
                 calib_probes: list | None = None, calib_out: str | None = None,
                 bc_aux_lambda: float = 0.0,
                 distill_anneal_actor_rollouts: int = 0, **kwargs):
        self.distill_beta = float(distill_beta)
        if not np.isfinite(self.distill_beta) or self.distill_beta < 0:
            raise ValueError(f"distill_beta 必须是有限非负数,实得 {distill_beta!r}")
        if (
            not isinstance(distill_anneal_actor_rollouts, int)
            or isinstance(distill_anneal_actor_rollouts, bool)
            or (
                distill_anneal_actor_rollouts != 0
                and distill_anneal_actor_rollouts < 2
            )
        ):
            raise ValueError(
                "distill_anneal_actor_rollouts 必须为 0 或 >=2 的整数")
        self.distill_anneal_actor_rollouts = int(
            distill_anneal_actor_rollouts)
        self._distill_actor_rollouts_completed = 0
        self._last_effective_distill_beta = self.distill_beta
        # E3 ④乙:辅助示范 CE 系数(镜像 β 形制;示范池经 mount_bc_aux_demos 挂载)
        self.bc_aux_lambda = float(bc_aux_lambda)
        if not np.isfinite(self.bc_aux_lambda) or self.bc_aux_lambda < 0:
            raise ValueError(f"bc_aux_lambda 必须是有限非负数,实得 {bc_aux_lambda!r}")
        self._bc_aux_obs = None
        self._bc_aux_actions = None
        self._bc_aux_masks = None
        self._bc_aux_rng = None
        self._bc_aux_positive = None
        self._bc_aux_negative = None
        self._bc_aux_anchor_probs = None
        self._bc_aux_validation_obs = None
        self._bc_aux_validation_actions = None
        self._bc_aux_validation_masks = None
        self._bc_aux_validation_anchor_probs = None
        self._bc_aux_permutations = {}
        self._bc_aux_cursors = {}
        self._bc_aux_train_calls = 0
        self._last_bc_aux_parts = None
        self._last_bc_aux_ce = None
        self.bc_aux_monitor_out = None
        # rev9 structural contextual mixture metadata.  Tensor surgery is
        # performed by train_ppo before any environment is created; keeping
        # the small JSON-compatible spec on the algorithm makes save/load and
        # every optimizer step fail-closed.
        self._bc_aux_circuit_spec = None
        self._bc_aux_eligible_states = 0
        self._bc_aux_requested_a12 = 0
        self._bc_aux_sampled_a12 = 0
        self._bc_aux_rejected_a12 = 0
        self._bc_aux_unexpected_sampled_a12 = 0
        self._bc_aux_expected_a12_mass = 0.0
        self._bc_aux_pending_requested_a12 = 0
        self._bc_aux_pending_executed_a12 = 0
        self._bc_aux_pending_action_receipts = []
        # Persisted proof that the most recently collected full rollout was
        # actually consumed by at least one PPO optimizer step.  A full buffer
        # alone only proves collection, not learning.
        self._ppo_optimizer_steps_completed = 0
        self._last_completed_ppo_rollout_steps = None
        self.gradient_clip_mode = GRADIENT_CLIP_GLOBAL
        self._critic_warmup_start_timesteps = None
        self._critic_warmup_until_timesteps = None
        self._critic_warmup_expected_rollouts = 0
        self._critic_warmup_rollouts_completed = 0
        self._critic_warmup_optimizer_steps_completed = 0
        self._critic_warmup_completed = False
        self._critic_warmup_actor_sha256 = None
        self._actor_optimizer_steps_completed = 0
        # Formal R7 learning-signal evidence.  ``transition_reward`` receipts
        # are collected in exact VecEnv/buffer order, then every joint rollout
        # records (a) a reward-centered actor gradient that cannot be produced
        # by critic/entropy/BC terms and (b) the pure clipped-PPO actor gradient
        # measured immediately before each real optimizer step.
        self._worker_onpolicy_pg_audit_required = False
        self._worker_onpolicy_pg_pending_receipts = []
        self._worker_onpolicy_pg_joint_rollouts = 0
        self._worker_onpolicy_pg_qualifying_rollouts = 0
        self._worker_onpolicy_pg_rollout_receipts = []
        self._worker_onpolicy_pg_collection_actor_sha256 = None
        # 不列入 _excluded_save_params：首次 aux 腿的策略头根锚必须随
        # checkpoint 跨 continuation 持久化，防逐腿小漂移累积成大退化。
        self.bc_aux_root_anchor_sd = None
        self.teacher_path = teacher_path
        self.teacher_sha256 = teacher_sha256
        if teacher_path:
            actual = hashlib.sha256(pathlib.Path(teacher_path).read_bytes()).hexdigest()
            if teacher_sha256 is not None and teacher_sha256 != actual:
                raise ValueError(
                    f"教师 SHA 不匹配: {actual} != {teacher_sha256}")
            self.teacher_sha256 = actual
        self.calib_probes = list(calib_probes or [])
        self.calib_out = calib_out
        self._calib_done = set()
        self._calib_tripped = False
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        # load() 流程:先 __dict__.update(data) 恢复 teacher_path,再调本方法——
        # fresh 与 resume 两条路径此处都成立(预注册 D4,审计 BLOCKER 4)
        self.teacher = None
        if getattr(self, "teacher_path", None):
            teacher_path = pathlib.Path(self.teacher_path)
            try:
                payload = teacher_path.read_bytes()
                actual_sha = hashlib.sha256(payload).hexdigest()
            except OSError as exc:
                raise ValueError(f"教师文件缺失/不可读: {teacher_path}") from exc
            expected_sha = getattr(self, "teacher_sha256", None)
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise ValueError(
                    "检查点缺少 teacher_sha256；旧检查点续训须由入口提供可信教师绑定")
            if actual_sha != expected_sha:
                raise ValueError(f"教师 SHA 漂移: {actual_sha} != {expected_sha}")
            # Hash and deserialize the exact same immutable payload.  Reopening
            # teacher_path here would permit an atomic replacement between the
            # integrity check and torch.load().
            self.teacher = build_teacher(payload).to(self.device)
            obs_shape = getattr(self.observation_space, "shape", None)
            obs_dim = int(np.prod(obs_shape)) if obs_shape else None
            n_actions = getattr(self.action_space, "n", None)
            asymmetric_teacher = (
                isinstance(
                    self.policy,
                    AsymmetricWorkerMaskableActorCriticPolicy)
                and obs_dim == ASYMMETRIC_WORKER_OBSERVATION_DIM
                and self.teacher[0].in_features
                == ASYMMETRIC_WORKER_LEGACY_DIM
            )
            if ((obs_dim != self.teacher[0].in_features
                 and not asymmetric_teacher)
                    or n_actions != self.teacher[-1].out_features):
                raise ValueError(
                    "教师与训练环境形状不匹配: "
                    f"teacher={self.teacher[0].in_features}→{self.teacher[-1].out_features}, "
                    f"env={obs_dim}→{n_actions}")
        # _excluded_save_params 成员在 load 后不存在,兜底重建
        # (E3:bc_aux_lambda 系持久化标量,旧 zip 无之亦兜底 0.0;示范池/专用流
        #  不入 zip,resume 由 train_ppo 案内重挂,λ 由 CLI 显式覆盖承 β 先例)
        for attr, dv in (("_calib_done", set()), ("_calib_tripped", False),
                         ("_last_distill_ce", None), ("_last_diverge", None),
                         ("_last_teacher_entropy", None),
                         ("_last_distill_kl", None),
                         ("_last_distill_tv", None),
                         ("distill_anneal_actor_rollouts", 0),
                         ("_distill_actor_rollouts_completed", 0),
                         ("_last_effective_distill_beta",
                          float(getattr(self, "distill_beta", 0.0))),
                         ("bc_aux_lambda", 0.0), ("_bc_aux_obs", None),
                         ("_bc_aux_actions", None), ("_bc_aux_masks", None),
                         ("_bc_aux_rng", None), ("_bc_aux_positive", None),
                         ("_bc_aux_negative", None),
                         ("_bc_aux_anchor_probs", None),
                         ("_bc_aux_validation_obs", None),
                         ("_bc_aux_validation_actions", None),
                         ("_bc_aux_validation_masks", None),
                         ("_bc_aux_validation_anchor_probs", None),
                         ("_bc_aux_permutations", {}),
                         ("_bc_aux_cursors", {}),
                         ("_bc_aux_train_calls", 0),
                         ("_last_bc_aux_parts", None),
                         ("_last_bc_aux_ce", None),
                         ("bc_aux_monitor_out", None),
                         ("_bc_aux_circuit_spec", None),
                         ("_bc_aux_eligible_states", 0),
                         ("_bc_aux_requested_a12", 0),
                         ("_bc_aux_sampled_a12", 0),
                         ("_bc_aux_rejected_a12", 0),
                         ("_bc_aux_unexpected_sampled_a12", 0),
                         ("_bc_aux_expected_a12_mass", 0.0),
                         ("_bc_aux_pending_requested_a12", 0),
                         ("_bc_aux_pending_executed_a12", 0),
                         ("_bc_aux_pending_action_receipts", []),
                         ("_ppo_optimizer_steps_completed", 0),
                         ("_last_completed_ppo_rollout_steps", None),
                         ("gradient_clip_mode", GRADIENT_CLIP_GLOBAL),
                         ("_critic_warmup_start_timesteps", None),
                         ("_critic_warmup_until_timesteps", None),
                         ("_critic_warmup_expected_rollouts", 0),
                         ("_critic_warmup_rollouts_completed", 0),
                         ("_critic_warmup_optimizer_steps_completed", 0),
                         ("_critic_warmup_completed", False),
                         ("_critic_warmup_actor_sha256", None),
                         ("_actor_optimizer_steps_completed", 0),
                         ("_worker_onpolicy_pg_audit_required", False),
                         ("_worker_onpolicy_pg_pending_receipts", []),
                         ("_worker_onpolicy_pg_joint_rollouts", 0),
                         ("_worker_onpolicy_pg_qualifying_rollouts", 0),
                         ("_worker_onpolicy_pg_rollout_receipts", []),
                         ("_worker_onpolicy_pg_collection_actor_sha256",
                          None),
                         ("bc_aux_root_anchor_sd", None)):
            if not hasattr(self, attr):
                setattr(self, attr, dv)

    def _effective_distill_beta(self, *, actor_frozen: bool) -> float:
        """Return a persisted actor-rollout schedule, excluding warmup."""
        beta = float(self.distill_beta)
        horizon = int(self.distill_anneal_actor_rollouts)
        completed = int(self._distill_actor_rollouts_completed)
        if (
            not math.isfinite(beta)
            or beta < 0.0
            or horizon < 0
            or completed < 0
        ):
            raise RuntimeError("distillation anneal 状态非法")
        if beta == 0.0 or horizon == 0 or actor_frozen:
            return beta
        if horizon < 2:
            raise RuntimeError(
                "distillation anneal horizon 必须 >=2")
        progress = min(completed, horizon - 1) / float(horizon - 1)
        return beta * (1.0 - progress)

    def configure_critic_migration(
            self, *, gradient_clip_mode: str,
            critic_warmup_steps: int) -> dict:
        """Install the persistent optimizer-update contract for a fresh critic.

        The caller must reset the critic and optimizer before invoking this
        method.  This method deliberately does not initialize either one; it
        closes the algorithm-side parameter partition and warmup accounting
        that ``train_ppo`` can bind into its migration receipt.
        """
        if gradient_clip_mode not in {
                GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
                GRADIENT_CLIP_ROOT_CONTEXT_CRITIC_V2,
        }:
            raise ValueError(
                "critic migration 必须使用分组梯度裁剪")
        if (not isinstance(critic_warmup_steps, int)
                or isinstance(critic_warmup_steps, bool)
                or critic_warmup_steps <= 0):
            raise ValueError("critic_warmup_steps 必须是正整数")
        quantum = int(self.n_steps) * int(self.n_envs)
        if quantum <= 0 or critic_warmup_steps % quantum != 0:
            raise ValueError(
                "critic_warmup_steps 必须整除完整 rollout 量子 "
                f"{quantum}")
        if getattr(self, "_bc_aux_circuit_spec", None) is not None \
                or float(getattr(self, "bc_aux_lambda", 0.0)) != 0.0:
            raise RuntimeError(
                "critic-only warmup 禁止并行 actor bc_aux 更新")
        if self._critic_warmup_start_timesteps is not None:
            raise RuntimeError("critic migration 已配置，禁止重置 warmup 边界")
        if self.policy.optimizer.state:
            raise RuntimeError(
                "critic migration 配置前必须重建 optimizer 并清空全部 state")
        if not isinstance(self.rollout_buffer, _AuditedMaskableRolloutBuffer):
            if (
                not isinstance(self.rollout_buffer, MaskableRolloutBuffer)
                or int(getattr(self.rollout_buffer, "pos", -1)) != 0
                or bool(getattr(self.rollout_buffer, "full", False))
                or isinstance(self.observation_space, spaces.Dict)
            ):
                raise RuntimeError(
                    "formal PG audit 无法从非空/非扁平 Maskable buffer "
                    "安全迁移")
            self.rollout_buffer_class = _AuditedMaskableRolloutBuffer
            self.rollout_buffer = _AuditedMaskableRolloutBuffer(
                self.n_steps,
                self.observation_space,
                self.action_space,
                self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
                **self.rollout_buffer_kwargs,
            )
        partition = strict_actor_critic_parameter_partition(
            self.policy, optimizer=self.policy.optimizer)
        extractor = self.policy.mlp_extractor
        if (isinstance(
                self.policy,
                AsymmetricWorkerMaskableActorCriticPolicy)
                and extractor.actor_context_enabled):
            raise RuntimeError(
                "critic migration 配置前 asymmetric actor context 必须关闭")
        start = int(self.num_timesteps)
        self.gradient_clip_mode = gradient_clip_mode
        self._critic_warmup_start_timesteps = start
        self._critic_warmup_until_timesteps = (
            start + int(critic_warmup_steps))
        self._critic_warmup_expected_rollouts = (
            int(critic_warmup_steps) // quantum)
        self._critic_warmup_rollouts_completed = 0
        self._critic_warmup_optimizer_steps_completed = 0
        self._critic_warmup_completed = False
        self._critic_warmup_actor_sha256 = _parameter_group_sha256(
            self.policy, partition["actor"])
        self._actor_optimizer_steps_completed = 0
        self._worker_onpolicy_pg_audit_required = True
        self._worker_onpolicy_pg_pending_receipts.clear()
        self._worker_onpolicy_pg_joint_rollouts = 0
        self._worker_onpolicy_pg_qualifying_rollouts = 0
        self._worker_onpolicy_pg_rollout_receipts.clear()
        self._worker_onpolicy_pg_collection_actor_sha256 = None
        return {
            "gradient_clip_mode": self.gradient_clip_mode,
            "warmup_start_timesteps":
                self._critic_warmup_start_timesteps,
            "warmup_until_timesteps":
                self._critic_warmup_until_timesteps,
            "warmup_steps": int(critic_warmup_steps),
            "warmup_rollout_quantum": quantum,
            "warmup_expected_rollouts":
                self._critic_warmup_expected_rollouts,
            "actor_sha256": self._critic_warmup_actor_sha256,
            "actor_parameter_tensors": len(partition["actor"]),
            "critic_parameter_tensors": len(partition["critic"]),
            "worker_onpolicy_pg_audit_schema":
                WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
        }

    def _assert_critic_migration_contract(self) -> dict:
        mode = getattr(self, "gradient_clip_mode", GRADIENT_CLIP_GLOBAL)
        if mode not in _GRADIENT_CLIP_MODES:
            raise RuntimeError(f"未知 gradient_clip_mode:{mode!r}")
        partition = strict_actor_critic_parameter_partition(
            self.policy, optimizer=self.policy.optimizer)
        start = self._critic_warmup_start_timesteps
        until = self._critic_warmup_until_timesteps
        if start is None:
            fields = (
                until,
                self._critic_warmup_expected_rollouts,
                self._critic_warmup_rollouts_completed,
                self._critic_warmup_optimizer_steps_completed,
                self._critic_warmup_completed,
                self._critic_warmup_actor_sha256,
            )
            if fields != (None, 0, 0, 0, False, None):
                raise RuntimeError("未配置 critic migration 却携残留 warmup 状态")
            if (
                self._worker_onpolicy_pg_audit_required is not False
                or self._worker_onpolicy_pg_pending_receipts
                or self._worker_onpolicy_pg_joint_rollouts != 0
                or self._worker_onpolicy_pg_qualifying_rollouts != 0
                or self._worker_onpolicy_pg_rollout_receipts
                or self._worker_onpolicy_pg_collection_actor_sha256
                is not None
            ):
                raise RuntimeError(
                    "未配置 critic migration 却携 formal PG audit 状态")
            return partition
        if mode not in {
                GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
                GRADIENT_CLIP_ROOT_CONTEXT_CRITIC_V2,
        }:
            raise RuntimeError("critic migration 的分组裁剪模式漂移")
        if (not isinstance(start, int) or not isinstance(until, int)
                or until <= start):
            raise RuntimeError("critic warmup 绝对步边界非法")
        quantum = int(self.n_steps) * int(self.n_envs)
        expected = (until - start) // quantum
        integer_fields = (
            self._critic_warmup_expected_rollouts,
            self._critic_warmup_rollouts_completed,
            self._critic_warmup_optimizer_steps_completed,
            self._actor_optimizer_steps_completed,
        )
        if ((until - start) % quantum != 0
                or any(not isinstance(value, int)
                       or isinstance(value, bool)
                       for value in integer_fields)
                or self._critic_warmup_expected_rollouts != expected
                or not 0 <= self._critic_warmup_rollouts_completed <= expected
                or self._critic_warmup_optimizer_steps_completed < 0
                or self._actor_optimizer_steps_completed < 0
                or not isinstance(self._critic_warmup_completed, bool)
                or not isinstance(self._critic_warmup_actor_sha256, str)
                or len(self._critic_warmup_actor_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in self._critic_warmup_actor_sha256)):
            raise RuntimeError("critic warmup 持久状态非法")
        receipts = self._worker_onpolicy_pg_rollout_receipts
        joint_rollouts = self._worker_onpolicy_pg_joint_rollouts
        qualifying_rollouts = self._worker_onpolicy_pg_qualifying_rollouts
        if (
            self._worker_onpolicy_pg_audit_required is not True
            or not isinstance(
                self.rollout_buffer, _AuditedMaskableRolloutBuffer)
            or not isinstance(self._worker_onpolicy_pg_pending_receipts, list)
            or not isinstance(receipts, list)
            or type(joint_rollouts) is not int
            or type(qualifying_rollouts) is not int
            or joint_rollouts != len(receipts)
            or not 0 <= qualifying_rollouts <= joint_rollouts
            or qualifying_rollouts
            != sum(
                isinstance(receipt, dict)
                and receipt.get("qualifies") is True
                for receipt in receipts)
            or not all(validate_worker_onpolicy_pg_receipt(
                receipt,
                expected_samples=int(self.n_steps) * int(self.n_envs),
            ) for receipt in receipts)
            or (
                self._worker_onpolicy_pg_collection_actor_sha256
                is not None
                and (
                    not isinstance(
                        self._worker_onpolicy_pg_collection_actor_sha256,
                        str,
                    )
                    or len(
                        self._worker_onpolicy_pg_collection_actor_sha256
                    ) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in
                        self._worker_onpolicy_pg_collection_actor_sha256
                    )
                )
            )
        ):
            raise RuntimeError("formal Worker on-policy PG 持久状态非法")
        return partition

    def _prepare_main_ppo_rollout(self) -> bool:
        partition = self._assert_critic_migration_contract()
        start = self._critic_warmup_start_timesteps
        if start is None:
            return False
        now = int(self.num_timesteps)
        until = int(self._critic_warmup_until_timesteps)
        completed = int(self._critic_warmup_rollouts_completed)
        expected = int(self._critic_warmup_expected_rollouts)
        quantum = int(self.n_steps) * int(self.n_envs)
        if now <= until:
            if (isinstance(
                    self.policy,
                    AsymmetricWorkerMaskableActorCriticPolicy)
                    and self.policy.mlp_extractor.actor_context_enabled):
                raise RuntimeError(
                    "critic warmup 期间 actor context 提前启用")
            required_endpoint = start + (completed + 1) * quantum
            if completed >= expected or now != required_endpoint:
                raise RuntimeError(
                    "critic warmup rollout 端点跳跃/重复:"
                    f"now={now},expected={required_endpoint}")
            if self._critic_warmup_completed:
                raise RuntimeError("critic warmup 提前标记 complete")
            return True
        if (not self._critic_warmup_completed
                or completed != expected
                or self._critic_warmup_optimizer_steps_completed <= 0):
            raise RuntimeError("actor 解冻前 critic warmup 未完整闭合")
        if (isinstance(
                self.policy,
                AsymmetricWorkerMaskableActorCriticPolicy)
                and not self.policy.mlp_extractor.actor_context_enabled):
            raise RuntimeError(
                "critic warmup 完成后 actor context 未原子启用")
        # Verify the frozen anchor exactly once at the warmup→joint-training
        # boundary.  After the first legitimate actor optimizer step the
        # current actor must differ, so comparing every later rollout to the
        # warmup anchor would falsely abort the second joint rollout.
        if self._actor_optimizer_steps_completed == 0:
            current = _parameter_group_sha256(
                self.policy, partition["actor"])
            if current != self._critic_warmup_actor_sha256:
                raise RuntimeError("critic warmup 期间 actor 权重发生漂移")
        return False

    def _worker_pg_gae_closure(self) -> dict:
        """Recompute GAE from the sealed pre-callback rollout arrays."""
        buffer = self.rollout_buffer
        if (
            not isinstance(buffer, _AuditedMaskableRolloutBuffer)
            or buffer.generator_ready
            or not isinstance(buffer._formal_gae_snapshot, dict)
        ):
            raise RuntimeError(
                "formal PG GAE 校验要求未展开的 audited rollout buffer")
        snapshot = buffer._formal_gae_snapshot
        expected_keys = {
            "last_values",
            "dones",
            *_AuditedMaskableRolloutBuffer._AUDIT_ARRAY_NAMES,
        }
        if set(snapshot) != expected_keys:
            raise RuntimeError("formal PG GAE 密封快照 schema 漂移")
        for name in _AuditedMaskableRolloutBuffer._AUDIT_ARRAY_NAMES:
            current = np.asarray(getattr(buffer, name))
            sealed = np.asarray(snapshot[name])
            if current.shape != sealed.shape or not np.array_equal(
                    current, sealed):
                raise RuntimeError(
                    f"formal PG rollout buffer.{name} 在 GAE 后被改写")

        rewards = np.asarray(snapshot["rewards"])
        values = np.asarray(snapshot["values"])
        episode_starts = np.asarray(snapshot["episode_starts"])
        sealed_advantages = np.asarray(snapshot["advantages"])
        sealed_returns = np.asarray(snapshot["returns"])
        last_values = np.asarray(snapshot["last_values"])
        dones = np.asarray(snapshot["dones"])
        expected_shape = (int(self.n_steps), int(self.n_envs))
        if (
            rewards.shape != expected_shape
            or values.shape != expected_shape
            or episode_starts.shape != expected_shape
            or sealed_advantages.shape != expected_shape
            or sealed_returns.shape != expected_shape
            or last_values.shape != (int(self.n_envs),)
            or dones.shape != (int(self.n_envs),)
            or not all(np.isfinite(array).all() for array in (
                rewards,
                values,
                episode_starts,
                sealed_advantages,
                sealed_returns,
                last_values,
            ))
        ):
            raise RuntimeError("formal PG GAE 密封数组形状/有限性异常")

        recomputed = np.zeros_like(sealed_advantages)
        last_gae_lam: np.ndarray | int = 0
        for step in reversed(range(int(self.n_steps))):
            if step == int(self.n_steps) - 1:
                next_non_terminal = (
                    1.0 - dones.astype(np.float32))
                next_values = last_values
            else:
                next_non_terminal = (
                    1.0 - episode_starts[step + 1])
                next_values = values[step + 1]
            delta = (
                rewards[step]
                + self.gamma * next_values * next_non_terminal
                - values[step]
            )
            last_gae_lam = (
                delta
                + self.gamma
                * self.gae_lambda
                * next_non_terminal
                * last_gae_lam
            )
            recomputed[step] = last_gae_lam
        recomputed_returns = recomputed + values
        advantage_delta = np.abs(
            recomputed.astype(np.float64)
            - sealed_advantages.astype(np.float64))
        return_delta = np.abs(
            recomputed_returns.astype(np.float64)
            - sealed_returns.astype(np.float64))
        advantage_max = float(advantage_delta.max(initial=0.0))
        return_max = float(return_delta.max(initial=0.0))
        if (
            not np.array_equal(recomputed, sealed_advantages)
            or not np.array_equal(recomputed_returns, sealed_returns)
        ):
            raise RuntimeError(
                "formal PG GAE/return 独立递推不闭合:"
                f"adv_max={advantage_max},return_max={return_max}")
        return {
            "advantage_max_abs_delta": advantage_max,
            "return_max_abs_delta": return_max,
        }

    def _begin_worker_onpolicy_pg_rollout(
            self, *, actor_frozen: bool) -> dict | None:
        """Consume exact Worker receipts and prime one joint-rollout audit."""
        if not self._worker_onpolicy_pg_audit_required:
            if self._worker_onpolicy_pg_pending_receipts:
                raise RuntimeError(
                    "formal PG audit 未启用却收到 reward receipts")
            return None
        collection_actor_sha256 = (
            self._worker_onpolicy_pg_collection_actor_sha256)
        current_actor_sha256 = actor_parameter_sha256(
            self.policy, optimizer=self.policy.optimizer)
        if (
            not isinstance(collection_actor_sha256, str)
            or current_actor_sha256 != collection_actor_sha256
        ):
            raise RuntimeError(
                "formal PG train() actor 与 rollout collection actor 不同")
        gae_closure = self._worker_pg_gae_closure()
        self._worker_onpolicy_pg_collection_actor_sha256 = None
        actions = th.as_tensor(
            self.rollout_buffer.actions,
            device=self.device,
        ).long().reshape(-1)
        receipts = self._worker_onpolicy_pg_pending_receipts
        if not isinstance(receipts, list) or len(receipts) != len(actions):
            raise RuntimeError(
                "formal PG rollout buffer 与 reward receipts 长度不闭合:"
                f"buffer={len(actions)},receipts="
                f"{len(receipts) if isinstance(receipts, list) else 'invalid'}")
        requested = th.as_tensor(
            [row["requested_action"] for row in receipts],
            dtype=actions.dtype,
            device=actions.device,
        )
        if not th.equal(requested, actions):
            mismatch = int(th.nonzero(
                requested != actions, as_tuple=False)[0].item())
            raise RuntimeError(
                "formal PG rollout action/reward receipt 错位:"
                f"index={mismatch},buffer={int(actions[mismatch])},"
                f"receipt={int(requested[mismatch])}")
        rewards_np = np.asarray(
            [row["transition_reward"] for row in receipts],
            dtype=np.float64,
        )
        if rewards_np.ndim != 1 or not np.isfinite(rewards_np).all():
            raise RuntimeError("formal PG transition_reward 序列非法")
        executed_np = np.asarray([
            -1 if row["executed_action"] is None
            else row["executed_action"]
            for row in receipts
        ], dtype=np.int64)
        combat_effect_np = np.asarray(
            [row["combat_effect"] for row in receipts],
            dtype=np.bool_,
        )
        timeout_np = np.asarray(
            [row["worker_no_progress_timeout"] for row in receipts],
            dtype=np.bool_,
        )
        timeout_base_np = np.asarray(
            [
                row["no_progress_timeout_base_failure_reward"]
                for row in receipts
            ],
            dtype=np.float64,
        )
        timeout_additional_np = np.asarray(
            [
                row["no_progress_timeout_additional_failure_reward"]
                for row in receipts
            ],
            dtype=np.float64,
        )
        timeout_total_np = np.asarray(
            [
                row["no_progress_timeout_failure_reward"]
                for row in receipts
            ],
            dtype=np.float64,
        )
        if (
            executed_np.shape != rewards_np.shape
            or combat_effect_np.shape != rewards_np.shape
            or timeout_np.shape != rewards_np.shape
            or timeout_base_np.shape != rewards_np.shape
            or timeout_additional_np.shape != rewards_np.shape
            or timeout_total_np.shape != rewards_np.shape
            or not np.isfinite(timeout_base_np).all()
            or not np.isfinite(timeout_additional_np).all()
            or not np.isfinite(timeout_total_np).all()
            or np.any((executed_np < -1) | (executed_np >= 15))
            or np.any(combat_effect_np & (executed_np != 9))
        ):
            raise RuntimeError(
                "formal PG executed/combat receipt 序列非法")
        buffer_rewards = np.asarray(
            self.rollout_buffer.rewards, dtype=np.float32).reshape(-1)
        expected_buffer_rewards = np.asarray(
            [row["expected_buffer_reward"] for row in receipts],
            dtype=np.float32,
        )
        if (
            len(buffer_rewards) != len(actions)
            or not np.isfinite(buffer_rewards).all()
            or not np.isfinite(expected_buffer_rewards).all()
        ):
            raise RuntimeError(
                "formal PG rollout buffer reward 形状/有限性异常")
        if not np.array_equal(buffer_rewards, expected_buffer_rewards):
            mismatch = int(np.flatnonzero(
                buffer_rewards != expected_buffer_rewards)[0])
            raise RuntimeError(
                "formal PG rollout buffer reward 与 "
                "transition_reward/TimeLimit bootstrap 不闭合:"
                f"index={mismatch},buffer="
                f"{float(buffer_rewards[mismatch])!r},expected="
                f"{float(expected_buffer_rewards[mismatch])!r},raw="
                f"{float(rewards_np[mismatch])!r},bootstrap="
                f"{receipts[mismatch]['time_limit_bootstrap_delta']!r}")
        observations_for_log_prob = th.as_tensor(
            self.rollout_buffer.observations,
            device=self.device,
        )
        masks_for_log_prob = th.as_tensor(
            self.rollout_buffer.action_masks,
            device=self.device,
        ).bool()
        observations_for_log_prob = observations_for_log_prob.reshape(
            -1, observations_for_log_prob.shape[-1])
        masks_for_log_prob = masks_for_log_prob.reshape(
            -1, masks_for_log_prob.shape[-1])
        sealed_log_probs = np.asarray(
            self.rollout_buffer._formal_gae_snapshot["log_probs"],
            dtype=np.float64,
        ).reshape(-1)
        with th.no_grad():
            recomputed_log_probs = (
                self.policy.get_distribution(
                    observations_for_log_prob,
                    action_masks=masks_for_log_prob,
                )
                .log_prob(actions)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
        if (
            recomputed_log_probs.shape != sealed_log_probs.shape
            or not np.isfinite(recomputed_log_probs).all()
            or not np.isfinite(sealed_log_probs).all()
        ):
            raise RuntimeError(
                "formal PG collection log-prob 形状/有限性异常")
        log_prob_delta = np.abs(
            recomputed_log_probs - sealed_log_probs)
        log_prob_max_abs_delta = float(
            log_prob_delta.max(initial=0.0))
        if not np.allclose(
            recomputed_log_probs,
            sealed_log_probs,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise RuntimeError(
                "formal PG 当前 actor/log-prob 与 collection 回执不闭合:"
                f"max_abs_delta={log_prob_max_abs_delta}")
        # Every collected rollout, including critic-only warmup, must consume
        # exactly one receipt stream.  Only joint rollouts generate actor-PG
        # evidence.
        self._worker_onpolicy_pg_pending_receipts.clear()
        if actor_frozen:
            return None

        observations = th.as_tensor(
            self.rollout_buffer.observations,
            device=self.device,
        )
        masks = th.as_tensor(
            self.rollout_buffer.action_masks,
            device=self.device,
        ).bool()
        observations = observations.reshape(
            -1, observations.shape[-1])
        masks = masks.reshape(-1, masks.shape[-1])
        if len(observations) != len(actions) or len(masks) != len(actions):
            raise RuntimeError("formal PG rollout tensor 行数不闭合")
        advantages_np = np.asarray(
            self.rollout_buffer.advantages,
            dtype=np.float64,
        ).reshape(-1)
        if (
            len(advantages_np) != len(actions)
            or not np.isfinite(advantages_np).all()
        ):
            raise RuntimeError("formal PG GAE advantage 序列非法")

        reward_mean = float(rewards_np.mean())
        reward_centered_np = rewards_np - reward_mean
        reward_variance = float(np.mean(
            np.square(reward_centered_np)))
        reward_centered_l2 = float(np.linalg.norm(
            reward_centered_np))
        combat_rewards_np = np.where(
            combat_effect_np, rewards_np, 0.0)
        combat_reward_centered_np = (
            combat_rewards_np - float(combat_rewards_np.mean()))
        combat_reward_centered_l2 = float(np.linalg.norm(
            combat_reward_centered_np))
        advantage_mean = float(advantages_np.mean())
        advantage_variance = float(np.mean(np.square(
            advantages_np - advantage_mean)))
        numeric = (
            reward_mean,
            reward_variance,
            reward_centered_l2,
            combat_reward_centered_l2,
            advantage_mean,
            advantage_variance,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise RuntimeError("formal PG rollout statistics 非有限")

        # This audit-only REINFORCE term uses the exact Worker transition
        # reward, centered by a scalar baseline.  It is evaluated before any
        # update on the rollout's collection policy and never reaches
        # ``optimizer.step``.  A critic/entropy/distillation signal therefore
        # cannot create this norm when rewards are constant.
        distribution = self.policy.get_distribution(
            observations, action_masks=masks)
        reward_log_prob = distribution.log_prob(actions)
        centered_rewards = th.as_tensor(
            reward_centered_np,
            dtype=reward_log_prob.dtype,
            device=reward_log_prob.device,
        )
        reward_pg_loss = -(
            centered_rewards.detach() * reward_log_prob
        ).mean()
        combat_centered_rewards = th.as_tensor(
            combat_reward_centered_np,
            dtype=reward_log_prob.dtype,
            device=reward_log_prob.device,
        )
        combat_reward_pg_loss = -(
            combat_centered_rewards.detach() * reward_log_prob
        ).mean()
        actor_parameters = strict_actor_critic_parameter_partition(
            self.policy,
            optimizer=self.policy.optimizer,
        )["actor"]
        if not isinstance(
                self.policy,
                AsymmetricWorkerMaskableActorCriticPolicy):
            raise RuntimeError(
                "formal Worker PG context receipt 要求 asymmetric policy")
        adapter = self.policy.mlp_extractor.context_adapter
        context_groups = adapter.named_parameter_groups()
        context_parameters = tuple(adapter.parameters())
        if not context_parameters:
            raise RuntimeError(
                "formal Worker PG context 参数集为空")
        reward_grad_norm = _finite_autograd_gradient_norm(
            reward_pg_loss,
            actor_parameters,
            retain_graph=True,
        )
        reward_context_gradients = _finite_autograd_gradients(
            reward_pg_loss,
            context_parameters,
            retain_graph=True,
        )
        reward_context_grad_norm = _finite_gradient_tuple_norm(
            reward_context_gradients)
        reward_encoder_grad_norm = _finite_gradient_tuple_norm(
            _select_parameter_gradients(
                context_parameters,
                reward_context_gradients,
                context_groups["encoder"],
            )
        )
        reward_interaction_grad_norm = _finite_gradient_tuple_norm(
            _select_parameter_gradients(
                context_parameters,
                reward_context_gradients,
                context_groups["interaction"],
            )
        )
        reward_output_grad_norm = _finite_gradient_tuple_norm(
            _select_parameter_gradients(
                context_parameters,
                reward_context_gradients,
                context_groups["output"],
            )
        )
        combat_reward_actor_gradients = _finite_autograd_gradients(
            combat_reward_pg_loss,
            actor_parameters,
            retain_graph=False,
        )
        combat_reward_grad_norm = _finite_gradient_tuple_norm(
            combat_reward_actor_gradients)
        combat_reward_context_gradients = _select_parameter_gradients(
            actor_parameters,
            combat_reward_actor_gradients,
            context_parameters,
        )
        combat_reward_context_grad_norm = _finite_gradient_tuple_norm(
            combat_reward_context_gradients)
        combat_reward_encoder_grad_norm = _finite_gradient_tuple_norm(
            _select_parameter_gradients(
                context_parameters,
                combat_reward_context_gradients,
                context_groups["encoder"],
            )
        )
        combat_reward_interaction_grad_norm = _finite_gradient_tuple_norm(
            _select_parameter_gradients(
                context_parameters,
                combat_reward_context_gradients,
                context_groups["interaction"],
            )
        )
        combat_reward_output_grad_norm = _finite_gradient_tuple_norm(
            _select_parameter_gradients(
                context_parameters,
                combat_reward_context_gradients,
                context_groups["output"],
            )
        )
        requested_action_counts = np.bincount(
            requested.detach().cpu().numpy(),
            minlength=15,
        )
        executed_action_counts = np.bincount(
            executed_np[executed_np >= 0],
            minlength=15,
        )
        combat_rewards = rewards_np[combat_effect_np]
        receipt = {
            "schema": WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
            "rollout_end_timesteps": int(self.num_timesteps),
            "collection_actor_sha256": collection_actor_sha256,
            "onpolicy_log_prob_source":
                "collection-policy-vs-MaskableRolloutBuffer.log_probs",
            "onpolicy_log_prob_max_abs_delta":
                log_prob_max_abs_delta,
            "transition_reward_source":
                "WorkerWindowEnv.info.transition_reward",
            "transition_reward_samples": int(len(rewards_np)),
            "transition_reward_nonzero_samples":
                int(np.count_nonzero(rewards_np)),
            "transition_reward_positive_samples":
                int(np.count_nonzero(rewards_np > 0.0)),
            "transition_reward_negative_samples":
                int(np.count_nonzero(rewards_np < 0.0)),
            "transition_reward_sum": float(rewards_np.sum()),
            "transition_reward_abs_sum":
                float(np.abs(rewards_np).sum()),
            "transition_reward_mean": reward_mean,
            "transition_reward_variance": reward_variance,
            "no_progress_timeout_samples":
                int(np.count_nonzero(timeout_np)),
            "no_progress_timeout_base_failure_reward_sum":
                float(timeout_base_np.sum()),
            "no_progress_timeout_additional_failure_reward_sum":
                float(timeout_additional_np.sum()),
            "no_progress_timeout_failure_reward_sum":
                float(timeout_total_np.sum()),
            "requested_action_counts": [
                int(value) for value in requested_action_counts],
            "executed_action_counts": [
                int(value) for value in executed_action_counts],
            "combat_effect_samples":
                int(np.count_nonzero(combat_effect_np)),
            "combat_transition_reward_nonzero_samples":
                int(np.count_nonzero(
                    combat_effect_np & (rewards_np != 0.0))),
            "combat_transition_reward_positive_samples":
                int(np.count_nonzero(combat_rewards > 0.0)),
            "combat_transition_reward_negative_samples":
                int(np.count_nonzero(combat_rewards < 0.0)),
            "combat_transition_reward_sum":
                float(combat_rewards.sum()),
            "combat_positive_advantage_samples":
                int(np.count_nonzero(
                    combat_effect_np & (advantages_np > 0.0))),
            "combat_transition_reward_abs_sum":
                float(np.abs(
                    rewards_np[combat_effect_np]).sum()),
            "combat_reward_centered_l2":
                combat_reward_centered_l2,
            "combat_reward_centered_actor_grad_norm":
                float(combat_reward_grad_norm),
            "combat_reward_centered_root_grad_norm": 0.0,
            "combat_reward_centered_context_grad_norm":
                float(combat_reward_context_grad_norm),
            "combat_reward_centered_context_output_grad_norm":
                float(combat_reward_output_grad_norm),
            "combat_reward_centered_context_encoder_grad_norm":
                float(combat_reward_encoder_grad_norm),
            "combat_reward_centered_context_interaction_grad_norm":
                float(combat_reward_interaction_grad_norm),
            "reward_centered_l2": reward_centered_l2,
            "reward_centered_actor_grad_norm":
                float(reward_grad_norm),
            "reward_centered_context_grad_norm":
                float(reward_context_grad_norm),
            "reward_centered_context_output_grad_norm":
                float(reward_output_grad_norm),
            "reward_centered_context_encoder_grad_norm":
                float(reward_encoder_grad_norm),
            "reward_centered_context_interaction_grad_norm":
                float(reward_interaction_grad_norm),
            "advantage_source":
                "independently-recomputed-SB3-GAE",
            "gae_recomputed_max_abs_delta":
                float(gae_closure["advantage_max_abs_delta"]),
            "return_recomputed_max_abs_delta":
                float(gae_closure["return_max_abs_delta"]),
            "gae_advantage_samples": int(len(advantages_np)),
            "gae_advantage_nonzero_samples":
                int(np.count_nonzero(advantages_np)),
            "gae_advantage_mean": advantage_mean,
            "gae_advantage_variance": advantage_variance,
            "pure_ppo_gradient_source":
                "clipped-PPO-surrogate-only",
            "pure_ppo_actor_grad_measurements": 0,
            "pure_ppo_actor_grad_norm_max": 0.0,
            "pure_ppo_actor_grad_norm_mean": 0.0,
            "pure_ppo_root_grad_measurements": 0,
            "pure_ppo_root_grad_norm_max": 0.0,
            "pure_ppo_root_grad_norm_mean": 0.0,
            "pure_ppo_context_grad_measurements": 0,
            "pure_ppo_context_grad_norm_max": 0.0,
            "pure_ppo_context_grad_norm_mean": 0.0,
            "pure_ppo_context_output_grad_measurements": 0,
            "pure_ppo_context_output_grad_norm_max": 0.0,
            "pure_ppo_context_output_grad_norm_mean": 0.0,
            "pure_ppo_context_encoder_grad_measurements": 0,
            "pure_ppo_context_encoder_grad_norm_max": 0.0,
            "pure_ppo_context_encoder_grad_norm_mean": 0.0,
            "pure_ppo_context_interaction_grad_measurements": 0,
            "pure_ppo_context_interaction_grad_norm_max": 0.0,
            "pure_ppo_context_interaction_grad_norm_mean": 0.0,
            "distill_context_grad_measurements": 0,
            "distill_context_grad_norm_max": 0.0,
            "combined_root_dot_pure_ppo_sum": 0.0,
            "pure_ppo_root_grad_sq_sum": 0.0,
            "combined_root_on_pure_ppo_projection": 0.0,
            "combined_context_dot_pure_ppo_sum": 0.0,
            "pure_ppo_context_grad_sq_sum": 0.0,
            "combined_context_on_pure_ppo_projection": 0.0,
            "pure_ppo_actor_dot_combat_reward_sum": 0.0,
            "pure_ppo_actor_on_combat_reward_projection": 0.0,
            "pure_ppo_root_dot_combat_reward_sum": 0.0,
            "pure_ppo_root_on_combat_reward_projection": 0.0,
            "pure_ppo_context_dot_combat_reward_sum": 0.0,
            "pure_ppo_context_on_combat_reward_projection": 0.0,
            "optimizer_delta_source":
                "rollout-start-to-post-main-PPO-parameters",
            "optimizer_delta_actor_l2": 0.0,
            "optimizer_delta_actor_dot_combat_reward_descent": 0.0,
            "optimizer_delta_actor_on_combat_reward_descent_projection":
                0.0,
            "optimizer_delta_actor_on_combat_reward_descent_cosine": 0.0,
            "optimizer_delta_root_l2": 0.0,
            "optimizer_delta_root_dot_combat_reward_descent": 0.0,
            "optimizer_delta_root_on_combat_reward_descent_projection":
                0.0,
            "optimizer_delta_root_on_combat_reward_descent_cosine": 0.0,
            "optimizer_delta_context_l2": 0.0,
            "optimizer_delta_context_dot_combat_reward_descent": 0.0,
            "optimizer_delta_context_on_combat_reward_descent_projection":
                0.0,
            "optimizer_delta_context_on_combat_reward_descent_cosine": 0.0,
            "optimizer_steps": 0,
            "qualifies": False,
            "kl_early_stopped": False,
        }
        return {
            "receipt": receipt,
            "actor_parameters": tuple(actor_parameters),
            "combat_reward_actor_gradients":
                tuple(combat_reward_actor_gradients),
        }

    def _complete_worker_onpolicy_pg_rollout(
            self, receipt: dict | None, *,
            kl_early_stopped: bool = False,
            pure_ppo_actor_grad_norms: list[float],
            pure_ppo_root_grad_norms: list[float],
            pure_ppo_context_grad_norms: list[float],
            pure_ppo_context_output_grad_norms: list[float],
            pure_ppo_context_encoder_grad_norms: list[float],
            pure_ppo_context_interaction_grad_norms: list[float],
            distill_context_grad_norms: list[float],
            combined_root_dot_pure_ppo: list[float],
            pure_ppo_root_grad_sq: list[float],
            combined_context_dot_pure_ppo: list[float],
            pure_ppo_context_grad_sq: list[float],
            pure_ppo_root_dot_combat_reward: list[float],
            pure_ppo_context_dot_combat_reward: list[float],
            combat_reward_root_grad_norm: float | None,
            optimizer_delta_actor_l2: float | None,
            optimizer_delta_root_l2: float | None,
            optimizer_delta_context_l2: float | None,
            optimizer_delta_actor_dot_combat_reward_descent: float | None,
            optimizer_delta_root_dot_combat_reward_descent: float | None,
            optimizer_delta_context_dot_combat_reward_descent:
                float | None,
            optimizer_steps: int) -> None:
        if receipt is None:
            if (
                pure_ppo_actor_grad_norms
                or pure_ppo_root_grad_norms
                or pure_ppo_context_grad_norms
                or pure_ppo_context_output_grad_norms
                or pure_ppo_context_encoder_grad_norms
                or pure_ppo_context_interaction_grad_norms
                or distill_context_grad_norms
                or combined_root_dot_pure_ppo
                or pure_ppo_root_grad_sq
                or combined_context_dot_pure_ppo
                or pure_ppo_context_grad_sq
                or pure_ppo_root_dot_combat_reward
                or pure_ppo_context_dot_combat_reward
                or combat_reward_root_grad_norm is not None
                or optimizer_delta_actor_l2 is not None
                or optimizer_delta_root_l2 is not None
                or optimizer_delta_context_l2 is not None
                or optimizer_delta_actor_dot_combat_reward_descent
                is not None
                or optimizer_delta_root_dot_combat_reward_descent
                is not None
                or optimizer_delta_context_dot_combat_reward_descent
                is not None
            ):
                raise RuntimeError(
                    "formal PG receipt 缺失却产生 gradient measurements")
            return
        # A4 修正案(2026-07-27):冻结配方自带 target_kl 早停,与满 epoch
        # 地板冲突系配方内部矛盾(腿3 182,248 步实证:KL 尖峰 rollout 于第
        # 7 步早停被误杀)。早停 rollout 豁免至 ≥1(活性仍保证),如实落旗。
        _floor = (
            1 if kl_early_stopped
            else WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT)
        if (
            type(optimizer_steps) is not int
            or optimizer_steps < _floor
        ):
            raise RuntimeError(
                "formal PG 每个 joint rollout 至少要求 "
                f"{_floor}"
                f" 个 actor optimizer steps(kl_early_stopped="
                f"{kl_early_stopped!r})，实得 {optimizer_steps!r}")
        if (
            len(pure_ppo_actor_grad_norms) != optimizer_steps
            or len(pure_ppo_root_grad_norms) != optimizer_steps
            or len(pure_ppo_context_grad_norms) != optimizer_steps
            or len(
                pure_ppo_context_output_grad_norms) != optimizer_steps
            or len(
                pure_ppo_context_encoder_grad_norms) != optimizer_steps
            or len(
                pure_ppo_context_interaction_grad_norms
            ) != optimizer_steps
            or len(distill_context_grad_norms) != optimizer_steps
            or len(combined_root_dot_pure_ppo) != optimizer_steps
            or len(pure_ppo_root_grad_sq) != optimizer_steps
            or len(combined_context_dot_pure_ppo) != optimizer_steps
            or len(pure_ppo_context_grad_sq) != optimizer_steps
            or len(pure_ppo_root_dot_combat_reward)
            != optimizer_steps
            or len(pure_ppo_context_dot_combat_reward)
            != optimizer_steps
            or not all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) >= 0.0
                for value in (
                    *pure_ppo_actor_grad_norms,
                    *pure_ppo_root_grad_norms,
                    *pure_ppo_context_grad_norms,
                    *pure_ppo_context_output_grad_norms,
                    *pure_ppo_context_encoder_grad_norms,
                    *pure_ppo_context_interaction_grad_norms,
                    *distill_context_grad_norms,
                    *pure_ppo_root_grad_sq,
                    *pure_ppo_context_grad_sq,
                ))
            or not all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in (
                    *combined_root_dot_pure_ppo,
                    *combined_context_dot_pure_ppo,
                    *pure_ppo_root_dot_combat_reward,
                    *pure_ppo_context_dot_combat_reward,
                ))
            or not isinstance(
                combat_reward_root_grad_norm, (int, float))
            or not math.isfinite(
                float(combat_reward_root_grad_norm))
            or float(combat_reward_root_grad_norm) < 0.0
            or not all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) >= 0.0
                for value in (
                    optimizer_delta_actor_l2,
                    optimizer_delta_root_l2,
                    optimizer_delta_context_l2,
                )
            )
            or not all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in (
                    optimizer_delta_actor_dot_combat_reward_descent,
                    optimizer_delta_root_dot_combat_reward_descent,
                    optimizer_delta_context_dot_combat_reward_descent,
                )
            )
        ):
            raise RuntimeError(
                "formal PG optimizer/gradient measurements 不闭合")
        norms = [float(value) for value in pure_ppo_actor_grad_norms]
        root_norms = [
            float(value) for value in pure_ppo_root_grad_norms]
        context_norms = [
            float(value) for value in pure_ppo_context_grad_norms]
        output_norms = [
            float(value)
            for value in pure_ppo_context_output_grad_norms]
        encoder_norms = [
            float(value)
            for value in pure_ppo_context_encoder_grad_norms]
        interaction_norms = [
            float(value)
            for value in pure_ppo_context_interaction_grad_norms]
        distill_norms = [
            float(value) for value in distill_context_grad_norms]
        root_dots = [
            float(value) for value in combined_root_dot_pure_ppo]
        root_squares = [
            float(value) for value in pure_ppo_root_grad_sq]
        context_dots = [
            float(value)
            for value in combined_context_dot_pure_ppo]
        context_squares = [
            float(value) for value in pure_ppo_context_grad_sq]
        pure_combat_root_dots = [
            float(value)
            for value in pure_ppo_root_dot_combat_reward]
        pure_combat_context_dots = [
            float(value)
            for value in pure_ppo_context_dot_combat_reward]
        receipt["pure_ppo_actor_grad_measurements"] = len(norms)
        receipt["pure_ppo_actor_grad_norm_max"] = float(max(norms))
        receipt["pure_ppo_actor_grad_norm_mean"] = float(np.mean(norms))
        receipt["pure_ppo_root_grad_measurements"] = len(root_norms)
        receipt["pure_ppo_root_grad_norm_max"] = float(max(root_norms))
        receipt["pure_ppo_root_grad_norm_mean"] = float(
            np.mean(root_norms))
        receipt["pure_ppo_context_grad_measurements"] = len(
            context_norms)
        receipt["pure_ppo_context_grad_norm_max"] = float(
            max(context_norms))
        receipt["pure_ppo_context_grad_norm_mean"] = float(
            np.mean(context_norms))
        receipt["pure_ppo_context_output_grad_measurements"] = len(
            output_norms)
        receipt["pure_ppo_context_output_grad_norm_max"] = float(
            max(output_norms))
        receipt["pure_ppo_context_output_grad_norm_mean"] = float(
            np.mean(output_norms))
        receipt["pure_ppo_context_encoder_grad_measurements"] = len(
            encoder_norms)
        receipt["pure_ppo_context_encoder_grad_norm_max"] = float(
            max(encoder_norms))
        receipt["pure_ppo_context_encoder_grad_norm_mean"] = float(
            np.mean(encoder_norms))
        receipt[
            "pure_ppo_context_interaction_grad_measurements"
        ] = len(interaction_norms)
        receipt[
            "pure_ppo_context_interaction_grad_norm_max"
        ] = float(max(interaction_norms))
        receipt[
            "pure_ppo_context_interaction_grad_norm_mean"
        ] = float(np.mean(interaction_norms))
        receipt["distill_context_grad_measurements"] = len(
            distill_norms)
        receipt["distill_context_grad_norm_max"] = float(
            max(distill_norms))
        root_dot_sum = float(math.fsum(root_dots))
        root_square_sum = float(math.fsum(root_squares))
        receipt["combined_root_dot_pure_ppo_sum"] = root_dot_sum
        receipt["pure_ppo_root_grad_sq_sum"] = root_square_sum
        receipt["combined_root_on_pure_ppo_projection"] = (
            root_dot_sum / root_square_sum
            if root_square_sum > 0.0 else 0.0)
        dot_sum = float(math.fsum(context_dots))
        square_sum = float(math.fsum(context_squares))
        receipt["combined_context_dot_pure_ppo_sum"] = dot_sum
        receipt["pure_ppo_context_grad_sq_sum"] = square_sum
        receipt["combined_context_on_pure_ppo_projection"] = (
            dot_sum / square_sum if square_sum > 0.0 else 0.0)
        receipt["combat_reward_centered_root_grad_norm"] = float(
            combat_reward_root_grad_norm)
        combat_actor_sq = (
            receipt["combat_reward_centered_actor_grad_norm"] ** 2)
        combat_root_sq = (
            receipt["combat_reward_centered_root_grad_norm"] ** 2)
        combat_context_sq = (
            receipt["combat_reward_centered_context_grad_norm"] ** 2)
        pure_combat_root_sum = float(
            math.fsum(pure_combat_root_dots))
        pure_combat_context_sum = float(
            math.fsum(pure_combat_context_dots))
        pure_combat_actor_sum = (
            pure_combat_root_sum + pure_combat_context_sum)
        receipt[
            "pure_ppo_actor_dot_combat_reward_sum"
        ] = pure_combat_actor_sum
        receipt[
            "pure_ppo_actor_on_combat_reward_projection"
        ] = (
            pure_combat_actor_sum
            / (optimizer_steps * combat_actor_sq)
            if combat_actor_sq > 0.0 else 0.0
        )
        receipt[
            "pure_ppo_root_dot_combat_reward_sum"
        ] = pure_combat_root_sum
        receipt[
            "pure_ppo_root_on_combat_reward_projection"
        ] = (
            pure_combat_root_sum
            / (optimizer_steps * combat_root_sq)
            if combat_root_sq > 0.0 else 0.0
        )
        receipt[
            "pure_ppo_context_dot_combat_reward_sum"
        ] = pure_combat_context_sum
        receipt[
            "pure_ppo_context_on_combat_reward_projection"
        ] = (
            pure_combat_context_sum
            / (optimizer_steps * combat_context_sq)
            if combat_context_sq > 0.0 else 0.0
        )
        delta_actor_dot = float(
            optimizer_delta_actor_dot_combat_reward_descent)
        delta_root_dot = float(
            optimizer_delta_root_dot_combat_reward_descent)
        delta_context_dot = float(
            optimizer_delta_context_dot_combat_reward_descent)
        delta_actor_norm = float(optimizer_delta_actor_l2)
        delta_root_norm = float(optimizer_delta_root_l2)
        delta_context_norm = float(optimizer_delta_context_l2)

        def _delta_cosine(dot, delta_norm, combat_grad_norm):
            denominator = delta_norm * combat_grad_norm
            return dot / denominator if denominator > 0.0 else 0.0

        receipt["optimizer_delta_actor_l2"] = delta_actor_norm
        receipt[
            "optimizer_delta_actor_dot_combat_reward_descent"
        ] = delta_actor_dot
        receipt[
            "optimizer_delta_actor_on_combat_reward_descent_projection"
        ] = (
            delta_actor_dot / combat_actor_sq
            if combat_actor_sq > 0.0 else 0.0
        )
        receipt[
            "optimizer_delta_actor_on_combat_reward_descent_cosine"
        ] = _delta_cosine(
            delta_actor_dot,
            delta_actor_norm,
            receipt["combat_reward_centered_actor_grad_norm"],
        )
        receipt["optimizer_delta_root_l2"] = delta_root_norm
        receipt[
            "optimizer_delta_root_dot_combat_reward_descent"
        ] = delta_root_dot
        receipt[
            "optimizer_delta_root_on_combat_reward_descent_projection"
        ] = (
            delta_root_dot / combat_root_sq
            if combat_root_sq > 0.0 else 0.0
        )
        receipt[
            "optimizer_delta_root_on_combat_reward_descent_cosine"
        ] = _delta_cosine(
            delta_root_dot,
            delta_root_norm,
            receipt["combat_reward_centered_root_grad_norm"],
        )
        receipt["optimizer_delta_context_l2"] = delta_context_norm
        receipt[
            "optimizer_delta_context_dot_combat_reward_descent"
        ] = delta_context_dot
        receipt[
            "optimizer_delta_context_on_combat_reward_descent_projection"
        ] = (
            delta_context_dot / combat_context_sq
            if combat_context_sq > 0.0 else 0.0
        )
        receipt[
            "optimizer_delta_context_on_combat_reward_descent_cosine"
        ] = _delta_cosine(
            delta_context_dot,
            delta_context_norm,
            receipt["combat_reward_centered_context_grad_norm"],
        )
        receipt["optimizer_steps"] = optimizer_steps
        receipt["kl_early_stopped"] = bool(kl_early_stopped)
        receipt["qualifies"] = _worker_onpolicy_pg_receipt_qualifies(
            receipt)
        if not validate_worker_onpolicy_pg_receipt(
                receipt,
                expected_samples=int(self.n_steps) * int(self.n_envs)):
            raise RuntimeError("formal Worker on-policy PG receipt 非法")
        committed = dict(receipt)
        self._worker_onpolicy_pg_rollout_receipts.append(committed)
        self._worker_onpolicy_pg_joint_rollouts += 1
        self._worker_onpolicy_pg_qualifying_rollouts += int(
            committed["qualifies"])

    def _apply_main_ppo_optimizer_step(
            self, loss: th.Tensor, *, actor_frozen: bool) -> dict:
        start = self._critic_warmup_start_timesteps
        should_freeze = (
            start is not None
            and int(self.num_timesteps)
            <= int(self._critic_warmup_until_timesteps)
            and not self._critic_warmup_completed
        )
        if bool(actor_frozen) is not should_freeze:
            raise RuntimeError(
                "main PPO actor_frozen 与持久 warmup 边界不一致")
        self.policy.optimizer.zero_grad()
        loss.backward()
        circuit_snapshot = (
            self._protect_bc_aux_circuit_before_step()
            if self._bc_aux_circuit_spec is not None else [])
        if self.gradient_clip_mode == GRADIENT_CLIP_GLOBAL:
            if actor_frozen:
                raise RuntimeError("global clip 禁止用于 critic-only warmup")
            total = float(th.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm,
                error_if_nonfinite=True).detach().cpu())
            if not math.isfinite(total):
                raise RuntimeError("global gradient norm 非有限")
            clip_record = {
                "actor_counterfactual_norm": None,
                "actor_preclip_norm": None,
                "critic_preclip_norm": None,
                "actor_clipped": None,
                "critic_clipped": None,
                "actor_frozen": False,
                "global_preclip_norm": total,
            }
        elif self.gradient_clip_mode in {
                GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
                GRADIENT_CLIP_ROOT_CONTEXT_CRITIC_V2,
        }:
            clip_record = clip_actor_critic_gradients(
                self.policy, self.policy.optimizer,
                self.max_grad_norm,
                actor_frozen=actor_frozen,
                separate_root_context=(
                    self.gradient_clip_mode
                    == GRADIENT_CLIP_ROOT_CONTEXT_CRITIC_V2
                ),
            )
            clip_record["global_preclip_norm"] = None
        else:
            raise RuntimeError(
                f"未知 gradient_clip_mode:{self.gradient_clip_mode!r}")
        self.policy.optimizer.step()
        self._ppo_optimizer_steps_completed += 1
        if actor_frozen:
            self._critic_warmup_optimizer_steps_completed += 1
            current = actor_parameter_sha256(
                self.policy, optimizer=self.policy.optimizer)
            if current != self._critic_warmup_actor_sha256:
                raise RuntimeError("critic-only optimizer step 改写 actor")
        else:
            self._actor_optimizer_steps_completed += 1
        self._project_bc_aux_adapter_weight()
        self._assert_bc_aux_circuit_unchanged(circuit_snapshot)
        return clip_record

    def _complete_main_ppo_rollout(
            self, *, actor_frozen: bool, optimizer_steps: int) -> None:
        if not actor_frozen:
            if optimizer_steps <= 0:
                if not self._calib_tripped:
                    raise RuntimeError(
                        "joint actor rollout 未执行 optimizer step")
                return
            self._distill_actor_rollouts_completed += 1
            return
        if optimizer_steps <= 0:
            if not self._calib_tripped:
                raise RuntimeError("critic warmup rollout 未执行 optimizer step")
            return
        self._critic_warmup_rollouts_completed += 1
        now = int(self.num_timesteps)
        if now == self._critic_warmup_until_timesteps:
            if (self._critic_warmup_rollouts_completed
                    != self._critic_warmup_expected_rollouts):
                raise RuntimeError("critic warmup 末端 rollout 计数不闭合")
            self._critic_warmup_completed = True
            if isinstance(
                    self.policy,
                    AsymmetricWorkerMaskableActorCriticPolicy):
                self.policy.mlp_extractor.enable_actor_context()

    def _excluded_save_params(self):
        # _last_*/_calib_* 不入 zip:β=0 腿的哨兵行必须报 null 而非上腿陈值;
        # _calib_tripped=True 若被 load 驮回,会在续训第一步误杀健康腿(审查团确认项)
        return super()._excluded_save_params() + [
            "teacher", "_last_distill_ce", "_last_diverge",
            "_calib_tripped", "_calib_done", "calib_record_only",
            "_bc_aux_obs", "_bc_aux_actions", "_bc_aux_masks",
            "_bc_aux_rng", "_bc_aux_positive", "_bc_aux_negative",
            "_bc_aux_anchor_probs", "_bc_aux_validation_obs",
            "_bc_aux_validation_actions", "_bc_aux_validation_masks",
            "_bc_aux_validation_anchor_probs", "_bc_aux_permutations",
            "_bc_aux_cursors", "_bc_aux_train_calls",
            "_last_bc_aux_parts", "_last_bc_aux_ce",
            "bc_aux_monitor_out", "_bc_aux_pending_requested_a12",
            "_bc_aux_pending_executed_a12",
            "_bc_aux_pending_action_receipts",
            "_worker_onpolicy_pg_pending_receipts",
            "_worker_onpolicy_pg_collection_actor_sha256"]

    def collect_rollouts(self, *args, **kwargs) -> bool:
        """Seal the collection-policy actor identity around all callbacks."""
        if not self._worker_onpolicy_pg_audit_required:
            return super().collect_rollouts(*args, **kwargs)
        if (
            self._worker_onpolicy_pg_pending_receipts
            or self._worker_onpolicy_pg_collection_actor_sha256 is not None
        ):
            raise RuntimeError(
                "formal PG 新 rollout 前仍有未消费 collection 状态")
        actor_sha256 = actor_parameter_sha256(
            self.policy, optimizer=self.policy.optimizer)
        completed = super().collect_rollouts(*args, **kwargs)
        actor_sha256_after = actor_parameter_sha256(
            self.policy, optimizer=self.policy.optimizer)
        if actor_sha256_after != actor_sha256:
            raise RuntimeError(
                "formal PG rollout 收集期间/rollout-end callback "
                "改写 actor")
        if completed:
            if (
                not isinstance(
                    self.rollout_buffer,
                    _AuditedMaskableRolloutBuffer,
                )
                or not isinstance(
                    self.rollout_buffer._formal_gae_snapshot, dict)
            ):
                raise RuntimeError(
                    "formal PG 完整 rollout 缺 GAE 密封快照")
            self._worker_onpolicy_pg_collection_actor_sha256 = (
                actor_sha256)
        return completed

    def _update_info_buffer(self, infos, dones=None) -> None:
        """Bind rollout rows to native reward/action execution receipts.

        The rollout buffer stores requested actions only.  A potion key can be
        rejected during hit/block recovery, so counting ``actions == 12``
        falsely described failed key presses as on-policy potion experience.
        WorkerWindowEnv publishes the native-certified primary
        ``executed_action`` in every transition info; consume that receipt at
        the same VecEnv step at which SB3 records the request.  R7 critic
        migration additionally consumes the exact pre-TimeLimit-bootstrap
        ``transition_reward`` so an entropy- or critic-only actor change cannot
        pass as Worker policy-reward learning.
        """
        super()._update_info_buffer(infos, dones)
        if self._worker_onpolicy_pg_audit_required:
            if not isinstance(infos, (list, tuple)):
                raise RuntimeError("formal PG rollout info batch 形状异常")
            if len(infos) != int(self.n_envs):
                raise RuntimeError(
                    "formal PG rollout info batch 与 n_envs 不闭合:"
                    f"{len(infos)} != {int(self.n_envs)}")
            n_actions = int(getattr(self.action_space, "n", 0))
            validated_rewards = []
            if dones is None:
                raise RuntimeError(
                    "formal PG rollout 缺少 dones，无法核对 TimeLimit")
            dones_array = np.asarray(dones)
            if dones_array.shape != (int(self.n_envs),):
                raise RuntimeError(
                    "formal PG rollout dones 与 n_envs 不闭合:"
                    f"{dones_array.shape} != {(int(self.n_envs),)}")
            for index, info in enumerate(infos):
                if not isinstance(info, dict) \
                        or "requested_action" not in info \
                        or "transition_reward" not in info:
                    raise RuntimeError(
                        "formal PG rollout 缺 WorkerWindowEnv "
                        "requested_action/transition_reward 回执")
                requested = info["requested_action"]
                reward = info["transition_reward"]
                if (
                    not isinstance(requested, (int, np.integer))
                    or isinstance(requested, (bool, np.bool_))
                ):
                    raise RuntimeError(
                        "formal PG requested_action 非普通整数")
                requested = int(requested)
                if not 0 <= requested < n_actions:
                    raise RuntimeError(
                        "formal PG requested_action 越界")
                if (
                    not isinstance(
                        reward,
                        (int, float, np.integer, np.floating),
                    )
                    or isinstance(reward, (bool, np.bool_))
                    or not math.isfinite(float(reward))
                ):
                    raise RuntimeError(
                        "formal PG transition_reward 非有限普通数")
                timeout_fields = (
                    "worker_wage",
                    "worker_no_progress_timeout",
                    "no_progress_timeout_base_failure_reward",
                    "no_progress_timeout_additional_failure_reward",
                    "no_progress_timeout_failure_reward",
                )
                if any(field not in info for field in timeout_fields):
                    raise RuntimeError(
                        "formal PG rollout 缺 Worker no-progress timeout "
                        "逐步分账")
                worker_wage = info["worker_wage"]
                timeout = info["worker_no_progress_timeout"]
                timeout_components = tuple(info[field] for field in (
                    "no_progress_timeout_base_failure_reward",
                    "no_progress_timeout_additional_failure_reward",
                    "no_progress_timeout_failure_reward",
                ))
                numeric_timeout_values = (
                    worker_wage, *timeout_components)
                if (
                    type(timeout) is not bool
                    or any(
                        not isinstance(
                            value,
                            (int, float, np.integer, np.floating),
                        )
                        or isinstance(value, (bool, np.bool_))
                        or not math.isfinite(float(value))
                        for value in numeric_timeout_values
                    )
                ):
                    raise RuntimeError(
                        "formal PG no-progress timeout 分账类型/有限性异常")
                worker_wage = float(worker_wage)
                timeout_base, timeout_additional, timeout_total = (
                    float(value) for value in timeout_components)
                if not timeout:
                    if (
                        timeout_base,
                        timeout_additional,
                        timeout_total,
                    ) != (0.0, 0.0, 0.0):
                        raise RuntimeError(
                            "formal PG 非 timeout transition 携失败成本")
                elif (
                    not bool(dones_array[index])
                    or info.get("TimeLimit.truncated", False) is not False
                    or info.get("time_limit_bootstrap_safe") is not False
                    # A3 修正案(2026-07-27 批):快进链可把「工人窗超时
                    # 罚款」与「整局预算于未结算态耗尽」桥进同一 transition
                    # (fast_forward_extras 尾部触底)。此同现合法:放行
                    # unsettled=True,但要求与 budget_exhausted 一致闭合。
                    # 其余八肢原封。(第一腿 119,776 步确定性复现在案。)
                    or info.get("unsettled_budget_terminal")
                    not in (False, True)
                    or (
                        info.get("unsettled_budget_terminal") is True
                        and info.get("budget_exhausted") is not True
                    )
                    or timeout_base >= 0.0
                    or timeout_additional > 0.0
                    or timeout_total
                    != timeout_base + timeout_additional
                    or timeout_total >= 0.0
                    or float(reward) != worker_wage + timeout_total
                    or any(
                        field not in info
                        or not isinstance(
                            info[field],
                            (int, float, np.integer, np.floating),
                        )
                        or isinstance(info[field], (bool, np.bool_))
                        or float(info[field]) != 0.0
                        for field in (
                            "existing_terminal_death_reward",
                            "additional_terminal_death_reward",
                            "total_terminal_death_reward",
                        )
                    )
                ):
                    raise RuntimeError(
                        "formal PG no-progress timeout "
                        "终止/惩罚/死亡互斥账不闭合")
                if "executed_action" not in info:
                    raise RuntimeError(
                        "formal PG rollout 缺 executed_action 回执")
                executed = info["executed_action"]
                if executed is not None and (
                    not isinstance(executed, (int, np.integer))
                    or isinstance(executed, (bool, np.bool_))
                ):
                    raise RuntimeError(
                        "formal PG executed_action 非普通整数/None")
                if executed is not None:
                    executed = int(executed)
                    if not 0 <= executed < n_actions:
                        raise RuntimeError(
                            "formal PG executed_action 越界")
                    if executed != requested:
                        raise RuntimeError(
                            "formal PG 请求/执行动作回执不一致")

                effect = info.get("action_effect_audit")
                if effect is None:
                    if not (
                        executed is None
                        and info.get("fuse_tripped") is True
                        and info.get("overridden") is True
                    ):
                        raise RuntimeError(
                            "formal PG 非 fuse transition "
                            "缺 action_effect_audit")
                    combat_effect = False
                else:
                    if (
                        not isinstance(effect, dict)
                        or set(effect)
                        != _WORKER_ACTION_EFFECT_AUDIT_KEYS
                    ):
                        raise RuntimeError(
                            "formal PG action_effect_audit schema 漂移")
                    attempts = effect["native_attempts"]
                    accepts = effect["native_accepts"]
                    reasons = effect["effect_reasons"]
                    if (
                        isinstance(
                            effect.get("requested_action"),
                            (bool, np.bool_),
                        )
                        or not isinstance(
                            effect.get("requested_action"),
                            (int, np.integer),
                        )
                        or int(effect["requested_action"]) != requested
                        or isinstance(attempts, (bool, np.bool_))
                        or not isinstance(attempts, (int, np.integer))
                        or isinstance(accepts, (bool, np.bool_))
                        or not isinstance(accepts, (int, np.integer))
                        or not 0 <= int(accepts) <= int(attempts)
                        or not isinstance(
                            effect.get("request_executed"), bool)
                        or not isinstance(
                            effect.get("material_effect"), bool)
                        or not isinstance(effect.get("same_scene"), bool)
                        or not isinstance(
                            effect.get("stall_cost_applied"), bool)
                        or not isinstance(reasons, tuple)
                        or any(
                            not isinstance(reason, str) or not reason
                            for reason in reasons
                        )
                        or len(set(reasons)) != len(reasons)
                        or bool(reasons)
                        != effect["material_effect"]
                        or effect["request_executed"]
                        != (requested == 0 or int(accepts) > 0)
                        or effect["stall_cost_applied"]
                        != (
                            effect["same_scene"]
                            and (
                                requested == 0
                                or not effect["request_executed"]
                            )
                        )
                        or (
                            executed is not None
                        ) != effect["request_executed"]
                        or info.get("fuse_tripped") is True
                        or info.get("overridden") is True
                    ):
                        raise RuntimeError(
                            "formal PG action_effect_audit 字段不守恒")
                    combat_effect = bool(
                        requested == 9
                        and executed == 9
                        and _WORKER_COMBAT_EFFECT_REASONS.intersection(
                            reasons)
                    )
                raw_reward = np.float32(float(reward))
                expected = np.asarray(
                    [raw_reward], dtype=np.float32)
                time_limit_bootstrap = bool(
                    bool(dones_array[index])
                    and info.get("terminal_observation") is not None
                    and info.get("TimeLimit.truncated", False)
                )
                if (
                    info.get("unsettled_budget_terminal") is True
                    and time_limit_bootstrap
                ):
                    raise RuntimeError(
                        "formal PG unsettled budget terminal "
                        "不得执行 TimeLimit bootstrap")
                if time_limit_bootstrap:
                    terminal_obs = self.policy.obs_to_tensor(
                        info["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(
                            terminal_obs)[0]
                    # Deliberately mirror sb3-contrib collect_rollouts:
                    # mutate one float32 reward cell with gamma*Tensor value,
                    # then compare that exact stored representation later.
                    expected[0] += self.gamma * terminal_value
                expected_reward = np.float32(expected[0])
                bootstrap_delta = np.float32(
                    expected_reward - raw_reward)
                validated_rewards.append({
                    "requested_action": requested,
                    "executed_action": executed,
                    "combat_effect": combat_effect,
                    "transition_reward": float(raw_reward),
                    "worker_no_progress_timeout": timeout,
                    "no_progress_timeout_base_failure_reward":
                        timeout_base,
                    "no_progress_timeout_additional_failure_reward":
                        timeout_additional,
                    "no_progress_timeout_failure_reward":
                        timeout_total,
                    "expected_buffer_reward": float(expected_reward),
                    "time_limit_bootstrap": time_limit_bootstrap,
                    "time_limit_bootstrap_delta": float(
                        bootstrap_delta),
                })
            # Commit only after the entire VecEnv batch validates, preserving
            # row order ``step-major, env-minor`` used by buffer flattening.
            self._worker_onpolicy_pg_pending_receipts.extend(
                validated_rewards)
        if self._bc_aux_circuit_spec is None:
            return
        if not isinstance(infos, (list, tuple)):
            raise RuntimeError("a12 rollout info batch 形状异常")
        if len(infos) != int(self.n_envs):
            raise RuntimeError(
                "a12 rollout info batch 与 n_envs 不闭合:"
                f"{len(infos)} != {int(self.n_envs)}")
        n_actions = int(getattr(self.action_space, "n", 0))
        if n_actions <= 0:
            raise RuntimeError("a12 rollout action_space 非 Discrete")
        validated = []
        for info in infos:
            if not isinstance(info, dict) \
                    or "requested_action" not in info \
                    or "executed_action" not in info:
                raise RuntimeError(
                    "a12 rollout 缺 WorkerWindowEnv 执行回执")
            requested = info["requested_action"]
            executed = info["executed_action"]
            if (
                not isinstance(requested, (int, np.integer))
                or isinstance(requested, (bool, np.bool_))
            ):
                raise RuntimeError("a12 rollout requested_action 非普通整数")
            requested = int(requested)
            if not 0 <= requested < n_actions:
                raise RuntimeError("a12 rollout requested_action 越界")
            if executed is not None and (
                not isinstance(executed, (int, np.integer))
                or isinstance(executed, (bool, np.bool_))
            ):
                raise RuntimeError("a12 rollout executed_action 非普通整数/None")
            if executed is not None:
                executed = int(executed)
                if not 0 <= executed < n_actions:
                    raise RuntimeError("a12 rollout executed_action 越界")
                if executed != requested:
                    raise RuntimeError(
                        "a12 rollout 请求/执行动作回执不一致")
            validated.append((requested, executed))

        # Commit only after the complete VecEnv info batch validates.  A bad
        # later environment must not leave a partially advanced audit stream.
        self._bc_aux_pending_action_receipts.extend(validated)
        self._bc_aux_pending_requested_a12 += sum(
            requested == _LEGACY_DISTILL_EXCLUDED_ACTION
            for requested, _ in validated)
        self._bc_aux_pending_executed_a12 += sum(
            executed == _LEGACY_DISTILL_EXCLUDED_ACTION
            for _, executed in validated)

    def _teacher_probs(self, obs: th.Tensor, action_masks: th.Tensor) -> th.Tensor:
        # KING predates the sign-multiplexed drink latch.  Decode only its
        # read-only input; the student still receives ``obs`` unchanged.
        teacher_obs = _legacy_worker_observation_view(obs)
        teacher_width = int(self.teacher[0].in_features)
        if teacher_obs.shape[-1] < teacher_width:
            raise ValueError("KING teacher observation 维度不足")
        teacher_obs = teacher_obs[..., :teacher_width]
        t_logits = self.teacher(teacher_obs)
        mask = action_masks.reshape(t_logits.shape).bool()
        return _masked_log_softmax_from_raw(t_logits, mask).exp()

    def _student_raw_action_logits(self, obs: th.Tensor) -> th.Tensor:
        """Return actor scores before Categorical normalization or masking."""
        actor_obs = (
            _legacy_worker_observation_view(obs)
            if self._bc_aux_circuit_spec is not None else obs)
        features = self.policy.extract_features(actor_obs)
        if self.policy.share_features_extractor:
            latent_pi = self.policy.mlp_extractor.forward_actor(features)
        else:
            # SB3 returns ``(pi_features, vf_features)`` when feature
            # extractors are not shared.
            pi_features, _ = features
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
        return self.policy.action_net(latent_pi)

    def _student_distillation_logits(self, obs: th.Tensor) -> th.Tensor:
        """Return the student branch that is semantically shared with KING.

        KING observes only the legacy 298-wide state and was trained against
        the old macro kernel.  Applying its CE to the full asymmetric actor
        would directly punish every context-conditioned deviation, including
        the phase/distance interactions introduced specifically to repair the
        combat policy.  Keep the legacy root and shared action head leashed,
        but leave the fresh context residual reward-driven.
        """
        if not isinstance(
                self.policy,
                AsymmetricWorkerMaskableActorCriticPolicy):
            return self._student_raw_action_logits(obs)
        features = self.policy.extract_features(obs)
        if not self.policy.share_features_extractor:
            pi_features, _ = features
        else:
            pi_features = features
        if (
            pi_features.ndim != 2
            or pi_features.shape[1]
            != ASYMMETRIC_WORKER_OBSERVATION_DIM
        ):
            raise ValueError(
                "asymmetric distillation observation 形状异常")
        latent_pi = self.policy.mlp_extractor.policy_net(
            pi_features[:, :ASYMMETRIC_WORKER_LEGACY_DIM])
        return self.policy.action_net(latent_pi)

    def mount_bc_aux_demos(self, obs, actions, masks,
                           rng: np.random.Generator) -> None:
        """挂载 rev4 校准 bank，并冻结首次 aux 根策略作为非触发态锚。

        bank 必须同时含 a12 正例与 ``m[12]=True,y!=12`` hard negatives；
        前者教“何时可喝”，后者和起点分布 KL 共同防“有药就喝”及“全 a9”。
        入口另保证这些行只来自 BC-v2 training episodes，原始 held-out
        episodes 不得进入优化器。rng 仅用于分组无放回轮转，不触碰训练 RNG。
        """
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions)
        masks = np.asarray(masks)
        n_actions = int(getattr(self.action_space, "n"))
        obs_dim = int(np.prod(self.observation_space.shape))
        if obs.ndim != 2 or len(obs) == 0 or obs.shape[1] != obs_dim:
            raise ValueError(
                f"bc_aux 示范观测形状异常: {obs.shape}(期望 (N,{obs_dim}),N≥1)")
        if (actions.ndim != 1 or len(actions) != len(obs)
                or not np.issubdtype(actions.dtype, np.integer)
                or not bool(((actions >= 0) & (actions < n_actions)).all())):
            raise ValueError("bc_aux 示范标签形状/类型/取值异常")
        if masks.shape != (len(obs), n_actions) or masks.dtype != np.bool_:
            raise ValueError(
                f"bc_aux 示范掩码形状/dtype 异常: {masks.shape},{masks.dtype}")
        if not bool(masks[np.arange(len(actions)), actions].all()):
            raise ValueError(
                "bc_aux 示范对存在标签被自身掩码禁止"
                "(掩位 log-prob 为 -1e8,on-manifold 破缺,fail-loud)")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("bc_aux 须挂专用 np.random.Generator 流(零染训练 RNG)")
        positive = np.flatnonzero(actions == 12)
        negative = np.flatnonzero(actions != 12)
        if len(positive) == 0 or len(negative) == 0:
            raise ValueError(
                "bc_aux rev4 bank 必须同时包含 a12 正例与非 a12 hard negative")
        if not bool(masks[:, 12].all()):
            raise ValueError("bc_aux rev4 bank 全部样本均须 m[12]=True")
        self._bc_aux_obs = th.as_tensor(obs, device=self.device)
        self._bc_aux_actions = th.as_tensor(actions.astype(np.int64),
                                            device=self.device)
        self._bc_aux_masks = th.as_tensor(masks, device=self.device)
        self._bc_aux_rng = rng
        self._bc_aux_positive = positive.astype(np.int64)
        self._bc_aux_negative = negative.astype(np.int64)
        self._bc_aux_permutations = {}
        self._bc_aux_cursors = {}
        self._bc_aux_train_calls = 0
        self._last_bc_aux_parts = None
        # continuation 不得把 KL 锚逐腿重置为“本腿起点”。首次 aux 腿的六张量
        # 根锚会随 checkpoint 持久化；后续腿即使 current 已漂移，也始终用该
        # 根策略在同一 bank/masks 上的概率。正例不消费此锚，仍可提高 a12。
        root = getattr(self, "bc_aux_root_anchor_sd", None)
        if root is None:
            was_training = self.policy.training
            self.policy.set_training_mode(False)
            with th.no_grad():
                root_masks = self._bc_aux_masks
                if self._bc_aux_circuit_spec is not None:
                    root_masks = _legacy_distillation_masks(root_masks)
                    root_logits = self._student_raw_action_logits(
                        self._bc_aux_obs)
                    self._bc_aux_anchor_probs = (
                        _masked_log_softmax_from_raw(
                            root_logits, root_masks)
                        .exp().detach().clone())
                else:
                    anchor_dist = self.policy.get_distribution(
                        _legacy_worker_observation_view(self._bc_aux_obs),
                        action_masks=root_masks)
                    self._bc_aux_anchor_probs = (
                        anchor_dist.distribution.logits.exp()
                        .detach().clone())
            self.policy.set_training_mode(was_training)
        else:
            if not isinstance(root, dict) \
                    or set(root) != set(_BC_AUX_POLICY_HEAD_KEYS):
                raise ValueError("bc_aux persistent root 策略头键集合异常")
            tensors = []
            for key in _BC_AUX_POLICY_HEAD_KEYS:
                value = root[key]
                if not isinstance(value, th.Tensor) \
                        or not bool(th.isfinite(value).all().item()):
                    raise ValueError(
                        f"bc_aux persistent root 张量异常:{key}")
                tensors.append(
                    value.detach().to(device=self.device, dtype=th.float32))
            w0, b0, w1, b1, wa, ba = tensors
            if (w0.ndim != 2 or w0.shape[1] != obs_dim
                    or b0.shape != (w0.shape[0],)
                    or w1.ndim != 2 or w1.shape[1] != w0.shape[0]
                    or b1.shape != (w1.shape[0],)
                    or wa.shape != (n_actions, w1.shape[0])
                    or ba.shape != (n_actions,)):
                raise ValueError("bc_aux persistent root 策略头形状异常")
            with th.no_grad():
                legacy_obs = _legacy_worker_observation_view(self._bc_aux_obs)
                hidden = th.tanh(legacy_obs @ w0.T + b0)
                hidden = th.tanh(hidden @ w1.T + b1)
                logits = hidden @ wa.T + ba
                root_masks = self._bc_aux_masks
                if self._bc_aux_circuit_spec is not None:
                    root_masks = _legacy_distillation_masks(root_masks)
                    root_logp = _masked_log_softmax_from_raw(
                        logits, root_masks)
                    self._bc_aux_anchor_probs = (
                        root_logp.exp().detach().clone())
                else:
                    masked = th.where(
                        root_masks, logits,
                        th.full_like(logits, HUGE_NEG))
                    self._bc_aux_anchor_probs = (
                        th.softmax(masked, dim=-1).detach().clone())

    def _bc_aux_root_probs_for(
            self, obs: th.Tensor, masks: th.Tensor) -> th.Tensor:
        """Evaluate the persistent (possibly narrower) legacy root head."""
        root = getattr(self, "bc_aux_root_anchor_sd", None)
        if not isinstance(root, dict) \
                or set(root) != set(_BC_AUX_POLICY_HEAD_KEYS):
            raise RuntimeError("a12 circuit validation 缺 persistent root")
        tensors = [
            root[key].detach().to(device=self.device, dtype=th.float32)
            for key in _BC_AUX_POLICY_HEAD_KEYS
        ]
        if not all(bool(value.isfinite().all().item()) for value in tensors):
            raise RuntimeError("a12 circuit root 含 NaN/Inf")
        w0, b0, w1, b1, wa, ba = tensors
        with th.no_grad():
            legacy_obs = _legacy_worker_observation_view(obs)
            hidden = th.tanh(legacy_obs @ w0.T + b0)
            hidden = th.tanh(hidden @ w1.T + b1)
            logits = hidden @ wa.T + ba
            root_masks = _legacy_distillation_masks(masks)
            return (
                _masked_log_softmax_from_raw(logits, root_masks)
                .exp().detach().clone())

    def mount_bc_aux_circuit_validation(
            self, obs, actions, masks) -> None:
        """Mount the nested-validation domain used only by rollout gates."""
        if self._bc_aux_circuit_spec is None:
            raise RuntimeError("validation 只适用于 active a12 circuit")
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions)
        masks = np.asarray(masks)
        obs_dim = int(np.prod(self.observation_space.shape))
        n_actions = int(self.action_space.n)
        if (
            obs.ndim != 2 or len(obs) == 0 or obs.shape[1] != obs_dim
            or actions.shape != (len(obs),)
            or not np.issubdtype(actions.dtype, np.integer)
            or masks.shape != (len(obs), n_actions)
            or masks.dtype != np.bool_
            or not bool(masks[
                np.arange(len(actions)), actions.astype(np.int64)].all())
            or not bool((actions == _LEGACY_DISTILL_EXCLUDED_ACTION).any())
            or not bool(((actions != _LEGACY_DISTILL_EXCLUDED_ACTION)
                         & masks[:, _LEGACY_DISTILL_EXCLUDED_ACTION]).any())
        ):
            raise ValueError("a12 circuit validation 输入形状/覆盖异常")
        self._bc_aux_validation_obs = th.as_tensor(
            obs, device=self.device)
        self._bc_aux_validation_actions = th.as_tensor(
            actions.astype(np.int64), device=self.device)
        self._bc_aux_validation_masks = th.as_tensor(
            masks, device=self.device)
        self._bc_aux_validation_anchor_probs = self._bc_aux_root_probs_for(
            self._bc_aux_validation_obs,
            self._bc_aux_validation_masks)

    def mount_bc_aux_circuit_fit(
            self, obs, actions, masks) -> None:
        """Mount the complete nested-fit distribution (never a 1:8 bank)."""
        if self._bc_aux_circuit_spec is None:
            raise RuntimeError("fit 只适用于 active a12 circuit")
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions)
        masks = np.asarray(masks)
        obs_dim = int(np.prod(self.observation_space.shape))
        n_actions = int(self.action_space.n)
        if (
            obs.ndim != 2 or len(obs) == 0 or obs.shape[1] != obs_dim
            or actions.shape != (len(obs),)
            or not np.issubdtype(actions.dtype, np.integer)
            or masks.shape != (len(obs), n_actions)
            or masks.dtype != np.bool_
            or not bool(masks[
                np.arange(len(actions)), actions.astype(np.int64)].all())
            or not bool((actions == _LEGACY_DISTILL_EXCLUDED_ACTION).any())
            or not bool(((actions != _LEGACY_DISTILL_EXCLUDED_ACTION)
                         & masks[:, _LEGACY_DISTILL_EXCLUDED_ACTION]).any())
        ):
            raise ValueError("a12 circuit fit 输入形状/覆盖异常")
        self._bc_aux_obs = th.as_tensor(obs, device=self.device)
        self._bc_aux_actions = th.as_tensor(
            actions.astype(np.int64), device=self.device)
        self._bc_aux_masks = th.as_tensor(masks, device=self.device)
        self._bc_aux_positive = np.flatnonzero(
            actions == _LEGACY_DISTILL_EXCLUDED_ACTION).astype(np.int64)
        self._bc_aux_negative = np.flatnonzero(
            actions != _LEGACY_DISTILL_EXCLUDED_ACTION).astype(np.int64)
        self._bc_aux_anchor_probs = self._bc_aux_root_probs_for(
            self._bc_aux_obs, self._bc_aux_masks)

    def _bc_aux_take(self, group: str, count: int) -> np.ndarray:
        """组内跨调用无放回轮转；耗尽才重洗，避免 199 正例反复抽中同一批。"""
        pool = (self._bc_aux_positive if group == "positive"
                else self._bc_aux_negative)
        if pool is None or len(pool) == 0:
            raise RuntimeError(f"bc_aux {group} 池为空")
        need = int(count)
        out = []
        while need > 0:
            permutation = self._bc_aux_permutations.get(group)
            cursor = int(self._bc_aux_cursors.get(group, 0))
            if permutation is None or cursor >= len(permutation):
                permutation = self._bc_aux_rng.permutation(pool)
                cursor = 0
                self._bc_aux_permutations[group] = permutation
            take = min(need, len(permutation) - cursor)
            out.append(permutation[cursor:cursor + take])
            cursor += take
            need -= take
            self._bc_aux_cursors[group] = cursor
        return np.concatenate(out).astype(np.int64, copy=False)

    def _bc_aux_ce_loss(self) -> th.Tensor:
        """rev5 校准损失：soft binary target + 非触发态根策略 KL。

        正负组分别求均值后按 1:3 混合，不让 8:1 bank 大小隐式改变权重。
        正例目标 0.65 而非 1.0，负例目标 0.01；KL 只施于负例，故不会把
        a12 正例重新焊回 KING 的零使用状态。
        """
        if self._bc_aux_obs is None or self._bc_aux_rng is None:
            raise RuntimeError(
                "λ_bc>0 但示范池未挂载(fail-loud 条款,反 P2 构造性零通路)")
        if self._bc_aux_anchor_probs is None:
            raise RuntimeError("bc_aux rev4 根策略锚未挂载")
        size = min(int(self.batch_size), int(self._bc_aux_obs.shape[0]))
        pos_n = min(
            len(self._bc_aux_positive),
            max(1, int(round(size * _BC_AUX_POSITIVE_FRACTION))))
        neg_n = min(len(self._bc_aux_negative), max(1, size - pos_n))
        pos_np = self._bc_aux_take("positive", pos_n)
        neg_np = self._bc_aux_take("negative", neg_n)
        index_np = np.concatenate([pos_np, neg_np])
        index = th.as_tensor(index_np, dtype=th.long, device=self.device)
        dist = self.policy.get_distribution(
            self._bc_aux_obs[index], action_masks=self._bc_aux_masks[index])
        logp = dist.distribution.logits
        # Consume the deployed distribution's action log-probability rather
        # than ``Categorical.logits[:, 12]``.  The exact-mixture distribution
        # carries an analytic a12 log-prob derivative; reading PyTorch's
        # redundantly re-normalized logits would reintroduce an O(1e-9)
        # gradient into unsupported non-a12 rows (notably a14), which Adam can
        # accumulate despite the mathematical mixture being independent.
        a12_actions = th.full(
            (len(index_np),),
            _LEGACY_DISTILL_EXCLUDED_ACTION,
            dtype=th.long,
            device=self.device,
        )
        # 方案A(2026-07-27 批,1-ULP 跨平台勘定):BCE 支路消费掩码后的
        # 解析 a12 log-prob 列,不再经 log_prob() 的直通前向——直通残差
        # (normalized−exact).detach() 是采样器 O(ulp) 归一化噪声,随
        # BLAS/SIMD 归约序漂移(Mac ARM vs Linux AVX 差 1 ULP);解析列
        # 对 a14 扰动逐位不变,梯度与直通反向同一张量,语义零变化。
        # 直通值仍由 PPO ratio 路径独占消费(那里需要与采样器逐位一致)。
        exact_original = getattr(dist, "_original_exact_log_probs", None)
        if exact_original is not None:
            mask12 = self._bc_aux_masks[index][
                :, _LEGACY_DISTILL_EXCLUDED_ACTION].to(th.bool)
            a12_logp = th.where(
                mask12,
                exact_original[:, _LEGACY_DISTILL_EXCLUDED_ACTION],
                th.full_like(
                    exact_original[:, _LEGACY_DISTILL_EXCLUDED_ACTION],
                    HUGE_NEG))
            p12 = a12_logp.exp().clamp(1e-7, 1.0 - 1e-7)
        else:
            p12 = dist.log_prob(a12_actions).exp().clamp(
                1e-7, 1.0 - 1e-7)
        pos = th.arange(pos_n, device=self.device)
        neg = th.arange(pos_n, pos_n + neg_n, device=self.device)
        pos_target = th.full_like(p12[pos], _BC_AUX_POSITIVE_TARGET)
        neg_target = th.full_like(p12[neg], _BC_AUX_NEGATIVE_TARGET)
        pos_loss = F.binary_cross_entropy(p12[pos], pos_target)
        neg_loss = F.binary_cross_entropy(p12[neg], neg_target)

        anchor = self._bc_aux_anchor_probs[index][neg]
        if self._bc_aux_circuit_spec is not None:
            negative_obs = self._bc_aux_obs[index][neg]
            root_support = _legacy_distillation_masks(
                self._bc_aux_masks[index][neg])
            current_logp = _masked_log_softmax_from_raw(
                self._student_raw_action_logits(negative_obs),
                root_support)
        else:
            current_logp = logp[neg]
        eps = th.finfo(anchor.dtype).eps
        anchor_kl = (anchor * (
            th.log(anchor.clamp_min(eps)) - current_logp
        )).sum(dim=-1).mean().clamp_min(0.0)
        loss = (_BC_AUX_POSITIVE_FRACTION * pos_loss
                + (1.0 - _BC_AUX_POSITIVE_FRACTION) * neg_loss
                + _BC_AUX_ANCHOR_KL_COEF * anchor_kl)
        self._last_bc_aux_parts = {
            "positive_bce": float(pos_loss.detach().cpu()),
            "negative_bce": float(neg_loss.detach().cpu()),
            "anchor_kl": float(anchor_kl.detach().cpu()),
            "positive_n": int(pos_n), "negative_n": int(neg_n),
        }
        return loss

    def _peek_bc_aux_ce_loss(self) -> th.Tensor:
        """构建 G-CAL 辅助图，但不消费生产示范流。

        rev5 的真实辅助更新在每轮 PPO epochs 之后独立执行。G-CAL 仍需在
        首个 PPO minibatch 上读取联合梯度方向；若直接调用采样器，末尾真实
        更新会悄悄消费第二批示范。这里完整快照专用 RNG、排列和游标，返回
        可求导图后恢复采样状态，使末尾重算严格取得同一批样本。
        """
        if self._bc_aux_rng is None:
            raise RuntimeError("bc_aux peek 前示范池未挂载")
        rng_state = copy.deepcopy(self._bc_aux_rng.bit_generator.state)
        permutations = {
            key: value.copy()
            for key, value in self._bc_aux_permutations.items()
        }
        cursors = dict(self._bc_aux_cursors)
        try:
            return self._bc_aux_ce_loss()
        finally:
            self._bc_aux_rng.bit_generator.state = rng_state
            self._bc_aux_permutations = permutations
            self._bc_aux_cursors = cursors

    def _apply_bc_aux_step(self) -> th.Tensor:
        """执行一次 rev5 独立辅助 optimizer step。

        旧实现把辅助项塞进首个 PPO minibatch，与 PPO/value/entropy/KING
        梯度共同裁剪；而 liveness preflight 做的是纯辅助步，因而系统性
        高估可学性。现在生产与预检共用这一原子顺序：
        zero_grad → λ·loss.backward → 全局裁剪 → optimizer.step。
        """
        if not np.isfinite(self.bc_aux_lambda) or self.bc_aux_lambda <= 0:
            raise RuntimeError("独立 bc_aux step 要求有限正 λ")
        aux_loss = self._bc_aux_ce_loss()
        weighted = self.bc_aux_lambda * aux_loss
        self.policy.optimizer.zero_grad()
        weighted.backward()
        th.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        return aux_loss

    def _bc_aux_circuit_protected_tensors(
            self) -> list[tuple[th.nn.Parameter, th.Tensor]]:
        """Return every frozen slice around the contextual a12 gate.

        The old non-a12 actor remains trainable.  Four added neurons and every
        cross-connection stay canonical zero and serve only as an unmistakable
        checkpoint topology marker.  Four otherwise-unused action-head cells
        are coefficients over stable raw combat features; action12's bias is
        their intercept.  Those five values are the only adapter parameters.
        The exact mixture maps g(s) into a bounded probability, so learning
        context can never leak a12 outside the visible predicate.
        """
        spec = self._bc_aux_circuit_spec
        if spec is None:
            return []
        required = {
            "schema_version", "base_width", "expanded_width",
            "action_index", "gate_feature_indices",
            "gate_parameter_columns",
            "hp_low", "hp_high", "boundary_epsilon",
            "initial_probability", "initial_gate_bias",
            "probability_min", "probability_max",
            "gate_parameter_abs_max",
        }
        if not isinstance(spec, dict) or set(spec) != required:
            raise RuntimeError("a12 circuit spec 字段不精确")
        if spec["schema_version"] != _BC_AUX_CIRCUIT_SCHEMA:
            raise RuntimeError("a12 circuit schema 漂移")
        base = int(spec["base_width"])
        width = int(spec["expanded_width"])
        action = int(spec["action_index"])
        policy_net = self.policy.mlp_extractor.policy_net
        if len(policy_net) < 3:
            raise RuntimeError("a12 circuit 策略 MLP 结构异常")
        w0, b0 = policy_net[0].weight, policy_net[0].bias
        w1, b1 = policy_net[2].weight, policy_net[2].bias
        wa, ba = self.policy.action_net.weight, self.policy.action_net.bias
        if (
            w0.shape[0] != width
            or b0.shape != (width,)
            or w1.shape != (width, width)
            or b1.shape != (width,)
            or wa.shape[1] != width
            or ba.shape != (wa.shape[0],)
            or not 0 < base < width
            or not 0 <= action < wa.shape[0]
        ):
            raise RuntimeError("a12 circuit 参数形状与 spec 不闭合")

        masks = []
        for parameter in (w0, b0, w1, b1, wa, ba):
            masks.append(th.zeros_like(parameter, dtype=th.bool))
        masks[0][base:] = True
        masks[1][base:] = True
        masks[2][base:, :] = True
        masks[2][:, base:] = True
        masks[3][base:] = True
        # All expanded output columns remain frozen except the four a12 cells
        # used explicitly as raw-feature gate coefficients.  Action12's old
        # latent row remains canonical zero, so a12 gradients cannot enter the
        # legacy actor; its bias is the fifth trainable gate parameter.
        masks[4][action, :base] = True
        masks[4][:, base:] = True
        parameter_columns = tuple(
            int(value) for value in spec["gate_parameter_columns"])
        feature_indices = tuple(
            int(value) for value in spec["gate_feature_indices"])
        if (
            len(parameter_columns) != width - base
            or len(feature_indices) != len(parameter_columns)
            or tuple(parameter_columns) != tuple(range(base, width))
        ):
            raise RuntimeError("a12 contextual gate 参数/特征映射不闭合")
        masks[4][action, list(parameter_columns)] = False
        return list(zip((w0, b0, w1, b1, wa, ba), masks))

    def _protect_bc_aux_circuit_before_step(
            self) -> list[tuple[th.nn.Parameter, th.Tensor, th.Tensor]]:
        """Zero circuit gradients *before clipping* and purge Adam momentum."""
        snapshots = []
        optimizer = self.policy.optimizer
        for parameter, protected in self._bc_aux_circuit_protected_tensors():
            if parameter.grad is not None:
                if parameter.grad.shape != parameter.shape:
                    raise RuntimeError("a12 circuit gradient 形状异常")
                parameter.grad.masked_fill_(protected, 0.0)
            state = optimizer.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = state.get(key)
                if value is not None:
                    if not isinstance(value, th.Tensor) \
                            or value.shape != parameter.shape:
                        raise RuntimeError(
                            f"a12 circuit optimizer {key} 形状异常")
                    value.masked_fill_(protected, 0.0)
            snapshots.append((
                parameter, protected,
                parameter.detach()[protected].clone()))
        return snapshots

    @staticmethod
    def _assert_bc_aux_circuit_unchanged(
            snapshots: list[tuple[th.nn.Parameter, th.Tensor, th.Tensor]]
    ) -> None:
        for parameter, protected, before in snapshots:
            after = parameter.detach()[protected]
            if not th.equal(after, before):
                raise RuntimeError(
                    "主 optimizer 改写了冻结 a12 circuit 参数")

    def _bc_aux_drift_for(
            self, obs, actions, masks, anchor) -> dict | None:
        """Behavior/sampling safety metrics for one immutable BC domain."""
        if obs is None or actions is None or masks is None or anchor is None:
            return None
        with th.no_grad():
            dist = self.policy.get_distribution(
                obs, action_masks=masks)
            probs = dist.distribution.logits.exp()
            pred = probs.argmax(dim=-1)
            if self._bc_aux_circuit_spec is not None:
                root_support = _legacy_distillation_masks(masks)
                current_root_logp = _masked_log_softmax_from_raw(
                    self._student_raw_action_logits(obs), root_support)
                current_root_probs = current_root_logp.exp()
                current_root_pred = current_root_probs.argmax(dim=-1)
            else:
                current_root_logp = th.log(
                    probs.clamp_min(th.finfo(probs.dtype).eps))
                current_root_probs = probs
                current_root_pred = pred
            anchor_pred = anchor.argmax(dim=-1)
            positive = actions == 12
            negative = ~positive
            legal_negative = negative & masks[:, 12]
            pred12 = pred == 12
            eps = th.finfo(probs.dtype).eps
            tp = int((pred12 & positive).sum().cpu())
            fp = int((pred12 & legal_negative).sum().cpu())
            pos_n = int(positive.sum().cpu())
            neg_n = int(legal_negative.sum().cpu())
            high_hp_negative = (
                legal_negative & (obs[:, 0] >= 0.65))
            high_hp_n = int(high_hp_negative.sum().cpu())
            high_hp_fp = int((pred12 & high_hp_negative).sum().cpu())
            anchor_pred12 = anchor_pred == 12
            anchor_fp = int(
                (anchor_pred12 & legal_negative).sum().cpu())
            anchor_high_hp_fp = int(
                (anchor_pred12 & high_hp_negative).sum().cpu())
            legal_negative_p12 = probs[:, 12][legal_negative]
            predicted_share_13 = float((pred == 13).float().mean().cpu())
            anchor_share_13 = float(
                (anchor_pred == 13).float().mean().cpu())
            return {
                "pairs": int(len(pred)),
                "true_a12": pos_n,
                "non_a12": neg_n,
                "predicted_share_12": float(pred12.float().mean().cpu()),
                "recall_12": tp / pos_n if pos_n else 0.0,
                "fpr_12": fp / neg_n if neg_n else 1.0,
                "anchor_fpr_12": anchor_fp / neg_n if neg_n else 1.0,
                "anchor_predicted_share_12": float(
                    anchor_pred12.float().mean().cpu()),
                "high_hp_non_a12": high_hp_n,
                "high_hp_false_drink_rate": (
                    high_hp_fp / high_hp_n if high_hp_n else 1.0),
                "anchor_high_hp_false_drink_rate": (
                    anchor_high_hp_fp / high_hp_n if high_hp_n else 1.0),
                "legal_negative_probability_12_mean": (
                    float(legal_negative_p12.mean().cpu())
                    if neg_n else 1.0),
                "legal_negative_probability_12_max": (
                    float(legal_negative_p12.max().cpu())
                    if neg_n else 1.0),
                "legal_negative_probability_12_sum": (
                    float(legal_negative_p12.sum().cpu())
                    if neg_n else float("inf")),
                "argmax_drift": float(
                    (current_root_pred != anchor_pred)
                    .float().mean().cpu()),
                "tv_mean": float(
                    (0.5 * (current_root_probs - anchor)
                     .abs().sum(dim=-1)).mean().cpu()),
                "kl_anchor_to_policy": float(
                    (anchor * (
                        th.log(anchor.clamp_min(eps))
                        - current_root_logp
                    )).sum(dim=-1).mean().clamp_min(0.0).cpu()),
                "predicted_share_13": predicted_share_13,
                "anchor_share_13": anchor_share_13,
                "a13_spillover": max(
                    0.0, predicted_share_13 - anchor_share_13),
            }

    def _bc_aux_bank_drift(self) -> dict | None:
        """Fixed fit-bank behavior relative to the persistent V28 root."""
        return self._bc_aux_drift_for(
            self._bc_aux_obs, self._bc_aux_actions,
            self._bc_aux_masks, self._bc_aux_anchor_probs)

    def _bc_aux_validation_drift(self) -> dict | None:
        """Nested-validation behavior; this domain never tunes the circuit."""
        return self._bc_aux_drift_for(
            self._bc_aux_validation_obs,
            self._bc_aux_validation_actions,
            self._bc_aux_validation_masks,
            self._bc_aux_validation_anchor_probs)

    def _project_bc_aux_adapter_weight(self) -> dict | None:
        """Project contextual gate parameters into a finite compact box."""
        spec = self._bc_aux_circuit_spec
        if spec is None:
            return None
        action = int(spec["action_index"])
        columns = tuple(
            int(value) for value in spec["gate_parameter_columns"])
        limit = float(spec["gate_parameter_abs_max"])
        weight = self.policy.action_net.weight
        bias = self.policy.action_net.bias
        gate_weight = weight[action, list(columns)]
        gate_bias = bias[action]
        if (
            not bool(th.isfinite(gate_weight).all().item())
            or not bool(th.isfinite(gate_bias).item())
        ):
            raise RuntimeError("a12 contextual gate 参数含 NaN/Inf")
        before_weight = gate_weight.detach().clone()
        before_bias = gate_bias.detach().clone()
        with th.no_grad():
            weight[action, list(columns)] = before_weight.clamp(
                -limit, limit)
            gate_bias.clamp_(-limit, limit)
        gate_weight = weight[action, list(columns)]
        clipped_weight = before_weight != gate_weight.detach()
        clipped_bias = bool((before_bias != gate_bias.detach()).item())
        weight_state = self.policy.optimizer.state.get(weight, {})
        if bool(clipped_weight.any().item()):
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = weight_state.get(key)
                if value is not None:
                    for column, clipped in zip(
                            columns, clipped_weight.tolist()):
                        if clipped:
                            value[action, column] = 0.0
        bias_state = self.policy.optimizer.state.get(bias, {})
        if clipped_bias:
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = bias_state.get(key)
                if value is not None:
                    value[action] = 0.0
        return {
            "gate_bias": float(gate_bias.detach().cpu()),
            "gate_weight_l2": float(
                th.linalg.vector_norm(gate_weight.detach()).cpu()),
            "gate_weight_abs_max": float(
                gate_weight.detach().abs().max().cpu()),
            "projected": (
                bool(clipped_weight.any().item()) or clipped_bias),
            "gate_parameter_abs_max": limit,
        }

    def _record_bc_aux_rollout_exposure(self) -> dict | None:
        """Count genuine on-policy adapter opportunities exactly once."""
        spec = self._bc_aux_circuit_spec
        if spec is None:
            return None
        obs = th.as_tensor(
            self.rollout_buffer.observations, device=self.device)
        actions = th.as_tensor(
            self.rollout_buffer.actions, device=self.device).long()
        masks = th.as_tensor(
            self.rollout_buffer.action_masks, device=self.device).bool()
        obs = obs.reshape(-1, obs.shape[-1])
        actions = actions.reshape(-1)
        masks = masks.reshape(-1, masks.shape[-1])
        action = int(spec["action_index"])
        eligible = (
            (obs[:, 0] >= (
                float(spec["hp_low"])
                - float(spec["boundary_epsilon"])))
            & (obs[:, 0] < (
                float(spec["hp_high"])
                - float(spec["boundary_epsilon"])))
            & (obs[:, _LEGACY_EXHAUSTED_FEATURE] >= 0.0)
            & masks[:, action]
        )
        receipts = self._bc_aux_pending_action_receipts
        if not isinstance(receipts, list) or len(receipts) != len(actions):
            raise RuntimeError(
                "a12 rollout buffer 与执行回执长度不闭合:"
                f"buffer={len(actions)},receipts="
                f"{len(receipts) if isinstance(receipts, list) else 'invalid'}")
        receipt_requested = th.as_tensor(
            [requested for requested, _ in receipts],
            dtype=actions.dtype,
            device=actions.device,
        )
        if not th.equal(receipt_requested, actions):
            mismatch = int(
                th.nonzero(receipt_requested != actions, as_tuple=False)[0]
                .item()
            )
            raise RuntimeError(
                "a12 rollout buffer 请求序列与执行回执错位:"
                f"index={mismatch},buffer={int(actions[mismatch])},"
                f"receipt={int(receipt_requested[mismatch])}")
        executed_actions = th.as_tensor(
            [-1 if executed is None else executed for _, executed in receipts],
            dtype=actions.dtype,
            device=actions.device,
        )
        requested_a12 = receipt_requested == action
        executed_a12 = executed_actions == action
        eligible_count = int(eligible.sum().cpu())
        unexpected_requested_count = int(
            ((~eligible) & requested_a12).sum().cpu())
        if unexpected_requested_count:
            raise RuntimeError(
                "a12 contextual mixture 在硬 eligibility 外请求动作12:"
                f"{unexpected_requested_count}")
        unexpected_count = int((~eligible & executed_a12).sum().cpu())
        pending_requested = int(requested_a12.sum().cpu())
        executed_count = int(executed_a12.sum().cpu())
        if (
            pending_requested != int(self._bc_aux_pending_requested_a12)
            or executed_count != int(self._bc_aux_pending_executed_a12)
            or not bool((executed_a12 <= requested_a12).all().item())
        ):
            raise RuntimeError(
                "a12 rollout buffer 请求与原生执行回执不闭合:"
                f"buffer={int(requested_a12.sum().cpu())},"
                f"info_requested={pending_requested},"
                f"info_executed={executed_count}")
        rejected_count = pending_requested - executed_count
        with th.no_grad():
            distribution = self.policy.get_distribution(
                obs, action_masks=masks)
            p12 = distribution.distribution.logits[:, action].exp()
            expected_mass = float(p12.sum().cpu())
        if not np.isfinite(expected_mass) or expected_mass < 0.0:
            raise RuntimeError("a12 rollout expected probability mass 非法")
        self._bc_aux_eligible_states += eligible_count
        self._bc_aux_requested_a12 += pending_requested
        # ``sampled`` is retained as the historical receipt key, but from
        # rev9 onward it means a sampled request whose native action audit
        # certified actual potion execution.
        self._bc_aux_sampled_a12 += executed_count
        self._bc_aux_rejected_a12 += rejected_count
        self._bc_aux_unexpected_sampled_a12 += unexpected_count
        self._bc_aux_expected_a12_mass += expected_mass
        self._bc_aux_pending_requested_a12 = 0
        self._bc_aux_pending_executed_a12 = 0
        self._bc_aux_pending_action_receipts.clear()
        return {
            "rollout_eligible_states": eligible_count,
            "rollout_requested_a12": pending_requested,
            "rollout_sampled_a12": executed_count,
            "rollout_rejected_a12": rejected_count,
            "rollout_unexpected_sampled_a12": unexpected_count,
            "cumulative_eligible_states":
                int(self._bc_aux_eligible_states),
            "cumulative_requested_a12":
                int(self._bc_aux_requested_a12),
            "cumulative_sampled_a12":
                int(self._bc_aux_sampled_a12),
            "cumulative_rejected_a12":
                int(self._bc_aux_rejected_a12),
            "cumulative_unexpected_sampled_a12":
                int(self._bc_aux_unexpected_sampled_a12),
            "rollout_expected_a12_mass": expected_mass,
            "cumulative_expected_a12_mass":
                float(self._bc_aux_expected_a12_mass),
        }

    def _bc_aux_rollout_monitor(self) -> dict | None:
        """每个 optimizer rollout 后检查毁损上界；不对 recall 设下限。"""
        circuit_active = self._bc_aux_circuit_spec is not None
        if self.bc_aux_lambda <= 0 and not circuit_active:
            return None
        bank = self._bc_aux_bank_drift()
        validation = (
            self._bc_aux_validation_drift() if circuit_active else None)
        reasons = []
        if bank is None:
            reasons.append("bc_aux_bank_missing")
        elif circuit_active:
            if validation is None:
                reasons.append("bc_aux_validation_missing")
            else:
                if validation["fpr_12"] > _BC_AUX_CIRCUIT_FPR_MAX:
                    reasons.append(
                        f"validation_fpr12>{_BC_AUX_CIRCUIT_FPR_MAX}")
                if (validation["predicted_share_12"]
                        > _BC_AUX_CIRCUIT_SHARE_MAX):
                    reasons.append(
                        "validation_predicted_share12>"
                        f"{_BC_AUX_CIRCUIT_SHARE_MAX}")
                if (validation["high_hp_false_drink_rate"]
                        > _BC_AUX_CIRCUIT_HIGH_HP_FPR_MAX):
                    reasons.append(
                        "validation_high_hp_false_drink_rate>"
                        f"{_BC_AUX_CIRCUIT_HIGH_HP_FPR_MAX}")
                if validation["tv_mean"] > _BC_AUX_CIRCUIT_TV_MAX:
                    reasons.append(
                        f"validation_tv>{_BC_AUX_CIRCUIT_TV_MAX}")
                if (validation["argmax_drift"]
                        > _BC_AUX_CIRCUIT_ARGMAX_DRIFT_MAX):
                    reasons.append(
                        "validation_argmax_drift>"
                        f"{_BC_AUX_CIRCUIT_ARGMAX_DRIFT_MAX}")
                if (validation[
                        "legal_negative_probability_12_mean"]
                        > _BC_AUX_CIRCUIT_LEGAL_NEG_P12_MEAN_MAX):
                    reasons.append(
                        "validation_legal_negative_p12_mean>"
                        f"{_BC_AUX_CIRCUIT_LEGAL_NEG_P12_MEAN_MAX}")
                if (validation[
                        "legal_negative_probability_12_max"]
                        > _BC_AUX_CIRCUIT_LEGAL_NEG_P12_MAX):
                    reasons.append(
                        "validation_legal_negative_p12_max>"
                        f"{_BC_AUX_CIRCUIT_LEGAL_NEG_P12_MAX}")
        else:
            allowed_fpr = max(
                _BC_AUX_ROLLOUT_FPR_MAX,
                bank["anchor_fpr_12"] + _BC_AUX_ROLLOUT_FPR_MAX)
            if bank["fpr_12"] > allowed_fpr:
                reasons.append(
                    f"bank_fpr12>{allowed_fpr}")
            allowed_share = max(
                _BC_AUX_ROLLOUT_PREDICTED_SHARE_MAX,
                bank["anchor_predicted_share_12"] + 0.10)
            if (bank["predicted_share_12"]
                    > allowed_share):
                reasons.append(
                    "bank_predicted_share12>"
                    f"{allowed_share}")
            allowed_high_hp = max(
                _BC_AUX_ROLLOUT_HIGH_HP_FPR_MAX,
                bank["anchor_high_hp_false_drink_rate"]
                + _BC_AUX_ROLLOUT_HIGH_HP_FPR_MAX)
            if (bank["high_hp_false_drink_rate"]
                    > allowed_high_hp):
                reasons.append(
                    "bank_high_hp_false_drink_rate>"
                    f"{allowed_high_hp}")
            # 小型 smoke bank 的单个 argmax 翻转占比很大；用 2/sqrt(N)
            # 统计容差，生产约 1.8k bank 时仍收紧到注册 5%。
            allowed_a13 = max(
                _BC_AUX_ROLLOUT_A13_SPILLOVER_MAX,
                2.0 / np.sqrt(max(1, bank["pairs"])))
            if bank["a13_spillover"] > allowed_a13:
                reasons.append(
                    f"bank_a13_spillover>{allowed_a13}")
        record = {
            "monitor": "bc-aux-rollout-boundary/2",
            "step": int(self.num_timesteps),
            "offpolicy_bank": bank,
            "nested_validation": validation,
            "adapter": (
                {
                    **(self._project_bc_aux_adapter_weight() or {}),
                    "cumulative_eligible_states":
                        int(self._bc_aux_eligible_states),
                    "cumulative_requested_a12":
                        int(self._bc_aux_requested_a12),
                    "cumulative_sampled_a12":
                        int(self._bc_aux_sampled_a12),
                    "cumulative_rejected_a12":
                        int(self._bc_aux_rejected_a12),
                    "cumulative_unexpected_sampled_a12":
                        int(self._bc_aux_unexpected_sampled_a12),
                    "cumulative_expected_a12_mass":
                        float(self._bc_aux_expected_a12_mass),
                }
                if circuit_active else None),
            "trip_reasons": reasons,
            "tripped": bool(reasons),
            "lower_recall_gate_applied": False,
        }
        if reasons:
            self._calib_tripped = True
            print(f"   [bc_aux rollout gate] {record}")
        if self.bc_aux_monitor_out:
            with open(self.bc_aux_monitor_out, "a") as stream:
                stream.write(json.dumps(record) + "\n")
        return record

    def _calib_probe(self, policy_loss, distill_ce, diverge, *,
                     effective_distill_beta: float | None = None,
                     teacher_entropy=None, distill_kl=None,
                     distill_tv=None, total_loss=None, bc_aux_loss=None):
        """生产 G-CAL：梯度全景 + a12 头 + 固定 off-policy bank 漂移。

        ``calib_record_only`` 只保留给旧协议的零侵入对照；新生产命令不携该
        旗，任一破坏性读数会在触发 minibatch 更新前停腿。
        """
        if effective_distill_beta is None:
            effective_distill_beta = float(self.distill_beta)
        params = [p for p in self.policy.parameters() if p.requires_grad]

        def gnorm(scalar):
            """Return ``(norm, status)`` without serialising NaN/Inf."""
            if scalar is None or not params or not scalar.requires_grad:
                return 0.0, "missing"
            grads = th.autograd.grad(
                scalar, params, retain_graph=True, allow_unused=True)
            live = [gradient for gradient in grads if gradient is not None]
            if not live:
                return 0.0, "missing"
            if not all(bool(th.isfinite(gradient).all().item())
                       for gradient in live):
                return None, "nonfinite"
            value = float(th.sqrt(th.stack(
                [(gradient ** 2).sum() for gradient in live]
            ).sum()).detach().cpu())
            if not np.isfinite(value):
                return None, "nonfinite"
            return value, ("zero" if value == 0.0 else "ok")

        def action12_grad(scalar):
            if scalar is None or not scalar.requires_grad:
                return None
            head = self.policy.action_net
            grads = th.autograd.grad(
                scalar, [head.weight, head.bias], retain_graph=True,
                allow_unused=True)
            flattened = []
            if grads[0] is not None and grads[0].shape[0] > 12:
                flattened.append(grads[0][12].reshape(-1))
            if grads[1] is not None and grads[1].shape[0] > 12:
                flattened.append(grads[1][12].reshape(-1))
            return th.cat(flattened) if flattened else None

        def vector_norm_status(vector):
            if vector is None or vector.numel() == 0:
                return 0.0, "missing"
            if not bool(th.isfinite(vector).all().item()):
                return None, "nonfinite"
            value = float(th.linalg.vector_norm(vector).detach().cpu())
            if not np.isfinite(value):
                return None, "nonfinite"
            return value, ("zero" if value == 0.0 else "ok")

        if total_loss is None:
            total_loss = policy_loss
        aux_term = (
            self.bc_aux_lambda * bc_aux_loss
            if bc_aux_loss is not None else None)
        aux_a12 = action12_grad(aux_term)
        other_a12 = action12_grad(
            total_loss - aux_term if aux_term is not None else total_loss)
        total_a12 = action12_grad(total_loss)

        def cosine(left, right):
            if left is None or right is None:
                return None
            if (not bool(th.isfinite(left).all().item())
                    or not bool(th.isfinite(right).all().item())):
                return None
            denom = th.linalg.vector_norm(left) * th.linalg.vector_norm(right)
            denom_value = float(denom.detach().cpu())
            if not np.isfinite(denom_value) or denom_value == 0.0:
                return None
            value = float((th.dot(left, right) / denom).detach().cpu())
            return value if np.isfinite(value) else None

        def projection_on_aux(total, aux):
            if total is None or aux is None:
                return None
            if (not bool(th.isfinite(total).all().item())
                    or not bool(th.isfinite(aux).all().item())):
                return None
            denom = th.dot(aux, aux)
            denom_value = float(denom.detach().cpu())
            if not np.isfinite(denom_value) or denom_value == 0.0:
                return None
            value = float((th.dot(total, aux) / denom).detach().cpu())
            return value if np.isfinite(value) else None

        pg_norm, _ = gnorm(policy_loss)
        ce_norm, _ = (
            gnorm(effective_distill_beta * distill_ce)
            if distill_ce.requires_grad else (0.0, "missing"))
        aux_norm, aux_status = gnorm(aux_term)
        total_norm, _ = gnorm(total_loss)
        aux_a12_norm, aux_a12_status = vector_norm_status(aux_a12)
        total_a12_norm, total_a12_status = vector_norm_status(total_a12)
        aux_other_cosine = cosine(aux_a12, other_a12)
        total_aux_projection = projection_on_aux(total_a12, aux_a12)

        def scalar_readout(scalar):
            if scalar is None:
                return None
            value = float(scalar.detach().cpu())
            return value if np.isfinite(value) else None

        bank = self._bc_aux_bank_drift()
        trip_reasons = []
        if diverge > 0.20:
            trip_reasons.append("teacher_diverge>0.20")
        if self.bc_aux_lambda > 0:
            # A liveness probe that merely records an absent/poisoned a12
            # signal is a false PASS.  Production must prove that the actual
            # combined update still has a strictly positive component along
            # the auxiliary a12 direction before touching the minibatch.
            if ("nonfinite" in {aux_status, aux_a12_status}):
                trip_reasons.append("aux_gradient_nonfinite")
            elif (aux_status in {"missing", "zero"}
                  or aux_a12_status in {"missing", "zero"}):
                trip_reasons.append("aux_gradient_missing")
            elif total_a12_status == "nonfinite":
                trip_reasons.append(
                    "total_on_aux_a12_projection_nonfinite")
            elif total_a12_status == "missing":
                trip_reasons.append(
                    "total_on_aux_a12_projection_missing")
            elif total_aux_projection is None:
                trip_reasons.append(
                    "total_on_aux_a12_projection_nonfinite")
            elif total_aux_projection <= 0.0:
                trip_reasons.append(
                    "total_on_aux_a12_projection<=0")
            if bank is None:
                trip_reasons.append("bc_aux_bank_missing")
            else:
                if bank["tv_mean"] > _BC_AUX_CALIB_TV_MAX:
                    trip_reasons.append(
                        f"bank_tv>{_BC_AUX_CALIB_TV_MAX}")
                if bank["argmax_drift"] > _BC_AUX_CALIB_ARGMAX_DRIFT_MAX:
                    trip_reasons.append(
                        f"bank_argmax_drift>{_BC_AUX_CALIB_ARGMAX_DRIFT_MAX}")
                if bank["fpr_12"] > _BC_AUX_CALIB_FPR_MAX:
                    trip_reasons.append(
                        f"bank_fpr12>{_BC_AUX_CALIB_FPR_MAX}")
                if (bank["predicted_share_12"]
                        > _BC_AUX_CALIB_PREDICTED_SHARE_MAX):
                    trip_reasons.append(
                        "bank_predicted_share12>"
                        f"{_BC_AUX_CALIB_PREDICTED_SHARE_MAX}")
        rec = {"step": int(self.num_timesteps),
               "g_pg": pg_norm,
               "g_ce": ce_norm,
               "g_aux": aux_norm,
               "g_aux_a12": aux_a12_norm,
               "g_total": total_norm,
               "g_total_a12": total_a12_norm,
               "g_aux_vs_other_a12_cosine": aux_other_cosine,
               "g_total_on_aux_a12_projection": total_aux_projection,
               "distill_ce": scalar_readout(distill_ce),
               "teacher_entropy": scalar_readout(teacher_entropy),
               "distill_kl": scalar_readout(distill_kl),
               "distill_tv": scalar_readout(distill_tv),
               "distill_beta_initial": float(self.distill_beta),
               "distill_beta_effective":
                   float(effective_distill_beta),
               "bc_aux_lambda": float(self.bc_aux_lambda),
               "bc_aux_loss": scalar_readout(bc_aux_loss),
               "bc_aux_parts": self._last_bc_aux_parts,
               "offpolicy_bank": bank,
               "teacher_diverge": round(diverge, 4),
               "trip_reasons": trip_reasons,
               "tripped": bool(trip_reasons)}
        if rec["tripped"] and not getattr(self, "calib_record_only", False):
            self._calib_tripped = True
        if self.calib_out:
            with open(self.calib_out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"   [G-CAL] {rec}")

    def train(self) -> None:
        # ===== sb3_contrib ppo_mask.py train() 诚实复写;“皮筋”段以 β>0 守卫 =====
        if not np.isfinite(self.distill_beta) or self.distill_beta < 0:
            raise ValueError(f"distill_beta 必须是有限非负数,实得 {self.distill_beta!r}")
        if not np.isfinite(self.bc_aux_lambda) or self.bc_aux_lambda < 0:
            raise ValueError(
                f"bc_aux_lambda 必须是有限非负数,实得 {self.bc_aux_lambda!r}")
        if self.bc_aux_lambda > 0 and (
                self._bc_aux_obs is None or self._bc_aux_rng is None):
            raise RuntimeError(
                "λ_bc>0 但示范池未挂载(fail-loud 条款,反 P2 构造性零通路)")
        if self._bc_aux_circuit_spec is not None and (
                self._bc_aux_obs is None
                or self._bc_aux_validation_obs is None):
            raise RuntimeError(
                "a12 circuit 在位但 fit/validation bank 未完整挂载")
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        actor_frozen_for_rollout = self._prepare_main_ppo_rollout()
        effective_distill_beta = self._effective_distill_beta(
            actor_frozen=actor_frozen_for_rollout)
        self._last_effective_distill_beta = float(
            effective_distill_beta)
        worker_pg_bundle = self._begin_worker_onpolicy_pg_rollout(
            actor_frozen=actor_frozen_for_rollout)
        worker_pg_receipt = (
            worker_pg_bundle["receipt"]
            if worker_pg_bundle is not None else None
        )
        worker_pg_actor_parameters = (
            strict_actor_critic_parameter_partition(
                self.policy,
                optimizer=self.policy.optimizer,
            )["actor"]
            if worker_pg_receipt is not None else ()
        )
        worker_pg_adapter = (
            self.policy.mlp_extractor.context_adapter
            if worker_pg_receipt is not None else None
        )
        worker_pg_context_parameters = (
            tuple(worker_pg_adapter.parameters())
            if worker_pg_adapter is not None else ()
        )
        worker_pg_context_parameter_ids = {
            id(parameter) for parameter in worker_pg_context_parameters}
        worker_pg_root_parameters = tuple(
            parameter
            for parameter in worker_pg_actor_parameters
            if id(parameter) not in worker_pg_context_parameter_ids
        )
        if worker_pg_receipt is not None and (
            not worker_pg_root_parameters
            or len(worker_pg_root_parameters)
            + len(worker_pg_context_parameters)
            != len(worker_pg_actor_parameters)
            or tuple(
                id(parameter)
                for parameter in worker_pg_bundle["actor_parameters"]
            ) != tuple(
                id(parameter)
                for parameter in worker_pg_actor_parameters
            )
        ):
            raise RuntimeError(
                "formal PG actor root/context 参数分区不闭合")
        worker_pg_actor_start = (
            tuple(
                parameter.detach().clone()
                for parameter in worker_pg_actor_parameters
            )
            if worker_pg_receipt is not None else ()
        )
        worker_pg_combat_actor_gradients = (
            tuple(worker_pg_bundle[
                "combat_reward_actor_gradients"])
            if worker_pg_receipt is not None else ()
        )
        if (
            worker_pg_receipt is not None
            and len(worker_pg_combat_actor_gradients)
            != len(worker_pg_actor_parameters)
        ):
            raise RuntimeError(
                "formal PG combat gradient/actor 参数数量不闭合")
        worker_pg_combat_root_gradients = (
            _select_parameter_gradients(
                worker_pg_actor_parameters,
                worker_pg_combat_actor_gradients,
                worker_pg_root_parameters,
            )
            if worker_pg_receipt is not None else ()
        )
        worker_pg_combat_context_gradients = (
            _select_parameter_gradients(
                worker_pg_actor_parameters,
                worker_pg_combat_actor_gradients,
                worker_pg_context_parameters,
            )
            if worker_pg_receipt is not None else ()
        )
        worker_pg_context_groups = (
            worker_pg_adapter.named_parameter_groups()
            if worker_pg_adapter is not None else {
                "encoder": (),
                "interaction": (),
                "output": (),
            }
        )
        pure_ppo_actor_grad_norms = []
        pure_ppo_root_grad_norms = []
        pure_ppo_context_grad_norms = []
        pure_ppo_context_output_grad_norms = []
        pure_ppo_context_encoder_grad_norms = []
        pure_ppo_context_interaction_grad_norms = []
        distill_context_grad_norms = []
        combined_root_dot_pure_ppo = []
        pure_ppo_root_grad_sq = []
        combined_context_dot_pure_ppo = []
        pure_ppo_context_grad_sq = []
        pure_ppo_root_dot_combat_reward = []
        pure_ppo_context_dot_combat_reward = []
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        distill_ces, teacher_entropies, distill_kls = [], [], []
        distill_tvs, diverges, t_confs = [], [], []
        bc_aux_ces = []
        actor_counterfactual_norms = []
        root_counterfactual_norms = []
        context_counterfactual_norms = []
        actor_clip_scales = []
        root_clip_scales = []
        context_clip_scales = []
        actor_preclip_norms = []
        root_preclip_norms = []
        context_preclip_norms = []
        critic_preclip_norms = []
        actor_grad_clipped = 0
        root_grad_clipped = 0
        context_grad_clipped = 0
        critic_grad_clipped = 0
        ppo_optimizer_steps_this_rollout = 0
        calib_due = ((effective_distill_beta > 0
                      or self.bc_aux_lambda > 0)
                     and self.calib_out is not None
                     and any(p not in self._calib_done and self.num_timesteps >= p
                             for p in self.calib_probes))
        bc_aux_applied = False
        adapter_exposure = (
            self._record_bc_aux_rollout_exposure()
            if self._bc_aux_circuit_spec is not None else None)
        if self.bc_aux_lambda > 0:
            aux_call_index = int(self._bc_aux_train_calls)
            self._bc_aux_train_calls += 1
            # 每个 rollout 更新至多消费一个辅助 batch（旧实现每个
            # minibatch×epoch 约 80 批）；校准点同样保证包含生产辅助项。
            bc_aux_due = (
                aux_call_index % _BC_AUX_UPDATE_EVERY == 0 or calib_due)
        else:
            bc_aux_due = False
            if self._bc_aux_circuit_spec is not None:
                self._bc_aux_train_calls += 1

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                distill_ce_for_calib = loss.new_zeros(())
                teacher_entropy_for_calib = None
                distill_kl_for_calib = None
                distill_tv_for_calib = None
                diverge_for_calib = 0.0

                # ===== 皮筋(v24 唯一插入段;β=0 整段跳过 → G-KL-B 逐位等价)=====
                if effective_distill_beta > 0:
                    if self.teacher is None:
                        raise RuntimeError("β>0 但教师未挂载(fail-loud 条款)")
                    distill_masks = _legacy_distillation_masks(
                        rollout_data.action_masks)
                    with th.no_grad():
                        t_probs = self._teacher_probs(
                            rollout_data.observations, distill_masks)
                    student_raw_logits = self._student_distillation_logits(
                        rollout_data.observations)
                    logp_all = _masked_log_softmax_from_raw(
                        student_raw_logits, distill_masks)
                    distill_ce = -(t_probs * logp_all).sum(dim=-1).mean()
                    distill_ce_for_calib = distill_ce
                    loss = loss + effective_distill_beta * distill_ce
                    distill_ces.append(distill_ce.item())
                    with th.no_grad():
                        t_logp = th.where(
                            t_probs > 0.0,
                            th.log(t_probs),
                            th.zeros_like(t_probs),
                        )
                        teacher_entropy = -(
                            t_probs * t_logp).sum(dim=-1).mean()
                        distill_kl = (
                            t_probs * (t_logp - logp_all)
                        ).sum(dim=-1).mean()
                        distill_tv = 0.5 * th.abs(
                            t_probs - logp_all.exp()
                        ).sum(dim=-1).mean()
                        teacher_entropy_for_calib = teacher_entropy
                        distill_kl_for_calib = distill_kl
                        distill_tv_for_calib = distill_tv
                        teacher_entropies.append(
                            teacher_entropy.item())
                        distill_kls.append(distill_kl.item())
                        distill_tvs.append(distill_tv.item())
                        dv = (logp_all.argmax(-1) != t_probs.argmax(-1)).float().mean().item()
                        diverge_for_calib = dv
                        diverges.append(dv)
                        t_confs.append(t_probs.max(dim=-1).values.mean().item())
                # ===== 皮筋段结束 =====

                # ===== rev5 辅助探针项（这里只窥视，不在 PPO minibatch 更新）=====
                bc_aux_ce = None
                probe_total_loss = loss
                if self.bc_aux_lambda > 0 and bc_aux_due and calib_due:
                    bc_aux_ce = self._peek_bc_aux_ce_loss()
                    # G-CAL 继续报告“若仍联合更新”的完整梯度，专门暴露
                    # PPO/KING 与 a12 目标的方向冲突；真实更新在 epochs 后
                    # 由独立 step 执行，避免共同裁剪把辅助方向抵消。
                    probe_total_loss = loss + self.bc_aux_lambda * bc_aux_ce
                # ===== 辅助探针段结束 =====

                # G-CAL 必须放在两项附加损失装配完之后；旧位置看不到 λaux、
                # total gradient 或 a12 头，是 current funcaux 假 PASS 的直接缝。
                if calib_due:
                    self._calib_probe(
                        policy_loss, distill_ce_for_calib,
                        diverge_for_calib,
                        effective_distill_beta=
                            effective_distill_beta,
                        teacher_entropy=
                            teacher_entropy_for_calib,
                        distill_kl=distill_kl_for_calib,
                        distill_tv=distill_tv_for_calib,
                        total_loss=probe_total_loss,
                        bc_aux_loss=bc_aux_ce)
                    for p in self.calib_probes:
                        if self.num_timesteps >= p:
                            self._calib_done.add(p)
                    calib_due = False

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                # 主动 G-CAL 已裁决停腿时，连当前 minibatch 也不应再更新。
                # 原实现要继续跑完本轮多个 epoch，到下一次 rollout 的
                # callback 才停，裁决后仍可改变权重数百次。
                if self._calib_tripped:
                    continue_training = False
                    break

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                if worker_pg_receipt is not None:
                    pure_actor_gradients = (
                        _finite_autograd_gradients(
                            policy_loss,
                            worker_pg_actor_parameters,
                            retain_graph=True,
                        )
                    )
                    combined_actor_gradients = (
                        _finite_autograd_gradients(
                            loss,
                            worker_pg_actor_parameters,
                            retain_graph=True,
                        )
                    )
                    pure_ppo_actor_grad_norms.append(
                        _finite_gradient_tuple_norm(
                            pure_actor_gradients))
                    pure_root_gradients = (
                        _select_parameter_gradients(
                            worker_pg_actor_parameters,
                            pure_actor_gradients,
                            worker_pg_root_parameters,
                        )
                    )
                    combined_root_gradients = (
                        _select_parameter_gradients(
                            worker_pg_actor_parameters,
                            combined_actor_gradients,
                            worker_pg_root_parameters,
                        )
                    )
                    pure_context_gradients = (
                        _select_parameter_gradients(
                            worker_pg_actor_parameters,
                            pure_actor_gradients,
                            worker_pg_context_parameters,
                        )
                    )
                    combined_context_gradients = (
                        _select_parameter_gradients(
                            worker_pg_actor_parameters,
                            combined_actor_gradients,
                            worker_pg_context_parameters,
                        )
                    )
                    distill_context_gradients = (
                        _finite_autograd_gradients(
                            effective_distill_beta
                            * distill_ce_for_calib,
                            worker_pg_context_parameters,
                            retain_graph=True,
                        )
                    )
                    pure_ppo_context_encoder_grad_norms.append(
                        _finite_gradient_tuple_norm(
                            _select_parameter_gradients(
                                worker_pg_context_parameters,
                                pure_context_gradients,
                                worker_pg_context_groups["encoder"],
                            )
                        )
                    )
                    pure_ppo_context_interaction_grad_norms.append(
                        _finite_gradient_tuple_norm(
                            _select_parameter_gradients(
                                worker_pg_context_parameters,
                                pure_context_gradients,
                                worker_pg_context_groups["interaction"],
                            )
                        )
                    )
                    pure_ppo_context_output_grad_norms.append(
                        _finite_gradient_tuple_norm(
                            _select_parameter_gradients(
                                worker_pg_context_parameters,
                                pure_context_gradients,
                                worker_pg_context_groups["output"],
                            )
                        )
                    )
                    pure_ppo_context_grad_norms.append(
                        _finite_gradient_tuple_norm(
                            pure_context_gradients)
                    )
                    pure_ppo_root_grad_norms.append(
                        _finite_gradient_tuple_norm(
                            pure_root_gradients)
                    )
                    distill_context_grad_norms.append(
                        _finite_gradient_tuple_norm(
                            distill_context_gradients)
                    )
                    combined_root_dot_pure_ppo.append(
                        _finite_gradient_tuple_dot(
                            combined_root_gradients,
                            pure_root_gradients,
                        )
                    )
                    pure_ppo_root_grad_sq.append(
                        _finite_gradient_tuple_dot(
                            pure_root_gradients,
                            pure_root_gradients,
                        )
                    )
                    combined_context_dot_pure_ppo.append(
                        _finite_gradient_tuple_dot(
                            combined_context_gradients,
                            pure_context_gradients,
                        )
                    )
                    pure_ppo_context_grad_sq.append(
                        _finite_gradient_tuple_dot(
                            pure_context_gradients,
                            pure_context_gradients,
                        )
                    )
                    pure_ppo_root_dot_combat_reward.append(
                        _finite_gradient_tuple_dot(
                            pure_root_gradients,
                            worker_pg_combat_root_gradients,
                        )
                    )
                    pure_ppo_context_dot_combat_reward.append(
                        _finite_gradient_tuple_dot(
                            pure_context_gradients,
                            worker_pg_combat_context_gradients,
                        )
                    )
                clip_record = self._apply_main_ppo_optimizer_step(
                    loss, actor_frozen=actor_frozen_for_rollout)
                if clip_record["actor_counterfactual_norm"] is not None:
                    actor_counterfactual_norms.append(
                        clip_record["actor_counterfactual_norm"])
                    root_counterfactual_norms.append(
                        clip_record["root_counterfactual_norm"])
                    context_counterfactual_norms.append(
                        clip_record["context_counterfactual_norm"])
                    actor_clip_scales.append(
                        clip_record["actor_clip_scale"])
                    root_clip_scales.append(
                        clip_record["root_clip_scale"])
                    context_clip_scales.append(
                        clip_record["context_clip_scale"])
                    actor_preclip_norms.append(
                        clip_record["actor_preclip_norm"])
                    root_preclip_norms.append(
                        clip_record["root_preclip_norm"])
                    context_preclip_norms.append(
                        clip_record["context_preclip_norm"])
                    critic_preclip_norms.append(
                        clip_record["critic_preclip_norm"])
                    actor_grad_clipped += int(
                        clip_record["actor_clipped"])
                    root_grad_clipped += int(
                        clip_record["root_clipped"])
                    context_grad_clipped += int(
                        clip_record["context_clipped"])
                    critic_grad_clipped += int(
                        clip_record["critic_clipped"])
                ppo_optimizer_steps_this_rollout += 1

            self._n_updates += 1
            if not continue_training:
                break

        self._complete_main_ppo_rollout(
            actor_frozen=actor_frozen_for_rollout,
            optimizer_steps=ppo_optimizer_steps_this_rollout)
        worker_pg_actor_delta = (
            tuple(
                parameter.detach() - start
                for parameter, start in zip(
                    worker_pg_actor_parameters,
                    worker_pg_actor_start,
                    strict=True,
                )
            )
            if worker_pg_receipt is not None else ()
        )
        worker_pg_root_delta = (
            _select_parameter_gradients(
                worker_pg_actor_parameters,
                worker_pg_actor_delta,
                worker_pg_root_parameters,
            )
            if worker_pg_receipt is not None else ()
        )
        worker_pg_context_delta = (
            _select_parameter_gradients(
                worker_pg_actor_parameters,
                worker_pg_actor_delta,
                worker_pg_context_parameters,
            )
            if worker_pg_receipt is not None else ()
        )
        worker_pg_actor_delta_l2 = (
            _finite_gradient_tuple_norm(worker_pg_actor_delta)
            if worker_pg_receipt is not None else None
        )
        worker_pg_root_delta_l2 = (
            _finite_gradient_tuple_norm(worker_pg_root_delta)
            if worker_pg_receipt is not None else None
        )
        worker_pg_context_delta_l2 = (
            _finite_gradient_tuple_norm(worker_pg_context_delta)
            if worker_pg_receipt is not None else None
        )
        # Optimizers descend loss gradients.  Positive values therefore mean
        # the *actual*, post-Adam/post-clipping parameter displacement has a
        # component along ``-g_combat_reward`` from rollout collection time.
        worker_pg_delta_root_on_combat_descent = (
            -_finite_gradient_tuple_dot(
                worker_pg_root_delta,
                worker_pg_combat_root_gradients,
            )
            if worker_pg_receipt is not None else None
        )
        worker_pg_delta_context_on_combat_descent = (
            -_finite_gradient_tuple_dot(
                worker_pg_context_delta,
                worker_pg_combat_context_gradients,
            )
            if worker_pg_receipt is not None else None
        )
        worker_pg_delta_actor_on_combat_descent = (
            worker_pg_delta_root_on_combat_descent
            + worker_pg_delta_context_on_combat_descent
            if worker_pg_receipt is not None else None
        )
        self._complete_worker_onpolicy_pg_rollout(
            worker_pg_receipt,
            kl_early_stopped=not continue_training,
            pure_ppo_actor_grad_norms=pure_ppo_actor_grad_norms,
            pure_ppo_root_grad_norms=pure_ppo_root_grad_norms,
            pure_ppo_context_grad_norms=
                pure_ppo_context_grad_norms,
            pure_ppo_context_output_grad_norms=
                pure_ppo_context_output_grad_norms,
            pure_ppo_context_encoder_grad_norms=
                pure_ppo_context_encoder_grad_norms,
            pure_ppo_context_interaction_grad_norms=
                pure_ppo_context_interaction_grad_norms,
            distill_context_grad_norms=
                distill_context_grad_norms,
            combined_root_dot_pure_ppo=
                combined_root_dot_pure_ppo,
            pure_ppo_root_grad_sq=
                pure_ppo_root_grad_sq,
            combined_context_dot_pure_ppo=
                combined_context_dot_pure_ppo,
            pure_ppo_context_grad_sq=
                pure_ppo_context_grad_sq,
            pure_ppo_root_dot_combat_reward=
                pure_ppo_root_dot_combat_reward,
            pure_ppo_context_dot_combat_reward=
                pure_ppo_context_dot_combat_reward,
            combat_reward_root_grad_norm=(
                _finite_gradient_tuple_norm(
                    worker_pg_combat_root_gradients)
                if worker_pg_receipt is not None else None
            ),
            optimizer_delta_actor_l2=worker_pg_actor_delta_l2,
            optimizer_delta_root_l2=worker_pg_root_delta_l2,
            optimizer_delta_context_l2=worker_pg_context_delta_l2,
            optimizer_delta_actor_dot_combat_reward_descent=
                worker_pg_delta_actor_on_combat_descent,
            optimizer_delta_root_dot_combat_reward_descent=
                worker_pg_delta_root_on_combat_descent,
            optimizer_delta_context_dot_combat_reward_descent=
                worker_pg_delta_context_on_combat_descent,
            optimizer_steps=ppo_optimizer_steps_this_rollout)

        # rev5：每个 rollout 至多一次纯辅助 optimizer step。target_kl 只
        # 提前结束 PPO epochs，不应吞掉本轮已注册的辅助更新；只有 G-CAL
        # fail-closed 裁决会禁止它。末步之后再跑行为 monitor，覆盖 target_kl
        # 看不到的 off-policy 跳变。
        if bc_aux_due and not self._calib_tripped:
            bc_aux_ce = self._apply_bc_aux_step()
            bc_aux_ces.append(bc_aux_ce.item())
            bc_aux_applied = True

        if (
            ppo_optimizer_steps_this_rollout > 0
            and not self._calib_tripped
        ):
            self._last_completed_ppo_rollout_steps = int(
                self.num_timesteps)

        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                           self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record(
            "train/ppo_optimizer_steps_completed",
            int(self._ppo_optimizer_steps_completed),
            exclude="tensorboard")
        self.logger.record(
            "train/actor_optimizer_steps_completed",
            int(self._actor_optimizer_steps_completed),
            exclude="tensorboard")
        self.logger.record(
            "train/critic_warmup_optimizer_steps_completed",
            int(self._critic_warmup_optimizer_steps_completed),
            exclude="tensorboard")
        self.logger.record(
            "train/critic_warmup_rollouts_completed",
            int(self._critic_warmup_rollouts_completed),
            exclude="tensorboard")
        self.logger.record(
            "train/critic_warmup_actor_frozen",
            int(actor_frozen_for_rollout))
        self.logger.record(
            "train/worker_onpolicy_pg_joint_rollouts",
            int(self._worker_onpolicy_pg_joint_rollouts),
            exclude="tensorboard")
        self.logger.record(
            "train/worker_onpolicy_pg_qualifying_rollouts",
            int(self._worker_onpolicy_pg_qualifying_rollouts),
            exclude="tensorboard")
        if worker_pg_receipt is not None:
            self.logger.record(
                "train/worker_reward_centered_actor_grad_norm",
                float(worker_pg_receipt[
                    "reward_centered_actor_grad_norm"]))
            self.logger.record(
                "train/pure_ppo_actor_grad_norm",
                float(worker_pg_receipt[
                    "pure_ppo_actor_grad_norm_mean"]))
        if actor_counterfactual_norms:
            count = len(actor_counterfactual_norms)
            self.logger.record(
                "train/actor_counterfactual_grad_norm",
                float(np.mean(actor_counterfactual_norms)))
            self.logger.record(
                "train/root_counterfactual_grad_norm",
                float(np.mean(root_counterfactual_norms)))
            self.logger.record(
                "train/context_counterfactual_grad_norm",
                float(np.mean(context_counterfactual_norms)))
            root_mean = float(np.mean(root_counterfactual_norms))
            context_mean = float(np.mean(
                context_counterfactual_norms))
            self.logger.record(
                "train/context_to_root_grad_ratio",
                context_mean / max(root_mean, 1e-30))
            self.logger.record(
                "train/actor_clip_scale",
                float(np.mean(actor_clip_scales)))
            self.logger.record(
                "train/root_clip_scale",
                float(np.mean(root_clip_scales)))
            self.logger.record(
                "train/context_clip_scale",
                float(np.mean(context_clip_scales)))
            self.logger.record(
                "train/actor_grad_norm",
                float(np.mean(actor_preclip_norms)))
            self.logger.record(
                "train/root_grad_norm",
                float(np.mean(root_preclip_norms)))
            self.logger.record(
                "train/context_grad_norm",
                float(np.mean(context_preclip_norms)))
            self.logger.record(
                "train/critic_grad_norm",
                float(np.mean(critic_preclip_norms)))
            self.logger.record(
                "train/actor_grad_clip_fraction",
                float(actor_grad_clipped / count))
            self.logger.record(
                "train/root_grad_clip_fraction",
                float(root_grad_clipped / count))
            self.logger.record(
                "train/context_grad_clip_fraction",
                float(context_grad_clipped / count))
            self.logger.record(
                "train/critic_grad_clip_fraction",
                float(critic_grad_clipped / count))
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        # v24 皮筋读数(跨 minibatch 均值,预注册 D4:不许只记末批)
        self.logger.record(
            "train/distill_beta_initial", self.distill_beta)
        self.logger.record(
            "train/distill_beta", effective_distill_beta)
        self.logger.record(
            "train/distill_actor_rollouts_completed",
            int(self._distill_actor_rollouts_completed),
            exclude="tensorboard")
        if distill_ces:
            self.logger.record("train/distill_ce", float(np.mean(distill_ces)))
            self.logger.record(
                "train/teacher_entropy",
                float(np.mean(teacher_entropies)))
            self.logger.record(
                "train/distill_kl", float(np.mean(distill_kls)))
            self.logger.record(
                "train/distill_tv", float(np.mean(distill_tvs)))
            self.logger.record("train/teacher_diverge", float(np.mean(diverges)))
            self.logger.record("train/teacher_top1_conf", float(np.mean(t_confs)))
            self._last_distill_ce = float(np.mean(distill_ces))
            self._last_teacher_entropy = float(
                np.mean(teacher_entropies))
            self._last_distill_kl = float(np.mean(distill_kls))
            self._last_distill_tv = float(np.mean(distill_tvs))
            self._last_diverge = float(np.mean(diverges))
        else:
            self._last_distill_ce = None    # β=0 腿:哨兵行报 null,双簿对账干净
            self._last_teacher_entropy = None
            self._last_distill_kl = None
            self._last_distill_tv = None
            self._last_diverge = None
        # E3 ④乙读数(跨 minibatch 均值,承皮筋"不许只记末批"口径)
        self.logger.record("train/bc_aux_lambda", self.bc_aux_lambda)
        self.logger.record("train/bc_aux_applied", int(bc_aux_applied))
        self.logger.record("train/bc_aux_train_calls",
                           int(self._bc_aux_train_calls))
        if bc_aux_ces:
            self.logger.record("train/bc_aux_ce", float(np.mean(bc_aux_ces)))
            for key, value in (self._last_bc_aux_parts or {}).items():
                self.logger.record(f"train/bc_aux_{key}", value)
            self._last_bc_aux_ce = float(np.mean(bc_aux_ces))
        else:
            self._last_bc_aux_ce = None
        # G-CAL 稀疏探针之外，每个完整 PPO rollout 更新后都读固定 bank。
        # 这里只检查过度饮药/a13 挤出的上界；最终原始 held-out 完整门由
        # train_ppo 发布事务执行。
        if self.bc_aux_lambda > 0:
            self._bc_aux_rollout_monitor()
        elif self._bc_aux_circuit_spec is not None:
            projection = self._project_bc_aux_adapter_weight()
            self.logger.record(
                "train/bc_aux_gate_bias",
                projection["gate_bias"])
            self.logger.record(
                "train/bc_aux_gate_weight_l2",
                projection["gate_weight_l2"])
            for key, value in (adapter_exposure or {}).items():
                self.logger.record(f"train/bc_aux_{key}", value)
            self._bc_aux_rollout_monitor()
