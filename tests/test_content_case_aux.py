"""内容案 E3 ④乙 辅助示范通路之自包含快速回归(PREREG-内容案-课⑤x④乙 E3/E7
对应件;不启动引擎/训练产物)。

覆盖(E3 施工面逐字):
- CLI 两旗解析与互不强制(单独给任一旗不强制对方;在位谓词 = λ_bc>0 ∧ demos);
- λ_bc=0 之零侵入(不加载/不采样/不进损失图;λ=0 双模型 policy+optimizer
  torch.equal,辅助分支实测零调用);
- v2 demos 专用验证器(镜像断言按世代分别成文:masks 键/形状/dtype、v2 禁 11
  允 12、标签-掩码一致)与 v1 面回归零破坏(v1 常量/验证器/加载器原封);
- 12 类子集过滤与图纸数据面断言(12 类示范对 m[12]=True);
- 辅助 CE 前向消费逐样本 masks(独 12 掩码 → CE 精确 0);
- demo minibatch 专用 rng 流(E1③ 形制固定偏移派生、与 p_skip 流族/训练种子域
  构造性不相交、在位/不在位两跑训练 RNG 状态轨迹逐点相等);
- 12 头梯度非零(小模型 autograd 实测 + 真 train() 指纹:λ>0 行 12 必变、
  λ=0 行 12 位级不变——反 P2 构造性零通路);
- 锚教师/β/锚公式零触碰之源码钉(圈 3/圈 5 双引)。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import gymnasium as gym
import numpy as np
import torch as th

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "python"))

from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

import train_ppo  # noqa: E402
from diablogym.worker_env import (  # noqa: E402
    legacy_worker_policy_observation_view,
)
from leashed_ppo import (  # noqa: E402
    _BC_AUX_SEED_OFFSET,
    _legacy_distillation_masks,
    _legacy_worker_observation_view,
    _masked_log_softmax_from_raw,
    A12MixtureMaskableActorCriticPolicy,
    LeashedMaskablePPO,
    build_teacher,
    derive_bc_aux_rng,
)
from train_ppo import (  # noqa: E402
    _BC_AUX_MAIN_LAMBDA,
    _BC_AUX_OBJECTIVE_REVISION,
    _BC_AUX_UPDATE_EVERY,
    _BC_V2_COLLECTION_EPISODES,
    _BC_V2_DEMOS_SCHEMA_VERSION,
    _BC_V2_REPORT_SCHEMA_VERSION,
    _BC_PASS_KEYS,
    _BC_REPORT_SCHEMA_VERSION,
    _WORKER_BC_FORBIDDEN_ACTIONS,
    _WORKER_BC_V2_FORBIDDEN_ACTIONS,
    _bc_aux_active,
    _bc_aux_circuit_spec,
    _bc_aux_structural_active,
    _bc_v2_holdout_indices,
    _bc_v2_training_indices,
    _bc_aux_liveness_call_plan,
    _build_bc_aux_training_bank,
    _persistent_bc_aux_root_anchor,
    _policy_head_snapshot,
    _expand_policy_with_bc_aux_circuit,
    _reset_policy_optimizer,
    _publish_model_final_with_bc_aux_gate,
    _run_bc_aux_liveness_preflight,
    _simulate_bc_aux_circuit_liveness,
    _simulate_bc_aux_liveness,
    bc_aux_behavior_metrics,
    _implementation_bundle_sha256,
    _filter_bc_aux_demo_pairs,
    _load_bc_aux_demos_v2,
    _parse_bc_aux_demos_v2,
    _load_dry_anchor_demos,
)
from eval_contract import PROTOCOL_VERSION  # noqa: E402
from diablogym.worker_env import (  # noqa: E402
    BC_RESERVED_SEED_RANGES,
    _P_SKIP_SEED_OFFSET,
    _derive_p_skip_rng,
    is_reserved_train_seed,
    sample_train_seed,
)

TRAIN_PPO = ROOT / "train" / "train_ppo.py"
LEASHED_PPO = ROOT / "train" / "leashed_ppo.py"
BC_WORKER = ROOT / "train" / "bc_worker.py"
CANONICAL_MANAGER = (
    ROOT / "train" / "models" / "v22-h-manager" / "policy.npz")
CANONICAL_MANAGER_SHA = hashlib.sha256(
    CANONICAL_MANAGER.read_bytes()).hexdigest()


def _run_cli(*extra_args):
    return subprocess.run(
        [sys.executable, str(TRAIN_PPO), *extra_args],
        text=True, capture_output=True, check=False)


def _ns(**kw):
    import types

    base = dict(
        bc_aux_lambda=0.0, bc_aux_demos=None, bc_aux_graft=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _write_v2_npz(path, n=8, labels=None, masks=None, obs_dim=298,
                  episode_ids=None, x_dtype=np.float32,
                  observations=None, **meta_override):
    x = (np.zeros((n, obs_dim), dtype=x_dtype)
         if observations is None
         else np.asarray(observations, dtype=x_dtype))
    if observations is None:
        x[: n // 2, min(297, obs_dim - 1)] = 1.0
    y = np.asarray(
        labels if labels is not None
        else ([12, 9, 9, 9] * ((n + 3) // 4))[:n],
        dtype=np.int64)
    if observations is None and obs_dim > 0:
        x[:, 0] = 0.8
        x[y == 12, 0] = 0.55
    if masks is None:
        masks = np.ones((n, 15), dtype=bool)
        masks[:, 11] = False
    ep = np.asarray(
        episode_ids if episode_ids is not None
        else [0] * (n // 2) + [1] * (n - n // 2), dtype=np.int64)
    generator = ROOT / "train" / "bc_worker.py"
    metadata = {
        "schema_version": np.asarray(_BC_V2_DEMOS_SCHEMA_VERSION),
        "protocol_version": np.asarray(PROTOCOL_VERSION, dtype=np.int64),
        "implementation_sha256": np.asarray(_implementation_bundle_sha256()),
        "generator_sha256": np.asarray(
            hashlib.sha256(generator.read_bytes()).hexdigest()),
        "manager_npz_sha256": np.asarray(CANONICAL_MANAGER_SHA),
        "teacher_generation": np.asarray(2, dtype=np.int64),
        "preventive_threshold": np.asarray(0.65, dtype=np.float64),
    }
    metadata.update(meta_override)
    np.savez_compressed(path, X=x, Y=y, episode_id=ep, masks=masks,
                        **metadata)
    return x, y, ep, masks


def _write_final_holdout_marker(
        out_dir, generation, seeds, report):
    """为已提交测试 bundle 写与生产 consumer 同构的 final one-shot 证据。"""
    marker, spec, pool_sha256 = train_ppo._bc_final_holdout_marker_path(
        out_dir, generation, seeds)
    provenance_keys = {
        "schema_version", "protocol_version", "implementation_sha256",
        "generator_sha256", "manager_npz_sha256",
    }
    if generation == 2:
        provenance_keys |= {"teacher_generation", "preventive_threshold"}
    record = {
        **spec,
        "pool_sha256": pool_sha256,
        "marker_schema_version":
            train_ppo._BC_FINAL_HOLDOUT_MARKER_SCHEMA,
        "started_at_ns": 1,
        "provenance": {
            key: report[key] for key in provenance_keys
        },
        "consumption_stage": "before_pool_collection",
    }
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":")).encode()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(payload)
    report["final_pool_sha256"] = pool_sha256
    report["final_holdout_marker_sha256"] = hashlib.sha256(
        payload).hexdigest()
    return marker


def _write_v2_pass_bundle(path):
    """构造足以通过生产 loader 的固定 384-episode 已提交 bundle。"""
    episodes = np.asarray(_BC_V2_COLLECTION_EPISODES, dtype=np.int64)
    groups = np.repeat(episodes, 80)
    n = len(groups)
    labels = np.full(n, 9, dtype=np.int64)
    heldout = _bc_v2_holdout_indices(groups)
    fit, _ = train_ppo._bc_v2_fit_validation_indices(groups)
    _, validation = train_ppo._bc_v2_fit_validation_indices(groups)
    positive = np.concatenate([fit[:120], validation[:1], heldout[:1]])
    labels[positive] = 12
    observations = np.zeros((n, 298), dtype=np.float32)
    observations[:, 0] = 0.8
    observations[:, 1] = -1.0
    observations[positive, 0] = 0.55
    observations[positive, 1] = 1.0
    positive_set = set(int(i) for i in positive)
    post_drink: list[int] = []
    for domain in (fit, validation, heldout):
        domain_positive = int((labels[domain] == 12).sum())
        candidates = [
            int(i) for i in domain
            if int(i) not in positive_set
        ][:domain_positive]
        post_drink.extend(candidates)
        observations[candidates, 0] = 0.55
        observations[candidates, 297] = -1.5
    masks = np.ones((n, 15), dtype=np.bool_)
    masks[:, 11] = False
    masks[np.asarray(post_drink, dtype=np.int64), 12] = False
    x, y, ep, masks = _write_v2_npz(
        path, n=n, labels=labels, masks=masks, episode_ids=groups,
        observations=observations)

    policy_sd = {
        "mlp_extractor.policy_net.0.weight":
            th.zeros((64, 298), dtype=th.float32),
        "mlp_extractor.policy_net.0.bias":
            th.zeros(64, dtype=th.float32),
        "mlp_extractor.policy_net.2.weight":
            th.zeros((64, 64), dtype=th.float32),
        "mlp_extractor.policy_net.2.bias":
            th.zeros(64, dtype=th.float32),
        "action_net.weight":
            th.zeros((15, 64), dtype=th.float32),
        "action_net.bias":
            th.full((15,), -5.0, dtype=th.float32),
    }
    policy_sd["mlp_extractor.policy_net.0.weight"][0, 1] = 5.0
    policy_sd["mlp_extractor.policy_net.2.weight"][0, 0] = 5.0
    policy_sd["action_net.weight"][12, 0] = 5.0
    policy_sd["action_net.weight"][9, 0] = -5.0
    policy_sd["action_net.bias"][12] = 0.0
    policy_sd["action_net.bias"][9] = 0.0
    policy = path.with_name("policy_sd.pt")
    th.save(policy_sd, policy)
    behavior = bc_aux_behavior_metrics(
        policy_sd, x, y, ep, masks, heldout_only=True)
    behavior_gate = train_ppo.bc_aux_behavior_gate(
        behavior, require_teacher_recall=False)
    if behavior_gate["verdict"] != "PASS":
        raise AssertionError(
            f"测试 bundle 行为夹具未过生产门:{behavior_gate['reasons']}")
    with th.no_grad():
        logits = train_ppo._policy_logits_from_sb3_state_dict(
            policy_sd, x[heldout])
        pred = th.where(
            th.from_numpy(masks[heldout]), logits,
            th.full_like(logits, -1e8)
        ).argmax(dim=-1).cpu().numpy()
    heldout_y = y[heldout]
    heldout_top1 = float((pred == heldout_y).mean())
    counts = np.bincount(y, minlength=15)
    class_recalls = {}
    for action in np.flatnonzero(counts >= 300):
        selected = heldout_y == action
        class_recalls[str(int(action))] = (
            float((pred[selected] == action).mean())
            if selected.any() else 0.0)
    _, validation = train_ppo._bc_v2_fit_validation_indices(ep)
    fit_behavior = bc_aux_behavior_metrics(
        policy_sd, x[fit], y[fit], ep[fit], masks[fit],
        heldout_only=False)
    calibration = {
        "schema_version": train_ppo._A12_CALIBRATION_SCHEMA_VERSION,
        "fit_scope": "nested-fit-episodes-only",
        "fit_pairs": int(len(fit)),
        "fit_episodes": int(len(np.unique(ep[fit]))),
        "validation_pairs_excluded": int(len(validation)),
        "validation_episodes_excluded":
            int(len(np.unique(ep[validation]))),
        "final_heldout_pairs_excluded": int(len(heldout)),
        "final_heldout_episodes_excluded":
            int(len(np.unique(ep[heldout]))),
        "hp_low": 0.5,
        "hp_high": 0.65,
        "hp_feature": train_ppo._A12_CALIBRATION_HP_FEATURE,
        "drink_latch_feature":
            train_ppo._A12_CALIBRATION_DRINK_LATCH_FEATURE,
        "predicate": train_ppo._A12_CALIBRATION_PREDICATE,
        "bias_12": float(policy_sd["action_net.bias"][12]),
        "target_recall_12":
            train_ppo._A12_CALIBRATION_TRAIN_RECALL_TARGET,
        "fit_metrics": {
            key: fit_behavior[key]
            for key in (
                "tp", "fp", "precision_12", "recall_12", "fpr_12",
                "predicted_share_12", "high_hp_false_drink_rate",
                "legal_negative_probability_12_mean",
                "legal_negative_probability_12_max",
                "a13_spillover",
            )
        },
    }
    report = {
        "schema_version": _BC_V2_REPORT_SCHEMA_VERSION,
        "data_gate": "PASS",
        "protocol_version": PROTOCOL_VERSION,
        "implementation_sha256": _implementation_bundle_sha256(),
        "generator_sha256": hashlib.sha256(
            (ROOT / "train" / "bc_worker.py").read_bytes()).hexdigest(),
        "manager_npz_sha256": hashlib.sha256(
            (ROOT / "train" / "models" / "v22-h-manager"
             / "policy.npz").read_bytes()).hexdigest(),
        "teacher_generation": 2,
        "preventive_threshold": 0.65,
        "pairs": n,
        "collection_episodes": 384,
        "held_out_pairs": int(len(heldout)),
        "held_out_episodes": [
            int(v) for v in sorted(np.unique(ep[heldout]))],
        "held_out_top1": heldout_top1,
        "class_recalls": class_recalls,
        "class_weighted_retry": False,
        "n12": int((y == 12).sum()),
        "n12_gate_min": 122,
        "n12_by_episode": {
            str(int(episode)): int((y[ep == episode] == 12).sum())
            for episode in np.unique(ep)
            if bool((y[ep == episode] == 12).any())
        },
        "recall_12": round(float(behavior["recall_12"]), 4),
        "recall_12_denominator": int(behavior["true_a12"]),
        "recall_12_gate_min": train_ppo._BC_V2_TEACHER_RECALL_MIN,
        "class_share_12": float((y == 12).mean()),
        "class_share_13": float((y == 13).mean()),
        "belt_economy": {
            "belt_mean_at_a12": 1.0,
            "belt_mean_overall": 1.0,
            "a13_pairs": int((y == 13).sum()),
        },
        "class_weights": {"9": 1.0, "12": 1.0},
        "demos_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "a12_behavior": behavior,
        "a12_behavior_gate": behavior_gate,
        "a12_calibration": calibration,
    }
    _write_final_holdout_marker(
        path.parent, 2, _BC_V2_COLLECTION_EPISODES, report)
    if set(report) != set(train_ppo._BC_V2_PASS_KEYS):
        raise AssertionError(
            "测试 bundle 回执键漂移:"
            f" missing={sorted(set(train_ppo._BC_V2_PASS_KEYS) - set(report))},"
            f" extra={sorted(set(report) - set(train_ppo._BC_V2_PASS_KEYS))}")
    path.with_name("bc_report_v2.json").write_text(json.dumps(report))
    return x, y, ep, masks


class TinyMasked15Env(gym.Env):
    """免引擎 15 动作微环境(Discrete(15) 使"12 头"真实存在)。"""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,),
                                       dtype=np.float32)
    action_space = gym.spaces.Discrete(15)
    metadata = {"render_modes": []}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}

    def action_masks(self):
        mask = np.ones(15, dtype=bool)
        mask[11] = mask[12] = False
        return mask


class TinyMasked298Env(TinyMasked15Env):
    """与正式 worker 同观测宽度，专测 legacy worker 兼容视图。"""

    observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(298,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return np.zeros(298, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(298, dtype=np.float32), 0.0, False, False, {}


def _make_model(lam=0.0, seed=7):
    env = DummyVecEnv([TinyMasked15Env])
    return LeashedMaskablePPO(
        "MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1,
        gamma=1.0, ent_coef=0.005, distill_beta=0.0, bc_aux_lambda=lam,
        seed=seed, device="cpu", verbose=0)


def _make_298_model(lam=0.0, seed=7):
    env = DummyVecEnv([TinyMasked298Env])
    return LeashedMaskablePPO(
        "MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1,
        gamma=1.0, ent_coef=0.005, distill_beta=0.0, bc_aux_lambda=lam,
        seed=seed, device="cpu", verbose=0)


def _fill_buffer(model, seed=11):
    # rollout 面 12 恒掩 → PPO 自身对 action_net 第 12 行零梯度(Adam 零梯度
    # 零更新)——12 头位级变动只可能来自 ④乙 辅助通路(反 P2 指纹构造)。
    rng = np.random.default_rng(seed)
    buf = model.rollout_buffer
    buf.reset()
    for index in range(model.n_steps):
        obs = rng.standard_normal((1, 4)).astype(np.float32)
        mask = np.ones((1, 15), dtype=bool)
        mask[:, 11] = mask[:, 12] = False
        action = np.asarray([int(rng.choice([9, 10, 13, 14]))])
        buf.add(obs, action, np.asarray([float(rng.standard_normal())]),
                np.asarray([index % 5 == 0]), th.zeros(1), th.zeros(1),
                action_masks=mask)
    buf.compute_returns_and_advantage(last_values=th.zeros(1),
                                      dones=np.zeros(1))


def _fill_298_exposure_buffer(model, actions):
    """Populate only the rollout fields consumed by the a12 receipt audit."""
    actions = list(actions)
    if len(actions) != model.n_steps:
        raise ValueError("actions 必须精确覆盖一个 rollout")
    if model.ep_info_buffer is None:
        model._setup_learn(total_timesteps=model.n_steps)
    buf = model.rollout_buffer
    buf.reset()
    for index, action in enumerate(actions):
        obs = np.zeros((1, 298), dtype=np.float32)
        obs[:, 0] = 0.55
        obs[:, 297] = float(index) / max(1, len(actions))
        mask = np.ones((1, 15), dtype=bool)
        mask[:, 11] = False
        buf.add(
            obs, np.asarray([action], dtype=np.int64),
            np.zeros(1, dtype=np.float32),
            np.asarray([index == 0], dtype=bool),
            th.zeros(1), th.zeros(1), action_masks=mask)


def _demo_bank(n=16, obs_dim=4, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, obs_dim)).astype(np.float32)
    positive_n = max(1, n // 4)
    y = np.full(n, 9, dtype=np.int64)
    y[:positive_n] = 12
    masks = np.ones((n, 15), dtype=bool)
    masks[:, 11] = False
    return x, y, masks


def _tree_equal(a, b):
    if isinstance(a, th.Tensor):
        return isinstance(b, th.Tensor) and th.equal(a, b)
    if isinstance(a, dict):
        return (isinstance(b, dict) and set(a) == set(b)
                and all(_tree_equal(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return (isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(_tree_equal(x, y) for x, y in zip(a, b)))
    return a == b


def _tiny_policy_sd(*, collapse=False, anchor=False):
    """2→1→1→15 的合法六张量；feature1 区分稀有 a12。"""
    sd = {
        "mlp_extractor.policy_net.0.weight":
            th.tensor([[0.0, 5.0]], dtype=th.float32),
        "mlp_extractor.policy_net.0.bias": th.zeros(1),
        "mlp_extractor.policy_net.2.weight":
            th.tensor([[5.0]], dtype=th.float32),
        "mlp_extractor.policy_net.2.bias": th.zeros(1),
        "action_net.weight": th.zeros((15, 1), dtype=th.float32),
        "action_net.bias": th.full((15,), -5.0, dtype=th.float32),
    }
    if collapse:
        sd["action_net.bias"][12] = 10.0
        sd["action_net.bias"][9] = 0.0
    elif anchor:
        sd["action_net.weight"].zero_()
        sd["action_net.bias"][9] = 1.0
    else:
        sd["action_net.weight"][12, 0] = 5.0
        sd["action_net.weight"][9, 0] = -5.0
        sd["action_net.bias"][12] = 0.0
        sd["action_net.bias"][9] = 0.0
    return sd


def _contextual_anchor_sd():
    """298→64→64→15 的 V28 形根锚；action9 为稳定非 a12 首选。"""
    state = {
        "mlp_extractor.policy_net.0.weight":
            th.zeros((64, 298), dtype=th.float32),
        "mlp_extractor.policy_net.0.bias":
            th.zeros(64, dtype=th.float32),
        "mlp_extractor.policy_net.2.weight":
            th.zeros((64, 64), dtype=th.float32),
        "mlp_extractor.policy_net.2.bias":
            th.zeros(64, dtype=th.float32),
        "action_net.weight":
            th.zeros((15, 64), dtype=th.float32),
        "action_net.bias":
            th.full((15,), -5.0, dtype=th.float32),
    }
    state["action_net.bias"][9] = 0.0
    return state


def _contextual_policy_sd(*, gate_bias=1.0, gate_coefficients=None):
    """把测试根锚无损扩为 rev9 68 宽，并只设置五个门控参数。"""
    anchor = _contextual_anchor_sd()
    state = {
        "mlp_extractor.policy_net.0.weight":
            th.zeros((68, 298), dtype=th.float32),
        "mlp_extractor.policy_net.0.bias":
            th.zeros(68, dtype=th.float32),
        "mlp_extractor.policy_net.2.weight":
            th.zeros((68, 68), dtype=th.float32),
        "mlp_extractor.policy_net.2.bias":
            th.zeros(68, dtype=th.float32),
        "action_net.weight":
            th.zeros((15, 68), dtype=th.float32),
        "action_net.bias": anchor["action_net.bias"].clone(),
    }
    state["mlp_extractor.policy_net.0.weight"][:64].copy_(
        anchor["mlp_extractor.policy_net.0.weight"])
    state["mlp_extractor.policy_net.0.bias"][:64].copy_(
        anchor["mlp_extractor.policy_net.0.bias"])
    state["mlp_extractor.policy_net.2.weight"][:64, :64].copy_(
        anchor["mlp_extractor.policy_net.2.weight"])
    state["mlp_extractor.policy_net.2.bias"][:64].copy_(
        anchor["mlp_extractor.policy_net.2.bias"])
    state["action_net.weight"][:, :64].copy_(
        anchor["action_net.weight"])
    spec = _bc_aux_circuit_spec()
    coefficients = (
        [0.0] * 4 if gate_coefficients is None
        else list(gate_coefficients))
    state["action_net.weight"][
        int(spec["action_index"]),
        list(spec["gate_parameter_columns"]),
    ] = th.as_tensor(coefficients, dtype=th.float32)
    state["action_net.bias"][int(spec["action_index"])] = float(gate_bias)
    return state


def _feature297_sensitive_sd(template=None):
    """策略方向只由 legacy exhausted feature297 决定：负→12，正→9。"""
    if template is None:
        state = {
            "mlp_extractor.policy_net.0.weight":
                th.zeros((1, 298), dtype=th.float32),
            "mlp_extractor.policy_net.0.bias": th.zeros(1),
            "mlp_extractor.policy_net.2.weight":
                th.zeros((1, 1), dtype=th.float32),
            "mlp_extractor.policy_net.2.bias": th.zeros(1),
            "action_net.weight": th.zeros((15, 1), dtype=th.float32),
            "action_net.bias": th.zeros(15, dtype=th.float32),
        }
    else:
        state = {
            key: template[key].detach().cpu().clone()
            for key in train_ppo._POLICY_HEAD_KEYS
        }
        for value in state.values():
            value.zero_()
    state["mlp_extractor.policy_net.0.weight"][0, 297] = 1.0
    state["mlp_extractor.policy_net.2.weight"][0, 0] = 1.0
    state["action_net.weight"][9, 0] = 5.0
    state["action_net.weight"][12, 0] = -5.0
    return state


class _FakePolicy:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


class _FakePublishModel:
    def __init__(
            self, state, step=1024, *, eligible_states=400,
            requested_a12=10, sampled_a12=10, rejected_a12=0,
            unexpected_sampled_a12=0,
            expected_a12_mass=20.0):
        self.policy = _FakePolicy(state)
        self.num_timesteps = step
        self._bc_aux_eligible_states = eligible_states
        self._bc_aux_requested_a12 = requested_a12
        self._bc_aux_sampled_a12 = sampled_a12
        self._bc_aux_rejected_a12 = rejected_a12
        self._bc_aux_unexpected_sampled_a12 = unexpected_sampled_a12
        self._bc_aux_expected_a12_mass = expected_a12_mass


def _publication_provenance(step=1024):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "implementation_sha256": "b" * 64,
        "manager_npz_sha256": "c" * 64,
        "resume_checkpoint_sha256": "d" * 64,
        "teacher_sha256": "e" * 64,
        "bc_aux_demos_sha256": "a" * 64,
        "bc_aux_liveness_preflight_sha256": "f" * 64,
        "training_contract_sha256": "1" * 64,
        "start_steps": 0,
        "target_global_steps": step,
        "seed": 304000,
        "optimizer_reset": True,
        "target_kl": 0.02,
        "distill_beta": 0.015625,
        "bc_aux_lambda": 0.0,
        "bc_aux_mode": "expanded-trainable-a12-contextual-mixture",
        "calib_record_only": False,
    }


def _behavior_fixture():
    # 每局 1000 对、恰 1 个 a12；无论 rng(23) 留出哪一整局，行为门均可判。
    # 至少 20 局，使固定 10% final holdout 含两局，能验证确定性可部署门。
    groups = np.repeat(np.arange(20, dtype=np.int64), 1000)
    n = len(groups)
    x = np.zeros((n, 298), dtype=np.float32)
    x[:, 0] = 0.8       # high-HP legal negatives
    y = np.full(n, 9, dtype=np.int64)
    for episode in range(20):
        index = episode * 1000
        x[index, 0] = 0.55
        x[index, 8] = 1.0
        x[index, 9] = 0.25
        x[index, 286] = 0.5
        y[index] = 12
    masks = np.ones((n, 15), dtype=bool)
    masks[:, 11] = False
    return x, y, groups, masks


class BcAuxActivationPredicateTests(unittest.TestCase):
    """rev6:正式在位谓词由 graft+demos 构成，旧 λ 路只用于迁移报错。"""

    def test_predicate_truth_table(self):
        self.assertFalse(_bc_aux_active(_ns()))
        self.assertFalse(_bc_aux_active(_ns(bc_aux_lambda=0.5)))
        self.assertFalse(_bc_aux_active(_ns(bc_aux_demos="demos.npz")))
        self.assertFalse(_bc_aux_active(_ns(bc_aux_graft=True)))
        self.assertTrue(_bc_aux_active(
            _ns(bc_aux_lambda=0.5, bc_aux_demos="demos.npz")))
        structural = _ns(
            bc_aux_graft=True, bc_aux_demos="demos.npz",
            bc_aux_lambda=0.0)
        self.assertTrue(_bc_aux_active(structural))
        self.assertTrue(_bc_aux_structural_active(structural))
        self.assertFalse(_bc_aux_structural_active(
            _ns(bc_aux_lambda=0.5, bc_aux_demos="demos.npz")))

    def test_structural_contract_keeps_zero_lambda_and_68_wide_actor(self):
        # 历史常量仍可读以便旧合同给出精确迁移错误；结构路径必须 λ=0。
        self.assertEqual(_BC_AUX_MAIN_LAMBDA, 0.015625)
        self.assertEqual(_BC_AUX_OBJECTIVE_REVISION, 11)
        self.assertEqual(train_ppo._CONTRACT_REVISION, 25)
        self.assertEqual(
            train_ppo._BC_AUX_BEHAVIOR_RECEIPT_SCHEMA_VERSION,
            "bc-aux-behavior/8")
        self.assertEqual(
            train_ppo._BC_AUX_CIRCUIT_KING_SUPPORT,
            "legal-non12-non14-renormalized")
        spec = _bc_aux_circuit_spec()
        self.assertEqual(spec, {
            "schema_version":
                "a12-onpolicy-contextual-mixture-adapter/1",
            "base_width": 64,
            "expanded_width": 68,
            "action_index": 12,
            "gate_feature_indices": [0, 8, 9, 286],
            "gate_parameter_columns": [64, 65, 66, 67],
            "hp_low": 0.5,
            "hp_high": 0.65,
            "boundary_epsilon": 1e-6,
            "initial_probability": 0.05,
            "initial_gate_bias": math.log((0.05 - 0.001) / (0.95 - 0.05)),
            "probability_min": 0.001,
            "probability_max": 0.95,
            "gate_parameter_abs_max": 8.0,
        })


class BcAuxLivenessPreflightTests(unittest.TestCase):
    """环境点火前 aux 必要条件探针只消费 training split，失败不发布。"""

    @staticmethod
    def _raw_fixture():
        rng = np.random.default_rng(81)
        groups = np.repeat(np.arange(20, dtype=np.int64), 10)
        x = rng.standard_normal((len(groups), 4)).astype(np.float32)
        y = np.full(len(groups), 9, dtype=np.int64)
        y[::10] = 12
        masks = np.ones((len(groups), 15), dtype=np.bool_)
        masks[:, 11] = False
        return x, y, groups, masks

    def test_production_leg_maps_exactly_to_244_aux_calls(self):
        plan = _bc_aux_liveness_call_plan(
            244 * 2_048, 512, 4)
        self.assertEqual(plan, {
            "rollout_quantum": 2_048,
            "train_calls": 244,
            "aux_optimizer_calls": 244,
            "update_every": 1,
        })

        # 不只测除法：隔离模型实际调用计数也须逐字等于计划。
        model = _make_model(lam=0.015625, seed=83)
        x, y, groups, masks = self._raw_fixture()
        result = _simulate_bc_aux_liveness(
            model, bank=_demo_bank(n=64),
            x=x, y=y, episode_id=groups, masks=masks,
            bc_aux_lambda=0.015625, seed=83, call_plan=plan)
        self.assertEqual(
            result["calls"]["actual_aux_optimizer_calls"], 244)
        self.assertEqual(model._bc_aux_train_calls, 244)
        model.get_env().close()

    def test_simulation_evaluation_excludes_every_heldout_episode(self):
        model = _make_model(lam=0.015625, seed=89)
        x, y, groups, masks = self._raw_fixture()
        heldout = _bc_v2_holdout_indices(groups)
        captured = {}
        real_metrics = train_ppo.bc_aux_behavior_metrics

        def capture(policy_sd, cx, cy, cep, cmasks, **kwargs):
            captured["episodes"] = np.unique(cep)
            captured["heldout_only"] = kwargs.get("heldout_only")
            return real_metrics(
                policy_sd, cx, cy, cep, cmasks, **kwargs)

        with mock.patch.object(
                train_ppo, "bc_aux_behavior_metrics",
                side_effect=capture):
            result = _simulate_bc_aux_liveness(
                model, bank=_demo_bank(n=64),
                x=x, y=y, episode_id=groups, masks=masks,
                bc_aux_lambda=0.015625, seed=89,
                call_plan=_bc_aux_liveness_call_plan(8, 8, 1))
        self.assertFalse(captured["heldout_only"])
        self.assertFalse(np.intersect1d(
            captured["episodes"], np.unique(groups[heldout])).size)
        self.assertEqual(result["heldout_rows_consumed"], 0)
        self.assertTrue(result["split"]["episode_disjoint"])
        model.get_env().close()

    def test_failed_preflight_writes_bound_receipt_and_removes_canonical(self):
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            source = _make_model(lam=0.0, seed=97)
            checkpoint = base / "start.zip"
            source.save(checkpoint)
            source.get_env().close()
            payload = checkpoint.read_bytes()
            checkpoint_sha = hashlib.sha256(payload).hexdigest()
            run_dir = base / "run"
            run_dir.mkdir()
            (run_dir / "model_final.zip").write_bytes(b"stale")
            args = types.SimpleNamespace(
                total_steps=8, n_steps=8, num_envs=1,
                bc_aux_lambda=0.015625, seed=97, device="cpu",
                distill_beta=0.015625,
                lr=3e-4, reset_optimizer=False, target_kl=None,
                ent_coef=0.005, gamma=1.0)
            x, y, groups, masks = self._raw_fixture()
            simulated_fail = {
                "status": "FAIL",
                "gate": {"verdict": "FAIL",
                         "reasons": ["training_recall_unreachable"]},
            }
            with mock.patch.object(
                    train_ppo, "_validate_model_recipe"), \
                    mock.patch.object(
                        train_ppo, "_simulate_bc_aux_liveness",
                        return_value=simulated_fail):
                with self.assertRaisesRegex(ValueError, "preflight FAIL"):
                    _run_bc_aux_liveness_preflight(
                        run_dir=run_dir, args=args,
                        resume_checkpoint_bytes=payload,
                        resume_checkpoint_sha256=checkpoint_sha,
                        bank=_demo_bank(n=64),
                        x=x, y=y, episode_id=groups, masks=masks,
                        demos_sha256="a" * 64,
                        manager_npz_sha256="b" * 64,
                        implementation_sha256="c" * 64,
                        batch_size=8)
            self.assertFalse((run_dir / "model_final.zip").exists())
            receipt = json.loads(
                (run_dir / "bc_aux_liveness_preflight.json").read_text())
            self.assertEqual(
                receipt["schema_version"],
                "bc-aux-liveness-preflight/4")
            self.assertEqual(receipt["status"], "FAIL")
            self.assertEqual(
                receipt["inputs"]["resume_checkpoint_sha256"],
                checkpoint_sha)
            self.assertEqual(
                receipt["config"]["aux_optimizer_calls"], 1)


class LegacySceneClockCompatibilityTests(unittest.TestCase):
    """旧 KING/root 只读恢复 v3 belt/exhausted，学生消费原始 v4 观测。"""

    def test_numpy_environment_and_torch_policy_decoders_are_identical(self):
        rng = np.random.default_rng(20260726)
        obs = rng.standard_normal((257, 298)).astype(np.float32)
        # Include every legal packed-belt combination and both latch signs,
        # plus clipping boundaries used by defensive replay/eval paths.
        for row, (heals, free) in enumerate(
                (divmod(index, 9) for index in range(81))):
            obs[row, 286] = heals / 8.0 + free / 128.0
            exhausted = float(row % 2)
            obs[row, 297] = (
                -(1.0 + exhausted) if row % 3 == 0 else exhausted)
        obs[81:85, 286] = np.asarray(
            [-np.inf, -0.25, 1.25, np.inf], dtype=np.float32)
        original = obs.copy()

        numpy_view = legacy_worker_policy_observation_view(obs)
        torch_view = _legacy_worker_observation_view(
            th.from_numpy(obs)).cpu().numpy()

        np.testing.assert_array_equal(numpy_view, torch_view)
        np.testing.assert_array_equal(obs, original)
        self.assertFalse(np.shares_memory(numpy_view, obs))

    def test_legacy_view_decodes_belt_and_exhausted_without_mutation(self):
        obs = th.zeros((3, 298), dtype=th.float32)
        obs[:, 286] = th.tensor([
            3.0 / 8.0 + 5.0 / 128.0, 1.0, 2.0 / 8.0])
        obs[:, 296] = th.tensor([1.0, 0.999, 1.0])
        obs[:, 297] = th.tensor([-1.5, 0.625, 0.0])
        original = obs.clone()
        decoded = _legacy_worker_observation_view(obs)
        th.testing.assert_close(
            decoded[:, 286], th.tensor([3.0 / 8.0, 1.0, 2.0 / 8.0]))
        th.testing.assert_close(
            decoded[:, 297], th.tensor([0.5, 0.625, 0.0]))
        # A saturated no-kill clock is independent of exhausted.
        self.assertEqual(float(decoded[2, 296]), 1.0)
        self.assertEqual(float(decoded[2, 297]), 0.0)
        self.assertTrue(th.equal(obs, original))
        self.assertNotEqual(decoded.data_ptr(), obs.data_ptr())
        small = th.randn(2, 4)
        self.assertIs(_legacy_worker_observation_view(small), small)

    def test_king_decodes_signed_latch_but_caller_observation_stays_signed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "legacy-teacher.pt"
            th.save(_feature297_sensitive_sd(), path)
            teacher = build_teacher(path)
            holder = types.SimpleNamespace(teacher=teacher)
            obs = th.zeros((2, 298), dtype=th.float32)
            obs[:, 286] = 3.0 / 8.0 + 5.0 / 128.0
            obs[:, 296] = 1.0
            obs[:, 297] = th.tensor([-1.5, 0.5])
            original = obs.clone()
            masks = th.ones((2, 15), dtype=th.bool)
            # Feeding the signed latch directly reproduces the legacy/OOD
            # action-12 flip; the KING wrapper must recover action 9 instead.
            self.assertEqual(int(teacher(obs)[0].argmax()), 12)
            probs = LeashedMaskablePPO._teacher_probs(
                holder, obs, masks)
            self.assertEqual(probs.argmax(dim=-1).tolist(), [9, 9])
            self.assertTrue(th.equal(obs, original))

    def test_persistent_root_decodes_but_aux_student_bank_keeps_signed_latch(self):
        env = DummyVecEnv([TinyMasked298Env])
        model = LeashedMaskablePPO(
            "MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1,
            gamma=1.0, distill_beta=0.0, bc_aux_lambda=0.015625,
            seed=37, device="cpu", verbose=0)
        self.addCleanup(env.close)
        model.bc_aux_root_anchor_sd = _feature297_sensitive_sd(
            model.policy.state_dict())
        x = np.zeros((8, 298), dtype=np.float32)
        x[:, 286] = 3.0 / 8.0 + 5.0 / 128.0
        x[:, 296] = 1.0
        x[:, 297] = -1.5
        y = np.full(8, 9, dtype=np.int64)
        y[:2] = 12
        masks = np.ones((8, 15), dtype=np.bool_)
        masks[:, 11] = False
        signed_logits = train_ppo._policy_logits_from_sb3_state_dict(
            model.bc_aux_root_anchor_sd, x)
        self.assertTrue(
            bool((signed_logits.argmax(dim=-1) == 12).all()))

        model.mount_bc_aux_demos(
            x, y, masks, rng=derive_bc_aux_rng(37))
        self.assertTrue(
            bool((model._bc_aux_obs[:, 297].cpu() == -1.5).all()))
        self.assertTrue(bool(
            (model._bc_aux_anchor_probs.argmax(dim=-1).cpu() == 9).all()))

    def test_offline_anchor_metrics_decode_root_only_not_current_student(self):
        state = _feature297_sensitive_sd()
        x = np.zeros((8, 298), dtype=np.float32)
        x[:, 0] = 0.6
        x[:, 286] = 3.0 / 8.0 + 5.0 / 128.0
        x[:, 296] = 1.0
        x[:, 297] = -1.5
        y = np.full(8, 9, dtype=np.int64)
        groups = np.arange(8, dtype=np.int64)
        masks = np.ones((8, 15), dtype=np.bool_)
        masks[:, 11] = False
        metrics = bc_aux_behavior_metrics(
            state, x, y, groups, masks, anchor_sd=state,
            heldout_only=False)
        self.assertEqual(metrics["predicted_a12"], 8)
        self.assertEqual(metrics["anchor"]["argmax_drift"], 1.0)
        action9 = metrics["anchor"]["critical_action_retention"]["9"]
        self.assertEqual(action9["support"], 8)
        self.assertEqual(action9["retained"], 0)


class StructuralAdapterAndKingFirewallTests(unittest.TestCase):
    """与校准阈值无关的结构回归：无损扩宽、持久化及 KING 支持集。"""

    @staticmethod
    def _raw_actor_logits(model, obs):
        with th.no_grad():
            features = model.policy.extract_features(obs)
            latent_pi = model.policy.mlp_extractor.forward_actor(features)
            return model.policy.action_net(latent_pi).detach().cpu()

    def test_expand_preserves_every_non_a12_logit_bitwise_and_roundtrips(self):
        model = _make_298_model(seed=401)
        obs = th.from_numpy(
            np.random.default_rng(401).standard_normal(
                (32, 298)).astype(np.float32))
        obs[:, 297] = th.linspace(-1.9, 0.9, len(obs))
        before = self._raw_actor_logits(model, obs)
        root = _persistent_bc_aux_root_anchor(model)
        root_sha = train_ppo._policy_head_sha256(root)

        spec = _expand_policy_with_bc_aux_circuit(model)
        _reset_policy_optimizer(model, 3e-4)
        self.assertIsInstance(
            model.policy, A12MixtureMaskableActorCriticPolicy)
        self.assertIs(
            model.policy_class, A12MixtureMaskableActorCriticPolicy)
        self.assertEqual(model.policy.bc_aux_mixture_spec, spec)
        self.assertEqual(
            model.policy_kwargs["bc_aux_mixture_spec"], spec)
        action = int(spec["action_index"])
        columns = list(spec["gate_parameter_columns"])
        coefficients = th.tensor(
            [0.125, -0.25, 0.5, -0.75], dtype=th.float32)
        gate_bias = th.tensor(-0.375, dtype=th.float32)
        with th.no_grad():
            model.policy.action_net.weight[action, columns].copy_(
                coefficients)
            model.policy.action_net.bias[action].copy_(gate_bias)
        expected_coefficients = (
            model.policy.action_net.weight[action, columns]
            .detach().clone())
        expected_bias = (
            model.policy.action_net.bias[action].detach().clone())

        # continuation 不只要留住参数，还必须留住这五个参数的 Adam 动量。
        optimizer = model.policy.optimizer
        weight = model.policy.action_net.weight
        bias = model.policy.action_net.bias
        weight_state = optimizer.state[weight]
        weight_state["step"] = th.tensor(7.0)
        weight_state["exp_avg"] = th.zeros_like(weight)
        weight_state["exp_avg_sq"] = th.zeros_like(weight)
        weight_state["exp_avg"][action, columns] = th.tensor(
            [0.01, 0.02, 0.03, 0.04])
        weight_state["exp_avg_sq"][action, columns] = th.tensor(
            [0.11, 0.12, 0.13, 0.14])
        bias_state = optimizer.state[bias]
        bias_state["step"] = th.tensor(7.0)
        bias_state["exp_avg"] = th.zeros_like(bias)
        bias_state["exp_avg_sq"] = th.zeros_like(bias)
        bias_state["exp_avg"][action] = 0.05
        bias_state["exp_avg_sq"][action] = 0.15
        expected_weight_moments = {
            key: weight_state[key][action, columns].detach().clone()
            for key in ("exp_avg", "exp_avg_sq")
        }
        expected_bias_moments = {
            key: bias_state[key][action].detach().clone()
            for key in ("exp_avg", "exp_avg_sq")
        }
        model._bc_aux_eligible_states = 321
        model._bc_aux_requested_a12 = 23
        model._bc_aux_sampled_a12 = 17
        model._bc_aux_rejected_a12 = 6
        model._bc_aux_unexpected_sampled_a12 = 0
        model._bc_aux_expected_a12_mass = 16.05
        model._bc_aux_pending_requested_a12 = 2
        model._bc_aux_pending_executed_a12 = 1
        model._bc_aux_pending_action_receipts = [
            (12, 12), (12, None)]
        after = self._raw_actor_logits(model, obs)
        keep = [action for action in range(15) if action != 12]
        self.assertTrue(th.equal(before[:, keep], after[:, keep]))
        self.assertEqual(
            model.policy.mlp_extractor.policy_net[0].weight.shape,
            (68, 298))
        self.assertEqual(
            model.policy.mlp_extractor.policy_net[2].weight.shape,
            (68, 68))
        self.assertEqual(model.policy.action_net.weight.shape, (15, 68))
        self.assertEqual(spec, model._bc_aux_circuit_spec)
        # 根锚有意保持旧 64 宽，不能随 live actor 扩宽而被覆写。
        self.assertEqual(root["action_net.weight"].shape, (15, 64))
        self.assertEqual(
            train_ppo._policy_head_sha256(model.bc_aux_root_anchor_sd),
            root_sha)

        expanded_sha = train_ppo._policy_head_sha256(
            _policy_head_snapshot(model.policy))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = pathlib.Path(directory) / "grafted.zip"
            model.save(checkpoint)
            loaded = LeashedMaskablePPO.load(
                checkpoint, env=DummyVecEnv([TinyMasked298Env]),
                device="cpu")
            self.assertIsInstance(
                loaded.policy, A12MixtureMaskableActorCriticPolicy)
            self.assertIs(
                loaded.policy_class, A12MixtureMaskableActorCriticPolicy)
            self.assertEqual(
                loaded.policy.action_net.weight.shape, (15, 68))
            self.assertEqual(loaded._bc_aux_circuit_spec, spec)
            self.assertEqual(loaded.policy.bc_aux_mixture_spec, spec)
            self.assertEqual(
                loaded.policy_kwargs["bc_aux_mixture_spec"], spec)
            self.assertTrue(th.equal(
                loaded.policy.action_net.weight[
                    action, columns].detach(),
                expected_coefficients))
            self.assertTrue(th.equal(
                loaded.policy.action_net.bias[action].detach(),
                expected_bias))
            loaded_weight_state = loaded.policy.optimizer.state[
                loaded.policy.action_net.weight]
            loaded_bias_state = loaded.policy.optimizer.state[
                loaded.policy.action_net.bias]
            for key in ("exp_avg", "exp_avg_sq"):
                self.assertTrue(th.equal(
                    loaded_weight_state[key][action, columns],
                    expected_weight_moments[key]))
                self.assertTrue(th.equal(
                    loaded_bias_state[key][action],
                    expected_bias_moments[key]))
            self.assertEqual(loaded._bc_aux_eligible_states, 321)
            self.assertEqual(loaded._bc_aux_requested_a12, 23)
            self.assertEqual(loaded._bc_aux_sampled_a12, 17)
            self.assertEqual(loaded._bc_aux_rejected_a12, 6)
            self.assertEqual(loaded._bc_aux_unexpected_sampled_a12, 0)
            self.assertEqual(loaded._bc_aux_expected_a12_mass, 16.05)
            self.assertEqual(loaded._bc_aux_pending_requested_a12, 0)
            self.assertEqual(loaded._bc_aux_pending_executed_a12, 0)
            self.assertEqual(loaded._bc_aux_pending_action_receipts, [])
            self.assertEqual(
                train_ppo._policy_head_sha256(
                    _policy_head_snapshot(loaded.policy)),
                expanded_sha)
            self.assertEqual(
                train_ppo._policy_head_sha256(
                    loaded.bc_aux_root_anchor_sd),
                root_sha)
            self.assertTrue(th.equal(
                self._raw_actor_logits(loaded, obs), after))
            # continuation 的幂等安装验证不能把现场上下文参数重置到初始化。
            self.assertEqual(
                _expand_policy_with_bc_aux_circuit(loaded), spec)
            self.assertTrue(th.equal(
                loaded.policy.action_net.weight[
                    action, columns].detach(),
                expected_coefficients))
            self.assertTrue(th.equal(
                loaded.policy.action_net.bias[action].detach(),
                expected_bias))
            loaded.get_env().close()
        model.get_env().close()

    def test_rollout_exposure_counts_only_native_executed_a12(self):
        model = _make_298_model(seed=403)
        self.addCleanup(model.get_env().close)
        _expand_policy_with_bc_aux_circuit(model)
        actions = [12, 12, 9, 10, 13, 14, 9, 10]
        _fill_298_exposure_buffer(model, actions)
        executed = [12, None, 9, 10, 13, 14, 9, 10]
        for requested, receipt in zip(actions, executed):
            model._update_info_buffer([{
                "requested_action": requested,
                "executed_action": receipt,
            }], np.asarray([False]))

        exposure = model._record_bc_aux_rollout_exposure()
        self.assertEqual(exposure["rollout_eligible_states"], 8)
        self.assertEqual(exposure["rollout_requested_a12"], 2)
        self.assertEqual(exposure["rollout_sampled_a12"], 1)
        self.assertEqual(exposure["rollout_rejected_a12"], 1)
        self.assertEqual(exposure["cumulative_requested_a12"], 2)
        self.assertEqual(exposure["cumulative_sampled_a12"], 1)
        self.assertEqual(exposure["cumulative_rejected_a12"], 1)
        self.assertEqual(model._bc_aux_pending_requested_a12, 0)
        self.assertEqual(model._bc_aux_pending_executed_a12, 0)
        self.assertEqual(model._bc_aux_pending_action_receipts, [])

    def test_rollout_receipts_fail_loud_when_missing_or_inconsistent(self):
        model = _make_298_model(seed=404)
        self.addCleanup(model.get_env().close)
        _expand_policy_with_bc_aux_circuit(model)
        model._setup_learn(total_timesteps=model.n_steps)
        with self.assertRaisesRegex(RuntimeError, "缺.*执行回执"):
            model._update_info_buffer(
                [{"requested_action": 12}], np.asarray([False]))
        self.assertEqual(model._bc_aux_pending_requested_a12, 0)
        self.assertEqual(model._bc_aux_pending_executed_a12, 0)
        self.assertEqual(model._bc_aux_pending_action_receipts, [])
        # Every call corresponds to exactly one VecEnv transition batch.
        # Accepting a short/long batch would allow receipts to shift across
        # timesteps when adjacent requested actions happen to be identical.
        with self.assertRaises(RuntimeError):
            model._update_info_buffer([
                {"requested_action": 12, "executed_action": 12},
                {"requested_action": 12, "executed_action": None},
            ], np.asarray([False]))
        self.assertEqual(model._bc_aux_pending_requested_a12, 0)
        self.assertEqual(model._bc_aux_pending_executed_a12, 0)
        self.assertEqual(model._bc_aux_pending_action_receipts, [])
        for label, info in (
            (
                "a12 request forged as another execution",
                {"requested_action": 12, "executed_action": 9},
            ),
            (
                "non-a12 request forged as a12 execution",
                {"requested_action": 9, "executed_action": 12},
            ),
            (
                "out-of-range request and execution",
                {"requested_action": 99, "executed_action": 99},
            ),
        ):
            with self.subTest(label=label), \
                    self.assertRaises(RuntimeError):
                model._update_info_buffer(
                    [info], np.asarray([False]))
            # A malformed batch is rejected atomically; it cannot poison the
            # next rollout's otherwise valid closure counters.
            self.assertEqual(model._bc_aux_pending_requested_a12, 0)
            self.assertEqual(model._bc_aux_pending_executed_a12, 0)
            self.assertEqual(model._bc_aux_pending_action_receipts, [])

    def test_rollout_buffer_and_info_receipts_must_close_exactly(self):
        for label, actions, infos in (
            (
                "buffer request lacks info receipt",
                [12, 9, 9, 9, 9, 9, 9, 9],
                [{"requested_action": 9, "executed_action": 9}] * 8,
            ),
            (
                "info forges request absent from buffer",
                [9] * 8,
                [{"requested_action": 12, "executed_action": 12}]
                + [{"requested_action": 9, "executed_action": 9}] * 7,
            ),
        ):
            with self.subTest(label=label):
                model = _make_298_model(seed=405)
                _expand_policy_with_bc_aux_circuit(model)
                _fill_298_exposure_buffer(model, actions)
                for info in infos:
                    model._update_info_buffer(
                        [info], np.asarray([False]))
                with self.assertRaisesRegex(
                        RuntimeError,
                        "buffer 请求(?:与原生执行回执不闭合|序列与执行回执错位)"):
                    model._record_bc_aux_rollout_exposure()
                model.get_env().close()

    def test_equal_a12_totals_cannot_hide_transition_receipt_swap(self):
        model = _make_298_model(seed=405)
        self.addCleanup(model.get_env().close)
        _expand_policy_with_bc_aux_circuit(model)
        _fill_298_exposure_buffer(
            model, [12, 9, 9, 9, 9, 9, 9, 9])
        # Keep the same aggregate a12 count while moving its alleged receipt
        # from transition 0 to transition 1.  Aggregate-only closure would
        # incorrectly certify this forged sequence.
        infos = (
            [{"requested_action": 9, "executed_action": 9},
             {"requested_action": 12, "executed_action": 12}]
            + [{"requested_action": 9, "executed_action": 9}] * 6
        )
        for info in infos:
            model._update_info_buffer([info], np.asarray([False]))
        with self.assertRaisesRegex(
                RuntimeError,
                "buffer 请求(?:与原生执行回执不闭合|序列与执行回执错位)"):
            model._record_bc_aux_rollout_exposure()

    def test_exact_mixture_probability_and_noneligible_zero(self):
        model = _make_298_model(seed=406)
        spec = _expand_policy_with_bc_aux_circuit(model)

        obs = th.zeros((7, 298), dtype=th.float32)
        obs[:, 0] = th.tensor(
            [0.5, 0.55, 0.649, 0.49, 0.65, 0.55, 0.55])
        obs[:, 297] = th.tensor(
            [0.0, 0.25, 1.0, 0.0, 0.0, -1.5, 0.0])
        masks = th.ones((len(obs), 15), dtype=th.bool)
        masks[:, 11] = False
        masks[6, 12] = False
        with th.no_grad():
            distribution = model.policy.get_distribution(
                obs, action_masks=masks)
            probabilities = distribution.distribution.probs

        th.testing.assert_close(
            probabilities[:3, 12],
            th.full((3,), 0.05, dtype=probabilities.dtype),
            rtol=0.0, atol=5e-7)
        self.assertFalse(bool(
            (probabilities[:3].argmax(dim=-1) == 12).any()))
        self.assertTrue(th.equal(
            probabilities[3:, 12],
            th.zeros(4, dtype=probabilities.dtype)))
        coefficients = model.policy.action_net.weight[
            int(spec["action_index"]),
            list(spec["gate_parameter_columns"]),
        ]
        self.assertTrue(th.equal(
            coefficients, th.zeros_like(coefficients)))
        self.assertAlmostEqual(
            float(model.policy.action_net.bias[12].detach()),
            float(spec["initial_gate_bias"]), places=6)
        model.get_env().close()

    def test_context_changes_probability_and_pmax_makes_deterministic_a12_reachable(self):
        model = _make_298_model(seed=406)
        spec = _expand_policy_with_bc_aux_circuit(model)
        action = int(spec["action_index"])
        columns = list(spec["gate_parameter_columns"])
        with th.no_grad():
            model.policy.action_net.weight[action, columns].zero_()
            # feature8（怪物密度）相同 hp 带内即可把门推向两端。
            model.policy.action_net.weight[action, columns[1]] = 8.0
            model.policy.action_net.bias[action] = 0.0
        obs = th.zeros((2, 298), dtype=th.float32)
        obs[:, 0] = 0.55
        obs[:, 8] = th.tensor([-1.0, 1.0])
        masks = th.ones((2, 15), dtype=th.bool)
        masks[:, 11] = False
        with th.no_grad():
            dist = model.policy.get_distribution(
                obs, action_masks=masks)
            probabilities = dist.distribution.probs
            deterministic = dist.get_actions(deterministic=True)
        self.assertLess(float(probabilities[0, action]), 0.002)
        self.assertGreater(float(probabilities[1, action]), 0.94)
        self.assertLessEqual(
            float(probabilities[1, action]),
            float(spec["probability_max"]) + 1e-7)
        self.assertNotEqual(int(deterministic[0]), action)
        self.assertEqual(int(deterministic[1]), action)
        model.get_env().close()

    def test_hard_predicate_outside_has_exact_zero_probability_and_gate_gradient(self):
        model = _make_298_model(seed=407)
        spec = _expand_policy_with_bc_aux_circuit(model)
        _reset_policy_optimizer(model, 3e-4)
        action = int(spec["action_index"])
        columns = list(spec["gate_parameter_columns"])
        with th.no_grad():
            model.policy.action_net.weight[action, columns].fill_(7.0)
            model.policy.action_net.bias[action] = 7.0
        obs = th.zeros((4, 298), dtype=th.float32)
        obs[:, 0] = th.tensor([0.49, 0.65, 0.55, 0.55])
        obs[:, 297] = th.tensor([0.0, 0.0, -1.0, 0.0])
        masks = th.ones((4, 15), dtype=th.bool)
        masks[:, 11] = False
        masks[3, action] = False
        dist = model.policy.get_distribution(obs, action_masks=masks)
        self.assertTrue(th.equal(
            dist.distribution.probs[:, action],
            th.zeros(4, dtype=th.float32)))
        loss = -dist.log_prob(th.full((4,), 9, dtype=th.long)).mean()
        model.policy.optimizer.zero_grad()
        loss.backward()
        self.assertTrue(th.equal(
            model.policy.action_net.weight.grad[action, columns],
            th.zeros(len(columns), dtype=th.float32)))
        self.assertEqual(
            float(model.policy.action_net.bias.grad[action]), 0.0)
        model.get_env().close()

    def test_a12_logprob_gradient_is_strictly_confined_to_five_gate_parameters(self):
        model = _make_298_model(seed=408)
        spec = _expand_policy_with_bc_aux_circuit(model)
        _reset_policy_optimizer(model, 3e-4)
        action = int(spec["action_index"])
        columns = list(spec["gate_parameter_columns"])
        obs = th.zeros((4, 298), dtype=th.float32)
        obs[:, 0] = 0.55
        obs[:, 8] = th.tensor([1.0, 2.0, 3.0, 4.0])
        obs[:, 9] = 0.5
        obs[:, 286] = 2.0
        masks = th.ones((4, 15), dtype=th.bool)
        masks[:, 11] = False
        dist = model.policy.get_distribution(obs, action_masks=masks)
        loss = -dist.log_prob(
            th.full((4,), action, dtype=th.long)).mean()
        model.policy.optimizer.zero_grad()
        loss.backward()

        action_weight_grad = model.policy.action_net.weight.grad
        action_bias_grad = model.policy.action_net.bias.grad
        expected_weight_live = th.zeros_like(
            action_weight_grad, dtype=th.bool)
        expected_weight_live[action, columns] = True
        self.assertTrue(bool(
            (action_weight_grad[expected_weight_live] != 0).all()))
        self.assertTrue(th.equal(
            action_weight_grad[~expected_weight_live],
            th.zeros_like(action_weight_grad[~expected_weight_live])))
        expected_bias_live = th.zeros_like(
            action_bias_grad, dtype=th.bool)
        expected_bias_live[action] = True
        self.assertNotEqual(float(action_bias_grad[action]), 0.0)
        self.assertTrue(th.equal(
            action_bias_grad[~expected_bias_live],
            th.zeros_like(action_bias_grad[~expected_bias_live])))
        for parameter in (
                model.policy.mlp_extractor.policy_net[0].weight,
                model.policy.mlp_extractor.policy_net[0].bias,
                model.policy.mlp_extractor.policy_net[2].weight,
                model.policy.mlp_extractor.policy_net[2].bias):
            self.assertTrue(
                parameter.grad is None
                or th.equal(parameter.grad, th.zeros_like(parameter.grad)))
        model.get_env().close()

    def test_non_a12_policy_loss_still_trains_legacy_actor(self):
        model = _make_298_model(seed=409)
        _expand_policy_with_bc_aux_circuit(model)
        _reset_policy_optimizer(model, 3e-3)
        obs = th.from_numpy(
            np.random.default_rng(409).standard_normal(
                (16, 298)).astype(np.float32))
        obs[:, 0] = 0.8  # 谓词外，纯旧 actor 更新。
        masks = th.ones((len(obs), 15), dtype=th.bool)
        masks[:, 11] = False
        before = model.policy.action_net.weight[9, :64].detach().clone()
        dist = model.policy.get_distribution(obs, action_masks=masks)
        loss = -dist.log_prob(
            th.full((len(obs),), 9, dtype=th.long)).mean()
        model.policy.optimizer.zero_grad()
        loss.backward()
        self.assertGreater(
            float(model.policy.action_net.weight.grad[9, :64]
                  .abs().sum()), 0.0)
        model.policy.optimizer.step()
        self.assertFalse(th.equal(
            before, model.policy.action_net.weight[9, :64].detach()))
        model.get_env().close()

    def test_offline_68_wide_forward_requires_exact_explicit_circuit_spec(self):
        model = _make_298_model(seed=410)
        spec = _expand_policy_with_bc_aux_circuit(model)
        state = _policy_head_snapshot(model.policy)
        obs = np.zeros((3, 298), dtype=np.float32)
        obs[:, 0] = 0.55
        masks = np.ones((3, 15), dtype=np.bool_)
        masks[:, 11] = False
        with self.assertRaisesRegex(ValueError, "不得仅凭 68 宽猜测"):
            train_ppo._policy_logits_from_sb3_state_dict(
                state, obs, action_masks=masks)
        wrong = json.loads(json.dumps(spec))
        wrong["probability_max"] = 0.25
        with self.assertRaisesRegex(ValueError, "不得仅凭 68 宽猜测"):
            train_ppo._policy_logits_from_sb3_state_dict(
                state, obs, action_masks=masks, circuit_spec=wrong)
        offline = train_ppo._policy_logits_from_sb3_state_dict(
            state, obs, action_masks=masks, circuit_spec=spec)
        with th.no_grad():
            live = model.policy.get_distribution(
                th.from_numpy(obs),
                action_masks=th.from_numpy(masks)).distribution.logits
        th.testing.assert_close(offline, live, rtol=0.0, atol=3e-7)
        model.get_env().close()

    def test_negative_latch_value_entrypoints_are_bitwise_identical(self):
        model = _make_298_model(seed=411)
        _expand_policy_with_bc_aux_circuit(model)
        obs = th.from_numpy(
            np.random.default_rng(407).standard_normal(
                (19, 298)).astype(np.float32))
        obs[:, 0] = 0.55
        obs[:, 297] = th.linspace(-2.0, -1.001, len(obs))
        masks = th.ones((len(obs), 15), dtype=th.bool)
        masks[:, 11] = False
        with th.no_grad():
            actions, forward_values, _ = model.policy.forward(
                obs, deterministic=True, action_masks=masks)
            evaluated_values, _, _ = model.policy.evaluate_actions(
                obs, actions, action_masks=masks)
            predicted_values = model.policy.predict_values(obs)
        self.assertTrue(th.equal(forward_values, evaluated_values))
        self.assertTrue(th.equal(forward_values, predicted_values))
        model.get_env().close()

    def test_exact_mixture_liveness_first_install_and_continuation(self):
        model = _make_298_model(seed=408)
        with tempfile.TemporaryDirectory() as directory:
            out = pathlib.Path(directory) / "runs" / "bc-worker-v2"
            out.mkdir(parents=True)
            x, y, groups, masks = _write_v2_pass_bundle(
                out / "demos.npz")
            first = _simulate_bc_aux_circuit_liveness(
                model, x=x, y=y, episode_id=groups, masks=masks,
                learning_rate=3e-4, call_plan={"train_calls": 244})
            self.assertEqual(first["status"], "PASS")
            self.assertEqual(
                first["simulation"],
                "isolated-exact-mixture-with-policy-gradient-canary")
            self.assertEqual(first["installation"], "first-install")
            self.assertEqual(
                first["calls"]["policy_gradient_canary_calls"], 1)
            self.assertIn("policy_gradient_canary", first)
            self.assertGreater(
                first["policy_gradient_canary"]["probability_12_delta"], 0.0)
            self.assertGreater(
                first["policy_gradient_canary"]["gate_bias_delta"], 0.0)
            self.assertTrue(
                first["policy_gradient_canary"]["state_restored"])
            self.assertEqual(
                first["calls"]["initial_adapter_calibrations"], 1)
            self.assertEqual(
                first["calls"]["trainable_adapter_parameters"], 5)
            self.assertEqual(
                first["calibration"]["initializer"],
                "exact-contextual-legal-support-mixture")

            spec = _bc_aux_circuit_spec()
            action = int(spec["action_index"])
            columns = list(spec["gate_parameter_columns"])
            with th.no_grad():
                model.policy.action_net.weight[action, columns] = th.tensor(
                    [0.1, -0.2, 0.3, -0.4])
                model.policy.action_net.bias[action] = -0.5
            preserved_coefficients = (
                model.policy.action_net.weight[action, columns]
                .detach().clone())
            preserved_bias = (
                model.policy.action_net.bias[action].detach().clone())
            continued = _simulate_bc_aux_circuit_liveness(
                model, x=x, y=y, episode_id=groups, masks=masks,
                learning_rate=3e-4, call_plan={"train_calls": 244})
            self.assertEqual(continued["status"], "PASS")
            self.assertEqual(
                continued["installation"], "preserved-continuation")
            self.assertEqual(
                continued["calls"]["initial_adapter_calibrations"], 0)
            self.assertEqual(
                continued["calls"]["policy_gradient_canary_calls"], 1)
            self.assertEqual(
                continued["calibration"]["initializer"],
                "preserved-continuation")
            self.assertTrue(th.equal(
                model.policy.action_net.weight[
                    action, columns].detach(),
                preserved_coefficients))
            self.assertTrue(th.equal(
                model.policy.action_net.bias[action].detach(),
                preserved_bias))
        model.get_env().close()

    def test_policy_gradient_canary_uses_live_distribution_and_restores_state(self):
        model = _make_298_model(seed=418)
        with tempfile.TemporaryDirectory() as directory:
            out = pathlib.Path(directory) / "runs" / "bc-worker-v2"
            out.mkdir(parents=True)
            x, y, groups, masks = _write_v2_pass_bundle(
                out / "demos.npz")
            spec = _expand_policy_with_bc_aux_circuit(model)
            _reset_policy_optimizer(model, 3e-4)
            train_ppo._calibrate_bc_aux_adapter_weight(
                model, x, y, groups, masks)
            policy_before = {
                key: value.detach().clone()
                for key, value in model.policy.state_dict().items()
            }
            optimizer_before = copy.deepcopy(
                model.policy.optimizer.state_dict())
            training_before = bool(model.policy.training)
            real_get_distribution = model.policy.get_distribution
            with mock.patch.object(
                    model.policy, "get_distribution",
                    wraps=real_get_distribution) as get_distribution:
                canary = train_ppo._run_bc_aux_policy_gradient_canary(
                    model, x=x, y=y, episode_id=groups,
                    masks=masks, spec=spec)

            # before/loss/after 三次都走真实 policy distribution；中间目标
            # 是真实 log_prob(action12)，而非离线 logits 捷径。
            self.assertGreaterEqual(get_distribution.call_count, 3)
            self.assertEqual(canary["optimizer_steps"], 1)
            self.assertGreater(canary["probability_12_delta"], 0.0)
            self.assertGreater(canary["gate_bias_delta"], 0.0)
            self.assertLess(canary["gate_bias_gradient"], 0.0)
            self.assertNotEqual(
                canary["start_policy_head_sha256"],
                canary["stepped_policy_head_sha256"])
            self.assertTrue(canary["state_restored"])
            self.assertTrue(all(
                th.equal(model.policy.state_dict()[key], value)
                for key, value in policy_before.items()))
            self.assertTrue(_tree_equal(
                model.policy.optimizer.state_dict(), optimizer_before))
            self.assertIs(bool(model.policy.training), training_before)
        model.get_env().close()

    def test_structural_aux_loss_has_exact_zero_a14_gradient_and_invariance(self):
        model = _make_298_model(lam=0.015625, seed=419)
        self.addCleanup(model.get_env().close)
        _expand_policy_with_bc_aux_circuit(model)
        _reset_policy_optimizer(model, 3e-4)
        rng = np.random.default_rng(419)
        x = rng.standard_normal((32, 298)).astype(np.float32)
        x[:, 0] = 0.55
        x[:, 297] = 0.0
        y = np.full(32, 9, dtype=np.int64)
        y[:8] = 12
        masks = np.ones((32, 15), dtype=np.bool_)
        masks[:, 11] = False
        original_masks = masks.copy()
        model.mount_bc_aux_demos(
            x, y, masks, rng=derive_bc_aux_rng(419))

        self.assertTrue(np.array_equal(
            model._bc_aux_masks.cpu().numpy(), original_masks))
        self.assertTrue(th.equal(
            model._bc_aux_anchor_probs[:, 12],
            th.zeros(32, dtype=th.float32)))
        self.assertTrue(th.equal(
            model._bc_aux_anchor_probs[:, 14],
            th.zeros(32, dtype=th.float32)))
        # 锚定后只扰动一个旧动作，保证 KL 分支确有非零训练信号；
        # a14 的零梯度不能由“整个 KL 恰为零”伪造。
        with th.no_grad():
            model.policy.action_net.bias[9] += 0.75

        loss = model._peek_bc_aux_ce_loss()
        grad_w, grad_b = th.autograd.grad(
            loss,
            [model.policy.action_net.weight,
             model.policy.action_net.bias],
        )
        self.assertGreater(float(grad_w[12].abs().sum()), 0.0)
        self.assertNotEqual(float(grad_b[12]), 0.0)
        self.assertGreater(float(grad_w[9].abs().sum()), 0.0)
        self.assertNotEqual(float(grad_b[9]), 0.0)
        self.assertTrue(th.equal(
            grad_w[14], th.zeros_like(grad_w[14])))
        self.assertEqual(float(grad_b[14]), 0.0)

        baseline = loss.detach().clone()
        with th.no_grad():
            model.policy.action_net.weight[14].zero_()
            model.policy.action_net.bias[14] = 1_000_000.0
        perturbed = model._peek_bc_aux_ce_loss().detach()
        self.assertTrue(th.equal(baseline, perturbed))

    def test_structural_root_gates_are_invariant_to_a14_policy_growth(self):
        model = _make_298_model(lam=0.015625, seed=420)
        self.addCleanup(model.get_env().close)
        spec = _expand_policy_with_bc_aux_circuit(model)
        _reset_policy_optimizer(model, 3e-4)
        rng = np.random.default_rng(420)
        x = rng.standard_normal((32, 298)).astype(np.float32)
        x[:, 0] = 0.55
        x[:, 297] = 0.0
        y = np.full(32, 9, dtype=np.int64)
        y[:8] = 12
        groups = np.repeat(np.arange(8, dtype=np.int64), 4)
        masks = np.ones((32, 15), dtype=np.bool_)
        masks[:, 11] = False
        model.mount_bc_aux_demos(
            x, y, masks, rng=derive_bc_aux_rng(420))

        runtime_before = model._bc_aux_bank_drift()
        state_before = _policy_head_snapshot(model.policy)
        offline_before = bc_aux_behavior_metrics(
            state_before, x, y, groups, masks,
            anchor_sd=model.bc_aux_root_anchor_sd,
            heldout_only=False, circuit_spec=spec)
        with th.no_grad():
            model.policy.action_net.weight[14].zero_()
            model.policy.action_net.bias[14] = 1_000_000.0
            full_pred = model.policy.get_distribution(
                th.from_numpy(x),
                action_masks=th.from_numpy(masks),
            ).distribution.probs.argmax(dim=-1)
        self.assertTrue(bool((full_pred == 14).all()))

        runtime_after = model._bc_aux_bank_drift()
        offline_after = bc_aux_behavior_metrics(
            _policy_head_snapshot(model.policy), x, y, groups, masks,
            anchor_sd=model.bc_aux_root_anchor_sd,
            heldout_only=False, circuit_spec=spec)
        for key in ("argmax_drift", "tv_mean", "kl_anchor_to_policy"):
            self.assertEqual(runtime_before[key], runtime_after[key], key)
            self.assertEqual(
                offline_before["anchor"][key],
                offline_after["anchor"][key],
                key,
            )
        self.assertEqual(
            offline_before["anchor"]["critical_action_retention"],
            offline_after["anchor"]["critical_action_retention"])

    def test_king_support_is_invariant_to_a12_a14_with_zero_gradients(self):
        model = _make_298_model(seed=409)
        obs = th.from_numpy(
            np.random.default_rng(409).standard_normal(
                (16, 298)).astype(np.float32))
        masks = th.ones((len(obs), 15), dtype=th.bool)
        masks[:, 11] = False
        original_masks = masks.clone()
        conditional = _legacy_distillation_masks(masks)
        self.assertTrue(th.equal(masks, original_masks))
        self.assertFalse(bool(conditional[:, 12].any()))
        self.assertFalse(bool(conditional[:, 14].any()))
        self.assertTrue(bool(conditional.any(dim=-1).all()))

        teacher_logits = th.from_numpy(
            np.random.default_rng(410).standard_normal(
                (len(obs), 15)).astype(np.float32))
        teacher_probs = _masked_log_softmax_from_raw(
            teacher_logits, conditional).exp()

        def distill_ce():
            student_raw_logits = model._student_raw_action_logits(obs)
            return -(
                teacher_probs * _masked_log_softmax_from_raw(
                    student_raw_logits, conditional)
            ).sum(dim=-1).mean()

        model.policy.optimizer.zero_grad()
        baseline = distill_ce()
        baseline.backward()
        self.assertTrue(th.equal(
            model.policy.action_net.weight.grad[12],
            th.zeros_like(model.policy.action_net.weight.grad[12])))
        self.assertEqual(
            float(model.policy.action_net.bias.grad[12]), 0.0)
        self.assertTrue(th.equal(
            model.policy.action_net.weight.grad[14],
            th.zeros_like(model.policy.action_net.weight.grad[14])))
        self.assertEqual(
            float(model.policy.action_net.bias.grad[14]), 0.0)

        with th.no_grad():
            model.policy.action_net.bias[12] += 1_000_000.0
            model.policy.action_net.weight[12].fill_(1_000_000.0)
            model.policy.action_net.bias[14] -= 1_000_000.0
            model.policy.action_net.weight[14].fill_(-1_000_000.0)
        perturbed = distill_ce().detach()
        self.assertTrue(th.equal(baseline.detach(), perturbed))
        model.get_env().close()


class BcAuxBehaviorPublicationTests(unittest.TestCase):
    """原始 held-out 行为硬门是 model_final 发布事务的一部分。"""

    def test_aux_training_split_is_episode_disjoint_from_final_gate(self):
        groups = np.repeat(np.arange(20, dtype=np.int64), 5)
        training = _bc_v2_training_indices(groups)
        heldout = _bc_v2_holdout_indices(groups)
        self.assertEqual(len(training) + len(heldout), len(groups))
        self.assertFalse(np.intersect1d(training, heldout).size)
        self.assertFalse(np.intersect1d(
            np.unique(groups[training]), np.unique(groups[heldout])).size)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([training, heldout])),
            np.arange(len(groups), dtype=np.int64))

    def test_aux_bank_contains_no_heldout_row_even_for_a12_positives(self):
        groups = np.repeat(np.arange(20, dtype=np.int64), 5)
        heldout = _bc_v2_holdout_indices(groups)
        training = _bc_v2_training_indices(groups)
        x = np.zeros((len(groups), 298), dtype=np.float32)
        # 行号作为不可碰撞指纹；held-out 正例若泄漏会直接出现在 bank。
        x[:, 0] = np.arange(len(groups), dtype=np.float32)
        y = np.full(len(groups), 9, dtype=np.int64)
        y[training[::5]] = 12
        y[heldout] = 12
        training_negative = training[y[training] != 12]
        post_drink = training_negative[:int((y[training] == 12).sum())]
        x[post_drink, 297] = -1.5
        masks = np.ones((len(groups), 15), dtype=np.bool_)
        masks[:, 11] = False
        masks[post_drink, 12] = False

        bank_x, bank_y, bank_masks = _build_bc_aux_training_bank(
            x, y, groups, masks)
        self.assertTrue((bank_y == 12).any())
        self.assertTrue((bank_y != 12).any())
        self.assertTrue(bank_masks[bank_y == 12, 12].all())
        # feature297<0 且 m12=False 的后饮证据只负责闭合覆盖，
        # 不得混进要求 m12=True 的 legacy 校准 bank。
        self.assertTrue(bank_masks[:, 12].all())
        self.assertFalse((bank_x[:, 297] < 0.0).any())
        self.assertFalse(np.isin(bank_x[:, 0], x[heldout, 0]).any())

    def test_fpr_denominator_excludes_structurally_illegal_a12_states(self):
        x = np.zeros((100, 2), dtype=np.float32)
        x[:, 0] = 0.8
        y = np.full(100, 9, dtype=np.int64)
        y[0] = 12
        groups = np.repeat(np.arange(2, dtype=np.int64), 50)
        masks = np.ones((100, 15), dtype=bool)
        masks[:, 11] = False
        masks[2:, 12] = False  # 97/99 negatives 对 a12 结构性不可达
        metrics = bc_aux_behavior_metrics(
            _tiny_policy_sd(collapse=True), x, y, groups, masks,
            heldout_only=False)
        self.assertEqual(metrics["all_non_a12"], 99)
        self.assertEqual(metrics["non_a12"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["tn"], 0)
        self.assertEqual(metrics["fpr_12"], 1.0)  # 禁被 99 稀释成约 .01

    def test_published_receipt_binds_successful_model_bytes(self):
        x, y, groups, masks = _behavior_fixture()
        model = _FakePublishModel(_contextual_policy_sd())
        anchor = _contextual_anchor_sd()
        with tempfile.TemporaryDirectory() as d:
            final = pathlib.Path(d) / "model_final.zip"

            def fake_save(_model, destination):
                pathlib.Path(destination).write_bytes(b"published")

            with mock.patch.object(
                    train_ppo, "_atomic_save_model",
                    side_effect=fake_save) as save:
                rec = _publish_model_final_with_bc_aux_gate(
                    model, final, x=x, y=y, episode_id=groups,
                    masks=masks, demos_sha256="a" * 64,
                    anchor_sd=anchor,
                    publication_provenance=_publication_provenance())
            self.assertEqual(rec["publication"], "PUBLISHED")
            self.assertEqual(rec["gate"]["verdict"], "PASS")
            self.assertTrue(final.is_file())
            self.assertEqual(
                rec["model_sha256"],
                hashlib.sha256(final.read_bytes()).hexdigest())
            self.assertEqual(save.call_count, 1)
            disk = json.loads(
                (final.parent / "bc_aux_behavior_receipt.json").read_text())
            self.assertEqual(disk["publication"], "PUBLISHED")
            self.assertEqual(disk["objective_revision"], 11)
            self.assertIs(
                disk["gate"]["thresholds"]["deployable_a12_required"],
                False)
            self.assertIsNone(
                disk["gate"]["thresholds"][
                    "deterministic_a12_episode_min"])
            self.assertIsNone(
                disk["gate"]["thresholds"][
                    "deterministic_a12_margin_min"])
            self.assertGreaterEqual(
                disk["metrics"]["predicted_a12_episodes"], 2)
            self.assertGreaterEqual(
                disk["metrics"]["predicted_a12_margin_min"], 1e-4)
            self.assertEqual(disk["exploration_evidence"], {
                "eligible_states": 400,
                "expected_a12_mass": 20.0,
                "requested_a12": 10,
                "sampled_a12": 10,
                "rejected_a12": 0,
                "unexpected_sampled_a12": 0,
                "minimum_expected_a12_mass": 20.0,
                "minimum_actual_a12_samples": 10,
                "information_status": "INFORMATIVE",
                "reasons": [],
            })
            self.assertEqual(
                disk["anchor"]["identity"], "bc-aux-root-policy")
            self.assertEqual(
                disk["provenance"]["target_global_steps"], 1024)
            self.assertEqual(
                disk["provenance"]["bc_aux_liveness_preflight_sha256"],
                "f" * 64)

    def test_rev10_publication_does_not_impose_deterministic_a12_quota(self):
        x, y, groups, masks = _behavior_fixture()
        spec = _bc_aux_circuit_spec()
        model = _FakePublishModel(_contextual_policy_sd(
            gate_bias=spec["initial_gate_bias"]))
        with tempfile.TemporaryDirectory() as d:
            final = pathlib.Path(d) / "model_final.zip"

            def fake_save(_model, destination):
                pathlib.Path(destination).write_bytes(b"published")

            with mock.patch.object(
                    train_ppo, "_atomic_save_model",
                    side_effect=fake_save):
                rec = _publish_model_final_with_bc_aux_gate(
                    model, final, x=x, y=y, episode_id=groups,
                    masks=masks, demos_sha256="a" * 64,
                    anchor_sd=_contextual_anchor_sd(),
                    publication_provenance=_publication_provenance())

        self.assertEqual(rec["publication"], "PUBLISHED")
        self.assertEqual(rec["metrics"]["predicted_a12"], 0)
        self.assertEqual(rec["metrics"]["predicted_a12_episodes"], 0)
        self.assertEqual(rec["metrics"]["predicted_a12_margin_min"], 0.0)
        self.assertEqual(rec["gate"]["verdict"], "PASS")
        self.assertIs(
            rec["gate"]["thresholds"]["deployable_a12_required"], False)

    def test_publication_refuses_insufficient_onpolicy_exploration(self):
        x, y, groups, masks = _behavior_fixture()
        model = _FakePublishModel(
            _contextual_policy_sd(), eligible_states=399,
            requested_a12=10, sampled_a12=9, rejected_a12=1,
            expected_a12_mass=19.95)
        with tempfile.TemporaryDirectory() as directory:
            final = pathlib.Path(directory) / "model_final.zip"
            with mock.patch.object(
                    train_ppo, "_atomic_save_model") as save:
                with self.assertRaisesRegex(
                        ValueError, "探索证据硬门 FAIL"):
                    _publish_model_final_with_bc_aux_gate(
                        model, final, x=x, y=y, episode_id=groups,
                        masks=masks, demos_sha256="a" * 64,
                        anchor_sd=_contextual_anchor_sd(),
                        publication_provenance=_publication_provenance())
            save.assert_not_called()
            receipt = json.loads(
                (final.parent / "bc_aux_behavior_receipt.json").read_text())
            self.assertEqual(receipt["publication"], "REFUSED")
            evidence = receipt["exploration_evidence"]
            self.assertEqual(
                evidence["information_status"],
                "INSUFFICIENT_OR_INVALID")
            self.assertEqual(evidence["sampled_a12"], 9)
            self.assertEqual(evidence["requested_a12"], 10)
            self.assertEqual(evidence["rejected_a12"], 1)
            self.assertIn("expected_a12_mass<20.0", evidence["reasons"])
            self.assertIn("sampled_a12<10", evidence["reasons"])

    def test_ten_rejected_requests_do_not_satisfy_sampled_gate(self):
        env = DummyVecEnv([TinyMasked298Env])
        model = LeashedMaskablePPO(
            "MlpPolicy", env, n_steps=10, batch_size=10, n_epochs=1,
            gamma=1.0, distill_beta=0.0, bc_aux_lambda=0.0,
            seed=413, device="cpu", verbose=0)
        try:
            _expand_policy_with_bc_aux_circuit(model)
            _fill_298_exposure_buffer(model, [12] * 10)
            for _ in range(10):
                model._update_info_buffer([{
                    "requested_action": 12,
                    "executed_action": None,
                }], np.asarray([False]))
            exposure = model._record_bc_aux_rollout_exposure()
            self.assertEqual(exposure["rollout_requested_a12"], 10)
            self.assertEqual(exposure["rollout_rejected_a12"], 10)
            self.assertEqual(exposure["rollout_sampled_a12"], 0)
            self.assertEqual(model._bc_aux_sampled_a12, 0)
        finally:
            env.close()

        x, y, groups, masks = _behavior_fixture()
        publish_model = _FakePublishModel(
            _contextual_policy_sd(), requested_a12=10,
            sampled_a12=0, rejected_a12=10,
            expected_a12_mass=20.0)
        with tempfile.TemporaryDirectory() as directory:
            final = pathlib.Path(directory) / "model_final.zip"
            with mock.patch.object(
                    train_ppo, "_atomic_save_model") as save:
                with self.assertRaisesRegex(
                        ValueError, "sampled_a12<10"):
                    _publish_model_final_with_bc_aux_gate(
                        publish_model, final, x=x, y=y,
                        episode_id=groups, masks=masks,
                        demos_sha256="a" * 64,
                        anchor_sd=_contextual_anchor_sd(),
                        publication_provenance=_publication_provenance())
            save.assert_not_called()
            evidence = json.loads(
                (final.parent / "bc_aux_behavior_receipt.json")
                .read_text())["exploration_evidence"]
            self.assertEqual(evidence["sampled_a12"], 0)
            self.assertIn("sampled_a12<10", evidence["reasons"])

    def test_publication_rejects_malformed_or_nonclosing_exploration_evidence(
            self):
        x, y, groups, masks = _behavior_fixture()
        mutations = (
            (
                "bool requested count",
                {"_bc_aux_requested_a12": True},
            ),
            (
                "fractional rejected count",
                {"_bc_aux_rejected_a12": 0.5},
            ),
            (
                "requested does not equal executed plus rejected",
                {
                    "_bc_aux_requested_a12": 11,
                    "_bc_aux_sampled_a12": 10,
                    "_bc_aux_rejected_a12": 0,
                },
            ),
            (
                "bool expected mass",
                {"_bc_aux_expected_a12_mass": True},
            ),
            (
                "string expected mass",
                {"_bc_aux_expected_a12_mass": "20.0"},
            ),
            (
                "expected mass exceeds eligible states",
                {"_bc_aux_expected_a12_mass": 400.5},
            ),
        )
        for label, mutation in mutations:
            with self.subTest(label=label), \
                    tempfile.TemporaryDirectory() as directory:
                model = _FakePublishModel(_contextual_policy_sd())
                for name, value in mutation.items():
                    setattr(model, name, value)
                final = pathlib.Path(directory) / "model_final.zip"
                with mock.patch.object(
                        train_ppo, "_atomic_save_model",
                        side_effect=lambda _model, destination:
                        pathlib.Path(destination).write_bytes(
                            b"must-not-publish")) as save, \
                        self.assertRaises(ValueError):
                    _publish_model_final_with_bc_aux_gate(
                        model, final, x=x, y=y, episode_id=groups,
                        masks=masks, demos_sha256="a" * 64,
                        anchor_sd=_contextual_anchor_sd(),
                        publication_provenance=_publication_provenance())
                save.assert_not_called()

    def test_save_failure_cannot_leave_pass_receipt_or_canonical_model(self):
        x, y, groups, masks = _behavior_fixture()
        model = _FakePublishModel(_contextual_policy_sd())
        anchor = _contextual_anchor_sd()
        with tempfile.TemporaryDirectory() as d:
            final = pathlib.Path(d) / "model_final.zip"
            with mock.patch.object(
                    train_ppo, "_atomic_save_model",
                    side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    _publish_model_final_with_bc_aux_gate(
                        model, final, x=x, y=y, episode_id=groups,
                        masks=masks, demos_sha256="a" * 64,
                        anchor_sd=anchor,
                        publication_provenance=_publication_provenance())
            self.assertFalse(final.exists())
            disk = json.loads(
                (final.parent / "bc_aux_behavior_receipt.json").read_text())
            self.assertEqual(disk["publication"], "SAVE_FAILED")
            self.assertIsNone(disk["model_sha256"])
            self.assertEqual(disk["save_error"]["type"], "OSError")

    def test_publication_refuses_incomplete_or_wrong_step_lineage(self):
        x, y, groups, masks = _behavior_fixture()
        model = _FakePublishModel(_contextual_policy_sd())
        anchor = _contextual_anchor_sd()
        with tempfile.TemporaryDirectory() as d:
            final = pathlib.Path(d) / "model_final.zip"
            incomplete = _publication_provenance()
            incomplete.pop("manager_npz_sha256")
            with self.assertRaisesRegex(ValueError, "谱系字段不完整"):
                _publish_model_final_with_bc_aux_gate(
                    model, final, x=x, y=y, episode_id=groups,
                    masks=masks, demos_sha256="a" * 64,
                    anchor_sd=anchor,
                    publication_provenance=incomplete)
            wrong_step = _publication_provenance(step=2048)
            with self.assertRaisesRegex(ValueError, "步数不闭合"):
                _publish_model_final_with_bc_aux_gate(
                    model, final, x=x, y=y, episode_id=groups,
                    masks=masks, demos_sha256="a" * 64,
                    anchor_sd=anchor,
                    publication_provenance=wrong_step)
            no_preflight = _publication_provenance()
            no_preflight["bc_aux_liveness_preflight_sha256"] = None
            with self.assertRaisesRegex(ValueError, "preflight_sha256 非法"):
                _publish_model_final_with_bc_aux_gate(
                    model, final, x=x, y=y, episode_id=groups,
                    masks=masks, demos_sha256="a" * 64,
                    anchor_sd=anchor,
                    publication_provenance=no_preflight)
            record_only = _publication_provenance()
            record_only["calib_record_only"] = True
            with self.assertRaisesRegex(ValueError, "禁止 calib_record_only"):
                _publish_model_final_with_bc_aux_gate(
                    model, final, x=x, y=y, episode_id=groups,
                    masks=masks, demos_sha256="a" * 64,
                    anchor_sd=anchor,
                    publication_provenance=record_only)
            self.assertFalse(final.exists())

    def test_final_gate_rejects_root_distribution_or_critical_action_loss(self):
        x, y, groups, masks = _behavior_fixture()
        metrics = train_ppo.bc_aux_behavior_metrics(
            _contextual_policy_sd(), x, y, groups, masks,
            anchor_sd=_contextual_anchor_sd(),
            circuit_spec=_bc_aux_circuit_spec())
        self.assertEqual(
            train_ppo.bc_aux_behavior_gate(
                metrics, require_root_anchor=True)["verdict"],
            "PASS")

        drifted = json.loads(json.dumps(metrics))
        drifted["anchor"]["tv_mean"] = 0.9
        gate = train_ppo.bc_aux_behavior_gate(
            drifted, require_root_anchor=True)
        self.assertEqual(gate["verdict"], "FAIL")
        self.assertTrue(any("tv_mean" in reason for reason in gate["reasons"]))

        erased = json.loads(json.dumps(metrics))
        erased["anchor"]["critical_action_retention"]["9"]["retention"] = 0.0
        gate = train_ppo.bc_aux_behavior_gate(
            erased, require_root_anchor=True)
        self.assertEqual(gate["verdict"], "FAIL")
        self.assertTrue(any(
            "action_9_retention" in reason for reason in gate["reasons"]))

    def test_strict_gate_rejects_bool_count_closure_and_zero_deployment(self):
        x, y, groups, masks = _behavior_fixture()
        spec = _bc_aux_circuit_spec()
        metrics = train_ppo.bc_aux_behavior_metrics(
            _contextual_policy_sd(), x, y, groups, masks,
            anchor_sd=_contextual_anchor_sd(), circuit_spec=spec)
        strict = dict(
            require_root_anchor=True,
            require_teacher_recall=False,
            require_deployable_a12=True,
        )
        self.assertEqual(
            train_ppo.bc_aux_behavior_gate(
                metrics, **strict)["verdict"], "PASS")

        forged_bool = json.loads(json.dumps(metrics))
        forged_bool["pairs"] = True
        bool_gate = train_ppo.bc_aux_behavior_gate(
            forged_bool, **strict)
        self.assertEqual(bool_gate["verdict"], "FAIL")
        self.assertIn("pairs_invalid", bool_gate["reasons"])

        forged_closure = json.loads(json.dumps(metrics))
        forged_closure["tp"] += 1
        closure_gate = train_ppo.bc_aux_behavior_gate(
            forged_closure, **strict)
        self.assertEqual(closure_gate["verdict"], "FAIL")
        self.assertIn("count_closure_invalid", closure_gate["reasons"])
        self.assertIn(
            "precision_12_count_mismatch", closure_gate["reasons"])

        initial = train_ppo.bc_aux_behavior_metrics(
            _contextual_policy_sd(
                gate_bias=spec["initial_gate_bias"]),
            x, y, groups, masks,
            anchor_sd=_contextual_anchor_sd(), circuit_spec=spec)
        zero_gate = train_ppo.bc_aux_behavior_gate(
            initial, **strict)
        self.assertEqual(initial["predicted_a12"], 0)
        self.assertEqual(zero_gate["verdict"], "FAIL")
        self.assertIn(
            "predicted_a12_episodes<2", zero_gate["reasons"])
        self.assertIn(
            "predicted_a12_margin_min<0.0001",
            zero_gate["reasons"])

    def test_deployable_gate_requires_two_episodes_and_positive_margin(self):
        x, y, groups, masks = _behavior_fixture()
        metrics = train_ppo.bc_aux_behavior_metrics(
            _contextual_policy_sd(), x, y, groups, masks,
            anchor_sd=_contextual_anchor_sd(),
            circuit_spec=_bc_aux_circuit_spec())
        self.assertGreaterEqual(metrics["predicted_a12_episodes"], 2)
        self.assertGreater(metrics["predicted_a12_margin_min"], 0.0)

        one_episode = json.loads(json.dumps(metrics))
        one_episode["predicted_a12_episodes"] = 1
        gate = train_ppo.bc_aux_behavior_gate(
            one_episode, require_root_anchor=True,
            require_teacher_recall=False,
            require_deployable_a12=True)
        self.assertEqual(gate["verdict"], "FAIL")
        self.assertIn("predicted_a12_episodes<2", gate["reasons"])

        zero_margin = json.loads(json.dumps(metrics))
        zero_margin["predicted_a12_margin_min"] = 0.0
        gate = train_ppo.bc_aux_behavior_gate(
            zero_margin, require_root_anchor=True,
            require_teacher_recall=False,
            require_deployable_a12=True)
        self.assertEqual(gate["verdict"], "FAIL")
        self.assertIn(
            "predicted_a12_margin_min<0.0001", gate["reasons"])

    def test_root_anchor_persists_across_continuation_mounts(self):
        model = _make_298_model(seed=67)
        root = _persistent_bc_aux_root_anchor(model)
        root_sha = train_ppo._policy_head_sha256(root)

        with th.no_grad():
            model.policy.action_net.bias[0] += 1.0
        resumed = _persistent_bc_aux_root_anchor(model)
        self.assertEqual(train_ppo._policy_head_sha256(resumed), root_sha)
        self.assertNotEqual(
            train_ppo._policy_head_sha256(
                _policy_head_snapshot(model.policy)),
            root_sha)
        model.get_env().close()

    def test_root_anchor_survives_real_sb3_save_load(self):
        model = _make_model(lam=0.0, seed=71)
        root = _persistent_bc_aux_root_anchor(model)
        expected = train_ppo._policy_head_sha256(root)
        with tempfile.TemporaryDirectory() as d:
            checkpoint = pathlib.Path(d) / "anchor.zip"
            model.save(checkpoint)
            loaded = LeashedMaskablePPO.load(
                checkpoint, env=DummyVecEnv([TinyMasked15Env]),
                device="cpu")
            recovered = _persistent_bc_aux_root_anchor(loaded)
            self.assertEqual(
                train_ppo._policy_head_sha256(recovered), expected)
            loaded.get_env().close()
        model.get_env().close()

    def test_fail_receipt_refuses_model_publish(self):
        x, y, groups, masks = _behavior_fixture()
        # rev10 允许 PPO 不把 a12 变成 argmax，但仍拒绝破坏根策略的候选。
        unsafe = _contextual_policy_sd(
            gate_bias=_bc_aux_circuit_spec()["initial_gate_bias"])
        unsafe["action_net.bias"][0] = 20.0
        model = _FakePublishModel(unsafe)
        anchor = _contextual_anchor_sd()
        with tempfile.TemporaryDirectory() as d:
            final = pathlib.Path(d) / "model_final.zip"
            with mock.patch.object(
                    train_ppo, "_atomic_save_model") as save:
                with self.assertRaisesRegex(ValueError, "拒绝发布 model_final"):
                    _publish_model_final_with_bc_aux_gate(
                        model, final, x=x, y=y, episode_id=groups,
                        masks=masks, demos_sha256="a" * 64,
                        anchor_sd=anchor,
                        publication_provenance=_publication_provenance())
            save.assert_not_called()
            self.assertFalse(final.exists())
            disk = json.loads(
                (final.parent / "bc_aux_behavior_receipt.json").read_text())
            self.assertEqual(disk["publication"], "REFUSED")
            self.assertEqual(disk["gate"]["verdict"], "FAIL")
            self.assertEqual(disk["metrics"]["predicted_a12"], 0)
            self.assertTrue(any(
                reason.startswith("root_anchor.")
                for reason in disk["gate"]["reasons"]))


class BcAuxDemoValidatorTests(unittest.TestCase):
    """E3②:v2 demos 专用验证器(镜像断言按世代分别成文)。"""

    def _tmp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        out = pathlib.Path(directory.name) / "runs" / "bc-worker-v2"
        out.mkdir(parents=True)
        return out

    def test_good_v2_file_roundtrip_returns_arrays_and_sha(self):
        p = self._tmp() / "demos.npz"
        x, y, ep, masks = _write_v2_pass_bundle(p)
        lx, ly, lep, lmasks, sha = _load_bc_aux_demos_v2(
            p, expected_manager_sha256=CANONICAL_MANAGER_SHA)
        self.assertEqual(sha, hashlib.sha256(p.read_bytes()).hexdigest())
        np.testing.assert_array_equal(lx, x)
        np.testing.assert_array_equal(ly, y)
        np.testing.assert_array_equal(lep, ep)
        np.testing.assert_array_equal(lmasks, masks)
        self.assertIn(12, ly)                       # v2 允 12

    def test_v2_pass_bundle_requires_final_holdout_marker(self):
        p = self._tmp() / "demos.npz"
        _write_v2_pass_bundle(p)
        marker, _, _ = train_ppo._bc_final_holdout_marker_path(
            p.parent, 2, _BC_V2_COLLECTION_EPISODES)
        marker.unlink()
        with self.assertRaisesRegex(
                ValueError, "one-shot marker 缺失/不可读"):
            _load_bc_aux_demos_v2(
                p, expected_manager_sha256=CANONICAL_MANAGER_SHA)

    def test_v2_final_holdout_marker_tamper_is_rejected(self):
        for mutation, message in (
            ("pool", "pool/时间/消费阶段身份不闭合"),
            ("marker_schema", "pool/时间/消费阶段身份不闭合"),
            ("provenance", "provenance 与 PASS report 不一致"),
            ("extra", "字段/schema 不精确"),
        ):
            with self.subTest(mutation=mutation):
                base = self._tmp() / mutation
                base.mkdir()
                p = base / "demos.npz"
                _write_v2_pass_bundle(p)
                marker, _, _ = train_ppo._bc_final_holdout_marker_path(
                    base, 2, _BC_V2_COLLECTION_EPISODES)
                record = json.loads(marker.read_text())
                if mutation == "pool":
                    record["pool_sha256"] = "0" * 64
                elif mutation == "marker_schema":
                    record["marker_schema_version"] = (
                        "bc-final-holdout-consumption/future")
                elif mutation == "provenance":
                    record["provenance"]["implementation_sha256"] = "0" * 64
                else:
                    record["unexpected"] = True
                marker.write_text(json.dumps(record))
                if mutation == "provenance":
                    report_path = p.with_name("bc_report_v2.json")
                    report = json.loads(report_path.read_text())
                    report["final_holdout_marker_sha256"] = hashlib.sha256(
                        marker.read_bytes()).hexdigest()
                    report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, message):
                    _load_bc_aux_demos_v2(
                        p, expected_manager_sha256=CANONICAL_MANAGER_SHA)

    def test_v2_report_marker_hash_tamper_is_rejected(self):
        p = self._tmp() / "demos.npz"
        _write_v2_pass_bundle(p)
        report_path = p.with_name("bc_report_v2.json")
        report = json.loads(report_path.read_text())
        report["final_holdout_marker_sha256"] = "0" * 64
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(
                ValueError, "PASS report 未精确绑定 final pool/marker"):
            _load_bc_aux_demos_v2(
                p, expected_manager_sha256=CANONICAL_MANAGER_SHA)

    def test_calibration_split_cannot_be_forged_by_preserving_totals(self):
        p = self._tmp() / "demos.npz"
        _write_v2_pass_bundle(p)
        report_path = p.with_name("bc_report_v2.json")
        report = json.loads(report_path.read_text())
        calibration = report["a12_calibration"]
        calibration["fit_pairs"] -= 80
        calibration["validation_pairs_excluded"] += 80
        calibration["fit_episodes"] -= 1
        calibration["validation_episodes_excluded"] += 1
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "三域确定性切分不一致"):
            _load_bc_aux_demos_v2(
                p, expected_manager_sha256=CANONICAL_MANAGER_SHA)

    def test_calibration_metrics_reject_bool_negative_and_unbound_values(self):
        mutations = (
            ("tp", -5),
            ("fp", True),
            ("precision_12", True),
            ("predicted_share_12", 999.0),
            ("legal_negative_probability_12_mean", True),
            ("legal_negative_probability_12_max", 999.0),
        )
        for key, value in mutations:
            base = self._tmp() / key
            base.mkdir()
            p = base / "demos.npz"
            _write_v2_pass_bundle(p)
            report_path = p.with_name("bc_report_v2.json")
            report = json.loads(report_path.read_text())
            report["a12_calibration"]["fit_metrics"][key] = value
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(
                    ValueError,
                    "fit (计数|指标)未绑定现场策略|"
                    "fit_metrics\\..*必须是数值"):
                _load_bc_aux_demos_v2(
                    p, expected_manager_sha256=CANONICAL_MANAGER_SHA)

    def test_fit_probability_receipt_mean_and_max_are_exactly_recomputed(self):
        for key in (
                "legal_negative_probability_12_mean",
                "legal_negative_probability_12_max"):
            with self.subTest(key=key):
                base = self._tmp() / key
                base.mkdir()
                p = base / "demos.npz"
                _write_v2_pass_bundle(p)
                report_path = p.with_name("bc_report_v2.json")
                report = json.loads(report_path.read_text())
                original = report["a12_calibration"]["fit_metrics"][key]
                self.assertGreater(original, 0.0)
                self.assertLessEqual(
                    original,
                    (train_ppo._A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX
                     if key.endswith("_mean")
                     else train_ppo._A12_LEGAL_NEGATIVE_PROBABILITY_MAX))
                # 仍是有限、[0,1] 且不超门的伪值，也必须因不等于现场
                # 六张量重算而拒绝，不能只做 schema/range 检查。
                report["a12_calibration"]["fit_metrics"][key] = (
                    original * 0.5)
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(
                        ValueError, "fit 指标未绑定现场策略"):
                    _load_bc_aux_demos_v2(
                        p, expected_manager_sha256=CANONICAL_MANAGER_SHA)

    def test_fit_probability_limit_rejects_argmax_safe_policy(self):
        p = self._tmp() / "demos.npz"
        x, y, ep, masks = _write_v2_pass_bundle(p)
        policy = th.load(
            p.with_name("policy_sd.pt"),
            map_location="cpu", weights_only=True)
        # 提高到刚越过 max 概率线，但仍远低于竞争动作：
        # argmax/FPR 继续为零，只有概率安全门能抓住。
        policy["action_net.bias"][12] = 3.1
        fit, _ = train_ppo._bc_v2_fit_validation_indices(ep)
        behavior = bc_aux_behavior_metrics(
            policy, x[fit], y[fit], ep[fit], masks[fit],
            heldout_only=False)
        self.assertEqual(behavior["fp"], 0)
        self.assertEqual(behavior["fpr_12"], 0.0)
        self.assertGreater(
            behavior["legal_negative_probability_12_max"],
            train_ppo._A12_LEGAL_NEGATIVE_PROBABILITY_MAX)

        report = json.loads(
            p.with_name("bc_report_v2.json").read_text())
        calibration = report["a12_calibration"]
        calibration["bias_12"] = float(
            policy["action_net.bias"][12].item())
        calibration["fit_metrics"] = {
            key: behavior[key]
            for key in (
                "tp", "fp", "precision_12", "recall_12", "fpr_12",
                "predicted_share_12", "high_hp_false_drink_rate",
                "legal_negative_probability_12_mean",
                "legal_negative_probability_12_max",
                "a13_spillover",
            )
        }
        with self.assertRaisesRegex(
                ValueError, "现场 fit 安全门未过"):
            train_ppo._validate_bc_v2_calibration_receipt(
                report, policy, x, y, ep, masks)

    def test_calibration_receipt_is_mandatory_at_ppo_entry(self):
        p = self._tmp() / "demos.npz"
        _write_v2_pass_bundle(p)
        report_path = p.with_name("bc_report_v2.json")
        report = json.loads(report_path.read_text())
        report.pop("a12_calibration")
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "字段/schema 不精确"):
            _load_bc_aux_demos_v2(
                p, expected_manager_sha256=CANONICAL_MANAGER_SHA)

    def test_bundle_manager_must_match_actual_training_manager(self):
        p = self._tmp() / "demos.npz"
        _write_v2_pass_bundle(p)
        with self.assertRaisesRegex(ValueError, "经理分布.*不一致"):
            _load_bc_aux_demos_v2(
                p, expected_manager_sha256="0" * 64)
        # 同一 bundle 对其真实采集经理仍可通过，证明不是硬编码拒绝非空输入。
        loaded = _load_bc_aux_demos_v2(
            p, expected_manager_sha256=CANONICAL_MANAGER_SHA)
        self.assertEqual(loaded[-1], hashlib.sha256(p.read_bytes()).hexdigest())

    def test_embedded_metadata_without_pass_report_is_rejected(self):
        p = self._tmp() / "demos.npz"
        _write_v2_npz(p)
        with self.assertRaisesRegex(ValueError, "PASS 回执缺失"):
            _load_bc_aux_demos_v2(
                p, expected_manager_sha256=CANONICAL_MANAGER_SHA)

    def test_bundle_rejects_running_report_and_byte_tamper(self):
        base = self._tmp()
        p = base / "demos.npz"
        _write_v2_pass_bundle(p)
        report_path = base / "bc_report_v2.json"
        report = json.loads(report_path.read_text())
        report["data_gate"] = "RUNNING"
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "未通过"):
            _load_bc_aux_demos_v2(
                p, expected_manager_sha256=CANONICAL_MANAGER_SHA)

        _write_v2_pass_bundle(p)
        with open(p, "ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "未绑定现场 demos"):
            _load_bc_aux_demos_v2(
                p, expected_manager_sha256=CANONICAL_MANAGER_SHA)

        _write_v2_pass_bundle(p)
        (base / "policy_sd.pt").write_bytes(b"tampered-policy")
        with self.assertRaisesRegex(ValueError, "未绑定现场 policy"):
            _load_bc_aux_demos_v2(
                p, expected_manager_sha256=CANONICAL_MANAGER_SHA)

    def test_hash_bound_arbitrary_policy_bytes_are_rejected(self):
        base = self._tmp()
        p = base / "demos.npz"
        _write_v2_pass_bundle(p)
        policy = base / "policy_sd.pt"
        report_path = base / "bc_report_v2.json"
        report = json.loads(report_path.read_text())
        policy.write_bytes(b"hash-bound-but-not-a-torch-state-dict")
        report["policy_sha256"] = hashlib.sha256(
            policy.read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "weights-only"):
            _load_bc_aux_demos_v2(
                p, expected_manager_sha256=CANONICAL_MANAGER_SHA)

    def test_parseable_policy_with_failing_behavior_is_rejected(self):
        base = self._tmp()
        p = base / "demos.npz"
        x, y, ep, masks = _write_v2_pass_bundle(p)
        policy = base / "policy_sd.pt"
        state = th.load(policy, map_location="cpu", weights_only=True)
        state["action_net.bias"][12] = 30.0
        th.save(state, policy)
        collapsed = bc_aux_behavior_metrics(
            state, x, y, ep, masks, heldout_only=True)
        self.assertEqual(
            train_ppo.bc_aux_behavior_gate(collapsed)["verdict"], "FAIL")
        report_path = base / "bc_report_v2.json"
        report = json.loads(report_path.read_text())
        report["policy_sha256"] = hashlib.sha256(
            policy.read_bytes()).hexdigest()
        # 攻击者同时伪造“读数来自该权重”，但仍把 gate 字段写成 PASS；
        # loader 必须以现场重算 gate 为准。
        report["a12_behavior"] = collapsed
        report["a12_behavior_gate"] = {"verdict": "PASS"}
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(
                ValueError,
                "a12_calibration bias_12 未绑定|"
                "held_out_top1.*重算不一致|现场权重重算.*未 PASS"):
            _load_bc_aux_demos_v2(
                p, expected_manager_sha256=CANONICAL_MANAGER_SHA)

    def test_forged_behavior_report_is_rejected(self):
        base = self._tmp()
        p = base / "demos.npz"
        _write_v2_pass_bundle(p)
        report_path = base / "bc_report_v2.json"
        report = json.loads(report_path.read_text())
        report["a12_behavior"]["tp"] += 1
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "a12_behavior.*重算不一致"):
            _load_bc_aux_demos_v2(
                p, expected_manager_sha256=CANONICAL_MANAGER_SHA)

    def test_policy_tensor_shape_and_finiteness_are_enforced(self):
        for mutation in ("shape", "nan"):
            base = self._tmp()
            p = base / f"{mutation}" / "demos.npz"
            p.parent.mkdir()
            _write_v2_pass_bundle(p)
            policy = p.with_name("policy_sd.pt")
            state = th.load(policy, map_location="cpu", weights_only=True)
            if mutation == "shape":
                state["action_net.bias"] = th.zeros(14)
            else:
                state["action_net.bias"][0] = float("nan")
            th.save(state, policy)
            report_path = p.with_name("bc_report_v2.json")
            report = json.loads(report_path.read_text())
            report["policy_sha256"] = hashlib.sha256(
                policy.read_bytes()).hexdigest()
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "形制/有限性异常"):
                _load_bc_aux_demos_v2(
                    p, expected_manager_sha256=CANONICAL_MANAGER_SHA)

    def test_legacy_v3_shape_without_embedded_provenance_rejected(self):
        p = self._tmp() / "legacy.npz"
        x = np.zeros((8, 298), dtype=np.float32)
        y = np.asarray([9, 12] * 4, dtype=np.int64)
        masks = np.ones((8, 15), dtype=bool)
        masks[:, 11] = False
        np.savez_compressed(
            p, X=x, Y=y,
            episode_id=np.asarray([0] * 4 + [1] * 4, dtype=np.int64),
            masks=masks)
        with self.assertRaisesRegex(ValueError, "schema/provenance"):
            _parse_bc_aux_demos_v2(p)

    def test_protocol_and_generator_provenance_tamper_rejected(self):
        base = self._tmp()
        stale = base / "stale.npz"
        _write_v2_npz(
            stale,
            protocol_version=np.asarray(PROTOCOL_VERSION - 1, dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "协议过期"):
            _parse_bc_aux_demos_v2(stale)
        forged = base / "forged.npz"
        _write_v2_npz(forged, generator_sha256=np.asarray("0" * 64))
        with self.assertRaisesRegex(ValueError, "generator_sha256"):
            _parse_bc_aux_demos_v2(forged)

    def test_missing_masks_key_rejected(self):
        p = self._tmp() / "demos.npz"
        x = np.zeros((6, 298), dtype=np.float32)
        y = np.asarray([9, 12] * 3, dtype=np.int64)
        ep = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        np.savez_compressed(p, X=x, Y=y, episode_id=ep)
        with self.assertRaisesRegex(ValueError, "masks"):
            _parse_bc_aux_demos_v2(p)

    def test_generation_conditioned_forbidden_labels(self):
        # v2 禁 11 允 12(守卫面不弱化;v1 禁 (11,12) 系 bc_worker 面原封)
        p = self._tmp() / "demos.npz"
        _write_v2_npz(p, labels=[9, 11] * 4)
        with self.assertRaisesRegex(ValueError, "禁采动作 11"):
            _parse_bc_aux_demos_v2(p)

    def test_label_out_of_range_rejected(self):
        p = self._tmp() / "demos.npz"
        _write_v2_npz(p, labels=[9, 15] * 4)
        with self.assertRaisesRegex(ValueError, "标签越界"):
            _parse_bc_aux_demos_v2(p)

    def test_masks_shape_and_dtype_rejected(self):
        base = self._tmp()
        p14 = base / "d14.npz"
        _write_v2_npz(p14, masks=np.ones((8, 14), dtype=bool))
        with self.assertRaisesRegex(ValueError, "masks 形状/dtype"):
            _parse_bc_aux_demos_v2(p14)
        pint = base / "dint.npz"
        _write_v2_npz(pint, masks=np.ones((8, 15), dtype=np.int8))
        with self.assertRaisesRegex(ValueError, "masks 形状/dtype"):
            _parse_bc_aux_demos_v2(pint)

    def test_label_masked_by_own_mask_rejected(self):
        p = self._tmp() / "demos.npz"
        masks = np.ones((8, 15), dtype=bool)
        masks[0, 12] = False                        # 第 0 行标签 12 被自身掩码禁止
        _write_v2_npz(p, masks=masks)
        with self.assertRaisesRegex(ValueError, "on-manifold 破缺"):
            _parse_bc_aux_demos_v2(p)

    def test_obs_shape_dtype_and_episode_diversity_mirrored(self):
        base = self._tmp()
        p297 = base / "d297.npz"
        _write_v2_npz(p297, obs_dim=297)
        with self.assertRaisesRegex(ValueError, "数组形状异常"):
            _parse_bc_aux_demos_v2(p297)
        p64 = base / "d64.npz"
        _write_v2_npz(p64, x_dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "dtype 异常"):
            _parse_bc_aux_demos_v2(p64)
        pep = base / "dep.npz"
        _write_v2_npz(pep, episode_ids=[0] * 8)
        with self.assertRaisesRegex(ValueError, "独立局数"):
            _parse_bc_aux_demos_v2(pep)


class BcAuxSubsetFilterTests(unittest.TestCase):
    """rev8:正例/合法负例入 bank；负闩+m12=False 只作覆盖证据。"""

    def test_filter_keeps_positives_and_hard_negatives(self):
        x = np.zeros((6, 298), dtype=np.float32)
        y = np.asarray([9, 12, 10, 9, 13, 9], dtype=np.int64)
        x[:, 0] = np.arange(6, dtype=np.float32)
        x[[0, 2], 297] = -1.5
        masks = np.ones((6, 15), dtype=bool)
        masks[[0, 2], 12] = False
        fx, fy, fmasks = _filter_bc_aux_demo_pairs(x, y, masks)
        self.assertEqual(list(fy[:1]), [12])
        self.assertEqual(set(fy[1:].tolist()), {9, 13})
        self.assertEqual(len(fy), 4)
        np.testing.assert_array_equal(fx[:1], x[[1]])
        self.assertTrue(fmasks[:, 12].all())
        self.assertFalse((fx[:, 297] < 0.0).any())

    def test_filter_fails_loud_without_class12(self):
        x = np.zeros((3, 298), dtype=np.float32)
        y = np.asarray([9, 10, 13], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "没有 12 类示范对"):
            _filter_bc_aux_demo_pairs(x, y, np.ones((3, 15), dtype=bool))

    def test_filter_asserts_m12_true_on_class12_pairs(self):
        # 图纸字面(G0-2b/E7):全部 12 类示范对断言 m[12]=True。
        x = np.zeros((2, 298), dtype=np.float32)
        y = np.asarray([12, 12], dtype=np.int64)
        masks = np.ones((2, 15), dtype=bool)
        masks[1, 12] = False
        with self.assertRaisesRegex(ValueError, r"m\[12\]=False"):
            _filter_bc_aux_demo_pairs(x, y, masks)

    def test_filter_fails_loud_without_legal_hard_negative(self):
        x = np.zeros((3, 298), dtype=np.float32)
        y = np.asarray([12, 9, 10], dtype=np.int64)
        masks = np.ones((3, 15), dtype=bool)
        masks[1:, 12] = False
        with self.assertRaisesRegex(ValueError, "hard negative"):
            _filter_bc_aux_demo_pairs(x, y, masks)

    def test_filter_fails_loud_without_visible_post_drink_negative(self):
        x = np.zeros((4, 298), dtype=np.float32)
        x[:, 0] = [0.55, 0.55, 0.8, 0.8]
        y = np.asarray([12, 9, 10, 13], dtype=np.int64)
        masks = np.ones((4, 15), dtype=bool)
        with self.assertRaisesRegex(ValueError, "可见后饮关闩证据"):
            _filter_bc_aux_demo_pairs(x, y, masks)

    def test_post_drink_evidence_requires_negative_latch_and_closed_mask(self):
        x = np.zeros((5, 298), dtype=np.float32)
        x[:, 0] = [0.55, 0.55, 0.55, 0.8, 0.8]
        y = np.asarray([12, 9, 10, 9, 13], dtype=np.int64)
        masks = np.ones((5, 15), dtype=bool)
        # 单有负闩但 m12 仍开、或单有关闭 m12 但闩非负，都不能算后饮证据。
        x[1, 297] = -1.5
        masks[2, 12] = False
        with self.assertRaisesRegex(ValueError, "可见后饮关闩证据"):
            _filter_bc_aux_demo_pairs(x, y, masks)


class BcAuxRngTests(unittest.TestCase):
    """E3③:demo minibatch 专用流,播种规则同 E1③ 形制(固定偏移确定性派生)。"""

    def test_fixed_offset_derivation_is_deterministic(self):
        self.assertEqual(
            derive_bc_aux_rng(0).bit_generator.state,
            np.random.default_rng(_BC_AUX_SEED_OFFSET).bit_generator.state)
        a = [derive_bc_aux_rng(304000).random() for _ in range(8)]
        b = [derive_bc_aux_rng(304000).random() for _ in range(8)]
        c = [derive_bc_aux_rng(304001).random() for _ in range(8)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_stream_disjoint_from_training_and_p_skip_families(self):
        # 偏移移出 [0, 2**32) 训练种子域;与 E1 p_skip 流族 (seed+rank)+2**33+26
        # 之偏移差 ≥ 2**33 >> rank,构造性不相交(种子级同流即同序列,故查种子)。
        self.assertGreaterEqual(_BC_AUX_SEED_OFFSET, 2**32)
        self.assertGreaterEqual(abs(_BC_AUX_SEED_OFFSET - _P_SKIP_SEED_OFFSET),
                                2**33)
        base = derive_bc_aux_rng(304000).bit_generator.state
        self.assertNotEqual(
            base, np.random.default_rng(304000).bit_generator.state)
        for rank in range(64):
            self.assertNotEqual(
                base, _derive_p_skip_rng(304000 + rank).bit_generator.state)

    def test_seed_none_mirrors_worker_env_fallback(self):
        self.assertIsInstance(derive_bc_aux_rng(None), np.random.Generator)

    def test_bc_seed_ranges_and_automatic_sampler_are_fail_closed(self):
        self.assertEqual(
            BC_RESERVED_SEED_RANGES,
            (
                (2000, 2128),
                (3000, 3384),
                (2_100_000, 2_100_128),
                (2_101_000, 2_101_384),
                (2_102_000, 2_102_128),
                (2_103_000, 2_103_384),
                (2_104_000, 2_104_128),
                (2_106_000, 2_106_128),
            ),
        )
        for seed, expected in (
                (99, False), (100, True), (483, True), (484, False),
                (999, False), (1000, True), (1383, True), (1384, False),
                (1999, False), (2000, True), (2127, True), (2128, False),
                (2999, False), (3000, True), (3383, True), (3384, False),
                (2_099_999, False),
                (2_100_000, True), (2_100_127, True),
                (2_100_128, False),
                (2_101_000, True), (2_101_383, True),
                (2_101_384, False),
                (2_102_000, True), (2_102_127, True),
                (2_102_128, False),
                (2_103_000, True), (2_103_383, True),
                (2_103_384, False),
                (2_103_999, False),
                (2_104_000, True), (2_104_127, True),
                (2_104_128, False),
                (2_105_999, False),
                (2_106_000, True), (2_106_127, True),
                (2_106_128, False)):
            self.assertIs(is_reserved_train_seed(seed), expected, seed)

        class ScriptedRng:
            def __init__(self):
                self.values = iter((
                    100, 483, 1000, 1383,
                    2000, 2127, 3000, 3383,
                    2_100_000, 2_100_127,
                    2_101_000, 2_101_383,
                    2_102_000, 2_102_127,
                    2_103_000, 2_103_383,
                    2_104_000, 2_104_127,
                    2_106_000, 2_106_127,
                    42))
                self.calls = 0

            def integers(self, low, high):
                self.assert_bounds = (low, high)
                self.calls += 1
                return next(self.values)

        rng = ScriptedRng()
        self.assertEqual(sample_train_seed(rng), 42)
        self.assertEqual(rng.calls, 21)
        self.assertEqual(rng.assert_bounds, (0, 2**31))


class BcAuxTrainLoopTests(unittest.TestCase):
    """E3①④⑤:辅助 CE 通路(免引擎缩样,TinyMasked15Env + DummyVecEnv)。"""

    def _model(self, lam=0.0, seed=7):
        model = _make_model(lam=lam, seed=seed)
        self.addCleanup(model.env.close)
        return model

    def test_ctor_rejects_bad_lambda(self):
        with self.assertRaisesRegex(ValueError, "bc_aux_lambda"):
            self._model(lam=-0.1)
        with self.assertRaisesRegex(ValueError, "bc_aux_lambda"):
            self._model(lam=float("nan"))

    def test_mount_validates_shapes_masks_and_rng(self):
        model = self._model()
        x, y, masks = _demo_bank()
        bad = masks.copy()
        bad[0, 12] = False                          # 标签 12 被自身掩码禁止
        with self.assertRaisesRegex(ValueError, "掩码禁止"):
            model.mount_bc_aux_demos(x, y, bad, rng=derive_bc_aux_rng(4))
        with self.assertRaisesRegex(ValueError, "示范观测形状"):
            model.mount_bc_aux_demos(x[:, :3], y, masks, derive_bc_aux_rng(4))
        with self.assertRaisesRegex(ValueError, "掩码形状"):
            model.mount_bc_aux_demos(x, y, masks[:, :14], derive_bc_aux_rng(4))
        with self.assertRaisesRegex(ValueError, "标签"):
            model.mount_bc_aux_demos(x, y.astype(np.float32), masks,
                                     derive_bc_aux_rng(4))
        with self.assertRaisesRegex(ValueError, "专用 np.random.Generator"):
            model.mount_bc_aux_demos(x, y, masks, rng=None)

    def test_resume_mount_uses_persistent_root_not_current_leg_start(self):
        model = self._model(lam=0.015625, seed=41)
        root = _persistent_bc_aux_root_anchor(model)
        x, y, masks = _demo_bank(n=64, seed=43)

        root_logits = train_ppo._policy_logits_from_sb3_state_dict(root, x)
        mask_t = th.as_tensor(masks)
        expected_root = th.softmax(
            th.where(mask_t, root_logits,
                     th.full_like(root_logits, -1e8)), dim=-1)
        with th.no_grad():
            # 模拟上一腿已经发生的显著漂移；persistent root 字段仍是首次根。
            model.policy.action_net.bias[12] += 30.0
            current = model.policy.get_distribution(
                th.as_tensor(x), action_masks=mask_t)
            current_probs = current.distribution.logits.exp().detach()

        recovered = _persistent_bc_aux_root_anchor(model)
        self.assertEqual(
            train_ppo._policy_head_sha256(recovered),
            train_ppo._policy_head_sha256(root))
        model.mount_bc_aux_demos(
            x, y, masks, rng=derive_bc_aux_rng(41))
        th.testing.assert_close(
            model._bc_aux_anchor_probs.cpu(), expected_root,
            rtol=1e-6, atol=1e-7)
        self.assertFalse(th.allclose(
            model._bc_aux_anchor_probs.cpu(), current_probs.cpu()))

    def test_lambda_positive_without_bank_fails_loud(self):
        # 反 P2:λ>0 而池未挂载不得静默空转(镜像 β>0 无教师条款)。
        model = self._model(lam=0.015625)
        model._setup_learn(total_timesteps=8)
        _fill_buffer(model)
        with self.assertRaisesRegex(RuntimeError, "示范池未挂载"):
            model.train()

    def test_zero_lambda_is_zero_intrusion(self):
        # λ=0:辅助分支零调用、专用流零消耗、policy+optimizer 与"池不在位"
        # 双胞胎逐张量 torch.equal(G0-2a 张量级恒等之免引擎缩样)。
        calls = []
        snapshots = []
        for mounted in (True, False):
            model = self._model(lam=0.0, seed=7)
            if mounted:
                rng = derive_bc_aux_rng(3)
                state_before = rng.bit_generator.state
                model.mount_bc_aux_demos(*_demo_bank(), rng=rng)
                original = model._bc_aux_ce_loss
                model._bc_aux_ce_loss = (
                    lambda: calls.append(1) or original())
            model._setup_learn(total_timesteps=8)
            _fill_buffer(model)
            model.train()
            self.assertIsNone(model._last_bc_aux_ce)
            snapshots.append((model.policy.state_dict(),
                              model.policy.optimizer.state_dict()))
            if mounted:
                self.assertEqual(rng.bit_generator.state, state_before)
        self.assertEqual(calls, [])
        (policy_a, optim_a), (policy_b, optim_b) = snapshots
        self.assertEqual(list(policy_a), list(policy_b))
        for key in policy_a:
            self.assertTrue(th.equal(policy_a[key], policy_b[key]), key)
        self.assertTrue(_tree_equal(optim_a, optim_b))

    def test_12_head_gradient_nonzero_via_autograd(self):
        # G0-2b 单测先立件:autograd 实测辅助 CE 对 action_net 第 12 行梯度非零。
        model = self._model(lam=0.015625)
        model.mount_bc_aux_demos(*_demo_bank(), rng=derive_bc_aux_rng(1))
        ce = model._bc_aux_ce_loss()
        grad_w, grad_b = th.autograd.grad(
            ce, [model.policy.action_net.weight, model.policy.action_net.bias])
        self.assertGreater(float(grad_w[12].abs().sum()), 0.0)
        self.assertNotEqual(float(grad_b[12]), 0.0)

    def test_12_head_train_fingerprint_lambda_gated(self):
        # 真 train() 指纹:rollout 面 12 恒掩时,PPO 自身对 12 头仍有 O(1e-10)
        # 级残留梯度——sb3_contrib MaskableCategorical 之 _original_logits 系
        # 未掩 logsumexp 归一化后逻辑值,掩前归一化项对 raw logit 12 留有
        # softmax_unmasked_12 微通道(施工实测呈报件,详交接单)。故指纹取
        # 数量级分离而非位级冻结:λ>0 之行 12 位移须比 λ=0 残留位移大 ≥1e2 倍。
        deltas = {}
        for lam in (0.015625, 0.0):
            model = self._model(lam=lam)
            model.mount_bc_aux_demos(*_demo_bank(), rng=derive_bc_aux_rng(2))
            model._setup_learn(total_timesteps=8)
            _fill_buffer(model)
            before = model.policy.action_net.weight.detach().clone()
            model.train()
            after = model.policy.action_net.weight.detach()
            deltas[lam] = float((after[12] - before[12]).abs().sum())
            self.assertFalse(th.equal(before[9], after[9]))   # 其余头照常在学
            if lam > 0:
                self.assertIsInstance(model._last_bc_aux_ce, float)
            else:
                self.assertIsNone(model._last_bc_aux_ce)
        self.assertLess(deltas[0.0], 1e-7)          # λ=0:仅掩码归一化残留
        self.assertGreater(deltas[0.015625], 1e-5)  # λ>0:辅助通路真实到达 12 头
        self.assertGreater(deltas[0.015625], 1e2 * deltas[0.0])

    def test_aux_forward_consumes_per_sample_masks(self):
        # rev2 全 bank 要求 m12=True；正例仅开 12、负例开 9/12 的可辨识
        # 花纹与全开掩码应产生不同校准损失，证明逐样本 masks 真被消费。
        model = self._model(lam=0.015625)
        x, y, _ = _demo_bank()
        structured = np.zeros((len(y), 15), dtype=bool)
        structured[:, 12] = True
        structured[y != 12, 9] = True
        model.mount_bc_aux_demos(x, y, structured, rng=derive_bc_aux_rng(5))
        ce_masked = float(model._bc_aux_ce_loss().detach())
        model.mount_bc_aux_demos(x, y, np.ones((len(y), 15), dtype=bool),
                                 rng=derive_bc_aux_rng(5))
        ce_open = float(model._bc_aux_ce_loss().detach())
        self.assertTrue(np.isfinite(ce_masked))
        self.assertTrue(np.isfinite(ce_open))
        self.assertGreater(abs(ce_masked - ce_open), 1e-4)

    def test_one_aux_batch_per_train_call_not_per_minibatch(self):
        model = self._model(lam=0.015625)
        model.mount_bc_aux_demos(*_demo_bank(), rng=derive_bc_aux_rng(8))
        model._setup_learn(total_timesteps=16)
        calls = 0
        original = model._bc_aux_ce_loss

        def counted():
            nonlocal calls
            calls += 1
            return original()

        model._bc_aux_ce_loss = counted
        for seed in (31, 32):
            _fill_buffer(model, seed=seed)
            model.train()
        self.assertEqual(_BC_AUX_UPDATE_EVERY, 1)
        self.assertEqual(calls, 2)  # 两次 train()，不是 n_epochs×minibatches

    def test_gcal_peek_does_not_consume_the_dedicated_aux_batch(self):
        model = self._model(lam=0.015625, seed=12)
        model.mount_bc_aux_demos(
            *_demo_bank(n=64), rng=derive_bc_aux_rng(12))
        before_rng = json.dumps(
            model._bc_aux_rng.bit_generator.state, sort_keys=True)
        before_permutations = {
            key: value.copy()
            for key, value in model._bc_aux_permutations.items()
        }
        before_cursors = dict(model._bc_aux_cursors)
        peek = model._peek_bc_aux_ce_loss()
        self.assertTrue(th.isfinite(peek))
        self.assertEqual(
            json.dumps(model._bc_aux_rng.bit_generator.state, sort_keys=True),
            before_rng)
        self.assertEqual(model._bc_aux_cursors, before_cursors)
        self.assertEqual(
            set(model._bc_aux_permutations), set(before_permutations))
        for key, value in before_permutations.items():
            np.testing.assert_array_equal(
                model._bc_aux_permutations[key], value)

        # 恢复后的真实独立步应消费恰好这一批，而非 G-CAL 后的第二批。
        actual = model._apply_bc_aux_step()
        self.assertTrue(th.isfinite(actual))
        self.assertNotEqual(
            json.dumps(model._bc_aux_rng.bit_generator.state, sort_keys=True),
            before_rng)

    def test_micro_optimization_raises_positive_p12_without_negative_collapse(self):
        model = self._model(lam=0.015625, seed=17)
        rng = np.random.default_rng(19)
        x = rng.normal(scale=0.1, size=(128, 4)).astype(np.float32)
        y = np.full(128, 9, dtype=np.int64)
        y[:32] = 12
        x[:32, 0] += 2.0
        x[32:, 0] -= 2.0
        masks = np.ones((128, 15), dtype=bool)
        masks[:, 11] = False
        model.mount_bc_aux_demos(x, y, masks, rng=derive_bc_aux_rng(19))

        def readout():
            with th.no_grad():
                dist = model.policy.get_distribution(
                    th.as_tensor(x), action_masks=th.as_tensor(masks))
                probs = dist.distribution.logits.exp()
                pred = probs.argmax(-1).cpu().numpy()
            return (float(probs[:32, 12].mean()),
                    float(probs[32:, 12].mean()),
                    float((pred[32:] == 12).mean()))

        before = readout()
        for _ in range(160):
            loss = model.bc_aux_lambda * model._bc_aux_ce_loss()
            model.policy.optimizer.zero_grad()
            loss.backward()
            model.policy.optimizer.step()
        after = readout()
        self.assertGreater(after[0], before[0] + 0.10)
        self.assertLess(after[1], before[1])
        self.assertLessEqual(after[2], 0.01)

    def test_gcal_records_aux_total_a12_grad_and_trips_bank_collapse(self):
        with tempfile.TemporaryDirectory() as d:
            model = self._model(lam=0.015625, seed=23)
            model.calib_out = str(pathlib.Path(d) / "calib.jsonl")
            model.mount_bc_aux_demos(
                *_demo_bank(n=64), rng=derive_bc_aux_rng(23))
            # anchor 已在 mount 冻结；随后构造“m12 合法便总喝”的坍缩。
            with th.no_grad():
                model.policy.action_net.bias[12] += 30.0
            aux = model._bc_aux_ce_loss()
            zero_pg = model.policy.action_net.bias.sum() * 0.0
            total = zero_pg + model.bc_aux_lambda * aux
            model._calib_probe(
                zero_pg, total.new_zeros(()), 0.0,
                total_loss=total, bc_aux_loss=aux)
            rec = json.loads(pathlib.Path(model.calib_out).read_text())
            self.assertEqual(rec["bc_aux_lambda"], 0.015625)
            self.assertGreater(rec["g_aux"], 0.0)
            self.assertGreater(rec["g_aux_a12"], 0.0)
            self.assertGreater(rec["g_total"], 0.0)
            self.assertGreater(rec["g_total_a12"], 0.0)
            self.assertIn("g_aux_vs_other_a12_cosine", rec)
            self.assertIn("g_total_on_aux_a12_projection", rec)
            self.assertGreater(
                rec["g_total_on_aux_a12_projection"], 0.0)
            self.assertGreater(rec["offpolicy_bank"]["fpr_12"], 0.9)
            self.assertTrue(rec["tripped"])
            self.assertTrue(model._calib_tripped)
            self.assertTrue(any("bank_fpr12" in reason
                                for reason in rec["trip_reasons"]))

    def test_gcal_trips_when_total_a12_gradient_opposes_aux_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self._model(lam=0.015625, seed=24)
            model.calib_out = str(pathlib.Path(directory) / "conflict.jsonl")
            model.mount_bc_aux_demos(
                *_demo_bank(n=64), rng=derive_bc_aux_rng(24))
            aux = model._bc_aux_ce_loss()
            aux_term = model.bc_aux_lambda * aux
            # Mirror an overwhelmingly opposed PPO/KING contribution:
            # other=-2*aux, therefore combined total=-aux and projection=-1.
            policy_loss = -2.0 * aux_term
            total = -aux_term
            model._calib_probe(
                policy_loss, total.new_zeros(()), 0.0,
                total_loss=total, bc_aux_loss=aux)
            record = json.loads(pathlib.Path(model.calib_out).read_text())
            self.assertAlmostEqual(
                record["g_total_on_aux_a12_projection"], -1.0, places=5)
            self.assertIn(
                "total_on_aux_a12_projection<=0",
                record["trip_reasons"])
            self.assertTrue(record["tripped"])
            self.assertTrue(model._calib_tripped)

    def test_gcal_fails_closed_on_missing_or_nonfinite_aux_gradient(self):
        for mode in ("missing", "nonfinite"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                model = self._model(
                    lam=0.015625, seed=25 if mode == "missing" else 26)
                model.calib_out = str(
                    pathlib.Path(directory) / f"{mode}.jsonl")
                model.mount_bc_aux_demos(
                    *_demo_bank(n=64),
                    rng=derive_bc_aux_rng(
                        25 if mode == "missing" else 26))
                factor = 0.0 if mode == "missing" else float("nan")
                aux = model.policy.action_net.bias[12] * factor
                total = model.bc_aux_lambda * aux
                model._calib_probe(
                    total, total.new_zeros(()), 0.0,
                    total_loss=total, bc_aux_loss=aux)
                record = json.loads(
                    pathlib.Path(model.calib_out).read_text())
                expected = (
                    "aux_gradient_missing" if mode == "missing"
                    else "aux_gradient_nonfinite")
                self.assertIn(expected, record["trip_reasons"])
                self.assertTrue(record["tripped"])
                self.assertTrue(model._calib_tripped)
                if mode == "nonfinite":
                    self.assertIsNone(record["g_aux"])
                    self.assertIsNone(record["g_aux_a12"])
                    self.assertIsNone(record["bc_aux_loss"])

    def test_rollout_monitor_ignores_low_recall_but_trips_new_overdrink(self):
        model = self._model(lam=0.015625, seed=29)
        model.mount_bc_aux_demos(
            *_demo_bank(n=128), rng=derive_bc_aux_rng(29))
        start = model._bc_aux_rollout_monitor()
        self.assertFalse(start["tripped"])
        self.assertFalse(start["lower_recall_gate_applied"])
        # 起点 recall 无论高低都不能作为中途停训理由。
        self.assertFalse(any("recall" in reason
                             for reason in start["trip_reasons"]))
        with th.no_grad():
            model.policy.action_net.bias[12] += 30.0
        collapsed = model._bc_aux_rollout_monitor()
        self.assertTrue(collapsed["tripped"])
        self.assertTrue(model._calib_tripped)
        self.assertTrue(any(
            "fpr12" in reason or "predicted_share12" in reason
            for reason in collapsed["trip_reasons"]))

    def test_dedicated_rng_stream_zero_contamination(self):
        # E3③/G0-2b 缩样:在位(λ>0+池挂载)/不在位两跑,训练 RNG(torch 全局 +
        # numpy 全局)状态轨迹逐点相等;专用流自身实被消费。
        trajectories = []
        consumed_states = []
        for mounted in (True, False):
            model = self._model(lam=0.015625 if mounted else 0.0, seed=7)
            rng = None
            if mounted:
                rng = derive_bc_aux_rng(304000)
                model.mount_bc_aux_demos(*_demo_bank(), rng=rng)
            model._setup_learn(total_timesteps=8)
            th.manual_seed(1234)
            np.random.seed(1234)
            states = []
            for round_seed in (21, 22):
                _fill_buffer(model, seed=round_seed)
                model.train()
                np_state = np.random.get_state()
                states.append((th.get_rng_state().numpy().tobytes(),
                               np_state[1].tobytes(), int(np_state[2])))
            trajectories.append(states)
            if mounted:
                consumed_states.append(rng.bit_generator.state)
        self.assertEqual(trajectories[0], trajectories[1])
        self.assertNotEqual(consumed_states[0],
                            derive_bc_aux_rng(304000).bit_generator.state)


class BcAuxCliTests(unittest.TestCase):
    """E3②/E7:CLI 旗解析、互不强制、fail-loud(子进程,引擎零点火)。"""

    def _tmp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        out = pathlib.Path(directory.name) / "runs" / "bc-worker-v2"
        out.mkdir(parents=True)
        return out

    def _worker_argv(self, base, *aux):
        manager = base / "manager.npz"
        if not manager.exists():
            # active aux 路径现在必须把 bundle 的采集经理与训练经理逐字对账。
            manager.write_bytes(CANONICAL_MANAGER.read_bytes())
        # 普通 Worker 必须显式声明 legacy policy view；结构化 A12 则必须
        # 保留 raw protocol-v4 gate，因此二者不能共携。
        observation_flags = (
            () if "--bc-aux-graft" in aux
            else ("--legacy-worker-policy-observation-view",)
        )
        return ("--worker", "--algo", "mppo", "--gamma", "1.0",
                "--max-steps", "3000", "--manager-npz", str(manager),
                "--seed", "7000", *observation_flags,
                *aux)                               # 7000 撞探针段:安全终止哨

    @staticmethod
    def _formal_aux_flags():
        return (
            "--resume-from",
            str(ROOT / "train" / "models" / "v28-worker-leg1"
                / "model_final.zip"),
            "--bc-aux-liveness-preflight",
            "--reset-optimizer",
            "--distill-beta", "0.015625",
        )

    def test_help_documents_rev6_graft_and_zero_lambda_surface(self):
        run = _run_cli("--help")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--bc-aux-lambda", run.stdout)
        self.assertIn("--bc-aux-demos", run.stdout)
        self.assertIn("--bc-aux-graft", run.stdout)
        self.assertIn("--reset-optimizer", run.stdout)
        self.assertIn("--bc-aux-liveness-preflight", run.stdout)

    def test_negative_lambda_fails_loud(self):
        run = _run_cli("--bc-aux-lambda=-0.5")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("--bc-aux-lambda 必须是有限非负数", run.stderr)

    def test_lambda_alone_does_not_force_demos(self):
        # λ 单独在位 → 通路不在位,无耦合报错;推进至下游种子纪律闸即证
        run = _run_cli(*self._worker_argv(self._tmp(),
                                          "--bc-aux-lambda", "0.015625"))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("种子纪律", run.stderr)
        self.assertNotIn("④乙", run.stderr)

    def test_explicit_bc_pool_seeds_and_rank_overlap_are_rejected(self):
        for seed in (
                100, 483, 1000, 1383,
                2000, 2127, 3000, 3383):
            argv = list(self._worker_argv(self._tmp()))
            argv[argv.index("--seed") + 1] = str(seed)
            run = _run_cli(*argv)
            self.assertNotEqual(run.returncode, 0, seed)
            self.assertIn("种子纪律", run.stderr)

        # 首 rank 自身安全也不够；seed+任一真实 env rank 撞池同样必须拒绝。
        argv = list(self._worker_argv(self._tmp()))
        argv[argv.index("--seed") + 1] = "1999"
        argv.extend(("--num-envs", "2"))
        run = _run_cli(*argv)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("种子纪律", run.stderr)

    def test_demos_alone_does_not_force_lambda_and_is_not_loaded(self):
        # demos 单独在位(路径故意不存在)→ 不在位即不加载、连存在性都不查(零侵入)
        run = _run_cli(*self._worker_argv(
            self._tmp(), "--bc-aux-demos", "/definitely/missing/demos.npz"))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("种子纪律", run.stderr)
        self.assertNotIn("④乙", run.stderr)

    def test_active_path_requires_worker_mppo(self):
        run = _run_cli("--worker", "--bc-aux-lambda", "0",
                       "--bc-aux-graft",
                       "--bc-aux-demos", "/missing/demos.npz")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("只适用于 --worker --algo mppo", run.stderr)

    def test_active_path_missing_file_fails_loud(self):
        run = _run_cli(*self._worker_argv(
            self._tmp(), "--bc-aux-lambda", "0",
            "--bc-aux-graft",
            "--bc-aux-demos", "/definitely/missing/demos.npz",
            *self._formal_aux_flags()))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("不存在", run.stderr)

    def test_active_path_rejects_v1_shaped_npz(self):
        base = self._tmp()
        v1 = base / "v1demos.npz"
        x = np.zeros((6, 298), dtype=np.float32)
        np.savez_compressed(
            v1, X=x, Y=np.asarray([9, 12] * 3, dtype=np.int64),
            episode_id=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64))
        run = _run_cli(*self._worker_argv(
            base, "--bc-aux-lambda", "0", "--bc-aux-graft",
            "--bc-aux-demos", str(v1),
            *self._formal_aux_flags()))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("masks", run.stderr)

    def test_active_path_with_valid_v2_passes_aux_validation(self):
        base = self._tmp()
        v2 = base / "v2demos.npz"
        _write_v2_pass_bundle(v2)
        run = _run_cli(*self._worker_argv(
            base, "--bc-aux-lambda", "0", "--bc-aux-graft",
            "--bc-aux-demos", str(v2),
            *self._formal_aux_flags()))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("种子纪律", run.stderr)        # 辅助面已全过,倒在安全哨
        self.assertNotIn("④乙", run.stderr)

    def test_active_path_requires_armed_calibration_and_liveness(self):
        base = self._tmp()
        missing = base / "not-read.npz"
        common = (
            "--bc-aux-lambda", "0",
            "--bc-aux-graft",
            "--bc-aux-demos", str(missing),
            "--resume-from",
            str(ROOT / "train" / "models" / "v28-worker-leg1"
                / "model_final.zip"),
            "--reset-optimizer",
            "--distill-beta", "0.015625",
        )
        no_preflight = _run_cli(*self._worker_argv(base, *common))
        self.assertNotEqual(no_preflight.returncode, 0)
        self.assertIn("--bc-aux-liveness-preflight", no_preflight.stderr)
        record_only = _run_cli(*self._worker_argv(
            base, *common, "--bc-aux-liveness-preflight",
            "--calib-record-only"))
        self.assertNotEqual(record_only.returncode, 0)
        self.assertIn("禁止 --calib-record-only", record_only.stderr)


class V1SurfaceRegressionTests(unittest.TestCase):
    """E3②:v1 面(:61 schema 常量/canonical 验证器/加载器)回归零破坏。"""

    def test_v1_constants_untouched(self):
        self.assertEqual(_BC_REPORT_SCHEMA_VERSION, 1)
        self.assertEqual(_WORKER_BC_FORBIDDEN_ACTIONS, (11, 12))
        self.assertEqual(_WORKER_BC_V2_FORBIDDEN_ACTIONS, (11,))
        self.assertEqual(_BC_PASS_KEYS["data_gate"], {
            "schema_version", "pairs", "held_out_top1", "held_out_pairs",
            "held_out_episodes", "class_recalls", "class_weighted_retry",
            "data_gate", "protocol_version", "implementation_sha256",
            "generator_sha256", "manager_npz_sha256", "policy_sha256",
            "demos_sha256", "final_pool_sha256",
            "final_holdout_marker_sha256",
        })

    def test_v1_loader_accepts_v1_npz_and_v2_validator_rejects_it(self):
        # 世代分账:同一 v1 形制文件——v1 加载器照常受理,v2 验证器 fail-loud。
        with tempfile.TemporaryDirectory() as directory:
            p = pathlib.Path(directory) / "demos.npz"
            x = np.zeros((6, 298), dtype=np.float32)
            x[0, 297] = 1.0
            np.savez_compressed(
                p, X=x, Y=np.asarray([9, 9, 10, 13, 9, 14], dtype=np.int64),
                episode_id=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64))
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            lx, ly, lsha = _load_dry_anchor_demos(p, sha)
            self.assertEqual(lsha, sha)
            self.assertEqual(lx.shape, (6, 298))
            self.assertEqual(len(ly), 6)
            with self.assertRaisesRegex(ValueError, "masks"):
                _parse_bc_aux_demos_v2(p)


class AnchorZeroTouchSourcePins(unittest.TestCase):
    """rev6 源码钉：KING 条件支持与结构化 graft 必须接进正式路径。"""

    def test_leashed_king_excludes_a12_and_circuit_is_step_firewalled(self):
        src = LEASHED_PPO.read_text()
        self.assertIn("distill_ce = -(t_probs * logp_all).sum(dim=-1).mean()",
                      src)
        self.assertIn(
            "loss = loss + effective_distill_beta * distill_ce", src)
        self.assertIn("distill_masks = _legacy_distillation_masks(", src)
        self.assertIn(
            "student_raw_logits = self._student_distillation_logits(", src)
        self.assertIn(
            "logp_all = _masked_log_softmax_from_raw(", src)
        self.assertIn("self._bc_aux_circuit_spec is not None", src)
        self.assertIn("self._protect_bc_aux_circuit_before_step()", src)
        self.assertIn(
            "self._assert_bc_aux_circuit_unchanged(circuit_snapshot)", src)

    def test_train_ppo_wiring_points_present(self):
        src = TRAIN_PPO.read_text()
        self.assertIn(
            "expected_manager_sha256=manager_npz_sha256", src)
        self.assertIn("_expand_policy_with_bc_aux_circuit(model)", src)
        self.assertIn("_calibrate_bc_aux_adapter_weight(", src)
        self.assertIn("model.mount_bc_aux_circuit_fit(*bc_aux_fit)", src)
        self.assertIn("model.mount_bc_aux_circuit_validation(", src)
        self.assertIn("model.bc_aux_lambda = args.bc_aux_lambda", src)
        self.assertIn(
            '"bc_aux_mode":\n'
            '                "expanded-trainable-a12-contextual-mixture"',
            src)
        self.assertIn(
            "observed_gate = bc_aux_behavior_gate(\n"
            "        observed_behavior, require_teacher_recall=False)",
            src)
        self.assertIn("_publish_model_final_with_bc_aux_gate(", src)

        producer_src = BC_WORKER.read_text()
        self.assertIn(
            "a12_behavior_gate = bc_aux_behavior_gate(\n"
            "        a12_behavior, require_teacher_recall=False)",
            producer_src)


if __name__ == "__main__":
    unittest.main()
