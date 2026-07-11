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
标定探针(G-CAL):到达 --calib-probes 指定全局步时,对首个 minibatch 用
autograd.grad 测 g_ce/g_pg 与 teacher_diverge,写 calib.jsonl;
diverge>20% 置 _calib_tripped,由哨兵回调终止本腿(驱动裁决重标定)。
v28:calib_record_only=True 时探针只记不裁(tripped 位照记入 jsonl,旗不武装
——续航起点分歧 41.5%,20% 阈值对定居点失义;面板 blocker 修正)。
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance

HUGE_NEG = -1e8


def build_teacher(sd_path: str) -> th.nn.Module:
    """从 SB3 键名 state_dict 组装冻结教师(PiHead 298→64→64→15 同构)。"""
    sd = th.load(sd_path, map_location="cpu")
    net = th.nn.Sequential(
        th.nn.Linear(sd["mlp_extractor.policy_net.0.weight"].shape[1], 64), th.nn.Tanh(),
        th.nn.Linear(64, 64), th.nn.Tanh(),
        th.nn.Linear(64, sd["action_net.weight"].shape[0]))
    with th.no_grad():
        net[0].weight.copy_(sd["mlp_extractor.policy_net.0.weight"])
        net[0].bias.copy_(sd["mlp_extractor.policy_net.0.bias"])
        net[2].weight.copy_(sd["mlp_extractor.policy_net.2.weight"])
        net[2].bias.copy_(sd["mlp_extractor.policy_net.2.bias"])
        net[4].weight.copy_(sd["action_net.weight"])
        net[4].bias.copy_(sd["action_net.bias"])
    net.eval()
    net.requires_grad_(False)
    return net


class LeashedMaskablePPO(MaskablePPO):
    def __init__(self, *args, distill_beta: float = 0.0, teacher_path: str | None = None,
                 calib_probes: list | None = None, calib_out: str | None = None, **kwargs):
        self.distill_beta = float(distill_beta)
        self.teacher_path = teacher_path
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
            self.teacher = build_teacher(self.teacher_path).to(self.device)
        # _excluded_save_params 成员在 load 后不存在,兜底重建
        for attr, dv in (("_calib_done", set()), ("_calib_tripped", False),
                         ("_last_distill_ce", None), ("_last_diverge", None)):
            if not hasattr(self, attr):
                setattr(self, attr, dv)

    def _excluded_save_params(self):
        # _last_*/_calib_* 不入 zip:β=0 腿的哨兵行必须报 null 而非上腿陈值;
        # _calib_tripped=True 若被 load 驮回,会在续训第一步误杀健康腿(审查团确认项)
        return super()._excluded_save_params() + [
            "teacher", "_last_distill_ce", "_last_diverge",
            "_calib_tripped", "_calib_done", "calib_record_only"]

    def _teacher_probs(self, obs: th.Tensor, action_masks: th.Tensor) -> th.Tensor:
        t_logits = self.teacher(obs)
        mask = action_masks.reshape(t_logits.shape).bool()
        t_logits = th.where(mask, t_logits, th.full_like(t_logits, HUGE_NEG))
        return th.softmax(t_logits, dim=-1)

    def _calib_probe(self, policy_loss, distill_ce, diverge):
        """G-CAL 探针:g_pg/g_ce 范数只记账;diverge>20% 置旗(哨兵回调终止本腿)。"""
        params = [p for p in self.policy.parameters() if p.requires_grad]

        def gnorm(scalar):
            grads = th.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
            return float(th.sqrt(sum((g ** 2).sum() for g in grads if g is not None)))

        rec = {"step": int(self.num_timesteps),
               "g_pg": gnorm(policy_loss),
               "g_ce": gnorm(self.distill_beta * distill_ce) if distill_ce.requires_grad else 0.0,
               "distill_ce": float(distill_ce), "teacher_diverge": round(diverge, 4),
               "tripped": diverge > 0.20}
        if rec["tripped"] and not getattr(self, "calib_record_only", False):
            self._calib_tripped = True   # v28 record_only:只记不裁,旗不武装
        if self.calib_out:
            with open(self.calib_out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"   [G-CAL] {rec}")

    def train(self) -> None:
        # ===== sb3_contrib ppo_mask.py train() 诚实复写;“皮筋”段以 β>0 守卫 =====
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        distill_ces, diverges, t_confs = [], [], []
        calib_due = (self.distill_beta > 0 and self.calib_out is not None
                     and any(p not in self._calib_done and self.num_timesteps >= p
                             for p in self.calib_probes))

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

                # ===== 皮筋(v24 唯一插入段;β=0 整段跳过 → G-KL-B 逐位等价)=====
                if self.distill_beta > 0:
                    assert self.teacher is not None, "β>0 但教师未挂载(fail-loud 条款)"
                    with th.no_grad():
                        t_probs = self._teacher_probs(rollout_data.observations,
                                                      rollout_data.action_masks)
                    dist = self.policy.get_distribution(
                        rollout_data.observations, action_masks=rollout_data.action_masks)
                    logp_all = dist.distribution.logits   # 归一化 log-probs(掩位≈-1e8)
                    distill_ce = -(t_probs * logp_all).sum(dim=-1).mean()
                    loss = loss + self.distill_beta * distill_ce
                    distill_ces.append(distill_ce.item())
                    with th.no_grad():
                        dv = (logp_all.argmax(-1) != t_probs.argmax(-1)).float().mean().item()
                        diverges.append(dv)
                        t_confs.append(t_probs.max(dim=-1).values.mean().item())
                    if calib_due:
                        self._calib_probe(policy_loss, distill_ce, dv)
                        for p in self.calib_probes:
                            if self.num_timesteps >= p:
                                self._calib_done.add(p)
                        calib_due = False
                # ===== 皮筋段结束 =====

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break
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
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        # v24 皮筋读数(跨 minibatch 均值,预注册 D4:不许只记末批)
        self.logger.record("train/distill_beta", self.distill_beta)
        if distill_ces:
            self.logger.record("train/distill_ce", float(np.mean(distill_ces)))
            self.logger.record("train/teacher_diverge", float(np.mean(diverges)))
            self.logger.record("train/teacher_top1_conf", float(np.mean(t_confs)))
            self._last_distill_ce = float(np.mean(distill_ces))
            self._last_diverge = float(np.mean(diverges))
        else:
            self._last_distill_ce = None    # β=0 腿:哨兵行报 null,双簿对账干净
            self._last_diverge = None
