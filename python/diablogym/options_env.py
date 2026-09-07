"""v22 策略脑/操作脑:OptionsEnv——冻结宏之上的 SMDP 包装器。

设计稿:docs/DESIGN.md v22 章(评审团 wf_66e41e30 合成,Lawrence 批准)。
核心承诺:
  - 操作脑 = 神谕 oracle_mountain 的内环逐字移植并冻结(平稳 SMDP);
  - "榨干→下潜"不写死在脚本里,升格为策略脑的决策(本章唯一考题);
  - 换层必归还控制权(显式不变量:每层至少一次新决策,16 层扩展性的地基);
  - γ_mgr=1.0 由训练侧保证,选项内奖励不折现累加——策略脑优化的量
    逐字等于神谕账本(3000 步不折现回报);
  - 喝药是脑干反射(hp<0.5∧belt>0 → 12),刻意不是选项(v12 幽灵防复活)。

选项词表 Discrete(3):
  0 FARM     清剿本层:有怪交战,无怪捡药/捡装/探索。终止:升级/无进展
             140/本 scene 累计 FARM 1800/换层。
  1 DIVE     战斗推进下一主线目标:挡路者(≤6 格)打穿,否则剧情/下楼宏。
             终止:换场景/降层/停滞 140。
  2 RESUPPLY 潜前补给:连续捡药。掩码:无地面药或腰带满。终止:满/无/零进展。
通用终止:死亡/截断/TAU_CAP=600 微拍(命中率入 info,>5% 报警)。
反藏身处:通常 FARM 为保底；但剧情目标已出现且无近敌时强制交权 DIVE，
避免 action10 越权。榨干旗置位且 DIVE 合法时同样强制交权，切断旧经理
依赖 action10 暗中下楼所形成的无限干窗；仅 DIVE 非法时保留 FARM，并强制
≥25 拍复访地板。

v23(docs/PREREG-v23.md):窗口循环的逐拍簿记(保险丝/反射/终止阶梯)抽成
共享方法——OptionsEnv(组装/评测)与 WorkerWindowEnv(在位训练)跑同一段
代码,消灭"第三份实现"。支持 workers={选项: 策略} 把某选项的脚本内环换成
可学习工人:
  - 反射所有权上提:工人永不观测"反射待发"态,窗口开始与每个工人动作之后
    由包装器排水(逐拍过保险丝/时钟/终止阶梯)。脚本路径为恒等变换
    (dispatch 首分支即同款检查,提前判定不改变动作序列)。
  - 工人工资 w_t = r_t − 换层奖金(唯一剥除项;下潜套利修复,教训16)。
    账本恒等式:Σw ≡ 窗口R − DESCEND_UNIT×ΣΔdlvl⁺。经理账本一字不动。
  - 工人观测 298 维 = 基础 295 +
    [τ/TAU_CAP, 旧无新杀钟/140, 旧 exhausted]。后两维逐字保持冻结
    V28 的 protocol-v3 语义；新“无正进展钟”与 scene FARM 累计预算只
    负责当前收窗动力学，不偷换冻结网络的输入契约。第三维另以符号公开
    本窗饮药历史：未饮仍为旧 0/1，本窗任一来源（工人主动或脑干反射）
    已饮则编码为 -1/-2，绝不让隐藏闩制造同观测异标签。
  - 工人动作恒掩 11(下楼归经理 DIVE 职权)，剧情交权态也掩 10；12 由
    drink_sovereignty 控制，只在可见的 hp∈[0.5,0.75)、腰带有药且本窗
    尚未发生任何饮药时合法。脑干反射本身不受此闩限制，紧急态仍可连续
    排水；闩只防工人在已经获救后又浪费一瓶。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from . import bridge
from .controller_wire import *  # noqa: F403 - re-export canonical wire schema
from .env import (
    DESCEND_UNIT,
    REWARD_ECONOMY_V1,
    DiabloGymEnv,
    _scene_identity,
    gear_combat_utility_value,
)

FARM, DIVE, RESUPPLY = 0, 1, 2
KILL_PATIENCE = 140   # 微拍:FARM 无正进展 / DIVE 无换层的停滞上限
TAU_CAP = 600         # 选项最长占用(straggler 税封顶)
REVISIT_FLOOR = 25    # 榨干旗下复选 FARM 的最小占用(堵秒终止搅拌键)
RESUPPLY_CAP = 60
# 本场景 FARM 的累计占用硬预算。短钟会被新格/伤害等真实进展清零，不能
# 同时承担“何时把控制权交回 DIVE”的职责；否则持续探索可把整个 3000 拍
# 局都吃完。1800 给出生层约六成总预算，固定种子定标仍可清 40--90 只怪，
# 同时为发现/执行 DIVE 留出至少 1200 拍。
FARM_SCENE_CAP = 1800
BLOCKER_RADIUS = 6    # 剧情交权与 DIVE 清路共用；禁止两套阈值留下 4..6 空档
GEAR_GRACE_MAX_DECISIONS = 3
# 主动喝药是“有限主权”而不是任意掉血即可挥霍药水。脑干继续独占 <0.5
# 的紧急排水；工人只在 [0.5, 0.75) 的安全包络内拥有选择权。上界刻意高于
# BC-v2 教师的主档 0.65 与唯一 OC 0.70：mask 只封明显浪费态，并保留
# [teacher_threshold, 0.75) 的真实合法 hard negatives 让策略学习边界，
# 不把教师的目标策略直接硬编码成动作合法性。
VOLUNTARY_DRINK_HP_LOW = 0.50
VOLUNTARY_DRINK_HP_HIGH = 0.75
N_EXTRA_WORKER = 3    # 工人观测追加维:τ 钟 / 旧无新杀钟 / 旧 exhausted
# 295 基础维之后的第三个 worker extra。非负值逐字保留旧 exhausted；
# ≤-1 表示本窗已发生至少一次饮药（主动或反射），绝不改变旧零饮策略
# 的任何输入；``-v-1`` 可无损还原旧 0/1。
WORKER_DRINK_LATCH_FEATURE = 297
WORKER_OBSERVATION_VIEW_RAW_V4 = "raw-v4"
WORKER_OBSERVATION_VIEW_LEGACY_V3 = "legacy-v3"
WORKER_OBSERVATION_VIEW_A12_OVERLAY = "legacy-v3-a12-overlay"
WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC = "dual-v4-asymmetric-v3"
# R13 教室改革:v5 = 完整 v4 向量逐位前缀 + 12 维窗口模式追加块。
# 旧 v4 布局与 sha 一字不动;v5 仅由 R13 预注册配方点名,默认路径不可达。
WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE = "dual-v5-window-mode-v1"
DUAL_WORKER_V5_APPENDIX_DIM = 12
WORKER_OBSERVATION_VIEWS = frozenset({
    WORKER_OBSERVATION_VIEW_RAW_V4,
    WORKER_OBSERVATION_VIEW_LEGACY_V3,
    WORKER_OBSERVATION_VIEW_A12_OVERLAY,
    WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
    WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE,
})
WORKER_ACTION12_PERMANENTLY_MASKED = "permanently-masked"
WORKER_ACTION12_ENVIRONMENT_MASK = "environment-mask"
WORKER_ACTION12_MODES = frozenset({
    WORKER_ACTION12_PERMANENTLY_MASKED,
    WORKER_ACTION12_ENVIRONMENT_MASK,
})
# ``dual-v4-asymmetric-v3`` layout, segment metadata and canonical hash live
# in controller_wire.py.  Runtime/wrapper/training code all import that frozen
# source rather than copying absolute indices.
MANAGER_OBSERVATION_VIEW_RAW_V4 = "raw-v4"
MANAGER_OBSERVATION_VIEW_LEGACY_V3 = "legacy-v3"
MANAGER_OBSERVATION_VIEWS = frozenset({
    MANAGER_OBSERVATION_VIEW_RAW_V4,
    MANAGER_OBSERVATION_VIEW_LEGACY_V3,
})


def _resolve_worker_drink_sovereignty(
        workers: dict,
        requested: bool | None) -> bool:
    """Bind the environment-wide action-12 semantics to tagged callbacks."""
    modes = []
    untagged = []
    for option, worker in workers.items():
        mode = getattr(worker, "diablogym_worker_action12_mode", None)
        if mode is None:
            untagged.append(option)
            continue
        if mode not in WORKER_ACTION12_MODES:
            raise ValueError(
                "Worker callback action12 mode 未注册:"
                f"option={option!r},mode={mode!r}")
        modes.append(mode)
    if modes and untagged:
        raise ValueError(
            "带 action12 contract 的 Worker 不得与无 contract callback "
            f"混用:untagged={untagged!r}")
    unique_modes = set(modes)
    if len(unique_modes) > 1:
        raise ValueError(
            "OptionsEnv 的全局 drink_sovereignty 无法表达多个 Worker "
            f"action12 mode:{sorted(unique_modes)!r}")
    derived = None
    if unique_modes:
        derived = (
            next(iter(unique_modes))
            == WORKER_ACTION12_ENVIRONMENT_MASK
        )
    if requested is None:
        return True if derived is None else derived
    configured = bool(requested)
    if derived is not None and configured != derived:
        raise ValueError(
            "OptionsEnv drink_sovereignty 与 Worker action12 contract "
            f"不一致:configured={configured},contract={next(iter(unique_modes))!r}")
    return configured


@dataclass(frozen=True)
class BeatOutcome:
    """一拍的可审计结果。

    fuse 命中时不再把请求动作静默替换成探索动作 10，而是拒绝本拍并让
    当前选项以 ``reason="fuse"`` 收窗。这样 rollout 中记录的请求动作与
    环境实际执行动作不会错标：``executed_action=None`` 明确表示未执行。
    """

    reason: str | None
    requested_action: int
    executed_action: int | None
    fuse_tripped: bool
    action14_audit: dict | None = None
    action_effect_audit: dict | None = None


@dataclass(frozen=True)
class WorkerStepOutcome:
    """工人提案及其反射尾部的合并审计结果。"""

    reason: str | None
    requested_action: int
    executed_action: int | None
    fuse_tripped: bool
    fuse_requested_action: int | None
    action14_audit: dict | None = None
    action_effect_audit: dict | None = None


_ACTION14_AUDIT_KEYS = frozenset({
    "accepted", "commit_attempts", "utility_before",
    "utility_after", "utility_delta",
})
_ACTION_EFFECT_AUDIT_KEYS = frozenset({
    "requested_action", "native_attempts", "native_accepts",
    "request_executed", "material_effect", "effect_reasons",
    "same_scene", "stall_cost_applied",
})


def _validated_action14_audit(info: dict) -> dict:
    """Validate the synchronous native gear-commit receipt from the base env."""
    audit = info.get("action14_audit")
    if not isinstance(audit, dict) or set(audit) != _ACTION14_AUDIT_KEYS:
        raise RuntimeError(
            "action14 缺失/损坏原生因果回执")
    accepted = audit["accepted"]
    integers = {
        key: audit[key]
        for key in (
            "commit_attempts", "utility_before",
            "utility_after", "utility_delta",
        )
    }
    if (
        not isinstance(accepted, bool)
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integers.values()
        )
        or integers["commit_attempts"] < 0
        or not 0 <= integers["utility_before"] <= 0xFFFFFFFF
        or not 0 <= integers["utility_after"] <= 0xFFFFFFFF
        or integers["utility_delta"] < 0
        or integers["utility_after"] - integers["utility_before"]
        != integers["utility_delta"]
        or accepted
        != (
            integers["commit_attempts"] >= 1
            and integers["utility_delta"] > 0
        )
    ):
        raise RuntimeError(
            "action14 原生因果回执字段不守恒")
    return {
        "accepted": accepted,
        **integers,
    }


def _validated_action_effect_audit(
        info: dict, requested_action: int) -> dict:
    """Validate the base environment's causal effect/stall receipt."""
    audit = info.get("action_effect_audit")
    resource = info.get("resource_action_audit")
    is_resource = isinstance(resource, dict) and resource.get("source") == "resupply-script"
    if is_resource and (requested_action != 0
            or not isinstance(resource.get("accepted"), bool)
            or type(resource.get("micro_steps")) is not int
            or resource["micro_steps"] < 0):
        raise RuntimeError("Invalid script-owned resource receipt")
    if (
        not isinstance(audit, dict)
        or set(audit) != _ACTION_EFFECT_AUDIT_KEYS
        or (
            isinstance(audit.get("requested_action"), bool)
            or not isinstance(audit.get("requested_action"), int)
        )
        or int(audit["requested_action"]) != int(requested_action)
        or isinstance(audit.get("native_attempts"), bool)
        or not isinstance(audit.get("native_attempts"), int)
        or isinstance(audit.get("native_accepts"), bool)
        or not isinstance(audit.get("native_accepts"), int)
        or not 0 <= audit["native_accepts"] <= audit["native_attempts"]
        or not isinstance(audit.get("request_executed"), bool)
        or not isinstance(audit.get("material_effect"), bool)
        or not isinstance(audit.get("same_scene"), bool)
        or not isinstance(audit.get("stall_cost_applied"), bool)
        or not isinstance(audit.get("effect_reasons"), tuple)
        or any(
            not isinstance(reason, str) or not reason
            for reason in audit["effect_reasons"]
        )
        or len(set(audit["effect_reasons"]))
        != len(audit["effect_reasons"])
        or bool(audit["effect_reasons"]) != audit["material_effect"]
        or audit["request_executed"]
        != (
            bool(resource["accepted"]) if is_resource else
            (int(requested_action) == 0 or audit["native_accepts"] > 0)
        )
        or audit["stall_cost_applied"]
        != (
            (audit["same_scene"] and not audit["request_executed"]
             and resource["micro_steps"] > 0) if is_resource else
            (audit["same_scene"] and (int(requested_action) == 0
                or not audit["request_executed"]))
        )
    ):
        raise RuntimeError("基础动作缺失/损坏因果效果回执")
    return {
        "requested_action": int(audit["requested_action"]),
        "native_attempts": int(audit["native_attempts"]),
        "native_accepts": int(audit["native_accepts"]),
        "request_executed": bool(audit["request_executed"]),
        "material_effect": bool(audit["material_effect"]),
        "effect_reasons": tuple(audit["effect_reasons"]),
        "same_scene": bool(audit["same_scene"]),
        "stall_cost_applied": bool(audit["stall_cost_applied"]),
    }


def _floor_heals(raw) -> bool:
    # 与基础 env 的动作目标/掩码共用唯一可见且可达口径；helper 对缺少
    # visible/reachable 的旧 fixture 向后兼容，避免另造第二套筛选真源。
    return bool(DiabloGymEnv._policy_floor_items(raw, "heal"))


def _reflex(raw) -> bool:
    """脑干反射条件(与 dispatch 首分支逐字同款)。"""
    return DiabloGymEnv._reflex_eligible(raw)


_DISTANCE_UNSET = object()


def dispatch(
    mode: str,
    raw,
    gear_available: bool,
    action_mask=None,
    nearest_engageable_distance=_DISTANCE_UNSET,
) -> int:
    """神谕内环逐字移植(纯函数,冻结)。mode ∈ {farm, dive, resupply}。
    注意:神谕农期的 stagnant>=140→11 子分支被刻意剔除——归策略脑管。"""
    hp = raw["hp"] / max(1, raw["max_hp"])
    belt = raw.get("belt_heals", 0)
    free_slots = DiabloGymEnv._belt_free_slots(raw)
    has_exact_action_mask = action_mask is not None
    if action_mask is not None:
        action_mask = np.asarray(action_mask, dtype=bool)
        if action_mask.shape != (15,):
            raise ValueError(
                "dispatch action_mask 必须是 (15,),收到 "
                f"{action_mask.shape}")
        monster_available = bool(action_mask[9])
        heal_available = bool(action_mask[13])
        gear_available = bool(gear_available) and bool(action_mask[14])
    else:
        monster_available = bool(
            DiabloGymEnv._policy_monsters(raw))
        heal_available = free_slots > 0 and _floor_heals(raw)
    if nearest_engageable_distance is _DISTANCE_UNSET:
        near = _nearest(raw)
    else:
        near = nearest_engageable_distance
        if near is not None:
            if isinstance(near, bool) or not isinstance(
                    near, (int, float, np.integer, np.floating)):
                raise TypeError(
                    "nearest_engageable_distance 必须是非负数或 None")
            near = float(near)
            if not np.isfinite(near) or near < 0:
                raise ValueError(
                    "nearest_engageable_distance 必须是有限非负数")
    if hp < 0.5 and belt > 0:          # 脑干反射,嵌在一切模式里
        return 12
    if mode == "resupply":
        # OptionsEnv 正常只会在 RESUPPLY 掩码为真时进入此分支；额外守卫
        # 让纯函数调用也绝不向“药少但被非药物塞满”的腰带发非法拾药。
        return 13 if heal_available else 0
    if mode == "dive":
        if belt <= 2 and heal_available:
            return 13
        if (
            monster_available
            and near is not None
            and near <= BLOCKER_RADIUS
        ):
            return 9
        # action 11 may cross a one-way scene boundary.  Drain a currently
        # executable upgrade first so loot from the cleared blocker is not
        # abandoned with the old scene.  Callers without the exact controller
        # mask have no atomic reachability proof and retain fail-closed a11.
        if has_exact_action_mask and gear_available:
            return 14
        return 11
    # farm
    if raw.get("progression_targets") and (near is None or near > 6):
        # Flat 神谕没有 manager，必须立即执行 DIVE；旧 action10 会在
        # progression 态确定性 wait，连续制造约 KILL_PATIENCE 个空标签。
        # 层级 Options 会在到达这里前由 _farm_handoff 收窗。若 Worker
        # 掩掉了 11，fail-closed 等待交权而不生成越权动作。
        if action_mask is not None and not bool(action_mask[11]):
            # A just-cleared blocker may expose both the one-way progression
            # target and a strict gear upgrade.  During the single Worker
            # grace decision the mask is exactly {0,14}; teach/execute 14
            # instead of turning that valuable state into an action0 label.
            return 14 if gear_available else 0
        return 11
    if monster_available:
        return 9
    if heal_available:
        return 13
    if gear_available:
        return 14
    return 10


def _nearest(raw):
    px, py = raw["player_x"], raw["player_y"]
    ds = [max(abs(m["x"] - px), abs(m["y"] - py))
          for m in DiabloGymEnv._policy_monsters(raw)]
    return min(ds) if ds else None


def _farm_handoff(
    raw,
    nearest_engageable_distance=_DISTANCE_UNSET,
) -> bool:
    """剧情目标已出现且无近敌时，FARM 必须把职权交还给经理 DIVE。"""
    near = (
        _nearest(raw)
        if nearest_engageable_distance is _DISTANCE_UNSET
        else nearest_engageable_distance
    )
    return (bool(raw.get("progression_targets"))
            and (near is None or near > BLOCKER_RADIUS))


class OptionsEnv(gym.Env):
    """step(option) 把选项跑到终止,返回(303 维观测, 不折现累计奖励, ...)。"""

    N_EXTRA_MGR = 8  # time_remaining/旧无新杀钟/本层杀/旧本层耗时/上选项one-hot(3)/上选项τ

    def __init__(self, max_steps: int = 3000, workers: dict | None = None,
                 drink_sovereignty: bool | None = None,
                 dive_stall_protocol: str = "no-progress-v1",
                 worker_observation_view: str = WORKER_OBSERVATION_VIEW_RAW_V4,
                 manager_observation_view: str = (
                     MANAGER_OBSERVATION_VIEW_RAW_V4),
                 dive_live_sovereignty: bool = False,
                 worker_descend_bonus_fraction: float = 0.0,
                 farm_scene_cap: int = FARM_SCENE_CAP,
                 reset_layer_clock_on_window: bool = False,
                 resource_protocol: str = "off",
                 resource_purchase_mode: str = "full",
                 resource_service_policy: str = "legacy-v1",
                 resource_calibration=None,
                 resource_readiness_law: str = "veto-v1",
                 worker_time_protocol: str = "legacy",
                 resource_retreat: str = "off",
                 resource_portal: str = "off",
                 **env_kwargs):
        super().__init__()
        from .resource_protocol import (
            validate_resource_config, validate_resource_calibration, RESOURCE_FARM_CAP,
            validate_readiness_law)
        self.resource_protocol, self.resource_purchase_mode = validate_resource_config(
            resource_protocol, resource_purchase_mode)
        from .resource_sustain import validate_service_policy
        self.resource_service_policy = validate_service_policy(
            self.resource_protocol, self.resource_purchase_mode, resource_service_policy)
        self.resource_calibration = validate_resource_calibration(
            self.resource_protocol, resource_calibration)
        # R17.1 ruling 3: readiness law; default veto-v1 keeps env kwargs byte-identical.
        self.resource_readiness_law = validate_readiness_law(
            self.resource_protocol, resource_readiness_law)
        if self.resource_readiness_law != "veto-v1":
            env_kwargs["resource_readiness_law"] = self.resource_readiness_law
        # R18-A retreat-v1: return-to-town interface (default off = byte-identical kwargs).
        from .resource_protocol import validate_retreat_protocol
        self.resource_retreat = validate_retreat_protocol(
            self.resource_protocol, self.resource_readiness_law, resource_retreat)
        if self.resource_retreat != "off":
            env_kwargs["resource_retreat"] = self.resource_retreat
        # R18-F portal-v1 (default off = byte-identical kwargs).
        from .resource_protocol import validate_portal_protocol
        self.resource_portal = validate_portal_protocol(
            self.resource_protocol, self.resource_readiness_law,
            self.resource_retreat, resource_portal)
        if self.resource_portal != "off":
            env_kwargs["resource_portal"] = self.resource_portal
        from .completion_clock import COMPLETION_PROTOCOLS, COMPLETION_RECIPES
        if worker_time_protocol not in ("legacy", *COMPLETION_PROTOCOLS):
            raise ValueError("Unknown worker_time_protocol")
        self.worker_time_protocol = worker_time_protocol
        if worker_time_protocol in COMPLETION_PROTOCOLS:
            recipe = COMPLETION_RECIPES[worker_time_protocol]
            if (type(max_steps) is not int
                    or max_steps != recipe.actor_denominator
                    or self.resource_protocol != "l2-town-v1"
                    or self.resource_purchase_mode != "full"
                    or self.resource_service_policy not in ("sustain-v6", "sustain-loot-v1")
                    or self.resource_calibration is not None):
                raise ValueError("completion-l2-v1 requires legacy observation denominator and l2-town-v1/full with sustain-v6 or sustain-loot-v1 without calibration")
        if self.resource_service_policy == "sustain-loot-v1" and self.worker_time_protocol not in COMPLETION_PROTOCOLS:
            raise ValueError("sustain-loot-v1 requires an explicit completion-l2 time protocol")
        env_kwargs["resource_protocol"] = self.resource_protocol
        env_kwargs["resource_purchase_mode"] = self.resource_purchase_mode
        ordinary_scope = self.resource_service_policy in ("sustain-v5", "sustain-v6", "sustain-loot-v1")
        requested_scope = env_kwargs.pop("resource_ordinary_armor_scope", ordinary_scope)
        if requested_scope is not ordinary_scope:
            raise ValueError("resource_ordinary_armor_scope must match resource_service_policy")
        if ordinary_scope:
            env_kwargs["resource_ordinary_armor_scope"] = True
        preserve_equipment = self.resource_service_policy in ("sustain-v6", "sustain-loot-v1")
        requested_preservation = env_kwargs.pop(
            "resource_preserve_equipment_readiness", preserve_equipment)
        if requested_preservation is not preserve_equipment:
            raise ValueError("resource_preserve_equipment_readiness must match resource_service_policy")
        if preserve_equipment:
            env_kwargs["resource_preserve_equipment_readiness"] = True
        loot_economy = self.resource_service_policy == "sustain-loot-v1"
        requested_loot = env_kwargs.pop("resource_loot_economy", loot_economy)
        if requested_loot is not loot_economy:
            raise ValueError("resource_loot_economy must match resource_service_policy")
        if loot_economy:
            env_kwargs["resource_loot_economy"] = True
        if self.resource_protocol != "off":
            farm_scene_cap = (RESOURCE_FARM_CAP if self.resource_calibration is None
                              else self.resource_calibration.farm_scene_microstep_cap)
        self._workers = workers or {}
        # R16 修宪(审计簇 C6/C3):出生层场景预算改为可配置(默认 1800 逐位
        # 旧法);FARM 开窗时可选清零无进展钟——argmax 体制下 DIVE stall 后
        # layer_clock 被继承,下一 FARM 窗一个决策即再榨干(时钟继承)。
        _cap = int(farm_scene_cap)
        if _cap <= 0:
            raise ValueError(
                f"farm_scene_cap 必须为正整数,收到 {farm_scene_cap!r}")
        self.farm_scene_cap = _cap
        self.reset_layer_clock_on_window = bool(reset_layer_clock_on_window)
        # R13 主权移交(默认 False = 旧法逐位不变):仅在 live DIVE 窗内
        # 解禁 a11 与 1-8 踏 trigger 格;a10 剧情掩码与 FARM/RESUPPLY 窗
        # 掩码在任何取值下一字不动。
        self.dive_live_sovereignty = bool(dive_live_sovereignty)
        # R14 乙案:live DIVE 窗内工人保留的下楼奖金比例(默认 0 = 全额
        # 剥薪旧法);仅与主权移交同时生效,记账见 _win_beat。
        _f = float(worker_descend_bonus_fraction)
        if not np.isfinite(_f) or not 0.0 <= _f <= 1.0:
            raise ValueError(
                "worker_descend_bonus_fraction 必须在 [0,1] 内,"
                f"收到 {worker_descend_bonus_fraction!r}")
        if _f > 0.0 and not self.dive_live_sovereignty:
            raise ValueError(
                "worker_descend_bonus_fraction 仅与 dive_live_sovereignty "
                "同时生效(非 live-DIVE 法域无下楼可酬)")
        self.worker_descend_bonus_fraction = _f
        self.descend_bonus_kept_total = 0.0
        # v32 喝药主权(④丙):默认 True = 新协议常态,工人可主动按 12;
        # 0.5 反射(_drain/dispatch 内嵌)一字不动,永为兜底。False 系
        # 对照腿/旧协议复现专用旋钮。带部署合约的 Worker 可在 None
        # 默认下成为单一真源；若调用方又显式给值，两者必须一致。
        self.drink_sovereignty = _resolve_worker_drink_sovereignty(
            self._workers, drink_sovereignty)
        # E-fix 修 B(甲形态):DIVE 停滞钟协议。"no-progress-v1" = 新协议
        # 常态:唯「历史最小目标距严格改善 ∨ 击杀 ∨ positive_progress
        # 全集」给 DIVE 窗续命;KILL_PATIENCE 常数(冻结观测归一化分母)
        # 一字不动,TAU_CAP=600 仍为硬顶。裸位移不算进展(极限环每拍都在
        # 位移,裸位移判据会把死窗烧到 TAU_CAP)。"tau-v3" = 旧协议纯耗时
        # 收窗,系对照腿/旧档案位级重放专用端点。
        if dive_stall_protocol not in ("tau-v3", "no-progress-v1"):
            raise ValueError(
                "dive_stall_protocol 必须是 'tau-v3' 或 'no-progress-v1'，"
                f"收到 {dive_stall_protocol!r}")
        self.dive_stall_protocol = dive_stall_protocol
        if worker_observation_view not in WORKER_OBSERVATION_VIEWS:
            raise ValueError(
                "worker_observation_view 必须是 "
                f"{sorted(WORKER_OBSERVATION_VIEWS)} 之一，收到 "
                f"{worker_observation_view!r}")
        self.worker_observation_view = worker_observation_view
        if manager_observation_view not in MANAGER_OBSERVATION_VIEWS:
            raise ValueError(
                "manager_observation_view 必须是 "
                f"{sorted(MANAGER_OBSERVATION_VIEWS)} 之一，收到 "
                f"{manager_observation_view!r}")
        self.manager_observation_view = manager_observation_view
        env_kwargs.setdefault("descend_ladder", True)
        env_kwargs.setdefault("death_ladder", True)
        env_kwargs.setdefault("start_in_dungeon", True)
        env_kwargs.setdefault("include_raw", False)
        if not bool(env_kwargs["descend_ladder"]):
            raise ValueError(
                "OptionsEnv 要求 descend_ladder=True；Worker 工资会从基础"
                "奖励中扣除同一深度奖金，关闭它会凭空制造负工资")
        required_controller_snapshot = (
            worker_observation_view in (
                WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
                WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE,
            )
        )
        requested_controller_snapshot = env_kwargs.setdefault(
            "controller_snapshot_enabled", required_controller_snapshot)
        if bool(requested_controller_snapshot) != required_controller_snapshot:
            raise ValueError(
                "controller_snapshot_enabled 必须与 worker_observation_view "
                "一致；只有 dual 家族视图可启用固定 controller wire")
        self.env = DiabloGymEnv(max_steps=max_steps, **env_kwargs)
        if self.resource_calibration is not None:
            self.env._resource_calibration = self.resource_calibration
        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(3)
        base = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(base + self.N_EXTRA_MGR,), dtype=np.float32)
        self._last_base_obs = None
        self._win = None
        self._reset_wrapper_state()

    # ---- wrapper 状态(跨选项持续)----
    def _reset_wrapper_state(self):
        from .resource_protocol import ResourceService
        service_type = ResourceService
        if getattr(self, "resource_service_policy", "legacy-v1") == "sustain-v2":
            from .resource_sustain import SustainResourceService
            service_type = SustainResourceService
        elif getattr(self, "resource_service_policy", "legacy-v1") == "sustain-v3":
            from .resource_sustain_gold import SustainGoldMemoryService
            service_type = SustainGoldMemoryService
        elif getattr(self, "resource_service_policy", "legacy-v1") == "sustain-v4":
            from .resource_sustain_gold import SustainGoldExtendedService
            service_type = SustainGoldExtendedService
        elif getattr(self, "resource_service_policy", "legacy-v1") == "sustain-v5":
            from .resource_sustain_armor import SustainOrdinaryArmorService
            service_type = SustainOrdinaryArmorService
        elif getattr(self, "resource_service_policy", "legacy-v1") == "sustain-v6":
            from .resource_sustain_armor import SustainEquipmentReadinessService
            service_type = SustainEquipmentReadinessService
        if getattr(self, "worker_time_protocol", "legacy").startswith("completion-l2"):
            from .resource_sustain_completion import SustainCompletionService
            from .completion_clock import CompletionClock, COMPLETION_RECIPES
            service_type = SustainCompletionService
            self._completion_clock = CompletionClock(COMPLETION_RECIPES[self.worker_time_protocol])
            self.env.max_steps = self._completion_clock.state.physical_deadline
            self.env._completion_prefix_active = False
            self.env._completion_time_failure = None
            self._completion_previous_scene = self._completion_scene(
                getattr(self.env, "_raw", None))
            self.env.resource_time_callback = self._completion_time_tick
            self.env.resource_time_info_callback = self.completion_time_telemetry
        if getattr(self, "resource_service_policy", "legacy-v1") == "sustain-loot-v1":
            from .resource_sustain_loot import SustainLootService
            service_type = SustainLootService
        self.resource_service = service_type(
            getattr(self, "resource_purchase_mode", "full"),
            calibration=getattr(self, "resource_calibration", None))
        if getattr(self, "resource_service_policy", "legacy-v1") == "sustain-loot-v1":
            service = self.resource_service
            self.env.resource_observation_callback = lambda inner, raw, steps: service.observe(raw, steps)
            if getattr(self.env, "_raw", None) is not None:
                service.observe(self.env._raw, int(self.env._steps))
        elif getattr(self, "resource_service_policy", "legacy-v1") in ("sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6"):
            memory = self.resource_service.gold_memory
            self.env.resource_observation_callback = lambda inner, raw, steps: memory.observe(raw, steps)
            if getattr(self.env, "_raw", None) is not None:
                memory.observe(self.env._raw, int(self.env._steps))
        self._resource_farm_ledgers = {}
        self._resource_command = None
        # R18-A retreat-v1: a fresh scripted retreat service per episode; None when off.
        if getattr(self, "resource_retreat", "off") != "off":
            from .resource_retreat import RetreatService
            self.retreat_service = RetreatService()
        else:
            self.retreat_service = None
        # R18-F portal-v1: a fresh scripted portal service per episode; None when off.
        if getattr(self, "resource_portal", "off") != "off":
            from .resource_portal import PortalService
            self.portal_service = PortalService()
        else:
            self.portal_service = None
        # 冻结的 V28 worker 与 M29 manager 都在 db7d26c 的 protocol-v3
        # 状态上训练。新协议可以改变收窗动力学，却不能把同宽度列静默换义；
        # 因此旧无新杀钟/榨干旗/本层起点独立维护，只供冻结网络观测。
        self._legacy_layer_clock = 0
        self._legacy_exhausted = False
        self._legacy_layer_steps0 = 0
        self._layer_steps0 = 0
        self.layer_clock = 0          # 本层无正进展微拍数(战斗/探索/物资/换层清零)
        self.exhausted = False        # 榨干旗(任一可验证正进展/换层清除)
        self.farm_scene_steps = 0     # 当前 scene 累计 FARM 微拍(正进展不清零)
        raw = getattr(self.env, "_raw", None)
        self._farm_scene = _scene_identity(raw) if raw is not None else None
        self._fuse_sig = None         # B4 保险丝签名(跨选项边界持续)
        self._fuse = 0
        self._fuse_recovery_pending = False
        self._layer_kills0 = 0
        self._last_opt = -1
        self._last_tau = 0
        self._cap_hits = 0
        self._decisions = 0
        self.mode_seq = []
        self._win = None

    @staticmethod
    def _completion_scene(raw):
        if raw is None:
            return None
        return {key: raw[key] for key in ("engine_level", "is_set_level")}

    def completion_time_telemetry(self):
        """Detached clock facts only; no native query, reset, reward or outcome."""
        if getattr(self, "worker_time_protocol", "legacy") == "legacy":
            return None
        from dataclasses import asdict
        clock = self._completion_clock
        return {"protocol": self.worker_time_protocol,
                "recipe": clock.recipe.as_dict(),
                "clock": asdict(clock.state),
                "effective_physical_limit": int(self.env.max_steps),
                "observation_denominator": int(self.max_steps),
                "prefix_budget_active": bool(getattr(self.env, "_completion_prefix_active", False))}

    def _completion_time_tick(self, inner, raw, micro_step):
        clock = self._completion_clock
        target = self._completion_scene(raw)
        prefix_active = bool(getattr(inner, "_completion_prefix_active", False))
        expected = (getattr(inner, "_completion_prefix_deadline", None) if prefix_active
                    else clock.state.physical_deadline)
        if expected is None or int(inner.max_steps) != expected:
            raise RuntimeError("completion physical limit changed outside its owner")
        entering = (clock.state.first_arrival_micro_step is None
                    and target["engine_level"] == 2 and not target["is_set_level"])
        if entering:
            if prefix_active:
                # The engine already completed this tick; it is an engineering
                # stop, not a settled/credited prefix or permission to extend it.
                inner._completion_time_failure = {
                    "reason": "L2_before_learner_handoff",
                    "actual_native_microstep": int(micro_step),
                    "source_scene": self._completion_previous_scene,
                    "observed_scene": target,
                    "window_incomplete": True,
                    "physical_deadline_unchanged": int(inner.max_steps),
                }
                raise RuntimeError("completion L2 arrival before learner handoff")
            transition = raw.get("resource_state", {}).get("transition", {})
            # R17.1 ruling 3: under coach-v03 a forced-unready arrival is legal
            # (accepted, pretransition_ready False, accounted, no escrow); only
            # the receipt's existence/acceptance and the 1->2 geometry are law.
            ready_required = getattr(self, "resource_readiness_law", "veto-v1") != "coach-v03"
            if (not transition.get("accepted")
                    or (ready_required and not transition.get("pretransition_ready"))
                    or transition.get("source_depth") != 1 or transition.get("target_depth") != 2
                    or transition.get("source_is_set") or transition.get("target_is_set")):
                raise RuntimeError("completion L2 arrival lacks the live native readiness receipt")
        clock.observe_native_tick(micro_step, self._completion_previous_scene, target)
        self._completion_previous_scene = target
        if entering:
            inner.max_steps = clock.state.physical_deadline

    def _mark_exhausted(self) -> None:
        """原子发布当前协议的榨干态。

        ``layer_clock``/``exhausted`` 只负责当前收窗与掩码；冻结网络读取
        独立的 protocol-v3 状态，避免 scene cap 或正向进度把旧输入换义。
        """
        self.exhausted = True
        self.layer_clock = max(int(self.layer_clock), KILL_PATIENCE)

    def _clear_exhausted(self) -> None:
        """清除交权态；调用方必须已经确认 scene/预算允许继续 FARM。"""
        self.exhausted = False

    def _sig(self, a, raw):
        # fuse 只能识别“请求动作确实没有正进展”，不能把站桩输出误判成
        # 卡死。玩家 hp/mana 刻意不入签名：敌方持续打人不是请求动作的
        # 正进展，否则真正卡墙会靠掉血不断续命直到死亡。摘要全部排序，
        # 禁止 bridge 列表枚举顺序抖动重置/触发保险丝。
        monsters = tuple(sorted(
            (
             DiabloGymEnv._monster_generation_key(m),
             int(m.get("x", 0)), int(m.get("y", 0)),
             int(m.get("hp", 0)), int(m.get("max_hp", 0)))
            for m in DiabloGymEnv._policy_monsters(raw)
        ))
        def item_word(item, field, bits, *, default=None):
            return DiabloGymEnv._controller_uint(
                item.get(field, default),
                name=f"fuse floor_item.{field}",
                bits=bits,
            )

        floor_items = tuple(sorted(
            (
             item_word(it, "active_id", 7),
             int(it.get("x", 0)), int(it.get("y", 0)),
             bool(it.get("heal")), bool(it.get("gear")),
             item_word(it, "combat_utility_hi", 16),
             item_word(it, "combat_utility_lo", 16),
             item_word(it, "seed_hi", 16, default=0),
             item_word(it, "seed_lo", 16, default=0),
             item_word(it, "create_info", 16, default=0),
             item_word(it, "base_id", 16, default=0))
            for it in raw.get("floor_items", ())
            if bool(it.get("visible", True)) and bool(it.get("reachable", True))
        ))
        progression = tuple(sorted(
            (str(p.get("kind", "")), str(p.get("action", "")),
             int(p.get("x", 0)), int(p.get("y", 0)),
             int(p.get("goal_x", 0)), int(p.get("goal_y", 0)),
             bool(p.get("exact")))
            for p in raw.get("progression_targets", ())
        ))
        combat_floor = tuple(sorted(
            (
                (
                    tuple(int(part) for part in key)
                    if isinstance(key, tuple) and len(key) == 3
                    else (int(key), 0, 0)
                ),
                int(value[0]),
                int(value[1]),
            )
            for key, value in getattr(
                self.env, "_combat_hp_floor", {}
            ).items()
        ))
        return (a, raw["player_x"], raw["player_y"], raw.get("belt_heals", 0),
                raw.get("future_x", raw["player_x"]),
                raw.get("future_y", raw["player_y"]),
                DiabloGymEnv._belt_free_slots(raw),
                raw.get("xp", 0), raw.get("gold", 0),
                gear_combat_utility_value(raw, "fuse_signature"),
                raw["char_level"],
                self.env._ep_kills, _scene_identity(raw),
                int(getattr(self.env, "exploration_progress", 0)),
                combat_floor,
                monsters, floor_items, progression)

    def _tick_layer_clock(
        self,
        kills_before,
        scene_before,
        steps_delta,
        *,
        positive_progress: bool = False,
    ):
        """同步推进旧观测钟与当前 FARM“无正进展”钟。

        kill 只是进展的一种。action10 新踏足/开普通软墙、怪物最低血线
        继续下降、成功取得腰带补给或穿上装备同样证明当前层尚未榨干；
        若仍只认 kill，12 个探索宏左右就会在边疆尚存时误报 exhausted。
        自己掉血、怪物回血后重打一段已付血线、重复走旧格都不清钟。
        """
        raw = self.env._raw
        scene_changed = _scene_identity(raw) != scene_before

        # protocol-v3 的逐拍语义：仅换 scene 或新杀清钟；探索、伤害、
        # 拾取、穿装等后来新增的 positive_progress 不得影响冻结输入。
        if scene_changed:
            self._legacy_layer_clock = 0
            self._legacy_exhausted = False
            self._legacy_layer_steps0 = self.env._steps
        elif self.env._ep_kills > kills_before:
            self._legacy_layer_clock = 0
            self._legacy_exhausted = False
        else:
            self._legacy_layer_clock += steps_delta

        if scene_changed:
            self._sync_farm_scene(_scene_identity(raw))
            self.layer_clock = 0
            self._clear_exhausted()
            self._layer_kills0 = self.env._ep_kills
            self._layer_steps0 = self.env._steps
        elif self.env._ep_kills > kills_before or positive_progress:
            # 累计预算是独立的交权闸，不能被一块新地板或下一刀重新打开。
            if (getattr(self, "farm_scene_steps", 0)
                    < getattr(self, "farm_scene_cap", FARM_SCENE_CAP)):
                self.layer_clock = 0
                self._clear_exhausted()
            else:
                self._mark_exhausted()
        else:
            self.layer_clock += steps_delta
            # cap 可能恰在一个没有正进展的宏内命中；在 _win_term 之前就
            # 原子发布，保证任何调试/worker 观测都不会看见“钟未满但旗已满”
            # 的中间态。
            if (getattr(self, "farm_scene_steps", 0)
                    >= getattr(self, "farm_scene_cap", FARM_SCENE_CAP)):
                self._mark_exhausted()

    def _sync_farm_scene(self, scene) -> None:
        """scene identity 变化时原子清空累计 FARM 预算（主层/任务图均适用）。"""
        scene = tuple(scene)
        if getattr(self, "_farm_scene", None) != scene:
            if getattr(self, "resource_protocol", "off") != "off":
                previous = getattr(self, "_farm_scene", None)
                if previous is not None:
                    self._resource_farm_ledgers[previous] = int(self.farm_scene_steps)
                self._farm_scene = scene
                self.farm_scene_steps = int(self._resource_farm_ledgers.get(scene, 0))
            else:
                self._farm_scene = scene
                self.farm_scene_steps = 0

    # ---- gym 接口 ----
    def reset(self, *, seed=None, options=None):
        # OptionsEnv 自身也是 Gym Env，必须建立它自己的 np_random；
        # 只给内层 env 传 seed 会被 env_checker 判为不符合 Gymnasium 契约。
        super().reset(seed=seed)
        if getattr(self, "worker_time_protocol", "legacy").startswith("completion-l2"):
            self._completion_clock.reset()
            self.env.max_steps = self._completion_clock.state.physical_deadline
        if getattr(self, "resource_service_policy", "legacy-v1") in ("sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6", "sustain-loot-v1"):
            self.env.resource_observation_callback = None
        obs, info = self.env.reset(seed=seed, options=options)
        self._reset_wrapper_state()
        self._last_base_obs = obs
        return self._mgr_obs(obs), info

    def _controller_action_context(
        self,
    ) -> tuple[np.ndarray, int | None]:
        """Delegate to the real env; retain a narrow pure-fixture fallback."""
        builder = getattr(self.env, "controller_action_context", None)
        if callable(builder):
            return builder()
        # Unit-test shells predating the controller wire have no native map.
        # Production OptionsEnv always constructs DiabloGymEnv above and can
        # never enter this branch.
        raw = self.env._raw
        mask = np.ones(15, dtype=bool)
        mask[9] = bool(DiabloGymEnv._policy_monsters(raw))
        mask[12] = (
            int(raw.get("belt_heals", 0)) > 0
            and int(raw.get("hp", 0)) < int(raw.get("max_hp", 0))
        )
        mask[13] = (
            DiabloGymEnv._belt_free_slots(raw) > 0
            and bool(DiabloGymEnv._policy_floor_items(raw, "heal"))
        )
        mask[14] = bool(
            DiabloGymEnv._policy_floor_items(raw, "gear"))
        return mask, _nearest(raw)

    def action_masks(self) -> np.ndarray:
        if self._last_base_obs is None:
            raise gym.error.ResetNeeded("OptionsEnv.action_masks() 前必须 reset()")
        self.env._ensure_active(allow_ended=True)
        raw = self.env._raw
        self._sync_farm_scene(_scene_identity(raw))
        m = np.ones(3, dtype=bool)
        # DIVE = 推进下一项主线目标，而不再狭义等于 NEXT 楼梯：任务入口、
        # Vile 机关、L16 开门机关及 set-level 返回都属于同一职权。
        transition = (bridge.WM_DIABRTNLVL if raw.get("is_set_level")
                      else bridge.WM_DIABNEXTLVL)
        m[DIVE] = bool(raw.get("progression_targets")) or any(
            t.get("msg") == transition for t in raw.get("triggers", []))
        controller_mask, nearest = self._controller_action_context()
        # RESUPPLY must mean "a13 can execute now", not merely "a potion
        # exists somewhere in the full raw list".  The latter opened endless
        # two-wait windows for a visible potion just outside radius 12.
        m[RESUPPLY] = bool(controller_mask[13])
        # 剧情目标是 DIVE 的专属职权。榨干态也必须在 DIVE 合法时交权：
        # v4 已移除 action10“无 frontier 就偷偷下楼”的旧泄漏，而冻结旧 H
        # 曾依赖该泄漏，即使 layer_clock 饱和仍会连续复选 FARM，直至整局
        # 截断。若 DIVE 当前非法则 FARM 仍作保底，保证掩码永不全假，并
        # 继续由 REVISIT_FLOOR 约束干层复访。
        forced_dive = _farm_handoff(
            raw, nearest) or (self.exhausted and bool(m[DIVE]))
        m[FARM] = not forced_dive
        if getattr(self, "resource_protocol", "off") != "off":
            from .resource_protocol import progression_allowed, native_readiness, RESOURCE_FARM_CAP
            law = getattr(self, "resource_readiness_law", "veto-v1")
            # veto-v1 masks DIVE on the native verdict; coach-v03 leaves the frozen
            # DIVE legality untouched (progression_allowed returns True under it).
            m[DIVE] = bool(m[DIVE] and progression_allowed(raw, law))
            service = self.resource_service
            if getattr(self, "resource_service_policy", "legacy-v1") == "sustain-loot-v1":
                alive = sum(1 for monster in raw.get("monsters", [])
                            if int(monster.get("hp", 0)) > 0 and int(monster.get("type", -1)) != 109)
                service.maybe_start(raw, self.env._steps,
                    farm_scene_steps=self.farm_scene_steps,
                    cleared=alive == 0 and int(raw.get("monster_kill_total", 0)) > 0)
            elif (not service.attempted and int(raw["dungeon_level"]) == 1
                    and not raw.get("is_set_level") and not native_readiness(raw)["ready"]):
                alive = sum(1 for monster in raw.get("monsters", [])
                            if int(monster.get("hp", 0)) > 0 and int(monster.get("type", -1)) != 109)
                cleared = alive == 0 and int(raw.get("monster_kill_total", 0)) > 0
                calibration = getattr(self, "resource_calibration", None)
                farm_trigger = (RESOURCE_FARM_CAP if calibration is None
                                else calibration.farm_scene_microstep_cap)
                if cleared or self.farm_scene_steps >= farm_trigger:
                    service.start(raw, self.env._steps, "cleared" if cleared else "farm_cap",
                                  farm_scene_steps=self.farm_scene_steps)
            retreat = getattr(self, "retreat_service", None)
            portal = getattr(self, "portal_service", None)
            if (portal is not None and portal.awaiting_return and not portal.active
                    and not service.active and int(raw.get("dungeon_level", 0)) == 0
                    and not raw.get("is_set_level")
                    and getattr(portal, "_town_visit_served", -1) != len(portal.attempts)):
                # R18-F portal-v1: a portal arrival in town runs the ordinary town
                # itinerary (heal / armor / potions / sell / scroll) BEFORE the return
                # leg. The loot service normally departs from L1; started here it
                # skips the stairs walk (outbound phase: depth 0 -> smith/healer) and
                # is closed at its return phase by the RESUPPLY-loop intercept, after
                # which the portal return trigger fires. One itinerary per visit.
                portal._town_visit_served = len(portal.attempts)
                if hasattr(service, "_begin_trip"):
                    # sustain-loot-v1 forbids direct start(); its trip entry is _begin_trip
                    # (maybe_start cannot fire in town by design). The visit counts as a trip.
                    service._begin_trip(raw, int(self.env._steps), "portal_arrival",
                                        int(self.farm_scene_steps), ["portal_arrival"])
                else:
                    service.start(raw, self.env._steps, "portal_arrival",
                                  farm_scene_steps=self.farm_scene_steps)
                # The collect phase (gold pickup) belongs to L1 and rejects town as an
                # unexpected scene; a portal visit enters the itinerary at outbound
                # (depth 0 -> sell / smith / healer / potions), which is exactly what
                # collect does when it finds nothing to pick up.
                service.phase = service._last_phase = "outbound"
                bridge.configure_town_service(True)
            if (portal is not None and not portal.active and not service.active
                    and (retreat is None or not retreat.active)):
                # R18-F portal-v1: consulted BEFORE the walking retreat. Its
                # outbound law is the retreat law PLUS "a scroll is carried", so
                # the coach prefers the portal exactly when the pair has one and
                # falls through to the stairs retreat otherwise. It also owns the
                # town-side legs (buy the scroll from Adria; walk back through).
                trigger = portal.trigger_reason(raw, int(self.env._steps))
                if trigger is not None:
                    portal.start(raw, int(self.env._steps), bridge, trigger)
            if (retreat is not None and not retreat.active and not service.active
                    and (portal is None or not portal.active)):
                # R18-A retreat-v1: the manager-side law fires here, exactly where the
                # town trip starts; the mask then collapses to RESUPPLY (= retreat).
                trigger = retreat.trigger_reason(raw, int(self.env._steps))
                if trigger is not None:
                    retreat.start(raw, int(self.env._steps), bridge, trigger)
            if portal is not None and portal.active:
                m[:] = False
                m[RESUPPLY] = True
            elif retreat is not None and retreat.active:
                m[:] = False
                m[RESUPPLY] = True
            elif service.active:
                m[:] = False
                m[RESUPPLY] = True
            elif law == "coach-v03":
                # R17.1 ruling 3: the frozen forced-descent law stands (escape hatch
                # kept); forced-unready descents are recorded by the engine receipt
                # and paid no escrow (worker_env settlement reads pretransition_ready_law).
                m[FARM] = not forced_dive
                m[RESUPPLY] = bool(controller_mask[13])
            else:
                # veto-v1: no exhausted/cleared bypass of the native admission verdict.
                m[FARM] = not (forced_dive and m[DIVE])
                m[RESUPPLY] = bool(controller_mask[13])
            self.env._resource_dive_authority = bool(
                self.dive_live_sovereignty and (self._win or {}).get("opt") == DIVE
                and m[DIVE])
        return m

    def resource_option_choice(self, mask=None):
        if getattr(self, "resource_protocol", "off") == "off":
            raise RuntimeError("Resource manager requires l2-town-v1")
        from .resource_protocol import law_ready
        mask = self.action_masks() if mask is None else mask
        portal = getattr(self, "portal_service", None)
        if portal is not None and portal.active:
            return RESUPPLY
        retreat = getattr(self, "retreat_service", None)
        if retreat is not None and retreat.active:
            return RESUPPLY
        if self.resource_service.active:
            return RESUPPLY
        raw = self.env._raw
        # veto-v1: native seven-condition verdict; coach-v03: six-condition law
        # (health excluded). The town-trip trigger above keeps the full native
        # verdict so low HP still sends the manager to Pepin (ruling 3).
        if mask[DIVE] and (raw.get("is_set_level") or raw.get("progression_targets")
                           or law_ready(raw, getattr(self, "resource_readiness_law", "veto-v1"))):
            return DIVE
        for option in (FARM, RESUPPLY, DIVE):
            if mask[option]:
                return option
        raise RuntimeError("Resource manager has no legal option")

    # ---- 共享窗口核(v23:OptionsEnv 与 WorkerWindowEnv 唯一实现)----
    def _dive_target_distance(self, raw) -> int | None:
        """DIVE 当前主线目标的 Chebyshev 距离(与 action_masks 同口径:
        有剧情目标取各目标最小距,否则取 transition 触发器最近者)。"""
        px, py = raw["player_x"], raw["player_y"]
        targets = raw.get("progression_targets") or []
        if targets:
            return min(
                max(abs(int(t["goal_x"]) - px), abs(int(t["goal_y"]) - py))
                for t in targets)
        transition = (bridge.WM_DIABRTNLVL if raw.get("is_set_level")
                      else bridge.WM_DIABNEXTLVL)
        stairs = [t for t in raw.get("triggers", [])
                  if t.get("msg") == transition]
        if not stairs:
            return None
        return min(
            max(abs(t["x"] - px), abs(t["y"] - py)) for t in stairs)

    def _win_begin(self, option: int):
        if not self.action_space.contains(option):
            raise ValueError(f"选项必须是 {self.action_space}中的整数，收到 {option!r}")
        option = int(option)
        if getattr(self, "resource_protocol", "off") != "off":
            self.env._resource_dive_authority = False
        if not self.action_masks()[option]:
            raise ValueError(f"选项 {option} 被掩码却被选择")
        raw = self.env._raw
        if (option == FARM
                and getattr(self, "reset_layer_clock_on_window", False)):
            # R16(C3 时钟继承):新 FARM 窗从零计无进展,不承接上一
            # DIVE stall 的余额;榨干旗与 legacy 钟不动(冻结观测语义)。
            self.layer_clock = 0
        self._win = {
            # _decisions 只在 _win_end 增一，因此同一底层局内这是稳定、
            # 单调且在窗口开始时即可对外报告的标识。
            "window_id": self._decisions + 1,
            "opt": option,
            "mode": ("farm", "dive", "resupply")[option],
            "t0": self.env._steps,
            "clvl0": raw["char_level"],
            "dlvl0": raw["dungeon_level"],
            "scene0": _scene_identity(raw),
            "kills0": int(self.env._ep_kills),
            "floor": REVISIT_FLOOR if (option == FARM and self.exhausted) else 0,
            "dive_best_d": (self._dive_target_distance(raw)
                            if option == DIVE else None),
            "dive_last_progress_tau": 0,
            "resupply_stall": 0,
            "R": 0.0, "W": 0.0, "bonus": 0.0,
            # 这两本账只覆盖 _win_step_worker：开窗前的 fuse recovery /
            # 脑干排水不属于网络 transition，不能混入 PPO 实际领取的工资
            # 或“工人伴随击杀”。WorkerWindowEnv.step 返回的 wage 与这里
            # 逐步累加的增量使用同一个 W 差分，二者必须严格同源。
            "worker_wage": 0.0, "worker_kills": 0,
            "worker_action14_requests": 0,
            "worker_action14_native_successes": 0,
            "worker_action14_gear_utility_delta": 0,
            "no_effect_requests": 0,
            "worker_no_effect_requests": 0,
            "executed_requests": 0,
            # After a FARM kill exposes progression, give the Worker exactly
            # one visible {decline, equip} decision before manager handoff.
            "gear_grace_pending": False,
            "gear_grace_consumed": False,
            "gear_grace_opportunities": 0,
            "gear_grace_decisions": 0,
            "beats": 0, "overrides": 0, "fuse_trips": 0,
            # attempts includes hit-recovery rejections; drains counts only
            # native-accepted potion uses.  Conflating them previously made
            # failed rescues look successful in evaluation.
            "drain_attempts": 0, "drains": 0,
            "voluntary_drinks": 0,
            "recovery_actions": 0, "last_recovery_action": None,
            "last_requested_action": None, "last_executed_action": None,
            "fuse_requested_action": None,
            "done": False, "trunc": False, "last_info": {},
        }
        if getattr(self, "resource_protocol", "off") != "off":
            self._win["resource_depth0"] = int(self.env._resource_max_main_depth)
            retreat = getattr(self, "retreat_service", None)
            self._win["retreat_window"] = bool(
                retreat is not None and retreat.active and option == RESUPPLY)
            portal = getattr(self, "portal_service", None)
            self._win["portal_window"] = bool(
                portal is not None and portal.active and option == RESUPPLY)

    def _beat(self, a: int, *, worker_authority: bool = False):
        """一拍:保险丝 → env.step → 观测缓存 → 停滞钟。

        返回 ``(r, done, trunc, info, audit, lvl_before, belt_free_before)``。
        保险丝命中时不执行任何基础动作，``audit.executed_action`` 为 None；
        调用方必须立即以 ``reason="fuse"`` 结束当前窗口。
        """
        raw = self.env._raw
        requested = int(a)
        sig = (("resource", tuple(self._resource_command), int(self.env._steps))
               if getattr(self, "_resource_command", None) is not None
               else self._sig(a, raw))
        if sig == self._fuse_sig:
            self._fuse += 1
            if self._fuse >= 25:
                # 旧行为在这里静默执行动作 10，但 PPO/BC 仍把本 transition
                # 标成 requested，形成动作—结果错标。现在拒绝本拍并归还
                # manager；清空签名使下一窗口从干净保险丝状态开始。
                self._fuse = 0
                self._fuse_sig = None
                # 保险丝拒拍不执行任何原生动作;a14 的回执强制(每个请求必带
                # 因果回执)要求这里也发布显式拒绝回执,否则消费端 fail-closed。
                # (2026-07-27 修复:07-25 加回执强制时漏改本早退分支——
                # prepare-bc 教师连按 a14 至 25 拍触发保险丝即崩。)
                fuse_action14_audit = None
                if requested == 14:
                    fuse_utility = gear_combat_utility_value(
                        raw, "action14_fuse_reject")
                    fuse_action14_audit = {
                        "accepted": False,
                        "commit_attempts": 0,
                        "utility_before": fuse_utility,
                        "utility_after": fuse_utility,
                        "utility_delta": 0,
                    }
                audit = BeatOutcome(
                    reason="fuse",
                    requested_action=requested,
                    executed_action=None,
                    fuse_tripped=True,
                    action14_audit=fuse_action14_audit,
                )
                return 0.0, False, False, {}, audit, \
                    raw["dungeon_level"], DiabloGymEnv._belt_free_slots(raw)
        else:
            self._fuse = 0
        self._fuse_sig = sig
        kills_b = self.env._ep_kills
        lvl_b = raw["dungeon_level"]
        scene_b = _scene_identity(raw)
        self._sync_farm_scene(scene_b)
        steps_b = self.env._steps
        belt_free_b = DiabloGymEnv._belt_free_slots(raw)
        exploration_b = int(getattr(self.env, "exploration_progress", 0))
        gear_utility_b = gear_combat_utility_value(
            raw, "options_before")
        gold_b = int(raw.get("gold", 0))
        combat_floor_b = dict(getattr(self.env, "_combat_hp_floor", {}))
        if getattr(self, "_resource_command", None) is not None:
            obs, r, done, trunc, info = self.env.step_resource(self._resource_command)
        elif worker_authority:
            obs, r, done, trunc, info = self.env.step(
                a, worker_authority=True)
        else:
            obs, r, done, trunc, info = self.env.step(a)
        self._last_base_obs = obs
        current = self.env._raw
        action14_audit = (
            _validated_action14_audit(info)
            if requested == 14 else None
        )
        action_effect_audit = _validated_action_effect_audit(
            info, requested)
        steps_delta = self.env._steps - steps_b
        current_scene = _scene_identity(current)
        if current_scene != scene_b:
            self._sync_farm_scene(current_scene)
        elif (getattr(self, "_win", None) is not None
              and int(self._win.get("opt", -1)) == FARM):
            self.farm_scene_steps += steps_delta
        combat_floor = getattr(self.env, "_combat_hp_floor", {})
        new_damage_floor = any(
            mid in combat_floor and int(combat_floor[mid][0]) < int(before[0])
            for mid, before in combat_floor_b.items()
        )
        positive_progress = (
            int(getattr(self.env, "exploration_progress", 0)) > exploration_b
            or new_damage_floor
            or (
                action14_audit is not None
                and action14_audit["accepted"]
            )
            or gear_combat_utility_value(
                current, "options_after") > gear_utility_b
            or DiabloGymEnv._belt_free_slots(current) < belt_free_b
            or int(current.get("gold", 0)) > gold_b
        )
        self._tick_layer_clock(
            kills_b,
            scene_b,
            steps_delta,
            positive_progress=positive_progress,
        )
        _dive_w = getattr(self, "_win", None)
        if (_dive_w is not None and _dive_w.get("opt") == DIVE
                and getattr(self, "dive_stall_protocol", "tau-v3")
                == "no-progress-v1"):
            w = _dive_w
            d_now = self._dive_target_distance(current)
            best = w.get("dive_best_d")
            improved = (
                d_now is not None
                and (best is None or d_now < best)
            )
            if improved:
                w["dive_best_d"] = d_now
            if (improved
                    or int(self.env._ep_kills) > int(kills_b)
                    or positive_progress):
                w["dive_last_progress_tau"] = self.env._steps - w["t0"]
        executed_action = (
            int(a)
            if action_effect_audit["request_executed"]
            else None
        )
        if requested == 12:
            drink_audit = info.get("action12_audit")
            if (
                not isinstance(drink_audit, dict)
                or set(drink_audit) != {
                    "accepted", "accepted_belt_before",
                    "belt_before", "consumed", "belt_after",
                }
                or not isinstance(drink_audit["accepted"], bool)
                or not isinstance(drink_audit["consumed"], bool)
                or drink_audit["accepted"] != drink_audit["consumed"]
            ):
                raise RuntimeError(
                    "action12 缺失/损坏真实执行回执")
            executed_action = 12 if drink_audit["consumed"] else None
        elif requested == 14:
            # A14 is a macro: an accepted safe walk/open is a real execution
            # even when this decision has not reached the final gear commit.
            # Keep completion in action14_audit; only require the one-way
            # implication that a successful commit was truly executed.
            if action14_audit["accepted"] and executed_action != 14:
                raise RuntimeError(
                    "action14 成功提交缺少通用执行回执")
        audit = BeatOutcome(
            reason=None,
            requested_action=requested,
            executed_action=executed_action,
            fuse_tripped=False,
            action14_audit=action14_audit,
            action_effect_audit=action_effect_audit,
        )
        return float(r), done, trunc, info, audit, lvl_b, belt_free_b

    def _win_term(self, done, trunc, belt_free_b):
        """七级终止阶梯(顺序与 v22 逐行同构)。返回 reason 或 None。"""
        w = self._win
        raw = self.env._raw
        tau = self.env._steps - w["t0"]
        if done or trunc:
            return "death" if raw.get("dead") else "end"
        if _scene_identity(raw) != w["scene0"]:
            # Resource service returns town -> visited L1. Only a newly reached
            # main depth is descent; a scene return must not inflate statistics.
            if getattr(self, "resource_protocol", "off") != "off":
                return ("descend" if int(self.env._resource_max_main_depth)
                        > w["resource_depth0"] else "scene")
            # 进入/离开任务副本同样必须归还控制权，但只有主线深度增加
            # 才叫 descend、才可领取下潜奖金。
            return ("descend" if raw["dungeon_level"] > w["dlvl0"]
                    else "scene")
        opt = w["opt"]
        portal = getattr(self, "portal_service", None)
        if portal is not None:
            # R18-F portal-v1 (constructed only when the flag is on): a portal
            # window closes when the script hands back; any other window closes
            # the beat a portal leg becomes legal so the mask can take the word.
            if w.get("portal_window"):
                return None if portal.active else "portal_complete"
            if (opt != RESUPPLY and not portal.active
                    and portal.trigger_reason(raw, self.env._steps) is not None):
                return "portal_trigger"
        retreat = getattr(self, "retreat_service", None)
        if retreat is not None:
            # R18-A retreat-v1 (constructed only when the flag is on): a retreat
            # window closes when the script hands back; any other window closes
            # the beat the retreat law fires so the mask can take the word.
            if w.get("retreat_window"):
                return None if retreat.active else "retreat_complete"
            if (opt != RESUPPLY and not retreat.active
                    and retreat.trigger_reason(raw, self.env._steps) is not None):
                return "retreat_trigger"
        # FARM 的最后一只近敌被清掉后，剧情目标可能在同一拍成为当前
        # 状态；必须先于复访地板/停滞钟收窗，不能让 action10 越权操作。
        if opt == FARM:
            controller_mask, nearest = self._controller_action_context()
            if _farm_handoff(raw, nearest):
                exact_gear = bool(controller_mask[14])
                if (
                    not w.get("gear_grace_consumed", False)
                    and exact_gear
                ):
                    if (
                        int(w.get("gear_grace_decisions", 0))
                        >= GEAR_GRACE_MAX_DECISIONS
                    ):
                        w["gear_grace_consumed"] = True
                        w["gear_grace_pending"] = False
                        return "handoff"
                    if not w.get("gear_grace_pending", False):
                        w["gear_grace_pending"] = True
                        if int(w.get(
                                "gear_grace_opportunities", 0)) == 0:
                            w["gear_grace_opportunities"] = 1
                    return None
                w["gear_grace_pending"] = False
                return "handoff"
        if tau < w["floor"]:
            return None
        if opt == FARM and raw["char_level"] > w["clvl0"]:
            return "levelup"
        # db7d26c 只在 FARM 的这一层级、并且在 done/scene/floor/levelup
        # 之后发布旧 exhausted。即使当前 no-progress 因真实正向进度而
        # 不收窗，冻结 V28 仍应看到它在旧协议此刻会收到的 296/297。
        if opt == FARM and self._legacy_layer_clock >= KILL_PATIENCE:
            self._legacy_exhausted = True
        if (opt == FARM
                and (self.layer_clock >= KILL_PATIENCE
                     or self.farm_scene_steps
                     >= getattr(self, "farm_scene_cap", FARM_SCENE_CAP))):
            self._mark_exhausted()
            return "exhausted"
        if opt == DIVE:
            if (getattr(self, "dive_stall_protocol", "tau-v3")
                    == "no-progress-v1"):
                if (tau - int(w.get("dive_last_progress_tau", 0))
                        >= KILL_PATIENCE):
                    return "stall"
            elif tau >= KILL_PATIENCE:
                return "stall"
        if opt == RESUPPLY and getattr(self, "resource_protocol", "off") != "off" and self.resource_service.attempted:
            if self.resource_service.active:
                return None
            return "resource_complete"
        if opt == RESUPPLY:
            belt_free = DiabloGymEnv._belt_free_slots(raw)
            if belt_free >= belt_free_b:
                w["resupply_stall"] += 1
            else:
                w["resupply_stall"] = 0
            exact_heal_available = bool(
                self._controller_action_context()[0][13])
            if (belt_free <= 0
                    or not exact_heal_available
                    or w["resupply_stall"] >= 2 or tau >= RESUPPLY_CAP):
                return "done"
        if tau >= TAU_CAP:
            self._cap_hits += 1
            return "cap"
        return None

    def _win_beat(self, a: int, *, worker_authority: bool = False):
        """一拍 + 账本(经理 R / 工人 W)+ 终止判定。"""
        w = self._win
        # Narrow compatibility for hand-built unit-test windows.  Production
        # windows initialize every ledger in _win_begin.
        for key in (
            "no_effect_requests",
            "worker_no_effect_requests",
            "executed_requests",
            "gear_grace_opportunities",
            "gear_grace_decisions",
        ):
            w.setdefault(key, 0)
        w.setdefault("gear_grace_pending", False)
        w.setdefault("gear_grace_consumed", False)
        r, done, trunc, info, audit, lvl_b, belt_free_b = self._beat(
            a, worker_authority=worker_authority)
        w["last_requested_action"] = audit.requested_action
        w["last_executed_action"] = audit.executed_action
        if audit.fuse_tripped:
            # 拒绝提案不消耗基础微拍、不产生伪奖励；只登记可审计的
            # manager 介入。dry revisit 的最小占用地板仍是硬不变量：
            # 地板未满时本拍只拒绝、不收窗；清空后的 fuse 允许下一提案
            # 正常推进，避免零微步死循环。
            w["overrides"] += 1
            w["fuse_trips"] += 1
            w["fuse_requested_action"] = audit.requested_action
            tau = self.env._steps - w["t0"]
            reason = "fuse" if tau >= w["floor"] else None
            if reason is not None:
                # 拒绝拍本身不偷执行任何动作。下一经理窗口开头再执行一拍
                # 显式登记的脚本恢复，避免同一坏几何状态被经理反复重选。
                self._fuse_recovery_pending = True
            return BeatOutcome(
                reason=reason,
                requested_action=audit.requested_action,
                executed_action=None,
                fuse_tripped=True,
                # 2026-07-27 修复第二洞:窗口级 fuse 包装器必须传播内层
                # a14 拒绝回执(_beat 保险丝路径已合成),否则 a14 请求在
                # 此路径回执为 None,消费端 fail-closed(2_106 池阵亡原因)。
                action14_audit=audit.action14_audit,
                action_effect_audit=audit.action_effect_audit,
            )
        w["beats"] += 1
        if (
            audit.requested_action != 0
            and audit.action_effect_audit is not None
            and not audit.action_effect_audit["request_executed"]
        ):
            w["no_effect_requests"] += 1
        elif (
            audit.action_effect_audit is not None
            and audit.action_effect_audit["request_executed"]
            and audit.requested_action != 0
        ):
            w["executed_requests"] += 1
        w["R"] += r
        cur_lvl = self.env._raw["dungeon_level"]
        # 剥薪单价跟随经济规格(v1 时 == DESCEND_UNIT,数值逐位同旧;
        # R10 v2 换单价后恒等式 Σw ≡ 窗口R − unit×ΣΔdlvl⁺ 继续成立,
        # 防"临战下楼避罚"套利语义不变)。测试替身/旧包装无该属性时
        # 回退 v1——与历史行为逐位一致。
        descend_unit = getattr(
            self.env, "reward_economy", REWARD_ECONOMY_V1).descend_unit
        bonus = descend_unit * sum(range(lvl_b, cur_lvl)) if cur_lvl > lvl_b else 0.0
        if getattr(self, "resource_protocol", "off") != "off":
            bonus = float(info["resource_protocol"]["new_main_depth_bonus"])
        # R14 乙案(主席批「先做乙 我想看看在乙的条件下 是否仍会出发
        # 裸奔哥的情况」):live DIVE 窗内工人保留 fraction 的下楼奖金。
        # w["bonus"] 语义=实际剥除额,恒等式 R≡W+bonus 逐位继续成立;
        # fraction=0(默认)或非 live-DIVE 窗时本路径逐位同旧法。
        if (bonus > 0.0
                and getattr(
                    self, "worker_descend_bonus_fraction", 0.0) > 0.0
                and getattr(self, "dive_live_sovereignty", False)
                and w.get("opt") == DIVE):
            _kept = bonus * self.worker_descend_bonus_fraction
            bonus -= _kept
            self.descend_bonus_kept_total = float(getattr(
                self, "descend_bonus_kept_total", 0.0)) + _kept
        w["bonus"] += bonus
        w["W"] += r - bonus
        w["done"], w["trunc"], w["last_info"] = done, trunc, info
        return BeatOutcome(
            reason=self._win_term(done, trunc, belt_free_b),
            requested_action=audit.requested_action,
            executed_action=audit.executed_action,
            fuse_tripped=False,
            action14_audit=audit.action14_audit,
            action_effect_audit=audit.action_effect_audit,
        )

    def _drain(self):
        """反射排水:hp<0.5∧belt>0 时由包装器逐拍喝药(工人路径专用;
        每一拍照常过保险丝/时钟/终止阶梯——喝药拍可跨榨干/CAP/死亡)。

        返回使排水结束窗口的 BeatOutcome；排水完成且窗口仍活跃则返回 None。
        """
        w = self._win
        while _reflex(self.env._raw):
            w["drain_attempts"] += 1
            outcome = self._win_beat(12)
            if outcome.executed_action == 12:
                w["drains"] += 1
            if outcome.reason is not None:
                return outcome
        return None

    def _win_step_worker(self, a: int):
        """工人一步 = 工人动作一拍 + 反射尾部排水。"""
        if not self.env.action_space.contains(a):
            raise ValueError(f"工人动作必须是 {self.env.action_space}中的整数，收到 {a!r}")
        a = int(a)
        if (
            1 <= a <= 8
            and a in DiabloGymEnv._protected_walk_actions(self.env._raw)
            and not (
                getattr(self, "dive_live_sovereignty", False)
                and (self._win or {}).get("opt") == DIVE
            )   # R13:live DIVE 窗内踏格即职权,与掩码放行同条件
        ):
            raise ValueError(
                f"工人动作 {a} 试图踏入 DIVE 专属 trigger/剧情格")
        if not self._worker_masks()[a]:
            raise ValueError(f"工人动作 {a} 被掩码却被执行")
        wage_before = float(self._win["W"])
        kills_before = int(self.env._ep_kills)
        grace_decision = bool(
            self._win.get("gear_grace_pending", False))
        if grace_decision:
            self._win["gear_grace_pending"] = False
            # a0 explicitly declines.  a14 keeps the grace open until the
            # exact native gear commit removes the target, or the bounded
            # retry budget is exhausted; one macro may need several decisions
            # to traverse the fixed radius-12 path.
            self._win["gear_grace_consumed"] = a == 0
            self._win["gear_grace_decisions"] += 1
        primary = (
            self._win_beat(a, worker_authority=True)
            if 1 <= a <= 8
            else self._win_beat(a)
        )
        if grace_decision and primary.fuse_tripped:
            # Fuse rejection executes no base action, so it cannot consume the
            # one learning opportunity.
            self._win["gear_grace_pending"] = True
            self._win["gear_grace_consumed"] = False
            self._win["gear_grace_decisions"] -= 1
        if a == 12 and primary.executed_action == 12:
            # 这是工人主动饮药，不含 _drain 的脑干反射。计数只用于审计；
            # 计数会在下一策略观测的 feature 297 符号中公开，并关闭本窗
            # 后续 worker-owned a12；脑干 _drain 仍可在紧急态连续排水。
            # 这样“每窗至多一次工人饮药”既是可见状态也是执行约束。
            self._win["voluntary_drinks"] += 1
        if a == 14:
            self._win.setdefault("worker_action14_requests", 0)
            self._win.setdefault(
                "worker_action14_native_successes", 0)
            self._win.setdefault(
                "worker_action14_gear_utility_delta", 0)
            self._win["worker_action14_requests"] += 1
            gear_audit = primary.action14_audit
            if gear_audit is None:
                raise RuntimeError("action14 缺少原生装备提交回执")
            if gear_audit["accepted"]:
                if (
                    primary.executed_action != 14
                    or gear_audit["utility_delta"] <= 0
                ):
                    raise RuntimeError(
                        "action14 成功提交未发布真实执行回执")
                self._win[
                    "worker_action14_native_successes"
                ] += 1
                self._win[
                    "worker_action14_gear_utility_delta"
                ] += int(gear_audit["utility_delta"])
                if grace_decision:
                    # The grace budget exists to let one fixed pickup macro
                    # finish walking to its target.  Once the native atomic
                    # commit succeeds, hand control back even if another
                    # upgrade happens to become the new nearest target.
                    self._win["gear_grace_consumed"] = True
        if (
            a != 0
            and primary.action_effect_audit is not None
            and not primary.action_effect_audit["request_executed"]
        ):
            self._win["worker_no_effect_requests"] += 1
        ending = primary
        if primary.reason is None:
            drain_ending = self._drain()
            if drain_ending is not None:
                ending = drain_ending
        # 与 WorkerWindowEnv.step 的 policy reward 完全相同：当前工人提案
        # 及其反射尾部所造成的 W 增量。开窗恢复/排水发生在调用本方法之前，
        # 因而不会被错误归给网络。
        self._win["worker_wage"] += float(self._win["W"]) - wage_before
        worker_kills = int(self.env._ep_kills) - kills_before
        if worker_kills < 0:
            raise RuntimeError("单局击杀计数在 worker transition 内回退")
        self._win["worker_kills"] += worker_kills
        fuse = primary if primary.fuse_tripped else (
            ending if ending.fuse_tripped else None)
        return WorkerStepOutcome(
            reason=ending.reason,
            requested_action=a,
            executed_action=primary.executed_action,
            fuse_tripped=fuse is not None,
            fuse_requested_action=(None if fuse is None
                                   else fuse.requested_action),
            action14_audit=primary.action14_audit,
            action_effect_audit=primary.action_effect_audit,
        )

    def _consume_fuse_recovery(self):
        """在下一窗口开头执行一拍可审计恢复；不归因成被拒的 worker 动作。"""
        if not self._fuse_recovery_pending:
            return None
        mode = self._win["mode"]
        if mode == "farm":
            raw = self.env._raw
            controller_mask, _nearest = (
                self._controller_action_context())
            # FARM recovery 也不得借 action10 操作剧情；有剧情时继续清理
            # 可见 blocker（无怪的 handoff 态本就不会开 FARM）。
            action = (
                9 if (raw.get("progression_targets")
                      and bool(controller_mask[9]))
                else (0 if raw.get("progression_targets") else 10)
            )
        else:
            action = {"dive": 11, "resupply": 0}[mode]
        self._fuse_recovery_pending = False
        self._win["recovery_actions"] += 1
        self._win["last_recovery_action"] = action
        return self._win_beat(action)

    def _win_end(self, reason: str):
        """收窗:经理状态机推进 + option_extra。返回 (extra, base_info, done, trunc)。"""
        if reason is None:
            raise RuntimeError("选项尚无终止原因却尝试收窗")
        w = self._win
        for key in (
            "no_effect_requests",
            "worker_no_effect_requests",
            "executed_requests",
            "gear_grace_opportunities",
            "gear_grace_decisions",
        ):
            w.setdefault(key, 0)
        tau = self.env._steps - w["t0"]
        terminal_info = (
            w["last_info"] if isinstance(w.get("last_info"), dict) else {})
        safe_time_limit = (
            terminal_info.get("time_limit_bootstrap_safe") is True)
        unsettled_time_limit = (
            terminal_info.get("unsettled_budget_terminal") is True)
        native_terminal = bool(
            self.env._raw.get("dead")
            or self.env._raw.get("game_over")
            or self.env._raw.get("victory")
        )
        budget_boundary = bool(
            not native_terminal
            and safe_time_limit != unsettled_time_limit
        )
        self._decisions += 1
        self._last_opt, self._last_tau = w["opt"], tau
        self.mode_seq.append("FDR"[w["opt"]] + ("†" if reason == "death" else ""))
        extra = {
            "window_id": w["window_id"],
            "opt": w["opt"], "tau": tau, "reason": reason,
            "micro_steps": self.env._steps, "decisions": self._decisions,
            "cap_hits": self._cap_hits, "mode_seq": "".join(self.mode_seq),
            "R": w["R"], "W": w["W"], "bonus": w["bonus"],
            "kills_delta": int(self.env._ep_kills) - int(w["kills0"]),
            "worker_wage": w["worker_wage"],
            "worker_kills": w["worker_kills"],
            "worker_action14_requests":
                w["worker_action14_requests"],
            "worker_action14_native_successes":
                w["worker_action14_native_successes"],
            "worker_action14_gear_utility_delta":
                w["worker_action14_gear_utility_delta"],
            "no_effect_requests": w["no_effect_requests"],
            "worker_no_effect_requests": w["worker_no_effect_requests"],
            "executed_requests": w["executed_requests"],
            "gear_grace_opportunities": w["gear_grace_opportunities"],
            "gear_grace_decisions": w["gear_grace_decisions"],
            "beats": w["beats"], "overrides": w["overrides"],
            "fuse_trips": w["fuse_trips"],
            "drain_attempts": w["drain_attempts"],
            "drains": w["drains"],
            "voluntary_drinks": w["voluntary_drinks"],
            "recovery_actions": w["recovery_actions"],
            "last_recovery_action": w["last_recovery_action"],
            "last_requested_action": w["last_requested_action"],
            "last_executed_action": w["last_executed_action"],
            "fuse_requested_action": w["fuse_requested_action"],
            "dlvl0": w["dlvl0"], "dlvl_end": self.env._raw["dungeon_level"],
            "farm_scene_steps": self.farm_scene_steps,
            "dry": w["floor"] > 0,      # 开窗时榨干旗在位(干层复访窗)
            "base_done": w["done"] or w["trunc"],
            "base_trunc": w["trunc"] and not w["done"],
            "no_progress_micro_steps": int(self.layer_clock),
            "budget_boundary": budget_boundary,
            "timeout_without_progress": bool(
                budget_boundary
                and int(self.layer_clock) >= KILL_PATIENCE
            ),
        }
        if (not all(math.isfinite(float(extra[key]))
                    for key in ("R", "W", "bonus", "worker_wage"))
                or not math.isclose(
                    float(extra["R"]),
                    float(extra["W"]) + float(extra["bonus"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12)):
            raise RuntimeError(
                "选项回报分账异常: "
                f"R={extra['R']}, W={extra['W']}, bonus={extra['bonus']}, "
                f"worker_wage={extra['worker_wage']}")
        if (extra["kills_delta"] < 0
                or not 0 <= extra["worker_kills"] <= extra["kills_delta"]):
            raise RuntimeError(
                "选项击杀分账异常: "
                f"kills_delta={extra['kills_delta']}, "
                f"worker_kills={extra['worker_kills']}")
        if (
            not 0
            <= extra["worker_action14_native_successes"]
            <= extra["worker_action14_requests"]
            or extra["worker_action14_gear_utility_delta"] < 0
            or (
                extra["worker_action14_native_successes"] == 0
            ) != (
                extra["worker_action14_gear_utility_delta"] == 0
            )
        ):
            raise RuntimeError(
                "选项 action14 原生回执分账异常: "
                f"requests={extra['worker_action14_requests']}, "
                "successes="
                f"{extra['worker_action14_native_successes']}, "
                "utility_delta="
                f"{extra['worker_action14_gear_utility_delta']}")
        if (
            not 0 <= extra["worker_no_effect_requests"]
            <= extra["no_effect_requests"] <= extra["beats"]
            or not 0 <= extra["executed_requests"] <= extra["beats"]
            or (
                extra["no_effect_requests"]
                + extra["executed_requests"]
                > extra["beats"]
            )
            or not 0 <= extra["gear_grace_opportunities"] <= 1
            or not 0 <= extra["gear_grace_decisions"] <= (
                GEAR_GRACE_MAX_DECISIONS
                * max(1, extra["gear_grace_opportunities"])
            )
        ):
            raise RuntimeError(
                "选项动作效果/装备学习窗口分账异常:"
                f"no_effect={extra['no_effect_requests']},"
                f"worker_no_effect={extra['worker_no_effect_requests']},"
                f"executed={extra['executed_requests']},"
                f"beats={extra['beats']},"
                f"gear_grace={extra['gear_grace_decisions']}/"
                f"{extra['gear_grace_opportunities']}")
        base_info, done, trunc = w["last_info"], w["done"], w["trunc"]
        self._win = None
        return extra, base_info, done, trunc

    # ---- 工人视角(v23)----
    def _worker_obs(self) -> np.ndarray:
        w = self._win
        tau = self.env._steps - w["t0"] if w is not None else self._last_tau
        # V28/KING/root 的 actor 与 critic 均按 protocol-v3 训练：296 是
        # 无新杀钟，297 是旧 exhausted。当前 no-progress/scene-cap 状态
        # 仍驱动窗口，但不得进入这两个冻结输入槽。
        legacy_exhausted = 1.0 if self._legacy_exhausted else 0.0
        exhausted_and_latch = legacy_exhausted
        if w is not None and (
            int(w.get("voluntary_drinks", 0)) > 0
            or int(w.get("drains", 0)) > 0
        ):
            # 符号域与未饮的 0/1 严格分离；``-v-1`` 可无损恢复旧值。
            exhausted_and_latch = -(1.0 + legacy_exhausted)
        extra = np.asarray([
            min(1.0, tau / TAU_CAP),
            min(1.0, self._legacy_layer_clock / KILL_PATIENCE),
            exhausted_and_latch,
        ], dtype=np.float32)
        return np.concatenate([np.asarray(self._last_base_obs, dtype=np.float32), extra])

    def _worker_v5_window_appendix(self, w, tau, raw) -> np.ndarray:
        """R13 v5 追加块(12 维):窗口模式感知 + DIVE 进展/价值分解特征。

        仅 dual-v5-window-mode-v1 视图构造本块;v4 前缀在调用方逐位拼装,
        本函数零 RNG 消耗、零状态写入。零填充迁移的 actor/critic 对本块
        初始权重为零,故行为自 v4 起点连续(D0 门的实现前提)。
        """
        mode = (w or {}).get("mode")
        best_d = (w or {}).get("dive_best_d")
        stall = 0.0
        if w is not None and w.get("opt") == DIVE:
            stall = min(1.0, max(0.0, (
                float(tau) - float(w.get("dive_last_progress_tau", 0))
            ) / max(1, KILL_PATIENCE)))
        window_kills = 0.0
        worker_wage_frac = 0.0
        drinks_frac = 0.0
        if w is not None:
            window_kills = min(1.0, max(0.0, (
                float(self.env._ep_kills) - float(w.get("kills0", 0))
            ) / 50.0))
            worker_wage_frac = min(1.0, max(
                -1.0, float(w.get("worker_wage", 0.0)) / 50.0))
            drinks_frac = min(1.0, max(0.0, (
                float(w.get("drains", 0))
                + float(w.get("voluntary_drinks", 0))
            ) / 5.0))
        dlvl = float(raw.get("dungeon_level", 0))
        clvl = float(raw.get("char_level", 0))
        hp_frac = min(1.0, max(0.0, (
            float(raw.get("hp", 0)) / max(1.0, float(raw.get("max_hp", 0))))))
        appendix = np.asarray([
            1.0 if mode == "farm" else 0.0,       # [0] 模式 one-hot
            1.0 if mode == "dive" else 0.0,       # [1]
            1.0 if mode == "resupply" else 0.0,   # [2]
            (min(1.0, max(0.0, float(best_d) / 50.0))
             if best_d is not None else 0.0),     # [3] 主线目标最近距
            stall,                                # [4] DIVE 停滞钟
            min(1.0, max(0.0, float(tau) / max(1, TAU_CAP))),  # [5] 窗龄
            min(1.0, max(0.0, dlvl / 15.0)),      # [6] 地牢层
            min(1.0, max(-1.0, (clvl - dlvl) / 10.0)),  # [7] 等级余量
            window_kills,                         # [8] 窗内击杀
            hp_frac,                              # [9] 血量分数
            worker_wage_frac,                     # [10] 窗内工人工资
            drinks_frac,                          # [11] 窗内饮药
        ], dtype=np.float32)
        if (
            appendix.shape != (DUAL_WORKER_V5_APPENDIX_DIM,)
            or not np.isfinite(appendix).all()
        ):
            raise RuntimeError(
                "dual Worker v5 追加块形状/有限性漂移:"
                f"shape={appendix.shape},"
                f"finite={np.isfinite(appendix).all()}")
        return appendix

    def _worker_policy_observation(
            self,
            view: str,
            *,
            skip_dry_probability: float = 0.0,
    ) -> np.ndarray:
        """Build a declared Worker representation at the lossless raw edge.

        A complete protocol-v3 row cannot be reconstructed from the v4 vector
        after invisible/unreachable monsters and items have been filtered.
        Rebuild the 295-wide base from native ``raw``.  The A12 representation
        overlays only the two reversible v4 fields used by its adapter:
        packed belt economy (286) and the signed drink latch (297).  Thus the
        inherited V28/KING actor can recover an exact v3 row while the adapter
        still observes its resource/latch state.  The asymmetric representation
        instead keeps that exact 298-wide legacy row intact, then appends the
        current v4 base, wrapper dynamics, real fuse streak, exact Worker /
        Manager masks, and the immutable controller snapshot according to the
        constants above.
        """
        if view not in WORKER_OBSERVATION_VIEWS:
            raise ValueError(
                f"Worker policy observation view 未注册:{view!r}")
        current = np.asarray(self._worker_obs(), dtype=np.float32)
        if current.shape != (298,):
            raise RuntimeError(
                f"Worker raw-v4 观测形状漂移:{current.shape} != (298,)")
        if view == WORKER_OBSERVATION_VIEW_RAW_V4:
            return current

        raw = getattr(self.env, "_raw", None)
        if not isinstance(raw, dict):
            raise RuntimeError(
                "lossless Worker policy observation requires active native raw")
        legacy_base = self.env._legacy_policy_vectorize(self.env._raw)
        legacy_result = np.concatenate([
            legacy_base,
            np.asarray([
                current[295],
                current[296],
                1.0 if self._legacy_exhausted else 0.0,
            ], dtype=np.float32),
        ]).astype(np.float32, copy=False)
        if legacy_result.shape != (298,):
            raise RuntimeError(
                "Worker legacy 观测形状漂移:"
                f"{legacy_result.shape} != (298,)")
        if view in (
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
            WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE,
        ):
            try:
                p_skip = float(skip_dry_probability)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "dual Worker skip_dry_probability 必须是 [0,1] 有限数"
                ) from exc
            if not math.isfinite(p_skip) or not 0.0 <= p_skip <= 1.0:
                raise ValueError(
                    "dual Worker skip_dry_probability 必须是 [0,1] 有限数")
            current_base = np.asarray(
                self._last_base_obs, dtype=np.float32)
            if current_base.shape != (295,):
                raise RuntimeError(
                    "dual Worker current-v4 base 形状漂移:"
                    f"{current_base.shape} != (295,)")
            snapshot_vectorizer = getattr(
                self.env, "controller_snapshot_vector", None)
            if not callable(snapshot_vectorizer):
                raise RuntimeError(
                    "dual Worker v3 requires controller_snapshot_vector()")
            controller_snapshot = np.asarray(
                snapshot_vectorizer(), dtype=np.float32)
            if (
                controller_snapshot.shape
                != (CONTROLLER_SNAPSHOT_VECTOR_DIM,)
                or not np.isfinite(controller_snapshot).all()
            ):
                raise RuntimeError(
                    "dual Worker controller snapshot 形状/有限性漂移:"
                    f"shape={controller_snapshot.shape},"
                    f"finite={np.isfinite(controller_snapshot).all()}")

            # action_masks() synchronizes the scene-budget identity.  Do this
            # before reading any context scalar so masks and context describe
            # one post-sync state rather than opposite sides of a scene reset.
            manager_mask = np.asarray(
                self.action_masks(), dtype=bool)
            worker_mask = np.asarray(
                self._worker_masks(), dtype=bool)
            if manager_mask.shape != (3,) or worker_mask.shape != (15,):
                raise RuntimeError(
                    "dual Worker mask 形状漂移:"
                    f"worker={worker_mask.shape},manager={manager_mask.shape}")

            w = self._win
            tau = (
                int(self.env._steps) - int(w["t0"])
                if w is not None else int(self._last_tau)
            )
            drank = bool(
                w is not None
                and (
                    int(w.get("voluntary_drinks", 0)) > 0
                    or int(w.get("drains", 0)) > 0
                )
            )
            dry_floor_remaining = 0.0
            if w is not None:
                dry_floor_remaining = min(
                    1.0,
                    max(
                        0.0,
                        (int(w.get("floor", 0)) - tau)
                        / max(1, REVISIT_FLOOR),
                    ),
                )

            calibration = getattr(self, "resource_calibration", None)
            farm_scene_denominator = (
                calibration.worker_farm_scene_denominator if calibration is not None
                else getattr(self, "farm_scene_cap", FARM_SCENE_CAP))
            context = np.asarray([
                min(1.0, max(0.0, self.layer_clock / KILL_PATIENCE)),
                1.0 if self.exhausted else 0.0,
                min(
                    1.0,
                    max(
                        0.0,
                        self.farm_scene_steps / max(1, farm_scene_denominator),
                    ),
                ),
                min(
                    1.0,
                    max(
                        0.0,
                        1.0 - self.env._steps / max(1, self.max_steps),
                    ),
                ),
                min(
                    1.0,
                    max(
                        0.0,
                        (
                            int(self.env._ep_kills)
                            - int(self._layer_kills0)
                        ) / 50.0,
                    ),
                ),
                min(
                    1.0,
                    max(
                        0.0,
                        (
                            int(self.env._steps)
                            - int(self._legacy_layer_steps0)
                        ) / 1500.0,
                    ),
                ),
                dry_floor_remaining,
                1.0 if drank else 0.0,
                p_skip,
            ], dtype=np.float32)
            if context.shape != (9,):
                raise RuntimeError(
                    f"dual Worker context 形状漂移:{context.shape} != (9,)")

            # ``_fuse`` counts repeats after the first matching request and
            # trips at 25.  A matching signature with counter zero is already
            # armed and one request closer to the trip than ``_fuse_sig=None``;
            # v1 encoded both as zero.  Publish (counter+1)/25 while armed so
            # the observation is Markov at that boundary.  Merely remembering
            # a previous action is not evidence of a current streak:
            # movement/entity/item changes invalidate it.
            fuse_streak = np.zeros(15, dtype=np.float32)
            fuse_signature = self._fuse_sig
            if (
                isinstance(fuse_signature, tuple)
                and fuse_signature
                and isinstance(fuse_signature[0], (int, np.integer))
                and not isinstance(fuse_signature[0], (bool, np.bool_))
            ):
                fuse_action = int(fuse_signature[0])
                if (
                    0 <= fuse_action < 15
                    and self._sig(fuse_action, raw) == fuse_signature
                ):
                    fuse_counter = int(self._fuse)
                    if not 0 <= fuse_counter < 25:
                        raise RuntimeError(
                            "dual Worker fuse counter 必须在 [0,24]:"
                            f"{fuse_counter}")
                    fuse_streak[fuse_action] = (
                        fuse_counter + 1) / 25.0

            result = np.concatenate([
                legacy_result,
                current_base,
                context,
                fuse_streak,
                worker_mask.astype(np.float32),
                manager_mask.astype(np.float32),
                controller_snapshot,
            ]).astype(np.float32, copy=False)
            expected_dim = DUAL_WORKER_OBSERVATION_DIM
            if view == WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE:
                # R13:v4 前缀已在上方完整拼装(逐位同 v4),此处仅追加。
                result = np.concatenate([
                    result,
                    self._worker_v5_window_appendix(w, tau, raw),
                ]).astype(np.float32, copy=False)
                expected_dim += DUAL_WORKER_V5_APPENDIX_DIM
            if (
                result.shape != (expected_dim,)
                or not np.isfinite(result).all()
            ):
                raise RuntimeError(
                    "dual Worker policy observation 形状/有限性漂移:"
                    f"shape={result.shape},finite={np.isfinite(result).all()}")
            return result

        result = legacy_result.copy()
        if view == WORKER_OBSERVATION_VIEW_A12_OVERLAY:
            # All other base fields remain exact v3.  In particular features
            # 8/9 and the entity/map/item channels must not retain v4 filtering:
            # that information loss cannot be undone inside the policy.
            # Main ticks must use the exact v3 heal classification (which
            # historically included Healing scrolls); otherwise flooring in
            # the inherited actor cannot recover its training input.  The v4
            # free-slot count occupies the reversible sub-tick.  Actual
            # drinkability remains enforced by the native v4 action mask.
            free_slots = min(
                8, max(0, int(self.env._raw.get("belt_free_slots", 0))))
            result[286] = legacy_base[286] + free_slots / 128.0
            result[WORKER_DRINK_LATCH_FEATURE] = current[
                WORKER_DRINK_LATCH_FEATURE]
        else:
            result[WORKER_DRINK_LATCH_FEATURE] = (
                1.0 if self._legacy_exhausted else 0.0)
        return result

    def _worker_masks_and_distance(
        self,
    ) -> tuple[np.ndarray, int | None]:
        if getattr(self, "resource_protocol", "off") != "off":
            from .resource_protocol import progression_allowed
            self.env._resource_dive_authority = bool(
                self.dive_live_sovereignty and (self._win or {}).get("opt") == DIVE
                and progression_allowed(self.env._raw))
        m, nearest = self._controller_action_context()
        m = np.array(m, dtype=bool)
        raw = self.env._raw
        # R13 主权移交:live DIVE 窗内工人获得 a11 与踏格职权(默认关,
        # 旧法逐位不变)。FARM/RESUPPLY 窗以及旗关路径维持恒掩——
        # 临战下楼规避死亡成本的逃跑漏洞只在「下楼本就是本窗任务」的
        # DIVE 窗内不成立。
        _dive_live_window = (
            getattr(self, "dive_live_sovereignty", False)
            and (self._win or {}).get("opt") == DIVE
        )
        if not _dive_live_window:
            m[11] = False   # 主线推进归经理(DIVE 职权)
            # 1..8 也能直接踏上相邻 trigger/剧情站位。仅剥掉换层奖金无法
            # 封权：临战下楼可规避巨额死亡成本，Worker 会因此学会逃跑。
            for action in DiabloGymEnv._protected_walk_actions(raw):
                m[action] = False
        # action10 会优先处理剧情白名单；只要目标存在就必须掩掉，哪怕
        # 近敌尚需 FARM 清理，也不能让学习工人越过 manager DIVE。
        m[10] = m[10] and not bool(
            self.env._raw.get("progression_targets"))
        if bool((self._win or {}).get("gear_grace_pending", False)):
            # Two legal actions are essential: masking to a14 alone would make
            # its probability exactly one and produce zero policy gradient.
            # a0 is an explicit, costly decline; a14 is the exact native
            # upgrade receipt the Worker must learn to choose.
            grace = np.zeros_like(m)
            grace[0] = True
            grace[14] = bool(m[14])
            return grace, nearest
        if self.drink_sovereignty:
            # 主权只在当前可见的安全包络内开放。用整数交叉乘避免
            # 0.5/0.75 边界因浮点舍入在采集与部署间翻转。成功的主动饮
            # 或反射饮都会置可见闩，并关闭本窗后续 worker-owned a12；
            # 紧急 _drain() 自身仍可继续排水。
            hp = int(raw.get("hp", 0))
            max_hp = max(1, int(raw.get("max_hp", 0)))
            in_safe_envelope = (
                2 * hp >= max_hp
                and 4 * hp < 3 * max_hp
            )
            # A reflex drain is just as much a consumed potion as a voluntary
            # one.  The old latch counted only voluntary presses, so the
            # worker could immediately spend another bottle after a successful
            # brainstem rescue.  Keep emergency _drain() unrestricted (it
            # bypasses worker masks), but allow at most one worker-owned drink
            # before this option window returns control to the manager.
            already_drank = (
                int((self._win or {}).get("voluntary_drinks", 0)) > 0
                or int((self._win or {}).get("drains", 0)) > 0
            )
            m[12] = (
                m[12]
                and raw.get("belt_heals", 0) > 0
                and in_safe_envelope
                and not already_drank
            )
        else:
            m[12] = False   # 喝药归脑干(v32 前旧协议;反射仍兜底)
        return m, nearest

    def _worker_masks(self) -> np.ndarray:
        return self._worker_masks_and_distance()[0]

    def _validate_worker_action12_contract(
            self, worker, declared_view: str) -> None:
        """Keep a dual observation's embedded mask equal to deployment.

        Historical 298-wide policies do not embed their action mask.  The
        asymmetric lossless dual view does, so allowing a callback to apply a
        different action-12 contract after the observation is built would
        present the actor with a state that contradicts its actual legal
        actions.
        """
        expected_mode = (
            WORKER_ACTION12_ENVIRONMENT_MASK
            if self.drink_sovereignty
            else WORKER_ACTION12_PERMANENTLY_MASKED
        )
        declared_mode = getattr(
            worker, "diablogym_worker_action12_mode", None)
        if declared_mode is None:
            if declared_view == WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC:
                raise RuntimeError(
                    "dual Worker callback 缺少 action12 deployment contract")
            return
        if declared_mode != expected_mode:
            raise RuntimeError(
                "Worker action12 contract 与 OptionsEnv "
                "drink_sovereignty 不一致:"
                f"callback={declared_mode!r},environment={expected_mode!r}")

    def step(self, option: int):
        if self._win is not None:
            raise RuntimeError("上一个选项窗口尚未收束")
        self.env._ensure_active()
        self._win_begin(option)
        mode = self._win["mode"]
        worker = self._workers.get(int(option))
        ending = self._consume_fuse_recovery()
        retreat = getattr(self, "retreat_service", None)
        portal = getattr(self, "portal_service", None)
        # R18-F portal-v1 extends the two-rung chain to three; the town service
        # is still the default owner of a RESUPPLY window.
        service = (portal if portal is not None and portal.active
                   else retreat if retreat is not None and retreat.active
                   else self.resource_service)
        if (getattr(self, "resource_protocol", "off") != "off"
                and int(option) == RESUPPLY and service.active):
            while ending is None or ending.reason is None:
                if (portal is not None and service is portal and not portal.active
                        and self.resource_service.active):
                    # R18-F: a witch leg run inside a town trip has completed; the
                    # town service resumes its own (return) phase.
                    service = self.resource_service
                if (getattr(self, "resource_readiness_law", "veto-v1") == "coach-v03"
                        and not getattr(service, "settle_exempt", False)):
                    # R17.1 ruling 3 (T0′ crash, seed 2133003): a deadline-truncated
                    # service walk can leave the player mid-tile; the service's next
                    # decision — and, at completion, the worker's first decision —
                    # must start from a decision-idle player (frozen worker law).
                    # Settle through the audited script-owned wait command
                    # (bounded).  Inert under veto-v1, so R18–R23 receipts stand.
                    settle = 0
                    while (not self.env._decision_idle(self.env._raw)
                           and not self.env._raw.get("dead") and settle < 40
                           and (ending is None or ending.reason is None)):
                        self._resource_command = ("wait",)
                        try:
                            ending = self._win_beat(0)
                        finally:
                            self._resource_command = None
                        service.record_steps(
                            self.env._steps, service.phase)
                        settle += 1
                    if ending is not None and ending.reason is not None:
                        break
                if (portal is not None and not portal.active and not portal.awaiting_return
                        and service is self.resource_service and service.active
                        and getattr(service, "phase", None) == "return"
                        and int(self.env._raw["dungeon_level"]) == 0
                        and portal.trigger_reason(self.env._raw, int(self.env._steps)) == "buy_scroll"):
                    # R18-F portal-v1: the witch leg runs INSIDE the town trip, after the
                    # ordinary itinerary (heal / armor / potions / sell) and before the walk
                    # back, so the scroll is bought last with the belt already full. The
                    # town service never hands back while in town, so this is the only
                    # moment the purchase can happen. Decided BEFORE the town service is
                    # asked for a command (a discarded command would leave its pending
                    # request without a receipt). The portal script owns the next
                    # commands; the town service resumes its return phase afterwards.
                    portal.start(self.env._raw, int(self.env._steps), bridge, "buy_scroll")
                    service = portal
                synthesized = False
                if (portal is not None and portal.awaiting_return
                        and service is self.resource_service and service.active
                        and getattr(service, "phase", None) == "return"
                        and int(self.env._raw["dungeon_level"]) == 0):
                    # R18-F portal-v1. ResourceService's return phase walks to the
                    # cathedral stairs and completes only on L1, which would throw away
                    # the depth the scroll was bought to keep. Close the town service
                    # here (its itinerary is finished by construction at this phase),
                    # decided BEFORE asking it for a command so no pending request is
                    # left without a receipt; the next window's manager law hands the
                    # return leg to the portal script.
                    service.active = False
                    service.phase = "done"
                    service.reason = "portal_return_pending"
                    bridge.configure_town_service(False)
                    command = ("complete",)
                    synthesized = True
                else:
                    command = service.command(self.env, bridge)
                self.env._resource_service_deadline = (
                    service.start_steps + service.service_microstep_cap)
                if service is self.resource_service and getattr(self, "resource_service_policy", "legacy-v1") in ("sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6", "sustain-loot-v1"):
                    self.env._resource_service_deadline = self.resource_service.command_microstep_deadline
                self._resource_command = command
                try:
                    ending = self._win_beat(0)
                finally:
                    self._resource_command = None
                receipt = self._win.get("last_info", {}).get("resource_action_audit", {})
                if not synthesized:
                    service.receipt(command, receipt)
                if (portal is not None and service is portal and command[0] == "buy"
                        and receipt.get("accepted") and self.resource_service.active):
                    # R18-F: the scroll was bought inside the town trip; book it in the
                    # town service ledger so the trip cash reconciliation stays exact.
                    self.resource_service.purchases += 1
                    self.resource_service.gold_spent += int(receipt.get("price", 0) or 0)
                service.record_steps(self.env._steps, service.phase)
            extra, base_info, done, trunc = self._win_end(ending.reason)
            self._finish_loot_episode(done, trunc, base_info)
            extra["resource"] = self.resource_service.telemetry()
            if retreat is not None:
                extra["retreat"] = retreat.telemetry()
            if portal is not None:
                extra["portal"] = portal.telemetry()
            info = dict(base_info)
            info["option_extra"] = extra
            return self._mgr_obs(self._last_base_obs), extra["R"], done, trunc, info
        if worker is not None:
            if ending is None or ending.reason is None:
                ending = self._drain()  # 工人首个观测必须是无反射态
            while ending is None:
                declared_view = getattr(
                    worker,
                    "diablogym_worker_observation_view",
                    self.worker_observation_view,
                )
                self._validate_worker_action12_contract(
                    worker, declared_view)
                a = worker(
                    self._worker_policy_observation(declared_view),
                    self._worker_masks(),
                )
                outcome = self._win_step_worker(a)
                if outcome.reason is not None:
                    ending = BeatOutcome(
                        reason=outcome.reason,
                        requested_action=outcome.requested_action,
                        executed_action=outcome.executed_action,
                        fuse_tripped=outcome.fuse_tripped,
                        action14_audit=outcome.action14_audit,
                        action_effect_audit=outcome.action_effect_audit,
                    )
        else:
            while ending is None or ending.reason is None:
                raw = self.env._raw
                action_mask, nearest = self._controller_action_context()
                action_mask = np.asarray(action_mask, dtype=bool)
                grace_decision = bool(
                    self._win.get("gear_grace_pending", False))
                if grace_decision:
                    # Scripted Options evaluation consumes the same single
                    # opportunity, without charging it to Worker counters.
                    self._win["gear_grace_pending"] = False
                    self._win["gear_grace_consumed"] = False
                    self._win["gear_grace_decisions"] += 1
                    a = 14 if bool(action_mask[14]) else 0
                    if a == 0:
                        self._win["gear_grace_consumed"] = True
                else:
                    a = 12 if _reflex(raw) else dispatch(
                        mode,
                        raw,
                        bool(action_mask[14]),
                        action_mask=action_mask,
                        nearest_engageable_distance=nearest,
                    )
                ending = self._win_beat(a)
                if grace_decision and ending.fuse_tripped:
                    self._win["gear_grace_pending"] = True
                    self._win["gear_grace_consumed"] = False
                    self._win["gear_grace_decisions"] -= 1
        extra, base_info, done, trunc = self._win_end(ending.reason)
        self._finish_loot_episode(done, trunc, base_info)
        info = dict(base_info)
        if getattr(self, "resource_protocol", "off") != "off":
            extra["resource"] = self.resource_service.telemetry()
            retreat = getattr(self, "retreat_service", None)
            if retreat is not None:
                extra["retreat"] = retreat.telemetry()
            portal = getattr(self, "portal_service", None)
            if portal is not None:
                extra["portal"] = portal.telemetry()
        info["option_extra"] = extra
        return self._mgr_obs(self._last_base_obs), extra["R"], done, trunc, info

    def _finish_loot_episode(self, done, trunc, base_info):
        if (getattr(self, "resource_service_policy", "legacy-v1") != "sustain-loot-v1"
                or not (done or trunc)):
            return
        raw = self.env._raw
        reason = (getattr(self.env, "_resource_terminal_reason", None)
                  or base_info.get("terminal_reason")
                  or ("death" if raw.get("dead") or int(raw.get("hp", 0)) <= 0
                      else "episode_terminated" if done else "episode_truncated"))
        self.resource_service.finish_episode(raw, int(self.env._steps), str(reason))

    def _mgr_obs(self, base_obs) -> np.ndarray:
        view = getattr(
            self, "manager_observation_view",
            MANAGER_OBSERVATION_VIEW_RAW_V4)
        if view not in MANAGER_OBSERVATION_VIEWS:
            raise RuntimeError(
                f"manager observation view 未注册:{view!r}")
        if view == MANAGER_OBSERVATION_VIEW_LEGACY_V3:
            # M29 was trained on the same protocol-v3 base as V28.  Restoring
            # only belt feature 286 is insufficient: v4 also filtered monster
            # count/nearest/slots, the 121-cell monster map, and item slots.
            # Rebuild before information is lost at the vector boundary.
            rebuild_legacy = getattr(
                self.env, "_legacy_policy_vectorize", None)
            raw = getattr(self.env, "_raw", None)
            if not callable(rebuild_legacy) or raw is None:
                raise RuntimeError(
                    "legacy-v3 manager view requires a lossless native raw "
                    "record and legacy vectorizer")
            manager_base = rebuild_legacy(raw)
        else:
            manager_base = np.asarray(base_obs, dtype=np.float32).copy()
        if manager_base.shape != (295,):
            raise ValueError(
                f"manager 基础观测必须为 (295,)，收到 {manager_base.shape}")
        if view == MANAGER_OBSERVATION_VIEW_LEGACY_V3:
            # The vectorizer already used native legacy_belt_heals.  Keep this
            # assertion local so no future refactor can reintroduce a packed
            # v4 sub-tick into M29's feature 286.
            expected_belt = min(
                8, max(0, int(self.env._raw["legacy_belt_heals"]))) / 8.0
            if not np.isclose(
                    manager_base[286], expected_belt, rtol=0.0, atol=1e-7):
                raise RuntimeError(
                    "legacy-v3 manager belt feature was not reconstructed "
                    "from native legacy_belt_heals")
        one_hot = [0.0, 0.0, 0.0]
        if self._last_opt >= 0:
            one_hot[self._last_opt] = 1.0
        if view == MANAGER_OBSERVATION_VIEW_LEGACY_V3:
            clock = self._legacy_layer_clock
            layer_steps0 = self._legacy_layer_steps0
        else:
            clock = self.layer_clock
            layer_steps0 = self._layer_steps0
        extra = np.asarray([
            max(0.0, 1.0 - self.env._steps / max(1, self.max_steps)), # 余时
            min(1.0, clock / KILL_PATIENCE),
            min(1.0, (self.env._ep_kills - self._layer_kills0) / 50.0),
            min(1.0, (self.env._steps - layer_steps0) / 1500.0),
            *one_hot,
            min(1.0, self._last_tau / TAU_CAP),
        ], dtype=np.float32)
        return np.concatenate([manager_base, extra])

    def close(self):
        self.env.close()


class StagnationClockWrapper(gym.Wrapper):
    """恶魔臂 F 专用:295+1=296 维平面包装。

    Clock progress uses the same factual signals as OptionsEnv.  Counting only
    kills taught the flat oracle to abandon a high-HP monster after 140 beats
    even while every attack was lowering its conserved HP floor.
    """

    def __init__(self, env: DiabloGymEnv):
        super().__init__(env)
        base = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(base + 1,), dtype=np.float32)
        self._clock = 0
        self._kills = 0
        self._scene = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._clock, self._kills = 0, 0
        self._scene = _scene_identity(self.env._raw)
        return self._obs(obs), info

    def action_masks(self):
        return self.env.action_masks()

    def step(self, action):
        steps_b = self.env._steps
        exploration_b = int(getattr(
            self.env, "exploration_progress", 0))
        gear_utility_b = gear_combat_utility_value(
            self.env._raw, "flat_clock_before")
        belt_free_b = DiabloGymEnv._belt_free_slots(self.env._raw)
        gold_b = int(self.env._raw.get("gold", 0))
        combat_floor_b = dict(getattr(
            self.env, "_combat_hp_floor", {}))
        obs, r, done, trunc, info = self.env.step(action)
        raw = self.env._raw
        combat_floor = getattr(self.env, "_combat_hp_floor", {})
        new_damage_floor = any(
            key in combat_floor
            and int(combat_floor[key][0]) < int(before[0])
            for key, before in combat_floor_b.items()
        )
        positive_progress = (
            int(getattr(self.env, "exploration_progress", 0))
            > exploration_b
            or new_damage_floor
            or gear_combat_utility_value(
                raw, "flat_clock_after") > gear_utility_b
            or DiabloGymEnv._belt_free_slots(raw) < belt_free_b
            or int(raw.get("gold", 0)) > gold_b
        )
        if _scene_identity(raw) != self._scene or self.env._ep_kills > self._kills:
            self._clock = 0
            self._scene = _scene_identity(raw)
            self._kills = self.env._ep_kills
        elif positive_progress:
            self._clock = 0
        else:
            self._clock += self.env._steps - steps_b
        return self._obs(obs), r, done, trunc, info

    def _obs(self, base_obs):
        return np.concatenate([np.asarray(base_obs, dtype=np.float32),
                               np.asarray([min(1.0, self._clock / KILL_PATIENCE)],
                                          dtype=np.float32)])
