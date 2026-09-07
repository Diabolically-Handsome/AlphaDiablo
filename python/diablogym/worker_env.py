"""v23 WorkerWindowEnv:FARM 操作脑的在位训练环境(docs/PREREG-v23.md)。

一个 Gym episode = 一个真实底层游戏，而不是一个 FARM 窗口。
  - reset():同局快进——经理(numpy 前向,argmax+经理掩码)逐窗决策,
    DIVE/RESUPPLY 窗口由脚本内环跑完(OptionsEnv.step 原路径,簿记全同源);
    遇 FARM 即开窗、排水反射,把首个无反射观测交给工人。基础局死/截断则
    滚入新局(含出生快进的天然代价)。跨窗口 wrapper 状态(停滞钟/榨干旗/
    保险丝/上选项)绝不清零——经理 303 维观测的状态机与 OptionsEnv 同一段代码。
  - step(a):工人一拍 + 反射尾部排水(共享窗口核 _win_step_worker)。
    FARM 自然收窗时，同一 transition 内继续推进冻结经理/脚本，直至下一个
    可学习 FARM；策略奖励等于当前 FARM 工资。中间快进窗口及下一窗开场
    反射的正收益只进 info/stats 审计，不能把脚本 DIVE 的战斗/下楼收益
    错发给最后一个工人动作；同一 transition 若在 manager/script 阶段
    真实死亡，可由显式 ``terminal-death-only`` 模式且只传回底层终局
    拍的死亡罚分（默认 ``none`` 精确保留旧回报）。
    自然窗口边界不再伪造 terminal，PPO 会对真实下一 FARM 状态 bootstrap。
    底层真实结束为 terminated；基础局在 idle 决策边界用尽 3000 步时，
    有进展的安全边界才 truncated。OptionsEnv 证明为 no-progress timeout
    的边界在 Worker 层升格为带死亡等价失败成本的 terminated。若上限恰好
    切在未编码进 298 维的走格/受击动画中，底层仍 fail-closed 为
    unsettled_budget_terminal，禁止 SB3 用别名 terminal_observation
    做错误的 TimeLimit bootstrap。
  - 训练种子:显式采样器永久拒采历史已烧 BC 池 [100,484)、
    [1000,1384) 及全部现行登记的 BC/评测池。已消费的 protocol-v7 BC 池
    [2100000,2100128)、[2101000,2101384) 永久保留在拒采表；当前 active
    v1/v2 池分别为 [2142000,2142128)、[2143000,2143384)。统一开发/终考
    种子银行为 [2110000,2130000)，与两代 BC 池构造性不交叠。
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import time
import hashlib
import io
import json
import pathlib
import re
from types import MappingProxyType
from typing import Mapping

import gymnasium as gym
import numpy as np

from .env import REWARD_ECONOMY_V1, terminal_death_reward_component
from .options_env import (
    DIVE,
    DUAL_WORKER_OBSERVATION_DIM,
    DUAL_WORKER_ACTION_MASK_SLICE,
    DUAL_WORKER_MANAGER_MASK_SLICE,
    DUAL_WORKER_V5_APPENDIX_DIM,
    FARM,
    KILL_PATIENCE,
    RESUPPLY,
    MANAGER_OBSERVATION_VIEW_LEGACY_V3,
    N_EXTRA_WORKER,
    WORKER_OBSERVATION_VIEW_A12_OVERLAY,
    WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
    WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE,
    WORKER_OBSERVATION_VIEW_LEGACY_V3,
    WORKER_OBSERVATION_VIEW_RAW_V4,
    WORKER_OBSERVATION_VIEWS,
    WORKER_ACTION12_ENVIRONMENT_MASK,
    WORKER_ACTION12_PERMANENTLY_MASKED,
    OptionsEnv,
)

# 单一真源：训练 CLI 的首局 seed/rank 与环境自动滚局都必须读这张表。
# 12000 池在 protocol-v4 代码冻结前预留，只能在训练完成后用于一次
# paired generalization evaluation；v7 的大块银行一次性预留未来已冻结
# dev/final 子池，避免每次扩池都改变 implementation identity。把任一块
# 留在训练采样路径中都会使“新鲜留出”主张失效。
EVAL_RESERVED_SEED_RANGES = (
    (7000, 7032),
    (9000, 9032),
    (12000, 12032),
    (2_110_000, 2_130_000),
)
# 这些早期 BC 池已经被历史生产者打开。它们不再授权给当前 bc-v1/v2
# producer，但必须永久留在普通训练拒采域，防止“旧 heldout”被重新采样
# 后又错误标为 fresh。BC_RESERVED_SEED_RANGES 还保留较新的 burned 池，
# 并登记当前 active 池；它是训练拒采表，不等同于 producer 授权表。
HISTORICAL_BURNED_BC_SEED_RANGES = (
    (100, 484),
    (1000, 1384),
)
BC_RESERVED_SEED_RANGES = (
    (2000, 2128),
    (3000, 3384),
    (2_100_000, 2_100_128),
    (2_101_000, 2_101_384),
    (2_102_000, 2_102_128),
    (2_103_000, 2_103_384),
    # 2026-07-27:2_102_000..127 于 WSL 首采时被 a14 保险丝回执缺失崩溃
    # 一次性烧毁(标记已落,纪律正确咬合);2_104_000..127 随后被候选门 FAIL
    # 一次性烧毁(A2 修正案缘起);v1 活动段推进至 2_106 段。
    (2_104_000, 2_104_128),
    (2_106_000, 2_106_128),
    (2_108_000, 2_108_128),
    # 2026-07-27 立法(批文「同意」):2_108/2_103 随 A3 实现束变更作废;
    # BC 立法块 2_140_000-2_158_xxx 保留(避开评测银行 2_11x 万段),
    # 活动对 v1=2_140 / v2=2_141,后续每对消费即追表。
    (2_140_000, 2_140_128),
    (2_141_000, 2_141_384),
    # 2026-07-27:2_140/2_141 随 A4 实现束变更作废,追表推进(立法块内)。
    (2_142_000, 2_142_128),
    (2_143_000, 2_143_384),
)
TRAIN_RESERVED_SEED_RANGES = (
    *HISTORICAL_BURNED_BC_SEED_RANGES,
    *BC_RESERVED_SEED_RANGES,
    *EVAL_RESERVED_SEED_RANGES,
)
_MAX_EMPTY_FARM_EPISODES = 8
_FAST_FORWARD_REWARD_CREDIT_MODES = frozenset({
    "none",
    "terminal-death-only",
})
# R13 教室改革:farm-only = 旧法(非 FARM 窗脚本快进);farm-dive-v1 =
# DIVE 窗 live 采集(主权移交预注册配方专用)。RESUPPLY/跳干窗恒脚本。
_LEARNING_WINDOW_SCOPES = frozenset({
    "farm-only",
    "farm-dive-v1",
    "earned-dive-suffix-v1",
})
_LIVE_DIVE_WINDOW_SCOPES = frozenset({
    "farm-dive-v1", "earned-dive-suffix-v1",
})
_PREFIX_WORKER_SHA256 = (
    "7e31dc5402caed733443abb9fef383c877d93b3623b199ba4b5c6293092592f0")
_MAX_PREFIX_ZERO_TICK_WINDOWS = 32


class PrefixBudgetExceeded(RuntimeError):
    """A lifetime collection limit was reached; no learner terminal is emitted."""

    def __init__(self, reason, audit=None):
        self.reason = str(reason)
        self.audit = copy.deepcopy(audit)
        message = self.reason
        if audit is not None:
            message += "\nEARNED_PREFIX_AUDIT=" + json.dumps(
                audit, ensure_ascii=False, sort_keys=True,
                default=lambda value: value.tolist() if isinstance(value, np.ndarray)
                else value.item() if isinstance(value, np.generic) else str(value))
        super().__init__(message)


# R15 战备门教练(主席批「就按您说的办」):readiness-v1 = 分层战备表
# 放行(短板逻辑,任一维不达标不下楼;榨干层为逃生口)。
_MANAGER_HEURISTICS = frozenset({
    "level-margin-1",
    "readiness-v1",
    # R15 修正案一:v1 的榨干逃生口被工人白嫖(故意不杀怪触发无进展
    # 榨干旗骗开下楼门,kills 35→11、死亡率 87%);v2 改为真实清场
    # 逃生——本层存活怪 ≤25% 才放行,通行证必须杀出来。
    "readiness-v2",
    # R16 修宪(审计簇 C1/C2):v3 = 战备表 v0.2(引擎算术反推)+ 击杀
    # 口径清场(花名册只含活怪,v2 判据是死代码)。
    "readiness-v3",
})
# R16 战备表 v0.2(审计裁定 C1:v0.1 的 L2 门 HP90⇔clvl4⇔8040XP 超过 L1
# 全清经验池;AC 列刻度错位——出生战士 AC 7 对 L1 门 12 只有 0.58)。
# 口径:第 d 层门槛 = 榨取良好的工人在第 d−1 层能达到的水平(L1-L4 按
# 实测/引擎算术;L5+ 沿 v0.1 递推,待深层数据再校准)。HP 列删除
# (3体:2力 自动加点下 HP 是 clvl 的确定函数,冗余)。
_READINESS_V2_CLVL = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 19)
_READINESS_V2_AC = (7, 9, 11, 13, 16, 19, 22, 26, 30, 35, 40, 45, 50, 55, 60, 60)
_READINESS_V2_DMG = (5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 26)
_READINESS_V2_FIRE_RESIST = {13: 30, 14: 30, 15: 50, 16: 50}
_GOLEM_TYPE = 109   # 花名册中 4 个 golem 占位槽(mlvl12/exp0/hp1,永不死)
# R17.0 仪表(合议庭更正二.3):托管战备门逐次过门遥测的有界上限
# (每 env;训练臂 ~244 次下楼,4096 足够;满后只记不裁地停止追加)。
_DESCEND_ESCROW_GATE_LOG_CAP = 4096


def readiness_power_ratio_v2(raw, target_dlvl) -> float:
    """R16 v0.2 战斗力比值(三维短板 + 深层火抗)。≥1.0 放行。"""
    d = max(1, min(16, int(target_dlvl)))
    i = d - 1
    ratios = [
        float(raw.get("char_level", 0)) / _READINESS_V2_CLVL[i],
        float(raw.get("armor_class", 0)) / _READINESS_V2_AC[i],
        (float(raw.get("item_max_damage", 0))
         + float(raw.get("damage_mod", 0))) / _READINESS_V2_DMG[i],
    ]
    fire_gate = _READINESS_V2_FIRE_RESIST.get(d)
    if fire_gate is not None:
        ratios.append(float(raw.get("fire_resist", 0)) / fire_gate)
    return min(ratios)


def roster_alive_count(raw) -> int:
    """本层花名册存活怪数(剔除 golem 占位槽)。花名册只含活怪。"""
    return sum(
        1 for m in (raw.get("monsters") or [])
        if int(m.get("hp", 0)) > 0 and int(m.get("type", -1)) != _GOLEM_TYPE)


def readiness_clear_ratio(raw, kills_at_floor_entry) -> float:
    """R16 击杀口径清场比 = 本层击杀 / (本层击杀 + 存活非 golem)。

    花名册在桥侧导出时已剔除死者(C2),故用 monster_kill_total 相对
    进层基数计本层击杀;idle 无法抬高该比值,通行证必须杀出来。
    """
    kills_here = max(
        0, int(raw.get("monster_kill_total", 0)) - int(kills_at_floor_entry))
    alive = roster_alive_count(raw)
    denom = kills_here + alive
    return (kills_here / denom) if denom > 0 else 1.0


def readiness_floor_cleared(raw, fraction: float = 0.25) -> bool:
    """R15 v2 逃生判据:本层存活怪占比 ≤ fraction(容忍锁门房/不可达残怪)。

    monsters 为整层花名册(死者保留 hp≤0),idle 白嫖无法改变占比——
    唯一降占比的方式就是真的去杀。花名册缺失时保守返回 False。
    """
    monsters = raw.get("monsters") or []
    total = len(monsters)
    if total <= 0:
        return False
    alive = sum(1 for m in monsters if int(m.get("hp", 0)) > 0)
    return alive <= max(2.0, fraction * total)
# R15《分层战备表》v0.1 下界(r15-READINESS-TABLE-draft.md,三十年攻略
# 考古交叉:Jarulf 机制圣经 + monstdat 源码 + 社区配置指南)。
# index = dlvl−1(目标层 1..16)。
_READINESS_CLVL = (1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 22)
_READINESS_HP = (70, 90, 100, 120, 140, 150, 160, 180,
                 200, 210, 220, 240, 260, 280, 300, 350)
_READINESS_AC = (12, 15, 30, 35, 40, 45, 50, 55,
                 60, 65, 70, 80, 90, 100, 110, 110)
_READINESS_DMG = (5, 8, 10, 12, 14, 16, 17, 19,
                  22, 26, 25, 27, 29, 31, 33, 33)
_READINESS_FIRE_RESIST = {13: 50, 14: 50, 15: 75, 16: 75}
_READINESS_MAGIC_RESIST = {15: 50, 16: 50}


def readiness_power_ratio(raw, target_dlvl) -> float:
    """R15 战斗力比值:min(四维达标比,深层另并入抗性比)。

    ≥1.0 = 战备达标可放行。短板逻辑:HP 够而伤害不够=打不出硬直被围殴,
    伤害够而 HP 不够=一波带走——任一短板都应拦下。字段名经 live 探针
    实证(probe_r15_raw):char_level/max_hp/armor_class/
    item_max_damage+damage_mod/fire_resist/magic_resist。
    """
    d = max(1, min(16, int(target_dlvl)))
    i = d - 1
    ratios = [
        float(raw.get("char_level", 0)) / _READINESS_CLVL[i],
        float(raw.get("max_hp", 0)) / _READINESS_HP[i],
        float(raw.get("armor_class", 0)) / _READINESS_AC[i],
        (float(raw.get("item_max_damage", 0))
         + float(raw.get("damage_mod", 0))) / _READINESS_DMG[i],
    ]
    fire_gate = _READINESS_FIRE_RESIST.get(d)
    if fire_gate is not None:
        ratios.append(float(raw.get("fire_resist", 0)) / fire_gate)
    magic_gate = _READINESS_MAGIC_RESIST.get(d)
    if magic_gate is not None:
        ratios.append(float(raw.get("magic_resist", 0)) / magic_gate)
    return min(ratios)
_SEED_SCOPES = frozenset({"train", "bc-v1", "bc-v2", "replay"})
_CURRENT_BC_V1_RANGE = (2_142_000, 2_142_128)
_CURRENT_BC_V2_RANGE = (2_143_000, 2_143_384)
_LEGACY_BELT_FEATURE = 286
_LEGACY_EXHAUSTED_FEATURE = 297

# Worker NPZ 是部署协议，不只是六块无语义的矩阵。经理 NPZ 仍保持历史
# 六成员格式；298→15 的 Worker 必须另外携带这一个 canonical JSON 标量。
WORKER_NPZ_CONTRACT_MEMBER = "worker_contract_json"
WORKER_NPZ_SCHEMA = "diablogym-worker-npz/1"
WORKER_NPZ_ROLE = "worker"
WORKER_NPZ_REPRESENTATION = "plain-maskable-mlp-argmax"
_WORKER_WEIGHT_MEMBERS = frozenset({"w0", "b0", "w1", "b1", "wa", "ba"})
_WORKER_CONTRACT_FIELDS = frozenset({
    "schema",
    "role",
    "representation",
    "observation_view",
    "action12_mode",
    "source_checkpoint_sha256",
    "source_training_contract_sha256",
})
_WORKER_OBSERVATION_VIEWS = frozenset({
    WORKER_OBSERVATION_VIEW_LEGACY_V3,
    WORKER_OBSERVATION_VIEW_RAW_V4,
})
_WORKER_ACTION12_MODES = frozenset({
    WORKER_ACTION12_PERMANENTLY_MASKED,
    WORKER_ACTION12_ENVIRONMENT_MASK,
})
_LOWER_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_sha256(value: object, field: str, *, nullable: bool) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or _LOWER_SHA256_RE.fullmatch(value) is None:
        suffix = "或 null" if nullable else ""
        raise ValueError(f"Worker NPZ metadata.{field} 必须是 64 位小写 SHA256{suffix}")


def validate_worker_npz_contract(contract: object) -> dict:
    """Validate and copy the complete Worker deployment contract.

    Observation selection happens before this network API is called.  In
    particular, ``legacy-v3`` means a complete protocol-v3 vector produced
    at the environment/raw-state boundary; it never claims that a v4
    298-vector can be losslessly decoded back into v3.
    """
    if not isinstance(contract, dict):
        raise ValueError("Worker NPZ metadata 必须是 JSON object")
    fields = set(contract)
    if fields != _WORKER_CONTRACT_FIELDS:
        missing = sorted(_WORKER_CONTRACT_FIELDS.difference(fields))
        extra = sorted(fields.difference(_WORKER_CONTRACT_FIELDS))
        raise ValueError(
            "Worker NPZ metadata 字段必须精确匹配 schema；"
            f"missing={missing}, extra={extra}")
    checked = dict(contract)
    for field, expected in (
        ("schema", WORKER_NPZ_SCHEMA),
        ("role", WORKER_NPZ_ROLE),
        ("representation", WORKER_NPZ_REPRESENTATION),
    ):
        if checked[field] != expected:
            raise ValueError(
                f"Worker NPZ metadata.{field} 必须为 {expected!r}，"
                f"收到 {checked[field]!r}")
    view = checked["observation_view"]
    if not isinstance(view, str) or view not in _WORKER_OBSERVATION_VIEWS:
        raise ValueError(
            "Worker NPZ metadata.observation_view 必须是 "
            f"{sorted(_WORKER_OBSERVATION_VIEWS)} 之一")
    action12_mode = checked["action12_mode"]
    if (not isinstance(action12_mode, str)
            or action12_mode not in _WORKER_ACTION12_MODES):
        raise ValueError(
            "Worker NPZ metadata.action12_mode 必须是 "
            f"{sorted(_WORKER_ACTION12_MODES)} 之一")
    _validate_sha256(
        checked["source_checkpoint_sha256"],
        "source_checkpoint_sha256",
        nullable=False,
    )
    _validate_sha256(
        checked["source_training_contract_sha256"],
        "source_training_contract_sha256",
        nullable=True,
    )
    return checked


def make_worker_npz_contract(
        *,
        observation_view: str,
        action12_mode: str,
        source_checkpoint_sha256: str,
        source_training_contract_sha256: str | None) -> dict:
    """Build the sole accepted Worker NPZ metadata schema."""
    return validate_worker_npz_contract({
        "schema": WORKER_NPZ_SCHEMA,
        "role": WORKER_NPZ_ROLE,
        "representation": WORKER_NPZ_REPRESENTATION,
        "observation_view": observation_view,
        "action12_mode": action12_mode,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_training_contract_sha256": source_training_contract_sha256,
    })


def canonical_worker_npz_contract_json(contract: object) -> str:
    """Return the only byte-level JSON spelling accepted inside a Worker NPZ."""
    return _canonical_json_text(validate_worker_npz_contract(contract))


def _parse_worker_npz_contract(value: np.ndarray, path: pathlib.Path) -> dict:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(
            f"{path} 的 {WORKER_NPZ_CONTRACT_MEMBER} 必须是单个 JSON 字符串标量")
    scalar = array.item()
    if isinstance(scalar, bytes):
        try:
            encoded = scalar.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{path} 的 {WORKER_NPZ_CONTRACT_MEMBER} 不是 UTF-8") from exc
    elif isinstance(scalar, str):
        encoded = scalar
    else:
        raise ValueError(
            f"{path} 的 {WORKER_NPZ_CONTRACT_MEMBER} 必须是 JSON 字符串")

    def reject_duplicate_keys(pairs):
        document = {}
        for key, item in pairs:
            if key in document:
                raise ValueError(f"Worker NPZ metadata 含重复字段 {key!r}")
            document[key] = item
        return document

    def reject_nonfinite(token):
        raise ValueError(f"Worker NPZ metadata 禁止非有限 JSON 数字 {token}")

    try:
        document = json.loads(
            encoded,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(
            f"{path} 的 {WORKER_NPZ_CONTRACT_MEMBER} 不是严格 JSON") from exc
    checked = validate_worker_npz_contract(document)
    canonical = _canonical_json_text(checked)
    if encoded != canonical:
        raise ValueError(
            f"{path} 的 {WORKER_NPZ_CONTRACT_MEMBER} 必须使用 canonical JSON")
    return checked


def legacy_worker_policy_observation_view(observation: np.ndarray) -> np.ndarray:
    """Decode the reversible A12 overlay into a protocol-v3 Worker row.

    Formal Worker boundaries first reconstruct every non-overlay feature from
    lossless native state.  This array-only helper therefore decodes only the
    two deliberately reversible overlay fields: packed belt slots at 286 and
    the signed drink latch at 297.  It must not be used to claim that an
    arbitrary filtered raw-v4 vector can recover the old monster/item channels.
    Arrays that predate this Worker contract are returned unchanged, matching
    ``leashed_ppo._legacy_worker_observation_view``.
    """
    value = np.asarray(observation)
    if value.ndim == 0 or value.shape[-1] <= _LEGACY_EXHAUSTED_FEATURE:
        return value
    legacy = value.copy()
    packed_belt = legacy[..., _LEGACY_BELT_FEATURE]
    legacy[..., _LEGACY_BELT_FEATURE] = (
        np.floor(np.clip(
            packed_belt * np.float32(8.0),
            np.float32(0.0),
            np.float32(8.0),
        ))
        / np.float32(8.0)
    )
    encoded_exhausted = legacy[..., _LEGACY_EXHAUSTED_FEATURE]
    legacy[..., _LEGACY_EXHAUSTED_FEATURE] = np.where(
        encoded_exhausted < np.float32(0.0),
        -encoded_exhausted - np.float32(1.0),
        encoded_exhausted,
    )
    return legacy

# E1 ⑤A(PREREG-内容案):p 抽签专用流之固定偏移——承 dry-anchor rng(26) 先例取 26,
# 加 2**33 移出 [0, 2**32) 训练种子域,杜绝专用流与任何 env 训练 RNG 同种子。
_P_SKIP_SEED_OFFSET = 2**33 + 26


@dataclass(frozen=True)
class _AdvanceOutcome:
    """冻结 manager 从窗口边界推进到下一学习态的完整结果。"""

    obs: np.ndarray | None
    reward: float
    terminated: bool
    truncated: bool
    extras: tuple[dict, ...]
    opening_reward: float = 0.0
    terminal_base_info: dict | None = None
    opening_recovery_action: int | None = None
    # 只含真实终局底层拍的死亡罚分，不含同拍 XP/伤害/走位等奖励，也不含
    # 此前 manager/script 窗口的任何回报。非死亡终局恒为 0。
    terminal_death_reward: float = 0.0
    # OptionsEnv 已证明该预算边界在 KILL_PATIENCE 内没有进展。Worker 把
    # 这类 safe base TimeLimit 升格为不可 bootstrap 的策略失败终止。
    timeout_without_progress: bool = False


def _coerce_p_skip(value) -> float:
    """E1 ⑤A:skip_dry 布尔升格为跳过概率 p_skip(True≡1.0,False≡0.0,向后兼容)。"""
    p = float(value)
    if not (np.isfinite(p) and 0.0 <= p <= 1.0):
        raise ValueError(f"skip_dry/p_skip 必须在 [0, 1] 内，收到 {value!r}")
    return p


def _coerce_fast_forward_reward_credit(value) -> str:
    """只接受具名信用模式，避免 bool/拼写错误静默改变训练目标。"""
    if not isinstance(value, str) or value not in _FAST_FORWARD_REWARD_CREDIT_MODES:
        raise ValueError(
            "fast_forward_reward_credit 必须是 "
            f"{sorted(_FAST_FORWARD_REWARD_CREDIT_MODES)} 之一，收到 {value!r}")
    return value


def _coerce_learning_window_scope(value) -> str:
    """R13 教室改革:只接受具名课堂范围,默认 farm-only 逐位复现旧法。"""
    if not isinstance(value, str) or value not in _LEARNING_WINDOW_SCOPES:
        raise ValueError(
            "learning_window_scope 必须是 "
            f"{sorted(_LEARNING_WINDOW_SCOPES)} 之一，收到 {value!r}")
    return value


def _coerce_depth_shaping_unit(value) -> float:
    """R13.2 甲案:势函数深度塑形单位(φ=unit×(dungeon_level−1))。

    仅训练期工人 policy_reward 层生效;0.0(默认)= 旧法零触碰。
    """
    unit = float(value)
    if not np.isfinite(unit) or not 0.0 <= unit <= 100.0:
        raise ValueError(
            f"depth_shaping_unit 必须是 [0,100] 内有限数，收到 {value!r}")
    return unit


def _coerce_additional_terminal_death_cost(value) -> float:
    """额外死亡成本以非负 cost 配置，内部统一转换成负 reward。"""
    cost = float(value)
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(
            "additional_terminal_death_cost 必须是有限非负数，收到 "
            f"{value!r}")
    return cost


def _coerce_seed_scope(value) -> str:
    """Make permission to consume held-out seeds explicit at construction."""
    if not isinstance(value, str) or value not in _SEED_SCOPES:
        raise ValueError(
            f"seed_scope 必须是 {sorted(_SEED_SCOPES)} 之一，收到 {value!r}")
    return value


def _seed_in_half_open_range(seed: int, registered: tuple[int, int]) -> bool:
    return registered[0] <= int(seed) < registered[1]


def _derive_p_skip_rng(seed: int) -> np.random.Generator:
    """E1③ 逐 env 播种:p 抽签专用流以固定偏移自 episode 种子确定性派生(零染训练 RNG)。"""
    return np.random.default_rng(int(seed) + _P_SKIP_SEED_OFFSET)


class NumpyManager:
    """冻结 v22-H 的 numpy 前向(MlpPolicy(64,64) 策略侧;G0' 与 SB3 逐位对账)。"""

    def __init__(self, npz_path: str | pathlib.Path,
                 expected_sha256: str | None = None):
        path = pathlib.Path(npz_path)
        payload = path.read_bytes()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
                raise ValueError("NumpyManager expected_sha256 必须是 64 位字符串")
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"{path} SHA256 不匹配: {actual_sha256} != {expected_sha256}")
        # Hash and parse the exact same immutable byte string.  Opening the
        # path once for validation and again for np.load would leave a replace
        # race in which subprocesses could silently consume different brains.
        self.source_sha256 = actual_sha256
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            files = tuple(z.files)
            if len(files) != len(set(files)):
                raise ValueError(f"{path} NPZ 含重复成员")
            members = set(files)
            missing = _WORKER_WEIGHT_MEMBERS.difference(members)
            if missing:
                raise ValueError(f"{path} 缺少权重: {sorted(missing)}")
            allowed = _WORKER_WEIGHT_MEMBERS | {WORKER_NPZ_CONTRACT_MEMBER}
            extra = members.difference(allowed)
            if extra:
                raise ValueError(f"{path} 含未登记 NPZ 成员: {sorted(extra)}")
            self._worker_contract = (
                _parse_worker_npz_contract(
                    z[WORKER_NPZ_CONTRACT_MEMBER], path)
                if WORKER_NPZ_CONTRACT_MEMBER in members
                else None
            )
            self.w0, self.b0 = z["w0"].astype(np.float32), z["b0"].astype(np.float32)
            self.w1, self.b1 = z["w1"].astype(np.float32), z["b1"].astype(np.float32)
            self.wa, self.ba = z["wa"].astype(np.float32), z["ba"].astype(np.float32)
        if (self.w0.ndim != 2 or self.b0.shape != (self.w0.shape[0],)
                or self.w1.ndim != 2 or self.w1.shape[1] != self.w0.shape[0]
                or self.b1.shape != (self.w1.shape[0],)
                or self.wa.ndim != 2 or self.wa.shape[1] != self.w1.shape[0]
                or self.ba.shape != (self.wa.shape[0],)):
            raise ValueError(f"{path} 权重形状不构成可连接的 MLP")
        if not all(np.isfinite(a).all()
                   for a in (self.w0, self.b0, self.w1, self.b1, self.wa, self.ba)):
            raise ValueError(f"{path} 权重包含 NaN/Inf")
        self._io_shape = (int(self.w0.shape[1]), int(self.wa.shape[0]))
        if self._worker_contract is not None and self._io_shape != (298, 15):
            raise ValueError(
                f"{path} 携 Worker metadata，但权重形状为 "
                f"{self._io_shape[0]}→{self._io_shape[1]}，必须为 298→15")
        if self._worker_contract is not None:
            self._worker_contract = MappingProxyType(self._worker_contract)

    def require_io_shape(self, observation_dim: int, action_count: int,
                         label: str = "NumpyManager") -> None:
        actual = self._io_shape
        expected = (int(observation_dim), int(action_count))
        if actual != expected:
            raise ValueError(
                f"{label} 输入→动作形状必须为 {expected[0]}→{expected[1]}，"
                f"实际 {actual[0]}→{actual[1]}")

    @property
    def worker_contract(self) -> Mapping | None:
        """Immutable deployment metadata, or ``None`` for historical NPZs."""
        return self._worker_contract

    def require_worker_contract(self) -> Mapping:
        if self._io_shape != (298, 15):
            raise ValueError(
                "Worker NPZ 权重形状必须为 298→15，"
                f"实际 {self._io_shape[0]}→{self._io_shape[1]}")
        if self._worker_contract is None:
            raise ValueError(
                "298→15 Worker NPZ 缺少严格 worker_contract_json metadata；"
                "旧六成员 NPZ 不再允许静默部署")
        return self._worker_contract

    @property
    def worker_observation_view(self) -> str:
        return str(self.require_worker_contract()["observation_view"])

    @property
    def worker_action12_mode(self) -> str:
        return str(self.require_worker_contract()["action12_mode"])

    def _raw_logits(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.w0.shape[1],):
            raise ValueError(f"观测形状应为 {(self.w0.shape[1],)}，收到 {obs.shape}")
        h = np.tanh(self.w0 @ obs + self.b0)
        h = np.tanh(self.w1 @ h + self.b1)
        logits = self.wa @ h + self.ba
        if not np.isfinite(logits).all():
            raise FloatingPointError("NumpyManager 前向产生 NaN/Inf logits")
        return logits

    def forensic_worker_logits(self, policy_obs: np.ndarray) -> np.ndarray:
        """Inspect an old contractless Worker only for parity/archaeology.

        No observation selection or deployment mask is applied.  Callers such
        as the historical BC state-dict adapter must separately declare the
        OptionsEnv view and apply its frozen mask contract; Worker NPZ
        deployment must use ``worker_logits``/``worker_callback`` instead.
        """
        actual = (int(self.w0.shape[1]), int(self.wa.shape[0]))
        if actual != (298, 15):
            raise ValueError(
                "forensic_worker_logits 只接受 298→15 Worker，"
                f"实际 {actual[0]}→{actual[1]}")
        return self._raw_logits(policy_obs)

    def _reject_generic_worker_inference(self) -> None:
        actual = getattr(
            self,
            "_io_shape",
            (int(self.w0.shape[1]), int(self.wa.shape[0])),
        )
        if actual == (298, 15):
            raise ValueError(
                "298→15 Worker 禁止调用通用 logits/choose："
                "请使用 worker_logits/choose_worker 并显式声明已提供的 observation_view")

    def logits(self, obs: np.ndarray) -> np.ndarray:
        self._reject_generic_worker_inference()
        return self._raw_logits(obs)

    def choose(self, obs: np.ndarray, mask: np.ndarray) -> int:
        lg = self.logits(obs)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != lg.shape:
            raise ValueError(f"掩码形状应为 {lg.shape}，收到 {mask.shape}")
        if not mask.any():
            raise ValueError("动作掩码不能全为 False")
        lg = np.where(mask, lg, -np.inf)
        return int(np.argmax(lg))

    def worker_logits(
            self,
            policy_obs: np.ndarray,
            *,
            observation_view: str) -> np.ndarray:
        """Forward an already-selected Worker policy observation.

        The explicit view label prevents callers from presenting a raw-v4
        vector to a legacy-v3 network (or vice versa).  This method performs
        no observation conversion because the complete v3 view must be built
        from the original raw game state at the environment boundary.
        """
        contract = self.require_worker_contract()
        expected_view = contract["observation_view"]
        if not isinstance(observation_view, str) or observation_view != expected_view:
            raise ValueError(
                "Worker policy observation_view 不匹配："
                f"NPZ 要求 {expected_view!r}，调用者声明 {observation_view!r}")
        return self._raw_logits(policy_obs)

    def worker_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        """Apply the NPZ's action-12 deployment contract without mutation."""
        contract = self.require_worker_contract()
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != (15,):
            raise ValueError(f"Worker 动作掩码形状应为 (15,)，收到 {mask.shape}")
        policy_mask = mask.copy()
        if contract["action12_mode"] == WORKER_ACTION12_PERMANENTLY_MASKED:
            policy_mask[12] = False
        if not policy_mask.any():
            raise ValueError("Worker 动作掩码应用部署合约后不能全为 False")
        return policy_mask

    def choose_worker(
            self,
            policy_obs: np.ndarray,
            raw_mask: np.ndarray,
            *,
            observation_view: str) -> int:
        logits = self.worker_logits(
            policy_obs,
            observation_view=observation_view,
        )
        policy_mask = self.worker_mask(raw_mask)
        return int(np.argmax(np.where(policy_mask, logits, -np.inf)))

    def worker_callback(self):
        """Return an OptionsEnv callback carrying its lossless view request."""
        view = self.worker_observation_view

        def policy(policy_obs: np.ndarray, raw_mask: np.ndarray) -> int:
            return self.choose_worker(
                policy_obs,
                raw_mask,
                observation_view=view,
            )

        # OptionsEnv reads this before constructing the callback's observation.
        # A plain lambda without this declaration would fall back to an
        # environment default and reintroduce a same-shape semantic mismatch.
        policy.diablogym_worker_observation_view = view
        policy.diablogym_worker_action12_mode = self.worker_action12_mode
        return policy


def is_reserved_eval_seed(seed: int) -> bool:
    """训练路径不得消费任何已登记评测池种子。"""
    value = int(seed)
    return any(lo <= value < hi for lo, hi in EVAL_RESERVED_SEED_RANGES)


def is_reserved_train_seed(seed: int) -> bool:
    """Whether an ordinary training reset would contaminate any held-out pool."""
    value = int(seed)
    return any(lo <= value < hi for lo, hi in TRAIN_RESERVED_SEED_RANGES)


def sample_train_seed(rng: np.random.Generator) -> int:
    """训练种子采样器:拒采历史已烧及全部现行登记的 BC/评测池。"""
    while True:
        s = int(rng.integers(0, 2**31))
        if not is_reserved_train_seed(s):
            return s


class WorkerWindowEnv(gym.Env):
    """SB3 视角:显式 policy view,Discrete(15),episode=一个真实底层游戏。

    历史 bool 继续映射到两个 298-wide 兼容视图；lossless dual
    ``dual-v4-asymmetric-v3`` 必须由 ``policy_observation_view`` 点名，
    防止旧 CLI 因默认值变化静默扩维。
    """

    metadata = {"render_modes": []}

    def __init__(self, manager_npz: str | None, max_steps: int = 3000,
                 rng_seed: int | None = None, log_windows: bool = False,
                 skip_dry: float | bool = False, manager_sha256: str | None = None,
                 manager_heuristic: str | None = None,
                 drink_sovereignty: bool = True,
                 legacy_policy_observation_view: bool = False,
                 policy_observation_view: str | None = None,
                 fast_forward_reward_credit: str = "none",
                 additional_terminal_death_cost: float = 0.0,
                 seed_scope: str = "train",
                 learning_window_scope: str = "farm-only",
                 depth_shaping_unit: float = 0.0,
                 worker_descend_bonus_fraction: float = 0.0,
                 worker_descend_escrow_fraction: float = 0.0,
                 worker_descend_escrow_power: float = 1.0,
                 worker_descend_escrow_readiness_gate: bool = False,
                 worker_descend_escrow_readiness_table: str = "v1",
                 worker_hp_loss_price: float = 0.0,
                 worker_potion_pickup_bonus: float = 0.0,
                 worker_no_progress_timeout_credit: str = "death-equivalent",
                 resource_protocol: str = "off",
                 resource_purchase_mode: str = "full",
                 resource_service_policy: str = "legacy-v1",
                 resource_readiness_law: str = "veto-v1",
                 worker_time_protocol: str = "legacy",
                 prefix_worker=None,
                 prefix_worker_sha256: str | None = None,
                 prefix_max_attempts: int | None = None,
                 prefix_max_microsteps: int | None = None,
                 **env_kwargs):
        super().__init__()
        if "resource_calibration" in env_kwargs:
            raise ValueError("Resource calibration is diagnostic-only; use OptionsEnv, not WorkerWindowEnv")
        from .resource_protocol import validate_resource_config
        self.resource_protocol, self.resource_purchase_mode = validate_resource_config(
            resource_protocol, resource_purchase_mode)
        from .resource_sustain import validate_service_policy
        self.resource_service_policy = validate_service_policy(
            self.resource_protocol, self.resource_purchase_mode, resource_service_policy)
        env_kwargs["resource_protocol"] = self.resource_protocol
        env_kwargs["resource_purchase_mode"] = self.resource_purchase_mode
        if self.resource_service_policy != "legacy-v1":
            env_kwargs["resource_service_policy"] = self.resource_service_policy
        # R17.1 ruling 3: readiness law passthrough (default keeps kwargs byte-identical).
        from .resource_protocol import validate_readiness_law
        self.resource_readiness_law = validate_readiness_law(
            self.resource_protocol, resource_readiness_law)
        if self.resource_readiness_law != "veto-v1":
            env_kwargs["resource_readiness_law"] = self.resource_readiness_law
        # R13 教室改革主旗:默认 farm-only 逐位复现旧法(非 FARM 窗脚本
        # 快进);farm-dive-v1 使 DIVE 窗成为一等 live 学习窗并向
        # OptionsEnv 移交窗内主权(a11/踏格)。
        self.learning_window_scope = _coerce_learning_window_scope(
            learning_window_scope)
        if worker_time_protocol not in ("legacy", "completion-l2-v1"):
            raise ValueError("Unknown worker_time_protocol")
        self.worker_time_protocol = worker_time_protocol
        if worker_time_protocol != "legacy":
            if self.learning_window_scope != "earned-dive-suffix-v1":
                raise ValueError("completion-l2-v1 requires earned-dive-suffix-v1")
            env_kwargs["worker_time_protocol"] = worker_time_protocol
        prefix_values = (prefix_worker, prefix_worker_sha256,
                         prefix_max_attempts, prefix_max_microsteps)
        if self.learning_window_scope == "earned-dive-suffix-v1":
            if (self.resource_protocol != "l2-town-v1"
                    or self.resource_purchase_mode != "full"
                    or self.resource_service_policy != "sustain-v6"):
                raise ValueError("earned-dive-suffix-v1 requires l2-town-v1/full/sustain-v6")
            if (not callable(prefix_worker)
                    or prefix_worker_sha256 != _PREFIX_WORKER_SHA256
                    or getattr(prefix_worker, "source_sha256", None)
                    != prefix_worker_sha256):
                raise ValueError("earned prefix requires the exact verified original R16 callback")
            if (not callable(getattr(prefix_worker, "episode_reseed", None))
                    or getattr(prefix_worker, "on_beat", None) is not None):
                raise ValueError("prefix callback requires episode_reseed and no on_beat hook")
            if (getattr(prefix_worker, "diablogym_worker_observation_view", None)
                    != WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC
                    or getattr(prefix_worker, "diablogym_worker_action12_mode", None)
                    != WORKER_ACTION12_ENVIRONMENT_MASK
                    or not drink_sovereignty):
                raise ValueError("prefix callback requires dual-v4-asymmetric-v3/environment-mask")
            for name, value in (("prefix_max_attempts", prefix_max_attempts),
                                ("prefix_max_microsteps", prefix_max_microsteps)):
                if (isinstance(value, (bool, np.bool_))
                        or not isinstance(value, (int, np.integer)) or int(value) <= 0):
                    raise ValueError(f"{name} requires an explicit positive integer")
            if _coerce_p_skip(skip_dry) != 0.0 or "workers" in env_kwargs:
                raise ValueError("earned prefix forbids skip_dry and external workers")
            self.prefix_worker = prefix_worker
            self.prefix_worker_sha256 = prefix_worker_sha256
            self.prefix_max_attempts = int(prefix_max_attempts)
            self.prefix_max_microsteps = int(prefix_max_microsteps)
        elif any(value is not None for value in prefix_values):
            raise ValueError("prefix arguments require earned-dive-suffix-v1")
        # R13.2 甲案(主席 2026-08-31 批「按甲来说吧」):势函数深度塑形。
        # F=unit×Δdepth 逐 transition;true terminal 退款 −unit×(d−1)
        # (φ(absorbing)=0);truncation 不退款由自举吸收。只进工人
        # policy_reward,不进经理账本/工资恒等式/考卷计分——剥薪法条
        # (下楼奖金归经理)零触碰,策略不变性由势函数定理背书。
        self.depth_shaping_unit = _coerce_depth_shaping_unit(
            depth_shaping_unit)
        if (self.depth_shaping_unit > 0.0
                and self.learning_window_scope not in _LIVE_DIVE_WINDOW_SCOPES):
            raise ValueError(
                "depth_shaping_unit 仅适用于 farm-dive-v1 教室"
                "(旧法无 live 下楼可塑)")
        # R14 乙案:有界下楼奖金返还,仅限 farm-dive-v1 教室(法域校验
        # 由 OptionsEnv 二次把关);记账/恒等式见 options_env._win_beat。
        if (float(worker_descend_bonus_fraction) > 0.0
                and self.learning_window_scope not in _LIVE_DIVE_WINDOW_SCOPES):
            raise ValueError(
                "worker_descend_bonus_fraction 仅适用于 farm-dive-v1 教室")
        # R14.2 丁案(主席夜间条例「乙不行直接按丁走」):下楼费托管发放。
        # 新最深层的下楼费 ×fraction 入托管;下一个非死亡收窗时 vest 进
        # policy_reward;死亡(live 或快进段)全额罚没——R11 认证的
        # 杀怪锁「没收型反自杀阀门」同款部件,工人可控自身生存故可学。
        # 全程只走 policy_reward 层:W 账本/恒等式/经理账本零触碰。
        _ef = float(worker_descend_escrow_fraction)
        if not np.isfinite(_ef) or not 0.0 <= _ef <= 1.0:
            raise ValueError(
                "worker_descend_escrow_fraction 必须在 [0,1] 内,"
                f"收到 {worker_descend_escrow_fraction!r}")
        if _ef > 0.0 and self.learning_window_scope not in _LIVE_DIVE_WINDOW_SCOPES:
            raise ValueError(
                "worker_descend_escrow_fraction 仅适用于 farm-dive-v1 教室")
        if _ef > 0.0 and float(worker_descend_bonus_fraction) > 0.0:
            raise ValueError(
                "乙案(bonus_fraction)与丁案(escrow_fraction)互斥,"
                "不得同时启用")
        self.descend_escrow_fraction = _ef
        # R14.3 戊案(主席夜令「挖掘一条能用的曲线」):托管工钱凸曲线。
        # 每层托管额 = fraction × unit × d^power;power=1.0 走原整数和
        # 路径逐位同旧。实测风险 h(d)≈0.23×1.53^(d−1) 几何上涨,线性
        # 工钱必卡盈亏平衡层,凸度须追上风险凸度(拟合 p≈1.6)。
        _ep = float(worker_descend_escrow_power)
        if not np.isfinite(_ep) or not 1.0 <= _ep <= 3.0:
            raise ValueError(
                "worker_descend_escrow_power 必须在 [1,3] 内,"
                f"收到 {worker_descend_escrow_power!r}")
        if _ep != 1.0 and _ef <= 0.0:
            raise ValueError(
                "worker_descend_escrow_power ≠ 1 仅与托管(escrow_fraction"
                ">0)同时生效")
        self.descend_escrow_power = _ep
        # R15 修正案二(工资侧执法):战备条件托管——escrow 只为
        # readiness>=1 的下楼计账。掩码法(榨干强制交权)能逼工人进
        # DIVE 窗,但未达标的下楼分文不入托管:强制得动人,骗不到钱。
        self.descend_escrow_readiness_gate = bool(
            worker_descend_escrow_readiness_gate)
        if (self.descend_escrow_readiness_gate
                and self.descend_escrow_fraction <= 0.0):
            raise ValueError(
                "descend_escrow_readiness_gate 仅与托管(escrow_fraction>0)"
                "同时生效")
        # R17.0 修正案一(合议庭更正二.1):托管战备门的尺子选择器。
        # v1 = 既有 readiness_power_ratio(v0.1 表,旧法逐位);v2 =
        # readiness_power_ratio_v2(v0.2 表,与 readiness-v3 教练同尺)。
        # 默认 v1 零漂移;v2 仅与战备门同时生效(否则尺子无处可用)。
        _table = str(worker_descend_escrow_readiness_table)
        if _table not in ("v1", "v2"):
            raise ValueError(
                "worker_descend_escrow_readiness_table 必须是 v1/v2,收到 "
                f"{worker_descend_escrow_readiness_table!r}")
        if _table != "v1" and not self.descend_escrow_readiness_gate:
            raise ValueError(
                "worker_descend_escrow_readiness_table=v2 仅与战备门"
                "(descend_escrow_readiness_gate)同时生效")
        self.descend_escrow_readiness_table = _table
        self._descend_escrow = 0.0
        # R16 修宪(审计簇 C7/C8):血量经济计价与终局记账修正。
        # hp_loss_price:每损失 1 点 HP 扣 price(非对称,只罚失血不奖回血
        # ——势函数式对称记账不改最优解,见甲案教训);
        # potion_pickup_bonus:a13 成功拾药即奖(腰带计数增加);
        # no_progress_timeout_credit="zero":无进展超时不再按死亡等价计罚
        # (幸存溢价不再为负)。三者默认关=旧法逐位不变。
        _hp_price = float(worker_hp_loss_price)
        _pot_bonus = float(worker_potion_pickup_bonus)
        if not np.isfinite(_hp_price) or _hp_price < 0.0:
            raise ValueError(
                f"worker_hp_loss_price 必须是有限非负数,收到 {_hp_price!r}")
        if not np.isfinite(_pot_bonus) or _pot_bonus < 0.0:
            raise ValueError(
                "worker_potion_pickup_bonus 必须是有限非负数,收到 "
                f"{_pot_bonus!r}")
        if worker_no_progress_timeout_credit not in (
                "death-equivalent", "zero"):
            raise ValueError(
                "worker_no_progress_timeout_credit 必须是 death-equivalent/"
                f"zero,收到 {worker_no_progress_timeout_credit!r}")
        self.hp_loss_price = _hp_price
        self.potion_pickup_bonus = _pot_bonus
        self.no_progress_timeout_credit = worker_no_progress_timeout_credit
        # 默认 none 精确复现旧版“所有快进奖励均只审计”的策略回报；
        # terminal-death-only 只跨冻结策略边界传递既有死亡罚分。额外死亡
        # cost 独立于该模式，在直接/快进真实 death 上都恰好追加一次。
        self.fast_forward_reward_credit = (
            _coerce_fast_forward_reward_credit(fast_forward_reward_credit))
        self.additional_terminal_death_cost = (
            _coerce_additional_terminal_death_cost(
                additional_terminal_death_cost))
        self.seed_scope = _coerce_seed_scope(seed_scope)
        legacy_view = bool(legacy_policy_observation_view)
        if policy_observation_view is None:
            resolved_policy_view = (
                WORKER_OBSERVATION_VIEW_LEGACY_V3
                if legacy_view else WORKER_OBSERVATION_VIEW_A12_OVERLAY
            )
        else:
            if (
                not isinstance(policy_observation_view, str)
                or policy_observation_view not in WORKER_OBSERVATION_VIEWS
            ):
                raise ValueError(
                    "policy_observation_view 必须是 "
                    f"{sorted(WORKER_OBSERVATION_VIEWS)} 之一，收到 "
                    f"{policy_observation_view!r}")
            resolved_policy_view = policy_observation_view
            if (
                legacy_view
                and resolved_policy_view
                != WORKER_OBSERVATION_VIEW_LEGACY_V3
            ):
                raise ValueError(
                    "legacy_policy_observation_view=True 与显式 "
                    f"policy_observation_view={resolved_policy_view!r} 冲突")
        self.policy_observation_view = resolved_policy_view
        if (self.learning_window_scope == "earned-dive-suffix-v1"
                and resolved_policy_view != WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC):
            raise ValueError("earned prefix requires the existing dual-v4-asymmetric-v3 learner view")
        # Retain this public compatibility attribute for existing contracts
        # and probes.  New code must bind the string-valued view above.
        self.legacy_policy_observation_view = (
            resolved_policy_view == WORKER_OBSERVATION_VIEW_LEGACY_V3)
        # R12 修正案二:脚本教练(仓库正典启发式,原文即假设门脚本教师
        # 规则的 margin-1 版)。魔鬼教练零 FARM 窗的结构性失败(台账
        # 2026-08-30)证明深层课堂需要"潜下去之后开课"的教练。
        if manager_heuristic is not None:
            if manager_heuristic not in _MANAGER_HEURISTICS:
                raise ValueError(
                    "manager_heuristic 只支持 "
                    f"{sorted(_MANAGER_HEURISTICS)},收到 "
                    f"{manager_heuristic!r}")
            if manager_npz is not None:
                raise ValueError("manager_heuristic 与 manager_npz 互斥")
            self.mgr = None
            self.manager_heuristic = manager_heuristic
        else:
            if manager_npz is None:
                raise ValueError("必须给 manager_npz 或 manager_heuristic")
            self.manager_heuristic = None
            self.mgr = NumpyManager(manager_npz, expected_sha256=manager_sha256)
            self.mgr.require_io_shape(303, 3, "WorkerWindow manager")
        self.oe = OptionsEnv(max_steps=max_steps,
                             drink_sovereignty=drink_sovereignty,
                             worker_observation_view=(
                                 self.policy_observation_view),
                             manager_observation_view=(
                                 MANAGER_OBSERVATION_VIEW_LEGACY_V3),
                             dive_live_sovereignty=(
                                 self.learning_window_scope
                                 in _LIVE_DIVE_WINDOW_SCOPES),
                             worker_descend_bonus_fraction=(
                                 float(worker_descend_bonus_fraction)),
                             **env_kwargs)
        if (self.depth_shaping_unit > 0.0
                or self.descend_escrow_fraction > 0.0):
            # 单调势/托管的自变量 _econ_episode_max_depth 只在 v2+ 经济下
            # 逐拍维护;v1 下恒 0 会使其静默空转,fail-closed 拒绝。
            _econ = getattr(self.oe.env, "reward_economy", None)
            if getattr(_econ, "name", "v1") == "v1":
                raise ValueError(
                    "depth_shaping_unit/descend_escrow_fraction 依赖 v2+ "
                    "经济的本局最深层追踪器,v1 经济下拒绝启用")
        base = self.oe.env.observation_space.shape[0]
        if (
            self.policy_observation_view
            == WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC
        ):
            observation_dim = DUAL_WORKER_OBSERVATION_DIM
        elif (
            self.policy_observation_view
            == WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE
        ):
            observation_dim = (
                DUAL_WORKER_OBSERVATION_DIM + DUAL_WORKER_V5_APPENDIX_DIM)
        else:
            observation_dim = base + N_EXTRA_WORKER
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(observation_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(15)
        self._rng = np.random.default_rng(rng_seed)
        self._alive = False
        self._episode_seed: int | None = None
        self.log_windows = log_windows
        # v26 绿洲 → E1 ⑤A:干层复访窗跳过概率 p_skip(bool 升格 float,1.0/0.0
        # 位级保留原布尔语义);p 抽签走自有专用流 _p_rng,零染训练 RNG(_rng)。
        self.skip_dry = _coerce_p_skip(skip_dry)
        # VecEnv may auto-reset a terminated environment before callbacks
        # regain control.  A rollout-boundary curriculum change therefore
        # cannot be applied safely by an after-step callback.  The callback
        # registers the next probability plus a per-environment Worker-step
        # countdown; reset() deliberately preserves both fields.
        self._pending_skip_dry_probability: float | None = None
        self._pending_skip_dry_remaining_env_steps = 0
        self._p_rng = (_derive_p_skip_rng(rng_seed) if rng_seed is not None
                       else np.random.default_rng())
        self.window_log = []      # log_windows=True 时:全部窗口(含快进窗)按序入册
        self.stats = {"windows": 0, "dry": 0, "fresh": 0, "ff_windows": 0,
                      "ff_dry": 0, "episodes": 0, "reseeds": 0,
                      "reasons": {}, "ff_reasons": {}}
        # 不删旧统计键；追加三本奖励/终结账，便于审计 reset 初始快进与
        # worker transition 内快进，不再把隐藏后果混为一谈。
        self.stats.update({"ff_terminals": 0, "transition_ff_reward": 0.0,
                           "reset_ff_reward": 0.0, "manual_ff_reward": 0.0})
        self.stats.update({
            "interrupted_resets": 0,
            "manual_ff_calls": 0,
            "direct_terminal_deaths": 0,
            "transition_ff_terminal_deaths": 0,
            "reset_ff_terminal_deaths": 0,
            "manual_ff_terminal_deaths": 0,
            "direct_existing_terminal_death_reward": 0.0,
            "direct_additional_terminal_death_reward": 0.0,
            "transition_ff_terminal_death_reward": 0.0,
            "transition_ff_additional_terminal_death_reward": 0.0,
            "credited_ff_terminal_death_reward": 0.0,
            "reset_ff_terminal_death_reward": 0.0,
            "reset_ff_additional_terminal_death_reward": 0.0,
            "additional_terminal_death_reward": 0.0,
            "direct_no_progress_timeouts": 0,
            "transition_ff_no_progress_timeouts": 0,
            "reset_ff_no_progress_timeouts": 0,
            "manual_ff_no_progress_timeouts": 0,
            "direct_no_progress_timeout_failure_reward": 0.0,
            "transition_ff_no_progress_timeout_failure_reward": 0.0,
            "reset_ff_no_progress_timeout_failure_reward": 0.0,
            "manual_ff_no_progress_timeout_failure_reward": 0.0,
            "credited_no_progress_timeout_failure_reward": 0.0,
        })

        if self.learning_window_scope == "earned-dive-suffix-v1":
            self._initialize_prefix_state()

    # ---- 内务 ----
    def _policy_observation(self, observation: np.ndarray) -> np.ndarray:
        """Apply the explicitly selected policy view at the lossless edge."""
        # A few pure transition probes instantiate the environment with
        # ``object.__new__`` to avoid native-engine setup.  Missing means the
        # constructor's A12-overlay default.  Real environments always rebuild
        # the complete v3 base from ``oe.env._raw``; the array-only fallback is
        # intentionally limited to synthetic transition tests.
        view = getattr(self, "policy_observation_view", None)
        if view is None:
            # Compatibility for pure transition probes created with
            # ``object.__new__`` and for pre-selector callers.
            view = (
                WORKER_OBSERVATION_VIEW_LEGACY_V3
                if getattr(
                    self, "legacy_policy_observation_view", False)
                else WORKER_OBSERVATION_VIEW_A12_OVERLAY
            )
        options_env = getattr(self, "oe", None)
        if (
            options_env is not None
            and callable(getattr(
                options_env, "_worker_policy_observation", None))
            and getattr(getattr(options_env, "env", None), "_raw", None)
            is not None
        ):
            return options_env._worker_policy_observation(
                view,
                skip_dry_probability=(
                    getattr(self, "skip_dry", 0.0)
                    if view in (
                        WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
                        WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE,
                    )
                    else 0.0
                ),
            )
        if view in (
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
            WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE,
        ):
            raise RuntimeError(
                f"{view} requires lossless active OptionsEnv raw")
        if view == WORKER_OBSERVATION_VIEW_LEGACY_V3:
            return legacy_worker_policy_observation_view(observation)
        return np.asarray(observation)

    def _new_episode(self, seed=None, options=None):
        if self._earned_prefix_enabled():
            self._prefix_require_capacity()
        if seed is None:
            if self.seed_scope in {"bc-v1", "bc-v2"}:
                raise RuntimeError(
                    f"{self.seed_scope} 示范局结束后禁止自动滚入未登记 seed")
            s = sample_train_seed(self._rng)
        else:
            s = int(seed)
            if self.seed_scope == "train" and is_reserved_train_seed(s):
                raise ValueError(f"训练 Worker reset 拒绝保留种子 {s}")
            expected = (
                _CURRENT_BC_V1_RANGE
                if self.seed_scope == "bc-v1"
                else _CURRENT_BC_V2_RANGE
                if self.seed_scope == "bc-v2"
                else None
            )
            if expected is not None and not _seed_in_half_open_range(
                    s, expected):
                raise ValueError(
                    f"{self.seed_scope} 只允许登记池 "
                    f"[{expected[0]},{expected[1]})，收到 {s}")
        # p_skip 是逐“真实底层局”派生，而不是只在外部 reset(seed) 时
        # 派生一次后跨 auto-reset 连续消费。这样任一 episode_seed 都能
        # 单独重放其课程抽签，不依赖此前经历了多少局/多少 dry 窗。
        self._p_rng = _derive_p_skip_rng(s)
        try:
            if self._earned_prefix_enabled():
                self._prefix_begin_attempt(s)
            self.oe.reset(seed=s, options=options)
            if self._earned_prefix_enabled():
                self._prefix_after_reset()
        except Exception as exc:
            if self._earned_prefix_enabled():
                self._prefix_abort_error(exc)
            raise
        self._episode_seed = s
        self._alive = True
        # R14.2 丁案:托管余额按局清零(死亡罚没/终局 vest 已在上局结清;
        # 防御性清零防跨局泄漏)。
        self._descend_escrow = 0.0
        self.stats["episodes"] += 1

    def _log(self, extra, fast_forward: bool):
        if self.log_windows:
            self.window_log.append(dict(extra, ff=fast_forward))
        reason = str(extra.get("reason", "unknown"))
        reasons = self.stats["reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        if fast_forward:
            self.stats["ff_windows"] += 1
            ff_reasons = self.stats["ff_reasons"]
            ff_reasons[reason] = ff_reasons.get(reason, 0) + 1

    def _mgr_choose(self) -> int:
        if getattr(self, "resource_protocol", "off") != "off":
            return self.oe.resource_option_choice()
        if self.manager_heuristic is not None:
            # level-margin-1:榨干 或 高一级即潜,否则农;
            # readiness-v1(R15):战备表短板达标才潜,榨干为逃生口
            # (楼层榨干=战备无从提升,best-effort 下行)。掩码回退按
            # FARM->DIVE->RESUPPLY 合法序,与 G0 探针同款。
            raw = self.oe.env._raw
            mask = self.oe.action_masks()
            if self.manager_heuristic == "readiness-v3":
                # R16:v0.2 战备表 + 击杀口径清场逃生。进层基数按
                # dungeon_level 变化重置(getattr 容缺护 __new__ 壳)。
                # G0-3/4/5 定标裁决:阈值 0.75 会在楼层仍有 25% 活怪时开 DIVE
                # 窗,a11 主线被怪堵路→失速(57 窗 39 stall);改为 1.0
                # (花名册无活怪即刻下潜,不必等榨干旗的 KILL_PATIENCE),
                # 不可达残怪仍由榨干旗兜底。
                dlvl = int(raw["dungeon_level"])
                floor_state = getattr(self, "_readiness_floor_state", None)
                if floor_state is None or floor_state[0] != dlvl:
                    floor_state = (dlvl, int(raw.get("monster_kill_total", 0)))
                    self._readiness_floor_state = floor_state
                ready = readiness_power_ratio_v2(raw, dlvl + 1) >= 1.0
                cleared = readiness_clear_ratio(raw, floor_state[1]) >= 1.0
                want = DIVE if (ready or cleared) else FARM
            elif self.manager_heuristic == "readiness-v2":
                ready = readiness_power_ratio(
                    raw, int(raw["dungeon_level"]) + 1) >= 1.0
                want = (DIVE if (ready or readiness_floor_cleared(raw))
                        else FARM)
            elif self.manager_heuristic == "readiness-v1":
                ready = readiness_power_ratio(
                    raw, int(raw["dungeon_level"]) + 1) >= 1.0
                want = (DIVE if (ready or self.oe.exhausted) else FARM)
            else:
                want = (DIVE if (self.oe.exhausted
                                 or int(raw["char_level"])
                                 >= int(raw["dungeon_level"]) + 1)
                        else FARM)
            if mask[want]:
                return int(want)
            for cand in (FARM, DIVE, RESUPPLY):
                if mask[cand]:
                    return int(cand)
            raise RuntimeError("经理三动作全被掩,协议不变量被破坏")
        mobs = self.oe._mgr_obs(self.oe._last_base_obs)
        return self.mgr.choose(mobs, self.oe.action_masks())

    def set_skip_dry_p(self, p: float | bool) -> float:
        """Set the live probability only while no boundary switch is armed."""
        pending = getattr(
            self, "_pending_skip_dry_probability", None)
        remaining = getattr(
            self, "_pending_skip_dry_remaining_env_steps", 0)
        if pending is not None or remaining != 0:
            raise RuntimeError(
                "已有 rollout-boundary p_skip 切换倒计时，"
                "禁止直接覆盖当前概率")
        self.skip_dry = _coerce_p_skip(p)
        return self.skip_dry

    def schedule_skip_dry_p(
            self, p: float | bool, remaining_env_steps: int) -> dict:
        """Register an atomic probability switch after N Worker actions.

        The countdown is local to one environment, not the global VecEnv
        sample count.  The Nth action and its reflex tail run under the current
        value; the pending value becomes live immediately afterwards, before
        next-state construction, next-window selection, or VecEnv auto-reset.
        """
        if (
            not isinstance(remaining_env_steps, (int, np.integer))
            or isinstance(remaining_env_steps, (bool, np.bool_))
            or int(remaining_env_steps) <= 0
        ):
            raise ValueError(
                "rollout-boundary p_skip remaining_env_steps 必须是正整数")
        pending = getattr(
            self, "_pending_skip_dry_probability", None)
        remaining = getattr(
            self, "_pending_skip_dry_remaining_env_steps", 0)
        if pending is not None or remaining != 0:
            raise RuntimeError(
                "rollout-boundary p_skip 切换已登记，禁止重叠倒计时")
        self._pending_skip_dry_probability = _coerce_p_skip(p)
        self._pending_skip_dry_remaining_env_steps = int(
            remaining_env_steps)
        return self.skip_dry_schedule_state()

    def skip_dry_schedule_state(self) -> dict:
        """Return a pickle-safe curriculum state for callback audits."""
        pending = getattr(
            self, "_pending_skip_dry_probability", None)
        remaining = getattr(
            self, "_pending_skip_dry_remaining_env_steps", 0)
        if pending is None:
            if remaining != 0:
                raise RuntimeError(
                    "p_skip pending 概率缺失但倒计时非零")
            normalized_pending = None
        else:
            if (
                not isinstance(remaining, (int, np.integer))
                or isinstance(remaining, (bool, np.bool_))
                or int(remaining) <= 0
            ):
                raise RuntimeError("p_skip pending 倒计时状态非法")
            normalized_pending = float(pending)
            remaining = int(remaining)
        return {
            "current_probability": float(self.skip_dry),
            "pending_probability": normalized_pending,
            "remaining_env_steps": int(remaining),
        }

    def _tick_skip_dry_schedule(self) -> bool:
        """Advance the countdown after one successful Worker action."""
        state = self.skip_dry_schedule_state()
        pending = state["pending_probability"]
        if pending is None:
            return False
        remaining = int(state["remaining_env_steps"]) - 1
        if remaining > 0:
            self._pending_skip_dry_remaining_env_steps = remaining
            return False
        # Commit before any observation/advance/reset can expose a mixed
        # old-selection/new-feature state.
        self.skip_dry = _coerce_p_skip(pending)
        self._pending_skip_dry_probability = None
        self._pending_skip_dry_remaining_env_steps = 0
        return True

    def _skip_dry_draw(self) -> bool:
        """p_skip 抽签:端点 1.0/0.0 不耗流(位级复刻原布尔行为);中间值走专用流。"""
        p = self.skip_dry
        if p >= 1.0:
            return True
        if p <= 0.0:
            return False
        return float(self._p_rng.random()) < p

    @staticmethod
    def _base_boundary(done, trunc) -> tuple[bool, bool]:
        """把底层结束映射为互斥 Gymnasium terminated/truncated。"""
        terminated = bool(done)
        truncated = bool(trunc) and not terminated
        return terminated, truncated

    @classmethod
    def _worker_boundary(
            cls, done, trunc, option_extra: dict,
            base_info: dict | None) -> tuple[bool, bool, bool]:
        """将无进展预算耗尽映射成 Worker 策略失败，而非 TimeLimit。

        OptionsEnv 是 no-progress 事实的单一真源；这里只改变 Worker Gym
        边界语义。正常有进展的安全预算边界仍可 bootstrap，动画未结清的
        unsettled 边界仍沿用底层 fail-closed terminated。
        """
        terminated, truncated = cls._base_boundary(done, trunc)
        if not isinstance(option_extra, dict):
            raise RuntimeError("Worker 终局缺少 option_extra")
        if not isinstance(base_info, dict):
            raise RuntimeError("Worker 终局缺少底层 base_info")
        required = {
            "base_done",
            "base_trunc",
            "budget_boundary",
            "no_progress_micro_steps",
            "timeout_without_progress",
        }
        missing = required.difference(option_extra)
        if missing:
            raise RuntimeError(
                "Worker 终局 option_extra 缺字段:"
                f"{sorted(missing)}")
        marker = option_extra["timeout_without_progress"]
        budget_boundary = option_extra["budget_boundary"]
        base_done = option_extra["base_done"]
        base_trunc = option_extra["base_trunc"]
        no_progress_micro_steps = option_extra[
            "no_progress_micro_steps"]
        if (
            not isinstance(marker, (bool, np.bool_))
            or not isinstance(budget_boundary, (bool, np.bool_))
            or not isinstance(base_done, (bool, np.bool_))
            or not isinstance(base_trunc, (bool, np.bool_))
        ):
            raise RuntimeError(
                "Worker 终局边界标志必须是 bool")
        if (
            isinstance(no_progress_micro_steps, (bool, np.bool_))
            or not isinstance(
                no_progress_micro_steps, (int, np.integer))
            or int(no_progress_micro_steps) < 0
        ):
            raise RuntimeError(
                "option_extra.no_progress_micro_steps "
                "必须是非负普通整数")
        if (
            bool(base_done) is not True
            or bool(base_trunc) != truncated
        ):
            raise RuntimeError(
                "Worker 终局 base_done/base_trunc 与底层返回不闭合:"
                f"done={done!r},trunc={trunc!r},"
                f"base_done={base_done!r},base_trunc={base_trunc!r}")

        terminal = dict(base_info)
        safe_time_limit = bool(
            not terminated
            and truncated
            and bool(base_trunc)
            and terminal.get("time_limit_bootstrap_safe") is True
            and terminal.get("unsettled_budget_terminal") is False
        )
        unsettled_terminal = bool(
            terminated
            and not truncated
            and not bool(base_trunc)
            and terminal.get("time_limit_bootstrap_safe") is False
            and terminal.get("unsettled_budget_terminal") is True
        )
        derived_budget_boundary = (
            safe_time_limit or unsettled_terminal)
        if bool(budget_boundary) != derived_budget_boundary:
            raise RuntimeError(
                "Worker 终局 budget_boundary 与底层边界事实不闭合:"
                f"terminated={terminated},truncated={truncated},"
                f"budget_boundary={budget_boundary!r},"
                "time_limit_bootstrap_safe="
                f"{terminal.get('time_limit_bootstrap_safe')!r},"
                "unsettled_budget_terminal="
                f"{terminal.get('unsettled_budget_terminal')!r}")
        # A Gymnasium truncation is only legitimate when the lower layer
        # explicitly certified a settled TimeLimit boundary.  Without this
        # independent check, a simultaneous producer bug could omit both
        # base markers and publish ``budget_boundary=False``; the two false
        # values would agree and silently re-enable value bootstrap.
        if truncated and not safe_time_limit:
            raise RuntimeError(
                "Worker truncated 终局缺少显式安全 TimeLimit 证明:"
                "time_limit_bootstrap_safe="
                f"{terminal.get('time_limit_bootstrap_safe')!r},"
                "unsettled_budget_terminal="
                f"{terminal.get('unsettled_budget_terminal')!r}")

        expected_timeout = bool(
            derived_budget_boundary
            and int(no_progress_micro_steps) >= KILL_PATIENCE
        )
        timeout_without_progress = bool(marker)
        if timeout_without_progress != expected_timeout:
            raise RuntimeError(
                "option_extra.timeout_without_progress 与 "
                "budget/no-progress 事实不闭合:"
                f"marker={timeout_without_progress},"
                f"budget_boundary={derived_budget_boundary},"
                f"no_progress_micro_steps={no_progress_micro_steps},"
                f"threshold={KILL_PATIENCE}")
        if not timeout_without_progress:
            return terminated, truncated, False

        if safe_time_limit == unsettled_terminal:
            raise RuntimeError(
                "no-progress timeout 只能来自 settled safe TimeLimit 或 "
                "fail-closed unsettled terminal："
                f"terminated={terminated},truncated={truncated},"
                f"base_done={option_extra.get('base_done')!r},"
                f"base_trunc={option_extra.get('base_trunc')!r},"
                f"budget_boundary={option_extra.get('budget_boundary')!r},"
                "time_limit_bootstrap_safe="
                f"{terminal.get('time_limit_bootstrap_safe')!r},"
                "unsettled_budget_terminal="
                f"{terminal.get('unsettled_budget_terminal')!r}")
        return True, False, True

    @staticmethod
    def _option_base_info(info: dict) -> dict:
        """从 OptionsEnv.step info 中剥出底层 env 的原始 info。"""
        return {key: value for key, value in dict(info).items()
                if key != "option_extra"}

    @staticmethod
    def _attach_terminal_base_info(
            info: dict, base_info: dict | None, *,
            terminated: bool, truncated: bool) -> None:
        """封装底层终局，并在顶层发布最终 Worker bootstrap 语义。"""
        if bool(terminated) == bool(truncated):
            raise RuntimeError(
                "Worker 终局必须恰有一个 terminated/truncated")
        terminal = dict(base_info or {})
        info["terminal_base_info"] = terminal
        for key, value in terminal.items():
            info.setdefault(key, value)
        # A settled base TimeLimit may have been promoted to a no-progress
        # policy failure.  Keep that lower-layer fact only in
        # terminal_base_info; top-level consumers must see the final Worker
        # boundary and can bootstrap iff the wrapper itself returned truncation.
        info["time_limit_bootstrap_safe"] = bool(truncated)

    def _terminal_death_reward(
            self, terminated: bool, truncated: bool,
            base_info: dict | None) -> float:
        """重建底层终局拍中可安全跨越冻结策略边界的死亡分量。

        OptionsEnv 的窗口 ``R`` 只公开整窗总和，终局拍可能同时含 XP、伤害、
        走位等正负项。把总和或 ``min(R, 0)`` 交给 worker 都会泄漏冻结策略
        的行为信用。这里依底层已落盘的 episode_extra 与实际 death_ladder
        配置，只重建 DiabloGymEnv._reward 的死亡加项；任何非死亡终局为 0。
        """
        if not (terminated or truncated):
            return 0.0
        terminal = dict(base_info or {})
        episode = terminal.get("episode_extra")
        if not isinstance(episode, dict):
            raise RuntimeError("底层终局缺少 episode_extra，无法核对死亡罚分")
        died = bool(episode.get("died", False))
        if not died:
            return 0.0
        if not terminated or truncated:
            raise RuntimeError("died=True 却未映射为互斥 terminated")

        base_env = getattr(self.oe, "env", None)
        if base_env is None or not hasattr(base_env, "death_ladder"):
            raise RuntimeError("无法读取底层 death_ladder 配置")
        try:
            reward = terminal_death_reward_component(
                dead=True,
                dungeon_level=episode["depth"],
                death_ladder=bool(base_env.death_ladder),
                economy=getattr(base_env, "reward_economy",
                                REWARD_ECONOMY_V1),
            )
            depth = int(episode["depth"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("死亡终局缺少有效 depth") from exc
        raw = getattr(base_env, "_raw", None)
        if isinstance(raw, dict):
            if not bool(raw.get("dead", False)):
                raise RuntimeError("episode_extra.died=True 但底层 raw.dead=False")
            raw_depth = int(raw.get("dungeon_level", depth))
            if raw_depth != depth:
                raise RuntimeError(
                    f"死亡终局 depth 账不一致: info={depth},raw={raw_depth}")
        return reward

    def _additional_terminal_death_reward(
            self, terminated: bool, truncated: bool,
            base_info: dict | None) -> float:
        """将非负配置 cost 只映射到真实死亡终局；非死亡/截断恒为 0。"""
        if not terminated or truncated:
            return 0.0
        episode = dict(base_info or {}).get("episode_extra")
        if not isinstance(episode, dict):
            raise RuntimeError("底层终局缺少 episode_extra，无法追加死亡成本")
        return (
            -self.additional_terminal_death_cost
            if bool(episode.get("died", False))
            else 0.0
        )

    def _no_progress_timeout_failure_components(
            self, timeout_without_progress: bool,
            base_info: dict | None) -> tuple[float, float, float]:
        """返回无进展失败的 (基础死亡等价项, 额外项, 总项)。"""
        if not isinstance(timeout_without_progress, (bool, np.bool_)):
            raise RuntimeError("timeout_without_progress 必须是 bool")
        if not bool(timeout_without_progress):
            return 0.0, 0.0, 0.0
        if getattr(self, "no_progress_timeout_credit",
                   "death-equivalent") == "zero":
            # R16(C8):超时≠死亡——终局仍为 terminated,但不计死亡等价罚,
            # 活到终局的轨迹不再被记成"L1 死亡还惨"。
            return 0.0, 0.0, 0.0
        terminal = dict(base_info or {})
        episode = terminal.get("episode_extra")
        if not isinstance(episode, dict):
            raise RuntimeError(
                "no-progress timeout 缺少 episode_extra，无法核对当前深度")
        if bool(episode.get("died", False)):
            raise RuntimeError("no-progress timeout 不得同时标记真实死亡")
        base_env = getattr(self.oe, "env", None)
        if base_env is None or not hasattr(base_env, "death_ladder"):
            raise RuntimeError("无法读取底层 death_ladder 配置")
        try:
            base_failure = terminal_death_reward_component(
                dead=True,
                dungeon_level=episode["depth"],
                death_ladder=bool(base_env.death_ladder),
                economy=getattr(base_env, "reward_economy",
                                REWARD_ECONOMY_V1),
            )
            depth = int(episode["depth"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "no-progress timeout 缺少有效当前 depth") from exc
        raw = getattr(base_env, "_raw", None)
        if isinstance(raw, dict):
            if bool(raw.get("dead", False)):
                raise RuntimeError(
                    "no-progress timeout 但底层 raw.dead=True")
            raw_depth = int(raw.get("dungeon_level", depth))
            if raw_depth != depth:
                raise RuntimeError(
                    "no-progress timeout depth 账不一致："
                    f"info={depth},raw={raw_depth}")
        additional_failure = -float(self.additional_terminal_death_cost)
        total_failure = float(base_failure) + additional_failure
        if (
            not all(np.isfinite(value) for value in (
                base_failure, additional_failure, total_failure))
            or base_failure >= 0.0
            or additional_failure > 0.0
            or total_failure != float(base_failure) + additional_failure
        ):
            raise RuntimeError(
                "no-progress timeout 失败成本分账异常："
                f"base={base_failure},additional={additional_failure},"
                f"total={total_failure}")
        return float(base_failure), additional_failure, total_failure

    @staticmethod
    def _attach_no_progress_timeout_audit(
            info: dict, timeout_without_progress: bool,
            components: tuple[float, float, float],
            credit_mode: str = "death-equivalent") -> None:
        base_failure, additional_failure, total_failure = components
        timeout = bool(timeout_without_progress)
        if credit_mode not in ("death-equivalent", "zero"):
            raise RuntimeError(
                f"no_progress_timeout_credit 非法: {credit_mode!r}")
        # R16(C8)零罚金法域:超时仍为 True,分账恒 (0,0,0)——旧法下超时
        # 必带负 base_failure,故该特例不掩盖任何旧法违规;且只在
        # credit_mode=="zero" 下放行(fail-closed),法域随 info 下传给
        # leashed_ppo 的 formal PG 守门员。
        zero_credit_timeout = (timeout and credit_mode == "zero"
                               and components == (0.0, 0.0, 0.0))
        if (
            (not timeout and components != (0.0, 0.0, 0.0))
            or (timeout and not zero_credit_timeout and not (
                base_failure < 0.0
                and additional_failure <= 0.0
                and total_failure == base_failure + additional_failure
            ))
        ):
            raise RuntimeError(
                "Worker no-progress timeout 审计分账不闭合")
        info.update({
            "worker_no_progress_timeout": timeout,
            "no_progress_timeout_base_failure_reward": base_failure,
            "no_progress_timeout_additional_failure_reward": (
                additional_failure),
            "no_progress_timeout_failure_reward": total_failure,
            "no_progress_timeout_credit": credit_mode,
        })

    def _record_no_progress_timeout(
            self, scope: str, total_failure: float, *,
            credited: bool) -> None:
        if scope not in {"direct", "transition_ff", "reset_ff", "manual_ff"}:
            raise RuntimeError(f"非法 no-progress timeout 账本 scope={scope!r}")
        if not np.isfinite(total_failure) or (
                total_failure >= 0.0
                and not (
                    total_failure == 0.0
                    and getattr(self, "no_progress_timeout_credit",
                                "death-equivalent") == "zero")):
            raise RuntimeError(
                "no-progress timeout 总失败成本必须有限且为负")
        count_key = f"{scope}_no_progress_timeouts"
        reward_key = f"{scope}_no_progress_timeout_failure_reward"
        self.stats[count_key] = int(self.stats.get(count_key, 0)) + 1
        self.stats[reward_key] = (
            float(self.stats.get(reward_key, 0.0)) + total_failure)
        if credited:
            self.stats["credited_no_progress_timeout_failure_reward"] = (
                float(self.stats.get(
                    "credited_no_progress_timeout_failure_reward", 0.0))
                + total_failure)

    def _earned_prefix_enabled(self) -> bool:
        return getattr(self, "learning_window_scope", "farm-only") == "earned-dive-suffix-v1"

    def _initialize_prefix_state(self):
        self._prefix_original_physical_limit = int(self.oe.env.max_steps)
        self._prefix_observation_limit = int(self.oe.max_steps)
        self._prefix_deadline = None
        self._prefix_handed_off = False
        self._prefix_failure = None
        self._prefix_attempt = None
        self._prefix_attempts = []
        self._prefix_lifetime_attempts = 0
        self._prefix_lifetime_microsteps = 0
        self.stats.update({
            "prefix_attempts": 0, "prefix_microsteps": 0,
            "prefix_handoffs": 0, "prefix_no_handoff_episodes": 0,
            "prefix_parent_decisions": 0, "prefix_raw_reward": 0.0,
            "prefix_worker_wage": 0.0, "prefix_budget_exhausted": False,
        })

        def parent(observation, action_mask):
            self._prefix_validate_limits(active=True)
            action = self.prefix_worker(observation, action_mask)
            if not self.action_space.contains(action) or not bool(action_mask[int(action)]):
                raise RuntimeError("fixed prefix callback returned an illegal action")
            self.stats["prefix_parent_decisions"] += 1
            self._prefix_attempt["parent_decisions"] += 1
            self._prefix_attempt["requested_actions"][int(action)] += 1
            return action

        for name in ("diablogym_worker_observation_view", "diablogym_worker_action12_mode"):
            setattr(parent, name, getattr(self.prefix_worker, name))
        self._prefix_policy = parent

    def get_prefix_ledger(self) -> dict:
        """Return detached audit data; this performs no native or policy query."""
        if not self._earned_prefix_enabled():
            return {"enabled": False}
        return copy.deepcopy({
            "enabled": True, "scope": self.learning_window_scope,
            "prefix_worker_sha256": self.prefix_worker_sha256,
            "prefix_max_attempts": self.prefix_max_attempts,
            "prefix_max_microsteps": self.prefix_max_microsteps,
            "budget_scope": "per-env-lifetime-across-resets",
            "budget_start": "after_normal_env_reset",
            "bootstrap_navigation_in_budget": False,
            "bootstrap_note": "Every attempt executes the normal reset/navigation; its wall time is included below.",
            "original_physical_limit": self._prefix_original_physical_limit,
            "observation_limit": self._prefix_observation_limit,
            "lifetime": {
                **{key: value for key, value in self.stats.items() if key.startswith("prefix_")},
                "prefix_attempts": self._prefix_lifetime_attempts,
                "prefix_microsteps": self._prefix_lifetime_microsteps,
            },
            "failure": self._prefix_failure,
            "attempts": self._prefix_attempts,
        })

    def _prefix_snapshot(self):
        raw = getattr(self.oe.env, "_raw", None)
        if not isinstance(raw, dict):
            return None
        keys = ("dungeon_level", "is_set_level", "set_level_number", "player_x", "player_y",
                "future_x", "future_y", "player_mode", "walkpath0", "walkpath", "dest_action",
                "dead", "game_over", "victory", "hp", "max_hp", "xp", "char_level",
                "gold", "belt_heals", "belt_heal_kinds", "armor_class", "equipped_items",
                "gear_combat_profile", "monster_kill_total")
        state = {key: copy.deepcopy(raw[key]) for key in keys if key in raw}
        state["resource_state"] = {key: copy.deepcopy(value)
                                   for key, value in raw.get("resource_state", {}).items()
                                   if key in ("enabled", "readiness", "transition", "service_trip", "service_authorized",
                                              "ordinary_armor_scope", "preserve_equipment_readiness")}
        return state

    def _prefix_validate_limits(self, *, active):
        expected = self._prefix_deadline if active else self._prefix_original_physical_limit
        if (not active and getattr(self, "worker_time_protocol", "legacy") == "completion-l2-v1"):
            expected = self.oe._completion_clock.state.physical_deadline
        if (expected is None or int(self.oe.env.max_steps) != expected
                or int(self.oe.max_steps) != self._prefix_observation_limit):
            raise RuntimeError("earned prefix physical/observation limit changed outside its owner")

    def _prefix_budget_exception(self, reason):
        audit = self.get_prefix_ledger()
        audit["attempt_summary"] = [
            {key: attempt.get(key) for key in
             ("attempt", "episode_seed", "status", "microsteps", "raw_reward", "worker_wage", "wall_seconds")}
            for attempt in audit["attempts"]]
        audit["attempts"] = audit["attempts"][-1:]
        return PrefixBudgetExceeded(reason, audit)

    def _prefix_require_capacity(self):
        if self._prefix_failure is not None:
            raise self._prefix_budget_exception(f"earned prefix is sealed after failure: {self._prefix_failure}")
        if (self._prefix_lifetime_attempts >= self.prefix_max_attempts
                or self._prefix_lifetime_microsteps >= self.prefix_max_microsteps):
            self.stats["prefix_budget_exhausted"] = True
            self._prefix_failure = ("attempt_budget_exhausted"
                                    if self._prefix_lifetime_attempts >= self.prefix_max_attempts
                                    else "microstep_budget_exhausted")
            raise self._prefix_budget_exception(self._prefix_failure)

    def _prefix_begin_attempt(self, seed):
        self._prefix_validate_limits(active=False)
        self._prefix_handed_off = False
        self._prefix_lifetime_attempts += 1
        self.stats["prefix_attempts"] = self._prefix_lifetime_attempts
        self._prefix_wall_start = time.perf_counter()
        self._prefix_attempt = {
            "attempt": self.stats["prefix_attempts"], "episode_seed": int(seed),
            "status": "resetting", "parent_decisions": 0, "requested_actions": [0] * 15,
            "microsteps": 0, "raw_reward": 0.0, "worker_wage": 0.0, "windows": [],
            "budget_start": "after_normal_env_reset", "bootstrap_navigation_in_budget": False,
        }
        self._prefix_attempts.append(self._prefix_attempt)
        self.prefix_worker.episode_reseed(int(seed))

    def _prefix_after_reset(self):
        self._prefix_validate_limits(active=False)
        clock = int(self.oe.env._resource_actual_microsteps)
        if int(self.oe.env._steps) != clock:
            raise RuntimeError("earned prefix requires aligned settled physical clocks")
        remaining = self.prefix_max_microsteps - self._prefix_lifetime_microsteps
        self._prefix_deadline = min(self._prefix_original_physical_limit, clock + remaining)
        if self._prefix_deadline <= clock:
            raise RuntimeError("normal reset returned beyond the prefix physical budget")
        self.oe.env.max_steps = self._prefix_deadline
        if getattr(self, "worker_time_protocol", "legacy") == "completion-l2-v1":
            self.oe.env._completion_prefix_active = True
            self.oe.env._completion_prefix_deadline = self._prefix_deadline
        self._prefix_accounted_clock = clock
        self._prefix_zero_tick_windows = 0
        self._prefix_attempt.update({
            "status": "prefix", "start_native_microstep": clock,
            "start_env_steps": int(self.oe.env._steps),
            "physical_deadline": self._prefix_deadline,
            "reset_wall_seconds": time.perf_counter() - self._prefix_wall_start,
            "initial_state": self._prefix_snapshot(),
        })

    def _prefix_account_clock(self):
        self._prefix_validate_limits(active=True)
        clock = int(self.oe.env._resource_actual_microsteps)
        if (clock < self._prefix_accounted_clock or clock > self._prefix_deadline
                or int(self.oe.env._steps) != clock):
            raise RuntimeError("earned prefix actual microstep accounting violated its physical deadline")
        delta = clock - self._prefix_accounted_clock
        self._prefix_lifetime_microsteps += delta
        self.stats["prefix_microsteps"] = self._prefix_lifetime_microsteps
        self._prefix_attempt["microsteps"] += delta
        self._prefix_accounted_clock = clock
        return delta

    def _prefix_restore_limit(self):
        if self._prefix_deadline is not None:
            self._prefix_validate_limits(active=True)
            self.oe.env.max_steps = self._prefix_original_physical_limit
            self._prefix_deadline = None
            if getattr(self, "worker_time_protocol", "legacy") == "completion-l2-v1":
                self.oe.env._completion_prefix_active = False
                self.oe.env._completion_prefix_deadline = None

    def _prefix_finish_attempt(self, status, **details):
        self._prefix_attempt.update({
            "status": status, "end_native_microstep": int(self.oe.env._resource_actual_microsteps),
            "end_env_steps": int(self.oe.env._steps),
            "wall_seconds": time.perf_counter() - self._prefix_wall_start,
            "final_state": self._prefix_snapshot(), **copy.deepcopy(details),
        })
        self._prefix_restore_limit()
        if status != "handoff":
            self.stats["prefix_no_handoff_episodes"] += 1

    def _prefix_abort_error(self, exc):
        # Unexpected failures remain failures: no fabricated terminal, observation,
        # payroll completion or retry. Do not overwrite a foreign changed deadline.
        if self._prefix_failure is None:
            self._prefix_failure = f"engineering_error:{type(exc).__name__}:{exc}"
        if self._prefix_attempt is not None and self._prefix_attempt.get("status") in ("prefix", "resetting"):
            self._prefix_attempt.update({
                "status": "engineering_error", "error": str(exc),
                "wall_seconds": time.perf_counter() - self._prefix_wall_start,
                "final_state": self._prefix_snapshot(),
                "window_incomplete": self.oe._win is not None,
                "observed_native_microstep": getattr(self.oe.env, "_resource_actual_microsteps", None),
            })
            if getattr(self.oe.env, "_completion_time_failure", None) is not None:
                self._prefix_attempt["completion_time_failure"] = copy.deepcopy(
                    self.oe.env._completion_time_failure)
        if (self._prefix_deadline is not None
                and int(self.oe.env.max_steps) == self._prefix_deadline
                and int(self.oe.max_steps) == self._prefix_observation_limit):
            self._prefix_restore_limit()
        self._alive = False

    def _prefix_record_window(self, extra, base_info, start_clock):
        delta = self._prefix_account_clock()
        self._prefix_zero_tick_windows = (self._prefix_zero_tick_windows + 1 if delta == 0 else 0)
        self._prefix_attempt["windows"].append({
            "start_native_microstep": int(start_clock),
            "end_native_microstep": int(self.oe.env._resource_actual_microsteps),
            "extra": copy.deepcopy(extra), "terminal_info": copy.deepcopy(base_info)
            if extra.get("base_done") else None,
        })
        for field, stat, value in (("raw_reward", "prefix_raw_reward", float(extra["R"])),
                                   ("worker_wage", "prefix_worker_wage", float(extra["worker_wage"]))):
            self._prefix_attempt[field] += value
            self.stats[stat] += value
        if self._prefix_zero_tick_windows >= _MAX_PREFIX_ZERO_TICK_WINDOWS:
            raise RuntimeError("earned prefix repeated zero-native-tick windows without a learner handoff")

    def _prefix_open_dive(self):
        # Match the real Options worker opening. A failed opening verdict runs
        # this entire window with the parent; it is never closed and reopened.
        self.oe._win_begin(DIVE)
        ending = self.oe._consume_fuse_recovery()
        if ending is None or ending.reason is None:
            ending = self.oe._drain()
        observation = None
        if ending is None:
            observation = self.oe._worker_policy_observation(self.policy_observation_view)
            mask = np.asarray(observation[DUAL_WORKER_ACTION_MASK_SLICE])
            manager_mask = np.asarray(observation[DUAL_WORKER_MANAGER_MASK_SLICE])
            if (mask.shape != (15,) or manager_mask.shape != (3,)
                    or not np.all((mask == 0) | (mask == 1))
                    or not np.all((manager_mask == 0) | (manager_mask == 1))):
                raise RuntimeError("earned prefix requires the canonical embedded action masks")
            from .resource_protocol import native_readiness
            raw = self.oe.env._raw
            eligible = (int(raw["dungeon_level"]) == 1 and not raw.get("is_set_level")
                        and not any(raw.get(key) for key in ("dead", "game_over", "victory"))
                        and self.oe.env._decision_idle(raw)
                        and native_readiness(raw)["ready"]
                        and bool(manager_mask[DIVE]) and bool(np.any(mask)))
            self._prefix_attempt.setdefault("dive_openings", []).append({
                "window_id": int(self.oe._win["window_id"]),
                "native_microstep": int(self.oe.env._resource_actual_microsteps),
                "eligible": bool(eligible), "state": self._prefix_snapshot(),
                "worker_mask": mask.astype(bool).tolist(),
                "manager_mask": manager_mask.astype(bool).tolist(),
            })
            if eligible:
                self._prefix_account_clock()
                if self._prefix_lifetime_microsteps >= self.prefix_max_microsteps:
                    raise RuntimeError("prefix handoff reached an unclosed physical budget boundary")
                win = self.oe._win
                opening_reward = float(win["R"])
                self.stats["prefix_raw_reward"] += opening_reward
                self._prefix_attempt["raw_reward"] += opening_reward
                self.stats["prefix_handoffs"] += 1
                self.stats["windows"] += 1
                self.stats["fresh"] += 1
                self.stats["dive_live_windows"] = int(self.stats.get("dive_live_windows", 0)) + 1
                self._prefix_finish_attempt(
                    "handoff", window_id=int(win["window_id"]),
                    opening_reward=opening_reward,
                    learner_observation_sha256=hashlib.sha256(observation.tobytes()).hexdigest(),
                    handoff_window={key: copy.deepcopy(win[key]) for key in
                                    ("window_id", "opt", "t0", "R", "W", "worker_wage", "beats")},
                    learner_reward_credited=0.0)
                self._prefix_handed_off = True
                return observation, None
        while ending is None:
            if observation is None:
                observation = self.oe._worker_policy_observation(self.policy_observation_view)
            action = self._prefix_policy(observation, self.oe._worker_masks())
            outcome = self.oe._win_step_worker(action)
            if outcome.reason is not None:
                ending = outcome
            observation = None
        return None, self.oe._win_end(ending.reason)

    def _advance_earned_prefix(self) -> _AdvanceOutcome:
        if not self._alive or self._prefix_attempt is None:
            raise RuntimeError("earned prefix requires a real reset before advancement")
        reward = 0.0
        extras = []
        try:
            while True:
                self._prefix_validate_limits(active=True)
                start_clock = int(self.oe.env._resource_actual_microsteps)
                option = self._mgr_choose()
                if self.oe._win is not None:
                    raise RuntimeError("earned prefix cannot replace an open window")
                if option == DIVE:
                    observation, closed = self._prefix_open_dive()
                    if observation is not None:
                        return _AdvanceOutcome(
                            observation, reward + float(self._prefix_attempt["opening_reward"]),
                            False, False, tuple(extras),
                            opening_reward=float(self._prefix_attempt["opening_reward"]),
                            opening_recovery_action=(self.oe._win["last_recovery_action"]
                                                     if self.oe._win["recovery_actions"] else None))
                    extra, base_info, done, trunc = closed
                else:
                    original_workers = self.oe._workers
                    if option == FARM:
                        self.oe._workers = {**original_workers, FARM: self._prefix_policy}
                    try:
                        _, _, done, trunc, info = self.oe.step(option)
                    finally:
                        self.oe._workers = original_workers
                    extra = info["option_extra"]
                    base_info = self._option_base_info(info)
                self._prefix_record_window(extra, base_info, start_clock)
                reward += float(extra["R"])
                extras.append(extra)
                self._log(extra, fast_forward=True)
                if self._prefix_lifetime_microsteps >= self.prefix_max_microsteps:
                    self.stats["prefix_budget_exhausted"] = True
                    self._prefix_failure = "microstep_budget_exhausted"
                    self._prefix_finish_attempt(
                        "microstep_budget_exhausted", terminal_info=base_info,
                        terminal_option_extra=extra, base_done=bool(done), base_truncated=bool(trunc))
                    self._alive = False
                    raise self._prefix_budget_exception(self._prefix_failure)
                raw = self.oe.env._raw
                if int(raw["dungeon_level"]) >= 2 and not raw.get("is_set_level"):
                    self._prefix_failure = "early_main_depth_before_handoff"
                    self._prefix_finish_attempt(self._prefix_failure, terminal_info=base_info,
                                                terminal_option_extra=extra)
                    self._alive = False
                    raise RuntimeError(self._prefix_failure)
                if done or trunc:
                    self._alive = False
                    self.stats["ff_terminals"] += 1
                    terminated, truncated, timeout = self._worker_boundary(done, trunc, extra, base_info)
                    self._prefix_finish_attempt(
                        "prefix_terminal", reason=extra.get("reason"),
                        terminated=terminated, truncated=truncated,
                        timeout_without_progress=timeout, terminal_info=base_info,
                        terminal_option_extra=extra)
                    return _AdvanceOutcome(
                        None, reward, terminated, truncated, tuple(extras),
                        terminal_base_info=base_info,
                        terminal_death_reward=self._terminal_death_reward(terminated, truncated, base_info),
                        timeout_without_progress=timeout)
        except Exception as exc:
            self._prefix_abort_error(exc)
            raise

    def _advance_to_learning_window(self) -> _AdvanceOutcome:
        """推进到下一可学习 FARM，并完整返回中间后果，绝不滚新局。

        DIVE/RESUPPLY、课程跳过的 dry FARM、以及下一 FARM 开窗反射均是
        上一 worker 动作之后发生的环境动力学。其原始奖励在这里累计供
        守恒审计；``step()`` 会在同一 transition 返回真实下一态/终止，
        但不会把这些冻结策略收益记给 worker 动作。
        """
        if (self._earned_prefix_enabled()
                and not self._prefix_handed_off):
            return self._advance_earned_prefix()
        if not self._alive:
            return _AdvanceOutcome(None, 0.0, False, False, ())
        if self.oe._win is not None:
            raise RuntimeError("推进下一 FARM 前当前窗口必须已收束")
        reward = 0.0
        extras = []
        while True:
            opt = self._mgr_choose()
            if opt == FARM:
                dry = self.oe.exhausted
                if dry and self._skip_dry_draw():
                    # v26 绿洲:干层复访窗(榨干旗在位)由脚本内环代跑——
                    # 与 DIVE/RESUPPLY 同路,簿记同源,不成为学习 episode
                    # (E1 ⑤A:以 p_skip 概率代跑;p=1.0 恒代跑 ≡ 原 skip_dry=True)
                    _, r, done, trunc, info = self.oe.step(FARM)
                    extra = info["option_extra"]
                    reward += float(r)
                    extras.append(extra)
                    self._log(extra, fast_forward=True)
                    self.stats["ff_dry"] += 1
                    if done or trunc:
                        self._alive = False
                        self.stats["ff_terminals"] += 1
                        base_info = self._option_base_info(info)
                        terminated, truncated, timeout = (
                            self._worker_boundary(
                                done, trunc, extra, base_info))
                        return _AdvanceOutcome(
                            self._policy_observation(self.oe._worker_obs()),
                            reward, terminated, truncated,
                            tuple(extras),
                            terminal_base_info=base_info,
                            terminal_death_reward=self._terminal_death_reward(
                                terminated, truncated, base_info),
                            timeout_without_progress=timeout)
                    continue
                self.oe._win_begin(FARM)
                # 上一 worker 提案若触发 fuse，拒绝拍本身仍不执行动作；
                # 恢复拍在这个新经理窗口开头显式入账，再做反射排水。
                ending = self.oe._consume_fuse_recovery()
                if ending is None or ending.reason is None:
                    ending = self.oe._drain()
                if ending is None:
                    # 开窗反射发生在工人取得下一观测之前；第一拍 worker
                    # 会从当前 W 起算，故必须在这里归给上一 transition，
                    # 否则这部分奖励仍会落入账外。
                    opening_reward = float(self.oe._win["R"])
                    reward += opening_reward
                    self.stats["windows"] += 1
                    self.stats["dry" if dry else "fresh"] += 1
                    return _AdvanceOutcome(
                        self._policy_observation(self.oe._worker_obs()),
                        reward, False, False,
                        tuple(extras), opening_reward=opening_reward,
                        opening_recovery_action=(
                            self.oe._win["last_recovery_action"]
                            if self.oe._win["recovery_actions"] else None))
                # 排水拍直接终结了窗口(死亡/榨干/CAP……):按快进窗入册,继续找
                extra, base_info, done, trunc = self.oe._win_end(ending.reason)
                reward += float(extra["R"])
                extras.append(extra)
                self._log(extra, fast_forward=True)
                if done or trunc:
                    self._alive = False
                    self.stats["ff_terminals"] += 1
                    terminated, truncated, timeout = (
                        self._worker_boundary(
                            done, trunc, extra, base_info))
                    return _AdvanceOutcome(
                        self._policy_observation(self.oe._worker_obs()),
                        reward, terminated, truncated,
                        tuple(extras), terminal_base_info=base_info,
                        terminal_death_reward=self._terminal_death_reward(
                            terminated, truncated, base_info),
                        timeout_without_progress=timeout)
            elif (opt == DIVE
                  and getattr(self, "learning_window_scope", "farm-only")
                  in _LIVE_DIVE_WINDOW_SCOPES):
                # R13 教室改革:DIVE 窗与 FARM 同构 live 开窗,工人逐微拍
                # 进梯度;开窗恢复拍/反射排水仍归上一 transition,与 FARM
                # 同法同账。RESUPPLY 与跳干窗保持脚本快进(下方 else)。
                self.oe._win_begin(DIVE)
                ending = self.oe._consume_fuse_recovery()
                if ending is None or ending.reason is None:
                    ending = self.oe._drain()
                if ending is None:
                    opening_reward = float(self.oe._win["R"])
                    reward += opening_reward
                    self.stats["windows"] += 1
                    self.stats["fresh"] += 1
                    self.stats["dive_live_windows"] = (
                        int(self.stats.get("dive_live_windows", 0)) + 1)
                    return _AdvanceOutcome(
                        self._policy_observation(self.oe._worker_obs()),
                        reward, False, False,
                        tuple(extras), opening_reward=opening_reward,
                        opening_recovery_action=(
                            self.oe._win["last_recovery_action"]
                            if self.oe._win["recovery_actions"] else None))
                # 排水拍直接终结了 DIVE 窗:按快进窗入册,继续找
                extra, base_info, done, trunc = self.oe._win_end(ending.reason)
                reward += float(extra["R"])
                extras.append(extra)
                self._log(extra, fast_forward=True)
                if done or trunc:
                    self._alive = False
                    self.stats["ff_terminals"] += 1
                    terminated, truncated, timeout = (
                        self._worker_boundary(
                            done, trunc, extra, base_info))
                    return _AdvanceOutcome(
                        self._policy_observation(self.oe._worker_obs()),
                        reward, terminated, truncated,
                        tuple(extras), terminal_base_info=base_info,
                        terminal_death_reward=self._terminal_death_reward(
                            terminated, truncated, base_info),
                        timeout_without_progress=timeout)
            else:
                _, r, done, trunc, info = self.oe.step(opt)   # 脚本内环,同源簿记
                extra = info["option_extra"]
                reward += float(r)
                extras.append(extra)
                self._log(extra, fast_forward=True)
                if done or trunc:
                    self._alive = False
                    self.stats["ff_terminals"] += 1
                    base_info = self._option_base_info(info)
                    terminated, truncated, timeout = (
                        self._worker_boundary(
                            done, trunc, extra, base_info))
                    return _AdvanceOutcome(
                        self._policy_observation(self.oe._worker_obs()),
                        reward, terminated, truncated,
                        tuple(extras),
                        terminal_base_info=base_info,
                        terminal_death_reward=self._terminal_death_reward(
                            terminated, truncated, base_info),
                        timeout_without_progress=timeout)

    def next_window(self):
        """手动推进到本局下一 FARM；局尽返回 None，绝不滚新局。

        新训练路径由 ``step()`` 直接消费完整 _AdvanceOutcome。此兼容口仅供
        示范/探针手动遍历；若调用方使用它，快进奖励会明确记入
        ``manual_ff_reward``，而不会伪装成已交给 PPO。
        """
        self.stats["manual_ff_calls"] = int(
            self.stats.get("manual_ff_calls", 0)) + 1
        outcome = self._advance_to_learning_window()
        self.stats["manual_ff_reward"] += float(outcome.reward)
        if float(outcome.terminal_death_reward) < 0.0:
            self.stats["manual_ff_terminal_deaths"] += 1
        if outcome.timeout_without_progress:
            timeout_components = (
                self._no_progress_timeout_failure_components(
                    True, outcome.terminal_base_info))
            self._record_no_progress_timeout(
                "manual_ff", timeout_components[2], credited=False)
        if outcome.terminated or outcome.truncated:
            return None
        return outcome.obs

    # ---- gym 接口 ----
    def reset(self, *, seed=None, options=None):
        # SB3/VecEnv 只会通过 reset(seed) 给环境定种。若不用这个
        # seed 重置工人的局种子采样器，那么 --seed 只能控制首局，
        # 后续自动滚局仍来自系统熵，整段训练无法复现。
        if self._earned_prefix_enabled():
            self._prefix_require_capacity()
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
            # E1③ 逐 env 播种钉死:p 抽签专用流以固定偏移自 episode 种子确定性派生
            # (SB3 以 seed+rank 逐 env 定种,复现主张由此成立;零染训练 RNG)。
            self._p_rng = _derive_p_skip_rng(seed)
        # Gym 允许调用方在 episode 尚未结束时主动 reset。此时 _win 仍在，
        # 若沿用同一底层局直接 next_window()，会覆盖未结算窗口并把经理状态、
        # 工资账本和新 Gym episode 串在一起。显式放弃旧局，从干净边界重开。
        interrupted_window = self.oe._win is not None
        if interrupted_window:
            self.stats["interrupted_resets"] = int(
                self.stats.get("interrupted_resets", 0)) + 1
            self._alive = False
        if seed is not None or not self._alive:
            self._new_episode(seed, options=options)
        empty_episodes = 0
        while True:
            outcome = self._advance_to_learning_window()
            self.stats["reset_ff_reward"] += float(outcome.reward)
            reset_terminal_death = float(outcome.terminal_death_reward)
            if reset_terminal_death < 0.0:
                reset_additional_terminal_death = (
                    self._additional_terminal_death_reward(
                        outcome.terminated,
                        outcome.truncated,
                        outcome.terminal_base_info))
                self.stats["reset_ff_terminal_deaths"] += 1
                self.stats["reset_ff_terminal_death_reward"] += (
                    reset_terminal_death)
                # reset 快进发生在 PPO 取得本局首态之前，不能把额外 cost
                # 记到任何 worker transition；这里只保留反事实审计账。
                self.stats[
                    "reset_ff_additional_terminal_death_reward"
                ] += reset_additional_terminal_death
            if outcome.timeout_without_progress:
                reset_timeout_components = (
                    self._no_progress_timeout_failure_components(
                        True, outcome.terminal_base_info))
                # reset 快进先于任一 Worker 观测/动作，只记反事实失败成本；
                # 绝不把它塞进下一局第一拍或凭空制造 transition。
                self._record_no_progress_timeout(
                    "reset_ff", reset_timeout_components[2],
                    credited=False)
            if (outcome.obs is not None
                    and not (outcome.terminated or outcome.truncated)):
                return outcome.obs, {
                    "episode_seed": self._episode_seed,
                    "window_id": int(self.oe._win["window_id"]),
                    "reset_fast_forward_reward": float(outcome.reward),
                    "reset_fast_forward_extras": list(outcome.extras),
                    "reset_recovery_action": outcome.opening_recovery_action,
                }
            self.stats["reseeds"] += 1    # 兜底滚局(显式种子局零 FARM 窗时也会走到——
            empty_episodes += 1
            if (not self._earned_prefix_enabled()
                    and empty_episodes >= _MAX_EMPTY_FARM_EPISODES):
                raise RuntimeError(
                    "冻结 manager 连续 "
                    f"{empty_episodes} 局未产生 FARM 窗口，拒绝无限滚局；"
                    "manager="
                    + (f"heuristic:{self.manager_heuristic}"
                       if self.mgr is None
                       else f"sha256={self.mgr.source_sha256}"))
            self._new_episode()           # BC 侧用 stats 断言封死示范池逃逸口)

    def _descend_escrow_settlement(self, d_before, close_reason) -> float:
        """R14.2 丁案:收窗结算,返回本 transition 应 vest 的金额(旗关恒 0)。

        非死亡收窗:先 vest 全部既有托管,再把本窗新最深层下楼费×fraction
        存入托管(候下一次非死亡收窗);死亡收窗:全额罚没。快进段死亡
        的罚没由 continuation 分支单独执行。只进 policy_reward,零触碰
        W 账本/恒等式/经理账本(R11 没收型阀门的工人侧移植)。
        """
        f = getattr(self, "descend_escrow_fraction", 0.0)
        if f <= 0.0:
            return 0.0
        pending = float(getattr(self, "_descend_escrow", 0.0))
        if close_reason == "death":
            if pending:
                self.stats["descend_escrow_forfeited"] = float(
                    self.stats.get(
                        "descend_escrow_forfeited", 0.0)) + pending
            self._descend_escrow = 0.0
            return 0.0
        vest = pending
        if vest:
            self.stats["descend_escrow_vested"] = float(
                self.stats.get("descend_escrow_vested", 0.0)) + vest
        max_after = max(1, int(getattr(
            self.oe.env, "_econ_episode_max_depth", 1)))
        new_escrow = 0.0
        if max_after > max(1, int(d_before)):
            _unit = float(getattr(getattr(
                self.oe.env, "reward_economy", None),
                "descend_unit", 0.0))
            _pw = float(getattr(self, "descend_escrow_power", 1.0))
            if _pw == 1.0:
                _floors = float(
                    sum(range(max(1, int(d_before)), max_after)))
            else:
                # R14.3 戊案:凸曲线 hazard pay——d^power 逐层计价
                _floors = float(sum(
                    d ** _pw
                    for d in range(max(1, int(d_before)), max_after)))
            new_escrow = f * _unit * _floors
            if (new_escrow > 0.0
                    and getattr(
                        self, "descend_escrow_readiness_gate", False)):
                # R15 修正案二:未达标下楼分文不入(角色面板在换层前后
                # 不变,收窗时按抵达层评估等价于下楼时评估)。
                raw = getattr(self.oe.env, "_raw", None) or {}
                # R17.0 修正案一:尺子选择器(v1 旧尺逐位;v2 与教练同尺)
                _table = getattr(
                    self, "descend_escrow_readiness_table", "v1")
                _ratio_v1 = readiness_power_ratio(raw, max_after)
                _ratio_v2 = readiness_power_ratio_v2(raw, max_after)
                _ratio = _ratio_v2 if _table == "v2" else _ratio_v1
                if getattr(self, "resource_protocol", "off") != "off":
                    receipts = getattr(self.oe.env, "_resource_transition_receipts", [])
                    receipt = next((item for item in reversed(receipts)
                                    if int(item.get("target_depth", -1)) == max_after), None)
                    if getattr(self, "resource_readiness_law", "veto-v1") == "coach-v03":
                        # Six-condition law (health excluded): a forced-unready
                        # descent is accepted by the engine but vests nothing.
                        _ratio = (1.0 if receipt and receipt.get("accepted")
                                  and receipt.get("pretransition_ready_law") else 0.0)
                        _table = "native-l2-town-v1/coach-v03"
                    else:
                        _ratio = 1.0 if receipt and receipt.get("accepted") and receipt.get("pretransition_ready") else 0.0
                        _table = "native-l2-town-v1"
                # R17.0 仪表(合议庭更正二.3):逐次过门遥测——两把尺的
                # 比值与面板(clvl/HP/AC/dmg)、抵达层、判定、应记托管额;
                # 有界 list 只记不裁,旗关(无战备门)不触。审计回调
                # (R13DiveAuditCallback)整体重写 r17_descend_gate.jsonl。
                _gate_log = self.stats.setdefault(
                    "descend_escrow_gate_log", [])
                if len(_gate_log) < _DESCEND_ESCROW_GATE_LOG_CAP:
                    _gate_log.append({
                        "depth": int(max_after),
                        "char_level": int(raw.get("char_level", 0)),
                        "max_hp": int(raw.get("max_hp", 0)),
                        "armor_class": int(raw.get("armor_class", 0)),
                        "dmg": (int(raw.get("item_max_damage", 0))
                                + int(raw.get("damage_mod", 0))),
                        "ratio_v1": round(float(_ratio_v1), 4),
                        "ratio_v2": round(float(_ratio_v2), 4),
                        "table": str(_table),
                        "passed": bool(_ratio >= 1.0),
                        "escrow": round(float(new_escrow), 3),
                    })
                if _ratio < 1.0:
                    self.stats["descend_escrow_unready_denied"] = float(
                        self.stats.get(
                            "descend_escrow_unready_denied", 0.0)
                    ) + new_escrow
                    new_escrow = 0.0
                else:
                    # R17.0:过门下楼计数(两把尺均计),供审计统计
                    # 「达标下楼次数」而非只看 vest 金额。
                    self.stats["descend_escrow_ready_vested_count"] = (
                        int(self.stats.get(
                            "descend_escrow_ready_vested_count", 0)) + 1)
        self._descend_escrow = new_escrow
        return vest

    def _forfeit_descend_escrow_on_ff_death(self) -> None:
        """R14.2:快进段(RESUPPLY/跳干窗)真实死亡同样罚没托管。"""
        if getattr(self, "descend_escrow_fraction", 0.0) <= 0.0:
            return
        pending = float(getattr(self, "_descend_escrow", 0.0))
        if pending:
            self.stats["descend_escrow_forfeited"] = float(
                self.stats.get(
                    "descend_escrow_forfeited", 0.0)) + pending
        self._descend_escrow = 0.0

    def _hp_economy_component(self, hp_before, belt_before) -> float:
        """R16(C7):血量经济分量(旗关恒 0.0)。

        −hp_loss_price × max(0, hp_before − hp_now)(非对称:只罚失血);
        +potion_pickup_bonus × max(0, belt_heals_now − belt_before)。
        只进 policy_reward,零触碰工资恒等式/经理账本。
        """
        price = float(getattr(self, "hp_loss_price", 0.0))
        bonus = float(getattr(self, "potion_pickup_bonus", 0.0))
        if price <= 0.0 and bonus <= 0.0:
            return 0.0
        raw = getattr(self.oe.env, "_raw", None) or {}
        component = 0.0
        if price > 0.0:
            lost = max(0, int(hp_before) - int(raw.get("hp", hp_before)))
            if lost:
                charge = -price * float(lost)
                component += charge
                self.stats["hp_loss_charged"] = float(
                    self.stats.get("hp_loss_charged", 0.0)) + charge
        if bonus > 0.0:
            gained = max(0, int(raw.get("belt_heals", belt_before))
                         - int(belt_before))
            if gained:
                credit = bonus * float(gained)
                component += credit
                self.stats["potion_pickup_credited"] = float(
                    self.stats.get("potion_pickup_credited", 0.0)) + credit
        return component

    def _depth_shaping_component(
            self, d_before, terminated, truncated) -> float:
        """R13.2 甲案:本 transition 的势函数塑形分量(旗关恒 0.0)。

        φ = unit×(本局最深层−1)(单调势,理论审查修正:live DIVE 窗豁免
        踏格保护后上行转移存在,current-level 势会重罚上行;单调势上行
        不扣、循环免疫、新进度才计酬)。F = φ(s′)−φ(s);true terminal
        退款 −φ(s_T) 使 φ(absorbing)=0;truncation 不退款(critic 学
        V−φ,自举精确对冲——审查证明这是唯一自洽组合)。只进
        policy_reward,零触碰工资恒等式/经理账本(合宪槽位:
        additional_terminal_death_cost 同层)。守恒:任何 true-terminal
        轨迹 ΣF ≡ 0。
        """
        unit = getattr(self, "depth_shaping_unit", 0.0)
        if unit <= 0.0:
            return 0.0
        max_after = max(1, int(getattr(
            self.oe.env, "_econ_episode_max_depth", 1)))
        delta = unit * float(max_after - max(1, int(d_before)))
        if delta != 0.0:
            self.stats["depth_shaping_credited"] = float(
                self.stats.get("depth_shaping_credited", 0.0)) + delta
        refund = 0.0
        if terminated and not truncated:
            refund = -unit * float(max_after - 1)
            if refund != 0.0:
                self.stats["depth_shaping_refunded"] = float(
                    self.stats.get("depth_shaping_refunded", 0.0)) + refund
        return delta + refund

    def step(self, action):
        if self._earned_prefix_enabled():
            if self._prefix_failure is not None or not self._prefix_handed_off:
                raise gym.error.ResetNeeded("earned prefix has not handed a live window to the learner")
            self._prefix_validate_limits(active=False)
        if self.oe._win is None:
            raise gym.error.ResetNeeded("WorkerWindowEnv.step() 前必须 reset() 开启 FARM 窗口")
        if not self.action_space.contains(action):
            raise ValueError(f"动作必须是 {self.action_space}中的整数，收到 {action!r}")
        win = self.oe._win
        window_id = int(win["window_id"])
        w_before = win["W"]
        # R13.2:塑形基准=本局最深层(单调势;旗关零读取,护 __new__ 壳
        # 与零触碰纪律)。reason-None 拍换层必触发 scene 收窗,故 Δφ≠0
        # 只出现在收窗 transition,两个 policy_reward 组装点覆盖全部情形。
        _d_before = (
            max(1, int(getattr(
                self.oe.env, "_econ_episode_max_depth", 1)))
            if (getattr(self, "depth_shaping_unit", 0.0) > 0.0
                or getattr(
                    self, "descend_escrow_fraction", 0.0) > 0.0)
            else 0)
        # R16(C7):血量经济基准(旗关零读取)
        if (getattr(self, "hp_loss_price", 0.0) > 0.0
                or getattr(self, "potion_pickup_bonus", 0.0) > 0.0):
            _raw0 = getattr(self.oe.env, "_raw", None) or {}
            _hp_before = int(_raw0.get("hp", 0))
            _belt_before = int(_raw0.get("belt_heals", 0))
        else:
            _hp_before = _belt_before = 0
        outcome = self.oe._win_step_worker(action)
        wage = win["W"] - w_before
        if (getattr(self, "learning_window_scope", "farm-only")
                in _LIVE_DIVE_WINDOW_SCOPES):
            # R13 分窗计量账(旗关路径零触碰):live 微拍按窗口模式分桶,
            # farm_live_steps + dive_live_steps ≡ num_timesteps 增量。
            _in_dive = int(win.get("opt", FARM)) == DIVE
            _step_key = "dive_live_steps" if _in_dive else "farm_live_steps"
            self.stats[_step_key] = int(self.stats.get(_step_key, 0)) + 1
            # R14 乙案审计:镜像 OptionsEnv 的下楼奖金保留累计额
            _kept = float(getattr(
                self.oe, "descend_bonus_kept_total", 0.0))
            if _kept:
                self.stats["descend_bonus_kept"] = round(_kept, 6)
            if _in_dive:
                if int(outcome.requested_action) == 11:
                    self.stats["dive_live_a11_requests"] = (
                        int(self.stats.get("dive_live_a11_requests", 0)) + 1)
                if outcome.executed_action == 11:
                    self.stats["dive_live_a11_executed"] = (
                        int(self.stats.get(
                            "dive_live_a11_executed", 0)) + 1)
        # The action belongs to the current rollout.  Its resulting state
        # belongs to the next one, so an armed curriculum switch commits here:
        # after action execution, but before observation construction,
        # next-window advancement, or a terminal return that VecEnv auto-resets.
        self._tick_skip_dry_schedule()
        overridden = bool(outcome.fuse_tripped)   # 兼容 BC 剔除键；现语义=拒绝提案
        obs = self._policy_observation(
            self.oe._worker_obs())           # 收窗前取终观测(窗口仍持有 τ)
        info = {
            "episode_seed": self._episode_seed,
            "window_id": window_id,
            "farm_window_end": False,
            "worker_wage": float(wage),
            "transition_reward": float(wage),
            "requested_action": int(outcome.requested_action),
            "executed_action": outcome.executed_action,
            "action_effect_audit": getattr(
                outcome, "action_effect_audit", None),
            "action14_audit": getattr(
                outcome, "action14_audit", None),
            "overridden": overridden,
            "fuse_tripped": bool(outcome.fuse_tripped),
            "fuse_requested_action": outcome.fuse_requested_action,
            "fast_forward_reward_credit_mode": (
                self.fast_forward_reward_credit),
            "additional_terminal_death_cost": (
                self.additional_terminal_death_cost),
            # R17.0 修正案一(合议庭更正二.2):policy_reward 三个新法分量
            # 的逐 transition 回执(旗关恒 0.0),供 leashed_ppo formal-PG
            # 守门员核对 reward == wage + timeout + shaping + vest + hp_econ
            # (求和顺序与下方两处组装点一致,浮点逐位可核)。
            "worker_depth_shaping_reward": 0.0,
            "worker_descend_escrow_vest": 0.0,
            "worker_hp_economy_reward": 0.0,
        }
        self._attach_no_progress_timeout_audit(
            info, False, (0.0, 0.0, 0.0),
            credit_mode=getattr(self, "no_progress_timeout_credit",
                                "death-equivalent"))
        _hp_econ = self._hp_economy_component(_hp_before, _belt_before)
        info["worker_hp_economy_reward"] = float(_hp_econ)
        if outcome.reason is None:
            if _hp_econ:
                info["transition_reward"] = float(wage) + _hp_econ
                return obs, float(wage) + _hp_econ, False, False, info
            return obs, wage, False, False, info
        extra, base_info, done, trunc = self.oe._win_end(outcome.reason)
        self._log(extra, fast_forward=False)
        if (getattr(self, "learning_window_scope", "farm-only")
                in _LIVE_DIVE_WINDOW_SCOPES):
            # R13 审计:live 收窗记模式;live DIVE 收窗按 reason 分桶
            # (descend=课堂成功/stall=空转/death=深层死亡)。
            info["window_mode"] = (
                ("farm", "dive", "resupply")[int(extra.get("opt", FARM))])
            if int(extra.get("opt", FARM)) == DIVE:
                _reason_key = {
                    "descend": "dive_live_descends",
                    "stall": "dive_live_stalls",
                    "death": "dive_live_deaths",
                }.get(extra.get("reason"))
                if _reason_key is not None:
                    self.stats[_reason_key] = (
                        int(self.stats.get(_reason_key, 0)) + 1)
        # R14.2 丁案:收窗即结算(非死亡 vest 既有托管+新下楼入托管;
        # 死亡罚没)。旗关恒 0.0,零触碰。
        _escrow_vest = self._descend_escrow_settlement(
            _d_before, extra.get("reason"))
        info["worker_descend_escrow_vest"] = float(_escrow_vest)
        info.update({
            "farm_window_end": True,
            "option_extra": extra,
            "fast_forward_reward": 0.0,
            "credited_fast_forward_reward": 0.0,
            "existing_terminal_death_reward": 0.0,
            "additional_terminal_death_reward": 0.0,
            "total_terminal_death_reward": 0.0,
            "credited_fast_forward_terminal_death_reward": 0.0,
            "fast_forward_extras": [],
            "next_window_id": None,
            "next_window_opening_reward": 0.0,
            "next_window_recovery_action": None,
            "terminal_option_extra": extra if (done or trunc) else None,
            "terminal_base_info": None,
        })
        if done or trunc:
            self._alive = False
            terminated, truncated, timeout_without_progress = (
                self._worker_boundary(done, trunc, extra, base_info))
            # 此死亡分量已经由 _win_step_worker 计入 wage。只公开审计，
            # 不走 fast-forward credit，避免同一底层死亡拍重复计罚。
            existing_terminal_death = self._terminal_death_reward(
                terminated, truncated, base_info)
            additional_terminal_death = (
                self._additional_terminal_death_reward(
                    terminated, truncated, base_info))
            total_terminal_death = (
                existing_terminal_death + additional_terminal_death)
            timeout_components = (
                self._no_progress_timeout_failure_components(
                    timeout_without_progress, base_info))
            self._attach_no_progress_timeout_audit(
                info, timeout_without_progress, timeout_components,
                credit_mode=getattr(self, "no_progress_timeout_credit",
                                    "death-equivalent"))
            info.update({
                "existing_terminal_death_reward": existing_terminal_death,
                "additional_terminal_death_reward": additional_terminal_death,
                "total_terminal_death_reward": total_terminal_death,
            })
            if existing_terminal_death < 0.0:
                self.stats["direct_terminal_deaths"] += 1
                self.stats[
                    "direct_existing_terminal_death_reward"
                ] += existing_terminal_death
                self.stats[
                    "direct_additional_terminal_death_reward"
                ] += additional_terminal_death
            self.stats["additional_terminal_death_reward"] += (
                additional_terminal_death)
            if timeout_without_progress:
                self._record_no_progress_timeout(
                    "direct", timeout_components[2], credited=True)
            _shaping = self._depth_shaping_component(
                _d_before, terminated, truncated)
            info["worker_depth_shaping_reward"] = float(_shaping)
            policy_reward = (
                float(wage)
                + additional_terminal_death
                + timeout_components[2]
                + _shaping
                + _escrow_vest
                + _hp_econ
            )
            info["transition_reward"] = policy_reward
            # R17.1(W/P 复核 H3/M1):统一死亡分量回执键 + 收据 schema 标记,
            # 令 leashed_ppo 守门员对每一笔 transition 做六项恒等式
            # (wage + death + timeout + shaping + vest + hp_econ,顺序同上)。
            info["worker_credited_terminal_death_reward"] = float(
                additional_terminal_death)
            info["worker_policy_reward_receipt"] = "v1"
            self._attach_terminal_base_info(
                info, base_info,
                terminated=terminated, truncated=truncated)
            return obs, policy_reward, terminated, truncated, info

        # 自然 FARM 边界不是 Gym terminal。冻结 manager/script 的全部后果
        # 在本次 transition 内推进并返回，原始奖励只进审计账；随后返回
        # 真实下一 FARM 观测供价值函数 bootstrap。
        continuation = self._advance_to_learning_window()
        self.stats["transition_ff_reward"] += float(continuation.reward)
        ff_terminal_death = float(continuation.terminal_death_reward)
        if (not np.isfinite(ff_terminal_death)
                or ff_terminal_death > 0.0
                or (ff_terminal_death != 0.0
                    and not continuation.terminated)):
            raise RuntimeError(
                "快进终局死亡分量异常: "
                f"reward={ff_terminal_death},"
                f"terminated={continuation.terminated},"
                f"truncated={continuation.truncated}")
        if ff_terminal_death < 0.0:
            # R14.2 丁案:快进段真实死亡同罚没托管(本 transition 收窗
            # 时新存入的部分)——死亡即失,与 live 死亡同法。
            self._forfeit_descend_escrow_on_ff_death()
        additional_terminal_death = (
            self._additional_terminal_death_reward(
                continuation.terminated,
                continuation.truncated,
                continuation.terminal_base_info))
        total_terminal_death = (
            ff_terminal_death + additional_terminal_death)
        timeout_components = (
            self._no_progress_timeout_failure_components(
                continuation.timeout_without_progress,
                continuation.terminal_base_info))
        credited_existing_terminal_death = (
            ff_terminal_death
            if self.fast_forward_reward_credit == "terminal-death-only"
            else 0.0
        )
        credited_terminal_death = (
            credited_existing_terminal_death
            + additional_terminal_death)
        self.stats["transition_ff_terminal_death_reward"] += (
            ff_terminal_death)
        if ff_terminal_death < 0.0:
            self.stats["transition_ff_terminal_deaths"] += 1
            self.stats[
                "transition_ff_additional_terminal_death_reward"
            ] += additional_terminal_death
        self.stats["credited_ff_terminal_death_reward"] += (
            credited_terminal_death)
        self.stats["additional_terminal_death_reward"] += (
            additional_terminal_death)
        if continuation.timeout_without_progress:
            self._record_no_progress_timeout(
                "transition_ff", timeout_components[2], credited=True)
        self._attach_no_progress_timeout_audit(
            info, continuation.timeout_without_progress,
            timeout_components,
            credit_mode=getattr(self, "no_progress_timeout_credit",
                                "death-equivalent"))
        info.update({
            "fast_forward_reward": float(continuation.reward),
            # 冻结 manager/script 不是 worker 动作。保留完整原始奖励供
            # 守恒审计，策略只领取真实终局死亡分量或 no-progress 失败
            # 成本；否则 a0 等到 exhausted 便可把 DIVE 战斗/下楼正收益
            # 记在自己名下。
            "credited_fast_forward_reward": (
                credited_terminal_death + timeout_components[2]),
            "existing_terminal_death_reward": ff_terminal_death,
            "additional_terminal_death_reward": additional_terminal_death,
            "total_terminal_death_reward": total_terminal_death,
            "credited_fast_forward_terminal_death_reward": (
                credited_terminal_death),
            "fast_forward_extras": list(continuation.extras),
            "next_window_opening_reward": float(continuation.opening_reward),
            "next_window_recovery_action": continuation.opening_recovery_action,
            "terminal_option_extra": (
                continuation.extras[-1]
                if ((continuation.terminated or continuation.truncated)
                    and continuation.extras)
                else None),
        })
        _shaping = self._depth_shaping_component(
            _d_before,
            continuation.terminated, continuation.truncated)
        info["worker_depth_shaping_reward"] = float(_shaping)
        policy_reward = (
            float(wage)
            + credited_terminal_death
            + timeout_components[2]
            + _shaping
            + _escrow_vest
            + _hp_econ
        )
        info["transition_reward"] = policy_reward
        # R17.1(W/P 复核 H3/M1):同上,快进分支的死亡分量是 credited_terminal_death。
        info["worker_credited_terminal_death_reward"] = float(credited_terminal_death)
        info["worker_policy_reward_receipt"] = "v1"
        if continuation.terminated or continuation.truncated:
            self._attach_terminal_base_info(
                info, continuation.terminal_base_info,
                terminated=continuation.terminated,
                truncated=continuation.truncated)
            return (continuation.obs, policy_reward, continuation.terminated,
                    continuation.truncated, info)
        if continuation.obs is None or self.oe._win is None:
            raise RuntimeError("自然 FARM 边界后未得到下一学习态，也未得到真实终止")
        info["next_window_id"] = int(self.oe._win["window_id"])
        return continuation.obs, policy_reward, False, False, info

    def action_masks(self) -> np.ndarray:
        if self.oe._win is None:
            raise gym.error.ResetNeeded("WorkerWindowEnv.action_masks() 前必须 reset()")
        return self.oe._worker_masks()

    def close(self):
        self.oe.close()
