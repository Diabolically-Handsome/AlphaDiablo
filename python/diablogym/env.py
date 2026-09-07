"""DiabloGymEnv —— Gymnasium 包装(v0:结构化向量观测 + 离散动作)。

观测向量(float32,长度 12 + K*4 + 2*(2R+1)² + 9,R=5 时共 295):
  [hp/maxhp, mana/maxmana, xp(log1p/10), gold/1000, char_level/50,
   dungeon_level/16, player_x/112, player_y/112,
   可见可达怪数/50, 最近可见可达怪距离/30(无怪=1),
   下一项必需主线目标方向 dx/56, dy/56(普通层即下行楼梯;无则 0,0)]
  + K 个最近可见可达怪物的 (dx/20, dy/20, hp/max_hp, 1存在标志)
  + 11×11 局部地图两通道(可走性、可见可达怪物占位)——run4 教训:没有空间感知,
    奖励再好也是"盲人拿完美账本"(隔墙锁定、穿墙塑形、找不到房门)
  + [腰带治疗药数/8 + 空槽数/128,
     最近可见可达地面治疗药 dx/20, dy/20(截断至 ±1), 存在标志]
    (v4 在原 belt scalar 内无歧义编码两个 0..8 整数域：治疗药仍占 1/8
    主刻度，空槽只占其下 1/16 子刻度，最大扰动 0.0625。由此喝药/捡药
    的全部资源前置条件均可观测，而不改变 295 维 shape)
  + [护甲值/50(截断至 1), 最近可见可达整套战力升级装备 dx/20,
     dy/20(截断至 ±1), 存在标志]
    (存在标志由原生 PlanGearUpgrade 生成：候选先鉴定并模拟占用槽替换、
    双戒指选择及单/双手切换；只有替换后整套保守战力严格增长才为 1)
  + [min(2, 角色等级/max(1,地牢层数))/2]——v19 强弱仪表

动作(Discrete(15)):
  0      明确等待:取消遗留寻路/追击/destAction；攻击动画立即中止，已提交的
         单格移动在本次 step 的计费 settle 拍内自然收尾，但不会继续上一
         动作的长路径或重复攻击(v4)
  1-8    朝八方向走一格(寻路)
  9      交战宏:锁定最近的可见可达怪物持续追击,直到它死/自己死/换层/
         超时(≤10 拍)
         (v2 教训:单拍攻击会被下一个走位动作打断,策略学不会"坚持进攻")
  10     探索宏:若存在通关必需剧情目标则 fail-closed 等待/交还经理，
         绝不代替 action11 越权推进；否则走向 25×25 视野内最近的
         "可走且未踏足"边疆点;发现猎物
         (最近怪 ≤6 格)立即交还控制权;顺路开启通往新区域的普通闭门，
         无边疆时也可处理挡路桶，但绝不踩任何上/下楼 trigger
         (run5 教训:出生区无可达怪时,反应式策略不会"换个房间找")
  11     主线推进宏(v11+):优先完成法杖台/法杖/L15 任务入口/Vile 书与
         法阵/L16 机关的严格白名单,无待办时再走下行或任务返回触发点。
         与探索宏不同,发现猎物**不**打断——这是策略主动选择的撤离/换层键
         (困局的逃生舱 + 清层后的下一章按钮);12 拍后控制权自然归还。
         (v10 教训:困局是死的 0,多给时间没用——得给一扇门)
  12     喝药键(v12):腰带有治疗类药水就喝一瓶(引擎手柄快捷键同路),
         没有则为空拍。v12 曾刻意不把腰带药数放进观测(保 286 维历代
         可复评),结果 99.5% 的按键落在空腰带上(教训十一"瓶盲"),
         v13 起腰带药数与最近地面药方向入观测。
  13     捡药宏(v13):沿观测绑定的 radius-12 固定快照逐个走 4 向安全步，
         遇普通门先在相邻格开启；必须精确站到目标格才提交原生拾取，原生
         再校验物品身份与腰带容量。无目标/无空槽或安全路径不完整则等待。
  14     捡装备宏(v14):使用同一逐步安全路径走向 PlanGearUpgrade 认可的
         严格整套升级；精确到位后原生重验物品身份与换装计划，复制身体槽
         并原子替换（含占用槽、武器/盾牌、双戒指与双手切换）。CalcPlrInv
         后若保守整套战力未严格增长则完整回滚；不走 AutoEquip 的背包回退。

动作掩码与路径安全:
  action_masks 只动态约束 9/12/13/14：9 要有 radius-12 快照内可见且局部
  可接敌的编码怪；12 要有腰带治疗药且确实掉血；13 要有腰带空槽和有效
  治疗药目标；14 要有有效整套升级目标。其余键保持合法。掩码只消灭确定性
  空按，不承诺宏能避开之后发生的动态阻挡、受击或时间上限。
  dual Worker 执行生成观测时安装的同一快照；旧 295/298 视图只在 mask/step
  边界抓取只读快照。控制器只用该 radius-12 视图规划 4 向相邻步，hazard、
  explosive softwall 与受保护剧情格一律不可穿；物品宏还要求路径完整抵达
  精确目标格，禁止把原生近距离命令当作跨危险格的二次寻路捷径。

决策边界(v4):除死亡/胜利/总步数边界外，step 返回时玩家必须处于
PM_STAND，且 future==tile、walkpath/destAction 均为空。结清走格、受击等
隐藏动画所用的 engine beats 全部计入 micro steps、奖励差分和 Options τ；
295/298/303 维观测因而无需加入不可见的执行态，也不会发生“同一 tile、
不同 pending 命令”这一类状态别名。若 max_steps 恰在动画中耗尽，不能越
预算结算；该边界 fail-closed 为 terminal 并禁止价值 bootstrap。只有 idle
的 max_steps 边界才是标准 truncation。

action9/10 是可恢复的有记忆控制器：失败目标轮转、粘性 frontier 与已踏足
集合属于宏的内部调度状态，不是策略可选择的另一种动作；它们不得改变
action mask，且失败表耗尽后必须开启下一轮，不能永久删除合法目标。形式上，
若把 15 个键当作完全原子的 flat-MDP 动作，这些调度表（连同游戏未观测区域）
仍是部分可观测状态；严格展开会需要目标方向及探索地图新通道并使旧 295 维
权重失效。当前接口把它们明确归入 option controller，只把会改变经理 mask
或 FARM 收窗时刻的 wrapper 交权钟/预算计数直接放入 298/303 维策略观测；
“下一格是否首次踏足”仍随探索控制器的 visited map 属于这项明确的部分可观测
抽象，而不是伪称能由 295 维完整重建。

奖励(v4,状态势函数守恒):
  +Φ(q_before)-Φ(q_after), Φ(q)=0.75q-0.125q²
                                              q 为怪物本 lifetime 已付最低
                                              HP/maxHP；只为新最低血线付一次，
                                              回血重打不重复。满血到 0 的伤害
                                              塑形恒为 0.625，不随切刀、settle
                                              拍数或攻击节奏改变
  +1.0 * Δmonster_kill_total                  原生单调击杀总账；宏内生成又死亡、
                                              或击杀后同拍换场景也不会漏记
  +0.01 * ΔXP                                  真实目标(升级)
  +8.0  * Δ地牢层                               扁平模式；深度阶梯模式对跨过的
                                              每个 N→N+1 结算 8N
  +min(1, max(0, Δgear_utility)/4096)          武器伤害、命中、AC、抗性、
                                              格挡、词缀与耐久共用原生整套
                                              uint32 总账；只奖增长，负 Δ 不罚
  +0.005 * 固定存活目标的 player-only 接近差   走远时可为负
  -0.002  同场景请求 action0 等待
  -0.002  同场景原生未执行的任意非零请求        已真实执行但原地的攻击/喝药/
                                              拾取操作不收这笔固定罚分
  -2.0 死亡(死亡阶梯模式为 -8×当前层)  +10.0 通关
  历史教训:v0 的掉血惩罚→面壁塌缩;v1 的"怪贴脸也计分"→站桩钓鱼。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import math
import os
import pathlib
import shutil
import tempfile
from collections import deque
from contextlib import contextmanager

import gymnasium as gym
import numpy as np

from . import bridge, nav
from .controller_wire import *  # noqa: F403 - single schema source, re-exported

# 八方向(等距地牢的 tile 坐标系)
_DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
_K_MONSTERS = 8
_MAP_RADIUS = 5  # 11×11 局部地图

# 换层奖金单价(_reward 的 Δ地牢层项;v23 起为具名常量——工人工资剥薪
# 需要在包装器侧按同一公式反算,数字只许存在一份)
DESCEND_UNIT = 8.0
GEAR_COMBAT_UTILITY_REWARD_SCALE = 4096.0
GEAR_COMBAT_UTILITY_REWARD_CAP = 1.0
STALL_ACTION_REWARD = -0.002


@dataclass(frozen=True)
class TerminalDeathRewardSpec:
    """Immutable source of truth for the native terminal-death component."""

    flat_cost: float
    ladder_cost_per_depth: float


TERMINAL_DEATH_REWARD_SPEC = TerminalDeathRewardSpec(
    flat_cost=2.0,
    ladder_cost_per_depth=8.0,
)


@dataclass(frozen=True)
class RewardEconomy:
    """R10 经济法案:全部工资常量的单一真源(v1=历史逐位不变)。

    v2(R10 深度经济,主席批文 2026-08-27,记账全额归经理):
      - A 下楼奖金:descend_unit 上调(仍按 v17 递进公式 8×N 结构,换单价);
      - B 深度乘数:farm 收入(xp+击杀)×(1+kill_depth_beta×(d-1));
      - C 主席版死亡罚金:flat + c0×gamma^(d-1),随深度递减(拧反 v1 的
        ladder 递增恐惧项);
      - D 反躺平:同层滞留超 idle_threshold_steps 后 farm 收入 ×idle_factor
        (抵达新的本局最深层即重置);
      - E1 装备重定价:scale 降 / cap 升,一次真实升级≈数只怪。
    """

    name: str
    descend_unit: float
    kill_depth_beta: float
    death_flat: float
    death_c0: float
    death_gamma: float
    death_uses_ladder_spec: bool
    gear_scale: float
    gear_cap: float
    idle_threshold_steps: int
    idle_kill_factor: float
    # R11 主席版杀怪锁:>0 时,下楼奖金记入"未解锁"托管;该层击杀满
    # 此数解锁;死于解锁前全额罚没(gamma=1.0 下与到账托管数学等价)。
    # 0 = 关闭(v1/v2 语义不变)。
    descend_vest_kills: int
    # R16 修宪(C8,2026-09-01)反躺平重定义,两个默认关字段:
    #   idle_counts_micro_beats: D 条款 idle 钟按底层微拍(self._steps 增量)
    #     计数,而非按 _reward 调用次数(=决策/宏)计数;
    #   idle_reset_on_kill: 本决策发生原生击杀即把 idle 钟清零(先按清零前
    #     的钟结算本决策的 farm 乘数,再清零;新最深层清零照旧)。
    # v1/v2/v3/v3b 均取默认 False → 旧行为逐位不变。
    idle_counts_micro_beats: bool = False
    idle_reset_on_kill: bool = False


REWARD_ECONOMY_V1 = RewardEconomy(
    name="v1",
    descend_unit=DESCEND_UNIT,
    kill_depth_beta=0.0,
    death_flat=TERMINAL_DEATH_REWARD_SPEC.flat_cost,
    death_c0=0.0,
    death_gamma=1.0,
    death_uses_ladder_spec=True,
    gear_scale=GEAR_COMBAT_UTILITY_REWARD_SCALE,
    gear_cap=GEAR_COMBAT_UTILITY_REWARD_CAP,
    idle_threshold_steps=0,
    idle_kill_factor=1.0,
    descend_vest_kills=0,
)

REWARD_ECONOMY_V2 = RewardEconomy(
    name="v2",
    # rev2(G0 校准 2026-08-27):rev1 探针 DIVE-FARM=-29.3 未翻符——
    # B 乘数无差别肥了守旧派。下楼奖 24→48、beta 0.5→0.25、
    # 反躺平 900→300 步。终值以冻结预注册为准。
    descend_unit=48.0,
    kill_depth_beta=0.25,
    death_flat=2.0,
    death_c0=24.0,
    death_gamma=0.7,
    death_uses_ladder_spec=False,
    gear_scale=1024.0,
    gear_cap=12.0,
    idle_threshold_steps=300,
    idle_kill_factor=0.5,
    descend_vest_kills=0,
)

# R11 试跑版(2026-08-28 深夜主席令「今晚先试跑一轮」):v2 + 杀怪锁 K=3。
# 单变量纪律:除 vest_kills 外与 v2 逐字相同,便于明晨干净归因。
REWARD_ECONOMY_V3 = RewardEconomy(
    name="v3",
    descend_unit=48.0,
    kill_depth_beta=0.25,
    death_flat=2.0,
    death_c0=24.0,
    death_gamma=0.7,
    death_uses_ladder_spec=False,
    gear_scale=1024.0,
    gear_cap=12.0,
    idle_threshold_steps=300,
    idle_kill_factor=0.5,
    descend_vest_kills=3,
)

# R11 试跑二(二分搜索:K=3 端点证伪潜行,退一格):v3 仅改 vest_kills=1。
REWARD_ECONOMY_V3B = RewardEconomy(
    name="v3b",
    descend_unit=48.0,
    kill_depth_beta=0.25,
    death_flat=2.0,
    death_c0=24.0,
    death_gamma=0.7,
    death_uses_ladder_spec=False,
    gear_scale=1024.0,
    gear_cap=12.0,
    idle_threshold_steps=300,
    idle_kill_factor=0.5,
    descend_vest_kills=1,
)

# R16 修宪 v4(C8 判词:v2 的 D 条款按决策计数、只在新最深层清零、击杀不
# 清零,>300 后 farm 收入减半——专打在本层磨等级的轨迹)。v4 = v2 全部参数
# (dataclasses.replace 保证逐字段同源)+ 反躺平重定义:idle 钟按底层微拍
# 计数且击杀发生即清零。v2 本身一字不动。
REWARD_ECONOMY_V4 = dataclasses.replace(
    REWARD_ECONOMY_V2,
    name="v4",
    idle_counts_micro_beats=True,
    idle_reset_on_kill=True,
)

REWARD_ECONOMIES = {
    "v1": REWARD_ECONOMY_V1,
    "v2": REWARD_ECONOMY_V2,
    "v3": REWARD_ECONOMY_V3,
    "v3b": REWARD_ECONOMY_V3B,
    "v4": REWARD_ECONOMY_V4,
}


def gear_combat_utility_value(raw, label: str) -> int:
    """Validate and return the native whole-loadout uint32 utility."""
    if "gear_combat_utility" not in raw:
        raise RuntimeError(
            f"{label} 缺少原生 gear_combat_utility")
    value = raw["gear_combat_utility"]
    if isinstance(value, (bool, np.bool_)):
        raise RuntimeError(
            f"{label}.gear_combat_utility 必须是非负整数")
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"{label}.gear_combat_utility 必须是非负整数") from exc
    if (
        not math.isfinite(numeric)
        or numeric != float(integer)
        or integer < 0
        or integer > 0xFFFFFFFF
    ):
        raise RuntimeError(
            f"{label}.gear_combat_utility 必须是 uint32")
    return integer


def gear_upgrade_reward_delta_component(delta: int) -> float:
    """Bounded shaping for one causally attributed native utility increase."""
    if isinstance(delta, (bool, np.bool_)) or not isinstance(
            delta, (int, np.integer)):
        raise RuntimeError("gear utility delta 必须是非负整数")
    delta = int(delta)
    if not 0 <= delta <= 0xFFFFFFFF:
        raise RuntimeError("gear utility delta 必须是 uint32")
    return min(
        GEAR_COMBAT_UTILITY_REWARD_CAP,
        float(delta) / GEAR_COMBAT_UTILITY_REWARD_SCALE,
    )


def gear_upgrade_reward_component(previous, current) -> float:
    """Bounded positive shaping from the native gear comparator's own ledger."""

    delta = max(
        0,
        gear_combat_utility_value(current, "current")
        - gear_combat_utility_value(previous, "previous"),
    )
    return gear_upgrade_reward_delta_component(delta)


@dataclass(frozen=True)
class _ControllerMonster:
    """One canonical visible monster row in the controller snapshot."""

    monster_id: int
    generation_key: tuple[int, int, int]
    monster_type: int
    x: int
    y: int
    future_x: int
    future_y: int
    hp: int
    max_hp: int
    ledger_low: int
    ledger_max: int
    blocked: bool
    visible: bool
    native_reachable: bool
    locally_engageable: bool
    dynamic_quantities: tuple[float, ...]
    combat_flags: int


@dataclass(frozen=True)
class _ControllerMissile:
    """One canonical radius-12 projectile slot, already float32-safe."""

    quantities: tuple[float, ...]


@dataclass(frozen=True)
class _ControllerItemTarget:
    x: int
    y: int
    active_id: int
    seed_hi: int
    seed_lo: int
    create_info: int
    base_id: int
    heal_kind: int
    gear_quantities: tuple[float, ...]
    effect_flags: int
    dam_ac_flags: int


@dataclass(frozen=True)
class _ControllerSnapshot:
    """Immutable state shared by one Worker observation and its macro action."""

    raw_identity: int
    steps: int
    scene: tuple[int, bool, int]
    player_x: int
    player_y: int
    player_future_x: int
    player_future_y: int
    walkable: tuple[int, ...]
    visible_monster: tuple[int, ...]
    physical_monster: tuple[int, ...]
    softwall: tuple[int, ...]
    closed_door: tuple[int, ...]
    hazard: tuple[int, ...]
    explosive_softwall: tuple[int, ...]
    visited_mask: tuple[int, ...]
    blocked_mask: tuple[int, ...]
    protected_mask: tuple[int, ...]
    visited_tiles: frozenset[tuple[int, int]]
    explore_blocked_targets: frozenset[tuple[int, int]]
    protected_tiles: frozenset[tuple[int, int]]
    sticky_target: tuple[int, int] | None
    monsters: tuple[_ControllerMonster, ...]
    monster_overflow_quantities: tuple[float, ...]
    candidates: tuple[_ControllerMonster, ...]
    missiles: tuple[_ControllerMissile, ...]
    missile_overflow_quantities: tuple[float, ...]
    heal_target: _ControllerItemTarget | None
    gear_target: _ControllerItemTarget | None
    belt_slot_kinds: tuple[int, ...]
    exact_quantities: tuple[float, ...]
    combat_quantities: tuple[float, ...]
    combat_effect_flags: int
    combat_dam_ac_flags: int
    equipped_quantities: tuple[float, ...]


def terminal_death_reward_component(
        *, dead: bool, dungeon_level, death_ladder: bool,
        economy: RewardEconomy = REWARD_ECONOMY_V1) -> float:
    """Return only the terminal-death component of the native reward.

    This pure function is shared by :class:`DiabloGymEnv` when it settles the
    real transition and by ``WorkerWindowEnv`` when it reconstructs that one
    component across a frozen manager/script boundary.  No XP, combat,
    movement, progression, or victory credit is included.

    economy=v1(默认)逐位复现历史行为;economy=v2 时死亡罚金改为主席版
    递减函数 flat + c0×gamma^(max(d,1)-1),death_ladder 分支被 v2 覆盖。
    """
    if not isinstance(dead, (bool, np.bool_)):
        raise TypeError(f"dead 必须是 bool，收到 {dead!r}")
    if not dead:
        return 0.0
    if not isinstance(death_ladder, (bool, np.bool_)):
        raise TypeError(
            f"death_ladder 必须是 bool，收到 {death_ladder!r}")
    if isinstance(dungeon_level, (bool, np.bool_)):
        raise ValueError(
            f"死亡终局 dungeon_level 必须是非负整数，收到 {dungeon_level!r}")
    try:
        depth = int(dungeon_level)
        numeric_depth = float(dungeon_level)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"死亡终局 dungeon_level 必须是非负整数，收到 {dungeon_level!r}"
        ) from exc
    if (not math.isfinite(numeric_depth)
            or numeric_depth != float(depth)
            or depth < 0):
        raise ValueError(
            f"死亡终局 dungeon_level 必须是非负整数，收到 {dungeon_level!r}")
    if not economy.death_uses_ladder_spec:
        # v2 主席版:一层死罚最重,越深越轻(2026-08-27 批文)。
        cost = economy.death_flat + (
            economy.death_c0
            * (economy.death_gamma ** (max(depth, 1) - 1)))
        return -float(cost)
    cost = (
        TERMINAL_DEATH_REWARD_SPEC.ladder_cost_per_depth * depth
        if bool(death_ladder)
        else TERMINAL_DEATH_REWARD_SPEC.flat_cost
    )
    return -float(cost)


_DEFAULT_ASSETS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "build" / "engine" / "devilutionx.app" / "Contents" / "Resources"
)
_TEMP_SAVE_LEGACY_PREFIX = "diablogym-saves-"
_TEMP_SAVE_PREFIX = "diablogym-saves-v2-"
_TEMP_SAVE_LOCK = ".owner.lock"
_TEMP_SAVE_REGISTRY_LOCK = (
    f".diablogym-saves-v2.{os.getuid() if hasattr(os, 'getuid') else 0}.global.lock"
)


def _scene_identity(raw) -> tuple[int, bool, int]:
    """主线深度相同的任务副本仍是另一张地图，不能跨图做差分。"""
    depth = int(raw["dungeon_level"])
    is_set = bool(raw.get("is_set_level", False))
    set_id = int(raw.get("set_level_id", 0)) if is_set else 0
    return depth, is_set, set_id


@contextmanager
def _temp_save_registry_lock(root: pathlib.Path):
    """Serialize scratch publication and reclamation across spawn workers.

    The registry file is intentionally persistent and outside the scratch glob.
    Unlinking it would let waiters retain the old inode while newcomers lock a
    new inode, splitting the critical section in two.
    """
    import fcntl

    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(root / _TEMP_SAVE_REGISTRY_LOCK, flags, 0o600)
    try:
        registry = os.fdopen(fd, "a+", encoding="utf-8")
    except Exception:
        os.close(fd)
        raise
    acquired = False
    try:
        fcntl.flock(registry.fileno(), fcntl.LOCK_EX)
        acquired = True
        yield
    finally:
        if acquired:
            fcntl.flock(registry.fileno(), fcntl.LOCK_UN)
        registry.close()


def _cleanup_stale_temp_save_dirs_locked(base: pathlib.Path) -> int:
    """Reclaim stale scratch while the caller holds the registry lock."""
    import fcntl

    removed = 0
    for candidate in base.glob(f"{_TEMP_SAVE_LEGACY_PREFIX}*"):
        if not candidate.is_dir():
            continue
        marker = candidate / _TEMP_SAVE_LOCK
        if not marker.is_file():
            # A v2 creator publishes the directory and owner marker while
            # holding the same registry lock.  Therefore a visible markerless
            # v2 directory can only be debris from a crashed creator.  Older
            # directories predate that invariant and remain fail-closed.
            if candidate.name.startswith(_TEMP_SAVE_PREFIX):
                try:
                    shutil.rmtree(candidate)
                    removed += 1
                except OSError:
                    pass
            continue
        try:
            owner = open(marker, "a+", encoding="utf-8")
        except OSError:
            continue
        acquired = False
        try:
            try:
                fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                continue
            try:
                shutil.rmtree(candidate)
                removed += 1
            except OSError:
                # The owner may have completed cleanup after our path lookup;
                # genuine I/O failures remain for a later startup to retry.
                pass
        finally:
            if acquired:
                fcntl.flock(owner.fileno(), fcntl.LOCK_UN)
            owner.close()
    return removed


def _cleanup_stale_temp_save_dirs(root: pathlib.Path | None = None) -> int:
    """回收被 SIGKILL 遗留、且已没有进程持锁的新式 scratch 目录。

    v2 的创建/清理由全局事务锁串行化，因此可回收崩溃留下的无 marker
    半成品；旧版本没有这项所有权证据，宁可保留也不猜。flock 由内核在
    进程死亡时释放，因此 PID 复用不会导致误删或漏删。
    """
    base = (pathlib.Path(root) if root is not None
            else pathlib.Path(tempfile.gettempdir()))
    with _temp_save_registry_lock(base):
        return _cleanup_stale_temp_save_dirs_locked(base)


def _create_locked_temp_save_dir():
    import fcntl

    base = pathlib.Path(tempfile.gettempdir())
    with _temp_save_registry_lock(base):
        _cleanup_stale_temp_save_dirs_locked(base)
        directory = tempfile.TemporaryDirectory(
            prefix=_TEMP_SAVE_PREFIX, dir=base)
        owner = open(pathlib.Path(directory.name) / _TEMP_SAVE_LOCK,
                     "a+", encoding="utf-8")
        try:
            fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner.write(f"pid={os.getpid()}\n")
            owner.flush()
            os.fsync(owner.fileno())
        except Exception:
            owner.close()
            directory.cleanup()
            raise
    return directory, owner


class DiabloGymEnv(gym.Env):
    metadata = {"render_modes": []}

    # R10 经济法案的类级默认:v1 = 历史逐位不变。既有测试/法证脚本常以
    # __new__ 裸构造实例直呼 _reward,类级默认保证该路径永远落在 v1;
    # 正常 __init__ 会按入参覆写实例属性。
    reward_economy = REWARD_ECONOMY_V1
    _econ_steps_on_level = 0
    _econ_idle_prev_steps = 0  # R16 v4:上次 _reward 时的 self._steps(微拍计数用)
    _econ_episode_max_depth = 0
    _econ_unvested = 0.0
    _econ_kills_on_floor = 0
    _econ_prev_epkills = 0

    # DevilutionX 是进程内全局单例，不是可重入的多实例引擎。同一
    # 进程可以顺序复用多个 wrapper，但不能交错 step；多环境必须用
    # SubprocVecEnv 之类的多进程方案。在这里显式记账，把静默串状态
    # 变成响亮的异常。
    _engine_initialized = False
    _engine_pid: int | None = None
    _engine_config: tuple[str, str, str, int] | None = None
    _active_token = None
    _temp_save_dir: tempfile.TemporaryDirectory | None = None
    _temp_save_lock = None
    _atfork_registered = False

    @classmethod
    def _after_fork_child(cls) -> None:
        """子进程不得析构父进程仍在使用的 scratch 或原生引擎。"""
        directory, cls._temp_save_dir = cls._temp_save_dir, None
        owner, cls._temp_save_lock = cls._temp_save_lock, None
        if directory is not None:
            finalizer = getattr(directory, "_finalizer", None)
            if finalizer is not None and finalizer.alive:
                finalizer.detach()  # 只取消子进程副本；不能 rmtree 父进程目录
        if owner is not None:
            owner.close()

    def __init__(
        self,
        assets_dir: str | None = None,
        save_dir: str | None = None,
        data_dir: str | None = None,
        ticks_per_step: int = 4,
        max_steps: int = 5000,
        start_in_dungeon: bool = False,
        include_raw: bool = True,
        descend_ladder: bool = False,
        death_ladder: bool = False,
        hero_class: int = 0,
        controller_snapshot_enabled: bool = False,
        descend_fallback_promotion: bool = True,
        reward_economy: str = "v1",
        explore_global_fallback: bool = False,
        progress_far_tiles: int = 0,
        explore_global_hunt: bool = False,
        resource_protocol: str = "off",
        resource_purchase_mode: str = "full",
        dive_blocker_recovery: str = "off",
        resource_ordinary_armor_scope: bool = False,
        resource_preserve_equipment_readiness: bool = False,
        resource_readiness_law: str = "veto-v1",
    ):
        super().__init__()
        from .resource_protocol import validate_resource_config, validate_ordinary_armor_scope
        self.resource_protocol, self.resource_purchase_mode = validate_resource_config(
            resource_protocol, resource_purchase_mode)
        self.resource_ordinary_armor_scope = validate_ordinary_armor_scope(
            self.resource_protocol, self.resource_purchase_mode, resource_ordinary_armor_scope)
        from .resource_protocol import validate_equipment_readiness_preservation
        self._resource_preserve_equipment_readiness = validate_equipment_readiness_preservation(
            self.resource_protocol, resource_preserve_equipment_readiness)
        # R17.1 ruling 3: readiness law (veto-v1 = R18-R23 native veto, bit-identical;
        # coach-v03 = six-condition coach + accounted forced descents).
        from .resource_protocol import validate_readiness_law
        self.resource_readiness_law = validate_readiness_law(
            self.resource_protocol, resource_readiness_law)
        if dive_blocker_recovery not in ("off", "adjacent-v1"):
            raise ValueError("dive_blocker_recovery must be off or adjacent-v1")
        if dive_blocker_recovery != "off" and self.resource_protocol == "off":
            raise ValueError("dive_blocker_recovery requires the resource protocol")
        self.dive_blocker_recovery = dive_blocker_recovery
        self._resource_pending_command = None
        self._resource_terminal_reason = None
        self._resource_dive_authority = False
        self._resource_actual_microsteps = 0
        self._resource_scene_ledgers = {}
        self._resource_max_main_depth = 0
        self._resource_transition_receipts = []
        self._resource_reward_depth_before = 0
        self._resource_step_bonus = 0.0
        # R16 修宪(2026-09-01)两把默认关的开关,默认值下代码路径逐位不变:
        #   explore_global_fallback(C5):a10 在 25×25 窗内无边疆/软墙候选时,
        #     不再直接 wait,而是全图 BFS 找最近的未踏足可达格或存活怪占位
        #     格,取其路径落在窗内的最远局部可达前缀点作本次 frontier;
        #   progress_far_tiles(C6):>0 时只有与既有"进展锚点"切比雪夫距离
        #     ≥ 该值的新格才推进 exploration_progress(踱步不算),0 = 旧法
        #     (任何首次踏足格都算)。
        if not isinstance(explore_global_fallback, (bool, np.bool_)):
            raise TypeError(
                "explore_global_fallback 必须是 bool，收到 "
                f"{explore_global_fallback!r}")
        self._explore_global_fallback = bool(explore_global_fallback)
        if (isinstance(progress_far_tiles, bool)
                or not isinstance(progress_far_tiles, (int, np.integer))
                or int(progress_far_tiles) < 0):
            raise ValueError(
                "progress_far_tiles 必须是非负整数(0=旧法)，收到 "
                f"{progress_far_tiles!r}")
        self._progress_far_tiles = int(progress_far_tiles)
        # R16 C5 附加变体(默认关,探针发现 fallback 单独几乎不触发):a10 在
        # 25×25 窗内没有任何可见怪时,不等局部边疆耗尽,直接全图 BFS 朝最近
        # 存活怪(含隔墙/未照亮的)推进(仍只取窗内航点、走既有逐步走格
        # 机制);全图无可达怪才回到局部边疆探索。
        if not isinstance(explore_global_hunt, (bool, np.bool_)):
            raise TypeError(
                "explore_global_hunt 必须是 bool，收到 "
                f"{explore_global_hunt!r}")
        self._explore_global_hunt = bool(explore_global_hunt)
        if (isinstance(ticks_per_step, bool)
                or not isinstance(ticks_per_step, (int, np.integer))
                or int(ticks_per_step) <= 0):
            raise ValueError(f"ticks_per_step 必须是正整数，收到 {ticks_per_step!r}")
        if (isinstance(max_steps, bool)
                or not isinstance(max_steps, (int, np.integer))
                or int(max_steps) <= 0):
            raise ValueError(f"max_steps 必须是正整数，收到 {max_steps!r}")
        if (isinstance(hero_class, bool)
                or not isinstance(hero_class, (int, np.integer))
                or int(hero_class) != 0):
            raise ValueError(
                "当前动作/自动加点契约只支持 hero_class=0(战士)；"
                f"收到 {hero_class!r}")
        if not isinstance(controller_snapshot_enabled, (bool, np.bool_)):
            raise TypeError(
                "controller_snapshot_enabled 必须是 bool，收到 "
                f"{controller_snapshot_enabled!r}")
        # E-fix 修 A(甲形态):避怪规划"成功但踏不上楼梯"时按秩比较提升
        # 宽容规划(_macro_progression 同款既有立法);False = 旧行为端点,
        # 系对照腿/旧档案位级重放专用。
        if not isinstance(descend_fallback_promotion, (bool, np.bool_)):
            raise TypeError(
                "descend_fallback_promotion 必须是 bool，收到 "
                f"{descend_fallback_promotion!r}")
        self._descend_fallback_promotion = bool(descend_fallback_promotion)

        assets = str(pathlib.Path(assets_dir or _DEFAULT_ASSETS).expanduser().resolve())
        data = str(pathlib.Path(
            data_dir
            or pathlib.Path.home() / "Library/Application Support/diasurgical/devilution"
        ).expanduser().resolve())
        cls = DiabloGymEnv
        pid = os.getpid()
        if not cls._atfork_registered and hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=cls._after_fork_child)
            cls._atfork_registered = True
        if cls._engine_initialized and cls._engine_pid != pid:
            raise RuntimeError(
                "DiabloGym 引擎已在父进程初始化，不能 fork 后复用；"
                "fork 子进程只能立即 exec/os._exit，多环境训练请使用 spawn")

        # bridge.init() 是一个长 C++ 调用，SIGINT 可能恰在它成功返回、
        # Python 尚未来得及写三个 class 属性时转成 KeyboardInterrupt。
        # 原生配置是提交事实源；每次构造都先据此修复可能被异步异常撕裂的
        # Python 账本，不能误删原生仍在使用的临时存档目录。
        native_config = bridge.engine_config()
        if native_config is not None:
            recovered = (str(native_config[0]), str(native_config[1]),
                         str(native_config[2]), int(native_config[3]))
            cls._engine_config = recovered
            cls._engine_pid = pid
            cls._engine_initialized = True
        elif cls._engine_initialized:
            raise RuntimeError(
                "DiabloGym Python/原生单例账本不一致：Python 标为已初始化，"
                "原生桥却未初始化")
        if cls._engine_initialized:
            if cls._engine_config is None:
                raise RuntimeError("DiabloGym 引擎单例状态损坏: 已初始化但配置缺失")
            _, old_saves, _, _ = cls._engine_config
            requested_save = (str(pathlib.Path(save_dir).expanduser().resolve())
                              if save_dir is not None else old_saves)
            requested = (assets, requested_save, data, int(hero_class))
            if requested != cls._engine_config:
                raise RuntimeError(
                    "DevilutionX 是进程内单例，不能用不同的 assets/save/data/"
                    f"hero_class 重复初始化；已有={cls._engine_config!r}, 请求={requested!r}")
            saves = old_saves
        else:
            if save_dir is not None:
                saves = str(pathlib.Path(save_dir).expanduser().resolve())
            else:
                # 存档 scratch 要活到进程内引擎退出，但不应像
                # mkdtemp 那样在每次多进程训练结束后永久遗留磁盘垃圾。
                cls._temp_save_dir, cls._temp_save_lock = _create_locked_temp_save_dir()
                saves = cls._temp_save_dir.name
            try:
                bridge.init(assets_dir=assets, save_dir=saves, data_dir=data,
                            hero_class=int(hero_class))
            except BaseException:
                # 若异步异常发生在原生提交之后，保留 native 与 scratch，
                # 并把 Python 账本补齐；只有原生确实未提交时才回滚磁盘。
                committed = bridge.engine_config()
                if committed is not None:
                    cls._engine_config = (
                        str(committed[0]), str(committed[1]),
                        str(committed[2]), int(committed[3]))
                    cls._engine_pid = pid
                    cls._engine_initialized = True
                elif save_dir is None and cls._temp_save_dir is not None:
                    cls._temp_save_dir.cleanup()
                    cls._temp_save_dir = None
                    if cls._temp_save_lock is not None:
                        cls._temp_save_lock.close()
                        cls._temp_save_lock = None
                raise
            cls._engine_config = (assets, saves, data, int(hero_class))
            cls._engine_pid = pid
            # 提交位必须最后写：若 SIGINT 落在配置/PID 两次赋值之间，下一次
            # 构造会从原生 engine_config 恢复；反过来先置 True 会把缺失 PID
            # 误判成 fork，甚至进不到恢复逻辑。
            cls._engine_initialized = True

        self.ticks_per_step = int(ticks_per_step)
        self.max_steps = int(max_steps)
        self.start_in_dungeon = start_in_dungeon
        self.include_raw = include_raw
        # v17 深水区:下楼奖金层数递进(N→N+1 付 8×N;False = v6-v16 的扁平 8.0,
        # 旧章金标准的世界规则不动)
        self.descend_ladder = descend_ladder
        # v18:死亡成本与阶梯同步定价(死在 N 层罚 8×N;False = 恒 -2.0)。
        # 教训十六:阶梯 8/16/24 对上死亡 -2,冲刺期望值稳赚(+5.8),
        # "活着抵达"必须在拍卖行里赢过"摸到深度"
        self.death_ladder = death_ladder
        # R10 经济法案:v1 = 全部历史常量逐位不变(默认);v2 = 深度经济
        # (A/B/C/D/E1,主席批文 2026-08-27)。字符串入口,单一真源在
        # REWARD_ECONOMIES。
        if reward_economy not in REWARD_ECONOMIES:
            raise ValueError(
                f"reward_economy 必须是 {sorted(REWARD_ECONOMIES)},"
                f" 收到 {reward_economy!r}")
        self.reward_economy = REWARD_ECONOMIES[reward_economy]
        # D 条款状态:同层滞留步数与本局最深层(reset 时清零)。
        self._econ_steps_on_level = 0
        self._econ_episode_max_depth = 0
        # 旧 295/298 视图不消费 controller wire，默认不为每个决策额外抓取
        # 25×25 地图，也不要求旧 raw 具备新协议字段。dual Worker 在 wrapper
        # 构造时显式开启；宏动作仍会在按键边界做一次局部、非缓存快照。
        self._controller_snapshot_enabled = bool(
            controller_snapshot_enabled)
        side = 2 * _MAP_RADIUS + 1
        self.action_space = gym.spaces.Discrete(15)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(12 + _K_MONSTERS * 4 + 2 * side * side + 9,), dtype=np.float32,
        )  # +9 = v13 药 4 维 + v14 装备 4 维 + v19 强弱仪表 1 维
        self._token = object()
        self._raw = None
        self._native_generation: int | None = None
        self._episode_seed: int | None = None
        self._episode_ended = True
        self._steps = 0
        self._ep_kills = 0
        self._ep_start_xp = 0
        self._visited: set[tuple[int, int]] = set()
        # 单局单调探索进展钟：任一动作首次踏入新格或 action10 真正打开
        # 一处软墙各加 1。Options FARM 用前后差重置“无进展”钟，避免
        # teacher(action10) 与 learned worker(方向键/追击) 使用两套终止
        # 语义。换场景不回绕，reset 才归零。
        self._exploration_progress = 0
        self._softwalls_opened = 0
        # action10 的边疆目标必须跨宏保持。若每 12 拍都按“离当前位置
        # 最近”重选，迷宫分叉会形成稳定 2-cycle：走向 A 后 B 更近，
        # 走向 B 后 A 又更近，尚有大量房间却最终报 exhausted。
        self._explore_target: tuple[int, int] | None = None
        self._explore_blocked_targets: set[tuple[int, int]] = set()
        # v4:原生 FindPath 的 reachable 是一张瞬时几何快照；怪物/玩家
        # 动画、动态占位仍可能让某个目标在实际追击中无进展。记住本场景
        # 已证失败的目标，使下一次 action9 优先轮转，而不是永远被稳定的
        # ActiveMonsters/id 顺序吸回同一个等距目标。
        self._engage_blocked_keys: set[tuple[int, int, int]] = set()
        # v4:每个怪物 lifetime 本场景已经付过钱的最低血线。键是
        # (active slot, rndItemSeed hi, lo)，不能只用会被运行时刷怪复用
        # 的 slot id；值为 (lowest_hp, denominator_max_hp)。
        self._combat_hp_floor: dict[
            tuple[int, int, int], tuple[int, int]
        ] = {}
        self._controller_snapshot: _ControllerSnapshot | None = None

    @property
    def resource_preserve_equipment_readiness(self):
        """Constructor-only protocol identity; reset reapplies the same native flag."""
        return getattr(self, "_resource_preserve_equipment_readiness", False)

    def _validate_native_resource_flags(self, raw):
        from .resource_protocol import (
            validate_native_armor_scope, validate_native_equipment_readiness_preservation)
        validate_native_armor_scope(raw, getattr(self, "resource_ordinary_armor_scope", False))
        validate_native_equipment_readiness_preservation(
            raw, getattr(self, "resource_preserve_equipment_readiness", False))
        from .resource_protocol import validate_native_readiness_law
        validate_native_readiness_law(raw, getattr(self, "resource_readiness_law", "veto-v1"))

    # ---------- gymnasium API ----------

    def reset(self, *, seed: int | None = None, options=None):
        if (DiabloGymEnv._engine_initialized
                and DiabloGymEnv._engine_pid != os.getpid()):
            raise RuntimeError(
                "禁止在 fork 子进程 reset 父进程已初始化的 DevilutionX；"
                "多环境训练必须使用 spawn")
        super().reset(seed=seed)
        # R10 经济 v2 的 D 条款状态按局清零(v1 下为无害恒零)。
        self._econ_steps_on_level = 0
        self._econ_idle_prev_steps = 0  # R16 v4 微拍基线(仅 v4 读取)
        self._econ_episode_max_depth = 0
        # R11 杀怪锁状态按局清零。
        self._econ_unvested = 0.0
        self._econ_kills_on_floor = 0
        self._econ_prev_epkills = 0
        actual_seed = seed if seed is not None else int(self.np_random.integers(2**31))
        actual_seed = int(actual_seed)
        if not 0 <= actual_seed <= np.iinfo(np.uint32).max:
            raise ValueError(f"seed 必须在 uint32 范围 [0, 2**32-1] 内，收到 {actual_seed}")
        try:
            DiabloGymEnv._active_token = self._token
            self._configure_native_resource_protocol()
            self._raw = bridge.reset(seed=actual_seed)
            self._validate_native_resource_flags(self._raw)
            self._native_generation = int(bridge.episode_generation())
            if self.start_in_dungeon:
                # 城镇布局固定,脚本化走到教堂楼梯(约 500-900 tick,~0.05s)
                if getattr(self, "resource_preserve_equipment_readiness", False):
                    self._raw = nav.descend_to_dungeon(
                        bridge, raw_validator=self._validate_native_resource_flags)
                else:
                    self._raw = nav.descend_to_dungeon(bridge)
                self._validate_native_resource_flags(self._raw)
            if self._native_monster_kill_delta(
                    self._raw, self._raw) is None:
                raise RuntimeError(
                    "当前原生桥缺少 monster_kill_total；"
                    "无法完整统计宏内 spawn→death")
            self._steps = 0
            self._episode_seed = actual_seed
            self._episode_ended = False
            self._ep_kills = 0
            self._ep_start_xp = int(self._raw["xp"])
            self._visited = {(self._raw["player_x"], self._raw["player_y"])}
            # R16 C6 进展锚点与足迹同源起步(仅 progress_far_tiles>0 时读取)。
            self._resource_actual_microsteps = 0
            self._resource_terminal_reason = None
            self._resource_pending_command = None
            self._resource_dive_authority = False
            self._resource_scene_ledgers = {}
            self._resource_max_main_depth = int(self._raw["dungeon_level"])
            self._resource_transition_receipts = []
            self._resource_reward_depth_before = self._resource_max_main_depth
            self._resource_step_bonus = 0.0
            self._progress_anchors = set(self._visited)
            self._exploration_progress = 0
            self._softwalls_opened = 0
            self._explore_target = None
            self._explore_blocked_targets = set()
            self._engage_blocked_keys = set()
            self._reset_combat_ledger(self._raw)
            self._controller_snapshot = (
                self._capture_controller_snapshot(self._raw)
                if self._controller_snapshot_enabled else None
            )
            obs = self._vectorize(self._raw)
            info = self._info(self._raw)
        except BaseException:
            # 导航/观测构造也是 reset 事务的一部分；中途失败时不得
            # 留下一个看似可 step 的半初始化 episode。
            try:
                bridge.end_game()
            except Exception:
                pass
            if DiabloGymEnv._active_token is self._token:
                DiabloGymEnv._active_token = None
            self._raw = None
            self._native_generation = None
            self._episode_seed = None
            self._episode_ended = True
            self._exploration_progress = 0
            self._softwalls_opened = 0
            self._explore_target = None
            self._explore_blocked_targets = set()
            self._engage_blocked_keys = set()
            self._combat_hp_floor = {}
            self._controller_snapshot = None
            raise
        return obs, info

    @staticmethod
    def _policy_monsters(raw) -> list[dict]:
        """策略可消费的怪物子集；全层 monsters 仍留给奖励/击杀账。

        缺 visible/reachable 的分支只服务旧的纯 Python 合成 fixture；
        v4 原生 raw 始终显式提供两字段。
        """
        return [
            m for m in raw.get("monsters", ())
            if bool(m.get("visible", True)) and bool(m.get("reachable", True))
        ]

    @staticmethod
    def _policy_floor_items(raw, flag: str) -> list[dict]:
        if flag not in {"heal", "gear"}:
            raise ValueError(f"未知地面物品策略标志: {flag!r}")
        return [
            it for it in raw.get("floor_items", ())
            if bool(it.get(flag))
            and bool(it.get("visible", True))
            and bool(it.get("reachable", True))
        ]

    @staticmethod
    def _belt_free_slots(raw) -> int:
        if "belt_free_slots" in raw:
            return max(0, int(raw["belt_free_slots"]))
        # 仅兼容 protocol-v3 合成 fixture/旧桥的近似；v4 原生字段
        # 才能区分“2 瓶药+6 件其他腰带物品”和“2 瓶药+6 个空格”。
        return max(0, 8 - int(raw.get("belt_heals", 0)))

    @staticmethod
    def _reflex_eligible(raw) -> bool:
        """Shared emergency handoff predicate for every multi-beat macro."""
        return (
            2 * int(raw.get("hp", 0))
            < max(1, int(raw.get("max_hp", 0)))
            and int(raw.get("belt_heals", 0)) > 0
        )

    @classmethod
    def _belt_observation_scalar(cls, raw) -> float:
        """在既有一个 scalar 中无歧义公开治疗药数与真实空槽数。

        heals/free 都是原生腰带的 0..8 整数。heals 使用历史 1/8 主刻度，
        free 使用 1/128 子刻度，因此相邻 heals 桶之间仍留有 1/16 间隔；
        旧策略输入最多只偏移 8/128=0.0625，shape 与旧主刻度不变。
        """
        heals = min(8, max(0, int(raw.get("belt_heals", 0))))
        free = min(8, cls._belt_free_slots(raw))
        return heals / 8.0 + free / 128.0

    @staticmethod
    def _controller_binary_channel(values, name: str) -> tuple[int, ...]:
        expected = CONTROLLER_SNAPSHOT_CELLS
        try:
            frozen = tuple(int(value) for value in values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"controller snapshot {name} 通道不可整数化") from exc
        if len(frozen) != expected or any(value not in (0, 1) for value in frozen):
            raise RuntimeError(
                f"controller snapshot {name} 必须是 {expected} 个 0/1，"
                f"收到 len={len(frozen)}")
        return frozen

    @staticmethod
    def _controller_uint(value, *, name: str, bits: int) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise RuntimeError(
                f"controller snapshot {name} 必须是 uint{bits} 整数")
        try:
            integer = int(value)
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"controller snapshot {name} 必须是 uint{bits} 整数") from exc
        if (
            not math.isfinite(numeric)
            or numeric != float(integer)
            or not 0 <= integer < (1 << bits)
        ):
            raise RuntimeError(
                f"controller snapshot {name} 必须是 uint{bits} 整数")
        return integer

    @classmethod
    def _monster_generation_key(cls, monster) -> tuple[int, int, int]:
        """Stable native monster lifetime key, never serialized to policy."""
        if not isinstance(monster, dict):
            raise RuntimeError("monster generation 需要 dict")
        monster_id = cls._controller_uint(
            monster.get("id"), name="monster.id", bits=16)
        seed_hi = cls._controller_uint(
            monster.get("rnd_item_seed_hi"),
            name=f"monster[{monster_id}].rnd_item_seed_hi",
            bits=16,
        )
        seed_lo = cls._controller_uint(
            monster.get("rnd_item_seed_lo"),
            name=f"monster[{monster_id}].rnd_item_seed_lo",
            bits=16,
        )
        return monster_id, seed_hi, seed_lo

    @staticmethod
    def _controller_int32_words(value, *, name: str) -> tuple[float, float]:
        if isinstance(value, (bool, np.bool_)):
            raise RuntimeError(
                f"controller snapshot {name} 必须是 int32 整数")
        try:
            integer = int(value)
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"controller snapshot {name} 必须是 int32 整数") from exc
        if (
            not math.isfinite(numeric)
            or numeric != float(integer)
            or not -(1 << 31) <= integer < (1 << 32)
        ):
            raise RuntimeError(
                f"controller snapshot {name} 必须落在 32-bit word 范围")
        unsigned = integer & 0xFFFFFFFF
        return (
            float((unsigned >> 16) & 0xFFFF) / 65536.0,
            float(unsigned & 0xFFFF) / 65536.0,
        )

    @staticmethod
    def _controller_target(
        raw,
        flag: str,
        *,
        player_x: int,
        player_y: int,
        reachable_tiles: frozenset[tuple[int, int]],
    ) -> _ControllerItemTarget | None:
        targets = [
            item for item in raw.get("floor_items", ())
            if bool(item.get(flag)) and bool(item.get("visible", True))
            and (
                int(item["x"]), int(item["y"])
            ) in reachable_tiles
            and abs(int(item["x"]) - player_x)
            <= CONTROLLER_SNAPSHOT_RADIUS
            and abs(int(item["y"]) - player_y)
            <= CONTROLLER_SNAPSHOT_RADIUS
        ]
        if not targets:
            return None

        def active_id(item) -> int:
            if "active_id" not in item:
                raise RuntimeError(
                    f"controller snapshot {flag} 目标缺少 active_id")
            return DiabloGymEnv._controller_uint(
                item["active_id"], name=f"{flag}.active_id", bits=7)

        target = min(
            targets,
            key=lambda item: (
                max(
                    abs(int(item["x"]) - player_x),
                    abs(int(item["y"]) - player_y),
                ),
                int(item["x"]),
                int(item["y"]),
                active_id(item),
            ),
        )
        selected_id = active_id(target)
        identity = {}
        for field in ("seed_hi", "seed_lo", "create_info", "base_id"):
            if field not in target:
                raise RuntimeError(
                    f"controller snapshot {flag} 目标缺少 {field}")
            identity[field] = DiabloGymEnv._controller_uint(
                target[field],
                name=f"{flag}.{field}",
                bits=16,
            )
        if flag == "heal":
            if "heal_kind" not in target:
                raise RuntimeError(
                    "controller snapshot heal 目标缺少 heal_kind")
            heal_kind = DiabloGymEnv._controller_uint(
                target["heal_kind"], name="heal.heal_kind", bits=3)
            if not 1 <= heal_kind <= CONTROLLER_SNAPSHOT_INSTANT_HEAL_KINDS:
                raise RuntimeError(
                    "controller snapshot heal_kind 必须是 [1,4] 整数")
            gear_quantities: tuple[float, ...] = ()
            effect_flags = 0
            dam_ac_flags = 0
        else:
            missing = [
                field for field in CONTROLLER_SNAPSHOT_GEAR_FIELDS
                if field not in target
            ]
            if missing:
                raise RuntimeError(
                    "controller snapshot gear 目标缺字段:"
                    + ",".join(missing))
            bool_fields = {
                "identified", "effects_active", "stat_usable"}
            gear_quantities = tuple(
                (
                    float(bool(target[field]))
                    if field in bool_fields else float(target[field])
                ) / scale
                for field, scale in zip(
                    CONTROLLER_SNAPSHOT_GEAR_FIELDS,
                    CONTROLLER_SNAPSHOT_GEAR_SCALES,
                    strict=True,
                )
            )
            if not all(math.isfinite(value) for value in gear_quantities):
                raise RuntimeError(
                    "controller snapshot gear 精确量含 NaN/Inf")
            effect_flags = DiabloGymEnv._controller_uint(
                target.get("effect_flags"),
                name="gear.effect_flags",
                bits=CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS,
            )
            dam_ac_flags = DiabloGymEnv._controller_uint(
                target.get("effect_dam_ac_flags"),
                name="gear.effect_dam_ac_flags",
                bits=CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS,
            )
            heal_kind = 0
        return _ControllerItemTarget(
            x=int(target["x"]),
            y=int(target["y"]),
            active_id=selected_id,
            seed_hi=identity["seed_hi"],
            seed_lo=identity["seed_lo"],
            create_info=identity["create_info"],
            base_id=identity["base_id"],
            heal_kind=heal_kind,
            gear_quantities=gear_quantities,
            effect_flags=effect_flags,
            dam_ac_flags=dam_ac_flags,
        )

    def _capture_controller_snapshot(self, raw) -> _ControllerSnapshot:
        """Read the radius-12 controller state exactly once at a decision edge."""
        if not isinstance(raw, dict):
            raise RuntimeError("controller snapshot 需要 active native raw")
        radius = CONTROLLER_SNAPSHOT_RADIUS
        side = CONTROLLER_SNAPSHOT_SIDE
        local_map = bridge.local_map(radius=radius)
        walkable = self._controller_binary_channel(
            local_map.get("walkable", ()), "walkable")
        physical_monster = self._controller_binary_channel(
            local_map.get("monster", ()), "physical_monster")
        softwall = self._controller_binary_channel(
            local_map.get("door", ()), "softwall")
        closed_door = self._controller_binary_channel(
            local_map.get("closed_door", local_map.get("door", ())),
            "closed_door",
        )
        hazard = self._controller_binary_channel(
            local_map.get("hazard", ()), "hazard")
        explosive_softwall = self._controller_binary_channel(
            local_map.get("explosive_softwall", ()),
            "explosive_softwall",
        )

        px, py = int(raw["player_x"]), int(raw["player_y"])
        player_future_x = int(raw.get("future_x", px))
        player_future_y = int(raw.get("future_y", py))
        def in_window(point) -> bool:
            return (
                abs(int(point[0]) - px) <= radius
                and abs(int(point[1]) - py) <= radius
            )

        # planner 与 wire 必须消费完全同一片 25×25 事实。若保留窗外一格
        # 的 visited/protected halo，near_visited 膨胀或门的 unseen-side
        # 判定会让 radius13 历史改变 radius12 边缘动作，却不出现在观测。
        protected = frozenset(
            (int(x), int(y))
            for x, y in self._explore_protected_tiles(raw)
            if in_window((x, y))
        )
        visited = frozenset(
            (int(x), int(y))
            for x, y in getattr(self, "_visited", set())
            if in_window((x, y))
        )
        explore_blocked = frozenset(
            (int(x), int(y))
            for x, y in getattr(self, "_explore_blocked_targets", set())
            if in_window((x, y))
        )

        def tile_mask(tiles) -> tuple[int, ...]:
            return tuple(
                1 if (px + dx, py + dy) in tiles else 0
                for dy in range(-radius, radius + 1)
                for dx in range(-radius, radius + 1)
            )

        def map_index(x: int, y: int) -> int:
            return (y - py + radius) * side + (x - px + radius)

        raw_monsters = raw.get("monsters")
        if not isinstance(raw_monsters, (list, tuple)):
            raise RuntimeError(
                "controller snapshot 缺少原生 monsters 列表")
        visible_local_monsters: list[dict] = []
        for monster_index, monster in enumerate(raw_monsters):
            if not isinstance(monster, dict):
                raise RuntimeError(
                    "controller snapshot monsters"
                    f"[{monster_index}] 必须是 dict")
            missing_identity = [
                field for field in (
                    "id", "x", "y", "hp", "max_hp",
                    "visible", "reachable",
                    "rnd_item_seed_hi", "rnd_item_seed_lo",
                )
                if field not in monster
            ]
            if missing_identity:
                raise RuntimeError(
                    "controller snapshot monsters"
                    f"[{monster_index}] 缺字段:"
                    + ",".join(missing_identity))
            # Hidden/native-inactive monsters may remain in the immutable raw
            # collision snapshot for macro planning, but neither identity nor
            # coordinates may enter an actor row or actor occupancy channel.
            if (
                bool(monster["visible"])
                and in_window((monster["x"], monster["y"]))
            ):
                visible_local_monsters.append(monster)

        visible_monster_values = [0] * CONTROLLER_SNAPSHOT_CELLS
        for monster in visible_local_monsters:
            visible_monster_values[
                map_index(int(monster["x"]), int(monster["y"]))
            ] = 1
        visible_monster = tuple(visible_monster_values)

        def local_reachable(
            *,
            allow_softwalls: bool,
            avoid_monsters: bool,
        ) -> frozenset[tuple[int, int]]:
            start = (px, py)
            reached = {start}
            queue = deque([start])
            while queue:
                cx, cy = queue.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    point = (cx + dx, cy + dy)
                    if (
                        point in reached
                        or not in_window(point)
                        or point in protected
                    ):
                        continue
                    i = map_index(*point)
                    if hazard[i] or explosive_softwall[i]:
                        continue
                    if (
                        not walkable[i]
                        and not (allow_softwalls and softwall[i])
                    ):
                        continue
                    if avoid_monsters and physical_monster[i]:
                        continue
                    reached.add(point)
                    queue.append(point)
            return frozenset(reached)

        engage_reachable = local_reachable(
            allow_softwalls=False, avoid_monsters=True)
        pickup_reachable = local_reachable(
            allow_softwalls=True, avoid_monsters=False)

        def locally_engageable(monster) -> bool:
            target = (
                int(monster.get("future_x", monster["x"])),
                int(monster.get("future_y", monster["y"])),
            )
            # The native controller attack guard and _macro_engage both bind
            # the command to this observation window.  A monster may still
            # have its visible current tile on the radius-12 edge while its
            # already-committed future tile lies at radius 13; keep its row,
            # but do not advertise an action-9 candidate that execution must
            # reject.
            if not in_window(target):
                return False
            return any(
                max(abs(rx - target[0]), abs(ry - target[1])) <= 1
                for rx, ry in engage_reachable
            )

        ranked_monsters = [
            (monster, locally_engageable(monster))
            for monster in visible_local_monsters
        ]
        ranked_monsters.sort(key=lambda entry: (
            not entry[1],
            max(
                abs(
                    int(entry[0].get("future_x", entry[0]["x"]))
                    - player_future_x
                ),
                abs(
                    int(entry[0].get("future_y", entry[0]["y"]))
                    - player_future_y
                ),
            ),
            int(entry[0]["id"]),
        ))
        selected_monsters = ranked_monsters[
            :CONTROLLER_SNAPSHOT_MONSTER_LIMIT]
        overflow_monsters = ranked_monsters[
            CONTROLLER_SNAPSHOT_MONSTER_LIMIT:]

        blocked_keys = getattr(self, "_engage_blocked_keys", set())
        ledger = getattr(self, "_combat_hp_floor", {})
        monster_rows = []
        candidates = []
        for monster, is_locally_engageable in selected_monsters:
            monster_id = int(monster["id"])
            generation_key = self._monster_generation_key(monster)
            hp = max(0, int(monster["hp"]))
            max_hp = max(1, int(monster["max_hp"]))
            ledger_low, ledger_max = ledger.get(
                generation_key, (hp, max_hp))
            missing_dynamics = [
                field
                for field in (
                    CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_FIELDS
                    + CONTROLLER_SNAPSHOT_MONSTER_INT32_FIELDS
                )
                if field not in monster
            ]
            if missing_dynamics:
                raise RuntimeError(
                    "controller snapshot monster"
                    f"[{monster_id}] 缺动态战斗字段:"
                    + ",".join(missing_dynamics))
            dynamic_quantities_list = list(
                (
                    float(bool(monster[field]))
                    if field in {"anim_petrified", "is_invalid"}
                    else float(monster[field])
                ) / scale
                for field, scale in zip(
                    CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_FIELDS,
                    CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_SCALES,
                    strict=True,
                )
            )
            for field in CONTROLLER_SNAPSHOT_MONSTER_INT32_FIELDS:
                dynamic_quantities_list.extend(
                    self._controller_int32_words(
                        monster[field],
                        name=f"monster[{monster_id}].{field}",
                    )
                )
            dynamic_quantities = tuple(dynamic_quantities_list)
            if not all(
                math.isfinite(value) for value in dynamic_quantities
            ):
                raise RuntimeError(
                    "controller snapshot monster"
                    f"[{monster_id}] 动态战斗量含 NaN/Inf")
            combat_flags = self._controller_uint(
                monster.get("combat_flags"),
                name=f"monster[{monster_id}].combat_flags",
                bits=CONTROLLER_SNAPSHOT_MONSTER_FLAG_BITS,
            )
            row = _ControllerMonster(
                monster_id=monster_id,
                generation_key=generation_key,
                monster_type=int(monster.get("type", 0)),
                x=int(monster["x"]),
                y=int(monster["y"]),
                future_x=int(monster.get("future_x", monster["x"])),
                future_y=int(monster.get("future_y", monster["y"])),
                hp=hp,
                max_hp=max_hp,
                ledger_low=max(0, int(ledger_low)),
                ledger_max=max(1, int(ledger_max)),
                blocked=generation_key in blocked_keys,
                visible=True,
                native_reachable=bool(monster["reachable"]),
                locally_engageable=is_locally_engageable,
                dynamic_quantities=dynamic_quantities,
                combat_flags=combat_flags,
            )
            monster_rows.append(row)
            if is_locally_engageable:
                # Action 9 must never target a row that the actor did not
                # receive.  Engageable rows sort first, so this remains useful
                # even in pathological over-capacity scenes.
                candidates.append(row)

        overflow_values = (
            len(ranked_monsters),
            len(overflow_monsters),
            sum(is_engageable for _, is_engageable in ranked_monsters),
            sum(is_engageable for _, is_engageable in overflow_monsters),
            sum(max(0, int(monster["hp"]))
                for monster, _ in overflow_monsters),
            sum(max(1, int(monster["max_hp"]))
                for monster, _ in overflow_monsters),
            min(
                (
                    max(
                        abs(
                            int(monster.get("future_x", monster["x"]))
                            - player_future_x
                        ),
                        abs(
                            int(monster.get("future_y", monster["y"]))
                            - player_future_y
                        ),
                    )
                    for monster, _ in overflow_monsters
                ),
                default=0,
            ),
            max(
                (
                    max(
                        max(0, int(monster["max_damage"])),
                        max(0, int(monster["max_damage_special"])),
                    )
                    for monster, _ in overflow_monsters
                ),
                default=0,
            ),
        )
        monster_overflow_quantities = tuple(
            float(value) / scale
            for value, scale in zip(
                overflow_values,
                CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_SCALES,
                strict=True,
            )
        )

        raw_missiles = raw.get("missiles")
        if not isinstance(raw_missiles, (list, tuple)):
            raise RuntimeError(
                "controller snapshot 缺少原生 missiles 列表")
        missile_required = (
            CONTROLLER_SNAPSHOT_MISSILE_DIRECT_FIELDS
            + CONTROLLER_SNAPSHOT_MISSILE_INT32_FIELDS
        )
        missile_bool_fields = {
            "deleted", "draw", "light", "pre", "hit",
            "limit_reached", "hostile",
            "visible", "source_visible", "start_visible",
        }
        encoded_monster_ids = {
            entry.monster_id for entry in monster_rows}

        def missile_source_is_visible(missile) -> bool:
            if not bool(missile["source_visible"]):
                return False
            # Native MissileSource::Monster is 1.  A lit-but-out-of-window or
            # over-capacity source still has no actor entity row, so exposing
            # its engine index would be an identity side channel.
            return (
                int(missile["source_type"]) != 1
                or int(missile["source_id"]) in encoded_monster_ids
            )

        def missile_policy_value(missile, field):
            if field == "source_visible":
                return missile_source_is_visible(missile)
            if (
                field == "source_id"
                and not missile_source_is_visible(missile)
            ):
                return -1
            if (
                field in {"start_dx", "start_dy"}
                and not bool(missile["start_visible"])
            ):
                return 0
            return missile[field]

        local_missiles = []
        for missile_index, missile in enumerate(raw_missiles):
            if not isinstance(missile, dict):
                raise RuntimeError(
                    f"controller snapshot missiles[{missile_index}] 必须是 dict")
            missing = [
                field for field in missile_required
                if field not in missile
            ]
            if missing:
                raise RuntimeError(
                    f"controller snapshot missiles[{missile_index}] 缺字段:"
                    + ",".join(missing))
            try:
                tile_dx = int(missile["tile_dx"])
                tile_dy = int(missile["tile_dy"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    "controller snapshot missile tile offset 非整数") from exc
            if abs(tile_dx) > radius or abs(tile_dy) > radius:
                continue
            if not bool(missile["visible"]):
                continue
            local_missiles.append(missile)

        def missile_sort_key(missile):
            return (
                0 if bool(missile["hostile"]) else 1,
                max(abs(int(missile["tile_dx"])),
                    abs(int(missile["tile_dy"]))),
                max(0, int(missile["duration"])),
                int(missile["type"]),
                int(missile_policy_value(missile, "source_id")),
                tuple(
                    int(bool(missile[field]))
                    if field in missile_bool_fields
                    else int(missile_policy_value(missile, field))
                    for field in missile_required
                ),
            )

        local_missiles.sort(key=missile_sort_key)
        selected_missiles = local_missiles[
            :CONTROLLER_SNAPSHOT_MISSILE_LIMIT]
        overflow_missiles = local_missiles[
            CONTROLLER_SNAPSHOT_MISSILE_LIMIT:]
        missile_slots = []
        for missile_index, missile in enumerate(selected_missiles):
            quantities: list[float] = []
            for field, scale in zip(
                CONTROLLER_SNAPSHOT_MISSILE_DIRECT_FIELDS,
                CONTROLLER_SNAPSHOT_MISSILE_DIRECT_SCALES,
                strict=True,
            ):
                value = (
                    float(bool(missile_policy_value(missile, field)))
                    if field in missile_bool_fields
                    else float(missile_policy_value(missile, field))
                )
                quantities.append(value / scale)
            for field in CONTROLLER_SNAPSHOT_MISSILE_INT32_FIELDS:
                quantities.extend(self._controller_int32_words(
                    missile_policy_value(missile, field),
                    name=f"missile[{missile_index}].{field}",
                ))
            expected = CONTROLLER_SNAPSHOT_MISSILE_FIELDS - 1
            if (
                len(quantities) != expected
                or not all(math.isfinite(value) for value in quantities)
            ):
                raise RuntimeError(
                    "controller snapshot missile slot 形状/有限性异常")
            missile_slots.append(_ControllerMissile(tuple(quantities)))

        overflow_hostile = [
            missile for missile in overflow_missiles
            if bool(missile["hostile"])
        ]
        overflow_values = (
            len(local_missiles),
            len(overflow_missiles),
            sum(bool(missile["hostile"]) for missile in local_missiles),
            len(overflow_hostile),
            sum(abs(int(missile["damage"]))
                for missile in overflow_missiles),
            max(
                (abs(int(missile["damage"]))
                 for missile in overflow_missiles),
                default=0,
            ),
            min(
                (max(abs(int(missile["tile_dx"])),
                     abs(int(missile["tile_dy"])))
                 for missile in overflow_hostile),
                default=0,
            ),
            min(
                (max(0, int(missile["duration"]))
                 for missile in overflow_hostile),
                default=0,
            ),
            sum(bool(missile["deleted"]) for missile in local_missiles),
        )
        missile_overflow_quantities = tuple(
            float(value) / scale
            for value, scale in zip(
                overflow_values,
                CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_SCALES,
                strict=True,
            )
        )

        if "belt_slot_kinds" not in raw:
            raise RuntimeError(
                "controller snapshot 缺少原生 belt_slot_kinds；"
                "不能混淆腰带 empty/other 或猜测 action12 消耗顺序")
        try:
            belt_slot_kinds = tuple(
                int(kind) for kind in raw["belt_slot_kinds"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "controller snapshot belt_slot_kinds 不可整数化") from exc
        if (
            len(belt_slot_kinds) != CONTROLLER_SNAPSHOT_BELT_SLOTS
            or any(
                kind < 0 or kind >= CONTROLLER_SNAPSHOT_BELT_KINDS
                for kind in belt_slot_kinds
            )
        ):
            raise RuntimeError(
                "controller snapshot belt_slot_kinds 必须是 8 个 [0,5] 整数")

        missing_exact = [
            field for field in CONTROLLER_SNAPSHOT_EXACT_FIELDS
            if field not in raw
        ]
        if missing_exact:
            raise RuntimeError(
                "controller snapshot 缺少 player/scene/quest 精确量:"
                + ",".join(missing_exact))
        exact_quantities = tuple(
            (
                float(bool(raw[field]))
                if field in {
                    "is_set_level", "monotonic_quest_turn_in_used"}
                else float(raw[field])
            ) / scale
            for field, scale in zip(
                CONTROLLER_SNAPSHOT_EXACT_FIELDS,
                CONTROLLER_SNAPSHOT_EXACT_SCALES,
                strict=True,
            )
        )
        if not all(math.isfinite(value) for value in exact_quantities):
            raise RuntimeError("controller snapshot player/scene/quest 精确量含 NaN/Inf")

        missing_combat = [
            field for field in CONTROLLER_SNAPSHOT_COMBAT_FIELDS
            if field not in raw
        ]
        if missing_combat:
            raise RuntimeError(
                "controller snapshot 缺少当前有效战斗量:"
                + ",".join(missing_combat))
        combat_quantities = tuple(
            (
                float(bool(raw[field]))
                if field == "block_enabled" else float(raw[field])
            ) / scale
            for field, scale in zip(
                CONTROLLER_SNAPSHOT_COMBAT_FIELDS,
                CONTROLLER_SNAPSHOT_COMBAT_SCALES,
                strict=True,
            )
        )
        if not all(math.isfinite(value) for value in combat_quantities):
            raise RuntimeError(
                "controller snapshot 当前有效战斗量含 NaN/Inf")
        combat_effect_flags = self._controller_uint(
            raw.get("item_effect_flags"),
            name="item_effect_flags",
            bits=CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS,
        )
        combat_dam_ac_flags = self._controller_uint(
            raw.get("item_dam_ac_flags"),
            name="item_dam_ac_flags",
            bits=CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS,
        )

        equipped_items = raw.get("equipped_items")
        if (
            not isinstance(equipped_items, (list, tuple))
            or len(equipped_items) != CONTROLLER_SNAPSHOT_EQUIPPED_SLOTS
        ):
            raise RuntimeError(
                "controller snapshot equipped_items 必须恰有 7 个槽位")
        equipped_quantities_list: list[float] = []
        equipped_bool_fields = {
            "identified", "effects_active", "stat_usable"}
        for slot_index, item in enumerate(equipped_items):
            if not isinstance(item, dict):
                raise RuntimeError(
                    "controller snapshot equipped_items"
                    f"[{slot_index}] 必须是 dict")
            missing = [
                field for field in (
                    ("present",)
                    + CONTROLLER_SNAPSHOT_GEAR_FIELDS
                    + ("effect_flags", "effect_dam_ac_flags")
                )
                if field not in item
            ]
            if missing:
                raise RuntimeError(
                    "controller snapshot equipped_items"
                    f"[{slot_index}] 缺字段:" + ",".join(missing))
            equipped_quantities_list.append(
                1.0 if bool(item["present"]) else 0.0)
            equipped_quantities_list.extend(
                (
                    float(bool(item[field]))
                    if field in equipped_bool_fields else float(item[field])
                ) / scale
                for field, scale in zip(
                    CONTROLLER_SNAPSHOT_GEAR_FIELDS,
                    CONTROLLER_SNAPSHOT_GEAR_SCALES,
                    strict=True,
                )
            )
            effect_flags = self._controller_uint(
                item["effect_flags"],
                name=f"equipped[{slot_index}].effect_flags",
                bits=CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS,
            )
            dam_ac_flags = self._controller_uint(
                item["effect_dam_ac_flags"],
                name=f"equipped[{slot_index}].effect_dam_ac_flags",
                bits=CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS,
            )
            equipped_quantities_list.extend(
                1.0 if effect_flags & (1 << bit) else 0.0
                for bit in range(CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS)
            )
            equipped_quantities_list.extend(
                1.0 if dam_ac_flags & (1 << bit) else 0.0
                for bit in range(CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS)
            )
        equipped_quantities = tuple(equipped_quantities_list)
        if (
            len(equipped_quantities)
            != CONTROLLER_SNAPSHOT_EQUIPPED_SLOTS
            * CONTROLLER_SNAPSHOT_EQUIPPED_FIELDS
            or not all(
                math.isfinite(value) for value in equipped_quantities)
        ):
            raise RuntimeError(
                "controller snapshot equipped_items 精确量形状/有限性异常")
        sticky = getattr(self, "_explore_target", None)
        sticky_target = (
            (int(sticky[0]), int(sticky[1]))
            if sticky is not None else None
        )
        return _ControllerSnapshot(
            raw_identity=id(raw),
            steps=int(getattr(self, "_steps", 0)),
            scene=_scene_identity(raw),
            player_x=px,
            player_y=py,
            player_future_x=player_future_x,
            player_future_y=player_future_y,
            walkable=walkable,
            visible_monster=visible_monster,
            physical_monster=physical_monster,
            softwall=softwall,
            closed_door=closed_door,
            hazard=hazard,
            explosive_softwall=explosive_softwall,
            visited_mask=tile_mask(visited),
            blocked_mask=tile_mask(explore_blocked),
            protected_mask=tile_mask(protected),
            visited_tiles=visited,
            explore_blocked_targets=explore_blocked,
            protected_tiles=protected,
            sticky_target=sticky_target,
            monsters=tuple(monster_rows),
            monster_overflow_quantities=monster_overflow_quantities,
            candidates=tuple(candidates),
            missiles=tuple(missile_slots),
            missile_overflow_quantities=missile_overflow_quantities,
            heal_target=self._controller_target(
                raw,
                "heal",
                player_x=px,
                player_y=py,
                reachable_tiles=pickup_reachable,
            ),
            gear_target=self._controller_target(
                raw,
                "gear",
                player_x=px,
                player_y=py,
                reachable_tiles=pickup_reachable,
            ),
            belt_slot_kinds=belt_slot_kinds,
            exact_quantities=exact_quantities,
            combat_quantities=combat_quantities,
            combat_effect_flags=combat_effect_flags,
            combat_dam_ac_flags=combat_dam_ac_flags,
            equipped_quantities=equipped_quantities,
        )

    def _controller_snapshot_for(self, raw) -> _ControllerSnapshot:
        """Return an already-installed decision snapshot without mutation."""
        snapshot = getattr(self, "_controller_snapshot", None)
        if (
            snapshot is None
            or snapshot.raw_identity != id(raw)
            or snapshot.steps != int(getattr(self, "_steps", 0))
            or snapshot.scene != _scene_identity(raw)
        ):
            raise RuntimeError(
                "controller snapshot 未安装或已过期；"
                "观测/掩码读取不得隐式抓取另一张地图")
        return snapshot

    def controller_snapshot_vector(self) -> np.ndarray:
        """Return the exact fixed controller wire appended to the dual Worker."""
        self._ensure_active(allow_ended=True)
        snapshot = self._controller_snapshot_for(self._raw)
        values: list[float] = []
        softwall_kind = tuple(
            (
                softwall
                + 2 * closed_door
                + 4 * explosive
            ) / CONTROLLER_SNAPSHOT_SOFTWALL_KIND_DENOMINATOR
            for softwall, closed_door, explosive in zip(
                snapshot.softwall,
                snapshot.closed_door,
                snapshot.explosive_softwall,
                strict=True,
            )
        )
        for channel in (
            snapshot.walkable,
            snapshot.visible_monster,
            softwall_kind,
            snapshot.visited_mask,
            snapshot.blocked_mask,
            snapshot.protected_mask,
            snapshot.hazard,
        ):
            values.extend(float(value) for value in channel)
        for candidate in snapshot.monsters:
            values.extend((
                1.0,
                float(candidate.monster_id) / 200.0,
                float(candidate.monster_type) / 200.0,
                float(candidate.x - snapshot.player_x) / 112.0,
                float(candidate.y - snapshot.player_y) / 112.0,
                float(candidate.future_x - snapshot.player_future_x) / 112.0,
                float(candidate.future_y - snapshot.player_future_y) / 112.0,
                float(candidate.hp) / 1024.0,
                float(candidate.max_hp) / 1024.0,
                float(candidate.ledger_low) / 1024.0,
                float(candidate.ledger_max) / 1024.0,
                1.0 if candidate.blocked else 0.0,
                1.0 if candidate.visible else 0.0,
                1.0 if candidate.native_reachable else 0.0,
                1.0 if candidate.locally_engageable else 0.0,
            ))
            values.extend(candidate.dynamic_quantities)
            values.extend(
                1.0 if candidate.combat_flags & (1 << bit) else 0.0
                for bit in range(CONTROLLER_SNAPSHOT_MONSTER_FLAG_BITS)
            )
        values.extend(
            [0.0] * (
                CONTROLLER_SNAPSHOT_MONSTER_DIM
                - CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_DIM
                - len(snapshot.monsters)
                * CONTROLLER_SNAPSHOT_MONSTER_FIELDS
            )
        )
        values.extend(snapshot.monster_overflow_quantities)
        for missile in snapshot.missiles:
            values.append(1.0)
            values.extend(missile.quantities)
        values.extend(
            [0.0] * (
                CONTROLLER_SNAPSHOT_MISSILE_LIMIT
                - len(snapshot.missiles)
            ) * CONTROLLER_SNAPSHOT_MISSILE_FIELDS
        )
        values.extend(snapshot.missile_overflow_quantities)
        for kind in snapshot.belt_slot_kinds:
            values.extend(
                1.0 if kind == expected else 0.0
                for expected in range(CONTROLLER_SNAPSHOT_BELT_KINDS)
            )
        values.extend(snapshot.exact_quantities)
        values.extend(snapshot.combat_quantities)
        values.extend(
            1.0 if snapshot.combat_effect_flags & (1 << bit) else 0.0
            for bit in range(CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS)
        )
        values.extend(
            1.0 if snapshot.combat_dam_ac_flags & (1 << bit) else 0.0
            for bit in range(CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS)
        )
        values.extend(snapshot.equipped_quantities)
        if snapshot.sticky_target is None:
            values.extend((0.0, 0.0, 0.0))
        else:
            values.extend((
                1.0,
                float(snapshot.sticky_target[0] - snapshot.player_x) / 112.0,
                float(snapshot.sticky_target[1] - snapshot.player_y) / 112.0,
            ))
        heal_target = snapshot.heal_target
        if heal_target is None:
            values.extend([0.0] * CONTROLLER_SNAPSHOT_HEAL_TARGET_DIM)
        else:
            values.extend((
                1.0,
                float(heal_target.x - snapshot.player_x) / 112.0,
                float(heal_target.y - snapshot.player_y) / 112.0,
                float(heal_target.active_id) / 128.0,
                float(heal_target.heal_kind)
                / CONTROLLER_SNAPSHOT_INSTANT_HEAL_KINDS,
            ))

        gear_target = snapshot.gear_target
        if gear_target is None:
            values.extend([0.0] * CONTROLLER_SNAPSHOT_GEAR_TARGET_DIM)
        else:
            values.extend((
                1.0,
                float(gear_target.x - snapshot.player_x) / 112.0,
                float(gear_target.y - snapshot.player_y) / 112.0,
            ))
            values.extend(gear_target.gear_quantities)
            values.extend(
                1.0 if gear_target.effect_flags & (1 << bit) else 0.0
                for bit in range(CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS)
            )
            values.extend(
                1.0 if gear_target.dam_ac_flags & (1 << bit) else 0.0
                for bit in range(CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS)
            )
        result = np.asarray(values, dtype=np.float32)
        if (
            result.shape != (CONTROLLER_SNAPSHOT_VECTOR_DIM,)
            or not np.isfinite(result).all()
        ):
            raise RuntimeError(
                "controller snapshot wire 形状/有限性漂移:"
                f"shape={result.shape},finite={np.isfinite(result).all()}")
        return result

    @property
    def exploration_progress(self) -> int:
        """本局首次踏足格 + action10 已打开软墙的单调计数。"""
        return int(self._exploration_progress)

    @property
    def softwalls_opened(self) -> int:
        """本局由 action10 实际打开的 closed door / blocking barrel 数。"""
        return int(self._softwalls_opened)

    def action_masks(self) -> np.ndarray:
        """v4 无效动作掩码(MaskablePPO 协议方法)。

        9 只在 radius-12 快照内存在可见且局部可接敌的已编码怪物时合法；
        每行另行公开原生 reachable，不能把两种可达性混为一个事实。
        12 要求腰带有治疗药且当前确实掉血；13 要求有腰带空槽和快照内
        有效治疗药目标；14 要求快照内存在 PlanGearUpgrade 认可的严格
        整套升级。其余动作保持合法。掩码消灭确定性空按，但不承诺宏在
        动态怪物阻挡、受击或时间上限内必然完成。

        这是 protocol-v4 的行为语义断点：旧策略即使动作/观测 shape 相同，
        logits 也会被重新归一化，必须重新基线化，不能与 v3 排行榜混用。
        """
        self._ensure_active(allow_ended=True)
        mask, _nearest = self.controller_action_context()
        return mask

    def _decision_controller_snapshot(self, raw) -> _ControllerSnapshot:
        """Return the exact snapshot that a controller action would execute.

        The dual view installs its snapshot while producing the observation;
        silently recapturing it here would break observation/action atomicity.
        Legacy/flat views do not expose the controller wire, so they capture a
        read-only snapshot at the mask edge, exactly as ``step`` does.
        """
        snapshot = getattr(self, "_controller_snapshot", None)
        snapshot_is_current = (
            snapshot is not None
            and snapshot.raw_identity == id(raw)
            and snapshot.steps == int(getattr(self, "_steps", 0))
            and snapshot.scene == _scene_identity(raw)
        )
        if snapshot_is_current:
            return snapshot
        if getattr(self, "_controller_snapshot_enabled", False):
            return self._controller_snapshot_for(raw)
        return self._capture_controller_snapshot(raw)

    def controller_action_context(
        self,
    ) -> tuple[np.ndarray, int | None]:
        """Exact action mask plus nearest locally engageable monster distance.

        Scripted teachers need the distance from the same radius-12 candidate
        set as action 9.  Using the global raw monster list here made a hidden
        or locally blocked monster suppress DIVE handoff and generate labels
        for a different target than the macro could execute.
        """
        self._ensure_active(allow_ended=True)
        mask = np.ones(15, dtype=bool)
        raw = self._raw
        snapshot = self._decision_controller_snapshot(raw)
        # Flat/base policies receive the same hard authority boundary as the
        # hierarchical Worker.  A nominal adjacent CMD_WALKXY may otherwise
        # re-plan around a blocked edge and step on a stair/quest trigger.
        if getattr(self, "resource_protocol", "off") != "off":
            from .resource_protocol import progression_allowed
            protected = self._resource_direction_protected_tiles(raw)
            px, py = int(raw["player_x"]), int(raw["player_y"])
            for action, (dx, dy) in enumerate(_DIRS, start=1):
                if (px + dx, py + dy) in protected:
                    mask[action] = False
            mask[11] = progression_allowed(
                raw, getattr(self, "resource_readiness_law", "veto-v1"))
        else:
            for action in self._protected_walk_actions(raw):
                mask[action] = False
        mask[9] = bool(snapshot.candidates)
        mask[12] = (
            int(raw.get("belt_heals", 0)) > 0
            and int(raw.get("hp", 0)) < int(raw.get("max_hp", 0))
        )
        mask[13] = (
            self._belt_free_slots(raw) > 0
            and snapshot.heal_target is not None
        )
        mask[14] = snapshot.gear_target is not None
        nearest = min(
            (
                max(
                    abs(candidate.future_x - snapshot.player_future_x),
                    abs(candidate.future_y - snapshot.player_future_y),
                )
                for candidate in snapshot.candidates
            ),
            default=None,
        )
        return mask, nearest

    def controller_action_masks(self) -> np.ndarray:
        """Backward-compatible name for the now-canonical exact mask."""
        return self.action_masks()

    def _configure_native_resource_protocol(self):
        enabled = getattr(self, "resource_protocol", "off") != "off"
        if hasattr(bridge, "configure_resource_protocol"):
            bridge.end_game()
            if getattr(self, "resource_readiness_law", "veto-v1") == "coach-v03":
                # New keyword: only reachable with a bridge built after R17.1 ruling 3.
                bridge.configure_resource_protocol(enabled,
                    ordinary_armor_scope=getattr(self, "resource_ordinary_armor_scope", False),
                    preserve_equipment_readiness=getattr(
                        self, "resource_preserve_equipment_readiness", False),
                    readiness_advisory=True)
            elif getattr(self, "resource_preserve_equipment_readiness", False):
                bridge.configure_resource_protocol(enabled,
                    ordinary_armor_scope=getattr(self, "resource_ordinary_armor_scope", False),
                    preserve_equipment_readiness=True)
            elif getattr(self, "resource_ordinary_armor_scope", False):
                bridge.configure_resource_protocol(enabled, ordinary_armor_scope=True)
            else:
                # Preserve the original one-argument ABI for every old policy.
                bridge.configure_resource_protocol(enabled)
        elif enabled:
            raise RuntimeError("Native bridge lacks the resource protocol; rebuild required")

    def _step_native(self):
        raw = bridge.step(ticks=self.ticks_per_step)
        if getattr(self, "resource_protocol", "off") != "off":
            self._resource_actual_microsteps += 1
        DiabloGymEnv._validate_native_resource_flags(self, raw)
        if getattr(self, "resource_protocol", "off") != "off":
            time_callback = getattr(self, "resource_time_callback", None)
            if time_callback is not None:
                time_callback(self, raw, self._resource_actual_microsteps)
            observer = getattr(self, "resource_observation_callback", None)
            if observer is not None:
                observer(self, raw, self._resource_actual_microsteps)
            callback = getattr(self, "resource_tick_callback", None)
            if callback is not None:
                callback(self, raw, self._resource_actual_microsteps)
        return raw

    def step_resource(self, command):
        """Execute a script-owned service command through normal env accounting.

        The internal wait action number is never a Worker-labelled transition;
        resource_action_audit identifies the actual source and command.
        """
        if getattr(self, "resource_protocol", "off") == "off":
            raise RuntimeError("Resource command requires l2-town-v1")
        if self._resource_pending_command is not None:
            raise RuntimeError("Nested resource command")
        self._resource_pending_command = tuple(command)
        try:
            return self.step(0)
        finally:
            self._resource_pending_command = None

    def _execute_resource_command(self, command):
        kind, *args = command
        receipt = {"accepted": False, "reason": kind, "price": 0}
        if kind in ("finish", "complete"):
            if kind == "finish":
                self._resource_terminal_reason = str(args[0])
            return self._raw, 0, receipt
        if kind in ("attack_monster", "drink", "unequip"):
            # New script-only commands. Keep the legacy service dispatch below
            # untouched; every mutation here is bounded by actual native clocks.
            if kind == "unequip" and self.resource_purchase_mode != "full":
                raise ValueError("Unequip is restricted to the full resource arm")
            if kind == "attack_monster":
                if (len(args) != 2 or type(args[1]) is not int or args[1] <= 0):
                    raise ValueError("attack_monster requires id and positive microstep budget")
                command_budget = min(12, args[1])
            else:
                command_budget = 1
            clock_before = self._resource_actual_microsteps
            fixed_deadline = clock_before + command_budget

            def time_available():
                return self._resource_actual_microsteps < min(
                    fixed_deadline, self.max_steps,
                    int(getattr(self, "_resource_service_deadline", self.max_steps)))

            if not time_available():
                receipt["reason"] = "resource_command_deadline"
                return self._raw, 0, receipt
            before = self._raw
            if (before.get("dead") or before.get("game_over") or before.get("victory")
                    or int(before.get("hp", 0)) <= 0):
                receipt["reason"] = "resource_command_terminal"
                return before, 0, receipt
            start_scene = _scene_identity(before)
            px, py = int(before["player_x"]), int(before["player_y"])
            if kind == "attack_monster":
                target_before = next((monster for monster in before.get("monsters", ())
                                      if int(monster.get("id", -1)) == int(args[0])
                                      and bool(monster.get("visible", False))
                                      and not bool(monster.get("is_invalid", False))
                                      and int(monster.get("type", -1)) != 109
                                      and int(monster.get("hp", 0)) > 0), None)
                target_key = (None if target_before is None else tuple(
                    target_before.get(name) for name in
                    ("id", "type", "rnd_item_seed_hi", "rnd_item_seed_lo")))
                adjacent = target_before is not None and max(
                    abs(int(target_before.get("future_x", target_before["x"]))
                        - int(before.get("future_x", px))),
                    abs(int(target_before.get("future_y", target_before["y"]))
                        - int(before.get("future_y", py)))) <= 1
                result = (bridge.act_controller_attack_monster(int(args[0]), px, py, 1)
                          if adjacent else 0)
                receipt["accepted"] = int(result) == 1
                receipt["monster_id"] = int(args[0])
                receipt["microstep_budget"] = command_budget
            elif kind == "drink":
                kinds = before.get("belt_heal_kinds")
                belt_before = (sum(int(value) in (1, 2, 3, 4) for value in kinds)
                               if kinds is not None else int(before.get(
                                   "resource_state", {}).get("readiness", {}).get("belt_heals", 0)))
                # Same FIFO fence and certified return convention as action12:
                # the result is zero or the PRE-drink belt count, not bool(1).
                bridge.act_wait()
                result = bridge.act_drink()
                if not isinstance(result, (bool, np.bool_, int, np.integer)):
                    raise RuntimeError("resource drink native receipt must be an integer")
                accepted = int(result)
                if accepted not in (0, belt_before):
                    raise RuntimeError("resource drink native receipt disagrees with pre-drink belt")
                receipt.update(accepted=accepted > 0, accepted_belt_before=accepted,
                               belt_before=belt_before, hp_before=int(before.get("hp", 0)),
                               consumed=accepted > 0)
            else:
                result = bridge.act_unequip_equipped_item(*args)
                if not isinstance(result, dict):
                    raise RuntimeError("resource unequip native receipt must be a dictionary")
                receipt = dict(result)
            if not receipt.get("accepted") and kind != "drink":
                bridge.act_wait()
            raw = self._step_native()
            beats = 1
            if kind == "attack_monster" and receipt.get("accepted"):
                # Do not cancel a swing after its first native beat. Preserve
                # its ordinary animation, but never chase an out-of-range or
                # no-longer-visible target and never cross a live deadline.
                while (time_available()
                       and not (raw.get("dead") or raw.get("game_over") or raw.get("victory"))
                       and _scene_identity(raw) == start_scene
                       and not self._decision_idle(raw)):
                    target = next((monster for monster in raw.get("monsters", ())
                                   if int(monster.get("id", -1)) == int(args[0])
                                   and bool(monster.get("visible", False))
                                   and int(monster.get("hp", 0)) > 0), None)
                    if (target is None or tuple(target.get(name) for name in
                            ("id", "type", "rnd_item_seed_hi", "rnd_item_seed_lo")) != target_key):
                        break
                    if max(
                            abs(int(target.get("future_x", target["x"]))
                                - int(raw.get("future_x", raw["player_x"]))),
                            abs(int(target.get("future_y", target["y"]))
                                - int(raw.get("future_y", raw["player_y"])))) > 1:
                        break
                    raw = self._step_native()
                    beats += 1
            if kind == "drink":
                kinds = raw.get("belt_heal_kinds")
                belt_after = (sum(int(value) in (1, 2, 3, 4) for value in kinds)
                              if kinds is not None else int(raw.get(
                                  "resource_state", {}).get("readiness", {}).get("belt_heals", 0)))
                hp_after = int(raw.get("hp", 0))
                receipt.update(belt_after=belt_after, hp_after=hp_after,
                               belt_consumed_observed=belt_after < receipt["belt_before"],
                               post_tick_effect_visible=(belt_after < receipt["belt_before"]
                                                         or hp_after > receipt["hp_before"]))
                # Native ActDrink already certifies synchronous consumption or
                # HP increase. An intervening attack/auto-refill may hide that
                # effect in the net post-tick delta; retain both facts explicitly.
            return raw, beats, receipt
        px, py = int(self._raw["player_x"]), int(self._raw["player_y"])
        if kind == "walk":
            result = bridge.act_explore_walk(*args, (), px, py, self._DESCEND_RADIUS)
        elif kind == "open":
            result = bridge.act_controller_operate(*args, px, py, self._DESCEND_RADIUS)
        elif kind == "gold":
            result = bridge.act_pickup_gold_at(*args)
        elif kind == "talk":
            result = bridge.act_talk_towner(*args)
        elif kind == "dismiss":
            result = bridge.act_dismiss_dialog()
        elif kind == "buy":
            result = bridge.act_buy_store_item(*args)
        elif kind == "equip":
            result = bridge.act_equip_inventory_item(*args)
        elif kind == "repair":
            if self.resource_purchase_mode != "full":
                raise ValueError("Paid repair is restricted to the full resource arm")
            result = bridge.act_repair_equipped_item(*args)
        elif kind == "wait":
            result = bridge.act_wait()
        else:
            raise ValueError(f"Unknown resource command: {kind!r}")
        if isinstance(result, dict):
            receipt = dict(result)
        else:
            receipt["accepted"] = int(result) == 1
        # Rejections still advance a real, explicit wait. This avoids zero-time
        # service retry loops while never inventing a phantom microstep.
        if not receipt.get("accepted"):
            bridge.act_wait()
        clock_before = self._resource_actual_microsteps
        start_scene = _scene_identity(self._raw)
        raw = self._step_native()
        beats = 1
        if kind == "open" and receipt.get("accepted"):
            # Non-explosive barrels share the planner's softwall channel.
            # Let the accepted native operation finish its attack animation
            # before the generic settle fence cancels further intent. Without
            # this, a one-beat command starts a swing and immediately cancels
            # it before impact on every retry (seed 2114004, object type 57).
            deadline = min(self.max_steps,
                           int(getattr(self, "_resource_service_deadline", self.max_steps)),
                           clock_before + 12)
            while (self._resource_actual_microsteps < deadline
                   and not (raw.get("dead") or raw.get("game_over") or raw.get("victory"))
                   and _scene_identity(raw) == start_scene
                   and not bool(bridge.probe_tile(int(args[0]), int(args[1]))["walkable"])
                   and not self._decision_idle(raw)
                   and (getattr(self, "_resource_calibration", None) is None
                        or self._resource_actual_microsteps < self.max_steps)):
                raw = self._step_native()
                beats += 1
        return raw, beats, receipt

    def step(self, action: int, *, worker_authority: bool = False):
        self._ensure_active()
        if not self.action_space.contains(action):
            raise ValueError(f"动作必须是 {self.action_space}中的整数，收到 {action!r}")
        prev = self._raw
        resource_command = getattr(self, "_resource_pending_command", None)
        resource_receipt = None
        resource_clock_before = getattr(self, "_resource_actual_microsteps", 0)
        self._resource_reward_depth_before = getattr(self, "_resource_max_main_depth", 0)
        self._resource_step_bonus = 0.0
        action = int(action)
        if getattr(self, "dive_blocker_recovery", "off") != "off":
            self._dive_blocker_audit = None  # receipt is strictly per action
        remaining = self.max_steps - self._steps
        if resource_command is not None:
            remaining = min(remaining, max(0, int(getattr(
                self, "_resource_service_deadline", self.max_steps)) - self._resource_actual_microsteps))
            if resource_command[0] == "attack_monster":
                if (len(resource_command) != 3 or type(resource_command[2]) is not int
                        or resource_command[2] <= 0):
                    raise ValueError("attack_monster requires id and positive microstep budget")
                remaining = min(remaining, 12, resource_command[2])
            elif resource_command[0] in ("drink", "unequip"):
                remaining = min(remaining, 1)
        drink_audit = None
        action14_audit = None
        native_execution = {"attempts": 0, "accepts": 0}
        exploration_before = int(
            getattr(self, "_exploration_progress", 0))
        softwalls_before = int(
            getattr(self, "_softwalls_opened", 0))
        controller_snapshot = None
        engage_target_generation_key = None
        if action in {9, 10, 13, 14}:
            # dual Worker 必须执行它刚看到的已安装快照；旧 295/298
            # 视图未公开 controller wire，按键边界局部抓取一次即可，且
            # 不写入缓存，保持 action_masks/普通观测纯读和旧模式轻量。
            controller_snapshot = (
                self._controller_snapshot_for(prev)
                if getattr(self, "_controller_snapshot_enabled", False)
                else self._capture_controller_snapshot(prev)
            )
        if resource_command is not None:
            self._raw, micro, resource_receipt = self._execute_resource_command(resource_command)
            if resource_command[0] not in ("finish", "complete"):
                native_execution["attempts"] = 1
                native_execution["accepts"] = int(bool(resource_receipt.get("accepted")))
        elif action == 9:
            engage_candidate = self._canonical_engage_candidate(
                controller_snapshot)
            if engage_candidate is not None:
                engage_target_generation_key = (
                    engage_candidate.generation_key)
            self._raw, micro = self._macro_engage(
                max_beats=min(10, remaining),
                controller_snapshot=controller_snapshot,
                execution_audit=native_execution,
            )
        elif action == 10:
            self._raw, micro = self._macro_explore(
                max_beats=min(12, remaining),
                controller_snapshot=controller_snapshot,
                execution_audit=native_execution,
            )
        elif action == 11:
            self._raw, micro = self._macro_descend(
                max_beats=min(12, remaining),
                execution_audit=native_execution,
            )
        elif action == 12:
            # 喝药本身不走网络命令层，必须先用 wait 栅栏取消上一动作可能
            # 留下的追击；否则药水键的一拍仍会继续走路/攻击并吞掉其奖励。
            belt_before = int(prev.get("belt_heals", 0))
            bridge.act_wait()
            accepted_raw = bridge.act_drink()
            if not isinstance(
                    accepted_raw, (bool, np.bool_, int, np.integer)):
                raise RuntimeError(
                    "action12 原生回执必须是整数，收到 "
                    f"{accepted_raw!r}")
            accepted = int(accepted_raw)
            if accepted not in (0, belt_before):
                raise RuntimeError(
                    "action12 原生回执与请求前腰带数不一致:"
                    f"accepted={accepted},before={belt_before}")
            self._raw = self._step_native()
            micro = 1
            # ``act_drink`` can be rejected while the player is in hit/block
            # recovery even though the visible hp/belt predicate is true.
            # A requested key is not an executed drink until a bottle really
            # leaves the belt.  Keep this audit through the later settle so
            # wrappers never count failed emergency attempts as rescues.
            drink_audit = {
                "accepted": accepted > 0,
                "accepted_belt_before": accepted,
                "belt_before": belt_before,
                "consumed": accepted > 0,
            }
            self._record_native_execution(
                native_execution,
                1 if accepted > 0 else 0,
                "action12 drink",
            )
        elif action == 13:
            self._raw, micro = self._macro_pickup(
                "heal",
                max_beats=min(12, remaining),
                controller_snapshot=controller_snapshot,
                execution_audit=native_execution,
            )
        elif action == 14:
            utility = gear_combat_utility_value(
                prev, "action14_request")
            action14_audit = {
                "accepted": False,
                "commit_attempts": 0,
                "utility_before": utility,
                "utility_after": utility,
                "utility_delta": 0,
            }
            self._raw, micro = self._macro_pickup(
                "gear",
                max_beats=min(12, remaining),
                controller_snapshot=controller_snapshot,
                action14_audit=action14_audit,
                execution_audit=native_execution,
            )
        else:
            accepted = self._apply_action(
                action, worker_authority=bool(worker_authority))
            if action != 0:
                self._record_native_execution(
                    native_execution, accepted, "direction")
            self._raw = self._step_native()
            micro = 1
        dive_recovery_audit = getattr(self, "_dive_blocker_audit", None)
        if dive_recovery_audit is not None:
            dive_recovery_audit["attack_core_micro_steps"] = dive_recovery_audit["micro_steps"]
            dive_recovery_audit["core_micro_steps"] = self._resource_actual_microsteps - resource_clock_before
            dive_recovery_audit["core_after_microstep"] = self._resource_actual_microsteps
        # 295 维策略观测不含 future/mode/path。所有非终局决策边界必须把
        # 本动作已经提交的单格/硬直动画结清到 PM_STAND；这些 settle 拍
        # 属于本动作，照常占用 max_steps、奖励差分及 Options 的 τ/时钟。
        if resource_command is None or resource_command[0] not in ("finish", "complete"):
            self._raw, micro = self._settle_to_idle(
                self._raw,
                micro,
                max_beats=remaining,
                start_scene=_scene_identity(prev),
            )
        if getattr(self, "resource_protocol", "off") != "off":
            micro = self._resource_actual_microsteps - resource_clock_before
        authorized_retreat = (
            getattr(self, "resource_protocol", "off") != "off"
            and int(prev["dungeon_level"]) == 1
            and int(self._raw["dungeon_level"]) == 0
            and bool(self._raw.get("resource_state", {}).get("service_trip")))
        if (self.start_in_dungeon and not authorized_retreat
                and self._raw["dungeon_level"] < prev["dungeon_level"]):
            # 未来若出现新的回城/向上传送路径，宁可终止训练也
            # 不能把 depth=0 空耗轨迹静默喂给 PPO。原生触发层已封住
            # 常见上楼与回城楼梯，这里是第二道 fail-closed 不变量。
            bad_transition = (prev["dungeon_level"], self._raw["dungeon_level"])
            try:
                bridge.end_game()
            finally:
                if DiabloGymEnv._active_token is self._token:
                    DiabloGymEnv._active_token = None
                self._raw = None
                self._native_generation = None
                self._episode_ended = True
                self._engage_blocked_keys = set()
            raise RuntimeError(
                f"DiabloGym 禁止地牢层级回退: {bad_transition[0]}→{bad_transition[1]}")
        self._steps += micro
        same_scene = _scene_identity(self._raw) == _scene_identity(prev)
        if not same_scene and getattr(self, "resource_protocol", "off") != "off":
            names = ("_visited", "_progress_anchors", "_explore_target",
                     "_explore_blocked_targets", "_engage_blocked_keys", "_combat_hp_floor")
            self._resource_scene_ledgers[_scene_identity(prev)] = {
                name: getattr(self, name) for name in names}
        if not same_scene:
            # 新主层或任务副本:足迹清零。不同地图共用同一坐标系,不清的话
            # 探索宏会把旧图足迹当"已踏足",边疆逻辑整层失效；伤害最低
            # 血线也必须换账本，怪物 id 只在单场景内有意义。
            self._visited = set()
            self._progress_anchors = set()  # R16 C6:锚点与足迹同场景清零
            self._explore_target = None
            self._explore_blocked_targets = set()
            self._engage_blocked_keys = set()
            self._reset_combat_ledger(self._raw)
            if getattr(self, "resource_protocol", "off") != "off":
                for name, value in self._resource_scene_ledgers.get(
                        _scene_identity(self._raw), {}).items():
                    setattr(self, name, value)
        if getattr(self, "resource_protocol", "off") != "off" and not self._raw.get("is_set_level"):
            depth = int(self._raw["dungeon_level"])
            if depth > self._resource_max_main_depth:
                receipt = dict(self._raw.get("resource_state", {}).get("transition", {}))
                if getattr(self, "resource_readiness_law", "veto-v1") == "coach-v03":
                    # Forced-unready descents are legal under the frozen mask law;
                    # the receipt must exist and be accepted, readiness is accounting.
                    if depth > 1 and not receipt.get("accepted"):
                        raise RuntimeError("New main depth lacks an accepted transition receipt")
                elif depth > 1 and (not receipt.get("accepted")
                        or not receipt.get("pretransition_ready")):
                    raise RuntimeError("New main depth lacks an accepted pretransition readiness receipt")
                self._resource_transition_receipts.append(receipt)
                self._resource_max_main_depth = depth
            if self._resource_max_main_depth > self._resource_reward_depth_before:
                self._econ_steps_on_level = 0
            self._econ_episode_max_depth = self._resource_max_main_depth
        self._record_visit((self._raw["player_x"], self._raw["player_y"]))

        # Native MonsterDeath is the only event-complete kill ledger.  Endpoint
        # list differences still miss a monster that spawns and dies inside
        # one multi-beat macro, and slot generation keys cannot recover an
        # entity absent from both observations.
        native_kills = self._native_monster_kill_delta(prev, self._raw)
        if native_kills is None:
            # Compatibility for old synthetic fixtures only.  Current native
            # observations always carry monster_kill_total.
            native_kills = (
                self._disappeared_monster_generations(prev, self._raw)
                if same_scene else 0
            )
        self._ep_kills += native_kills

        effect_reasons = self._action_effect_reasons(
            prev,
            self._raw,
            requested_action=action,
            engage_target_generation_key=engage_target_generation_key,
            native_kills=native_kills,
            exploration_before=exploration_before,
            softwalls_before=softwalls_before,
            drink_audit=drink_audit,
            action14_audit=action14_audit,
        )
        action_effective = bool(effect_reasons)
        request_executed = bool(
            action == 0
            or native_execution["accepts"] > 0
        )
        stall_cost_applied = bool(
            _scene_identity(self._raw) == _scene_identity(prev)
            and (action == 0 or not request_executed)
        )
        if resource_command is not None:
            request_executed = bool((resource_receipt or {}).get("accepted"))
            stall_cost_applied = bool(same_scene and not request_executed and micro > 0)
        same_scene_for_reward = bool(
            _scene_identity(self._raw) == _scene_identity(prev))

        reward = self._reward(
            prev,
            self._raw,
            requested_action=None if resource_command is not None else action,
            engage_target_generation_key=engage_target_generation_key,
            action_executed=request_executed,
            action14_utility_delta=(
                int(action14_audit["utility_delta"])
                if action14_audit is not None else None
            ),
        )
        if resource_command is not None and resource_command[0] in ("finish", "complete"):
            reward = 0.0
            stall_cost_applied = False
        # 奖励会推进 combat ledger；下一决策的候选账本必须在这之后冻结。
        self._controller_snapshot = (
            self._capture_controller_snapshot(self._raw)
            if getattr(self, "_controller_snapshot_enabled", False)
            else None
        )
        # SB3 会对每个 Gym ``truncated`` 的 terminal_observation 自动加入
        # gamma*V(s_T)。若预算恰在走格/受击动画中耗尽，295 维观测却没有
        # future/mode/path，价值网络看到的是一个与“可立即重新决策的 idle
        # 状态”完全别名的 busy 状态；此时 bootstrap 会凭空假定玩家能跳过
        # 已提交动画并立刻行动。不能越过 max_steps 偷结算动画，也不能把
        # POMDP 状态冒充合法 TimeLimit 状态。因此把这一种表示层边界定义为
        # fail-closed terminal；真正 idle 的时间上限仍是标准 truncation，
        # 保留正确的 TimeLimit bootstrap。
        (terminated, truncated, budget_exhausted, decision_idle,
         unsettled_budget_terminal) = self._episode_boundary(
             self._raw, self._steps, self.max_steps)

        resource_failure_terminal = bool(
            getattr(self, "resource_protocol", "off") != "off"
            and self._resource_terminal_reason is not None
        )
        if resource_failure_terminal:
            # A failed service itinerary ends this protocol episode. It is
            # not a settled TimeLimit that SB3 may bootstrap through, even
            # when the player is alive or the episode clock also runs out.
            terminated, truncated = True, False
        info = self._info(self._raw)
        if getattr(self, "resource_protocol", "off") != "off":
            info["resource_protocol"] = {
                "protocol": self.resource_protocol, "purchase_mode": self.resource_purchase_mode,
                "max_main_depth": self._resource_max_main_depth,
                "terminal_reason": self._resource_terminal_reason,
                "new_main_depth_bonus": self._resource_step_bonus,
                "curriculum_boundary": int(self._raw["dungeon_level"]) >= 2,
                "readiness": dict(self._raw["resource_state"]["readiness"]),
                "transition": dict(self._raw["resource_state"].get("transition", {})),
            }
            calibration = getattr(self, "_resource_calibration", None)
            if calibration is not None:
                info["resource_protocol"]["calibration"] = calibration.metadata()
            if resource_command is not None:
                info["resource_action_audit"] = {
                    "source": "resupply-script", "command": resource_command,
                    "micro_steps": micro, **(resource_receipt or {})}
        info["action_effect_audit"] = {
            "requested_action": action,
            "native_attempts": int(native_execution["attempts"]),
            "native_accepts": int(native_execution["accepts"]),
            "request_executed": request_executed,
            "material_effect": action_effective,
            "effect_reasons": tuple(effect_reasons),
            "same_scene": same_scene_for_reward,
            "stall_cost_applied": stall_cost_applied,
        }
        if dive_recovery_audit is not None:
            info["dive_blocker_recovery_audit"] = {
                **dive_recovery_audit,
                "transition_before_microstep": resource_clock_before,
                "after_microstep": self._resource_actual_microsteps,
                "micro_steps": micro,
                "settle_micro_steps": micro - dive_recovery_audit["core_micro_steps"],
                "decision_idle": self._decision_idle(self._raw),
            }
        if drink_audit is not None:
            belt_after = int(self._raw.get("belt_heals", 0))
            info["action12_audit"] = {
                **drink_audit,
                "belt_after": belt_after,
            }
        if action14_audit is not None:
            info["action14_audit"] = dict(action14_audit)
        if budget_exhausted:
            info.update({
                "budget_exhausted": True,
                "decision_idle": bool(decision_idle),
                "unsettled_budget_terminal": unsettled_budget_terminal,
                "time_limit_bootstrap_safe": bool(truncated),
            })
        if resource_failure_terminal:
            info.update({
                "resource_failure_terminal": True,
                "decision_idle": bool(decision_idle),
                "unsettled_budget_terminal": unsettled_budget_terminal,
                "time_limit_bootstrap_safe": False,
            })
        if terminated or truncated:
            info["episode_extra"] = {
                "xp": int(self._raw["xp"]) - self._ep_start_xp,
                "kills": self._ep_kills,
                "char_level": self._raw["char_level"],
                "depth": self._raw["dungeon_level"],
                "died": bool(self._raw["dead"]),
                "gold": self._raw["gold"],
            }
        obs = self._vectorize(self._raw)
        if terminated or truncated:
            self._episode_ended = True
        return obs, reward, terminated, truncated, info

    def _info(self, raw):
        info = {"episode_seed": self._episode_seed}
        time_info = getattr(self, "resource_time_info_callback", None)
        if time_info is not None:
            info["completion_time"] = time_info()
        if self.include_raw:
            # info 属于调用方；不得把内部奖励/宏状态依赖的可变 raw
            # 直接泄露出去，否则回调或调试代码修改 info["raw"] 会篡改下一拍奖励。
            # 新协议会继续追加 list[dict]（equipped/missiles 等），因此不能
            # 维护一个容易漏字段的白名单。只递归复制 Python 可变容器；
            # 数字/字符串保持共享不可变对象，成本仍远低于复制原生状态。
            def clone_mutable(value):
                if isinstance(value, dict):
                    return {
                        key: clone_mutable(child)
                        for key, child in value.items()
                    }
                if isinstance(value, list):
                    return [clone_mutable(child) for child in value]
                if isinstance(value, tuple):
                    return tuple(clone_mutable(child) for child in value)
                if isinstance(value, set):
                    return {clone_mutable(child) for child in value}
                return value

            snapshot = clone_mutable(raw)
            info["raw"] = snapshot
        return info

    def _ensure_active(self, *, allow_ended: bool = False) -> None:
        if (DiabloGymEnv._engine_initialized
                and DiabloGymEnv._engine_pid != os.getpid()):
            raise RuntimeError(
                "禁止在 fork 子进程使用父进程已初始化的 DevilutionX；"
                "多环境训练必须使用 spawn")
        if self._raw is None:
            raise gym.error.ResetNeeded("step/action_masks 前必须先调用 reset()")
        if DiabloGymEnv._active_token is not self._token:
            raise RuntimeError(
                "检测到同进程多个 DiabloGymEnv 交错使用；引擎是全局单例。"
                "请顺序 reset/使用，或改用 SubprocVecEnv")
        if int(bridge.episode_generation()) != self._native_generation:
            raise RuntimeError(
                "引擎已被直接 bridge.reset() 或其他 wrapper 重置，"
                "当前环境缓存已失效；请对本环境重新 reset()")
        if self._episode_ended and not allow_ended:
            raise gym.error.ResetNeeded("episode 已终止/截断，继续 step() 前必须 reset()")

    def close(self):
        if (DiabloGymEnv._engine_initialized
                and DiabloGymEnv._engine_pid != os.getpid()):
            # 子进程继承的是父进程多线程引擎的一份不安全快照；绝不能
            # 在这里进入 SDL/NetClose/Lua。OS 会回收子进程地址空间。
            if DiabloGymEnv._active_token is self._token:
                DiabloGymEnv._active_token = None
            self._raw = None
            self._native_generation = None
            self._episode_ended = True
            self._exploration_progress = 0
            self._softwalls_opened = 0
            self._explore_target = None
            self._explore_blocked_targets = set()
            self._engage_blocked_keys = set()
            self._combat_hp_floor = {}
            self._controller_snapshot = None
            return
        if DiabloGymEnv._active_token is self._token:
            bridge.end_game()
            DiabloGymEnv._active_token = None
        self._raw = None
        self._native_generation = None
        self._episode_ended = True
        self._exploration_progress = 0
        self._softwalls_opened = 0
        self._explore_target = None
        self._explore_blocked_targets = set()
        self._engage_blocked_keys = set()
        self._combat_hp_floor = {}
        self._controller_snapshot = None

    # ---------- 内部 ----------

    def _record_visit(self, pos) -> bool:
        """登记真实落脚点；首次踏足同时推进 Options 的探索进展钟。

        R16 C6(progress_far_tiles>0):足迹登记不变,但只有与既有"进展锚点"
        (起点 + 历次已计进展的格)切比雪夫距离 ≥ progress_far_tiles 的新格才
        推进 exploration_progress,并自身成为新锚点。沿走廊直行每 far 格计
        一次;在足迹附近踱步永远计不到。锚点若按"全部足迹"衡量,直行时脚
        后跟永远距离 1,任何走动都计不到——故以锚点集而非足迹集作参照。
        """
        point = (int(pos[0]), int(pos[1]))
        if point in self._visited:
            return False
        self._visited.add(point)
        far = int(getattr(self, "_progress_far_tiles", 0))
        if far > 0:
            anchors = getattr(self, "_progress_anchors", None)
            if anchors is None:
                anchors = set(self._visited)
                anchors.discard(point)
                self._progress_anchors = anchors
            if any(
                (point[0] + dx, point[1] + dy) in anchors
                for dx in range(1 - far, far)
                for dy in range(1 - far, far)
            ):
                return True
            anchors.add(point)
        self._exploration_progress += 1
        return True

    def _apply_action(
        self,
        action: int,
        *,
        worker_authority: bool = False,
    ) -> int:
        obs = self._raw
        px, py = obs["player_x"], obs["player_y"]
        if action == 0:
            bridge.act_wait()
            return 1
        elif 1 <= action <= 8:
            dx, dy = _DIRS[action - 1]
            # All learned direction keys use the same exact-one-step native
            # commit.  Restricting this to Worker left flat/base training able
            # to invoke unrestricted path replanning and cross protected
            # progression tiles under a nominal adjacent action.
            direction_protected = (self._resource_direction_protected_tiles(obs)
                if getattr(self, "resource_protocol", "off") != "off"
                else self._explore_protected_tiles(obs))
            protected = sorted(
                point
                for point in direction_protected
                if max(abs(point[0] - px), abs(point[1] - py)) <= 1
            )
            return bridge.act_explore_walk(
                px + dx,
                py + dy,
                protected,
                px,
                py,
                1,
            )
        raise RuntimeError(f"_apply_action 收到未知原子动作 {action}")

    @staticmethod
    def _record_native_execution(
        audit: dict,
        result,
        label: str,
    ) -> bool:
        """Record one checked native 0/1 command receipt."""
        if (
            isinstance(result, (bool, np.bool_))
            or isinstance(result, (int, np.integer))
        ):
            accepted = int(result)
        else:
            raise RuntimeError(
                f"{label} 原生执行回执必须是整数 0/1，"
                f"收到 {result!r}")
        if accepted not in (0, 1):
            raise RuntimeError(
                f"{label} 原生执行回执必须是 0/1，收到 {accepted}")
        audit["attempts"] = int(audit.get("attempts", 0)) + 1
        audit["accepts"] = int(audit.get("accepts", 0)) + accepted
        return bool(accepted)

    def _wait_step(self):
        """取消旧命令并消耗一个标准 micro-step；供无目标/不可达宏返回。"""
        bridge.act_wait()
        raw = self._step_native()
        if (not raw.get("dead") and not raw.get("game_over") and not raw.get("victory")
                and int(raw.get("dest_action", bridge.ACTION_NONE)) != bridge.ACTION_NONE):
            raise RuntimeError(
                f"wait step 后仍有 destAction={raw.get('dest_action')}")
        if (not raw.get("dead") and not raw.get("game_over") and not raw.get("victory")
                and int(raw.get("walkpath0", bridge.WALK_NONE)) != bridge.WALK_NONE):
            raise RuntimeError(
                f"wait step 后仍有 walkpath0={raw.get('walkpath0')}")
        return raw, 1

    @staticmethod
    def _finish_macro(raw, beats: int, start_scene):
        """宏结束时先清路径/destAction；外层统一结算隐藏动画。

        wait 的 loopback FIFO 栅栏保证下一策略动作不再继承路径或攻击；
        当前单格/受击动画所需的有成本拍由 ``_settle_to_idle`` 统一推进，
        不在各宏里复制循环，也不免费消耗游戏时间。
        """
        if (raw.get("dead") or raw.get("game_over") or raw.get("victory")
                or _scene_identity(raw) != start_scene):
            return raw, beats
        if not bridge.act_wait():
            return raw, beats
        refreshed = bridge.observe()
        if raw.get("resource_state", {}).get("preserve_equipment_readiness") is True:
            from .resource_protocol import (
                validate_native_armor_scope, validate_native_equipment_readiness_preservation)
            validate_native_armor_scope(
                refreshed, raw["resource_state"].get("ordinary_armor_scope", False))
            validate_native_equipment_readiness_preservation(refreshed, True)
        if (int(refreshed.get("dest_action", bridge.ACTION_NONE)) != bridge.ACTION_NONE
                or int(refreshed.get("walkpath0", bridge.WALK_NONE)) != bridge.WALK_NONE):
            raise RuntimeError(
                "宏返回前 wait 未清空原生命令状态: "
                f"dest_action={refreshed.get('dest_action')}, "
                f"walkpath0={refreshed.get('walkpath0')}")
        return refreshed, beats

    @staticmethod
    def _engage_distance(raw, monster) -> int:
        """与 CMD_ATTACKID 完全同口径的追击距离（future→future）。

        player/monster 的 tile 会在一格走路动画开始时先改，而 future 是
        引擎已经提交的终点。用 tile 比较会把一段合法的 8~10 tick 动画
        误报为“原地不动”。
        """
        px = int(raw.get("future_x", raw["player_x"]))
        py = int(raw.get("future_y", raw["player_y"]))
        mx = int(monster.get("future_x", monster["x"]))
        my = int(monster.get("future_y", monster["y"]))
        return max(abs(mx - px), abs(my - py))

    def _select_engage_target(
        self,
        raw,
        *,
        exclude: set[tuple[int, int, int]] | None = None,
        allow_blocked_cycle: bool = False,
        candidate_keys: tuple[tuple[int, int, int], ...] | None = None,
    ):
        """取最近且未证失败的目标；稳定 tie 用 id 打破，便于可复现审计。

        同一宏内失败目标绝不重选。跨宏保留失败集合；只有当前所有候选都
        已轮过一遍时才清一轮重试，避免单只动态怪永久失去可攻击性。
        """
        excluded = exclude or set()
        policy_monsters = self._policy_monsters(raw)
        if candidate_keys is None:
            candidates = [
                monster for monster in policy_monsters
                if self._monster_generation_key(monster) not in excluded
            ]
        else:
            # a9 的候选宇宙在策略观测时已固定。宏内只允许按该 canonical
            # 顺序过滤死亡/失去可达性的成员，绝不把第 33 个怪物偷偷补进来。
            by_key = {
                self._monster_generation_key(monster): monster
                for monster in policy_monsters
            }
            candidates = [
                by_key[generation_key]
                for generation_key in candidate_keys
                if generation_key in by_key
                and generation_key not in excluded
            ]
        if not candidates:
            return None
        if candidate_keys is None:
            px = int(raw.get("future_x", raw["player_x"]))
            py = int(raw.get("future_y", raw["player_y"]))
            candidates.sort(key=lambda m: (
                abs(int(m.get("future_x", m["x"])) - px)
                + abs(int(m.get("future_y", m["y"])) - py),
                int(m["id"]),
            ))

        policy_keys = (
            {self._monster_generation_key(m) for m in policy_monsters}
            if candidate_keys is None
            else set(candidate_keys)
        )
        # 消失/离开可见可达集合的 generation 不能污染新生命周期；同一
        # native slot id 被 spawn 复用时，rndItemSeed 会产生新 key。
        self._engage_blocked_keys.intersection_update(policy_keys)
        for monster in candidates:
            if (
                self._monster_generation_key(monster)
                not in self._engage_blocked_keys
            ):
                return monster
        if not allow_blocked_cycle:
            return None
        self._engage_blocked_keys.difference_update(
            self._monster_generation_key(m) for m in candidates)
        return candidates[0]

    @staticmethod
    def _movement_engine_busy(raw) -> bool:
        """玩家仍在执行/排队走路；tile 暂停不代表动画或路径停了。"""
        walking_modes = {
            bridge.PM_WALK_NORTHWARDS,
            bridge.PM_WALK_SOUTHWARDS,
            bridge.PM_WALK_SIDEWAYS,
        }
        return (
            int(raw.get("walkpath0", bridge.WALK_NONE)) != bridge.WALK_NONE
            or int(raw.get("player_mode", -1)) in walking_modes
            or (
                int(raw.get("future_x", raw["player_x"])) != int(raw["player_x"])
                or int(raw.get("future_y", raw["player_y"])) != int(raw["player_y"])
            )
        )

    @classmethod
    def _engage_engine_busy(cls, raw) -> bool:
        """追击包已排队、仍在走一格或挥刀时，几何不变不等于卡死。"""
        return (
            cls._movement_engine_busy(raw)
            or int(raw.get("dest_action", bridge.ACTION_NONE))
            == bridge.ACTION_ATTACKMON
            or int(raw.get("player_mode", -1)) == bridge.PM_ATTACK
        )

    @staticmethod
    def _decision_idle(raw) -> bool:
        """策略可观测边界：不存在任何未入 295 维向量的玩家执行态。"""
        return (
            int(raw.get("player_mode", -1)) == bridge.PM_STAND
            and int(raw.get("dest_action", bridge.ACTION_NONE))
            == bridge.ACTION_NONE
            and int(raw.get("walkpath0", bridge.WALK_NONE))
            == bridge.WALK_NONE
            and int(raw.get("future_x", raw["player_x"]))
            == int(raw["player_x"])
            and int(raw.get("future_y", raw["player_y"]))
            == int(raw["player_y"])
        )

    @classmethod
    def _episode_boundary(cls, raw, steps: int, max_steps: int):
        """分类原生终局、可 bootstrap 时限与不可观测的预算中断。

        返回 ``(terminated, truncated, budget_exhausted, decision_idle,
        unsettled_budget_terminal)``。后两种 Gym 边界严格互斥。
        """
        native_terminated = bool(
            raw.get("dead") or raw.get("game_over") or raw.get("victory"))
        budget_exhausted = int(steps) >= int(max_steps)
        decision_idle = cls._decision_idle(raw)
        unsettled_budget_terminal = bool(
            budget_exhausted and not native_terminated and not decision_idle)
        terminated = bool(native_terminated or unsettled_budget_terminal)
        truncated = bool(budget_exhausted and not terminated)
        return (
            terminated,
            truncated,
            budget_exhausted,
            decision_idle,
            unsettled_budget_terminal,
        )

    def _settle_to_idle(
        self,
        raw,
        beats: int,
        *,
        max_beats: int,
        start_scene,
    ):
        """取消长命令并用有成本的 engine beats 结清隐藏动画。

        死亡/胜利立即交给外层终止逻辑；换 scene 后仍在新图继续结算，
        保证 manager 的首个观测也为 idle。若总步数预算先耗尽，允许返回
        busy raw，但同一次 ``step`` 会把它标成
        ``unsettled_budget_terminal`` 而非可 bootstrap 的 truncation；
        它既不会成为下一次策略决策输入，也不会被价值函数当作 idle 别名。
        """
        beats = int(beats)
        max_beats = int(max_beats)
        terminal = (
            raw.get("dead") or raw.get("game_over") or raw.get("victory")
        )
        calibration = getattr(self, "_resource_calibration", None)
        resource_command = getattr(self, "_resource_pending_command", None)
        bounded_resource = bool(resource_command and resource_command[0]
                                in ("attack_monster", "drink", "unequip"))

        bounded_dive = getattr(self, "_dive_blocker_audit", None) is not None
        completion_time = getattr(self, "resource_time_callback", None) is not None
        settle_clock_start = (self._resource_actual_microsteps - beats
                              if completion_time else None)

        def settle_limit():
            if not completion_time:
                return max_beats
            physical = self.max_steps - settle_clock_start
            # A new arrival may extend or shorten the physical horizon inside
            # this very action. Only its pending animation gets the new room;
            # resource-command budgets retain their existing tighter limits.
            return physical if resource_command is None else min(max_beats, physical)

        def resource_time_available():
            return ((not bounded_resource or self._resource_actual_microsteps < min(
                self.max_steps,
                int(getattr(self, "_resource_service_deadline", self.max_steps))))
                and (not bounded_dive or self._resource_actual_microsteps < self.max_steps)
                and (not completion_time or self._resource_actual_microsteps < self.max_steps))

        if (terminal or beats >= settle_limit() or not resource_time_available()
                or (calibration is not None
                    and self._resource_actual_microsteps >= self.max_steps)):
            return raw, beats

        # ActWait 立即清 path/dest/攻击；已经提交的单格走路、受击/格挡
        # 动画不能硬切，只能继续 game_loop。首次调用也是 FIFO 栅栏，保证
        # 旧网络包不会在下一拍重新装回长命令。
        settle_scene = _scene_identity(raw)
        bridge.act_wait()
        while (beats < settle_limit()
               and not self._decision_idle(raw)
               and resource_time_available()
               and (calibration is None or self._resource_actual_microsteps < self.max_steps)):
            raw = self._step_native()
            beats += 1
            current_scene = _scene_identity(raw)
            if current_scene == start_scene:
                self._record_visit((raw["player_x"], raw["player_y"]))
            if (raw.get("dead") or raw.get("game_over") or raw.get("victory")
                    or not resource_time_available()
                    or (calibration is not None
                        and self._resource_actual_microsteps >= self.max_steps)):
                break
            if current_scene != settle_scene:
                # 换图后的首个 manager/worker 观测同样必须是 idle；重新在
                # 新场景落 FIFO 栅栏，继续结算而不是把 PM_NEWLVL 泄出去。
                settle_scene = current_scene
                bridge.act_wait()
        return raw, beats

    @staticmethod
    def _canonical_engage_candidate(
        snapshot: _ControllerSnapshot,
    ) -> _ControllerMonster | None:
        """Return exactly the candidate action 9 will bind for this snapshot."""
        if not snapshot.candidates:
            return None
        return next(
            (
                candidate
                for candidate in snapshot.candidates
                if not candidate.blocked
            ),
            snapshot.candidates[0],
        )

    def _macro_engage(
        self,
        max_beats: int = 10,
        *,
        controller_snapshot: _ControllerSnapshot | None = None,
        execution_audit: dict | None = None,
    ):
        """追击一个可接敌目标；真停滞时在同一 action9 预算内轮换。

        战士的一格走路约需 8~10 engine tick，挥刀到伤害帧还会再用十余
        tick。旧实现仅看两次 4-tick 采样的 tile/HP，恰在首刀伤害帧前
        调用 ActWait 中止攻击，形成永久不掉血。现在只有引擎已经空闲且
        连续两拍既未接近也未伤害才判停滞；pending attack、未完成走格与
        PM_ATTACK 都是正在执行，不得提前取消。
        """
        snapshot = (
            controller_snapshot
            if controller_snapshot is not None
            else self._capture_controller_snapshot(self._raw)
        )
        candidates = list(snapshot.candidates)
        candidate = self._canonical_engage_candidate(snapshot)
        if candidate is None:
            return self._wait_step()
        if candidate.blocked:
            # “全已轮过”这一事实已在 snapshot 每槽 blocked bit 中公开；
            # action9 执行时原子开启下一轮，仍按 wire canonical 首项选择。
            self._engage_blocked_keys.difference_update(
                entry.generation_key for entry in candidates)
        tid = int(candidate.monster_id)
        target_key = candidate.generation_key
        target = next(
            (
                monster for monster in self._raw.get("monsters", ())
                if self._monster_generation_key(monster) == target_key
                and bool(monster.get("visible", True))
            ),
            None,
        )
        if target is None:
            self._engage_blocked_keys.add(target_key)
            return self._wait_step()

        start_scene = _scene_identity(self._raw)
        raw = self._raw
        path = self._plan_controller_path(
            snapshot,
            candidate.future_x,
            candidate.future_y,
            avoid_monsters=True,
            allow_softwalls=False,
        )
        initial_distance = max(
            abs(int(raw.get("future_x", raw["player_x"]))
                - int(target.get("future_x", target["x"]))),
            abs(int(raw.get("future_y", raw["player_y"]))
                - int(target.get("future_y", target["y"]))),
        )
        if path is None and initial_distance > 1:
            self._engage_blocked_keys.add(target_key)
            return self._wait_step()

        protected = sorted(snapshot.protected_tiles)
        pi = 0
        active_step: tuple[int, int] | None = None
        beats = 0
        last_hp = int(target["hp"])
        last_distance = self._engage_distance(raw, target)
        target_start_hp = last_hp
        target_start_distance = last_distance
        target_progress = False
        idle_stall = 0

        for beats in range(1, max_beats + 1):
            current_target = next(
                (monster for monster in raw.get("monsters", ())
                 if self._monster_generation_key(monster) == target_key),
                None,
            )
            if current_target is None:
                self._engage_blocked_keys.discard(target_key)
                break
            target_x = int(current_target.get(
                "future_x", current_target["x"]))
            target_y = int(current_target.get(
                "future_y", current_target["y"]))
            if (
                not bool(current_target.get("visible", True))
                or abs(target_x - snapshot.player_x)
                > CONTROLLER_SNAPSHOT_RADIUS
                or abs(target_y - snapshot.player_y)
                > CONTROLLER_SNAPSHOT_RADIUS
            ):
                self._engage_blocked_keys.add(target_key)
                break

            player_x = int(raw.get("future_x", raw["player_x"]))
            player_y = int(raw.get("future_y", raw["player_y"]))
            adjacent = max(
                abs(player_x - target_x), abs(player_y - target_y)) <= 1
            if self._decision_idle(raw):
                if adjacent:
                    accepted = bridge.act_controller_attack_monster(
                        tid,
                        snapshot.player_x,
                        snapshot.player_y,
                        CONTROLLER_SNAPSHOT_RADIUS,
                    )
                    if execution_audit is not None:
                        self._record_native_execution(
                            execution_audit, accepted, "action9 attack")
                elif active_step is None:
                    if path is None or pi >= len(path):
                        self._engage_blocked_keys.add(target_key)
                        break
                    nx, ny, is_softwall = path[pi]
                    if is_softwall:
                        self._engage_blocked_keys.add(target_key)
                        break
                    if abs(nx - player_x) + abs(ny - player_y) != 1:
                        self._engage_blocked_keys.add(target_key)
                        break
                    accepted = bridge.act_explore_walk(
                        nx,
                        ny,
                        protected,
                        snapshot.player_x,
                        snapshot.player_y,
                        CONTROLLER_SNAPSHOT_RADIUS,
                    )
                    if execution_audit is not None:
                        accepted = self._record_native_execution(
                            execution_audit, accepted, "action9 walk")
                    else:
                        accepted = int(accepted) == 1
                    if not accepted:
                        self._engage_blocked_keys.add(target_key)
                        break
                    active_step = (nx, ny)

            raw = self._step_native()
            if _scene_identity(raw) == start_scene:
                self._record_visit((raw["player_x"], raw["player_y"]))
            if (raw["dead"] or _scene_identity(raw) != start_scene):
                break
            # action9 is a multi-beat controller, while the wrapper's
            # emergency potion reflex runs only after control returns.  The
            # old loop could keep attacking for the rest of its ten-beat
            # budget after HP had already crossed below 50%, killing players
            # that still had several potions.  End the macro at the first
            # observed reflex-eligible state so the normal tail drain gets a
            # chance; no potion is consumed or hidden inside action9.
            if self._reflex_eligible(raw):
                break

            cur_target = next(
                (
                    m for m in raw["monsters"]
                    if self._monster_generation_key(m) == target_key
                ),
                None,
            )
            if cur_target is None:
                # 死亡/消失/slot 复用都是旧 generation 成功结束；新怪必须
                # 等下一次策略观测，不能继承本宏的攻击。
                self._engage_blocked_keys.discard(target_key)
                break

            cur_hp = int(cur_target["hp"])
            cur_distance = self._engage_distance(raw, cur_target)
            position = (int(raw["player_x"]), int(raw["player_y"]))
            if active_step is not None and position == active_step:
                pi += 1
                active_step = None
            hp_progress = cur_hp < last_hp
            distance_progress = cur_distance < last_distance
            if hp_progress or distance_progress:
                target_progress = True
                idle_stall = 0
                self._engage_blocked_keys.discard(target_key)
            elif self._engage_engine_busy(raw):
                # 动画/命令仍活跃；尤其不能在 PM_ATTACK 的伤害帧前止损。
                idle_stall = 0
            else:
                idle_stall += 1

            # 固定快照路径不允许在宏中重规划/换目标；真 idle 连续无进展
            # 就记失败并把控制权交还，由下一份观测重新决定。
            if idle_stall >= 2:
                self._engage_blocked_keys.add(target_key)
                break

            last_hp = cur_hp
            last_distance = cur_distance

        # 已耗尽整个宏仍既没打掉血也没比目标起点更接近，下一 action9
        # 优先尝试别的候选。正在执行的动画本次仍允许完整预算，绝不因
        # 这项跨宏轮转规则提前 ActWait。
        if (not raw.get("dead")
                and _scene_identity(raw) == start_scene):
            surviving = next(
                (m for m in raw.get("monsters", ())
                 if self._monster_generation_key(m) == target_key), None)
            if surviving is not None:
                effective_progress = (
                    target_progress
                    or int(surviving["hp"]) < target_start_hp
                    or self._engage_distance(raw, surviving)
                    < target_start_distance
                )
                if not effective_progress:
                    self._engage_blocked_keys.add(target_key)

        return self._finish_macro(raw, beats, start_scene)

    _EXPLORE_RADIUS = CONTROLLER_SNAPSHOT_RADIUS  # 25×25 固定控制快照

    _PROGRESSION_PRIORITY = {
        "lazarus_stand": 0,
        "lazarus_staff": 1,
        "vile_entrance": 2,
        "vile_book": 3,
        "vile_center_circle": 4,
        "diablo_switch": 5,
    }

    @staticmethod
    def _progression_present(raw, target) -> bool:
        """坐标+种类是单场景内稳定身份；目标消失即本次交互已提交。"""
        return any(
            p.get("kind") == target.get("kind")
            and int(p.get("x", -1)) == int(target["x"])
            and int(p.get("y", -1)) == int(target["y"])
            for p in raw.get("progression_targets", ())
        )

    @staticmethod
    def _progression_ready(raw, target) -> bool:
        px, py = int(raw["player_x"]), int(raw["player_y"])
        tx, ty = int(target["x"]), int(target["y"])
        gx, gy = int(target["goal_x"]), int(target["goal_y"])
        action = target["action"]
        if action == "walk":
            return (px, py) == (gx, gy)
        if action == "pickup":
            # CMD_GOTOAGETITEM may otherwise install a new multi-tile route
            # after the safe planner has finished.  Walk onto the exact item
            # tile first, then the pickup command has no route left to invent.
            return (px, py) == (tx, ty)
        if action == "operate" and bool(target.get("exact")):
            return (px, py) == (gx, gy)
        if action == "operate":
            return max(abs(px - tx), abs(py - ty)) <= 1
        raise RuntimeError(f"未知剧情目标动作: {action!r}")

    @staticmethod
    def _issue_progression(
        target,
        *,
        center_x: int,
        center_y: int,
        radius: int,
    ) -> bool:
        action = target["action"]
        if action == "operate":
            result = bridge.act_controller_operate(
                int(target["x"]),
                int(target["y"]),
                center_x,
                center_y,
                radius,
            )
        elif action == "pickup":
            result = bridge.act_pickup_progression(
                int(target["x"]), int(target["y"]))
        elif action == "walk":
            result = bridge.act_explore_walk(
                int(target["goal_x"]),
                int(target["goal_y"]),
                (),
                center_x,
                center_y,
                radius,
            )
        else:
            raise RuntimeError(f"未知剧情目标动作: {action!r}")
        if not isinstance(result, (bool, np.bool_, int, np.integer)):
            raise RuntimeError(
                "action11 progression 原生回执必须是整数 0/1，"
                f"收到 {result!r}")
        accepted = int(result)
        if accepted not in (0, 1):
            raise RuntimeError(
                "action11 progression 原生回执必须是 0/1，"
                f"收到 {accepted}")
        return accepted == 1

    def _macro_progression(
        self,
        max_beats: int = 12,
        *,
        execution_audit: dict | None = None,
    ):
        """推进严格白名单中的下一项通关必需交互。

        仅 action 11 / DIVE 经理调用此宏；action 10 / FARM 在剧情态会
        fail-closed 交还控制权，不能越权推进剧情。Vile 书要求精确站圈；
        普通机关只要相邻。全局 BFS 仍只把门/桶当软墙，不传送、不穿墙。
        """
        raw = self._raw
        targets = [dict(p) for p in raw.get("progression_targets", ())]
        if not targets:
            return self._wait_step()

        candidates = []
        for target in targets:
            required = {"kind", "action", "x", "y", "goal_x", "goal_y", "exact"}
            if set(target) != required:
                raise RuntimeError(f"剧情目标 schema 异常: {target!r}")
            ready = self._progression_ready(raw, target)
            path = [] if ready else self._plan_descend_path(
                raw, int(target["goal_x"]), int(target["goal_y"]),
                avoid_monsters=True)
            px, py = int(raw["player_x"]), int(raw["player_y"])

            def assess(candidate_path):
                ex, ey = ((candidate_path[-1][0], candidate_path[-1][1])
                          if candidate_path else (px, py))
                if target["action"] == "operate" and not bool(target["exact"]):
                    remaining_ = max(abs(ex - int(target["x"])),
                                     abs(ey - int(target["y"])))
                    limit = 1
                elif target["action"] == "pickup":
                    remaining_ = max(abs(ex - int(target["x"])),
                                     abs(ey - int(target["y"])))
                    limit = 0
                else:
                    remaining_ = max(abs(ex - int(target["goal_x"])),
                                     abs(ey - int(target["goal_y"])))
                    limit = 0
                reachable_ = ready or (
                    candidate_path is not None and remaining_ <= limit)
                return remaining_, reachable_

            remaining, reachable = assess(path)
            if not ready and not reachable:
                # “避怪 BFS”即使只能走到怪物墙前也会返回一条 partial path，
                # 不是 None。若只在 None 时回退，L16 会把真正可达的第二个
                # switch 错判为远目标，反复走向另一扇尚未开放的墙。
                fallback = self._plan_descend_path(
                    raw, int(target["goal_x"]), int(target["goal_y"]))
                fallback_remaining, fallback_reachable = assess(fallback)
                old_rank = (not reachable, remaining,
                            len(path) if path is not None else 10**9)
                new_rank = (not fallback_reachable, fallback_remaining,
                            len(fallback) if fallback is not None else 10**9)
                if new_rank < old_rank:
                    path = fallback
                    remaining, reachable = fallback_remaining, fallback_reachable
            priority = self._PROGRESSION_PRIORITY.get(target["kind"], 999)
            candidates.append((not reachable, priority, remaining,
                               len(path) if path is not None else 10**9,
                               target, path))

        _, _, _, _, target, path = min(candidates, key=lambda c: c[:4])
        if path is None and not self._progression_ready(raw, target):
            return self._wait_step()

        start_scene = _scene_identity(raw)
        center_x = int(raw["player_x"])
        center_y = int(raw["player_y"])
        pi = 0
        command = None  # ("open"/"walk"/"progress", x, y, path_index)
        last_pos = (raw["player_x"], raw["player_y"])
        stall = 0
        beats = 0
        for beats in range(1, max_beats + 1):
            if not self._progression_present(raw, target):
                break
            if command is None:
                if self._progression_ready(raw, target):
                    accepted = self._issue_progression(
                        target,
                        center_x=center_x,
                        center_y=center_y,
                        radius=self._DESCEND_RADIUS,
                    )
                    if execution_audit is not None:
                        accepted = self._record_native_execution(
                            execution_audit,
                            1 if accepted else 0,
                            "action11 progression",
                        )
                    if not accepted:
                        break
                    command = ("progress", int(target["x"]), int(target["y"]), pi)
                elif pi >= len(path):
                    break
                else:
                    j = pi
                    command = (
                        ("open", path[j][0], path[j][1], j)
                        if path[j][2]
                        else ("walk", path[j][0], path[j][1], j)
                    )
                    if command[0] == "open":
                        accepted = bridge.act_controller_operate(
                            command[1],
                            command[2],
                            center_x,
                            center_y,
                            self._DESCEND_RADIUS,
                        )
                    else:
                        accepted = bridge.act_explore_walk(
                            command[1],
                            command[2],
                            (),
                            center_x,
                            center_y,
                            self._DESCEND_RADIUS,
                        )
                    if execution_audit is not None:
                        accepted = self._record_native_execution(
                            execution_audit, accepted,
                            f"action11 {command[0]}")
                    else:
                        accepted = int(accepted) == 1
                    if not accepted:
                        break

            raw = self._step_native()
            pos = (raw["player_x"], raw["player_y"])
            if _scene_identity(raw) == start_scene:
                self._record_visit(pos)
            if raw["dead"] or _scene_identity(raw) != start_scene:
                break
            if self._reflex_eligible(raw):
                break
            if not self._progression_present(raw, target):
                break

            if command[0] == "open":
                if bridge.probe_tile(command[1], command[2])["walkable"]:
                    path[command[3]] = (command[1], command[2], False)
                    pi = command[3]
                    command = None
                    stall = 0
                    last_pos = pos
                    continue
            elif command[0] == "walk":
                # act_explore_walk now proves and installs exactly one adjacent
                # edge.  Do not consume that edge while the player's tile is
                # merely still adjacent to its endpoint (which is also true at
                # the starting tile); wait until the animation really lands.
                if pos == (command[1], command[2]):
                    pi = command[3] + 1
                    command = None
                    stall = 0
                    last_pos = pos
                    continue

            if pos == last_pos:
                stall += 1
                if stall == 3:
                    if command[0] == "open":
                        accepted = bridge.act_controller_operate(
                            command[1],
                            command[2],
                            center_x,
                            center_y,
                            self._DESCEND_RADIUS,
                        )
                    elif command[0] == "walk":
                        accepted = bridge.act_explore_walk(
                            command[1],
                            command[2],
                            (),
                            center_x,
                            center_y,
                            self._DESCEND_RADIUS,
                        )
                    else:
                        accepted = self._issue_progression(
                            target,
                            center_x=center_x,
                            center_y=center_y,
                            radius=self._DESCEND_RADIUS,
                        )
                        accepted = 1 if accepted else 0
                    if execution_audit is not None:
                        accepted = self._record_native_execution(
                            execution_audit, accepted,
                            f"action11 retry {command[0]}")
                    else:
                        accepted = int(accepted) == 1
                    if not accepted:
                            break
                if stall >= 6:
                    break
            else:
                stall = 0
            last_pos = pos
        return self._finish_macro(raw, beats, start_scene)

    def _resource_direction_protected_tiles(self, raw):
        protected = self._explore_protected_tiles(raw)
        if not getattr(self, "_resource_dive_authority", False):
            return protected
        from .resource_protocol import progression_allowed
        if not progression_allowed(raw, getattr(self, "resource_readiness_law", "veto-v1")):
            return protected
        transition = bridge.WM_DIABRTNLVL if raw.get("is_set_level") else bridge.WM_DIABNEXTLVL
        allowed = {(int(t["x"]), int(t["y"])) for t in raw.get("triggers", [])
                   if t.get("msg") == transition}
        for target in raw.get("progression_targets", []):
            allowed.add((int(target["x"]), int(target["y"])))
            allowed.add((int(target["goal_x"]), int(target["goal_y"])))
        return protected - allowed

    @staticmethod
    def _explore_protected_tiles(raw) -> set[tuple[int, int]]:
        protected = {
            (int(t["x"]), int(t["y"]))
            for t in raw.get("triggers", ())
        }
        for target in raw.get("progression_targets", ()):
            protected.add((int(target["x"]), int(target["y"])))
            protected.add((int(target["goal_x"]), int(target["goal_y"])))
        return protected

    @classmethod
    def _protected_walk_actions(cls, raw) -> frozenset[int]:
        """Adjacent direction keys that would trespass on DIVE-only tiles."""
        px, py = int(raw["player_x"]), int(raw["player_y"])
        protected = cls._explore_protected_tiles(raw)
        return frozenset(
            action
            for action, (dx, dy) in enumerate(_DIRS, start=1)
            if (px + dx, py + dy) in protected
        )

    def _plan_explore_step(
        self,
        raw,
        *,
        blocked_softwalls: set[tuple[int, int]] | None = None,
        controller_snapshot: _ControllerSnapshot | None = None,
    ):
        """为 action10 规划原有边疆点，或通向普通软墙的可达站位。

        规划只穿当前 walkable 连通域；trigger 与剧情目标/goal 是硬禁区，
        因而探索不会踩楼梯或借普通 operate 越过 DIVE 的剧情职权。8 步内
        的普通闭门可在经过时优先处理；blocking barrel 只在连通域已经没有
        未踏足边疆时兜底。已有边疆保持原 action10 的远路点连续寻路语义。
        """
        snapshot = controller_snapshot
        if snapshot is not None:
            px, py = snapshot.player_x, snapshot.player_y
        else:
            px, py = int(raw["player_x"]), int(raw["player_y"])
        r = self._EXPLORE_RADIUS
        side = 2 * r + 1
        if snapshot is None:
            lm = bridge.local_map(radius=r)
            walk = lm["walkable"]
            occupied = lm["monster"]
            softwall = lm.get("door", [0] * (side * side))
            ordinary_door = lm.get("closed_door", softwall)
            hazard = lm.get("hazard", [0] * (side * side))
            explosive_softwall = lm.get(
                "explosive_softwall", [0] * (side * side))
            protected = self._explore_protected_tiles(raw)
            visited = self._visited
            blocked_targets = getattr(
                self, "_explore_blocked_targets", set())
            sticky = getattr(self, "_explore_target", None)
        else:
            walk = snapshot.walkable
            occupied = snapshot.physical_monster
            softwall = snapshot.softwall
            ordinary_door = snapshot.closed_door
            hazard = snapshot.hazard
            explosive_softwall = snapshot.explosive_softwall
            protected = set(snapshot.protected_tiles)
            visited = set(snapshot.visited_tiles)
            blocked_targets = set(snapshot.explore_blocked_targets)
            sticky = snapshot.sticky_target
        blocked = blocked_softwalls or set()

        # 旧版仅按目标格自身 walkable + 几何距离挑边疆，可能选中五格外但
        # 实际要绕 160 格迷宫才能抵达的地板；引擎 100 格路径上限拒绝后，
        # 每次 action10 又会选同一格永久面壁。先在局部窗做保守 4 向连通域，
        # 只把本次原生寻路确实可到的格作为候选。怪物占位与职权禁区不能穿过。
        reachable = {(px, py)}
        depth = {(px, py): 0}
        queue = deque([(px, py)])

        def local_index(tx, ty):
            return (ty - py + r) * side + (tx - px + r)

        def in_window(tx, ty):
            return abs(tx - px) <= r and abs(ty - py) <= r

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if ((nx, ny) in reachable
                        or not in_window(nx, ny)
                        or (nx, ny) in protected):
                    continue
                i = local_index(nx, ny)
                if (
                    not walk[i]
                    or occupied[i]
                    or hazard[i]
                    or explosive_softwall[i]
                ):
                    continue
                reachable.add((nx, ny))
                depth[(nx, ny)] = depth[(cx, cy)] + 1
                queue.append((nx, ny))

        # 候选:可走、离玩家 ≥5 格、且不在足迹邻域(±1)内的边疆点
        near_visited = visited | {
            (x + dx, y + dy) for x, y in visited for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        }
        candidates = []
        retry_candidates = []
        for tx, ty in reachable:
            d_player = max(abs(tx - px), abs(ty - py))
            if (d_player >= 5
                    and (tx, ty) not in near_visited):
                entry = (d_player, tx, ty)
                if (tx, ty) in blocked_targets:
                    retry_candidates.append(entry)
                else:
                    candidates.append(entry)

        # 搜索当前连通域边界上真正连接另一块潜在空间的普通软墙。
        # local_map["door"] 精确包含 closedDoor 或 solid breakable barrel；
        # progression/trigger 坐标仍由 protected 双重排除。
        softwall_candidates = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                door = (px + dx, py + dy)
                if (door in blocked
                        or door in protected
                        or not softwall[local_index(*door)]
                        or explosive_softwall[local_index(*door)]
                        or hazard[local_index(*door)]):
                    continue
                approaches = []
                has_unseen_side = False
                for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor = (door[0] + ox, door[1] + oy)
                    if neighbor in reachable:
                        approaches.append(neighbor)
                        continue
                    if neighbor in protected:
                        continue
                    if not in_window(*neighbor):
                        has_unseen_side = True
                        continue
                    ni = local_index(*neighbor)
                    # 怪物占位是动态的；地板本身存在便足以证明门后不是实墙。
                    if (
                        not hazard[ni]
                        and not explosive_softwall[ni]
                        and (walk[ni] or softwall[ni])
                    ):
                        has_unseen_side = True
                if not approaches or not has_unseen_side:
                    continue
                approach = min(
                    approaches,
                    key=lambda p: (depth[p], p[0], p[1]),
                )
                softwall_candidates.append((
                    0 if ordinary_door[local_index(*door)] else 1,
                    depth[approach],
                    max(abs(door[0] - px), abs(door[1] - py)),
                    door[0], door[1], approach,
                ))

        chosen_softwall = None
        if softwall_candidates:
            # 有普通边疆时只顺路处理 8 步内的真门；桶保留给连通域已经
            # 完全无边疆时的最后软墙兜底，避免 FARM 沿途沉迷砸桶。
            nearby_doors = [
                entry for entry in softwall_candidates
                if entry[0] == 0 and entry[1] <= 8
            ]
            if nearby_doors:
                chosen_softwall = min(nearby_doors)
            elif not candidates:
                chosen_softwall = min(softwall_candidates)
        if chosen_softwall is not None:
            _, _, _, door_x, door_y, approach = chosen_softwall
            if approach != (px, py):
                return ("approach", approach[0], approach[1])
            return ("open", door_x, door_y)

        # frontier 失败记忆只负责“先轮换别的候选”，不能把动态占位造成的
        # 一次失败永久升级成整场景黑名单。当前可用的未阻塞候选耗尽后，
        # 原子开启下一轮并重试仍然可达/未踏足的目标；否则同一个 hidden
        # set 会让完全相同的可见地图从“可探索”永久变成 wait。
        if not candidates and retry_candidates:
            self._explore_blocked_targets.difference_update(
                (tx, ty) for _, tx, ty in retry_candidates)
            candidates = retry_candidates

        # 边疆目标跨 action10 保持，直到真正抵达/失效。不能在每个宏的
        # 新当前位置上重做“最近点”贪心，否则分叉两侧会互相抢占最近名次，
        # 形成 seed7002 实锤的 A↔B 永久往返。
        if sticky is not None:
            sticky = (int(sticky[0]), int(sticky[1]))
            if (sticky in protected
                    or sticky in blocked_targets
                    or sticky in visited
                    or sticky not in reachable
                    or max(abs(sticky[0] - px), abs(sticky[1] - py)) <= 1):
                self._explore_target = None
            else:
                return ("frontier", sticky[0], sticky[1])
        # R16 C5 附加 hunt(默认关):窗内没有任何可见怪(隔墙/未照亮的怪不
        # 算可见)时优先全图寻怪;窗内一旦有可见怪就交还局部逻辑/a9 掩码/反射
        # (探针:以"无可接敌候选"为闸会在可见但暂不可接敌的怪旁 a9↔a10 振荡,
        # seed7002 击杀 96→48;以"窗内无占位"为闸则隔墙暗怪把 hunt 锁死,
        # seed9005 原地徘徊)。全图无可达存活怪(None)才落回局部边疆/回退。
        if getattr(self, "_explore_global_hunt", False):
            if snapshot is not None:
                visible_in_window = any(snapshot.visible_monster)
            else:
                visible_in_window = any(
                    bool(m.get("visible"))
                    and in_window(int(m["x"]), int(m["y"]))
                    for m in raw.get("monsters", ())
                )
            if not visible_in_window:
                command = self._plan_explore_global_waypoint(
                    raw, px, py, reachable, blocked_targets,
                    monsters_only=True)
                if command is not None:
                    self._explore_target = (
                        (command[1], command[2])
                        if command[0] == "frontier" else None)
                    return command
        if candidates:
            _, tx, ty = min(candidates)  # 最近的边疆点(便宜且稳)
            self._explore_target = (tx, ty)
            return ("frontier", tx, ty)
        # R16 C5(默认关):窗内无候选时全图 BFS 回退,取通往最近的未踏足
        # 可达格/存活怪占位格路径上、落在本窗内且局部可达的最远前缀点作为
        # 本次 frontier(前缀被窗内软墙截断则改发 approach/open);随后交给
        # 既有的逐步走格/粘性目标机制。仍无候选才返回 None。
        if getattr(self, "_explore_global_fallback", False):
            command = self._plan_explore_global_waypoint(
                raw, px, py, reachable, blocked_targets)
            if command is not None:
                self._explore_target = (
                    (command[1], command[2])
                    if command[0] == "frontier" else None)
                return command
        self._explore_target = None
        return None

    def _plan_explore_global_waypoint(
        self,
        raw,
        px: int,
        py: int,
        reachable,
        blocked_targets,
        *,
        monsters_only: bool = False,
    ):
        """R16 C5:全图 4 向 BFS(同 _plan_descend_path 的口径:关门视为可通,
        hazard/explosive-softwall/trigger/剧情格为墙)找最近目标,返回一条
        与局部规划同款的命令:("frontier", x, y) = 其路径在 25×25 窗内、且
        属于本次局部 reachable 连通域的最远前缀点;若前缀被窗内软墙截断则
        返回 ("approach", x, y) / ("open", x, y);无目标/无前缀返回 None。

        目标(BFS 首个弹出即最近):
          - 存活怪占位格(radius-112 monster 通道,dMonster≠0);
          - 窗外、未踏足且不在全局足迹 ±1 光环内的可走格。
        窗内的可走未踏足格已由局部候选穷举(离玩家 <5 或在光环内的不算),
        不再重复作为目标。怪物占位格只作目标不再扩展。整段无随机数。
        monsters_only=True(hunt 变体)只把怪物占位格当目标。
        前缀点若已在 blocked_targets 中则退到更近的前缀点;全部被阻塞时
        与局部候选同款:清除这些阻塞记忆后仍取最远前缀点,避免同一可见
        地图永久退化成 wait。
        """
        r = self._EXPLORE_RADIUS
        big = self._DESCEND_RADIUS
        side = 2 * big + 1
        lm = bridge.local_map(radius=big)
        walk, door = lm["walkable"], lm["door"]
        hazard = lm.get("hazard", [0] * (side * side))
        explosive = lm.get("explosive_softwall", [0] * (side * side))
        mon = lm["monster"]
        protected = self._explore_protected_tiles(raw)
        visited = getattr(self, "_visited", set())

        def idx(tx, ty):
            return (ty - py + big) * side + (tx - px + big)

        def is_goal(tx, ty):
            if mon[idx(tx, ty)]:
                return True
            if monsters_only or (abs(tx - px) <= r and abs(ty - py) <= r):
                return False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if (tx + dx, ty + dy) in visited:
                        return False
            return True

        start = (px, py)
        prev = {start: None}
        queue = deque([start])
        goal = None
        while queue and goal is None:
            cx, cy = queue.popleft()
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + ddx, cy + ddy
                if (abs(nx - px) > big or abs(ny - py) > big
                        or (nx, ny) in prev or (nx, ny) in protected):
                    continue
                i = idx(nx, ny)
                if hazard[i] or explosive[i]:
                    continue
                if not walk[i] and not door[i]:
                    continue
                prev[(nx, ny)] = (cx, cy)
                if is_goal(nx, ny):
                    goal = (nx, ny)
                    break
                queue.append((nx, ny))
        if goal is None:
            return None
        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        prefix = []
        for point in path:
            if point not in reachable:
                break
            prefix.append(point)
        if len(prefix) < len(path):
            # 前缀被窗内软墙(普通闭门/非爆炸实心桶,BFS 视作可通)截断:改发
            # 既有的 approach/open 命令交给 act_controller_operate,而不是把
            # 航点停在软墙前一格——否则每次都选同一航点永久打转(seed7002
            # hunt 实锤:桶前同一航点重复 25 次)。怪物堵门则仍走航点。
            blocker = path[len(prefix)]
            bi = idx(*blocker)
            if (abs(blocker[0] - px) <= r and abs(blocker[1] - py) <= r
                    and door[bi] and not mon[bi]):
                approach = prefix[-1] if prefix else start
                if approach == start:
                    return ("open", blocker[0], blocker[1])
                return ("approach", approach[0], approach[1])
        if not prefix:
            return None
        for point in reversed(prefix):
            if point not in blocked_targets:
                return ("frontier", point[0], point[1])
        self._explore_blocked_targets.difference_update(prefix)
        return ("frontier", prefix[-1][0], prefix[-1][1])

    def _macro_explore(
        self,
        max_beats: int = 12,
        *,
        controller_snapshot: _ControllerSnapshot | None = None,
        execution_audit: dict | None = None,
    ):
        """探索宏:只执行观测时 radius-12 快照选出的一个确定命令。

        剧情推进是 a11/经理的专属职权；a10 在剧情态 fail-closed 为 wait。
        普通探索绝不走上/下楼 trigger，也不操作 progression 坐标。门打开
        或接近站位完成后立即交还策略，由下一份快照决定下一步，禁止同一
        观测后的第二次 local_map 重规划。
        """
        if self._raw.get("progression_targets"):
            self._explore_target = None
            return self._wait_step()
        raw = self._raw
        snapshot = (
            controller_snapshot
            if controller_snapshot is not None
            else self._capture_controller_snapshot(raw)
        )
        start_scene = _scene_identity(raw)
        command = self._plan_explore_step(
            raw,
            blocked_softwalls=set(),
            controller_snapshot=snapshot,
        )
        if command is None:
            return self._wait_step()

        protected = sorted(snapshot.protected_tiles)
        path = None
        if command[0] != "open":
            path = self._plan_controller_path(
                snapshot,
                command[1],
                command[2],
                avoid_monsters=True,
                allow_softwalls=False,
            )
            if path is None:
                path = self._plan_controller_path(
                    snapshot,
                    command[1],
                    command[2],
                    avoid_monsters=False,
                    allow_softwalls=False,
                )
            if path is None:
                return self._wait_step()
        elif max(
            abs(int(raw["player_x"]) - int(command[1])),
            abs(int(raw["player_y"]) - int(command[2])),
        ) > 1:
            raise RuntimeError(
                "controller snapshot 规划出非相邻 operate")

        pi = 0
        active_step: tuple[int, int] | None = None
        command_issued = False
        last_pos = (int(raw["player_x"]), int(raw["player_y"]))
        stall = 0
        beats = 0
        while beats < max_beats:
            if command[0] == "open":
                if not command_issued and self._decision_idle(raw):
                    accepted = bridge.act_controller_operate(
                        command[1],
                        command[2],
                        snapshot.player_x,
                        snapshot.player_y,
                        CONTROLLER_SNAPSHOT_RADIUS,
                    )
                    if execution_audit is not None:
                        accepted = self._record_native_execution(
                            execution_audit, accepted, "action10 open")
                    else:
                        accepted = int(accepted) == 1
                    if not accepted:
                        break
                    command_issued = True
            elif active_step is None and self._decision_idle(raw):
                if pi >= len(path):
                    break
                nx, ny, is_softwall = path[pi]
                if is_softwall:
                    raise RuntimeError(
                        "action10 普通走路路径意外穿过未打开软墙")
                if abs(nx - int(raw["player_x"])) + abs(
                        ny - int(raw["player_y"])) != 1:
                    raise RuntimeError(
                        "controller snapshot 路径不是 4 向相邻步")
                accepted = bridge.act_explore_walk(
                    nx,
                    ny,
                    protected,
                    snapshot.player_x,
                    snapshot.player_y,
                    CONTROLLER_SNAPSHOT_RADIUS,
                )
                if execution_audit is not None:
                    accepted = self._record_native_execution(
                        execution_audit, accepted, "action10 walk")
                else:
                    accepted = int(accepted) == 1
                if not accepted:
                    break
                active_step = (nx, ny)

            raw = self._step_native()
            beats += 1
            pos = (int(raw["player_x"]), int(raw["player_y"]))
            if _scene_identity(raw) == start_scene:
                self._record_visit(pos)
            nd = self._nearest_dist(raw)
            if (
                raw["dead"]
                or _scene_identity(raw) != start_scene
                or self._reflex_eligible(raw)
                or (nd is not None and nd <= 6)
            ):
                break

            if command[0] == "open":
                if bridge.probe_tile(command[1], command[2])["walkable"]:
                    self._softwalls_opened += 1
                    self._exploration_progress += 1
                    break
                interaction_busy = (
                    self._movement_engine_busy(raw)
                    or int(raw.get("player_mode", -1)) == bridge.PM_ATTACK
                    or int(raw.get("dest_action", bridge.ACTION_NONE))
                    != bridge.ACTION_NONE
                )
            else:
                interaction_busy = self._movement_engine_busy(raw)
                if active_step is not None and pos == active_step:
                    pi += 1
                    active_step = None
                    stall = 0
                    reached = (
                        pos == (int(command[1]), int(command[2]))
                        if command[0] == "approach"
                        else max(
                            abs(pos[0] - int(command[1])),
                            abs(pos[1] - int(command[2])),
                        ) <= 1
                    )
                    if reached:
                        if (
                            command[0] == "frontier"
                            and getattr(self, "_explore_target", None)
                            == (int(command[1]), int(command[2]))
                        ):
                            self._explore_target = None
                        break
                    last_pos = pos
                    continue

            if pos != last_pos or interaction_busy:
                stall = 0
            else:
                stall += 1
                if stall == 3 and self._decision_idle(raw):
                    if command[0] == "open":
                        accepted = bridge.act_controller_operate(
                            command[1],
                            command[2],
                            snapshot.player_x,
                            snapshot.player_y,
                            CONTROLLER_SNAPSHOT_RADIUS,
                        )
                        if execution_audit is not None:
                            accepted = self._record_native_execution(
                                execution_audit, accepted,
                                "action10 reopen")
                        else:
                            accepted = int(accepted) == 1
                        if not accepted:
                            break
                    elif active_step is not None:
                        accepted = bridge.act_explore_walk(
                            active_step[0],
                            active_step[1],
                            protected,
                            snapshot.player_x,
                            snapshot.player_y,
                            CONTROLLER_SNAPSHOT_RADIUS,
                        )
                        if execution_audit is not None:
                            self._record_native_execution(
                                execution_audit, accepted,
                                "action10 rewalk")
                if stall >= 6:
                    break
            if command[0] == "frontier" and stall >= 2:
                failed = (int(command[1]), int(command[2]))
                self._explore_blocked_targets.add(failed)
                if getattr(self, "_explore_target", None) == failed:
                    self._explore_target = None
                break
            last_pos = pos
        return self._finish_macro(raw, beats, start_scene)

    _DESCEND_RADIUS = 112  # 规划窗覆盖全图(地牢 112×112):有的层联通回廊会绕大圈,
                           # 40 格窗曾在 seed 9005 上漏掉西侧绕行路线。每次按键只规划一次,
                           # C++ 端一次调用出图,开销在毫秒级,换全局最优值得

    def _plan_descend_path(self, raw, sx, sy, avoid_monsters: bool = False):
        """全局窗 4 向 BFS(关着的门视为可通行),返回去往"可达且离楼梯最近的格"
        的路径 [(x, y, 是否关门), ...](不含起点)。None = 可达域内没有比脚下
        更接近楼梯的格子(真·被困)。4 向保证引擎寻路必然接受每段(斜穿墙角
        引擎会拒绝);贪心"只挑更近的格"会死在凹形迷宫里,BFS 允许先绕远。

        火焰/酸池等 hazard 与 explosive-softwall 永远视为墙；``door``
        同时包含普通门和 blocking barrels，不能把爆炸桶误当安全软墙。

        avoid_monsters=True 时把怪物占位格视为墙(v14 修复:引擎寻路拒绝穿怪,
        规划器若怪物盲,遇到闲置怪堵走廊会陷入"重规划出同一条路"的失速死循环
        ——9024 号种子的 1 血骷髅当场抓获;调用方应在返回 None 时退回
        avoid_monsters=False 保底,行为最坏退化为旧版失速交还)。"""
        px, py = raw["player_x"], raw["player_y"]
        r = self._DESCEND_RADIUS
        side = 2 * r + 1
        lm = bridge.local_map(radius=r)
        walk, door = lm["walkable"], lm["door"]
        hazard = lm.get("hazard", [0] * (side * side))
        explosive = lm.get(
            "explosive_softwall", [0] * (side * side))
        mon = lm["monster"] if avoid_monsters else None

        def idx(tx, ty):
            return (ty - py + r) * side + (tx - px + r)

        start = (px, py)
        prev = {start: None}
        depth = {start: 0}
        best = (max(abs(sx - px), abs(sy - py)), 0, start)
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + ddx, cy + ddy
                if abs(nx - px) > r or abs(ny - py) > r or (nx, ny) in prev:
                    continue
                i = idx(nx, ny)
                if hazard[i] or explosive[i]:
                    continue
                if not walk[i] and not door[i]:
                    continue
                if mon is not None and mon[i]:
                    continue  # 怪物占位=墙(引擎寻路拒绝穿怪;见 docstring)
                prev[(nx, ny)] = (cx, cy)
                depth[(nx, ny)] = depth[(cx, cy)] + 1
                d_stairs = max(abs(sx - nx), abs(sy - ny))
                if (d_stairs, depth[(nx, ny)]) < best[:2]:
                    best = (d_stairs, depth[(nx, ny)], (nx, ny))
                queue.append((nx, ny))
        if best[2] == start:
            return None
        path = []
        cur = best[2]
        while cur != start:
            path.append((cur[0], cur[1], bool(door[idx(*cur)])))
            cur = prev[cur]
        path.reverse()
        return path

    @staticmethod
    def _plan_controller_path(
        snapshot: _ControllerSnapshot,
        target_x: int,
        target_y: int,
        *,
        avoid_monsters: bool,
        allow_softwalls: bool = True,
    ):
        """Plan only from the observation-bound radius-12 controller snapshot.

        Targets beyond the window are not deleted: BFS advances to the locally
        reachable cell that most reduces Chebyshev distance, then the next
        policy decision receives a re-centered snapshot.  This preserves
        ordinary long-range reachability without consulting an unobserved
        radius-112 map behind the Worker's back.
        """
        px, py = snapshot.player_x, snapshot.player_y
        radius = CONTROLLER_SNAPSHOT_RADIUS
        side = CONTROLLER_SNAPSHOT_SIDE

        def index(x, y):
            return (y - py + radius) * side + (x - px + radius)

        start = (px, py)
        predecessors = {start: None}
        depths = {start: 0}
        best = (
            max(abs(int(target_x) - px), abs(int(target_y) - py)),
            0,
            start,
        )
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                point = (nx, ny)
                if (
                    abs(nx - px) > radius
                    or abs(ny - py) > radius
                    or point in predecessors
                    or point in snapshot.protected_tiles
                ):
                    continue
                i = index(nx, ny)
                if snapshot.hazard[i] or snapshot.explosive_softwall[i]:
                    continue
                if (
                    not snapshot.walkable[i]
                    and not (
                        allow_softwalls and snapshot.softwall[i])
                ):
                    continue
                if avoid_monsters and snapshot.physical_monster[i]:
                    continue
                predecessors[point] = (cx, cy)
                depths[point] = depths[(cx, cy)] + 1
                distance = max(
                    abs(int(target_x) - nx), abs(int(target_y) - ny))
                if (distance, depths[point]) < best[:2]:
                    best = (distance, depths[point], point)
                queue.append(point)
        if best[2] == start:
            return None
        path = []
        current = best[2]
        while current != start:
            path.append((
                current[0],
                current[1],
                bool(snapshot.softwall[index(*current)]),
            ))
            current = predecessors[current]
        path.reverse()
        return path

    @staticmethod
    def _descend_blocker_health(monster):
        if "hp_fixed_hi" in monster and "hp_fixed_lo" in monster:
            return (int(monster["hp_fixed_hi"]) << 16) + int(monster["hp_fixed_lo"])
        return 64 * int(monster.get("hp", 0))

    @classmethod
    def _descend_observed_blocker(cls, raw, target_x, target_y):
        """Bind only a visible live hostile occupying this exact next edge.

        A moving monster can reserve the edge with future while its tile still
        lies elsewhere. Adjacency uses future on both sides, matching native
        radius-one attack validation; this never permits a chase.
        """
        px = int(raw.get("future_x", raw["player_x"]))
        py = int(raw.get("future_y", raw["player_y"]))
        edge = (int(target_x), int(target_y))
        for monster in raw.get("monsters", ()):
            if (not bool(monster.get("visible", False))
                    or bool(monster.get("is_invalid", False))
                    or bool(monster.get("is_player_minion", False))
                    or int(monster.get("type", -1)) == 109
                    or cls._descend_blocker_health(monster) <= 0):
                continue
            tile = (int(monster["x"]), int(monster["y"]))
            future = (int(monster.get("future_x", tile[0])),
                      int(monster.get("future_y", tile[1])))
            if edge in (tile, future) and max(abs(future[0] - px), abs(future[1] - py)) <= 1:
                return monster
        return None

    def _macro_descend_blocker(self, raw, blocker, *, beats_before,
                              macro_deadline, start_scene, execution_audit):
        """One native adjacent attack command, then return for real replanning.

        The command may span multiple ordinary attack frames within the core
        budget; this does not promise exactly one swing or one damage event.

        There is no cross-action recovery state. The attack core shares this
        a11 call's remaining budget; normal idle settle is charged separately
        and remains bounded by the live episode deadline. Receipts distinguish
        native acceptance, observed damage and actual committed displacement.
        """
        start_clock = self._resource_actual_microsteps
        initial_position = (int(raw["player_x"]), int(raw["player_y"]))
        key = self._monster_generation_key(blocker)
        initial_hp = self._descend_blocker_health(blocker)
        receipt = {"source": "dive-blocker:adjacent-v1", "monster_generation": key,
                   "before_microstep": start_clock, "macro_deadline": int(macro_deadline),
                   "accepted": False, "monster_hp_fixed_before": initial_hp,
                   "monster_hp_fixed_after": initial_hp, "damage_observed": False,
                   "target_alive_observed": True, "position_changed": False,
                   "after_microstep": start_clock, "micro_steps": 0,
                   "reason": "deadline"}
        self._dive_blocker_audit = receipt

        def time_available():
            return self._resource_actual_microsteps < min(macro_deadline, self.max_steps)

        def canonical_ready(observed):
            return bool(observed.get("resource_state", {}).get("readiness", {}).get("ready", False))

        if not time_available() or not canonical_ready(raw):
            receipt["reason"] = "deadline" if not time_available() else "readiness_lost"
            return self._finish_macro(raw, beats_before, start_scene)
        result = bridge.act_controller_attack_monster(
            int(blocker["id"]), int(raw["player_x"]), int(raw["player_y"]), 1)
        if execution_audit is not None:
            accepted = self._record_native_execution(execution_audit, result, "action11 blocker attack")
        else:
            accepted = int(result) == 1
        receipt["accepted"] = bool(accepted)
        if not accepted:
            bridge.act_wait()
        receipt["reason"] = "attack_rejected" if not accepted else "macro_cap"
        beats = int(beats_before)
        while time_available():
            raw = self._step_native()
            beats += 1
            if _scene_identity(raw) == start_scene:
                self._record_visit((raw["player_x"], raw["player_y"]))
            if raw.get("dead") or raw.get("game_over") or raw.get("victory"):
                receipt["reason"] = "terminal"
                break
            if _scene_identity(raw) != start_scene:
                receipt["reason"] = "scene_changed"
                break
            target = next((m for m in raw.get("monsters", ())
                           if self._monster_generation_key(m) == key), None)
            if target is not None:
                current_hp = self._descend_blocker_health(target)
                receipt["monster_hp_fixed_after"] = current_hp
                receipt["damage_observed"] = receipt["damage_observed"] or current_hp < initial_hp
                receipt["target_alive_observed"] = current_hp > 0
            else:
                # Disappearance / generation replacement is not proof of a kill.
                receipt["monster_hp_fixed_after"] = None
                receipt["target_alive_observed"] = None
            if not canonical_ready(raw):
                receipt["reason"] = "readiness_lost"
                break
            if not accepted:
                break
            if target is None or self._descend_blocker_health(target) <= 0:
                receipt["reason"] = "target_changed" if target is None else "target_dead_observed"
                break
            if (not bool(target.get("visible", False))
                    or self._engage_distance(raw, target) > 1):
                receipt["reason"] = "target_left_adjacency"
                break
            if self._decision_idle(raw):
                receipt["reason"] = "swing_complete"
                break
        receipt["after_microstep"] = self._resource_actual_microsteps
        receipt["micro_steps"] = self._resource_actual_microsteps - start_clock
        receipt["position_changed"] = (
            int(raw["player_x"]), int(raw["player_y"])) != initial_position
        return self._finish_macro(raw, beats, start_scene)

    def _macro_descend(
        self,
        max_beats: int = 12,
        *,
        execution_audit: dict | None = None,
    ):
        """下楼宏:全局 BFS 规划一次,逐个 4 向相邻安全步走向下行楼梯。

        门只在已经相邻时操作；walk/operate 都使用原生受约束控制器入口，
        禁止 CMD_WALKXY/CMD_OPOBJXY 在执行阶段另算最短路穿过 hazard。
        地牢房间靠门连通,而关着的门在 walkable 通道里长得和墙一样，
        因而路径仍必须显式携带门状态。

        发现猎物不打断(这是主动撤离键);换层/阵亡/持续失速提前结束;
        12 拍耗尽自然归还控制权,下次按键重新规划。全程无随机数,确定性。
        """
        if self._raw.get("progression_targets"):
            return self._macro_progression(
                max_beats=max_beats,
                execution_audit=execution_audit,
            )
        raw = self._raw
        transition = (bridge.WM_DIABRTNLVL if raw.get("is_set_level")
                      else bridge.WM_DIABNEXTLVL)
        stairs = [t for t in raw.get("triggers", [])
                  if t["msg"] == transition]
        if not stairs:
            return self._wait_step()
        px, py = raw["player_x"], raw["player_y"]
        st = min(stairs, key=lambda t: max(abs(t["x"] - px), abs(t["y"] - py)))
        sx, sy = st["x"], st["y"]
        start_scene = _scene_identity(raw)

        path = self._plan_descend_path(raw, sx, sy, avoid_monsters=True)
        if (getattr(self, "_descend_fallback_promotion", False)
                and path is not None):
            # E-fix 修 A:避怪 BFS 的口袋可达域会造出"原路返回更近"的
            # partial path,与宽容规划形成确定性极限环(R8 seed 2122004:
            # 6 窗 24 微步周期钉死原地)。与 _macro_progression 的既有
            # 立法同款:避怪路径踏不上楼梯格时咨询宽容规划,终点严格
            # 更近才提升;能踏上楼梯(remaining==0)时第二次 BFS 不发生,
            # 行为逐位等旧。
            def _remaining(candidate):
                if not candidate:
                    return max(abs(sx - int(px)), abs(sy - int(py)))
                tail = candidate[-1]
                return max(abs(sx - int(tail[0])), abs(sy - int(tail[1])))
            remaining = _remaining(path)
            if remaining > 0:
                fallback = self._plan_descend_path(raw, sx, sy)
                if fallback is not None and _remaining(fallback) < remaining:
                    path = fallback
        if path is None:
            path = self._plan_descend_path(raw, sx, sy)  # 怪物封死唯一通路:退回旧行为
        if path is None:
            return self._wait_step()  # 真被困:原地一拍,交还控制权

        pi = 0            # 路径消费指针
        target = None     # (kind, x, y, path_index)
        center_x, center_y = int(px), int(py)
        stall = 0
        beats = 0
        last_pos = (px, py)
        blocker_recovery = (
            getattr(self, "dive_blocker_recovery", "off") == "adjacent-v1"
            and int(raw.get("dungeon_level", 0)) > 0
            and not bool(raw.get("is_set_level", False))
        )
        macro_deadline = (self._resource_actual_microsteps + max_beats
                          if blocker_recovery else None)
        for beats in range(1, max_beats + 1):
            if blocker_recovery:
                next_edge = (target[1:3] if target is not None else
                             path[pi][:2] if pi < len(path) else None)
                blocker = (self._descend_observed_blocker(raw, *next_edge)
                           if next_edge is not None else None)
                if blocker is not None:
                    return self._macro_descend_blocker(
                        raw, blocker, beats_before=beats - 1,
                        macro_deadline=macro_deadline, start_scene=start_scene,
                        execution_audit=execution_audit)
            if target is None:
                if pi >= len(path):
                    break  # 路径走完(最近可达格≠楼梯时会发生),交还控制权
                j = pi
                target = (
                    ("open", path[j][0], path[j][1], j)
                    if path[j][2]
                    else ("walk", path[j][0], path[j][1], j)
                )
                if target[0] == "open":
                    accepted = bridge.act_controller_operate(
                        target[1],
                        target[2],
                        center_x,
                        center_y,
                        self._DESCEND_RADIUS,
                    )
                else:
                    accepted = bridge.act_explore_walk(
                        target[1],
                        target[2],
                        (),
                        center_x,
                        center_y,
                        self._DESCEND_RADIUS,
                    )
                if execution_audit is not None:
                    accepted = self._record_native_execution(
                        execution_audit, accepted,
                        f"action11 descend {target[0]}")
                else:
                    accepted = int(accepted) == 1
                if not accepted:
                    break
            raw = self._step_native()
            pos = (raw["player_x"], raw["player_y"])
            if _scene_identity(raw) == start_scene:
                self._record_visit(pos)
            if raw["dead"] or _scene_identity(raw) != start_scene:
                break  # 换层成功(或阵亡);足迹由 step() 统一按层重置
            if self._reflex_eligible(raw):
                break
            if pos == (sx, sy):
                continue  # 已站上楼梯格,等触发换层——站桩不算失速
            if target[0] == "open":
                # 开门型目标:门格真的变可走才算完成(贴脸≠已开,动画要几拍)
                if bridge.probe_tile(target[1], target[2])["walkable"]:
                    path[target[3]] = (target[1], target[2], False)
                    pi = target[3]  # 从门所在格继续消费路径
                    target = None
                    stall = 0
                    last_pos = pos
                    continue
            elif pos == (target[1], target[2]):
                # Every native walk request is one exact controller edge.
                # The old multi-hop macro accepted "within one tile", which
                # could advance pi before the committed animation had landed.
                pi = target[3] + 1  # 到达路点,继续下一段
                target = None
                stall = 0
                last_pos = pos
                continue
            if pos == last_pos:
                stall += 1
                if stall == 3 and target is not None:
                    # 命令可能被打断(被怪撞开路径等):原地重发一次
                    if target[0] == "open":
                        accepted = bridge.act_controller_operate(
                            target[1],
                            target[2],
                            center_x,
                            center_y,
                            self._DESCEND_RADIUS,
                        )
                    else:
                        accepted = bridge.act_explore_walk(
                            target[1],
                            target[2],
                            (),
                            center_x,
                            center_y,
                            self._DESCEND_RADIUS,
                        )
                    if execution_audit is not None:
                        self._record_native_execution(
                            execution_audit, accepted,
                            f"action11 descend retry {target[0]}")
                if stall >= 6:
                    break  # 重发后仍无进展 → 交还控制权,下次按键重新规划
            else:
                stall = 0
            last_pos = pos
        return self._finish_macro(raw, beats, start_scene)

    def _macro_pickup(
        self,
        kind: str = "heal",
        max_beats: int = 12,
        *,
        controller_snapshot: _ControllerSnapshot | None = None,
        action14_audit: dict | None = None,
        execution_audit: dict | None = None,
    ):
        """捡取宏(v13 药 / v14 装备):只消费观测绑定的目标与 radius-12 地图。

        沿固定快照路径开门走向该目标物，并在精确站上目标格后才提交。
        wrapper 与两个原生定点入口都要求玩家 future 精确同格，把整条
        移动轨迹锁在固定快照逐步验证的路径上；公开便利入口同样不能
        选择邻近物品绕过这项证明。

        原生提交再次核验 active id、坐标、seed、create info 与 base id。
        装备还会重算 PlanGearUpgrade，复制整套身体槽后原子替换并以
        CalcPlrInv 复验，失败完整回滚；不走 AutoEquip/背包回退。成功
        的 0/1 原生回执会在提交后立刻同步 observe，记录严格为正的整套
        战力 delta，后续受击/耐久损失不能篡改这条因果凭证。目标
        消失或失效即结束，药另有腰带数上涨的快速判据。阵亡/换层/路径
        耗尽/持续失速提前结束；12 拍耗尽自然归还控制权，下次按键重新
        规划。全程无随机数，确定性。
        """
        raw = self._raw
        flag = "heal" if kind == "heal" else "gear"
        snapshot = (
            controller_snapshot
            if controller_snapshot is not None
            else self._capture_controller_snapshot(raw)
        )
        target_point = (
            snapshot.heal_target if kind == "heal"
            else snapshot.gear_target
        )
        if target_point is None or (
                kind == "heal" and self._belt_free_slots(raw) <= 0):
            return self._wait_step()

        if kind == "gear":
            utility = gear_combat_utility_value(
                raw, "action14_macro_start")
            if action14_audit is None:
                action14_audit = {
                    "accepted": False,
                    "commit_attempts": 0,
                    "utility_before": utility,
                    "utility_after": utility,
                    "utility_delta": 0,
                }
            expected_audit_keys = {
                "accepted", "commit_attempts", "utility_before",
                "utility_after", "utility_delta",
            }
            if (
                set(action14_audit) != expected_audit_keys
                or action14_audit["accepted"] is not False
                or int(action14_audit["commit_attempts"]) != 0
                or int(action14_audit["utility_delta"]) != 0
            ):
                raise RuntimeError(
                    "action14_audit 初始状态损坏")
        elif action14_audit is not None:
            raise RuntimeError(
                "action14_audit 只能用于装备拾取宏")

        px, py = int(raw["player_x"]), int(raw["player_y"])
        hx, hy = target_point.x, target_point.y

        def act():
            if kind == "heal":
                accepted = bridge.act_pickup_at(
                    target_point.active_id,
                    hx,
                    hy,
                    target_point.seed_hi,
                    target_point.seed_lo,
                    target_point.create_info,
                    target_point.base_id,
                )
            else:
                utility_before = gear_combat_utility_value(
                    raw, "action14_before_native_commit")
                accepted_raw = bridge.act_pickup_gear_at(
                    target_point.active_id,
                    hx,
                    hy,
                    target_point.seed_hi,
                    target_point.seed_lo,
                    target_point.create_info,
                    target_point.base_id,
                )
                if not isinstance(
                        accepted_raw, (bool, np.bool_, int, np.integer)):
                    raise RuntimeError(
                        "action14 原生回执必须是整数 0/1，收到 "
                        f"{accepted_raw!r}")
                accepted = int(accepted_raw)
                if accepted not in (0, 1):
                    raise RuntimeError(
                        "action14 原生回执必须是 0/1，收到 "
                        f"{accepted}")
                action14_audit["commit_attempts"] += 1
                if accepted:
                    if action14_audit["accepted"]:
                        raise RuntimeError(
                            "action14 同一策略动作出现重复成功提交")
                    committed = bridge.observe()
                    if getattr(self, "resource_preserve_equipment_readiness", False):
                        self._validate_native_resource_flags(committed)
                    utility_after = gear_combat_utility_value(
                        committed, "action14_after_native_commit")
                    utility_delta = utility_after - utility_before
                    if utility_delta <= 0:
                        raise RuntimeError(
                            "action14 原生接受后整套战力未严格增长:"
                            f"{utility_before}->{utility_after}")
                    action14_audit.update({
                        "accepted": True,
                        "utility_before": utility_before,
                        "utility_after": utility_after,
                        "utility_delta": utility_delta,
                    })
            if execution_audit is not None:
                return int(self._record_native_execution(
                    execution_audit, accepted,
                    f"action{13 if kind == 'heal' else 14} pickup"))
            return int(accepted)

        start_belt = int(raw["belt_heals"])
        start_scene = _scene_identity(raw)
        pickup_radius = 0
        near0 = max(abs(hx - px), abs(hy - py)) <= pickup_radius
        path = self._plan_controller_path(
            snapshot, hx, hy, avoid_monsters=True)
        if path is None:
            path = self._plan_controller_path(
                snapshot, hx, hy, avoid_monsters=False)
        center_index = CONTROLLER_SNAPSHOT_CELLS // 2
        same_tile_safe = (
            (px, py) == (hx, hy)
            and not snapshot.hazard[center_index]
            and not snapshot.explosive_softwall[center_index]
        )
        item_path_complete = (
            same_tile_safe
            or (
                path is not None
                and bool(path)
                and (int(path[-1][0]), int(path[-1][1])) == (hx, hy)
            )
        )
        if (
            path is None and not near0
            or not item_path_complete
        ):
            return self._wait_step()

        protected = sorted(snapshot.protected_tiles)
        pi = 0
        target: tuple[str, int, int] | None = None
        stall = 0
        beats = 0
        last_pos = (px, py)
        for beats in range(1, max_beats + 1):
            if target is None and self._decision_idle(raw):
                cur = (int(raw["player_x"]), int(raw["player_y"]))
                near = (
                    max(abs(hx - cur[0]), abs(hy - cur[1]))
                    <= pickup_radius
                )
                door_pending = bool(path) and any(
                    point[2] for point in path[pi:])
                if near and not door_pending:
                    target = ("pick", hx, hy)
                    act()
                elif path is None or pi >= len(path):
                    break
                else:
                    nx, ny, is_softwall = path[pi]
                    if abs(nx - cur[0]) + abs(ny - cur[1]) != 1:
                        raise RuntimeError(
                            "controller pickup 路径不是 4 向相邻步")
                    target = (
                        ("open", nx, ny)
                        if is_softwall else ("walk", nx, ny)
                    )
                    if is_softwall:
                        accepted = bridge.act_controller_operate(
                            nx,
                            ny,
                            snapshot.player_x,
                            snapshot.player_y,
                            CONTROLLER_SNAPSHOT_RADIUS,
                        )
                        if execution_audit is not None:
                            accepted = self._record_native_execution(
                                execution_audit, accepted,
                                f"action{13 if kind == 'heal' else 14} open")
                        else:
                            accepted = int(accepted) == 1
                        if not accepted:
                            break
                    else:
                        accepted = bridge.act_explore_walk(
                            nx,
                            ny,
                            protected,
                            snapshot.player_x,
                            snapshot.player_y,
                            CONTROLLER_SNAPSHOT_RADIUS,
                        )
                        if execution_audit is not None:
                            accepted = self._record_native_execution(
                                execution_audit, accepted,
                                f"action{13 if kind == 'heal' else 14} walk")
                        else:
                            accepted = int(accepted) == 1
                        if not accepted:
                            break

            raw = self._step_native()
            pos = (int(raw["player_x"]), int(raw["player_y"]))
            if _scene_identity(raw) == start_scene:
                self._record_visit(pos)
            if raw["dead"] or _scene_identity(raw) != start_scene:
                break
            if self._reflex_eligible(raw):
                break
            if kind == "heal" and int(raw["belt_heals"]) > start_belt:
                break

            tracked_items = [
                it for it in raw.get("floor_items", ())
                if (
                    int(it.get("active_id", -1))
                    == target_point.active_id
                    and int(it.get("seed_hi", -1))
                    == target_point.seed_hi
                    and int(it.get("seed_lo", -1))
                    == target_point.seed_lo
                    and int(it.get("create_info", -1))
                    == target_point.create_info
                    and int(it.get("base_id", -1))
                    == target_point.base_id
                )
            ]
            if not tracked_items:
                break
            if not any(bool(it.get(flag)) for it in tracked_items):
                break
            if not any(it.get("visible", True) for it in tracked_items):
                break

            if target is not None and target[0] == "pick":
                if self._decision_idle(raw):
                    target = None
            elif target is not None and target[0] == "open":
                if bridge.probe_tile(target[1], target[2])["walkable"]:
                    path[pi] = (target[1], target[2], False)
                    target = None
                    stall = 0
                    last_pos = pos
                    continue
            elif (
                target is not None
                and target[0] == "walk"
                and pos == (target[1], target[2])
            ):
                pi += 1
                target = None
                stall = 0
                last_pos = pos
                continue

            interaction_busy = (
                not self._decision_idle(raw)
                or self._movement_engine_busy(raw)
            )
            if pos != last_pos or interaction_busy:
                stall = 0
            else:
                stall += 1
                if (
                    stall == 3
                    and target is not None
                    and self._decision_idle(raw)
                ):
                    if target[0] == "open":
                        accepted = bridge.act_controller_operate(
                            target[1],
                            target[2],
                            snapshot.player_x,
                            snapshot.player_y,
                            CONTROLLER_SNAPSHOT_RADIUS,
                        )
                        if execution_audit is not None:
                            accepted = self._record_native_execution(
                                execution_audit, accepted,
                                f"action{13 if kind == 'heal' else 14} reopen")
                        else:
                            accepted = int(accepted) == 1
                        if not accepted:
                            break
                    elif target[0] == "pick":
                        act()
                    else:
                        accepted = bridge.act_explore_walk(
                            target[1],
                            target[2],
                            protected,
                            snapshot.player_x,
                            snapshot.player_y,
                            CONTROLLER_SNAPSHOT_RADIUS,
                        )
                        if execution_audit is not None:
                            self._record_native_execution(
                                execution_audit, accepted,
                                f"action{13 if kind == 'heal' else 14} rewalk")
                if stall >= 6:
                    break
            last_pos = pos
        return self._finish_macro(raw, beats, start_scene)

    @classmethod
    def _nearest_monster(cls, obs):
        px, py = obs["player_x"], obs["player_y"]
        best, best_d = None, None
        for m in cls._policy_monsters(obs):
            d = abs(m["x"] - px) + abs(m["y"] - py)
            if best_d is None or d < best_d:
                best, best_d = m, d
        return best

    @classmethod
    def _nearest_dist(cls, raw):
        px, py = raw["player_x"], raw["player_y"]
        dists = [
            max(abs(m["x"] - px), abs(m["y"] - py))
            for m in cls._policy_monsters(raw)
        ]
        return min(dists) if dists else None

    @classmethod
    def _player_approach_delta(
        cls,
        prev,
        cur,
        *,
        target_generation_key: tuple[int, int, int] | None = None,
    ) -> int | None:
        """Player-only distance change to one fixed surviving generation.

        Recomputing two independent nearest-monster minima lets monster motion
        and target switching masquerade as player progress.  By default select
        the nearest generation at the start; action 9 instead supplies the
        canonical generation selected by its installed controller snapshot.
        Hold that generation's current endpoint fixed and compare only the
        player's two positions.  A killed/despawned/reused-slot target receives
        no approach shaping; combat reward already credits a real kill.
        """
        previous_monsters = cls._policy_monsters(prev)
        if not previous_monsters:
            return None
        px0, py0 = int(prev["player_x"]), int(prev["player_y"])
        if target_generation_key is None:
            target = min(
                previous_monsters,
                key=lambda monster: (
                    max(
                        abs(int(monster["x"]) - px0),
                        abs(int(monster["y"]) - py0),
                    ),
                    cls._monster_generation_key(monster),
                ),
            )
        else:
            target = next(
                (
                    monster
                    for monster in previous_monsters
                    if cls._monster_generation_key(monster)
                    == target_generation_key
                ),
                None,
            )
            if target is None:
                return None
        target_key = cls._monster_generation_key(target)
        current = next(
            (
                monster
                for monster in cls._policy_monsters(cur)
                if cls._monster_generation_key(monster) == target_key
            ),
            None,
        )
        if current is None:
            return None
        tx, ty = int(current["x"]), int(current["y"])
        px1, py1 = int(cur["player_x"]), int(cur["player_y"])
        return (
            max(abs(tx - px0), abs(ty - py0))
            - max(abs(tx - px1), abs(ty - py1))
        )

    @classmethod
    def _disappeared_monster_generations(cls, prev, cur) -> int:
        """Count same-scene monster lifetimes that ended during a transition."""
        current = {
            cls._monster_generation_key(monster)
            for monster in cur.get("monsters", ())
        }
        return sum(
            cls._monster_generation_key(monster) not in current
            for monster in prev.get("monsters", ())
        )

    @staticmethod
    def _native_monster_kill_delta(prev, cur) -> int | None:
        """Validated delta of the process/hero-monotonic native kill ledger."""
        before_present = "monster_kill_total" in prev
        after_present = "monster_kill_total" in cur
        if not before_present and not after_present:
            return None
        if before_present != after_present:
            raise RuntimeError(
                "monster_kill_total 只出现在 transition 一端")

        def value(raw, label):
            candidate = raw["monster_kill_total"]
            if isinstance(candidate, bool):
                raise RuntimeError(
                    f"{label}.monster_kill_total 不得为 bool")
            try:
                integer = int(candidate)
                numeric = float(candidate)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"{label}.monster_kill_total 必须是非负整数") from exc
            if (
                not math.isfinite(numeric)
                or numeric != float(integer)
                or integer < 0
            ):
                raise RuntimeError(
                    f"{label}.monster_kill_total 必须是非负整数")
            return integer

        before = value(prev, "prev")
        after = value(cur, "cur")
        if after < before:
            raise RuntimeError(
                "原生 monster_kill_total 在 episode 内回退:"
                f"{before}->{after}")
        return after - before

    def _reset_combat_ledger(self, raw) -> None:
        """以当前场景血线为零点；reset/换图时调用，绝不跨 lifetime 域。"""
        self._combat_hp_floor = {
            self._monster_generation_key(m): (
                max(0, int(m["hp"])),
                max(1, int(m["max_hp"])),
            )
            for m in raw.get("monsters", ())
        }

    def _combat_reward(self, prev, cur) -> float:
        """以怪物本场景新最低 HP 的势函数差计伤害，已付区间不重复给钱。

        ledger 只保留当前仍存活 lifetime；新生成的怪从首次观测血线建账。
        generation 消失时只把其尚未支付的伤害势结至 0；收头单位来自原生
        单调 monster_kill_total，而非由 active-list 消失推断（缺少该字段
        的旧合成 fixture 才回退到 generation 消失计数）。slot 被同拍复用
        时旧、新 generation 分别结账，绝不把新怪 HP 当作旧怪回血。
        """
        if not hasattr(self, "_combat_hp_floor"):
            self._combat_hp_floor = {}
        if not self._combat_hp_floor and prev.get("monsters"):
            self._reset_combat_ledger(prev)

        prev_by_generation = {
            self._monster_generation_key(m): m
            for m in prev.get("monsters", ())
        }
        cur_by_generation = {
            self._monster_generation_key(m): m
            for m in cur.get("monsters", ())
        }
        next_floor: dict[tuple[int, int, int], tuple[int, int]] = {}
        r = 0.0

        for generation_key, m in prev_by_generation.items():
            ledger_low, ledger_max = self._combat_hp_floor.get(
                generation_key,
                (max(0, int(m["hp"])), max(1, int(m["max_hp"]))),
            )
            # 若调用方在两个 env.step 之间直接推进了 bridge，prev 已经低于
            # 账本时只下调零点、不追付环境之外发生的伤害。
            paid_floor = min(ledger_low, max(0, int(m["hp"])))
            denominator = max(ledger_max, int(m["max_hp"]), 1)
            current = cur_by_generation.get(generation_key)
            hp_after = max(0, int(current["hp"])) if current is not None else 0
            new_floor = min(paid_floor, hp_after)
            newly_credited = paid_floor - new_floor
            if newly_credited > 0:
                # Pay the difference of one bounded health potential.  The
                # old endpoint multiplier overpaid a 100→0 one-shot relative
                # to the exact same damage split across several transitions,
                # making reward depend on ticks_per_step and attack cadence.
                q_before = paid_floor / denominator
                q_after = new_floor / denominator
                r += (
                    0.75 * (q_before - q_after)
                    - 0.125 * (
                        q_before * q_before - q_after * q_after
                    )
                )
            if current is None:
                continue
            next_floor[generation_key] = (
                new_floor,
                max(denominator, int(current["max_hp"]), 1),
            )

        # 本拍新生成/首次进入 active list 的怪不能凭首次观测的残血领奖。
        for generation_key, m in cur_by_generation.items():
            if generation_key not in prev_by_generation:
                next_floor[generation_key] = (
                    max(0, int(m["hp"])),
                    max(1, int(m["max_hp"])),
                )

        self._combat_hp_floor = next_floor
        native_kills = self._native_monster_kill_delta(prev, cur)
        r += (
            native_kills
            if native_kills is not None
            else self._disappeared_monster_generations(prev, cur)
        )
        return r

    @staticmethod
    def _progression_effect_signature(raw) -> tuple:
        """Stable visible progression identity for causal action receipts."""
        return tuple(sorted(
            (
                str(target.get("kind", "")),
                str(target.get("action", "")),
                int(target.get("x", 0)),
                int(target.get("y", 0)),
                int(target.get("goal_x", 0)),
                int(target.get("goal_y", 0)),
                bool(target.get("exact", False)),
            )
            for target in raw.get("progression_targets", ())
        ))

    def _action_effect_reasons(
        self,
        prev,
        cur,
        *,
        requested_action: int,
        engage_target_generation_key: tuple[int, int, int] | None,
        native_kills: int,
        exploration_before: int,
        softwalls_before: int,
        drink_audit: dict | None,
        action14_audit: dict | None,
    ) -> tuple[str, ...]:
        """Return only effects causally attributable to the requested action.

        Enemy motion and player damage are deliberately absent.  They are
        world consequences, not proof that an explore/attack/pickup request
        executed, and previously let stalled requests masquerade as progress.
        """
        action = int(requested_action)
        if action == 0:
            return ()

        reasons: list[str] = []
        scene_changed = _scene_identity(cur) != _scene_identity(prev)
        position_changed = (
            (
                int(cur.get("player_x", 0)),
                int(cur.get("player_y", 0)),
                int(cur.get("future_x", cur.get("player_x", 0))),
                int(cur.get("future_y", cur.get("player_y", 0))),
            )
            != (
                int(prev.get("player_x", 0)),
                int(prev.get("player_y", 0)),
                int(prev.get("future_x", prev.get("player_x", 0))),
                int(prev.get("future_y", prev.get("player_y", 0))),
            )
        )
        if scene_changed:
            reasons.append("scene")

        if 1 <= action <= 8:
            if position_changed:
                reasons.append("move")
        elif action == 9:
            if position_changed:
                reasons.append("move")
            if int(native_kills) > 0:
                reasons.append("kill")
            target_key = engage_target_generation_key
            if target_key is not None:
                previous = next(
                    (
                        monster
                        for monster in prev.get("monsters", ())
                        if self._monster_generation_key(monster) == target_key
                    ),
                    None,
                )
                current = next(
                    (
                        monster
                        for monster in cur.get("monsters", ())
                        if self._monster_generation_key(monster) == target_key
                    ),
                    None,
                )
                if (
                    previous is not None
                    and current is not None
                    and int(current.get("hp", 0))
                    < int(previous.get("hp", 0))
                ):
                    reasons.append("target_damage")
                elif (
                    previous is not None
                    and current is None
                    and int(native_kills) > 0
                ):
                    reasons.append("target_removed")
        elif action == 10:
            if position_changed:
                reasons.append("move")
            if int(getattr(
                    self, "_exploration_progress", 0)) > int(
                        exploration_before):
                reasons.append("exploration")
            if int(getattr(self, "_softwalls_opened", 0)) > int(
                    softwalls_before):
                reasons.append("softwall")
        elif action == 11:
            if position_changed:
                reasons.append("move")
            recovery = getattr(self, "_dive_blocker_audit", None)
            if (getattr(self, "dive_blocker_recovery", "off") == "adjacent-v1"
                    and recovery is not None
                    and recovery.get("accepted") is True
                    and recovery.get("damage_observed") is True):
                reasons.append("blocker_damage")
            if (
                self._progression_effect_signature(cur)
                != self._progression_effect_signature(prev)
            ):
                reasons.append("progression")
        elif action == 12:
            if (
                isinstance(drink_audit, dict)
                and drink_audit.get("consumed") is True
            ):
                reasons.append("drink")
        elif action == 13:
            if (
                self._belt_free_slots(cur) < self._belt_free_slots(prev)
                or int(cur.get("belt_heals", 0))
                > int(prev.get("belt_heals", 0))
            ):
                reasons.append("heal_pickup")
        elif action == 14:
            if (
                isinstance(action14_audit, dict)
                and action14_audit.get("accepted") is True
                and int(action14_audit.get("utility_delta", 0)) > 0
            ):
                reasons.append("gear_commit")
        return tuple(dict.fromkeys(reasons))

    def _reward(
        self,
        prev,
        cur,
        requested_action: int | None = None,
        *,
        engage_target_generation_key: tuple[int, int, int] | None = None,
        action_executed: bool | None = None,
        action14_utility_delta: int | None = None,
    ) -> float:
        cls = type(self)
        econ = self.reward_economy
        xp_term = 0.01 * (cur["xp"] - prev["xp"])
        r = xp_term
        dl = cur["dungeon_level"] - prev["dungeon_level"]
        actual_prev, actual_cur = prev, cur
        if getattr(self, "resource_protocol", "off") != "off":
            before = self._resource_reward_depth_before
            after = self._resource_max_main_depth
            dl = max(0, after - before)
            paid = econ.descend_unit * (sum(range(before, after))
                    if self.descend_ladder else dl)
            self._resource_step_bonus = float(paid)
            # Preserve the remainder of the old accounting using equivalent
            # synthetic monotonic endpoints; raw game state is never changed.
            prev = dict(prev, dungeon_level=before)
            cur = dict(cur, dungeon_level=after)
        if self.descend_ladder and dl > 0:
            # v17:深度递进——每个 N→N+1 付 8×N(L1→2 仍是 8,锚定旧章;
            # L2→3 付 16、L3→4 付 24……越深越值钱,给"往下活着"一个未来)
            r += DESCEND_UNIT * sum(range(prev["dungeon_level"], cur["dungeon_level"]))
        else:
            r += DESCEND_UNIT * dl
        if dl > 0 and econ.descend_unit != DESCEND_UNIT:
            # R10 v2 A 条款:下楼单价上调,以对 v1 的精确增量追加
            # (层数为整数,差价乘积在 float64 中无损;v1 路径零触碰)。
            if self.descend_ladder:
                r += (econ.descend_unit - DESCEND_UNIT) * sum(
                    range(prev["dungeon_level"], cur["dungeon_level"]))
            else:
                r += (econ.descend_unit - DESCEND_UNIT) * dl
        if econ.descend_vest_kills > 0:
            # R11 杀怪锁(主席版,2026-08-28):下楼奖金先记"未解锁"托管;
            # 该层击杀满 vest_kills 只解锁(托管清零、钱留账);死于解锁前
            # 全额罚没。gamma=1.0 下与真托管到账数学等价,且不动工资剥薪
            # 恒等式。超时局终不罚(主席原文只判死刑)。
            if dl > 0:
                paid = econ.descend_unit * (
                    sum(range(prev["dungeon_level"], cur["dungeon_level"]))
                    if self.descend_ladder else dl)
                self._econ_unvested += float(paid)
                self._econ_kills_on_floor = 0
            kills_now = int(getattr(self, "_ep_kills", 0))
            kill_delta = kills_now - self._econ_prev_epkills
            self._econ_prev_epkills = kills_now
            if kill_delta > 0 and dl == 0 and self._econ_unvested > 0.0:
                self._econ_kills_on_floor += kill_delta
                if self._econ_kills_on_floor >= econ.descend_vest_kills:
                    self._econ_unvested = 0.0
            if bool(cur["dead"]) and self._econ_unvested > 0.0:
                r -= self._econ_unvested
                self._econ_unvested = 0.0
        prev, cur = actual_prev, actual_cur
        # The native replacement comparator, observation and this shaping
        # consume one shared uint32 ledger.  Unlike the old ΔAC-only reward,
        # weapon damage/to-hit, shields/block, resistances, affixes and usable
        # durability all receive credit.  Strict replacement makes growth
        # monotonic at pickup time; the cap bounds any rare large unique.
        if action14_utility_delta is None:
            gear_term = gear_upgrade_reward_component(prev, cur)
            r += gear_term
        else:
            # Action 14 publishes the comparator result synchronously at the
            # native commit.  Its later settle endpoint may already include
            # durability loss or other combat changes, so endpoint Δutility
            # is not a causal receipt and must neither erase nor double-pay
            # the accepted upgrade.
            gear_term = gear_upgrade_reward_delta_component(
                action14_utility_delta)
            r += gear_term
        if econ.name != "v1" and gear_term > 0.0:
            # R10 v2 E1 条款:装备重定价(scale 降/cap 升)。a14 因果分支
            # 有精确 Δutility 可重算;(prev,cur) 分支按 v1 分量等比换算,
            # 仅 v1 饱和(=cap)时保守低估——预注册已如实登记该近似。
            if action14_utility_delta is not None:
                repriced = min(
                    econ.gear_cap,
                    float(int(action14_utility_delta)) / econ.gear_scale)
            else:
                repriced = min(
                    econ.gear_cap,
                    gear_term * (GEAR_COMBAT_UTILITY_REWARD_SCALE
                                 / econ.gear_scale))
            r += repriced - gear_term
        same_scene = _scene_identity(cur) == _scene_identity(prev)
        combat_term = 0.0
        if same_scene:
            combat_term = self._combat_reward(prev, cur)
            r += combat_term
        else:
            # Damage ledgers are scene-local, but a monster killed earlier in
            # the same macro must not lose its terminal unit merely because
            # the final micro-beat also crossed a level/quest boundary.
            native_kills = self._native_monster_kill_delta(prev, cur)
            if native_kills is not None:
                combat_term = float(native_kills)
                r += native_kills
        # 接近塑形:仅当本拍请求了非等待动作且是"自己走近"才有奖励。
        # ActWait 为避免破坏引擎占位，允许上一拍已提交的单格动画自然收尾；
        # 若只比较前后坐标，这段旧动作位移会被错记到当前 action0，真实
        # a0-only 探针因此偶发 +0.005~+0.02。requested_action 把信用归因
        # 钉回请求动作：等待永不领接近塑形，并且即使旧步收尾仍付 -0.002。
        # XP/击杀/真实伤害等环境事件仍按事实记账，不因动作号而抹掉。
        if same_scene:
            moved = (cur["player_x"], cur["player_y"]) != (prev["player_x"], prev["player_y"])
            if requested_action == 0 or action_executed is False:
                # A rejected/stalled non-zero request is behaviorally the same
                # as an explicit wait.  Charging only action0 let action10 (and
                # stale masked requests) consume the TimeLimit at zero cost.
                r += STALL_ACTION_REWARD
            elif requested_action is not None:
                if not moved:
                    approach_delta = None
                elif requested_action == 9:
                    # action 9 的路径/攻击已经绑定 controller snapshot 的
                    # canonical generation。若这里重新按 tile 选最近怪，
                    # 移动怪或局部不可接敌的 decoy 会把朝真实目标的走位
                    # 反向记成负奖励。无候选时也必须 fail closed，不能
                    # 回退到任意 raw 怪物给一次空 action 伪造进展。
                    approach_delta = (
                        cls._player_approach_delta(
                            prev,
                            cur,
                            target_generation_key=(
                                engage_target_generation_key),
                        )
                        if engage_target_generation_key is not None
                        else None
                    )
                else:
                    approach_delta = cls._player_approach_delta(prev, cur)
                if approach_delta is not None:
                    r += 0.005 * approach_delta
        r += terminal_death_reward_component(
            dead=bool(cur["dead"]),
            dungeon_level=cur["dungeon_level"],
            death_ladder=bool(self.death_ladder),
            economy=econ,
        )
        if econ.name != "v1":
            # R10 v2 B×D 条款:深度乘数与反躺平,只作用于正向 farm 收入
            # (xp+击杀),不触碰下楼/装备/塑形/死亡/胜利各项;记账归经理。
            d_now = max(int(cur["dungeon_level"]), 1)
            if econ.idle_counts_micro_beats:
                # R16 v4:idle 钟按底层微拍计数。step() 在调用 _reward 前已
                # 把本决策全部拍(含 settle)累进 self._steps,取其增量。
                steps_now = int(getattr(self, "_steps", 0))
                idle_tick = max(
                    0, steps_now - int(getattr(
                        self, "_econ_idle_prev_steps", 0)))
                self._econ_idle_prev_steps = steps_now
            else:
                idle_tick = 1
            if int(cur["dungeon_level"]) > self._econ_episode_max_depth:
                self._econ_episode_max_depth = int(cur["dungeon_level"])
                self._econ_steps_on_level = 0
            elif dl == 0:
                self._econ_steps_on_level += idle_tick
            else:
                self._econ_steps_on_level = 0
            mult = 1.0 + econ.kill_depth_beta * (d_now - 1)
            if (econ.idle_threshold_steps > 0
                    and self._econ_steps_on_level
                    > econ.idle_threshold_steps):
                mult *= econ.idle_kill_factor
            farm = xp_term + combat_term
            if farm > 0.0:
                r += farm * (mult - 1.0)
            if econ.idle_reset_on_kill:
                # R16 v4:击杀发生即清零(结算在前:终结长时间空转的那一杀
                # 仍按清零前的钟计价;此后钟重新起算)。
                idle_kills = self._native_monster_kill_delta(prev, cur)
                if idle_kills is not None and idle_kills > 0:
                    self._econ_steps_on_level = 0
        if cur["victory"]:
            r += 10.0
        return float(r)

    @classmethod
    def _legacy_policy_vectorize(cls, obs) -> np.ndarray:
        """Reconstruct the exact protocol-v3 295-wide policy observation.

        Protocol-v4 deliberately hid invisible/unreachable entities from the
        live observation.  Frozen V28/KING/M29 networks were trained before
        that semantic break, so their compatibility view cannot be recovered
        from the already-filtered 295-vector: monster count/nearest/slots, the
        121-cell monster channel, and floor-item slots may all have lost
        information.  Rebuild those fields from the still-lossless native
        ``raw`` record and local map instead of pretending that decoding the
        packed belt scalar alone restores v3.

        Keep this implementation literal and independent of v4 policy helpers.
        Calling ``_policy_monsters`` or ``_policy_floor_items`` here would
        silently reintroduce the distribution shift this boundary exists to
        prevent.
        """
        if "monsters" not in obs or "floor_items" not in obs:
            raise RuntimeError(
                "protocol-v3 compatibility view requires lossless native "
                "monsters and floor_items records")
        if "legacy_belt_heals" not in obs:
            raise RuntimeError(
                "protocol-v3 compatibility view requires native "
                "legacy_belt_heals; refusing a v4 belt fallback")
        missing_legacy_item = [
            index for index, item in enumerate(obs["floor_items"])
            if "legacy_heal" not in item
        ]
        if missing_legacy_item:
            raise RuntimeError(
                "protocol-v3 compatibility view requires native legacy_heal "
                "on every floor item; missing indices "
                f"{missing_legacy_item[:8]}")
        px, py = obs["player_x"], obs["player_y"]
        all_monsters = list(obs["monsters"])
        nearest = None
        if all_monsters:
            nearest = min(
                max(abs(m["x"] - px), abs(m["y"] - py))
                for m in all_monsters
            )
        advance = list(obs.get("progression_targets", ()))
        if advance:
            st = min(advance, key=lambda t: max(
                abs(t["goal_x"] - px), abs(t["goal_y"] - py)))
            sx, sy = st["goal_x"], st["goal_y"]
        else:
            transition = (bridge.WM_DIABRTNLVL if obs.get("is_set_level")
                          else bridge.WM_DIABNEXTLVL)
            stairs = [t for t in obs.get("triggers", ())
                      if t["msg"] == transition]
            st = min(stairs, key=lambda t: max(
                abs(t["x"] - px), abs(t["y"] - py))) if stairs else None
            sx, sy = (st["x"], st["y"]) if st is not None else (px, py)
        if advance or st is not None:
            stair_dx, stair_dy = (sx - px) / 56.0, (sy - py) / 56.0
        else:
            stair_dx = stair_dy = 0.0
        vec = [
            obs["hp"] / max(1, obs["max_hp"]),
            obs["mana"] / max(1, obs["max_mana"]),
            math.log1p(obs["xp"]) / 10.0,
            obs["gold"] / 1000.0,
            obs["char_level"] / 50.0,
            obs["dungeon_level"] / 16.0,
            px / 112.0,
            py / 112.0,
            min(1.0, len(all_monsters) / 50.0),
            min(1.0, nearest / 30.0) if nearest is not None else 1.0,
            stair_dx,
            stair_dy,
        ]
        monsters = sorted(
            all_monsters,
            key=lambda m: abs(m["x"] - px) + abs(m["y"] - py),
        )[:_K_MONSTERS]
        for monster in monsters:
            vec += [
                (monster["x"] - px) / 20.0,
                (monster["y"] - py) / 20.0,
                monster["hp"] / max(1, monster["max_hp"]),
                1.0,
            ]
        vec += [0.0, 0.0, 0.0, 0.0] * (
            _K_MONSTERS - len(monsters))
        local_map = bridge.local_map(radius=_MAP_RADIUS)
        vec += [float(value) for value in local_map["walkable"]]
        vec += [float(value) for value in local_map["monster"]]
        heals = [
            item for item in obs["floor_items"]
            if bool(item["legacy_heal"])
        ]
        legacy_belt_heals = obs["legacy_belt_heals"]
        if heals:
            heal = min(heals, key=lambda item: max(
                abs(item["x"] - px), abs(item["y"] - py)))
            vec += [
                legacy_belt_heals / 8.0,
                max(-1.0, min(1.0, (heal["x"] - px) / 20.0)),
                max(-1.0, min(1.0, (heal["y"] - py) / 20.0)),
                1.0,
            ]
        else:
            vec += [legacy_belt_heals / 8.0, 0.0, 0.0, 0.0]
        gears = [
            item for item in obs["floor_items"]
            if item.get("gear")
        ]
        armor_class = max(
            0.0, min(1.0, obs.get("armor_class", 0) / 50.0))
        if gears:
            gear = min(gears, key=lambda item: max(
                abs(item["x"] - px), abs(item["y"] - py)))
            vec += [
                armor_class,
                max(-1.0, min(1.0, (gear["x"] - px) / 20.0)),
                max(-1.0, min(1.0, (gear["y"] - py) / 20.0)),
                1.0,
            ]
        else:
            vec += [armor_class, 0.0, 0.0, 0.0]
        vec += [
            min(
                2.0,
                obs["char_level"] / max(1, obs["dungeon_level"]),
            ) / 2.0
        ]
        result = np.asarray(vec, dtype=np.float32)
        if result.shape != (295,):
            raise RuntimeError(
                "protocol-v3 policy observation 形状漂移:"
                f"{result.shape} != (295,)")
        return result

    @classmethod
    def _vectorize(cls, obs) -> np.ndarray:
        px, py = obs["player_x"], obs["player_y"]
        policy_monsters = cls._policy_monsters(obs)
        nearest = cls._nearest_dist(obs)
        advance = list(obs.get("progression_targets", ()))
        if advance:
            st = min(advance, key=lambda t: max(
                abs(t["goal_x"] - px), abs(t["goal_y"] - py)))
            sx, sy = st["goal_x"], st["goal_y"]
        else:
            transition = (bridge.WM_DIABRTNLVL if obs.get("is_set_level")
                          else bridge.WM_DIABNEXTLVL)
            stairs = [t for t in obs.get("triggers", [])
                      if t["msg"] == transition]
            st = min(stairs, key=lambda t: max(
                abs(t["x"] - px), abs(t["y"] - py))) if stairs else None
            sx, sy = (st["x"], st["y"]) if st is not None else (px, py)
        if advance or st is not None:
            stair_dx, stair_dy = (sx - px) / 56.0, (sy - py) / 56.0
        else:
            stair_dx = stair_dy = 0.0
        vec = [
            obs["hp"] / max(1, obs["max_hp"]),
            obs["mana"] / max(1, obs["max_mana"]),
            math.log1p(obs["xp"]) / 10.0,
            obs["gold"] / 1000.0,
            obs["char_level"] / 50.0,
            obs["dungeon_level"] / 16.0,
            px / 112.0,
            py / 112.0,
            min(1.0, len(policy_monsters) / 50.0),
            min(1.0, nearest / 30.0) if nearest is not None else 1.0,
            stair_dx,
            stair_dy,
        ]
        monsters = sorted(
            policy_monsters, key=lambda m: abs(m["x"] - px) + abs(m["y"] - py)
        )[:_K_MONSTERS]
        for m in monsters:
            vec += [(m["x"] - px) / 20.0, (m["y"] - py) / 20.0, m["hp"] / max(1, m["max_hp"]), 1.0]
        vec += [0.0, 0.0, 0.0, 0.0] * (_K_MONSTERS - len(monsters))
        lm = bridge.local_map(radius=_MAP_RADIUS)
        vec += [float(v) for v in lm["walkable"]]
        # local_map 的 monster 通道是物理碰撞事实源，会包含墙后/未照亮怪；
        # 策略观测必须与 token/a9 共用可见可达真源，不能从这 121 位侧信道
        # 偷看全知占位。规划宏仍可用原始通道避免穿过实体。
        policy_tiles = {(int(m["x"]), int(m["y"])) for m in policy_monsters}
        vec += [
            1.0 if (px + dx, py + dy) in policy_tiles else 0.0
            for dy in range(-_MAP_RADIUS, _MAP_RADIUS + 1)
            for dx in range(-_MAP_RADIUS, _MAP_RADIUS + 1)
        ]
        heals = cls._policy_floor_items(obs, "heal")
        belt_scalar = cls._belt_observation_scalar(obs)
        if heals:  # v13:瓶盲修复——喝药/捡药两个键的前置条件入观测
            h = min(heals, key=lambda it: max(abs(it["x"] - px), abs(it["y"] - py)))
            vec += [belt_scalar,
                    max(-1.0, min(1.0, (h["x"] - px) / 20.0)),
                    max(-1.0, min(1.0, (h["y"] - py) / 20.0)), 1.0]
        else:
            vec += [belt_scalar, 0.0, 0.0, 0.0]
        gears = cls._policy_floor_items(obs, "gear")
        ac = max(0.0, min(1.0, obs.get("armor_class", 0) / 50.0))
        if gears:  # v14:装备章——捡装备键的前置条件入观测(教训十一验收单)
            g = min(gears, key=lambda it: max(abs(it["x"] - px), abs(it["y"] - py)))
            vec += [ac,
                    max(-1.0, min(1.0, (g["x"] - px) / 20.0)),
                    max(-1.0, min(1.0, (g["y"] - py) / 20.0)), 1.0]
        else:
            vec += [ac, 0.0, 0.0, 0.0]
        # v19:强弱仪表(教训五族)。"够不够强、该不该下"的决策变量是
        # 等级/层数之比,但 dim5 的 char_level/50 让 1 级和 3 级只差 0.04,
        # 对策略近乎不可见——农到多强才下楼,得先看得见"多强"。
        # 比值 1.0 = 等级与层数持平,>1 越级碾压,<1 越级送死;封顶 2 归一。
        vec += [min(2.0, obs["char_level"] / max(1, obs["dungeon_level"])) / 2.0]
        return np.asarray(vec, dtype=np.float32)
