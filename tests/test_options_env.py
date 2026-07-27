"""v22 G0 单元测试:OptionsEnv 状态机角落 + 不变量(纯脚本断言,无 pytest 依赖)。"""
import pathlib
import sys
import types

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from diablogym import OptionsEnv, bridge
from diablogym.options_env import (
    BLOCKER_RADIUS,
    BeatOutcome,
    DIVE,
    FARM,
    FARM_SCENE_CAP,
    GEAR_GRACE_MAX_DECISIONS,
    KILL_PATIENCE,
    MANAGER_OBSERVATION_VIEW_LEGACY_V3,
    RESUPPLY,
    REVISIT_FLOOR,
    StagnationClockWrapper,
    VOLUNTARY_DRINK_HP_HIGH,
    VOLUNTARY_DRINK_HP_LOW,
    WORKER_DRINK_LATCH_FEATURE,
    dispatch,
)

env = OptionsEnv(max_steps=3000)
try:
    OptionsEnv(max_steps=1, descend_ladder=False)
except ValueError as exc:
    assert "descend_ladder=True" in str(exc)
else:
    raise AssertionError(
        "OptionsEnv 允许关闭 descend_ladder，却仍从 Worker 工资扣深度奖金")

# --- 1. 形状与掩码基本面 ---
obs, _ = env.reset(seed=7000)
assert obs.shape == (303,), obs.shape
assert obs.dtype == np.float32 and env.observation_space.contains(obs)
assert env.np_random is not None, "OptionsEnv.reset(seed) 未建立自身 RNG"
m = env.action_masks()
assert m.shape == (3,) and m.dtype == bool and m[FARM], m
print("G0.1 PASS: obs 303 维,掩码 3 位,普通态 FARM 保底为真")

# --- 1b. 拾药合法性必须看真实腰带空位，而不是治疗药数量近似 ---
full_belt = {
    "hp": 60, "max_hp": 70, "belt_heals": 2, "belt_free_slots": 0,
    "player_x": 10, "player_y": 10, "char_level": 1,
    "dungeon_level": 1, "is_set_level": False,
    "monsters": [], "progression_targets": [], "triggers": [],
    "floor_items": [{
        "heal": True, "visible": True, "reachable": True, "x": 11, "y": 10,
    }],
}
for mode in ("farm", "dive", "resupply"):
    assert dispatch(mode, full_belt, False) != 13, mode

progression_only = {
    **full_belt,
    "belt_free_slots": 6,
    "floor_items": [],
    "progression_targets": [{
        "kind": "diablo_switch", "action": "operate",
        "x": 20, "y": 20, "goal_x": 20, "goal_y": 20,
    }],
}
flat_mask = np.ones(15, dtype=bool)
assert dispatch(
    "farm", progression_only, False, action_mask=flat_mask,
    nearest_engageable_distance=None,
) == 11, "flat teacher 不得采 action10→wait 剧情空标签"

fixture = OptionsEnv.__new__(OptionsEnv)
fixture._last_base_obs = np.zeros(295, dtype=np.float32)
fixture.exhausted = False
fixture.env = types.SimpleNamespace(
    _raw=full_belt, _steps=1, _ep_kills=0,
    _ensure_active=lambda **kwargs: None,
)
assert not fixture.action_masks()[RESUPPLY]
fixture._win = {
    "t0": 0, "scene0": (1, False, 0), "dlvl0": 1, "clvl0": 1,
    "floor": 0, "opt": RESUPPLY, "resupply_stall": 0,
}
fixture._cap_hits = 0
fixture.layer_clock = 0
assert fixture._win_term(False, False, 1) == "done"

far_heal = {
    **progression_only,
    "progression_targets": [],
    "floor_items": [{
        "heal": True, "visible": True, "reachable": True,
        "x": 23, "y": 10,
    }],
}
far_controller_mask = np.ones(15, dtype=bool)
far_controller_mask[13] = False
far_fixture = OptionsEnv.__new__(OptionsEnv)
far_fixture._last_base_obs = np.zeros(295, dtype=np.float32)
far_fixture.exhausted = False
far_fixture.env = types.SimpleNamespace(
    _raw=far_heal, _steps=1, _ep_kills=0,
    _ensure_active=lambda **kwargs: None,
    controller_action_context=lambda: (
        far_controller_mask.copy(), None),
)
assert not far_fixture.action_masks()[RESUPPLY], (
    "radius13 药水不得开启必然两拍空转的 RESUPPLY 窗")
print("G0.1b PASS: 2药+6非药导致 free_slots=0 时拾药掩码/调度/收窗均禁13")

# --- 1b2. 榨干态只在 DIVE 合法时强制交权；非法时 FARM 保底 ---
plain_exhausted = {
    **full_belt,
    "floor_items": [],
    "triggers": [],
}
fixture.env._raw = plain_exhausted
fixture.exhausted = True
mm = fixture.action_masks()
assert not mm[DIVE] and mm[FARM] and mm.any(), mm

dive_exhausted = {
    **plain_exhausted,
    "triggers": [{
        "msg": bridge.WM_DIABNEXTLVL, "x": 80, "y": 80,
    }],
}
fixture.env._raw = dive_exhausted
mm = fixture.action_masks()
assert mm[DIVE] and not mm[FARM] and mm.any(), mm

# 新杀与换场景都必须清掉 exhausted；清旗后普通态 FARM 恢复合法。
fixture.layer_clock = KILL_PATIENCE
fixture.env._ep_kills = 1
fixture._layer_kills0 = 0
fixture._legacy_layer_clock = KILL_PATIENCE
fixture._legacy_exhausted = True
fixture._legacy_layer_steps0 = 0
fixture._tick_layer_clock(
    kills_before=0, scene_before=(1, False, 0), steps_delta=1)
assert not fixture.exhausted and fixture.layer_clock == 0
assert not fixture._legacy_exhausted and fixture._legacy_layer_clock == 0
assert fixture.action_masks()[FARM]

fixture.exhausted = True
fixture.layer_clock = KILL_PATIENCE
fixture.env._raw = {**dive_exhausted, "dungeon_level": 2}
fixture._tick_layer_clock(
    kills_before=fixture.env._ep_kills,
    scene_before=(1, False, 0), steps_delta=1)
assert not fixture.exhausted and fixture.layer_clock == 0
assert fixture.action_masks()[FARM]
print("G0.1b2 PASS: exhausted+DIVE 强制交权；DIVE 非法时 FARM 保底；"
      "新杀/换场景清旗恢复")

# --- 1b3. 冻结 V28/M29 的旧状态语义必须与新收窗状态物理隔离 ---
alias = OptionsEnv.__new__(OptionsEnv)
manager_raw = {
    **plain_exhausted,
    "belt_free_slots": 6,
    "legacy_belt_heals": 2,
}
alias._last_base_obs = base_alias = np.zeros(295, dtype=np.float32)
base_alias[286] = 2 / 8 + 6 / 128  # current packed heals/free scalar

def rebuild_legacy_manager(_raw):
    legacy = base_alias.copy()
    legacy[286] = 2 / 8
    return legacy

alias.env = types.SimpleNamespace(
    _raw=manager_raw, _steps=0, _ep_kills=0,
    _ensure_active=lambda **kwargs: None,
    _legacy_policy_vectorize=rebuild_legacy_manager,
)
alias.max_steps = 3000
alias.manager_observation_view = MANAGER_OBSERVATION_VIEW_LEGACY_V3
alias._reset_wrapper_state()
alias._last_base_obs = base_alias
alias._win = {"t0": 0, "voluntary_drinks": 0, "drains": 0}
manager_reset = alias._mgr_obs(base_alias)
worker_reset = alias._worker_obs()
assert alias._legacy_layer_clock == 0
assert not alias._legacy_exhausted and alias._legacy_layer_steps0 == 0
assert manager_reset[286] == 2 / 8
assert worker_reset[286] == base_alias[286]
assert manager_reset[296] == 0.0 and manager_reset[298] == 0.0
assert worker_reset[296] == 0.0 and worker_reset[297] == 0.0

# 非击杀正向进度只清“当前 no-progress”钟；冻结网络仍逐拍看到旧无新杀钟。
alias.layer_clock = 19
alias.env._steps = 17
alias._tick_layer_clock(
    kills_before=0, scene_before=(1, False, 0), steps_delta=17,
    positive_progress=True)
manager_progress = alias._mgr_obs(base_alias)
worker_progress = alias._worker_obs()
assert alias.layer_clock == 0
assert alias._legacy_layer_clock == 17 and not alias._legacy_exhausted
assert np.isclose(manager_progress[296], 17 / KILL_PATIENCE)
assert np.isclose(manager_progress[298], 17 / 1500)
assert np.isclose(worker_progress[296], 17 / KILL_PATIENCE)
assert worker_progress[297] == 0.0

# 新杀按旧契约清 296/297，但旧 layer_steps0 只在换 scene 时重置。
alias.env._steps = 23
alias.env._ep_kills = 1
alias._tick_layer_clock(
    kills_before=0, scene_before=(1, False, 0), steps_delta=6,
    positive_progress=True)
manager_kill = alias._mgr_obs(base_alias)
assert alias._legacy_layer_clock == 0 and not alias._legacy_exhausted
assert manager_kill[296] == 0.0
assert np.isclose(manager_kill[298], 23 / 1500)

# 换 scene 同时重置旧钟、旧层起点与本层击杀基线。
alias.env._steps = 31
alias.env._raw = {**manager_raw, "dungeon_level": 2}
alias._tick_layer_clock(
    kills_before=1, scene_before=(1, False, 0), steps_delta=8,
    positive_progress=True)
manager_scene = alias._mgr_obs(base_alias)
assert alias._legacy_layer_clock == 0 and not alias._legacy_exhausted
assert alias._legacy_layer_steps0 == 31 and alias._layer_kills0 == 1
assert manager_scene[296] == 0.0 and manager_scene[297] == 0.0
assert manager_scene[298] == 0.0

# 新 scene 内的正向进度仍只清当前钟；旧 296 与旧层耗时继续增长。
alias.env._steps = 38
alias._tick_layer_clock(
    kills_before=1, scene_before=(2, False, 0), steps_delta=7,
    positive_progress=True)
manager_scene_progress = alias._mgr_obs(base_alias)
assert alias.layer_clock == 0 and alias._legacy_layer_clock == 7
assert np.isclose(manager_scene_progress[296], 7 / KILL_PATIENCE)
assert np.isclose(manager_scene_progress[298], 7 / 1500)

# scene-cap/current exhausted 只改变内部动力学，不得污染冻结输入。旧钟跨过
# 140 后则按 V28 的旧语义发布 296=1、297=1；饮闩用 -2 无损复用该位。
alias.farm_scene_steps = FARM_SCENE_CAP
manager_before_current_cap = alias._mgr_obs(base_alias)
worker_before_current_cap = alias._worker_obs()
alias._mark_exhausted()
assert alias.exhausted and alias.layer_clock == KILL_PATIENCE
assert np.array_equal(alias._mgr_obs(base_alias), manager_before_current_cap)
assert np.array_equal(alias._worker_obs(), worker_before_current_cap)
alias.env._steps += KILL_PATIENCE
alias._tick_layer_clock(
    kills_before=1, scene_before=(2, False, 0),
    steps_delta=KILL_PATIENCE, positive_progress=True)
alias._win.update({
    "opt": FARM, "scene0": (2, False, 0), "floor": 0, "clvl0": 1,
})
assert alias._win_term(False, False, 0) == "exhausted"
worker_legacy_dry = alias._worker_obs()
assert alias._legacy_exhausted
assert worker_legacy_dry[296] == 1.0 and worker_legacy_dry[297] == 1.0
alias._win["voluntary_drinks"] = 1
worker_after_drink = alias._worker_obs()
assert WORKER_DRINK_LATCH_FEATURE == 297
assert worker_after_drink[297] == -2.0
assert -worker_after_drink[297] - 1.0 == worker_legacy_dry[297]
alias._win["voluntary_drinks"] = 0
print("G0.1b3 PASS: reset/新杀/换scene/正向进度逐拍钉死 V28/M29 旧语义；"
      "current no-progress/scene-cap 仅留内部，饮闩在旧 exhausted 位无损复用")

# --- 1c. FARM 不得借 action10 操作剧情；主动饮药能力边界必须可见 ---
handoff_raw = dict(full_belt)
handoff_raw.update({
    "belt_free_slots": 6,
    "floor_items": [],
    "progression_targets": [{
        "kind": "vile_book", "action": "operate",
        "x": 20, "y": 20, "goal_x": 20, "goal_y": 21,
    }],
})
fixture.env._raw = handoff_raw
fixture.exhausted = False
mm = fixture.action_masks()
assert mm[DIVE] and not mm[FARM], mm

for distance, dive_action, farm_legal in (
        (4, 9, True), (BLOCKER_RADIUS, 9, True),
        (BLOCKER_RADIUS + 1, 11, False)):
    boundary_raw = {
        **handoff_raw,
        "monsters": [{
            "id": 1, "x": 10 + distance, "y": 10,
            "hp": 10, "max_hp": 10, "visible": True, "reachable": True,
        }],
    }
    assert dispatch("dive", boundary_raw, False) == dive_action
    fixture.env._raw = boundary_raw
    assert bool(fixture.action_masks()[FARM]) is farm_legal

# DIVE keeps survival and blocker clearing ahead of gear, but must drain an
# exact a14 before the one-way progression action 11. Once the target has
# disappeared the same state advances normally. A legacy call without an
# exact controller mask cannot prove pickup reachability and remains a11.
dive_gear_mask = np.ones(15, dtype=bool)
dive_gear_raw = {
    **handoff_raw,
    "hp": 60, "max_hp": 100, "belt_heals": 3,
}
assert dispatch(
    "dive", dive_gear_raw, True, action_mask=dive_gear_mask,
    nearest_engageable_distance=None,
) == 14
dive_gear_mask[14] = False
assert dispatch(
    "dive", dive_gear_raw, False, action_mask=dive_gear_mask,
    nearest_engageable_distance=None,
) == 11
assert dispatch(
    "dive", dive_gear_raw, True,
    nearest_engageable_distance=None,
) == 11

dive_priority_mask = np.ones(15, dtype=bool)
assert dispatch(
    "dive", {**dive_gear_raw, "hp": 40, "belt_heals": 3}, True,
    action_mask=dive_priority_mask,
    nearest_engageable_distance=None,
) == 12
assert dispatch(
    "dive", {**dive_gear_raw, "belt_heals": 2}, True,
    action_mask=dive_priority_mask,
    nearest_engageable_distance=None,
) == 13
assert dispatch(
    "dive", dive_gear_raw, True,
    action_mask=dive_priority_mask,
    nearest_engageable_distance=BLOCKER_RADIUS,
) == 9

worker_fixture = OptionsEnv.__new__(OptionsEnv)
worker_fixture.env = types.SimpleNamespace(
    _raw=handoff_raw,
    action_space=gym.spaces.Discrete(15),
    action_masks=lambda: np.ones(15, dtype=bool),
)
worker_fixture.drink_sovereignty = True
worker_fixture._win = {"voluntary_drinks": 0}
# handoff_raw 的 60/70≈0.857 已在明显浪费区；先换成安全包络内状态验证
# 主权仍然可用，随后逐点钉死两端的闭/开边界。
worker_fixture.env._raw = {**handoff_raw, "hp": 45}
wm = worker_fixture._worker_masks()
assert not wm[10] and wm[12], wm
for distance in (4, BLOCKER_RADIUS, BLOCKER_RADIUS + 1):
    worker_fixture.env._raw = {
        **handoff_raw,
        "monsters": [{
            "id": 1, "x": 10 + distance, "y": 10,
            "hp": 10, "max_hp": 10, "visible": True, "reachable": True,
        }],
    }
    assert not worker_fixture._worker_masks()[10], distance
for hp, expected in ((49, False), (50, True), (74, True),
                     (75, False), (99, False)):
    worker_fixture.env._raw = {
        **handoff_raw, "hp": hp, "max_hp": 100,
    }
    assert bool(worker_fixture._worker_masks()[12]) is expected, (
        hp, worker_fixture._worker_masks())
assert VOLUNTARY_DRINK_HP_LOW == 0.5
assert VOLUNTARY_DRINK_HP_HIGH == 0.75

worker_fixture.env._raw = {**handoff_raw, "hp": 60, "max_hp": 100}
worker_fixture._win["voluntary_drinks"] = 1
assert not worker_fixture._worker_masks()[12], (
    "本窗成功主动饮药后，公开的 feature297 闩必须关闭后续 worker m12")

# Worker 不能用普通方向键绕过 a11，踩相邻楼梯来逃避死亡罚。
escape_raw = {
    **handoff_raw,
    "progression_targets": [],
    "triggers": [{"x": 11, "y": 10, "msg": 123}],
}
worker_fixture.env._raw = escape_raw
escape_mask = worker_fixture._worker_masks()
assert not escape_mask[3] and escape_mask[7], escape_mask

# mask 与执行入口必须是同一条硬边界：0.74 可真实执行，且执行后若仍处
# 安全包络，本窗闩也必须禁止第二瓶；0.75 即使调用方绕过 MaskablePPO
# 直接 step 也要 fail-loud。
executed = []
execute_fixture = OptionsEnv.__new__(OptionsEnv)
execute_fixture.env = types.SimpleNamespace(
    _raw={**handoff_raw, "hp": 74, "max_hp": 100},
    _ep_kills=0,
    action_space=gym.spaces.Discrete(15),
    action_masks=lambda: np.ones(15, dtype=bool),
)
execute_fixture.drink_sovereignty = True
execute_fixture._win = {
    "W": 0.0, "worker_wage": 0.0, "worker_kills": 0,
    "voluntary_drinks": 0,
}
execute_fixture._win_beat = lambda a: (
    executed.append(a) or BeatOutcome(None, a, a, False))
execute_fixture._drain = lambda: None
outcome = execute_fixture._win_step_worker(12)
assert outcome.executed_action == 12 and executed == [12]
assert execute_fixture._win["voluntary_drinks"] == 1
assert not execute_fixture._worker_masks()[12]
try:
    execute_fixture._win_step_worker(12)
except ValueError as exc:
    assert "被掩码却被执行" in str(exc), exc
else:
    raise AssertionError("主动饮闩未阻止同一窗口第二瓶")
assert execute_fixture._win["voluntary_drinks"] == 1
execute_fixture.env._raw = {**handoff_raw, "hp": 75, "max_hp": 100}
try:
    execute_fixture._win_step_worker(12)
except ValueError as exc:
    assert "被掩码却被执行" in str(exc), exc
else:
    raise AssertionError("hp=0.75 的主动饮绕过了 Worker 安全包络")
assert executed == [12], executed

execute_fixture.env._raw = escape_raw
try:
    execute_fixture._win_step_worker(3)
except ValueError as exc:
    assert "DIVE 专属" in str(exc), exc
else:
    raise AssertionError("Worker 方向键绕过 mask 踩入下楼 trigger")
assert executed == [12], executed

drink_fixture = OptionsEnv.__new__(OptionsEnv)
drink_fixture.env = types.SimpleNamespace(
    action_space=gym.spaces.Discrete(15), _ep_kills=0)
drink_fixture._win = {
    "W": 0.0, "worker_wage": 0.0, "worker_kills": 0,
    "voluntary_drinks": 0,
}
drink_fixture._worker_masks = lambda: np.ones(15, dtype=bool)
drink_fixture._win_beat = lambda a: BeatOutcome(None, a, a, False)
drink_fixture._drain = lambda: None
drink_fixture._win_step_worker(12)
assert drink_fixture._win["voluntary_drinks"] == 1

# 精确归因只取当前 policy transition 的 W/kill 增量：开窗恢复/排水留下的
# 既有窗口账不能再次归给网络；当前动作后的反射尾部则属于同一 transition。
ledger_fixture = OptionsEnv.__new__(OptionsEnv)
ledger_fixture.env = types.SimpleNamespace(
    action_space=gym.spaces.Discrete(15), _ep_kills=4)
ledger_fixture._win = {
    "W": 7.5, "worker_wage": 0.25, "worker_kills": 1,
    "voluntary_drinks": 0,
}
ledger_fixture._worker_masks = lambda: np.ones(15, dtype=bool)


def _ledger_primary(a):
    ledger_fixture._win["W"] += 1.25
    ledger_fixture.env._ep_kills += 2
    return BeatOutcome(None, a, a, False)


def _ledger_tail():
    ledger_fixture._win["W"] += 0.5
    ledger_fixture.env._ep_kills += 1
    return BeatOutcome("done", 12, 12, False)


ledger_fixture._win_beat = _ledger_primary
ledger_fixture._drain = _ledger_tail
ledger_outcome = ledger_fixture._win_step_worker(9)
assert ledger_outcome.reason == "done"
assert ledger_fixture._win["W"] == 9.25
assert ledger_fixture._win["worker_wage"] == 2.0
assert ledger_fixture._win["worker_kills"] == 4
print("G0.1c PASS: 剧情交权/DIVE 共用 blocker≤6 边界(4/6/7已测)；"
      "工人主动12能力仅依可见[0.5,0.75)+有药，feature297 显式公开"
      "本窗是否已主动饮；"
      "worker 工资/击杀仅归因当前动作及反射尾部")

# --- 1c2. FARM 击杀后装备机会必须先暴露给 Worker，且不能单动作 mask ---
grace_fixture = OptionsEnv.__new__(OptionsEnv)
grace_mask = np.ones(15, dtype=bool)
grace_mask[11] = False
grace_fixture.env = types.SimpleNamespace(
    _raw={
        **handoff_raw,
        "gear_combat_utility": 100,
        "floor_items": [{
            "gear": True, "heal": False, "visible": True,
            "reachable": True, "x": 11, "y": 10,
        }],
    },
    _steps=1,
    _ep_kills=0,
    action_space=gym.spaces.Discrete(15),
)
grace_fixture._controller_action_context = lambda: (
    grace_mask.copy(), None)
grace_fixture.drink_sovereignty = True
grace_fixture._win = {
    "opt": FARM, "scene0": (1, False, 0), "t0": 0,
    "clvl0": 1, "dlvl0": 1, "floor": 0,
    "gear_grace_pending": False,
    "gear_grace_consumed": False,
    "gear_grace_opportunities": 0,
    "gear_grace_decisions": 0,
    "W": 0.0, "worker_wage": 0.0, "worker_kills": 0,
    "worker_action14_requests": 0,
    "worker_action14_native_successes": 0,
    "worker_action14_gear_utility_delta": 0,
    "worker_no_effect_requests": 0,
    "voluntary_drinks": 0,
}
grace_fixture.layer_clock = 0
grace_fixture.farm_scene_steps = 0
grace_fixture._legacy_layer_clock = 0
grace_fixture._cap_hits = 0
assert grace_fixture._win_term(False, False, 0) is None
assert grace_fixture._win["gear_grace_pending"]
masked = grace_fixture._worker_masks()
assert np.flatnonzero(masked).tolist() == [0, 14], masked

partial_effect = {
    "requested_action": 14,
    "native_attempts": 1,
    "native_accepts": 1,
    "request_executed": True,
    "material_effect": True,
    "effect_reasons": ("move",),
    "same_scene": True,
    "stall_cost_applied": False,
}
partial_gear = {
    "accepted": False, "commit_attempts": 0,
    "utility_before": 100, "utility_after": 100,
    "utility_delta": 0,
}


def _partial_grace_beat(action):
    assert int(action) == 14
    return BeatOutcome(
        grace_fixture._win_term(False, False, 0),
        14, 14, False,
        action14_audit=partial_gear,
        action_effect_audit=partial_effect,
    )


grace_fixture._win_beat = _partial_grace_beat
grace_fixture._drain = lambda: None
for decision in range(1, GEAR_GRACE_MAX_DECISIONS + 1):
    outcome = grace_fixture._win_step_worker(14)
    assert grace_fixture._win["gear_grace_decisions"] == decision
    if decision < GEAR_GRACE_MAX_DECISIONS:
        assert outcome.reason is None
        assert grace_fixture._win["gear_grace_pending"]
    else:
        assert outcome.reason == "handoff"
        assert not grace_fixture._win["gear_grace_pending"]
assert grace_fixture._win["gear_grace_opportunities"] == 1
assert grace_fixture._win["worker_action14_requests"] == (
    GEAR_GRACE_MAX_DECISIONS)
assert grace_fixture._win["worker_action14_native_successes"] == 0
print("G0.1c2 PASS: post-kill gear grace 向 Worker 暴露 {0,14}，"
      "partial a14 可有界续走且不伪造 native equip success")

# --- 1d. fuse 签名必须识别真实战斗进展，且怪列表顺序无关 ---
fuse_raw = {
    "hp": 70, "max_hp": 70, "mana": 20, "max_mana": 20,
    "xp": 0, "gold": 50, "armor_class": 0,
    "gear_combat_utility": 0, "char_level": 1,
    "belt_heals": 0, "belt_free_slots": 8,
    "player_x": 10, "player_y": 10, "dungeon_level": 1,
    "is_set_level": False, "floor_items": [], "progression_targets": [],
    "monsters": [
        {"id": 2, "x": 11, "y": 10, "hp": 100, "max_hp": 100,
         "visible": True, "reachable": True,
         "rnd_item_seed_hi": 0, "rnd_item_seed_lo": 3},
        {"id": 1, "x": 12, "y": 10, "hp": 100, "max_hp": 100,
         "visible": True, "reachable": True,
         "rnd_item_seed_hi": 0, "rnd_item_seed_lo": 2},
    ],
}


class _FuseBase:
    action_space = gym.spaces.Discrete(15)

    def __init__(self, raw, *, target_damage, player_damage=False,
                 exploration_progress=False, material_progress=False):
        self._raw = {
            **raw,
            "monsters": [dict(m) for m in raw["monsters"]],
        }
        self._ep_kills = 0
        self._steps = 0
        self._exploration_progress = 0
        self._combat_hp_floor = {
            int(m["id"]): (int(m["hp"]), int(m["max_hp"]))
            for m in self._raw["monsters"]
        }
        self.target_damage = target_damage
        self.player_damage = player_damage
        self.add_exploration_progress = exploration_progress
        self.material_progress = material_progress

    @property
    def exploration_progress(self):
        return self._exploration_progress

    @staticmethod
    def _ensure_active(**_kwargs):
        return None

    def step(self, action):
        utility_before = int(self._raw["gear_combat_utility"])
        if self.target_damage:
            self._raw["monsters"][0]["hp"] -= 1
            mid = int(self._raw["monsters"][0]["id"])
            _, maximum = self._combat_hp_floor[mid]
            self._combat_hp_floor[mid] = (
                int(self._raw["monsters"][0]["hp"]), maximum)
        if self.player_damage:
            self._raw["hp"] -= 1
        if self.add_exploration_progress:
            self._exploration_progress += 1
        if self.material_progress:
            self._raw["gear_combat_utility"] += 1
        self._steps += 1
        info = {}
        material_effect = bool(
            self.target_damage
            or self.add_exploration_progress
            or self.material_progress
        )
        effect_reasons = (
            ("fixture_progress",) if material_effect else ())
        info["action_effect_audit"] = {
            "requested_action": int(action),
            "native_attempts": 0 if int(action) == 0 else 1,
            "native_accepts": 0 if int(action) == 0 else 1,
            "request_executed": True,
            "material_effect": material_effect,
            "effect_reasons": effect_reasons,
            "same_scene": True,
            "stall_cost_applied": int(action) == 0,
        }
        if int(action) == 14:
            utility_after = int(self._raw["gear_combat_utility"])
            accepted = utility_after > utility_before
            info["action14_audit"] = {
                "accepted": accepted,
                "commit_attempts": 1 if accepted else 0,
                "utility_before": utility_before,
                "utility_after": utility_after,
                "utility_delta": utility_after - utility_before,
            }
        return np.zeros(295, dtype=np.float32), 0.0, False, False, info


def _fuse_probe(*, target_damage, player_damage=False,
                exploration_progress=False, material_progress=False):
    probe = OptionsEnv.__new__(OptionsEnv)
    probe.env = _FuseBase(
        fuse_raw, target_damage=target_damage, player_damage=player_damage,
        exploration_progress=exploration_progress,
        material_progress=material_progress)
    probe._fuse_sig = None
    probe._fuse = 0
    probe.layer_clock = 0
    probe.exhausted = False
    probe._layer_kills0 = 0
    probe._legacy_layer_clock = 0
    probe._legacy_exhausted = False
    probe._legacy_layer_steps0 = 0
    probe._last_base_obs = np.zeros(295, dtype=np.float32)
    return probe


flat_damage_base = _FuseBase(fuse_raw, target_damage=True)
flat_clock = StagnationClockWrapper.__new__(StagnationClockWrapper)
flat_clock.env = flat_damage_base
flat_clock._clock = KILL_PATIENCE
flat_clock._kills = 0
flat_clock._scene = (1, False, 0)
flat_clock.step(9)
assert flat_clock._clock == 0, (
    "flat 停滞钟在怪物最低血线继续下降时错误强制下潜")

flat_static_base = _FuseBase(fuse_raw, target_damage=False)
flat_clock.env = flat_static_base
flat_clock._clock = 0
flat_clock._kills = 0
flat_clock._scene = (1, False, 0)
flat_clock.step(9)
assert flat_clock._clock == 1


ordered = _fuse_probe(target_damage=False)
sig_ordered = ordered._sig(9, ordered.env._raw)
ordered.env._raw["monsters"].reverse()
assert ordered._sig(9, ordered.env._raw) == sig_ordered
ordered.env._raw["monsters"][0]["hp"] -= 1
assert ordered._sig(9, ordered.env._raw) != sig_ordered

# native slot 复用时 rndItemSeed 才是生命周期边界；同 id 新 generation
# 与同坐标新 active item 都必须打破 fuse，且地面装备效用变化不可别名。
identity_probe = _fuse_probe(target_damage=False)
identity_raw = identity_probe.env._raw
identity_sig = identity_probe._sig(9, identity_raw)
identity_raw["monsters"][0]["rnd_item_seed_lo"] += 1
assert identity_probe._sig(9, identity_raw) != identity_sig

identity_raw["floor_items"] = [{
    "active_id": 4,
    "x": 10, "y": 10,
    "heal": False, "gear": True,
    "visible": True, "reachable": True,
    "seed_hi": 1, "seed_lo": 2, "create_info": 3, "base_id": 4,
    "combat_utility_hi": 0, "combat_utility_lo": 7,
}]
item_sig = identity_probe._sig(14, identity_raw)
identity_raw["floor_items"][0]["active_id"] = 5
assert identity_probe._sig(14, identity_raw) != item_sig
identity_raw["floor_items"][0]["active_id"] = 4
identity_raw["floor_items"][0]["combat_utility_lo"] = 8
assert identity_probe._sig(14, identity_raw) != item_sig
identity_raw["floor_items"][0]["combat_utility_lo"] = 7
identity_raw["floor_items"][0]["seed_lo"] = 9
assert identity_probe._sig(14, identity_raw) != item_sig
del identity_raw["floor_items"][0]["combat_utility_hi"]
try:
    identity_probe._sig(14, identity_raw)
except RuntimeError as exc:
    assert "combat_utility_hi" in str(exc)
else:
    raise AssertionError("fuse 接受了缺 combat_utility_hi 的地面物品")

progressing = _fuse_probe(target_damage=True)
progressing.exhausted = True
for _ in range(30):
    _, _, _, _, audit, _, _ = progressing._beat(9)
    assert not audit.fuse_tripped
assert progressing.layer_clock == 0 and not progressing.exhausted

exploring = _fuse_probe(
    target_damage=False, exploration_progress=True)
exploring.exhausted = True
for _ in range(30):
    _, _, _, _, audit, _, _ = exploring._beat(10)
    assert not audit.fuse_tripped
assert exploring.layer_clock == 0 and not exploring.exhausted

unreachable_damage = _fuse_probe(target_damage=True)
unreachable_damage.env._raw["monsters"][0]["reachable"] = False
for _ in range(30):
    _, _, _, _, audit, _, _ = unreachable_damage._beat(9)
    assert not audit.fuse_tripped

equipping = _fuse_probe(
    target_damage=False, material_progress=True)
equipping.exhausted = True
_, _, _, _, equip_audit, _, _ = equipping._beat(14)
assert equip_audit.executed_action == 14
assert equip_audit.action14_audit["utility_delta"] == 1
assert equipping.layer_clock == 0 and not equipping.exhausted

# Scripted DIVE a14 is manager-owned even when that successful equip reaches
# the DIVE stall boundary on the same beat. Publish it as the actual last
# action without inflating Worker action14 participation ledgers.
scripted_dive_equip = _fuse_probe(
    target_damage=False, material_progress=True)
scripted_dive_equip.env._steps = KILL_PATIENCE - 1
scripted_dive_equip._decisions = 0
scripted_dive_equip._cap_hits = 0
scripted_dive_equip.mode_seq = []
scripted_dive_equip._win = {
    "window_id": 1,
    "opt": DIVE,
    "mode": "dive",
    "t0": 0,
    "clvl0": 1,
    "dlvl0": 1,
    "scene0": (1, False, 0),
    "kills0": 0,
    "floor": 0,
    "resupply_stall": 0,
    "R": 0.0,
    "W": 0.0,
    "bonus": 0.0,
    "worker_wage": 0.0,
    "worker_kills": 0,
    "worker_action14_requests": 0,
    "worker_action14_native_successes": 0,
    "worker_action14_gear_utility_delta": 0,
    "beats": 0,
    "overrides": 0,
    "fuse_trips": 0,
    "drain_attempts": 0,
    "drains": 0,
    "voluntary_drinks": 0,
    "recovery_actions": 0,
    "last_recovery_action": None,
    "last_requested_action": None,
    "last_executed_action": None,
    "fuse_requested_action": None,
    "done": False,
    "trunc": False,
    "last_info": {},
}
scripted_outcome = scripted_dive_equip._win_beat(14)
assert scripted_outcome.reason == "stall"
assert scripted_outcome.executed_action == 14
assert scripted_outcome.action14_audit["utility_delta"] == 1
scripted_extra, _, _, _ = scripted_dive_equip._win_end(
    scripted_outcome.reason)
assert scripted_extra["last_executed_action"] == 14
assert scripted_extra["worker_action14_requests"] == 0
assert scripted_extra["worker_action14_native_successes"] == 0
assert scripted_extra["worker_action14_gear_utility_delta"] == 0

declining_endpoint = _fuse_probe(target_damage=False)
declining_endpoint.env._raw["gear_combat_utility"] = 100
declining_endpoint.exhausted = True


def _accepted_gear_then_endpoint_declines(action):
    assert int(action) == 14
    declining_endpoint.env._raw["gear_combat_utility"] = 63
    declining_endpoint.env._steps += 1
    return np.zeros(295, dtype=np.float32), 0.0, False, False, {
        "action_effect_audit": {
            "requested_action": 14,
            "native_attempts": 1,
            "native_accepts": 1,
            "request_executed": True,
            "material_effect": True,
            "effect_reasons": ("gear_commit",),
            "same_scene": True,
            "stall_cost_applied": False,
        },
        "action14_audit": {
            "accepted": True,
            "commit_attempts": 1,
            "utility_before": 100,
            "utility_after": 137,
            "utility_delta": 37,
        },
    }


declining_endpoint.env.step = _accepted_gear_then_endpoint_declines
_, _, _, _, declining_audit, _, _ = declining_endpoint._beat(14)
assert declining_audit.executed_action == 14
assert declining_audit.action14_audit["utility_delta"] == 37
assert declining_endpoint.layer_clock == 0
assert not declining_endpoint.exhausted

endpoint_only = _fuse_probe(target_damage=False)
endpoint_only.env._raw["gear_combat_utility"] = 100


def _endpoint_rises_without_native_receipt(action):
    assert int(action) == 14
    endpoint_only.env._raw["gear_combat_utility"] = 137
    endpoint_only.env._steps += 1
    return np.zeros(295, dtype=np.float32), 0.0, False, False, {
        "action_effect_audit": {
            "requested_action": 14,
            "native_attempts": 1,
            "native_accepts": 0,
            "request_executed": False,
            "material_effect": False,
            "effect_reasons": (),
            "same_scene": True,
            "stall_cost_applied": True,
        },
        "action14_audit": {
            "accepted": False,
            "commit_attempts": 1,
            "utility_before": 100,
            "utility_after": 100,
            "utility_delta": 0,
        },
    }


endpoint_only.env.step = _endpoint_rises_without_native_receipt
_, _, _, _, endpoint_only_audit, _, _ = endpoint_only._beat(14)
assert endpoint_only_audit.executed_action is None
assert not endpoint_only_audit.action14_audit["accepted"]

static = _fuse_probe(target_damage=False)
trips = 0
for _ in range(30):
    _, _, _, _, audit, _, _ = static._beat(9)
    trips += int(audit.fuse_tripped)
assert trips == 1

under_attack = _fuse_probe(target_damage=False, player_damage=True)
attack_trips = 0
for _ in range(30):
    _, _, _, _, audit, _, _ = under_attack._beat(9)
    attack_trips += int(audit.fuse_tripped)
assert attack_trips == 1
assert under_attack.layer_clock == 29, under_attack.layer_clock
print("G0.1d PASS: 新最低血线/探索/穿装清正进展钟；纯掉血不清且仍fuse；"
      "摘要与列表顺序无关")

# --- 1e. 短期停滞钟与 scene-local FARM 总预算必须分工 ---
continuous = _fuse_probe(
    target_damage=False, exploration_progress=True)
continuous.farm_scene_steps = FARM_SCENE_CAP - 3
continuous._farm_scene = (1, False, 0)
continuous._win = {
    "opt": FARM, "scene0": (1, False, 0),
    "floor": 0, "clvl0": 1, "t0": 0,
}
for _ in range(3):
    continuous._beat(10)
assert continuous.layer_clock == KILL_PATIENCE
assert continuous.farm_scene_steps == FARM_SCENE_CAP
assert continuous.exhausted
assert continuous._win_term(False, False, 8) == "exhausted"
assert continuous.exhausted
# 当前没有 DIVE 目标时 FARM 必须仍合法，不能因累计预算制造全假掩码；
# 复访窗会继续探索，目标一旦出现便由 exhausted 强制交权。
cm = continuous.action_masks()
assert cm[FARM] and not cm[DIVE] and cm.any(), cm

fighting = _fuse_probe(target_damage=True)
fighting.farm_scene_steps = 0
fighting._farm_scene = (1, False, 0)
fighting._win = {
    "opt": FARM, "scene0": (1, False, 0),
    "floor": 0, "clvl0": 1, "t0": 0,
}
for _ in range(KILL_PATIENCE + 20):
    fighting._beat(9)
assert fighting.layer_clock == 0
assert fighting.farm_scene_steps == KILL_PATIENCE + 20
assert fighting._win_term(False, False, 8) is None
assert not fighting.exhausted

# 任务副本与主层都由完整 scene identity 分账；换图清累计预算和榨干旗。
fighting.farm_scene_steps = FARM_SCENE_CAP
fighting.exhausted = True
fighting.env._raw["dungeon_level"] = 2
fighting._tick_layer_clock(
    fighting.env._ep_kills, (1, False, 0), 1)
assert fighting.farm_scene_steps == 0
assert fighting._farm_scene == (2, False, 0)
assert not fighting.exhausted
print("G0.1e PASS: 持续探索命中 scene FARM 总预算；真实战斗不触发短钟；"
      "DIVE 非法保底与换 scene 清账成立")

# --- 2. 换层必归还(显式不变量)---
obs, _ = env.reset(seed=7000)
descended = False
seed7000_fuses = 0
seed7000_recoveries = 0
for _ in range(40):
    if not env.action_masks()[DIVE]:
        break
    lvl_b = env.env._raw["dungeon_level"]
    obs, R, done, trunc, info = env.step(DIVE)
    ex = info["option_extra"]
    seed7000_fuses += ex["fuse_trips"]
    seed7000_recoveries += ex["recovery_actions"]
    if env.env._raw["dungeon_level"] != lvl_b:
        assert ex["reason"] == "descend", ex
        descended = True
        break
    if done or trunc:
        break
assert descended, "40 次 DIVE 未见换层"
assert seed7000_fuses == 0 and seed7000_recoveries == 0, (
    seed7000_fuses, seed7000_recoveries)
print("G0.2 PASS: seed7000 DIVE 真战斗/换层且 fuse=recovery=0")

# --- 3. 榨干旗 + 复选地板 ---
obs, _ = env.reset(seed=7001)
reason = None
for _ in range(60):
    obs, R, done, trunc, info = env.step(FARM)
    reason = info["option_extra"]["reason"]
    if reason == "exhausted" or done or trunc:
        break
if reason == "exhausted":
    assert env.exhausted
    mask = env.action_masks()
    if mask[DIVE]:
        assert not mask[FARM], mask
        print("G0.3 PASS: 榨干旗置位且 DIVE 合法，FARM 被强制交权")
    else:
        assert mask[FARM], mask
        obs, R, done, trunc, info = env.step(FARM)
        assert info["option_extra"]["tau"] >= REVISIT_FLOOR or done or trunc, info["option_extra"]
        print(f"G0.3 PASS: DIVE 非法时 FARM 保底复访 "
              f"{info['option_extra']['tau']} 拍(地板 {REVISIT_FLOOR})")
else:
    print(f"G0.3 SKIP: 该种子未触发榨干(reason={reason}),不变量由 G0.4 模糊测试兜底")

# --- 4. 掩码永不全假 + 随机模糊 200 决策 ---
rng = np.random.default_rng(0)
obs, _ = env.reset(seed=7002)
taus = []
for i in range(200):
    m = env.action_masks()
    assert m.any(), f"决策 {i}:掩码全假"
    opt = int(rng.choice(np.flatnonzero(m)))
    obs, R, done, trunc, info = env.step(opt)
    assert env.env._steps <= env.max_steps, (env.env._steps, env.max_steps)
    taus.append(info["option_extra"]["tau"])
    assert info["option_extra"]["tau"] >= 1
    if done or trunc:
        obs, _ = env.reset(seed=7002 + i)
print(f"G0.4 PASS: 200 决策模糊测试,τ 中位 {sorted(taus)[len(taus)//2]}")

# --- 5. τ 与 env._steps 差分一致 ---
obs, _ = env.reset(seed=7003)
s0 = env.env._steps
obs, R, done, trunc, info = env.step(FARM)
assert info["option_extra"]["tau"] == env.env._steps - s0
print("G0.5 PASS: τ 与微步差分逐位一致")

# --- 5b. 宏动作不得越过基础局微步上限 ---
short = OptionsEnv(max_steps=1)
obs, _ = short.reset(seed=7004)
obs, R, done, trunc, info = short.step(FARM)
assert short.env._steps == short.max_steps == 1, short.env._steps
assert done or trunc
print("G0.5b PASS: 宏动作在 max_steps 剩余预算内截断,无微步越界")

# --- 5c. fuse 不得静默把请求动作替换为 10 ---
obs, _ = env.reset(seed=7005)
env._win_begin(FARM)
requested = 9
env._fuse_sig = env._sig(requested, env.env._raw)
env._fuse = 24
steps_before = env.env._steps
outcome = env._win_beat(requested)
assert outcome.reason == "fuse", outcome
assert outcome.requested_action == requested
assert outcome.executed_action is None and outcome.fuse_tripped
assert env.env._steps == steps_before, "fuse 拒绝提案却推进了基础微拍"
extra, _, done, trunc = env._win_end(outcome.reason)
assert not done and not trunc
assert extra["reason"] == "fuse" and extra["fuse_trips"] == 1
assert extra["overrides"] == 1 and extra["beats"] == 0
assert extra["last_requested_action"] == requested
assert extra["last_executed_action"] is None
assert extra["fuse_requested_action"] == requested
assert isinstance(extra["window_id"], int) and extra["window_id"] > 0
print("G0.5c PASS: fuse 拒绝请求并显式收窗;executed_action=None,未偷执行动作10")

# --- 6. MaskablePPO 冒烟(γ=1)---
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

vec = DummyVecEnv([lambda: Monitor(OptionsEnv(max_steps=600))])
smoke = MaskablePPO("MlpPolicy", vec, n_steps=16, batch_size=16, gamma=1.0,
                    gae_lambda=0.95, seed=22, device="cpu", verbose=0)
smoke.learn(total_timesteps=32, progress_bar=False)
vec.close()
print("G0.6 PASS: MaskablePPO γ=1 在 OptionsEnv 上冒烟完成")
print("G0 ALL PASS")
