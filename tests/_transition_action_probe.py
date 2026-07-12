"""真实资源回归：拍尾换层即时提交，旧场景动作不得泄漏。"""

from __future__ import annotations

import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from diablogym import DiabloGymEnv, bridge  # noqa: E402
from diablogym.nav import walk_to  # noqa: E402


def descend_without_pending_frame(seed: int):
    """从城镇下楼；任何 bridge.step 返回 PM_NEWLVL 旧场景都立即失败。"""
    raw = bridge.reset(seed=seed)
    stair = next(t for t in raw["triggers"]
                 if t["msg"] == bridge.WM_DIABNEXTLVL)
    neighbor = next(
        (stair["x"] + dx, stair["y"] + dy)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (-1, -1), (1, -1), (-1, 1))
        if bridge.probe_tile(stair["x"] + dx,
                             stair["y"] + dy)["walkable"]
    )
    raw, _ = walk_to(bridge, *neighbor)
    if raw["dungeon_level"] != 0:
        raise AssertionError("走向楼梯相邻格时意外提前换层")

    bridge.act_walk(stair["x"], stair["y"])
    for call_index in range(1, 301):
        raw = bridge.step(ticks=1)
        if raw["player_mode"] == bridge.PM_NEWLVL:
            raise AssertionError(
                f"bridge.step 向 Python 暴露了待处理 PM_NEWLVL 尾帧: {raw}")
        if raw["dungeon_level"] == 1:
            # 深度在这次调用中已变为 1：换层与其奖励可由
            # 上层调用者归给真正踩中楼梯的动作。
            return raw, call_index
    raise AssertionError("城镇楼梯在 300 tick 内未进入 L1")


def level_fingerprint(raw):
    return (
        raw["dungeon_level"],
        raw["player_x"], raw["player_y"],
        raw["future_x"], raw["future_y"],
        raw["dest_action"], raw["walkpath0"],
    )


def queue_probe_warp_to_l2():
    """探针直接调 StartNewLvl，故意构造 Python 正常永远看不到的 guard 窗。"""
    bridge.probe_warp_main_level(2)
    pending = bridge.observe()
    if (pending["player_mode"] != bridge.PM_NEWLVL
            or pending["dungeon_level"] != 1):
        raise AssertionError(f"探针未构造 L1→L2 待处理尾帧: {pending}")
    return pending


env = DiabloGymEnv(max_steps=100, include_raw=False)
try:
    # 普通 Step 路径：换层必须在触发它的同一次调用里完成。
    env.reset(seed=81234)
    _, transition_call = descend_without_pending_frame(81234)

    # 第一跑用探针构造 guard 窗，不发动作，记下 L2 首帧。
    queue_probe_warp_to_l2()
    baseline = bridge.step(ticks=1)
    if baseline["dungeon_level"] != 2:
        raise AssertionError(f"基准跑未进入 L2: {baseline['dungeon_level']}")
    px, py = baseline["player_x"], baseline["player_y"]
    target = next(
        (px + dx, py + dy)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if bridge.probe_tile(px + dx, py + dy)["walkable"]
    )
    expected = level_fingerprint(baseline)

    # 第二跑在同一探针尾帧调用全部训练动作入口。它们
    # 必须是空操作，不能在 SyncLoad 后改写 L2 的 future/destAction。
    _, transition_call_again = descend_without_pending_frame(81234)
    pending = queue_probe_warp_to_l2()
    belt_before = pending["belt_heals"]
    bridge.act_walk(*target)
    bridge.act_attack_monster(0)
    bridge.act_attack_tile(*target)
    bridge.act_operate(*target)
    if bridge.act_drink() != 0:
        raise AssertionError("换层期 act_drink 未被拒绝")
    if bridge.act_pickup() != 0:
        raise AssertionError("换层期 act_pickup 未被拒绝")
    if bridge.act_pickup_gear() != 0:
        raise AssertionError("换层期 act_pickup_gear 未被拒绝")
    if bridge.act_pickup_progression(*target) != 0:
        raise AssertionError("换层期 act_pickup_progression 未被拒绝")
    if bridge.sweep_backpack_gear() != 0:
        raise AssertionError("换层期 sweep_backpack_gear 未被拒绝")
    if bridge.observe()["belt_heals"] != belt_before:
        raise AssertionError("换层期动作改写了旧场景药水状态")

    actual = bridge.step(ticks=1)
    if level_fingerprint(actual) != expected:
        raise AssertionError(
            "旧场景动作泄漏到 L2 首帧:\n"
            f"expected={expected}\nactual={level_fingerprint(actual)}")

    # SyncLoad 必须同时提交本地 join 状态；否则 _pLvlChanging
    # guard 会把新层第一个合法动作也误吞。
    bridge.act_walk(*target)
    accepted = bridge.step(ticks=1)
    if (accepted["future_x"], accepted["future_y"]) != target:
        raise AssertionError(
            "新层第一个合法动作被 _pLvlChanging guard 误吞: "
            f"target={target}, raw={accepted}")

    if transition_call_again != transition_call:
        raise AssertionError(
            f"同 seed 换层调用不确定: {transition_call} != {transition_call_again}")
    print(
        "PASS: 拍尾换层同 step 提交，不暴露 PM_NEWLVL，"
        "guard 不泄漏旧动作也不误吞新动作")
finally:
    env.close()
