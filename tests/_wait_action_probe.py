"""真实资源回归：v4 wait 必须截断旧 walk/attack，宏返回不得泄漏命令。"""

from __future__ import annotations

import pathlib
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import DiabloGymEnv, bridge  # noqa: E402
from diablogym.env import _DIRS  # noqa: E402


SEED = 7001


def find_actionable_monster(env: DiabloGymEnv, seed: int = SEED):
    env.reset(seed=seed)
    bridge.probe_invincible(True)
    for _ in range(20):
        target = env._nearest_monster(env._raw)
        if target is not None:
            return target
        _, _, terminated, truncated, _ = env.step(10)
        if terminated or truncated:
            break
    raise AssertionError("定向种子未在 20 次探索内产生可见可达怪物")


def target_hp(raw, monster_id: int):
    target = next((m for m in raw["monsters"] if m["id"] == monster_id), None)
    return None if target is None else target["hp"]


env = DiabloGymEnv(start_in_dungeon=True, max_steps=1000, include_raw=False)
try:
    # 0. 满血饮药既要被 mask，也要在绕过 Gym mask 直调原生时 fail closed。
    env.reset(seed=SEED)
    bridge.probe_invincible(True)
    full = env._raw
    assert full["hp"] == full["max_hp"] and full["belt_heals"] > 0, full
    assert not env.action_masks()[12], env.action_masks()
    belt0 = full["belt_heals"]
    assert bridge.act_drink() == 0
    assert bridge.step(ticks=1)["belt_heals"] == belt0

    # 0b. action10 的原生安全寻路在执行层硬拒全部 trigger；即使调用方
    # 忘记把楼梯放进 protected_tiles，也不能把探索键变成下楼键。
    env.reset(seed=7000)
    down = next(
        t for t in env._raw["triggers"]
        if t["msg"] == bridge.WM_DIABNEXTLVL)
    protected = [(int(t["x"]), int(t["y"])) for t in env._raw["triggers"]]
    assert bridge.act_explore_walk(
        down["x"],
        down["y"],
        protected,
        env._raw["player_x"],
        env._raw["player_y"],
        12,
    ) == 0
    stayed = bridge.step(ticks=4)
    assert stayed["dungeon_level"] == 1, stayed

    # 0c. 真实 committed-move 决策边界：方向键首拍只推进 4 ticks，但
    # env.step 必须把自然收尾拍计入本动作后才返回 idle；下一 action0
    # 不再继承 tile 变化，也不得冒领旧动作的接近塑形。
    env.reset(seed=7000)
    bridge.probe_invincible(True)
    px, py = env._raw["player_x"], env._raw["player_y"]
    direction = next(
        action for action, (dx, dy) in enumerate(_DIRS, 1)
        if bridge.probe_tile(px + dx, py + dy)["walkable"]
        and not bridge.probe_tile(px + dx, py + dy)["monster"]
    )
    steps_before = env._steps
    env.step(direction)
    moved = (env._raw["player_x"], env._raw["player_y"]) != (px, py)
    assert moved, env._raw
    assert env._steps - steps_before > 1, env._steps - steps_before
    assert env._decision_idle(env._raw), env._raw
    pos_before = (env._raw["player_x"], env._raw["player_y"])
    ledger_before = (
        env._raw["xp"], env._ep_kills, env._raw["dungeon_level"],
        env._raw["armor_class"], env._raw["gold"],
    )
    _, wait_reward, terminated, truncated, _ = env.step(0)
    assert not terminated and not truncated
    assert env._decision_idle(env._raw), env._raw
    assert (env._raw["player_x"], env._raw["player_y"]) == pos_before
    ledger_after = (
        env._raw["xp"], env._ep_kills, env._raw["dungeon_level"],
        env._raw["armor_class"], env._raw["gold"],
    )
    assert ledger_after == ledger_before, (ledger_before, ledger_after)
    assert abs(wait_reward - (-0.002)) < 1e-12, wait_reward

    # 1. 目标格被怪物占据、不可供玩家站立；FindPath 必须仍把目标格当
    # destination 接受，否则 reachable/a9 mask 会把真实近战目标全部屏蔽。
    target = find_actionable_monster(env)
    raw = env._raw
    tile = bridge.probe_tile(target["x"], target["y"])
    assert tile["monster"] != 0, tile
    assert target["visible"] and target["reachable"], target
    assert env.action_masks()[9], env.action_masks()
    t0 = time.perf_counter()
    for _ in range(2000):
        bridge.observe()
    observe_rate = 2000 / (time.perf_counter() - t0)
    # v4 raw 现在为百余只怪保留完整战斗/动画账，并为七个装备槽编码词缀；
    # 本机真实资源约 2.1k/s。1k/s 保险线仍能抓住“每只怪都跑全图寻路”
    # 这类数量级回退，同时不会把有意增加的 Markov 状态当性能故障。
    assert observe_rate >= 1000, observe_rate

    # 2. 同一网络批次 attack→wait：wait 的 FIFO 栅栏必须压过前面的攻击包。
    tid = target["id"]
    hp0 = target["hp"]
    pos0 = (raw["player_x"], raw["player_y"])
    bridge.act_attack_monster(tid)
    assert bridge.act_wait() == 1
    trace = [bridge.step(ticks=1) for _ in range(24)]
    assert all(r["dest_action"] == bridge.ACTION_NONE for r in trace)
    assert all(r["walkpath0"] == bridge.WALK_NONE for r in trace)
    positions = [pos0] + [(r["player_x"], r["player_y"]) for r in trace]
    # find_actionable_monster 的最后一个探索拍可能已经提交一格走路；
    # wait 允许这一格自然收尾，但不能生成第二格追击路径。
    assert len(set(positions)) <= 2, positions
    assert len(set(positions[-12:])) == 1, positions[-12:]
    hp_after = target_hp(trace[-1], tid)
    assert hp_after == hp0, (hp0, hp_after)

    # 3. 先让攻击追击命令真实生效，再 wait。已经提交的一个走路动画可以
    # 收尾，但此后不得继续长路径或在怪物靠近后自动挥击。
    target = find_actionable_monster(env)
    tid = target["id"]
    hp0 = target["hp"]
    bridge.act_attack_monster(tid)
    started = bridge.step(ticks=4)
    assert started["dest_action"] != bridge.ACTION_NONE or started["walkpath0"] != bridge.WALK_NONE, started
    assert bridge.act_wait() == 1
    trace = [bridge.step(ticks=1) for _ in range(48)]
    assert all(r["dest_action"] == bridge.ACTION_NONE for r in trace)
    assert all(r["walkpath0"] == bridge.WALK_NONE for r in trace)
    positions = [(r["player_x"], r["player_y"]) for r in trace]
    assert len(set(positions)) <= 2, positions
    assert len(set(positions[-12:])) == 1, positions[-12:]
    hp_after = target_hp(trace[-1], tid)
    assert hp_after == hp0, (hp0, hp_after)

    # 4. Python 交战宏无论止损还是耗尽，交回策略的 raw 必须已清命令。
    find_actionable_monster(env)
    _, _, terminated, truncated, _ = env.step(9)
    if not terminated and not truncated:
        assert env._raw["dest_action"] == bridge.ACTION_NONE, env._raw
        assert env._raw["walkpath0"] == bridge.WALK_NONE, env._raw

    # 5. seed7000 的首个 explore 曾稳定在 2 beat 后误判 stall：此时 tile
    # 尚未改变，但 PM_WALK/future 已经在推进一格。修复后同一个 action10
    # 必须允许动画完成并真实走出多格，不能靠下一次动作继承残步。
    env.reset(seed=7000)
    bridge.probe_invincible(True)
    explore_start = (env._raw["player_x"], env._raw["player_y"])
    steps_before = env._steps
    env.step(10)
    explore_end = (env._raw["player_x"], env._raw["player_y"])
    explore_beats = env._steps - steps_before
    assert explore_beats > 2, (explore_beats, explore_start, explore_end)
    assert explore_end != explore_start, (explore_beats, explore_start, explore_end)

    # 6. seed7000 DIVE 的真实战斗阻塞回归。旧边界在 action11×3 后会
    # 泄漏 tile/future 不同的未完成单格，并让后续 stop-loss 在首刀前
    # ActWait。合法的逐格路径会随相邻边/怪物时序优化而改变，所以这里
    # 钉执行不变量而非脆弱黄金坐标：每拍必须 idle，两个 blocker 随后
    # 必须由两次 action9 真实清掉。
    env.reset(seed=7000)
    bridge.probe_invincible(True)
    for _ in range(3):
        env.step(11)
    assert env._decision_idle(env._raw), env._raw
    blockers = {
        int(m["id"]): m for m in env._policy_monsters(env._raw)
        if int(m["id"]) in {47, 96}
    }
    assert set(blockers) == {47, 96}, blockers
    env.step(9)
    assert env._decision_idle(env._raw), env._raw
    after_first_ids = {int(m["id"]) for m in env._raw["monsters"]}
    assert len({47, 96} & after_first_ids) == 1, (
        "第一次 action9 未恰好清掉一只 blocker", env._raw)
    env.step(9)
    assert env._decision_idle(env._raw), env._raw
    after_second_ids = {int(m["id"]) for m in env._raw["monsters"]}
    assert not ({47, 96} & after_second_ids), (
        "seed7000 的两只 DIVE blocker 未被真实战斗清掉", env._raw)

    # 7. seed 7006 是旧版稳定复现：action9 返回时剩一刀未结，下一次
    # action0 会杀怪并吃到 +2.628。v4 wait 必须让这枚 no-op 不再补刀。
    target = find_actionable_monster(env, seed=7006)
    tid = target["id"]
    env.step(9)
    after_macro = next(
        (m for m in env._raw["monsters"] if m["id"] == tid), None)
    if after_macro is not None:
        hp_after_macro = after_macro["hp"]
        env.step(0)
        after_wait = next(
            (m for m in env._raw["monsters"] if m["id"] == tid), None)
        assert after_wait is not None, "action0 继承旧刀杀死了目标"
        assert after_wait["hp"] >= hp_after_macro, (
            hp_after_macro, after_wait["hp"])

    print(
        "PASS: occupied monster reachable/a9 mask 成立；"
        "attack→wait/a0 无后续追击或补刀；committed-move 不串账；"
        "宏返回命令态清空；"
        "explore 不误杀走路动画；seed7000 双 blocker 由 action9 清除；"
        f"Observe={observe_rate:.0f}/s"
    )
finally:
    env.close()
