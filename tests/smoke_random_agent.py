"""DiabloGym v0 冒烟测试:随机 agent + 确定性验证。

验证链:引擎初始化 → reset(seed) → 随机动作 N 步 → 观测在变 → 同种子可复现。
用法(仓库根目录):  .venv/bin/python tests/smoke_random_agent.py
"""

import os
import pathlib
import signal
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import numpy as np

from diablogym import DiabloGymEnv, bridge


def snapshot(raw):
    """取观测中的确定性指纹:玩家位置 + 前 5 个怪物的位置/血量。"""
    mons = [(m["id"], m["x"], m["y"], m["hp"]) for m in raw["monsters"][:5]]
    return (raw["player_x"], raw["player_y"], raw["dungeon_level"], tuple(mons))


def main():
    print("== DiabloGym v0 冒烟测试 ==")
    env = DiabloGymEnv(ticks_per_step=4, max_steps=1000)

    # --- 1. reset 与初始观测 ---
    obs, info = env.reset(seed=42)
    raw = info["raw"]
    assert info["episode_seed"] == 42
    print(f"reset(seed=42): 城镇位置 ({raw['player_x']},{raw['player_y']}) "
          f"HP {raw['hp']}/{raw['max_hp']} 金币 {raw['gold']} "
          f"层 {raw['dungeon_level']} 怪物数 {len(raw['monsters'])}")
    assert obs.shape == env.observation_space.shape, "观测向量形状不对"
    assert obs.dtype == np.float32 and env.observation_space.contains(obs)

    # info 是调用方所有的快照，修改它不得篡改环境内部奖励基线。
    real_x = env._raw["player_x"]
    raw["player_x"] = 999
    raw["monsters"].clear()
    assert env._raw["player_x"] == real_x, "info['raw'] 泄露了内部可变状态"

    # --- 2. 随机走 300 步 ---
    rng = np.random.default_rng(0)
    t0 = time.time()
    total_reward, positions = 0.0, set()
    for step in range(300):
        action = int(rng.integers(0, 10))
        obs, reward, terminated, truncated, info = env.step(action)
        raw = info["raw"]
        total_reward += reward
        positions.add((raw["player_x"], raw["player_y"]))
        if step % 100 == 0:
            print(f"  step {step:4d}: pos ({raw['player_x']},{raw['player_y']}) "
                  f"HP {raw['hp']} XP {raw['xp']} 层 {raw['dungeon_level']}")
        if terminated:
            print(f"  episode 终止于 step {step}(dead={raw['dead']})")
            break
    dt = time.time() - t0
    ticks = (step + 1) * env.ticks_per_step
    print(f"随机 {step + 1} 步({ticks} tick)耗时 {dt:.2f}s "
          f"≈ {ticks / dt:.0f} tick/s(实时为 20 tick/s,加速 {ticks / dt / 20:.0f}x)")
    assert len(positions) > 3, f"玩家几乎没动过(只到过 {len(positions)} 个格子)—— 动作注入可能失效"
    print(f"PASS: 玩家移动过 {len(positions)} 个格子,动作注入有效")

    # --- 2b. 宏动作定向冒烟(11 下楼 / 12 喝药 / 13 捡药 / 14 捡装备):每个
    # 新引擎代码路径都可能埋着无头雷(教训:蝙蝠俯冲/屠夫台词),CI 必须真踩一遍 ---
    if not terminated:
        for macro in (11, 12, 13, 14):
            obs, reward, terminated, truncated, info = env.step(macro)
            if terminated or truncated:
                break
        print("PASS: 宏动作 11/12/13/14 定向冒烟无崩溃")

    # --- 3. 确定性:同种子同世界,异种子异世界 ---
    _, info_a = env.reset(seed=123)
    snap_a = snapshot(info_a["raw"])
    _, info_b = env.reset(seed=123)
    snap_b = snapshot(info_b["raw"])
    _, info_c = env.reset(seed=456)
    snap_c = snapshot(info_c["raw"])
    assert snap_a == snap_b, f"同种子初始世界不一致!\n{snap_a}\n{snap_b}"
    print("PASS: seed=123 两次 reset 初始世界一致(确定性成立)")
    if snap_a == snap_c:
        print("WARN: seed=123 与 seed=456 初始世界相同(城镇布局本就固定,属正常;下地牢后才分化)")
    else:
        print("PASS: 不同种子初始世界不同")

    # --- 4. Gym/原生边界与精确截断 ---
    short = DiabloGymEnv(ticks_per_step=4, max_steps=1, include_raw=False)
    short.reset(seed=7)
    try:
        env.step(0)
    except RuntimeError as exc:
        assert "交错" in str(exc)
    else:
        raise AssertionError("同进程全局引擎被多环境静默交错使用")
    _, _, terminated, truncated, cap_info = short.step(10)
    # 最长 12 拍的宏也只能用剩余 1 拍。若这一拍停在 295 维未编码的
    # walk/future/mode 中间态，必须 fail-closed terminal，不能让 SB3
    # 把别名 terminal_observation 当成安全 TimeLimit 状态 bootstrap。
    assert (terminated or truncated) and not (terminated and truncated)
    assert short._steps == short.max_steps == 1
    if cap_info["decision_idle"]:
        assert truncated and cap_info["time_limit_bootstrap_safe"]
        assert not cap_info["unsettled_budget_terminal"]
    else:
        assert terminated and not truncated
        assert cap_info["unsettled_budget_terminal"]
        assert not cap_info["time_limit_bootstrap_safe"]
    try:
        short.step(0)
    except Exception as exc:
        assert exc.__class__.__name__ == "ResetNeeded"
    else:
        raise AssertionError("episode 截断后仍可继续 step")
    short.reset(seed=8)
    for bad_call in (
        lambda: bridge.step(ticks=0),
        lambda: bridge.local_map(radius=-1),
        lambda: bridge.probe_tile(-1, 0),
    ):
        try:
            bad_call()
        except (ValueError, IndexError):
            pass
        else:
            raise AssertionError("原生边界未拒绝非法参数")

    # 普通 step 现在会在返回前结算拍尾换层事件，因此用
    # 探针直接排队 StartNewLvl，继续验证"待处理事件后立即
    # reset" 不会把上局换层泄漏给新英雄。
    bridge.reset(seed=81)
    bridge.probe_warp_main_level(1)
    pending = bridge.observe()
    assert (pending["player_mode"] == bridge.PM_NEWLVL
            and pending["dungeon_level"] == 0), "探针未排队换层事件"
    bridge.reset(seed=82)
    assert bridge.step(ticks=1)["dungeon_level"] == 0, "上局换层事件泄漏到新局"

    # 任务只允许向下推进。历史上 FARM 的 explore 会在 seed 7023
    # 误踩 L1 上楼格回城，使余下整局变成 depth=0 的空耗样本。
    env.start_in_dungeon = True
    _, backtrack_info = env.reset(seed=7023)
    upstairs = next(t for t in backtrack_info["raw"]["triggers"]
                    if t["msg"] == bridge.WM_DIABPREVLVL)
    bridge.act_walk(upstairs["x"], upstairs["y"])
    backtrack_trace = [bridge.step(ticks=1) for _ in range(120)]
    assert any((r["player_x"], r["player_y"])
               == (upstairs["x"], upstairs["y"]) for r in backtrack_trace)
    assert min(r["dungeon_level"] for r in backtrack_trace) == 1, \
        "地牢上楼/回城触发未被封住，训练轨迹退回 depth=0"

    # fork 会复制已经启动线程的 SDL/network/Lua 内存，却不会复制那些线程。
    # 子进程必须拒绝一切继承状态，并在普通 interpreter exit 时跳过原生析构；
    # 父进程随后仍须可用。
    if hasattr(os, "fork"):
        scratch = pathlib.Path(DiabloGymEnv._engine_config[1])
        assert scratch.is_dir(), "父进程 scratch 在 fork 前已丢失"
        sys.stdout.flush()
        child = os.fork()
        if child == 0:
            try:
                rejected = 0
                for inherited_call in (
                    lambda: env._ensure_active(),
                    lambda: env.reset(seed=1),
                    bridge.observe,
                    bridge.engine_config,
                ):
                    try:
                        inherited_call()
                    except RuntimeError as exc:
                        if "fork" in str(exc):
                            rejected += 1
                if rejected != 4:
                    raise RuntimeError(f"fork 子进程只拒绝了 {rejected}/4 个入口")
                env.close()  # 只清 wrapper，绝不能进入继承的原生析构
            except BaseException as exc:
                print(f"fork child failure: {exc}", file=sys.stderr, flush=True)
                os._exit(3)
            os._exit(0)  # fork child 的唯一安全终点（另一条是 exec）

        deadline = time.monotonic() + 10.0
        status = None
        while time.monotonic() < deadline:
            waited, candidate = os.waitpid(child, os.WNOHANG)
            if waited == child:
                status = candidate
                break
            time.sleep(0.01)
        if status is None:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)
            raise AssertionError("fork 子进程 os._exit 死锁")
        assert os.waitstatus_to_exitcode(status) == 0, status
        assert scratch.is_dir(), "fork child 清除了父进程仍在使用的 scratch"

        # 防御性验证：误用 SystemExit/普通 exit 不能继续跑继承来的 C++ 静态
        # 析构（Lua cross-TU UAF）；原生 atexit 必须 fail-closed 为失败码。
        sys.stdout.flush()
        unsafe_child = os.fork()
        if unsafe_child == 0:
            raise SystemExit(0)
        _, unsafe_status = os.waitpid(unsafe_child, 0)
        assert os.waitstatus_to_exitcode(unsafe_status) == 1, unsafe_status
        assert scratch.is_dir(), "普通退出的 fork child 清除了父 scratch"

        env.reset(seed=7024)
        bridge.step(ticks=1)
        print("PASS: fork 子进程拒绝继承引擎，os._exit/普通退出均不析构父状态")
    env.close()

    short.close()
    try:
        bridge.observe()
    except RuntimeError:
        pass
    else:
        raise AssertionError("end_game 后 observe 未拒绝无效状态")
    print("PASS: 观测隔离、精确截断、原生边界/生命周期守卫成立")

    print("\n== 全部通过:桥、动作、观测、确定性 OK ==")


if __name__ == "__main__":
    main()
