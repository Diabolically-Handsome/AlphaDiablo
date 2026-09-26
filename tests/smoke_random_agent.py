"""DiabloGym v0 smoke test: random agent + determinism check.

Chain checked: engine init → reset(seed) → N random steps → the observation changes → the same seed reproduces.
Usage (repository root):  .venv/bin/python tests/smoke_random_agent.py
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
    """Deterministic fingerprint of the observation: player position + position/HP of the first 5 monsters."""
    mons = [(m["id"], m["x"], m["y"], m["hp"]) for m in raw["monsters"][:5]]
    return (raw["player_x"], raw["player_y"], raw["dungeon_level"], tuple(mons))


def main():
    print("== DiabloGym v0 smoke test ==")
    env = DiabloGymEnv(ticks_per_step=4, max_steps=1000)

    # --- 1. reset and the initial observation ---
    obs, info = env.reset(seed=42)
    raw = info["raw"]
    assert info["episode_seed"] == 42
    print(f"reset(seed=42): town position ({raw['player_x']},{raw['player_y']}) "
          f"HP {raw['hp']}/{raw['max_hp']} gold {raw['gold']} "
          f"level {raw['dungeon_level']} monsters {len(raw['monsters'])}")
    assert obs.shape == env.observation_space.shape, "wrong observation vector shape"
    assert obs.dtype == np.float32 and env.observation_space.contains(obs)

    # info is a snapshot owned by the caller; modifying it must not tamper with the environment's internal reward baseline.
    real_x = env._raw["player_x"]
    raw["player_x"] = 999
    raw["monsters"].clear()
    assert env._raw["player_x"] == real_x, "info['raw'] leaked internal mutable state"

    # --- 2. 300 random steps ---
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
                  f"HP {raw['hp']} XP {raw['xp']} level {raw['dungeon_level']}")
        if terminated:
            print(f"  episode terminated at step {step} (dead={raw['dead']})")
            break
    dt = time.time() - t0
    ticks = (step + 1) * env.ticks_per_step
    print(f"{step + 1} random steps ({ticks} tick) took {dt:.2f}s "
          f"≈ {ticks / dt:.0f} tick/s (real time is 20 tick/s, {ticks / dt / 20:.0f}x speed-up)")
    assert len(positions) > 3, f"the player barely moved (only {len(positions)} tiles visited); action injection may be broken"
    print(f"PASS: the player visited {len(positions)} tiles; action injection works")

    # --- 2b. targeted smoke of the macro actions (11 descend / 12 drink / 13 potion pickup / 14 gear pickup): every
    # new engine code path may hide a headless mine (lessons: bat dive/Butcher dialogue), so CI must really step on each ---
    if not terminated:
        for macro in (11, 12, 13, 14):
            obs, reward, terminated, truncated, info = env.step(macro)
            if terminated or truncated:
                break
        print("PASS: targeted smoke of macro actions 11/12/13/14 without crashes")

    # --- 3. determinism: same seed same world, different seed different world ---
    _, info_a = env.reset(seed=123)
    snap_a = snapshot(info_a["raw"])
    _, info_b = env.reset(seed=123)
    snap_b = snapshot(info_b["raw"])
    _, info_c = env.reset(seed=456)
    snap_c = snapshot(info_c["raw"])
    assert snap_a == snap_b, f"initial world differs for the same seed!\n{snap_a}\n{snap_b}"
    print("PASS: two resets with seed=123 give the same initial world (determinism holds)")
    if snap_a == snap_c:
        print("WARN: seed=123 and seed=456 give the same initial world (normal: the town layout is fixed; worlds diverge after entering the dungeon)")
    else:
        print("PASS: different seeds give different initial worlds")

    # --- 4. Gym/native boundaries and exact truncation ---
    short = DiabloGymEnv(ticks_per_step=4, max_steps=1, include_raw=False)
    short.reset(seed=7)
    try:
        env.step(0)
    except RuntimeError as exc:
        assert "interleaved" in str(exc)
    else:
        raise AssertionError("the in-process global engine was silently interleaved by several environments")
    _, _, terminated, truncated, cap_info = short.step(10)
    # A macro of up to 12 ticks can only use the 1 remaining tick. If that tick stops in a walk/future/mode
    # intermediate state not encoded in the 295 dims, it must be a fail-closed terminal; SB3 must not
    # bootstrap the aliased terminal_observation as a safe TimeLimit state.
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
        raise AssertionError("step still possible after the episode was truncated")
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
            raise AssertionError("the native boundary did not refuse invalid arguments")

    # Ordinary step now settles the end-of-tick level-change event before returning, so the
    # probe queues StartNewLvl directly to keep verifying that "reset right after a pending
    # event" does not leak the previous game's level change to the new hero.
    bridge.reset(seed=81)
    bridge.probe_warp_main_level(1)
    pending = bridge.observe()
    assert (pending["player_mode"] == bridge.PM_NEWLVL
            and pending["dungeon_level"] == 0), "the probe did not queue a level-change event"
    bridge.reset(seed=82)
    assert bridge.step(ticks=1)["dungeon_level"] == 0, "the previous game's level-change event leaked into the new game"

    # The task only allows progressing downward. Historically FARM's explore on seed 7023
    # stepped on the L1 up-stairs by mistake and returned to town, turning the rest of the game into wasted depth=0 samples.
    env.start_in_dungeon = True
    _, backtrack_info = env.reset(seed=7023)
    upstairs = next(t for t in backtrack_info["raw"]["triggers"]
                    if t["msg"] == bridge.WM_DIABPREVLVL)
    bridge.act_walk(upstairs["x"], upstairs["y"])
    backtrack_trace = [bridge.step(ticks=1) for _ in range(120)]
    assert any((r["player_x"], r["player_y"])
               == (upstairs["x"], upstairs["y"]) for r in backtrack_trace)
    assert min(r["dungeon_level"] for r in backtrack_trace) == 1, \
        "dungeon up-stairs/town-return trigger not sealed; training trajectory fell back to depth=0"

    # fork copies the memory of SDL/network/Lua threads already started, but not the threads themselves.
    # The child must refuse all inherited state and skip native destructors on a normal interpreter exit;
    # the parent must stay usable afterwards.
    if hasattr(os, "fork"):
        scratch = pathlib.Path(DiabloGymEnv._engine_config[1])
        assert scratch.is_dir(), "the parent's scratch was already lost before fork"
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
                    raise RuntimeError(f"the fork child refused only {rejected}/4 entry points")
                env.close()  # only clears the wrapper; must never enter the inherited native destructors
            except BaseException as exc:
                print(f"fork child failure: {exc}", file=sys.stderr, flush=True)
                os._exit(3)
            os._exit(0)  # the only safe end point of a fork child (the other is exec)

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
            raise AssertionError("fork child deadlocked in os._exit")
        assert os.waitstatus_to_exitcode(status) == 0, status
        assert scratch.is_dir(), "the fork child removed the scratch still used by the parent"

        # Defensive check: a mistaken SystemExit/normal exit must not go on to run the inherited C++ static
        # destructors (Lua cross-TU UAF); the native atexit must fail closed with a failure code.
        sys.stdout.flush()
        unsafe_child = os.fork()
        if unsafe_child == 0:
            raise SystemExit(0)
        _, unsafe_status = os.waitpid(unsafe_child, 0)
        assert os.waitstatus_to_exitcode(unsafe_status) == 1, unsafe_status
        assert scratch.is_dir(), "a normally exiting fork child removed the parent's scratch"

        env.reset(seed=7024)
        bridge.step(ticks=1)
        print("PASS: the fork child refuses the inherited engine; os._exit/normal exit destroy no parent state")
    env.close()

    short.close()
    try:
        bridge.observe()
    except RuntimeError:
        pass
    else:
        raise AssertionError("observe did not refuse the invalid state after end_game")
    print("PASS: observation isolation, exact truncation, native boundary/lifecycle guards hold")

    print("\n== All passed: bridge, actions, observation, determinism OK ==")


if __name__ == "__main__":
    main()
