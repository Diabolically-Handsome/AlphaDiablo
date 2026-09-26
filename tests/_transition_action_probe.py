"""Real-resource regression: an end-of-tick level change is committed immediately, and old-scene actions must not leak."""

from __future__ import annotations

import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from diablogym import DiabloGymEnv, bridge  # noqa: E402
from diablogym.nav import walk_to  # noqa: E402


def descend_without_pending_frame(seed: int):
    """Descend from town; any bridge.step that returns the old PM_NEWLVL scene fails immediately."""
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
        raise AssertionError("level changed unexpectedly early while walking to the tile next to the stairs")

    bridge.act_walk(stair["x"], stair["y"])
    for call_index in range(1, 301):
        raw = bridge.step(ticks=1)
        if raw["player_mode"] == bridge.PM_NEWLVL:
            raise AssertionError(
                f"bridge.step exposed a pending PM_NEWLVL tail frame to Python: {raw}")
        if raw["dungeon_level"] == 1:
            # The depth became 1 within this call: the level change and its reward can be
            # credited by the caller to the action that really stepped on the stairs.
            return raw, call_index
    raise AssertionError("the town stairs did not enter L1 within 300 ticks")


def level_fingerprint(raw):
    return (
        raw["dungeon_level"],
        raw["player_x"], raw["player_y"],
        raw["future_x"], raw["future_y"],
        raw["dest_action"], raw["walkpath0"],
    )


def queue_probe_warp_to_l2():
    """The probe calls StartNewLvl directly, deliberately building a guard window Python normally never sees."""
    bridge.probe_warp_main_level(2)
    pending = bridge.observe()
    if (pending["player_mode"] != bridge.PM_NEWLVL
            or pending["dungeon_level"] != 1):
        raise AssertionError(f"the probe did not build an L1→L2 pending tail frame: {pending}")
    return pending


env = DiabloGymEnv(max_steps=100, include_raw=False)
try:
    # Ordinary Step path: the level change must complete in the same call that triggered it.
    env.reset(seed=81234)
    _, transition_call = descend_without_pending_frame(81234)

    # The first run builds the guard window with the probe, sends no action, and records the first L2 frame.
    queue_probe_warp_to_l2()
    baseline = bridge.step(ticks=1)
    if baseline["dungeon_level"] != 2:
        raise AssertionError(f"the baseline run did not enter L2: {baseline['dungeon_level']}")
    px, py = baseline["player_x"], baseline["player_y"]
    target = next(
        (px + dx, py + dy)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if bridge.probe_tile(px + dx, py + dy)["walkable"]
    )
    expected = level_fingerprint(baseline)

    # The second run calls every training action entry point at the same probe tail frame. They
    # must be no-ops and must not rewrite L2's future/destAction after SyncLoad.
    _, transition_call_again = descend_without_pending_frame(81234)
    pending = queue_probe_warp_to_l2()
    belt_before = pending["belt_heals"]
    if bridge.act_wait() != 0:
        raise AssertionError("act_wait was not refused during the level change")
    bridge.act_walk(*target)
    if bridge.act_explore_walk(
            *target, [], pending["player_x"], pending["player_y"], 12) != 0:
        raise AssertionError("act_explore_walk was not refused during the level change")
    bridge.act_attack_monster(0)
    if bridge.act_controller_attack_monster(
            0, pending["player_x"], pending["player_y"], 12) != 0:
        raise AssertionError("act_controller_attack_monster was not refused during the level change")
    bridge.act_attack_tile(*target)
    bridge.act_operate(*target)
    if bridge.act_controller_operate(
            *target, pending["player_x"], pending["player_y"], 12) != 0:
        raise AssertionError("act_controller_operate was not refused during the level change")
    if bridge.act_drink() != 0:
        raise AssertionError("act_drink was not refused during the level change")
    if bridge.act_pickup() != 0:
        raise AssertionError("act_pickup was not refused during the level change")
    if bridge.act_pickup_gear() != 0:
        raise AssertionError("act_pickup_gear was not refused during the level change")
    if bridge.act_pickup_progression(*target) != 0:
        raise AssertionError("act_pickup_progression was not refused during the level change")
    if bridge.observe()["belt_heals"] != belt_before:
        raise AssertionError("an action during the level change rewrote the old scene's potion state")

    actual = bridge.step(ticks=1)
    if level_fingerprint(actual) != expected:
        raise AssertionError(
            "old-scene action leaked into the first L2 frame:\n"
            f"expected={expected}\nactual={level_fingerprint(actual)}")

    # SyncLoad must commit the local join state at the same time; otherwise the _pLvlChanging
    # guard would also swallow the new level's first legal action.
    bridge.act_walk(*target)
    accepted = bridge.step(ticks=1)
    if (accepted["future_x"], accepted["future_y"]) != target:
        raise AssertionError(
            "the new level's first legal action was swallowed by the _pLvlChanging guard: "
            f"target={target}, raw={accepted}")

    if transition_call_again != transition_call:
        raise AssertionError(
            f"level change call is not deterministic for the same seed: {transition_call} != {transition_call_again}")
    print(
        "PASS: the end-of-tick level change is committed in the same step and PM_NEWLVL is not exposed; "
        "the guard neither leaks old actions nor swallows new ones")
finally:
    env.close()
