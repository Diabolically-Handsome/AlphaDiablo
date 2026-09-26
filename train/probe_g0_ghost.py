"""G0-ghost regression probe for the v12 defect (PREREG-v32 W-G0; run after the E1 work).

Assertions:
  A. Autonomy on: m[12] passes through the base environment mask (legal when belt>0); pressing 12 drinks normally
     (belt decreases, beats booked as usual), and after drinking down to belt=0 m[12] turns illegal (guards against a ghost empty drink).
  B. Autonomy off (control-leg knob): m[12] is always False (the old protocol reproduced verbatim).
  C. Reflex predicate verbatim: the _reflex semantics (hp<0.5 and belt>0) were not touched by E1 (synthetic raw
     unit assertions; drain behavior on the real distribution is backstopped by the bit-level wage hash of G0-identity:
     the actions/wages of drain beats are all inside the hash).
  D. Termination discipline: a step after the window has ended must be refused (the wrapper layer guarantees no drinking after death).
Exit code 0 = everything passed.
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym.options_env import OptionsEnv, _reflex  # noqa: E402
from diablogym.worker_env import WorkerWindowEnv       # noqa: E402

H_NPZ = str(ROOT / "train" / "models" / "v22-h-manager" / "policy.npz")


def check(cond, msg):
    if not cond:
        print(f"G0-ghost FAIL: {msg}")
        sys.exit(1)
    print(f"  ok: {msg}")


def main():
    # C. Reflex predicate unit assertions (semantics verbatim before and after E1)
    check(_reflex({"hp": 49, "max_hp": 100, "belt_heals": 1}) is True,
          "reflex predicate: below half HP with a potion -> True")
    check(_reflex({"hp": 50, "max_hp": 100, "belt_heals": 3}) is False,
          "reflex predicate: exactly half HP -> False (threshold semantics unchanged)")
    check(_reflex({"hp": 10, "max_hp": 100, "belt_heals": 0}) is False,
          "reflex predicate: no potion -> False")

    # B. Autonomy off: the old protocol reproduced verbatim
    env_off = OptionsEnv(max_steps=3000, drink_sovereignty=False)
    env_off.reset(seed=424242)
    m_off = env_off._worker_masks()
    check(bool(m_off[12]) is False, "autonomy off: m[12] always False")
    check(bool(m_off[11]) is False, "autonomy off: m[11] always False (authority unchanged)")
    env_off.close()

    # A/D. Autonomy on: walk one live window
    env = WorkerWindowEnv(H_NPZ, max_steps=3000, rng_seed=0,
                          seed_scope="replay",
                          drink_sovereignty=True)
    obs, _ = env.reset(seed=424242)
    check(obs is not None, "autonomy on: reset returns the first window observation")
    raw = env.oe.env._raw
    m = env.oe._worker_masks()
    check(bool(m[11]) is False, "autonomy on: m[11] still always masked (DIVE belongs to the manager)")
    belt0 = raw.get("belt_heals", 0)
    check(bool(m[12]) == (belt0 > 0),
          f"autonomy on: m[12] is gated on belt (belt={belt0} -> {bool(m[12])})")
    drinks = 0
    reason_ended = False
    # First get a potion: pick one up within a 600-beat budget (press 13 when legal, otherwise clear monsters/explore), then test drinking
    for _ in range(600):
        raw = env.oe.env._raw
        if raw.get("belt_heals", 0) > 0:
            break
        m = env.oe._worker_masks()
        a = 13 if m[13] else (9 if m[9] else 10)
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            nxt = env.next_window()
            if nxt is None:
                reason_ended = True
                break
    for _ in range(40):                       # Drink while there are potions, until the belt is empty
        if reason_ended:
            break
        m = env.oe._worker_masks()
        raw = env.oe.env._raw
        belt_before = raw.get("belt_heals", 0)
        if not m[12]:
            check(belt_before == 0,
                  f"m[12] illegal <=> belt=0 (belt={belt_before})")
            break
        beats_b = env.oe._win["beats"]
        ov_b = env.oe._win["overrides"]
        steps_b = env.oe.env._steps
        obs2, w, term, trunc, info = env.step(12)
        raw2 = env.oe.env._raw
        check(raw2.get("belt_heals", 0) < belt_before,
              f"active drink: belt {belt_before}->{raw2.get('belt_heals', 0)} strictly decreasing")
        check(np.isfinite(w), "active drink: wage beat is finite")
        check(env.oe.env._steps >= steps_b + 1, "active drink: micro-steps advance (clock semantics)")
        if not (term or trunc):
            check(env.oe._win["beats"] >= beats_b + 1, "active drink: beat ledger increments")
            check(env.oe._win["overrides"] == ov_b, "active drink: no fuse-override beat")
        drinks += 1
        if term or trunc:
            reason_ended = True
            break
    check(drinks > 0, "a real drink happened (fail closed: idling is a FAIL, the v30 empty-probe discipline)")
    print(f"  active drinks: {drinks} beats (belt emptied or the window ended naturally)")
    if not reason_ended:
        # D. Termination discipline: after forcing the window to its end, a step must be refused
        for _ in range(5000):
            m = env.oe._worker_masks()
            a = 9 if m[9] else int(np.flatnonzero(m)[0])
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                reason_ended = True
                break
    check(reason_ended, "the window can end naturally")
    nxt = env.next_window()
    while nxt is not None:                    # Drive unconditionally to the end of the episode, then assert the refusal
        m = env.oe._worker_masks()
        a = 9 if m[9] else int(np.flatnonzero(m)[0])
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            nxt = env.next_window()
    try:
        env.step(9)
        check(False, "a step after the episode ended should be refused")
    except Exception:
        check(True, "a step after episode end/death is refused (the wrapper layer guarantees no drinking)")
    env.close()

    # E. Bridge proof 3, live: script mode with autonomy on/off is bit-equivalent + dispatch never returns 12
    from diablogym import options_env as oe_mod
    orig = oe_mod.dispatch
    seen12 = {"n": 0}

    def spy(mode, raw, gear):
        a = orig(mode, raw, gear)
        if a == 12:
            seen12["n"] += 1
        return a

    traces = {}
    for sv in (True, False):
        oe_mod.dispatch = spy
        try:
            e = oe_mod.OptionsEnv(max_steps=3000, drink_sovereignty=sv)
            e.reset(seed=424242)
            tr = []
            done = trunc = False
            while not (done or trunc):
                _, r, done, trunc, info = e.step(0)   # FARM fallback slot
                ex = info["option_extra"]
                tr.append((ex["tau"], round(ex["R"], 6), round(ex["W"], 6),
                           ex["beats"], ex["overrides"], ex["reason"]))
            traces[sv] = tr
            e.close()
        finally:
            oe_mod.dispatch = orig
    check(traces[True] == traces[False],
          "assertion E: in script mode, autonomy on/off are bit-equal window by window over the whole episode (dispatch does not consume the mask)")
    check(seen12["n"] == 0,
          "assertion E: dispatch never returned 12 (the reflex drinks first; dynamic proof that branch 12 is unreachable)")

    print("G0-ghost PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
