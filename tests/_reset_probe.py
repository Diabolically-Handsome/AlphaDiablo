"""Bisection: do walk commands still work after several resets?"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from diablogym import DiabloGymEnv, bridge

DiabloGymEnv()


def probe_walk(tag):
    obs = bridge.observe()
    sx, sy = obs["player_x"], obs["player_y"]
    bridge.act_walk(sx + 4, sy + 4)
    obs = bridge.step(ticks=60)  # more than enough for a 4-tile diagonal walk
    moved = max(abs(obs["player_x"] - sx), abs(obs["player_y"] - sy))
    print(f"{tag}: walked {moved} tiles from ({sx},{sy}) → ({obs['player_x']},{obs['player_y']})  mode={obs['player_mode']}")
    return moved


print("== Experiment A: repeated resets in town only ==")
bridge.reset(seed=1001)
a1 = probe_walk("A-reset#1")
bridge.reset(seed=1001)
a2 = probe_walk("A-reset#2")
bridge.reset(seed=1001)
a3 = probe_walk("A-reset#3")

print(f"\nConclusion A: reset#1 walked {a1} tiles, #2 walked {a2} tiles, #3 walked {a3} tiles")
if a2 >= 3 and a3 >= 3:
    print("→ repeated town-only resets work; _piece_probe covers the 'reset after entering the dungeon' path")
else:
    raise AssertionError(
        f"the second reset itself is broken: NetInit/teardown re-entrancy problem (moves={a1,a2,a3})")
