"""R18-D aggro cap (hold-v1): stop pulling new monsters while a crowd is on us.

Chairman's order (2026-09-07 01:05): "拿到 3-5 只怪的仇恨之后,就在原地处理完再打别的".
Data (T0-double-prime): at the moment the retreat law fired, 7.5 alive monsters
stood within 6 tiles on average.

Rule (env.controller_action_context, after the a9 mask): on main L2+ (v1 scope,
so the L1 prefix stays byte-identical to the retreat arm), when the flag is on,
action 9 is executable (there is something to fight) and at least ``count``
visible alive monsters stand within ``radius`` tiles, action 10 (explore /
global hunt = the only pull) is masked. The ``mask[9]`` guard keeps the rule
from ever masking the last useful action (no deadlock into the exhausted
stall). Default ``"off"`` is byte-identical.
"""
from dataclasses import dataclass

AGGRO_CAP_PROTOCOLS = ("off", "hold-v1")


@dataclass(frozen=True)
class AggroCapPolicy:
    count: int = 5     # chairman: 3-5; T0-double-prime mean at retreat trigger 7.5
    radius: int = 6

    def __post_init__(self):
        if type(self.count) is not int or self.count < 1:
            raise ValueError("count must be a positive int")
        if type(self.radius) is not int or self.radius < 1:
            raise ValueError("radius must be a positive int")

    def as_dict(self):
        return {"count": self.count, "radius": self.radius}


def validate_aggro_cap(value):
    if value not in AGGRO_CAP_PROTOCOLS:
        raise ValueError(f"Unknown aggro_cap {value!r}; expected one of {AGGRO_CAP_PROTOCOLS}")
    return value


def visible_alive_monsters_within(raw, radius):
    """Visible, alive, non-golem monsters within Chebyshev ``radius`` of the player.
    (Unlike resource_retreat.alive_monsters_within this applies the visibility
    filter, so the count matches what action 9 can see.)"""
    px, py = int(raw["player_x"]), int(raw["player_y"])
    count = 0
    for monster in raw.get("monsters", ()):
        if int(monster.get("hp", 0)) <= 0 or int(monster.get("type", -1)) == 109:
            continue
        if bool(monster.get("is_invalid", False)) or not bool(monster.get("visible", True)):
            continue
        if "x" not in monster or "y" not in monster:
            continue
        if max(abs(int(monster["x"]) - px), abs(int(monster["y"]) - py)) <= radius:
            count += 1
    return count


def cap_fires(raw, policy, mask9):
    return bool(mask9) and visible_alive_monsters_within(raw, policy.radius) >= policy.count
