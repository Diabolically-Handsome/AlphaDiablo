"""R18-E engagement selection (threat-v1): which monster action 9 should hit.

Chairman's order (2026-09-07 01:05): "择敌要做,优先攻击血量少、威胁大的怪".
Today action 9 binds the wire-canonical first engageable row (nearest, then id).
Under ``engagement_priority="threat-v1"`` the selector ranks the same candidate
set by (ranged first, threat weight, runtime damage, lowest HP, distance, id).
The wire order, the observation and the reward key are untouched; the rule is a
function of the frozen controller snapshot (plus the decision floor: v1 applies on
main L2+ only, so the L1 prefix stays byte-identical to the retreat arm) and the
a9 reward's approach term and the macro always agree on the target. Default
``"off"`` is byte-identical.

AI ids follow DevilutionX ``enum class MonsterAI`` (Source/monstdat.h). The table
is keyed on the runtime ``ai`` field, never on the base type: unique monsters
override ai/hp/damage on the same base type (e.g. Deadeye is a skeleton type
with the goat-archer AI). Unknown ids degrade to weight 1.0.
"""
ENGAGEMENT_PROTOCOLS = ("off", "threat-v1")

# Ranged / caster AIs (kite or shoot; hit them first, they do not come to us).
RANGED_AI = frozenset({3, 7, 9, 14, 16, 19, 20, 25, 35, 36, 37, 38})

# Melee threat weights by AI id on the early floors (1.0 = ordinary zombie).
THREAT_BY_AI = {
    0: 1.0,   # Zombie
    1: 1.4,   # Fat (bloated/overlord family: heavy hits)
    2: 1.2,   # SkeletonMelee
    3: 3.0,   # SkeletonRanged
    4: 1.6,   # Scavenger (fast pack hunters, eat corpses to heal)
    5: 1.8,   # Rhino (charges)
    6: 1.5,   # GoatMelee
    7: 3.0,   # GoatRanged
    8: 0.8,   # Fallen (weak, flee)
    9: 3.0,   # Magma
    10: 2.5,  # SkeletonKing
    11: 1.3,  # Bat (hit and run)
    12: 1.5,  # Gargoyle
    13: 3.0,  # Butcher
    14: 2.5,  # Succubus
    15: 1.5,  # Sneak (hidden)
    16: 2.5,  # Storm
    17: 2.0,  # FireMan
    19: 2.5,  # Acid
    20: 2.5,  # AcidUnique
    25: 2.5,  # Counselor
}


def validate_engagement_priority(value):
    if value not in ENGAGEMENT_PROTOCOLS:
        raise ValueError(
            f"Unknown engagement_priority {value!r}; expected one of {ENGAGEMENT_PROTOCOLS}")
    return value


def threat_weight(ai):
    return THREAT_BY_AI.get(int(ai), 1.0)


def engage_sort_key(candidate, player_x, player_y):
    """Total, deterministic order: ranged first, heavier threat, heavier hitter,
    lowest current HP (finish the nearly dead), then the pre-existing (distance, id)."""
    ai = int(getattr(candidate, "ai", 0))
    distance = max(abs(int(candidate.future_x) - int(player_x)),
                   abs(int(candidate.future_y) - int(player_y)))
    return (0 if ai in RANGED_AI else 1,
            -threat_weight(ai),
            -int(getattr(candidate, "max_damage", 0)),
            int(candidate.hp),
            distance,
            int(candidate.monster_id))


def choose_engage_candidate(snapshot):
    """threat-v1 selector over the frozen snapshot: unblocked candidates first;
    when every candidate is blocked the same pool is ranked (the canonical selector
    takes the first row there), so the blocked-cycle reset in _macro_engage still runs."""
    candidates = list(snapshot.candidates)
    if not candidates:
        return None
    px, py = int(snapshot.player_future_x), int(snapshot.player_future_y)
    unblocked = [c for c in candidates if not c.blocked]
    pool = unblocked or candidates
    return min(pool, key=lambda c: engage_sort_key(c, px, py))
