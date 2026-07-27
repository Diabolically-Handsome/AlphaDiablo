"""真实资源回归：a14 严格升级、自动鉴定且不写入隐藏背包。"""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import DiabloGymEnv, bridge  # noqa: E402


def stand_on(spawned):
    raw = bridge.observe()
    target = (int(spawned["x"]), int(spawned["y"]))
    if (
        (int(raw["player_x"]), int(raw["player_y"])) == target
        and (int(raw["future_x"]), int(raw["future_y"])) == target
        and int(raw["player_mode"]) == int(bridge.PM_STAND)
    ):
        return
    bridge.act_walk(*target)
    for _ in range(128):
        raw = bridge.step(ticks=1)
        if (
            (int(raw["player_x"]), int(raw["player_y"])) == target
            and (int(raw["future_x"]), int(raw["future_y"])) == target
            and int(raw["player_mode"]) == int(bridge.PM_STAND)
        ):
            return
    raise AssertionError(("failed to stand exactly on test gear", target, raw))


def pickup_gear(spawned):
    stand_on(spawned)
    return bridge.act_pickup_gear_at(
        spawned["active_id"],
        spawned["x"],
        spawned["y"],
        spawned["seed_hi"],
        spawned["seed_lo"],
        spawned["create_info"],
        spawned["base_id"],
    )


env = DiabloGymEnv(start_in_dungeon=True, max_steps=100, include_raw=True)
try:
    # Developer-only ordinary seeds: outside every train/eval/BC reserved pool.
    env.reset(seed=424242)
    before = bridge.observe()
    sword = next(
        item for item in before["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )
    sword_base_id = int(sword["base_id"])
    utility_before = int(before["gear_combat_utility"])
    inventory_before = bridge.probe_inventory_item_count()

    weak = bridge.probe_spawn_test_gear(
        sword_base_id, 1, 1, 0, 0)
    weak_raw = next(
        item for item in bridge.observe()["floor_items"]
        if item["active_id"] == weak["active_id"]
    )
    assert not weak_raw["gear"], weak_raw
    assert pickup_gear(weak) == 0
    assert int(bridge.observe()["gear_combat_utility"]) == utility_before

    strong = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30, 0, 40)
    strong_raw = next(
        item for item in bridge.observe()["floor_items"]
        if item["active_id"] == strong["active_id"]
    )
    assert strong_raw["gear"], strong_raw
    assert not strong_raw["identified"], strong_raw
    assert strong_raw["effect_damage"] == 40, strong_raw
    assert int(strong_raw["combat_utility"]) > int(sword["combat_utility"])

    # Adjacent is no longer enough: navigation must finish on the exact item
    # tile before native commit can consume it.
    assert bridge.act_pickup_gear_at(
        strong["active_id"],
        strong["x"],
        strong["y"],
        strong["seed_hi"],
        strong["seed_lo"],
        strong["create_info"],
        strong["base_id"],
    ) == 0
    assert bridge.act_pickup_gear() == 0
    stand_on(strong)
    # active item slots are reusable.  A stale snapshot with the same id and
    # coordinates but a different stable identity must fail before planning.
    assert bridge.act_pickup_gear_at(
        strong["active_id"],
        strong["x"],
        strong["y"],
        int(strong["seed_hi"]) ^ 1,
        strong["seed_lo"],
        strong["create_info"],
        strong["base_id"],
    ) == 0
    assert any(
        item["active_id"] == strong["active_id"]
        for item in bridge.observe()["floor_items"]
    )
    assert pickup_gear(strong) == 1
    after = bridge.observe()
    assert int(after["gear_combat_utility"]) > utility_before, after
    equipped_weapon = next(
        item for item in after["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )
    assert equipped_weapon["identified"], equipped_weapon
    assert equipped_weapon["effects_active"], equipped_weapon
    assert equipped_weapon["effect_damage"] == 40, equipped_weapon
    assert bridge.probe_inventory_item_count() == inventory_before
    assert all(
        item["active_id"] != strong["active_id"]
        for item in after["floor_items"]
    )
    assert any(
        item["active_id"] == weak["active_id"]
        for item in after["floor_items"]
    )

    quest = bridge.probe_spawn_test_gear(
        bridge.IDI_LAZSTAFF, 100, 120, 0, 100)
    quest_raw = next(
        item for item in bridge.observe()["floor_items"]
        if item["active_id"] == quest["active_id"]
    )
    assert not quest_raw["gear"], quest_raw
    assert pickup_gear(quest) == 0

    # Fresh state for adversarial whole-loadout semantics.  The old per-item
    # sum accepted every rejected candidate below.
    env.reset(seed=424243)
    before = bridge.observe()
    sword = next(
        item for item in before["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )
    sword_base_id = int(sword["base_id"])

    def floor_item(spawned):
        return next(
            item for item in bridge.observe()["floor_items"]
            if item["active_id"] == spawned["active_id"]
        )

    fastest = int(bridge.ITEM_EFFECT_FASTEST_ATTACK)
    speed_anchor = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30, 0, 0, fastest, 75, 75, 75)
    assert floor_item(speed_anchor)["gear"], floor_item(speed_anchor)
    assert pickup_gear(speed_anchor) == 1
    anchor_profile = dict(bridge.probe_gear_combat_profile())
    assert anchor_profile["attack_speed_tier"] == 4, anchor_profile
    assert (
        anchor_profile["fire_resist"],
        anchor_profile["lightning_resist"],
        anchor_profile["magic_resist"],
    ) == (75, 75, 75), anchor_profile
    anchor_equipped = next(
        item for item in bridge.observe()["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )

    def assert_reject(spawned, reason):
        candidate = floor_item(spawned)
        assert not candidate["gear"], (reason, candidate)
        assert pickup_gear(spawned) == 0
        after_profile = dict(bridge.probe_gear_combat_profile())
        assert after_profile == anchor_profile, (
            reason, anchor_profile, after_profile)
        return candidate

    # Counterexample 1: +1 panel damage used to outweigh a Fastest bit because
    # all four speed tiers were each just +192 in the per-item sum.
    speed_loss = bridge.probe_spawn_test_gear(
        sword_base_id, 21, 30, 0, 0, 0, 75, 75, 75)
    speed_loss_raw = assert_reject(speed_loss, "attack-speed downgrade")
    # ``combat_utility`` on one item is only a compact actor/debug descriptor;
    # it cannot know whole-player damage modifiers or animation metadata.
    # The authoritative ``gear`` bit and commit gate consume the simulated
    # whole-loadout profile above.
    assert int(speed_loss_raw["effect_flags"]) == 0, speed_loss_raw
    assert int(anchor_equipped["effect_flags"]) & fastest, anchor_equipped

    # Counterexample 2: ZeroResistance is an OR-ed global curse applied after
    # resistance aggregation; it clears all three already-capped 75 values.
    zero_res = bridge.probe_spawn_test_gear(
        sword_base_id, 30, 40, 0, 0,
        fastest | int(bridge.ITEM_EFFECT_ZERO_RESISTANCE),
        75, 75, 75)
    assert_reject(zero_res, "ZeroResistance")

    # No installed action casts, consumes mana or shoots arrows.  These raw
    # affixes remain observable but must neither create gear utility nor cross
    # the pickup gate on an otherwise identical weapon.
    no_op_only_flags = (
        int(bridge.ITEM_EFFECT_NO_MANA)
        | int(bridge.ITEM_EFFECT_STEAL_MANA5)
        | int(bridge.ITEM_EFFECT_RANDOM_ARROW_VELOCITY)
        | int(bridge.ITEM_EFFECT_FIRE_ARROWS)
        | int(bridge.ITEM_EFFECT_MULTIPLE_ARROWS)
        | int(bridge.ITEM_EFFECT_LIGHTNING_ARROWS)
    )
    no_op_flags = fastest | no_op_only_flags
    no_op = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30, 0, 0, no_op_flags, 75, 75, 75,
        magic_bonus=50, mana_bonus_points=50, spell_level_bonus=3)
    no_op_raw = assert_reject(no_op, "spell/mana/arrow no-op affixes")
    assert int(no_op_raw["combat_utility"]) == int(
        anchor_equipped["combat_utility"]), (no_op_raw, anchor_equipped)

    # Knockback executes, but is not a monotonic upgrade for an action set that
    # must walk back into melee after pushing the target away.  Keep it visible
    # for the policy, but never accept an otherwise identical item solely for it.
    knockback = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30, 0, 0,
        fastest | int(bridge.ITEM_EFFECT_KNOCKBACK),
        75, 75, 75)
    knockback_raw = assert_reject(knockback, "situational Knockback")
    assert int(knockback_raw["combat_utility"]) == int(
        anchor_equipped["combat_utility"]), (knockback_raw, anchor_equipped)

    drain_life = bridge.probe_spawn_test_gear(
        sword_base_id, 100, 120, 0, 0,
        fastest | int(bridge.ITEM_EFFECT_DRAIN_LIFE),
        75, 75, 75)
    assert_reject(drain_life, "DrainLife")

    # Hellfire's Decay removes the weapon over repeated hits, Peril applies
    # self-damage, and Doppelganger can clone the struck monster.  A huge paper
    # damage number must not introduce any of these real combat curses.
    for name, dam_ac_flag in (
        ("Decay", bridge.ITEM_DAM_AC_DECAY),
        ("Peril", bridge.ITEM_DAM_AC_PERIL),
        ("Doppelganger", bridge.ITEM_DAM_AC_DOPPELGANGER),
    ):
        cursed = bridge.probe_spawn_test_gear(
            sword_base_id, 100, 120, 0, 0, fastest, 75, 75, 75,
            dam_ac_flags=int(dam_ac_flag))
        assert_reject(cursed, name)

    # Redundant tier bits and resistance above the 75 cap are not counted as
    # extra whole-loadout power (the old item sum counted both repeatedly).
    duplicate_speed = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30, 0, 0,
        fastest | int(bridge.ITEM_EFFECT_QUICK_ATTACK),
        75, 75, 75)
    duplicate_speed_raw = assert_reject(
        duplicate_speed, "redundant attack-speed bit")
    assert int(duplicate_speed_raw["combat_utility"]) == int(
        anchor_equipped["combat_utility"])

    over_cap = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30, 0, 0, fastest, 100, 100, 100)
    over_cap_raw = assert_reject(over_cap, "redundant over-cap resistance")
    assert int(over_cap_raw["combat_utility"]) > int(
        anchor_equipped["combat_utility"])

    # Rejected candidates deliberately remain on the floor.  Reset onto a
    # third ordinary developer seed so the positive control is adjacent rather
    # than consuming a radius-2 spill slot.
    env.reset(seed=424244)
    fresh = bridge.observe()
    sword_base_id = int(next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )["base_id"])
    speed_anchor = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30, 0, 0, fastest, 75, 75, 75)
    assert pickup_gear(speed_anchor) == 1
    anchor_profile = dict(bridge.probe_gear_combat_profile())

    # A true Pareto improvement still commits and the same whole-loadout score
    # is visible through Observe and the native commit gate.
    pareto = bridge.probe_spawn_test_gear(
        sword_base_id, 21, 31, 0, 0, fastest, 75, 75, 75)
    assert floor_item(pareto)["gear"], floor_item(pareto)
    assert pickup_gear(pareto) == 1
    pareto_profile = dict(bridge.probe_gear_combat_profile())
    assert pareto_profile["utility"] > anchor_profile["utility"], (
        anchor_profile, pareto_profile)
    assert pareto_profile["physical_min"] > anchor_profile["physical_min"]
    assert pareto_profile["physical_max"] > anchor_profile["physical_max"]
    assert int(bridge.observe()["gear_combat_utility"]) == int(
        pareto_profile["utility"])

    # There is no stochastic/float noise in the native integer simulation.
    # A real +1 resistance gain is only +512 in this ledger and used to be
    # discarded by the arbitrary 1024 minimum-gain threshold.
    env.reset(seed=424253)
    fresh = bridge.observe()
    starter_sword = next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )
    small_before = dict(bridge.probe_gear_combat_profile())
    small_resist = bridge.probe_spawn_test_gear(
        int(starter_sword["base_id"]),
        int(starter_sword["min_damage"]),
        int(starter_sword["max_damage"]),
        fire_resistance=1)
    assert floor_item(small_resist)["gear"], floor_item(small_resist)
    assert pickup_gear(small_resist) == 1
    small_after = dict(bridge.probe_gear_combat_profile())
    assert small_after["fire_resist"] == small_before["fire_resist"] + 1
    assert (
        small_after["utility"] - small_before["utility"]
    ) == 512, (small_before, small_after)

    # Light radius controls IsTileLit, hence which monsters/items enter the
    # action graph.  It is real task utility rather than cosmetic metadata.
    env.reset(seed=424254)
    fresh = bridge.observe()
    starter_sword = next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )
    light_before = dict(bridge.probe_gear_combat_profile())
    radiance = bridge.probe_spawn_test_gear(
        int(starter_sword["base_id"]),
        int(starter_sword["min_damage"]),
        int(starter_sword["max_damage"]),
        light_bonus=1)
    assert floor_item(radiance)["gear"], floor_item(radiance)
    assert pickup_gear(radiance) == 1
    light_after = dict(bridge.probe_gear_combat_profile())
    assert light_after["light_radius"] == light_before["light_radius"] + 1
    assert light_after["utility"] > light_before["utility"], (
        light_before, light_after)

    # NoMana/mana-steal/arrow/spell/magic affixes are unpriced, not forbidden:
    # enough genuine melee damage may still make the replacement an upgrade.
    env.reset(seed=424255)
    fresh = bridge.observe()
    sword_base_id = int(next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )["base_id"])
    no_op_before = dict(bridge.probe_gear_combat_profile())
    no_op_damage = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30, effect_flags=no_op_only_flags,
        magic_bonus=50, mana_bonus_points=50, spell_level_bonus=3)
    assert floor_item(no_op_damage)["gear"], floor_item(no_op_damage)
    assert pickup_gear(no_op_damage) == 1
    no_op_after = dict(bridge.probe_gear_combat_profile())
    assert no_op_after["physical_max"] > no_op_before["physical_max"]
    assert no_op_after["magic"] > no_op_before["magic"]
    assert no_op_after["max_mana_fixed"] > no_op_before["max_mana_fixed"]
    assert no_op_after["mana_steal_tier"] == 2
    assert no_op_after["utility"] > no_op_before["utility"]

    # The disaster gates must not regress into a full Pareto gate.  A large
    # real damage gain may safely trade two points of hit chance while attack
    # timing, curses, block and capped resistances remain intact.
    env.reset(seed=424247)
    fresh = bridge.observe()
    sword_base_id = int(next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )["base_id"])
    hit_anchor = bridge.probe_spawn_test_gear(
        sword_base_id, 6, 10, to_hit_bonus=10)
    assert pickup_gear(hit_anchor) == 1
    hit_anchor_profile = dict(bridge.probe_gear_combat_profile())
    damage_trade = bridge.probe_spawn_test_gear(
        sword_base_id, 10, 16, to_hit_bonus=8)
    assert floor_item(damage_trade)["gear"], floor_item(damage_trade)
    assert pickup_gear(damage_trade) == 1
    damage_trade_profile = dict(bridge.probe_gear_combat_profile())
    assert (
        damage_trade_profile["melee_piercing_to_hit"]
        < hit_anchor_profile["melee_piercing_to_hit"]
    ), (hit_anchor_profile, damage_trade_profile)
    assert (
        damage_trade_profile["utility"]
        >= hit_anchor_profile["utility"] + 4096
    ), (hit_anchor_profile, damage_trade_profile)

    # The old field-by-field gates also locked +damage-taken, capped resistance,
    # hit recovery and life steal forever.  Each is now a priced scalar loss:
    # a sufficiently large genuine damage gain can trade it, while the lethal
    # curse/current-HP gates above remain non-negotiable.
    env.reset(seed=424256)
    fresh = bridge.observe()
    sword_base_id = int(next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )["base_id"])
    get_hit_anchor = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30)
    assert pickup_gear(get_hit_anchor) == 1
    get_hit_before = dict(bridge.probe_gear_combat_profile())
    get_hit_trade = bridge.probe_spawn_test_gear(
        sword_base_id, 30, 40, get_hit_penalty=1)
    assert floor_item(get_hit_trade)["gear"], floor_item(get_hit_trade)
    assert pickup_gear(get_hit_trade) == 1
    get_hit_after = dict(bridge.probe_gear_combat_profile())
    assert get_hit_after["get_hit"] > get_hit_before["get_hit"]
    assert get_hit_after["utility"] > get_hit_before["utility"]

    env.reset(seed=424257)
    fresh = bridge.observe()
    sword_base_id = int(next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )["base_id"])
    resist_anchor = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30, fire_resistance=75,
        lightning_resistance=75, magic_resistance=75)
    assert pickup_gear(resist_anchor) == 1
    resist_before = dict(bridge.probe_gear_combat_profile())
    resist_trade = bridge.probe_spawn_test_gear(
        sword_base_id, 30, 40, fire_resistance=74,
        lightning_resistance=74, magic_resistance=74)
    assert floor_item(resist_trade)["gear"], floor_item(resist_trade)
    assert pickup_gear(resist_trade) == 1
    resist_after = dict(bridge.probe_gear_combat_profile())
    assert resist_before["fire_resist"] == 75, resist_before
    assert resist_after["fire_resist"] == 74, resist_after
    assert resist_after["utility"] > resist_before["utility"]

    env.reset(seed=424258)
    fresh = bridge.observe()
    sword_base_id = int(next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )["base_id"])
    recovery_life_flags = (
        int(bridge.ITEM_EFFECT_FASTEST_HIT_RECOVERY)
        | int(bridge.ITEM_EFFECT_STEAL_LIFE5)
    )
    recovery_anchor = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30, effect_flags=recovery_life_flags)
    assert pickup_gear(recovery_anchor) == 1
    recovery_before = dict(bridge.probe_gear_combat_profile())
    recovery_trade = bridge.probe_spawn_test_gear(
        sword_base_id, 40, 50)
    assert floor_item(recovery_trade)["gear"], floor_item(recovery_trade)
    assert pickup_gear(recovery_trade) == 1
    recovery_after = dict(bridge.probe_gear_combat_profile())
    assert recovery_before["hit_recovery_tier"] == 3, recovery_before
    assert recovery_before["life_steal_tier"] == 2, recovery_before
    assert recovery_after["hit_recovery_tier"] == 0, recovery_after
    assert recovery_after["life_steal_tier"] == 0, recovery_after
    assert recovery_after["utility"] > recovery_before["utility"]

    # Life steal is not a flag bounty: DevilutionX heals a percentage of the
    # physical damage of each successful melee hit.  These developer-only
    # seeds are outside every train/eval/BC pool and lock the whole-loadout
    # score to that physical hit-throughput contract.
    life3 = int(bridge.ITEM_EFFECT_STEAL_LIFE3)
    life5 = int(bridge.ITEM_EFFECT_STEAL_LIFE5)
    random_life = int(bridge.ITEM_EFFECT_RANDOM_STEAL_LIFE)

    def life_profile(seed, min_damage, max_damage, flags):
        env.reset(seed=seed)
        fresh = bridge.observe()
        base_id = int(next(
            item for item in fresh["equipped_items"]
            if item["present"] and item["item_class"] == 1
        )["base_id"])
        spawned = bridge.probe_spawn_test_gear(
            base_id, min_damage, max_damage, effect_flags=flags)
        raw = floor_item(spawned)
        assert raw["gear"], (seed, min_damage, max_damage, flags, raw)
        local_utility = int(raw["combat_utility"])
        assert pickup_gear(spawned) == 1
        return local_utility, dict(bridge.probe_gear_combat_profile())

    plain_local, plain_life_profile = life_profile(
        424263, 20, 30, 0)
    life3_local, life3_profile = life_profile(
        424263, 20, 30, life3)
    life5_local, life5_profile = life_profile(
        424263, 20, 30, life5)
    random_local, random_life_profile = life_profile(
        424263, 20, 30, random_life)
    stacked_local, stacked_life_profile = life_profile(
        424263, 20, 30, life5 | random_life)
    assert (
        stacked_life_profile["utility"]
        > random_life_profile["utility"]
        > life5_profile["utility"]
        > life3_profile["utility"]
        > plain_life_profile["utility"]
    ), (
        plain_life_profile, life3_profile, life5_profile,
        random_life_profile, stacked_life_profile)
    # The compact actor/debug hint is non-authoritative, but it must not teach
    # the opposite ordering from the native whole-loadout gate.
    assert (
        stacked_local > random_local > life5_local > life3_local
        > plain_local
    ), (
        plain_local, life3_local, life5_local, random_local, stacked_local)
    assert life3_profile["life_steal_tier"] == 1, life3_profile
    assert life5_profile["life_steal_tier"] == 2, life5_profile
    assert random_life_profile["life_steal_tier"] == 0, (
        random_life_profile)
    assert (
        stacked_life_profile["effect_flags"] & (life5 | random_life)
    ) == (life5 | random_life), stacked_life_profile

    # The same life-steal percentage must become more valuable when physical
    # damage per hit rises.  This rejects the former constant 8192/16384/2048
    # implementation for both fixed and random steal, while identical hit
    # chance and animation timing isolate the physical-throughput dependency.
    _, low_plain = life_profile(424264, 10, 10, 0)
    _, low_fixed = life_profile(424264, 10, 10, life5)
    _, low_random = life_profile(424264, 10, 10, random_life)
    _, high_plain = life_profile(424264, 100, 100, 0)
    _, high_fixed = life_profile(424264, 100, 100, life5)
    _, high_random = life_profile(424264, 100, 100, random_life)
    low_fixed_gain = low_fixed["utility"] - low_plain["utility"]
    high_fixed_gain = high_fixed["utility"] - high_plain["utility"]
    low_random_gain = low_random["utility"] - low_plain["utility"]
    high_random_gain = high_random["utility"] - high_plain["utility"]
    assert 0 < low_fixed_gain < high_fixed_gain, (
        low_plain, low_fixed, high_plain, high_fixed)
    assert 0 < low_random_gain < high_random_gain, (
        low_plain, low_random, high_plain, high_random)

    # Regression for the concrete strict reversal found during audit.  The
    # candidate has both higher physical damage and the stronger expected
    # random steal (~6.25% versus 5%), so action 14 must not retain the weaker
    # fixed-steal weapon.
    env.reset(seed=424265)
    fresh = bridge.observe()
    sword_base_id = int(next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )["base_id"])
    fixed_anchor = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30, effect_flags=life5)
    assert pickup_gear(fixed_anchor) == 1
    fixed_anchor_profile = dict(bridge.probe_gear_combat_profile())
    stronger_random = bridge.probe_spawn_test_gear(
        sword_base_id, 21, 30, effect_flags=random_life)
    assert floor_item(stronger_random)["gear"], floor_item(stronger_random)
    assert pickup_gear(stronger_random) == 1
    stronger_random_profile = dict(bridge.probe_gear_combat_profile())
    assert (
        stronger_random_profile["physical_min"]
        > fixed_anchor_profile["physical_min"]
    ), (fixed_anchor_profile, stronger_random_profile)
    assert (
        stronger_random_profile["utility"]
        > fixed_anchor_profile["utility"]
    ), (fixed_anchor_profile, stronger_random_profile)
    assert (
        stronger_random_profile["effect_flags"] & random_life
    ) != 0, stronger_random_profile

    # Sword and mace invert the animal/undead 1.5x/0.5x modifiers.  The old
    # per-dimension Pareto gate made every class swap impossible even when the
    # new weapon had substantially higher average damage.
    env.reset(seed=424248)
    sword_profile = dict(bridge.probe_gear_combat_profile())
    club = bridge.probe_spawn_test_gear(
        int(bridge.IDI_WARRCLUB), 5, 9)
    assert floor_item(club)["gear"], floor_item(club)
    assert pickup_gear(club) == 1
    club_profile = dict(bridge.probe_gear_combat_profile())
    assert club_profile["animal_max"] < sword_profile["animal_max"], (
        sword_profile, club_profile)
    assert club_profile["undead_max"] > sword_profile["undead_max"], (
        sword_profile, club_profile)
    assert club_profile["utility"] >= sword_profile["utility"] + 4096

    # Below the nonlinear 75% cap, an ordinary AC/resistance trade-off can
    # progress as well.
    env.reset(seed=424249)
    shield_anchor = bridge.probe_spawn_test_gear(
        int(bridge.IDI_WARRSHLD), 0, 0, 10, 0, 0, 10, 0, 0)
    assert pickup_gear(shield_anchor) == 1
    shield_anchor_profile = dict(bridge.probe_gear_combat_profile())
    shield_trade = bridge.probe_spawn_test_gear(
        int(bridge.IDI_WARRSHLD), 0, 0, 20, 0, 0, 5, 0, 0)
    assert floor_item(shield_trade)["gear"], floor_item(shield_trade)
    assert pickup_gear(shield_trade) == 1
    shield_trade_profile = dict(bridge.probe_gear_combat_profile())
    assert shield_trade_profile["fire_resist"] < shield_anchor_profile[
        "fire_resist"]
    assert shield_trade_profile["armor"] > shield_anchor_profile["armor"]
    assert shield_trade_profile["utility"] >= (
        shield_anchor_profile["utility"] + 4096)

    # Non-durable jewelry has no durability reserve.  A blank ring in an
    # empty slot must not manufacture exactly one upgrade margin merely by
    # existing.
    env.reset(seed=424250)
    blank_ring = bridge.probe_spawn_test_gear(
        int(bridge.IDI_INFRARING), 0, 0)
    blank_ring_raw = floor_item(blank_ring)
    assert int(blank_ring_raw["max_durability"]) == 0, blank_ring_raw
    assert not blank_ring_raw["gear"], blank_ring_raw
    assert pickup_gear(blank_ring) == 0

    # Durability is an absolute remaining-event resource.  The old ratio-only
    # reserve assigned both 1/1 and 200/200 exactly +4096, accepting a one-hit
    # item as if it had two hundred hits left.
    env.reset(seed=424259)
    fragile_ring = bridge.probe_spawn_test_gear(
        int(bridge.IDI_INFRARING), 0, 0, fire_resistance=1,
        durability=1, max_durability=1)
    fragile_raw = floor_item(fragile_ring)
    assert pickup_gear(fragile_ring) == 1
    fragile_profile = dict(bridge.probe_gear_combat_profile())
    durable_ring = bridge.probe_spawn_test_gear(
        int(bridge.IDI_INFRARING), 0, 0, fire_resistance=1,
        durability=200, max_durability=200)
    durable_raw = floor_item(durable_ring)
    assert int(durable_raw["combat_utility"]) > int(
        fragile_raw["combat_utility"]), (fragile_raw, durable_raw)
    assert durable_raw["gear"], durable_raw
    assert pickup_gear(durable_ring) == 1
    durable_profile = dict(bridge.probe_gear_combat_profile())
    assert durable_profile["utility"] > fragile_profile["utility"], (
        fragile_profile, durable_profile)

    # The whole-loadout reserve must remain strictly ordered above 32 as well.
    # The previous compact cap made an otherwise identical 200/200 weapon
    # invisible after equipping 32/32, even though it survives 168 additional
    # native durability-loss events.  Use one occupied weapon slot so a spare
    # ring slot cannot masquerade as evidence for replacement ordering.
    env.reset(seed=424259)
    fresh = bridge.observe()
    sword_base_id = int(next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )["base_id"])
    medium_weapon = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30, durability=32, max_durability=32)
    assert floor_item(medium_weapon)["gear"], floor_item(medium_weapon)
    assert pickup_gear(medium_weapon) == 1
    medium_profile = dict(bridge.probe_gear_combat_profile())
    long_lived_weapon = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30, durability=200, max_durability=200)
    assert floor_item(long_lived_weapon)["gear"], floor_item(
        long_lived_weapon)
    assert pickup_gear(long_lived_weapon) == 1
    long_lived_profile = dict(bridge.probe_gear_combat_profile())
    assert (
        long_lived_profile["utility"] - medium_profile["utility"]
    ) == 16 * (200 - 32), (medium_profile, long_lived_profile)

    indestructible_weapon = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30,
        durability=int(bridge.DUR_INDESTRUCTIBLE),
        max_durability=int(bridge.DUR_INDESTRUCTIBLE))
    assert floor_item(indestructible_weapon)["gear"], floor_item(
        indestructible_weapon)
    assert pickup_gear(indestructible_weapon) == 1
    indestructible_profile = dict(bridge.probe_gear_combat_profile())
    assert indestructible_profile["utility"] > long_lived_profile["utility"], (
        long_lived_profile, indestructible_profile)

    # Scalar trade-offs must really be reachable.  The Warrior starts with a
    # shield and a 16-frame one-handed attack; the old hard Pareto veto made
    # every two-handed axe (no block, 20 frames) impossible forever, even when
    # its damage dwarfed the entire starting loadout.  A weak axe remains a
    # net loss under the same score, while a genuinely strong one commits.
    env.reset(seed=424251)
    one_hand_profile = dict(bridge.probe_gear_combat_profile())
    weak_two_hand = bridge.probe_spawn_test_gear(
        int(bridge.IDI_CLEAVER), 1, 1)
    assert not floor_item(weak_two_hand)["gear"], floor_item(weak_two_hand)
    assert pickup_gear(weak_two_hand) == 0
    strong_two_hand = bridge.probe_spawn_test_gear(
        int(bridge.IDI_CLEAVER), 200, 255)
    assert floor_item(strong_two_hand)["gear"], floor_item(strong_two_hand)
    assert pickup_gear(strong_two_hand) == 1
    two_hand_profile = dict(bridge.probe_gear_combat_profile())
    assert one_hand_profile["block_enabled"], one_hand_profile
    assert not two_hand_profile["block_enabled"], two_hand_profile
    assert (
        two_hand_profile["attack_cycle_frames"]
        > one_hand_profile["attack_cycle_frames"]
    ), (one_hand_profile, two_hand_profile)
    assert two_hand_profile["utility"] > one_hand_profile["utility"], (
        one_hand_profile, two_hand_profile)
    assert two_hand_profile["physical_max"] > one_hand_profile[
        "physical_max"]

    # Per-hit damage and speed must be scored as one throughput quantity.
    # The former additive model charged a fixed price for losing Fastest, so at
    # high damage it accepted a slower weapon whose real damage/frame fell.
    env.reset(seed=424260)
    fresh = bridge.observe()
    sword_base_id = int(next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )["base_id"])
    fast_high = bridge.probe_spawn_test_gear(
        sword_base_id, 200, 220, effect_flags=fastest)
    assert pickup_gear(fast_high) == 1
    fast_high_profile = dict(bridge.probe_gear_combat_profile())
    slow_slightly_higher = bridge.probe_spawn_test_gear(
        sword_base_id, 205, 225)
    assert not floor_item(slow_slightly_higher)["gear"], floor_item(
        slow_slightly_higher)
    assert pickup_gear(slow_slightly_higher) == 0
    unchanged_fast_profile = dict(bridge.probe_gear_combat_profile())
    assert unchanged_fast_profile == fast_high_profile
    assert (
        (
            fast_high_profile["physical_min"]
            + fast_high_profile["physical_max"]
        )
        * 16
        > (
            fast_high_profile["physical_min"] + 5
            + fast_high_profile["physical_max"] + 5
        )
        * fast_high_profile["attack_cycle_frames"]
    ), fast_high_profile

    # The reverse direction must stay reachable: a slightly weaker-per-hit
    # Fastest weapon is accepted when its true damage/frame is higher.
    env.reset(seed=424261)
    fresh = bridge.observe()
    sword_base_id = int(next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )["base_id"])
    slow_high = bridge.probe_spawn_test_gear(
        sword_base_id, 200, 220)
    assert pickup_gear(slow_high) == 1
    slow_high_profile = dict(bridge.probe_gear_combat_profile())
    fast_slightly_lower = bridge.probe_spawn_test_gear(
        sword_base_id, 195, 215, effect_flags=fastest)
    assert floor_item(fast_slightly_lower)["gear"], floor_item(
        fast_slightly_lower)
    assert pickup_gear(fast_slightly_lower) == 1
    fast_lower_profile = dict(bridge.probe_gear_combat_profile())
    assert (
        (
            fast_lower_profile["physical_min"]
            + fast_lower_profile["physical_max"]
        )
        * slow_high_profile["attack_cycle_frames"]
        > (
            slow_high_profile["physical_min"]
            + slow_high_profile["physical_max"]
        )
        * fast_lower_profile["attack_cycle_frames"]
    ), (slow_high_profile, fast_lower_profile)
    assert fast_lower_profile["utility"] > slow_high_profile["utility"]

    # Elemental weapon explosions are real action-9 damage.  Each fire
    # explosion retries collision for nine animation frames and lightning for
    # seven; even under the conservative resistant-target quarter-damage
    # portfolio, a Lightning 2-20 starter sword must not be replaced by a
    # plain sword whose physical range rises only from 2-6 to 4-8.
    env.reset(seed=424260)
    fresh = bridge.observe()
    sword_base_id = int(next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )["base_id"])
    lightning_weapon = bridge.probe_spawn_test_gear(
        sword_base_id, 2, 6,
        lightning_min_damage=2, lightning_max_damage=20)
    assert floor_item(lightning_weapon)["gear"], floor_item(lightning_weapon)
    assert pickup_gear(lightning_weapon) == 1
    lightning_profile = dict(bridge.probe_gear_combat_profile())
    assert (
        lightning_profile["lightning_min"],
        lightning_profile["lightning_max"],
    ) == (2, 20), lightning_profile
    plain_physical = bridge.probe_spawn_test_gear(
        sword_base_id, 4, 8)
    assert not floor_item(plain_physical)["gear"], floor_item(plain_physical)
    assert pickup_gear(plain_physical) == 0
    assert dict(bridge.probe_gear_combat_profile()) == lightning_profile

    # Magic is not generally useful to this 15-action Warrior, but it raises
    # WeaponExplosion's native magic-to-hit.  It must be neutral without
    # elemental damage (covered above) and positive when such damage exists.
    magic_lightning = bridge.probe_spawn_test_gear(
        sword_base_id, 2, 6, magic_bonus=1,
        lightning_min_damage=2, lightning_max_damage=20)
    assert floor_item(magic_lightning)["gear"], floor_item(magic_lightning)
    assert pickup_gear(magic_lightning) == 1
    magic_lightning_profile = dict(bridge.probe_gear_combat_profile())
    assert (
        magic_lightning_profile["magic_to_hit"]
        == lightning_profile["magic_to_hit"] + 1
    ), (lightning_profile, magic_lightning_profile)
    assert magic_lightning_profile["utility"] > lightning_profile["utility"]

    # Damage and melee hit chance must also be coupled.  Against the same
    # target-neutral reference, 100 damage at a clamped 95% hit probability
    # exceeds 110 damage at 81%; the old additive score accepted the latter.
    env.reset(seed=424260)
    fresh = bridge.observe()
    sword_base_id = int(next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )["base_id"])
    accurate_weapon = bridge.probe_spawn_test_gear(
        sword_base_id, 100, 100, to_hit_bonus=20)
    assert floor_item(accurate_weapon)["gear"], floor_item(accurate_weapon)
    assert pickup_gear(accurate_weapon) == 1
    accurate_profile = dict(bridge.probe_gear_combat_profile())
    inaccurate_paper_damage = bridge.probe_spawn_test_gear(
        sword_base_id, 110, 110)
    assert not floor_item(inaccurate_paper_damage)["gear"], floor_item(
        inaccurate_paper_damage)
    assert pickup_gear(inaccurate_paper_damage) == 0
    assert dict(bridge.probe_gear_combat_profile()) == accurate_profile

    # Removing +HP/+VIT subtracts an absolute amount from current and maximum
    # life.  "Projected alive" alone is not enough: 101 HP on +100 HP gear
    # would become a nonfatal but catastrophic 1 HP.  The health-fraction gate
    # must defer this swap while injured, then allow the exact same high-damage
    # item after healing to full so it does not become another permanent lock.
    env.reset(seed=424252)
    fresh = bridge.observe()
    sword_base_id = int(next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )["base_id"])
    life_anchor = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30, magic_damage_bonus=40,
        life_bonus_points=100)
    assert pickup_gear(life_anchor) == 1
    anchor_raw = bridge.observe()
    anchor_utility = int(anchor_raw["gear_combat_utility"])
    anchor_profile = dict(bridge.probe_gear_combat_profile())
    anchor_max_hp = int(anchor_profile["max_hp_fixed"]) >> 6
    assert anchor_max_hp > 101, anchor_profile
    anchor_weapon = next(
        item for item in anchor_raw["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )
    anchor_identity = (
        int(anchor_weapon["base_id"]),
        int(anchor_weapon["seed_hi"]),
        int(anchor_weapon["seed_lo"]),
        int(anchor_weapon["create_info"]),
    )
    assert bridge.probe_set_current_hit_points(101) == 101 * 64
    low_profile = dict(bridge.probe_gear_combat_profile())
    assert low_profile["current_hp_fixed"] == 101 * 64, low_profile
    assert low_profile["current_hp_fixed"] - 100 * 64 == 64
    catastrophic_damage = bridge.probe_spawn_test_gear(
        sword_base_id, 200, 255, magic_damage_bonus=255)
    catastrophic_raw = floor_item(catastrophic_damage)
    assert not catastrophic_raw["gear"], catastrophic_raw
    assert pickup_gear(catastrophic_damage) == 0
    rejected = bridge.observe()
    rejected_weapon = next(
        item for item in rejected["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )
    assert (
        int(rejected_weapon["base_id"]),
        int(rejected_weapon["seed_hi"]),
        int(rejected_weapon["seed_lo"]),
        int(rejected_weapon["create_info"]),
    ) == anchor_identity, (anchor_weapon, rejected_weapon)
    assert int(rejected["hp"]) == 101, rejected
    assert int(rejected["gear_combat_utility"]) == anchor_utility, rejected
    assert any(
        item["active_id"] == catastrophic_damage["active_id"]
        for item in rejected["floor_items"]
    ), rejected

    assert bridge.probe_set_current_hit_points(anchor_max_hp) == int(
        anchor_profile["max_hp_fixed"])
    assert floor_item(catastrophic_damage)["gear"], floor_item(
        catastrophic_damage)
    assert pickup_gear(catastrophic_damage) == 1
    healed_trade = dict(bridge.probe_gear_combat_profile())
    assert healed_trade["current_hp_fixed"] == healed_trade["max_hp_fixed"]
    assert (
        anchor_profile["max_hp_fixed"] - healed_trade["max_hp_fixed"]
    ) == 100 * 64, (anchor_profile, healed_trade)
    assert healed_trade["utility"] > anchor_profile["utility"]

    # At a still lower HPBase the same removal crosses Diablo's fixed-point
    # death boundary.  Planning must reject before touching live equipment and
    # preserve the exact body/resource state.
    env.reset(seed=424262)
    fresh = bridge.observe()
    sword_base_id = int(next(
        item for item in fresh["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )["base_id"])
    life_anchor = bridge.probe_spawn_test_gear(
        sword_base_id, 20, 30, magic_damage_bonus=40,
        life_bonus_points=100)
    assert pickup_gear(life_anchor) == 1
    fatal_anchor_raw = bridge.observe()
    fatal_anchor_utility = int(fatal_anchor_raw["gear_combat_utility"])
    fatal_anchor_weapon = next(
        item for item in fatal_anchor_raw["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )
    fatal_anchor_identity = (
        int(fatal_anchor_weapon["base_id"]),
        int(fatal_anchor_weapon["seed_hi"]),
        int(fatal_anchor_weapon["seed_lo"]),
        int(fatal_anchor_weapon["create_info"]),
    )
    assert bridge.probe_set_current_hit_points(1) == 64
    fatal_profile = dict(bridge.probe_gear_combat_profile())
    assert fatal_profile["current_hp_fixed"] == 64, fatal_profile
    suicidal_damage = bridge.probe_spawn_test_gear(
        sword_base_id, 200, 255, magic_damage_bonus=255)
    assert not floor_item(suicidal_damage)["gear"], floor_item(
        suicidal_damage)
    assert pickup_gear(suicidal_damage) == 0
    fatal_rejected = bridge.observe()
    fatal_rejected_weapon = next(
        item for item in fatal_rejected["equipped_items"]
        if item["present"] and item["item_class"] == 1
    )
    assert (
        int(fatal_rejected_weapon["base_id"]),
        int(fatal_rejected_weapon["seed_hi"]),
        int(fatal_rejected_weapon["seed_lo"]),
        int(fatal_rejected_weapon["create_info"]),
    ) == fatal_anchor_identity, (
        fatal_anchor_weapon, fatal_rejected_weapon)
    assert int(fatal_rejected["hp"]) == 1, fatal_rejected
    assert int(fatal_rejected["gear_combat_utility"]) == (
        fatal_anchor_utility), fatal_rejected

    print(
        "PASS: a14 拒绝弱装/任务物并严格提升 whole-loadout utility；"
        "法力/法术/箭矢/Knockback、重复攻速与超 cap 不制造伪升级；"
        "ZeroResistance/DrainLife/Decay/Peril/Doppelganger、致死及低血比换装硬拒；"
        "微小抗性、光照、高伤换低命中/+受伤/满抗/恢复/吸血、剑锤类别、"
        "AC/单抗、强双手武器、攻速/命中耦合吞吐及元素爆炸均正确排序；"
        "Random/5%/3% 吸血按物理命中吞吐单调排序且可叠加；"
        "1/1、32/32、200/200 与不可毁耐久严格区分，"
        "弱双手/空白珠宝不伪升级"
    )
finally:
    env.close()
