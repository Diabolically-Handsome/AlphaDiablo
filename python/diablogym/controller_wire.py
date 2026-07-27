"""Pure, immutable schema for the dual Worker/controller observation wire.

This module deliberately imports neither Gym nor the native bridge.  Runtime
capture, policy code, migration receipts and structured encoders all consume
the same constants and canonical hash instead of copying dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


def _divide_by(scale: float) -> str:
    return f"divide_by:{format(float(scale), '.17g')}"


def _scaled_encodings(
    fields: tuple[str, ...],
    scales: tuple[float, ...],
    *,
    binary_fields: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    if len(fields) != len(scales):
        raise RuntimeError(
            "wire field/scale 数量不闭合:"
            f"{len(fields)}!={len(scales)}")
    return tuple(
        "binary_0_or_1" if field in binary_fields else _divide_by(scale)
        for field, scale in zip(fields, scales, strict=True)
    )


CONTROLLER_SNAPSHOT_RADIUS = 12
CONTROLLER_SNAPSHOT_SIDE = 2 * CONTROLLER_SNAPSHOT_RADIUS + 1
CONTROLLER_SNAPSHOT_CELLS = CONTROLLER_SNAPSHOT_SIDE ** 2
CONTROLLER_SNAPSHOT_SOFTWALL_KIND_DENOMINATOR = 7.0
CONTROLLER_SNAPSHOT_MAP_CHANNELS = (
    "walkable",
    # Only currently visible occupants are policy-observable.  The immutable
    # snapshot keeps the full physical collision plane separately for local
    # macro planning, but that privileged plane must never enter this wire.
    "visible_monster",
    # Normalized bit-pack.  Decode with round(value * 7), then bits 0/1/2
    # are softwall/closed_door/explosive_softwall respectively.
    "softwall_kind",
    "visited",
    "blocked",
    "protected",
    "hazard",
)
CONTROLLER_SNAPSHOT_MAP_FIELD_ENCODINGS = (
    "binary_0_or_1",
    "binary_0_or_1",
    "normalized_bitpack_div_7:"
    "softwall+2*closed_door+4*explosive_softwall",
    "binary_0_or_1",
    "binary_0_or_1",
    "binary_0_or_1",
    "binary_0_or_1",
)
CONTROLLER_SNAPSHOT_MAP_DIM = (
    len(CONTROLLER_SNAPSHOT_MAP_CHANNELS) * CONTROLLER_SNAPSHOT_CELLS)

CONTROLLER_SNAPSHOT_MONSTER_LIMIT = 64
_CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_SCHEMA = (
    ("hp_fixed_hi", 65536.0),
    ("hp_fixed_lo", 65536.0),
    ("max_hp_fixed_hi", 65536.0),
    ("max_hp_fixed_lo", 65536.0),
    ("mode", 18.0),
    ("direction", 8.0),
    ("anim_frame", 32.0),
    ("anim_tick", 32.0),
    ("anim_ticks_per_frame", 32.0),
    ("anim_num_frames", 32.0),
    ("anim_progress", 128.0),
    ("anim_petrified", 1.0),
    ("enemy_dx", 112.0),
    ("enemy_dy", 112.0),
    ("old_dx", 112.0),
    ("old_dy", 112.0),
    ("min_damage", 255.0),
    ("max_damage", 255.0),
    ("min_damage_special", 255.0),
    ("max_damage_special", 255.0),
    ("armor_class", 255.0),
    ("resistance", 65535.0),
    ("unique_type", 255.0),
    ("reduce_strength", 255.0),
    ("reduce_magic", 255.0),
    ("reduce_dexterity", 255.0),
    ("reduce_vitality", 255.0),
    ("reduce_max_hp", 255.0),
    ("reduce_max_mana", 255.0),
    ("monster_level", 255.0),
    ("to_hit", 65535.0),
    ("to_hit_special", 65535.0),
    ("experience_hi", 65536.0),
    ("experience_lo", 65536.0),
    # These are live ProcessMonsters state-machine inputs, not save-only
    # bookkeeping.  Signed integer ranges are divided by powers of two so
    # every native int8/int16 value remains exactly representable in float32.
    ("goal", 8.0),
    ("goal_var1", 32768.0),
    ("goal_var2", 128.0),
    ("goal_var3", 128.0),
    ("var1", 32768.0),
    ("var2", 32768.0),
    ("var3", 128.0),
    ("last_dx", 112.0),
    ("last_dy", 112.0),
    ("active_for_ticks", 256.0),
    ("path_count", 256.0),
    ("enemy_id", 256.0),
    ("is_invalid", 1.0),
    ("monster_level_type", 16.0),
    ("ai", 64.0),
    ("intelligence", 256.0),
    ("leader", 256.0),
    ("leader_relation", 4.0),
    ("pack_size", 256.0),
    ("talk_msg", 65536.0),
)
CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_FIELDS = tuple(
    field for field, _ in _CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_SCHEMA)
CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_SCALES = tuple(
    scale for _, scale in _CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_SCHEMA)
CONTROLLER_SNAPSHOT_MONSTER_INT32_FIELDS = ("temp_dx", "temp_dy")
CONTROLLER_SNAPSHOT_MONSTER_WORD_FIELDS = tuple(
    word
    for field in CONTROLLER_SNAPSHOT_MONSTER_INT32_FIELDS
    for word in (f"{field}_hi", f"{field}_lo")
)
CONTROLLER_SNAPSHOT_MONSTER_FLAG_BITS = 13
CONTROLLER_SNAPSHOT_MONSTER_BASE_FIELDS = (
    "present",
    "monster_id",
    "monster_type",
    "tile_dx",
    "tile_dy",
    "future_dx",
    "future_dy",
    "hp",
    "max_hp",
    "ledger_low",
    "ledger_max",
    "blocked",
    "visible",
    "native_reachable",
    "locally_engageable",
)
CONTROLLER_SNAPSHOT_MONSTER_ROW_FIELDS = (
    CONTROLLER_SNAPSHOT_MONSTER_BASE_FIELDS
    + CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_FIELDS
    + CONTROLLER_SNAPSHOT_MONSTER_WORD_FIELDS
    + tuple(
        f"combat_flag_bit_{bit}"
        for bit in range(CONTROLLER_SNAPSHOT_MONSTER_FLAG_BITS)
    )
)
CONTROLLER_SNAPSHOT_MONSTER_FIELDS = len(
    CONTROLLER_SNAPSHOT_MONSTER_ROW_FIELDS)
CONTROLLER_SNAPSHOT_MONSTER_ROW_ENCODINGS = (
    (
        "binary_0_or_1",
        _divide_by(200.0),
        _divide_by(200.0),
        _divide_by(112.0),
        _divide_by(112.0),
        _divide_by(112.0),
        _divide_by(112.0),
        _divide_by(1024.0),
        _divide_by(1024.0),
        _divide_by(1024.0),
        _divide_by(1024.0),
        "binary_0_or_1",
        "binary_0_or_1",
        "binary_0_or_1",
        "binary_0_or_1",
    )
    + _scaled_encodings(
        CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_FIELDS,
        CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_SCALES,
        binary_fields=frozenset({"anim_petrified", "is_invalid"}),
    )
    + tuple(
        encoding
        for _field in CONTROLLER_SNAPSHOT_MONSTER_INT32_FIELDS
        for encoding in (
            "signed_int32_twos_complement_hi16_divide_by:65536",
            "signed_int32_twos_complement_lo16_divide_by:65536",
        )
    )
    + ("binary_0_or_1",) * CONTROLLER_SNAPSHOT_MONSTER_FLAG_BITS
)
CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_FIELDS = (
    "visible_count",
    "overflow_count",
    "engageable_count",
    "engageable_overflow_count",
    "overflow_hp_sum",
    "overflow_max_hp_sum",
    "overflow_nearest_distance",
    "overflow_max_damage",
)
CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_SCALES = (
    256.0,
    256.0,
    256.0,
    256.0,
    65536.0,
    65536.0,
    float(CONTROLLER_SNAPSHOT_RADIUS),
    255.0,
)
CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_ENCODINGS = _scaled_encodings(
    CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_FIELDS,
    CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_SCALES,
)
CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_DIM = len(
    CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_FIELDS)
CONTROLLER_SNAPSHOT_MONSTER_DIM = (
    CONTROLLER_SNAPSHOT_MONSTER_LIMIT * CONTROLLER_SNAPSHOT_MONSTER_FIELDS
    + CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_DIM
)

CONTROLLER_SNAPSHOT_MISSILE_LIMIT = 32
_CONTROLLER_SNAPSHOT_MISSILE_DIRECT_SCHEMA = (
    ("type", 128.0),
    ("visible", 1.0),
    ("source_visible", 1.0),
    ("start_visible", 1.0),
    ("tile_dx", float(CONTROLLER_SNAPSHOT_RADIUS)),
    ("tile_dy", float(CONTROLLER_SNAPSHOT_RADIUS)),
    ("start_dx", 112.0),
    ("start_dy", 112.0),
    ("direction", 16.0),
    ("deleted", 1.0),
    ("draw", 1.0),
    ("pre", 1.0),
    ("caster", 4.0),
    ("hit", 1.0),
    ("source_type", 4.0),
    ("hostile", 1.0),
    ("anim_type", 256.0),
    ("anim_flags", 4.0),
    # _mirnd is a misleading upstream name: after creation it is the current
    # deterministic BPath phase and is read/advanced by ProcessMissiles.
    ("random", 16.0),
    ("limit_reached", 1.0),
)
CONTROLLER_SNAPSHOT_MISSILE_DIRECT_FIELDS = tuple(
    field for field, _ in _CONTROLLER_SNAPSHOT_MISSILE_DIRECT_SCHEMA)
CONTROLLER_SNAPSHOT_MISSILE_DIRECT_SCALES = tuple(
    scale for _, scale in _CONTROLLER_SNAPSHOT_MISSILE_DIRECT_SCHEMA)
CONTROLLER_SNAPSHOT_MISSILE_INT32_FIELDS = (
    "offset_x", "offset_y",
    "velocity_x", "velocity_y",
    "traveled_x", "traveled_y",
    "spell_level",
    "anim_delay", "anim_len", "anim_frame",
    "duration", "source_id", "damage", "distance",
    "anim_count", "anim_add",
)
CONTROLLER_SNAPSHOT_MISSILE_WORD_FIELDS = tuple(
    word
    for field in CONTROLLER_SNAPSHOT_MISSILE_INT32_FIELDS
    for word in (f"{field}_hi", f"{field}_lo")
)
CONTROLLER_SNAPSHOT_MISSILE_ROW_FIELDS = (
    ("present",)
    + CONTROLLER_SNAPSHOT_MISSILE_DIRECT_FIELDS
    + CONTROLLER_SNAPSHOT_MISSILE_WORD_FIELDS
)
CONTROLLER_SNAPSHOT_MISSILE_FIELDS = len(
    CONTROLLER_SNAPSHOT_MISSILE_ROW_FIELDS)
CONTROLLER_SNAPSHOT_MISSILE_ROW_ENCODINGS = (
    ("binary_0_or_1",)
    + _scaled_encodings(
        CONTROLLER_SNAPSHOT_MISSILE_DIRECT_FIELDS,
        CONTROLLER_SNAPSHOT_MISSILE_DIRECT_SCALES,
        binary_fields=frozenset({
            "visible", "source_visible", "start_visible",
            "deleted", "draw", "pre", "hit", "hostile",
            "limit_reached",
        }),
    )
    + tuple(
        encoding
        for _field in CONTROLLER_SNAPSHOT_MISSILE_INT32_FIELDS
        for encoding in (
            "signed_int32_twos_complement_hi16_divide_by:65536",
            "signed_int32_twos_complement_lo16_divide_by:65536",
        )
    )
)
CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_FIELDS = (
    "local_count",
    "overflow_count",
    "hostile_count",
    "hostile_overflow_count",
    "overflow_abs_damage_sum",
    "overflow_max_abs_damage",
    "overflow_nearest_hostile_distance",
    "overflow_nearest_hostile_duration",
    "deleted_count",
)
CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_SCALES = (
    256.0, 256.0, 256.0, 256.0,
    65536.0, 65536.0,
    float(CONTROLLER_SNAPSHOT_RADIUS), 65536.0,
    256.0,
)
CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_ENCODINGS = _scaled_encodings(
    CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_FIELDS,
    CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_SCALES,
)
CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_DIM = len(
    CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_FIELDS)
CONTROLLER_SNAPSHOT_MISSILE_DIM = (
    CONTROLLER_SNAPSHOT_MISSILE_LIMIT * CONTROLLER_SNAPSHOT_MISSILE_FIELDS
    + CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_DIM
)

CONTROLLER_SNAPSHOT_BELT_SLOTS = 8
CONTROLLER_SNAPSHOT_INSTANT_HEAL_KINDS = 4
CONTROLLER_SNAPSHOT_BELT_KINDS = 6
CONTROLLER_SNAPSHOT_BELT_ROW_FIELDS = (
    "empty",
    "other",
    "heal_minor",
    "heal_full",
    "rejuvenation_minor",
    "rejuvenation_full",
)
CONTROLLER_SNAPSHOT_BELT_DIM = (
    CONTROLLER_SNAPSHOT_BELT_SLOTS * CONTROLLER_SNAPSHOT_BELT_KINDS)

CONTROLLER_SNAPSHOT_EXACT_FIELDS = (
    "hp",
    "max_hp",
    "mana",
    "max_mana",
    "hp_fixed_hi",
    "hp_fixed_lo",
    "max_hp_fixed_hi",
    "max_hp_fixed_lo",
    "mana_fixed_hi",
    "mana_fixed_lo",
    "max_mana_fixed_hi",
    "max_mana_fixed_lo",
    "armor_class",
    "is_set_level",
    "set_level_id",
    "engine_level",
    "level_type",
    "betrayer_quest_active",
    "betrayer_quest_stage",
    "betrayer_portal_stage",
    "monotonic_quest_turn_in_used",
)
CONTROLLER_SNAPSHOT_EXACT_SCALES = (
    1024.0, 1024.0, 1024.0, 1024.0,
    65536.0, 65536.0,
    65536.0, 65536.0,
    65536.0, 65536.0,
    65536.0, 65536.0,
    200.0,
    1.0,
    16.0,
    16.0,
    4.0,
    3.0,
    16.0,
    16.0,
    1.0,
)
CONTROLLER_SNAPSHOT_EXACT_ENCODINGS = _scaled_encodings(
    CONTROLLER_SNAPSHOT_EXACT_FIELDS,
    CONTROLLER_SNAPSHOT_EXACT_SCALES,
    binary_fields=frozenset({
        "is_set_level", "monotonic_quest_turn_in_used",
    }),
)
CONTROLLER_SNAPSHOT_EXACT_DIM = len(CONTROLLER_SNAPSHOT_EXACT_FIELDS)

CONTROLLER_SNAPSHOT_COMBAT_FIELDS = (
    "hero_class",
    "strength",
    "magic",
    "dexterity",
    "vitality",
    "melee_to_hit",
    "melee_piercing_to_hit",
    "block_chance",
    "item_min_damage",
    "item_max_damage",
    "damage_mod",
    "item_bonus_damage",
    "item_bonus_to_hit",
    "item_bonus_damage_mod",
    "item_get_hit",
    "item_enemy_ac",
    "magic_resist",
    "fire_resist",
    "lightning_resist",
    "item_fire_min",
    "item_fire_max",
    "item_lightning_min",
    "item_lightning_max",
    "block_enabled",
    "gear_combat_utility_hi",
    "gear_combat_utility_lo",
)
CONTROLLER_SNAPSHOT_COMBAT_SCALES = (
    8.0,
    750.0, 750.0, 750.0, 750.0,
    300.0, 300.0, 300.0,
    1024.0, 1024.0, 1024.0,
    200.0, 300.0, 1024.0,
    200.0, 200.0,
    100.0, 100.0, 100.0,
    1024.0, 1024.0, 1024.0, 1024.0,
    1.0,
    65536.0, 65536.0,
)
CONTROLLER_SNAPSHOT_COMBAT_ENCODINGS = _scaled_encodings(
    CONTROLLER_SNAPSHOT_COMBAT_FIELDS,
    CONTROLLER_SNAPSHOT_COMBAT_SCALES,
    binary_fields=frozenset({"block_enabled"}),
)
CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS = 32
CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS = 8
CONTROLLER_SNAPSHOT_EQUIPPED_SLOTS = 7
CONTROLLER_SNAPSHOT_STICKY_DIM = 3

_CONTROLLER_SNAPSHOT_GEAR_SCHEMA = (
    ("active_id", 128.0),
    ("item_type", 16.0),
    ("equip_loc", 8.0),
    ("base_ac", 200.0),
    ("identified", 1.0),
    ("quality", 2.0),
    ("durability", 255.0),
    ("max_durability", 255.0),
    ("effects_active", 1.0),
    ("min_damage", 200.0),
    ("max_damage", 200.0),
    ("effect_damage", 200.0),
    ("effect_to_hit", 300.0),
    ("effect_ac_percent", 200.0),
    ("effect_strength", 100.0),
    ("effect_magic", 100.0),
    ("effect_dexterity", 100.0),
    ("effect_vitality", 100.0),
    ("effect_fire_resist", 100.0),
    ("effect_lightning_resist", 100.0),
    ("effect_magic_resist", 100.0),
    ("effect_mana", 1024.0),
    ("effect_hp", 1024.0),
    ("effect_damage_mod", 1024.0),
    ("effect_get_hit", 200.0),
    ("effect_light", 16.0),
    ("effect_spell_level", 16.0),
    ("effect_enemy_ac", 200.0),
    ("effect_fire_min", 1024.0),
    ("effect_fire_max", 1024.0),
    ("effect_lightning_min", 1024.0),
    ("effect_lightning_max", 1024.0),
    ("item_class", 8.0),
    ("min_strength", 255.0),
    ("min_magic", 255.0),
    ("min_dexterity", 255.0),
    ("stat_usable", 1.0),
    ("combat_utility_hi", 65536.0),
    ("combat_utility_lo", 65536.0),
)
CONTROLLER_SNAPSHOT_GEAR_FIELDS = tuple(
    field for field, _ in _CONTROLLER_SNAPSHOT_GEAR_SCHEMA)
CONTROLLER_SNAPSHOT_GEAR_SCALES = tuple(
    scale for _, scale in _CONTROLLER_SNAPSHOT_GEAR_SCHEMA)
CONTROLLER_SNAPSHOT_GEAR_ENCODINGS = _scaled_encodings(
    CONTROLLER_SNAPSHOT_GEAR_FIELDS,
    CONTROLLER_SNAPSHOT_GEAR_SCALES,
    binary_fields=frozenset({
        "identified", "effects_active", "stat_usable",
    }),
)
CONTROLLER_SNAPSHOT_EFFECT_FLAG_FIELDS = tuple(
    f"effect_flag_bit_{bit}"
    for bit in range(CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS)
)
CONTROLLER_SNAPSHOT_DAM_AC_FLAG_FIELDS = tuple(
    f"dam_ac_flag_bit_{bit}"
    for bit in range(CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS)
)
CONTROLLER_SNAPSHOT_COMBAT_PREFIX_FIELDS = (
    CONTROLLER_SNAPSHOT_COMBAT_FIELDS
    + CONTROLLER_SNAPSHOT_EFFECT_FLAG_FIELDS
    + CONTROLLER_SNAPSHOT_DAM_AC_FLAG_FIELDS
)
CONTROLLER_SNAPSHOT_COMBAT_PREFIX_ENCODINGS = (
    CONTROLLER_SNAPSHOT_COMBAT_ENCODINGS
    + ("binary_0_or_1",) * CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS
    + ("binary_0_or_1",) * CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS
)
CONTROLLER_SNAPSHOT_EQUIPPED_ROW_FIELDS = (
    ("present",)
    + CONTROLLER_SNAPSHOT_GEAR_FIELDS
    + CONTROLLER_SNAPSHOT_EFFECT_FLAG_FIELDS
    + CONTROLLER_SNAPSHOT_DAM_AC_FLAG_FIELDS
)
CONTROLLER_SNAPSHOT_EQUIPPED_ROW_ENCODINGS = (
    ("binary_0_or_1",)
    + CONTROLLER_SNAPSHOT_GEAR_ENCODINGS
    + ("binary_0_or_1",) * CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS
    + ("binary_0_or_1",) * CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS
)
CONTROLLER_SNAPSHOT_EQUIPPED_FIELDS = len(
    CONTROLLER_SNAPSHOT_EQUIPPED_ROW_FIELDS)
CONTROLLER_SNAPSHOT_COMBAT_DIM = (
    len(CONTROLLER_SNAPSHOT_COMBAT_FIELDS)
    + CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS
    + CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS
    + CONTROLLER_SNAPSHOT_EQUIPPED_SLOTS
    * CONTROLLER_SNAPSHOT_EQUIPPED_FIELDS
)
CONTROLLER_SNAPSHOT_HEAL_TARGET_FIELDS = (
    "present", "tile_dx", "tile_dy", "active_id", "heal_kind")
CONTROLLER_SNAPSHOT_HEAL_TARGET_DIM = len(
    CONTROLLER_SNAPSHOT_HEAL_TARGET_FIELDS)
CONTROLLER_SNAPSHOT_GEAR_TARGET_FIELDS = (
    ("present", "tile_dx", "tile_dy")
    + CONTROLLER_SNAPSHOT_GEAR_FIELDS
    + CONTROLLER_SNAPSHOT_EFFECT_FLAG_FIELDS
    + CONTROLLER_SNAPSHOT_DAM_AC_FLAG_FIELDS
)
CONTROLLER_SNAPSHOT_GEAR_TARGET_DIM = len(
    CONTROLLER_SNAPSHOT_GEAR_TARGET_FIELDS)
CONTROLLER_SNAPSHOT_HEAL_TARGET_ENCODINGS = (
    "binary_0_or_1",
    _divide_by(112.0),
    _divide_by(112.0),
    _divide_by(128.0),
    _divide_by(CONTROLLER_SNAPSHOT_INSTANT_HEAL_KINDS),
)
CONTROLLER_SNAPSHOT_GEAR_TARGET_ENCODINGS = (
    (
        "binary_0_or_1",
        _divide_by(112.0),
        _divide_by(112.0),
    )
    + CONTROLLER_SNAPSHOT_GEAR_ENCODINGS
    + ("binary_0_or_1",) * CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS
    + ("binary_0_or_1",) * CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS
)
CONTROLLER_SNAPSHOT_ITEM_TARGET_DIM = (
    CONTROLLER_SNAPSHOT_HEAL_TARGET_DIM
    + CONTROLLER_SNAPSHOT_GEAR_TARGET_DIM
)
CONTROLLER_SNAPSHOT_VECTOR_DIM = (
    CONTROLLER_SNAPSHOT_MAP_DIM
    + CONTROLLER_SNAPSHOT_MONSTER_DIM
    + CONTROLLER_SNAPSHOT_MISSILE_DIM
    + CONTROLLER_SNAPSHOT_BELT_DIM
    + CONTROLLER_SNAPSHOT_EXACT_DIM
    + CONTROLLER_SNAPSHOT_COMBAT_DIM
    + CONTROLLER_SNAPSHOT_STICKY_DIM
    + CONTROLLER_SNAPSHOT_ITEM_TARGET_DIM
)

# Absolute dual Worker layout.  Prefix fields remain frozen V28/current wrapper
# state; controller segments begin at 635.
DUAL_WORKER_CONTROLLER_START = 635
DUAL_WORKER_OBSERVATION_DIM = (
    DUAL_WORKER_CONTROLLER_START + CONTROLLER_SNAPSHOT_VECTOR_DIM)
DUAL_WORKER_LEGACY_SLICE = slice(0, 298)
DUAL_WORKER_CURRENT_V4_BASE_SLICE = slice(298, 593)
DUAL_WORKER_CURRENT_LAYER_CLOCK_FEATURE = 593
DUAL_WORKER_CURRENT_EXHAUSTED_FEATURE = 594
DUAL_WORKER_FARM_SCENE_FRACTION_FEATURE = 595
DUAL_WORKER_TIME_REMAINING_FEATURE = 596
DUAL_WORKER_LAYER_KILLS_FEATURE = 597
DUAL_WORKER_LEGACY_LAYER_TIME_FEATURE = 598
DUAL_WORKER_DRY_FLOOR_REMAINING_FEATURE = 599
DUAL_WORKER_DRINK_LATCH_FEATURE = 600
DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE = 601
DUAL_WORKER_FUSE_STREAK_SLICE = slice(602, 617)
DUAL_WORKER_ACTION_MASK_SLICE = slice(617, 632)
DUAL_WORKER_MANAGER_MASK_SLICE = slice(632, 635)
DUAL_WORKER_CONTROLLER_MAP_SLICE = slice(
    DUAL_WORKER_CONTROLLER_START,
    DUAL_WORKER_CONTROLLER_START + CONTROLLER_SNAPSHOT_MAP_DIM,
)
DUAL_WORKER_CONTROLLER_MONSTER_SLICE = slice(
    DUAL_WORKER_CONTROLLER_MAP_SLICE.stop,
    DUAL_WORKER_CONTROLLER_MAP_SLICE.stop + CONTROLLER_SNAPSHOT_MONSTER_DIM,
)
DUAL_WORKER_CONTROLLER_MISSILE_SLICE = slice(
    DUAL_WORKER_CONTROLLER_MONSTER_SLICE.stop,
    DUAL_WORKER_CONTROLLER_MONSTER_SLICE.stop
    + CONTROLLER_SNAPSHOT_MISSILE_DIM,
)
DUAL_WORKER_CONTROLLER_BELT_SLICE = slice(
    DUAL_WORKER_CONTROLLER_MISSILE_SLICE.stop,
    DUAL_WORKER_CONTROLLER_MISSILE_SLICE.stop + CONTROLLER_SNAPSHOT_BELT_DIM,
)
DUAL_WORKER_CONTROLLER_EXACT_SLICE = slice(
    DUAL_WORKER_CONTROLLER_BELT_SLICE.stop,
    DUAL_WORKER_CONTROLLER_BELT_SLICE.stop + CONTROLLER_SNAPSHOT_EXACT_DIM,
)
DUAL_WORKER_CONTROLLER_COMBAT_SLICE = slice(
    DUAL_WORKER_CONTROLLER_EXACT_SLICE.stop,
    DUAL_WORKER_CONTROLLER_EXACT_SLICE.stop + CONTROLLER_SNAPSHOT_COMBAT_DIM,
)
DUAL_WORKER_CONTROLLER_STICKY_SLICE = slice(
    DUAL_WORKER_CONTROLLER_COMBAT_SLICE.stop,
    DUAL_WORKER_CONTROLLER_COMBAT_SLICE.stop
    + CONTROLLER_SNAPSHOT_STICKY_DIM,
)
DUAL_WORKER_CONTROLLER_ITEM_TARGET_SLICE = slice(
    DUAL_WORKER_CONTROLLER_STICKY_SLICE.stop,
    DUAL_WORKER_CONTROLLER_STICKY_SLICE.stop
    + CONTROLLER_SNAPSHOT_ITEM_TARGET_DIM,
)


@dataclass(frozen=True)
class WireSegmentSpec:
    name: str
    start: int
    stop: int
    shape: tuple[int, ...]
    field_names: tuple[str, ...]
    semantic_tags: tuple[str, ...]
    row_width: int | None = None
    row_count: int | None = None
    row_offset: int = 0
    prefix_field_names: tuple[str, ...] = ()
    prefix_field_encodings: tuple[str, ...] = ()
    tail_field_names: tuple[str, ...] = ()
    tail_field_encodings: tuple[str, ...] = ()
    field_encodings: tuple[str, ...] = ()

    @property
    def width(self) -> int:
        return self.stop - self.start

    def canonical_payload(self) -> dict:
        return {
            "name": self.name,
            "start": self.start,
            "stop": self.stop,
            "width": self.width,
            "shape": list(self.shape),
            "field_names": list(self.field_names),
            "semantic_tags": list(self.semantic_tags),
            "row_width": self.row_width,
            "row_count": self.row_count,
            "row_offset": self.row_offset,
            "prefix_field_names": list(self.prefix_field_names),
            "prefix_field_encodings": list(
                self.prefix_field_encodings),
            "tail_field_names": list(self.tail_field_names),
            "tail_field_encodings": list(self.tail_field_encodings),
            "field_encodings": list(self.field_encodings),
        }


@dataclass(frozen=True)
class DualWorkerLayoutSpec:
    schema: str
    observation_dim: int
    controller_start: int
    p_skip_semantic_index: int
    segments: tuple[WireSegmentSpec, ...]
    excluded_high_entropy_fields: tuple[str, ...]
    banned_rng_tag_violations: tuple[str, ...]

    def canonical_payload(self) -> dict:
        return {
            "schema": self.schema,
            "observation_dim": self.observation_dim,
            "controller_start": self.controller_start,
            "p_skip_semantic_index": self.p_skip_semantic_index,
            "segments": [
                segment.canonical_payload() for segment in self.segments
            ],
            "excluded_high_entropy_fields": list(
                self.excluded_high_entropy_fields),
            "banned_rng_tag_violations": list(
                self.banned_rng_tag_violations),
        }


def _segment(
    name: str,
    bounds: slice,
    *,
    shape: tuple[int, ...],
    field_names: tuple[str, ...],
    semantic_tags: tuple[str, ...],
    row_width: int | None = None,
    row_count: int | None = None,
    row_offset: int = 0,
    prefix_field_names: tuple[str, ...] = (),
    prefix_field_encodings: tuple[str, ...] = (),
    tail_field_names: tuple[str, ...] = (),
    tail_field_encodings: tuple[str, ...] = (),
    field_encodings: tuple[str, ...] = (),
) -> WireSegmentSpec:
    return WireSegmentSpec(
        name=name,
        start=int(bounds.start),
        stop=int(bounds.stop),
        shape=shape,
        field_names=field_names,
        semantic_tags=semantic_tags,
        row_width=row_width,
        row_count=row_count,
        row_offset=row_offset,
        prefix_field_names=prefix_field_names,
        prefix_field_encodings=prefix_field_encodings,
        tail_field_names=tail_field_names,
        tail_field_encodings=tail_field_encodings,
        field_encodings=field_encodings,
    )


_PREFIX_SCALAR_FIELDS = (
    "current_layer_clock",
    "current_exhausted",
    "farm_scene_fraction",
    "time_remaining",
    "layer_kills",
    "legacy_layer_time",
    "dry_floor_remaining",
    "drink_latch",
    "skip_dry_probability",
)
_PREFIX_SCALAR_ENCODINGS = (
    "clip_0_1;divide_by:140",
    "binary_0_or_1",
    "clip_0_1;divide_by:1800",
    "clip_0_1;one_minus_divide_by:runtime_max_steps",
    "clip_0_1;divide_by:50",
    "clip_0_1;divide_by:1500",
    "clip_0_1;divide_by:25",
    "binary_0_or_1",
    "probability_0_to_1",
)
DUAL_WORKER_LAYOUT_SEGMENTS = (
    _segment(
        "legacy_v3",
        DUAL_WORKER_LEGACY_SLICE,
        shape=(298,),
        field_names=("legacy_v3_vector",),
        semantic_tags=("legacy", "actor", "critic"),
        field_encodings=("opaque_frozen_legacy_v3_vector",),
    ),
    _segment(
        "current_v4_base",
        DUAL_WORKER_CURRENT_V4_BASE_SLICE,
        shape=(295,),
        field_names=("current_v4_base_vector",),
        semantic_tags=("current", "actor", "critic", "wrapper_state"),
        field_encodings=("opaque_current_v4_base_vector",),
    ),
    _segment(
        "wrapper_scalars",
        slice(593, 602),
        shape=(9,),
        field_names=_PREFIX_SCALAR_FIELDS,
        # p_skip (absolute index 601) is removed by the actor's field-level
        # exclusion.  The other eight current wrapper scalars are legitimate
        # actor inputs and must not disappear through segment-wide tagging.
        semantic_tags=("current", "actor", "critic", "wrapper_state"),
        field_encodings=_PREFIX_SCALAR_ENCODINGS,
    ),
    _segment(
        "fuse_streak",
        DUAL_WORKER_FUSE_STREAK_SLICE,
        shape=(15,),
        field_names=tuple(f"action_{i}" for i in range(15)),
        semantic_tags=("current", "actor", "critic", "controller_memory"),
        field_encodings=("armed_repeat_count_plus_1_divide_by:25",) * 15,
    ),
    _segment(
        "action_mask",
        DUAL_WORKER_ACTION_MASK_SLICE,
        shape=(15,),
        field_names=tuple(f"action_{i}" for i in range(15)),
        semantic_tags=("current", "actor", "critic", "legality"),
        field_encodings=("binary_0_or_1",) * 15,
    ),
    _segment(
        "manager_mask",
        DUAL_WORKER_MANAGER_MASK_SLICE,
        shape=(3,),
        field_names=("farm", "dive", "resupply"),
        semantic_tags=("current", "actor", "critic", "legality"),
        field_encodings=("binary_0_or_1",) * 3,
    ),
    _segment(
        "controller_map",
        DUAL_WORKER_CONTROLLER_MAP_SLICE,
        shape=(
            len(CONTROLLER_SNAPSHOT_MAP_CHANNELS),
            CONTROLLER_SNAPSHOT_SIDE,
            CONTROLLER_SNAPSHOT_SIDE,
        ),
        field_names=CONTROLLER_SNAPSHOT_MAP_CHANNELS,
        semantic_tags=("controller", "actor", "critic", "spatial", "observable"),
        field_encodings=CONTROLLER_SNAPSHOT_MAP_FIELD_ENCODINGS,
    ),
    _segment(
        "controller_monsters",
        DUAL_WORKER_CONTROLLER_MONSTER_SLICE,
        shape=(CONTROLLER_SNAPSHOT_MONSTER_DIM,),
        field_names=CONTROLLER_SNAPSHOT_MONSTER_ROW_FIELDS,
        semantic_tags=("controller", "actor", "critic", "entity_rows", "combat"),
        row_width=CONTROLLER_SNAPSHOT_MONSTER_FIELDS,
        row_count=CONTROLLER_SNAPSHOT_MONSTER_LIMIT,
        tail_field_names=CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_FIELDS,
        tail_field_encodings=(
            CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_ENCODINGS),
        field_encodings=CONTROLLER_SNAPSHOT_MONSTER_ROW_ENCODINGS,
    ),
    _segment(
        "controller_missiles",
        DUAL_WORKER_CONTROLLER_MISSILE_SLICE,
        shape=(CONTROLLER_SNAPSHOT_MISSILE_DIM,),
        field_names=CONTROLLER_SNAPSHOT_MISSILE_ROW_FIELDS,
        semantic_tags=("controller", "actor", "critic", "entity_rows", "projectile"),
        row_width=CONTROLLER_SNAPSHOT_MISSILE_FIELDS,
        row_count=CONTROLLER_SNAPSHOT_MISSILE_LIMIT,
        tail_field_names=CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_FIELDS,
        tail_field_encodings=(
            CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_ENCODINGS),
        field_encodings=CONTROLLER_SNAPSHOT_MISSILE_ROW_ENCODINGS,
    ),
    _segment(
        "controller_belt",
        DUAL_WORKER_CONTROLLER_BELT_SLICE,
        shape=(
            CONTROLLER_SNAPSHOT_BELT_SLOTS,
            CONTROLLER_SNAPSHOT_BELT_KINDS,
        ),
        field_names=CONTROLLER_SNAPSHOT_BELT_ROW_FIELDS,
        semantic_tags=("controller", "actor", "critic", "categorical_rows", "resource"),
        row_width=CONTROLLER_SNAPSHOT_BELT_KINDS,
        row_count=CONTROLLER_SNAPSHOT_BELT_SLOTS,
        field_encodings=("binary_one_hot_0_or_1",)
        * CONTROLLER_SNAPSHOT_BELT_KINDS,
    ),
    _segment(
        "controller_exact",
        DUAL_WORKER_CONTROLLER_EXACT_SLICE,
        shape=(CONTROLLER_SNAPSHOT_EXACT_DIM,),
        field_names=CONTROLLER_SNAPSHOT_EXACT_FIELDS,
        semantic_tags=("controller", "actor", "critic", "player", "scene"),
        field_encodings=CONTROLLER_SNAPSHOT_EXACT_ENCODINGS,
    ),
    _segment(
        "controller_combat",
        DUAL_WORKER_CONTROLLER_COMBAT_SLICE,
        shape=(CONTROLLER_SNAPSHOT_COMBAT_DIM,),
        field_names=tuple(
            f"equipped.{field}"
            for field in CONTROLLER_SNAPSHOT_EQUIPPED_ROW_FIELDS
        ),
        semantic_tags=("controller", "actor", "critic", "combat", "equipment_rows"),
        row_width=CONTROLLER_SNAPSHOT_EQUIPPED_FIELDS,
        row_count=CONTROLLER_SNAPSHOT_EQUIPPED_SLOTS,
        row_offset=len(CONTROLLER_SNAPSHOT_COMBAT_PREFIX_FIELDS),
        prefix_field_names=CONTROLLER_SNAPSHOT_COMBAT_PREFIX_FIELDS,
        prefix_field_encodings=(
            CONTROLLER_SNAPSHOT_COMBAT_PREFIX_ENCODINGS),
        field_encodings=CONTROLLER_SNAPSHOT_EQUIPPED_ROW_ENCODINGS,
    ),
    _segment(
        "controller_sticky",
        DUAL_WORKER_CONTROLLER_STICKY_SLICE,
        shape=(CONTROLLER_SNAPSHOT_STICKY_DIM,),
        field_names=("present", "tile_dx", "tile_dy"),
        semantic_tags=("controller", "actor", "critic", "controller_memory"),
        field_encodings=(
            "binary_0_or_1",
            _divide_by(112.0),
            _divide_by(112.0),
        ),
    ),
    _segment(
        "controller_item_targets",
        DUAL_WORKER_CONTROLLER_ITEM_TARGET_SLICE,
        shape=(CONTROLLER_SNAPSHOT_ITEM_TARGET_DIM,),
        field_names=tuple(
            f"heal.{field}"
            for field in CONTROLLER_SNAPSHOT_HEAL_TARGET_FIELDS
        ) + tuple(
            f"gear.{field}"
            for field in CONTROLLER_SNAPSHOT_GEAR_TARGET_FIELDS
        ),
        semantic_tags=("controller", "actor", "critic", "action_target", "equipment"),
        field_encodings=(
            CONTROLLER_SNAPSHOT_HEAL_TARGET_ENCODINGS
            + CONTROLLER_SNAPSHOT_GEAR_TARGET_ENCODINGS
        ),
    ),
)

_EXCLUDED_HIGH_ENTROPY_FIELDS = (
    "monster.rnd_item_seed_hi",
    "monster.rnd_item_seed_lo",
    "monster.ai_seed_hi",
    "monster.ai_seed_lo",
    "missile.light_id",
    "missile.light",
    "missile.uniq_trans",
    "missile.anim_width",
    "missile.anim_width2",
    # These remain in diagnostic raw state but are deliberately excluded from
    # the actor wire.  Upstream aliases var1..var7 by MissileID: some are
    # transition counters/directions, while others are unlit absolute
    # coordinates or a Monster slot id.  lastCollisionTargetHash likewise
    # folds collision entity identity.  A uniform safe encoding is impossible
    # without a complete type-aware visibility contract.
    "missile.last_collision_target_hash",
    "missile.var1",
    "missile.var2",
    "missile.var3",
    "missile.var4",
    "missile.var5",
    "missile.var6",
    "missile.var7",
    "gear.seed_hi",
    "gear.seed_lo",
    "gear.create_info",
    "gear.base_id",
    "gear.misc_id",
    "gear.spell_id",
    "gear.charges",
    "gear.max_charges",
)
_BANNED_TAGS = frozenset({"rng", "drop_seed", "save_only"})


def _banned_policy_wire_violations(
    segments: tuple[WireSegmentSpec, ...],
) -> tuple[str, ...]:
    """Return actual tag/field leaks, not merely documentation mismatches."""
    violations = [
        f"{segment.name}:tag:{tag}"
        for segment in segments
        for tag in segment.semantic_tags
        if tag in _BANNED_TAGS
    ]
    namespace_by_segment = {
        "controller_monsters": "monster",
        "controller_missiles": "missile",
    }
    banned_fields = frozenset(_EXCLUDED_HIGH_ENTROPY_FIELDS)
    # Leaf matching is deliberate: a future refactor must not evade the guard
    # merely by moving ``ai_seed_hi`` into a generic combat segment and
    # forgetting its source namespace in field_names.
    namespace_bound_leaves = frozenset({
        "last_collision_target_hash",
        "var1", "var2", "var3", "var4", "var5", "var6", "var7",
    })
    banned_leaf_fields = frozenset(
        field.rpartition(".")[2]
        for field in banned_fields
        if field.rpartition(".")[2] not in namespace_bound_leaves
    )
    for segment in segments:
        namespace = namespace_by_segment.get(segment.name)
        for field in (
            segment.prefix_field_names
            + segment.field_names
            + segment.tail_field_names
        ):
            normalized = {field}
            if namespace is not None and not field.startswith(
                f"{namespace}."
            ):
                normalized.add(f"{namespace}.{field}")
            if field.startswith("equipped."):
                normalized.add(f"gear.{field.removeprefix('equipped.')}")
            if (
                normalized & banned_fields
                or field.rpartition(".")[2] in banned_leaf_fields
            ):
                violations.append(
                    f"{segment.name}:field:{field}")
    return tuple(violations)


_BANNED_RNG_TAG_VIOLATIONS = _banned_policy_wire_violations(
    DUAL_WORKER_LAYOUT_SEGMENTS)

DUAL_WORKER_LAYOUT = DualWorkerLayoutSpec(
    schema="diablogym-dual-worker-layout/4",
    observation_dim=DUAL_WORKER_OBSERVATION_DIM,
    controller_start=DUAL_WORKER_CONTROLLER_START,
    p_skip_semantic_index=DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE,
    segments=DUAL_WORKER_LAYOUT_SEGMENTS,
    excluded_high_entropy_fields=_EXCLUDED_HIGH_ENTROPY_FIELDS,
    banned_rng_tag_violations=_BANNED_RNG_TAG_VIOLATIONS,
)
DUAL_WORKER_LAYOUT_SPEC = DUAL_WORKER_LAYOUT


def _layout_canonical_json(layout: DualWorkerLayoutSpec) -> str:
    return json.dumps(
        layout.canonical_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _layout_sha256(layout: DualWorkerLayoutSpec) -> str:
    return hashlib.sha256(
        _layout_canonical_json(layout).encode("utf-8")
    ).hexdigest()


DUAL_WORKER_LAYOUT_CANONICAL_JSON = _layout_canonical_json(
    DUAL_WORKER_LAYOUT)
DUAL_WORKER_LAYOUT_SHA256 = _layout_sha256(DUAL_WORKER_LAYOUT)
DUAL_WORKER_LAYOUT_FROZEN_SHA256 = (
    "6463990a5732c366f19ae460576f74fc186cf4ad5ddb3015e704140f4c2586c9"
)


def _validate_layout_spec(layout: DualWorkerLayoutSpec) -> None:
    expected_start = 0
    for segment in layout.segments:
        if segment.start != expected_start or segment.stop <= segment.start:
            raise RuntimeError(
                "dual Worker layout segment 未无缝覆盖:"
                f"{segment.name}={segment.start}:{segment.stop},"
                f"expected_start={expected_start}"
            )
        if (
            not segment.shape
            or any(
                not isinstance(axis, int)
                or isinstance(axis, bool)
                or axis <= 0
                for axis in segment.shape
            )
            or math.prod(segment.shape) != segment.width
        ):
            raise RuntimeError(
                "dual Worker segment shape/width 未闭合:"
                f"{segment.name}.shape={segment.shape},"
                f"width={segment.width}"
            )
        if len(segment.field_encodings) != len(segment.field_names):
            raise RuntimeError(
                "dual Worker field encoding 数量与字段不一致:"
                f"{segment.name}={len(segment.field_encodings)}"
                f"!={len(segment.field_names)}"
            )
        if (
            len(segment.prefix_field_encodings)
            != len(segment.prefix_field_names)
        ):
            raise RuntimeError(
                "dual Worker prefix encoding 数量与字段不一致:"
                f"{segment.name}="
                f"{len(segment.prefix_field_encodings)}"
                f"!={len(segment.prefix_field_names)}"
            )
        if (
            len(segment.tail_field_encodings)
            != len(segment.tail_field_names)
        ):
            raise RuntimeError(
                "dual Worker tail encoding 数量与字段不一致:"
                f"{segment.name}={len(segment.tail_field_encodings)}"
                f"!={len(segment.tail_field_names)}"
            )
        if segment.row_width is None:
            if (
                segment.row_count is not None
                or segment.row_offset != 0
                or segment.prefix_field_names
                or segment.tail_field_names
            ):
                raise RuntimeError(
                    "dual Worker 非行 segment 含 row/prefix/tail 元数据:"
                    f"{segment.name}")
        else:
            if (
                segment.row_count is None
                or segment.row_width <= 0
                or segment.row_count <= 0
                or len(segment.field_names) != segment.row_width
                or segment.row_offset != len(segment.prefix_field_names)
                or (
                    segment.row_offset
                    + segment.row_count * segment.row_width
                    + len(segment.tail_field_names)
                    != segment.width
                )
            ):
                raise RuntimeError(
                    "dual Worker row/prefix/tail 维度未闭合:"
                    f"{segment.name}")
        expected_start = segment.stop
    if expected_start != layout.observation_dim:
        raise RuntimeError("dual Worker layout 未覆盖最终维度")
    if layout.p_skip_semantic_index != 601:
        raise RuntimeError("p_skip semantic index 漂移")
    field_violations = _banned_policy_wire_violations(layout.segments)
    if (
        layout.banned_rng_tag_violations
        or field_violations
    ):
        raise RuntimeError(
            "controller policy wire 含 RNG/drop/save-only 字段或 tag:"
            + ",".join(
                layout.banned_rng_tag_violations + field_violations)
        )


def _validate_layout() -> None:
    declared_rows = (
        ("monster row", CONTROLLER_SNAPSHOT_MONSTER_ROW_FIELDS),
        ("missile direct", CONTROLLER_SNAPSHOT_MISSILE_DIRECT_FIELDS),
        ("missile int32", CONTROLLER_SNAPSHOT_MISSILE_INT32_FIELDS),
        ("missile row", CONTROLLER_SNAPSHOT_MISSILE_ROW_FIELDS),
    )
    for label, names in declared_rows:
        duplicates = tuple(
            name for index, name in enumerate(names)
            if name in names[:index]
        )
        if duplicates:
            raise RuntimeError(
                f"controller wire {label} 含重复字段:"
                + ",".join(duplicates)
            )
    overlap = set(CONTROLLER_SNAPSHOT_MISSILE_DIRECT_FIELDS).intersection(
        CONTROLLER_SNAPSHOT_MISSILE_INT32_FIELDS)
    if overlap:
        raise RuntimeError(
            "controller wire missile direct/int32 重复编码:"
            + ",".join(sorted(overlap))
        )
    if CONTROLLER_SNAPSHOT_VECTOR_DIM != 12377:
        raise RuntimeError("controller wire 维度漂移")
    if DUAL_WORKER_OBSERVATION_DIM != 13012:
        raise RuntimeError("dual Worker 维度漂移")
    if DUAL_WORKER_LAYOUT_SHA256 != DUAL_WORKER_LAYOUT_FROZEN_SHA256:
        raise RuntimeError(
            "dual Worker canonical layout SHA 漂移:"
            f"{DUAL_WORKER_LAYOUT_SHA256}"
            f"!={DUAL_WORKER_LAYOUT_FROZEN_SHA256}"
        )
    _validate_layout_spec(DUAL_WORKER_LAYOUT)


_validate_layout()
