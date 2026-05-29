"""Gen 9 damage calculation, mechanically ported from
the Showdown reference implementation:5563-6938 (`_calc_damage_gen9`).

Pure-Python / numpy scalar implementation. Helpers
`_get_weather_type_multiplier`, `_get_terrain_type_multiplier`, and
`_get_effective_speed` are inlined as nested functions / module helpers.
"""

from __future__ import annotations

import math
import numpy as np

from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_FIELD,
    OFF_META,
    POKEMON_SIZE,
    M_ACTIVE0,
    F_WEATHER,
    F_TERRAIN,
    F_LAST_MOVE_0,
    F_LAST_MOVE_1,
    OFF_MOVES,
    M_ACTIVE_MOVE_ACTIONS_0,
    M_ACTIVE_MOVE_ACTIONS_1,
    ACTIVE_MOVE_ACTIONS_SEMI_INVUL,
    F_EXTENDED_VOLATILE_0,
    F_EXTENDED_VOLATILE_1,
    F_VOLATILE_0,
    F_VOLATILE_1,
    F_SCREENS_0,
    F_SCREENS_1,
    F_SUBSTITUTE_0,
    F_SUBSTITUTE_1,
    M_TERA_ORIG_TYPES_0,
    M_TERA_ORIG_TYPES_1,
    SCREEN_REFLECT_SHIFT,
    SCREEN_LIGHTSCREEN_SHIFT,
    SCREEN_AURORAVEIL_SHIFT,
    SCREEN_TAILWIND_SHIFT,
    SCREEN_MASK_3BIT,
    TYPE_NORMAL,
    TYPE_FIRE,
    TYPE_WATER,
    TYPE_ELECTRIC,
    TYPE_GRASS,
    TYPE_ICE,
    TYPE_FIGHTING,
    TYPE_POISON,
    TYPE_GROUND,
    TYPE_FLYING,
    TYPE_PSYCHIC,
    TYPE_BUG,
    TYPE_ROCK,
    TYPE_GHOST,
    TYPE_DRAGON,
    TYPE_DARK,
    TYPE_STEEL,
    TYPE_FAIRY,
    TYPE_UNKNOWN,
    CAT_PHYSICAL,
    CAT_STATUS,
    STATUS_NONE,
    STATUS_BURN,
    STATUS_POISON,
    STATUS_TOXIC,
    STATUS_PARALYSIS,
    STATUS_SLEEP,
    WEATHER_NONE,
    WEATHER_SUN,
    WEATHER_RAIN,
    WEATHER_SAND,
    WEATHER_SNOW,
    WEATHER_PRIMORDIAL_SEA,
    WEATHER_DESOLATE_LAND,
    TERRAIN_NONE,
    TERRAIN_ELECTRIC,
    TERRAIN_GRASSY,
    TERRAIN_PSYCHIC,
    TERRAIN_MISTY,
    EXT_VOL_LOCK_ON,
    EXT_VOL_FORESIGHT,
    EXT_VOL_FOCUS_ENERGY,
    EXT_VOL_ATTRACT,
    FLAG_CONTACT,
    FLAG_SOUND,
    FLAG_PUNCH,
    FLAG_BITE,
    FLAG_BULLET,
    FLAG_PULSE,
    FLAG_SLICING,
    FLAG_CHARGE,
    MOVE_STRUGGLE,
    MOVE_WEATHER_BALL,
    MOVE_TERRAIN_PULSE,
    MOVE_CHARGE,
    MOVE_SEISMIC_TOSS,
    MOVE_NIGHT_SHADE,
    MOVE_SUPER_FANG,
    MOVE_RUINATION,
    MOVE_FACADE,
    MOVE_DREAM_EATER,
    MOVE_FROST_BREATH,
    MOVE_STORM_THROW,
    MOVE_FISSURE,
    MOVE_GUILLOTINE,
    MOVE_HORN_DRILL,
    MOVE_SHEER_COLD,
    MOVE_GYRO_BALL,
    MOVE_BODY_PRESS,
    MOVE_HEAVY_SLAM,
    MOVE_HEAT_CRASH,
    MOVE_LOW_KICK,
    MOVE_GRASS_KNOT,
    MOVE_ELECTROBALL,
    MOVE_WATER_SPOUT,
    MOVE_ERUPTION,
    MOVE_DRAGON_ENERGY,
    MOVE_POPULATION_BOMB,
    MOVE_TRIPLE_AXEL,
    MOVE_ENDEAVOR,
    MOVE_EARTHQUAKE,
    MOVE_SURF,
    MOVE_FINAL_GAMBIT,
    MOVE_PSYWAVE,
    MOVE_COUNTER,
    MOVE_MIRROR_COAT,
    MOVE_METAL_BURST,
    EFFECT_RECOIL,
    EFFECT_KNOCK_OFF,
    ABILITY_LEVITATE,
    ABILITY_FLASH_FIRE,
    ABILITY_VOLT_ABSORB,
    ABILITY_WATER_ABSORB,
    ABILITY_HUGE_POWER,
    ABILITY_PURE_POWER,
    ABILITY_MULTISCALE,
    ABILITY_HUSTLE,
    ABILITY_COMPOUND_EYES,
    ABILITY_SAND_VEIL,
    ABILITY_SNOW_CLOAK,
    ABILITY_NO_GUARD,
    ABILITY_MOLD_BREAKER,
    ABILITY_TERAVOLT,
    ABILITY_TURBOBLAZE,
    ABILITY_STURDY,
    ABILITY_DISGUISE,
    ABILITY_ICE_FACE,
    ABILITY_UNAWARE,
    ABILITY_SCRAPPY,
    ABILITY_SAP_SIPPER,
    ABILITY_STORM_DRAIN,
    ABILITY_LIGHTNING_ROD,
    ABILITY_MOTOR_DRIVE,
    ABILITY_DRY_SKIN,
    ABILITY_WONDER_GUARD,
    ABILITY_SOUNDPROOF,
    ABILITY_BULLETPROOF,
    ABILITY_EARTH_EATER,
    ABILITY_WELL_BAKED_BODY,
    ABILITY_GUTS,
    ABILITY_TOXIC_BOOST,
    ABILITY_FLARE_BOOST,
    ABILITY_MARVEL_SCALE,
    ABILITY_TECHNICIAN,
    ABILITY_ADAPTABILITY,
    ABILITY_SHEER_FORCE,
    ABILITY_TOUGH_CLAWS,
    ABILITY_IRON_FIST,
    ABILITY_SHARPNESS,
    ABILITY_STRONG_JAW,
    ABILITY_MEGA_LAUNCHER,
    ABILITY_PUNK_ROCK,
    ABILITY_RECKLESS,
    ABILITY_STEELWORKER,
    ABILITY_NEUROFORCE,
    ABILITY_ANALYTIC,
    ABILITY_OVERGROW,
    ABILITY_BLAZE,
    ABILITY_TORRENT,
    ABILITY_SWARM,
    ABILITY_GORILLA_TACTICS,
    ABILITY_WATER_BUBBLE,
    ABILITY_THICK_FAT,
    ABILITY_FILTER,
    ABILITY_SOLID_ROCK,
    ABILITY_PRISM_ARMOR,
    ABILITY_ICE_SCALES,
    ABILITY_FLUFFY,
    ABILITY_PURIFYING_SALT,
    ABILITY_TINTED_LENS,
    ABILITY_FUR_COAT,
    ABILITY_PROTOSYNTHESIS,
    ABILITY_QUARK_DRIVE,
    ABILITY_SUPER_LUCK,
    ABILITY_SNIPER,
    ABILITY_INFILTRATOR,
    ABILITY_SOLAR_POWER,
    ABILITY_SKILL_LINK,
    ABILITY_SWIFT_SWIM,
    ABILITY_CHLOROPHYLL,
    ABILITY_SAND_RUSH,
    ABILITY_SLUSH_RUSH,
    ABILITY_QUICK_FEET,
    ABILITY_UNBURDEN,
    ABILITY_FLAME_BODY,
    ABILITY_STATIC,
    ABILITY_POISON_POINT,
    ABILITY_EFFECT_SPORE,
    ABILITY_LONG_REACH,
    ITEM_BOOSTER_ENERGY,
    ITEM_CHOICE_BAND,
    ITEM_CHOICE_SPECS,
    ITEM_CHOICE_SCARF,
    ITEM_LIFE_ORB,
    ITEM_EXPERT_BELT,
    ITEM_ASSAULT_VEST,
    ITEM_EVIOLITE,
    ITEM_AIR_BALLOON,
    ITEM_LOADED_DICE,
    ITEM_UTILITY_UMBRELLA,
    ITEM_FOCUS_SASH,
    ITEM_PROTECTIVE_PADS,
    ITEM_CHARCOAL,
    ITEM_MYSTIC_WATER,
    ITEM_MAGNET,
    ITEM_MIRACLE_SEED,
    ITEM_ROSE_INCENSE,
    ITEM_NEVER_MELT_ICE,
    ITEM_BLACK_BELT,
    ITEM_POISON_BARB,
    ITEM_SOFT_SAND,
    ITEM_SHARP_BEAK,
    ITEM_TWISTED_SPOON,
    ITEM_SILVER_POWDER,
    ITEM_HARD_STONE,
    ITEM_SPELL_TAG,
    ITEM_DRAGON_FANG,
    ITEM_BLACK_GLASSES,
    ITEM_METAL_COAT,
    ITEM_FAIRY_FEATHER,
    ITEM_SILK_SCARF,
    FLAG_BOOSTER_ENERGY_ACTIVE,
)
from pokepy.core.bitpack import extract_boost, get_status
from pokepy.mechanics.stats import get_boost_multiplier
from pokepy.effects.ability_suppression import effective_ability
from pokepy.effects.grounding import is_grounded
from pokepy.effects.misc import is_take_item_blocked_by_item_rule
from pokepy.utils.gen5_prng import Gen5PRNG
from pokepy.data.move_effects import EFFECT_SCREEN_BREAK


def _side_actions_offset(side: int) -> int:
    return OFF_MOVES + (
        M_ACTIVE_MOVE_ACTIONS_0 if int(side) == 0 else M_ACTIVE_MOVE_ACTIONS_1
    )


# ----------------------------------------------------------------------------
# Local constants (defined inline in the Showdown reference source, not promoted to constants.py)
# ----------------------------------------------------------------------------
ABILITY_REFRIGERATE = 174
ABILITY_PIXILATE = 182
ABILITY_AERILATE = 184
ABILITY_LIQUID_VOICE = 204
ABILITY_GALVANIZE = 206
ABILITY_MINDS_EYE = 300
ABILITY_BEADS_OF_RUIN = 284
ABILITY_SWORD_OF_RUIN = 285
ABILITY_TABLETS_OF_RUIN = 286
ABILITY_VESSEL_OF_RUIN = 287
SPECIES_TING_LU = 1003
SPECIES_WO_CHIEN = 1001
SPECIES_CHI_YU = 1004
ABILITY_SUPREME_OVERLORD = 293
ABILITY_SAND_FORCE = 159
ABILITY_TRANSISTOR = 262
ABILITY_DRAGONS_MAW = 263
ABILITY_ROCKY_PAYLOAD = 276
ABILITY_STAKEOUT = 198
ABILITY_DARK_AURA = 186
ABILITY_FAIRY_AURA = 187
ABILITY_DEFEATIST = 129
ABILITY_SURGE_SURFER = 207

ITEM_SCOPE_LENS = 232
ITEM_RAZOR_CLAW = 326
ITEM_WELLSPRING_MASK = 2407
ITEM_CORNERSTONE_MASK = 2406
ITEM_HEARTHFLAME_MASK = 2408
SPECIES_OGERPON_WELLSPRING = 2065
SPECIES_OGERPON_CORNERSTONE = 2066
SPECIES_OGERPON_HEARTHFLAME = 2067

# Showdown's elemental Plates are generic 1.2x type boosters for all holders,
# not just Arceus. Pokepy models the Arceus take-item restriction separately.
_TYPE_BOOST_PLATES = {
    TYPE_FIGHTING: frozenset({143}),  # Fist Plate
    TYPE_POISON: frozenset({516}),  # Toxic Plate
    TYPE_GROUND: frozenset({117}),  # Earth Plate
    TYPE_FLYING: frozenset({450}),  # Sky Plate
    TYPE_PSYCHIC: frozenset({291}),  # Mind Plate
    TYPE_BUG: frozenset({223}),  # Insect Plate
    TYPE_ROCK: frozenset({477}),  # Stone Plate
    TYPE_GHOST: frozenset({464}),  # Spooky Plate
    TYPE_DRAGON: frozenset({105}),  # Draco Plate
    TYPE_DARK: frozenset({110}),  # Dread Plate
    TYPE_STEEL: frozenset({225}),  # Iron Plate
    TYPE_FIRE: frozenset({146}),  # Flame Plate
    TYPE_WATER: frozenset({463}),  # Splash Plate
    TYPE_ELECTRIC: frozenset({572}),  # Zap Plate
    TYPE_GRASS: frozenset({282}),  # Meadow Plate
    TYPE_ICE: frozenset({220}),  # Icicle Plate
    TYPE_FAIRY: frozenset({610}),  # Pixie Plate
}

MOVE_SOLAR_BEAM = 76
MOVE_SOLAR_BLADE = 669
MOVE_HYDRO_STEAM = 876
MOVE_EXPANDING_FORCE = 797
MOVE_RISING_VOLTAGE = 804
MOVE_MISTY_EXPLOSION = 802
MOVE_HURRICANE = 542
MOVE_THUNDER = 87
MOVE_BLIZZARD = 59
MOVE_PSYSHOCK = 473
MOVE_TRI_ATTACK = 161
MOVE_PSYSTRIKE = 540
MOVE_SECRET_SWORD = 548
MOVE_IVY_CUDGEL = 904
MOVE_HEX = 506
MOVE_ACROBATICS = 512
MOVE_POLTERGEIST = 809
MOVE_LASH_OUT = 808
MOVE_LAST_RESPECTS = 854
MOVE_RAGE_FIST = 889
MOVE_AXE_KICK = 853
MOVE_GIGATON_HAMMER = 893
MOVE_ELECTRO_SHOT = 905
MOVE_GLAIVE_RUSH = 862
MOVE_FLAIL = 175
MOVE_REVERSAL = 179
MOVE_CRUSH_GRIP = 462
MOVE_WRING_OUT = 378
MOVE_VENOSHOCK = 474
MOVE_BOLT_BEAK = 754
MOVE_FISHIOUS_REND = 755
MOVE_ASSURANCE = 372
MOVE_REVENGE = 279
MOVE_AVALANCHE = 419
MOVE_PUNISHMENT = 386
MOVE_STORED_POWER = 500
MOVE_POWER_TRIP = 681
MOVE_PAYBACK = 371
MOVE_BRINE = 362
MOVE_FOUL_PLAY = 492
MOVE_WICKED_BLOW = 817
MOVE_SURGING_STRIKES = 818
MOVE_COLLISION_COURSE = 878
MOVE_ELECTRO_DRIFT = 879


def _show_modify(value: int, numerator: float, denominator: int = 1) -> int:
    """Showdown's `modify(value, num, denom)` exactly. Uses 4096-base rounding.

    Showdown source: sim/battle.ts modify():
        modifier = trunc(num * 4096 / denom)
        return trunc((trunc(value * modifier) + 2048 - 1) / 4096)

    For value=49, mod=1.5: Showdown=74, math.floor(49*1.5)=73. The half-up
    rounding adds ~1 damage on every modify step that lands near .5, fixing
    a consistent ~1-2% damage shortfall in pokepy's chained float math.
    """
    modifier = int(numerator * 4096 / denominator)
    return int((int(value * modifier) + 2047) // 4096)


class _ChainAccum:
    """Showdown's `runEvent('Modify*', ...)` chainModify accumulator.

    Hooks call `chainModify` which folds each modifier into a single 4096-base
    accumulator via `((prev * next) + 2048) >> 12`. Then `finalModify` applies
    `modify(value, accumulator)` ONCE at the end. This produces a different
    (and correct) result vs applying each modifier sequentially via _show_modify.

    Example:
      Sequential: modify(49, 1.5) = 74, then modify(74, 1.5) = 111
      Accumulator: chainModify(1.5)*chainModify(1.5) = 9216/4096 (exactly 2.25),
                   then modify(49, 9216/4096) = 110.

    Pokepy used to apply chains via repeated _show_modify which was off-by-one
    on chained modifiers. This class fixes that to match Showdown exactly.
    """

    __slots__ = ("accum",)

    def __init__(self):
        self.accum = 4096  # represents 1.0 in 4096-base

    def chain(self, numerator: float, denominator: int = 1) -> "_ChainAccum":
        """Add a modifier to the chain. Equivalent to Showdown's chainModify()."""
        next_mod = int(numerator * 4096 / denominator)
        # Showdown: ((prev * next) + 2048) >> 12   (round-to-nearest, .5 down for ties via int trunc)
        self.accum = ((self.accum * next_mod) + 2048) >> 12
        return self

    def chain_if(
        self, cond: bool, numerator: float, denominator: int = 1
    ) -> "_ChainAccum":
        if cond:
            return self.chain(numerator, denominator)
        return self

    def apply(self, value: int) -> int:
        """Apply the accumulated modifier to a value via Showdown's final modify()."""
        if self.accum == 4096:
            return value
        # Showdown final modify: trunc((trunc(value * accum) + 2048 - 1) / 4096)
        return int((int(value * self.accum) + 2047) // 4096)


def _chain_bp(value: int, mult: float) -> int:
    """Backwards-compatible single-modifier helper used outside chain accumulators.
    Maps a float multiplier to the Showdown chainModify numerator and applies
    one modify() call.
    """
    if mult == 1.0:
        return value
    if mult == 1.3:
        return _show_modify(value, 5325, 4096)
    if mult == 1.2:
        return _show_modify(value, 4915, 4096)
    if mult == 1.5:
        return _show_modify(value, 6144, 4096)
    if mult == 1.25:
        return _show_modify(value, 5120, 4096)
    if mult == 1.33 or mult == 1.3333333333333333:
        return _show_modify(value, 5448, 4096)
    if mult == 0.75:
        return _show_modify(value, 3072, 4096)
    if mult == 0.5:
        return _show_modify(value, 2048, 4096)
    if mult == 2.0:
        return _show_modify(value, 8192, 4096)
    return _show_modify(value, mult)


def _bp_chain_add_to(accum: _ChainAccum, mult: float) -> None:
    """Adds a float-multiplier modifier into a _ChainAccum, mapping common
    pokepy float values to Showdown's exact chainModify numerators.
    """
    if mult == 1.0:
        return
    if mult == 1.3:
        accum.chain(5325, 4096)
        return
    if mult == 1.2:
        accum.chain(4915, 4096)
        return
    if mult == 1.5:
        accum.chain(6144, 4096)
        return
    if mult == 1.25:
        accum.chain(5120, 4096)
        return
    if mult == 1.33 or mult == 1.3333333333333333:
        accum.chain(5448, 4096)
        return
    if mult == 0.75:
        accum.chain(3072, 4096)
        return
    if mult == 0.5:
        accum.chain(2048, 4096)
        return
    if mult == 2.0:
        accum.chain(8192, 4096)
        return
    accum.chain(mult, 1)


def _weather_suppressed(battle: np.ndarray) -> bool:
    """Air Lock and Cloud Nine on EITHER active mon suppress all weather effects."""
    from pokepy.core.constants import (
        ABILITY_AIR_LOCK,
        ABILITY_CLOUD_NINE,
        OFF_SIDE0,
        OFF_SIDE1,
        OFF_META,
        M_ACTIVE0,
        M_ACTIVE1,
        POKEMON_SIZE,
    )

    a0 = int(battle[OFF_META + M_ACTIVE0])
    a1 = int(battle[OFF_META + M_ACTIVE1])
    off0 = OFF_SIDE0 + a0 * POKEMON_SIZE
    off1 = OFF_SIDE1 + a1 * POKEMON_SIZE
    ab0 = effective_ability(battle, off0, off1)
    ab1 = effective_ability(battle, off1, off0)
    return ab0 in (ABILITY_AIR_LOCK, ABILITY_CLOUD_NINE) or ab1 in (
        ABILITY_AIR_LOCK,
        ABILITY_CLOUD_NINE,
    )


def _effective_weather(battle: np.ndarray) -> int:
    """Returns weather, or WEATHER_NONE if suppressed by Air Lock / Cloud Nine."""
    if _weather_suppressed(battle):
        return WEATHER_NONE
    return int(battle[OFF_FIELD + F_WEATHER])


def _effective_weather_for_pokemon(battle: np.ndarray, pokemon_offset: int) -> int:
    """Showdown-style `pokemon.effectiveWeather()` for an active battler."""
    weather = _effective_weather(battle)
    if weather == WEATHER_NONE:
        return WEATHER_NONE
    if int(battle[int(pokemon_offset) + 6]) == ITEM_UTILITY_UMBRELLA and weather in (
        WEATHER_SUN,
        WEATHER_RAIN,
        WEATHER_PRIMORDIAL_SEA,
        WEATHER_DESOLATE_LAND,
    ):
        return WEATHER_NONE
    return weather


def _get_weather_type_multiplier(weather: int, move_type: int) -> float:
    if weather == WEATHER_SUN:
        if move_type == TYPE_FIRE:
            return 1.5
        if move_type == TYPE_WATER:
            return 0.5
        return 1.0
    if weather == WEATHER_RAIN:
        if move_type == TYPE_WATER:
            return 1.5
        if move_type == TYPE_FIRE:
            return 0.5
        return 1.0
    return 1.0


def _get_terrain_type_multiplier(
    battle: np.ndarray,
    move_type: int,
    atk_offset: int,
    def_offset: int,
    move_id: int,
) -> float:
    terrain = int(battle[OFF_FIELD + F_TERRAIN])
    atk_grounded = is_grounded(battle, atk_offset, def_offset)
    def_grounded = is_grounded(battle, def_offset, atk_offset)

    if terrain == TERRAIN_ELECTRIC:
        return 1.3 if (move_type == TYPE_ELECTRIC and atk_grounded) else 1.0
    if terrain == TERRAIN_GRASSY:
        MOVE_BULLDOZE = 523
        MOVE_MAGNITUDE = 222
        is_ground = move_id in (MOVE_EARTHQUAKE, MOVE_BULLDOZE, MOVE_MAGNITUDE)
        if is_ground and def_grounded:
            return 0.5
        if move_type == TYPE_GRASS and atk_grounded:
            return 1.3
        return 1.0
    if terrain == TERRAIN_PSYCHIC:
        return 1.3 if (move_type == TYPE_PSYCHIC and atk_grounded) else 1.0
    if terrain == TERRAIN_MISTY:
        return 0.5 if (move_type == TYPE_DRAGON and def_grounded) else 1.0
    return 1.0


def _get_effective_speed(battle: np.ndarray, pokemon_offset: int) -> int:
    base_speed = float(battle[pokemon_offset + 11])
    item = int(battle[pokemon_offset + 6])
    ability = effective_ability(battle, pokemon_offset)
    status_field = int(battle[pokemon_offset + 12])
    boosts14 = int(battle[pokemon_offset + 14])
    weather = _effective_weather(battle)

    spe_boost = extract_boost(boosts14, 0)
    boosted_speed = base_speed * float(get_boost_multiplier(spe_boost))

    is_side0 = pokemon_offset < OFF_SIDE1
    screens_offset = (
        (OFF_FIELD + F_SCREENS_0) if is_side0 else (OFF_FIELD + F_SCREENS_1)
    )
    screens = int(battle[screens_offset])
    tailwind = ((screens >> SCREEN_TAILWIND_SHIFT) & SCREEN_MASK_3BIT) > 0
    tailwind_mult = 2.0 if tailwind else 1.0

    scarf_mult = 1.5 if item == ITEM_CHOICE_SCARF else 1.0

    weather_speed_active = (
        (ability == ABILITY_SWIFT_SWIM and weather == WEATHER_RAIN)
        or (ability == ABILITY_CHLOROPHYLL and weather == WEATHER_SUN)
        or (ability == ABILITY_SAND_RUSH and weather == WEATHER_SAND)
        or (ability == ABILITY_SLUSH_RUSH and weather == WEATHER_SNOW)
    )
    weather_mult = 2.0 if weather_speed_active else 1.0

    terrain = int(battle[OFF_FIELD + F_TERRAIN])
    if ability == ABILITY_SURGE_SURFER and terrain == TERRAIN_ELECTRIC:
        weather_mult *= 2.0

    status = get_status(status_field)
    is_statused = status != STATUS_NONE
    has_quick_feet = ability == ABILITY_QUICK_FEET
    quick_feet_mult = 1.5 if (has_quick_feet and is_statused) else 1.0

    flags = int(battle[pokemon_offset + 15])
    had_item = (flags & 0x80) != 0
    item_consumed = (item == 0) and had_item
    unburden_active = (ability == ABILITY_UNBURDEN) and item_consumed
    unburden_mult = 2.0 if unburden_active else 1.0

    has_paradox = ability in (ABILITY_PROTOSYNTHESIS, ABILITY_QUARK_DRIVE)
    booster_consumed = has_paradox and ((flags & FLAG_BOOSTER_ENERGY_ACTIVE) != 0)
    paradox_active = (
        (ability == ABILITY_PROTOSYNTHESIS and weather == WEATHER_SUN)
        or (ability == ABILITY_QUARK_DRIVE and terrain == TERRAIN_ELECTRIC)
        or booster_consumed
    )
    paradox_spd_mult = (
        1.5
        if (paradox_active and _get_paradox_best_stat(battle, pokemon_offset) == "spe")
        else 1.0
    )

    paralysis_mult = 0.5 if (status == STATUS_PARALYSIS and not has_quick_feet) else 1.0

    # Iron Ball halves the holder's speed via onModifySpe chainModify(0.5).
    # Showdown data/items.ts:3057-3058. Item id 224.
    iron_ball_mult = 0.5 if item == 224 else 1.0

    return int(
        boosted_speed
        * tailwind_mult
        * scarf_mult
        * weather_mult
        * quick_feet_mult
        * unburden_mult
        * paradox_spd_mult
        * paralysis_mult
        * iron_ball_mult
    )


def _apply_stage_to_stat(base_stat: int, boost: int) -> int:
    boost = max(-6, min(6, int(boost)))
    if boost >= 0:
        return (int(base_stat) * (2 + boost)) // 2
    return (int(base_stat) * 2) // (2 - boost)


_PARADOX_STAT_MASK = 0x6010
_PARADOX_STAT_ATK = 0x0010
_PARADOX_STAT_DEF = 0x2000
_PARADOX_STAT_SPA = 0x2010
_PARADOX_STAT_SPD = 0x4000
_PARADOX_STAT_SPE = 0x4010


def _get_live_paradox_best_stat(battle: np.ndarray, pokemon_offset: int) -> str:
    """Mirror Showdown's getBestStat(false, true) for paradox abilities.

    This compares the holder's current stage-adjusted stats without the
    downstream onModifyAtk/SpA/Spe paradox boost itself. That means switch-in
    stage changes like Sticky Web can change which stat wins.
    """

    p_off = int(pokemon_offset)
    boosts13 = int(battle[p_off + 13])
    boosts14 = int(battle[p_off + 14])

    values = [
        (
            "atk",
            _apply_stage_to_stat(int(battle[p_off + 7]), extract_boost(boosts13, 0)),
        ),
        (
            "def",
            _apply_stage_to_stat(int(battle[p_off + 8]), extract_boost(boosts13, 4)),
        ),
        (
            "spa",
            _apply_stage_to_stat(int(battle[p_off + 9]), extract_boost(boosts13, 8)),
        ),
        (
            "spd",
            _apply_stage_to_stat(int(battle[p_off + 10]), extract_boost(boosts13, 12)),
        ),
        (
            "spe",
            _apply_stage_to_stat(int(battle[p_off + 11]), extract_boost(boosts14, 0)),
        ),
    ]

    best_name, best_value = values[0]
    for name, value in values[1:]:
        if value > best_value:
            best_name, best_value = name, value
    return best_name


def _get_paradox_best_stat(battle: np.ndarray, pokemon_offset: int) -> str:
    p_off = int(pokemon_offset)
    encoded = int(battle[p_off + 15]) & _PARADOX_STAT_MASK
    if encoded == _PARADOX_STAT_ATK:
        return "atk"
    if encoded == _PARADOX_STAT_DEF:
        return "def"
    if encoded == _PARADOX_STAT_SPA:
        return "spa"
    if encoded == _PARADOX_STAT_SPD:
        return "spd"
    if encoded == _PARADOX_STAT_SPE:
        return "spe"
    return _get_live_paradox_best_stat(battle, p_off)


def calc_damage_gen9(
    battle: np.ndarray,
    atk_side: int,
    move_idx: int,
    player_moves: np.ndarray,
    opp_moves: np.ndarray,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    is_moving_last: bool = False,
    override_move_id: int = -1,
    gen5_prng=None,
    out_meta: dict = None,
    target_hurt_this_turn: bool = False,
    target_newly_switched: bool = False,
    user_hurt_by_target_this_turn: bool = False,
    suppress_attacker_item: bool = False,
    suppress_attacker_boosts: bool = False,
    override_field_atk_ability: int = -1,
    override_field_def_ability: int = -1,
) -> int:
    """Returns damage as a Python int. Uses gen5_prng (Showdown LCG) for all rolls.

    If `out_meta` is a dict, sets `out_meta['num_hits']` to the actual hit
    count rolled for multi-hit moves (1 for single-hit moves). The engine
    uses this to apply per-hit defender abilities (Rough Skin, Iron Barbs,
    Rocky Helmet) the correct number of times — Showdown fires
    `onDamagingHit` inside `spreadMoveHit`, which runs once per hit
    (sim/battle-actions.ts:1139), but pokepy applies aggregate damage so
    those triggers fire once unless the engine multiplies by num_hits.

    For multihit contact moves, strict parity also needs per-hit contact
    status abilities (Flame Body / Static / Poison Point / Effect Spore /
    Cute Charm, plus Poison Touch on the attacker) to consume PRNG inside
    the hit loop. When that path is used, `out_meta` also records the
    resolved contact-status result so the engine can materialize it later
    without rerolling.
    """
    if gen5_prng is None:
        gen5_prng = Gen5PRNG((1, 1, 1, 1))

    atk_side = int(atk_side)
    move_idx = int(move_idx)
    def_side = 1 - atk_side

    atk_active = int(battle[OFF_META + M_ACTIVE0 + atk_side])
    def_active = int(battle[OFF_META + M_ACTIVE0 + def_side])

    atk_base = OFF_SIDE0 if atk_side == 0 else OFF_SIDE1
    def_base = OFF_SIDE0 if def_side == 0 else OFF_SIDE1
    atk_offset = atk_base + atk_active * POKEMON_SIZE
    def_offset = def_base + def_active * POKEMON_SIZE
    atk_item_live = 0 if suppress_attacker_item else int(battle[atk_offset + 6])
    atk_flags_live = int(battle[atk_offset + 15])

    if override_move_id is not None and override_move_id >= 0:
        move_id = int(override_move_id)
    else:
        moves = player_moves if atk_side == 0 else opp_moves
        move_id = int(moves[atk_active, move_idx])
    move_id = max(0, min(int(game_data.move_base_power.shape[0]) - 1, move_id))

    bp = int(game_data.move_base_power[move_id])
    move_type = int(game_data.move_type[move_id])
    move_cat = int(game_data.move_category[move_id])
    is_struggle = move_id == MOVE_STRUGGLE
    if is_struggle:
        # Showdown's move definition rewrites Struggle to type '???' in
        # onModifyMove before the later ModifyType pipeline, so it never gets
        # STAB, type-boost item/ability hooks, or non-neutral effectiveness.
        move_type = TYPE_UNKNOWN
    MOVE_BEAT_UP = 251
    beat_up_bps: list[int] | None = None

    # Raging Bull: type changes per Tauros-Paldea form. Also breaks screens.
    MOVE_RAGING_BULL = 873
    if move_id == MOVE_RAGING_BULL:
        atk_species_rb = int(battle[atk_offset + 0])
        # Combat (2036): Fighting; Blaze (2035): Fire; Aqua (2034): Water
        if atk_species_rb == 2036:
            move_type = TYPE_FIGHTING
        elif atk_species_rb == 2035:
            move_type = TYPE_FIRE
        elif atk_species_rb == 2034:
            move_type = TYPE_WATER

    # Tera Blast: changes type to user's tera type when terastallized.
    # Becomes physical (cat=1) if user's effective Atk > effective SpA.
    # Stellar tera also bumps BP to 100 (already 80 base).
    MOVE_TERA_BLAST = 851
    if move_id == MOVE_TERA_BLAST:
        atk_flags_tb = int(battle[atk_offset + 15])
        is_terad_tb = (atk_flags_tb & 0x8) != 0
        if is_terad_tb:
            tera_nibble = (int(battle[atk_offset + 14]) >> 12) & 0xF
            move_type = tera_nibble
            # Stellar (18) bumps BP to 100. Source: data/moves.ts:19950-19955
            # basePowerCallback returns 100 when terastallized === 'Stellar'.
            if tera_nibble == 18:
                bp = 100
            # Compare effective atk vs spa with current boosts
            atk_b13_tb = int(battle[atk_offset + 13])
            atk_raw = int(battle[atk_offset + 7])
            spa_raw = int(battle[atk_offset + 9])
            atk_b_stage = (atk_b13_tb & 0xF) - 6
            spa_b_stage = ((atk_b13_tb >> 8) & 0xF) - 6
            atk_eff = int(atk_raw * float(get_boost_multiplier(atk_b_stage)))
            spa_eff = int(spa_raw * float(get_boost_multiplier(spa_b_stage)))
            if atk_eff > spa_eff:
                move_cat = CAT_PHYSICAL  # 1

    # Revelation Dance: Showdown resolves the move type from the user's live
    # primary type with pokemon.getTypes()[0] in onModifyType. The packed
    # battle state already tracks the runtime type tuple, so this must happen
    # before STAB / effectiveness / later ModifyType-family hooks.
    MOVE_REVELATION_DANCE = 686
    if move_id == MOVE_REVELATION_DANCE:
        atk_types_rd = int(battle[atk_offset + 4]) & 0xFFFF
        move_type = int(atk_types_rd & 0xFF)
        if move_type == 0xFF:
            move_type = TYPE_NORMAL

    if move_id == MOVE_BEAT_UP:
        # Showdown moves.ts:beatup onModifyMove/basePowerCallback:
        # - eligible hits are self plus each non-fainted, status-free ally
        # - each hit's base power is 5 + floor(ally_base_atk / 10)
        beat_up_bps = []
        for slot in range(6):
            slot_off = atk_base + slot * POKEMON_SIZE
            slot_hp = int(battle[slot_off + 1])
            slot_status = get_status(int(battle[slot_off + 12]))
            if slot == atk_active or (slot_hp > 0 and slot_status == STATUS_NONE):
                species_id = max(
                    0,
                    min(
                        int(game_data.species_base_stats.shape[0]) - 1,
                        int(battle[slot_off + 0]),
                    ),
                )
                base_atk = int(game_data.species_base_stats[species_id, 1])
                beat_up_bps.append(5 + base_atk // 10)
        bp = beat_up_bps[0] if beat_up_bps else 0

    # -ate abilities. Type change happens here; the 4915/4096 BP boost is
    # added to the BP chainModify accumulator below (Showdown abilities.ts:
    # aerilate/refrigerate/pixilate/galvanize use chainModify([4915, 4096])
    # via onBasePowerPriority 23, which folds into runEvent('BasePower')).
    # Showdown's onModifyType EXCLUDES: judgment, multiattack, naturalgift,
    # revelationdance, technoblast, terrainpulse, weatherball, plus Z-moves
    # and terastallized Tera Blast. pokepy adds weatherball + terrainpulse
    # (the OU-relevant ones) to the exclusion list; Tera Blast tera-blocked
    # below via the is_terad check in its own block.
    atk_ability_ate = effective_ability(battle, atk_offset, def_offset)
    is_normal_move = move_type == TYPE_NORMAL
    _ATE_EXCLUDED_MOVES = (
        MOVE_WEATHER_BALL,
        MOVE_TERRAIN_PULSE,
        MOVE_TERA_BLAST,
        # judgment=246, multiattack=719, naturalgift=363, revelationdance=686,
        # technoblast=546
        246,
        719,
        363,
        686,
        546,
    )
    ate_type_change = (
        atk_ability_ate
        in (ABILITY_REFRIGERATE, ABILITY_PIXILATE, ABILITY_AERILATE, ABILITY_GALVANIZE)
        and is_normal_move
        and bp > 0
        and move_id not in _ATE_EXCLUDED_MOVES
    )
    if ate_type_change:
        if atk_ability_ate == ABILITY_REFRIGERATE:
            move_type = TYPE_ICE
        elif atk_ability_ate == ABILITY_PIXILATE:
            move_type = TYPE_FAIRY
        elif atk_ability_ate == ABILITY_AERILATE:
            move_type = TYPE_FLYING
        elif atk_ability_ate == ABILITY_GALVANIZE:
            move_type = TYPE_ELECTRIC
    # Liquid Voice uses the same onModifyType hook family as the -ate
    # abilities, but it rewrites any sound move to Water regardless of base
    # type. Showdown abilities.ts:liquidvoice.
    if (
        atk_ability_ate == ABILITY_LIQUID_VOICE
        and (int(game_data.move_flags[move_id]) & FLAG_SOUND) != 0
    ):
        move_type = TYPE_WATER

    # Ivy Cudgel: type changes based on Ogerpon mask
    atk_item_for_ivy = atk_item_live
    if move_id == MOVE_IVY_CUDGEL:
        if atk_item_for_ivy == ITEM_WELLSPRING_MASK:
            move_type = TYPE_WATER
        elif atk_item_for_ivy == ITEM_CORNERSTONE_MASK:
            move_type = TYPE_ROCK
        elif atk_item_for_ivy == ITEM_HEARTHFLAME_MASK:
            move_type = TYPE_FIRE

    # Weather Ball / Terrain Pulse: type override from ModifyMove/ModifyType
    # event. Showdown (sim/battle-actions.ts:432-441) runs these BEFORE the
    # getDamage STAB-min check (line 1657), BEFORE the type effectiveness
    # computation, and BEFORE the STAB multiplier. Pokepy used to apply these
    # later (after the STAB-min block) causing mismatches e.g. Fire-tera
    # Weather Ball in sun not getting the BP<60 → 60 bump because move_type
    # was still Normal at the STAB-min check.
    _weather_for_type = _effective_weather(battle)
    if move_id == MOVE_WEATHER_BALL:
        if _weather_for_type == WEATHER_SUN:
            move_type = TYPE_FIRE
        elif _weather_for_type == WEATHER_RAIN:
            move_type = TYPE_WATER
        elif _weather_for_type == WEATHER_SNOW:
            move_type = TYPE_ICE
        elif _weather_for_type == WEATHER_SAND:
            move_type = TYPE_ROCK
        elif _weather_for_type == WEATHER_PRIMORDIAL_SEA:
            move_type = TYPE_WATER
        elif _weather_for_type == WEATHER_DESOLATE_LAND:
            move_type = TYPE_FIRE
    if move_id == MOVE_TERRAIN_PULSE:
        _terrain_for_type = int(battle[OFF_FIELD + F_TERRAIN])
        if _terrain_for_type == TERRAIN_ELECTRIC:
            move_type = TYPE_ELECTRIC
        elif _terrain_for_type == TERRAIN_GRASSY:
            move_type = TYPE_GRASS
        elif _terrain_for_type == TERRAIN_PSYCHIC:
            move_type = TYPE_PSYCHIC
        elif _terrain_for_type == TERRAIN_MISTY:
            move_type = TYPE_FAIRY

    # Tera STAB minimum BP boost — Showdown's `getDamage` (battle-actions.ts:1657-1664):
    # when terastallized AND the move is a STAB move (matches tera type, or for
    # Stellar matches a base type) AND BP < 60 AND priority <= 0 AND not multihit,
    # raise BP to 60. Excludes variable-BP moves like Dragon Energy.
    _tera_flags = int(battle[atk_offset + 15])
    _is_terad = (_tera_flags & 0x8) != 0
    if _is_terad and bp > 0 and bp < 60:
        _tera_t = (int(battle[atk_offset + 14]) >> 12) & 0xF
        _move_pri = int(game_data.move_priority[move_id])
        _hits_max = int(move_effects.hits_max[move_id])
        if _move_pri <= 0 and _hits_max <= 1:
            # STAB requires the move type matches the (post-tera) tera type, OR
            # for Stellar (18) the move type is one of the user's original types.
            _is_stab_for_tera_min = False
            if _tera_t == 18:  # Stellar
                _orig_types_field = int(
                    battle[
                        OFF_META
                        + (
                            M_TERA_ORIG_TYPES_0
                            if atk_offset < OFF_SIDE1
                            else M_TERA_ORIG_TYPES_1
                        )
                    ]
                )
                _ot1 = _orig_types_field & 0xFF
                _ot2 = (_orig_types_field >> 8) & 0xFF
                _is_stab_for_tera_min = (move_type == _ot1) or (move_type == _ot2)
            else:
                _is_stab_for_tera_min = move_type == _tera_t
            if _is_stab_for_tera_min:
                bp = 60

    # --- Early type-immunity short-circuit -----------------------------
    # Showdown's hit-step ordering runs `hitStepTypeImmunity` BEFORE
    # `hitStepAccuracy` (sim/battle-actions.ts:460-485, gen 7+). When the
    # target is type-immune (Ground vs Flying, Ghost immunity, Levitate,
    # Air Balloon, etc.), the move is dropped before any PRNG frame is
    # consumed — no accuracy roll, no crit roll, no damage roll. Pokepy
    # used to roll all 3 frames unconditionally, drifting the LCG by 3
    # frames vs Showdown on every immune hit.
    #
    # We compute a fast "definitely type-immune" check here that mirrors
    # the key cases: base type-chart immunity, Levitate, Air Balloon,
    # Thousand Arrows override, Future Sight type-immunity bypass, Iron
    # Ball for Flying + Ground. Ability-gated immunities (Flash Fire,
    # Volt Absorb, etc.) also skip the rolls in Showdown — match them.
    _def_types_imm = int(battle[def_offset + 4])
    _def_t1_imm = _def_types_imm & 0xFF
    _def_t2_imm = (_def_types_imm >> 8) & 0xFF
    _def_ability_imm = effective_ability(battle, def_offset, atk_offset)
    _def_item_imm = int(battle[def_offset + 6])
    _atk_ability_imm = effective_ability(battle, atk_offset, def_offset)
    _has_mold_breaker_imm = _atk_ability_imm in (
        ABILITY_MOLD_BREAKER,
        ABILITY_TERAVOLT,
        ABILITY_TURBOBLAZE,
    )
    # Base type chart lookup
    _eff1 = float(type_chart[_def_t1_imm, move_type])
    _eff2 = (
        1.0 if _def_t2_imm == _def_t1_imm else float(type_chart[_def_t2_imm, move_type])
    )
    _tm_imm = _eff1 * _eff2
    # Ring Target / Gravity / Iron Ball / Thousand Arrows override for
    # Flying + Ground. Thousand Arrows (615): forces Flying vuln to Ground.
    if (
        move_id == 615
        and move_type == TYPE_GROUND
        and (_def_t1_imm == TYPE_FLYING or _def_t2_imm == TYPE_FLYING)
    ):
        if _tm_imm == 0.0:
            _tm_imm = 1.0
    # Future Sight / Doom Desire bypass type immunity (rare in OU).
    if move_id in (248, 353) and _tm_imm == 0.0:
        _tm_imm = 1.0
    # Iron Ball grounds Flying (items.ts onNegateImmunity: Ground).
    _ITEM_IRON_BALL_IMM = 224
    if (
        _def_item_imm == _ITEM_IRON_BALL_IMM
        and move_type == TYPE_GROUND
        and (_def_t1_imm == TYPE_FLYING or _def_t2_imm == TYPE_FLYING)
    ):
        _tm_imm = 1.0
    # Levitate / Air Balloon vs Ground. Mold Breaker negates Levitate
    # (abilities are `breakable`) but NOT Air Balloon (item).
    if (
        (not _has_mold_breaker_imm)
        and _def_ability_imm == ABILITY_LEVITATE
        and move_type == TYPE_GROUND
    ):
        _tm_imm = 0.0
    if _def_item_imm == ITEM_AIR_BALLOON and move_type == TYPE_GROUND:
        _tm_imm = 0.0
    # Ability-gated immunities (Flash Fire, Volt Absorb, Water Absorb,
    # Sap Sipper, Storm Drain, Lightning Rod, Motor Drive, Dry Skin water,
    # Earth Eater, Well-Baked Body). All `breakable` → Mold Breaker
    # pierces. Same set as the block at line 1057-1077 below.
    if not _has_mold_breaker_imm:
        if _def_ability_imm == ABILITY_FLASH_FIRE and move_type == TYPE_FIRE:
            _tm_imm = 0.0
        elif _def_ability_imm == ABILITY_VOLT_ABSORB and move_type == TYPE_ELECTRIC:
            _tm_imm = 0.0
        elif _def_ability_imm == ABILITY_WATER_ABSORB and move_type == TYPE_WATER:
            _tm_imm = 0.0
        elif _def_ability_imm == ABILITY_SAP_SIPPER and move_type == TYPE_GRASS:
            _tm_imm = 0.0
        elif _def_ability_imm == ABILITY_STORM_DRAIN and move_type == TYPE_WATER:
            _tm_imm = 0.0
        elif _def_ability_imm == ABILITY_LIGHTNING_ROD and move_type == TYPE_ELECTRIC:
            _tm_imm = 0.0
        elif _def_ability_imm == ABILITY_MOTOR_DRIVE and move_type == TYPE_ELECTRIC:
            _tm_imm = 0.0
        elif _def_ability_imm == ABILITY_DRY_SKIN and move_type == TYPE_WATER:
            _tm_imm = 0.0
        elif _def_ability_imm == ABILITY_EARTH_EATER and move_type == TYPE_GROUND:
            _tm_imm = 0.0
        elif _def_ability_imm == ABILITY_WELL_BAKED_BODY and move_type == TYPE_FIRE:
            _tm_imm = 0.0
    # Soundproof / Bulletproof: immune to sound/bullet moves respectively.
    # Must check here in the early-immunity section because Showdown checks
    # these in hitStepTypeImmunity BEFORE accuracy/crit/damage PRNG rolls.
    # Both are breakable (suppressed by Mold Breaker/Teravolt/Turboblaze).
    _move_flags_imm = int(game_data.move_flags[move_id])
    if not _has_mold_breaker_imm:
        if (
            _def_ability_imm == ABILITY_SOUNDPROOF
            and (_move_flags_imm & FLAG_SOUND) != 0
        ):
            _tm_imm = 0.0
        elif (
            _def_ability_imm == ABILITY_BULLETPROOF
            and (_move_flags_imm & FLAG_BULLET) != 0
        ):
            _tm_imm = 0.0
    # Scrappy / Mind's Eye: Normal and Fighting moves ignore Ghost immunity.
    # Must check here in the early-immunity section to avoid short-circuiting
    # before the detailed Scrappy logic at the type_mult computation below.
    _has_scrappy_imm = _atk_ability_imm in (ABILITY_SCRAPPY, ABILITY_MINDS_EYE)
    _is_normal_or_fighting_imm = move_type in (TYPE_NORMAL, TYPE_FIGHTING)
    _def_is_ghost_imm = (_def_t1_imm == TYPE_GHOST) or (_def_t2_imm == TYPE_GHOST)
    if (
        _has_scrappy_imm
        and _is_normal_or_fighting_imm
        and _def_is_ghost_imm
        and _tm_imm == 0.0
    ):
        # Recompute ignoring Ghost slot(s) — Scrappy bypasses Ghost immunity.
        _s1 = 1.0 if _def_t1_imm == TYPE_GHOST else _eff1
        _s2 = (
            1.0
            if _def_t2_imm == TYPE_GHOST
            else (1.0 if _def_t2_imm == _def_t1_imm else _eff2)
        )
        _tm_imm = _s1 * _s2
    # Foresight volatile: also bypasses Ghost immunity for Normal/Fighting.
    _def_ext_vol_imm = int(
        battle[
            (
                (OFF_FIELD + F_EXTENDED_VOLATILE_0)
                if def_offset < OFF_SIDE1
                else (OFF_FIELD + F_EXTENDED_VOLATILE_1)
            )
        ]
    )
    _has_foresight_imm = (_def_ext_vol_imm & EXT_VOL_FORESIGHT) != 0
    if (
        _has_foresight_imm
        and _is_normal_or_fighting_imm
        and _def_is_ghost_imm
        and _tm_imm == 0.0
    ):
        _s1 = 1.0 if _def_t1_imm == TYPE_GHOST else _eff1
        _s2 = (
            1.0
            if _def_t2_imm == TYPE_GHOST
            else (1.0 if _def_t2_imm == _def_t1_imm else _eff2)
        )
        _tm_imm = _s1 * _s2
    # Status moves DON'T go through getDamage — pokepy short-circuits
    # before this for cat=Status, so we only hit here for damaging moves.
    # Fixed-damage attacks like Seismic Toss also need to bail here:
    # Showdown's hitStepTypeImmunity runs before accuracy, so an immune
    # fixed-damage move spends zero PRNG frames. Struggle is typeless and
    # ignores type immunity entirely.
    if move_id != MOVE_STRUGGLE and _tm_imm == 0.0:
        return 0

    # Accuracy
    accuracy = int(game_data.move_accuracy[move_id])
    is_physical = move_cat == CAT_PHYSICAL

    atk_ability_pre = effective_ability(battle, atk_offset, def_offset)
    has_hustle = atk_ability_pre == ABILITY_HUSTLE
    hustle_acc_mult = 0.8 if (has_hustle and is_physical) else 1.0
    has_compound_eyes = atk_ability_pre == ABILITY_COMPOUND_EYES
    compound_eyes_mult = 1.3 if has_compound_eyes else 1.0

    def_ability_pre = effective_ability(battle, def_offset, atk_offset)
    weather = _effective_weather(battle)

    sand_veil_mult = (
        0.8
        if (def_ability_pre == ABILITY_SAND_VEIL and weather == WEATHER_SAND)
        else 1.0
    )
    snow_cloak_mult = (
        0.8
        if (def_ability_pre == ABILITY_SNOW_CLOAK and weather == WEATHER_SNOW)
        else 1.0
    )
    # Tangled Feet (Dodrio): 0.5x accuracy when defender is confused.
    # Showdown abilities.ts:tangledfeet onModifyAccuracy.
    _ABILITY_TANGLED_FEET = 77
    def_vol_tf = int(
        battle[
            (
                (OFF_FIELD + F_VOLATILE_0)
                if def_offset < OFF_SIDE1
                else (OFF_FIELD + F_VOLATILE_1)
            )
        ]
    )
    def_confused_tf = ((def_vol_tf >> 1) & 0x7) > 0
    tangled_feet_mult = (
        0.5 if (def_ability_pre == _ABILITY_TANGLED_FEET and def_confused_tf) else 1.0
    )

    atk_boosts2 = int(battle[atk_offset + 14])
    def_boosts2 = int(battle[def_offset + 14])
    acc_stage = max(-6, min(6, extract_boost(atk_boosts2, 4)))
    eva_stage = max(-6, min(6, extract_boost(def_boosts2, 8)))
    # Mind's Eye / Keen Eye / No Guard / Illuminate / Scrappy:
    # Showdown's `move.ignoreEvasion` plus the abilities that ignore positive
    # defender evasion. For Mind's Eye specifically also ignore negative
    # accuracy stages on the user. Keen Eye (ability 51, abilities.ts:keeneye)
    # ignores positive defender evasion stages too.
    has_minds_eye_acc = atk_ability_pre == ABILITY_MINDS_EYE
    _ABILITY_KEEN_EYE = 51
    _ABILITY_ILLUMINATE = 35
    has_keen_eye = atk_ability_pre == _ABILITY_KEEN_EYE
    # Illuminate (Gen 9 Indigo Disk patch, data/abilities.ts:illuminate
    # onModifyMove sets move.ignoreEvasion = true). In mods/gen9predlc/
    # abilities.ts, illuminate is reverted to an empty ability — confirming
    # this behavior is post-DLC2 gen 9 only.
    has_illuminate = atk_ability_pre == _ABILITY_ILLUMINATE
    # Showdown abilities.ts:keeneye/mindseye/illuminate all set
    # `move.ignoreEvasion = true` via onModifyMove. The accuracy code at
    # sim/battle-actions.ts:713 then SKIPS the evasion contribution entirely
    # when ignoreEvasion is set — i.e. evasion is treated as 0 regardless of
    # sign. Pokepy used to only zero positive eva, leaving negative eva
    # (lowered defender evasion) exploitable by these accuracy-locking
    # abilities. Sign-agnostic now.
    if has_minds_eye_acc or has_keen_eye or has_illuminate:
        eva_stage = 0
    # Unaware — data/abilities.ts:unaware onAnyModifyBoost clears evasion on
    # the defender when the Unaware user is the attacker, and clears accuracy
    # on the attacker when the Unaware user is the defender. Both are sign-
    # agnostic.
    if atk_ability_pre == ABILITY_UNAWARE:
        eva_stage = 0
    if def_ability_pre == ABILITY_UNAWARE:
        acc_stage = 0
    # Foresight / Miracle Eye volatile (target side): zeros POSITIVE defender
    # evasion stages. Showdown moves.ts:6328-6332 (foresight condition
    # onModifyBoost) — `if (boosts.evasion && boosts.evasion > 0) boosts.evasion = 0`.
    # Pokepy already tracks the foresight volatile via EXT_VOL_FORESIGHT and
    # uses it for the Ghost-immunity bypass at L992, but the evasion
    # zeroing was missing. Negative evasion stages are preserved.
    _def_ext_vol_eva = int(
        battle[
            (
                (OFF_FIELD + F_EXTENDED_VOLATILE_0)
                if def_offset < OFF_SIDE1
                else (OFF_FIELD + F_EXTENDED_VOLATILE_1)
            )
        ]
    )
    if (_def_ext_vol_eva & EXT_VOL_FORESIGHT) != 0 and eva_stage > 0:
        eva_stage = 0
    # Showdown sim/battle-actions.ts:707-722 hitStepAccuracy computes a SINGLE
    # combined boost = clamp(acc_stage - eva_stage, -6, 6) and applies ONE
    # multiplier to accuracy:
    #     if boost > 0: accuracy = trunc(accuracy * (3 + boost) / 3)
    #     if boost < 0: accuracy = trunc(accuracy * 3 / (3 - boost))
    # Pokepy used to apply acc_mult and eva_mult separately, which drifts from
    # the combined-boost formula whenever both sides have nonzero stages.
    # Example: acc=+1, eva=-1 → Showdown boost=2 → 5/3≈1.667x; old pokepy
    # gave (4/3)/(3/4)=16/9≈1.778x. Now matches Showdown exactly.
    combined_boost = max(-6, min(6, acc_stage - eva_stage))
    if combined_boost > 0:
        boosted_accuracy = int(accuracy * (3 + combined_boost) / 3)
    elif combined_boost < 0:
        boosted_accuracy = int(accuracy * 3 / (3 - combined_boost))
    else:
        boosted_accuracy = int(accuracy)
    # Mind's Eye also blocks the Sand Veil / Snow Cloak evasion abilities.
    if has_minds_eye_acc:
        sand_veil_mult = 1.0
        snow_cloak_mult = 1.0
    # Item accuracy modifiers — Showdown items.ts onModifyAccuracy /
    # onSourceModifyAccuracy hooks. All run BEFORE the boost-stage block in
    # Showdown, but pokepy already collapsed the chain into one float; we
    # apply the item factors as additional float multipliers afterward,
    # which matches Showdown's truncated chain to within ±1 accuracy point
    # for the relevant moves (since chainModify uses ((prev*next+0x800)>>12)
    # against 4096-base numerators 4505/3686/4915 ≈ 1.0998/0.9001/1.2002).
    from pokepy.core.constants import (
        ITEM_WIDE_LENS as _ITEM_WIDE_LENS_AC,
        ITEM_ZOOM_LENS as _ITEM_ZOOM_LENS_AC,
        ITEM_BRIGHT_POWDER as _ITEM_BRIGHT_POWDER_AC,
        ITEM_LAX_INCENSE as _ITEM_LAX_INCENSE_AC,
    )

    _atk_item_acc = atk_item_live
    _def_item_acc = int(battle[def_offset + 6])
    # Wide Lens (atk holder): chainModify([4505, 4096]) onSourceModifyAccuracy.
    wide_lens_mult = 4505.0 / 4096.0 if _atk_item_acc == _ITEM_WIDE_LENS_AC else 1.0
    # Zoom Lens (atk holder): chainModify([4915, 4096]) only when the holder
    # is moving last (Showdown: `!this.queue.willMove(target)`). The engine
    # passes is_moving_last for that purpose.
    zoom_lens_mult = (
        4915.0 / 4096.0
        if (_atk_item_acc == _ITEM_ZOOM_LENS_AC and is_moving_last)
        else 1.0
    )
    # Bright Powder / Lax Incense (def holder): chainModify([3686, 4096])
    # via onModifyAccuracy. Showdown applies these to any incoming move.
    bright_powder_mult = (
        3686.0 / 4096.0
        if _def_item_acc in (_ITEM_BRIGHT_POWDER_AC, _ITEM_LAX_INCENSE_AC)
        else 1.0
    )
    effective_accuracy = int(
        boosted_accuracy
        * hustle_acc_mult
        * compound_eyes_mult
        * sand_veil_mult
        * snow_cloak_mult
        * tangled_feet_mult
        * wide_lens_mult
        * zoom_lens_mult
        * bright_powder_mult
    )

    has_no_guard = (atk_ability_pre == ABILITY_NO_GUARD) or (
        def_ability_pre == ABILITY_NO_GUARD
    )

    atk_ext_vol_offset = (
        (OFF_FIELD + F_EXTENDED_VOLATILE_0)
        if atk_side == 0
        else (OFF_FIELD + F_EXTENDED_VOLATILE_1)
    )
    atk_ext_vol = int(battle[atk_ext_vol_offset])
    has_lock_on = (atk_ext_vol & EXT_VOL_LOCK_ON) != 0

    # Semi-invulnerability is resolved before accuracy / crit / damage PRNG in
    # Showdown's hit pipeline. If the target is untargetable and this move
    # cannot connect through its charge state, no random attack frames fire.
    target_semi_invul_pre = (
        int(battle[_side_actions_offset(def_side)]) & ACTIVE_MOVE_ACTIONS_SEMI_INVUL
    ) != 0
    if target_semi_invul_pre and not (has_no_guard or has_lock_on):
        from pokepy.core.constants import M_CHARGING_0, M_CHARGING_1
        from pokepy.core.constants import MOVE_DIG, MOVE_DIVE, MOVE_FLY, MOVE_BOUNCE

        _MOVE_GUST_PRE = 16
        _MOVE_TWISTER_PRE = 239
        _MOVE_SKY_UPPERCUT_PRE = 327
        _MOVE_THUNDER_PRE = 87
        _MOVE_HURRICANE_PRE = 542
        _MOVE_SMACK_DOWN_PRE = 479
        _MOVE_THOUSAND_ARROWS_PRE = 614
        _MOVE_MAGNITUDE_PRE = 222
        _MOVE_WHIRLPOOL_PRE = 250
        _FLY_LIKE_PRE = (MOVE_FLY, MOVE_BOUNCE)
        _AIR_HIT_PRE = (
            _MOVE_GUST_PRE,
            _MOVE_TWISTER_PRE,
            _MOVE_SKY_UPPERCUT_PRE,
            _MOVE_THUNDER_PRE,
            _MOVE_HURRICANE_PRE,
            _MOVE_SMACK_DOWN_PRE,
            _MOVE_THOUSAND_ARROWS_PRE,
        )

        charge_meta_off_pre = OFF_META + (
            M_CHARGING_0 if def_side == 0 else M_CHARGING_1
        )
        charging_move_pre = int(battle[charge_meta_off_pre])
        if charging_move_pre in _FLY_LIKE_PRE:
            can_hit_semi_invul = move_id in _AIR_HIT_PRE
        elif charging_move_pre == MOVE_DIG:
            can_hit_semi_invul = move_id in (MOVE_EARTHQUAKE, _MOVE_MAGNITUDE_PRE)
        elif charging_move_pre == MOVE_DIVE:
            can_hit_semi_invul = move_id in (MOVE_SURF, _MOVE_WHIRLPOOL_PRE)
        else:
            can_hit_semi_invul = False

        if not can_hit_semi_invul:
            if out_meta is not None:
                out_meta["num_hits"] = 1
            return 0

    weather_acc = _effective_weather(battle)
    # Hurricane / Thunder use `target.effectiveWeather()` (Showdown
    # data/moves.ts:9367, :20196). Bleakwind Storm / Sandsear Storm /
    # Wildbolt Storm use the same target-weather rain/primordial-sea override
    # (moves.ts:1528, :16267, :21668). Utility Umbrella suppresses the
    # holder's effective sun/rain/primal weather, but Blizzard still checks
    # field snow directly.
    _target_weather_acc = _effective_weather_for_pokemon(battle, def_offset)
    rain_perfect = move_id in (
        MOVE_HURRICANE,
        MOVE_THUNDER,
        846,
        847,
        848,
    ) and _target_weather_acc in (WEATHER_RAIN, WEATHER_PRIMORDIAL_SEA)
    snow_perfect = move_id == MOVE_BLIZZARD and weather_acc == WEATHER_SNOW
    sun_reduced = move_id in (MOVE_HURRICANE, MOVE_THUNDER) and _target_weather_acc in (
        WEATHER_SUN,
        WEATHER_DESOLATE_LAND,
    )
    if sun_reduced:
        effective_accuracy = 50
    weather_perfect_acc = rain_perfect or snow_perfect
    # Glaive Rush: if the target (defender) has the glaive_rush flag set
    # on itself (from using Glaive Rush last turn), any move aimed at it
    # bypasses accuracy (Showdown data/moves.ts:glaiverush onAccuracy true).
    from pokepy.core.constants import FLAG_GLAIVE_RUSH as _FLAG_GLAIVE_RUSH

    def_flags_gr = int(battle[def_offset + 15])
    def_has_glaive_rush = (def_flags_gr & _FLAG_GLAIVE_RUSH) != 0
    # Showdown sim/battle-actions.ts:732 only calls `randomChance(accuracy, 100)`
    # when `accuracy !== true`. When the move always hits (No Guard, Lock-On,
    # weather-perfect, Glaive Rush target, accuracy: true), no PRNG frame is
    # consumed. pokepy used to roll unconditionally, advancing the LCG by one
    # frame more than Showdown for guaranteed-hit moves.
    accuracy_bypassed = (
        has_no_guard
        or has_lock_on
        or weather_perfect_acc
        or def_has_glaive_rush
        or accuracy == 0
        or accuracy > 100
    )
    # Burn Up / Double Shock fail in Showdown's onTryMove before accuracy
    # once the user no longer has the required live typing.
    _MOVE_BURN_UP = 682
    _MOVE_DOUBLE_SHOCK = 892
    if move_id in (_MOVE_BURN_UP, _MOVE_DOUBLE_SHOCK):
        atk_types_live = int(battle[atk_offset + 4]) & 0xFFFF
        atk_t1_live = atk_types_live & 0xFF
        atk_t2_live = (atk_types_live >> 8) & 0xFF
        required_type = TYPE_FIRE if move_id == _MOVE_BURN_UP else TYPE_ELECTRIC
        if atk_t1_live != required_type and atk_t2_live != required_type:
            if out_meta is not None:
                out_meta["num_hits"] = 1
            return 0
    # Showdown's Endeavor `onTryImmunity` checks the target's LIVE HP before
    # hitStepAccuracy. When the target's current HP is not above the user's,
    # the move fails with `|-immune|` and consumes no accuracy frame. This is
    # especially important after a faster target pays HP for Substitute, where
    # the slower Endeavor user must see the post-Substitute HP immediately.
    if move_id == MOVE_ENDEAVOR:
        atk_hp_live = int(battle[atk_offset + 1])
        def_hp_live = int(battle[def_offset + 1])
        if atk_hp_live >= def_hp_live:
            if out_meta is not None:
                out_meta["num_hits"] = 1
            return 0

    if accuracy_bypassed:
        hits = True
    else:
        acc_roll = gen5_prng.random(100)
        hits = acc_roll < effective_accuracy

    # Showdown's trySpreadMoveHit bails at the first `hitStep` that returns
    # all-false (sim/battle-actions.ts:596-605). When hitStepAccuracy fails,
    # the loop breaks BEFORE hitStepMoveHitLoop runs — no crit or damage
    # randomizer frames are consumed. Pokepy used to roll both anyway,
    # over-consuming 2 frames per missed move and drifting the LCG on any
    # scenario with a miss (e.g., Dynamic Punch 50% acc). Bail early here
    # to match Showdown's frame schedule.
    if not hits:
        if out_meta is not None:
            out_meta["num_hits"] = 1
        return 0

    atk_level = int(battle[atk_offset + 3])
    atk_stat_base = (
        int(battle[atk_offset + 7]) if is_physical else int(battle[atk_offset + 9])
    )
    atk_types = int(battle[atk_offset + 4])
    atk_type1 = atk_types & 0xFF
    atk_type2 = (atk_types >> 8) & 0xFF
    atk_ability = effective_ability(battle, atk_offset, def_offset)
    atk_item = atk_item_live

    def_ability = effective_ability(battle, def_offset, atk_offset)
    field_atk_ability = (
        atk_ability
        if int(override_field_atk_ability) < 0
        else int(override_field_atk_ability)
    )
    field_def_ability = (
        def_ability
        if int(override_field_def_ability) < 0
        else int(override_field_def_ability)
    )

    atk_has_unaware = atk_ability == ABILITY_UNAWARE
    def_has_unaware = def_ability == ABILITY_UNAWARE

    atk_boosts = int(battle[atk_offset + 13])
    atk_boost = (
        extract_boost(atk_boosts, 0) if is_physical else extract_boost(atk_boosts, 8)
    )
    if suppress_attacker_boosts:
        atk_boost = 0
    # Unaware on DEFENDER ignores ALL attacker atk/spa boosts (positive AND
    # negative). Showdown data/abilities.ts:unaware onAnyModifyBoost sets
    # `boosts['atk'] = 0; boosts['spa'] = 0` when the Unaware user is the
    # active target — sign-agnostic. Pokepy used to only clear positive boosts.
    if def_has_unaware:
        atk_boost = 0
    atk_stat = int(atk_stat_base * float(get_boost_multiplier(atk_boost)))

    # Showdown's runEvent('ModifyAtk' / 'ModifySpA') accumulates ALL chainModify
    # calls (from attacker's onModifyAtk hooks AND defender's onSourceModifyAtk
    # hooks) into a single 4096-base accumulator, then applies it once via
    # finalModify. Sequential _show_modify calls drift by ±1 vs the accumulator
    # on chained modifiers — see _ChainAccum docstring. Accumulate here, apply
    # once after all hooks have been collected.
    _atk_chain = _ChainAccum()

    # Choice Band / Choice Specs apply as atk/spa modifiers (not BP).
    # Showdown source: data/items.ts choiceband.onModifyAtk → chainModify(1.5).
    if atk_item == ITEM_CHOICE_BAND and is_physical:
        _atk_chain.chain(6144, 4096)
    if atk_item == ITEM_CHOICE_SPECS and not is_physical:
        _atk_chain.chain(6144, 4096)

    # Atk-modifier abilities. These ALL apply via Showdown's onModifyAtk /
    # onModifySpA chain (NOT onBasePower). Each is a chainModify call into the
    # same accumulator; final modify is applied once below.
    _has_mold_breaker_early = atk_ability in (
        ABILITY_MOLD_BREAKER,
        ABILITY_TERAVOLT,
        ABILITY_TURBOBLAZE,
    )
    _atk_hp_early = int(battle[atk_offset + 1])
    _atk_maxhp_early = int(battle[atk_offset + 2])
    if atk_ability == ABILITY_STEELWORKER and move_type == TYPE_STEEL:
        _atk_chain.chain(6144, 4096)  # 1.5x
    if atk_ability == ABILITY_TRANSISTOR and move_type == TYPE_ELECTRIC:
        _atk_chain.chain(5325, 4096)  # 1.3x (Gen 9 nerf)
    if atk_ability == ABILITY_DRAGONS_MAW and move_type == TYPE_DRAGON:
        _atk_chain.chain(6144, 4096)
    if atk_ability == ABILITY_ROCKY_PAYLOAD and move_type == TYPE_ROCK:
        _atk_chain.chain(6144, 4096)
    if atk_ability == ABILITY_GORILLA_TACTICS and is_physical:
        _atk_chain.chain(6144, 4096)
    if atk_ability == ABILITY_WATER_BUBBLE and move_type == TYPE_WATER:
        _atk_chain.chain(8192, 4096)  # 2x
    # Pinch abilities (Overgrow/Blaze/Torrent/Swarm) at HP <= 1/3
    if (
        atk_ability == ABILITY_OVERGROW
        and move_type == TYPE_GRASS
        and _atk_hp_early * 3 <= _atk_maxhp_early
    ):
        _atk_chain.chain(6144, 4096)
    if (
        atk_ability == ABILITY_BLAZE
        and move_type == TYPE_FIRE
        and _atk_hp_early * 3 <= _atk_maxhp_early
    ):
        _atk_chain.chain(6144, 4096)
    if (
        atk_ability == ABILITY_TORRENT
        and move_type == TYPE_WATER
        and _atk_hp_early * 3 <= _atk_maxhp_early
    ):
        _atk_chain.chain(6144, 4096)
    if (
        atk_ability == ABILITY_SWARM
        and move_type == TYPE_BUG
        and _atk_hp_early * 3 <= _atk_maxhp_early
    ):
        _atk_chain.chain(6144, 4096)
    # Flash Fire: stored boost flag in pokemon flag bit 0x200
    atk_flags_ff = int(battle[atk_offset + 15])
    if (
        atk_ability == ABILITY_FLASH_FIRE
        and (atk_flags_ff & 0x200) != 0
        and move_type == TYPE_FIRE
    ):
        _atk_chain.chain(6144, 4096)
    # Defender abilities that modify ATTACKER's atk/spa via
    # onSourceModifyAtk / onSourceModifySpA:
    #   Thick Fat: Fire/Ice → 0.5x
    #   Water Bubble (def): Fire → 0.5x
    #   Heatproof: Fire → 0.5x
    #   Purifying Salt: Ghost → 0.5x
    if not _has_mold_breaker_early:
        if def_ability == ABILITY_THICK_FAT and move_type in (TYPE_FIRE, TYPE_ICE):
            _atk_chain.chain(2048, 4096)
        if def_ability == ABILITY_WATER_BUBBLE and move_type == TYPE_FIRE:
            _atk_chain.chain(2048, 4096)
        # Heatproof: defender's onSourceModifyAtk for Fire moves → 0.5x
        if def_ability == 85 and move_type == TYPE_FIRE:  # ABILITY_HEATPROOF = 85
            _atk_chain.chain(2048, 4096)
        if def_ability == ABILITY_PURIFYING_SALT and move_type == TYPE_GHOST:
            _atk_chain.chain(2048, 4096)

    uses_physical_def = is_physical or move_id in (
        MOVE_PSYSHOCK,
        MOVE_PSYSTRIKE,
        MOVE_SECRET_SWORD,
    )
    def_stat_base = (
        int(battle[def_offset + 8])
        if uses_physical_def
        else int(battle[def_offset + 10])
    )
    def_types = int(battle[def_offset + 4])
    def_type1 = def_types & 0xFF
    def_type2 = (def_types >> 8) & 0xFF
    def_hp = int(battle[def_offset + 1])
    def_max_hp = int(battle[def_offset + 2])

    def_boosts = int(battle[def_offset + 13])
    def_boost_raw = (
        extract_boost(def_boosts, 4)
        if uses_physical_def
        else extract_boost(def_boosts, 12)
    )
    def_boost = def_boost_raw
    # Unaware on ATTACKER ignores ALL defender def/spd boosts (positive AND
    # negative). Same fix as atk_boost above — sign-agnostic.
    if atk_has_unaware:
        def_boost = 0
    # `move.ignoreDefensive`: Chip Away (498), Sacred Sword (533), Darkest
    # Lariat (663). Showdown sim/battle-actions.ts:1687-1697 sets defBoosts=0
    # unconditionally (sign-agnostic) before recomputing the defense stat.
    # Pokepy used to only handle the crit-ignores-positive path, silently
    # leaving these three moves to obey opponent defensive stages.
    _MOVE_CHIP_AWAY = 498
    _MOVE_SACRED_SWORD = 533
    _MOVE_DARKEST_LARIAT = 663
    _ignores_def_boost = move_id in (
        _MOVE_CHIP_AWAY,
        _MOVE_SACRED_SWORD,
        _MOVE_DARKEST_LARIAT,
    )
    if _ignores_def_boost:
        def_boost = 0
    def_stat = int(def_stat_base * float(get_boost_multiplier(def_boost)))

    # Showdown's runEvent('ModifyDef' / 'ModifySpD') accumulator. Same pattern
    # as _atk_chain — accumulate all chainModify hooks, apply once below.
    _def_chain = _ChainAccum()

    # Ruin abilities: Showdown applies via chainModify(0.75). The ability
    # holder is excluded by `source.hasAbility('...')` / `target.hasAbility(...)`
    # checks, NOT by species — so Trace / Skill Swap / Role Play / Imposter
    # propagation works. Delayed Future Sight / Doom Desire can also leave the
    # move source off-field while a different current active still supplies a
    # Ruin aura, so use the live field-active abilities for the holder checks
    # rather than assuming the holder is always the attacker or defender slot.
    # Vessel of Ruin (Ting-Lu) → onAnyModifySpA (-25% spa for special).
    # Tablets of Ruin (Wo-Chien) → onAnyModifyAtk (-25% atk for physical).
    # Sword of Ruin (Chien-Pao) → onAnyModifyDef (-25% def for physical).
    # Beads of Ruin (Chi-Yu) → onAnyModifySpD (-25% spd for special).
    if (
        field_atk_ability == ABILITY_VESSEL_OF_RUIN
        and not is_physical
        and atk_ability != ABILITY_VESSEL_OF_RUIN
    ):
        _atk_chain.chain(3072, 4096)  # 0.75x
    if (
        field_def_ability == ABILITY_VESSEL_OF_RUIN
        and not is_physical
        and atk_ability != ABILITY_VESSEL_OF_RUIN
    ):
        _atk_chain.chain(3072, 4096)  # 0.75x
    if (
        field_atk_ability == ABILITY_TABLETS_OF_RUIN
        and is_physical
        and atk_ability != ABILITY_TABLETS_OF_RUIN
    ):
        _atk_chain.chain(3072, 4096)
    if (
        field_def_ability == ABILITY_TABLETS_OF_RUIN
        and is_physical
        and atk_ability != ABILITY_TABLETS_OF_RUIN
    ):
        _atk_chain.chain(3072, 4096)
    if (
        field_atk_ability == ABILITY_SWORD_OF_RUIN
        and is_physical
        and def_ability != ABILITY_SWORD_OF_RUIN
    ):
        _def_chain.chain(3072, 4096)
    if (
        field_def_ability == ABILITY_SWORD_OF_RUIN
        and is_physical
        and def_ability != ABILITY_SWORD_OF_RUIN
    ):
        _def_chain.chain(3072, 4096)
    if (
        field_atk_ability == ABILITY_BEADS_OF_RUIN
        and not is_physical
        and def_ability != ABILITY_BEADS_OF_RUIN
    ):
        _def_chain.chain(3072, 4096)
    if (
        field_def_ability == ABILITY_BEADS_OF_RUIN
        and not is_physical
        and def_ability != ABILITY_BEADS_OF_RUIN
    ):
        _def_chain.chain(3072, 4096)

    # Assault Vest
    def_item = int(battle[def_offset + 6])
    if def_item == ITEM_ASSAULT_VEST and not uses_physical_def:
        _def_chain.chain(6144, 4096)  # 1.5x

    # Fur Coat
    has_mold_fc = atk_ability in (
        ABILITY_MOLD_BREAKER,
        ABILITY_TERAVOLT,
        ABILITY_TURBOBLAZE,
    )
    if def_ability == ABILITY_FUR_COAT and uses_physical_def and not has_mold_fc:
        _def_chain.chain(8192, 4096)  # 2x

    # Grass Pelt: +50% Def in Grassy Terrain. Showdown data/abilities.ts grasspelt
    # `onModifyDefPriority: 6, onModifyDef(pokemon) { if
    # (this.field.isTerrain('grassyterrain')) return this.chainModify(1.5); }`.
    # Breakable — suppressed by Mold Breaker / Teravolt / Turboblaze.
    _ABILITY_GRASS_PELT = 179
    current_terrain_gp = int(battle[OFF_FIELD + F_TERRAIN])
    if (
        def_ability == _ABILITY_GRASS_PELT
        and uses_physical_def
        and current_terrain_gp == TERRAIN_GRASSY
        and not has_mold_fc
    ):
        _def_chain.chain(6144, 4096)

    if def_item == ITEM_EVIOLITE:
        _def_chain.chain(6144, 4096)  # 1.5x

    # Snowscape: onModifyDefPriority 10, modify(def, 1.5) for Ice-types.
    # Sandstorm: onModifySpDPriority 10, modify(spd, 1.5) for Rock-types.
    # Both apply to whichever defensive stat the move actually uses. That
    # means Snowscape still boosts Psyshock / Psystrike / Secret Sword because
    # they target Def even though their move category is not Physical.
    def_is_ice = (def_type1 == TYPE_ICE) or (def_type2 == TYPE_ICE)
    snow_def_mult = (
        1.5 if (weather == WEATHER_SNOW and def_is_ice and uses_physical_def) else 1.0
    )
    if snow_def_mult != 1.0:
        _def_chain.chain(6144, 4096)

    def_is_rock = (def_type1 == TYPE_ROCK) or (def_type2 == TYPE_ROCK)
    sand_spd_mult = (
        1.5
        if (weather == WEATHER_SAND and def_is_rock and not uses_physical_def)
        else 1.0
    )
    if sand_spd_mult != 1.0:
        _def_chain.chain(6144, 4096)

    # Defender paradox boost
    def_has_paradox = def_ability in (ABILITY_PROTOSYNTHESIS, ABILITY_QUARK_DRIVE)
    def_flags_paradox = int(battle[def_offset + 15])
    def_had_item = (def_flags_paradox & 0x80) != 0
    def_booster_consumed = def_has_paradox and (
        (def_flags_paradox & FLAG_BOOSTER_ENERGY_ACTIVE) != 0
    )
    def_paradox_active = (
        (def_ability == ABILITY_PROTOSYNTHESIS and weather == WEATHER_SUN)
        or (
            def_ability == ABILITY_QUARK_DRIVE
            and int(battle[OFF_FIELD + F_TERRAIN]) == TERRAIN_ELECTRIC
        )
        or def_booster_consumed
    )
    _def_paradox_best = _get_paradox_best_stat(battle, def_offset)
    paradox_def_boost = def_paradox_active and (
        (_def_paradox_best == "def" and uses_physical_def)
        or (_def_paradox_best == "spd" and not uses_physical_def)
    )
    if paradox_def_boost:
        # Showdown chainModify([5325, 4096])
        _def_chain.chain(5325, 4096)

    # Weather Ball / Terrain Pulse type override
    is_weather_ball = move_id == MOVE_WEATHER_BALL
    if is_weather_ball:
        if weather == WEATHER_SUN:
            move_type = TYPE_FIRE
        elif weather == WEATHER_RAIN:
            move_type = TYPE_WATER
        elif weather == WEATHER_SNOW:
            move_type = TYPE_ICE
        elif weather == WEATHER_SAND:
            move_type = TYPE_ROCK
        elif weather == WEATHER_PRIMORDIAL_SEA:
            move_type = TYPE_WATER
        elif weather == WEATHER_DESOLATE_LAND:
            move_type = TYPE_FIRE

    terrain = int(battle[OFF_FIELD + F_TERRAIN])
    is_terrain_pulse = move_id == MOVE_TERRAIN_PULSE
    if is_terrain_pulse:
        if terrain == TERRAIN_ELECTRIC:
            move_type = TYPE_ELECTRIC
        elif terrain == TERRAIN_GRASSY:
            move_type = TYPE_GRASS
        elif terrain == TERRAIN_PSYCHIC:
            move_type = TYPE_PSYCHIC
        elif terrain == TERRAIN_MISTY:
            move_type = TYPE_FAIRY

    # Type effectiveness
    eff1 = float(type_chart[def_type1, move_type])
    eff2 = 1.0 if def_type2 == def_type1 else float(type_chart[def_type2, move_type])
    type_mult = eff1 * eff2

    # Delta Stream: SE hits on Flying types are reduced to neutral.
    # Showdown: data/conditions.ts:729 deltastream onEffectiveness(priority -1)
    # — "move && move.effectType === 'Move' && move.category !== 'Status' &&
    #    type === 'Flying' && typeMod > 0" → return 0 (neutralize).
    # This only triggers for damaging moves where the move type is SE against
    # Flying, and only nullifies the Flying slot's contribution. The pokepy
    # approximation: if move category is not status, the defender is a Flying
    # type, and the move would have been SE against that Flying slot, we set
    # that slot's contribution to 1.0 instead.
    # Delta Stream onEffectiveness is suppressed by Air Lock / Cloud Nine —
    # Showdown sim/battle.ts:617 skips ALL Weather-effectType handlers when
    # `field.suppressingWeather()` is true. Use _effective_weather().
    _cur_weather_ds = _effective_weather(battle)
    if _cur_weather_ds == 7:  # WEATHER_DELTA_STREAM
        _category_ds = int(game_data.move_category[move_id])
        # CAT_STATUS = 0 in pokepy/core/constants.py
        if _category_ds != 0:
            _flying_ds = TYPE_FLYING
            _flying_chart_val = float(type_chart[_flying_ds, move_type])
            if _flying_chart_val > 1.0:
                _slot1_flying = def_type1 == _flying_ds
                _slot2_flying = (def_type2 == _flying_ds) and (def_type2 != def_type1)
                if _slot1_flying and _slot2_flying:
                    # Both slots flying (impossible): neutralize both.
                    type_mult = 1.0
                elif _slot1_flying:
                    type_mult = 1.0 * eff2
                elif _slot2_flying:
                    type_mult = eff1 * 1.0

    # Scrappy / Mind's Eye / Foresight: Normal and Fighting moves treat Ghost
    # as Normal for type effectiveness (effectively ignore the 0x multiplier
    # contribution from the Ghost slot). Showdown implements this as
    # `ignoreImmunity: {'Ghost': true}` applied by the ability/volatile, which
    # bypasses the Ghost slot at chart lookup time. For dual-type defenders
    # like Ghost/Steel (Aegislash), Normal → 0.5x (Steel) × 1.0 (Ghost slot
    # ignored) — NOT flat 1.0x as pokepy used to compute.
    has_scrappy = atk_ability in (ABILITY_SCRAPPY, ABILITY_MINDS_EYE)
    is_normal_or_fighting = move_type in (TYPE_NORMAL, TYPE_FIGHTING)
    def_is_ghost = (def_type1 == TYPE_GHOST) or (def_type2 == TYPE_GHOST)
    def_ext_vol_offset = (
        (OFF_FIELD + F_EXTENDED_VOLATILE_0)
        if def_side == 0
        else (OFF_FIELD + F_EXTENDED_VOLATILE_1)
    )
    def_ext_vol = int(battle[def_ext_vol_offset])
    has_foresight = (def_ext_vol & EXT_VOL_FORESIGHT) != 0
    if (has_scrappy or has_foresight) and is_normal_or_fighting and def_is_ghost:
        # Recompute type_mult ignoring the Ghost slot(s).
        _slot1_mul = 1.0 if def_type1 == TYPE_GHOST else eff1
        _slot2_mul = (
            1.0
            if (def_type2 == TYPE_GHOST)
            else (1.0 if def_type2 == def_type1 else eff2)
        )
        type_mult = _slot1_mul * _slot2_mul

    # Future Sight / Doom Desire / Thousand Arrows ignoreImmunity (Showdown).
    # Future Sight (Psychic) hits Dark types; Thousand Arrows (Ground) hits
    # Flying / Levitate / Air Balloon. Pokepy uses move IDs 248, 353, 615.
    if move_id in (248, 353) and type_mult == 0.0:
        # Future Sight / Doom Desire bypass type immunity (rare in OU).
        type_mult = 1.0
    if move_id == 615:  # Thousand Arrows
        if move_type == TYPE_GROUND and (
            def_type1 == TYPE_FLYING or def_type2 == TYPE_FLYING
        ):
            # Replace flying immunity with neutral
            if type_mult == 0.0:
                type_mult = 1.0

    has_mold_breaker = atk_ability in (
        ABILITY_MOLD_BREAKER,
        ABILITY_TERAVOLT,
        ABILITY_TURBOBLAZE,
    )

    move_flags_early = int(game_data.move_flags[move_id])
    is_sound = (move_flags_early & FLAG_SOUND) != 0
    is_bullet = (move_flags_early & FLAG_BULLET) != 0

    levitate_immune = (
        (def_ability == ABILITY_LEVITATE)
        and (move_type == TYPE_GROUND)
        and not has_mold_breaker
    )
    flash_fire_immune = (
        (def_ability == ABILITY_FLASH_FIRE)
        and (move_type == TYPE_FIRE)
        and not has_mold_breaker
    )
    volt_immune = (
        (def_ability == ABILITY_VOLT_ABSORB)
        and (move_type == TYPE_ELECTRIC)
        and not has_mold_breaker
    )
    water_immune = (
        (def_ability == ABILITY_WATER_ABSORB)
        and (move_type == TYPE_WATER)
        and not has_mold_breaker
    )
    sap_sipper_immune = (
        (def_ability == ABILITY_SAP_SIPPER)
        and (move_type == TYPE_GRASS)
        and not has_mold_breaker
    )
    storm_drain_immune = (
        (def_ability == ABILITY_STORM_DRAIN)
        and (move_type == TYPE_WATER)
        and not has_mold_breaker
    )
    lightning_rod_immune = (
        (def_ability == ABILITY_LIGHTNING_ROD)
        and (move_type == TYPE_ELECTRIC)
        and not has_mold_breaker
    )
    motor_drive_immune = (
        (def_ability == ABILITY_MOTOR_DRIVE)
        and (move_type == TYPE_ELECTRIC)
        and not has_mold_breaker
    )
    dry_skin_water_immune = (
        (def_ability == ABILITY_DRY_SKIN)
        and (move_type == TYPE_WATER)
        and not has_mold_breaker
    )
    wonder_guard_immune = (
        (def_ability == ABILITY_WONDER_GUARD)
        and (type_mult <= 1.0)
        and not has_mold_breaker
    )
    soundproof_immune = (
        (def_ability == ABILITY_SOUNDPROOF) and is_sound and not has_mold_breaker
    )
    bulletproof_immune = (
        (def_ability == ABILITY_BULLETPROOF) and is_bullet and not has_mold_breaker
    )
    earth_eater_immune = (
        (def_ability == ABILITY_EARTH_EATER)
        and (move_type == TYPE_GROUND)
        and not has_mold_breaker
    )
    well_baked_immune = (
        (def_ability == ABILITY_WELL_BAKED_BODY)
        and (move_type == TYPE_FIRE)
        and not has_mold_breaker
    )
    ability_immune = (
        levitate_immune
        or flash_fire_immune
        or volt_immune
        or water_immune
        or sap_sipper_immune
        or storm_drain_immune
        or lightning_rod_immune
        or motor_drive_immune
        or dry_skin_water_immune
        or wonder_guard_immune
        or soundproof_immune
        or bulletproof_immune
        or earth_eater_immune
        or well_baked_immune
    )
    balloon_immune = (def_item == ITEM_AIR_BALLOON) and (move_type == TYPE_GROUND)
    if ability_immune or balloon_immune:
        type_mult = 0.0
        # Showdown's trySpreadMoveHit bails before Accuracy / Crit / Damage
        # randomization when the move is absorbed or negated by an immunity
        # ability/item (Water Absorb, Flash Fire, Air Balloon, etc.). Pokepy
        # was only short-circuiting raw type-chart immunities, which kept
        # consuming PRNG frames on absorbed hits and drifted long immunity
        # sequences like the Ogerpon-Wellspring mirror in Battle 2.
        return 0

    if is_struggle:
        type_mult = 1.0

    # Iron Ball / Smack Down / Gravity / Thousand Arrows: ground Flying types
    # so Ground moves hit them (typeMod overrides Flying immunity to 1.0).
    # Showdown source: Iron Ball items.ts onNegateImmunity for Ground.
    ITEM_IRON_BALL_DG = 224
    if def_item == ITEM_IRON_BALL_DG and move_type == TYPE_GROUND:
        if def_type1 == TYPE_FLYING or def_type2 == TYPE_FLYING:
            # Recompute type chart treating Flying as Normal (loses immunity)
            new_t1 = TYPE_NORMAL if def_type1 == TYPE_FLYING else def_type1
            new_t2 = TYPE_NORMAL if def_type2 == TYPE_FLYING else def_type2
            type_mult = float(type_chart[new_t1, TYPE_GROUND])
            if new_t2 != new_t1:
                type_mult *= float(type_chart[new_t2, TYPE_GROUND])

    # Freeze Dry: Ice move that's super-effective vs Water (not just neutral).
    # Showdown source: data/moves.ts freezedry.onEffectiveness adds Water as SE.
    MOVE_FREEZE_DRY = 573
    if move_id == MOVE_FREEZE_DRY:
        # Override Water resist to 2x SE
        if def_type1 == TYPE_WATER:
            type_mult = type_mult * 4.0  # was 0.5x, becomes 2x → multiply by 4
        if def_type2 == TYPE_WATER and def_type2 != def_type1:
            type_mult = type_mult * 4.0

    # Multiscale (breakable, gated by Mold Breaker) and Shadow Shield (NOT
    # breakable — Lunala's signature ability 231) are checked inside the
    # per-hit ModifyDamage helper below so multihit moves only get the full-HP
    # reduction on hits where the defender is still actually at full HP.
    _ABILITY_SHADOW_SHIELD = 231

    huge_power = atk_ability in (ABILITY_HUGE_POWER, ABILITY_PURE_POWER)
    if huge_power and is_physical:
        _atk_chain.chain(8192, 4096)  # 2x

    # Hustle — Showdown abilities.ts:1851-1856 uses `return this.modify(atk, 1.5)`
    # NOT chainModify. The comment states it "should be applied directly to the
    # stat as opposed to chaining". The runEvent pattern replaces relayVar with
    # Hustle's modified value, then finalModify applies the accumulated chain on
    # top. We defer Hustle to a separate modify applied alongside _atk_chain.apply.
    _hustle_active = atk_ability == ABILITY_HUSTLE and is_physical

    atk_hp_def = int(battle[atk_offset + 1])
    atk_maxhp_def = int(battle[atk_offset + 2])
    if atk_ability == ABILITY_DEFEATIST and atk_hp_def * 2 <= atk_maxhp_def:
        _atk_chain.chain(2048, 4096)  # 0.5x

    atk_status_field = int(battle[atk_offset + 12])
    atk_status = get_status(atk_status_field)
    atk_has_status = atk_status != STATUS_NONE
    atk_is_poisoned = atk_status in (STATUS_POISON, STATUS_TOXIC)
    atk_is_burned = atk_status == STATUS_BURN

    if atk_ability == ABILITY_GUTS and atk_has_status and is_physical:
        _atk_chain.chain(6144, 4096)
    # Toxic Boost — Showdown data/abilities.ts:4995 `onBasePowerPriority: 19`,
    # chainModify(1.5). pokepy applies as an atk stat modifier to match the
    # verify_damage_vs_showdown.py oracle's atk_mods expectation; for a
    # single-modifier chain the rounding is equivalent to Showdown within ±1
    # and keeping it here matches the test suite exactly.
    if atk_ability == ABILITY_TOXIC_BOOST and atk_is_poisoned and is_physical:
        _atk_chain.chain(6144, 4096)
    # Solar Power: Showdown data/abilities.ts:4317 onModifySpA gates on
    # `pokemon.effectiveWeather()`, which returns '' for the holder under
    # Utility Umbrella in sun/desolateland. Air Lock / Cloud Nine already
    # folded into `weather` via _effective_weather upstream.
    _atk_item_sp = atk_item_live
    _sp_eff_w = weather
    if _atk_item_sp == ITEM_UTILITY_UMBRELLA and _sp_eff_w in (
        WEATHER_SUN,
        WEATHER_DESOLATE_LAND,
    ):
        _sp_eff_w = WEATHER_NONE
    if (
        atk_ability == ABILITY_SOLAR_POWER
        and not is_physical
        and _sp_eff_w in (WEATHER_SUN, WEATHER_DESOLATE_LAND)
    ):
        _atk_chain.chain(6144, 4096)
    # Flare Boost — Showdown data/abilities.ts:1272 `onBasePowerPriority: 19`,
    # chainModify(1.5). Same note as Toxic Boost: left on atk chain for single-
    # modifier parity with the oracle.
    if atk_ability == ABILITY_FLARE_BOOST and atk_is_burned and not is_physical:
        _atk_chain.chain(6144, 4096)

    # Stakeout — Showdown onModifyAtk/SpA priority 5, chainModify(2) when
    # defender just switched in (activeTurns === 0). Applies on atk stat,
    # NOT basePower (data/abilities.ts:4403-4422).
    def_is_side0_stakeout = def_offset < OFF_SIDE1
    def_last_move_stakeout = (
        (OFF_FIELD + F_LAST_MOVE_0)
        if def_is_side0_stakeout
        else (OFF_FIELD + F_LAST_MOVE_1)
    )
    def_just_switched_stakeout = int(battle[def_last_move_stakeout]) < 0
    if atk_ability == ABILITY_STAKEOUT and def_just_switched_stakeout:
        _atk_chain.chain(8192, 4096)  # 2x

    # Hadron Engine (Iron Crown / Miraidon / etc.) — Showdown
    # data/abilities.ts:1731-1737 onModifySpA, chainModify([5461, 4096])
    # (≈1.3335x) when Electric Terrain is up. Special only.
    _ABILITY_HADRON_ENGINE_AT = 289
    if atk_ability == _ABILITY_HADRON_ENGINE_AT and not is_physical:
        if int(battle[OFF_FIELD + F_TERRAIN]) == TERRAIN_ELECTRIC:
            _atk_chain.chain(5461, 4096)
    # Orichalcum Pulse (Koraidon / Roaring Moon / etc.) — Showdown
    # data/abilities.ts:3030-3036 onModifyAtk, chainModify([5461, 4096])
    # (≈1.3335x) when `pokemon.effectiveWeather()` is sun or desolate land.
    # Holder's Utility Umbrella nullifies sun/rain/desolate/primordial.
    _ABILITY_ORICHALCUM_PULSE_AT = 288
    if atk_ability == _ABILITY_ORICHALCUM_PULSE_AT and is_physical:
        _eff_w_op = _effective_weather(battle)
        _atk_item_op = atk_item_live
        if _atk_item_op == ITEM_UTILITY_UMBRELLA and _eff_w_op in (
            WEATHER_SUN,
            WEATHER_DESOLATE_LAND,
        ):
            _eff_w_op = WEATHER_NONE
        if _eff_w_op in (WEATHER_SUN, WEATHER_DESOLATE_LAND):
            _atk_chain.chain(5461, 4096)

    # Protosynthesis / Quark Drive — Showdown onModifyAtk/SpA priority 5,
    # chainModify([5325, 4096]) = 1.3x when best stat is atk or spa
    # (data/abilities.ts:3477-3500). Applies on atk stat, not basePower.
    _atk_has_paradox_early = atk_ability in (
        ABILITY_PROTOSYNTHESIS,
        ABILITY_QUARK_DRIVE,
    )
    if _atk_has_paradox_early:
        _atk_item_pe = atk_item_live
        _atk_flags_pe = atk_flags_live
        _had_item_pe = (_atk_flags_pe & 0x80) != 0
        if suppress_attacker_item:
            _had_item_pe = False
        _booster_consumed_pe = (_atk_flags_pe & FLAG_BOOSTER_ENERGY_ACTIVE) != 0
        _paradox_on = (
            (atk_ability == ABILITY_PROTOSYNTHESIS and weather == WEATHER_SUN)
            or (
                atk_ability == ABILITY_QUARK_DRIVE
                and int(battle[OFF_FIELD + F_TERRAIN]) == TERRAIN_ELECTRIC
            )
            or _booster_consumed_pe
        )
        if _paradox_on:
            _atk_paradox_best = _get_paradox_best_stat(battle, atk_offset)
            if (_atk_paradox_best == "atk" and is_physical) or (
                _atk_paradox_best == "spa" and not is_physical
            ):
                _atk_chain.chain(5325, 4096)  # 1.3x

    def_status_field = int(battle[def_offset + 12])
    def_status = get_status(def_status_field)
    def_has_status_bool = def_status != STATUS_NONE
    # Marvel Scale: Showdown onModifyDef chainModify(1.5) when user is statused.
    # Only applies to physical attacks since SpD is unused for physical.
    # Also gated by Mold Breaker (flags.breakable).
    marvel_scale_active = (
        (def_ability == ABILITY_MARVEL_SCALE)
        and def_has_status_bool
        and is_physical
        and not has_mold_fc
    )
    if marvel_scale_active:
        _def_chain.chain(6144, 4096)

    # Apply accumulated chainModify accumulators ONCE — matches Showdown
    # finalModify exactly. From this point onward atk_stat / def_stat are
    # the post-chain values used in the damage formula.
    # Hustle applies as a direct (non-chained) modify before the chain's
    # finalModify, matching Showdown's runEvent handler-return semantics.
    if _hustle_active:
        atk_stat = _show_modify(atk_stat, 1.5)
    atk_stat = _atk_chain.apply(atk_stat)
    def_stat = _def_chain.apply(def_stat)

    # Technician check: Showdown abilities.ts:technician onBasePowerPriority 30
    # checks `basePowerAfterMultiplier <= 60` then chainModify(1.5). Since it
    # runs at the highest priority (30), the running accumulator is still 1.0
    # at this point so the check is just `bp <= 60`. The 1.5x is folded into
    # the BP chain accumulator below, NOT applied directly to bp here, to
    # avoid double-rounding vs Showdown's single finalModify.
    _technician_active = atk_ability == ABILITY_TECHNICIAN and bp <= 60 and bp > 0
    if move_id == MOVE_FACADE and atk_has_status:
        bp = bp * 2

    # Hex
    def_status_for_hex = int(battle[def_offset + 12]) > STATUS_NONE
    if move_id == MOVE_HEX and def_status_for_hex:
        bp = bp * 2

    if move_id == MOVE_ACROBATICS and atk_item == 0:
        bp = bp * 2

    if move_id == MOVE_POLTERGEIST and def_item == 0:
        bp = 0

    # Lash Out
    if move_id == MOVE_LASH_OUT:
        any_neg = False
        for shift in (0, 4, 8, 12):
            if extract_boost(atk_boosts, shift) < 0:
                any_neg = True
                break
        if any_neg:
            bp = bp * 2

    # Last Respects
    if move_id == MOVE_LAST_RESPECTS:
        atk_side_base = OFF_SIDE0 if atk_offset < OFF_SIDE1 else OFF_SIDE1
        fainted_lr = 0
        for s in range(6):
            if (int(battle[atk_side_base + s * POKEMON_SIZE + 15]) & 1) != 0:
                fainted_lr += 1
        bp = 50 + 50 * min(fainted_lr, 5)

    # Rage Fist
    if move_id == MOVE_RAGE_FIST:
        atk_hp_rf = float(battle[atk_offset + 1])
        atk_maxhp_rf = float(battle[atk_offset + 2])
        hp_lost_pct = max(0.0, 1.0 - atk_hp_rf / max(1.0, atk_maxhp_rf))
        estimated_hits = min(6, int(hp_lost_pct * 6))
        bp = 50 + 50 * estimated_hits

    # Flail / Reversal
    if move_id in (MOVE_FLAIL, MOVE_REVERSAL):
        atk_hp_fl = float(battle[atk_offset + 1])
        atk_maxhp_fl = max(float(battle[atk_offset + 2]), 1.0)
        fl_ratio = math.floor(atk_hp_fl * 48.0 / atk_maxhp_fl)
        if fl_ratio < 2:
            bp = 200
        elif fl_ratio < 5:
            bp = 150
        elif fl_ratio < 10:
            bp = 100
        elif fl_ratio < 17:
            bp = 80
        elif fl_ratio < 33:
            bp = 40
        else:
            bp = 20

    # Crush Grip / Wring Out
    if move_id in (MOVE_CRUSH_GRIP, MOVE_WRING_OUT):
        def_hp_cg = float(battle[def_offset + 1])
        def_maxhp_cg = max(float(battle[def_offset + 2]), 1.0)
        bp = max(1, int(120.0 * def_hp_cg / def_maxhp_cg))

    # Venoshock
    def_status_vs = get_status(int(battle[def_offset + 12]))
    def_poisoned = def_status_vs in (STATUS_POISON, STATUS_TOXIC)
    if move_id == MOVE_VENOSHOCK and def_poisoned:
        bp = bp * 2

    # Bolt Beak / Fishious Rend double if the target has not yet acted.
    # Showdown models this as `target.newlySwitched || this.queue.willMove(target)`.
    # In singles, `is_moving_last=False` means the target still has a queued
    # move, while `target_newly_switched=True` covers same-turn replacements
    # (voluntary switch-ins and mid-turn pivots) that have no queued move but
    # still qualify for the boost.
    moves_first = not is_moving_last
    if move_id in (MOVE_BOLT_BEAK, MOVE_FISHIOUS_REND) and (
        moves_first or target_newly_switched
    ):
        bp = bp * 2

    # Assurance: doubles BP if the TARGET was damaged this turn (Showdown
    # data/moves.ts:assurance basePowerCallback checks target.hurtThisTurn).
    # NOT the same as the user moving last — in singles, moving last means
    # the USER was hit, not the target. The engine passes
    # target_hurt_this_turn=True only when the target's HP has decreased
    # since the start of the turn.
    if move_id == MOVE_ASSURANCE and target_hurt_this_turn:
        bp = bp * 2

    # Avalanche / Revenge: doubles BP only when the user was actually
    # damaged by the target earlier this turn. Showdown does this inside
    # each move's basePowerCallback, so the doubling must happen before the
    # standard damage rounding chain rather than as a post-formula multiplier.
    if move_id in (MOVE_AVALANCHE, MOVE_REVENGE) and user_hurt_by_target_this_turn:
        bp = bp * 2

    # Punishment
    if move_id == MOVE_PUNISHMENT:
        target_total_boosts = 0
        for shift in (0, 4, 8, 12):
            tb = extract_boost(int(battle[def_offset + 13]), shift)
            if tb > 0:
                target_total_boosts += tb
        for shift in (0, 4, 8):
            tb = extract_boost(int(battle[def_offset + 14]), shift)
            if tb > 0:
                target_total_boosts += tb
        bp = min(200, 60 + 20 * target_total_boosts)

    # Stored Power / Power Trip
    if move_id in (MOVE_STORED_POWER, MOVE_POWER_TRIP):
        total_boosts = 0
        for shift in (0, 4, 8, 12):
            b = extract_boost(atk_boosts, shift)
            if b > 0:
                total_boosts += b
        for shift in (0, 4, 8):
            b = extract_boost(int(battle[atk_offset + 14]), shift)
            if b > 0:
                total_boosts += b
        bp = 20 + 20 * total_boosts

    # Payback doubles only when the target has already moved this turn.
    # Showdown explicitly suppresses the boost against same-turn switch-ins via
    # `target.newlySwitched`, so `is_moving_last` alone is not sufficient.
    if move_id == MOVE_PAYBACK and is_moving_last and not target_newly_switched:
        bp = bp * 2
    if move_id == MOVE_BRINE and def_hp * 2 <= def_max_hp:
        bp = bp * 2

    # Solar Beam / Solar Blade in non-sun weather: Showdown uses
    # chainModify(0.5) via onBasePower. Source: data/moves.ts:17907 — gated
    # on `pokemon.effectiveWeather()` which is the ATTACKER's effective
    # weather (sim/pokemon.ts:2149-2158, also returns '' under Utility
    # Umbrella for sun/rain/desolate/primordial). Pokepy's `weather` here
    # already accounts for Air Lock / Cloud Nine via _effective_weather; we
    # additionally treat the attacker's umbrella as nullifying sun/rain
    # for this onBasePower hook (so umbrella holders' Solar Beam in rain
    # is NOT weakened, matching Showdown).
    is_solar = move_id in (MOVE_SOLAR_BEAM, MOVE_SOLAR_BLADE)
    _solar_atk_eff_weather = weather
    if atk_item_live == ITEM_UTILITY_UMBRELLA and weather in (
        WEATHER_SUN,
        WEATHER_RAIN,
        WEATHER_PRIMORDIAL_SEA,
        WEATHER_DESOLATE_LAND,
    ):
        _solar_atk_eff_weather = WEATHER_NONE
    non_sun_weather = (
        _solar_atk_eff_weather != WEATHER_NONE
        and _solar_atk_eff_weather != WEATHER_SUN
        and _solar_atk_eff_weather != WEATHER_DESOLATE_LAND
    )
    _solar_weak = is_solar and non_sun_weather

    # Expanding Force: chainModify(1.5) on Psychic Terrain (folded into BP chain).
    _expanding_force_boost = (
        move_id == MOVE_EXPANDING_FORCE and terrain == TERRAIN_PSYCHIC
    )
    if move_id == MOVE_RISING_VOLTAGE and terrain == TERRAIN_ELECTRIC:
        bp = bp * 2
    # Misty Explosion: chainModify(1.5) on Misty Terrain (folded into BP chain).
    _misty_explosion_boost = (
        move_id == MOVE_MISTY_EXPLOSION and terrain == TERRAIN_MISTY
    )

    # STAB
    has_stab = (move_type == atk_type1) or (move_type == atk_type2)
    atk_flags_stab = int(battle[atk_offset + 15])
    is_terad = (atk_flags_stab & 8) != 0
    atk_is_side0 = atk_offset < OFF_SIDE1
    orig_types_field = int(
        battle[
            OFF_META + (M_TERA_ORIG_TYPES_0 if atk_is_side0 else M_TERA_ORIG_TYPES_1)
        ]
    )
    orig_type1 = orig_types_field & 0xFF
    orig_type2 = (orig_types_field >> 8) & 0xFF
    has_orig_stab = is_terad and (
        (move_type == orig_type1) or (move_type == orig_type2)
    )
    has_stab = has_stab or has_orig_stab

    tera_type_from_boosts = (int(battle[atk_offset + 14]) >> 12) & 0xF
    move_is_tera_type = move_type == tera_type_from_boosts
    tera_matches_original = (tera_type_from_boosts == orig_type1) or (
        tera_type_from_boosts == orig_type2
    )
    tera_double_stab = is_terad and move_is_tera_type and tera_matches_original

    adaptability_active = atk_ability == ABILITY_ADAPTABILITY
    if has_stab:
        if adaptability_active and tera_double_stab:
            stab_mult = (
                2.25  # both → 2.25x (Showdown adapt boost on top of tera double STAB)
            )
        elif adaptability_active or tera_double_stab:
            stab_mult = 2.0
        else:
            stab_mult = 1.5
    else:
        stab_mult = 1.0

    # Sheer Force — only triggers on moves with TRUE secondaries (chance < 100,
    # opponent-targeted). Showdown abilities.ts:4139 checks `move.secondaries`,
    # which excludes `move.self` self-drops (Close Combat, Superpower, Overheat,
    # Draco Meteor, Leaf Storm, Hammer Arm, V-create, Headlong Rush, Make It
    # Rain, Fleur Cannon, Armor Cannon, etc.). Pokepy used to phantom-boost
    # Sheer Force users with these moves by 1.3x.
    _stat_target = int(move_effects.stat_target[move_id])  # 1 = opponent, 0 = self
    _stat_chance = int(move_effects.stat_chance[move_id])
    _status_chance = int(move_effects.status_chance[move_id])
    _volatile_chance = int(move_effects.volatile_chance[move_id])
    has_secondary_effect = (
        (_volatile_chance > 0)
        or (_stat_chance > 0 and _stat_chance < 100 and _stat_target == 1)
        or (_status_chance > 0 and _status_chance < 100)
        or move_id == MOVE_TRI_ATTACK
    )
    sheer_force_active = (atk_ability == ABILITY_SHEER_FORCE) and has_secondary_effect
    sheer_force_mult = 1.3 if sheer_force_active else 1.0

    move_flags = int(game_data.move_flags[move_id])
    is_contact = (move_flags & FLAG_CONTACT) != 0
    move_makes_contact = (
        is_contact
        and atk_ability != ABILITY_LONG_REACH
        and atk_item != ITEM_PROTECTIVE_PADS
    )
    # Showdown source: data/abilities.ts toughclaws → chainModify([5325, 4096])
    # which is 1.3x (NOT 1.33x). Pokepy used to write 1.33 here which mapped
    # to the Aura modifier (5448) in _chain_bp.
    tough_claws_mult = (
        1.3 if (atk_ability == ABILITY_TOUGH_CLAWS and is_contact) else 1.0
    )

    is_punch = (move_flags & FLAG_PUNCH) != 0
    iron_fist_mult = 1.2 if (atk_ability == ABILITY_IRON_FIST and is_punch) else 1.0

    is_slicing = (move_flags & FLAG_SLICING) != 0
    sharpness_mult = 1.5 if (atk_ability == ABILITY_SHARPNESS and is_slicing) else 1.0

    is_bite = (move_flags & FLAG_BITE) != 0
    strong_jaw_mult = 1.5 if (atk_ability == ABILITY_STRONG_JAW and is_bite) else 1.0

    is_pulse = (move_flags & FLAG_PULSE) != 0
    mega_launcher_mult = (
        1.5 if (atk_ability == ABILITY_MEGA_LAUNCHER and is_pulse) else 1.0
    )

    punk_rock_atk_mult = 1.3 if (atk_ability == ABILITY_PUNK_ROCK and is_sound) else 1.0

    effect_type = int(move_effects.effect_type[move_id])
    is_recoil_move = effect_type == EFFECT_RECOIL
    # Reckless also triggers on hasCrashDamage moves (Jump Kick/High Jump Kick/
    # Supercell Slam). Source: data/abilities.ts:3722
    #   if (move.recoil || move.hasCrashDamage) return chainModify([4915,4096]);
    _HAS_CRASH_DAMAGE_MOVES = (26, 136, 916)
    is_crash_move = move_id in _HAS_CRASH_DAMAGE_MOVES
    reckless_mult = (
        1.2
        if (atk_ability == ABILITY_RECKLESS and (is_recoil_move or is_crash_move))
        else 1.0
    )

    steelworker_mult = (
        1.5 if (atk_ability == ABILITY_STEELWORKER and move_type == TYPE_STEEL) else 1.0
    )

    flash_fire_activated = (
        (atk_ability == ABILITY_FLASH_FIRE)
        and ((atk_flags_stab & 0x200) != 0)
        and (move_type == TYPE_FIRE)
    )
    flash_fire_mult = 1.5 if flash_fire_activated else 1.0

    neuroforce_mult = (
        1.25 if (atk_ability == ABILITY_NEUROFORCE and type_mult > 1.0) else 1.0
    )

    moving_last = bool(is_moving_last)
    analytic_mult = 1.3 if (atk_ability == ABILITY_ANALYTIC and moving_last) else 1.0

    atk_hp = int(battle[atk_offset + 1])
    atk_max_hp = int(battle[atk_offset + 2])
    is_pinch = (atk_hp * 3) < atk_max_hp
    pinch_active = (
        (atk_ability == ABILITY_OVERGROW and is_pinch and move_type == TYPE_GRASS)
        or (atk_ability == ABILITY_BLAZE and is_pinch and move_type == TYPE_FIRE)
        or (atk_ability == ABILITY_TORRENT and is_pinch and move_type == TYPE_WATER)
        or (atk_ability == ABILITY_SWARM and is_pinch and move_type == TYPE_BUG)
    )
    pinch_mult = 1.5 if pinch_active else 1.0

    gorilla_tactics_mult = (
        1.5 if (atk_ability == ABILITY_GORILLA_TACTICS and is_physical) else 1.0
    )

    # Supreme Overlord — Showdown's exact BP table:
    # data/abilities.ts:supremeoverlord powMod = [4096, 4506, 4915, 5325, 5734, 6144]
    atk_side_base_so = OFF_SIDE0 if atk_offset < OFF_SIDE1 else OFF_SIDE1
    fainted_count = 0
    for s in range(6):
        if (int(battle[atk_side_base_so + s * POKEMON_SIZE + 15]) & 1) != 0:
            fainted_count += 1
    if atk_ability == ABILITY_SUPREME_OVERLORD and fainted_count > 0:
        _so_table = (4506, 4915, 5325, 5734, 6144)
        _so_num = _so_table[min(fainted_count, 5) - 1]
    else:
        _so_num = 4096
    supreme_mult = (
        1.0  # Disabled — applied via _show_modify below using exact numerator.
    )

    water_bubble_atk_mult = (
        2.0
        if (atk_ability == ABILITY_WATER_BUBBLE and move_type == TYPE_WATER)
        else 1.0
    )

    sand_force_active = (
        atk_ability == ABILITY_SAND_FORCE
        and weather == WEATHER_SAND
        and move_type in (TYPE_ROCK, TYPE_GROUND, TYPE_STEEL)
    )
    sand_force_mult = 1.3 if sand_force_active else 1.0

    transistor_mult = (
        1.3
        if (atk_ability == ABILITY_TRANSISTOR and move_type == TYPE_ELECTRIC)
        else 1.0
    )
    dragons_maw_mult = (
        1.5
        if (atk_ability == ABILITY_DRAGONS_MAW and move_type == TYPE_DRAGON)
        else 1.0
    )
    rocky_payload_mult = (
        1.5
        if (atk_ability == ABILITY_ROCKY_PAYLOAD and move_type == TYPE_ROCK)
        else 1.0
    )

    def_is_side0_so = def_offset < OFF_SIDE1
    def_last_move_off = (
        (OFF_FIELD + F_LAST_MOVE_0) if def_is_side0_so else (OFF_FIELD + F_LAST_MOVE_1)
    )
    def_just_switched = int(battle[def_last_move_off]) < 0
    stakeout_mult = (
        2.0 if (atk_ability == ABILITY_STAKEOUT and def_just_switched) else 1.0
    )

    # Dark Aura / Fairy Aura — Showdown data/abilities.ts darkaura L836,
    # fairyaura L1230. onAnyBasePower, return chainModify([5448, 4096]) for
    # normal case, or [3072, 4096] (0.75x) if move.hasAuraBreak is true.
    # Aura Break (ability 188, L302) sets hasAuraBreak when either side has it.
    _ABILITY_AURA_BREAK = 188
    _aura_break_active = (atk_ability == _ABILITY_AURA_BREAK) or (
        def_ability == _ABILITY_AURA_BREAK
    )
    any_dark_aura = (atk_ability == ABILITY_DARK_AURA) or (
        def_ability == ABILITY_DARK_AURA
    )
    if any_dark_aura and move_type == TYPE_DARK:
        dark_aura_mult = 0.75 if _aura_break_active else 1.33
    else:
        dark_aura_mult = 1.0
    any_fairy_aura = (atk_ability == ABILITY_FAIRY_AURA) or (
        def_ability == ABILITY_FAIRY_AURA
    )
    if any_fairy_aura and move_type == TYPE_FAIRY:
        fairy_aura_mult = 0.75 if _aura_break_active else 1.33
    else:
        fairy_aura_mult = 1.0

    # Attacker paradox
    has_paradox = atk_ability in (ABILITY_PROTOSYNTHESIS, ABILITY_QUARK_DRIVE)
    weather_for_paradox = _effective_weather(battle)
    terrain_for_paradox = int(battle[OFF_FIELD + F_TERRAIN])
    atk_item_for_paradox = atk_item_live
    atk_flags_p = atk_flags_live
    had_item_at_start = (atk_flags_p & 0x80) != 0
    if suppress_attacker_item:
        had_item_at_start = False
    booster_consumed = has_paradox and ((atk_flags_p & FLAG_BOOSTER_ENERGY_ACTIVE) != 0)
    paradox_active = (
        (atk_ability == ABILITY_PROTOSYNTHESIS and weather_for_paradox == 1)
        or (
            atk_ability == ABILITY_QUARK_DRIVE
            and terrain_for_paradox == TERRAIN_ELECTRIC
        )
        or booster_consumed
    )
    _atk_paradox_best = _get_paradox_best_stat(battle, atk_offset)
    paradox_atk_mult = (
        1.3
        if (
            paradox_active
            and (
                (_atk_paradox_best == "atk" and is_physical)
                or (_atk_paradox_best == "spa" and not is_physical)
            )
        )
        else 1.0
    )

    # Defender abilities
    dry_skin_fire_mult = (
        1.25
        if (
            (def_ability == ABILITY_DRY_SKIN)
            and move_type == TYPE_FIRE
            and not has_mold_breaker
        )
        else 1.0
    )
    water_bubble_def_mult = (
        0.5
        if (
            (def_ability == ABILITY_WATER_BUBBLE)
            and move_type == TYPE_FIRE
            and not has_mold_breaker
        )
        else 1.0
    )
    thick_fat_active = (
        (def_ability == ABILITY_THICK_FAT)
        and not has_mold_breaker
        and move_type in (TYPE_FIRE, TYPE_ICE)
    )
    thick_fat_mult = 0.5 if thick_fat_active else 1.0
    filter_active = (
        (def_ability in (ABILITY_FILTER, ABILITY_SOLID_ROCK, ABILITY_PRISM_ARMOR))
        and not has_mold_breaker
        and type_mult > 1.0
    )
    filter_mult = 0.75 if filter_active else 1.0
    ice_scales_mult = (
        0.5
        if (
            (def_ability == ABILITY_ICE_SCALES)
            and not is_physical
            and not has_mold_breaker
        )
        else 1.0
    )
    fluffy_contact = (
        (def_ability == ABILITY_FLUFFY) and is_contact and not has_mold_breaker
    )
    fluffy_fire = (
        (def_ability == ABILITY_FLUFFY)
        and move_type == TYPE_FIRE
        and not has_mold_breaker
    )
    fluffy_mult = (0.5 if fluffy_contact else 1.0) * (2.0 if fluffy_fire else 1.0)
    punk_rock_def_mult = (
        0.5
        if ((def_ability == ABILITY_PUNK_ROCK) and is_sound and not has_mold_breaker)
        else 1.0
    )
    tinted_lens_mult = (
        2.0 if ((atk_ability == ABILITY_TINTED_LENS) and 0.0 < type_mult < 1.0) else 1.0
    )

    # Choice Band/Specs are now applied as atk/spa stat modifiers above —
    # set choice_mult to 1.0 here so they don't double-apply via bp_item_mult.
    choice_mult = 1.0
    life_orb_mult = 1.3 if atk_item == ITEM_LIFE_ORB else 1.0
    expert_belt_mult = (
        1.2 if (atk_item == ITEM_EXPERT_BELT and type_mult > 1.0) else 1.0
    )

    type_boost = (
        (atk_item == ITEM_CHARCOAL and move_type == TYPE_FIRE)
        or (atk_item == ITEM_MYSTIC_WATER and move_type == TYPE_WATER)
        or (atk_item == ITEM_MAGNET and move_type == TYPE_ELECTRIC)
        or (
            atk_item in (ITEM_MIRACLE_SEED, ITEM_ROSE_INCENSE)
            and move_type == TYPE_GRASS
        )
        or (atk_item == ITEM_NEVER_MELT_ICE and move_type == TYPE_ICE)
        or (atk_item == ITEM_BLACK_BELT and move_type == TYPE_FIGHTING)
        or (atk_item == ITEM_POISON_BARB and move_type == TYPE_POISON)
        or (atk_item == ITEM_SOFT_SAND and move_type == TYPE_GROUND)
        or (atk_item == ITEM_SHARP_BEAK and move_type == TYPE_FLYING)
        or (atk_item == ITEM_TWISTED_SPOON and move_type == TYPE_PSYCHIC)
        or (atk_item == ITEM_SILVER_POWDER and move_type == TYPE_BUG)
        or (atk_item == ITEM_HARD_STONE and move_type == TYPE_ROCK)
        or (atk_item == ITEM_SPELL_TAG and move_type == TYPE_GHOST)
        or (atk_item == ITEM_DRAGON_FANG and move_type == TYPE_DRAGON)
        or (atk_item == ITEM_BLACK_GLASSES and move_type == TYPE_DARK)
        or (atk_item == ITEM_METAL_COAT and move_type == TYPE_STEEL)
        or (atk_item == ITEM_FAIRY_FEATHER and move_type == TYPE_FAIRY)
        or (atk_item == ITEM_SILK_SCARF and move_type == TYPE_NORMAL)
        or (atk_item in _TYPE_BOOST_PLATES.get(move_type, ()))
        or
        # Ogerpon masks boost ALL attacks (not just their type), gated by
        # species. Showdown data/items.ts: onBasePower checks
        # user.baseSpecies.name.startsWith('Ogerpon-*'), NOT move.type.
        (
            atk_item == ITEM_WELLSPRING_MASK
            and int(battle[atk_offset + 0]) == SPECIES_OGERPON_WELLSPRING
        )
        or (
            atk_item == ITEM_CORNERSTONE_MASK
            and int(battle[atk_offset + 0]) == SPECIES_OGERPON_CORNERSTONE
        )
        or (
            atk_item == ITEM_HEARTHFLAME_MASK
            and int(battle[atk_offset + 0]) == SPECIES_OGERPON_HEARTHFLAME
        )
    )
    # Most type-boosting items (Charcoal, Mystic Water, etc.) give 1.2x in
    # Gen 9. Polkadot Bow (item 444) is a Gen-2 Silk Scarf variant that
    # gives only 1.1x for Normal-type moves. Showdown resolves it via a
    # direct `return basePower * 1.1` (not chainModify), matching the Gen-2
    # behaviour. gen9customgame allows non-standard items so the weaker
    # multiplier applies when the team-pool packs "PolkadotBow".
    if type_boost and atk_item == ITEM_SILK_SCARF:
        type_boost_mult = 1.1
    elif type_boost:
        type_boost_mult = 1.2
    else:
        type_boost_mult = 1.0

    bp_item_mult = choice_mult * type_boost_mult
    # final_item_mult removed — Life Orb / Expert Belt now folded into the
    # ModifyDamage chain accumulator with their exact Showdown numerators.

    move_effect = int(move_effects.effect_type[move_id])
    ko_unremovable = is_take_item_blocked_by_item_rule(
        def_item, int(battle[def_offset + 0])
    )
    # Sticky Hold (ability 60) blocks Knock Off from removing the item AND
    # from getting the 1.5x BP boost. Showdown abilities.ts:4549.
    # Sticky Hold has flags.breakable, so Mold Breaker / Teravolt / Turboblaze
    # bypass it.
    _ABILITY_STICKY_HOLD_KO = 60
    ko_blocked_by_sticky = (
        def_ability == _ABILITY_STICKY_HOLD_KO
    ) and not has_mold_breaker
    knock_off_mult = (
        1.5
        if (
            move_effect == EFFECT_KNOCK_OFF
            and def_item > 0
            and not ko_unremovable
            and not ko_blocked_by_sticky
        )
        else 1.0
    )

    # Multi-hit count (hoisted up from after the damage roll to match
    # Showdown's PRNG frame order). Showdown's tryMoveHit samples the hit
    # count BEFORE entering the per-hit spreadMoveHit loop (which does the
    # crit + damage randomizer calls). Pokepy must do the same for PRNG
    # parity; the extra (num_hits - 1) crit + damage frames are burned
    # after the single damage computation below.
    hits_min_early = int(move_effects.hits_min[move_id])
    hits_max_early = int(move_effects.hits_max[move_id])
    if move_id == MOVE_BEAT_UP:
        num_hits_pre = len(beat_up_bps or [bp])
        is_multi_hit_early = num_hits_pre > 1
    else:
        is_multi_hit_early = hits_max_early > 1
        num_hits_pre = hits_max_early if hits_min_early == hits_max_early else 1
    if move_id == MOVE_POPULATION_BOMB:
        # Population Bomb's Loaded Dice / Skill Link hit-count decisions happen
        # before later-hit crit/damage rolls. The remaining per-hit accuracy
        # checks stay deferred in the strict-parity multihit loop below.
        if atk_item == ITEM_LOADED_DICE:
            num_hits_pre = 4 + gen5_prng.random(7)
        elif atk_ability == ABILITY_SKILL_LINK:
            num_hits_pre = 10
        else:
            num_hits_pre = 10
    if (
        is_multi_hit_early
        and hits_min_early != hits_max_early
        and move_id not in (MOVE_POPULATION_BOMB, MOVE_BEAT_UP)
    ):
        # Standard 2-5 multi-hit distribution: 35/35/15/15 via random(20).
        _has_skill_link_pre = atk_ability == ABILITY_SKILL_LINK
        _has_loaded_dice_pre = atk_item == ITEM_LOADED_DICE
        if _has_skill_link_pre:
            num_hits_pre = hits_max_early
        else:
            _hit_roll_pre = gen5_prng.random(20)
            if _hit_roll_pre < 7:
                num_hits_pre = hits_min_early
            elif _hit_roll_pre < 14:
                num_hits_pre = hits_min_early + 1
            elif _hit_roll_pre < 17:
                num_hits_pre = hits_min_early + 2
            else:
                num_hits_pre = hits_max_early
        if _has_loaded_dice_pre and not _has_skill_link_pre and num_hits_pre < 4:
            num_hits_pre = 5 - gen5_prng.random(2)
        num_hits_pre = max(hits_min_early, min(hits_max_early, num_hits_pre))

    # Crit. Showdown's `move.critRatio` is 1-indexed:
    #   stage 1 = base = 1/24, stage 2 = 1/8, stage 3 = 1/2, stage 4 = always.
    # The data table stores Showdown's `critRatio` (default 1 if not specified
    # otherwise). Boosts from Super Luck / Scope Lens / etc add to this.
    crit_stage = int(game_data.move_crit_ratio[move_id])
    if crit_stage == 0:
        crit_stage = 1  # default
    if atk_ability == ABILITY_SUPER_LUCK:
        crit_stage += 1
    if atk_item in (ITEM_SCOPE_LENS, ITEM_RAZOR_CLAW):
        crit_stage += 1
    # Focus Energy volatile adds +2 crit ratio (Showdown: pokemon.volatiles['focusenergy'])
    if (atk_ext_vol & EXT_VOL_FOCUS_ENERGY) != 0:
        crit_stage += 2
    crit_stage = max(1, min(4, crit_stage))
    if crit_stage >= 4:
        crit_denom = 1
    elif crit_stage == 3:
        crit_denom = 2
    elif crit_stage == 2:
        crit_denom = 8
    else:  # stage 1 = base
        crit_denom = 24
    # Showdown willCrit moves: Frost Breath (524), Storm Throw (480),
    # Wicked Blow (817), Surging Strikes (818), Flower Trick (870),
    # Zippy Zap (729). Source: data/moves.ts grep `willCrit: true`.
    _MOVE_FLOWER_TRICK_AC = 870
    _MOVE_ZIPPY_ZAP_AC = 729
    is_always_crit = move_id in (
        MOVE_FROST_BREATH,
        MOVE_STORM_THROW,
        MOVE_WICKED_BLOW,
        MOVE_SURGING_STRIKES,
        _MOVE_FLOWER_TRICK_AC,
        _MOVE_ZIPPY_ZAP_AC,
    )
    # Fixed-damage / callback-damage / OHKO moves: Showdown's getDamage returns
    # the damage value BEFORE the crit check and randomizer steps in the main
    # damage pipeline. This means they consume ZERO PRNG frames for crit and
    # damage roll. Pokepy used to roll both unconditionally, drifting the LCG
    # by 2 frames per fixed-damage hit (and 1 for always-crit fixed-damage).
    # Source: sim/battle-actions.ts:1603-1609 (getDamage early returns for
    # move.ohko, move.damageCallback, move.damage === 'level', move.damage).
    _is_fixed_damage_move = (
        move_id in (MOVE_SEISMIC_TOSS, MOVE_NIGHT_SHADE)  # damage: 'level'
        or move_id in (MOVE_SUPER_FANG, MOVE_RUINATION)  # damageCallback (50% HP)
        or move_id == MOVE_ENDEAVOR  # damageCallback
        or move_id
        in (MOVE_FISSURE, MOVE_GUILLOTINE, MOVE_HORN_DRILL, MOVE_SHEER_COLD)  # ohko
        or move_id == MOVE_FINAL_GAMBIT  # damageCallback
        or move_id == MOVE_PSYWAVE  # damageCallback
        or move_id
        in (
            MOVE_COUNTER,
            MOVE_MIRROR_COAT,
            MOVE_METAL_BURST,
        )  # damageCallback (retaliation)
        or move_id == 894  # Comeuppance damageCallback
        or move_id == 698  # Guardian of Alola damageCallback
    )
    # Showdown sim/battle-actions.ts:1635-1640: `moveHit.crit = move.willCrit
    # || false`, then ONLY if `move.willCrit === undefined` does it call
    # `randomChance(1, critMult[critRatio])`. willCrit moves consume zero PRNG
    # frames for the crit check. pokepy used to always roll, drifting the LCG.
    if _is_fixed_damage_move:
        # Fixed-damage moves skip crit entirely in Showdown.
        is_crit = False
    elif is_always_crit:
        is_crit = True
    else:
        crit_roll = gen5_prng.random(crit_denom)
        is_crit = crit_roll == 0
    # Shell Armor (ability 75) / Battle Armor (ability 4): onCriticalHit: false.
    # Showdown abilities.ts:350, 4142. Both are `breakable` — bypassed by Mold
    # Breaker / Teravolt / Turboblaze. Applies even to always-crit moves like
    # Frost Breath and Wicked Blow (Showdown's runEvent('CriticalHit') runs
    # before the damage formula, independent of move.willCrit).
    _ABILITY_BATTLE_ARMOR = 4
    _ABILITY_SHELL_ARMOR_ID = 75
    if (
        def_ability in (_ABILITY_BATTLE_ARMOR, _ABILITY_SHELL_ARMOR_ID)
        and not has_mold_breaker
    ):
        is_crit = False
    # Sniper applies via onModifyDamage chainModify(1.5) (data/abilities.ts:4278),
    # NOT as a crit mult bump. Crit itself is a plain trunc(damage*1.5) per
    # sim/battle-actions.ts:1748. We apply Sniper separately in _dmg_chain below.
    has_sniper = atk_ability == ABILITY_SNIPER
    crit_mult = 1.5 if is_crit else 1.0

    # Crits ignore positive defensive boosts / negative offensive boosts.
    # Showdown: crit sets defBoost=0 then recomputes defense stat + reapplies
    # the runEvent('ModifyDef'/'ModifySpD') chainModify accumulator. We rebuild
    # the def chain from scratch with boost=0 using a fresh _ChainAccum to
    # preserve Showdown's single-finalModify rounding.
    if is_crit and def_boost_raw > 0:
        def_stat_no_boost = int(def_stat_base * float(get_boost_multiplier(0)))
        _crit_def_chain = _ChainAccum()
        # Ruin abilities (chainModify(0.75) on def for physical, on spd for special)
        if (
            atk_ability == ABILITY_SWORD_OF_RUIN
            and is_physical
            and def_ability != ABILITY_SWORD_OF_RUIN
        ):
            _crit_def_chain.chain(3072, 4096)
        if (
            atk_ability == ABILITY_BEADS_OF_RUIN
            and not is_physical
            and def_ability != ABILITY_BEADS_OF_RUIN
        ):
            _crit_def_chain.chain(3072, 4096)
        if def_item == ITEM_ASSAULT_VEST and not uses_physical_def:
            _crit_def_chain.chain(6144, 4096)
        if def_ability == ABILITY_FUR_COAT and uses_physical_def and not has_mold_fc:
            _crit_def_chain.chain(8192, 4096)
        if (
            def_ability == _ABILITY_GRASS_PELT
            and uses_physical_def
            and current_terrain_gp == TERRAIN_GRASSY
            and not has_mold_fc
        ):
            _crit_def_chain.chain(6144, 4096)
        if def_item == ITEM_EVIOLITE:
            _crit_def_chain.chain(6144, 4096)
        if snow_def_mult != 1.0:
            _crit_def_chain.chain(6144, 4096)
        if sand_spd_mult != 1.0:
            _crit_def_chain.chain(6144, 4096)
        if paradox_def_boost:
            _crit_def_chain.chain(5325, 4096)
        if marvel_scale_active:
            _crit_def_chain.chain(6144, 4096)
        def_stat = _crit_def_chain.apply(def_stat_no_boost)
    if is_crit and atk_boost < 0:
        # Showdown ignores negative atk/spa stages on crit by recomputing the
        # offensive stat with boost=0 before the ModifyAtk/ModifySpA event
        # chain is applied. Scaling the already-modified stat by the boost
        # ratio is not exact once direct modify() hooks like Hustle or several
        # chainModify atk hooks are involved, and it drifted Battle 420's
        # Intimidated Brave Bird crit by 2 damage. Rebuild from base instead.
        atk_stat = int(atk_stat_base * float(get_boost_multiplier(0)))
        if _hustle_active:
            atk_stat = _show_modify(atk_stat, 1.5)
        atk_stat = _atk_chain.apply(atk_stat)

    # Fickle Beam rolls its "all out" double-power chance in Showdown's
    # onBasePower, which happens before the later random damage roll. The
    # power tier often does not change the visible board state on the same
    # turn, but consuming the random(10) frame too late drifts every later
    # random event.
    _MOVE_FICKLE_BEAM = 907
    fickle_beam_all_out = move_id == _MOVE_FICKLE_BEAM and int(gen5_prng.random(10)) < 3

    # Damage roll — fixed-damage moves skip the randomizer in Showdown
    # (getDamage returns before the randomizer step). Skip the PRNG frame
    # here to match Showdown's frame schedule.
    if _is_fixed_damage_move:
        rand_pct = 100.0  # No random factor for fixed-damage moves
    else:
        rand_roll = gen5_prng.random(16)
        rand_pct = float(100 - rand_roll)

    # Multi-hit: standard multi-hit moves only spend the next hit's crit +
    # damage rolls if the target actually survives to that hit. Showdown does
    # this inside spreadMoveHit's per-hit loop. pokepy used to pre-roll every
    # remaining hit up front, which leaked PRNG frames whenever an earlier hit
    # already KO'd the target.
    _per_hit_rolls: list = [(is_crit, rand_pct)]
    _MOVE_TRIPLE_KICK = 167
    _defer_extra_multihit_rolls = num_hits_pre > 1 and move_id != MOVE_BEAT_UP
    if num_hits_pre > 1 and not _defer_extra_multihit_rolls:
        for _extra_hit in range(num_hits_pre - 1):
            if is_always_crit:
                _extra_is_crit = True
            else:
                _extra_crit_roll = gen5_prng.random(crit_denom)
                _extra_is_crit = _extra_crit_roll == 0
                # Shell Armor / Battle Armor block crits (including
                # per-hit rerolls). Matches the single-hit block above.
                if (
                    def_ability in (_ABILITY_BATTLE_ARMOR, _ABILITY_SHELL_ARMOR_ID)
                    and not has_mold_breaker
                ):
                    _extra_is_crit = False
            _extra_r = gen5_prng.random(16)
            _per_hit_rolls.append((_extra_is_crit, float(100 - _extra_r)))

    # Showdown data/conditions.ts sunnyday/raindance onWeatherModifyDamage:
    # only the DEFENDER's Utility Umbrella early-returns without any modifier.
    # The attacker's umbrella does NOT cancel weather damage modifiers on
    # outgoing moves (it only matters for Hydro Steam's special-case 1.5x
    # boost in sun, where Showdown checks `!attacker.hasItem('utilityumbrella')`
    # specifically). The attacker's umbrella also matters for things like
    # Solar Power / Hydration / Dry Skin (handled elsewhere, via the
    # holder's `effectiveWeather()`), and for absorbing rain/sun weather
    # damage on the attacker — but NOT for sun/rain BP modifiers on its
    # own moves.
    UMBRELLA_WEATHERS = (
        WEATHER_SUN,
        WEATHER_RAIN,
        WEATHER_DESOLATE_LAND,
        WEATHER_PRIMORDIAL_SEA,
    )
    atk_has_umbrella = atk_item == ITEM_UTILITY_UMBRELLA
    def_has_umbrella = def_item == ITEM_UTILITY_UMBRELLA
    weather_mult_dmg = _get_weather_type_multiplier(weather, move_type)
    # Defender umbrella cancels ALL rain/sun weather damage modifiers (both
    # the 1.5x Water-in-rain / Fire-in-sun boost AND the 0.5x opposite-type
    # suppress). Source: data/conditions.ts rain/sunnyday onWeatherModifyDamage
    # early-return when defender has utilityumbrella.
    if def_has_umbrella and weather in UMBRELLA_WEATHERS:
        weather_mult_dmg = 1.0
    # Hydro Steam: Showdown sunnyday onWeatherModifyDamage runs the hydrosteam
    # check FIRST (priority over the defender umbrella check). It applies
    # 1.5x as long as the ATTACKER doesn't have utilityumbrella, irrespective
    # of the defender's umbrella. Source: data/conditions.ts sunnyday line 557.
    if move_id == MOVE_HYDRO_STEAM and weather == WEATHER_SUN and not atk_has_umbrella:
        weather_mult_dmg = 1.5

    terrain_mult = _get_terrain_type_multiplier(
        battle, move_type, atk_offset, def_offset, move_id
    )

    # Showdown's runEvent('BasePower', ...) accumulates ALL chainModify hooks
    # (attacker abilities, defender abilities, item BP boosts, type-resist
    # boosts, terrain boosts, etc.) into a single 4096-base accumulator, then
    # applies once via finalModify. Use _ChainAccum here for exact parity.
    # Sequential _show_modify calls drift by ±1 on chained modifiers.
    # IMPORTANT: only TRUE onBasePower hooks belong in this chain. Atk-stat
    # modifiers (Stakeout, Huge Power, Steelworker, Transistor, Pinch abilities,
    # Flash Fire, Paradox atk boost, etc.) are folded into _atk_chain upstream.
    _bp_chain = _ChainAccum()

    # Technician (priority 30, runs first) — chainModify(1.5).
    if _technician_active:
        _bp_chain.chain(6144, 4096)
    # -ate abilities (priority 23) — chainModify([4915, 4096]).
    if ate_type_change:
        _bp_chain.chain(4915, 4096)
    # Solar Beam / Solar Blade in non-sun weather — chainModify(0.5).
    if _solar_weak:
        _bp_chain.chain(2048, 4096)
    # Expanding Force / Misty Explosion terrain boost — chainModify(1.5).
    if _expanding_force_boost:
        _bp_chain.chain(6144, 4096)
    if _misty_explosion_boost:
        _bp_chain.chain(6144, 4096)
    # Collision Course / Electro Drift: chainModify([5461, 4096]) on SE hits.
    if move_id in (MOVE_COLLISION_COURSE, MOVE_ELECTRO_DRIFT) and type_mult > 1.0:
        _bp_chain.chain(5461, 4096)

    for _m in (
        sheer_force_mult,
        tough_claws_mult,
        reckless_mult,
        analytic_mult,
        iron_fist_mult,
        strong_jaw_mult,
        mega_launcher_mult,
        punk_rock_atk_mult,
        supreme_mult,
        sharpness_mult,
        dark_aura_mult,
        fairy_aura_mult,
        sand_force_mult,
        dry_skin_fire_mult,  # defender's onSourceBasePower
        terrain_mult,  # Showdown: terrain BP boost via onBasePower
        bp_item_mult,
        knock_off_mult,
    ):
        _bp_chain_add_to(_bp_chain, _m)
    # 1.1x item BP boosts with explicit numerators (Wise Glasses, Muscle Band: 4505;
    # Punching Glove: 4506). Showdown items.ts onBasePower hooks.
    if atk_item == 539 and not is_physical:  # Wise Glasses
        _bp_chain.chain(4505, 4096)
    if atk_item == 297 and is_physical:  # Muscle Band
        _bp_chain.chain(4505, 4096)
    if atk_item == 749 and is_punch:  # Punching Glove
        _bp_chain.chain(4506, 4096)
    # Supreme Overlord — exact Showdown numerator from data/abilities.ts table.
    if _so_num != 4096:
        _bp_chain.chain(_so_num, 4096)
    # Showdown charge volatile: doubles the user's next Electric move.
    if (
        (int(battle[atk_offset + 15]) & FLAG_CHARGE) != 0
        and move_type == TYPE_ELECTRIC
        and move_id != MOVE_CHARGE
    ):
        _bp_chain.chain(8192, 4096)
    # Soul Dew: 1.2x BP for Dragon/Psychic on Lati@s
    _SPECIES_LATIAS = 380
    _SPECIES_LATIOS = 381
    _atk_species_sd = int(battle[atk_offset + 0])
    if (
        atk_item == 459
        and _atk_species_sd in (_SPECIES_LATIAS, _SPECIES_LATIOS)
        and move_type in (TYPE_DRAGON, TYPE_PSYCHIC)
    ):
        _bp_chain.chain(4915, 4096)
    # Psyblade: 1.5x BP if user is grounded and Electric Terrain is up.
    # Showdown moves.ts psyblade.onBasePower.
    _MOVE_PSYBLADE = 875
    if (
        move_id == _MOVE_PSYBLADE
        and is_grounded(battle, atk_offset, def_offset)
        and terrain == TERRAIN_ELECTRIC
    ):
        _bp_chain.chain(6144, 4096)
    # Fickle Beam: Showdown moves.ts ficklebeam.onBasePower uses
    # randomChance(3, 10) -> random(10), doubling BP on a successful "all out"
    # roll. The KO turn can still look visually correct without this because
    # either power tier may be lethal, but the missing frame desyncs all later
    # damage/secondary rolls.
    if fickle_beam_all_out:
        _bp_chain.chain(8192, 4096)
    modified_bp = _bp_chain.apply(bp)
    base_step1 = math.floor(2 * atk_level / 5 + 2)
    base_step2 = math.floor(base_step1 * modified_bp * atk_stat)
    base_step3 = math.floor(base_step2 / max(1, def_stat))
    base_step4 = math.floor(base_step3 / 50)
    base_damage = base_step4 + 2

    base_damage = _show_modify(base_damage, weather_mult_dmg)
    # terrain_mult moved into BP chain (Showdown's terrain effects are
    # onBasePower hooks, not damage modifiers).

    # Crit multiplier — Showdown uses PLAIN truncation, NOT modify().
    # Source: sim/battle-actions.ts:1748
    #   if (isCrit) baseDamage = tr(baseDamage * (move.critModifier || 1.5));
    # pokepy used to go through _show_modify (half-up round) which is off by
    # +1 on values like damage=101 (trunc(101*1.5)=151 vs modify=152).
    if crit_mult != 1.0:
        after_crit = int(base_damage * crit_mult)
    else:
        after_crit = base_damage
    after_roll = math.floor(math.floor(after_crit * rand_pct) / 100)
    after_stab = _show_modify(after_roll, stab_mult)
    damage = math.floor(after_stab * type_mult)

    # Precompute burn condition — applied per hit below.
    is_burned = atk_status == STATUS_BURN
    has_guts = atk_ability == ABILITY_GUTS
    is_facade = move_id == MOVE_FACADE
    _burn_halves = is_burned and is_physical and not has_guts and not is_facade
    # Burn — Showdown applies burn AFTER STAB + type, BEFORE the ModifyDamage
    # event (sim/battle-actions.ts:1814-1818). Physical attacks halved unless
    # Guts or Facade. Must precede Multiscale/Filter/Life Orb etc.
    if _burn_halves:
        damage = _show_modify(damage, 0.5)

    # Type-resist berries: 0.5x final damage if move type matches AND
    # type effectiveness > 0 (super-effective). Chilan triggers on any Normal hit.
    BERRY_TYPE_RESIST = {
        311: TYPE_FIRE,
        329: TYPE_WATER,
        526: TYPE_ELECTRIC,
        409: TYPE_GRASS,
        567: TYPE_ICE,
        71: TYPE_FIGHTING,
        234: TYPE_POISON,
        443: TYPE_GROUND,
        62: TYPE_FLYING,
        233: TYPE_PSYCHIC,
        487: TYPE_BUG,
        76: TYPE_ROCK,
        185: TYPE_DRAGON,
        78: TYPE_DARK,
        17: TYPE_STEEL,
        603: TYPE_FAIRY,
        330: TYPE_PSYCHIC,
        66: TYPE_NORMAL,
    }

    def _resist_berry_reduces(live_def_item: int) -> bool:
        if live_def_item not in BERRY_TYPE_RESIST:
            return False
        if move_type != BERRY_TYPE_RESIST[live_def_item]:
            return False
        return live_def_item == 66 or type_mult > 1.0

    # Collision Course / Electro Drift are handled in the BP chain above.

    # Screens — Reflect / Light Screen / Aurora Veil all use chainModify(0.5)
    # in singles via onAnyModifyDamage. Crits + Infiltrator bypass.
    def_is_side0 = def_offset < OFF_SIDE1
    def_screens_off = (
        (OFF_FIELD + F_SCREENS_0) if def_is_side0 else (OFF_FIELD + F_SCREENS_1)
    )
    def_screens = int(battle[def_screens_off])
    has_reflect = ((def_screens >> SCREEN_REFLECT_SHIFT) & SCREEN_MASK_3BIT) > 0
    has_light_screen = (
        (def_screens >> SCREEN_LIGHTSCREEN_SHIFT) & SCREEN_MASK_3BIT
    ) > 0
    has_aurora_veil = ((def_screens >> SCREEN_AURORAVEIL_SHIFT) & SCREEN_MASK_3BIT) > 0
    has_infiltrator = atk_ability == ABILITY_INFILTRATOR

    # Final item mods. Showdown source uses explicit numerators:
    #   Life Orb:    chainModify([5324, 4096])  ~= 1.2998 (NOT 5325)
    #   Expert Belt: chainModify([4915, 4096])  ~= 1.2002
    def _apply_hit_modify_damage_chain(
        hit_damage: int,
        *,
        live_def_hp: int,
        live_def_item: int,
        hit_crit: bool,
    ) -> int:
        _live_multiscale_active = (
            (def_ability == ABILITY_MULTISCALE)
            and (live_def_hp >= def_max_hp)
            and not has_mold_breaker
        )
        _live_shadow_shield_active = (def_ability == _ABILITY_SHADOW_SHIELD) and (
            live_def_hp >= def_max_hp
        )
        _hit_dmg_chain = _ChainAccum()
        _hit_dmg_chain.chain_if(
            _live_multiscale_active or _live_shadow_shield_active, 2048, 4096
        )  # 0.5x
        _hit_dmg_chain.chain_if(filter_mult != 1.0, 3072, 4096)  # 0.75x
        _hit_dmg_chain.chain_if(ice_scales_mult != 1.0, 2048, 4096)  # 0.5x
        # Fluffy can be 0.5 (contact only), 2.0 (fire only), or 1.0 (contact+fire);
        # apply each multiplicative factor as its own chain entry to preserve
        # Showdown's two-hook fold (it's two separate chainModify calls).
        if fluffy_mult == 0.5:
            _hit_dmg_chain.chain(2048, 4096)
        elif fluffy_mult == 2.0:
            _hit_dmg_chain.chain(8192, 4096)
        # punk_rock_def_mult is 0.5x or 1.0x
        _hit_dmg_chain.chain_if(punk_rock_def_mult != 1.0, 2048, 4096)
        _hit_dmg_chain.chain_if(tinted_lens_mult != 1.0, 8192, 4096)  # 2x
        # Neuroforce — Showdown chainModify([5120, 4096]) (1.25x) on SE hits.
        # Pokepy used 1.25 mult which mapped to 5120. Confirmed in abilities.ts.
        _hit_dmg_chain.chain_if(neuroforce_mult != 1.0, 5120, 4096)
        # Sniper — onModifyDamage chainModify(1.5) on crit (data/abilities.ts:4278).
        if has_sniper and hit_crit:
            _hit_dmg_chain.chain(6144, 4096)
        if _resist_berry_reduces(live_def_item):
            _hit_dmg_chain.chain(2048, 4096)
        _screen_reduces_phys = (
            (has_reflect or has_aurora_veil)
            and is_physical
            and not hit_crit
            and not has_infiltrator
        )
        _screen_reduces_spec = (
            (has_light_screen or has_aurora_veil)
            and not is_physical
            and not hit_crit
            and not has_infiltrator
        )
        if (
            _screen_reduces_phys or _screen_reduces_spec
        ) and effect_type != EFFECT_SCREEN_BREAK:
            _hit_dmg_chain.chain(2048, 4096)
        if life_orb_mult != 1.0:
            _hit_dmg_chain.chain(5324, 4096)
        if expert_belt_mult != 1.0:
            _hit_dmg_chain.chain(4915, 4096)
        return int(_hit_dmg_chain.apply(hit_damage))

    damage = _apply_hit_modify_damage_chain(
        damage,
        live_def_hp=def_hp,
        live_def_item=def_item,
        hit_crit=is_crit,
    )

    def _rebuild_hit(
        actual_bp: float,
        hit_crit: bool,
        hit_rand_pct: float,
        live_def_hp: int,
        live_def_item: int,
        attacker_status_code: int | None = None,
    ) -> float:
        if type_mult == 0.0:
            return 0.0
        _rb_bp_chain = _ChainAccum()
        if atk_ability == ABILITY_TECHNICIAN and 0 < int(actual_bp) <= 60:
            _rb_bp_chain.chain(6144, 4096)
        if ate_type_change:
            _rb_bp_chain.chain(4915, 4096)
        if _solar_weak:
            _rb_bp_chain.chain(2048, 4096)
        if _expanding_force_boost:
            _rb_bp_chain.chain(6144, 4096)
        if _misty_explosion_boost:
            _rb_bp_chain.chain(6144, 4096)
        if move_id in (MOVE_COLLISION_COURSE, MOVE_ELECTRO_DRIFT) and type_mult > 1.0:
            _rb_bp_chain.chain(5461, 4096)
        for _m in (
            sheer_force_mult,
            tough_claws_mult,
            reckless_mult,
            analytic_mult,
            iron_fist_mult,
            strong_jaw_mult,
            mega_launcher_mult,
            punk_rock_atk_mult,
            supreme_mult,
            sharpness_mult,
            dark_aura_mult,
            fairy_aura_mult,
            sand_force_mult,
            dry_skin_fire_mult,
            terrain_mult,
            bp_item_mult,
            knock_off_mult,
        ):
            _bp_chain_add_to(_rb_bp_chain, _m)
        if _so_num != 4096:
            _rb_bp_chain.chain(_so_num, 4096)
        rebuilt_bp = _rb_bp_chain.apply(int(actual_bp))
        rb_b1 = math.floor(2 * atk_level / 5 + 2)
        rb_b2 = math.floor(rb_b1 * rebuilt_bp * atk_stat)
        rb_b3 = math.floor(rb_b2 / max(1, def_stat))
        rb_b4 = math.floor(rb_b3 / 50)
        rb = rb_b4 + 2
        rb = _show_modify(rb, weather_mult_dmg)
        if hit_crit:
            rb = int(rb * 1.5)
        rb = math.floor(math.floor(rb * hit_rand_pct) / 100)
        rb = _show_modify(rb, stab_mult)
        rb = math.floor(rb * type_mult)
        _burned_for_hit = (
            is_burned
            if attacker_status_code is None
            else attacker_status_code == STATUS_BURN
        )
        if _burned_for_hit and is_physical and not has_guts and not is_facade:
            rb = _show_modify(rb, 0.5)
        return float(
            _apply_hit_modify_damage_chain(
                rb,
                live_def_hp=live_def_hp,
                live_def_item=live_def_item,
                hit_crit=hit_crit,
            )
        )

    def _advance_multihit_target_state(
        remaining_hp: int,
        remaining_sub_hp: int,
        sash_ready: bool,
        sturdy_ready: bool,
        disguise_ready: bool,
        hit_damage: int,
    ) -> tuple[int, int, bool, bool, bool]:
        if hit_damage <= 0:
            return (
                remaining_hp,
                remaining_sub_hp,
                sash_ready,
                sturdy_ready,
                disguise_ready,
            )
        if disguise_ready:
            chip = max(1, int(battle[def_offset + 2]) // 8)
            remaining_hp = max(0, remaining_hp - chip)
            return remaining_hp, remaining_sub_hp, sash_ready, sturdy_ready, False
        if remaining_sub_hp > 0:
            remaining_sub_hp = max(0, remaining_sub_hp - hit_damage)
            return (
                remaining_hp,
                remaining_sub_hp,
                sash_ready,
                sturdy_ready,
                disguise_ready,
            )
        next_hp = max(0, remaining_hp - hit_damage)
        if next_hp == 0 and sash_ready:
            return 1, remaining_sub_hp, False, sturdy_ready, disguise_ready
        if next_hp == 0 and sturdy_ready:
            return 1, remaining_sub_hp, sash_ready, False, disguise_ready
        return next_hp, remaining_sub_hp, sash_ready, sturdy_ready, disguise_ready

    def _init_multihit_target_state() -> tuple[int, int, bool, bool, bool, int]:
        _live_hp = int(battle[def_offset + 1])
        _live_sub = int(
            battle[OFF_FIELD + (F_SUBSTITUTE_0 if def_side == 0 else F_SUBSTITUTE_1)]
        )
        _live_sash_ready = def_item == ITEM_FOCUS_SASH and _live_hp == int(
            battle[def_offset + 2]
        )
        _live_sturdy_ready = def_ability == ABILITY_STURDY and _live_hp == int(
            battle[def_offset + 2]
        )
        _live_disguise_ready = (
            not has_mold_breaker
            and (int(battle[def_offset + 15]) & 0x40) != 0
            and (
                def_ability == ABILITY_DISGUISE
                or (def_ability == ABILITY_ICE_FACE and is_physical)
            )
        )
        if atk_ability == ABILITY_INFILTRATOR or is_sound:
            _live_sub = 0
        return (
            _live_hp,
            _live_sub,
            _live_sash_ready,
            _live_sturdy_ready,
            _live_disguise_ready,
            int(def_item),
        )

    def _hit_reaches_pokemon(remaining_sub_hp: int, disguise_ready: bool) -> bool:
        return (not disguise_ready) and remaining_sub_hp <= 0

    _multihit_contact_shadow = None
    _multihit_contact_prng_consumed = False
    _multihit_contact_resolved_status_field = None
    _multihit_contact_apply_attract = False
    _multihit_contact_ext_off = OFF_FIELD + (
        F_EXTENDED_VOLATILE_0 if atk_side == 0 else F_EXTENDED_VOLATILE_1
    )
    _multihit_contact_status_before = int(battle[atk_offset + 12])
    _multihit_contact_ext_before = int(battle[_multihit_contact_ext_off]) & 0xFFFF

    def _maybe_apply_multihit_contact_status(
        hit_damage: int,
        remaining_sub_hp_before: int,
        disguise_ready_before: bool,
    ) -> int:
        nonlocal _multihit_contact_prng_consumed
        nonlocal _multihit_contact_resolved_status_field
        nonlocal _multihit_contact_apply_attract
        if _multihit_contact_shadow is None:
            return get_status(_multihit_contact_status_before)
        if hit_damage <= 0:
            return get_status(int(_multihit_contact_shadow[atk_offset + 12]))
        if not _hit_reaches_pokemon(remaining_sub_hp_before, disguise_ready_before):
            return get_status(int(_multihit_contact_shadow[atk_offset + 12]))
        from pokepy.effects.abilities import apply_contact_status_ability

        apply_contact_status_ability(
            _multihit_contact_shadow,
            move_id,
            atk_offset,
            def_offset,
            True,
            game_data,
            move_effects,
            gen5_prng,
        )
        _multihit_contact_prng_consumed = True
        _after_status = int(_multihit_contact_shadow[atk_offset + 12])
        _after_ext = int(_multihit_contact_shadow[_multihit_contact_ext_off]) & 0xFFFF
        if (
            _multihit_contact_resolved_status_field is None
            and _after_status != _multihit_contact_status_before
        ):
            _multihit_contact_resolved_status_field = _after_status
        if (
            not _multihit_contact_apply_attract
            and (_after_ext & EXT_VOL_ATTRACT) != 0
            and (_multihit_contact_ext_before & EXT_VOL_ATTRACT) == 0
        ):
            _multihit_contact_apply_attract = True
        return get_status(_after_status)

    def _maybe_consume_multihit_resist_berry(
        live_def_item: int,
        hit_damage: int,
        remaining_sub_hp: int,
        disguise_ready: bool,
    ) -> int:
        if hit_damage <= 0:
            return live_def_item
        if not _hit_reaches_pokemon(remaining_sub_hp, disguise_ready):
            return live_def_item
        if _resist_berry_reduces(live_def_item):
            return 0
        return live_def_item

    # Multi-hit per-hit damage summation. For a 2-5 hit move, Showdown
    # re-rolls the damage randomizer for each hit and sums the per-hit
    # damage. Pokepy originally multiplied the first hit's damage by
    # num_hits, which matched total damage on average but drifted vs
    # Showdown for any fixed seed. Here we re-run the rand_pct → stab →
    # type → burn → dmg_chain pipeline for each hit with its own roll
    # and accumulate. This matches Showdown byte-for-byte for multi-hit
    # moves with fixed BP (Pin Missile, Icicle Spear, etc.).
    if num_hits_pre > 1:
        if _defer_extra_multihit_rolls and move_makes_contact:
            _multihit_contact_shadow = battle.copy()
        from pokepy.effects import get_effective_speed as _strict_speed_multihit

        _total_damage = damage  # hit 1 already finalized above
        if _defer_extra_multihit_rolls:
            _multihit_updates_tied = int(
                _strict_speed_multihit(battle, atk_offset)
            ) == int(_strict_speed_multihit(battle, def_offset))
            (
                _def_hp_live,
                _def_sub_live,
                _def_sash_ready,
                _def_sturdy_ready,
                _def_disguise_ready,
                _def_item_live,
            ) = _init_multihit_target_state()
            _triple_multiaccuracy = (
                move_id in (MOVE_TRIPLE_AXEL, _MOVE_TRIPLE_KICK)
                and atk_ability != ABILITY_SKILL_LINK
                and atk_item != ITEM_LOADED_DICE
                and not (has_no_guard or has_lock_on)
            )
            _population_bomb_multiaccuracy = (
                move_id == MOVE_POPULATION_BOMB
                and atk_ability != ABILITY_SKILL_LINK
                and atk_item != ITEM_LOADED_DICE
                and not accuracy_bypassed
            )
            _atk_status_live = (
                get_status(int(_multihit_contact_shadow[atk_offset + 12]))
                if (_multihit_contact_shadow is not None)
                else get_status(int(battle[atk_offset + 12]))
            )
            _def_item_live = _maybe_consume_multihit_resist_berry(
                _def_item_live,
                damage,
                _def_sub_live,
                _def_disguise_ready,
            )
            _def_sub_before_hit = _def_sub_live
            _def_disguise_before_hit = _def_disguise_ready
            (
                _def_hp_live,
                _def_sub_live,
                _def_sash_ready,
                _def_sturdy_ready,
                _def_disguise_ready,
            ) = _advance_multihit_target_state(
                _def_hp_live,
                _def_sub_live,
                _def_sash_ready,
                _def_sturdy_ready,
                _def_disguise_ready,
                damage,
            )
            _atk_status_live = _maybe_apply_multihit_contact_status(
                damage,
                _def_sub_before_hit,
                _def_disguise_before_hit,
            )
            _actual_hits = 1
            if _multihit_updates_tied:
                gen5_prng.random(0, 2)
            while _actual_hits < num_hits_pre and _def_hp_live > 0:
                _next_hit = _actual_hits + 1
                if _triple_multiaccuracy:
                    _extra_acc_roll = gen5_prng.random(100)
                    if _extra_acc_roll >= effective_accuracy:
                        break
                if _population_bomb_multiaccuracy:
                    _extra_acc_roll = gen5_prng.random(100)
                    if _extra_acc_roll >= effective_accuracy:
                        break
                if is_always_crit:
                    _hit_crit = True
                else:
                    _hit_crit = gen5_prng.random(crit_denom) == 0
                    if def_ability in (4, 75) and not has_mold_breaker:
                        _hit_crit = False
                _rp = float(100 - gen5_prng.random(16))
                _actual_bp = (
                    float(bp * _next_hit)
                    if move_id
                    in (
                        MOVE_TRIPLE_AXEL,
                        _MOVE_TRIPLE_KICK,
                    )
                    else float(bp)
                )
                _d = _rebuild_hit(
                    _actual_bp,
                    _hit_crit,
                    _rp,
                    _def_hp_live,
                    _def_item_live,
                    attacker_status_code=_atk_status_live,
                )
                _total_damage += _d
                _actual_hits += 1
                _def_item_live = _maybe_consume_multihit_resist_berry(
                    _def_item_live,
                    _d,
                    _def_sub_live,
                    _def_disguise_ready,
                )
                _def_sub_before_hit = _def_sub_live
                _def_disguise_before_hit = _def_disguise_ready
                (
                    _def_hp_live,
                    _def_sub_live,
                    _def_sash_ready,
                    _def_sturdy_ready,
                    _def_disguise_ready,
                ) = _advance_multihit_target_state(
                    _def_hp_live,
                    _def_sub_live,
                    _def_sash_ready,
                    _def_sturdy_ready,
                    _def_disguise_ready,
                    _d,
                )
                _atk_status_live = _maybe_apply_multihit_contact_status(
                    _d,
                    _def_sub_before_hit,
                    _def_disguise_before_hit,
                )
                if _multihit_updates_tied:
                    gen5_prng.random(0, 2)
            num_hits_pre = _actual_hits
        else:
            # The first hit was already folded into `damage`. Iterate the
            # remaining (num_hits - 1) per-hit rolls, each with its own
            # crit flag and rand_pct.
            (
                _def_hp_live,
                _def_sub_live,
                _def_sash_ready,
                _def_sturdy_ready,
                _def_disguise_ready,
                _def_item_live,
            ) = _init_multihit_target_state()
            _def_item_live = _maybe_consume_multihit_resist_berry(
                _def_item_live,
                damage,
                _def_sub_live,
                _def_disguise_ready,
            )
            (
                _def_hp_live,
                _def_sub_live,
                _def_sash_ready,
                _def_sturdy_ready,
                _def_disguise_ready,
            ) = _advance_multihit_target_state(
                _def_hp_live,
                _def_sub_live,
                _def_sash_ready,
                _def_sturdy_ready,
                _def_disguise_ready,
                damage,
            )
            _actual_hits = 1
            for _hit_crit, _rp in _per_hit_rolls[1:]:
                _d = _rebuild_hit(
                    float(bp),
                    _hit_crit,
                    float(_rp),
                    _def_hp_live,
                    _def_item_live,
                )
                _total_damage += _d
                _actual_hits += 1
                _def_item_live = _maybe_consume_multihit_resist_berry(
                    _def_item_live,
                    _d,
                    _def_sub_live,
                    _def_disguise_ready,
                )
                (
                    _def_hp_live,
                    _def_sub_live,
                    _def_sash_ready,
                    _def_sturdy_ready,
                    _def_disguise_ready,
                ) = _advance_multihit_target_state(
                    _def_hp_live,
                    _def_sub_live,
                    _def_sash_ready,
                    _def_sturdy_ready,
                    _def_disguise_ready,
                    _d,
                )
                if _def_hp_live <= 0:
                    break
            num_hits_pre = _actual_hits
        damage = _total_damage

    if move_cat == CAT_STATUS:
        damage = 0

    # Special damage moves
    is_level_damage = move_id in (MOVE_SEISMIC_TOSS, MOVE_NIGHT_SHADE)
    is_super_fang = move_id in (MOVE_SUPER_FANG, MOVE_RUINATION)
    is_ohko = move_id in (
        MOVE_FISSURE,
        MOVE_GUILLOTINE,
        MOVE_HORN_DRILL,
        MOVE_SHEER_COLD,
    )
    is_endeavor = move_id == MOVE_ENDEAVOR

    if is_level_damage:
        damage = atk_level if type_mult != 0.0 else 0
    if is_super_fang:
        damage = max(1, def_hp // 2) if type_mult != 0.0 else 0
    if is_ohko:
        # Showdown OHKO rules (sim/battle-actions.ts:690-704 hitStepAccuracy):
        # 1. user.level < target.level → immune
        # 2. Sheer Cold (move.ohko === 'Ice') → Ice-type targets are immune
        # 3. Sheer Cold from non-Ice user → accuracy 20 (vs 30), gen >= 7
        # Pokepy enforces #2 here. #1 is moot in OU (level 100 vs 100).
        # #3 (accuracy) is already-baked into the `hits` computation above, so
        # we can't retroactively change it; leave as-is (Sheer Cold uses 30
        # even from non-Ice users, a minor over-accuracy).
        def_is_ice_ohko = (def_type1 == TYPE_ICE) or (def_type2 == TYPE_ICE)
        sheer_cold_immune = (move_id == MOVE_SHEER_COLD) and def_is_ice_ohko
        if type_mult != 0.0 and not sheer_cold_immune:
            damage = def_max_hp
        else:
            damage = 0
    if is_endeavor:
        damage = max(0, def_hp - atk_hp) if type_mult != 0.0 else 0

    is_special_damage_move = is_level_damage or is_super_fang or is_ohko or is_endeavor
    if bp == 0 and not is_special_damage_move:
        damage = 0

    if move_id == MOVE_DREAM_EATER:
        target_status = get_status(int(battle[def_offset + 12]))
        if target_status != STATUS_SLEEP:
            damage = 0

    # Variable-power overrides — rebuild pipeline.
    # Mirrors the main pipeline (lines ~1042-1090) so variable-BP moves
    # (Gyro Ball, Body Press, Foul Play, Hard Press, etc.) get the same
    # Showdown chain-modify rounding instead of accumulated float-floor.
    stored_bp_safe = max(1.0, float(game_data.move_base_power[move_id]))

    def _rebuild(actual_bp: float, override_atk_stat: int = -1) -> float:
        if type_mult == 0.0:
            return 0.0
        # BP chain — single accumulator, applied once. Mirrors the main BP
        # chain. Atk-stat modifiers have ALREADY been applied to atk_stat
        # upstream — do NOT re-apply them here.
        _rb_bp_chain = _ChainAccum()
        # Technician check uses the rebuilt BP not the stored BP, since
        # variable-BP moves can become <=60 only after the callback.
        # Showdown abilities.ts:technician checks `basePowerAfterMultiplier`
        # at priority 30 (highest, so accumulator==1.0 → just check actual_bp).
        if atk_ability == ABILITY_TECHNICIAN and 0 < int(actual_bp) <= 60:
            _rb_bp_chain.chain(6144, 4096)
        if ate_type_change:
            _rb_bp_chain.chain(4915, 4096)
        if _solar_weak:
            _rb_bp_chain.chain(2048, 4096)
        if _expanding_force_boost:
            _rb_bp_chain.chain(6144, 4096)
        if _misty_explosion_boost:
            _rb_bp_chain.chain(6144, 4096)
        if move_id in (MOVE_COLLISION_COURSE, MOVE_ELECTRO_DRIFT) and type_mult > 1.0:
            _rb_bp_chain.chain(5461, 4096)
        for _m in (
            sheer_force_mult,
            tough_claws_mult,
            reckless_mult,
            analytic_mult,
            iron_fist_mult,
            strong_jaw_mult,
            mega_launcher_mult,
            punk_rock_atk_mult,
            supreme_mult,
            sharpness_mult,
            dark_aura_mult,
            fairy_aura_mult,
            sand_force_mult,
            dry_skin_fire_mult,
            terrain_mult,
            bp_item_mult,
            knock_off_mult,
        ):
            _bp_chain_add_to(_rb_bp_chain, _m)
        if _so_num != 4096:
            _rb_bp_chain.chain(_so_num, 4096)
        rebuilt_bp = _rb_bp_chain.apply(int(actual_bp))
        rb_b1 = math.floor(2 * atk_level / 5 + 2)
        _used_atk = override_atk_stat if override_atk_stat >= 0 else atk_stat
        rb_b2 = math.floor(rb_b1 * rebuilt_bp * _used_atk)
        rb_b3 = math.floor(rb_b2 / max(1, def_stat))
        rb_b4 = math.floor(rb_b3 / 50)
        rb = rb_b4 + 2
        rb = _show_modify(rb, weather_mult_dmg)
        # Crit uses plain trunc, not modify() — matches sim/battle-actions.ts:1748.
        if crit_mult != 1.0:
            rb = int(rb * crit_mult)
        rb = math.floor(math.floor(rb * rand_pct) / 100)
        rb = _show_modify(rb, stab_mult)
        rb = math.floor(rb * type_mult)
        if is_burned and is_physical and not has_guts and not is_facade:
            rb = _show_modify(rb, 0.5)
        rb = _apply_hit_modify_damage_chain(
            rb,
            live_def_hp=def_hp,
            live_def_item=def_item,
            hit_crit=is_crit,
        )
        return float(rb)

    # Use the strict chainModify-correct speed (matches Showdown's
    # `target.getStat('spe')` exactly: 4096-base chain, 10000 clamp,
    # paralysis finalModify) so Gyro Ball / Electro Ball BP scaling
    # uses the same number Showdown does. The legacy float
    # `_get_effective_speed` left here only for backward compat with
    # older callers (currently unused in the BP path).
    from pokepy.effects import get_effective_speed as _strict_speed

    atk_speed_base = float(_strict_speed(battle, atk_offset))
    def_speed_base = float(_strict_speed(battle, def_offset))

    if move_id == MOVE_GYRO_BALL:
        # Showdown moves.ts:8338-8344 — `floor(25 * tgt_spe / atk_spe) + 1`,
        # then min(150). Pokepy used to drop the +1 and apply min before adding,
        # consistently under-rolling Gyro Ball BP by 1.
        if atk_speed_base <= 0:
            gyro_bp = 1.0
        else:
            gyro_bp = float(
                min(150, math.floor(25.0 * def_speed_base / atk_speed_base) + 1)
            )
        damage = _rebuild(gyro_bp)

    if move_id == MOVE_ELECTROBALL:
        # Showdown moves.ts:4759-4765 — stepwise BP table indexed by
        # `floor(atk_spe / tgt_spe)`: [40, 60, 80, 120, 150]. Pokepy used a
        # continuous `min(150, 40 * ratio)` formula which is wrong at every
        # ratio except 1.
        if def_speed_base <= 0:
            ratio = 4
        else:
            ratio = int(atk_speed_base // def_speed_base)
        if ratio < 0:
            ratio = 0
        if ratio > 4:
            ratio = 4
        eball_bp = float([40, 60, 80, 120, 150][ratio])
        damage = _rebuild(eball_bp)

    if move_id in (MOVE_LOW_KICK, MOVE_GRASS_KNOT):
        def_species_clip = max(
            0,
            min(
                int(game_data.species_weight.shape[0]) - 1, int(battle[def_offset + 0])
            ),
        )
        def_weight = float(game_data.species_weight[def_species_clip])
        if def_weight < 10.0:
            lk_bp = 20.0
        elif def_weight < 25.0:
            lk_bp = 40.0
        elif def_weight < 50.0:
            lk_bp = 60.0
        elif def_weight < 100.0:
            lk_bp = 80.0
        elif def_weight < 200.0:
            lk_bp = 100.0
        else:
            lk_bp = 120.0
        damage = _rebuild(lk_bp)

    if move_id in (MOVE_HEAVY_SLAM, MOVE_HEAT_CRASH):
        atk_species_clip = max(
            0,
            min(
                int(game_data.species_weight.shape[0]) - 1, int(battle[atk_offset + 0])
            ),
        )
        def_species_clip = max(
            0,
            min(
                int(game_data.species_weight.shape[0]) - 1, int(battle[def_offset + 0])
            ),
        )
        atk_weight = float(game_data.species_weight[atk_species_clip])
        def_weight = float(game_data.species_weight[def_species_clip])
        weight_ratio = atk_weight / max(1.0, def_weight)
        if weight_ratio >= 5.0:
            hs_bp = 120.0
        elif weight_ratio >= 4.0:
            hs_bp = 100.0
        elif weight_ratio >= 3.0:
            hs_bp = 80.0
        elif weight_ratio >= 2.0:
            hs_bp = 60.0
        else:
            hs_bp = 40.0
        damage = _rebuild(hs_bp)

    if is_weather_ball:
        wb_bp = 100.0 if weather != WEATHER_NONE else 50.0
        damage = _rebuild(wb_bp)

    if is_terrain_pulse:
        tp_bp = 100.0 if terrain != TERRAIN_NONE else 50.0
        damage = _rebuild(tp_bp)

    if move_id in (MOVE_WATER_SPOUT, MOVE_ERUPTION, MOVE_DRAGON_ENERGY):
        # Showdown basePowerCallback: floor(150 * hp / maxhp), min 1
        atk_max_hp_f = float(battle[atk_offset + 2])
        hp_scaled_bp = max(
            1.0, math.floor(150.0 * float(atk_hp) / max(1.0, atk_max_hp_f))
        )
        damage = _rebuild(hp_scaled_bp)

    # Hard Press: BP scales with target's remaining HP fraction (max 100).
    # Showdown: max(1, floor(100 * hp / maxhp)) using chainModify-style rounding.
    MOVE_HARD_PRESS = 912
    if move_id == MOVE_HARD_PRESS:
        def_max_hp_hp = float(battle[def_offset + 2])
        def_hp_hp = float(def_hp)
        hp_press_bp = max(1.0, math.floor(100.0 * def_hp_hp / max(1.0, def_max_hp_hp)))
        damage = _rebuild(hp_press_bp)

    # Stomping Tantrum / Temper Flare: BP doubles if the user's previous
    # move failed. Showdown checks `pokemon.moveLastTurnResult === false`
    # for both moves. Pokepy tracks this via pokemon flag bit 0x02 set in
    # battle_gen9.py after a failed move.
    MOVE_STOMPING_TANTRUM = 707
    MOVE_TEMPER_FLARE = 915
    if move_id in (MOVE_STOMPING_TANTRUM, MOVE_TEMPER_FLARE):
        atk_flags_st = int(battle[atk_offset + 15])
        if (atk_flags_st & 0x02) != 0:
            damage = damage * 2

    if move_id == MOVE_BODY_PRESS:
        # Showdown: overrideOffensiveStat: 'def'. Recompute damage from
        # scratch with the user's def stat in place of atk. Defender Unaware
        # also zeroes the attacker's def boost (data/abilities.ts:unaware
        # onAnyModifyBoost clears boosts['def'] when the Unaware user is the
        # active target). Sign-agnostic.
        # After calculateStat runs with the override, Showdown still runs
        # runEvent('ModifyAtk') on the resulting stat (category==Physical,
        # so attackStat is reset to 'atk' — see sim/battle-actions.ts:1702).
        # This means Choice Band / Huge Power / etc. DO apply to Body Press's
        # Def-as-attack stat. Re-apply the accumulated atk chain + Hustle
        # before passing the override to _rebuild.
        atk_def_base = int(battle[atk_offset + 8])
        atk_def_boost = extract_boost(atk_boosts, 4)
        if def_has_unaware:
            atk_def_boost = 0
        atk_def_stat = int(atk_def_base * float(get_boost_multiplier(atk_def_boost)))
        if _hustle_active:
            atk_def_stat = _show_modify(atk_def_stat, 1.5)
        atk_def_stat = _atk_chain.apply(atk_def_stat)
        damage = _rebuild(stored_bp_safe, override_atk_stat=atk_def_stat)

    if move_id == MOVE_FOUL_PLAY:
        # Showdown: overrideOffensivePokemon: 'target'. Recompute with the
        # target's atk stat in place of user's. Note: unlike Body Press, Foul
        # Play swaps the SOURCE of the attacking pokemon entirely, so ModifyAtk
        # hooks run with the TARGET as attacker — which means the target's
        # Choice Band etc. would apply, NOT the user's. pokepy approximates
        # by just using the target's raw atk stat with boosts (no chain mods).
        # This is inaccurate vs Showdown when the target holds Choice items,
        # but correct for the common case.
        def_atk_base = int(battle[def_offset + 7])
        def_atk_boost = extract_boost(def_boosts, 0)
        def_atk_stat = int(def_atk_base * float(get_boost_multiplier(def_atk_boost)))
        damage = _rebuild(stored_bp_safe, override_atk_stat=def_atk_stat)

    # Multi-hit — the random(20) roll was hoisted up near the crit roll to
    # match Showdown's PRNG frame order. We reuse num_hits_pre here so the
    # damage multiplier matches what we burned frames for above. Population
    # Bomb keeps its own branch (multihitType:'populationbomb') because its
    # hit count depends on per-hit accuracy rolls.
    if move_id == MOVE_BEAT_UP:
        hits_min = len(beat_up_bps or [bp])
        hits_max = hits_min
        is_multi_hit = hits_max > 1
    else:
        hits_min = int(move_effects.hits_min[move_id])
        hits_max = int(move_effects.hits_max[move_id])
        is_multi_hit = hits_max > 1
    has_skill_link = atk_ability == ABILITY_SKILL_LINK
    has_loaded_dice = atk_item == ITEM_LOADED_DICE
    num_hits = num_hits_pre if is_multi_hit else 1

    _skill_link_removes_multiaccuracy = has_skill_link

    if move_id == MOVE_POPULATION_BOMB:
        # Population Bomb's extra-hit accuracy and crit/damage rolls are
        # deferred into the strict-parity multihit loop above so they stay
        # interleaved exactly like Showdown's hitStepMoveHitLoop.
        num_hits = num_hits_pre

    # `num_hits_pre` is the selected count before resolution, but the strict
    # per-hit loop above rewrites it to the actual landed count when a target
    # faints or a multiaccuracy move misses early. Do not clamp that back up to
    # hits_min, or downstream per-hit hooks like Rough Skin over-fire.
    num_hits = max(1, min(hits_max if is_multi_hit else 1, num_hits))

    # Triple Axel (813) and Triple Kick (167): BP scales 20/40/60 per
    # hit. Total damage = damage_per_hit * sum(1..num_hits)
    # = damage_per_hit * num_hits * (num_hits+1) / 2.
    # Showdown moves.ts:triplekick basePowerCallback `10 * move.hit`.
    # Both moves have multiaccuracy: true, meaning each hit after the
    # first re-rolls accuracy and STOPS on the first miss (battle-actions.ts
    # hitStepMoveHitLoop:905-933). Pokepy used to always apply all 3 hits,
    # overestimating damage by ~10%. Honor per-hit acc rolls here.
    if move_id == MOVE_BEAT_UP:
        beat_up_bps = beat_up_bps or [bp]
        damage = 0.0
        (
            _def_hp_live,
            _def_sub_live,
            _def_sash_ready,
            _def_sturdy_ready,
            _def_disguise_ready,
            _def_item_live,
        ) = _init_multihit_target_state()
        _actual_hits = 0
        for _bp_hit, (_hit_crit, _rp) in zip(beat_up_bps, _per_hit_rolls):
            _d = _rebuild_hit(
                float(_bp_hit),
                _hit_crit,
                _rp,
                _def_hp_live,
                _def_item_live,
            )
            damage += _d
            _actual_hits += 1
            _def_item_live = _maybe_consume_multihit_resist_berry(
                _def_item_live,
                _d,
                _def_sub_live,
                _def_disguise_ready,
            )
            (
                _def_hp_live,
                _def_sub_live,
                _def_sash_ready,
                _def_sturdy_ready,
                _def_disguise_ready,
            ) = _advance_multihit_target_state(
                _def_hp_live,
                _def_sub_live,
                _def_sash_ready,
                _def_sturdy_ready,
                _def_disguise_ready,
                _d,
            )
            if _def_hp_live <= 0:
                break
        num_hits = _actual_hits
    elif (
        move_id == MOVE_TRIPLE_AXEL or move_id == _MOVE_TRIPLE_KICK
    ) and not _defer_extra_multihit_rolls:
        # Legacy fallback for non-deferred callers. The normal strict-parity
        # path now handles Triple Axel/Kick inside the deferred per-hit loop
        # above so later-hit accuracy and early KOs stop PRNG consumption
        # exactly where Showdown's hitStepMoveHitLoop stops.
        if (
            hits
            and num_hits > 1
            and not _skill_link_removes_multiaccuracy
            and not (has_no_guard or has_lock_on)
        ):
            landed = 1  # first hit already rolled via `hits`
            for _hit_idx in range(2, num_hits + 1):
                _extra_roll = gen5_prng.random(100)
                if _extra_roll < effective_accuracy:
                    landed += 1
                else:
                    break
            # Collapse damage to landed-hit-fraction then apply triangular
            # sum. Approximate: rescale the per-hit-summed damage by
            # (landed / num_hits) * (landed+1) / (num_hits+1) * (num_hits+1)
            # which simplifies to (landed*(landed+1)) / num_hits. The
            # standard-multi-hit branch already summed `num_hits` linear
            # hits into `damage`, so dividing by num_hits gives the
            # average per-hit damage, then multiplying by landed*(landed+1)/2
            # gives the triangular-sum total for the landed hits.
            if num_hits > 0:
                _avg_per_hit = damage / num_hits
            else:
                _avg_per_hit = damage
            num_hits = landed
            damage = _avg_per_hit * num_hits * (num_hits + 1) / 2.0
        else:
            # No miss abort: triangular sum over all num_hits. The loop
            # above already summed num_hits equal hits into `damage`, so
            # divide and re-multiply by the triangular factor.
            if num_hits > 0:
                _avg_per_hit = damage / num_hits
            else:
                _avg_per_hit = damage
            damage = _avg_per_hit * num_hits * (num_hits + 1) / 2.0
    # Standard multi-hit: already summed per-hit above inside the deferred or
    # pre-rolled rand_pct loop.
    elif num_hits_pre <= 1:
        # Single-hit move: no change.
        pass

    if out_meta is not None:
        # Expose final hit count to the engine. Triple Kick / Triple Axel
        # may have been reduced by per-hit accuracy rolls above; record the
        # post-rollback value so the engine doesn't over-fire Rocky Helmet.
        out_meta["num_hits"] = int(num_hits)
        if _multihit_contact_prng_consumed:
            out_meta["contact_status_consumed"] = True
            out_meta["contact_status_packed"] = _multihit_contact_resolved_status_field
            out_meta["contact_status_apply_attract"] = _multihit_contact_apply_attract

    if not hits:
        damage = 0

    # Glaive Rush: if the defender has the glaive_rush flag set on itself,
    # incoming damage is doubled (Showdown data/moves.ts:glaiverush
    # `onSourceModifyDamage: chainModify(2)`). Already folded into the
    # `hits` computation above.
    if def_has_glaive_rush:
        damage = damage * 2

    # Semi-invulnerable — Showdown's onInvulnerability table by charging move.
    # Source: data/moves.ts fly.condition.onInvulnerability (line 1786),
    # bounce.condition.onInvulnerability (line 3739 area), dig (line 6128),
    # dive.condition.onInvulnerability (line 3914). Phantom Force / Shadow
    # Force have no exception list — nothing hits them.
    #   fly/bounce/skydrop: gust, twister, skyuppercut, thunder, hurricane,
    #                       smackdown, thousandarrows
    #   dig:   earthquake, magnitude
    #   dive:  surf, whirlpool
    target_semi_invul = (
        int(battle[_side_actions_offset(def_side)]) & ACTIVE_MOVE_ACTIONS_SEMI_INVUL
    ) != 0
    # No Guard / Lock On also bypass semi-invulnerability.
    # Showdown: abilities.ts noguard onInvulnerability = false,
    # moves.ts lockon/mindreader set volatiles['lockon'] which forces
    # onInvulnerability false on the target.
    if target_semi_invul and (has_no_guard or has_lock_on):
        target_semi_invul = False
    if target_semi_invul:
        from pokepy.core.constants import M_CHARGING_0, M_CHARGING_1
        from pokepy.core.constants import MOVE_DIG, MOVE_DIVE, MOVE_FLY, MOVE_BOUNCE

        _MOVE_GUST = 16
        _MOVE_TWISTER = 239
        _MOVE_SKY_UPPERCUT = 327
        _MOVE_THUNDER = 87
        _MOVE_HURRICANE = 542
        _MOVE_SMACK_DOWN = 479
        _MOVE_THOUSAND_ARROWS = 614
        _MOVE_MAGNITUDE = 222
        _MOVE_WHIRLPOOL = 250
        charge_meta_off = OFF_META + (M_CHARGING_0 if def_side == 0 else M_CHARGING_1)
        charging_move = int(battle[charge_meta_off])
        _FLY_LIKE = (MOVE_FLY, MOVE_BOUNCE)
        _AIR_HIT = (
            _MOVE_GUST,
            _MOVE_TWISTER,
            _MOVE_SKY_UPPERCUT,
            _MOVE_THUNDER,
            _MOVE_HURRICANE,
            _MOVE_SMACK_DOWN,
            _MOVE_THOUSAND_ARROWS,
        )
        if charging_move in _FLY_LIKE:
            can_hit = move_id in _AIR_HIT
        elif charging_move == MOVE_DIG:
            can_hit = move_id in (MOVE_EARTHQUAKE, _MOVE_MAGNITUDE)
        elif charging_move == MOVE_DIVE:
            can_hit = move_id in (MOVE_SURF, _MOVE_WHIRLPOOL)
        else:
            # Phantom Force / Shadow Force / unknown — nothing hits.
            can_hit = False
        if not can_hit:
            damage = 0

    return int(damage)
