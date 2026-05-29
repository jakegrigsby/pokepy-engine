"""Ability-triggered effects.

Real ports of:

All functions mutate `battle: np.ndarray` in place. Stateful PRNG via `Gen5PRNG`.
"""

from __future__ import annotations

from pokepy.effects._common import np, MultiFormatState, Gen5PRNG
from pokepy.effects.ability_suppression import effective_ability
from pokepy.effects.misc import is_take_item_blocked_by_item_rule
from pokepy.effects.status_apply import _try_apply_status
from pokepy.core.bitpack import (
    apply_boost_to_packed,
    extract_boost,
    get_status,
    get_status_turns,
    set_status,
)
from pokepy.core.constants import (
    OFF_SIDE1,
    OFF_FIELD,
    OFF_META,
    OFF_MOVES,
    ACTIVE_MOVE_ACTIONS_COUNT_MASK,
    F_WEATHER,
    F_TERRAIN,
    F_SUBSTITUTE_0,
    F_SUBSTITUTE_1,
    M_WEATHER_TURNS,
    M_TERRAIN_TURNS,
    STATUS_NONE,
    STATUS_BURN,
    STATUS_PARALYSIS,
    STATUS_POISON,
    STATUS_SLEEP,
    WEATHER_NONE,
    WEATHER_SUN,
    WEATHER_RAIN,
    WEATHER_SAND,
    WEATHER_SNOW,
    WEATHER_PRIMORDIAL_SEA,
    WEATHER_DESOLATE_LAND,
    WEATHER_DELTA_STREAM,
    TERRAIN_ELECTRIC,
    TERRAIN_GRASSY,
    TERRAIN_PSYCHIC,
    TERRAIN_MISTY,
    TYPE_ELECTRIC,
    TYPE_WATER,
    TYPE_GRASS,
    TYPE_FIRE,
    TYPE_GROUND,
    TYPE_FLYING,
    TYPE_POISON,
    TYPE_STEEL,
    FLAG_CONTACT,
    CAT_STATUS,
    EFFECT_SWITCH,
    ABILITY_WATER_VEIL,
    ABILITY_WATER_BUBBLE,
    ABILITY_THERMAL_EXCHANGE,
    ABILITY_SPEED_BOOST,
    ABILITY_SHED_SKIN,
    ABILITY_HYDRATION,
    ABILITY_INTIMIDATE,
    ABILITY_CLEAR_BODY,
    ABILITY_WHITE_SMOKE,
    ABILITY_FULL_METAL_BODY,
    ABILITY_INNER_FOCUS,
    ABILITY_NEUTRALIZING_GAS,
    ABILITY_OWN_TEMPO,
    ABILITY_OBLIVIOUS,
    ABILITY_SCRAPPY,
    ABILITY_CONTRARY,
    ABILITY_SIMPLE,
    ABILITY_DEFIANT,
    ABILITY_COMPETITIVE,
    ABILITY_TRACE,
    ABILITY_DROUGHT,
    ABILITY_DRIZZLE,
    ABILITY_SAND_STREAM,
    ABILITY_SNOW_WARNING,
    ABILITY_PRIMORDIAL_SEA,
    ABILITY_DESOLATE_LAND,
    ABILITY_DELTA_STREAM,
    ABILITY_ORICHALCUM_PULSE,
    ABILITY_HADRON_ENGINE,
    ABILITY_ELECTRIC_SURGE,
    ABILITY_GRASSY_SURGE,
    ABILITY_PSYCHIC_SURGE,
    ABILITY_MISTY_SURGE,
    ABILITY_DOWNLOAD,
    ABILITY_INTREPID_SWORD,
    ABILITY_DAUNTLESS_SHIELD,
    ABILITY_PROTOSYNTHESIS,
    ABILITY_QUARK_DRIVE,
    FLAG_BOOSTER_ENERGY_ACTIVE,
    ABILITY_REGENERATOR,
    ABILITY_NATURAL_CURE,
    ABILITY_VOLT_ABSORB,
    ABILITY_WATER_ABSORB,
    ABILITY_DRY_SKIN,
    ABILITY_SAP_SIPPER,
    ABILITY_STORM_DRAIN,
    ABILITY_LIGHTNING_ROD,
    ABILITY_MOTOR_DRIVE,
    ABILITY_FLASH_FIRE,
    ABILITY_MOXIE,
    ABILITY_BEAST_BOOST,
    ABILITY_SOUL_HEART,
    ABILITY_CHILLING_NEIGH,
    ABILITY_GRIM_NEIGH,
    ABILITY_FLAME_BODY,
    ABILITY_STATIC,
    ABILITY_POISON_POINT,
    ABILITY_EFFECT_SPORE,
    ABILITY_OVERCOAT,
    ABILITY_LONG_REACH,
    ITEM_DAMP_ROCK,
    ITEM_HEAT_ROCK,
    ITEM_ICY_ROCK,
    ITEM_SMOOTH_ROCK,
    ITEM_BOOSTER_ENERGY,
    ITEM_SAFETY_GOGGLES,
    ITEM_PROTECTIVE_PADS,
    ITEM_WEAKNESS_POLICY,
    ITEM_TERRAIN_EXTENDER,
    MOVE_ATTRACT,
    M_ACTIVE_MOVE_ACTIONS_0,
    M_ACTIVE_MOVE_ACTIONS_1,
)
from pokepy.data.type_charts import MODERN_TYPE_CHART

# Embody Aspect (Ogerpon forms) — not in core constants.py, but used by switch-in
ABILITY_EMBODY_ASPECT_TEAL = 301  # +1 Spe
ABILITY_EMBODY_ASPECT_WELLSPRING = 302  # +1 SpD
ABILITY_EMBODY_ASPECT_HEARTHFLAME = 303  # +1 Atk
ABILITY_EMBODY_ASPECT_CORNERSTONE = 304  # +1 Def

# Terrain Seeds use the packed item ids from pokepy's item table.
ITEM_ELECTRIC_SEED = 881
ITEM_PSYCHIC_SEED = 882
ITEM_MISTY_SEED = 883
ITEM_GRASSY_SEED = 884
ITEM_ABILITY_SHIELD = 1881
ABILITY_STICKY_HOLD = 60
ABILITY_MAGICIAN = 170
ABILITY_CUTE_CHARM = 56

_PARADOX_STAT_MASK = 0x6010
_PARADOX_STAT_ATK = 0x0010
_PARADOX_STAT_DEF = 0x2000
_PARADOX_STAT_SPA = 0x2010
_PARADOX_STAT_SPD = 0x4000
_PARADOX_STAT_SPE = 0x4010


def _apply_stage_to_stat(base_stat: int, boost: int) -> int:
    boost = max(-6, min(6, int(boost)))
    if boost >= 0:
        return (int(base_stat) * (2 + boost)) // 2
    return (int(base_stat) * 2) // (2 - boost)


def _encode_paradox_best_stat_flag(battle: np.ndarray, pokemon_offset: int) -> int:
    """Mirror Showdown's getBestStat(false, true) at paradox activation time."""

    poff = int(pokemon_offset)
    boosts13 = int(battle[poff + 13])
    boosts14 = int(battle[poff + 14])

    stats = [
        (
            _PARADOX_STAT_ATK,
            _apply_stage_to_stat(int(battle[poff + 7]), extract_boost(boosts13, 0)),
        ),
        (
            _PARADOX_STAT_DEF,
            _apply_stage_to_stat(int(battle[poff + 8]), extract_boost(boosts13, 4)),
        ),
        (
            _PARADOX_STAT_SPA,
            _apply_stage_to_stat(int(battle[poff + 9]), extract_boost(boosts13, 8)),
        ),
        (
            _PARADOX_STAT_SPD,
            _apply_stage_to_stat(int(battle[poff + 10]), extract_boost(boosts13, 12)),
        ),
        (
            _PARADOX_STAT_SPE,
            _apply_stage_to_stat(int(battle[poff + 11]), extract_boost(boosts14, 0)),
        ),
    ]

    best_flag, best_value = stats[0]
    for flag, value in stats[1:]:
        if value > best_value:
            best_flag, best_value = flag, value
    return best_flag


def apply_booster_energy_update(battle: np.ndarray, pokemon_offset: int) -> None:
    """Mirror Booster Energy's active-mon `onUpdate` consumption.

    Showdown's item hook consumes Booster Energy whenever an active
    Protosynthesis / Quark Drive holder is on the field without the matching
    sun / Electric Terrain support, then starts the ability volatile from the
    item. This can happen on switch-in, after a weather/terrain change, or
    after weather/terrain expires at end of turn.
    """

    poff = int(pokemon_offset)
    if int(battle[poff + 1]) <= 0:
        return

    ability = int(battle[poff + 5])
    if ability not in (ABILITY_PROTOSYNTHESIS, ABILITY_QUARK_DRIVE):
        return
    if int(battle[poff + 6]) != ITEM_BOOSTER_ENERGY:
        return

    cur_weather = int(battle[OFF_FIELD + F_WEATHER])
    cur_terrain = int(battle[OFF_FIELD + F_TERRAIN])
    field_activates = (
        ability == ABILITY_PROTOSYNTHESIS and cur_weather == WEATHER_SUN
    ) or (ability == ABILITY_QUARK_DRIVE and cur_terrain == TERRAIN_ELECTRIC)
    if field_activates:
        return

    cur_flags = int(battle[poff + 15]) & ~_PARADOX_STAT_MASK
    cur_flags |= FLAG_BOOSTER_ENERGY_ACTIVE
    cur_flags |= _encode_paradox_best_stat_flag(battle, poff)
    if cur_flags >= 0x8000:
        cur_flags -= 0x10000
    battle[poff + 6] = 0
    battle[poff + 15] = cur_flags


def apply_magician_from_move(
    battle: np.ndarray,
    move_id: int,
    user_offset: int,
    target_offset: int,
    hit: bool,
    damage_dealt: int,
    game_data,
    move_effects,
) -> None:
    """Mirror Showdown Magician (`onAfterMoveSecondarySelf`) in singles.

    Showdown runs Magician after the move finishes resolving, even if the
    hit KO'd the target. The source must still be item-less, the move must
    be a damaging non-pivot move that actually hit a foe, and the target's
    item must be removable. Doubles speed-sorted multi-target steals are not
    needed for the current parity set; this mirrors the singles case.
    """

    move_id = int(move_id)
    user_offset = int(user_offset)
    target_offset = int(target_offset)
    hit = bool(hit)
    damage_dealt = int(damage_dealt)

    if not hit or damage_dealt <= 0:
        return
    if int(battle[user_offset + 1]) <= 0:
        return
    if int(battle[user_offset + 5]) != ABILITY_MAGICIAN:
        return
    if int(battle[user_offset + 6]) != 0:
        return
    if int(game_data.move_category[move_id]) == CAT_STATUS:
        return
    if int(move_effects.effect_type[move_id]) == EFFECT_SWITCH:
        return

    target_item = int(battle[target_offset + 6])
    if target_item <= 0:
        return
    target_species = int(battle[target_offset + 0])

    if is_take_item_blocked_by_item_rule(target_item, target_species):
        return
    if int(battle[target_offset + 5]) == ABILITY_STICKY_HOLD:
        return

    battle[target_offset + 6] = 0
    battle[user_offset + 6] = target_item


def apply_speed_boost(battle: np.ndarray, pokemon_offset: int, game_data) -> None:
    """Port of _apply_speed_boost (line ~7737).

    Speed Boost ability: +1 Speed at end of turn, but not on the switch-in
    turn. Showdown gates this on `pokemon.activeTurns`.
    """
    poff = int(pokemon_offset)
    ability = int(battle[poff + 5])
    hp = int(battle[poff + 1])
    if ability != ABILITY_SPEED_BOOST or hp <= 0:
        return
    active_move_actions = (
        M_ACTIVE_MOVE_ACTIONS_0 if poff < OFF_SIDE1 else M_ACTIVE_MOVE_ACTIONS_1
    )
    if (
        int(battle[OFF_MOVES + active_move_actions]) & ACTIVE_MOVE_ACTIONS_COUNT_MASK
    ) <= 0:
        return
    boosts14 = int(battle[poff + 14])
    battle[poff + 14] = apply_boost_to_packed(boosts14, 0, 1)  # Spe at shift 0


def apply_shed_skin_hydration(
    battle: np.ndarray,
    pokemon_offset: int,
    game_data,
    gen5_prng: Gen5PRNG,
) -> None:
    """Port of _apply_shed_skin_hydration (line ~7784).

    Shed Skin: 33% chance to cure status at end of turn.
    Hydration: cure status at end of turn while raining.
    """
    poff = int(pokemon_offset)
    ability = int(battle[poff + 5])
    hp = int(battle[poff + 1])
    status_field = int(battle[poff + 12])
    status = get_status(status_field)
    weather = int(battle[OFF_FIELD + F_WEATHER])

    has_shed_skin = ability == ABILITY_SHED_SKIN
    has_hydration = ability == ABILITY_HYDRATION
    is_alive = hp > 0
    has_status = status != STATUS_NONE

    if not (is_alive and has_status and (has_shed_skin or has_hydration)):
        return

    # Hydration: cures in rain (raindance OR primordialsea) at the holder's
    # `effectiveWeather()`. Showdown data/abilities.ts:1872 hydration includes
    # both 'raindance' and 'primordialsea'. Holder's effectiveWeather() returns
    # '' under Air Lock / Cloud Nine on either active mon, OR when the holder
    # has Utility Umbrella (data/items.ts:7444). Pokepy previously only
    # accepted plain rain and ignored both suppressors.
    _ITEM_UTILITY_UMBRELLA_HY = 718
    holder_item = int(battle[poff + 6])
    has_umbrella = holder_item == _ITEM_UTILITY_UMBRELLA_HY
    # Check Air Lock / Cloud Nine on either active mon. We don't have
    # _weather_suppressed in this module, so check inline.
    from pokepy.core.constants import (
        OFF_META as _OM_HY,
        M_ACTIVE0 as _MA0_HY,
        M_ACTIVE1 as _MA1_HY,
        OFF_SIDE0 as _OS0_HY,
        OFF_SIDE1 as _OS1_HY,
        POKEMON_SIZE as _PS_HY,
        ABILITY_AIR_LOCK as _AL_HY,
        ABILITY_CLOUD_NINE as _CN_HY,
    )

    _a0_hy = int(battle[_OM_HY + _MA0_HY])
    _a1_hy = int(battle[_OM_HY + _MA1_HY])
    _ab0_hy = int(battle[_OS0_HY + _a0_hy * _PS_HY + 5])
    _ab1_hy = int(battle[_OS1_HY + _a1_hy * _PS_HY + 5])
    weather_suppressed = _ab0_hy in (_AL_HY, _CN_HY) or _ab1_hy in (_AL_HY, _CN_HY)
    # Utility Umbrella only suppresses sun/rain (not primordial), but
    # primordial weather is itself an "ability" weather and Utility Umbrella
    # also suppresses it per data/items.ts:7444.
    rain_active = weather in (WEATHER_RAIN, WEATHER_PRIMORDIAL_SEA)
    hydration_cures = (
        has_hydration and rain_active and not weather_suppressed and not has_umbrella
    )

    # Shed Skin: 33% chance — only roll if it could matter (only advance
    # PRNG when the result is observable).
    # Showdown uses `this.randomChance(33, 100)` — NOT `this.randomChance(1, 3)`
    # — so the true rate is exactly 33/100, not 33.333%. Source:
    # pokemon-showdown/data/abilities.ts shedskin.
    shed_cures = False
    if has_shed_skin:
        roll = gen5_prng.random(100)
        shed_cures = roll < 33

    if hydration_cures or shed_cures:
        battle[poff + 12] = set_status(STATUS_NONE, 0)


def apply_terrain_seed_item(battle: np.ndarray, pokemon_offset: int) -> None:
    """Consume a matching Terrain Seed and apply its stat boost."""
    poff = int(pokemon_offset)
    if int(battle[poff + 1]) <= 0:
        return

    terrain = int(battle[OFF_FIELD + F_TERRAIN])
    item = int(battle[poff + 6])
    boost_shift = -1
    if item == ITEM_ELECTRIC_SEED and terrain == TERRAIN_ELECTRIC:
        boost_shift = 4
    elif item == ITEM_GRASSY_SEED and terrain == TERRAIN_GRASSY:
        boost_shift = 4
    elif item == ITEM_PSYCHIC_SEED and terrain == TERRAIN_PSYCHIC:
        boost_shift = 12
    elif item == ITEM_MISTY_SEED and terrain == TERRAIN_MISTY:
        boost_shift = 12

    if boost_shift < 0:
        return

    battle[poff + 13] = apply_boost_to_packed(int(battle[poff + 13]), boost_shift, 1)
    battle[poff + 6] = 0


def _switch_in_ability_is_suppressed(
    battle: np.ndarray,
    switcher_offset: int,
    opponent_offset: int,
) -> bool:
    """Mirror Showdown's active Neutralizing Gas gate for ability Start events."""
    s_off = int(switcher_offset)
    o_off = int(opponent_offset)

    if int(battle[s_off + 6]) == ITEM_ABILITY_SHIELD:
        return False

    switcher_ability = int(battle[s_off + 5])
    if switcher_ability == ABILITY_NEUTRALIZING_GAS:
        return False

    opponent_alive = int(battle[o_off + 1]) > 0
    opponent_ability = int(battle[o_off + 5])
    return opponent_alive and opponent_ability == ABILITY_NEUTRALIZING_GAS


def _consume_field_change_each_event_frame(
    battle: np.ndarray,
    switcher_offset: int,
    opponent_offset: int,
    gen5_prng,
) -> None:
    """Mirror Showdown's WeatherChange / TerrainChange eachEvent speedSort."""
    if gen5_prng is None:
        return

    s_off = int(switcher_offset)
    o_off = int(opponent_offset)
    if int(battle[s_off + 1]) <= 0 or int(battle[o_off + 1]) <= 0:
        return

    from pokepy import effects as fx

    if fx.get_effective_speed(battle, s_off) == fx.get_effective_speed(battle, o_off):
        gen5_prng.random(0, 2)


def apply_switch_in_ability(
    battle: np.ndarray,
    switcher_offset: int,
    opponent_offset: int,
    did_switch: bool,
    gen5_prng=None,
) -> None:
    """Port of _apply_switch_in_ability (line ~7987).

    Handles all switch-in abilities: Intimidate (+ Defiant/Competitive
    counters), Embody Aspect, Trace, Drought/Drizzle/Sand Stream/Snow Warning/
    Primordial Sea/Desolate Land/Delta Stream/Orichalcum Pulse weather setters,
    Electric/Grassy/Psychic/Misty Surge + Hadron Engine terrain setters,
    Download, Intrepid Sword, Dauntless Shield, Protosynthesis/Quark Drive
    Booster Energy consumption.
    """
    if not did_switch:
        return

    s_off = int(switcher_offset)
    o_off = int(opponent_offset)

    # Showdown routes switch-in abilities through Pokemon#ignoringAbility, so a
    # live opposing Neutralizing Gas suppresses all regular Start handlers
    # unless the switcher holds Ability Shield or is the Neutralizing Gas user.
    if _switch_in_ability_is_suppressed(battle, s_off, o_off):
        return

    switcher_ability = int(battle[s_off + 5])
    opponent_ability = int(battle[o_off + 5])

    # ----- Intimidate ---------------------------------------------------
    has_intimidate = switcher_ability == ABILITY_INTIMIDATE
    # Hyper Cutter ONLY blocks Atk drops (not all stat drops). Showdown source:
    # data/abilities.ts hypercutter `onTryBoost: { atk: ... }`. Pokepy treats it
    # as a full Clear Body block since Intimidate only drops atk.
    from pokepy.core.constants import (
        ABILITY_HYPER_CUTTER,
        ABILITY_MIRROR_ARMOR,
        ABILITY_GUARD_DOG,
    )

    opp_has_clear_body = opponent_ability in (
        ABILITY_CLEAR_BODY,
        ABILITY_WHITE_SMOKE,
        ABILITY_FULL_METAL_BODY,
        ABILITY_INNER_FOCUS,
        ABILITY_OWN_TEMPO,
        ABILITY_OBLIVIOUS,
        ABILITY_SCRAPPY,
        ABILITY_HYPER_CUTTER,
    )
    opp_has_mirror_armor = opponent_ability == ABILITY_MIRROR_ARMOR
    opp_has_guard_dog = opponent_ability == ABILITY_GUARD_DOG
    opp_has_contrary = opponent_ability == ABILITY_CONTRARY
    opp_has_simple = opponent_ability == ABILITY_SIMPLE
    # Clear Amulet on the opponent blocks Intimidate's atk drop. Showdown
    # data/items.ts:1064-1084 clearamulet onTryBoost.
    _ITEM_CLEAR_AMULET_IN = 747
    opp_has_clear_amulet = int(battle[o_off + 6]) == _ITEM_CLEAR_AMULET_IN

    # Guard Dog: Intimidate's atk drop is converted into +1 atk boost.
    # Showdown data/abilities.ts guarddog: deletes boost.atk and does
    # this.boost({atk: 1}, target, target, null, false, true).
    if opp_has_guard_dog:
        intimidate_change = 1
    elif opp_has_clear_body or opp_has_clear_amulet:
        intimidate_change = 0
    elif opp_has_contrary:
        intimidate_change = 1
    elif opp_has_simple:
        intimidate_change = -2
    else:
        intimidate_change = -1

    # Substitute blocks Intimidate
    switcher_is_side0 = s_off < OFF_SIDE1
    opp_sub_offset = (
        OFF_FIELD + F_SUBSTITUTE_1 if switcher_is_side0 else OFF_FIELD + F_SUBSTITUTE_0
    )
    opp_has_sub_intim = int(battle[opp_sub_offset]) > 0

    # Mirror Armor (Corviknight): reflect Intimidate back to switcher.
    # Showdown data/abilities.ts:2588-2609 mirrorarmor onTryBoost: deletes
    # the boost on the target and calls `this.boost(negativeBoost, source,
    # target, null, true)` — the reflected boost goes through the normal
    # boost() pipeline on the SWITCHER, so the switcher's own onTryBoost
    # handlers (Clear Body, Hyper Cutter, Clear Amulet, Contrary, etc.) and
    # onAfterEachBoost handlers (Defiant, Competitive) ALL apply.
    # Mirror Armor itself is excluded from re-bouncing via the
    # `effect.name === 'Mirror Armor'` check.
    sw_item = int(battle[s_off + 6])
    sw_has_clear_body = switcher_ability in (
        ABILITY_CLEAR_BODY,
        ABILITY_WHITE_SMOKE,
        ABILITY_FULL_METAL_BODY,
        ABILITY_INNER_FOCUS,
        ABILITY_OWN_TEMPO,
        ABILITY_OBLIVIOUS,
        ABILITY_SCRAPPY,
        ABILITY_HYPER_CUTTER,
    )
    sw_has_clear_amulet = sw_item == _ITEM_CLEAR_AMULET_IN
    sw_has_contrary = switcher_ability == ABILITY_CONTRARY
    sw_has_simple = switcher_ability == ABILITY_SIMPLE
    sw_has_guard_dog = switcher_ability == ABILITY_GUARD_DOG

    if has_intimidate and opp_has_mirror_armor and not opp_has_sub_intim:
        # Reflected to switcher. Apply switcher's defenses to the -1 atk.
        if sw_has_guard_dog:
            reflect_change = 1
        elif sw_has_clear_body or sw_has_clear_amulet:
            reflect_change = 0
        elif sw_has_contrary:
            reflect_change = 1
        elif sw_has_simple:
            reflect_change = -2
        else:
            reflect_change = -1
        if reflect_change != 0:
            sw_boosts13_ma = int(battle[s_off + 13])
            battle[s_off + 13] = apply_boost_to_packed(
                sw_boosts13_ma, 0, reflect_change
            )
            # Defiant / Competitive on switcher when stat lowered.
            if reflect_change < 0:
                if switcher_ability == ABILITY_DEFIANT:
                    sw_b = int(battle[s_off + 13])
                    battle[s_off + 13] = apply_boost_to_packed(sw_b, 0, 2)
                elif switcher_ability == ABILITY_COMPETITIVE:
                    sw_b = int(battle[s_off + 13])
                    battle[s_off + 13] = apply_boost_to_packed(sw_b, 8, 2)
        intimidate_applied = False
    else:
        intimidate_applied = (
            has_intimidate and (intimidate_change != 0) and (not opp_has_sub_intim)
        )

    opp_boosts13 = int(battle[o_off + 13])
    if intimidate_applied:
        opp_boosts13 = apply_boost_to_packed(opp_boosts13, 0, intimidate_change)

    stat_was_lowered = intimidate_applied and (intimidate_change < 0)

    # Defiant: +2 Atk on stat lower
    if stat_was_lowered and opponent_ability == ABILITY_DEFIANT:
        opp_boosts13 = apply_boost_to_packed(opp_boosts13, 0, 2)

    # Competitive: +2 SpA on stat lower
    if stat_was_lowered and opponent_ability == ABILITY_COMPETITIVE:
        opp_boosts13 = apply_boost_to_packed(opp_boosts13, 8, 2)

    battle[o_off + 13] = opp_boosts13

    # Adrenaline Orb — Showdown items.ts:111. Triggers on any Intimidate
    # event hitting the holder, regardless of whether the atk drop actually
    # landed (Hyper Cutter / Clear Body / etc. set boost.atk to undefined,
    # which Showdown's `boost.atk === 0` check distinguishes from "already
    # at -6"). Pokepy approximates: trigger when Intimidate was attempted
    # (`has_intimidate`) and the holder is below +6 Spe AND was not Subbed
    # AND the atk drop wasn't capped to 0 (intimidate_change != 0). The
    # +1 Spe boost is applied via the item's `boosts: { spe: 1 }`.
    from pokepy.core.constants import ITEM_ADRENALINE_ORB

    if has_intimidate and not opp_has_sub_intim:
        opp_item_ao = int(battle[o_off + 6])
        if opp_item_ao == ITEM_ADRENALINE_ORB and int(battle[o_off + 1]) > 0:
            opp_boosts14_ao = int(battle[o_off + 14])
            spe_stage_ao = (opp_boosts14_ao >> 0) & 0xF
            # Showdown also blocks if `boost.atk === 0` (i.e. already at -6
            # in pre-bounded form). Atk slot at shift 0 of opp_boosts13.
            atk_stage_ao = (int(battle[o_off + 13]) >> 0) & 0xF
            atk_at_floor = atk_stage_ao == 0  # 0 nibble == -6 stage
            spe_at_max = spe_stage_ao == 12  # 12 nibble == +6 stage
            if (not spe_at_max) and (not atk_at_floor):
                opp_boosts14_ao = apply_boost_to_packed(opp_boosts14_ao, 0, 1)
                battle[o_off + 14] = opp_boosts14_ao
                battle[o_off + 6] = 0  # consume orb

    # ----- Embody Aspect (Ogerpon) --------------------------------------
    # Showdown: only fires when Ogerpon TERASTALLIZES into its mask form, and
    # only once per battle (latched on volatile.embodied). Pokepy proxy: only
    # fire if `tera_used` flag (bit 3) is set on this mon, AND once per battle
    # (use bit 8 = "once_per_battle ability triggered" as the latch).
    sw_flags_ea = int(battle[s_off + 15])
    is_terad_ea = (sw_flags_ea & 0x8) != 0
    already_embodied = (sw_flags_ea & 0x100) != 0
    is_embody = switcher_ability in (
        ABILITY_EMBODY_ASPECT_HEARTHFLAME,
        ABILITY_EMBODY_ASPECT_CORNERSTONE,
        ABILITY_EMBODY_ASPECT_WELLSPRING,
        ABILITY_EMBODY_ASPECT_TEAL,
    )
    if is_embody and is_terad_ea and not already_embodied:
        sw_boosts13_ea = int(battle[s_off + 13])
        sw_boosts14_ea = int(battle[s_off + 14])
        if switcher_ability == ABILITY_EMBODY_ASPECT_HEARTHFLAME:
            sw_boosts13_ea = apply_boost_to_packed(sw_boosts13_ea, 0, 1)  # +1 Atk
        if switcher_ability == ABILITY_EMBODY_ASPECT_CORNERSTONE:
            sw_boosts13_ea = apply_boost_to_packed(sw_boosts13_ea, 4, 1)  # +1 Def
        if switcher_ability == ABILITY_EMBODY_ASPECT_WELLSPRING:
            sw_boosts13_ea = apply_boost_to_packed(sw_boosts13_ea, 12, 1)  # +1 SpD
        if switcher_ability == ABILITY_EMBODY_ASPECT_TEAL:
            sw_boosts14_ea = apply_boost_to_packed(sw_boosts14_ea, 0, 1)  # +1 Spe
        battle[s_off + 13] = sw_boosts13_ea
        battle[s_off + 14] = sw_boosts14_ea
        # Latch
        battle[s_off + 15] = sw_flags_ea | 0x100

    # ----- Trace --------------------------------------------------------
    # Showdown data/abilities.ts trace: cannot trace any ability with the
    # `notrace: 1` flag. The notrace list (gen 9) includes: As One forms,
    # Battle Bond, Comatose, Commander, Disguise, Embody Aspect (all 4),
    # Flower Gift, Forecast, Gulp Missile, Hadron Engine, Hunger Switch,
    # Ice Face, Illusion, Imposter, Multitype, Neutralizing Gas, No Ability,
    # Orichalcum Pulse, Poison Puppeteer, Power Construct, Power of Alchemy,
    # Protosynthesis, Quark Drive, Receiver, RKS System, Schooling, Shields
    # Down, Stance Change, Tera Shift, Teraform Zero, Trace itself, Wonder
    # Guard, Zen Mode, Zero to Hero. Pokepy keeps the small set that are in
    # constants.py + a couple of common locals.
    _NOTRACE_ABILITIES = frozenset(
        (
            ABILITY_TRACE,
            # multitype, stance change, schooling, shields down, comatose, etc.
            121,  # multitype
            176,  # stancechange
            208,  # schooling
            197,  # shieldsdown
            213,  # comatose
            209,  # disguise
            248,  # iceface
            211,  # powerconstruct
            281,  # protosynthesis
            282,  # quarkdrive
            288,  # orichalcumpulse
            289,  # hadronengine
            256,  # neutralizinggas
            150,  # imposter
            25,  # wonderguard
            161,  # zenmode
            258,  # hungerswitch
            241,  # gulpmissile
            59,  # forecast (gen-specific id but listed)
            301,
            302,
            303,
            304,  # embody aspect (all four)
        )
    )
    has_trace = switcher_ability == ABILITY_TRACE
    opp_ability_trace = int(battle[o_off + 5])
    can_copy = (opp_ability_trace > 0) and (opp_ability_trace not in _NOTRACE_ABILITIES)
    # Ability Shield (item 746) on the switcher blocks the ability change.
    # Showdown data/items.ts:2-20 abilityshield onSetAbility returns null.
    _ITEM_ABILITY_SHIELD_TR = 746
    if int(battle[s_off + 6]) == _ITEM_ABILITY_SHIELD_TR:
        can_copy = False
    if has_trace and can_copy:
        # Showdown Trace samples from adjacent foes even in singles, so the
        # one-target case still consumes a hidden `random(1)` frame before the
        # copied ability is applied.
        if gen5_prng is not None:
            gen5_prng.random(1)
        battle[s_off + 5] = opp_ability_trace
        switcher_ability = opp_ability_trace  # subsequent checks see traced ability

    # ----- Weather setters ----------------------------------------------
    # Showdown: sim/field.ts:45-53 setWeather rejects a same-weather override
    # from abilities in gen > 5 (no reset to 5 turns if weather is already
    # that type). Primal abilities (Primordial Sea / Desolate Land / Delta
    # Stream) also block any non-primal override via onAnySetWeather
    # (data/abilities.ts:926/950/974).
    current_weather = int(battle[OFF_FIELD + F_WEATHER])
    has_drought = switcher_ability == ABILITY_DROUGHT
    has_drizzle = switcher_ability == ABILITY_DRIZZLE
    has_sand_stream = switcher_ability == ABILITY_SAND_STREAM
    has_snow_warning = switcher_ability == ABILITY_SNOW_WARNING
    has_primordial_sea = switcher_ability == ABILITY_PRIMORDIAL_SEA
    has_desolate_land = switcher_ability == ABILITY_DESOLATE_LAND
    has_delta_stream = switcher_ability == ABILITY_DELTA_STREAM
    has_orichalcum = switcher_ability == ABILITY_ORICHALCUM_PULSE

    # Determine the weather the new switch-in WANTS to set. If another
    # primal weather is up, a non-primal weather cannot override it.
    wanted_weather = 0
    if has_drought or has_orichalcum:
        wanted_weather = WEATHER_SUN
    elif has_drizzle:
        wanted_weather = WEATHER_RAIN
    elif has_sand_stream:
        wanted_weather = WEATHER_SAND
    elif has_snow_warning:
        wanted_weather = WEATHER_SNOW
    elif has_primordial_sea:
        wanted_weather = WEATHER_PRIMORDIAL_SEA
    elif has_desolate_land:
        wanted_weather = WEATHER_DESOLATE_LAND
    elif has_delta_stream:
        wanted_weather = WEATHER_DELTA_STREAM

    primal_weather = has_primordial_sea or has_desolate_land or has_delta_stream
    primal_weathers_set = (
        WEATHER_PRIMORDIAL_SEA,
        WEATHER_DESOLATE_LAND,
        WEATHER_DELTA_STREAM,
    )
    current_is_primal = current_weather in primal_weathers_set

    # If current weather is already primal, only another primal may replace
    # it (Showdown onAnySetWeather check). If same weather id, Showdown
    # returns false for ability-set, so skip entirely (no duration reset).
    ability_sets_weather = wanted_weather > 0
    blocked_by_primal = (
        ability_sets_weather
        and current_is_primal
        and (wanted_weather not in primal_weathers_set)
    )
    same_weather_id = ability_sets_weather and (wanted_weather == current_weather)

    weather_changed = (
        ability_sets_weather and not blocked_by_primal and not same_weather_id
    )
    if weather_changed:
        final_weather = wanted_weather
        battle[OFF_FIELD + F_WEATHER] = final_weather

        switcher_item_w = int(battle[s_off + 6])
        has_weather_rock = (
            (final_weather == WEATHER_RAIN and switcher_item_w == ITEM_DAMP_ROCK)
            or (final_weather == WEATHER_SUN and switcher_item_w == ITEM_HEAT_ROCK)
            or (final_weather == WEATHER_SNOW and switcher_item_w == ITEM_ICY_ROCK)
            or (final_weather == WEATHER_SAND and switcher_item_w == ITEM_SMOOTH_ROCK)
        )
        if primal_weather:
            new_weather_turns = 0  # permanent
        else:
            new_weather_turns = 8 if has_weather_rock else 5
        battle[OFF_META + M_WEATHER_TURNS] = new_weather_turns
        _consume_field_change_each_event_frame(
            battle,
            s_off,
            o_off,
            gen5_prng,
        )

    # ----- Terrain setters ----------------------------------------------
    current_terrain = int(battle[OFF_FIELD + F_TERRAIN])
    has_electric_surge = switcher_ability == ABILITY_ELECTRIC_SURGE
    has_grassy_surge = switcher_ability == ABILITY_GRASSY_SURGE
    has_psychic_surge = switcher_ability == ABILITY_PSYCHIC_SURGE
    has_misty_surge = switcher_ability == ABILITY_MISTY_SURGE
    has_hadron = switcher_ability == ABILITY_HADRON_ENGINE

    final_terrain = current_terrain
    if has_electric_surge:
        final_terrain = TERRAIN_ELECTRIC
    if has_grassy_surge:
        final_terrain = TERRAIN_GRASSY
    if has_psychic_surge:
        final_terrain = TERRAIN_PSYCHIC
    if has_misty_surge:
        final_terrain = TERRAIN_MISTY
    if has_hadron:
        final_terrain = TERRAIN_ELECTRIC

    ability_sets_terrain = (
        has_electric_surge
        or has_grassy_surge
        or has_psychic_surge
        or has_misty_surge
        or has_hadron
    )
    terrain_changed = ability_sets_terrain and final_terrain != current_terrain
    if terrain_changed:
        battle[OFF_FIELD + F_TERRAIN] = final_terrain
        switcher_item = int(battle[s_off + 6])
        terrain_turns = 8 if switcher_item == ITEM_TERRAIN_EXTENDER else 5
        battle[OFF_META + M_TERRAIN_TURNS] = terrain_turns
        _consume_field_change_each_event_frame(
            battle,
            s_off,
            o_off,
            gen5_prng,
        )

    # ----- Download -----------------------------------------------------
    has_download = switcher_ability == ABILITY_DOWNLOAD
    sw_boosts13 = int(battle[s_off + 13])
    if has_download:
        opp_def = int(battle[o_off + 8])
        opp_spd = int(battle[o_off + 10])
        if opp_def < opp_spd:
            sw_boosts13 = apply_boost_to_packed(sw_boosts13, 0, 1)  # +1 Atk
        else:
            sw_boosts13 = apply_boost_to_packed(sw_boosts13, 8, 1)  # +1 SpA

    # ----- Intrepid Sword / Dauntless Shield (once per battle) ----------
    sw_flags = int(battle[s_off + 15])
    one_time_used = (sw_flags & 0x100) != 0
    if switcher_ability == ABILITY_INTREPID_SWORD and not one_time_used:
        sw_boosts13 = apply_boost_to_packed(sw_boosts13, 0, 1)  # +1 Atk
        new_flags = sw_flags | 0x100
        if new_flags >= 0x8000:
            new_flags -= 0x10000
        battle[s_off + 15] = new_flags
        sw_flags = new_flags

    has_dauntless_shield = switcher_ability == ABILITY_DAUNTLESS_SHIELD
    dauntless_used = (sw_flags & 0x100) != 0
    should_dauntless = has_dauntless_shield and (not dauntless_used)
    if should_dauntless:
        sw_boosts13 = apply_boost_to_packed(sw_boosts13, 4, 1)  # +1 Def
        new_flags = sw_flags | 0x100
        if new_flags >= 0x8000:
            new_flags -= 0x10000
        battle[s_off + 15] = new_flags

    battle[s_off + 13] = sw_boosts13

    # ----- Imposter (Ditto) — transform into opponent ------------------
    # Showdown: pokemon.transformInto(target) copies species, types, ability,
    # stats (except HP), moves, boosts, and the transformed flag. Doesn't
    # copy item, level, HP, or tera type (sim/pokemon.ts:1238-1340).
    # Pokepy approximates by copying species/types/ability/stats and boost
    # stages from the opponent. Move slots aren't synced — that requires
    # touching team_moves[] which is outside this function's scope; this
    # remains a known gap. Tera type lives in the upper 4 bits of slot
    # +14, which we preserve so a transformed Ditto keeps its own tera.
    from pokepy.core.constants import ABILITY_IMPOSTER as _ABILITY_IMPOSTER

    if switcher_ability == _ABILITY_IMPOSTER:
        # Don't transform into a substituted opponent (Showdown blocks via
        # `pokemon.volatiles['substitute']` check at sim/pokemon.ts:1241).
        opp_has_sub_imp = int(battle[opp_sub_offset]) > 0
        # Showdown also blocks transforming into a tera'd Ogerpon /
        # Terapagos and a Stellar-tera Ditto. Pokepy doesn't track tera
        # type at this granularity for the species check; Ogerpon /
        # Terapagos are rare in OU so the gap is acceptable.
        if not opp_has_sub_imp:
            # Copy species, types, ability, base stats, boosts (but NOT HP,
            # NOT item, NOT tera, NOT moves). Slot 14's upper 4 bits hold
            # the tera type — preserve them on the switcher.
            for off in (0, 4, 5, 7, 8, 9, 10, 11, 13):
                battle[s_off + off] = int(battle[o_off + off])
            self_tera = int(battle[s_off + 14]) & -4096
            opp_b14 = int(battle[o_off + 14]) & 4095
            new_b14 = (opp_b14 | self_tera) & 0xFFFF
            if new_b14 >= 0x8000:
                new_b14 -= 0x10000
            battle[s_off + 14] = new_b14

    # ----- Booster Energy consumption (Protosynthesis / Quark Drive) ----
    has_paradox_ability = switcher_ability in (
        ABILITY_PROTOSYNTHESIS,
        ABILITY_QUARK_DRIVE,
    )
    switcher_item = int(battle[s_off + 6])
    cur_weather2 = int(battle[OFF_FIELD + F_WEATHER])
    cur_terrain2 = int(battle[OFF_FIELD + F_TERRAIN])
    proto_has_sun = (
        switcher_ability == ABILITY_PROTOSYNTHESIS and cur_weather2 == WEATHER_SUN
    )
    quark_has_eterrain = (
        switcher_ability == ABILITY_QUARK_DRIVE and cur_terrain2 == TERRAIN_ELECTRIC
    )
    field_activates = proto_has_sun or quark_has_eterrain
    should_consume_booster = (
        has_paradox_ability
        and switcher_item == ITEM_BOOSTER_ENERGY
        and not field_activates
    )
    if has_paradox_ability:
        cur_flags = int(battle[s_off + 15]) & ~_PARADOX_STAT_MASK
        if should_consume_booster:
            battle[s_off + 6] = 0
            # Mark the per-entry paradox boost as active. This must stay distinct
            # from the generic had-item bit so Knock Off on a non-Booster item
            # never reactivates Protosynthesis / Quark Drive.
            cur_flags |= FLAG_BOOSTER_ENERGY_ACTIVE
        if field_activates or should_consume_booster:
            cur_flags |= _encode_paradox_best_stat_flag(battle, s_off)
        if cur_flags >= 0x8000:
            cur_flags -= 0x10000
        battle[s_off + 15] = cur_flags

    apply_terrain_seed_item(battle, s_off)
    if ability_sets_terrain:
        apply_terrain_seed_item(battle, o_off)

    apply_booster_energy_update(battle, s_off)
    # The opponent only needs a reactive paradox update here if this switch-in
    # actually changed the live weather or terrain. Calling it unconditionally
    # on simultaneous lead entries can consume the opponent's Booster Energy
    # before their own switch-in ability resolves, then let the later call wipe
    # the encoded best-stat bits.
    if weather_changed or terrain_changed:
        apply_booster_energy_update(battle, o_off)


def apply_switch_in_ability_with_trace_reaction(
    battle: np.ndarray,
    switcher_offset: int,
    opponent_offset: int,
    did_switch: bool,
    gen5_prng=None,
) -> None:
    """Apply a unilateral switch-in and let an active foe react with Trace.

    In Showdown, Trace can fire when the opposing active changes even if the
    Trace holder itself did not switch in on that turn. For single-sided
    switch events, resolve the switcher's own on-entry ability first, then let
    the already-active foe copy the new entrant if it currently has Trace.
    """
    apply_switch_in_ability(
        battle,
        switcher_offset,
        opponent_offset,
        did_switch,
        gen5_prng=gen5_prng,
    )
    if not did_switch:
        return

    o_off = int(opponent_offset)
    if int(battle[o_off + 1]) <= 0:
        return
    if int(battle[o_off + 5]) != ABILITY_TRACE:
        return

    apply_switch_in_ability(
        battle,
        o_off,
        int(switcher_offset),
        True,
        gen5_prng=gen5_prng,
    )


def apply_regenerator_on_switch_out(
    battle: np.ndarray,
    pokemon_offset: int,
    did_switch: bool,
) -> None:
    """Port of _apply_regenerator_on_switch_out (line ~8309).

    Heal 33% max HP when switching out (rounded down, minimum 1).
    """
    if not did_switch:
        return
    poff = int(pokemon_offset)
    # Showdown routes switch-out handlers through the user's currently
    # effective ability state, so Neutralizing Gas suppression disables
    # Regenerator / Natural Cure unless Ability Shield keeps them active.
    ability = int(effective_ability(battle, poff))
    if ability != ABILITY_REGENERATOR:
        return
    current_hp = int(battle[poff + 1])
    max_hp = int(battle[poff + 2])
    if current_hp <= 0:
        return
    heal_amount = max(int(max_hp / 3), 1)
    new_hp = min(max_hp, current_hp + heal_amount)
    battle[poff + 1] = new_hp


def apply_natural_cure_on_switch_out(
    battle: np.ndarray,
    pokemon_offset: int,
    did_switch: bool,
) -> None:
    """Port of _apply_natural_cure_on_switch_out (line ~8340).

    Clear status when switching out.
    """
    if not did_switch:
        return
    poff = int(pokemon_offset)
    ability = int(effective_ability(battle, poff))
    if ability != ABILITY_NATURAL_CURE:
        return
    status_field = int(battle[poff + 12])
    status = get_status(status_field)
    if status == STATUS_NONE:
        return
    battle[poff + 12] = 0


def apply_absorb_ability_healing(
    battle: np.ndarray,
    defender_offset: int,
    move_type: int,
    hit: bool,
) -> None:
    """Port of _apply_absorb_ability_healing (line ~8368).

    Healing absorbs:
      - Volt Absorb: Electric -> heal 25%
      - Water Absorb: Water -> heal 25%
      - Dry Skin: Water -> heal 25%
    Stat absorbs:
      - Sap Sipper: Grass -> +1 Atk
      - Storm Drain: Water -> +1 SpA
      - Lightning Rod: Electric -> +1 SpA
      - Motor Drive: Electric -> +1 Spe
    Flash Fire: Fire -> set 0x200 flag.

    NOTE: Showdown fires absorb side-effects via onTryHit, BEFORE damage is
    rolled — so the absorb triggers even though the type immunity zeroes
    damage. Callers must pass `hit=True` for "move was executed" (not
    blocked by sleep/freeze/full para/flinch/sub absorb), NOT "damage > 0".
    Pokepy historically passed the latter, which silently no-op'd every
    absorb (Lightning Rod, Storm Drain, Sap Sipper, Motor Drive, Volt/
    Water Absorb, Dry Skin, Flash Fire) — fixed at battle_gen9.py call site.
    """
    if not hit:
        return
    d_off = int(defender_offset)
    mt = int(move_type)
    def_ability = int(battle[d_off + 5])
    current_hp = int(battle[d_off + 1])
    max_hp = int(battle[d_off + 2])

    volt_absorb_heal = def_ability == ABILITY_VOLT_ABSORB and mt == TYPE_ELECTRIC
    water_absorb_heal = def_ability == ABILITY_WATER_ABSORB and mt == TYPE_WATER
    dry_skin_heal = def_ability == ABILITY_DRY_SKIN and mt == TYPE_WATER
    # Earth Eater — onTryHit for Ground: heal baseMaxhp/4 (data/abilities.ts:1111).
    _ABILITY_EARTH_EATER_AE = 297
    earth_eater_heal = def_ability == _ABILITY_EARTH_EATER_AE and mt == TYPE_GROUND

    should_heal = (
        volt_absorb_heal or water_absorb_heal or dry_skin_heal or earth_eater_heal
    ) and current_hp > 0
    if should_heal:
        heal_amount = max(int(max_hp / 4), 1)
        battle[d_off + 1] = min(max_hp, current_hp + heal_amount)

    sap_sipper_active = def_ability == ABILITY_SAP_SIPPER and mt == TYPE_GRASS
    storm_drain_active = def_ability == ABILITY_STORM_DRAIN and mt == TYPE_WATER
    lightning_rod_active = def_ability == ABILITY_LIGHTNING_ROD and mt == TYPE_ELECTRIC
    motor_drive_active = def_ability == ABILITY_MOTOR_DRIVE and mt == TYPE_ELECTRIC
    # Well-Baked Body — onTryHit for Fire: boost({def: 2}) (data/abilities.ts:5389).
    _ABILITY_WELL_BAKED_BODY_WB = 273
    well_baked_active = def_ability == _ABILITY_WELL_BAKED_BODY_WB and mt == TYPE_FIRE
    # Wind Rider — onTryHit for Wind moves: boost({atk: 1}) (data/abilities.ts:5421).
    # Pokepy doesn't track FLAG_WIND per-move yet, so we only trigger when caller
    # passes move_type_is_wind via FLAG_WIND detection in engine (handled there).

    boosts13 = int(battle[d_off + 13])
    boosts14 = int(battle[d_off + 14])
    if sap_sipper_active:
        boosts13 = apply_boost_to_packed(boosts13, 0, 1)  # +1 Atk
    if storm_drain_active or lightning_rod_active:
        boosts13 = apply_boost_to_packed(boosts13, 8, 1)  # +1 SpA
    if motor_drive_active:
        boosts14 = apply_boost_to_packed(boosts14, 0, 1)  # +1 Spe
    if well_baked_active:
        boosts13 = apply_boost_to_packed(boosts13, 4, 2)  # +2 Def
    battle[d_off + 13] = boosts13
    battle[d_off + 14] = boosts14

    # Flash Fire: bit 9 (0x200) on flags
    if def_ability == ABILITY_FLASH_FIRE and mt == TYPE_FIRE:
        flags = int(battle[d_off + 15])
        new_flags = flags | 0x200
        if new_flags >= 0x8000:
            new_flags -= 0x10000
        battle[d_off + 15] = new_flags


def apply_weakness_policy(
    battle: np.ndarray,
    defender_offset: int,
    move_type: int,
    hit: bool,
    damage_dealt: int,
    move_id: int = -1,
) -> None:
    """Port of _apply_weakness_policy (line ~8460).

    Trigger if hit by a super-effective move while holding Weakness Policy:
    +2 Atk and +2 SpA, then consume the item.

    Showdown items.ts:weaknesspolicy gates on `move.damage` (numeric
    fixed-damage like Dragon Rage / Sonic Boom) and `move.damageCallback`
    (Seismic Toss, Night Shade, Super Fang, Endeavor, Final Gambit,
    Counter, Mirror Coat, Metal Burst, Pain Split, Psywave). Those moves
    don't trigger Weakness Policy because they're not "real" SE damage.
    """
    if not hit:
        return
    d_off = int(defender_offset)
    mt = int(move_type)
    def_item = int(battle[d_off + 6])
    if def_item != ITEM_WEAKNESS_POLICY:
        return
    if int(damage_dealt) <= 0:
        return
    if int(battle[d_off + 1]) <= 0:
        return

    # Skip fixed-damage moves (Seismic Toss, Night Shade, Super Fang,
    # Endeavor, Final Gambit, Counter, Mirror Coat, Metal Burst, Pain Split,
    # Psywave, Dragon Rage, Sonic Boom, Bide).
    _FIXED_DAMAGE_MOVES = frozenset(
        (
            69,  # seismictoss
            100,  # nightshade
            162,  # superfang
            877,  # ruination
            283,  # endeavor
            710,  # finalgambit
            68,  # counter
            243,  # mirrorcoat
            484,  # metalburst
            220,  # painsplit
            149,  # psywave
            82,  # dragonrage
            49,  # sonicboom
            117,  # bide
        )
    )
    if int(move_id) in _FIXED_DAMAGE_MOVES:
        return

    types_packed = int(battle[d_off + 4]) & 0xFFFF
    type1 = types_packed & 0xFF
    type2 = (types_packed >> 8) & 0xFF

    eff1 = float(MODERN_TYPE_CHART[type1, mt])
    eff2 = 1.0 if type2 == type1 else float(MODERN_TYPE_CHART[type2, mt])
    type_mult = eff1 * eff2

    if type_mult <= 1.0:
        return

    boosts13 = int(battle[d_off + 13])
    boosts13 = apply_boost_to_packed(boosts13, 0, 2)  # +2 Atk
    boosts13 = apply_boost_to_packed(boosts13, 8, 2)  # +2 SpA
    battle[d_off + 13] = boosts13
    battle[d_off + 6] = 0  # consume item


def apply_ko_boost_ability(
    battle: np.ndarray,
    attacker_offset: int,
    target_fainted: bool,
    hit: bool,
) -> None:
    """Port of _apply_ko_boost_ability (line ~8520).

    On KOing an opponent:
      - Moxie / Chilling Neigh: +1 Atk
      - Soul-Heart / Grim Neigh: +1 SpA
      - Beast Boost: +1 to highest base stat (excluding HP)
    """
    if not (hit and target_fainted):
        return
    a_off = int(attacker_offset)
    if int(battle[a_off + 1]) <= 0:
        return

    atk_ability = int(battle[a_off + 5])
    has_moxie = atk_ability in (ABILITY_MOXIE, ABILITY_CHILLING_NEIGH)
    has_soul_heart = atk_ability in (ABILITY_SOUL_HEART, ABILITY_GRIM_NEIGH)
    has_beast_boost = atk_ability == ABILITY_BEAST_BOOST

    if not (has_moxie or has_soul_heart or has_beast_boost):
        return

    base_atk = int(battle[a_off + 7])
    base_def = int(battle[a_off + 8])
    base_spa = int(battle[a_off + 9])
    base_spd = int(battle[a_off + 10])
    base_spe = int(battle[a_off + 11])

    highest_is_atk = (
        base_atk >= base_def
        and base_atk >= base_spa
        and base_atk >= base_spd
        and base_atk >= base_spe
    )
    highest_is_def = (
        (not highest_is_atk)
        and base_def >= base_spa
        and base_def >= base_spd
        and base_def >= base_spe
    )
    highest_is_spa = (
        (not highest_is_atk)
        and (not highest_is_def)
        and base_spa >= base_spd
        and base_spa >= base_spe
    )
    highest_is_spd = (
        (not highest_is_atk)
        and (not highest_is_def)
        and (not highest_is_spa)
        and base_spd >= base_spe
    )
    highest_is_spe = (
        (not highest_is_atk)
        and (not highest_is_def)
        and (not highest_is_spa)
        and (not highest_is_spd)
    )

    boosts13 = int(battle[a_off + 13])
    boosts14 = int(battle[a_off + 14])

    boost_atk = has_moxie or (has_beast_boost and highest_is_atk)
    if boost_atk:
        boosts13 = apply_boost_to_packed(boosts13, 0, 1)

    if has_beast_boost and highest_is_def:
        boosts13 = apply_boost_to_packed(boosts13, 4, 1)

    boost_spa = has_soul_heart or (has_beast_boost and highest_is_spa)
    if boost_spa:
        boosts13 = apply_boost_to_packed(boosts13, 8, 1)

    if has_beast_boost and highest_is_spd:
        boosts13 = apply_boost_to_packed(boosts13, 12, 1)

    if has_beast_boost and highest_is_spe:
        boosts14 = apply_boost_to_packed(boosts14, 0, 1)

    battle[a_off + 13] = boosts13
    battle[a_off + 14] = boosts14


def _effect_spore_roll_blocked(battle: np.ndarray, attacker_offset: int) -> bool:
    """Return True when Showdown would skip Effect Spore's chance roll."""
    a_off = int(attacker_offset)
    if get_status(int(battle[a_off + 12])) != STATUS_NONE:
        return True

    attacker_types = int(battle[a_off + 4]) & 0xFFFF
    a_t1 = attacker_types & 0xFF
    a_t2 = (attacker_types >> 8) & 0xFF
    attacker_ability = int(battle[a_off + 5])
    attacker_item = int(battle[a_off + 6])
    return (
        a_t1 == TYPE_GRASS
        or a_t2 == TYPE_GRASS
        or attacker_ability == ABILITY_OVERCOAT
        or attacker_item == ITEM_SAFETY_GOGGLES
    )


def apply_contact_status_ability(
    battle: np.ndarray,
    move_id: int,
    attacker_offset: int,
    defender_offset: int,
    hit: bool,
    game_data,
    move_effects=None,
    gen5_prng: Gen5PRNG = None,
    prerolled_rolls: list = None,
) -> None:
    """Port of _apply_contact_status_ability (line ~10975).

    Defender's contact ability rolls 30% to inflict status on attacker:
      - Flame Body -> Burn
      - Static -> Paralysis
      - Poison Point -> Poison
      - Effect Spore -> 10% sleep / 10% poison / 10% paralysis
      - Cute Charm -> Attract

    Mummy / Wandering Spirit are handled in damage-time logic, not here.

    NOTE: Pokepy's battle_gen9.py calls this as
        apply_contact_status_ability(battle, move_id, atk_off, def_off, hit, gen5_prng)
    so we accept `game_data` as the 6th positional arg (which the engine passes
    as the prng) and treat the resulting `gen5_prng` keyword as optional.

    If ``prerolled_rolls`` is provided, the function pops PRNG values from
    the list instead of calling ``prng.random(100)`` live.  This supports
    the preroll-based PRNG ordering used by battle_gen9.py to keep Flame
    Body / Static / Poison Point / Effect Spore rolls at the correct
    position in the PRNG stream (between the first mover's secondaries
    and the second mover's damage calc).
    """
    # The integration in pokepy/engine/battle_gen9.py calls:
    #   apply_contact_status_ability(battle, move_id0, user0_off, p1_off,
    #                                hit0 and damage0_after_flinch > 0, gen5_prng)
    # i.e. the 6th positional arg is actually the Gen5PRNG. To stay compatible
    # with both call styles we look at the type of `game_data`.
    if isinstance(game_data, Gen5PRNG):
        prng = game_data
        gd = None
    else:
        prng = gen5_prng
        gd = game_data

    if not hit or prng is None:
        return

    if gd is None:
        # Need move_flags lookup; load lazily
        from pokepy.data.loader import load_game_data

        gd = load_game_data()

    mid = int(move_id)
    a_off = int(attacker_offset)
    d_off = int(defender_offset)

    move_flags = int(gd.move_flags[mid])
    is_contact = (move_flags & FLAG_CONTACT) != 0

    atk_ability = int(battle[a_off + 5])
    atk_item = int(battle[a_off + 6])
    if atk_ability == ABILITY_LONG_REACH:
        is_contact = False
    if atk_item == ITEM_PROTECTIVE_PADS:
        is_contact = False

    if not is_contact:
        return

    def_ability = int(battle[d_off + 5])

    # ----- Poison Touch (attacker ability) --------------------------------
    # Showdown data/abilities.ts poisontouch: on source's damaging contact
    # hit, 30% chance to poison the target. Shield Dust / Covert Cloak on
    # the target block (treated as secondary effect).
    _ABILITY_POISON_TOUCH = 143
    _ABILITY_SHIELD_DUST = 19
    _ITEM_COVERT_CLOAK = 1885
    if atk_ability == _ABILITY_POISON_TOUCH:
        target_hp_pt = int(battle[d_off + 1])
        target_status_pt = get_status(int(battle[d_off + 12]))
        target_item_pt = int(battle[d_off + 6])
        target_has_block = (
            def_ability == _ABILITY_SHIELD_DUST or target_item_pt == _ITEM_COVERT_CLOAK
        )
        # Check target type immunity to poison (Poison / Steel). Corrosion
        # on attacker bypasses, matching Showdown's trySetStatus path.
        _ABILITY_CORROSION = 212
        _TYPE_POISON = TYPE_POISON
        _TYPE_STEEL = TYPE_STEEL
        tt_packed = int(battle[d_off + 4]) & 0xFFFF
        tt1 = tt_packed & 0xFF
        tt2 = (tt_packed >> 8) & 0xFF
        type_immune_pt = (
            tt1 == _TYPE_POISON
            or tt2 == _TYPE_POISON
            or tt1 == _TYPE_STEEL
            or tt2 == _TYPE_STEEL
        ) and atk_ability != _ABILITY_CORROSION
        should_roll_pt = not target_has_block
        if should_roll_pt:
            roll_pt = prerolled_rolls.pop(0) if prerolled_rolls else prng.random(10)
            if (
                roll_pt < 3
                and target_hp_pt > 0
                and target_status_pt == STATUS_NONE
                and not type_immune_pt
            ):
                battle[d_off + 12] = set_status(STATUS_POISON, 0)

    has_flame_body = def_ability == ABILITY_FLAME_BODY
    has_static = def_ability == ABILITY_STATIC
    has_poison_point = def_ability == ABILITY_POISON_POINT
    has_effect_spore = def_ability == ABILITY_EFFECT_SPORE
    has_cute_charm = def_ability == ABILITY_CUTE_CHARM
    has_contact_ability = (
        has_flame_body
        or has_static
        or has_poison_point
        or has_effect_spore
        or has_cute_charm
    )
    if not has_contact_ability:
        return

    if int(battle[a_off + 1]) <= 0:
        return

    # 30% roll — Showdown: Flame Body/Static/Poison Point/Cute Charm use
    # randomChance(3, 10) → random(10) < 3. Effect Spore uses random(100)
    # with thresholds at 11/21/30.
    if has_effect_spore and _effect_spore_roll_blocked(battle, a_off):
        return
    if prerolled_rolls:
        roll = prerolled_rolls.pop(0)
    elif has_effect_spore:
        roll = prng.random(100)
    else:
        roll = prng.random(10)
    # For Flame Body/Static/Poison Point: roll >= 3 fails (out of 10).
    # For Effect Spore: roll >= 30 fails (out of 100). Check below.
    if has_effect_spore:
        if roll >= 30:
            return
    else:
        if roll >= 3:
            return

    if has_cute_charm:
        if move_effects is None:
            from pokepy.data.loader import load_move_effect_data

            move_effects = load_move_effect_data()
        from pokepy.effects.volatiles import apply_extended_volatile

        attacker_side = 0 if a_off < OFF_SIDE1 else 1
        defender_side = 0 if d_off < OFF_SIDE1 else 1
        apply_extended_volatile(
            battle,
            MOVE_ATTRACT,
            defender_side,
            attacker_side,
            True,
            game_data=gd,
            move_effects=move_effects,
            gen5_prng=prng,
        )
        return

    # Shield Dust / Covert Cloak block all contact secondary statuses.
    # Showdown handles these via runEvent('TryHit' ,...) which Shield Dust
    # applies as `this.add('-immune', target, 'Shield Dust')`. Pokepy
    # already gates Flame Body / Static / Poison Point by rolling but
    # missed the block path. Block by returning early once we know the
    # ability tried to fire.
    _SHIELD_DUST_FS = 19
    _ITEM_COVERT_CLOAK_FS = 816
    if int(battle[a_off + 5]) == _SHIELD_DUST_FS or atk_item == _ITEM_COVERT_CLOAK_FS:
        return

    # Contact status abilities consume their chance roll first, then let
    # Showdown's shared `trySetStatus` gate decide whether the status can
    # actually stick. Reuse that same gate here so Comatose, Purifying Salt,
    # Safeguard, terrain, and type/ability immunities all stay aligned.
    if has_flame_body:
        _try_apply_status(
            battle,
            None,
            a_off,
            STATUS_BURN,
            gd,
            prng,
            user_offset=d_off,
            is_status_move=False,
        )
        return
    if has_static:
        _try_apply_status(
            battle,
            None,
            a_off,
            STATUS_PARALYSIS,
            gd,
            prng,
            user_offset=d_off,
            is_status_move=False,
        )
        return
    if has_poison_point:
        _try_apply_status(
            battle,
            None,
            a_off,
            STATUS_POISON,
            gd,
            prng,
            user_offset=d_off,
            is_status_move=False,
        )
        return
    if has_effect_spore:
        # Showdown abilities.ts:1127-1134 checks the source's status and
        # powder immunity before it spends the random(100) frame. The shared
        # guard above keeps both live and prerolled Effect Spore paths aligned
        # with that ordering.
        if roll < 11:
            prerolled_sleep_turns = None
            if prerolled_rolls:
                prerolled_sleep_turns = int(prerolled_rolls.pop(0))
            _try_apply_status(
                battle,
                None,
                a_off,
                STATUS_SLEEP,
                gd,
                prng,
                user_offset=d_off,
                is_status_move=False,
                prerolled_sleep_turns=prerolled_sleep_turns,
            )
        elif roll < 21:
            _try_apply_status(
                battle,
                None,
                a_off,
                STATUS_PARALYSIS,
                gd,
                prng,
                user_offset=d_off,
                is_status_move=False,
            )
        else:  # roll < 30
            _try_apply_status(
                battle,
                None,
                a_off,
                STATUS_POISON,
                gd,
                prng,
                user_offset=d_off,
                is_status_move=False,
            )


def apply_resolved_contact_status_ability(
    battle: np.ndarray,
    attacker_offset: int,
    defender_offset: int,
    game_data,
    move_effects=None,
    gen5_prng: Gen5PRNG = None,
    *,
    resolved_status_field: int | None = None,
    apply_attract: bool = False,
) -> bool:
    """Materialize a contact-status result whose PRNG was already consumed.

    `calc_damage_gen9` uses this for multihit contact moves where Showdown
    resolves Flame Body / Static / Poison Point / Effect Spore / Cute Charm
    inside the per-hit loop. The damage calc burns the PRNG frames on a shadow
    battle, records the final status/volatile result in `out_meta`, and the
    engine later calls this helper at the real timing point without rerolling.
    """
    a_off = int(attacker_offset)
    d_off = int(defender_offset)
    before_status = int(battle[a_off + 12])
    before_status_code = get_status(before_status)

    if apply_attract:
        if move_effects is None:
            from pokepy.data.loader import load_move_effect_data

            move_effects = load_move_effect_data()
        from pokepy.effects.volatiles import apply_extended_volatile

        attacker_side = 0 if a_off < OFF_SIDE1 else 1
        defender_side = 0 if d_off < OFF_SIDE1 else 1
        apply_extended_volatile(
            battle,
            MOVE_ATTRACT,
            defender_side,
            attacker_side,
            True,
            game_data=game_data,
            move_effects=move_effects,
            gen5_prng=gen5_prng,
        )

    if resolved_status_field is not None:
        status_code = get_status(int(resolved_status_field))
        if status_code != STATUS_NONE:
            prerolled_sleep_turns = None
            if status_code == STATUS_SLEEP:
                prerolled_sleep_turns = int(
                    get_status_turns(int(resolved_status_field))
                )
            _try_apply_status(
                battle,
                None,
                a_off,
                status_code,
                game_data,
                gen5_prng,
                user_offset=d_off,
                is_status_move=False,
                prerolled_sleep_turns=prerolled_sleep_turns,
            )

    return get_status(int(battle[a_off + 12])) != before_status_code
