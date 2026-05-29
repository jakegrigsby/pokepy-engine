"""Stat-change effects (port of _apply_stat_changes_from_move).

(line ~7225). Mutates `battle` in place. Advances `gen5_prng` only when the
move actually has a stat-change effect (matches Showdown's behavior).
"""

from __future__ import annotations

from pokepy.effects._common import np, MultiFormatState, Gen5PRNG
from pokepy.core.bitpack import apply_boost_to_packed
from pokepy.effects.ability_suppression import effective_ability
from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_FIELD,
    OFF_META,
    M_ACTIVE0,
    M_ACTIVE1,
    POKEMON_SIZE,
    F_WEATHER,
    ABILITY_SERENE_GRACE,
    ABILITY_CLEAR_BODY,
    ABILITY_WHITE_SMOKE,
    ABILITY_FULL_METAL_BODY,
    ABILITY_CONTRARY,
    ABILITY_SIMPLE,
    ABILITY_DEFIANT,
    ABILITY_COMPETITIVE,
    ABILITY_AIR_LOCK,
    ABILITY_CLOUD_NINE,
    ITEM_UTILITY_UMBRELLA,
    WEATHER_NONE,
    WEATHER_SUN,
    WEATHER_RAIN,
    WEATHER_PRIMORDIAL_SEA,
    WEATHER_DESOLATE_LAND,
)

# IDs not exported from constants.py — defined locally to mirror the Showdown reference usage
ITEM_CLEAR_AMULET = 1882
ABILITY_MIRROR_ARMOR = 240
ABILITY_OPPORTUNIST = 290
MOVE_GROWTH = 74
MOVE_SKULL_BASH = 130
MOVE_CLANGING_SCALES = 691
MOVE_CLANGOROUS_SOULBLAZE = 728
MOVE_SCALE_SHOT = 799
MOVE_METEOR_BEAM = 800
MOVE_ELECTRO_SHOT = 905
_UMBRELLA_EFFECTIVE_WEATHERS = (
    WEATHER_SUN,
    WEATHER_RAIN,
    WEATHER_PRIMORDIAL_SEA,
    WEATHER_DESOLATE_LAND,
)


def _effective_weather_for_pokemon(
    battle: np.ndarray,
    pokemon_offset: int,
) -> int:
    """Showdown-style `pokemon.effectiveWeather()` for active users.

    Growth uses `pokemon.effectiveWeather()` in Showdown, so the holder's
    Utility Umbrella suppresses sun/rain/primal weather for this move while
    Air Lock / Cloud Nine on either active mon suppress all weather effects.
    """
    weather = int(battle[OFF_FIELD + F_WEATHER])
    if weather == WEATHER_NONE:
        return WEATHER_NONE

    active0 = int(battle[OFF_META + M_ACTIVE0])
    active1 = int(battle[OFF_META + M_ACTIVE1])
    off0 = OFF_SIDE0 + active0 * POKEMON_SIZE
    off1 = OFF_SIDE1 + active1 * POKEMON_SIZE
    ab0 = effective_ability(battle, off0, off1)
    ab1 = effective_ability(battle, off1, off0)
    if ab0 in (ABILITY_AIR_LOCK, ABILITY_CLOUD_NINE) or ab1 in (
        ABILITY_AIR_LOCK,
        ABILITY_CLOUD_NINE,
    ):
        return WEATHER_NONE

    if (
        int(battle[int(pokemon_offset) + 6]) == ITEM_UTILITY_UMBRELLA
        and weather in _UMBRELLA_EFFECTIVE_WEATHERS
    ):
        return WEATHER_NONE
    return weather


def get_live_move_stat_change_spec(
    battle: np.ndarray,
    move_id: int,
    move_effects,
    user_offset: int,
) -> tuple[list[int], int, int, bool]:
    """Return Showdown-style live stat-change metadata for a move.

    The static move tables miss a few runtime behaviors that matter for
    strict parity:
    - Growth upgrades to +2 Atk / +2 SpA under effective sun.
    - Top-level `selfBoost` moves such as Clanging Scales are not encoded in
      pokepy's stat-change arrays, but Showdown still applies them in the
      same move-resolution window.

    Returns `(stat_changes, stat_target, stat_chance, is_selfboost_like)`.
    """
    move_id = int(move_id)
    sc_arr = move_effects.stat_changes[move_id]
    stat_changes = [int(sc_arr[i]) for i in range(7)]
    stat_target = int(move_effects.stat_target[move_id])  # 0=self, 1=opponent
    stat_chance = int(move_effects.stat_chance[move_id])
    is_selfboost_like = False

    # Showdown upgrades Growth to +2 Atk / +2 SpA when the user's
    # effectiveWeather() is sun or desolate land.
    if move_id == MOVE_GROWTH and stat_target == 0:
        effective_weather = _effective_weather_for_pokemon(battle, user_offset)
        if effective_weather in (WEATHER_SUN, WEATHER_DESOLATE_LAND):
            stat_changes[0] = 2
            stat_changes[2] = 2

    # Top-level Showdown `selfBoost` moves are not represented in pokepy's
    # static move-effect arrays. Synthesize them here so both the immediate
    # pre-apply timing and the later shared stat-change block can reuse the
    # same live move spec.
    if move_id == MOVE_CLANGING_SCALES:
        stat_changes[1] = -1
        stat_target = 0
        stat_chance = 100
        is_selfboost_like = True
    elif move_id == MOVE_CLANGOROUS_SOULBLAZE:
        stat_changes[0] = 1
        stat_changes[1] = 1
        stat_changes[2] = 1
        stat_changes[3] = 1
        stat_changes[4] = 1
        stat_target = 0
        stat_chance = 100
        is_selfboost_like = True
    elif move_id == MOVE_SCALE_SHOT:
        is_selfboost_like = True

    return stat_changes, stat_target, stat_chance, is_selfboost_like


def apply_direct_stat_changes(
    battle: np.ndarray,
    source_offset: int,
    target_offset: int,
    stat_changes: list[int] | tuple[int, ...],
    *,
    stat_target: int = 1,
    allow_mirror_armor: bool = True,
) -> None:
    """Apply a direct stat-stage effect without secondary PRNG.

    This is the shared Showdown-style boost pipeline for non-move callers
    such as Sticky Web's onSwitchIn hook, which still needs the normal
    onTryBoost / onAfterEachBoost reactions (Clear Amulet, Contrary,
    Mirror Armor, Defiant, Competitive, etc.) even though no secondary
    effect roll occurs.
    """
    source_offset = int(source_offset)
    target_offset = int(target_offset)
    stat_target = int(stat_target)
    pokemon_offset = source_offset if stat_target == 0 else target_offset

    target_ability = effective_ability(
        battle,
        pokemon_offset,
        target_offset if stat_target == 0 else source_offset,
    )

    has_clear_body = target_ability in (
        ABILITY_CLEAR_BODY,
        ABILITY_WHITE_SMOKE,
        ABILITY_FULL_METAL_BODY,
    )
    is_opponent_drop = stat_target == 1

    has_contrary = target_ability == ABILITY_CONTRARY
    has_simple = target_ability == ABILITY_SIMPLE

    target_item = int(battle[pokemon_offset + 6])
    has_clear_amulet = target_item == ITEM_CLEAR_AMULET
    has_mirror_armor = (
        allow_mirror_armor
        and is_opponent_drop
        and target_ability == ABILITY_MIRROR_ARMOR
    )
    _ABILITY_HYPER_CUTTER = 52
    _ABILITY_BIG_PECKS = 145
    _ABILITY_KEEN_EYE = 51
    _ABILITY_MINDS_EYE = 300
    _ABILITY_ILLUMINATE = 35
    has_hyper_cutter = target_ability == _ABILITY_HYPER_CUTTER
    has_big_pecks = target_ability == _ABILITY_BIG_PECKS
    has_keen_eye = target_ability in (
        _ABILITY_KEEN_EYE,
        _ABILITY_MINDS_EYE,
        _ABILITY_ILLUMINATE,
    )

    reflected = [0] * 7  # [atk, def, spa, spd, spe, acc, eva]

    def process_stat_change(idx: int, change: int) -> int:
        nonlocal reflected
        if is_opponent_drop and idx == 0 and has_hyper_cutter and change < 0:
            return 0
        if is_opponent_drop and idx == 1 and has_big_pecks and change < 0:
            return 0
        if is_opponent_drop and idx == 5 and has_keen_eye and change < 0:
            return 0
        if is_opponent_drop and (has_clear_body or has_clear_amulet) and change < 0:
            return 0
        if has_mirror_armor and change < 0:
            reflected[idx] = change
            return 0
        if has_contrary:
            change = -change
        if has_simple:
            change = change * 2
        return change

    boosts13 = int(battle[pokemon_offset + 13])
    boosts14 = int(battle[pokemon_offset + 14])

    atk_change = process_stat_change(0, int(stat_changes[0]))
    if atk_change != 0:
        boosts13 = apply_boost_to_packed(boosts13, 0, atk_change)
    def_change = process_stat_change(1, int(stat_changes[1]))
    if def_change != 0:
        boosts13 = apply_boost_to_packed(boosts13, 4, def_change)
    spa_change = process_stat_change(2, int(stat_changes[2]))
    if spa_change != 0:
        boosts13 = apply_boost_to_packed(boosts13, 8, spa_change)
    spd_change = process_stat_change(3, int(stat_changes[3]))
    if spd_change != 0:
        boosts13 = apply_boost_to_packed(boosts13, 12, spd_change)
    spe_change = process_stat_change(4, int(stat_changes[4]))
    if spe_change != 0:
        boosts14 = apply_boost_to_packed(boosts14, 0, spe_change)
    acc_change = process_stat_change(5, int(stat_changes[5]))
    if acc_change != 0:
        boosts14 = apply_boost_to_packed(boosts14, 4, acc_change)
    eva_change = process_stat_change(6, int(stat_changes[6]))
    if eva_change != 0:
        boosts14 = apply_boost_to_packed(boosts14, 8, eva_change)

    battle[pokemon_offset + 13] = boosts13
    battle[pokemon_offset + 14] = boosts14

    any_stat_lowered = is_opponent_drop and (
        atk_change < 0
        or def_change < 0
        or spa_change < 0
        or spd_change < 0
        or spe_change < 0
        or acc_change < 0
        or eva_change < 0
    )
    if any_stat_lowered:
        boosts13_after = int(battle[pokemon_offset + 13])
        if target_ability == ABILITY_DEFIANT:
            boosts13_after = apply_boost_to_packed(boosts13_after, 0, 2)
        if target_ability == ABILITY_COMPETITIVE:
            boosts13_after = apply_boost_to_packed(boosts13_after, 8, 2)
        battle[pokemon_offset + 13] = boosts13_after

    if any(c != 0 for c in reflected):
        apply_direct_stat_changes(
            battle,
            target_offset,
            source_offset,
            reflected,
            stat_target=1,
            allow_mirror_armor=False,
        )


def apply_stat_changes_from_move(
    battle: np.ndarray,
    move_id: int,
    user_offset: int,
    target_offset: int,
    hit: bool,
    game_data,
    move_effects,
    gen5_prng: Gen5PRNG,
    prerolled_roll: "int | None" = None,
) -> None:
    """Port of MultiFormatFastEnv._apply_stat_changes_from_move (line ~7225).

    If `prerolled_roll` is not None, use it as the secondary-chance random
    value (range [0, 100)) instead of consuming a PRNG frame here. This is
    used by the engine to inject secondary rolls at the correct PRNG
    position (between each move's damage roll and the next move's damage
    roll) to match Showdown's per-move secondaryRoll ordering.
    """
    move_id = int(move_id)
    user_offset = int(user_offset)
    target_offset = int(target_offset)
    hit = bool(hit)

    stat_changes, stat_target, stat_chance, is_selfboost_like = (
        get_live_move_stat_change_spec(
            battle,
            move_id,
            move_effects,
            user_offset,
        )
    )

    # Serene Grace: double secondary effect chance
    user_ability_se = effective_ability(battle, user_offset, target_offset)
    if user_ability_se == ABILITY_SERENE_GRACE:
        stat_chance = min(stat_chance * 2, 100)

    # Sheer Force (data/abilities.ts:4122) deletes `move.secondaries` if the
    # move has any. Stat-change secondaries (chance < 100) are removed and
    # never roll a random in Showdown. Primary stat changes (chance == 100,
    # e.g. Bulk Up's atk/def boosts) are NOT secondaries and are unaffected.
    # Primary self-drops on damaging moves (Close Combat) come through
    # `move.self` which Sheer Force ALSO deletes — but we can't distinguish
    # those from move-level boosts in pokepy's data, so we keep it
    # conservative and only suppress true secondaries (chance < 100).
    from pokepy.core.constants import ABILITY_SHEER_FORCE as _ABILITY_SHEER_FORCE_SC

    has_sheer_force = user_ability_se == _ABILITY_SHEER_FORCE_SC
    if has_sheer_force and stat_chance < 100:
        return

    # Shield Dust (data/abilities.ts:shielddust) and Covert Cloak
    # (data/items.ts:covertcloak) filter out non-self secondaries via
    # `onModifySecondaries` BEFORE the secondaryRoll — so no PRNG frame is
    # consumed when the target has either. Opponent-target stat changes on
    # damaging moves are ALWAYS secondaries in Showdown (even 100%-chance
    # ones like Skitter Smack, Lunge, Breaking Swipe — they're in the
    # `secondary` block of moves.ts, not a top-level `boosts`). The only
    # top-level `boosts` on opponent-target moves come from STATUS-category
    # moves (Growl, Leer, Fake Tears, Memento, ...), which have
    # effect_type == EFFECT_STAT_CHANGE in pokepy and are unaffected.
    from pokepy.data.move_effects import (
        EFFECT_STAT_CHANGE as _EFFECT_STAT_CHANGE_SC,
        EFFECT_STATUS as _EFFECT_STATUS_SC,
        EFFECT_DEFOG as _EFFECT_DEFOG_SC,
    )
    from pokepy.core.constants import EFFECT_SWITCH as _EFFECT_SWITCH_SC

    _ABILITY_SHIELD_DUST_SC = 19
    _ITEM_COVERT_CLOAK_SC = 750
    move_effect_type_sc = int(move_effects.effect_type[move_id])
    is_primary_stat_move = move_effect_type_sc in (
        _EFFECT_STAT_CHANGE_SC,
        _EFFECT_STATUS_SC,
    )
    if stat_target == 1 and not is_primary_stat_move:
        target_ability_sd = effective_ability(battle, target_offset, user_offset)
        target_item_sd = int(battle[target_offset + 6])
        if (
            target_ability_sd == _ABILITY_SHIELD_DUST_SC
            or target_item_sd == _ITEM_COVERT_CLOAK_SC
        ):
            return

    # Only advance PRNG if move has any stat changes
    has_any_stat_change = (
        hit and (stat_chance > 0) and any(c != 0 for c in stat_changes)
    )
    if not has_any_stat_change:
        return

    # Showdown only calls secondaryRoll (the random(100) gate) for TRUE
    # secondaries — i.e. entries in `move.self.boosts` (Close Combat) or
    # `move.secondaries` (Crunch's def drop) — NOT for primary status
    # moves that apply boosts via top-level `move.boosts` (Calm Mind,
    # Bulk Up, Swords Dance, Dragon Dance, ...). Primary stat moves
    # resolve in `moveHit` via a direct `this.battle.boost` call without
    # consuming a PRNG frame. Pokepy used to roll `random(100)` for every
    # stat-change move, drifting the LCG by one frame per primary boost.
    _is_primary_status_boost = (
        move_effect_type_sc == _EFFECT_STAT_CHANGE_SC
        or move_effect_type_sc == _EFFECT_DEFOG_SC
        or move_effect_type_sc
        == _EFFECT_SWITCH_SC  # Parting Shot: onHit boost, no PRNG
        or move_id in (MOVE_SKULL_BASH, MOVE_METEOR_BEAM, MOVE_ELECTRO_SHOT)
    )
    if _is_primary_status_boost or is_selfboost_like:
        # Showdown top-level `selfBoost` resolves directly from moveResult,
        # not through the secondary-effect randomChance path.
        should_apply = True
    else:
        if prerolled_roll is None:
            roll = gen5_prng.random(100)
        else:
            roll = int(prerolled_roll)
        should_apply = roll < stat_chance
        if not should_apply:
            return

    # Showdown still spends the secondaryRoll frame even if the target faints
    # to the primary hit, but `battle.boost(...)` then sees no live target and
    # the secondary stat drop no-ops. Mirror that here for true
    # opponent-targeted damaging stat changes while preserving primary status
    # moves (Growl, Parting Shot, etc.) and self boosts/drops.
    if (
        stat_target == 1
        and not is_primary_stat_move
        and int(battle[target_offset + 1]) <= 0
    ):
        return

    # Determine which Pokemon to modify
    pokemon_offset = user_offset if stat_target == 0 else target_offset

    # Get target ability for ability checks
    target_ability = effective_ability(
        battle,
        pokemon_offset,
        target_offset if stat_target == 0 else user_offset,
    )

    has_clear_body = target_ability in (
        ABILITY_CLEAR_BODY,
        ABILITY_WHITE_SMOKE,
        ABILITY_FULL_METAL_BODY,
    )
    is_opponent_drop = stat_target == 1

    has_contrary = target_ability == ABILITY_CONTRARY
    has_simple = target_ability == ABILITY_SIMPLE

    target_item_sa = int(battle[pokemon_offset + 6])
    has_clear_amulet = target_item_sa == ITEM_CLEAR_AMULET
    has_mirror_armor = target_ability == ABILITY_MIRROR_ARMOR
    # Hyper Cutter ONLY blocks Atk drops (idx 0). Showdown source:
    # data/abilities.ts hypercutter `onTryBoost: { if (boost.atk && boost.atk < 0) ... }`
    from pokepy.core.constants import ABILITY_HYPER_CUTTER

    has_hyper_cutter = target_ability == ABILITY_HYPER_CUTTER
    # Big Pecks ONLY blocks Def drops (idx 1). Showdown source:
    # data/abilities.ts bigpecks `onTryBoost: { if (boost.def && boost.def < 0) ... }`
    _ABILITY_BIG_PECKS = 145
    has_big_pecks = target_ability == _ABILITY_BIG_PECKS
    # Keen Eye / Mind's Eye / Illuminate block accuracy drops. Showdown source:
    # data/abilities.ts keeneye / mindseye / illuminate onTryBoost. Illuminate
    # gained this effect in Gen 9 Indigo Disk (see mods/gen9predlc/abilities.ts
    # which reverts illuminate to an empty ability).
    _ABILITY_KEEN_EYE = 51
    _ABILITY_MINDS_EYE = 300
    _ABILITY_ILLUMINATE = 35
    has_keen_eye_sc = target_ability in (
        _ABILITY_KEEN_EYE,
        _ABILITY_MINDS_EYE,
        _ABILITY_ILLUMINATE,
    )

    # Mirror Armor reflects negative stat changes back to the source (Showdown:
    # data/abilities.ts mirrorarmor onTryBoost). Track reflected changes here
    # and apply them after the main stat-change loop to the user_offset.
    reflected = [0] * 7  # [atk, def, spa, spd, spe, acc, eva]

    def process_stat_change(idx: int, change: int) -> int:
        nonlocal reflected
        # Hyper Cutter blocks Atk drops only (idx 0 = atk)
        if is_opponent_drop and has_hyper_cutter and idx == 0 and change < 0:
            return 0
        # Big Pecks blocks Def drops only (idx 1 = def)
        if is_opponent_drop and has_big_pecks and idx == 1 and change < 0:
            return 0
        # Keen Eye / Mind's Eye block accuracy drops only (idx 5 = accuracy)
        if is_opponent_drop and has_keen_eye_sc and idx == 5 and change < 0:
            return 0
        # Clear Body / Clear Amulet block negative changes from opponent (no reflect)
        if is_opponent_drop and (has_clear_body or has_clear_amulet) and change < 0:
            return 0
        # Mirror Armor: reflect negative changes back to source
        if is_opponent_drop and has_mirror_armor and change < 0:
            reflected[idx] = change
            return 0
        if has_contrary:
            change = -change
        if has_simple:
            change = change * 2
        return change

    # Get current boost values
    boosts13 = int(battle[pokemon_offset + 13])
    boosts14 = int(battle[pokemon_offset + 14])

    # Apply each stat change in order
    # offset 13: atk@0, def@4, spa@8, spd@12
    # offset 14: spe@0, acc@4, eva@8
    atk_change = process_stat_change(0, stat_changes[0])
    if atk_change != 0:
        boosts13 = apply_boost_to_packed(boosts13, 0, atk_change)
    def_change = process_stat_change(1, stat_changes[1])
    if def_change != 0:
        boosts13 = apply_boost_to_packed(boosts13, 4, def_change)
    spa_change = process_stat_change(2, stat_changes[2])
    if spa_change != 0:
        boosts13 = apply_boost_to_packed(boosts13, 8, spa_change)
    spd_change = process_stat_change(3, stat_changes[3])
    if spd_change != 0:
        boosts13 = apply_boost_to_packed(boosts13, 12, spd_change)
    spe_change = process_stat_change(4, stat_changes[4])
    if spe_change != 0:
        boosts14 = apply_boost_to_packed(boosts14, 0, spe_change)
    acc_change = process_stat_change(5, stat_changes[5])
    if acc_change != 0:
        boosts14 = apply_boost_to_packed(boosts14, 4, acc_change)
    eva_change = process_stat_change(6, stat_changes[6])
    if eva_change != 0:
        boosts14 = apply_boost_to_packed(boosts14, 8, eva_change)

    battle[pokemon_offset + 13] = boosts13
    battle[pokemon_offset + 14] = boosts14

    # Mirror Armor: apply reflected drops to the source (user_offset). Note:
    # the source's own Clear Body / Mirror Armor does NOT block reflection
    # (Showdown: 'effect.name === Mirror Armor' is the early-return check).
    if any(c != 0 for c in reflected):
        src_b13 = int(battle[user_offset + 13])
        src_b14 = int(battle[user_offset + 14])
        for shift, c in (
            (0, reflected[0]),
            (4, reflected[1]),
            (8, reflected[2]),
            (12, reflected[3]),
        ):
            if c != 0:
                src_b13 = apply_boost_to_packed(src_b13, shift, c)
        for shift, c in ((0, reflected[4]), (4, reflected[5]), (8, reflected[6])):
            if c != 0:
                src_b14 = apply_boost_to_packed(src_b14, shift, c)
        battle[user_offset + 13] = src_b13
        battle[user_offset + 14] = src_b14

    # Defiant/Competitive: +2 Atk/SpA when any stat is lowered by opponent
    any_stat_lowered = is_opponent_drop and (
        atk_change < 0
        or def_change < 0
        or spa_change < 0
        or spd_change < 0
        or spe_change < 0
        or acc_change < 0
        or eva_change < 0
    )

    has_defiant = target_ability == ABILITY_DEFIANT
    has_competitive = target_ability == ABILITY_COMPETITIVE

    if any_stat_lowered:
        boosts13_after = int(battle[pokemon_offset + 13])
        if has_defiant:
            boosts13_after = apply_boost_to_packed(boosts13_after, 0, 2)  # +2 Atk
        if has_competitive:
            boosts13_after = apply_boost_to_packed(boosts13_after, 8, 2)  # +2 SpA
        battle[pokemon_offset + 13] = boosts13_after

    # Mirror Armor reflection is handled in the new reflected[] block above
    # (which runs immediately after the boost application).

    # Opportunist: copy positive self-boosts to the opponent
    # Mirror Herb: same effect, but consumes the item
    if stat_target == 0:
        if user_offset < OFF_SIDE1:
            opp_active = int(battle[OFF_META + M_ACTIVE1])
            opp_of_user_offset = OFF_SIDE1 + opp_active * POKEMON_SIZE
        else:
            opp_active = int(battle[OFF_META + M_ACTIVE0])
            opp_of_user_offset = OFF_SIDE0 + opp_active * POKEMON_SIZE
        opp_has_opportunist = int(battle[opp_of_user_offset + 5]) == ABILITY_OPPORTUNIST
        from pokepy.core.constants import ITEM_MIRROR_HERB

        opp_has_mirror_herb = int(battle[opp_of_user_offset + 6]) == ITEM_MIRROR_HERB
        if opp_has_opportunist or opp_has_mirror_herb:
            opp_b13 = int(battle[opp_of_user_offset + 13])
            opp_b14 = int(battle[opp_of_user_offset + 14])
            any_pos = False
            for stat_idx, shift, is_14 in [
                (0, 0, False),
                (1, 4, False),
                (2, 8, False),
                (3, 12, False),
                (4, 0, True),
            ]:
                raw_c = stat_changes[stat_idx]
                if raw_c > 0:
                    any_pos = True
                    if is_14:
                        opp_b14 = apply_boost_to_packed(opp_b14, shift, raw_c)
                    else:
                        opp_b13 = apply_boost_to_packed(opp_b13, shift, raw_c)
            battle[opp_of_user_offset + 13] = opp_b13
            battle[opp_of_user_offset + 14] = opp_b14
            if any_pos and opp_has_mirror_herb and not opp_has_opportunist:
                battle[opp_of_user_offset + 6] = 0  # consume Mirror Herb
