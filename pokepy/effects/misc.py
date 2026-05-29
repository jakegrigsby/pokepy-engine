"""Misc move effects: Knock Off, Trick, Rapid Spin, Defog, Haze, Clear Smog,
Psych Up, Screens.

Port of MultiFormatFastEnv._apply_knock_off_from_move and friends
(the Showdown reference implementation).
"""

from __future__ import annotations

from pokepy.effects._common import np, MultiFormatState, Gen5PRNG
from pokepy.core.bitpack import apply_boost_to_packed
from pokepy.core.constants import (
    EFFECT_CLEAR_SMOG,
    EFFECT_DEFOG,
    EFFECT_HAZE,
    EFFECT_KNOCK_OFF,
    EFFECT_PSYCH_UP,
    EFFECT_RAPID_SPIN,
    F_CHOICE_LOCK_0,
    F_CHOICE_LOCK_1,
    F_HAZARDS_0,
    F_HAZARDS_1,
    F_LEECH_SEED_0,
    F_LEECH_SEED_1,
    F_SCREENS_0,
    F_SCREENS_1,
    F_WEATHER,
    ITEM_LIGHT_CLAY,
    M_ACTIVE0,
    M_ACTIVE1,
    MOVE_AURORA_VEIL,
    MOVE_LIGHT_SCREEN,
    MOVE_LUCKY_CHANT,
    MOVE_MIST,
    MOVE_REFLECT,
    MOVE_SAFEGUARD,
    MOVE_SWITCHEROO,
    MOVE_TAILWIND,
    MOVE_TRICK,
    NEUTRAL_BOOSTS_13,
    NEUTRAL_BOOSTS_14,
    OFF_FIELD,
    OFF_META,
    OFF_SIDE0,
    OFF_SIDE1,
    POKEMON_SIZE,
    SCREEN_AURORAVEIL_SHIFT,
    SCREEN_LIGHTSCREEN_SHIFT,
    SCREEN_LUCKYCHANT_SHIFT,
    SCREEN_MASK_2BIT,
    SCREEN_MASK_3BIT,
    SCREEN_MIST_SHIFT,
    SCREEN_REFLECT_SHIFT,
    SCREEN_SAFEGUARD_SHIFT,
    SCREEN_TAILWIND_SHIFT,
    WEATHER_SNOW,
    ITEM_BOOSTER_ENERGY,
    screen_clear_mask,
)


def _to_int16(val: int) -> int:
    val = int(val) & 0xFFFF
    if val >= 0x8000:
        val -= 0x10000
    return val


_PARADOX_SPECIES = frozenset(
    {
        984,  # Great Tusk
        985,  # Scream Tail
        986,  # Brute Bonnet
        987,  # Flutter Mane
        988,  # Slither Wing
        989,  # Sandy Shocks
        990,  # Iron Treads
        991,  # Iron Bundle
        992,  # Iron Hands
        993,  # Iron Jugulis
        994,  # Iron Moth
        995,  # Iron Thorns
        1005,  # Roaring Moon
        1006,  # Iron Valiant
        1009,  # Walking Wake
        1010,  # Iron Leaves
        1020,  # Gouging Fire
        1021,  # Raging Bolt
        1022,  # Iron Boulder
        1023,  # Iron Crown
    }
)


def is_take_item_blocked_by_item_rule(item_id: int, holder_species: int) -> bool:
    """Mirror the item-level `onTakeItem` vetoes that matter in current Gen 9 parity.

    This intentionally models the item-data side of Showdown's `TakeItem`
    event only. Ability-side gates like Sticky Hold stay move / source
    dependent and are handled by the callers.
    """
    item_id = int(item_id)
    holder_species = int(holder_species)
    if item_id <= 0:
        return False
    # Ogerpon masks.
    if item_id in (758, 759, 760):
        return True
    # Arceus plates, Genesect drives, and Silvally memories while held by the
    # matching species.
    if holder_species == 493 and 185 <= item_id <= 202:
        return True
    if holder_species == 649 and 116 <= item_id <= 119:
        return True
    if holder_species == 773 and 596 <= item_id <= 613:
        return True
    # Booster Energy cannot be removed from a Paradox species.
    if item_id == ITEM_BOOSTER_ENERGY and holder_species in _PARADOX_SPECIES:
        return True
    return False


def apply_knock_off_from_move(
    battle: np.ndarray,
    move_id: int,
    target_offset: int,
    hit: bool,
    game_data=None,
    move_effects=None,
    user_offset: int | None = None,
    source_alive: bool | None = None,
) -> None:
    """Port of _apply_knock_off_from_move (line ~11062).

    Removes the target's item if the move is Knock Off, hit, target alive,
    USER alive (Showdown onAfterHit: `if (source.hp)` gate), sub didn't
    block the hit, and item is not an Ogerpon mask. Also clears the target
    side's choice lock when the item is removed.
    """
    if move_effects is None:
        return
    move_id = int(move_id)
    target_offset = int(target_offset)
    hit = bool(hit)

    move_effect = int(move_effects.effect_type[move_id])
    is_knock_off = move_effect == EFFECT_KNOCK_OFF
    if not is_knock_off or not hit:
        return

    # User-alive gate — Showdown knockoff onAfterHit: `if (source.hp)`.
    # Prevents the item from being knocked off if the user fainted to
    # Rocky Helmet / Iron Barbs / Aftermath mid-move.
    if source_alive is None and user_offset is not None:
        source_alive = int(battle[int(user_offset) + 1]) > 0
    if source_alive is False:
        return

    target_item = int(battle[target_offset + 6])
    target_hp = int(battle[target_offset + 1])
    target_ability = int(battle[target_offset + 5])
    target_species = int(battle[target_offset + 0])

    is_unremovable = is_take_item_blocked_by_item_rule(target_item, target_species)
    # Sticky Hold (abilities.ts:4549) blocks Knock Off from removing the
    # item AND from getting the 1.5x BP boost. Pokepy used to ignore the
    # ability entirely. Suppressed by Mold Breaker / Teravolt / Turboblaze
    # on the attacker.
    ABILITY_STICKY_HOLD = 60
    _MB_SET_KO = (104, 163, 164)  # moldbreaker, turboblaze, teravolt
    user_ab_ko = -1
    if user_offset is not None:
        user_ab_ko = int(battle[int(user_offset) + 5])
    has_sticky_hold = (target_ability == ABILITY_STICKY_HOLD) and (
        user_ab_ko not in _MB_SET_KO
    )
    should_remove = (
        (target_hp > 0)
        and (target_item > 0)
        and (not is_unremovable)
        and (not has_sticky_hold)
    )
    if not should_remove:
        return

    battle[target_offset + 6] = 0

    # Clear Choice Lock when item is knocked off
    target_is_side0 = target_offset < OFF_SIDE1
    lock_offset = OFF_FIELD + (F_CHOICE_LOCK_0 if target_is_side0 else F_CHOICE_LOCK_1)
    battle[lock_offset] = -1


def apply_trick_from_move(
    battle: np.ndarray,
    move_id: int,
    user_offset: int,
    target_offset: int,
    hit: bool,
    game_data=None,
    move_effects=None,
) -> bool:
    """Port of _apply_trick_from_move (line ~11102).

    Swaps held items between user and target if Trick/Switcheroo hit and
    the target is alive. Also clears Choice lock for both sides.

    Showdown gates:
      - onTryImmunity: fails if target has Sticky Hold (abilities.ts).
      - onHit: fails if either item is untakeable (mega stones on the
        correct holder, Arceus plates on Arceus, Genesect drives on
        Genesect, RKS memories on Silvally, Booster Energy, Z-crystals).
    """
    move_id = int(move_id)
    user_offset = int(user_offset)
    target_offset = int(target_offset)
    hit = bool(hit)

    is_trick = (move_id == MOVE_TRICK) or (move_id == MOVE_SWITCHEROO)
    if not is_trick or not hit:
        return False

    target_hp = int(battle[target_offset + 1])
    if target_hp <= 0:
        return False

    # Sticky Hold on target blocks Trick entirely. Showdown data/moves.ts
    # trick.onTryImmunity calls `target.hasAbility('stickyhold')`, which
    # uses pokemon.ts:hasAbility — that path is NOT bypassed by Mold
    # Breaker / Teravolt / Turboblaze (those work via the on-target
    # ability event ignore, not via direct hasAbility queries from a move's
    # own onTryImmunity). Verified by Showdown test mods/gen8 trick tests.
    ABILITY_STICKY_HOLD_TR = 60
    if int(battle[target_offset + 5]) == ABILITY_STICKY_HOLD_TR:
        return False

    user_item = int(battle[user_offset + 6])
    target_item = int(battle[target_offset + 6])

    user_species = int(battle[user_offset + 0])
    target_species = int(battle[target_offset + 0])
    if is_take_item_blocked_by_item_rule(user_item, user_species):
        return False
    if is_take_item_blocked_by_item_rule(target_item, target_species):
        return False

    battle[user_offset + 6] = target_item
    battle[target_offset + 6] = user_item

    # Clear Choice Lock for both sides
    user_is_side0 = user_offset < OFF_SIDE1
    user_lock_off = OFF_FIELD + (F_CHOICE_LOCK_0 if user_is_side0 else F_CHOICE_LOCK_1)
    target_lock_off = OFF_FIELD + (
        F_CHOICE_LOCK_1 if user_is_side0 else F_CHOICE_LOCK_0
    )
    battle[user_lock_off] = -1
    battle[target_lock_off] = -1
    return True


def apply_rapid_spin_from_move(
    battle: np.ndarray,
    move_id: int,
    user_offset: int,
    user_side: int,
    hit: bool,
    source_alive: bool | None = None,
    move_effects=None,
) -> None:
    """Port of _apply_rapid_spin_from_move (line ~11127).

    Clears all hazards and Leech Seed on the user's side, and gives the
    user +1 Speed (Gen 8+ effect).
    """
    if move_effects is None:
        return
    move_id = int(move_id)
    user_side = int(user_side)
    hit = bool(hit)

    move_effect = int(move_effects.effect_type[move_id])
    is_rapid_spin = move_effect == EFFECT_RAPID_SPIN
    if not is_rapid_spin or not hit:
        return
    if source_alive is None:
        source_alive = int(battle[int(user_offset) + 1]) > 0
    if not bool(source_alive):
        return

    hazard_offset = OFF_FIELD + (F_HAZARDS_0 if user_side == 0 else F_HAZARDS_1)
    battle[hazard_offset] = 0

    leech_seed_offset = OFF_FIELD + (
        F_LEECH_SEED_0 if user_side == 0 else F_LEECH_SEED_1
    )
    battle[leech_seed_offset] = 0

    # Rapid Spin / Mortal Spin also clear the user's `partiallytrapped`
    # volatile (Wrap, Bind, Fire Spin, etc.). Showdown data/moves.ts:
    # rapidspin onAfterHit removes 'partiallytrapped' from the user.
    from pokepy.core.constants import (
        F_EXTENDED_VOLATILE_0 as _FEV0,
        F_EXTENDED_VOLATILE_1 as _FEV1,
        OFF_MOVES as _OFF_MOVES_RS,
        M_PARTIAL_TRAP_TURNS_0 as _MPT0_RS,
        M_PARTIAL_TRAP_TURNS_1 as _MPT1_RS,
        EXT_VOL_PARTIAL_TRAP as _PT,
    )

    ext_off = OFF_FIELD + (_FEV0 if user_side == 0 else _FEV1)
    turns_off = _OFF_MOVES_RS + (_MPT0_RS if user_side == 0 else _MPT1_RS)
    cur_ext = int(battle[ext_off]) & 0xFFFF
    if (cur_ext & _PT) != 0:
        new_ext = cur_ext & ~_PT
        battle[ext_off] = _to_int16(new_ext)
        battle[turns_off] = 0

    # Rapid Spin's +1 Speed already comes through the generic self-stat
    # change path (`secondary.self.boosts`). Reapplying it here doubles the
    # boost and corrupts move order parity after repeated uses.


def apply_defog_from_move(
    battle: np.ndarray,
    move_id: int,
    hit: bool,
    move_effects=None,
    user_side: int = 0,
) -> None:
    """Port of _apply_defog_from_move (line ~11180).

    Showdown moves.ts:defog onHit:
      - Clears target's side conditions: reflect, lightscreen, auroraveil,
        safeguard, mist, AND all hazards (spikes, stealthrock, toxicspikes,
        stickyweb).
      - Clears SOURCE's side HAZARDS ONLY (not screens).
      - Clears terrain via `this.field.clearTerrain()`.
      - Drops target evasion by 1.
    """
    if move_effects is None:
        return
    move_id = int(move_id)
    hit = bool(hit)
    user_side = int(user_side)

    move_effect = int(move_effects.effect_type[move_id])
    is_defog = move_effect == EFFECT_DEFOG
    if not is_defog or not hit:
        return

    # Clear hazards on both sides.
    battle[OFF_FIELD + F_HAZARDS_0] = 0
    battle[OFF_FIELD + F_HAZARDS_1] = 0

    # Clear opponent's (target's) screens — Reflect / Light Screen / Aurora
    # Veil / Safeguard / Mist. User's own screens are preserved. Tailwind
    # and Lucky Chant are NOT listed in the Showdown removal list, so they
    # should survive. Pokepy's F_SCREENS_x packs all screens into one word,
    # so we selectively clear the five Defog-removable bitfields.
    from pokepy.core.constants import (
        SCREEN_REFLECT_SHIFT,
        SCREEN_LIGHTSCREEN_SHIFT,
        SCREEN_AURORAVEIL_SHIFT,
        SCREEN_SAFEGUARD_SHIFT,
        SCREEN_MIST_SHIFT,
        SCREEN_MASK_2BIT,
        SCREEN_MASK_3BIT,
    )

    opp_screens_offset = OFF_FIELD + (F_SCREENS_1 if user_side == 0 else F_SCREENS_0)
    screens = int(battle[opp_screens_offset]) & 0xFFFF
    screens &= ~(SCREEN_MASK_3BIT << SCREEN_REFLECT_SHIFT)
    screens &= ~(SCREEN_MASK_3BIT << SCREEN_LIGHTSCREEN_SHIFT)
    screens &= ~(SCREEN_MASK_3BIT << SCREEN_AURORAVEIL_SHIFT)
    screens &= ~(SCREEN_MASK_2BIT << SCREEN_SAFEGUARD_SHIFT)
    screens &= ~(SCREEN_MASK_2BIT << SCREEN_MIST_SHIFT)
    battle[opp_screens_offset] = _to_int16(screens)

    # Clear terrain (data/moves.ts:defog calls `this.field.clearTerrain()`).
    from pokepy.core.constants import F_TERRAIN, M_TERRAIN_TURNS

    battle[OFF_FIELD + F_TERRAIN] = 0
    battle[OFF_META + M_TERRAIN_TURNS] = 0

    # Drop target's evasion by 1 stage. Showdown moves.ts:3580 uses
    # `this.boost({ evasion: -1 })` which goes through the full boost chain —
    # blocked by Clear Body / White Smoke / Full Metal Body / Clear Amulet,
    # reversed by Contrary, reflected by Mirror Armor. Pokepy previously
    # bypassed all of these with a raw bitpack write.
    target_side = 1 if user_side == 0 else 0
    target_active = int(
        battle[OFF_META + (M_ACTIVE0 if target_side == 0 else M_ACTIVE1)]
    )
    target_base = OFF_SIDE0 if target_side == 0 else OFF_SIDE1
    target_off = target_base + target_active * POKEMON_SIZE
    target_ability_df = int(battle[target_off + 5])
    target_item_df = int(battle[target_off + 6])
    from pokepy.core.constants import (
        ABILITY_CLEAR_BODY as _ACB_DF,
        ABILITY_WHITE_SMOKE as _AWS_DF,
        ABILITY_FULL_METAL_BODY as _AFMB_DF,
        ABILITY_MIRROR_ARMOR as _AMA_DF,
        ABILITY_CONTRARY as _ACR_DF,
    )

    # Showdown items.ts:1064 clearamulet, num/spritenum 747. Pokepy uses
    # spritenum as its item id (verified vs other ITEM_CLEAR_AMULET defs).
    _ITEM_CLEAR_AMULET_DF = 747
    has_clear_body_df = target_ability_df in (_ACB_DF, _AWS_DF, _AFMB_DF)
    has_mirror_armor_df = target_ability_df == _AMA_DF
    has_contrary_df = target_ability_df == _ACR_DF
    has_clear_amulet_df = target_item_df == _ITEM_CLEAR_AMULET_DF

    if has_clear_body_df or has_clear_amulet_df:
        return
    if has_mirror_armor_df:
        # Reflect the evasion drop back to the user.
        user_active = int(
            battle[OFF_META + (M_ACTIVE0 if user_side == 0 else M_ACTIVE1)]
        )
        user_base = OFF_SIDE0 if user_side == 0 else OFF_SIDE1
        user_off = user_base + user_active * POKEMON_SIZE
        user_boosts14 = int(battle[user_off + 14])
        battle[user_off + 14] = apply_boost_to_packed(user_boosts14, 8, -1)
        return
    target_boosts14 = int(battle[target_off + 14])
    delta_df = +1 if has_contrary_df else -1
    battle[target_off + 14] = apply_boost_to_packed(target_boosts14, 8, delta_df)


def apply_court_change_from_move(
    battle: np.ndarray,
    move_id: int,
    move_executed: bool,
) -> None:
    """Court Change (Cinderace signature) — swap hazards, screens, and
    Tailwind between the two sides.

    Showdown source: data/moves.ts:courtchange `onHitField` swaps the
    `sideConditions` for both sides. Pokepy stores hazards in F_HAZARDS_*
    and screens (incl. Tailwind) in F_SCREENS_*. Swap both fields.
    """
    _MOVE_COURT_CHANGE = 756
    if int(move_id) != _MOVE_COURT_CHANGE or not move_executed:
        return
    h0 = int(battle[OFF_FIELD + F_HAZARDS_0])
    h1 = int(battle[OFF_FIELD + F_HAZARDS_1])
    battle[OFF_FIELD + F_HAZARDS_0] = h1
    battle[OFF_FIELD + F_HAZARDS_1] = h0
    from pokepy.core.constants import F_SCREENS_0 as _FS0, F_SCREENS_1 as _FS1

    s0 = int(battle[OFF_FIELD + _FS0])
    s1 = int(battle[OFF_FIELD + _FS1])
    battle[OFF_FIELD + _FS0] = s1
    battle[OFF_FIELD + _FS1] = s0


def apply_haze_from_move(
    battle: np.ndarray,
    move_id: int,
    hit: bool,
    move_effects=None,
) -> None:
    """Port of _apply_haze_from_move (line ~11226).

    Resets all stat changes on both active Pokemon. The packed neutral
    boosts value is 0x6666 (4 stats at neutral=6).
    """
    if move_effects is None:
        return
    move_id = int(move_id)
    hit = bool(hit)

    move_effect = int(move_effects.effect_type[move_id])
    is_haze = move_effect == EFFECT_HAZE
    if not is_haze or not hit:
        return

    active0 = int(battle[OFF_META + M_ACTIVE0])
    active1 = int(battle[OFF_META + M_ACTIVE1])
    p0_off = OFF_SIDE0 + active0 * POKEMON_SIZE
    p1_off = OFF_SIDE1 + active1 * POKEMON_SIZE

    neutral = _to_int16(0x6666)
    battle[p0_off + 13] = neutral
    battle[p1_off + 13] = neutral
    # Word 14 layout: spe(4) | acc(4)<<4 | eva(4)<<8 | tera_type(4)<<12.
    # Haze resets the boost nibbles (spe/acc/eva) but must preserve the
    # tera_type nibble in the top 4 bits.
    for off in (p0_off, p1_off):
        old14 = int(battle[off + 14]) & 0xFFFF
        tera_nibble = old14 & 0xF000
        battle[off + 14] = _to_int16(tera_nibble | 0x0666)


def apply_clear_smog_from_move(
    battle: np.ndarray,
    move_id: int,
    target_offset: int,
    hit: bool,
    move_effects=None,
) -> None:
    """Port of _apply_clear_smog_from_move (line ~11277).

    Resets the target's stat boosts to neutral (the damage portion is
    handled by the regular damage calculation).
    """
    if move_effects is None:
        return
    move_id = int(move_id)
    target_offset = int(target_offset)
    hit = bool(hit)

    move_effect = int(move_effects.effect_type[move_id])
    is_clear_smog = move_effect == EFFECT_CLEAR_SMOG
    if not is_clear_smog or not hit:
        return

    neutral = _to_int16(0x6666)
    battle[target_offset + 13] = neutral
    # Preserve tera_type nibble in the top 4 bits of word 14.
    old14 = int(battle[target_offset + 14]) & 0xFFFF
    tera_nibble = old14 & 0xF000
    battle[target_offset + 14] = _to_int16(tera_nibble | 0x0666)


def apply_psych_up_from_move(
    battle: np.ndarray,
    move_id: int,
    user_offset: int,
    target_offset: int,
    hit: bool,
    move_effects=None,
) -> None:
    """Port of _apply_psych_up_from_move (line ~11315).

    Copies the target's stat changes to the user.
    """
    if move_effects is None:
        return
    move_id = int(move_id)
    user_offset = int(user_offset)
    target_offset = int(target_offset)
    hit = bool(hit)

    move_effect = int(move_effects.effect_type[move_id])
    is_psych_up = move_effect == EFFECT_PSYCH_UP
    if not is_psych_up or not hit:
        return

    battle[user_offset + 13] = int(battle[target_offset + 13])
    # Copy spe/acc/eva boost nibbles from target but preserve user's
    # tera_type nibble in the top 4 bits of word 14.
    user_old14 = int(battle[user_offset + 14]) & 0xFFFF
    target14 = int(battle[target_offset + 14]) & 0xFFFF
    user_tera = user_old14 & 0xF000
    target_boosts = target14 & 0x0FFF
    battle[user_offset + 14] = _to_int16(user_tera | target_boosts)


def _set_screen_turns(
    screens: int, shift: int, turns: int, should_set: bool, mask: int
) -> int:
    """Set a single screen's turn counter if (a) the slot is empty and (b)
    `should_set` is True. Used by `apply_screen_from_move`. Mirrors the
    inline `set_screen_turns` helper at lines 11386-11393.
    """
    current_val = (screens >> shift) & mask
    can_set = (current_val == 0) and bool(should_set)
    if not can_set:
        return screens
    clear_mask = screen_clear_mask(mask, shift)
    cleared = screens & int(clear_mask)
    clamped_turns = min(int(turns), int(mask))
    new_screens = cleared | (clamped_turns << shift)
    return new_screens


def apply_screen_from_move(
    battle: np.ndarray,
    move_id: int,
    side: int,
    hit: bool,
    move_effects=None,
) -> None:
    """Port of _apply_screen_from_move (line ~11353).

    Handles Reflect / Light Screen / Aurora Veil / Tailwind / Safeguard /
    Mist / Lucky Chant. Aurora Veil only sets in snow/hail. Reflect /
    Light Screen / Aurora Veil last 5 turns (8 with Light Clay, capped to
    7 by the 3-bit field). Tailwind = 4 turns; the 2-bit screens default
    to 3 turns.
    """
    move_id = int(move_id)
    side = int(side)
    hit = bool(hit)

    screen_offset = OFF_FIELD + (F_SCREENS_0 if side == 0 else F_SCREENS_1)
    current_screens = int(battle[screen_offset]) & 0xFFFF
    weather = int(battle[OFF_FIELD + F_WEATHER])

    is_reflect = move_id == MOVE_REFLECT
    is_light_screen = move_id == MOVE_LIGHT_SCREEN
    is_aurora_veil = move_id == MOVE_AURORA_VEIL
    is_safeguard = move_id == MOVE_SAFEGUARD
    is_mist = move_id == MOVE_MIST
    is_lucky_chant = move_id == MOVE_LUCKY_CHANT
    is_tailwind = move_id == MOVE_TAILWIND

    # Aurora Veil only works in snow / hail
    snow_active = (weather == WEATHER_SNOW) or (weather == 5)  # 5 = legacy hail
    aurora_veil_can_set = is_aurora_veil and snow_active

    # Light Clay extends screens to 8 turns (capped to 7 in 3-bit field).
    user_active = int(battle[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])
    user_base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    user_item = int(battle[user_base + user_active * POKEMON_SIZE + 6])
    has_light_clay = user_item == ITEM_LIGHT_CLAY
    screen_turns = 7 if has_light_clay else 5

    new_screens = current_screens
    new_screens = _set_screen_turns(
        new_screens,
        SCREEN_REFLECT_SHIFT,
        screen_turns,
        hit and is_reflect,
        SCREEN_MASK_3BIT,
    )
    new_screens = _set_screen_turns(
        new_screens,
        SCREEN_LIGHTSCREEN_SHIFT,
        screen_turns,
        hit and is_light_screen,
        SCREEN_MASK_3BIT,
    )
    new_screens = _set_screen_turns(
        new_screens,
        SCREEN_AURORAVEIL_SHIFT,
        screen_turns,
        hit and aurora_veil_can_set,
        SCREEN_MASK_3BIT,
    )
    new_screens = _set_screen_turns(
        new_screens, SCREEN_TAILWIND_SHIFT, 4, hit and is_tailwind, SCREEN_MASK_3BIT
    )
    new_screens = _set_screen_turns(
        new_screens, SCREEN_SAFEGUARD_SHIFT, 3, hit and is_safeguard, SCREEN_MASK_2BIT
    )
    new_screens = _set_screen_turns(
        new_screens, SCREEN_MIST_SHIFT, 3, hit and is_mist, SCREEN_MASK_2BIT
    )
    new_screens = _set_screen_turns(
        new_screens,
        SCREEN_LUCKYCHANT_SHIFT,
        3,
        hit and is_lucky_chant,
        SCREEN_MASK_2BIT,
    )

    battle[screen_offset] = _to_int16(new_screens)
