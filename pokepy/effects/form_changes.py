"""Form-change abilities — Zen Mode, Stance Change, Schooling, Slow Start.

Port of multi_format_fast_env.py lines 4085-4274. These run after damage
calc each turn and rewrite the active pokemon's stats / types when an
ability triggers a form change.

Disguise / Ice Face are NOT here — they're handled in the damage code via
the flag bit 0x40 (already ported).
"""

from __future__ import annotations

import numpy as np

from pokepy.mechanics.stats import calc_stat_modern
from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_FIELD,
    OFF_META,
    POKEMON_SIZE,
    M_ACTIVE0B,
    M_ACTIVE1B,
    F_EXTENDED_VOLATILE_0,
    F_EXTENDED_VOLATILE_1,
    F_LAST_MOVE_0,
    F_LAST_MOVE_1,
    EXT_VOL_SCHOOLING_SOLO,
    TYPE_FIRE,
    TYPE_PSYCHIC,
    ABILITY_SHIELDS_DOWN,
    ABILITY_GULP_MISSILE,
    MOVE_SURF,
    MOVE_DIVE,
    MOVE_KINGS_SHIELD,
)

# Local ability constants
ABILITY_ZEN_MODE = 161
ABILITY_STANCE_CHANGE = 176
ABILITY_SCHOOLING = 208
ABILITY_SLOW_START = 112
_MINIOR_SPECIES = 774
_MINIOR_METEOR_BASE_STATS = (60, 60, 100, 60, 100, 60)
_SPECIES_CRAMORANT = 845
GULP_MISSILE_NONE = 0
GULP_MISSILE_GULPING = 1
GULP_MISSILE_GORGING = 2


def _to_int16(val: int) -> int:
    val = int(val) & 0xFFFF
    if val >= 0x8000:
        val -= 0x10000
    return val


def _gulp_missile_meta_offset(pokemon_offset: int) -> int:
    poff = int(pokemon_offset)
    return OFF_META + (M_ACTIVE0B if poff < OFF_SIDE1 else M_ACTIVE1B)


def get_gulp_missile_state(
    battle: np.ndarray,
    pokemon_offset: int,
) -> int:
    return int(battle[_gulp_missile_meta_offset(pokemon_offset)])


def clear_gulp_missile_state(
    battle: np.ndarray,
    pokemon_offset: int,
) -> None:
    battle[_gulp_missile_meta_offset(pokemon_offset)] = 0


def prime_gulp_missile_state_from_move(
    battle: np.ndarray,
    pokemon_offset: int,
    move_id: int,
    *,
    move_executed: bool,
    is_charge_turn: bool,
) -> None:
    """Track Cramorant's loaded Gulp Missile state in singles.

    Showdown models this as a forme change to Cramorant-Gulping/Gorging when
    Surf resolves or Dive starts charging. Pokepy's extracted species tables do
    not include those temporary forms, so store the same state in the otherwise
    unused singles `M_ACTIVE0B/M_ACTIVE1B` meta slots instead.
    """
    poff = int(pokemon_offset)
    if not bool(move_executed):
        return
    if int(battle[poff + 1]) <= 0:
        return
    if int(battle[poff + 5]) != ABILITY_GULP_MISSILE:
        return
    if int(battle[poff + 0]) != _SPECIES_CRAMORANT:
        return
    if get_gulp_missile_state(battle, poff) != GULP_MISSILE_NONE:
        return

    move_id = int(move_id)
    if move_id != MOVE_SURF and not (move_id == MOVE_DIVE and bool(is_charge_turn)):
        return

    hp = int(battle[poff + 1])
    max_hp = int(battle[poff + 2])
    state = GULP_MISSILE_GORGING if hp * 2 <= max_hp else GULP_MISSILE_GULPING
    battle[_gulp_missile_meta_offset(poff)] = np.int16(state)


def apply_shields_down_form_state(
    battle: np.ndarray,
    pokemon_offset: int,
    state,
    game_data,
) -> None:
    """Recompute Minior's live raw stats for its current Shields Down form."""
    if state is None:
        return

    poff = int(pokemon_offset)
    if int(battle[poff + 1]) <= 0 or int(battle[poff + 5]) != ABILITY_SHIELDS_DOWN:
        return

    if poff < OFF_SIDE1:
        slot = (poff - OFF_SIDE0) // POKEMON_SIZE
        species_id = int(state.team_species[slot])
        evs = state.team_evs_full[slot]
        ivs = state.team_ivs_full[slot]
        nature_mods = state.team_nature_mods[slot]
    else:
        slot = (poff - OFF_SIDE1) // POKEMON_SIZE
        species_id = int(state.opp_species[slot])
        evs = state.opp_evs_full[slot]
        ivs = state.opp_ivs_full[slot]
        nature_mods = state.opp_nature_mods[slot]

    if species_id != _MINIOR_SPECIES:
        return

    hp = int(battle[poff + 1])
    max_hp = int(battle[poff + 2])
    level = int(battle[poff + 3])
    target_base_stats = (
        _MINIOR_METEOR_BASE_STATS
        if hp * 2 > max_hp
        else tuple(int(x) for x in game_data.species_base_stats[species_id])
    )

    for stat_idx, battle_off in enumerate(range(7, 12), start=1):
        battle[poff + battle_off] = np.int16(
            calc_stat_modern(
                int(target_base_stats[stat_idx]),
                level,
                int(ivs[stat_idx]),
                int(evs[stat_idx]),
                False,
                float(nature_mods[stat_idx]),
            )
        )


def apply_stance_change_pre_move(
    battle: np.ndarray,
    p0_off: int,
    p1_off: int,
    move_id0: int,
    move_id1: int,
    is_switch0: bool,
    is_switch1: bool,
    game_data,
) -> None:
    """Aegislash stance change — MUST run before damage calc.

    Showdown source: data/abilities.ts:stancechange onModifyMovePriority=1
    (line 4434). The form change happens DURING move use, so the updated
    stats are used for the damage calc of this very move. Pokepy used to
    run stance change after damage, leaking Shield-form stats into the
    first attack each switch-in.
    """
    p0_off = int(p0_off)
    p1_off = int(p1_off)
    for poff, mid, is_sw in [
        (p0_off, int(move_id0), bool(is_switch0)),
        (p1_off, int(move_id1), bool(is_switch1)),
    ]:
        ab = int(battle[poff + 5])
        if ab != ABILITY_STANCE_CHANGE or is_sw:
            continue
        sc_bp = int(np.asarray(game_data.move_base_power)[mid])
        is_attack = sc_bp > 0
        is_kings = mid == MOVE_KINGS_SHIELD
        atk = int(battle[poff + 7])
        def_s = int(battle[poff + 8])
        spa = int(battle[poff + 9])
        spd = int(battle[poff + 10])
        blade_trig = is_attack and def_s > atk
        shield_trig = is_kings and atk > def_s
        if blade_trig or shield_trig:
            battle[poff + 7] = np.int16(def_s)
            battle[poff + 8] = np.int16(atk)
            battle[poff + 9] = np.int16(spd)
            battle[poff + 10] = np.int16(spa)


def apply_form_changes(
    battle: np.ndarray,
    p0_off: int,
    p1_off: int,
    move_id0: int,
    move_id1: int,
    is_switch0: bool,
    is_switch1: bool,
    game_data,
) -> None:
    """All form-change abilities that fire after damage application."""
    p0_off = int(p0_off)
    p1_off = int(p1_off)
    move_bp0 = int(np.asarray(game_data.move_base_power)[int(move_id0)])
    move_bp1 = int(np.asarray(game_data.move_base_power)[int(move_id1)])

    # ------ Zen Mode (Darmanitan): HP < 50% → Fire/Psychic + stat shuffle ------
    for poff in (p0_off, p1_off):
        ab = int(battle[poff + 5])
        if ab != ABILITY_ZEN_MODE:
            continue
        hp = int(battle[poff + 1])
        max_hp = int(battle[poff + 2])
        if hp <= 0 or hp * 2 >= max_hp:
            continue
        cur_types = int(battle[poff + 4])
        already_zen = ((cur_types >> 8) & 0xFF) == TYPE_PSYCHIC
        if already_zen:
            continue
        # Type → Fire / Psychic
        new_types = (TYPE_PSYCHIC << 8) | TYPE_FIRE
        battle[poff + 4] = _to_int16(new_types)
        # Stat scaling: Atk 140→30, Def 55→105, SpA 30→140, SpD 55→105, Spe 95→55
        atk = int(battle[poff + 7])
        def_s = int(battle[poff + 8])
        spa = int(battle[poff + 9])
        spd = int(battle[poff + 10])
        spe = int(battle[poff + 11])
        battle[poff + 7] = np.int16(atk * 30 // 140)
        battle[poff + 8] = np.int16(def_s * 105 // 55)
        battle[poff + 9] = np.int16(spa * 140 // 30)
        battle[poff + 10] = np.int16(spd * 105 // 55)
        battle[poff + 11] = np.int16(spe * 55 // 95)

    # Stance Change is now handled by apply_stance_change_pre_move which
    # fires BEFORE damage calc (matching Showdown onModifyMovePriority 1).
    # Keeping the post-damage call as a no-op to avoid double-swapping.

    # ------ Schooling (Wishiwashi): HP <= 25% → weak Solo form ------
    # Showdown abilities.ts:schooling onResidualOrder=29 does a formeChange
    # between Wishiwashi (solo) and Wishiwashi-School based on HP > 25%.
    # Pokepy stores stats directly in slots 7-11; there is no "base stat"
    # rollback channel. Divide current atk/spa by 3 ONCE per transition so
    # the stat mutation doesn't compound every turn the mon stays below
    # 25% HP. The transition marker lives on the active side's extended-
    # volatile field, sharing a bit with Libero/Protean's "used since switch"
    # marker; the abilities are mutually exclusive, so that bit is safe here.
    for poff in (p0_off, p1_off):
        ab = int(battle[poff + 5])
        if ab != ABILITY_SCHOOLING:
            continue
        hp = int(battle[poff + 1])
        max_hp = int(battle[poff + 2])
        if hp <= 0:
            continue
        ev_off = OFF_FIELD + (
            F_EXTENDED_VOLATILE_0 if poff < OFF_SIDE1 else F_EXTENDED_VOLATILE_1
        )
        ext_vol = int(battle[ev_off]) & 0xFFFF
        in_solo = (ext_vol & EXT_VOL_SCHOOLING_SOLO) != 0
        is_below = hp * 4 <= max_hp
        if is_below and not in_solo:
            atk = int(battle[poff + 7])
            spa = int(battle[poff + 9])
            battle[poff + 7] = np.int16(max(1, atk // 3))
            battle[poff + 9] = np.int16(max(1, spa // 3))
            new_ext = ext_vol | EXT_VOL_SCHOOLING_SOLO
            battle[ev_off] = np.int16(_to_int16(new_ext))
        elif (not is_below) and in_solo:
            atk = int(battle[poff + 7])
            spa = int(battle[poff + 9])
            battle[poff + 7] = np.int16(atk * 3)
            battle[poff + 9] = np.int16(spa * 3)
            new_ext = ext_vol & ~EXT_VOL_SCHOOLING_SOLO
            battle[ev_off] = np.int16(_to_int16(new_ext))

    # ------ Slow Start (Regigigas): halve Atk/Spe just after switch-in ------
    for poff in (p0_off, p1_off):
        ab = int(battle[poff + 5])
        if ab != ABILITY_SLOW_START:
            continue
        side = 0 if poff < OFF_SIDE1 else 1
        last = int(battle[OFF_FIELD + (F_LAST_MOVE_0 if side == 0 else F_LAST_MOVE_1)])
        if last < 0:  # just switched in
            atk = int(battle[poff + 7])
            spe = int(battle[poff + 11])
            battle[poff + 7] = np.int16(atk // 2)
            battle[poff + 11] = np.int16(spe // 2)
