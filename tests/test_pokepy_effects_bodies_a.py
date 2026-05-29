"""Smoke tests for the ported status / stat-change effect bodies.

Verifies the real ports of:
- pokepy.effects.status_apply.apply_status_from_move
- pokepy.effects.stat_changes.apply_stat_changes_from_move
- pokepy.effects.status_apply.apply_end_of_turn_status
"""

from __future__ import annotations

import numpy as np
import pytest

from pokepy.core.state import MultiFormatState
from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_FIELD,
    OFF_META,
    POKEMON_SIZE,
    M_ACTIVE0,
    M_ACTIVE1,
    F_TERRAIN,
    PHASE_BATTLE,
    NEUTRAL_BOOSTS_13,
    NEUTRAL_BOOSTS_14,
    STATUS_NONE,
    STATUS_PARALYSIS,
    STATUS_BURN,
    TYPE_NORMAL,
)
from pokepy.core.bitpack import (
    extract_boost,
    get_status,
    get_status_turns,
    set_status,
)
from pokepy.data.loader import load_game_data, load_move_effect_data
from pokepy.utils.gen5_prng import Gen5PRNG

from pokepy.effects.status_apply import (
    apply_status_from_move,
    apply_end_of_turn_status,
)
from pokepy.effects.stat_changes import apply_stat_changes_from_move


def _hand_state():
    """Build a minimal battle buffer with two normal-type Pokemon (one per side)."""
    gd = load_game_data()
    me = load_move_effect_data()
    state = MultiFormatState.create_empty(format_id=1)
    state.phase = np.int8(PHASE_BATTLE)

    bs = state.battle_state
    bs[OFF_META + M_ACTIVE0] = 0
    bs[OFF_META + M_ACTIVE1] = 0
    bs[OFF_FIELD + F_TERRAIN] = 0  # no terrain

    for side, base in [(0, OFF_SIDE0), (1, OFF_SIDE1)]:
        for slot in range(2):
            poff = base + slot * POKEMON_SIZE
            bs[poff + 0] = 1 + slot  # species_id (anything nonzero)
            bs[poff + 1] = 100  # current_hp
            bs[poff + 2] = 100  # max_hp
            bs[poff + 3] = 100  # level
            # type1=normal, type2=normal — explicitly NOT electric/ground/grass
            # so no immunity blocks our test moves
            bs[poff + 4] = (TYPE_NORMAL & 0xFF) | ((TYPE_NORMAL & 0xFF) << 8)
            bs[poff + 5] = 0  # ability (nothing relevant)
            bs[poff + 6] = 0  # item
            bs[poff + 7] = 80  # atk
            bs[poff + 8] = 80  # def
            bs[poff + 9] = 80  # spa
            bs[poff + 10] = 80  # spd
            bs[poff + 11] = 80  # spe
            bs[poff + 12] = set_status(STATUS_NONE, 0)
            bs[poff + 13] = NEUTRAL_BOOSTS_13
            bs[poff + 14] = NEUTRAL_BOOSTS_14
            bs[poff + 15] = 0  # flags
    return state, gd, me


def test_apply_status_thunder_wave_paralyzes_target():
    state, gd, me = _hand_state()
    bs = state.battle_state
    user_offset = OFF_SIDE0 + 0 * POKEMON_SIZE
    target_offset = OFF_SIDE1 + 0 * POKEMON_SIZE

    # Sanity: target starts unstatused
    assert get_status(int(bs[target_offset + 12])) == STATUS_NONE

    prng = Gen5PRNG()
    apply_status_from_move(
        battle=bs,
        move_id=86,  # Thunder Wave (status=2 paralysis, status_chance=100)
        target_offset=target_offset,
        hit=True,
        game_data=gd,
        move_effects=me,
        gen5_prng=prng,
        user_offset=user_offset,
    )

    new_status = get_status(int(bs[target_offset + 12]))
    assert new_status == STATUS_PARALYSIS, f"expected paralysis (2), got {new_status}"

    # User should remain unstatused (no synchronize)
    assert get_status(int(bs[user_offset + 12])) == STATUS_NONE


def test_apply_status_no_overwrite_existing():
    state, gd, me = _hand_state()
    bs = state.battle_state
    target_offset = OFF_SIDE1 + 0 * POKEMON_SIZE
    # Pre-burn the target
    bs[target_offset + 12] = set_status(STATUS_BURN, 0)

    prng = Gen5PRNG()
    apply_status_from_move(
        battle=bs,
        move_id=86,  # Thunder Wave
        target_offset=target_offset,
        hit=True,
        game_data=gd,
        move_effects=me,
        gen5_prng=prng,
        user_offset=OFF_SIDE0 + 0 * POKEMON_SIZE,
    )
    # Status should still be burn — Thunder Wave can't overwrite
    assert get_status(int(bs[target_offset + 12])) == STATUS_BURN


def test_apply_stat_changes_growl_lowers_target_atk():
    state, gd, me = _hand_state()
    bs = state.battle_state
    user_offset = OFF_SIDE0 + 0 * POKEMON_SIZE
    target_offset = OFF_SIDE1 + 0 * POKEMON_SIZE

    before = int(bs[target_offset + 13])
    assert extract_boost(before, 0) == 0  # neutral atk

    prng = Gen5PRNG()
    apply_stat_changes_from_move(
        battle=bs,
        move_id=45,  # Growl: stat_target=1, stat_changes=[-1,0,...], stat_chance=100
        user_offset=user_offset,
        target_offset=target_offset,
        hit=True,
        game_data=gd,
        move_effects=me,
        gen5_prng=prng,
    )

    after = int(bs[target_offset + 13])
    assert after != before, "growl should have changed packed boost"
    assert (
        extract_boost(after, 0) == -1
    ), f"expected target atk -1, got {extract_boost(after, 0)}"
    # User unaffected
    assert int(bs[user_offset + 13]) == NEUTRAL_BOOSTS_13


def test_end_of_turn_burn_damages_one_sixteenth():
    state, gd, me = _hand_state()
    bs = state.battle_state
    poff = OFF_SIDE0 + 0 * POKEMON_SIZE
    # Setup: burned, full HP
    bs[poff + 1] = 160
    bs[poff + 2] = 160
    bs[poff + 12] = set_status(STATUS_BURN, 0)

    prng = Gen5PRNG()
    apply_end_of_turn_status(
        battle=bs,
        pokemon_offset=poff,
        game_data=gd,
        move_effects=me,
        gen5_prng=prng,
    )

    # Burn = 1/16 max HP = 10
    assert int(bs[poff + 1]) == 150, f"expected 150 hp, got {int(bs[poff + 1])}"
    # Status preserved
    assert get_status(int(bs[poff + 12])) == STATUS_BURN
