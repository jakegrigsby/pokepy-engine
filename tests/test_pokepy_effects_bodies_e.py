"""Smoke tests for the ported volatile-status effect bodies.

Verifies the real ports of:
- pokepy.effects.volatiles.apply_leech_seed_from_move
- pokepy.effects.volatiles.apply_substitute_from_move
- pokepy.effects.volatiles.apply_perish_song_from_move
- pokepy.effects.volatiles.apply_destiny_bond_from_move
- pokepy.effects.volatiles.apply_confusion_from_move
- pokepy.effects.volatiles.apply_taunt_from_move
- pokepy.effects.volatiles.apply_encore_from_move
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
    F_LEECH_SEED_0,
    F_LEECH_SEED_1,
    F_SUBSTITUTE_0,
    F_SUBSTITUTE_1,
    F_PERISH_COUNT_0,
    F_PERISH_COUNT_1,
    F_DESTINY_BOND_0,
    F_DESTINY_BOND_1,
    F_VOLATILE_0,
    F_VOLATILE_1,
    F_LAST_MOVE_0,
    F_LAST_MOVE_1,
    PHASE_BATTLE,
    NEUTRAL_BOOSTS_13,
    NEUTRAL_BOOSTS_14,
    STATUS_NONE,
    TYPE_NORMAL,
)
from pokepy.core.bitpack import (
    set_status,
    get_confusion_turns,
    get_taunt_turns,
    get_encore_turns,
)
from pokepy.data.loader import load_game_data, load_move_effect_data
from pokepy.utils.gen5_prng import Gen5PRNG

from pokepy.effects.volatiles import (
    apply_leech_seed_from_move,
    apply_substitute_from_move,
    apply_perish_song_from_move,
    apply_destiny_bond_from_move,
    apply_confusion_from_move,
    apply_taunt_from_move,
    apply_encore_from_move,
)

def _hand_state():
    """Build a minimal battle buffer with two normal-type Pokemon (one per side)."""
    gd = load_game_data()
    me = load_move_effect_data()
    state = MultiFormatState.create_empty(format_id=1)
    state.phase = np.int8(PHASE_BATTLE)

    bs = state.battle_state
    bs[OFF_META + M_ACTIVE0] = 0
    bs[OFF_META + M_ACTIVE1] = 0

    for side, base in [(0, OFF_SIDE0), (1, OFF_SIDE1)]:
        for slot in range(2):
            poff = base + slot * POKEMON_SIZE
            bs[poff + 0] = 1 + slot  # species_id
            bs[poff + 1] = 100  # current_hp
            bs[poff + 2] = 100  # max_hp
            bs[poff + 3] = 100  # level
            # Normal/Normal — explicitly NOT grass/ghost so leech seed/curse aren't immune
            bs[poff + 4] = (TYPE_NORMAL & 0xFF) | ((TYPE_NORMAL & 0xFF) << 8)
            bs[poff + 5] = 0  # ability (no Magic Guard)
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

def test_apply_leech_seed_sets_target_flag():
    state, gd, me = _hand_state()
    bs = state.battle_state
    target_offset = OFF_SIDE1 + 0 * POKEMON_SIZE

    # Sanity: not seeded yet
    assert int(bs[OFF_FIELD + F_LEECH_SEED_1]) == 0

    apply_leech_seed_from_move(
        battle=bs,
        move_id=73,  # Leech Seed
        target_side=1,
        target_offset=target_offset,
        hit=True,
        game_data=gd,
        move_effects=me,
    )

    assert int(bs[OFF_FIELD + F_LEECH_SEED_1]) == 1, (
        "leech seed flag should be set on target side after Leech Seed hits"
    )
    # Side 0 not seeded
    assert int(bs[OFF_FIELD + F_LEECH_SEED_0]) == 0

def test_apply_substitute_pays_quarter_hp_and_sets_sub_field():
    state, gd, me = _hand_state()
    bs = state.battle_state
    user_offset = OFF_SIDE0 + 0 * POKEMON_SIZE

    starting_hp = int(bs[user_offset + 1])
    max_hp = int(bs[user_offset + 2])
    expected_cost = max(1, max_hp // 4)

    apply_substitute_from_move(
        battle=bs,
        move_id=164,  # Substitute
        user_side=0,
        user_offset=user_offset,
        game_data=gd,
        move_effects=me,
    )

    assert int(bs[OFF_FIELD + F_SUBSTITUTE_0]) == expected_cost, (
        f"substitute should equal max_hp//4 = {expected_cost}"
    )
    assert int(bs[user_offset + 1]) == starting_hp - expected_cost, (
        "user should have lost max_hp//4 HP"
    )

def test_apply_perish_song_sets_both_counters_to_4():
    state, gd, me = _hand_state()
    bs = state.battle_state

    assert int(bs[OFF_FIELD + F_PERISH_COUNT_0]) == 0
    assert int(bs[OFF_FIELD + F_PERISH_COUNT_1]) == 0

    apply_perish_song_from_move(
        battle=bs,
        move_id=195,  # Perish Song
        hit=True,
        move_effects=me,
    )

    assert int(bs[OFF_FIELD + F_PERISH_COUNT_0]) == 4
    assert int(bs[OFF_FIELD + F_PERISH_COUNT_1]) == 4

def test_apply_destiny_bond_sets_user_flag():
    state, gd, me = _hand_state()
    bs = state.battle_state

    assert int(bs[OFF_FIELD + F_DESTINY_BOND_0]) == 0

    apply_destiny_bond_from_move(
        battle=bs,
        move_id=194,  # Destiny Bond
        user_side=0,
        move_effects=me,
    )

    assert int(bs[OFF_FIELD + F_DESTINY_BOND_0]) == 1
    # Side 1 untouched
    assert int(bs[OFF_FIELD + F_DESTINY_BOND_1]) == 0

def test_apply_confusion_sets_volatile_turns_for_confuse_ray():
    state, gd, me = _hand_state()
    bs = state.battle_state

    # Sanity
    assert get_confusion_turns(int(bs[OFF_FIELD + F_VOLATILE_1])) == 0

    prng = Gen5PRNG()
    apply_confusion_from_move(
        battle=bs,
        move_id=109,  # Confuse Ray (volatile=VOLATILE_CONFUSION, chance=100)
        target_side=1,
        hit=True,
        game_data=gd,
        move_effects=me,
        gen5_prng=prng,
    )

    new_turns = get_confusion_turns(int(bs[OFF_FIELD + F_VOLATILE_1]))
    assert new_turns > 0, f"expected confusion turns > 0, got {new_turns}"
    assert 2 <= new_turns <= 5, (
        f"confusion lasts 2-5 turns, got {new_turns}"
    )

def test_apply_taunt_sets_taunt_turns():
    state, gd, me = _hand_state()
    bs = state.battle_state

    assert get_taunt_turns(int(bs[OFF_FIELD + F_VOLATILE_1])) == 0

    prng = Gen5PRNG()
    apply_taunt_from_move(
        battle=bs,
        move_id=269,  # Taunt
        target_side=1,
        hit=True,
        move_effects=me,
        gen5_prng=prng,
    )

    new_turns = get_taunt_turns(int(bs[OFF_FIELD + F_VOLATILE_1]))
    assert new_turns > 0, f"expected taunt turns > 0, got {new_turns}"
    assert new_turns == 3

def test_apply_encore_sets_encore_turns_when_target_has_last_move():
    state, gd, me = _hand_state()
    bs = state.battle_state

    # Encore requires the target to have a valid last move
    bs[OFF_FIELD + F_LAST_MOVE_1] = 0  # slot 0 (any 0..3 works)

    assert get_encore_turns(int(bs[OFF_FIELD + F_VOLATILE_1])) == 0

    prng = Gen5PRNG()
    apply_encore_from_move(
        battle=bs,
        move_id=227,  # Encore
        target_side=1,
        hit=True,
        move_effects=me,
        gen5_prng=prng,
    )

    new_turns = get_encore_turns(int(bs[OFF_FIELD + F_VOLATILE_1]))
    assert new_turns > 0, f"expected encore turns > 0, got {new_turns}"
    assert new_turns == 3
