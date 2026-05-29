"""Smoke tests for the ported effect bodies in:

- pokepy.effects.auto_switch
- pokepy.effects.hazards
- pokepy.effects.protect
- pokepy.effects.flinch
- pokepy.effects.misc

These verify the ports done by the bodies-F subagent against
9714-9940, 10148-10233, 11062-11417.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokepy.core.state import MultiFormatState
from pokepy.core.bitpack import (
    get_flinched,
    get_protect_active,
    get_spikes_layers,
    get_status,
    get_stealth_rock,
    set_status,
)
from pokepy.core.constants import (
    F_HAZARDS_0,
    F_HAZARDS_1,
    F_PROTECT_0,
    F_PROTECT_1,
    F_SCREENS_0,
    F_VOLATILE_1,
    M_ACTIVE0,
    M_ACTIVE1,
    NEUTRAL_BOOSTS_13,
    NEUTRAL_BOOSTS_14,
    OFF_FIELD,
    OFF_META,
    OFF_SIDE0,
    OFF_SIDE1,
    PHASE_BATTLE,
    POKEMON_SIZE,
    SCREEN_MASK_3BIT,
    SCREEN_REFLECT_SHIFT,
    STATUS_NONE,
    TYPE_FIRE,
    TYPE_NORMAL,
)
from pokepy.data.loader import load_game_data, load_move_effect_data
from pokepy.utils.gen5_prng import Gen5PRNG

from pokepy.effects.auto_switch import auto_switch, count_alive
from pokepy.effects.hazards import (
    apply_hazard_damage_on_switch,
    apply_hazard_from_move,
)
from pokepy.effects.protect import (
    apply_protect_from_move,
    check_protected,
)
from pokepy.effects.flinch import apply_flinch_from_move
from pokepy.effects.misc import (
    apply_knock_off_from_move,
    apply_screen_from_move,
)


def _hand_state(target_type1: int = TYPE_NORMAL):
    """Build a minimal battle buffer with one normal-type Pokemon per side."""
    gd = load_game_data()
    me = load_move_effect_data()
    state = MultiFormatState.create_empty(format_id=1)
    state.phase = np.int8(PHASE_BATTLE)

    bs = state.battle_state
    bs[OFF_META + M_ACTIVE0] = 0
    bs[OFF_META + M_ACTIVE1] = 0

    for side, base in [(0, OFF_SIDE0), (1, OFF_SIDE1)]:
        for slot in range(6):
            poff = base + slot * POKEMON_SIZE
            bs[poff + 0] = 1 + slot  # species_id
            bs[poff + 1] = 100  # current_hp
            bs[poff + 2] = 100  # max_hp
            bs[poff + 3] = 100  # level
            t1 = target_type1 if side == 1 else TYPE_NORMAL
            # Use type1 for both slots when type1 == TYPE_NORMAL (default);
            # otherwise pair with TYPE_NORMAL so we are mono-type for tests
            # that care about type-effectiveness multipliers (e.g. SR damage).
            t2 = TYPE_NORMAL if (target_type1 != TYPE_NORMAL and side == 1) else t1
            bs[poff + 4] = (t1 & 0xFF) | ((t2 & 0xFF) << 8)
            bs[poff + 5] = 0
            bs[poff + 6] = 1  # item (so knock-off has something to knock off)
            bs[poff + 7] = 80
            bs[poff + 8] = 80
            bs[poff + 9] = 80
            bs[poff + 10] = 80
            bs[poff + 11] = 80
            bs[poff + 12] = set_status(STATUS_NONE, 0)
            bs[poff + 13] = NEUTRAL_BOOSTS_13
            bs[poff + 14] = NEUTRAL_BOOSTS_14
            bs[poff + 15] = 0
    return state, gd, me


def test_apply_hazard_from_move_stealth_rock():
    state, gd, me = _hand_state()
    bs = state.battle_state
    # Side 1 should receive Stealth Rock (target_side=1).
    assert get_stealth_rock(int(bs[OFF_FIELD + F_HAZARDS_1])) == 0

    apply_hazard_from_move(
        battle=bs,
        move_id=446,  # Stealth Rock
        target_side=1,
        hit=True,
        game_data=gd,
        move_effects=me,
    )
    assert get_stealth_rock(int(bs[OFF_FIELD + F_HAZARDS_1])) == 1
    # Side 0 unaffected
    assert get_stealth_rock(int(bs[OFF_FIELD + F_HAZARDS_0])) == 0


def test_apply_hazard_damage_on_switch_fire_takes_quarter():
    state, gd, me = _hand_state(target_type1=TYPE_FIRE)
    bs = state.battle_state
    # Place Stealth Rock on side 1.
    apply_hazard_from_move(bs, 446, 1, True, gd, me)

    p_off = OFF_SIDE1 + 0 * POKEMON_SIZE
    pre_hp = int(bs[p_off + 1])
    assert pre_hp == 100

    apply_hazard_damage_on_switch(
        battle=bs,
        pokemon_offset=p_off,
        hazard_offset=OFF_FIELD + F_HAZARDS_1,
    )
    post_hp = int(bs[p_off + 1])
    # Fire is doubly weak to Rock (1, then 2x) -> SR_mult = 2.0,
    # raw damage = max_hp * 2.0 / 8 = 25 -> HP drops by 25.
    assert pre_hp - post_hp == 25


def test_apply_protect_from_move_returns_true_and_sets_active():
    state, gd, me = _hand_state()
    bs = state.battle_state
    prng = Gen5PRNG()

    success = apply_protect_from_move(
        battle=bs,
        move_id=182,  # Protect
        user_side=0,
        move_effects=me,
        gen5_prng=prng,
    )
    assert success is True
    assert get_protect_active(int(bs[OFF_FIELD + F_PROTECT_0])) == 1


def test_check_protected_false_for_unprotected_side():
    state, _, _ = _hand_state()
    bs = state.battle_state
    assert check_protected(bs, target_side=1) is False


def test_apply_flinch_from_move_fake_out_sets_flinch_bit():
    state, gd, me = _hand_state()
    bs = state.battle_state
    # Defender is side=1; attacker is side=0 (no Sheer Force).
    prng = Gen5PRNG()

    apply_flinch_from_move(
        battle=bs,
        move_id=252,  # Fake Out
        target_side=1,
        hit=True,
        move_effects=me,
        gen5_prng=prng,
    )
    # Fake Out has 100% flinch chance, so the bit must be set.
    assert get_flinched(int(bs[OFF_FIELD + F_VOLATILE_1])) is True


def test_auto_switch_returns_next_alive_slot():
    state, _, _ = _hand_state()
    bs = state.battle_state
    # Faint slot 0 of side 0.
    bs[OFF_SIDE0 + 0 * POKEMON_SIZE + 1] = 0
    bs[OFF_SIDE0 + 0 * POKEMON_SIZE + 15] = 1  # fainted flag

    new_active = auto_switch(bs, OFF_SIDE0, current_active=0, needs_switch=True)
    assert new_active == 1
    assert count_alive(bs, OFF_SIDE0) == 5


def test_apply_knock_off_clears_defender_item():
    state, gd, me = _hand_state()
    bs = state.battle_state
    target_off = OFF_SIDE1 + 0 * POKEMON_SIZE
    # Set a non-Ogerpon-mask item.
    bs[target_off + 6] = 200
    assert int(bs[target_off + 6]) == 200

    apply_knock_off_from_move(
        battle=bs,
        move_id=282,  # Knock Off
        target_offset=int(target_off),
        hit=True,
        game_data=gd,
        move_effects=me,
    )
    assert int(bs[target_off + 6]) == 0


def test_apply_screen_from_move_reflect_sets_counter():
    state, gd, me = _hand_state()
    bs = state.battle_state
    apply_screen_from_move(
        battle=bs,
        move_id=115,  # Reflect
        side=0,
        hit=True,
        move_effects=me,
    )
    screens = int(bs[OFF_FIELD + F_SCREENS_0]) & 0xFFFF
    reflect_turns = (screens >> SCREEN_REFLECT_SHIFT) & SCREEN_MASK_3BIT
    assert reflect_turns > 0
