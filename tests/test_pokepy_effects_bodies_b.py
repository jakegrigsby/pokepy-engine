"""Smoke tests for the ported end-of-turn / weather / terrain effect bodies.

Verifies the real ports of:
- pokepy.effects.end_of_turn.apply_end_of_turn_effects (orchestrator)
- pokepy.effects.end_of_turn.decrement_weather
- pokepy.effects.weather_terrain.apply_weather_from_move
- pokepy.effects.weather_terrain.apply_grassy_terrain_healing
- pokepy.effects.weather_terrain.apply_weather_damage
"""

from __future__ import annotations

import numpy as np

from pokepy.core.state import MultiFormatState
from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_FIELD,
    OFF_META,
    POKEMON_SIZE,
    M_ACTIVE0,
    M_ACTIVE1,
    M_WEATHER_TURNS,
    M_TERRAIN_TURNS,
    F_WEATHER,
    F_TERRAIN,
    PHASE_BATTLE,
    NEUTRAL_BOOSTS_13,
    NEUTRAL_BOOSTS_14,
    STATUS_NONE,
    STATUS_BURN,
    TYPE_NORMAL,
    WEATHER_NONE,
    WEATHER_SUN,
    WEATHER_RAIN,
    WEATHER_SAND,
    TERRAIN_NONE,
    TERRAIN_GRASSY,
)
from pokepy.core.bitpack import set_status, get_status
from pokepy.data.loader import load_game_data, load_move_effect_data, load_id_mappings
from pokepy.utils.gen5_prng import Gen5PRNG

from pokepy.effects.end_of_turn import (
    apply_end_of_turn_effects,
    decrement_weather,
    decrement_terrain,
    decrement_trick_room,
)
from pokepy.effects.weather_terrain import (
    apply_weather_from_move,
    apply_grassy_terrain_healing,
    apply_weather_damage,
    apply_weather_healing,
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
    bs[OFF_FIELD + F_WEATHER] = WEATHER_NONE
    bs[OFF_FIELD + F_TERRAIN] = TERRAIN_NONE

    for side, base in [(0, OFF_SIDE0), (1, OFF_SIDE1)]:
        for slot in range(2):
            poff = base + slot * POKEMON_SIZE
            bs[poff + 0] = 1 + slot
            bs[poff + 1] = 100  # current_hp
            bs[poff + 2] = 100  # max_hp
            bs[poff + 3] = 100  # level
            # type1=normal, type2=normal — no immunities to weather/terrain
            bs[poff + 4] = (TYPE_NORMAL & 0xFF) | ((TYPE_NORMAL & 0xFF) << 8)
            bs[poff + 5] = 0  # ability
            bs[poff + 6] = 0  # item
            bs[poff + 7] = 80  # atk
            bs[poff + 8] = 80  # def
            bs[poff + 9] = 80  # spa
            bs[poff + 10] = 80  # spd
            bs[poff + 11] = 80  # spe
            bs[poff + 12] = set_status(STATUS_NONE, 0)
            bs[poff + 13] = NEUTRAL_BOOSTS_13
            bs[poff + 14] = NEUTRAL_BOOSTS_14
            bs[poff + 15] = 0
    return state, gd, me

# -----------------------------------------------------------------------------
# Decrement helpers
# -----------------------------------------------------------------------------

def test_decrement_weather_reduces_turns_by_one():
    state, gd, me = _hand_state()
    bs = state.battle_state
    bs[OFF_FIELD + F_WEATHER] = WEATHER_SUN
    bs[OFF_META + M_WEATHER_TURNS] = 5

    decrement_weather(bs)

    assert int(bs[OFF_META + M_WEATHER_TURNS]) == 4
    assert int(bs[OFF_FIELD + F_WEATHER]) == WEATHER_SUN

def test_decrement_weather_clears_when_turns_hit_zero():
    state, gd, me = _hand_state()
    bs = state.battle_state
    bs[OFF_FIELD + F_WEATHER] = WEATHER_SUN
    bs[OFF_META + M_WEATHER_TURNS] = 1

    decrement_weather(bs)

    assert int(bs[OFF_META + M_WEATHER_TURNS]) == 0
    assert int(bs[OFF_FIELD + F_WEATHER]) == WEATHER_NONE

def test_decrement_weather_permanent_unchanged():
    state, gd, me = _hand_state()
    bs = state.battle_state
    bs[OFF_FIELD + F_WEATHER] = WEATHER_RAIN
    bs[OFF_META + M_WEATHER_TURNS] = 0  # ability-set permanent

    decrement_weather(bs)

    assert int(bs[OFF_META + M_WEATHER_TURNS]) == 0
    assert int(bs[OFF_FIELD + F_WEATHER]) == WEATHER_RAIN

def test_decrement_terrain_clears_at_zero():
    state, gd, me = _hand_state()
    bs = state.battle_state
    bs[OFF_FIELD + F_TERRAIN] = TERRAIN_GRASSY
    bs[OFF_META + M_TERRAIN_TURNS] = 1
    decrement_terrain(bs)
    assert int(bs[OFF_META + M_TERRAIN_TURNS]) == 0
    assert int(bs[OFF_FIELD + F_TERRAIN]) == TERRAIN_NONE

def test_decrement_trick_room_basic():
    state, gd, me = _hand_state()
    bs = state.battle_state
    from pokepy.core.constants import F_TRICK_ROOM
    bs[OFF_FIELD + F_TRICK_ROOM] = 5
    decrement_trick_room(bs)
    assert int(bs[OFF_FIELD + F_TRICK_ROOM]) == 4

# -----------------------------------------------------------------------------
# Move-triggered weather
# -----------------------------------------------------------------------------

def test_apply_weather_from_move_sunny_day_sets_sun():
    state, gd, me = _hand_state()
    bs = state.battle_state
    user_offset = OFF_SIDE0 + 0 * POKEMON_SIZE

    # Look up Sunny Day's id from the mapping (don't hardcode 241).
    mappings = load_id_mappings()
    sunny_day_id = mappings.move_to_idx["sunnyday"]

    # Sanity: the move_effects table has weather=WEATHER_SUN for Sunny Day.
    assert int(me.weather[sunny_day_id]) == WEATHER_SUN, (
        f"sunnyday move_effects.weather={int(me.weather[sunny_day_id])}, "
        f"expected {WEATHER_SUN}"
    )

    apply_weather_from_move(
        battle=bs,
        move_id=sunny_day_id,
        hit=True,
        game_data=gd,
        move_effects=me,
        user_offset=user_offset,
    )

    assert int(bs[OFF_FIELD + F_WEATHER]) == WEATHER_SUN
    # Default 5 turns (no Heat Rock equipped).
    assert int(bs[OFF_META + M_WEATHER_TURNS]) == 5

def test_apply_weather_from_move_miss_does_nothing():
    state, gd, me = _hand_state()
    bs = state.battle_state
    mappings = load_id_mappings()
    sunny_day_id = mappings.move_to_idx["sunnyday"]

    apply_weather_from_move(
        battle=bs,
        move_id=sunny_day_id,
        hit=False,
        game_data=gd,
        move_effects=me,
    )
    assert int(bs[OFF_FIELD + F_WEATHER]) == WEATHER_NONE

# -----------------------------------------------------------------------------
# Grassy terrain healing
# -----------------------------------------------------------------------------

def test_apply_grassy_terrain_healing_heals_one_sixteenth():
    state, gd, me = _hand_state()
    bs = state.battle_state
    poff = OFF_SIDE0 + 0 * POKEMON_SIZE
    # Set 160 max hp / 80 current hp so 160 // 16 = 10 heal step is exact.
    bs[poff + 1] = 80
    bs[poff + 2] = 160
    bs[OFF_FIELD + F_TERRAIN] = TERRAIN_GRASSY
    bs[OFF_META + M_TERRAIN_TURNS] = 5

    apply_grassy_terrain_healing(bs, poff, gd)

    # 160 / 16 = 10 hp restored
    assert int(bs[poff + 1]) == 90, f"expected 90 hp, got {int(bs[poff + 1])}"

def test_apply_grassy_terrain_healing_no_op_without_terrain():
    state, gd, me = _hand_state()
    bs = state.battle_state
    poff = OFF_SIDE0 + 0 * POKEMON_SIZE
    bs[poff + 1] = 80
    bs[poff + 2] = 160
    # Terrain explicitly NONE
    bs[OFF_FIELD + F_TERRAIN] = TERRAIN_NONE

    apply_grassy_terrain_healing(bs, poff, gd)

    assert int(bs[poff + 1]) == 80

# -----------------------------------------------------------------------------
# Weather damage standalone (sandstorm)
# -----------------------------------------------------------------------------

def test_apply_weather_damage_sandstorm_deals_one_sixteenth():
    state, gd, me = _hand_state()
    bs = state.battle_state
    poff = OFF_SIDE0 + 0 * POKEMON_SIZE
    bs[poff + 1] = 160
    bs[poff + 2] = 160
    bs[OFF_FIELD + F_WEATHER] = WEATHER_SAND
    bs[OFF_META + M_WEATHER_TURNS] = 5

    apply_weather_damage(bs, poff, gd)

    # 160 / 16 = 10
    assert int(bs[poff + 1]) == 150

# -----------------------------------------------------------------------------
# End-of-turn orchestrator
# -----------------------------------------------------------------------------

def test_end_of_turn_orchestrator_burn_and_sand_decrements_weather():
    state, gd, me = _hand_state()
    bs = state.battle_state

    # Burned Pokemon on side 0
    poff0 = OFF_SIDE0 + 0 * POKEMON_SIZE
    bs[poff0 + 1] = 160
    bs[poff0 + 2] = 160
    bs[poff0 + 12] = set_status(STATUS_BURN, 0)

    # Side 1 active is rock-typed so it ignores sand damage but still alive.
    from pokepy.core.constants import TYPE_ROCK
    poff1 = OFF_SIDE1 + 0 * POKEMON_SIZE
    bs[poff1 + 1] = 100
    bs[poff1 + 2] = 100
    bs[poff1 + 4] = (TYPE_ROCK & 0xFF) | ((TYPE_ROCK & 0xFF) << 8)

    # Sandstorm with 3 turns left
    bs[OFF_FIELD + F_WEATHER] = WEATHER_SAND
    bs[OFF_META + M_WEATHER_TURNS] = 3

    prng = Gen5PRNG()
    apply_end_of_turn_effects(
        battle=bs,
        active0=0,
        active1=0,
        game_data=gd,
        move_effects=me,
        gen5_prng=prng,
    )

    hp0_after = int(bs[poff0 + 1])
    hp1_after = int(bs[poff1 + 1])

    # Burned side 0 should lose HP. Burn alone is 1/16 = 10. Sand also
    # 1/16 = 10 (normal-type, not immune). Total 20.
    assert hp0_after < 160, (
        f"side 0 expected to lose HP from burn+sand, got {hp0_after}"
    )
    assert hp0_after == 140, f"expected 140 hp (burn 10 + sand 10), got {hp0_after}"

    # Side 1 (rock) immune to sand → still 100.
    assert hp1_after == 100, f"side 1 should be sand-immune, got {hp1_after}"

    # Weather decremented from 3 -> 2.
    assert int(bs[OFF_META + M_WEATHER_TURNS]) == 2
    assert int(bs[OFF_FIELD + F_WEATHER]) == WEATHER_SAND
