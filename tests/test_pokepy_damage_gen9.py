"""Smoke + sanity tests for pokepy.mechanics.damage_gen9.calc_damage_gen9."""

from __future__ import annotations

import numpy as np
import pytest

from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_META,
    POKEMON_SIZE,
    M_ACTIVE0,
    STATE_SIZE,
    TYPE_NORMAL,
    TYPE_GROUND,
    TYPE_DRAGON,
    TYPE_FLYING,
    TYPE_GRASS,
    ABILITY_LEVITATE,
    ABILITY_MOLD_BREAKER,
)
from pokepy.data.loader import load_game_data, load_move_effect_data, load_id_mappings
from pokepy.mechanics.damage_gen9 import calc_damage_gen9
from pokepy.utils.gen5_prng import Gen5PRNG


@pytest.fixture(scope="module")
def data():
    gd = load_game_data()
    me = load_move_effect_data()
    ids = load_id_mappings()
    return gd, me, ids


def _make_battle(
    atk_species=1,
    def_species=1,
    atk_type1=TYPE_NORMAL,
    atk_type2=TYPE_NORMAL,
    def_type1=TYPE_NORMAL,
    def_type2=TYPE_NORMAL,
    atk_ability=0,
    def_ability=0,
    level=100,
    atk_atk=200,
    atk_spa=200,
    def_def=200,
    def_spd=200,
    hp=300,
):
    b = np.zeros(STATE_SIZE, dtype=np.int16)
    # Side 0 active
    b[OFF_SIDE0 + 0] = atk_species
    b[OFF_SIDE0 + 1] = hp
    b[OFF_SIDE0 + 2] = hp
    b[OFF_SIDE0 + 3] = level
    b[OFF_SIDE0 + 4] = (atk_type1 & 0xFF) | ((atk_type2 & 0xFF) << 8)
    b[OFF_SIDE0 + 5] = atk_ability
    b[OFF_SIDE0 + 6] = 0
    b[OFF_SIDE0 + 7] = atk_atk
    b[OFF_SIDE0 + 8] = 200
    b[OFF_SIDE0 + 9] = atk_spa
    b[OFF_SIDE0 + 10] = 200
    b[OFF_SIDE0 + 11] = 200
    b[OFF_SIDE0 + 13] = 0x6666  # neutral
    b[OFF_SIDE0 + 14] = 0x0666
    # Side 1 active
    b[OFF_SIDE1 + 0] = def_species
    b[OFF_SIDE1 + 1] = hp
    b[OFF_SIDE1 + 2] = hp
    b[OFF_SIDE1 + 3] = level
    b[OFF_SIDE1 + 4] = (def_type1 & 0xFF) | ((def_type2 & 0xFF) << 8)
    b[OFF_SIDE1 + 5] = def_ability
    b[OFF_SIDE1 + 6] = 0
    b[OFF_SIDE1 + 7] = 200
    b[OFF_SIDE1 + 8] = def_def
    b[OFF_SIDE1 + 9] = 200
    b[OFF_SIDE1 + 10] = def_spd
    b[OFF_SIDE1 + 11] = 200
    b[OFF_SIDE1 + 13] = 0x6666
    b[OFF_SIDE1 + 14] = 0x0666
    # Active indices
    b[OFF_META + M_ACTIVE0 + 0] = 0
    b[OFF_META + M_ACTIVE0 + 1] = 0
    return b


def test_smoke_runs(data):
    gd, me, ids = data
    battle = _make_battle()
    moves = np.zeros((6, 4), dtype=np.int16)
    tackle = ids.move_to_idx.get("tackle", 33)
    moves[0, 0] = tackle
    dmg = calc_damage_gen9(
        battle,
        atk_side=0,
        move_idx=0,
        player_moves=moves,
        opp_moves=moves,
        game_data=gd,
        move_effects=me,
        type_chart=gd.type_chart,
        gen5_prng=Gen5PRNG((1, 2, 3, 4)),
    )
    assert isinstance(dmg, int)
    assert dmg >= 0


def test_status_move_zero_damage(data):
    gd, me, ids = data
    battle = _make_battle()
    moves = np.zeros((6, 4), dtype=np.int16)
    growl = ids.move_to_idx.get("growl", None)
    if growl is None:
        pytest.skip("growl missing")
    moves[0, 0] = growl
    dmg = calc_damage_gen9(
        battle,
        0,
        0,
        moves,
        moves,
        gd,
        me,
        gd.type_chart,
        gen5_prng=Gen5PRNG((1, 2, 3, 4)),
    )
    assert dmg == 0


def test_levitate_immune_to_ground(data):
    gd, me, ids = data
    eq = ids.move_to_idx.get("earthquake", None)
    if eq is None:
        pytest.skip("earthquake missing")
    battle = _make_battle(
        def_ability=ABILITY_LEVITATE, def_type1=TYPE_GRASS, def_type2=TYPE_GRASS
    )
    moves = np.zeros((6, 4), dtype=np.int16)
    moves[0, 0] = eq
    dmg = calc_damage_gen9(
        battle,
        0,
        0,
        moves,
        moves,
        gd,
        me,
        gd.type_chart,
        gen5_prng=Gen5PRNG((1, 2, 3, 4)),
    )
    assert dmg == 0


def test_mold_breaker_bypasses_levitate(data):
    gd, me, ids = data
    eq = ids.move_to_idx.get("earthquake", None)
    if eq is None:
        pytest.skip("earthquake missing")
    battle = _make_battle(
        atk_ability=ABILITY_MOLD_BREAKER,
        def_ability=ABILITY_LEVITATE,
        def_type1=TYPE_GRASS,
        def_type2=TYPE_GRASS,
    )
    moves = np.zeros((6, 4), dtype=np.int16)
    moves[0, 0] = eq
    dmg = calc_damage_gen9(
        battle,
        0,
        0,
        moves,
        moves,
        gd,
        me,
        gd.type_chart,
        gen5_prng=Gen5PRNG((1, 2, 3, 4)),
    )
    assert dmg > 0


def test_super_effective_more_than_neutral(data):
    gd, me, ids = data
    eq = ids.move_to_idx.get("earthquake", None)
    if eq is None:
        pytest.skip("earthquake missing")
    moves = np.zeros((6, 4), dtype=np.int16)
    moves[0, 0] = eq
    # neutral target (Normal)
    b1 = _make_battle()
    # super-effective target (Fire)
    from pokepy.core.constants import TYPE_FIRE

    b2 = _make_battle(def_type1=TYPE_FIRE, def_type2=TYPE_FIRE)
    seed = (1, 2, 3, 4)
    d1 = calc_damage_gen9(
        b1, 0, 0, moves, moves, gd, me, gd.type_chart, gen5_prng=Gen5PRNG(seed)
    )
    d2 = calc_damage_gen9(
        b2, 0, 0, moves, moves, gd, me, gd.type_chart, gen5_prng=Gen5PRNG(seed)
    )
    assert d2 > d1


def test_payback_does_not_boost_vs_newly_switched_target(data):
    gd, me, ids = data
    payback = ids.move_to_idx.get("payback", None)
    if payback is None:
        pytest.skip("payback missing")
    battle = _make_battle()
    moves = np.zeros((6, 4), dtype=np.int16)
    moves[0, 0] = payback
    seed = (1, 2, 3, 4)

    dmg_vs_new_switch = calc_damage_gen9(
        battle,
        0,
        0,
        moves,
        moves,
        gd,
        me,
        gd.type_chart,
        is_moving_last=True,
        target_newly_switched=True,
        gen5_prng=Gen5PRNG(seed),
    )
    dmg_vs_already_moved = calc_damage_gen9(
        battle,
        0,
        0,
        moves,
        moves,
        gd,
        me,
        gd.type_chart,
        is_moving_last=True,
        target_newly_switched=False,
        gen5_prng=Gen5PRNG(seed),
    )

    assert dmg_vs_already_moved > dmg_vs_new_switch


def test_fishious_rend_boosts_vs_newly_switched_target(data):
    gd, me, ids = data
    fishious_rend = ids.move_to_idx.get("fishiousrend", None)
    if fishious_rend is None:
        pytest.skip("fishiousrend missing")
    battle = _make_battle()
    moves = np.zeros((6, 4), dtype=np.int16)
    moves[0, 0] = fishious_rend
    seed = (1, 2, 3, 4)

    dmg_vs_new_switch = calc_damage_gen9(
        battle,
        0,
        0,
        moves,
        moves,
        gd,
        me,
        gd.type_chart,
        is_moving_last=True,
        target_newly_switched=True,
        gen5_prng=Gen5PRNG(seed),
    )
    dmg_vs_already_moved = calc_damage_gen9(
        battle,
        0,
        0,
        moves,
        moves,
        gd,
        me,
        gd.type_chart,
        is_moving_last=True,
        target_newly_switched=False,
        gen5_prng=Gen5PRNG(seed),
    )

    assert dmg_vs_new_switch > dmg_vs_already_moved
