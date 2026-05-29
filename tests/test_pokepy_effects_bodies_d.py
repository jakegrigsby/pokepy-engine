"""Smoke tests for ported effect bodies in pokepy.effects.

Covers:
- items.apply_leftovers_healing
- items.apply_sitrus_berry
- items.apply_lum_berry
- items.apply_life_orb_recoil
- recovery.apply_recovery_from_move (Recover)
- damage_modifiers.apply_recoil_drain_from_move (Brave Bird)
"""

from __future__ import annotations

import numpy as np
import pytest

from pokepy.core import constants as C
from pokepy.core.bitpack import set_status
from pokepy.data.loader import load_game_data, load_move_effect_data, load_id_mappings

from pokepy.effects.items import (
    apply_leftovers_healing,
    apply_sitrus_berry,
    apply_lum_berry,
    apply_life_orb_recoil,
)
from pokepy.effects.recovery import apply_recovery_from_move
from pokepy.effects.damage_modifiers import apply_recoil_drain_from_move


@pytest.fixture(scope="module")
def game_data():
    return load_game_data()


@pytest.fixture(scope="module")
def move_effects():
    return load_move_effect_data()


@pytest.fixture(scope="module")
def ids():
    return load_id_mappings()


def _empty_battle() -> np.ndarray:
    return np.zeros(C.STATE_SIZE, dtype=np.int16)


def _setup_pokemon(
    battle: np.ndarray,
    offset: int,
    *,
    hp: int = 100,
    max_hp: int = 100,
    item: int = 0,
    ability: int = 0,
    types: int = 0,
) -> None:
    battle[offset + 1] = hp
    battle[offset + 2] = max_hp
    battle[offset + 4] = types
    battle[offset + 5] = ability
    battle[offset + 6] = item


# -----------------------------------------------------------------------------
# items
# -----------------------------------------------------------------------------


def test_leftovers_heals_one_sixteenth(game_data):
    battle = _empty_battle()
    _setup_pokemon(battle, C.OFF_SIDE0, hp=50, max_hp=100, item=C.ITEM_LEFTOVERS)
    apply_leftovers_healing(battle, C.OFF_SIDE0, game_data)
    # 100 / 16 = 6 (int truncation), 50 + 6 = 56
    assert int(battle[C.OFF_SIDE0 + 1]) == 56


def test_leftovers_caps_at_max_hp(game_data):
    battle = _empty_battle()
    _setup_pokemon(battle, C.OFF_SIDE0, hp=99, max_hp=100, item=C.ITEM_LEFTOVERS)
    apply_leftovers_healing(battle, C.OFF_SIDE0, game_data)
    assert int(battle[C.OFF_SIDE0 + 1]) == 100


def test_leftovers_no_heal_without_item(game_data):
    battle = _empty_battle()
    _setup_pokemon(battle, C.OFF_SIDE0, hp=50, max_hp=100, item=0)
    apply_leftovers_healing(battle, C.OFF_SIDE0, game_data)
    assert int(battle[C.OFF_SIDE0 + 1]) == 50


def test_sitrus_berry_heals_25_percent_and_consumes(game_data):
    battle = _empty_battle()
    _setup_pokemon(battle, C.OFF_SIDE0, hp=49, max_hp=100, item=C.ITEM_SITRUS_BERRY)
    apply_sitrus_berry(battle, C.OFF_SIDE0, game_data)
    # threshold = 50, hp 49 < 50: triggers. heal = 100/4 = 25 -> hp 74
    assert int(battle[C.OFF_SIDE0 + 1]) == 74
    assert int(battle[C.OFF_SIDE0 + 6]) == 0  # consumed


def test_sitrus_berry_no_trigger_above_half(game_data):
    battle = _empty_battle()
    _setup_pokemon(battle, C.OFF_SIDE0, hp=80, max_hp=100, item=C.ITEM_SITRUS_BERRY)
    apply_sitrus_berry(battle, C.OFF_SIDE0, game_data)
    assert int(battle[C.OFF_SIDE0 + 1]) == 80
    assert int(battle[C.OFF_SIDE0 + 6]) == C.ITEM_SITRUS_BERRY


def test_lum_berry_clears_status_and_consumes(game_data):
    battle = _empty_battle()
    _setup_pokemon(battle, C.OFF_SIDE0, hp=80, max_hp=100, item=C.ITEM_LUM_BERRY)
    battle[C.OFF_SIDE0 + 12] = set_status(C.STATUS_POISON, 0)
    apply_lum_berry(battle, C.OFF_SIDE0, game_data)
    assert int(battle[C.OFF_SIDE0 + 12]) == 0
    assert int(battle[C.OFF_SIDE0 + 6]) == 0


def test_lum_berry_no_status_does_not_consume(game_data):
    battle = _empty_battle()
    _setup_pokemon(battle, C.OFF_SIDE0, hp=80, max_hp=100, item=C.ITEM_LUM_BERRY)
    apply_lum_berry(battle, C.OFF_SIDE0, game_data)
    # No status -> berry remains
    assert int(battle[C.OFF_SIDE0 + 6]) == C.ITEM_LUM_BERRY


def test_life_orb_recoil_one_tenth(game_data, move_effects):
    battle = _empty_battle()
    _setup_pokemon(battle, C.OFF_SIDE0, hp=100, max_hp=100, item=C.ITEM_LIFE_ORB)
    apply_life_orb_recoil(
        battle,
        C.OFF_SIDE0,
        damage_dealt=50,
        hit=True,
        game_data=game_data,
        move_id=None,
        move_effects=move_effects,
    )
    # 100 * 0.1 = 10, hp 100 - 10 = 90
    assert int(battle[C.OFF_SIDE0 + 1]) == 90


def test_life_orb_no_recoil_on_miss(game_data, move_effects):
    battle = _empty_battle()
    _setup_pokemon(battle, C.OFF_SIDE0, hp=100, max_hp=100, item=C.ITEM_LIFE_ORB)
    apply_life_orb_recoil(
        battle,
        C.OFF_SIDE0,
        damage_dealt=0,
        hit=False,
        game_data=game_data,
        move_id=None,
        move_effects=move_effects,
    )
    assert int(battle[C.OFF_SIDE0 + 1]) == 100


# -----------------------------------------------------------------------------
# recovery
# -----------------------------------------------------------------------------


def test_recover_heals_to_full(game_data, move_effects, ids):
    battle = _empty_battle()
    _setup_pokemon(battle, C.OFF_SIDE0, hp=50, max_hp=100)
    move_id = ids.move_to_idx["recover"]
    apply_recovery_from_move(
        battle,
        move_id,
        C.OFF_SIDE0,
        hit=True,
        game_data=game_data,
        move_effects=move_effects,
    )
    # Recover heal = 50% -> 50 + 50 = 100
    assert int(battle[C.OFF_SIDE0 + 1]) == 100


def test_recover_no_heal_on_miss(game_data, move_effects, ids):
    battle = _empty_battle()
    _setup_pokemon(battle, C.OFF_SIDE0, hp=50, max_hp=100)
    move_id = ids.move_to_idx["recover"]
    apply_recovery_from_move(
        battle,
        move_id,
        C.OFF_SIDE0,
        hit=False,
        game_data=game_data,
        move_effects=move_effects,
    )
    assert int(battle[C.OFF_SIDE0 + 1]) == 50


# -----------------------------------------------------------------------------
# damage_modifiers
# -----------------------------------------------------------------------------


def test_brave_bird_recoil(game_data, move_effects, ids):
    battle = _empty_battle()
    _setup_pokemon(battle, C.OFF_SIDE0, hp=100, max_hp=100)
    move_id = ids.move_to_idx["bravebird"]
    apply_recoil_drain_from_move(
        battle,
        move_id,
        C.OFF_SIDE0,
        damage_dealt=90,
        hit=True,
        game_data=game_data,
        move_effects=move_effects,
    )
    # Brave Bird recoil = round(33% of damage dealt) = round(0.33 * 90) = 30
    # (Showdown sim/battle-actions.ts:1396 uses Math.round). 100 - 30 = 70.
    expected = 100 - int(round(90 * 33 / 100.0))
    assert int(battle[C.OFF_SIDE0 + 1]) == expected
    assert expected == 70


def test_brave_bird_no_recoil_on_miss(game_data, move_effects, ids):
    battle = _empty_battle()
    _setup_pokemon(battle, C.OFF_SIDE0, hp=100, max_hp=100)
    move_id = ids.move_to_idx["bravebird"]
    apply_recoil_drain_from_move(
        battle,
        move_id,
        C.OFF_SIDE0,
        damage_dealt=0,
        hit=False,
        game_data=game_data,
        move_effects=move_effects,
    )
    assert int(battle[C.OFF_SIDE0 + 1]) == 100
