"""Ability suppression helpers for live active battles.

These mirror the subset of Showdown's `pokemon.ignoringAbility()` behavior
needed by pokepy's effects layer. Today this is primarily Neutralizing Gas
plus Ability Shield handling for single battles.
"""

from __future__ import annotations

from pokepy.effects._common import np
from pokepy.core.constants import (
    ABILITY_NEUTRALIZING_GAS,
    M_ACTIVE0,
    M_ACTIVE1,
    OFF_META,
    OFF_SIDE0,
    OFF_SIDE1,
    POKEMON_SIZE,
)

ITEM_ABILITY_SHIELD = 1881


def _active_opponent_offset(battle: np.ndarray, pokemon_offset: int) -> int:
    poff = int(pokemon_offset)
    if poff < OFF_SIDE1:
        return OFF_SIDE1 + int(battle[OFF_META + M_ACTIVE1]) * POKEMON_SIZE
    return OFF_SIDE0 + int(battle[OFF_META + M_ACTIVE0]) * POKEMON_SIZE


def ability_is_suppressed(
    battle: np.ndarray,
    pokemon_offset: int,
    opponent_offset: int | None = None,
) -> bool:
    """Return whether the mon's ability is suppressed by active Neutralizing Gas."""
    poff = int(pokemon_offset)
    if int(battle[poff + 1]) <= 0:
        return False
    if int(battle[poff + 6]) == ITEM_ABILITY_SHIELD:
        return False
    ability = int(battle[poff + 5])
    if ability <= 0 or ability == ABILITY_NEUTRALIZING_GAS:
        return False

    opp_off = (
        _active_opponent_offset(battle, poff)
        if opponent_offset is None
        else int(opponent_offset)
    )
    if opponent_offset is None and int(battle[opp_off + 1]) <= 0:
        return False
    return int(battle[opp_off + 5]) == ABILITY_NEUTRALIZING_GAS


def effective_ability(
    battle: np.ndarray,
    pokemon_offset: int,
    opponent_offset: int | None = None,
) -> int:
    """Return the currently effective ability id for this active mon."""
    if ability_is_suppressed(battle, pokemon_offset, opponent_offset):
        return 0
    return int(battle[int(pokemon_offset) + 5])
