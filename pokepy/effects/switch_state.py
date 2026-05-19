"""Shared switch-in state resets used by multiple battle pipelines."""

from __future__ import annotations

import numpy as np

from pokepy.core.constants import (
    FLAG_CHARGE,
    FLAG_BOOSTER_ENERGY_ACTIVE,
    NEUTRAL_BOOSTS_13,
    NEUTRAL_BOOSTS_14,
)

_PARADOX_STAT_MASK = 0x6010

def reset_incoming_switch_state(
    battle: np.ndarray,
    pokemon_offset: int,
    game_data,
    base_ability: int | None = None,
    state=None,
) -> None:
    """Restore Showdown's per-entry state for a Pokemon that just switched in.

    This mirrors the common `clearVolatile()` / `switchIn()` work that should
    happen regardless of whether the replacement arrived via a voluntary
    switch, a KO replacement, a pivot, or another forced switch path.
    """
    poff = int(pokemon_offset)

    tera = int(battle[poff + 14]) & -4096
    battle[poff + 13] = NEUTRAL_BOOSTS_13
    battle[poff + 14] = (NEUTRAL_BOOSTS_14 & 4095) | tera

    flags = int(battle[poff + 15])
    if (flags & 0x8) == 0:
        species = int(battle[poff + 0])
        type1 = int(game_data.species_types[species, 0])
        type2 = int(game_data.species_types[species, 1])
        if type2 < 0:
            type2 = type1
        battle[poff + 4] = np.int16(type1 | (type2 << 8))

    if base_ability is not None:
        battle[poff + 5] = np.int16(base_ability)

    flags &= ~(_PARADOX_STAT_MASK | FLAG_BOOSTER_ENERGY_ACTIVE | FLAG_CHARGE)
    if int(battle[poff + 6]) > 0:
        battle[poff + 15] = flags | 0x80
    else:
        battle[poff + 15] = flags & ~0x80

    from pokepy.effects.form_changes import clear_gulp_missile_state

    clear_gulp_missile_state(battle, poff)

    if state is not None:
        from pokepy.effects.form_changes import apply_shields_down_form_state

        apply_shields_down_form_state(battle, poff, state, game_data)
