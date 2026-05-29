"""Helpers for Showdown-style groundedness checks.

Pokepy currently models Flying typing, Levitate, Air Balloon, and Iron Ball
for groundedness. Gravity, Ingrain, Smack Down, Magnet Rise, Telekinesis, and
Roost's temporary type suppression edge cases are still out of scope.
"""

from __future__ import annotations

from pokepy.core.constants import TYPE_FLYING, ABILITY_LEVITATE, ITEM_AIR_BALLOON
from pokepy.effects.ability_suppression import effective_ability

_ITEM_IRON_BALL = 278


def is_grounded(battle, pokemon_offset: int, other_offset: int | None = None) -> bool:
    """Return True iff the Pokemon currently counts as grounded.

    When ``other_offset`` is supplied, Levitate is resolved through
    ``effective_ability`` so attack-time suppression is honored.
    """
    poff = int(pokemon_offset)
    item = int(battle[poff + 6])
    if item == _ITEM_IRON_BALL:
        return True
    if item == ITEM_AIR_BALLOON:
        return False

    types = int(battle[poff + 4]) & 0xFFFF
    type1 = types & 0xFF
    type2 = (types >> 8) & 0xFF
    if other_offset is None:
        ability = int(battle[poff + 5])
    else:
        ability = effective_ability(battle, poff, int(other_offset))

    is_flying = (type1 == TYPE_FLYING) or (type2 == TYPE_FLYING)
    has_levitate = ability == ABILITY_LEVITATE
    return (not is_flying) and (not has_levitate)
