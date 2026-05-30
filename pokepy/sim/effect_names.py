"""Data-driven ability/item id -> dispatch effect name mapping."""

from __future__ import annotations

from pokepy.core.constants import (
    ABILITY_HYDRATION,
    ABILITY_SHED_SKIN,
    ABILITY_SPEED_BOOST,
    ITEM_BLACK_SLUDGE,
    ITEM_LEFTOVERS,
)

# Ability id -> Showdown-style effect id for dispatch registry lookup.
ABILITY_EFFECT_NAMES: dict[int, str] = {
    ABILITY_SPEED_BOOST: "speedboost",
    ABILITY_SHED_SKIN: "shedskin",
    ABILITY_HYDRATION: "hydration",
}

# Item id -> dispatch effect id.
ITEM_EFFECT_NAMES: dict[int, str] = {
    ITEM_LEFTOVERS: "leftovers",
    ITEM_BLACK_SLUDGE: "blacksludge",
}


def ability_effect_name(ability_id: int) -> str | None:
    return ABILITY_EFFECT_NAMES.get(int(ability_id))


def item_effect_name(item_id: int) -> str | None:
    return ITEM_EFFECT_NAMES.get(int(item_id))
