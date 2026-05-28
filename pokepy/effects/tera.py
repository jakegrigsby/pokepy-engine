"""Terastallization activation for Gen 9."""

from __future__ import annotations

import numpy as np

from pokepy.core.constants import (
    OFF_META,
    OFF_SIDE0,
    OFF_SIDE1,
    POKEMON_SIZE,
    M_ACTIVE0,
    M_ACTIVE1,
    M_TERA_ORIG_TYPES_0,
    M_TERA_ORIG_TYPES_1,
)


def _side_base(side: int) -> int:
    return OFF_SIDE0 if side == 0 else OFF_SIDE1


def _active_slot(battle: np.ndarray, side: int) -> int:
    return int(battle[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])


def _pokemon_off(battle: np.ndarray, side: int, slot: int | None = None) -> int:
    if slot is None:
        slot = _active_slot(battle, side)
    return _side_base(side) + slot * POKEMON_SIZE


def side_can_tera(battle: np.ndarray, side: int) -> bool:
    """True if this side has not yet used Terastallization this battle."""
    side_base = _side_base(side)
    for slot in range(6):
        poff = side_base + slot * POKEMON_SIZE
        if int(battle[poff + 0]) < 0:
            continue
        flags = int(battle[poff + 15])
        if (flags & 0x8) != 0:
            return False
    return True


def activate_terastallization(
    battle: np.ndarray,
    side: int,
    *,
    team_tera: np.ndarray | None = None,
    active_slot: int | None = None,
) -> bool:
    """Terastallize the active Pokemon on `side`. Returns True if activated."""
    if not side_can_tera(battle, side):
        return False

    slot = active_slot if active_slot is not None else _active_slot(battle, side)
    if slot < 0:
        return False

    poff = _pokemon_off(battle, side, slot)
    if int(battle[poff + 0]) < 0:
        return False

    flags = int(battle[poff + 15])
    if (flags & 0x8) != 0:
        return False

    types_word = int(battle[poff + 4]) & 0xFFFF
    meta_off = OFF_META + (M_TERA_ORIG_TYPES_0 if side == 0 else M_TERA_ORIG_TYPES_1)
    battle[meta_off] = types_word

    tera_type = int(battle[poff + 14]) >> 12
    tera_type &= 0xF
    if team_tera is not None and 0 <= slot < len(team_tera):
        team_type = int(team_tera[slot])
        if team_type >= 0:
            tera_type = team_type

    if tera_type <= 0:
        return False

    new_types = (tera_type & 0xFF) | ((tera_type & 0xFF) << 8)
    battle[poff + 4] = np.int16(new_types if new_types < 0x8000 else new_types - 0x10000)

    battle[poff + 15] = flags | 0x8
    return True
