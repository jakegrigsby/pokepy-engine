"""Thin accessors over the packed int16 battle buffer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pokepy.core import bitpack
from pokepy.core.constants import (
    F_CHOICE_LOCK_0,
    F_CHOICE_LOCK_1,
    F_EXTENDED_VOLATILE_0,
    F_EXTENDED_VOLATILE_1,
    F_HAZARDS_0,
    F_HAZARDS_1,
    F_LEECH_SEED_0,
    F_LEECH_SEED_1,
    F_PROTECT_0,
    F_PROTECT_1,
    F_SCREENS_0,
    F_SCREENS_1,
    F_SUBSTITUTE_0,
    F_SUBSTITUTE_1,
    F_TERRAIN,
    F_TRICK_ROOM,
    F_TURN,
    F_VOLATILE_0,
    F_VOLATILE_1,
    F_WEATHER,
    M_ACTIVE0,
    M_ACTIVE1,
    OFF_FIELD,
    OFF_META,
    OFF_MOVES,
    OFF_SIDE0,
    OFF_SIDE1,
    POKEMON_SIZE,
)

if TYPE_CHECKING:
    pass


def _side_base(side: int) -> int:
    return OFF_SIDE0 if int(side) == 0 else OFF_SIDE1


def _field_side_off(side: int, off0: int, off1: int) -> int:
    return OFF_FIELD + (off0 if int(side) == 0 else off1)


class PokemonView:
    """Read/write view of one 16-slot Pokemon in the battle buffer."""

    __slots__ = ("battle", "base")

    def __init__(self, battle: np.ndarray, base_offset: int):
        self.battle = battle
        self.base = int(base_offset)

    @property
    def species(self) -> int:
        return int(self.battle[self.base + 0])

    @species.setter
    def species(self, val: int) -> None:
        self.battle[self.base + 0] = int(val)

    @property
    def hp(self) -> int:
        return int(self.battle[self.base + 1])

    @hp.setter
    def hp(self, val: int) -> None:
        self.battle[self.base + 1] = int(val)

    @property
    def max_hp(self) -> int:
        return int(self.battle[self.base + 2])

    @max_hp.setter
    def max_hp(self, val: int) -> None:
        self.battle[self.base + 2] = int(val)

    @property
    def level(self) -> int:
        return int(self.battle[self.base + 3])

    @property
    def type1(self) -> int:
        return int(self.battle[self.base + 4]) & 0xFF

    @property
    def type2(self) -> int:
        return (int(self.battle[self.base + 4]) >> 8) & 0xFF

    @property
    def ability(self) -> int:
        return int(self.battle[self.base + 5])

    @ability.setter
    def ability(self, val: int) -> None:
        self.battle[self.base + 5] = int(val)

    @property
    def item(self) -> int:
        return int(self.battle[self.base + 6])

    @item.setter
    def item(self, val: int) -> None:
        self.battle[self.base + 6] = int(val)

    @property
    def atk(self) -> int:
        return int(self.battle[self.base + 7])

    @property
    def def_(self) -> int:
        return int(self.battle[self.base + 8])

    @property
    def spa(self) -> int:
        return int(self.battle[self.base + 9])

    @property
    def spd(self) -> int:
        return int(self.battle[self.base + 10])

    @property
    def spe(self) -> int:
        return int(self.battle[self.base + 11])

    @property
    def status_field(self) -> int:
        return int(self.battle[self.base + 12])

    @status_field.setter
    def status_field(self, val: int) -> None:
        self.battle[self.base + 12] = int(val)

    @property
    def status(self) -> int:
        return bitpack.get_status(self.status_field)

    @status.setter
    def status(self, val: int) -> None:
        turns = bitpack.get_status_turns(self.status_field)
        self.status_field = bitpack.set_status(val, turns)

    @property
    def status_turns(self) -> int:
        return bitpack.get_status_turns(self.status_field)

    @status_turns.setter
    def status_turns(self, val: int) -> None:
        self.status_field = bitpack.set_status(self.status, val)

    @property
    def boosts_13(self) -> int:
        return int(self.battle[self.base + 13])

    @boosts_13.setter
    def boosts_13(self, val: int) -> None:
        self.battle[self.base + 13] = int(val)

    @property
    def boosts_14(self) -> int:
        return int(self.battle[self.base + 14])

    @boosts_14.setter
    def boosts_14(self, val: int) -> None:
        self.battle[self.base + 14] = int(val)

    @property
    def flags(self) -> int:
        return int(self.battle[self.base + 15])

    @flags.setter
    def flags(self, val: int) -> None:
        self.battle[self.base + 15] = int(val)

    @property
    def fainted(self) -> bool:
        return bool(self.flags & 0x01)

    @fainted.setter
    def fainted(self, val: bool) -> None:
        if val:
            self.flags = self.flags | 0x01
        else:
            self.flags = self.flags & ~0x01

    @property
    def is_active(self) -> bool:
        return bool(self.flags & 0x02)

    @is_active.setter
    def is_active(self, val: bool) -> None:
        if val:
            self.flags = self.flags | 0x02
        else:
            self.flags = self.flags & ~0x02

    def get_boost(self, stat: str) -> int:
        shifts = {
            "atk": 0,
            "def": 4,
            "spa": 8,
            "spd": 12,
            "spe": 0,
            "acc": 4,
            "eva": 8,
        }
        packed = (
            self.boosts_13 if stat in ("atk", "def", "spa", "spd") else self.boosts_14
        )
        return bitpack.extract_boost(packed, shifts[stat])

    def apply_boost(self, stat: str, delta: int) -> None:
        if stat in ("atk", "def", "spa", "spd"):
            self.boosts_13 = bitpack.apply_boost_to_packed(
                self.boosts_13, {"atk": 0, "def": 4, "spa": 8, "spd": 12}[stat], delta
            )
        else:
            self.boosts_14 = bitpack.apply_boost_to_packed(
                self.boosts_14, {"spe": 0, "acc": 4, "eva": 8}[stat], delta
            )

    @property
    def volatile(self) -> int:
        side = 0 if self.base < OFF_SIDE1 else 1
        off = _field_side_off(side, F_VOLATILE_0, F_VOLATILE_1)
        return int(self.battle[off])

    @volatile.setter
    def volatile(self, val: int) -> None:
        side = 0 if self.base < OFF_SIDE1 else 1
        off = _field_side_off(side, F_VOLATILE_0, F_VOLATILE_1)
        self.battle[off] = int(val)

    @property
    def extended_volatile(self) -> int:
        side = 0 if self.base < OFF_SIDE1 else 1
        off = _field_side_off(side, F_EXTENDED_VOLATILE_0, F_EXTENDED_VOLATILE_1)
        return int(self.battle[off])

    @extended_volatile.setter
    def extended_volatile(self, val: int) -> None:
        side = 0 if self.base < OFF_SIDE1 else 1
        off = _field_side_off(side, F_EXTENDED_VOLATILE_0, F_EXTENDED_VOLATILE_1)
        self.battle[off] = int(val)

    @property
    def side(self) -> int:
        return 0 if self.base < OFF_SIDE1 else 1

    @property
    def slot(self) -> int:
        return (self.base - _side_base(self.side)) // POKEMON_SIZE

    def offset(self) -> int:
        return self.base


class SideView:
    __slots__ = ("battle", "side")

    def __init__(self, battle: np.ndarray, side: int):
        self.battle = battle
        self.side = int(side)

    @property
    def base(self) -> int:
        return _side_base(self.side)

    def pokemon(self, slot: int) -> PokemonView:
        return PokemonView(self.battle, self.base + int(slot) * POKEMON_SIZE)

    def active_slot(self) -> int:
        meta = M_ACTIVE0 if self.side == 0 else M_ACTIVE1
        return int(self.battle[OFF_META + meta])

    def active(self) -> PokemonView:
        return self.pokemon(self.active_slot())

    @property
    def hazards(self) -> int:
        off = _field_side_off(self.side, F_HAZARDS_0, F_HAZARDS_1)
        return int(self.battle[off])

    @hazards.setter
    def hazards(self, val: int) -> None:
        off = _field_side_off(self.side, F_HAZARDS_0, F_HAZARDS_1)
        self.battle[off] = int(val)

    @property
    def protect(self) -> int:
        off = _field_side_off(self.side, F_PROTECT_0, F_PROTECT_1)
        return int(self.battle[off])

    @property
    def screens(self) -> int:
        off = _field_side_off(self.side, F_SCREENS_0, F_SCREENS_1)
        return int(self.battle[off])

    @property
    def choice_lock(self) -> int:
        off = _field_side_off(self.side, F_CHOICE_LOCK_0, F_CHOICE_LOCK_1)
        return int(self.battle[off])


class FieldView:
    __slots__ = ("battle",)

    def __init__(self, battle: np.ndarray):
        self.battle = battle

    @property
    def weather(self) -> int:
        return int(self.battle[OFF_FIELD + F_WEATHER])

    @weather.setter
    def weather(self, val: int) -> None:
        self.battle[OFF_FIELD + F_WEATHER] = int(val)

    @property
    def terrain(self) -> int:
        return int(self.battle[OFF_FIELD + F_TERRAIN])

    @terrain.setter
    def terrain(self, val: int) -> None:
        self.battle[OFF_FIELD + F_TERRAIN] = int(val)

    @property
    def trick_room(self) -> int:
        return int(self.battle[OFF_FIELD + F_TRICK_ROOM])

    @property
    def turn(self) -> int:
        return int(self.battle[OFF_FIELD + F_TURN])

    @turn.setter
    def turn(self, val: int) -> None:
        self.battle[OFF_FIELD + F_TURN] = int(val)

    def side(self, side: int) -> SideView:
        return SideView(self.battle, side)

    def active(self, side: int) -> PokemonView:
        return self.side(side).active()
