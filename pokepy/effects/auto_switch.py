"""Auto-switch helpers (post-faint switching, alive counting).

Port of MultiFormatFastEnv._auto_switch / _count_alive
(the Showdown reference implementation).

The pokepy port operates on a single-battle scalar `battle: np.ndarray[256]`
and matches the calling convention used by `pokepy.engine.battle_gen9`:

    auto_switch(battle, side_base, current_active, needs_switch=True) -> int
    count_alive(battle, side_base) -> int
"""
from __future__ import annotations

from pokepy.effects._common import np, MultiFormatState, Gen5PRNG
from pokepy.core.constants import POKEMON_SIZE

def auto_switch(
    battle: np.ndarray,
    base_offset: int,
    current_active: int,
    needs_switch: bool = True,
    order=None,
) -> int:
    """Find the first alive Pokemon on a side and return its slot index.

    Mirrors the source loop at lines 7016-7032: starts with `next_active =
    current_active`, walks slots 0..5, and on the first slot that is alive
    AND `needs_switch` is True AND we have not already selected a different
    slot, sets `next_active = i`. Returns `next_active`.
    """
    base_offset = int(base_offset)
    next_active = int(current_active)
    needs_switch = bool(needs_switch)

    if order is not None:
        seen = {int(current_active)}
        ordered_slots = [int(current_active)]
        for slot in order:
            slot_i = int(slot)
            if slot_i in seen:
                continue
            ordered_slots.append(slot_i)
            seen.add(slot_i)
        for i in ordered_slots[1:]:
            offset = base_offset + i * POKEMON_SIZE
            flags = int(battle[offset + 15])
            is_fainted = (flags & 0x1) != 0
            hp = int(battle[offset + 1])
            is_alive = (hp > 0) and (not is_fainted)
            should_select = needs_switch and is_alive and (next_active == int(current_active))
            if should_select:
                next_active = i

    for i in range(6):
        offset = base_offset + i * POKEMON_SIZE
        flags = int(battle[offset + 15])
        is_fainted = (flags & 0x1) != 0
        hp = int(battle[offset + 1])
        is_alive = (hp > 0) and (not is_fainted)
        should_select = needs_switch and is_alive and (next_active == int(current_active))
        if should_select:
            next_active = i
    return int(next_active)

def count_alive(battle: np.ndarray, base_offset: int) -> int:
    """Count alive (non-fainted, hp>0) Pokemon on a side.

    Port of `_count_alive` (lines 7034-7043).
    """
    base_offset = int(base_offset)
    alive = 0
    for i in range(6):
        offset = base_offset + i * POKEMON_SIZE
        flags = int(battle[offset + 15])
        hp = int(battle[offset + 1])
        if hp > 0 and (flags & 0x1) == 0:
            alive += 1
    return int(alive)
