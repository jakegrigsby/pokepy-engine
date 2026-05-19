"""Switch-in slot-condition helpers.

Showdown models Healing Wish and Lunar Dance as per-slot conditions whose
`onSwitchIn` / `onSwap` handlers fire before hazards and switch-in abilities.
Pokepy stores these as pending sentinels in `F_DESTINY_BOND_*`.
"""

from __future__ import annotations

import numpy as np

from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    POKEMON_SIZE,
    HEALING_WISH_PENDING,
    LUNAR_DANCE_PENDING,
)

def is_pending_wish_sentinel(pending_sentinel: int) -> bool:
    """Return True when the captured switch slot condition is a wish heal."""
    return int(pending_sentinel) in (HEALING_WISH_PENDING, LUNAR_DANCE_PENDING)

def apply_pending_wish_on_switch_in(
    battle: np.ndarray,
    side: int,
    pokemon_offset: int,
    state,
    game_data,
    pending_sentinel: int,
) -> bool:
    """Apply a captured Healing Wish / Lunar Dance sentinel to a switch-in.

    Returns True when the pending effect actually resolved and should be
    cleared. Returns False when there was no pending wish, the incoming target
    was fainted, or the slot condition should persist because it had nothing to
    heal / cure / restore yet.
    """
    pending_sentinel = int(pending_sentinel)
    healing_wish_pending = pending_sentinel == HEALING_WISH_PENDING
    lunar_dance_pending = pending_sentinel == LUNAR_DANCE_PENDING
    if not (healing_wish_pending or lunar_dance_pending):
        return False

    poff = int(pokemon_offset)
    if int(battle[poff + 1]) <= 0:
        return False

    needs_heal = int(battle[poff + 1]) < int(battle[poff + 2])
    needs_status = int(battle[poff + 12]) != 0
    needs_pp = False

    if lunar_dance_pending:
        side_base = OFF_SIDE0 if int(side) == 0 else OFF_SIDE1
        slot = (poff - side_base) // POKEMON_SIZE
        move_ids = state.team_moves if int(side) == 0 else state.opp_moves
        pp_table = state.team_pp if int(side) == 0 else state.opp_pp
        move_pp = np.asarray(game_data.move_pp)
        for j in range(4):
            mid = int(move_ids[slot, j])
            if mid < 0:
                continue
            max_pp = (int(move_pp[mid]) * 8) // 5
            if int(pp_table[slot, j]) < max_pp:
                needs_pp = True
                break

    if not (needs_heal or needs_status or needs_pp):
        return False

    battle[poff + 1] = int(battle[poff + 2])
    battle[poff + 12] = 0

    if lunar_dance_pending:
        side_base = OFF_SIDE0 if int(side) == 0 else OFF_SIDE1
        slot = (poff - side_base) // POKEMON_SIZE
        move_ids = state.team_moves if int(side) == 0 else state.opp_moves
        pp_table = state.team_pp if int(side) == 0 else state.opp_pp
        move_pp = np.asarray(game_data.move_pp)
        for j in range(4):
            mid = int(move_ids[slot, j])
            if mid >= 0:
                pp_table[slot, j] = (int(move_pp[mid]) * 8) // 5

    return True
