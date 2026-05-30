"""Shared helpers for the event turn loop."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from pokepy.core.constants import (
    M_ACTIVE0,
    M_ACTIVE1,
    OFF_META,
    OFF_SIDE0,
    OFF_SIDE1,
    POKEMON_SIZE,
    Phase,
)
from pokepy.core.state import MultiFormatState
from pokepy.effects import count_alive, get_effective_priority, get_effective_speed


def count_alive_side(battle: np.ndarray, side: int) -> int:
    base = OFF_SIDE0 if int(side) == 0 else OFF_SIDE1
    return count_alive(battle, base)


def active_slot(battle: np.ndarray, side: int) -> int:
    meta = M_ACTIVE0 if int(side) == 0 else M_ACTIVE1
    return int(battle[OFF_META + meta])


def move_id_for_side(state: MultiFormatState, side: int, move_slot: int) -> int:
    slot = int(move_slot)
    if slot < 0 or slot >= 4:
        return -1
    active = active_slot(state.battle_state, side)
    if side == 0:
        return int(state.team_moves[active, slot])
    return int(state.opp_moves[active, slot])


def action_speed(battle: np.ndarray, side: int, profile) -> int:
    from pokepy.core.constants import F_TRICK_ROOM, OFF_FIELD

    base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    poff = base + active_slot(battle, side) * POKEMON_SIZE
    spd = get_effective_speed(battle, poff)
    if int(battle[OFF_FIELD + F_TRICK_ROOM]) > 0:
        return max(1, 10000 - int(spd))
    return max(1, int(spd))


def move_priority(
    battle: np.ndarray,
    state: MultiFormatState,
    side: int,
    move_slot: int,
    game_data,
    prng,
) -> int | float:
    move_id = move_id_for_side(state, side, move_slot)
    if move_id < 0:
        return 0
    base = int(game_data.move_priority[move_id])
    poff = (OFF_SIDE0 if side == 0 else OFF_SIDE1) + active_slot(
        battle, side
    ) * POKEMON_SIZE
    return get_effective_priority(battle, move_id, base, poff, gen5_prng=prng)


def terminal_rewards(
    state: MultiFormatState, max_turns: int
) -> Tuple[np.float32, np.float32, bool, int]:
    battle = state.battle_state
    alive0 = count_alive_side(battle, 0)
    alive1 = count_alive_side(battle, 1)
    done = (alive0 == 0) or (alive1 == 0) or (int(state.turn) >= max_turns)
    if alive0 == 0 and alive1 > 0:
        winner = 1
    elif alive1 == 0 and alive0 > 0:
        winner = 0
    else:
        winner = -1
    reward0 = np.float32(0.0)
    if done and winner == 0:
        reward0 = np.float32(100.0)
    elif done and winner == 1:
        reward0 = np.float32(-100.0)
    reward1 = np.float32(-float(reward0))
    return reward0, reward1, bool(done), winner


def consume_startup_prng(state: MultiFormatState, profile, prng) -> None:
    if int(state.turn) != 0:
        return
    if getattr(state, "startup_prng_consumed", False):
        return
    startup_calls = getattr(state, "startup_prng_calls", ())
    if startup_calls:
        for call_args in startup_calls:
            prng.random(*tuple(int(a) for a in call_args))
        return
    from pokepy.sim.startup import consume_team_preview_queue_sort_frames

    consume_team_preview_queue_sort_frames(state.battle_state, prng)
    consume_endturn_quick_claw_roll(profile, prng)


def consume_endturn_quick_claw_roll(profile, prng) -> None:
    if profile.gen == 2:
        prng.random(256)
    elif profile.gen == 3:
        prng.random(5)


def sync_showdown_order(order_arr: np.ndarray, new_active_slot: int) -> None:
    new_active_slot = int(new_active_slot)
    idx = -1
    for i in range(len(order_arr)):
        if int(order_arr[i]) == new_active_slot:
            idx = i
            break
    if idx <= 0:
        return
    old_front = int(order_arr[0])
    order_arr[0] = np.int8(new_active_slot)
    order_arr[idx] = np.int8(old_front)


def base_ability_for_offset(state: MultiFormatState, pokemon_offset: int) -> int:
    poff = int(pokemon_offset)
    if poff < OFF_SIDE1:
        slot = (poff - OFF_SIDE0) // POKEMON_SIZE
        return int(state.team_abilities[slot])
    slot = (poff - OFF_SIDE1) // POKEMON_SIZE
    return int(state.opp_abilities[slot])


def clear_side_switch_field(battle: np.ndarray, side: int) -> None:
    from pokepy.core.constants import (
        F_DESTINY_BOND_0,
        F_DESTINY_BOND_1,
        F_DISABLE_TURNS_0,
        F_DISABLE_TURNS_1,
        F_EXTENDED_VOLATILE_0,
        F_EXTENDED_VOLATILE_1,
        F_LEECH_SEED_0,
        F_LEECH_SEED_1,
        F_PERISH_COUNT_0,
        F_PERISH_COUNT_1,
        F_SUBSTITUTE_0,
        F_SUBSTITUTE_1,
        F_VOLATILE_0,
        F_VOLATILE_1,
        F_YAWN_TURNS_0,
        F_YAWN_TURNS_1,
        OFF_FIELD,
    )

    offs = (
        (
            F_VOLATILE_0,
            F_LEECH_SEED_0,
            F_DISABLE_TURNS_0,
            F_EXTENDED_VOLATILE_0,
            F_DESTINY_BOND_0,
            F_SUBSTITUTE_0,
            F_YAWN_TURNS_0,
            F_PERISH_COUNT_0,
        )
        if side == 0
        else (
            F_VOLATILE_1,
            F_LEECH_SEED_1,
            F_DISABLE_TURNS_1,
            F_EXTENDED_VOLATILE_1,
            F_DESTINY_BOND_1,
            F_SUBSTITUTE_1,
            F_YAWN_TURNS_1,
            F_PERISH_COUNT_1,
        )
    )
    for off in offs:
        battle[OFF_FIELD + off] = 0


def set_forced_switch_phase(state: MultiFormatState, sides: Tuple[int, ...]) -> None:
    state.phase = np.int8(Phase.FORCED_SWITCH)
    if len(sides) == 2:
        state.forced_switch_side = np.int8(2)
    elif len(sides) == 1:
        state.forced_switch_side = np.int8(int(sides[0]))
    else:
        state.forced_switch_side = np.int8(-1)
