"""Phase 6 action mask tests."""

from __future__ import annotations

import numpy as np
import pytest

from pokepy.core.state import MultiFormatState
from pokepy.core.constants import (
    OFF_SIDE0, OFF_SIDE1, OFF_FIELD, OFF_META, OFF_MOVES, POKEMON_SIZE,
    M_ACTIVE0, M_ACTIVE1,
    M_CHARGING_0,
    M_LOCKED_MOVE_0, M_LOCKED_TURNS_0,
    M_RECHARGE_0,
    F_CHOICE_LOCK_0, F_CHOICE_LOCK_1, F_LAST_MOVE_0, F_LAST_MOVE_1,
    F_DISABLE_0, F_DISABLE_1,
    ITEM_ASSAULT_VEST,
    PHASE_BATTLE, PHASE_FORCED_SWITCH, NUM_BATTLE_ACTIONS,
)
from pokepy.data.loader import load_game_data, load_id_mappings
from pokepy.engine.action_mask import get_battle_action_mask, get_action_mask

def _hand_state():
    gd = load_game_data()
    state = MultiFormatState.create_empty(format_id=1)
    state.phase = np.int8(PHASE_BATTLE)
    # Two pokemon on each side
    for slot in range(2):
        state.team_species[slot] = 1 + slot
        state.team_moves[slot] = [33, 14, 22, 75]
        state.team_pp[slot] = [35, 40, 25, 25]
        state.opp_species[slot] = 4 + slot
        state.opp_moves[slot] = [33, 14, 22, 75]
        state.opp_pp[slot] = [35, 40, 25, 25]

    bs = state.battle_state
    bs[OFF_META + M_ACTIVE0] = 0
    bs[OFF_META + M_ACTIVE1] = 0
    # Sentinel fields default to -1 in real init
    for off in (F_CHOICE_LOCK_0, F_CHOICE_LOCK_1, F_LAST_MOVE_0, F_LAST_MOVE_1,
                F_DISABLE_0, F_DISABLE_1):
        bs[OFF_FIELD + off] = -1
    for side, base in [(0, OFF_SIDE0), (1, OFF_SIDE1)]:
        for slot in range(2):
            poff = base + slot * POKEMON_SIZE
            bs[poff + 0] = 1 + slot if side == 0 else 4 + slot
            bs[poff + 1] = 100  # current_hp
            bs[poff + 2] = 100  # max_hp
            bs[poff + 3] = 100
            bs[poff + 4] = 0  # type1=normal, type2=normal
            bs[poff + 5] = 0  # ability
            bs[poff + 6] = 0  # item
            bs[poff + 13] = 0x6666
            bs[poff + 14] = 0x0666
    return state, gd

def test_battle_mask_basic():
    state, gd = _hand_state()
    mask = get_battle_action_mask(state, side=0, game_data=gd)
    assert mask.shape == (10,)
    assert mask.dtype == bool
    # All 4 moves valid (have PP)
    assert mask[0:4].all()
    # Switch to slot 1 valid (alive, not active)
    assert mask[4 + 1]
    # Switch to slot 0 invalid (active)
    assert not mask[4 + 0]
    # Switches to empty slots invalid
    for s in range(2, 6):
        assert not mask[4 + s]

def test_battle_mask_no_pp_struggle():
    state, gd = _hand_state()
    state.team_pp[0] = [0, 0, 0, 0]
    mask = get_battle_action_mask(state, 0, gd)
    # Only move 0 valid (Struggle slot)
    assert mask[0]
    assert not mask[1] and not mask[2] and not mask[3]

def test_battle_mask_fainted_no_switch():
    state, gd = _hand_state()
    bs = state.battle_state
    # Faint slot 1
    bs[OFF_SIDE0 + 1 * POKEMON_SIZE + 1] = 0  # hp=0
    bs[OFF_SIDE0 + 1 * POKEMON_SIZE + 15] = 1  # fainted flag
    mask = get_battle_action_mask(state, 0, gd)
    assert not mask[4 + 1]

def test_forced_switch_no_moves():
    state, gd = _hand_state()
    state.phase = np.int8(PHASE_FORCED_SWITCH)
    mask = get_action_mask(state, side=0, game_data=gd)
    assert not mask[0:4].any()  # no moves
    assert mask[4 + 1]  # switch to slot 1 still legal

def test_battle_mask_lockedmove_forces_slot_and_blocks_switches():
    state, gd = _hand_state()
    bs = state.battle_state
    bs[OFF_MOVES + M_LOCKED_MOVE_0] = 14
    bs[OFF_MOVES + M_LOCKED_TURNS_0] = 2

    mask = get_battle_action_mask(state, 0, gd)

    assert mask.tolist()[:4] == [False, True, False, False]
    assert not mask[4:].any()

def test_battle_mask_charging_forces_slot_and_blocks_switches():
    state, gd = _hand_state()
    bs = state.battle_state
    bs[OFF_META + M_CHARGING_0] = 22

    mask = get_battle_action_mask(state, 0, gd)

    assert mask.tolist()[:4] == [False, False, True, False]
    assert not mask[4:].any()

def test_battle_mask_recharge_blocks_switches():
    state, gd = _hand_state()
    bs = state.battle_state
    bs[OFF_MOVES + M_RECHARGE_0] = 1

    mask = get_battle_action_mask(state, 0, gd)

    assert mask.tolist()[:4] == [True, False, False, False]
    assert not mask[4:].any()

def test_battle_mask_assault_vest_all_status_moves_falls_back_to_struggle_slot():
    state, gd = _hand_state()
    ids = load_id_mappings()
    bs = state.battle_state

    state.team_moves[0] = [
        ids.move_to_idx["protect"],
        ids.move_to_idx["recover"],
        ids.move_to_idx["taunt"],
        ids.move_to_idx["swordsdance"],
    ]
    bs[OFF_SIDE0 + 6] = ITEM_ASSAULT_VEST

    mask = get_battle_action_mask(state, 0, gd)

    assert mask.tolist()[:4] == [True, False, False, False]
    assert mask[4 + 1]
