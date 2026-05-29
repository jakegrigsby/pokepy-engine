"""Action mask construction for the Gen 9 OU battle phase.

Covers the BATTLE and FORCED_SWITCH branches only — pokepy v0 assumes
pre-built teams and skips the teambuilder phases (species/move/ability/
item/EV/preview select). Returns `bool[NUM_BATTLE_ACTIONS=10]` (4 moves +
6 switches), which is all the battle phase ever uses.
"""

from __future__ import annotations

import numpy as np

from pokepy.core.state import MultiFormatState
from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_FIELD,
    OFF_META,
    OFF_MOVES,
    POKEMON_SIZE,
    M_ACTIVE0,
    M_ACTIVE1,
    M_CHARGING_0,
    M_CHARGING_1,
    M_LOCKED_MOVE_0,
    M_LOCKED_MOVE_1,
    M_LOCKED_TURNS_0,
    M_LOCKED_TURNS_1,
    M_RECHARGE_0,
    M_RECHARGE_1,
    F_CHOICE_LOCK_0,
    F_CHOICE_LOCK_1,
    F_VOLATILE_0,
    F_VOLATILE_1,
    F_LAST_MOVE_0,
    F_LAST_MOVE_1,
    F_DISABLE_0,
    F_DISABLE_1,
    F_DISABLE_TURNS_0,
    F_DISABLE_TURNS_1,
    F_EXTENDED_VOLATILE_0,
    F_EXTENDED_VOLATILE_1,
    EXT_VOL_TORMENT,
    EXT_VOL_MEAN_LOOK,
    EXT_VOL_PARTIAL_TRAP,
    CAT_STATUS,
    TYPE_FLYING,
    TYPE_GHOST,
    TYPE_STEEL,
    FLAG_SOUND,
    ITEM_ASSAULT_VEST,
    ITEM_SHED_SHELL,
    ABILITY_LEVITATE,
    NUM_BATTLE_ACTIONS,
    PHASE_BATTLE,
    PHASE_FORCED_SWITCH,
)
from pokepy.core.bitpack import get_taunt_turns, get_encore_turns, get_throat_chop_turns
from pokepy.effects.grounding import is_grounded

# Trapping ability constants (defined locally per the Showdown reference convention)
ABILITY_SHADOW_TAG = 23
ABILITY_ARENA_TRAP = 71
ABILITY_MAGNET_PULL = 42


def get_battle_move_mask(
    state: MultiFormatState, side: int, game_data
) -> tuple[np.ndarray, bool]:
    """Compute the legal move-slot mask for one side during the BATTLE phase.

    Returns `(move_mask, forced_struggle)`, where `move_mask` is bool[4] for
    move slots and `forced_struggle` is true when Showdown-style DisableMove
    rules leave no usable moves, so slot 0 becomes the Struggle placeholder.
    Hard move locks (lockedmove / charging / recharge) are applied later by
    `get_battle_action_mask()` and the battle engine itself.
    """
    battle = state.battle_state
    side_base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    active = int(battle[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])

    moves_arr = state.team_moves if side == 0 else state.opp_moves
    pp_arr = state.team_pp if side == 0 else state.opp_pp

    f_choice = OFF_FIELD + (F_CHOICE_LOCK_0 if side == 0 else F_CHOICE_LOCK_1)
    f_vol = OFF_FIELD + (F_VOLATILE_0 if side == 0 else F_VOLATILE_1)
    f_last = OFF_FIELD + (F_LAST_MOVE_0 if side == 0 else F_LAST_MOVE_1)
    f_disable = OFF_FIELD + (F_DISABLE_0 if side == 0 else F_DISABLE_1)
    f_disable_turns = OFF_FIELD + (
        F_DISABLE_TURNS_0 if side == 0 else F_DISABLE_TURNS_1
    )
    f_extvol = OFF_FIELD + (
        F_EXTENDED_VOLATILE_0 if side == 0 else F_EXTENDED_VOLATILE_1
    )

    # PP-based mask (Struggle if all out)
    active_pp = pp_arr[active]
    total_pp = int(np.sum(active_pp))
    if total_pp > 0:
        pp_mask = np.array([int(active_pp[i]) > 0 for i in range(4)], dtype=bool)
    else:
        pp_mask = np.array([True, False, False, False], dtype=bool)

    # Move categories (for Taunt / Assault Vest)
    move_cats = np.zeros(4, dtype=np.int8)
    for i in range(4):
        mid = int(moves_arr[active, i])
        if mid >= 0:
            move_cats[i] = int(np.asarray(game_data.move_category)[mid])
    non_status_mask = move_cats != CAT_STATUS

    # Taunt: no status moves
    volatile = int(battle[f_vol])
    is_taunted = get_taunt_turns(volatile) > 0
    taunt_mask = non_status_mask if is_taunted else np.ones(4, dtype=bool)

    # Assault Vest: no status moves
    active_item = int(battle[side_base + active * POKEMON_SIZE + 6])
    has_av = active_item == ITEM_ASSAULT_VEST
    vest_mask = non_status_mask if has_av else np.ones(4, dtype=bool)

    # Encore: only last move
    encore_turns = get_encore_turns(volatile)
    last_move = int(battle[f_last])
    encore_mask = np.zeros(4, dtype=bool)
    if 0 <= last_move < 4:
        encore_mask[last_move] = True
    encore_applicable = (
        encore_turns > 0 and 0 <= last_move < 4 and int(active_pp[last_move]) > 0
    )

    # Throat Chop: disable sound moves for the next turn after the hit lands.
    throat_chop_turns = get_throat_chop_turns(volatile)
    throat_chop_mask = np.ones(4, dtype=bool)
    if throat_chop_turns > 0:
        for i in range(4):
            mid = int(moves_arr[active, i])
            if (
                mid >= 0
                and (int(np.asarray(game_data.move_flags)[mid]) & FLAG_SOUND) != 0
            ):
                throat_chop_mask[i] = False

    # Choice lock — cleared if the holder no longer has a Choice item
    # (Trick / Switcheroo / Knock Off can change items mid-battle).
    # Showdown conditions.ts:choicelock onBeforeMove `if (!pokemon.getItem().isChoice)`
    # removes the lock at move-time. Pokepy lazily clears the lock here in
    # the action mask: if the active mon's current item isn't a Choice item,
    # treat as not locked.
    from pokepy.core.constants import (
        ITEM_CHOICE_BAND,
        ITEM_CHOICE_SPECS,
        ITEM_CHOICE_SCARF,
    )

    cur_item = int(battle[side_base + active * POKEMON_SIZE + 6])
    is_holding_choice = cur_item in (
        ITEM_CHOICE_BAND,
        ITEM_CHOICE_SPECS,
        ITEM_CHOICE_SCARF,
    )
    choice_lock = int(battle[f_choice])
    if not is_holding_choice:
        choice_lock = -1  # Item lost via Trick / Knock Off / etc.; read-only here.
    is_choice_locked = choice_lock >= 0
    choice_mask = np.zeros(4, dtype=bool)
    if 0 <= choice_lock < 4:
        choice_mask[choice_lock] = True
    # Showdown: choicelock onDisableMove disables ALL non-locked moves; if the
    # locked move itself runs out of PP, getMoves returns [] and the caller
    # later forces Struggle (sim/pokemon.ts:1070-1073). Keep the locked slot
    # selected here so step_battle_gen9 can see `move_pp <= 0` on the chosen
    # move and route through its existing `must_struggle` override.
    if is_choice_locked and 0 <= choice_lock < 4 and int(active_pp[choice_lock]) <= 0:
        pp_mask = np.zeros(4, dtype=bool)
        pp_mask[choice_lock] = True
        choice_mask = np.zeros(4, dtype=bool)
        choice_mask[choice_lock] = True

    # Disable
    disable_move = int(battle[f_disable])
    disable_turns = int(battle[f_disable_turns])
    is_disabled = disable_move >= 0 and disable_turns > 0
    disable_mask = np.ones(4, dtype=bool)
    if is_disabled and 0 <= disable_move < 4:
        disable_mask[disable_move] = False

    # Torment
    ext_vol = int(battle[f_extvol])
    has_torment = (ext_vol & EXT_VOL_TORMENT) != 0
    torment_mask = np.ones(4, dtype=bool)
    if has_torment and 0 <= last_move < 4:
        torment_mask[last_move] = False

    # cantusetwice (Gigaton Hammer 893, Blood Moon 901):
    # Showdown's `flags: { cantusetwice: 1 }` means the move can't be used
    # two turns in a row from the same slot. Mirrors `disable_mask`.
    cantusetwice_mask = np.ones(4, dtype=bool)
    if 0 <= last_move < 4:
        last_mid = int(moves_arr[active, last_move])
        if last_mid == 893 or last_mid == 901:
            cantusetwice_mask[last_move] = False

    # Combine. When Choice lock conflicts with Encore (Encore forces a
    # different slot than the Choice-locked one), Showdown removes the
    # Choice lock at move-time (conditions.ts:choicelock onBeforeMove
    # `if (encored && encored !== this.effectState.move)` clears it).
    # Mirror that here so the action mask doesn't collapse to 0.
    move_mask = (
        pp_mask
        & taunt_mask
        & vest_mask
        & disable_mask
        & cantusetwice_mask
        & throat_chop_mask
    )
    if is_choice_locked and encore_applicable and 0 <= choice_lock < 4:
        if not bool(encore_mask[choice_lock]):
            # Encore forces a non-locked move — drop the Choice lock (read-only).
            is_choice_locked = False
            choice_lock = -1
    if is_choice_locked:
        move_mask = move_mask & choice_mask
    if encore_applicable:
        move_mask = move_mask & encore_mask
    if has_torment and 0 <= last_move < 4:
        move_mask = move_mask & torment_mask

    # Showdown: if every selectable move is disabled (PP exhausted, Disable +
    # Choice lock conflict, Taunt + Assault Vest, etc.), getMoves returns []
    # and getMoveRequestData() exposes only Struggle. Pokepy's convention:
    # slot 0 is the Struggle slot when no moves are usable.
    forced_struggle = not move_mask.any()
    if forced_struggle:
        move_mask = np.array([True, False, False, False], dtype=bool)

    return move_mask, forced_struggle


def get_battle_action_mask(state: MultiFormatState, side: int, game_data) -> np.ndarray:
    """Compute the legal-action mask for one side during the BATTLE phase.

    Returns bool[10]: indices 0-3 are moves, 4-9 are switches to team slots 0-5.
    """
    battle = state.battle_state
    side_base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    opp_base = OFF_SIDE1 if side == 0 else OFF_SIDE0
    active = int(battle[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])

    moves_arr = state.team_moves if side == 0 else state.opp_moves
    f_extvol = OFF_FIELD + (
        F_EXTENDED_VOLATILE_0 if side == 0 else F_EXTENDED_VOLATILE_1
    )

    move_mask, _ = get_battle_move_mask(state, side, game_data)

    # Showdown pokemon.ts:getLockedMove / getMoves:
    # lockedmove, twoturnmove, and recharge are hard locks that force the
    # active mon's move choice and also mark the mon as trapped, so switching
    # is illegal while any of them is active.
    hard_locked = False

    # Lockedmove (Outrage / Petal Dance / Thrash / Raging Fury) override.
    # Find the slot containing the locked move id and force that slot.
    f_locked_move = OFF_MOVES + (M_LOCKED_MOVE_0 if side == 0 else M_LOCKED_MOVE_1)
    f_locked_turns = OFF_MOVES + (M_LOCKED_TURNS_0 if side == 0 else M_LOCKED_TURNS_1)
    locked_mid = int(battle[f_locked_move])
    locked_turns = int(battle[f_locked_turns])
    if locked_mid >= 0 and locked_turns > 0:
        locked_slot = -1
        for i in range(4):
            if int(moves_arr[active, i]) == locked_mid:
                locked_slot = i
                break
        if locked_slot >= 0:
            move_mask = np.zeros(4, dtype=bool)
            move_mask[locked_slot] = True
            hard_locked = True

    # Two-turn charging moves (Solar Beam / Solar Blade / Meteor Beam / etc.)
    # use OFF_META.M_CHARGING_* to store the committed move id for the strike
    # turn. Mirror Showdown's twoturnmove lock here.
    f_charging = OFF_META + (M_CHARGING_0 if side == 0 else M_CHARGING_1)
    charging_mid = int(battle[f_charging])
    if charging_mid >= 0:
        charging_slot = -1
        for i in range(4):
            if int(moves_arr[active, i]) == charging_mid:
                charging_slot = i
                break
        if charging_slot >= 0:
            move_mask = np.zeros(4, dtype=bool)
            move_mask[charging_slot] = True
            hard_locked = True

    # Mustrecharge: Showdown conditions.ts:364 forces the user to skip
    # the next turn (only "recharge" available). Pokepy convention: slot 0.
    f_recharge = OFF_MOVES + (M_RECHARGE_0 if side == 0 else M_RECHARGE_1)
    if int(battle[f_recharge]) > 0:
        move_mask = np.array([True, False, False, False], dtype=bool)
        hard_locked = True

    # Switches: not active, not fainted, hp > 0
    switch_mask = np.zeros(6, dtype=bool)
    for slot in range(6):
        if slot == active:
            continue
        poff = side_base + slot * POKEMON_SIZE
        hp = int(battle[poff + 1])
        flags = int(battle[poff + 15])
        fainted = (flags & 0x1) != 0
        if hp > 0 and not fainted:
            switch_mask[slot] = True

    # Trapping. Showdown's TrapPokemon event resolves to pokemon.tryTrap()
    # which calls runStatusImmunity('trapped'); Ghost-type is immune
    # (data/typechart.ts ghost: trapped: 3) so it can ALWAYS switch out.
    # Shed Shell unconditionally clears trapped (data/items.ts:5635). The
    # tryTrap event runs at the start of every choice resolution, so the
    # source of the trap is checked fresh each turn.
    opp_active = int(battle[OFF_META + (M_ACTIVE1 if side == 0 else M_ACTIVE0)])
    opp_ability = int(battle[opp_base + opp_active * POKEMON_SIZE + 5])
    own_ability = int(battle[side_base + active * POKEMON_SIZE + 5])
    own_types = int(battle[side_base + active * POKEMON_SIZE + 4])
    own_type1 = own_types & 0xFF
    own_type2 = (own_types >> 8) & 0xFF
    own_item = int(battle[side_base + active * POKEMON_SIZE + 6])

    is_ghost = own_type1 == TYPE_GHOST or own_type2 == TYPE_GHOST
    is_steel = own_type1 == TYPE_STEEL or own_type2 == TYPE_STEEL
    shadow_trapped = (
        opp_ability == ABILITY_SHADOW_TAG and own_ability != ABILITY_SHADOW_TAG
    )
    arena_trapped = opp_ability == ABILITY_ARENA_TRAP and is_grounded(
        battle, side_base + active * POKEMON_SIZE
    )
    magnet_trapped = opp_ability == ABILITY_MAGNET_PULL and is_steel
    # Mean Look / Block / Spider Web / Thousand Waves / Anchor Shot /
    # Spirit Shackle all set EXT_VOL_MEAN_LOOK on the target side.
    # Wrap / Bind / Whirlpool / Fire Spin / Sand Tomb / Magma Storm /
    # Infestation / Clamp / Snap Trap / Thunder Cage all set
    # EXT_VOL_PARTIAL_TRAP. Both check Ghost immunity in Showdown via
    # runStatusImmunity('trapped').
    ext_vol_t = int(battle[f_extvol])
    mean_look_trapped = (ext_vol_t & EXT_VOL_MEAN_LOOK) != 0
    partial_trapped = (ext_vol_t & EXT_VOL_PARTIAL_TRAP) != 0
    has_shed = own_item == ITEM_SHED_SHELL
    is_trapped = (
        (
            shadow_trapped
            or arena_trapped
            or magnet_trapped
            or mean_look_trapped
            or partial_trapped
        )
        and not has_shed
        and not is_ghost
    )
    if hard_locked:
        switch_mask = np.zeros(6, dtype=bool)
    if is_trapped:
        switch_mask = np.zeros(6, dtype=bool)

    return np.concatenate([move_mask, switch_mask])


def get_action_mask(state: MultiFormatState, side: int, game_data) -> np.ndarray:
    """Phase-dispatched action mask. Currently supports BATTLE and FORCED_SWITCH only."""
    phase = int(state.phase)
    if phase == PHASE_BATTLE:
        return get_battle_action_mask(state, side, game_data)
    if phase == PHASE_FORCED_SWITCH:
        # Switches only (positions 4-9). Showdown sim/side.ts:932 only
        # checks pokemon.trapped during a 'move' request, NOT during a
        # forced 'switch' request — fainting / U-turn / Eject Button
        # always allow choosing a replacement, regardless of Shadow Tag /
        # Mean Look / partial trap / etc. Recompute the switch mask here
        # without the trapping clauses.
        return _forced_switch_mask(state, side)
    # Other phases (teambuilder) — not implemented in pokepy v0
    return np.zeros(NUM_BATTLE_ACTIONS, dtype=bool)


def _forced_switch_mask(state: MultiFormatState, side: int) -> np.ndarray:
    """Switch-only mask for FORCED_SWITCH phase. Trapping is ignored
    (Showdown sim/side.ts:932 only enforces trapping in 'move' requests).
    Active slot is excluded — for KO this is the fainted slot, for
    U-turn the outgoing pivot. Both are not legal switch targets."""
    battle = state.battle_state
    side_base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    active = int(battle[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])
    out = np.zeros(NUM_BATTLE_ACTIONS, dtype=bool)
    for slot in range(6):
        if slot == active:
            continue
        poff = side_base + slot * POKEMON_SIZE
        hp = int(battle[poff + 1])
        flags = int(battle[poff + 15])
        fainted = (flags & 0x1) != 0
        if hp > 0 and not fainted:
            out[4 + slot] = True
    return out
