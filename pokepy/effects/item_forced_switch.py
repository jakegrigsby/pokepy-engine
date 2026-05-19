"""Eject Button + Red Card item-triggered forced switches.

Port of multi_format_fast_env.py lines 4685-4714. These run after damage
each turn:
- Eject Button: when the holder is damaged, force the holder to switch
  out (consumes the item).
- Red Card: when the holder is hit, force the ATTACKER to switch
  (consumes the item).

For player 0 (the agent), Eject Button → set FORCED_SWITCH phase. Red Card
on side 1 means the agent (side 0) is forced out — also FORCED_SWITCH.
For side 1 (opponent), auto-switch via fx.auto_switch.
"""

from __future__ import annotations

import numpy as np

from pokepy.core.bitpack import get_status, set_status
from pokepy.core.constants import (
    OFF_SIDE0, OFF_SIDE1, OFF_FIELD, OFF_META, OFF_MOVES, M_ACTIVE0, M_ACTIVE1, POKEMON_SIZE,
    ITEM_EJECT_BUTTON, ITEM_RED_CARD,
    M_PARTIAL_TRAP_TURNS_0, M_PARTIAL_TRAP_TURNS_1,
    M_ACTIVE_MOVE_ACTIONS_0, M_ACTIVE_MOVE_ACTIONS_1,
    F_CHOICE_LOCK_0, F_LAST_MOVE_0, F_DISABLE_0, F_VOLATILE_0, F_LEECH_SEED_0,
    F_DISABLE_TURNS_0, F_EXTENDED_VOLATILE_0, F_DESTINY_BOND_0, F_SUBSTITUTE_0,
    F_YAWN_TURNS_0, F_PERISH_COUNT_0, F_HAZARDS_0,
    F_CHOICE_LOCK_1, F_LAST_MOVE_1, F_DISABLE_1, F_VOLATILE_1, F_LEECH_SEED_1,
    F_DISABLE_TURNS_1, F_EXTENDED_VOLATILE_1, F_DESTINY_BOND_1, F_SUBSTITUTE_1,
    F_YAWN_TURNS_1, F_PERISH_COUNT_1, F_HAZARDS_1,
    EXT_VOL_MEAN_LOOK, EXT_VOL_PARTIAL_TRAP,
    NEUTRAL_BOOSTS_13, NEUTRAL_BOOSTS_14,
    STATUS_TOXIC, CAT_STATUS,
)
from pokepy.effects.switch_slot_conditions import (
    apply_pending_wish_on_switch_in,
    is_pending_wish_sentinel,
)

def _clear_opp_switch_state(battle: np.ndarray, new_active1: int) -> None:
    """Clear field-level state when the opponent (side 1) auto-switches via
    Eject Button / Red Card / Wimp Out / Emergency Exit. Mirrors the
    post-faint cleanup at battle_gen9.py:2540-2554 — Showdown wipes
    volatiles and side-condition slots in `pokemon.clearVolatile()` /
    `pokemon.switchIn()`. Without this, a Choice-locked opponent that
    Eject-Buttoned out would still appear locked to its old move slot.
    """
    for off in (F_CHOICE_LOCK_1, F_LAST_MOVE_1, F_DISABLE_1):
        battle[OFF_FIELD + off] = -1
    battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1] = 0
    for off in (F_VOLATILE_1, F_LEECH_SEED_1, F_DISABLE_TURNS_1, F_EXTENDED_VOLATILE_1,
                F_DESTINY_BOND_1, F_SUBSTITUTE_1, F_YAWN_TURNS_1, F_PERISH_COUNT_1):
        battle[OFF_FIELD + off] = 0
    _clear_opponent_source_tied_lock_state(battle, side=1)
    new_p1 = OFF_SIDE1 + int(new_active1) * POKEMON_SIZE
    tera = int(battle[new_p1 + 14]) & -4096
    battle[new_p1 + 13] = NEUTRAL_BOOSTS_13
    battle[new_p1 + 14] = (NEUTRAL_BOOSTS_14 & 4095) | tera

def _clear_switched_out_pokemon_state(battle: np.ndarray, pokemon_off: int) -> None:
    """Mirror Showdown's per-Pokemon clearVolatile() work on switch-out.

    Red Card / Eject Button drags should clear the outgoing Pokemon's
    switch-out volatile flags such as Flash Fire and Glaive Rush, and reset
    its temporary boosts while preserving tera bits.
    """
    from pokepy.core.constants import (
        FLAG_BOOSTER_ENERGY_ACTIVE as _FLAG_BOOSTER_SW,
        FLAG_CHARGE as _FLAG_CHARGE_SW,
        FLAG_GLAIVE_RUSH as _FLAG_GLAIVE_RUSH_SW,
    )

    _PARADOX_STAT_MASK_SW = 0x6010
    _SWITCH_OUT_CLEAR_MASK = (
        _FLAG_GLAIVE_RUSH_SW
        | 0x200
        | _FLAG_CHARGE_SW
        | 0x02
        | _FLAG_BOOSTER_SW
        | _PARADOX_STAT_MASK_SW
    )
    poff = int(pokemon_off)
    battle[poff + 13] = NEUTRAL_BOOSTS_13
    tera = int(battle[poff + 14]) & -4096
    battle[poff + 14] = (NEUTRAL_BOOSTS_14 & 4095) | tera
    battle[poff + 15] = int(battle[poff + 15]) & ~_SWITCH_OUT_CLEAR_MASK

def _clear_opponent_source_tied_lock_state(battle: np.ndarray, side: int) -> None:
    """Clear target-side locks that end when the switching source leaves."""
    if int(side) == 0:
        opp_ext_vol = F_EXTENDED_VOLATILE_1
        opp_partial_trap_turns = M_PARTIAL_TRAP_TURNS_1
    else:
        opp_ext_vol = F_EXTENDED_VOLATILE_0
        opp_partial_trap_turns = M_PARTIAL_TRAP_TURNS_0

    battle[OFF_FIELD + opp_ext_vol] = np.int16(
        int(battle[OFF_FIELD + opp_ext_vol]) & ~(EXT_VOL_MEAN_LOOK | EXT_VOL_PARTIAL_TRAP)
    )
    battle[OFF_MOVES + opp_partial_trap_turns] = 0

def _apply_opp_switch_in_effects(
    battle: np.ndarray,
    new_active1: int,
    game_data,
    opp_base_abilities,
    pending_switch_slot_condition1: int = 0,
    state=None,
    gen5_prng=None,
) -> None:
    """Run hazards + switch-in ability for an opponent mon brought in via
    Eject Button / Red Card. Showdown's switchIn fires `hazard onStart`,
    then runs onSwitchIn (sim/battle.ts:507 speedSort) for the new mon.
    Without this the new opponent's Intimidate / weather setter / Trace /
    Download / Booster Energy never resolve. The previous mon's effects
    do NOT re-fire because they didn't switch.
    """
    from pokepy.effects.hazards import apply_hazard_damage_on_switch
    from pokepy.effects.abilities import apply_switch_in_ability_with_trace_reaction
    from pokepy.effects.switch_state import reset_incoming_switch_state

    new_p1 = OFF_SIDE1 + int(new_active1) * POKEMON_SIZE
    reset_incoming_switch_state(
        battle,
        new_p1,
        game_data,
        base_ability=int(opp_base_abilities[int(new_active1)]),
        state=state,
    )
    consumed_pending_switch_slot_condition1 = apply_pending_wish_on_switch_in(
        battle,
        1,
        new_p1,
        state,
        game_data,
        pending_switch_slot_condition1,
    )
    if (
        is_pending_wish_sentinel(pending_switch_slot_condition1)
        and not consumed_pending_switch_slot_condition1
    ):
        battle[OFF_FIELD + F_DESTINY_BOND_1] = np.int16(pending_switch_slot_condition1)
    apply_hazard_damage_on_switch(battle, new_p1, OFF_FIELD + F_HAZARDS_1)
    if get_status(int(battle[new_p1 + 12])) == STATUS_TOXIC:
        battle[new_p1 + 12] = set_status(STATUS_TOXIC, 0)
    if int(battle[new_p1 + 1]) > 0:
        active0 = int(battle[OFF_META + M_ACTIVE0])
        opp_for_intim = OFF_SIDE0 + active0 * POKEMON_SIZE
        apply_switch_in_ability_with_trace_reaction(
            battle,
            new_p1,
            opp_for_intim,
            True,
            gen5_prng=gen5_prng,
        )

def _consume_opp_item_switch_pre_switchin_frames(
    battle: np.ndarray,
    new_active1: int,
    gen5_prng,
) -> None:
    """Mirror hidden tied-speed frames for same-turn side-1 item auto-switches.

    Opponent-side Eject Button / Red Card replacements resolve as a real
    `switch` action followed by a queued `runSwitch`. On a tied singles board,
    that continuation spends four hidden `random(0, 2)` frames before the next
    visible turn:
      1. switch action post-action `eachEvent('Update')`
      2. `runSwitch` `speedSort(allActive)`
      3. `fieldEvent('SwitchIn')` handler speed-sort on the live active pair
      4. `runSwitch` post-action `eachEvent('Update')`

    These first three use the incoming Pokemon's neutral on-entry speed before
    hazards / switch-in effects mutate it.
    """
    if gen5_prng is None:
        return

    from pokepy import effects as fx

    new_p1 = OFF_SIDE1 + int(new_active1) * POKEMON_SIZE
    active0 = int(battle[OFF_META + M_ACTIVE0])
    p0_off = OFF_SIDE0 + active0 * POKEMON_SIZE
    if int(battle[new_p1 + 1]) <= 0 or int(battle[p0_off + 1]) <= 0:
        return

    pre_switch_in_tie = fx.get_effective_speed(battle, new_p1) == fx.get_effective_speed(battle, p0_off)
    if pre_switch_in_tie:
        for _ in range(3):
            gen5_prng.random(0, 2)

def _finalize_opp_item_switch_resume_frames(
    battle: np.ndarray,
    new_active1: int,
    gen5_prng,
) -> None:
    """Spend the post-runSwitch Update tie frame after switch-in effects."""
    if gen5_prng is None:
        return

    from pokepy import effects as fx

    new_p1 = OFF_SIDE1 + int(new_active1) * POKEMON_SIZE
    active0 = int(battle[OFF_META + M_ACTIVE0])
    p0_off = OFF_SIDE0 + active0 * POKEMON_SIZE
    if int(battle[new_p1 + 1]) <= 0 or int(battle[p0_off + 1]) <= 0:
        return
    if fx.get_effective_speed(battle, new_p1) == fx.get_effective_speed(battle, p0_off):
        gen5_prng.random(0, 2)

def _consume_player_item_switch_pre_switchin_frames(
    battle: np.ndarray,
    new_active0: int,
    gen5_prng,
) -> None:
    """Mirror hidden tied-speed frames for same-turn side-0 drags.

    Player-side Red Card drags resolve through the same switch-action +
    queued `runSwitch` path as the symmetric opponent-side item auto-switch.
    Before switch-in effects mutate the replacement, Showdown spends three
    tied-speed comparator frames when the incoming player replacement ties the
    current opposing active:
      1. switch action post-action `eachEvent('Update')`
      2. `runSwitch` `speedSort(allActive)`
      3. `fieldEvent('SwitchIn')` handler speed-sort on the live active pair
    """
    if gen5_prng is None:
        return

    from pokepy import effects as fx

    new_p0 = OFF_SIDE0 + int(new_active0) * POKEMON_SIZE
    active1 = int(battle[OFF_META + M_ACTIVE1])
    p1_off = OFF_SIDE1 + active1 * POKEMON_SIZE
    if int(battle[new_p0 + 1]) <= 0 or int(battle[p1_off + 1]) <= 0:
        return

    pre_switch_in_tie = fx.get_effective_speed(battle, new_p0) == fx.get_effective_speed(battle, p1_off)
    if pre_switch_in_tie:
        for _ in range(3):
            gen5_prng.random(0, 2)

def _finalize_player_item_switch_resume_frames(
    battle: np.ndarray,
    new_active0: int,
    gen5_prng,
) -> None:
    """Spend the post-runSwitch Update tie frame after player switch-in effects."""
    if gen5_prng is None:
        return

    from pokepy import effects as fx

    new_p0 = OFF_SIDE0 + int(new_active0) * POKEMON_SIZE
    active1 = int(battle[OFF_META + M_ACTIVE1])
    p1_off = OFF_SIDE1 + active1 * POKEMON_SIZE
    if int(battle[new_p0 + 1]) <= 0 or int(battle[p1_off + 1]) <= 0:
        return
    if fx.get_effective_speed(battle, new_p0) == fx.get_effective_speed(battle, p1_off):
        gen5_prng.random(0, 2)

def _clear_player_switch_state(battle: np.ndarray, new_active0: int) -> None:
    """Clear field-level state when the player (side 0) is forcibly dragged."""
    for off in (F_CHOICE_LOCK_0, F_LAST_MOVE_0, F_DISABLE_0):
        battle[OFF_FIELD + off] = -1
    battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0] = 0
    for off in (F_VOLATILE_0, F_LEECH_SEED_0, F_DISABLE_TURNS_0, F_EXTENDED_VOLATILE_0,
                F_DESTINY_BOND_0, F_SUBSTITUTE_0, F_YAWN_TURNS_0, F_PERISH_COUNT_0):
        battle[OFF_FIELD + off] = 0
    _clear_opponent_source_tied_lock_state(battle, side=0)
    new_p0 = OFF_SIDE0 + int(new_active0) * POKEMON_SIZE
    tera = int(battle[new_p0 + 14]) & -4096
    battle[new_p0 + 13] = NEUTRAL_BOOSTS_13
    battle[new_p0 + 14] = (NEUTRAL_BOOSTS_14 & 4095) | tera

def _apply_player_switch_in_effects(
    battle: np.ndarray,
    new_active0: int,
    game_data,
    team_base_abilities,
    pending_switch_slot_condition0: int = 0,
    state=None,
    gen5_prng=None,
) -> None:
    """Run hazards + switch-in ability for a player mon brought in via drag."""
    from pokepy.effects.hazards import apply_hazard_damage_on_switch
    from pokepy.effects.abilities import apply_switch_in_ability_with_trace_reaction
    from pokepy.effects.switch_state import reset_incoming_switch_state

    new_p0 = OFF_SIDE0 + int(new_active0) * POKEMON_SIZE
    reset_incoming_switch_state(
        battle,
        new_p0,
        game_data,
        base_ability=int(team_base_abilities[int(new_active0)]),
        state=state,
    )
    consumed_pending_switch_slot_condition0 = apply_pending_wish_on_switch_in(
        battle,
        0,
        new_p0,
        state,
        game_data,
        pending_switch_slot_condition0,
    )
    if (
        is_pending_wish_sentinel(pending_switch_slot_condition0)
        and not consumed_pending_switch_slot_condition0
    ):
        battle[OFF_FIELD + F_DESTINY_BOND_0] = np.int16(pending_switch_slot_condition0)
    apply_hazard_damage_on_switch(battle, new_p0, OFF_FIELD + F_HAZARDS_0)
    if get_status(int(battle[new_p0 + 12])) == STATUS_TOXIC:
        battle[new_p0 + 12] = set_status(STATUS_TOXIC, 0)
    if int(battle[new_p0 + 1]) > 0:
        active1 = int(battle[OFF_META + M_ACTIVE1])
        opp_for_intim = OFF_SIDE1 + active1 * POKEMON_SIZE
        apply_switch_in_ability_with_trace_reaction(
            battle,
            new_p0,
            opp_for_intim,
            True,
            gen5_prng=gen5_prng,
        )

def _to_int16(val: int) -> int:
    val = int(val) & 0xFFFF
    if val >= 0x8000:
        val -= 0x10000
    return val

def _sync_showdown_order_on_switch(order: np.ndarray, new_active: int) -> None:
    new_active = int(new_active)
    idx = -1
    for i in range(len(order)):
        if int(order[i]) == new_active:
            idx = i
            break
    if idx <= 0:
        return
    old_front = int(order[0])
    order[0] = np.int8(new_active)
    order[idx] = np.int8(old_front)

def _random_drag_switch_slot(
    battle: np.ndarray,
    side_off: int,
    active_slot: int,
    order: np.ndarray,
    gen5_prng,
) -> int:
    """Choose a Showdown-style random switch target for drag effects.

    Red Card / Dragon Tail style drags use Battle.getRandomSwitchable(), which
    samples uniformly from switchable bench mons in team order. This is not the
    same as pokepy's deterministic auto-switch helper.
    """
    _sync_showdown_order_on_switch(order, active_slot)
    switchable: list[int] = []
    for i in range(1, len(order)):
        slot = int(order[i])
        if slot == int(active_slot):
            continue
        slot_off = side_off + slot * POKEMON_SIZE
        if int(battle[slot_off + 1]) > 0 and (int(battle[slot_off + 15]) & 1) == 0:
            switchable.append(slot)
    if not switchable:
        return int(active_slot)
    choice_idx = int(gen5_prng.random(len(switchable)))
    return int(switchable[choice_idx])

def _red_card_triggered(move_id: int, did_hit: bool, game_data) -> bool:
    """Showdown's Red Card hook keys off a landed non-status move, not raw damage.

    This matters for hits that consume the attack via Disguise / Ice Face:
    the holder can still use Red Card even when the move's damage bucket does
    not contribute normal HP loss for the attack itself.
    """
    move_id = int(move_id)
    if move_id < 0 or not bool(did_hit):
        return False
    return int(game_data.move_category[move_id]) != CAT_STATUS

def apply_item_forced_switch(
    battle: np.ndarray,
    move_id0: int,
    move_id1: int,
    user0_off: int,
    user1_off: int,
    target0_off: int,
    target1_off: int,
    damage0: int,
    damage1: int,
    hit0: bool,
    hit1: bool,
    order0: np.ndarray,
    order1: np.ndarray,
    gen5_prng,
    game_data,
    state,
) -> bool:
    """Apply Eject Button + Red Card item triggers. Mutates battle.

    Returns True if player 0 needs to be forced to switch (Eject Button or
    Red Card consumed on player side). The caller (battle_gen9) should set
    PHASE_FORCED_SWITCH when this returns True.

    Side 1 (opponent) auto-switches via fx.auto_switch.
    """
    from pokepy.effects import auto_switch as _auto_switch_fn

    user0_off = int(user0_off)
    user1_off = int(user1_off)
    target0_off = int(target0_off)
    target1_off = int(target1_off)
    damage0 = int(damage0)
    damage1 = int(damage1)
    hit0 = bool(hit0)
    hit1 = bool(hit1)

    # In battle_gen9.py:
    #   target0_off = p1_off  (side 0's target = opponent)
    #   target1_off = p0_off  (side 1's target = player)
    p0_off = target1_off
    p1_off = target0_off

    # ----- Side 1 (opponent) Eject Button: damaged by p0 → auto-switch -----
    # Showdown's switchIn fires `BeforeSwitchOut` for any voluntary or
    # eject-button switch, which calls Regenerator / Natural Cure on the
    # outgoing mon. Pokepy used to skip these for item-forced switches.
    from pokepy.effects.abilities import (
        apply_regenerator_on_switch_out as _regen_out,
        apply_natural_cure_on_switch_out as _natcure_out,
    )
    p1_item = int(battle[p1_off + 6])
    if (
        p1_item == ITEM_EJECT_BUTTON
        and damage0 > 0
        and int(battle[p1_off + 1]) > 0
        and hit0
    ):
        bench_alive1 = 0
        for i in range(6):
            slot_off = OFF_SIDE1 + i * POKEMON_SIZE
            if slot_off == p1_off:
                continue
            if int(battle[slot_off + 1]) > 0 and (int(battle[slot_off + 15]) & 1) == 0:
                bench_alive1 += 1
        if bench_alive1 > 0:
            _regen_out(battle, p1_off, True)
            _natcure_out(battle, p1_off, True)
            _clear_switched_out_pokemon_state(battle, p1_off)
            active1 = int(battle[OFF_META + M_ACTIVE1])
            new_active1 = _auto_switch_fn(battle, OFF_SIDE1, active1)
            if new_active1 != active1:
                pending_switch_slot_condition1 = int(battle[OFF_FIELD + F_DESTINY_BOND_1])
                battle[OFF_META + M_ACTIVE1] = np.int16(new_active1)
                _sync_showdown_order_on_switch(order1, new_active1)
                # Showdown clears the side's volatile / choicelock state when
                # any Pokemon switches in (sim/pokemon.ts:clearVolatile + switchIn).
                _clear_opp_switch_state(battle, new_active1)
                _consume_opp_item_switch_pre_switchin_frames(
                    battle,
                    new_active1,
                    gen5_prng,
                )
                _apply_opp_switch_in_effects(
                    battle,
                    new_active1,
                    game_data,
                    state.opp_abilities,
                    pending_switch_slot_condition1=pending_switch_slot_condition1,
                    state=state,
                    gen5_prng=gen5_prng,
                )
                _finalize_opp_item_switch_resume_frames(
                    battle,
                    new_active1,
                    gen5_prng,
                )
            battle[p1_off + 6] = 0  # consume

    # ----- Side 1 (opponent) Red Card: hit by p0 → opponent auto-switches the
    #       attacker (side 0/player). For player-side that means FORCED_SWITCH.
    #       But Red Card on opponent forces side 0 OUT — handled below.
    # ----- Side 0 (player) Eject Button: damaged by p1 → FORCED_SWITCH -----
    p0_needs_switch = False
    p0_item = int(battle[p0_off + 6])
    if (
        p0_item == ITEM_EJECT_BUTTON
        and damage1 > 0
        and int(battle[p0_off + 1]) > 0
        and hit1
    ):
        bench_alive = 0
        for i in range(6):
            slot_off = OFF_SIDE0 + i * POKEMON_SIZE
            if slot_off == p0_off:
                continue
            if int(battle[slot_off + 1]) > 0 and (int(battle[slot_off + 15]) & 1) == 0:
                bench_alive += 1
        if bench_alive > 0:
            battle[p0_off + 6] = 0
            p0_needs_switch = True

    # Sheer Force suppresses onAfterMoveSecondary → Red Card never fires.
    # Showdown data/abilities.ts sheerforce: `move.hasSheerForce = true`
    # which skips the after-move-secondary chain entirely for a move with
    # any secondary effect.
    _ABILITY_SHEER_FORCE_RC = 125
    _ABILITY_MAGIC_GUARD_RC = 98  # Magic Guard doesn't block Red Card, only
    _ABILITY_SUCTION_CUPS_RC = 21
    _ABILITY_GUARD_DOG_RC = 275

    # ----- Side 0 (player) Red Card: hit by p1 → opponent (attacker) is dragged
    p0_item = int(battle[p0_off + 6])
    user1_ab = int(battle[user1_off + 5])
    user1_sf = user1_ab == _ABILITY_SHEER_FORCE_RC  # crude: no per-move check
    if (
        p0_item == ITEM_RED_CARD
        and int(battle[p0_off + 1]) > 0
        and _red_card_triggered(move_id1, hit1, game_data)
        # Suction Cups / Guard Dog on attacker prevent forced switch.
        and user1_ab not in (_ABILITY_SUCTION_CUPS_RC, _ABILITY_GUARD_DOG_RC)
    ):
        active1 = int(battle[OFF_META + M_ACTIVE1])
        opp_off_for_regen = OFF_SIDE1 + active1 * POKEMON_SIZE
        _regen_out(battle, opp_off_for_regen, True)
        _natcure_out(battle, opp_off_for_regen, True)
        _clear_switched_out_pokemon_state(battle, opp_off_for_regen)
        new_active1 = _random_drag_switch_slot(battle, OFF_SIDE1, active1, order1, gen5_prng)
        if new_active1 != active1:
            pending_switch_slot_condition1 = int(battle[OFF_FIELD + F_DESTINY_BOND_1])
            battle[OFF_META + M_ACTIVE1] = np.int16(new_active1)
            _sync_showdown_order_on_switch(order1, new_active1)
            if hasattr(state, "hidden_opp_switches"):
                state.hidden_opp_switches.append(int(new_active1))
            _clear_opp_switch_state(battle, new_active1)
            _consume_opp_item_switch_pre_switchin_frames(
                battle,
                new_active1,
                gen5_prng,
            )
            _apply_opp_switch_in_effects(
                battle,
                new_active1,
                game_data,
                state.opp_abilities,
                pending_switch_slot_condition1=pending_switch_slot_condition1,
                state=state,
                gen5_prng=gen5_prng,
            )
            _finalize_opp_item_switch_resume_frames(
                battle,
                new_active1,
                gen5_prng,
            )
        battle[p0_off + 6] = 0  # consume Red Card

    # ----- Side 1 (opponent) Red Card: hit by p0 → player is dragged out
    p1_item = int(battle[p1_off + 6])
    user0_ab = int(battle[user0_off + 5])
    if (
        p1_item == ITEM_RED_CARD
        and int(battle[p1_off + 1]) > 0
        and _red_card_triggered(move_id0, hit0, game_data)
        and user0_ab not in (_ABILITY_SUCTION_CUPS_RC, _ABILITY_GUARD_DOG_RC)
    ):
        bench_alive = 0
        for i in range(6):
            slot_off = OFF_SIDE0 + i * POKEMON_SIZE
            if slot_off == p0_off:
                continue
            if int(battle[slot_off + 1]) > 0 and (int(battle[slot_off + 15]) & 1) == 0:
                bench_alive += 1
        if bench_alive > 0:
            battle[p1_off + 6] = 0
            active0 = int(battle[OFF_META + M_ACTIVE0])
            own_off_for_regen = OFF_SIDE0 + active0 * POKEMON_SIZE
            _regen_out(battle, own_off_for_regen, True)
            _natcure_out(battle, own_off_for_regen, True)
            new_active0 = _random_drag_switch_slot(battle, OFF_SIDE0, active0, order0, gen5_prng)
            if new_active0 != active0:
                pending_switch_slot_condition0 = int(battle[OFF_FIELD + F_DESTINY_BOND_0])
                battle[OFF_META + M_ACTIVE0] = np.int16(new_active0)
                _sync_showdown_order_on_switch(order0, new_active0)
                _clear_player_switch_state(battle, new_active0)
                _consume_player_item_switch_pre_switchin_frames(
                    battle,
                    new_active0,
                    gen5_prng,
                )
                _apply_player_switch_in_effects(
                    battle,
                    new_active0,
                    game_data,
                    state.team_abilities,
                    pending_switch_slot_condition0=pending_switch_slot_condition0,
                    state=state,
                    gen5_prng=gen5_prng,
                )
                _finalize_player_item_switch_resume_frames(
                    battle,
                    new_active0,
                    gen5_prng,
                )

    return p0_needs_switch
