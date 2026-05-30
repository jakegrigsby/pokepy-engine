"""Switch-in, faint, forced-switch, and request model."""

from __future__ import annotations

from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np

from pokepy.core.constants import (
    M_ACTIVE0,
    M_ACTIVE1,
    OFF_META,
    OFF_SIDE0,
    OFF_SIDE1,
    PHASE_BATTLE,
    POKEMON_SIZE,
)
from pokepy.core.gen_profile import GenProfile, profile_for_gen
from pokepy.core.state import MultiFormatState
from pokepy.engine.dispatch import BitpackBattleContext, make_context
from pokepy.engine.switch_requests import SwitchRequest, resolve_switch_choices_sync


def run_switch_in_events(
    ctx: BitpackBattleContext,
    pokemon_offset: int,
    *,
    is_switch: bool = True,
) -> None:
    """Fire SwitchIn / AfterSwitchInSelf through dispatch + legacy effects."""
    from pokepy import effects as fx

    poff = int(pokemon_offset)
    ctx.single_event("SwitchIn", None, None, poff, relay_var=True)
    opp_active = ctx.battle[ctx.battle.shape[0] - 1]  # unused placeholder
    _ = opp_active
    if ctx.profile.has_abilities:
        from pokepy.core.constants import (
            OFF_META,
            M_ACTIVE0,
            M_ACTIVE1,
            OFF_SIDE0,
            OFF_SIDE1,
            POKEMON_SIZE,
        )

        if poff < OFF_SIDE1:
            opp_off = OFF_SIDE0 + int(ctx.battle[OFF_META + M_ACTIVE0]) * POKEMON_SIZE
            if poff >= OFF_SIDE0:
                opp_off = (
                    OFF_SIDE1 + int(ctx.battle[OFF_META + M_ACTIVE1]) * POKEMON_SIZE
                )
        else:
            opp_off = OFF_SIDE0 + int(ctx.battle[OFF_META + M_ACTIVE0]) * POKEMON_SIZE
        if int(ctx.battle[opp_off + 1]) > 0:
            fx.apply_switch_in_ability_with_trace_reaction(
                ctx.battle,
                poff,
                opp_off,
                is_switch,
                gen5_prng=ctx.gen5_prng,
                has_terrain=ctx.profile.has_terrain,
                ability_weather_limited=ctx.profile.ability_weather_limited,
            )
    from pokepy.core.constants import F_HAZARDS_0, F_HAZARDS_1, OFF_FIELD, OFF_SIDE1

    hazard_off = OFF_FIELD + (F_HAZARDS_1 if poff >= OFF_SIDE1 else F_HAZARDS_0)
    fx.apply_hazard_damage_on_switch(ctx.battle, poff, hazard_off)
    ctx.single_event("AfterSwitchInSelf", None, None, poff, relay_var=None)


def step_forced_switch_modular(
    state: MultiFormatState,
    action: int,
    side: int,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    gen5_prng,
    *,
    profile: Optional[GenProfile] = None,
) -> Tuple[np.float32, np.float32, bool]:
    """Forced-switch entry point implemented on modular bitpack dispatch."""
    prof = profile or profile_for_gen(getattr(state, "format_gen", 9) or 9)
    ctx = make_context(
        state,
        game_data,
        move_effects,
        type_chart,
        gen5_prng,
        profile=prof,
    )

    side_i = int(side)
    if side_i not in (0, 1):
        return np.float32(0.0), np.float32(0.0), bool(state.done)
    target_slot = int(action)
    if target_slot >= 4:
        target_slot -= 4
    target_slot = max(0, min(5, target_slot))

    active_meta = M_ACTIVE0 if side_i == 0 else M_ACTIVE1
    side_base = OFF_SIDE0 if side_i == 0 else OFF_SIDE1
    current_slot = int(state.battle_state[OFF_META + active_meta])
    current_off = side_base + current_slot * POKEMON_SIZE
    target_off = side_base + target_slot * POKEMON_SIZE

    # Validate forced-switch target: alive and not current active.
    if target_slot == current_slot or int(state.battle_state[target_off + 1]) <= 0:
        for s in range(6):
            if s == current_slot:
                continue
            cand_off = side_base + s * POKEMON_SIZE
            if int(state.battle_state[cand_off + 1]) > 0:
                target_slot = s
                target_off = cand_off
                break

    # Flip active pointers/flags.
    state.battle_state[current_off + 15] = np.int16(int(state.battle_state[current_off + 15]) & ~0x2)
    state.battle_state[target_off + 15] = np.int16(int(state.battle_state[target_off + 15]) | 0x2)
    state.battle_state[OFF_META + active_meta] = np.int16(target_slot)

    # Keep side order arrays in sync (Showdown front = active).
    order_arr = state.side_order0 if side_i == 0 else state.side_order1
    idx = -1
    for i in range(len(order_arr)):
        if int(order_arr[i]) == int(target_slot):
            idx = i
            break
    if idx > 0:
        old_front = int(order_arr[0])
        order_arr[0] = np.int8(target_slot)
        order_arr[idx] = np.int8(old_front)

    run_switch_in_events(ctx, target_off, is_switch=True)

    # Clear forced-switch bookkeeping and resume battle phase.
    state.forced_switch_side = np.int8(-1)
    state.phase = np.int8(PHASE_BATTLE)
    return np.float32(0.0), np.float32(0.0), bool(state.done)


__all__ = [
    "run_switch_in_events",
    "step_forced_switch_modular",
    "SwitchRequest",
    "resolve_switch_choices_sync",
]
