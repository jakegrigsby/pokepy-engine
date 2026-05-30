"""Single entry point for all generation turn execution."""

from __future__ import annotations

from typing import Dict, Generator, Tuple

import numpy as np

from pokepy.core.constants import M_ACTIVE0, M_ACTIVE1, OFF_META, Phase
from pokepy.core.gen_profile import GenProfile
from pokepy.core.state import MultiFormatState
from pokepy.engine.switch_requests import SwitchRequest, resolve_switch_choices_sync
from pokepy.sim.battle import Battle
from pokepy.sim.helpers import consume_startup_prng
from pokepy.sim.mods import apply_mod


def _make_battle(
    state: MultiFormatState,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    gen5_prng,
    *,
    profile: GenProfile,
) -> Battle:
    apply_mod(profile)
    return Battle(
        state,
        profile,
        gen5_prng,
        game_data=game_data,
        move_effects=move_effects,
        type_chart=type_chart,
    )


def inline_post_faint_switch_side1(
    state, game_data, gen5_prng, active0, active1, profile=None
):
    """Auto-switch side 1 after EOT faint (sync wrapper back-compat)."""
    from pokepy.engine.switch_requests import auto_switch_slot
    from pokepy.core.constants import M_ACTIVE1, OFF_META, Phase
    from pokepy.core.gen_profile import GEN9_PROFILE
    from pokepy.sim.battle import Battle
    from pokepy.sim.mods import apply_mod

    if profile is None:
        profile = GEN9_PROFILE
    apply_mod(profile)
    battle = Battle(
        state,
        profile,
        gen5_prng,
        game_data=game_data,
        move_effects=None,
        type_chart=None,
    )
    slot = auto_switch_slot(state.battle_state, 1, active1, order=state.side_order1)
    battle.switch_in(1, slot, forced=True)
    state.forced_switch_side = np.int8(-1)
    state.phase = np.int8(Phase.BATTLE)


def step_turn_event(
    state: MultiFormatState,
    action0: int,
    action1: int,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    gen5_prng,
    *,
    profile: GenProfile,
    defer_p1_forced_switch: bool = False,
    resolve_mid_turn_switch0=None,
    wants_tera0: bool = False,
    wants_tera1: bool = False,
    **kwargs,
) -> Tuple[np.float32, np.float32, bool]:
    consume_startup_prng(state, profile, gen5_prng)
    state.pending_opp_switch_in_slot = np.int8(-1)
    state.pending_opp_switch_action_speed = np.int16(0)
    battle = _make_battle(
        state, game_data, move_effects, type_chart, gen5_prng, profile=profile
    )
    if not profile.has_tera:
        wants_tera0 = wants_tera1 = False
    battle.wants_tera = (bool(wants_tera0), bool(wants_tera1))
    battle.commit_choices(action0, action1)
    gen = battle.turn_loop_iter(resolve_mid_turn_switch0=resolve_mid_turn_switch0)
    try:
        req = next(gen)
        while True:
            choices = resolve_switch_choices_sync(
                state,
                state.battle_state,
                req,
                side_order0=state.side_order0,
                side_order1=state.side_order1,
                resolve_mid_turn_switch0=resolve_mid_turn_switch0,
            )
            req = gen.send(choices)
    except StopIteration as stop:
        result = stop.value
    if (
        not defer_p1_forced_switch
        and int(state.phase) == Phase.FORCED_SWITCH
        and int(state.forced_switch_side) in (1, 2)
    ):
        from pokepy.engine.switch_requests import auto_switch_slot

        battle_obj = state.battle_state
        if int(state.forced_switch_side) == 2:
            for side in (1, 0):
                active = int(
                    battle_obj[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)]
                )
                slot = auto_switch_slot(
                    battle_obj,
                    side,
                    active,
                    order=state.side_order1 if side == 1 else state.side_order0,
                )
                battle.switch_in(side, slot, forced=True)
            state.forced_switch_side = np.int8(0)
        else:
            active1 = int(battle_obj[OFF_META + M_ACTIVE1])
            slot = auto_switch_slot(battle_obj, 1, active1, order=state.side_order1)
            battle.switch_in(1, slot, forced=True)
            state.forced_switch_side = np.int8(-1)
            state.phase = np.int8(Phase.BATTLE)
    return result


def step_turn_event_iter(
    state: MultiFormatState,
    action0: int,
    action1: int,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    gen5_prng,
    *,
    profile: GenProfile,
    resolve_mid_turn_switch0=None,
    wants_tera0: bool = False,
    wants_tera1: bool = False,
    **kwargs,
) -> Generator[SwitchRequest, Dict[int, int], Tuple[np.float32, np.float32, bool]]:
    consume_startup_prng(state, profile, gen5_prng)
    state.pending_opp_switch_in_slot = np.int8(-1)
    state.pending_opp_switch_action_speed = np.int16(0)
    battle = _make_battle(
        state, game_data, move_effects, type_chart, gen5_prng, profile=profile
    )
    if not profile.has_tera:
        wants_tera0 = wants_tera1 = False
    battle.wants_tera = (bool(wants_tera0), bool(wants_tera1))
    battle.commit_choices(action0, action1)
    loop = battle.turn_loop_iter(resolve_mid_turn_switch0=resolve_mid_turn_switch0)
    try:
        req = next(loop)
        while True:
            choices = yield req
            req = loop.send(choices)
    except StopIteration as stop:
        return stop.value


def step_forced_switch_event(
    state: MultiFormatState,
    action: int,
    side: int,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    gen5_prng,
    *,
    profile: GenProfile | None = None,
) -> Tuple[np.float32, np.float32, bool]:
    if profile is None:
        from pokepy.core.gen_profile import GEN9_PROFILE

        profile = GEN9_PROFILE
    apply_mod(profile)
    battle = Battle(
        state,
        profile,
        gen5_prng,
        game_data=game_data,
        move_effects=move_effects,
        type_chart=type_chart,
    )
    return battle.execute_forced_switch(action, side)
