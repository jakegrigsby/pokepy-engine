"""Tests for symmetric forced-switch decision points (subturns)."""

from __future__ import annotations

import pytest

from pokepy.core.constants import PHASE_FORCED_SWITCH, Phase
from pokepy.core.state import MultiFormatState
from pokepy.engine.battle_gen9 import (
    step_battle_gen9,
    step_battle_gen9_iter,
    step_forced_switch,
)
from pokepy.engine.switch_requests import SwitchRequest
from pokepy.obs.state_to_universal import state_to_universal_state


def test_battle_gen9_exports_generator_and_request():
    import inspect

    assert inspect.isgeneratorfunction(step_battle_gen9_iter)
    assert callable(step_battle_gen9)
    req = SwitchRequest((1,))
    assert req.sides == (1,)


def test_forced_switch_side_field_on_state():
    state = MultiFormatState.create_empty()
    assert int(state.forced_switch_side) == -1
    state.forced_switch_side = 1
    assert int(state.forced_switch_side) == 1


def test_state_to_universal_forced_switch_is_side_aware(gd, mappings):
    state = MultiFormatState.create_empty()
    state.phase = Phase.FORCED_SWITCH
    state.forced_switch_side = 1

    u0 = state_to_universal_state(
        state, gd, mappings, format_str="gen9ou", player_side=0
    )
    u1 = state_to_universal_state(
        state, gd, mappings, format_str="gen9ou", player_side=1
    )
    assert not u0.forced_switch
    assert u1.forced_switch

    state.forced_switch_side = 2
    u0 = state_to_universal_state(
        state, gd, mappings, format_str="gen9ou", player_side=0
    )
    u1 = state_to_universal_state(
        state, gd, mappings, format_str="gen9ou", player_side=1
    )
    assert u0.forced_switch
    assert u1.forced_switch


def test_sync_wrapper_auto_resolves_side1_post_faint(gd, me, type_chart, fresh_battle):
    """Sync API keeps side-1 post-faint inline (parity / battle_env back-compat)."""
    from conftest import MonSpec

    state, prng = fresh_battle(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=999,
    )
    battle = state.battle_state
    from pokepy.core.constants import (
        M_ACTIVE0,
        OFF_META,
        M_ACTIVE1,
        OFF_SIDE1,
        POKEMON_SIZE,
    )

    active1 = int(battle[OFF_META + M_ACTIVE1])
    p1 = OFF_SIDE1 + active1 * POKEMON_SIZE
    battle[p1 + 1] = 0  # faint side-1 active before turn ends bookkeeping

    # Minimal: set phase as if iter deferred side-1 faint.
    state.phase = PHASE_FORCED_SWITCH
    state.forced_switch_side = 1

    # Wrapper path is not used here; inline helper mirrors wrapper behavior.
    from pokepy.sim.event_turn import (
        inline_post_faint_switch_side1 as _inline_post_faint_switch_side1,
    )

    active0 = int(battle[OFF_META + M_ACTIVE0])
    prev_active1 = active1
    _inline_post_faint_switch_side1(state, gd, prng, active0, active1)
    new_active1 = int(battle[OFF_META + M_ACTIVE1])
    assert (
        new_active1 != prev_active1
        or int(battle[OFF_SIDE1 + new_active1 * POKEMON_SIZE + 1]) > 0
    )


def test_step_forced_switch_accepts_side1(gd, me, type_chart, fresh_battle):
    from conftest import MonSpec
    from pokepy.core.constants import OFF_META, M_ACTIVE1, OFF_SIDE1, POKEMON_SIZE

    state, prng = fresh_battle(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
    )
    battle = state.battle_state
    active1 = int(battle[OFF_META + M_ACTIVE1])
    battle[OFF_SIDE1 + active1 * POKEMON_SIZE + 1] = 0
    state.phase = PHASE_FORCED_SWITCH
    state.forced_switch_side = 1
    # Pick first bench slot (action encoding 4+slot).
    bench_slot = 1 if active1 == 0 else 0
    action = bench_slot + 4
    r0, r1, done = step_forced_switch(
        state,
        action,
        side=1,
        game_data=gd,
        move_effects=me,
        type_chart=type_chart,
        gen5_prng=prng,
    )
    assert int(state.forced_switch_side) == -1
    assert int(battle[OFF_META + M_ACTIVE1]) == bench_slot
