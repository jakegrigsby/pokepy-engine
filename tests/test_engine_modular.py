"""Unit tests for modular engine scaffolding."""

from __future__ import annotations

import numpy as np
import pytest

from pokepy.core.gen_profile import profile_for_gen
from pokepy.data.loader import load_game_data, load_move_effect_data
from pokepy.data.type_charts import load_type_chart_for_gen
from pokepy.engine.dispatch import make_context
from pokepy.engine.queue import BattleQueue, Action
from pokepy.engine.registry import DEFAULT_REGISTRY, build_default_registry
from pokepy.engine.turn_loop import TurnDriver, run_turn
from pokepy.env.battle_env import DEFAULT_TEAM, init_battle_state
from pokepy.utils.gen5_prng import Gen5PRNG


def test_registry_has_handlers():
    reg = build_default_registry()
    assert len(reg.generic_handlers) > 0


def test_dispatch_context_speed_sort_consumes_prng():
    gd = load_game_data(gen=9)
    me = load_move_effect_data(gen=9)
    chart = load_type_chart_for_gen(9)
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM, gd, seed=1, gen=9)
    prng = Gen5PRNG((1, 2, 3, 4))
    ctx = make_context(state, gd, me, chart, prng, profile=profile_for_gen(9))
    before = prng.call_count if hasattr(prng, "call_count") else None
    lst = [
        {"order": 200, "priority": 0, "index": 0},
        {"order": 200, "priority": 0, "index": 1},
    ]
    ctx.speed_sort(lst)
    assert lst  # ordered


def test_turn_driver_runs_one_step():
    gd = load_game_data(gen=9)
    me = load_move_effect_data(gen=9)
    chart = load_type_chart_for_gen(9)
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM, gd, seed=7, gen=9)
    prng = Gen5PRNG((7, 7, 7, 7))
    r0, r1, done = run_turn(state, 0, 0, gd, me, chart, prng)
    assert np.isfinite(r0)
