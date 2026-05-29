"""Gen2 parity smoke."""

import numpy as np

from pokepy.core.constants import (
    ABILITY_DROUGHT,
    F_WEATHER,
    OFF_FIELD,
    OFF_SIDE0,
    WEATHER_NONE,
)
from pokepy.core.gen_profile import GEN2_PROFILE
from pokepy.data.loader import load_game_data, load_move_effect_data
from pokepy.data.type_charts import load_type_chart_for_gen
from pokepy.engine import get_engine, step_battle
from pokepy.env.battle_env import DEFAULT_TEAM, init_battle_state
from pokepy.utils.gen5_prng import Gen5PRNG


def test_gen2_engine_registered():
    entry = get_engine(2)
    assert not entry.profile.has_abilities
    assert entry.profile.has_items
    assert not entry.profile.has_natures_evs
    assert entry.profile.phys_spec_mode == "type"
    assert entry.profile is GEN2_PROFILE


def test_gen2_init_strips_abilities():
    gd = load_game_data(gen=2)
    team = dict(
        DEFAULT_TEAM,
        abilities=[ABILITY_DROUGHT],
    )
    state = init_battle_state(team, team, gd, seed=1, gen=2)
    bs = state.battle_state
    assert int(bs[OFF_SIDE0 + 5]) == 0
    assert int(state.team_abilities[0]) == 0
    assert int(bs[OFF_FIELD + F_WEATHER]) == WEATHER_NONE


def test_gen2_step_battle_runs():
    gd = load_game_data(gen=2)
    me = load_move_effect_data(gen=2)
    chart = load_type_chart_for_gen(2)
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM, gd, seed=42, gen=2)
    prng = Gen5PRNG((42 & 0xFFFF, (42 >> 16) & 0xFFFF, 0, 0))
    for _ in range(5):
        r0, r1, done = step_battle(2, state, 0, 0, gd, me, chart, prng)
        assert np.isfinite(r0) and np.isfinite(r1)
        if done:
            break
