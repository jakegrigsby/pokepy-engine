"""Gen1 parity smoke."""

import numpy as np

from pokepy.core.gen_profile import GEN1_PROFILE
from pokepy.data.loader import load_game_data, load_move_effect_data
from pokepy.data.type_charts import load_type_chart_for_gen
from pokepy.engine import get_engine, step_battle
from pokepy.env.battle_env import DEFAULT_TEAM, init_battle_state
from pokepy.utils.gen5_prng import Gen5PRNG


def test_gen1_engine_registered():
    entry = get_engine(1)
    assert entry.profile.format_id == 0
    assert not entry.profile.has_items
    assert not entry.profile.has_abilities
    assert entry.profile.phys_spec_mode == "type"
    assert entry.profile is GEN1_PROFILE


def test_gen1_init_uses_combined_special():
    gd = load_game_data(gen=1)
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM, gd, seed=1, gen=1)
    bs = state.battle_state
    assert int(bs[0 + 5]) == 0  # no ability
    assert int(bs[0 + 6]) == 0  # no item
    spa = int(bs[0 + 9])
    spd = int(bs[0 + 10])
    assert spa == spd
    assert spa > 0


def test_gen1_step_battle_runs():
    gd = load_game_data(gen=1)
    me = load_move_effect_data(gen=1)
    chart = load_type_chart_for_gen(1)
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM, gd, seed=7, gen=1)
    prng = Gen5PRNG((7 & 0xFFFF, (7 >> 16) & 0xFFFF, 0, 0))
    for _ in range(5):
        r0, r1, done = step_battle(1, state, 0, 0, gd, me, chart, prng)
        assert np.isfinite(r0) and np.isfinite(r1)
        if done:
            break
