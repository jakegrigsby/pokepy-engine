"""Gen3 parity smoke — full golden suite requires team pool + Showdown CLI."""

import numpy as np

from pokepy.core.bitpack import get_spikes_layers, get_stealth_rock
from pokepy.core.constants import (
    ABILITY_DROUGHT,
    F_HAZARDS_1,
    F_WEATHER,
    M_WEATHER_TURNS,
    OFF_FIELD,
    OFF_META,
    OFF_SIDE0,
    OFF_SIDE1,
    WEATHER_SUN,
)
from pokepy.core.gen_profile import GEN3_PROFILE, GEN9_PROFILE
from pokepy.data.loader import load_game_data, load_move_effect_data
from pokepy.effects.abilities import apply_switch_in_ability
from pokepy.effects.hazards import apply_hazard_from_move
from pokepy.engine import get_engine, step_battle
from pokepy.env.battle_env import DEFAULT_TEAM, init_battle_state
from pokepy.utils.gen5_prng import Gen5PRNG


def test_gen3_engine_registered():
    entry = get_engine(3)
    assert entry.profile.phys_spec_mode == "type"
    assert entry.profile is GEN3_PROFILE


def test_gen3_profile_disables_gen4_plus_mechanics():
    assert not GEN3_PROFILE.has_terrain
    assert not GEN3_PROFILE.has_tera
    assert "stealthrock" not in GEN3_PROFILE.enabled_hazards
    assert "spikes" in GEN3_PROFILE.enabled_hazards
    assert not GEN3_PROFILE.ability_weather_limited


def test_gen3_hazard_gating_blocks_stealth_rock():
    # Stealth Rock is gen4+; use gen9 move metadata with gen3 hazard flags.
    gd = load_game_data(gen=9)
    me = load_move_effect_data(gen=9)
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM, gd, seed=1, gen=9)
    bs = state.battle_state
    apply_hazard_from_move(
        bs,
        446,  # Stealth Rock
        1,
        True,
        gd,
        me,
        enabled_hazards=GEN3_PROFILE.enabled_hazards,
    )
    assert get_stealth_rock(int(bs[OFF_FIELD + F_HAZARDS_1])) == 0


def test_gen3_hazard_gating_allows_spikes():
    gd = load_game_data(gen=3)
    me = load_move_effect_data(gen=3)
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM, gd, seed=1, gen=3)
    bs = state.battle_state
    apply_hazard_from_move(
        bs,
        191,  # Spikes
        1,
        True,
        gd,
        me,
        enabled_hazards=GEN3_PROFILE.enabled_hazards,
    )
    assert get_spikes_layers(int(bs[OFF_FIELD + F_HAZARDS_1])) == 1


def test_gen3_drought_sets_permanent_sun():
    gd = load_game_data(gen=3)
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM, gd, seed=1, gen=3)
    bs = state.battle_state
    p0 = OFF_SIDE0
    bs[p0 + 5] = ABILITY_DROUGHT
    apply_switch_in_ability(
        bs,
        p0,
        OFF_SIDE1,
        did_switch=True,
        ability_weather_limited=GEN3_PROFILE.ability_weather_limited,
    )
    assert int(bs[OFF_FIELD + F_WEATHER]) == WEATHER_SUN
    assert int(bs[OFF_META + M_WEATHER_TURNS]) == 0


def test_gen9_drought_still_limited():
    gd = load_game_data(gen=9)
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM, gd, seed=1, gen=9)
    bs = state.battle_state
    p0 = OFF_SIDE0
    bs[p0 + 5] = ABILITY_DROUGHT
    apply_switch_in_ability(
        bs,
        p0,
        OFF_SIDE1,
        did_switch=True,
        ability_weather_limited=GEN9_PROFILE.ability_weather_limited,
    )
    assert int(bs[OFF_META + M_WEATHER_TURNS]) == 5


def test_gen3_step_battle_runs():
    gd = load_game_data(gen=3)
    me = load_move_effect_data(gen=3)
    from pokepy.data.type_charts import load_type_chart_for_gen

    chart = load_type_chart_for_gen(3)
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM, gd, seed=99, gen=3)
    prng = Gen5PRNG((99 & 0xFFFF, (99 >> 16) & 0xFFFF, 0, 0))
    for _ in range(5):
        r0, r1, done = step_battle(3, state, 0, 0, gd, me, chart, prng)
        assert np.isfinite(r0) and np.isfinite(r1)
        if done:
            break
