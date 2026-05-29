"""Smoke tests for multi-gen engine registration and data loading."""

from __future__ import annotations

import numpy as np
import pytest

from pokepy.core.gen_profile import (
    GEN1_PROFILE,
    GEN4_PROFILE,
    GEN9_PROFILE,
    profile_for_gen,
    registered_gens,
)
from pokepy.data.loader import load_game_data, load_id_mappings
from pokepy.data.type_charts import load_type_chart_for_gen
from pokepy.engine import ENGINE_REGISTRY, get_engine, step_battle
from pokepy.env.battle_env import DEFAULT_TEAM, init_battle_state
from pokepy.utils.gen5_prng import Gen5PRNG


@pytest.mark.parametrize("gen", [1, 2, 3, 4, 9])
def test_registered_gen_has_engine_and_data(gen):
    assert gen in registered_gens()
    entry = get_engine(gen)
    assert entry.profile.gen == gen
    gd = load_game_data(gen=gen)
    assert gd.type_chart.shape[0] >= 15
    load_id_mappings(gen=gen)
    chart = load_type_chart_for_gen(gen)
    assert chart.ndim == 2


def test_gen9_profile_matches_legacy_crit():
    assert GEN9_PROFILE.crit_damage_mult == 1.5
    assert GEN9_PROFILE.crit_denom_for_stage(1) == 24
    assert GEN9_PROFILE.crit_denom_for_stage(4) == 1


def test_gen4_profile_flags():
    assert not GEN4_PROFILE.has_tera
    assert not GEN4_PROFILE.has_teampreview
    assert GEN4_PROFILE.crit_damage_mult == 2.0
    assert "stealthrock" in GEN4_PROFILE.enabled_hazards


def test_gen1_profile_flags():
    assert not GEN1_PROFILE.has_items
    assert not GEN1_PROFILE.has_abilities
    assert GEN1_PROFILE.phys_spec_mode == "type"


@pytest.mark.parametrize("gen", [1, 2, 3, 4, 9])
def test_step_battle_smoke(gen):
    gd = load_game_data(gen=gen)
    me = __import__(
        "pokepy.data.loader", fromlist=["load_move_effect_data"]
    ).load_move_effect_data(gen=gen)
    chart = load_type_chart_for_gen(gen)
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM, gd, seed=42, gen=gen)
    prng = Gen5PRNG((42 & 0xFFFF, (42 >> 16) & 0xFFFF, 0, 0))
    r0, r1, done = step_battle(
        gen, state, 0, 0, gd, me, chart, prng
    )
    assert np.isfinite(r0) and np.isfinite(r1)
