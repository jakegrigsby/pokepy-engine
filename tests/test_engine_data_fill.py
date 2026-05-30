"""Verify pokepy data tables are loadable for all target generations."""

from __future__ import annotations

import pytest

from pokepy.core.gen_profile import registered_gens
from pokepy.data.loader import load_game_data, load_id_mappings, load_move_effect_data
from pokepy.data.type_charts import load_type_chart_for_gen


@pytest.mark.parametrize("gen", sorted(registered_gens()))
def test_game_data_loads(gen):
    gd = load_game_data(gen=gen)
    assert gd.species_base_stats.shape[0] > 0
    assert gd.move_base_power.shape[0] > 0
    mappings = load_id_mappings(gen=gen)
    assert mappings.species_to_idx
    assert mappings.move_to_idx
    me = load_move_effect_data(gen=gen)
    assert me.effect_type.shape[0] == gd.move_base_power.shape[0]
    chart = load_type_chart_for_gen(gen)
    assert chart.ndim == 2
