"""Gen4 parity smoke + T2 live-diff entry point."""

import sys
from pathlib import Path

import numpy as np
import pytest

from pokepy.core.bitpack import get_stealth_rock, get_sticky_web
from pokepy.core.constants import F_HAZARDS_1, OFF_FIELD
from pokepy.core.gen_profile import GEN4_PROFILE, profile_for_format
from pokepy.data.loader import load_game_data, load_move_effect_data
from pokepy.effects.hazards import apply_hazard_from_move
from pokepy.engine import get_engine, step_battle
from pokepy.env.battle_env import DEFAULT_TEAM, init_battle_state
from pokepy.utils.gen5_prng import Gen5PRNG

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from parity_heuristic_e2e import run_live_diff  # noqa: E402

METAMON_ROOT = Path(__file__).resolve().parents[4]
SHOWDOWN_BIN = METAMON_ROOT / "server" / "pokemon-showdown" / "pokemon-showdown"


def test_gen4_type_chart_matches_showdown_psychic_self_resist():
    from pokepy.core.constants import TYPE_PSYCHIC
    from pokepy.data.type_charts import load_type_chart_for_gen

    chart = load_type_chart_for_gen(4)
    assert float(chart[TYPE_PSYCHIC, TYPE_PSYCHIC]) == 0.5
    entry = get_engine(4)
    assert entry.profile is GEN4_PROFILE


def test_gen4_profile_disables_gen9_only_mechanics():
    assert not GEN4_PROFILE.has_terrain
    assert not GEN4_PROFILE.has_tera
    assert "stickyweb" not in GEN4_PROFILE.enabled_hazards
    assert GEN4_PROFILE.hazard_enabled("stealthrock")


def test_gen4_hazard_gating_blocks_sticky_web():
    # Sticky Web is gen6+; use gen9 move metadata with gen4 hazard flags.
    gd = load_game_data(gen=9)
    me = load_move_effect_data(gen=9)
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM, gd, seed=1, gen=9)
    bs = state.battle_state
    apply_hazard_from_move(
        bs,
        564,  # Sticky Web
        1,
        True,
        gd,
        me,
        enabled_hazards=GEN4_PROFILE.enabled_hazards,
    )
    assert get_sticky_web(int(bs[OFF_FIELD + F_HAZARDS_1])) == 0


def test_gen4_hazard_gating_allows_stealth_rock():
    gd = load_game_data(gen=4)
    me = load_move_effect_data(gen=4)
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM, gd, seed=1, gen=4)
    bs = state.battle_state
    apply_hazard_from_move(
        bs,
        446,  # Stealth Rock
        1,
        True,
        gd,
        me,
        enabled_hazards=GEN4_PROFILE.enabled_hazards,
    )
    assert get_stealth_rock(int(bs[OFF_FIELD + F_HAZARDS_1])) == 1


def test_gen4_step_battle_runs():
    profile = profile_for_format("gen4ou")
    gd = load_game_data(gen=4)
    me = load_move_effect_data(gen=4)
    from pokepy.data.type_charts import load_type_chart_for_gen

    chart = load_type_chart_for_gen(4)
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM, gd, seed=99, gen=4)
    prng = Gen5PRNG((99 & 0xFFFF, (99 >> 16) & 0xFFFF, 0, 0))
    for _ in range(5):
        r0, r1, done = step_battle(4, state, 0, 0, gd, me, chart, prng)
        assert np.isfinite(r0) and np.isfinite(r1)
        if done:
            break


@pytest.mark.skipif(
    not SHOWDOWN_BIN.exists(),
    reason="Showdown CLI required for live-diff",
)
def test_gen4_live_diff_harness_runs():
    """Smoke: T2 harness executes and returns comparable rows."""
    team = dict(
        species=[65],
        moves=[[94, 247, 60, 113]],
        abilities=[0],
        items=[0],
        tera_types=[0],
        levels=[100],
        evs=[[0, 0, 0, 0, 0, 0]],
        ivs=[[31, 31, 31, 31, 31, 31]],
        natures=["Serious"],
    )
    py_rows, show_rows, mismatch, meta = run_live_diff(
        team, team, seed=999, n_turns=5, gen=4
    )
    assert meta["returncode"] == 0, meta
    assert py_rows
    assert show_rows
    assert mismatch is None, mismatch
