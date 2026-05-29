"""T2 live-diff parity: pokepy vs Showdown gen mods (multigen).

Run with Showdown CLI at server/pokemon-showdown/pokemon-showdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ENGINE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ENGINE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from parity_heuristic_e2e import (  # noqa: E402
    _parse_showdown_log,
    compare_battle_rows,
    run_live_diff,
)

METAMON_ROOT = Path(__file__).resolve().parents[4]
SHOWDOWN_BIN = METAMON_ROOT / "server" / "pokemon-showdown" / "pokemon-showdown"

pytestmark = pytest.mark.skipif(
    not SHOWDOWN_BIN.exists(),
    reason="Showdown CLI required for T2 live-diff tests",
)

_ZERO_EV = [[0, 0, 0, 0, 0, 0]]
_MAX_IV = [[31, 31, 31, 31, 31, 31]]


def _single_mon_team(
    species: int,
    moves: list,
    *,
    ability: int = 0,
    item: int = 0,
    nature: str = "Serious",
) -> dict:
    return dict(
        species=[species],
        moves=[moves],
        abilities=[ability],
        items=[item],
        tera_types=[0],
        levels=[100],
        evs=_ZERO_EV,
        ivs=_MAX_IV,
        natures=[nature],
    )


GEN4_ALAKAZAM_TEAM = _single_mon_team(65, [94, 247, 60, 113])  # Psychic, SB, Psybeam, LS
GEN4_STARMIE_TEAM = _single_mon_team(121, [56, 85, 94, 58], nature="Timid")  # Surf, Tbolt, Psychic, Ice Beam
GEN4_GENGAR_TEAM = _single_mon_team(94, [247, 126, 194, 85], nature="Timid")  # SB, Dream Eater, Shadow Punch, Tbolt

# Gen3 Psychic-only mirror (Shadow Ball is gen4+ in packed export).
GEN3_ALAKAZAM_TEAM = _single_mon_team(65, [94, 60, 113, 57])  # Psychic, Psybeam, LS, Surf
GEN3_STARMIE_TEAM = _single_mon_team(121, [56, 85, 94, 58], nature="Timid")  # Surf, Tbolt, Psychic, Ice Beam

# Gen2 Psychic-only mirror (same move set as gen3).
GEN2_ALAKAZAM_TEAM = _single_mon_team(65, [94, 60, 113, 57])  # Psychic, Psybeam, LS, Surf
GEN2_STARMIE_TEAM = _single_mon_team(121, [56, 85, 94, 58], nature="Timid")
GEN2_GENGAR_TEAM = _single_mon_team(94, [94, 85, 126, 137], nature="Timid")  # Psychic, Tbolt, Dream Eater, Night Shade

# Gen1 Psychic-only mirror (Surf is gen1-valid).
GEN1_ALAKAZAM_TEAM = _single_mon_team(65, [94, 60, 113, 57])  # Psychic, Psybeam, LS, Surf
GEN1_STARMIE_TEAM = _single_mon_team(121, [56, 85, 94, 58], nature="Timid")
GEN1_GENGAR_TEAM = _single_mon_team(94, [94, 85, 126, 137], nature="Timid")


def test_showdown_log_parser_reads_gen4_active_damage():
    raw = """
|switch|p1a: Alakazam|Alakazam, M|251/251
|switch|p1a: Alakazam|Alakazam, M|100/100
|switch|p2a: Alakazam|Alakazam, M|251/251
|turn|1
|move|p1a: Alakazam|Psychic|p2a: Alakazam
|split|p2
|-damage|p2a: Alakazam|176/251
|-damage|p2a: Alakazam|71/100
|move|p2a: Alakazam|Psychic|p1a: Alakazam
|split|p1
|-damage|p1a: Alakazam|176/251
|-damage|p1a: Alakazam|71/100
|turn|2
"""
    rows = _parse_showdown_log(raw)
    by_turn = {row["turn"]: row for row in rows}
    assert by_turn[1]["p0_hp"] == 176
    assert by_turn[1]["p0_max_hp"] == 251
    assert by_turn[1]["p1_hp"] == 176
    assert by_turn[1]["p1_max_hp"] == 251
    assert by_turn[1]["p0_action"] == "move Psychic"
    assert by_turn[1]["p1_action"] == "move Psychic"


def test_showdown_log_parser_records_faint_hp():
    raw = """
|switch|p1a: Alakazam|Alakazam, M|251/251
|switch|p2a: Alakazam|Alakazam, M|251/251
|turn|1
|move|p1a: Alakazam|Psychic|p2a: Alakazam
|-damage|p2a: Alakazam|167/251
|turn|2
|move|p1a: Alakazam|Psychic|p2a: Alakazam
|-damage|p2a: Alakazam|0 fnt
|faint|p2a: Alakazam
"""
    rows = _parse_showdown_log(raw)
    assert rows[-1]["turn"] == 2
    assert rows[-1]["p1_hp"] == 0


def test_compare_battle_rows_reports_first_hp_mismatch():
    py_rows = [{"type": "normal", "turn": 1, "p0_hp": 110, "p0_max_hp": 251, "p0_status": 0,
                "p1_hp": 119, "p1_max_hp": 251, "p1_status": 0}]
    show_rows = [{"type": "normal", "turn": 1, "p0_hp": 176, "p0_max_hp": 251, "p0_status": 0,
                  "p1_hp": 176, "p1_max_hp": 251, "p1_status": 0}]
    msg = compare_battle_rows(py_rows, show_rows)
    assert msg is not None
    assert "turn 1" in msg
    assert "p0_hp" in msg


def test_gen4_calc_damage_psychic_matches_showdown_roll():
    """Isolated damage calc parity for gen4 Psychic (type chart + formula)."""
    from pokepy.core.gen_profile import GEN4_PROFILE
    from pokepy.data.loader import load_game_data, load_move_effect_data
    from pokepy.data.type_charts import load_type_chart_for_gen
    from pokepy.env.battle_env import init_battle_state
    from pokepy.mechanics.damage_gen9 import calc_damage_gen9
    from pokepy.utils.gen5_prng import Gen5PRNG

    gd = load_game_data(gen=4)
    me = load_move_effect_data(gen=4)
    chart = load_type_chart_for_gen(4)
    state = init_battle_state(
        GEN4_ALAKAZAM_TEAM, GEN4_ALAKAZAM_TEAM, gd, seed=999, gen=4
    )
    prng = Gen5PRNG((999 & 0xFFFF, (999 >> 16) & 0xFFFF, 0, 0))
    dmg = calc_damage_gen9(
        state.battle_state,
        0,
        0,
        state.team_moves,
        state.opp_moves,
        gd,
        me,
        chart,
        gen5_prng=prng,
        profile=GEN4_PROFILE,
        override_move_id=94,
    )
    assert dmg == 75


def test_gen4_live_diff_alakazam_psychic_turn1():
    """Gen4 T2 gate: turn-1 end HP/status must match Showdown replay."""
    py_rows, show_rows, mismatch, meta = run_live_diff(
        GEN4_ALAKAZAM_TEAM,
        GEN4_ALAKAZAM_TEAM,
        seed=999,
        n_turns=1,
        gen=4,
    )
    assert meta["returncode"] == 0, meta
    assert not meta["timeout"]
    assert show_rows
    assert py_rows
    assert mismatch is None, mismatch


@pytest.mark.parametrize("seed", [999, 12345, 424242])
def test_gen4_live_diff_alakazam_mirror_full_battle(seed: int):
    """Gen4 mirror Psychic spam should match Showdown through battle end."""
    py_rows, show_rows, mismatch, meta = run_live_diff(
        GEN4_ALAKAZAM_TEAM,
        GEN4_ALAKAZAM_TEAM,
        seed=seed,
        n_turns=20,
        gen=4,
    )
    assert meta["returncode"] == 0, meta
    assert not meta["timeout"]
    assert show_rows
    assert py_rows
    assert mismatch is None, mismatch


@pytest.mark.parametrize(
    ("team", "seed"),
    [
        (GEN4_STARMIE_TEAM, 999),
        (GEN4_STARMIE_TEAM, 12345),
        (GEN4_GENGAR_TEAM, 999),
        (GEN4_GENGAR_TEAM, 12345),
    ],
)
def test_gen4_live_diff_special_mirror_full_battle(team, seed: int):
    """Additional gen4 special-mirror scenarios through battle end."""
    py_rows, show_rows, mismatch, meta = run_live_diff(
        team,
        team,
        seed=seed,
        n_turns=20,
        gen=4,
    )
    assert meta["returncode"] == 0, meta
    assert not meta["timeout"]
    assert show_rows
    assert py_rows
    assert mismatch is None, mismatch


@pytest.mark.parametrize("seed", [999, 12345, 424242, 500])
def test_gen3_live_diff_alakazam_mirror_full_battle(seed: int):
    """Gen3 mirror Psychic spam should match Showdown through battle end."""
    py_rows, show_rows, mismatch, meta = run_live_diff(
        GEN3_ALAKAZAM_TEAM,
        GEN3_ALAKAZAM_TEAM,
        seed=seed,
        n_turns=20,
        gen=3,
    )
    assert meta["returncode"] == 0, meta
    assert not meta["timeout"]
    assert show_rows
    assert py_rows
    assert mismatch is None, mismatch


@pytest.mark.parametrize(
    ("team", "seed"),
    [
        (GEN3_STARMIE_TEAM, 999),
        (GEN3_STARMIE_TEAM, 12345),
    ],
)
def test_gen3_live_diff_special_mirror_full_battle(team, seed: int):
    """Additional gen3 special-mirror scenarios through battle end."""
    py_rows, show_rows, mismatch, meta = run_live_diff(
        team,
        team,
        seed=seed,
        n_turns=20,
        gen=3,
    )
    assert meta["returncode"] == 0, meta
    assert not meta["timeout"]
    assert show_rows
    assert py_rows
    assert mismatch is None, mismatch


@pytest.mark.parametrize("seed", [999, 12345, 424242, 500])
def test_gen2_live_diff_alakazam_mirror_full_battle(seed: int):
    """Gen2 mirror Psychic spam should match Showdown through battle end."""
    py_rows, show_rows, mismatch, meta = run_live_diff(
        GEN2_ALAKAZAM_TEAM,
        GEN2_ALAKAZAM_TEAM,
        seed=seed,
        n_turns=20,
        gen=2,
    )
    assert meta["returncode"] == 0, meta
    assert not meta["timeout"]
    assert show_rows
    assert py_rows
    assert mismatch is None, mismatch


@pytest.mark.parametrize(
    ("team", "seed"),
    [
        (GEN2_STARMIE_TEAM, 999),
        (GEN2_STARMIE_TEAM, 12345),
        (GEN2_GENGAR_TEAM, 999),
        (GEN2_GENGAR_TEAM, 12345),
    ],
)
def test_gen2_live_diff_special_mirror_full_battle(team, seed: int):
    """Additional gen2 special-mirror scenarios through battle end."""
    py_rows, show_rows, mismatch, meta = run_live_diff(
        team,
        team,
        seed=seed,
        n_turns=20,
        gen=2,
    )
    assert meta["returncode"] == 0, meta
    assert not meta["timeout"]
    assert show_rows
    assert py_rows
    assert mismatch is None, mismatch


@pytest.mark.parametrize("seed", [999, 12345, 424242, 500])
def test_gen1_live_diff_alakazam_mirror_full_battle(seed: int):
    """Gen1 mirror Psychic spam should match Showdown through battle end."""
    py_rows, show_rows, mismatch, meta = run_live_diff(
        GEN1_ALAKAZAM_TEAM,
        GEN1_ALAKAZAM_TEAM,
        seed=seed,
        n_turns=20,
        gen=1,
    )
    assert meta["returncode"] == 0, meta
    assert not meta["timeout"]
    assert show_rows
    assert py_rows
    assert mismatch is None, mismatch


@pytest.mark.parametrize(
    ("team", "seed"),
    [
        (GEN1_STARMIE_TEAM, 999),
        (GEN1_STARMIE_TEAM, 12345),
        (GEN1_GENGAR_TEAM, 999),
        (GEN1_GENGAR_TEAM, 12345),
    ],
)
def test_gen1_live_diff_special_mirror_full_battle(team, seed: int):
    """Additional gen1 special-mirror scenarios through battle end."""
    py_rows, show_rows, mismatch, meta = run_live_diff(
        team,
        team,
        seed=seed,
        n_turns=20,
        gen=1,
    )
    assert meta["returncode"] == 0, meta
    assert not meta["timeout"]
    assert show_rows
    assert py_rows
    assert mismatch is None, mismatch
