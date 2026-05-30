"""Phase-A vertical slice: pokepy.showdown engine vs live Showdown.

This is the acceptance gate for the verbatim port skeleton (todo A5). It drives
the new object-model engine (``pokepy.showdown.Battle``) with a fixed
first-legal-move policy, replays the identical action log into the Showdown CLI,
and asserts the per-turn HP/status rows agree for the clean-turn prefix.

Phase B widens this to the full multigen battery; here we only require that one
gen9 mirror turn (startup PRNG + move pipeline) lands bit-exact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ENGINE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ENGINE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from showdown_ref import (  # noqa: E402
    compare_battle_rows,
    run_showdown,
    team_to_showdown_packed,
)

from pokepy.data.loader import load_id_mappings  # noqa: E402
from pokepy.showdown.adapter import run_engine_battle  # noqa: E402

METAMON_ROOT = Path(__file__).resolve().parents[4]
SHOWDOWN_BIN = METAMON_ROOT / "server" / "pokemon-showdown" / "pokemon-showdown"

pytestmark = pytest.mark.skipif(
    not SHOWDOWN_BIN.exists(),
    reason="Showdown CLI required for live-diff tests",
)

_ZERO_EV = [[0, 0, 0, 0, 0, 0]]
_MAX_IV = [[31, 31, 31, 31, 31, 31]]


def _single_mon_team(species, moves, *, nature="Serious"):
    return dict(
        species=[species],
        moves=[moves],
        abilities=[0],
        items=[0],
        tera_types=[0],
        levels=[100],
        evs=_ZERO_EV,
        ivs=_MAX_IV,
        natures=[nature],
    )


# Alakazam Psychic-spam mirror (Psychic=94, Psybeam=60, Light Screen=113, Surf=57).
GEN9_ALAKAZAM = _single_mon_team(65, [94, 60, 113, 57])


def _run(team0, team1, seed, n_turns, gen):
    mappings = load_id_mappings(gen=gen)
    rows, p0_actions, p1_actions = run_engine_battle(
        team0, team1, mappings, seed=seed, n_turns=n_turns, gen=gen
    )
    seed_tuple = (seed & 0xFFFF, (seed >> 16) & 0xFFFF, 0, 0)
    show_rows, _raw, meta = run_showdown(
        seed_tuple,
        team_to_showdown_packed(team0, mappings),
        team_to_showdown_packed(team1, mappings),
        p0_actions,
        p1_actions,
        n_turns,
        battle_format=f"gen{gen}ou",
    )
    return rows, show_rows, meta


def test_gen9_engine_live_diff_alakazam_psychic_turn1():
    rows, show_rows, meta = _run(GEN9_ALAKAZAM, GEN9_ALAKAZAM, seed=999, n_turns=1, gen=9)
    assert meta["returncode"] == 0, meta
    assert rows and show_rows
    mismatch = compare_battle_rows(rows, show_rows)
    assert mismatch is None, mismatch


@pytest.mark.parametrize("seed", [999, 12345, 424242])
def test_gen9_engine_live_diff_alakazam_mirror_prefix(seed):
    rows, show_rows, meta = _run(GEN9_ALAKAZAM, GEN9_ALAKAZAM, seed=seed, n_turns=10, gen=9)
    assert meta["returncode"] == 0, meta
    assert rows and show_rows
    mismatch = compare_battle_rows(rows, show_rows)
    assert mismatch is None, mismatch
