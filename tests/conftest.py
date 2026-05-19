"""Shared pytest fixtures for the pokepy mechanics test matrix.

These fixtures keep mechanic tests terse: a typical test reads
`scenario = make_battle(...)` then `step_one_turn(scenario, ...)` and asserts
on a single delta. The factory hides the bit-packing required to produce a
valid `MultiFormatState`.

All fixtures are session-scoped where possible to keep test runs fast — the
game data tables and ID mappings are ~10 MB and reusing them across tests
shaves several seconds off the suite.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from pokepy.core.constants import (
    OFF_FIELD, OFF_META, OFF_SIDE0, OFF_SIDE1, POKEMON_SIZE,
    M_ACTIVE0, M_ACTIVE1, F_WEATHER, F_TERRAIN, M_WEATHER_TURNS, M_TERRAIN_TURNS,
    F_HAZARDS_0, F_HAZARDS_1, F_SCREENS_0, F_SCREENS_1,
    PHASE_BATTLE, PHASE_FORCED_SWITCH,
    STATUS_BURN, STATUS_PARALYSIS, STATUS_SLEEP, STATUS_FREEZE, STATUS_POISON, STATUS_TOXIC,
)
from pokepy.data.loader import (
    GameData, IDMappings, MoveEffectData,
    load_game_data, load_id_mappings, load_move_effect_data,
)
from pokepy.data.type_charts import MODERN_TYPE_CHART
from pokepy.env.battle_env import init_battle_state
from pokepy.engine.battle_gen9 import step_battle_gen9, step_forced_switch
from pokepy.engine.action_mask import get_action_mask
from pokepy.utils.gen5_prng import Gen5PRNG

# ---------------------------------------------------------------------------
# Session-scoped data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gd() -> GameData:
    return load_game_data()

@pytest.fixture(scope="session")
def me() -> MoveEffectData:
    return load_move_effect_data()

@pytest.fixture(scope="session")
def mappings() -> IDMappings:
    return load_id_mappings()

@pytest.fixture(scope="session")
def type_chart() -> np.ndarray:
    return MODERN_TYPE_CHART

# ---------------------------------------------------------------------------
# Team / battle factories
# ---------------------------------------------------------------------------

@dataclass
class MonSpec:
    species: str
    moves: List[str]
    item: str = ""
    ability: str = ""
    tera_type: int = 0
    level: int = 100

def _resolve_ids(spec: MonSpec, mappings: IDMappings) -> Tuple[int, List[int], int, int]:
    species_id = mappings.species_to_idx[spec.species]
    move_ids = []
    for m in spec.moves:
        if not m:
            move_ids.append(-1)
        else:
            move_ids.append(mappings.move_to_idx[m])
    while len(move_ids) < 4:
        move_ids.append(-1)
    item_id = mappings.item_to_idx.get(spec.item, 0) if spec.item else 0
    ability_id = mappings.ability_to_idx.get(spec.ability, 0) if spec.ability else 0
    return species_id, move_ids[:4], item_id, ability_id

@pytest.fixture(scope="session")
def make_team(mappings):
    """Factory that builds a team-dict from a list of MonSpecs.

    Pads to 6 mons with copies of the last spec — pokepy expects exactly 6
    slots in init_battle_state.
    """
    def _factory(specs: List[MonSpec]) -> Dict[str, Any]:
        if not specs:
            raise ValueError("at least one mon required")
        full = list(specs)
        while len(full) < 6:
            full.append(specs[-1])
        species, moves, items, abilities, teras, levels = [], [], [], [], [], []
        for s in full[:6]:
            sid, mids, iid, aid = _resolve_ids(s, mappings)
            species.append(sid)
            moves.append(mids)
            items.append(iid)
            abilities.append(aid)
            teras.append(int(s.tera_type))
            levels.append(int(s.level))
        return dict(
            species=species,
            moves=moves,
            items=items,
            abilities=abilities,
            tera_types=teras,
            levels=levels,
        )
    return _factory

@pytest.fixture(scope="session")
def fresh_battle(gd, me, make_team):
    """Factory: build an initialised MultiFormatState from two MonSpec lists."""
    def _factory(team0_specs: List[MonSpec], team1_specs: List[MonSpec], seed: int = 12345):
        t0 = make_team(team0_specs)
        t1 = make_team(team1_specs)
        state = init_battle_state(t0, t1, gd, seed=seed)
        prng = Gen5PRNG((seed & 0xFFFF, (seed >> 16) & 0xFFFF, 0, 0))
        return state, prng
    return _factory

# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def step_turn(gd, me, type_chart):
    """Factory that runs one turn given (state, prng, action0, action1)."""
    def _step(state, prng, a0: int, a1: int):
        if int(state.phase) == PHASE_FORCED_SWITCH:
            return step_forced_switch(
                state, a0, side=0, game_data=gd, move_effects=me,
                type_chart=type_chart, gen5_prng=prng,
            )
        return step_battle_gen9(state, a0, a1, gd, me, type_chart, prng)
    return _step

# ---------------------------------------------------------------------------
# State accessors / asserters
# ---------------------------------------------------------------------------

def _active_off(state, side: int) -> int:
    base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    active = int(state.battle_state[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])
    return base + active * POKEMON_SIZE

@pytest.fixture
def hp_of():
    def _hp(state, side: int) -> int:
        return int(state.battle_state[_active_off(state, side) + 1])
    return _hp

@pytest.fixture
def max_hp_of():
    def _maxhp(state, side: int) -> int:
        return int(state.battle_state[_active_off(state, side) + 2])
    return _maxhp

@pytest.fixture
def status_of():
    def _status(state, side: int) -> int:
        word = int(state.battle_state[_active_off(state, side) + 12])
        return word & 0xFF
    return _status

@pytest.fixture
def boost_of():
    """Read a stat boost stage (atk/def/spa/spd/spe/acc/eva) for the active mon."""
    SLOT_BITS = {
        "atk": (13, 0), "def": (13, 4), "spa": (13, 8), "spd": (13, 12),
        "spe": (14, 0), "acc": (14, 4), "eva": (14, 8),
    }
    def _boost(state, side: int, stat: str) -> int:
        word_idx, shift = SLOT_BITS[stat]
        word = int(state.battle_state[_active_off(state, side) + word_idx]) & 0xFFFF
        nibble = (word >> shift) & 0xF
        return nibble - 6
    return _boost

@pytest.fixture
def field_word():
    """Read an OFF_FIELD slot by its constant offset."""
    def _read(state, off: int) -> int:
        return int(state.battle_state[OFF_FIELD + off])
    return _read

# ---------------------------------------------------------------------------
# Common shorthand teams (1-mon vs 1-mon)
# ---------------------------------------------------------------------------

@pytest.fixture
def vanilla_pair(fresh_battle):
    """Tackle vs Tackle, no items, no abilities, both active. Useful for
    isolating a single mechanic without confounders."""
    def _pair(seed=12345):
        return fresh_battle(
            [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
            [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
            seed=seed,
        )
    return _pair

# Re-export so tests can `from conftest import MonSpec`
__all__ = ["MonSpec"]
