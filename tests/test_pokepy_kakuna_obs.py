"""Phase 8 Kakuna obs adapter tests."""

from __future__ import annotations

import numpy as np
import pytest

from pokepy.core.state import MultiFormatState
from pokepy.core.constants import (
    OFF_SIDE0, OFF_SIDE1, OFF_META, M_ACTIVE0, M_ACTIVE1, POKEMON_SIZE,
)
from pokepy.data.loader import load_game_data, load_id_mappings
from pokepy.obs.universal import UniversalState, UniversalPokemon, UniversalMove
from pokepy.obs.observation_space import state_to_obs
from pokepy.obs.tokenizer import load_default_tokenizer
from pokepy.obs.kakuna_obs import build_kakuna_obs, build_rl2_features
from pokepy.obs.state_to_universal import state_to_universal_state

def _make_min_pokemon(name="bulbasaur") -> UniversalPokemon:
    return UniversalPokemon(
        name=name, hp_pct=1.0,
        types="grass poison", item="leftovers", ability="overgrow",
        lvl=100, status="nostatus", effect="noeffect",
        moves=[UniversalMove.blank() for _ in range(4)],
        base_atk=49, base_spa=65, base_def=49, base_spd=65, base_spe=45, base_hp=45,
        tera_type="grass", base_species=name,
    )

def _make_min_state() -> UniversalState:
    return UniversalState(
        format="gen9ou",
        player_active_pokemon=_make_min_pokemon("bulbasaur"),
        opponent_active_pokemon=_make_min_pokemon("charmander"),
        available_switches=[_make_min_pokemon(f"mon{i}") for i in range(4)],
        player_prev_move=UniversalMove.blank(),
        opponent_prev_move=UniversalMove.blank(),
        opponents_remaining=6,
        player_conditions="noconditions",
        opponent_conditions="noconditions",
        weather="noweather",
        battle_field="nofield",
        forced_switch=False,
        battle_won=False,
        battle_lost=False,
        can_tera=True,
        opponent_teampreview=["bulbasaur", "charmander"],
    )

def test_tokenizer_loads():
    tok = load_default_tokenizer()
    assert len(tok) == 2541
    assert tok.vocab["<blank>"] == 1
    assert tok.vocab["<anychoice>"] == 0

def test_state_to_obs_shapes():
    s = _make_min_state()
    text, numbers = state_to_obs(s)
    # OpponentMoveObservationSpace + Expanded + TeamPreview gives ~106 tokens
    word_count = len(text.split())
    assert word_count == 106, f"expected 106 tokens, got {word_count}"
    assert numbers.dtype == np.float32
    assert numbers.shape == (55,), f"expected (55,), got {numbers.shape}"

def test_build_kakuna_obs_shapes():
    s = _make_min_state()
    obs = build_kakuna_obs(s)
    assert set(obs.keys()) == {"text_tokens", "numbers", "illegal_actions"}
    assert obs["text_tokens"].dtype == np.int32
    assert obs["text_tokens"].shape == (106,)
    assert obs["numbers"].dtype == np.float32
    assert obs["numbers"].shape == (55,)
    assert obs["illegal_actions"].dtype == np.bool_
    assert obs["illegal_actions"].shape == (13,)

def test_kakuna_obs_known_tokens():
    s = _make_min_state()
    obs = build_kakuna_obs(s)
    # Should not all be -1 (we use real metamon vocab)
    assert (obs["text_tokens"] >= 0).sum() > 50, "most tokens should be in vocab"

def test_build_rl2_features_shape():
    rl2 = build_rl2_features(prev_reward=0.5, prev_action=3)
    assert rl2.shape == (14,)
    assert rl2.dtype == np.float32
    assert rl2[0] == 0.5
    assert rl2[1 + 3] == 1.0
    assert rl2[1] == 0.0

def test_state_to_universal_runs_on_empty_state():
    """Bridge from MultiFormatState to UniversalState shouldn't crash on a hand-built state."""
    gd = load_game_data()
    m = load_id_mappings()
    state = MultiFormatState.create_empty(format_id=1)

    # Hand-build minimal team data + battle_state to make at least one active pokemon
    state.team_species[0] = 1  # bulbasaur
    state.team_moves[0] = [33, 14, 22, 75]  # tackle, growl, vine whip, razor leaf-ish
    state.team_pp[0] = [35, 40, 25, 25]
    state.team_items[0] = 0
    state.team_abilities[0] = 0
    state.opp_species[0] = 4  # charmander
    state.opp_moves[0] = [10, 43, 52, 108]
    state.opp_pp[0] = [35, 30, 25, 20]

    # Battle buffer: place active pokemon at side0/0 and side1/0
    bs = state.battle_state
    bs[OFF_META + M_ACTIVE0] = 0
    bs[OFF_META + M_ACTIVE1] = 0
    # side0 active: species, hp, max_hp, level, types, ability, item, stats
    p0 = OFF_SIDE0 + 0 * POKEMON_SIZE
    bs[p0 + 0] = 1                           # species
    bs[p0 + 1] = 100                         # current hp
    bs[p0 + 2] = 100                         # max hp
    bs[p0 + 3] = 100                         # level
    bs[p0 + 4] = 4 | (7 << 8)                # grass | poison
    bs[p0 + 5] = 0                           # ability
    bs[p0 + 6] = 0                           # item
    for i, v in enumerate([60, 60, 60, 60, 60]):
        bs[p0 + 7 + i] = v
    bs[p0 + 13] = 0x6666                     # neutral boosts
    bs[p0 + 14] = 0x0666

    p1 = OFF_SIDE1 + 0 * POKEMON_SIZE
    bs[p1 + 0] = 4                           # charmander
    bs[p1 + 1] = 100
    bs[p1 + 2] = 100
    bs[p1 + 3] = 100
    bs[p1 + 4] = 1                           # fire
    bs[p1 + 13] = 0x6666
    bs[p1 + 14] = 0x0666
    for i, v in enumerate([60, 60, 60, 60, 60]):
        bs[p1 + 7 + i] = v

    us = state_to_universal_state(state, gd, m)
    assert us.player_active_pokemon.name != "<blank>"
    assert us.opponent_active_pokemon.name != "<blank>"
    assert us.format == "gen9ou"

    # And the obs builder runs end-to-end on a real (synthetic) state
    obs = build_kakuna_obs(us)
    assert obs["text_tokens"].shape == (106,)
    assert obs["numbers"].shape == (55,)
