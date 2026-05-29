"""Observation builders ported from metamon/interface.py.

Composes the OpponentMoveObservationSpace pipeline (which is the one Kakuna
trains against):
    OpponentMoveObservationSpace -> TeamPreviewObservationSpace ->
    ExpandedObservationSpace -> DefaultObservationSpace.

Output: dict with 'text' (str) and 'numbers' (np.float32[55]).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from pokepy.obs.universal import (
    UniversalMove,
    UniversalPokemon,
    UniversalState,
    clean_name,
)

# -----------------------------------------------------------------------------
# Consistent ordering helpers (sort by name) — same as metamon's
# -----------------------------------------------------------------------------


def consistent_pokemon_order(pokemon: List[UniversalPokemon]) -> List[UniversalPokemon]:
    return sorted(pokemon, key=lambda p: p.name)


def consistent_move_order(moves: List[UniversalMove]) -> List[UniversalMove]:
    return sorted(moves, key=lambda m: m.name)


# -----------------------------------------------------------------------------
# Move features (OpponentMove variant: 2 tokens for active moves, not 3)
# -----------------------------------------------------------------------------


def _move_string_features_active(move: UniversalMove) -> List[str]:
    # OpponentMove: just name + type for active moves (saves 4 tokens vs default)
    return [clean_name(move.name), clean_name(move.move_type)]


def _move_string_features_inactive(move: UniversalMove) -> List[str]:
    # Inactive (in moveset listings): just name
    return [clean_name(move.name)]


def _move_pad_string_active() -> List[str]:
    # OpponentMove: 2 blanks for active pad
    return ["<blank>", "<blank>"]


def _move_pad_string_inactive() -> List[str]:
    return ["<blank>"]


def _move_numerical_features_active(move: UniversalMove) -> List[float]:
    # 4 features: bp/200, accuracy, priority/5, pp_warning
    pp_ratio = move.current_pp / move.max_pp if move.max_pp > 0 else 0.0
    pp_warning = (
        (1 if pp_ratio >= 0.5 else 0)
        + (1 if pp_ratio >= 0.25 else 0)
        + (1 if pp_ratio > 0 else 0)
    )
    return [
        move.base_power / 200.0,
        float(move.accuracy),
        move.priority / 5.0,
        float(pp_warning),
    ]


def _move_pad_numerical_active() -> List[float]:
    return [-2.0] * 4


# -----------------------------------------------------------------------------
# Pokemon features (TeamPreview/Expanded extends Default with tera_type)
# -----------------------------------------------------------------------------


def _pokemon_string_features(pokemon: UniversalPokemon, active: bool) -> List[str]:
    # Default base: [name, item, ability, ...]
    out = [
        clean_name(pokemon.name),
        clean_name(pokemon.item),
        clean_name(pokemon.ability),
    ]
    if active:
        out += [pokemon.types, clean_name(pokemon.effect), clean_name(pokemon.status)]
    else:
        out += ["<moveset>"]
        move_num = -1
        for move_num, move in enumerate(consistent_move_order(pokemon.moves)):
            out += _move_string_features_inactive(move)
        while move_num < 3:
            out += _move_pad_string_inactive()
            move_num += 1
    # Expanded adds tera_type
    out.append(clean_name(pokemon.tera_type))
    return out


def _opponent_pokemon_string_features(
    pokemon: UniversalPokemon, active: bool
) -> List[str]:
    # OpponentMoveObservationSpace: append the opponent's revealed moves (4 slots)
    base = _pokemon_string_features(pokemon, active)
    moves = ["<blank>"] * 4
    for i, move in enumerate(consistent_move_order(pokemon.moves)[:4]):
        moves[i] = clean_name(move.name)
    return base + moves


def _pokemon_pad_string(active: bool) -> List[str]:
    # Expanded: 4 + (4 active / 5 inactive) blanks
    blanks = 4 + (4 if active else 5)
    return ["<blank>"] * blanks


def _pokemon_numerical_features(pokemon: UniversalPokemon, active: bool) -> List[float]:
    out = [pokemon.hp_pct]
    if active:
        out.append(pokemon.lvl / 100.0)
        out += [
            pokemon.base_atk / 255.0,
            pokemon.base_spa / 255.0,
            pokemon.base_def / 255.0,
            pokemon.base_spd / 255.0,
            pokemon.base_spe / 255.0,
            pokemon.base_hp / 255.0,
        ]
        out += [
            pokemon.atk_boost / 6.0,
            pokemon.spa_boost / 6.0,
            pokemon.def_boost / 6.0,
            pokemon.spd_boost / 6.0,
            pokemon.spe_boost / 6.0,
            pokemon.accuracy_boost / 6.0,
            pokemon.evasion_boost / 6.0,
        ]
    return out


def _pokemon_pad_numerical(active: bool) -> List[float]:
    blanks = 1 + (14 if active else 0)
    return [-2.0] * blanks


# -----------------------------------------------------------------------------
# Top-level state_to_obs (OpponentMoveObservationSpace)
# -----------------------------------------------------------------------------


def state_to_obs(state: UniversalState) -> Tuple[str, np.ndarray]:
    """Build the (text, numbers) pair from a UniversalState.

    Mirrors metamon OpponentMoveObservationSpace.state_to_obs (which inherits
    from TeamPreview -> Expanded -> Default), with the OpponentMove move-feature
    overrides applied.
    """
    player = state.player_active_pokemon
    opponent = state.opponent_active_pokemon

    # --- Player active ---
    player_str = ["<player>"] + _pokemon_string_features(player, active=True)
    numerical: List[float] = [
        state.opponents_remaining / 6.0
    ] + _pokemon_numerical_features(player, active=True)

    # --- Player active moves ---
    move_str: List[str] = []
    move_num = -1
    for move_num, move in enumerate(consistent_move_order(player.moves)):
        move_str += ["<move>"] + _move_string_features_active(move)
        numerical += _move_numerical_features_active(move)
    while move_num < 3:
        move_str += ["<move>"] + _move_pad_string_active()
        numerical += _move_pad_numerical_active()
        move_num += 1

    # --- Player switches ---
    switch_str: List[str] = []
    switch_num = -1
    for switch_num, switch in enumerate(
        consistent_pokemon_order(state.available_switches)
    ):
        switch_str += ["<switch>"] + _pokemon_string_features(switch, active=False)
        numerical += _pokemon_numerical_features(switch, active=False)
    while switch_num < 4:
        switch_str += ["<switch>"] + _pokemon_pad_string(active=False)
        numerical += _pokemon_pad_numerical(active=False)
        switch_num += 1

    # --- Opponent active ---
    force_switch = "<forcedswitch>" if state.forced_switch else "<anychoice>"
    opponent_str = ["<opponent>"] + _opponent_pokemon_string_features(
        opponent, active=True
    )
    numerical += _pokemon_numerical_features(opponent, active=True)

    # --- Conditions / prev moves ---
    global_str = [
        "<conditions>",
        state.weather,
        state.player_conditions,
        state.opponent_conditions,
    ]
    prev_move_str = (
        ["<player_prev>"]
        + _move_string_features_inactive(state.player_prev_move)
        + ["<opp_prev>"]
        + _move_string_features_inactive(state.opponent_prev_move)
    )

    full_text_list = (
        [f"<{state.format}>", force_switch]
        + player_str
        + move_str
        + switch_str
        + opponent_str
        + global_str
        + prev_move_str
    )

    # --- Expanded extras: any_opponent_asleep, any_opponent_frozen, can_tera ---
    # Pokepy doesn't carry the cross-turn 'any opponent ever asleep/frozen'
    # state, so we report only the current state.
    any_asleep = 1.0 if opponent.status == "slp" else 0.0
    any_frozen = 1.0 if opponent.status == "frz" else 0.0
    numerical += [any_asleep, any_frozen, 1.0 if state.can_tera else 0.0]

    # --- Expanded extras: revealed opponents (6 tokens, sorted) ---
    revealed = sorted({opponent.base_species or opponent.name})
    while len(revealed) < 6:
        revealed.append("<blank>")
    full_text_list += revealed[:6]

    # --- Team preview extras: opponent_teampreview (6 tokens, sorted) ---
    teampreview = sorted(state.opponent_teampreview)
    while len(teampreview) < 6:
        teampreview.append("<blank>")
    full_text_list += teampreview[:6]

    text = " ".join(full_text_list)
    numbers = np.array(numerical, dtype=np.float32)
    return text, numbers
