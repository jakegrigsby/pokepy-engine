"""Mutable battle state, mirroring MultiFormatState (single-battle slice).

The MultiFormatState carries a leading [batch] dim because it's a the reference
vmap target. Pokepy's scalar state holds the same fields without the batch
axis. A future numpy-batched variant (phase 10) will reintroduce the leading
axis sharing this same field layout.

The flat `battle_state` int16 buffer follows the byte layout in
pokepy/core/constants.py (POKEMON_SIZE, OFF_SIDE0, OFF_FIELD, OFF_META).
Helper accessors live in pokepy/core/bitpack.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _f
import numpy as np

from pokepy.core.constants import STATE_SIZE

def _zeros(shape, dtype):
    return np.zeros(shape, dtype=dtype)

def _full(shape, value, dtype):
    return np.full(shape, value, dtype=dtype)

@dataclass
class MultiFormatState:
    """Single-battle slice of MultiFormatState."""

    # Format identifier
    format_id: np.int8 = np.int8(1)  # default Gen 9 OU

    # Phase / step (teambuilder, preview, battle, forced switch, ended)
    phase: np.int8 = np.int8(0)
    phase_step: np.int16 = np.int16(0)

    # Player team
    team_species: np.ndarray = _f(default_factory=lambda: _full(6, -1, np.int16))
    team_moves: np.ndarray = _f(default_factory=lambda: _full((6, 4), -1, np.int16))
    team_items: np.ndarray = _f(default_factory=lambda: _full(6, -1, np.int16))
    team_abilities: np.ndarray = _f(default_factory=lambda: _full(6, -1, np.int16))
    team_evs: np.ndarray = _f(default_factory=lambda: _zeros(6, np.int8))
    team_evs_full: np.ndarray = _f(default_factory=lambda: _zeros((6, 6), np.int16))
    team_ivs_full: np.ndarray = _f(default_factory=lambda: _full((6, 6), 31, np.int16))
    team_nature_mods: np.ndarray = _f(default_factory=lambda: np.ones((6, 6), dtype=np.float32))
    team_tera: np.ndarray = _f(default_factory=lambda: _full(6, -1, np.int8))
    team_pp: np.ndarray = _f(default_factory=lambda: _zeros((6, 4), np.int8))

    # Opponent team
    opp_species: np.ndarray = _f(default_factory=lambda: _full(6, -1, np.int16))
    opp_moves: np.ndarray = _f(default_factory=lambda: _full((6, 4), -1, np.int16))
    opp_items: np.ndarray = _f(default_factory=lambda: _full(6, -1, np.int16))
    opp_abilities: np.ndarray = _f(default_factory=lambda: _full(6, -1, np.int16))
    opp_evs: np.ndarray = _f(default_factory=lambda: _zeros(6, np.int8))
    opp_evs_full: np.ndarray = _f(default_factory=lambda: _zeros((6, 6), np.int16))
    opp_ivs_full: np.ndarray = _f(default_factory=lambda: _full((6, 6), 31, np.int16))
    opp_nature_mods: np.ndarray = _f(default_factory=lambda: np.ones((6, 6), dtype=np.float32))
    opp_tera: np.ndarray = _f(default_factory=lambda: _full(6, -1, np.int8))
    opp_pp: np.ndarray = _f(default_factory=lambda: _zeros((6, 4), np.int8))

    # Opponent revealed (partial observability)
    opp_revealed: np.ndarray = _f(default_factory=lambda: _zeros(6, np.bool_))
    opp_moves_revealed: np.ndarray = _f(default_factory=lambda: _zeros((6, 4), np.bool_))

    # Player revealed (symmetric partial observability, as seen by side 1).
    # Populated by the same engine reveal logic that drives opp_* above; used
    # when a side-1 agent (e.g. Kakuna-as-opponent) needs to observe which
    # side-0 moves have actually been used so far.
    team_revealed: np.ndarray = _f(default_factory=lambda: _zeros(6, np.bool_))
    team_moves_revealed: np.ndarray = _f(default_factory=lambda: _zeros((6, 4), np.bool_))

    # Per-slot consumed berry tracking for mechanics such as Harvest.
    team_last_consumed_berry: np.ndarray = _f(default_factory=lambda: _zeros(6, np.int16))
    opp_last_consumed_berry: np.ndarray = _f(default_factory=lambda: _zeros(6, np.int16))
    # Showdown resets each active Pokemon's moveSlot.used flags on switch-in.
    # Last Resort reads those flags, so track the four move slots as a bitmask.
    team_move_used_masks: np.ndarray = _f(default_factory=lambda: _zeros(6, np.int8))
    opp_move_used_masks: np.ndarray = _f(default_factory=lambda: _zeros(6, np.int8))

    # Battle state flat int16 buffer
    battle_state: np.ndarray = _f(default_factory=lambda: _zeros(STATE_SIZE, np.int16))

    # Episode tracking
    turn: np.int16 = np.int16(0)
    done: np.bool_ = np.bool_(False)
    winner: np.int8 = np.int8(-1)

    # RNG (Gen 5 PRNG state shape kept for compatibility)
    rng: np.ndarray = _f(default_factory=lambda: np.array([0, 0], dtype=np.uint32))

    # Gen5 PRNG seed (Showdown-compatible damage rolls)
    gen5_seed: np.uint64 = np.uint64(0)

    # Forced switch state
    forced_switch_slot: np.int8 = np.int8(-1)
    forced_switch_hp: np.int16 = np.int16(0)
    forced_switch_original: np.int8 = np.int8(-1)
    forced_switch_action_speed: np.int16 = np.int16(0)
    # When the opponent auto-switches in while player 0 is still fainted and
    # awaiting a forced replacement, defer that opponent's on-switch ability
    # until step_forced_switch can resolve against the real live target.
    pending_opp_switch_in_slot: np.int8 = np.int8(-1)
    pending_opp_switch_action_speed: np.int16 = np.int16(0)
    # Light Clay screens are stored as 7-turn packed values and skip their
    # first end-of-turn decrement so they still last 8 total turns.
    screen_skip_decrement0: np.int8 = np.int8(0)
    screen_skip_decrement1: np.int8 = np.int8(0)

    # Showdown side.pokemon order (singles: active at index 0, bench after it).
    # Needed for getRandomSwitchable parity on drag effects such as Red Card.
    side_order0: np.ndarray = _f(default_factory=lambda: np.arange(6, dtype=np.int8))
    side_order1: np.ndarray = _f(default_factory=lambda: np.arange(6, dtype=np.int8))
    # Hidden Showdown startup PRNG calls that happen before the first visible
    # turn (team preview queue sort + lead switch-in startup). step_battle_gen9
    # replays these on turn 0 so the live Gen5 PRNG stays aligned.
    startup_prng_calls: tuple[tuple[int, ...], ...] = _f(default_factory=tuple)

    @classmethod
    def create_empty(cls, format_id: int = 1) -> "MultiFormatState":
        s = cls()
        s.format_id = np.int8(format_id)
        return s
