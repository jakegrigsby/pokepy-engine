"""Phase 7 env wrapper smoke tests — full integration."""

from __future__ import annotations

import numpy as np
import pytest

from pokepy.env import BattleEnv, init_battle_state, DEFAULT_TEAM


def test_battle_env_reset_and_obs_shapes():
    env = BattleEnv()
    obs = env.reset()
    assert set(obs.keys()) == {"text_tokens", "numbers", "illegal_actions"}
    assert obs["text_tokens"].shape == (106,)
    assert obs["numbers"].shape == (55,)
    assert obs["illegal_actions"].shape == (13,)


def test_battle_env_runs_a_full_battle():
    env = BattleEnv()
    env.reset()
    done = False
    rewards_seen = False
    for turn in range(200):
        obs, r0, r1, done = env.step(0, 0)
        if r0 != 0.0 or r1 != 0.0:
            rewards_seen = True
        if done:
            break
    assert done, "battle should finish within 200 turns of mash-attack"
    assert rewards_seen


def test_battle_env_action_mask_excludes_illegal_switches():
    env = BattleEnv()
    env.reset()
    mask = env.get_action_mask(side=0)
    # Single-pokemon team -> only move actions legal, no switches
    assert mask[:4].any()
    assert not mask[4:].any()


def test_init_battle_state_writes_buffer():
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM)
    from pokepy.core.constants import OFF_SIDE0, POKEMON_SIZE

    bs = state.battle_state
    # Slot 0 should have valid species + nonzero hp
    assert int(bs[OFF_SIDE0 + 0]) == 1  # bulbasaur
    assert int(bs[OFF_SIDE0 + 1]) > 0  # current_hp > 0
    assert int(bs[OFF_SIDE0 + 2]) > 0  # max_hp > 0


def test_init_battle_state_marks_leadoff_revealed_symmetrically():
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM)
    # Both reveal masks follow the same convention: the lead active slot is
    # considered revealed at battle start on both sides (opp_revealed for the
    # side-0 view, team_revealed for the side-1 view).
    assert bool(state.opp_revealed[0])
    assert bool(state.team_revealed[0])


def test_battle_step_reveals_both_sides_moves_symmetrically():
    """Engine-level symmetry: step_battle_gen9 must update both
    opp_moves_revealed (side-0 view of side 1) and team_moves_revealed
    (side-1 view of side 0) on a successful move execution."""
    env = BattleEnv()
    env.reset()
    # Confirm pre-step baseline: no move-slot is revealed on either side.
    assert not state_any_revealed(env.state.opp_moves_revealed)
    assert not state_any_revealed(env.state.team_moves_revealed)

    # Mash-attack a single turn with both sides picking raw move slot 0.
    # DEFAULT_TEAM's lead has a legal move at slot 0; both sides reveal it.
    env.step(0, 0)

    # The engine only writes on successful execution; with both Pokemon
    # picking a legal move at slot 0 of their lead (active0 == active1 == 0)
    # we expect exactly the (0, 0) entry on each side to flip to True.
    assert bool(env.state.opp_moves_revealed[0, 0])
    assert bool(env.state.team_moves_revealed[0, 0])


def state_any_revealed(arr: np.ndarray) -> bool:
    return bool(np.asarray(arr).any())
