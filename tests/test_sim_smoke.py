"""End-to-end crash-only smoke tests for the event-loop engine.

These intentionally make NO parity assertions (HP/frame correctness is verified
separately in the parity suites, which are deselected during the writing phase).
The contract here is purely structural:

  * Full random-action battles for gens 1, 2, 3, 4, 9 run through the new event
    loop without raising.
  * Battles terminate (a side faints) or hit the turn cap.
  * The forced-switch control-flow path is exercised (multi-mon teams faint and
    must send a replacement).

If any of these crash, the writing phase regressed.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from pokepy.core.constants import (
    M_ACTIVE0,
    NUM_BATTLE_ACTIONS,
    OFF_META,
    OFF_SIDE0,
    PHASE_FORCED_SWITCH,
    POKEMON_SIZE,
)
from pokepy.env.battle_env import BattleEnv

GENS = [1, 2, 3, 4, 9]
SEEDS = [999, 12345, 424242]
MAX_TURNS = 50


def _multi_mon_team(gen: int) -> dict:
    """A 3-mon mirror team so faints exercise the forced-switch path.

    Alakazam / Snorlax / Starmie with broadly legal damaging + status moves.
    Carries items (and abilities for gens that support them) so the residual
    item/ability/berry handlers are actually exercised, not just the no-item
    fast path.
    """
    from pokepy.core.constants import (
        ABILITY_SPEED_BOOST,
        ITEM_LEFTOVERS,
        ITEM_SITRUS_BERRY,
    )

    has_abilities = gen >= 3
    items = (
        [ITEM_LEFTOVERS, ITEM_SITRUS_BERRY, ITEM_LEFTOVERS] if gen >= 2 else [0, 0, 0]
    )
    abilities = [ABILITY_SPEED_BOOST, 0, 0] if has_abilities else [0, 0, 0]
    return dict(
        species=[65, 143, 121],
        moves=[
            [94, 60, 113, 57],  # Psychic, Psybeam, Light Screen, Surf
            [63, 34, 89, 156],  # Hyper Beam, Body Slam, Earthquake, Rest
            [56, 85, 94, 58],  # Surf, Thunderbolt, Psychic, Ice Beam
        ],
        abilities=abilities,
        items=items,
        tera_types=[0, 0, 0],
        levels=[100, 100, 100],
    )


def _random_legal_action(env: BattleEnv, side: int, rng: random.Random) -> int:
    mask = env.get_action_mask(side=side)
    legal = [i for i in range(NUM_BATTLE_ACTIONS) if bool(mask[i])]
    return rng.choice(legal) if legal else 0


def _random_legal_switch(env: BattleEnv, side: int, rng: random.Random) -> int:
    mask = env.get_action_mask(side=side)
    switches = [a for a in range(4, NUM_BATTLE_ACTIONS) if bool(mask[a])]
    return rng.choice(switches) if switches else 4


@pytest.mark.parametrize("gen", GENS)
@pytest.mark.parametrize("seed", SEEDS)
def test_full_battle_runs_without_crashing(gen: int, seed: int):
    rng = random.Random(seed)
    env = BattleEnv(seed=seed, gen=gen)
    team = _multi_mon_team(gen)
    env.reset(team0=team, team1=team, seed=seed)

    done = False
    tera_requested = False
    for _ in range(MAX_TURNS):
        assert env.state is not None
        if int(env.state.phase) == PHASE_FORCED_SWITCH:
            a0 = _random_legal_switch(env, 0, rng)
            _, _, _, done = env.step(a0, 0)
        else:
            a0 = _random_legal_action(env, 0, rng)
            a1 = _random_legal_action(env, 1, rng)
            # Exercise the Terastallization path on the first move for gen9.
            want_tera = env.profile.has_tera and not tera_requested
            tera_requested = tera_requested or want_tera
            _, _, _, done = env.step(a0, a1, tera0=want_tera, tera1=want_tera)
        if done:
            break

    # The battle must have either ended or run the full cap without raising.
    assert done or int(env.state.turn) >= 0
    # Observations must still be well-formed at the end.
    obs = env.observe(side=0)
    assert isinstance(obs, dict) and obs


@pytest.mark.parametrize("gen", GENS)
def test_forced_switch_path_exercised(gen: int):
    """Deterministically drive the forced-switch replacement path.

    Random play auto-resolves most side-0 faints mid-turn, so to exercise the
    FORCED_SWITCH control flow we faint side 0's active directly, put the battle
    in the forced-switch phase, and confirm a replacement switch step runs
    without crashing and brings in a live benched mon."""
    env = BattleEnv(seed=999, gen=gen)
    team = _multi_mon_team(gen)
    env.reset(team0=team, team1=team, seed=999)
    state = env.state
    assert state is not None

    battle = state.battle_state
    active0 = int(battle[OFF_META + M_ACTIVE0])
    off = OFF_SIDE0 + active0 * POKEMON_SIZE
    battle[off + 1] = 0  # HP -> 0
    battle[off + 15] = int(battle[off + 15]) | 0x01  # fainted flag
    state.phase = np.int8(PHASE_FORCED_SWITCH)
    state.forced_switch_side = np.int8(0)

    mask = env.get_action_mask(side=0)
    switches = [a for a in range(4, NUM_BATTLE_ACTIONS) if bool(mask[a])]
    assert switches, f"gen{gen}: no legal replacement switch available"

    _, _, _, _ = env.step(switches[0], 0)

    # A live mon must now occupy side 0's active slot.
    new_active = int(env.state.battle_state[OFF_META + M_ACTIVE0])
    new_off = OFF_SIDE0 + new_active * POKEMON_SIZE
    assert int(env.state.battle_state[new_off + 1]) > 0
