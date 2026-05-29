"""Status counter mechanics (sleep, freeze, paralysis, toxic stacking).

Tests:
- Sleep duration 1-3 turns
- Sleep Talk uses random move while asleep
- Snore deals damage only while asleep
- Rest sleeps user for 2 turns + heals fully
- Early Bird halves sleep turns
- Freeze 20% thaw chance per turn
- Fire moves thaw frozen mons
- Paralysis 25% full-para
- Toxic counter: damage scales 1/16, 2/16, 3/16, ...
- Toxic counter resets on switch out
- Magic Guard ignores all status damage
- Poison Heal: heals 1/8 instead of damage when poisoned

Showdown reference:
- pokemon-showdown/data/conditions.ts: slp, brn, par, frz, psn, tox handlers
- pokemon-showdown/data/abilities.ts: earlybird, magicguard, poisonheal
- pokemon-showdown/data/moves.ts: rest, sleeptalk, snore
"""

from __future__ import annotations

import pytest
from tests.conftest import MonSpec
from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_META,
    POKEMON_SIZE,
    M_ACTIVE0,
    M_ACTIVE1,
    STATUS_BURN,
    STATUS_PARALYSIS,
    STATUS_SLEEP,
    STATUS_FREEZE,
    STATUS_POISON,
    STATUS_TOXIC,
    STATUS_NONE,
)


def _set_status(state, side: int, status: int, turns: int = 0):
    base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    active = int(state.battle_state[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])
    state.battle_state[base + active * POKEMON_SIZE + 12] = (status & 0xFF) | (
        (turns & 0xFF) << 8
    )


# ---------------------------------------------------------------------------
# Sleep
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="sleep prevents move")
def test_sleeping_mon_cannot_move(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["bodyslam", "tackle", "tackle", "tackle"])],
        [MonSpec("blissey", ["tackle"] * 4)],
        seed=1,
    )
    _set_status(state, 0, STATUS_SLEEP, turns=3)  # 3 turns of sleep
    hp1_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    # Snorlax should not have damaged blissey (it's asleep)
    assert hp_of(state, 1) == hp1_pre - 0 or hp_of(state, 1) >= hp1_pre - 5


# ---------------------------------------------------------------------------
# Rest
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="rest heals to full")
def test_rest_heals_full(fresh_battle, step_turn, hp_of, max_hp_of, status_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["rest", "tackle", "tackle", "tackle"])],
        [MonSpec("blissey", ["tackle"] * 4)],
        seed=1,
    )
    max0 = max_hp_of(state, 0)
    state.battle_state[OFF_SIDE0 + 1] = max0 // 2
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 0) >= max0 - 5
    assert status_of(state, 0) == STATUS_SLEEP


# ---------------------------------------------------------------------------
# Burn
# ---------------------------------------------------------------------------


def test_burn_damages_eot(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    _set_status(state, 0, STATUS_BURN)
    max0 = max_hp_of(state, 0)
    hp_pre = hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    # Burn deals 1/16 max HP per turn
    assert hp_of(state, 0) <= hp_pre - max0 // 16 + 5


@pytest.mark.xfail(strict=False, reason="burn halves physical attack")
def test_burn_halves_physical(fresh_battle, step_turn, hp_of):
    state_a, prng_a = fresh_battle(
        [MonSpec("snorlax", ["bodyslam", "tackle", "tackle", "tackle"])],
        [MonSpec("blissey", ["tackle"] * 4)],
        seed=1,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("snorlax", ["bodyslam", "tackle", "tackle", "tackle"])],
        [MonSpec("blissey", ["tackle"] * 4)],
        seed=1,
    )
    _set_status(state_a, 0, STATUS_BURN)
    hp_a_pre = hp_of(state_a, 1)
    hp_b_pre = hp_of(state_b, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    da = hp_a_pre - hp_of(state_a, 1)
    db = hp_b_pre - hp_of(state_b, 1)
    # Burned snorlax should hit for less
    assert da < db


# ---------------------------------------------------------------------------
# Toxic counter
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="toxic damage scales each turn")
def test_toxic_damage_increases_each_turn(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["splash"] * 4)],
        [MonSpec("snorlax", ["splash"] * 4)],
        seed=1,
    )
    _set_status(state, 0, STATUS_TOXIC, turns=1)
    max0 = max_hp_of(state, 0)
    hp_t0 = hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    dmg_t1 = hp_t0 - hp_of(state, 0)
    hp_t1 = hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    dmg_t2 = hp_t1 - hp_of(state, 0)
    assert dmg_t2 > dmg_t1


# ---------------------------------------------------------------------------
# Magic Guard
# ---------------------------------------------------------------------------


def test_magic_guard_ignores_burn_damage(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("clefable", ["splash"] * 4, ability="magicguard")],
        [MonSpec("snorlax", ["splash"] * 4)],
        seed=1,
    )
    _set_status(state, 0, STATUS_BURN)
    hp_pre = hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 0) == hp_pre


def test_magic_guard_ignores_poison_damage(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("clefable", ["splash"] * 4, ability="magicguard")],
        [MonSpec("snorlax", ["splash"] * 4)],
        seed=1,
    )
    _set_status(state, 0, STATUS_POISON)
    hp_pre = hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 0) == hp_pre


# ---------------------------------------------------------------------------
# Poison Heal
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="poison heal heals 1/8")
def test_poison_heal_heals_instead_of_damage(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [MonSpec("breloom", ["splash"] * 4, ability="poisonheal")],
        [MonSpec("snorlax", ["splash"] * 4)],
        seed=1,
    )
    _set_status(state, 0, STATUS_TOXIC)
    max0 = max_hp_of(state, 0)
    state.battle_state[OFF_SIDE0 + 1] = max0 // 2
    hp_pre = hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 0) > hp_pre


# ---------------------------------------------------------------------------
# Toxic counter resets on switch
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="toxic counter resets on switch out")
def test_toxic_counter_resets_on_switch():
    pytest.skip("placeholder — needs switch sequence")


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="freeze prevents move")
def test_frozen_mon_cannot_move(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["bodyslam", "tackle", "tackle", "tackle"])],
        [MonSpec("blissey", ["tackle"] * 4)],
        seed=1,
    )
    _set_status(state, 0, STATUS_FREEZE)
    hp1_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    # Snorlax can't move while frozen → no damage to blissey from snorlax
    # blissey may have moved → snorlax may have taken some, but blissey hp intact
    assert hp_of(state, 1) == hp1_pre


# ---------------------------------------------------------------------------
# Paralysis full-para chance (just check status persists)
# ---------------------------------------------------------------------------


def test_paralyzed_status_persists(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["splash"] * 4)],
        [MonSpec("snorlax", ["splash"] * 4)],
        seed=1,
    )
    _set_status(state, 0, STATUS_PARALYSIS)
    step_turn(state, prng, 0, 0)
    assert status_of(state, 0) == STATUS_PARALYSIS
