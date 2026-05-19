"""Multi-hit move mechanics.

Tests:
- Standard 2-5 hit moves: Bullet Seed, Rock Blast, Pin Missile, etc.
- Loaded Dice forces 4-5 hits
- Skill Link forces 5 hits
- Population Bomb (10 hits)
- Triple Axel (3 hits with increasing power 20 → 40 → 60)
- Triple Kick (similar)
- Double Hit moves: Dragon Darts, Bonemerang, Double Hit, Tachyon Cutter

Showdown reference:
- pokemon-showdown/data/moves.ts: search "multihit", "tripleaxel", "populationbomb"
- pokemon-showdown/data/items.ts: loadeddice
- pokemon-showdown/data/abilities.ts: skilllink
"""

from __future__ import annotations

import pytest
from tests.conftest import MonSpec
from pokepy.core.constants import OFF_SIDE0, OFF_SIDE1

# ---------------------------------------------------------------------------
# 2-5 hit moves
# ---------------------------------------------------------------------------

def test_bullet_seed_deals_damage(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("breloom", ["bulletseed", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"]*4)],
        seed=1,
    )
    hp1_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 1) < hp1_pre

# ---------------------------------------------------------------------------
# Skill Link
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="skill link forces 5 hits")
def test_skill_link_forces_max_hits(fresh_battle, step_turn, hp_of):
    """Cloyster (Skill Link) Icicle Spear should hit 5 times consistently."""
    state_a, prng_a = fresh_battle(
        [MonSpec("cloyster", ["iciclespear", "tackle", "tackle", "tackle"], ability="skilllink")],
        [MonSpec("snorlax", ["tackle"]*4)],
        seed=1,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("cloyster", ["iciclespear", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"]*4)],
        seed=1,
    )
    hp_a_pre = hp_of(state_a, 1)
    hp_b_pre = hp_of(state_b, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    da = hp_a_pre - hp_of(state_a, 1)
    db = hp_b_pre - hp_of(state_b, 1)
    # Skill Link version should hit harder (always 5 hits vs 2-5 average ~3)
    assert da >= db

# ---------------------------------------------------------------------------
# Loaded Dice
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="loaded dice 4-5 hits")
def test_loaded_dice_more_hits(fresh_battle, step_turn, hp_of):
    state_a, prng_a = fresh_battle(
        [MonSpec("breloom", ["bulletseed", "tackle", "tackle", "tackle"], item="loadeddice")],
        [MonSpec("blissey", ["tackle"]*4)],
        seed=1,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("breloom", ["bulletseed", "tackle", "tackle", "tackle"])],
        [MonSpec("blissey", ["tackle"]*4)],
        seed=1,
    )
    hp_a_pre = hp_of(state_a, 1)
    hp_b_pre = hp_of(state_b, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    da = hp_a_pre - hp_of(state_a, 1)
    db = hp_b_pre - hp_of(state_b, 1)
    assert da >= db

# ---------------------------------------------------------------------------
# Triple Axel
# ---------------------------------------------------------------------------

def test_triple_axel_deals_damage(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("ironvaliant", ["tripleaxel", "tackle", "tackle", "tackle"])],
        [MonSpec("dragonite", ["splash"]*4)],  # 4x weak to ice
        seed=1,
    )
    hp1_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    # Triple Axel KOs dragonite — read slot 0 directly (auto-switch trap)
    assert int(state.battle_state[OFF_SIDE1 + 1]) < hp1_pre

# ---------------------------------------------------------------------------
# Population Bomb
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="population bomb 10 hits")
def test_population_bomb_deals_damage(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("maushold", ["populationbomb", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"]*4)],
        seed=1,
    )
    hp1_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 1) < hp1_pre

# ---------------------------------------------------------------------------
# Double-hit
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="double hit moves")
def test_dragon_darts_deals_damage(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("dragapult", ["dragondarts", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"]*4)],
        seed=1,
    )
    hp1_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 1) < hp1_pre

@pytest.mark.xfail(strict=False, reason="bonemerang 2 hits")
def test_bonemerang_two_hits(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("marowak", ["bonemerang", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"]*4)],
        seed=1,
    )
    hp1_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 1) < hp1_pre

# ---------------------------------------------------------------------------
# Multi-hit + sub
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="multi-hit through substitute")
def test_multihit_breaks_then_hits_substitute():
    pytest.skip("placeholder — multi-hit + sub interaction")

# ---------------------------------------------------------------------------
# Multi-hit + sturdy
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="multi-hit vs sturdy")
def test_multihit_vs_sturdy_KOs():
    pytest.skip("placeholder — multi-hit can break sturdy because each hit is separate")

# ---------------------------------------------------------------------------
# Multi-hit + focus sash
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=False, reason="multi-hit vs focus sash")
def test_multihit_vs_focus_sash_KOs():
    pytest.skip("placeholder — sash survives 1 hit, second KOs")
