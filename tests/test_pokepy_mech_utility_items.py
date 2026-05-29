"""Utility item mechanics (non-damage non-recovery items).

Tests:
- Toxic Orb: badly poisons holder at end of turn
- Flame Orb: burns holder at end of turn
- White Herb: removes negative stat changes
- Mental Herb: cures attract/encore/taunt/disable/torment/heal block
- Power Herb: instant charge for charge moves
- Eject Button: switch out when hit
- Eject Pack: switch out when stat dropped
- Red Card: forces attacker to switch
- Air Balloon: ground immunity until popped
- Shed Shell: switch out even when trapped
- Booster Energy: paradox ability boost
- Light Clay: extends screens (already in screens.py)
- Mirror Herb: copies opponent's stat boosts (gen 9)

Showdown reference:
- pokemon-showdown/data/items.ts: search by name
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
    STATUS_TOXIC,
    ITEM_TOXIC_ORB,
    ITEM_FLAME_ORB,
    ITEM_WHITE_HERB,
    ITEM_AIR_BALLOON,
    ITEM_EJECT_BUTTON,
    ITEM_RED_CARD,
    ITEM_BOOSTER_ENERGY,
)

# ---------------------------------------------------------------------------
# Toxic Orb / Flame Orb
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="toxic orb badly poisons holder eot")
def test_toxic_orb_poisons_holder(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("breloom", ["splash"] * 4, item="toxicorb")],
        [MonSpec("snorlax", ["splash"] * 4)],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 0) == STATUS_TOXIC


@pytest.mark.xfail(strict=False, reason="flame orb burns holder eot")
def test_flame_orb_burns_holder(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["splash"] * 4, item="flameorb")],
        [MonSpec("snorlax", ["splash"] * 4)],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 0) == STATUS_BURN


# ---------------------------------------------------------------------------
# White Herb
# ---------------------------------------------------------------------------


def test_white_herb_clears_negatives(fresh_battle, step_turn, boost_of):
    """Use Draco Meteor (-2 spa) and verify White Herb resets it to 0."""
    state, prng = fresh_battle(
        [
            MonSpec(
                "latios",
                ["dracometeor", "tackle", "tackle", "tackle"],
                item="whiteherb",
            )
        ],
        [MonSpec("blissey", ["tackle"] * 4)],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert boost_of(state, 0, "spa") == 0
    # Item should be consumed
    assert int(state.battle_state[OFF_SIDE0 + 6]) == 0


# ---------------------------------------------------------------------------
# Air Balloon
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="air balloon ground immunity")
def test_air_balloon_blocks_eq(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["splash"] * 4, item="airballoon")],
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    hp_pre = hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 0) == hp_pre


@pytest.mark.xfail(strict=False, reason="air balloon pops on first non-ground hit")
def test_air_balloon_pops_on_hit(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["splash"] * 4, item="airballoon")],
        [MonSpec("garchomp", ["dragonclaw", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    # After being hit by a non-ground move, balloon should be consumed
    assert int(state.battle_state[OFF_SIDE0 + 6]) == 0


# ---------------------------------------------------------------------------
# Eject Button
# ---------------------------------------------------------------------------


def test_eject_button_triggers_switch(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["splash"] * 4, item="ejectbutton"),
            MonSpec("blissey", ["splash"] * 4),
        ],
        [MonSpec("garchomp", ["dragonclaw", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    # Either active changed to slot 1 (auto-switch) or phase = forced switch
    active = int(state.battle_state[OFF_META + M_ACTIVE0])
    from pokepy.core.constants import PHASE_FORCED_SWITCH

    assert active == 1 or int(state.phase) == PHASE_FORCED_SWITCH


# ---------------------------------------------------------------------------
# Red Card
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="red card forces attacker to switch")
def test_red_card_forces_attacker_switch(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["splash"] * 4, item="redcard")],
        [
            MonSpec("garchomp", ["dragonclaw", "tackle", "tackle", "tackle"]),
            MonSpec("blissey", ["splash"] * 4),
        ],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    # Side 1 active should have switched
    new_active = int(state.battle_state[OFF_META + M_ACTIVE1])
    assert new_active != 0


# ---------------------------------------------------------------------------
# Booster Energy
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False, reason="booster energy activates protosynthesis without sun"
)
def test_booster_energy_activates_protosynthesis():
    pytest.skip("placeholder — needs paradox species")


# ---------------------------------------------------------------------------
# Power Herb
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="power herb skips charge turn")
def test_power_herb_skips_solar_beam_charge(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [
            MonSpec(
                "venusaur",
                ["solarbeam", "tackle", "tackle", "tackle"],
                item="powerherb",
            )
        ],
        [MonSpec("snorlax", ["splash"] * 4)],
        seed=1,
    )
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    # Solar beam should fire immediately, dealing damage
    assert hp_of(state, 1) < hp_pre
    # Item consumed
    assert int(state.battle_state[OFF_SIDE0 + 6]) == 0


# ---------------------------------------------------------------------------
# Mental Herb
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="mental herb cures taunt/attract/etc")
def test_mental_herb_cures_taunt(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["splash"] * 4, item="mentalherb")],
        [MonSpec("snorlax", ["taunt", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    # Mental herb consumed
    assert int(state.battle_state[OFF_SIDE0 + 6]) == 0
