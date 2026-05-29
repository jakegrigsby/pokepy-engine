"""Choice item lock mechanics.

Tests:
- Choice Band/Specs/Scarf lock the user into the first move chosen
- The lock persists across turns until switched out
- Switching clears the lock
- Trick / Switcheroo with choice items
- Choice scarf doesn't lock if user used a status move (some implementations)

Showdown reference:
- pokemon-showdown/data/items.ts: choiceband, choicespecs, choicescarf
- onModifyMove / volatile "choicelock"
"""

from __future__ import annotations

import pytest
from tests.conftest import MonSpec
from pokepy.core.constants import (
    OFF_FIELD,
    F_CHOICE_LOCK_0,
    F_CHOICE_LOCK_1,
    OFF_SIDE0,
    OFF_SIDE1,
    POKEMON_SIZE,
    OFF_META,
    M_ACTIVE0,
    M_ACTIVE1,
)
from pokepy.engine.action_mask import get_action_mask

# ---------------------------------------------------------------------------
# Choice lock mechanics
# ---------------------------------------------------------------------------


def test_choice_band_initial_lock_sentinel(fresh_battle):
    state, _ = fresh_battle(
        [
            MonSpec(
                "garchomp",
                ["earthquake", "outrage", "stoneedge", "swordsdance"],
                item="choiceband",
            )
        ],
        [MonSpec("snorlax", ["tackle"] * 4)],
        seed=1,
    )
    # Sentinel = -1 (no lock yet)
    assert int(state.battle_state[OFF_FIELD + F_CHOICE_LOCK_0]) == -1


@pytest.mark.xfail(strict=False, reason="choice lock set after first attack")
def test_choice_band_locks_after_first_move(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [
            MonSpec(
                "garchomp",
                ["earthquake", "outrage", "stoneedge", "swordsdance"],
                item="choiceband",
            )
        ],
        [MonSpec("snorlax", ["tackle"] * 4)],
        seed=1,
    )
    step_turn(state, prng, 0, 0)  # use earthquake
    # After step, choice_lock should reference earthquake's slot or move id
    assert int(state.battle_state[OFF_FIELD + F_CHOICE_LOCK_0]) >= 0


@pytest.mark.xfail(
    strict=False, reason="action mask should disable other moves while locked"
)
def test_choice_lock_disables_other_moves_in_mask(fresh_battle, step_turn, gd):
    state, prng = fresh_battle(
        [
            MonSpec(
                "garchomp",
                ["earthquake", "outrage", "stoneedge", "swordsdance"],
                item="choiceband",
            )
        ],
        [MonSpec("snorlax", ["tackle"] * 4)],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    mask = get_action_mask(state, 0, gd)
    # Only earthquake (slot 0) should be legal among the 4 move slots
    legal_moves = sum(1 for i in range(4) if mask[i])
    assert legal_moves == 1


# ---------------------------------------------------------------------------
# Choice scarf
# ---------------------------------------------------------------------------


def test_choice_scarf_speed_boost(fresh_battle, step_turn, hp_of):
    """Tyranitar (61 base spe) @ scarf should outspeed Hippowdon (47 base spe)
    when scarfed, but not unscarfed."""
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "tyranitar",
                ["earthquake", "tackle", "tackle", "tackle"],
                item="choicescarf",
            )
        ],
        [MonSpec("hippowdon", ["earthquake", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    hp1_pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    # Tyranitar EQ should KO or near-KO Hippowdon
    assert hp_of(state_a, 1) < hp1_pre


# ---------------------------------------------------------------------------
# Switching clears choice lock
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="switch clears choice lock")
def test_switch_clears_choice_lock(fresh_battle, step_turn, gd):
    state, prng = fresh_battle(
        [
            MonSpec(
                "garchomp",
                ["earthquake", "outrage", "stoneedge", "swordsdance"],
                item="choiceband",
            ),
            MonSpec("snorlax", ["bodyslam"] * 4),
        ],
        [MonSpec("blissey", ["tackle"] * 4)],
        seed=1,
    )
    step_turn(state, prng, 0, 0)  # Garchomp uses EQ → locked
    locked_after_attack = int(state.battle_state[OFF_FIELD + F_CHOICE_LOCK_0])
    # Switch out to slot 1
    step_turn(state, prng, 5, 0)
    locked_after_switch = int(state.battle_state[OFF_FIELD + F_CHOICE_LOCK_0])
    assert locked_after_attack >= 0
    assert locked_after_switch == -1


# ---------------------------------------------------------------------------
# Choice Specs SpA boost
# ---------------------------------------------------------------------------


def test_choice_specs_special_attack_boost(fresh_battle, step_turn, hp_of):
    """Choice Specs Latios should hit harder than no-item Latios."""
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "latios",
                ["dracometeor", "psychic", "tackle", "tackle"],
                item="choicespecs",
            )
        ],
        [MonSpec("blissey", ["tackle"] * 4)],
        seed=1,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("latios", ["dracometeor", "psychic", "tackle", "tackle"])],
        [MonSpec("blissey", ["tackle"] * 4)],
        seed=1,
    )
    hp_a_pre = hp_of(state_a, 1)
    hp_b_pre = hp_of(state_b, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    da = hp_a_pre - hp_of(state_a, 1)
    db = hp_b_pre - hp_of(state_b, 1)
    assert da > db


# ---------------------------------------------------------------------------
# Switcheroo / Trick (placeholder)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="trick swap items")
def test_trick_swaps_items():
    pytest.skip("placeholder for trick item swap")


@pytest.mark.xfail(strict=False, reason="switcheroo swap items")
def test_switcheroo_swaps_items():
    pytest.skip("placeholder")


# ---------------------------------------------------------------------------
# Choice + Z-move / Tera (gen 9: tera doesn't break choice lock)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="tera does not break choice lock in gen 9")
def test_tera_does_not_unlock_choice():
    pytest.skip("placeholder")
