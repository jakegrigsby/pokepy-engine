"""Contact-effect mechanics.

Tests interactions when a contact move hits the defender:
- Rocky Helmet: 1/6 max HP recoil to attacker
- Iron Barbs / Rough Skin: 1/8 max HP recoil to attacker
- Static: 30% paralyze
- Flame Body: 30% burn
- Poison Point: 30% poison
- Effect Spore: 33% to inflict poison/para/sleep
- Mummy: changes attacker ability
- Lingering Aroma: changes attacker ability
- Wandering Spirit: swaps abilities
- Tangling Hair / Cotton Down: -1 spe
- Gooey: -1 spe
- Stamina: +1 def on hit
- Weak Armor: -1 def +2 spe on physical hit
- Anger Shell: at <50%, +1 atk/spa/spe -1 def/spd
- Long Reach: contact moves don't make contact
- Protective Pads: protects user from contact effects

Showdown reference:
- pokemon-showdown/data/abilities.ts: search ability names
- pokemon-showdown/data/items.ts: rockyhelmet, protectivepads
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
    STATUS_POISON,
)

# ---------------------------------------------------------------------------
# Rocky Helmet
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="rocky helmet 1/6 recoil")
def test_rocky_helmet_recoil(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["dragonclaw", "tackle", "tackle", "tackle"])],
        [MonSpec("ferrothorn", ["tackle"] * 4, item="rockyhelmet")],
        seed=1,
    )
    hp0_pre = hp_of(state, 0)
    max0 = max_hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    # Garchomp's contact move hit; rocky helmet should chip 1/6 max HP
    helmet_dmg = max0 // 6
    actual = hp0_pre - hp_of(state, 0)
    assert actual >= helmet_dmg - 2  # allow off-by-1


# ---------------------------------------------------------------------------
# Iron Barbs / Rough Skin
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="iron barbs 1/8 recoil")
def test_iron_barbs_recoil(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["dragonclaw", "tackle", "tackle", "tackle"])],
        [MonSpec("ferrothorn", ["tackle"] * 4, ability="ironbarbs")],
        seed=1,
    )
    hp0_pre = hp_of(state, 0)
    max0 = max_hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    barbs_dmg = max0 // 8
    actual = hp0_pre - hp_of(state, 0)
    assert actual >= barbs_dmg - 2


@pytest.mark.xfail(strict=False, reason="rough skin 1/8 recoil")
def test_rough_skin_recoil(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["bodyslam", "tackle", "tackle", "tackle"])],
        [MonSpec("garchomp", ["tackle"] * 4, ability="roughskin")],
        seed=1,
    )
    hp0_pre = hp_of(state, 0)
    max0 = max_hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    actual = hp0_pre - hp_of(state, 0)
    # Body slam contact triggers rough skin
    assert actual >= (max0 // 8) - 2


# ---------------------------------------------------------------------------
# Static / Flame Body / Poison Point (RNG-driven, just check no crash)
# ---------------------------------------------------------------------------


def test_static_does_not_crash(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["dragonclaw", "tackle", "tackle", "tackle"])],
        [MonSpec("jolteon", ["tackle"] * 4, ability="static")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)


def test_flame_body_does_not_crash(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["dragonclaw", "tackle", "tackle", "tackle"])],
        [MonSpec("magcargo", ["tackle"] * 4, ability="flamebody")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)


def test_poison_point_does_not_crash(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["dragonclaw", "tackle", "tackle", "tackle"])],
        [MonSpec("nidoking", ["tackle"] * 4, ability="poisonpoint")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)


# ---------------------------------------------------------------------------
# Mummy
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="mummy changes attacker ability")
def test_mummy_changes_attacker_ability(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [
            MonSpec(
                "garchomp",
                ["dragonclaw", "tackle", "tackle", "tackle"],
                ability="sandveil",
            )
        ],
        [MonSpec("cofagrigus", ["tackle"] * 4, ability="mummy")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    # Garchomp's ability slot should now be Mummy id
    from pokepy.core.constants import ABILITY_MUMMY

    assert int(state.battle_state[OFF_SIDE0 + 5]) == ABILITY_MUMMY


# ---------------------------------------------------------------------------
# Wandering Spirit
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="wandering spirit swaps abilities")
def test_wandering_spirit_swaps(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [
            MonSpec(
                "garchomp",
                ["dragonclaw", "tackle", "tackle", "tackle"],
                ability="sandveil",
            )
        ],
        [MonSpec("runerigus", ["tackle"] * 4, ability="wanderingspirit")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    # Now garchomp should have wandering spirit and runerigus has sand veil
    from pokepy.core.constants import (
        ABILITY_WANDERING_SPIRIT,
        ABILITY_SAND_VEIL,
    )

    assert int(state.battle_state[OFF_SIDE0 + 5]) == ABILITY_WANDERING_SPIRIT
    assert int(state.battle_state[OFF_SIDE1 + 5]) == ABILITY_SAND_VEIL


# ---------------------------------------------------------------------------
# Stamina
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="stamina +1 def on hit")
def test_stamina_def_boost(fresh_battle, step_turn, boost_of):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["dragonclaw", "tackle", "tackle", "tackle"])],
        [MonSpec("mudsdale", ["tackle"] * 4, ability="stamina")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert boost_of(state, 1, "def") >= 1


# ---------------------------------------------------------------------------
# Weak Armor
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="weak armor +2 spe -1 def on physical")
def test_weak_armor_boosts(fresh_battle, step_turn, boost_of):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["dragonclaw", "tackle", "tackle", "tackle"])],
        [MonSpec("skarmory", ["tackle"] * 4, ability="weakarmor")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert boost_of(state, 1, "def") <= -1
    assert boost_of(state, 1, "spe") >= 2


# ---------------------------------------------------------------------------
# Anger Shell
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="anger shell at <50% HP")
def test_anger_shell_triggers_below_half():
    pytest.skip("placeholder — needs HP setup")


# ---------------------------------------------------------------------------
# Cotton Down
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="cotton down -1 spe to attacker")
def test_cotton_down_drops_attacker_spe(fresh_battle, step_turn, boost_of):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["dragonclaw", "tackle", "tackle", "tackle"])],
        [MonSpec("eldegoss", ["tackle"] * 4, ability="cottondown")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert boost_of(state, 0, "spe") <= -1


# ---------------------------------------------------------------------------
# Long Reach
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="long reach negates contact")
def test_long_reach_avoids_rocky_helmet(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [
            MonSpec(
                "decidueye",
                ["leafblade", "tackle", "tackle", "tackle"],
                ability="longreach",
            )
        ],
        [MonSpec("ferrothorn", ["tackle"] * 4, item="rockyhelmet")],
        seed=1,
    )
    hp0_pre = hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    # Decidueye should take 0 contact damage from helmet
    assert hp_of(state, 0) == hp0_pre


# ---------------------------------------------------------------------------
# Protective Pads
# ---------------------------------------------------------------------------


def test_protective_pads_avoid_helmet(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [
            MonSpec(
                "garchomp",
                ["dragonclaw", "tackle", "tackle", "tackle"],
                item="protectivepads",
            )
        ],
        [MonSpec("ferrothorn", ["splash"] * 4, item="rockyhelmet")],
        seed=1,
    )
    hp0_pre = hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 0) == hp0_pre
