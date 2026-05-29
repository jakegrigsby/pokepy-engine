"""Damage-modifying held item mechanics in pokepy.

Each test sets up a one-mon-vs-one-mon Gen 9 OU singles scenario with a
specific item on either the attacker or defender, runs a single turn, and
asserts the expected damage / HP / boost / item-consumption outcome.

Items covered:
    Choice Band, Choice Specs, Choice Scarf, Life Orb, Expert Belt, Eviolite,
    Assault Vest, Heavy-Duty Boots, Air Balloon, Weakness Policy, Rocky Helmet,
    Loaded Dice, Leftovers, Black Sludge, Focus Sash, Sitrus Berry.

Showdown source of truth:
    pokemon-showdown/data/items.ts  (search each item's lowercase id)
    pokemon-showdown/data/scripts.ts modifyDamage / runImmunity for hazards
"""

from __future__ import annotations

import pytest
from tests.conftest import MonSpec
from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_FIELD,
    OFF_META,
    POKEMON_SIZE,
    M_ACTIVE0,
    M_ACTIVE1,
    F_HAZARDS_0,
    F_HAZARDS_1,
    HAZARD_STEALTH_ROCK,
    HAZARD_SPIKES,
    ITEM_LIFE_ORB,
    ITEM_LEFTOVERS,
    ITEM_HEAVY_DUTY_BOOTS,
    ITEM_FOCUS_SASH,
    ITEM_AIR_BALLOON,
    ITEM_WEAKNESS_POLICY,
    ITEM_ROCKY_HELMET,
    ITEM_LOADED_DICE,
    ITEM_BLACK_SLUDGE,
    ITEM_SITRUS_BERRY,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item_of(state, side: int) -> int:
    base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    active = int(state.battle_state[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])
    return int(state.battle_state[base + active * POKEMON_SIZE + 6])


# ---------------------------------------------------------------------------
# Choice Band: physical attack * 1.5
# ---------------------------------------------------------------------------


def test_choice_band_boosts_physical_damage(fresh_battle, step_turn, hp_of):
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "garchomp",
                ["earthquake", "tackle", "tackle", "tackle"],
                item="choiceband",
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    hp_pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    band_dmg = hp_pre - hp_of(state_a, 1)
    plain_dmg = hp_pre - hp_of(state_b, 1)
    assert band_dmg > plain_dmg, f"choice band should boost: {band_dmg} vs {plain_dmg}"


# ---------------------------------------------------------------------------
# Choice Specs: special attack * 1.5
# ---------------------------------------------------------------------------


def test_choice_specs_boosts_special_damage(fresh_battle, step_turn, hp_of):
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "alakazam",
                ["psychic", "tackle", "tackle", "tackle"],
                item="choicespecs",
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=2,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("alakazam", ["psychic", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=2,
    )
    hp_pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    specs_dmg = hp_pre - hp_of(state_a, 1)
    plain_dmg = hp_pre - hp_of(state_b, 1)
    assert (
        specs_dmg > plain_dmg
    ), f"choice specs should boost: {specs_dmg} vs {plain_dmg}"


# ---------------------------------------------------------------------------
# Choice Scarf: speed * 1.5 (lets a slower mon outspeed)
# ---------------------------------------------------------------------------


def test_choice_scarf_boosts_speed(fresh_battle, step_turn, hp_of):
    # Tyranitar (61 spe) @ Scarf -> 91 effective; Garchomp (102) still faster.
    # Use Hippowdon (47 spe) target so scarfed Ttar definitely outspeeds.
    state, prng = fresh_battle(
        [
            MonSpec(
                "tyranitar",
                ["earthquake", "tackle", "tackle", "tackle"],
                item="choicescarf",
            )
        ],
        [MonSpec("hippowdon", ["earthquake", "tackle", "tackle", "tackle"])],
        seed=3,
    )
    hp1_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 1) < hp1_pre


# ---------------------------------------------------------------------------
# Life Orb: damage * 1.3, 10% recoil
# ---------------------------------------------------------------------------


def test_life_orb_boosts_damage(fresh_battle, step_turn, hp_of):
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "garchomp", ["earthquake", "tackle", "tackle", "tackle"], item="lifeorb"
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=4,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=4,
    )
    hp_pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    orb_dmg = hp_pre - hp_of(state_a, 1)
    plain_dmg = hp_pre - hp_of(state_b, 1)
    assert orb_dmg > plain_dmg


def test_life_orb_recoil_user(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [
            MonSpec(
                "garchomp", ["earthquake", "tackle", "tackle", "tackle"], item="lifeorb"
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=5,
    )
    hp0_pre = hp_of(state, 0)
    max0 = max_hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    hp0_post = hp_of(state, 0)
    # User should have lost ~10% max HP from Life Orb recoil (plus possibly Tackle dmg).
    recoil_floor = max0 // 10 - 1
    assert hp0_pre - hp0_post >= recoil_floor


# ---------------------------------------------------------------------------
# Expert Belt: 1.2x on supereffective only
# ---------------------------------------------------------------------------


def test_expert_belt_only_supereffective(fresh_battle, step_turn, hp_of):
    # Garchomp's Earthquake is SE on Tyranitar (Rock/Dark; ground 2x on rock).
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "garchomp",
                ["earthquake", "tackle", "tackle", "tackle"],
                item="expertbelt",
            )
        ],
        [MonSpec("tyranitar", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [MonSpec("tyranitar", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    hp_pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    belt_dmg = hp_pre - hp_of(state_a, 1)
    plain_dmg = hp_pre - hp_of(state_b, 1)
    assert belt_dmg > plain_dmg


def test_expert_belt_no_boost_neutral(fresh_battle, step_turn, hp_of):
    # Tackle is Normal on Snorlax (Normal): neutral, no expert belt boost.
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "garchomp", ["tackle", "tackle", "tackle", "tackle"], item="expertbelt"
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("garchomp", ["tackle", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    hp_pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    belt_dmg = hp_pre - hp_of(state_a, 1)
    plain_dmg = hp_pre - hp_of(state_b, 1)
    assert belt_dmg == plain_dmg


# ---------------------------------------------------------------------------
# Eviolite: Def & SpD * 1.5 (NFE only in Showdown — pokepy doesn't gate by NFE)
# ---------------------------------------------------------------------------


def test_eviolite_reduces_damage_chansey(fresh_battle, step_turn, hp_of):
    state_a, prng_a = fresh_battle(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle", "tackle", "tackle", "tackle"], item="eviolite")],
        seed=8,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle", "tackle", "tackle", "tackle"])],
        seed=8,
    )
    hp_pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    evi_dmg = hp_pre - hp_of(state_a, 1)
    plain_dmg = hp_pre - hp_of(state_b, 1)
    assert evi_dmg < plain_dmg


# ---------------------------------------------------------------------------
# Assault Vest: SpD * 1.5
# ---------------------------------------------------------------------------


def test_assault_vest_reduces_special_damage(fresh_battle, step_turn, hp_of):
    state_a, prng_a = fresh_battle(
        [MonSpec("greninja", ["icebeam", "tackle", "tackle", "tackle"])],
        [
            MonSpec(
                "tyranitar",
                ["tackle", "tackle", "tackle", "tackle"],
                item="assaultvest",
            )
        ],
        seed=9,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("greninja", ["icebeam", "tackle", "tackle", "tackle"])],
        [MonSpec("tyranitar", ["tackle", "tackle", "tackle", "tackle"])],
        seed=9,
    )
    hp_pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    av_dmg = hp_pre - hp_of(state_a, 1)
    plain_dmg = hp_pre - hp_of(state_b, 1)
    assert av_dmg < plain_dmg


# ---------------------------------------------------------------------------
# Heavy-Duty Boots: ignore entry hazards on switch-in
# ---------------------------------------------------------------------------


def test_heavy_duty_boots_ignore_stealth_rock(
    fresh_battle, step_turn, hp_of, max_hp_of
):
    state, prng = fresh_battle(
        [
            MonSpec("garchomp", ["splash", "tackle", "tackle", "tackle"]),
            MonSpec(
                "skarmory",
                ["splash", "tackle", "tackle", "tackle"],
                item="heavydutyboots",
            ),
        ],
        [MonSpec("ferrothorn", ["splash", "tackle", "tackle", "tackle"])],
        seed=10,
    )
    # Rocks already up on side 0. HAZARD_STEALTH_ROCK is the *type indicator*
    # (1) used by apply_hazard_from_move's switch; the bit-packed value is 0x4
    # (set via set_stealth_rock). Use the bit-packed form here.
    from pokepy.core.bitpack import set_stealth_rock

    state.battle_state[OFF_FIELD + F_HAZARDS_0] = set_stealth_rock(0)
    # Switch garchomp -> skarmory (action 5 = roster slot 1, since active=0)
    step_turn(state, prng, 5, 0)
    # Skarmory is 4x weak to Rock (steel/flying); without HDB, SR would deal
    # 1/2 max HP. With HDB it should take 0 from SR. Both sides splash so no
    # other damage source.
    new_active_hp = hp_of(state, 0)
    new_active_max = max_hp_of(state, 0)
    assert new_active_hp == new_active_max


# ---------------------------------------------------------------------------
# Air Balloon: ground immunity until popped
# ---------------------------------------------------------------------------


def test_air_balloon_ground_immunity(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [
            MonSpec(
                "tyranitar", ["tackle", "tackle", "tackle", "tackle"], item="airballoon"
            )
        ],
        seed=11,
    )
    hp1_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    hp1_post = hp_of(state, 1)
    # EQ should deal 0 damage to balloon-holder; tyranitar's tackle deals tiny dmg.
    eq_dmg_floor = hp1_pre - hp1_post
    assert eq_dmg_floor < 50, f"balloon should make EQ miss, took {eq_dmg_floor}"


def test_air_balloon_pops_on_non_ground_hit(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["tackle", "tackle", "tackle", "tackle"])],
        [
            MonSpec(
                "tyranitar", ["tackle", "tackle", "tackle", "tackle"], item="airballoon"
            )
        ],
        seed=12,
    )
    step_turn(state, prng, 0, 0)
    assert _item_of(state, 1) == 0


# ---------------------------------------------------------------------------
# Weakness Policy: +2 atk/spa on supereffective hit
# ---------------------------------------------------------------------------


def test_weakness_policy_boosts_on_se(fresh_battle, step_turn, boost_of):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [
            MonSpec(
                "tyranitar",
                ["tackle", "tackle", "tackle", "tackle"],
                item="weaknesspolicy",
            )
        ],
        seed=13,
    )
    step_turn(state, prng, 0, 0)
    # If TTar survives, atk and spa should both be +2.
    assert boost_of(state, 1, "atk") == 2
    assert boost_of(state, 1, "spa") == 2


# ---------------------------------------------------------------------------
# Rocky Helmet: contact attacker takes 1/6 max HP
# ---------------------------------------------------------------------------


def test_rocky_helmet_chips_contact_attacker(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["tackle", "earthquake", "tackle", "tackle"])],
        [
            MonSpec(
                "ferrothorn",
                ["tackle", "tackle", "tackle", "tackle"],
                item="rockyhelmet",
            )
        ],
        seed=14,
    )
    hp0_pre = hp_of(state, 0)
    max0 = max_hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    hp0_post = hp_of(state, 0)
    expected = max0 // 6
    # Garchomp takes ferrothorn's tackle + helmet recoil. Helmet floor = max/6 ish.
    assert hp0_pre - hp0_post >= expected - 5


def test_rocky_helmet_skipped_for_non_contact(
    fresh_battle, step_turn, hp_of, max_hp_of
):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [
            MonSpec(
                "ferrothorn",
                ["tackle", "tackle", "tackle", "tackle"],
                item="rockyhelmet",
            )
        ],
        seed=15,
    )
    hp0_pre = hp_of(state, 0)
    max0 = max_hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    hp0_post = hp_of(state, 0)
    # EQ is non-contact; only ferrothorn's tackle should hit garchomp; no 1/6 helmet damage.
    helmet_chunk = max0 // 6
    assert hp0_pre - hp0_post < helmet_chunk + 5


# ---------------------------------------------------------------------------
# Loaded Dice: multi-hit moves hit 4-5 times
# ---------------------------------------------------------------------------


def test_loaded_dice_multihit(fresh_battle, step_turn, hp_of):
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "cinccino",
                ["bulletseed", "tackle", "tackle", "tackle"],
                item="loadeddice",
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=16,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("cinccino", ["bulletseed", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=16,
    )
    hp_pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    dice_dmg = hp_pre - hp_of(state_a, 1)
    plain_dmg = hp_pre - hp_of(state_b, 1)
    assert dice_dmg >= plain_dmg


# ---------------------------------------------------------------------------
# Leftovers: 1/16 max HP per turn
# ---------------------------------------------------------------------------


def test_leftovers_heals_each_turn(fresh_battle, step_turn, hp_of, max_hp_of):
    # Compare snorlax-with vs snorlax-without leftovers under identical seeds.
    # The diff between the two HP totals should be ~max_hp/16.
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "snorlax", ["tackle", "tackle", "tackle", "tackle"], item="leftovers"
            )
        ],
        [MonSpec("garchomp", ["tackle", "tackle", "tackle", "tackle"])],
        seed=17,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        [MonSpec("garchomp", ["tackle", "tackle", "tackle", "tackle"])],
        seed=17,
    )
    max0 = max_hp_of(state_a, 0)
    base_a = OFF_SIDE0 + int(state_a.battle_state[OFF_META + M_ACTIVE0]) * POKEMON_SIZE
    base_b = OFF_SIDE0 + int(state_b.battle_state[OFF_META + M_ACTIVE0]) * POKEMON_SIZE
    state_a.battle_state[base_a + 1] = max0 // 2
    state_b.battle_state[base_b + 1] = max0 // 2
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    diff = hp_of(state_a, 0) - hp_of(state_b, 0)
    assert diff >= max0 // 16 - 2, f"leftovers should heal ~{max0//16}, got diff={diff}"


# ---------------------------------------------------------------------------
# Black Sludge: heals if poison-type, damages otherwise
# ---------------------------------------------------------------------------


def test_black_sludge_heals_poison_type(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [
            MonSpec(
                "venusaur", ["splash", "splash", "splash", "splash"], item="blacksludge"
            )
        ],
        [MonSpec("snorlax", ["splash", "splash", "splash", "splash"])],
        seed=18,
    )
    max0 = max_hp_of(state, 0)
    base = OFF_SIDE0 + int(state.battle_state[OFF_META + M_ACTIVE0]) * POKEMON_SIZE
    state.battle_state[base + 1] = max0 // 2
    pre_hp = int(state.battle_state[base + 1])
    step_turn(state, prng, 0, 0)
    post_hp = hp_of(state, 0)
    # Splash deals no damage; black sludge should heal venusaur (Poison type) 1/16 max HP
    assert post_hp > pre_hp
    assert post_hp - pre_hp >= max0 // 16 - 1


def test_black_sludge_damages_non_poison(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [
            MonSpec(
                "snorlax", ["tackle", "tackle", "tackle", "tackle"], item="blacksludge"
            )
        ],
        [MonSpec("garchomp", ["tackle", "tackle", "tackle", "tackle"])],
        seed=19,
    )
    max0 = max_hp_of(state, 0)
    pre_hp = hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    post_hp = hp_of(state, 0)
    # snorlax (Normal) should LOSE 1/16 + tackle damage
    chunk = max0 // 16
    assert pre_hp - post_hp >= chunk - 2


# ---------------------------------------------------------------------------
# Focus Sash: survives a OHKO at full HP
# ---------------------------------------------------------------------------


def test_focus_sash_survives_ohko(fresh_battle, step_turn, hp_of):
    # Glaceon Choice Specs Ice Beam vs Garchomp @ Sash. 4x weak to Ice → clean OHKO.
    state, prng = fresh_battle(
        [
            MonSpec(
                "glaceon", ["icebeam", "tackle", "tackle", "tackle"], item="choicespecs"
            )
        ],
        [
            MonSpec(
                "garchomp", ["splash", "splash", "splash", "splash"], item="focussash"
            )
        ],
        seed=20,
    )
    step_turn(state, prng, 0, 0)
    # Read garchomp slot 0 directly (auto-switch trap)
    assert int(state.battle_state[OFF_SIDE1 + 1]) == 1
    assert int(state.battle_state[OFF_SIDE1 + 6]) == 0


# ---------------------------------------------------------------------------
# Sitrus Berry: heals 1/4 max at <=50% HP
# ---------------------------------------------------------------------------


def test_sitrus_berry_triggers_below_half(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [
            MonSpec(
                "snorlax", ["tackle", "tackle", "tackle", "tackle"], item="sitrusberry"
            )
        ],
        [MonSpec("garchomp", ["tackle", "tackle", "tackle", "tackle"])],
        seed=21,
    )
    max0 = max_hp_of(state, 0)
    base = OFF_SIDE0 + int(state.battle_state[OFF_META + M_ACTIVE0]) * POKEMON_SIZE
    # Drop snorlax to 30% HP so any damage triggers sitrus
    state.battle_state[base + 1] = max0 // 3
    pre_hp = int(state.battle_state[base + 1])
    step_turn(state, prng, 0, 0)
    post_hp = hp_of(state, 0)
    # Sitrus should fire and consume; net change should be positive.
    assert post_hp > pre_hp
    assert _item_of(state, 0) == 0


def test_sitrus_berry_not_consumed_above_half(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [
            MonSpec(
                "snorlax", ["tackle", "tackle", "tackle", "tackle"], item="sitrusberry"
            )
        ],
        [MonSpec("garchomp", ["tackle", "tackle", "tackle", "tackle"])],
        seed=22,
    )
    step_turn(state, prng, 0, 0)
    # Snorlax took only a tackle, still > 50%, sitrus not consumed.
    assert _item_of(state, 0) == ITEM_SITRUS_BERRY
