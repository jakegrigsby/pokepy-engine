"""Offensive ability mechanics for pokepy.

Each test sets up two parallel one-mon-vs-one-mon Gen 9 OU singles scenarios
with the same seed and team, the only difference being the attacker's ability
(target ability vs a no-op control like ``keeneye``/``runaway``). After running
one turn the test compares HP deltas on the defender to confirm the ability
applied the expected damage multiplier.

A handful of abilities are exercised against a baseline scenario where their
trigger condition is missing (e.g., contact vs non-contact for Tough Claws,
pulse vs non-pulse for Mega Launcher) instead of an ability swap.

Showdown source of truth:
    pokemon-showdown/data/abilities.ts
    Search for: "Huge Power", "Pure Power", "Adaptability", "Sheer Force",
    "Tough Claws", "Iron Fist", "Strong Jaw", "Mega Launcher", "Punk Rock",
    "Reckless", "Steelworker", "Dragon's Maw", "Rocky Payload", "Stakeout",
    "Tinted Lens", "Sniper", "Water Bubble", "Neuroforce", "Analytic",
    "Gorilla Tactics", "Hustle", "Overgrow", "Blaze", "Torrent", "Swarm",
    "Solar Power", "Mold Breaker", "Teravolt", "Turboblaze", "Sharpness",
    "Skill Link".

pokepy implementation:
    pokepy/mechanics/damage_gen9.py — most attacker-side multipliers wired
        directly into the damage calc (huge_power, sheer_force, tough_claws,
        iron_fist, strong_jaw, mega_launcher, punk_rock, reckless, steelworker,
        dragons_maw, rocky_payload, neuroforce, analytic, pinch types,
        gorilla_tactics, water_bubble, sharpness, sniper, tinted_lens, ...).
    pokepy/effects/abilities.py — passive triggers (Solar Power EOT damage,
        choice locks, etc.).
"""

from __future__ import annotations

import pytest

from tests.conftest import MonSpec
from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_META,
    OFF_FIELD,
    POKEMON_SIZE,
    M_ACTIVE0,
    M_ACTIVE1,
    M_WEATHER_TURNS,
    F_WEATHER,
    F_LAST_MOVE_0,
    F_LAST_MOVE_1,
    WEATHER_SUN,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Use a deterministic no-op ability ("keeneye" only modifies accuracy/evasion
# interaction; "runaway" affects only escape from wild battles). Both have
# zero effect on damage rolls in pokepy's damage_gen9 path.
NO_OP = "keeneye"


def _dmg(state_pre_hp: int, state, hp_of_fn) -> int:
    return state_pre_hp - hp_of_fn(state, 1)


def _run_pair(
    fresh_battle,
    step_turn,
    hp_of,
    attacker,
    target,
    move,
    ability_a,
    ability_b=NO_OP,
    seed: int = 7,
    side_to_assert: int = 1,
):
    """Build two identical battles differing only in attacker ability and
    return (dmg_with_ability, dmg_without_ability) on side ``side_to_assert``."""
    state_a, prng_a = fresh_battle(
        [MonSpec(attacker, [move, "tackle", "tackle", "tackle"], ability=ability_a)],
        [MonSpec(target, ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=seed,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec(attacker, [move, "tackle", "tackle", "tackle"], ability=ability_b)],
        [MonSpec(target, ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=seed,
    )
    pre_a = hp_of(state_a, side_to_assert)
    pre_b = hp_of(state_b, side_to_assert)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    return (
        pre_a - hp_of(state_a, side_to_assert),
        pre_b - hp_of(state_b, side_to_assert),
    )


# ---------------------------------------------------------------------------
# 1. Huge Power — Atk * 2
# ---------------------------------------------------------------------------


def test_huge_power_doubles_physical_damage(fresh_battle, step_turn, hp_of):
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "azumarill",
        "snorlax",
        "aquajet",
        "hugepower",
        seed=11,
    )
    # Atk*2 should produce nearly 2x the damage. Allow for damage roll spread.
    assert boosted > plain
    assert boosted >= int(plain * 1.7)


# ---------------------------------------------------------------------------
# 2. Pure Power — same as Huge Power
# ---------------------------------------------------------------------------


def test_pure_power_doubles_physical_damage(fresh_battle, step_turn, hp_of):
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "medicham",
        "snorlax",
        "machpunch",
        "purepower",
        seed=12,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.7)


# ---------------------------------------------------------------------------
# 3. Adaptability — STAB 2.0 instead of 1.5
# ---------------------------------------------------------------------------


def test_adaptability_boosts_stab(fresh_battle, step_turn, hp_of):
    # Dragapult uses Dragon Pulse (dragon STAB) — adaptability vs no ability.
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "dragapult",
        "snorlax",
        "dragonpulse",
        "adaptability",
        seed=13,
    )
    # 2.0 / 1.5 = 1.33x
    assert boosted > plain
    assert boosted >= int(plain * 1.2)


# ---------------------------------------------------------------------------
# 4. Sheer Force — secondary effects removed, BP * 1.3
# ---------------------------------------------------------------------------


def test_sheer_force_boosts_secondary_effect_move(fresh_battle, step_turn, hp_of):
    # Iron Head has 30% flinch chance — Sheer Force eligible.
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "tyranitar",
        "snorlax",
        "ironhead",
        "sheerforce",
        seed=14,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.2)


# ---------------------------------------------------------------------------
# 5. Tough Claws — contact moves * 1.3
# ---------------------------------------------------------------------------


def test_tough_claws_boosts_contact_move(fresh_battle, step_turn, hp_of):
    # Close Combat is contact; baseline = no Tough Claws.
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "scizor",
        "snorlax",
        "closecombat",
        "toughclaws",
        seed=15,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.18)


def test_tough_claws_does_not_boost_non_contact(fresh_battle, step_turn, hp_of):
    # Earthquake is non-contact; toughclaws should match baseline within roll noise.
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "scizor",
        "snorlax",
        "rockslide",
        "toughclaws",
        seed=16,
    )
    # Rock Slide is non-contact, no Tough Claws boost expected.
    assert abs(boosted - plain) <= max(3, int(plain * 0.06))


# ---------------------------------------------------------------------------
# 6. Iron Fist — punch moves * 1.2
# ---------------------------------------------------------------------------


def test_iron_fist_boosts_punch_move(fresh_battle, step_turn, hp_of):
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "blaziken",
        "snorlax",
        "firepunch",
        "ironfist",
        seed=17,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.1)


# ---------------------------------------------------------------------------
# 7. Strong Jaw — bite moves * 1.5
# ---------------------------------------------------------------------------


def test_strong_jaw_boosts_bite_move(fresh_battle, step_turn, hp_of):
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "tyrantrum",
        "snorlax",
        "crunch",
        "strongjaw",
        seed=18,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.35)


# ---------------------------------------------------------------------------
# 8. Mega Launcher — pulse moves * 1.5
# ---------------------------------------------------------------------------


def test_mega_launcher_boosts_pulse_move(fresh_battle, step_turn, hp_of):
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "blastoise",
        "snorlax",
        "aurasphere",
        "megalauncher",
        seed=19,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.35)


# ---------------------------------------------------------------------------
# 9. Punk Rock — sound moves * 1.3 (offense side)
# ---------------------------------------------------------------------------


def test_punk_rock_boosts_sound_move(fresh_battle, step_turn, hp_of):
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "toxtricity",
        "snorlax",
        "boomburst",
        "punkrock",
        seed=20,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.2)


# ---------------------------------------------------------------------------
# 10. Reckless — recoil moves * 1.2
# ---------------------------------------------------------------------------


def test_reckless_boosts_recoil_move(fresh_battle, step_turn, hp_of):
    # Double-Edge has effect_type=RECOIL (10) in pokepy's move tables; pick a
    # carrier that gets STAB so the damage delta is large enough to clear roll
    # noise. (Head Smash is also recoil but has 80 accuracy, which adds RNG.)
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "staraptor",
        "snorlax",
        "doubleedge",
        "reckless",
        seed=21,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.1)


# ---------------------------------------------------------------------------
# 11. Steelworker — steel moves * 1.5
# ---------------------------------------------------------------------------


def test_steelworker_boosts_steel_move(fresh_battle, step_turn, hp_of):
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "duraludon",
        "snorlax",
        "flashcannon",
        "steelworker",
        seed=22,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.35)


# ---------------------------------------------------------------------------
# 12. Dragon's Maw — dragon moves * 1.5
# ---------------------------------------------------------------------------


def test_dragons_maw_boosts_dragon_move(fresh_battle, step_turn, hp_of):
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "dragapult",
        "snorlax",
        "dragonpulse",
        "dragonsmaw",
        seed=23,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.35)


# ---------------------------------------------------------------------------
# 13. Rocky Payload — rock moves * 1.5
# ---------------------------------------------------------------------------


def test_rocky_payload_boosts_rock_move(fresh_battle, step_turn, hp_of):
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "garganacl",
        "snorlax",
        "stoneedge",
        "rockypayload",
        seed=1,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.35)


# ---------------------------------------------------------------------------
# 14. Stakeout — 2x damage if target switched in this turn
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason="stakeout requires synthetic just-switched flag and engine support",
)
def test_stakeout_doubles_vs_just_switched(fresh_battle, step_turn, hp_of):
    # Force F_LAST_MOVE_1 = -1 (defender just switched / no last move).
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "grafaiai", ["tackle", "tackle", "tackle", "tackle"], ability="stakeout"
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=25,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("grafaiai", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=25,
    )
    state_a.battle_state[OFF_FIELD + F_LAST_MOVE_1] = -1
    state_b.battle_state[OFF_FIELD + F_LAST_MOVE_1] = -1
    pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    boosted = pre - hp_of(state_a, 1)
    plain = pre - hp_of(state_b, 1)
    assert boosted >= int(plain * 1.7)


# ---------------------------------------------------------------------------
# 15. Tinted Lens — resisted damage * 2
# ---------------------------------------------------------------------------


def test_tinted_lens_doubles_resisted_damage(fresh_battle, step_turn, hp_of):
    # Venomoth (bug) using X-Scissor (bug) into Heracross (bug/fighting) — bug
    # resisted by fighting half? bug vs bug-fighting: 0.5 * 1 = 0.5 resisted.
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "venomoth",
        "heracross",
        "xscissor",
        "tintedlens",
        seed=26,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.7)


# ---------------------------------------------------------------------------
# 16. Sniper — crit damage * 2.25 instead of 1.5
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="crit RNG dependent; Sniper bonus only on crit")
def test_sniper_boosts_crit_damage(fresh_battle, step_turn, hp_of):
    # Frost Breath always crits — Sniper should make damage > non-Sniper crit.
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "frosmoth",
        "snorlax",
        "frostbreath",
        "sniper",
        seed=27,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.4)


# ---------------------------------------------------------------------------
# 17. Water Bubble — water moves * 2
# ---------------------------------------------------------------------------


def test_water_bubble_doubles_water_damage(fresh_battle, step_turn, hp_of):
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "araquanid",
        "snorlax",
        "liquidation",
        "waterbubble",
        seed=28,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.7)


# ---------------------------------------------------------------------------
# 18. Neuroforce — supereffective * 1.25
# ---------------------------------------------------------------------------


def test_neuroforce_boosts_supereffective(fresh_battle, step_turn, hp_of):
    # Garchomp (ground/dragon) → Tyranitar (rock/dark): EQ is 4x SE.
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "garchomp",
        "tyranitar",
        "earthquake",
        "neuroforce",
        seed=29,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.15)


# ---------------------------------------------------------------------------
# 19. Analytic — moving last gives 1.3x
# ---------------------------------------------------------------------------


def test_analytic_boosts_when_moving_last(fresh_battle, step_turn, hp_of):
    # Slow attacker (Snorlax 30 spe) vs fast defender (Garchomp 102 spe) — Snorlax
    # moves last every turn → Analytic activates.
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "snorlax",
                ["bodyslam", "tackle", "tackle", "tackle"],
                ability="analytic",
            )
        ],
        [MonSpec("garchomp", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=30,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("snorlax", ["bodyslam", "tackle", "tackle", "tackle"], ability=NO_OP)],
        [MonSpec("garchomp", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=30,
    )
    pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    boosted = pre - hp_of(state_a, 1)
    plain = pre - hp_of(state_b, 1)
    assert boosted > plain
    assert boosted >= int(plain * 1.2)


# ---------------------------------------------------------------------------
# 20. Gorilla Tactics — Atk * 1.5
# ---------------------------------------------------------------------------


def test_gorilla_tactics_boosts_physical(fresh_battle, step_turn, hp_of):
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "darmanitan",
        "snorlax",
        "earthquake",
        "gorillatactics",
        seed=31,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.35)


# ---------------------------------------------------------------------------
# 21. Hustle — Atk * 1.5 (and physical accuracy * 0.8 — not asserted here)
# ---------------------------------------------------------------------------


def test_hustle_boosts_physical(fresh_battle, step_turn, hp_of):
    # Use earthquake (no accuracy issue at 100% base) so Hustle's accuracy hit
    # doesn't drop the move and confound the damage comparison.
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "dragonite",
        "snorlax",
        "earthquake",
        "hustle",
        seed=32,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.35)


# ---------------------------------------------------------------------------
# 22. Pinch abilities — Overgrow / Blaze / Torrent / Swarm
# ---------------------------------------------------------------------------


def _set_low_hp(state, side: int):
    base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    active = int(state.battle_state[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])
    off = base + active * POKEMON_SIZE
    max_hp = int(state.battle_state[off + 2])
    state.battle_state[off + 1] = max(1, max_hp // 4)  # ~25% HP, below 1/3


def test_overgrow_boosts_grass_when_low_hp(fresh_battle, step_turn, hp_of):
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "venusaur",
                ["energyball", "tackle", "tackle", "tackle"],
                ability="overgrow",
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=33,
    )
    state_b, prng_b = fresh_battle(
        [
            MonSpec(
                "venusaur", ["energyball", "tackle", "tackle", "tackle"], ability=NO_OP
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=33,
    )
    _set_low_hp(state_a, 0)
    _set_low_hp(state_b, 0)
    pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    boosted = pre - hp_of(state_a, 1)
    plain = pre - hp_of(state_b, 1)
    assert boosted > plain
    assert boosted >= int(plain * 1.3)


def test_blaze_boosts_fire_when_low_hp(fresh_battle, step_turn, hp_of):
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "blaziken",
                ["flamethrower", "tackle", "tackle", "tackle"],
                ability="blaze",
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=34,
    )
    state_b, prng_b = fresh_battle(
        [
            MonSpec(
                "blaziken",
                ["flamethrower", "tackle", "tackle", "tackle"],
                ability=NO_OP,
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=34,
    )
    _set_low_hp(state_a, 0)
    _set_low_hp(state_b, 0)
    pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    boosted = pre - hp_of(state_a, 1)
    plain = pre - hp_of(state_b, 1)
    assert boosted > plain
    assert boosted >= int(plain * 1.3)


def test_torrent_boosts_water_when_low_hp(fresh_battle, step_turn, hp_of):
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "swampert", ["surf", "tackle", "tackle", "tackle"], ability="torrent"
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=35,
    )
    state_b, prng_b = fresh_battle(
        [MonSpec("swampert", ["surf", "tackle", "tackle", "tackle"], ability=NO_OP)],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=35,
    )
    _set_low_hp(state_a, 0)
    _set_low_hp(state_b, 0)
    pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    boosted = pre - hp_of(state_a, 1)
    plain = pre - hp_of(state_b, 1)
    assert boosted > plain
    assert boosted >= int(plain * 1.3)


def test_swarm_boosts_bug_when_low_hp(fresh_battle, step_turn, hp_of):
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "heracross", ["xscissor", "tackle", "tackle", "tackle"], ability="swarm"
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=36,
    )
    state_b, prng_b = fresh_battle(
        [
            MonSpec(
                "heracross", ["xscissor", "tackle", "tackle", "tackle"], ability=NO_OP
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=36,
    )
    _set_low_hp(state_a, 0)
    _set_low_hp(state_b, 0)
    pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    boosted = pre - hp_of(state_a, 1)
    plain = pre - hp_of(state_b, 1)
    assert boosted > plain
    assert boosted >= int(plain * 1.3)


# ---------------------------------------------------------------------------
# 23. Solar Power — SpA * 1.5 in sun, takes 1/8 max HP EOT
# ---------------------------------------------------------------------------


def test_solar_power_boosts_special_in_sun(fresh_battle, step_turn, hp_of):
    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "venusaur",
                ["energyball", "tackle", "tackle", "tackle"],
                ability="solarpower",
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=37,
    )
    state_b, prng_b = fresh_battle(
        [
            MonSpec(
                "venusaur", ["energyball", "tackle", "tackle", "tackle"], ability=NO_OP
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=37,
    )
    for st in (state_a, state_b):
        st.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_SUN
        st.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    pre = hp_of(state_a, 1)
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    boosted = pre - hp_of(state_a, 1)
    plain = pre - hp_of(state_b, 1)
    assert boosted > plain
    assert boosted >= int(plain * 1.35)


@pytest.mark.xfail(
    strict=False, reason="EOT chip from Solar Power not yet wired in pokepy"
)
def test_solar_power_chip_in_sun(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [
            MonSpec(
                "venusaur",
                ["tackle", "tackle", "tackle", "tackle"],
                ability="solarpower",
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"], ability=NO_OP)],
        seed=38,
    )
    state.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_SUN
    state.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    pre = hp_of(state, 0)
    mx = max_hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    # Solar Power chip = 1/8 max HP at end of turn.
    chip = pre - hp_of(state, 0)
    assert chip >= mx // 8 - 2  # allow for rounding


# ---------------------------------------------------------------------------
# 24. Mold Breaker / Teravolt / Turboblaze — ignore defender ability
# ---------------------------------------------------------------------------


def test_mold_breaker_ignores_levitate(fresh_battle, step_turn, hp_of):
    # Excadrill EQ vs Levitate-blissey: with Mold Breaker the move lands;
    # without it, blissey is immune. Read slot 0 HP directly because the EQ
    # KOs blissey and `hp_of(state, 1)` then reads slot 1 (auto-switched).
    from pokepy.core.constants import OFF_SIDE1

    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "excadrill",
                ["earthquake", "tackle", "tackle", "tackle"],
                ability="moldbreaker",
            )
        ],
        [
            MonSpec(
                "blissey", ["tackle", "tackle", "tackle", "tackle"], ability="levitate"
            )
        ],
        seed=39,
    )
    state_b, prng_b = fresh_battle(
        [
            MonSpec(
                "excadrill", ["earthquake", "tackle", "tackle", "tackle"], ability=NO_OP
            )
        ],
        [
            MonSpec(
                "blissey", ["tackle", "tackle", "tackle", "tackle"], ability="levitate"
            )
        ],
        seed=39,
    )
    pre_a = int(state_a.battle_state[OFF_SIDE1 + 1])
    pre_b = int(state_b.battle_state[OFF_SIDE1 + 1])
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    breaker_dmg = pre_a - int(state_a.battle_state[OFF_SIDE1 + 1])
    plain_dmg = pre_b - int(state_b.battle_state[OFF_SIDE1 + 1])
    assert breaker_dmg > 0
    assert plain_dmg == 0


def test_teravolt_ignores_defender_ability(fresh_battle, step_turn, hp_of):
    # Read slot 0 HP directly to avoid KO/auto-switch confounding hp_of.
    from pokepy.core.constants import OFF_SIDE1

    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "zekrom",
                ["earthquake", "tackle", "tackle", "tackle"],
                ability="teravolt",
            )
        ],
        [
            MonSpec(
                "blissey", ["tackle", "tackle", "tackle", "tackle"], ability="levitate"
            )
        ],
        seed=40,
    )
    state_b, prng_b = fresh_battle(
        [
            MonSpec(
                "zekrom", ["earthquake", "tackle", "tackle", "tackle"], ability=NO_OP
            )
        ],
        [
            MonSpec(
                "blissey", ["tackle", "tackle", "tackle", "tackle"], ability="levitate"
            )
        ],
        seed=40,
    )
    pre_a = int(state_a.battle_state[OFF_SIDE1 + 1])
    pre_b = int(state_b.battle_state[OFF_SIDE1 + 1])
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    assert (pre_a - int(state_a.battle_state[OFF_SIDE1 + 1])) > 0
    assert (pre_b - int(state_b.battle_state[OFF_SIDE1 + 1])) == 0


def test_turboblaze_ignores_defender_ability(fresh_battle, step_turn, hp_of):
    # Read slot 0 HP directly to avoid KO/auto-switch confounding hp_of.
    from pokepy.core.constants import OFF_SIDE1

    state_a, prng_a = fresh_battle(
        [
            MonSpec(
                "reshiram",
                ["earthquake", "tackle", "tackle", "tackle"],
                ability="turboblaze",
            )
        ],
        [
            MonSpec(
                "blissey", ["tackle", "tackle", "tackle", "tackle"], ability="levitate"
            )
        ],
        seed=41,
    )
    state_b, prng_b = fresh_battle(
        [
            MonSpec(
                "reshiram", ["earthquake", "tackle", "tackle", "tackle"], ability=NO_OP
            )
        ],
        [
            MonSpec(
                "blissey", ["tackle", "tackle", "tackle", "tackle"], ability="levitate"
            )
        ],
        seed=41,
    )
    pre_a = int(state_a.battle_state[OFF_SIDE1 + 1])
    pre_b = int(state_b.battle_state[OFF_SIDE1 + 1])
    step_turn(state_a, prng_a, 0, 0)
    step_turn(state_b, prng_b, 0, 0)
    assert (pre_a - int(state_a.battle_state[OFF_SIDE1 + 1])) > 0
    assert (pre_b - int(state_b.battle_state[OFF_SIDE1 + 1])) == 0


# ---------------------------------------------------------------------------
# 25. Sharpness — slicing moves * 1.5
# ---------------------------------------------------------------------------


def test_sharpness_boosts_slicing_move(fresh_battle, step_turn, hp_of):
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "ironvaliant",
        "snorlax",
        "psychocut",
        "sharpness",
        seed=42,
    )
    assert boosted > plain
    assert boosted >= int(plain * 1.35)


# ---------------------------------------------------------------------------
# 26. Skill Link — multi-hit moves always hit max times
# ---------------------------------------------------------------------------


def test_skill_link_maximizes_multihit_damage(fresh_battle, step_turn, hp_of):
    # Icicle Spear hits 2-5 times normally; Skill Link forces 5. Average vs max
    # should produce strictly higher (or in pathological RNG, equal) damage,
    # but with the same seed Skill Link should produce strictly more hits.
    boosted, plain = _run_pair(
        fresh_battle,
        step_turn,
        hp_of,
        "cloyster",
        "snorlax",
        "iciclespear",
        "skilllink",
        seed=43,
    )
    assert boosted >= plain
    # 5 hits / avg ~3 hits = 1.66x
    assert boosted >= int(plain * 1.3)
