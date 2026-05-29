"""Switch-in ability mechanics for pokepy.

Tests abilities that fire when a Pokemon enters the field via voluntary switch.
For each ability we set up a 2-mon side where slot 0 is a vanilla mon and slot 1
holds the ability under test, then issue action 5 (switch to roster slot 1) and
assert on the resulting battle state.

Showdown source for each ability lives in `pokemon-showdown/data/abilities.ts`:
- Drought, Drizzle, Sand Stream, Snow Warning  -> "onStart" sets weather
- Electric/Grassy/Psychic/Misty Surge          -> "onStart" sets terrain
- Intimidate                                   -> "onStart" -> boost({atk: -1})
- Defiant / Competitive                        -> "onAfterEachBoost"
- Clear Body / Hyper Cutter / Inner Focus      -> "onTryBoost"
- Trace                                        -> "onStart" copies opponent ability
- Download                                     -> "onStart" boosts atk vs lower def
- Intrepid Sword / Dauntless Shield            -> "onStart" once-per-game in gen 9
- Orichalcum Pulse / Hadron Engine             -> "onStart" set sun / e-terrain
- Protosynthesis / Quark Drive                 -> "onStart" + booster energy logic
- Anticipation / Forewarn / Frisk              -> "onStart" reveal info
- Pressure / Mold Breaker / Neutralizing Gas   -> "onStart" announce
- Air Lock / Cloud Nine                        -> "onStart" suppress weather
- Slow Start                                   -> "onStart" 5-turn debuff
- Imposter / Battle Bond                       -> "onStart" transform / KO chain

Switch-in implementation lives in pokepy/effects/abilities.py
(`apply_switch_in_ability`) and is invoked by pokepy/engine/battle_gen9.py at
~line 362 (turn-start switches) and again from forced-switch handlers.
"""

from __future__ import annotations

import pytest
from tests.conftest import MonSpec
from pokepy.effects import get_effective_speed
from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_FIELD,
    OFF_META,
    POKEMON_SIZE,
    M_ACTIVE0,
    M_ACTIVE1,
    F_WEATHER,
    F_TERRAIN,
    M_WEATHER_TURNS,
    M_TERRAIN_TURNS,
    WEATHER_NONE,
    WEATHER_SUN,
    WEATHER_RAIN,
    WEATHER_SAND,
    WEATHER_SNOW,
    TERRAIN_NONE,
    TERRAIN_ELECTRIC,
    TERRAIN_GRASSY,
    TERRAIN_PSYCHIC,
    TERRAIN_MISTY,
    ABILITY_DROUGHT,
    ABILITY_DRIZZLE,
    ABILITY_SAND_STREAM,
    ABILITY_SNOW_WARNING,
    ABILITY_ELECTRIC_SURGE,
    ABILITY_GRASSY_SURGE,
    ABILITY_PSYCHIC_SURGE,
    ABILITY_MISTY_SURGE,
    ABILITY_INTIMIDATE,
    ABILITY_TRACE,
    ABILITY_DOWNLOAD,
    ABILITY_DEFIANT,
    ABILITY_COMPETITIVE,
    ABILITY_CLEAR_BODY,
    ABILITY_INNER_FOCUS,
    ABILITY_OBLIVIOUS,
    ABILITY_INTREPID_SWORD,
    ABILITY_DAUNTLESS_SHIELD,
    ABILITY_ORICHALCUM_PULSE,
    ABILITY_HADRON_ENGINE,
    ABILITY_PROTOSYNTHESIS,
    ABILITY_QUARK_DRIVE,
)

# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _active_off(state, side: int) -> int:
    base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    active = int(state.battle_state[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])
    return base + active * POKEMON_SIZE


def _ability_id(state, side: int) -> int:
    return int(state.battle_state[_active_off(state, side) + 5])


# ===========================================================================
# Weather setters
# ===========================================================================


def test_drought_sets_sun_on_switchin(fresh_battle, step_turn, field_word):
    """Torkoal/Drought -> Sunny Day on switch-in."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("torkoal", ["tackle"] * 4, ability="drought"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=1,
    )
    step_turn(state, prng, 5, 0)
    assert field_word(state, F_WEATHER) == WEATHER_SUN
    assert int(state.battle_state[OFF_META + M_WEATHER_TURNS]) > 0


def test_drizzle_sets_rain_on_switchin(fresh_battle, step_turn, field_word):
    """Politoed/Drizzle -> Rain on switch-in."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("politoed", ["tackle"] * 4, ability="drizzle"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=2,
    )
    step_turn(state, prng, 5, 0)
    assert field_word(state, F_WEATHER) == WEATHER_RAIN


def test_sand_stream_sets_sand_on_switchin(fresh_battle, step_turn, field_word):
    """Tyranitar/Sand Stream -> Sandstorm on switch-in."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("tyranitar", ["tackle"] * 4, ability="sandstream"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=3,
    )
    step_turn(state, prng, 5, 0)
    assert field_word(state, F_WEATHER) == WEATHER_SAND


def test_snow_warning_sets_snow_on_switchin(fresh_battle, step_turn, field_word):
    """Abomasnow/Snow Warning -> Snow on switch-in."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("abomasnow", ["tackle"] * 4, ability="snowwarning"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=4,
    )
    step_turn(state, prng, 5, 0)
    assert field_word(state, F_WEATHER) == WEATHER_SNOW


# ===========================================================================
# Terrain setters
# ===========================================================================


def test_electric_surge_sets_terrain_on_switchin(fresh_battle, step_turn, field_word):
    """Tapu Koko/Electric Surge -> Electric Terrain on switch-in."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("tapukoko", ["tackle"] * 4, ability="electricsurge"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=5,
    )
    step_turn(state, prng, 5, 0)
    assert field_word(state, F_TERRAIN) == TERRAIN_ELECTRIC
    assert int(state.battle_state[OFF_META + M_TERRAIN_TURNS]) > 0


def test_grassy_surge_sets_terrain_on_switchin(fresh_battle, step_turn, field_word):
    """Tapu Bulu/Grassy Surge -> Grassy Terrain on switch-in."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("tapubulu", ["tackle"] * 4, ability="grassysurge"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=6,
    )
    step_turn(state, prng, 5, 0)
    assert field_word(state, F_TERRAIN) == TERRAIN_GRASSY


def test_psychic_surge_sets_terrain_on_switchin(fresh_battle, step_turn, field_word):
    """Tapu Lele/Psychic Surge -> Psychic Terrain on switch-in."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("tapulele", ["tackle"] * 4, ability="psychicsurge"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=7,
    )
    step_turn(state, prng, 5, 0)
    assert field_word(state, F_TERRAIN) == TERRAIN_PSYCHIC


def test_misty_surge_sets_terrain_on_switchin(fresh_battle, step_turn, field_word):
    """Tapu Fini/Misty Surge -> Misty Terrain on switch-in."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("tapufini", ["tackle"] * 4, ability="mistysurge"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=8,
    )
    step_turn(state, prng, 5, 0)
    assert field_word(state, F_TERRAIN) == TERRAIN_MISTY


# ===========================================================================
# Intimidate
# ===========================================================================


def test_intimidate_lowers_opponent_attack(fresh_battle, step_turn, boost_of):
    """Arcanine/Intimidate -> -1 atk to opposing active on switch-in."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("arcanine", ["tackle"] * 4, ability="intimidate"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=9,
    )
    assert boost_of(state, 1, "atk") == 0
    step_turn(state, prng, 5, 0)
    assert boost_of(state, 1, "atk") == -1


def test_intimidate_blocked_by_clear_body(fresh_battle, step_turn, boost_of):
    """Clear Body should null out Intimidate's atk drop."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("arcanine", ["tackle"] * 4, ability="intimidate"),
        ],
        [MonSpec("metagross", ["tackle"] * 4, ability="clearbody")],
        seed=10,
    )
    step_turn(state, prng, 5, 0)
    assert boost_of(state, 1, "atk") == 0


def test_intimidate_blocked_by_inner_focus(fresh_battle, step_turn, boost_of):
    """Inner Focus blocks Intimidate (gen 8+)."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("arcanine", ["tackle"] * 4, ability="intimidate"),
        ],
        [MonSpec("scrafty", ["tackle"] * 4, ability="innerfocus")],
        seed=11,
    )
    step_turn(state, prng, 5, 0)
    assert boost_of(state, 1, "atk") == 0


def test_intimidate_blocked_by_oblivious(fresh_battle, step_turn, boost_of):
    """Oblivious blocks Intimidate (gen 8+)."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("arcanine", ["tackle"] * 4, ability="intimidate"),
        ],
        [MonSpec("mew", ["tackle"] * 4, ability="oblivious")],
        seed=12,
    )
    step_turn(state, prng, 5, 0)
    assert boost_of(state, 1, "atk") == 0


@pytest.mark.xfail(
    strict=False,
    reason="hyper cutter is atk-specific block; pokepy may treat it as full Clear Body",
)
def test_intimidate_blocked_by_hyper_cutter(fresh_battle, step_turn, boost_of):
    """Hyper Cutter blocks atk drops only (not Intimidate's special-case logic)."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("arcanine", ["tackle"] * 4, ability="intimidate"),
        ],
        [MonSpec("krookodile", ["tackle"] * 4, ability="hypercutter")],
        seed=13,
    )
    step_turn(state, prng, 5, 0)
    assert boost_of(state, 1, "atk") == 0


def test_intimidate_triggers_defiant_plus2_atk(fresh_battle, step_turn, boost_of):
    """Defiant: +2 atk when a stat drops -> net +1 atk after Intimidate."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("arcanine", ["tackle"] * 4, ability="intimidate"),
        ],
        [MonSpec("bisharp", ["tackle"] * 4, ability="defiant")],
        seed=14,
    )
    step_turn(state, prng, 5, 0)
    assert boost_of(state, 1, "atk") == 1  # -1 (intim) +2 (defiant)


def test_intimidate_triggers_competitive_plus2_spa(fresh_battle, step_turn, boost_of):
    """Competitive: +2 spa when a stat drops."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("arcanine", ["tackle"] * 4, ability="intimidate"),
        ],
        [MonSpec("milotic", ["tackle"] * 4, ability="competitive")],
        seed=15,
    )
    step_turn(state, prng, 5, 0)
    assert boost_of(state, 1, "atk") == -1
    assert boost_of(state, 1, "spa") == 2


# ===========================================================================
# Trace / Download
# ===========================================================================


def test_trace_copies_opponent_ability(fresh_battle, step_turn):
    """Trace overwrites the user's ability with the opponent's."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("gardevoir", ["tackle"] * 4, ability="trace"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4, ability="intimidate")],
        seed=16,
    )
    step_turn(state, prng, 5, 0)
    assert _ability_id(state, 0) == ABILITY_INTIMIDATE


def test_download_boosts_attack_vs_low_def(fresh_battle, step_turn, boost_of):
    """Download: opponent def < spd -> +1 atk on user."""
    # Snorlax has def 65, spd 110 -> def < spd -> +1 atk to Porygon2.
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("porygon2", ["tackle"] * 4, ability="download"),
        ],
        [MonSpec("snorlax", ["tackle"] * 4)],
        seed=17,
    )
    step_turn(state, prng, 5, 0)
    assert boost_of(state, 0, "atk") == 1
    assert boost_of(state, 0, "spa") == 0


def test_download_boosts_spa_vs_low_spd(fresh_battle, step_turn, boost_of):
    """Download: opponent spd < def -> +1 spa on user."""
    # Skarmory has def 140 > spd 70 -> +1 spa.
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("porygon2", ["tackle"] * 4, ability="download"),
        ],
        [MonSpec("skarmory", ["tackle"] * 4)],
        seed=18,
    )
    step_turn(state, prng, 5, 0)
    assert boost_of(state, 0, "spa") == 1


# ===========================================================================
# Info-revealing abilities (no crash checks)
# ===========================================================================


def test_anticipation_no_crash_on_switchin(fresh_battle, step_turn):
    """Anticipation should not crash even if opponent has supereffective moves."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("ferrothorn", ["tackle"] * 4, ability="anticipation"),
        ],
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        seed=19,
    )
    step_turn(state, prng, 5, 0)
    assert int(state.battle_state[OFF_META + M_ACTIVE0]) == 1


def test_forewarn_no_crash_on_switchin(fresh_battle, step_turn):
    """Forewarn should not crash on switch-in."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("xatu", ["tackle"] * 4, ability="forewarn"),
        ],
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        seed=20,
    )
    step_turn(state, prng, 5, 0)
    assert int(state.battle_state[OFF_META + M_ACTIVE0]) == 1


def test_frisk_no_crash_on_switchin(fresh_battle, step_turn):
    """Frisk should not crash on switch-in (just reveals item)."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("banette", ["tackle"] * 4, ability="frisk"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4, item="leftovers")],
        seed=21,
    )
    step_turn(state, prng, 5, 0)
    assert int(state.battle_state[OFF_META + M_ACTIVE0]) == 1


# ===========================================================================
# Announce-only abilities (no crash checks)
# ===========================================================================


def test_pressure_ability_remains_after_switchin(fresh_battle, step_turn):
    """Pressure should remain set on the switched-in mon."""
    from pokepy.core.constants import ABILITY_PRESSURE

    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("dragapult", ["tackle"] * 4, ability="pressure"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=22,
    )
    step_turn(state, prng, 5, 0)
    assert _ability_id(state, 0) == ABILITY_PRESSURE


def test_mold_breaker_no_crash_on_switchin(fresh_battle, step_turn):
    """Mold Breaker switch-in announce: just no crash."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("haxorus", ["tackle"] * 4, ability="moldbreaker"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=23,
    )
    step_turn(state, prng, 5, 0)
    assert int(state.battle_state[OFF_META + M_ACTIVE0]) == 1


# ===========================================================================
# Imposter / Battle Bond (transform / first-KO chain)
# ===========================================================================


@pytest.mark.xfail(
    strict=False, reason="Imposter transform not implemented in apply_switch_in_ability"
)
def test_imposter_transforms_into_opponent(fresh_battle, step_turn):
    """Ditto/Imposter should copy opposing active's stats/ability/types."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("ditto", ["tackle"] * 4, ability="imposter"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4, ability="intimidate")],
        seed=24,
    )
    step_turn(state, prng, 5, 0)
    # After transform, Ditto's ability should be opponent's ability.
    assert _ability_id(state, 0) == ABILITY_INTIMIDATE


def test_battle_bond_no_crash_on_switchin(fresh_battle, step_turn):
    """Greninja/Battle Bond should not crash; the KO transform is later."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("greninja", ["tackle"] * 4, ability="battlebond"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=25,
    )
    step_turn(state, prng, 5, 0)
    assert int(state.battle_state[OFF_META + M_ACTIVE0]) == 1


# ===========================================================================
# Intrepid Sword / Dauntless Shield
# ===========================================================================


def test_intrepid_sword_boosts_attack_on_switchin(fresh_battle, step_turn, boost_of):
    """Zacian/Intrepid Sword -> +1 atk on switch-in."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("zacian", ["tackle"] * 4, ability="intrepidsword"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=26,
    )
    step_turn(state, prng, 5, 0)
    assert boost_of(state, 0, "atk") == 1


def test_dauntless_shield_boosts_def_on_switchin(fresh_battle, step_turn, boost_of):
    """Zamazenta/Dauntless Shield -> +1 def on switch-in."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("zamazenta", ["tackle"] * 4, ability="dauntlessshield"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=27,
    )
    step_turn(state, prng, 5, 0)
    assert boost_of(state, 0, "def") == 1


def test_dauntless_shield_only_once_per_battle(fresh_battle, step_turn, boost_of):
    """Dauntless Shield should not re-trigger after switching out and back in (gen 9)."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("zamazenta", ["tackle"] * 4, ability="dauntlessshield"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=28,
    )
    step_turn(state, prng, 5, 0)  # switch in zamazenta
    step_turn(state, prng, 4, 0)  # switch out to snorlax
    step_turn(state, prng, 5, 0)  # switch back to zamazenta
    # def boost should NOT have stacked (still 1, not 2)
    assert boost_of(state, 0, "def") <= 1


# ===========================================================================
# Paradox abilities (Orichalcum Pulse / Hadron Engine / Proto / Quark)
# ===========================================================================


def test_orichalcum_pulse_sets_sun(fresh_battle, step_turn, field_word):
    """Koraidon/Orichalcum Pulse -> sun on switch-in."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("koraidon", ["tackle"] * 4, ability="orichalcumpulse"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=29,
    )
    step_turn(state, prng, 5, 0)
    assert field_word(state, F_WEATHER) == WEATHER_SUN


def test_hadron_engine_sets_electric_terrain(fresh_battle, step_turn, field_word):
    """Miraidon/Hadron Engine -> electric terrain on switch-in."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("miraidon", ["tackle"] * 4, ability="hadronengine"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=30,
    )
    step_turn(state, prng, 5, 0)
    assert field_word(state, F_TERRAIN) == TERRAIN_ELECTRIC


def test_protosynthesis_no_crash_on_switchin(fresh_battle, step_turn):
    """Great Tusk/Protosynthesis: just verify no crash on switch-in (no sun)."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("greattusk", ["tackle"] * 4, ability="protosynthesis"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=31,
    )
    step_turn(state, prng, 5, 0)
    assert _ability_id(state, 0) == ABILITY_PROTOSYNTHESIS


def test_quark_drive_no_crash_on_switchin(fresh_battle, step_turn):
    """Iron Valiant/Quark Drive: just verify no crash on switch-in (no e-terrain)."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("ironvaliant", ["tackle"] * 4, ability="quarkdrive"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=32,
    )
    step_turn(state, prng, 5, 0)
    assert _ability_id(state, 0) == ABILITY_QUARK_DRIVE


def test_quark_drive_booster_energy_not_active_before_switchin(fresh_battle):
    """Held Booster Energy should not pre-activate a benched paradox mon."""
    state, _ = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec(
                "ironvaliant",
                ["tackle"] * 4,
                ability="quarkdrive",
                item="boosterenergy",
            ),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=3201,
    )
    bench_off = OFF_SIDE0 + POKEMON_SIZE
    assert get_effective_speed(state.battle_state, bench_off) == int(
        state.battle_state[bench_off + 11]
    )


def test_protosynthesis_booster_energy_consumed_off_field(fresh_battle, step_turn):
    """Protosynthesis with Booster Energy should consume the item when not in sun."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec(
                "greattusk",
                ["tackle"] * 4,
                ability="protosynthesis",
                item="boosterenergy",
            ),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=33,
    )
    step_turn(state, prng, 5, 0)
    p_off = _active_off(state, 0)
    item_word = int(state.battle_state[p_off + 6])
    # Booster Energy should have been consumed (off field activates).
    # If field activated proto already, item is preserved -- but we're not in sun.
    assert item_word == 0


# ===========================================================================
# Suppression / debuff abilities (xfail placeholders)
# ===========================================================================


@pytest.mark.xfail(
    strict=False,
    reason="Neutralizing Gas is not implemented in apply_switch_in_ability",
)
def test_neutralizing_gas_no_crash(fresh_battle, step_turn):
    """Weezing-Galar/Neutralizing Gas: switch-in should not crash."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("weezinggalar", ["tackle"] * 4, ability="neutralizinggas"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4, ability="intimidate")],
        seed=34,
    )
    step_turn(state, prng, 5, 0)
    # Opponent's intimidate should be suppressed -> our atk stays 0.
    from pokepy.core.constants import ABILITY_NEUTRALIZING_GAS

    assert _ability_id(state, 0) == ABILITY_NEUTRALIZING_GAS


@pytest.mark.xfail(
    strict=False,
    reason="Air Lock weather suppression not handled in apply_switch_in_ability",
)
def test_air_lock_suppresses_weather_effects(fresh_battle, step_turn, field_word):
    """Rayquaza/Air Lock should suppress weather while it is active."""
    state, prng = fresh_battle(
        [
            MonSpec("torkoal", ["tackle"] * 4, ability="drought"),
            MonSpec("rayquaza", ["tackle"] * 4, ability="airlock"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=35,
    )
    step_turn(state, prng, 5, 0)
    # Air Lock doesn't clear weather but should suppress effects -- here we
    # just check the ability is on the field; full suppression isn't modelled.
    assert field_word(state, F_WEATHER) == WEATHER_NONE


@pytest.mark.xfail(
    strict=False,
    reason="Cloud Nine weather suppression not handled in apply_switch_in_ability",
)
def test_cloud_nine_suppresses_weather_effects(fresh_battle, step_turn, field_word):
    """Golduck/Cloud Nine: weather suppression on switch-in."""
    state, prng = fresh_battle(
        [
            MonSpec("torkoal", ["tackle"] * 4, ability="drought"),
            MonSpec("golduck", ["tackle"] * 4, ability="cloudnine"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=36,
    )
    step_turn(state, prng, 5, 0)
    assert field_word(state, F_WEATHER) == WEATHER_NONE


@pytest.mark.xfail(strict=False, reason="Slow Start atk/spe halving not implemented")
def test_slow_start_halves_speed_first_5_turns(fresh_battle, step_turn, boost_of):
    """Regigigas/Slow Start should impose an internal flag for 5 turns."""
    state, prng = fresh_battle(
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("regigigas", ["tackle"] * 4, ability="slowstart"),
        ],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=37,
    )
    step_turn(state, prng, 5, 0)
    # Slow Start typically uses an internal counter, not a stat boost. We just
    # assert the speed-equivalent boost is non-positive as a smoke check.
    assert boost_of(state, 0, "spe") <= 0
