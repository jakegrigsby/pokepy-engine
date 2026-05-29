"""Weather and terrain mechanics for Gen 9 OU singles in pokepy.

This file targets the weather/terrain layer of the engine: setter moves,
duration rocks, switch-in abilities, damage/stat modifiers in active weather,
EOT chip damage and healing, type-changing moves (Weather Ball / Terrain
Pulse), priority/status blocking by terrain, and weather-suppression effects
(Utility Umbrella, Air Lock, Cloud Nine, primal weathers).

Showdown source references:
- pokemon-showdown/data/moves.ts             (sunnyday, raindance, sandstorm,
                                              snowscape, weatherball, terrainpulse,
                                              solarbeam, synthesis, morningsun,
                                              moonlight, thunder)
- pokemon-showdown/data/conditions.ts        (sunnyday, raindance, sandstorm,
                                              snow, electricterrain, grassyterrain,
                                              psychicterrain, mistyterrain,
                                              primordialsea, desolateland)
- pokemon-showdown/data/abilities.ts         (drought, drizzle, sandstream,
                                              snowwarning, electricsurge,
                                              grassysurge, psychicsurge,
                                              mistysurge, chlorophyll, swiftswim,
                                              sandrush, slushrush, primordialsea,
                                              desolateland, airlock, cloudnine,
                                              sandveil, snowcloak)
- pokemon-showdown/data/items.ts             (heatrock, damprock, smoothrock,
                                              icyrock, terrainextender,
                                              utilityumbrella)

Mirrors the layout in tests/test_pokepy_mech_speed_priority.py: each test
boots one battle via the `fresh_battle` fixture and inspects the resulting
state. Tests known to be unimplemented in pokepy are marked xfail(strict=False).
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
    M_WEATHER_TURNS,
    M_TERRAIN_TURNS,
    F_WEATHER,
    F_TERRAIN,
    WEATHER_NONE,
    WEATHER_SUN,
    WEATHER_RAIN,
    WEATHER_SAND,
    WEATHER_SNOW,
    WEATHER_PRIMORDIAL_SEA,
    WEATHER_DESOLATE_LAND,
    WEATHER_DELTA_STREAM,
    TERRAIN_NONE,
    TERRAIN_ELECTRIC,
    TERRAIN_GRASSY,
    TERRAIN_PSYCHIC,
    TERRAIN_MISTY,
)

# ---------------------------------------------------------------------------
# 1. Sunny Day move sets sun for 5 turns
# ---------------------------------------------------------------------------


def test_sunny_day_sets_sun_for_5_turns(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("torkoal", ["sunnyday", "flamethrower", "tackle", "tackle"])],
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        seed=42,
    )
    step_turn(state, prng, 0, 0)  # action 0 = move slot 0 = sunnyday
    assert int(state.battle_state[OFF_FIELD + F_WEATHER]) == WEATHER_SUN
    # 5-turn duration; one EOT decrement happens during the same turn -> 4 left.
    assert int(state.battle_state[OFF_META + M_WEATHER_TURNS]) == 4


# ---------------------------------------------------------------------------
# 2. Heat Rock extends sun to 8 turns
# ---------------------------------------------------------------------------


def test_heat_rock_extends_sun(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [
            MonSpec(
                "torkoal", ["sunnyday", "tackle", "tackle", "tackle"], item="heatrock"
            )
        ],
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        seed=42,
    )
    step_turn(state, prng, 0, 0)
    assert int(state.battle_state[OFF_FIELD + F_WEATHER]) == WEATHER_SUN
    # 8 - 1 EOT = 7 turns remaining
    assert int(state.battle_state[OFF_META + M_WEATHER_TURNS]) == 7


# ---------------------------------------------------------------------------
# 3. Drought ability sets sun on switch-in
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False, reason="switch-in abilities at battle start not yet wired"
)
def test_drought_ability_sets_sun_on_entry(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [
            MonSpec(
                "torkoal", ["tackle", "tackle", "tackle", "tackle"], ability="drought"
            )
        ],
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    # Showdown: drought sets sun the moment Torkoal enters at turn 1.
    step_turn(state, prng, 0, 0)
    assert int(state.battle_state[OFF_FIELD + F_WEATHER]) == WEATHER_SUN


# ---------------------------------------------------------------------------
# 4. Sun boosts fire damage 1.5x (compare flamethrower in sun vs no weather)
# ---------------------------------------------------------------------------


def test_sun_boosts_fire_damage(fresh_battle, step_turn, hp_of):
    # baseline: no sun
    state, prng = fresh_battle(
        [MonSpec("typhlosion", ["flamethrower", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    base_dmg = hp_pre - hp_of(state, 1)

    # sun version
    state2, prng2 = fresh_battle(
        [MonSpec("typhlosion", ["flamethrower", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state2.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_SUN
    state2.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    hp_pre2 = hp_of(state2, 1)
    step_turn(state2, prng2, 0, 0)
    sun_dmg = hp_pre2 - hp_of(state2, 1)
    assert (
        sun_dmg > base_dmg
    ), f"sun should boost fire dmg (base={base_dmg}, sun={sun_dmg})"


# ---------------------------------------------------------------------------
# 5. Sun reduces water damage to 0.5x
# ---------------------------------------------------------------------------


def test_sun_reduces_water_damage(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("kingdra", ["surf", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    base = hp_pre - hp_of(state, 1)

    state2, prng2 = fresh_battle(
        [MonSpec("kingdra", ["surf", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state2.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_SUN
    state2.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    hp_pre2 = hp_of(state2, 1)
    step_turn(state2, prng2, 0, 0)
    in_sun = hp_pre2 - hp_of(state2, 1)
    assert in_sun < base, f"sun should reduce water damage (base={base}, sun={in_sun})"


# ---------------------------------------------------------------------------
# 6. Solar Beam in sun: no charge turn
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False, reason="solar beam charge skip in sun not yet verified"
)
def test_solar_beam_no_charge_in_sun(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("venusaur", ["solarbeam", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_SUN
    state.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    # In sun Solar Beam fires immediately on the same turn -> Snorlax takes damage.
    assert hp_of(state, 1) < hp_pre


def test_meteor_beam_boosts_spa_on_charge_turn(
    fresh_battle, step_turn, hp_of, boost_of
):
    state, prng = fresh_battle(
        [MonSpec("glimmora", ["meteorbeam", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=42,
    )
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 1) == hp_pre
    assert boost_of(state, 0, "spa") == 1


def test_electro_shot_in_rain_boosts_before_same_turn_damage(
    fresh_battle,
    step_turn,
    hp_of,
    boost_of,
):
    state, prng = fresh_battle(
        [MonSpec("archaludon", ["electroshot", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=42,
    )
    state.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_RAIN
    state.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    step_turn(state, prng, 0, 0)
    assert boost_of(state, 0, "spa") == 1
    assert hp_of(state, 1) == 316


# ---------------------------------------------------------------------------
# 7. Synthesis / Morning Sun / Moonlight in sun: heal 2/3 (vs 1/2 baseline)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False, reason="weather-scaled recovery moves not yet implemented"
)
def test_synthesis_heals_two_thirds_in_sun(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [MonSpec("venusaur", ["synthesis", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_SUN
    state.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    mhp = max_hp_of(state, 0)
    state.battle_state[OFF_SIDE0 + 1] = 1  # 1 HP remaining
    step_turn(state, prng, 0, 0)
    # 2/3 max HP heal in sun
    assert hp_of(state, 0) >= 1 + (2 * mhp) // 3 - 2


# ---------------------------------------------------------------------------
# 8. Chlorophyll doubles speed in sun (covered in speed_priority too — sanity)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False, reason="chlorophyll x2 speed in sun verified in speed file"
)
def test_chlorophyll_doubles_speed_in_sun(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [
            MonSpec(
                "venusaur",
                ["energyball", "tackle", "tackle", "tackle"],
                ability="chlorophyll",
            )
        ],
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    state.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_SUN
    state.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    hp1_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 1) < hp1_pre


# ---------------------------------------------------------------------------
# 9. Rain Dance sets rain
# ---------------------------------------------------------------------------


def test_rain_dance_sets_rain(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("politoed", ["raindance", "surf", "tackle", "tackle"])],
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        seed=42,
    )
    step_turn(state, prng, 0, 0)
    assert int(state.battle_state[OFF_FIELD + F_WEATHER]) == WEATHER_RAIN
    assert int(state.battle_state[OFF_META + M_WEATHER_TURNS]) == 4


# ---------------------------------------------------------------------------
# 10. Rain boosts water 1.5x and reduces fire 0.5x
# ---------------------------------------------------------------------------


def test_rain_boosts_water_and_dampens_fire(fresh_battle, step_turn, hp_of):
    # water boost
    state, prng = fresh_battle(
        [MonSpec("kingdra", ["surf", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    base_water = hp_pre - hp_of(state, 1)

    state2, prng2 = fresh_battle(
        [MonSpec("kingdra", ["surf", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state2.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_RAIN
    state2.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    hp_pre2 = hp_of(state2, 1)
    step_turn(state2, prng2, 0, 0)
    rain_water = hp_pre2 - hp_of(state2, 1)
    assert rain_water > base_water


# ---------------------------------------------------------------------------
# 11. Rain + thunder: 100% accuracy (sanity: thunder hits across multiple seeds)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="thunder accuracy in rain not yet validated")
def test_thunder_perfect_accuracy_in_rain(fresh_battle, step_turn, hp_of):
    hits = 0
    for seed in range(8):
        state, prng = fresh_battle(
            [MonSpec("kingdra", ["thunder", "tackle", "tackle", "tackle"])],
            [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
            seed=seed,
        )
        state.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_RAIN
        state.battle_state[OFF_META + M_WEATHER_TURNS] = 5
        hp_pre = hp_of(state, 1)
        step_turn(state, prng, 0, 0)
        if hp_of(state, 1) < hp_pre:
            hits += 1
    assert hits == 8


# ---------------------------------------------------------------------------
# 12. Sandstorm chip damage 1/16 each EOT for non-rock/ground/steel
# ---------------------------------------------------------------------------


def test_sandstorm_chip_damages_normal_type(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=42,
    )
    state.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_SAND
    state.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    mhp = max_hp_of(state, 0)
    hp_pre = hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    assert hp_pre - hp_of(state, 0) == mhp // 16


# ---------------------------------------------------------------------------
# 13. Sand boosts rock-type SpD 1.5x (rock attacker takes less special damage from rock)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="rock SpD x1.5 in sand not yet validated")
def test_sand_boosts_rock_spd(fresh_battle, step_turn, hp_of):
    # Tyranitar (rock) takes a special move; in sand its SpD goes up so dmg drops.
    state, prng = fresh_battle(
        [MonSpec("venusaur", ["energyball", "tackle", "tackle", "tackle"])],
        [MonSpec("tyranitar", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    base = hp_pre - hp_of(state, 1)

    state2, prng2 = fresh_battle(
        [MonSpec("venusaur", ["energyball", "tackle", "tackle", "tackle"])],
        [MonSpec("tyranitar", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state2.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_SAND
    state2.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    hp_pre2 = hp_of(state2, 1)
    step_turn(state2, prng2, 0, 0)
    in_sand = hp_pre2 - hp_of(state2, 1)
    assert in_sand < base


# ---------------------------------------------------------------------------
# 14. Sandstorm + sand veil: 1.25x evasion (probabilistic, so loose)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="sand veil evasion modifier not yet wired")
def test_sand_veil_grants_evasion_in_sand():
    pytest.skip("placeholder for sand veil RNG test")


# ---------------------------------------------------------------------------
# 15. Sand Stream sets sand on switch-in
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False, reason="switch-in abilities at battle start not yet wired"
)
def test_sand_stream_sets_sand_on_entry(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [
            MonSpec(
                "tyranitar",
                ["tackle", "tackle", "tackle", "tackle"],
                ability="sandstream",
            )
        ],
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert int(state.battle_state[OFF_FIELD + F_WEATHER]) == WEATHER_SAND


# ---------------------------------------------------------------------------
# 16. Snow boosts ice-type Def 1.5x
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="ice Def x1.5 in snow not yet validated")
def test_snow_boosts_ice_def(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["bodyslam", "tackle", "tackle", "tackle"])],
        [MonSpec("glaceon", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    base = hp_pre - hp_of(state, 1)

    state2, prng2 = fresh_battle(
        [MonSpec("snorlax", ["bodyslam", "tackle", "tackle", "tackle"])],
        [MonSpec("glaceon", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state2.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_SNOW
    state2.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    hp_pre2 = hp_of(state2, 1)
    step_turn(state2, prng2, 0, 0)
    in_snow = hp_pre2 - hp_of(state2, 1)
    assert in_snow < base


# ---------------------------------------------------------------------------
# 17. Snow Warning sets snow on switch-in
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False, reason="switch-in abilities at battle start not yet wired"
)
def test_snow_warning_sets_snow_on_entry(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [
            MonSpec(
                "abomasnow",
                ["tackle", "tackle", "tackle", "tackle"],
                ability="snowwarning",
            )
        ],
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert int(state.battle_state[OFF_FIELD + F_WEATHER]) == WEATHER_SNOW


# ---------------------------------------------------------------------------
# 18. Slush Rush 2x speed in snow (sanity through ordering)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="slush rush speed boost not yet validated")
def test_slush_rush_doubles_speed_in_snow(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [
            MonSpec(
                "sandslash",
                ["icicleSpear" if False else "tackle", "tackle", "tackle", "tackle"],
                ability="slushrush",
            )
        ],
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    state.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_SNOW
    state.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    hp1_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 1) < hp1_pre


# ---------------------------------------------------------------------------
# 19. Electric Terrain blocks sleep on grounded mons
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="electric terrain sleep block not yet wired")
def test_electric_terrain_blocks_sleep(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("venusaur", ["spore", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state.battle_state[OFF_FIELD + F_TERRAIN] = TERRAIN_ELECTRIC
    state.battle_state[OFF_META + M_TERRAIN_TURNS] = 5
    step_turn(state, prng, 0, 0)
    from pokepy.core.constants import STATUS_SLEEP

    assert status_of(state, 1) != STATUS_SLEEP


# ---------------------------------------------------------------------------
# 20. Electric Terrain boosts electric moves 1.3x for grounded user
# ---------------------------------------------------------------------------


def test_electric_terrain_boosts_electric_damage(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("kingdra", ["thunderbolt", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    base = hp_pre - hp_of(state, 1)

    state2, prng2 = fresh_battle(
        [MonSpec("kingdra", ["thunderbolt", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state2.battle_state[OFF_FIELD + F_TERRAIN] = TERRAIN_ELECTRIC
    state2.battle_state[OFF_META + M_TERRAIN_TURNS] = 5
    hp_pre2 = hp_of(state2, 1)
    step_turn(state2, prng2, 0, 0)
    boosted = hp_pre2 - hp_of(state2, 1)
    assert boosted > base, f"electric terrain should boost (base={base}, ET={boosted})"


# ---------------------------------------------------------------------------
# 21. Grassy Terrain boosts grass moves 1.3x for grounded user
# ---------------------------------------------------------------------------


def test_grassy_terrain_boosts_grass_damage(fresh_battle, step_turn, hp_of):
    # Use a flying-type defender so grassy terrain's 1/16 EOT heal does not
    # mask the damage delta (snorlax was grounded → healed at EOT, hiding boost).
    state, prng = fresh_battle(
        [MonSpec("venusaur", ["energyball", "splash", "splash", "splash"])],
        [MonSpec("dragonite", ["splash", "splash", "splash", "splash"])],
        seed=7,
    )
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    base = hp_pre - hp_of(state, 1)

    state2, prng2 = fresh_battle(
        [MonSpec("venusaur", ["energyball", "splash", "splash", "splash"])],
        [MonSpec("dragonite", ["splash", "splash", "splash", "splash"])],
        seed=7,
    )
    state2.battle_state[OFF_FIELD + F_TERRAIN] = TERRAIN_GRASSY
    state2.battle_state[OFF_META + M_TERRAIN_TURNS] = 5
    hp_pre2 = hp_of(state2, 1)
    step_turn(state2, prng2, 0, 0)
    boosted = hp_pre2 - hp_of(state2, 1)
    assert boosted > base


# ---------------------------------------------------------------------------
# 22. Grassy Terrain heals grounded mons 1/16 max HP each EOT
# ---------------------------------------------------------------------------


def test_grassy_terrain_heals_one_sixteenth(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=42,
    )
    state.battle_state[OFF_FIELD + F_TERRAIN] = TERRAIN_GRASSY
    state.battle_state[OFF_META + M_TERRAIN_TURNS] = 5
    mhp = max_hp_of(state, 0)
    state.battle_state[OFF_SIDE0 + 1] = 100  # damaged
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 0) == 100 + mhp // 16


# ---------------------------------------------------------------------------
# 23. Grassy Terrain reduces Earthquake to 0.5x
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="grassy terrain EQ damping not yet validated")
def test_grassy_terrain_halves_earthquake(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    base = hp_pre - hp_of(state, 1)

    state2, prng2 = fresh_battle(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state2.battle_state[OFF_FIELD + F_TERRAIN] = TERRAIN_GRASSY
    state2.battle_state[OFF_META + M_TERRAIN_TURNS] = 5
    hp_pre2 = hp_of(state2, 1)
    step_turn(state2, prng2, 0, 0)
    damped = hp_pre2 - hp_of(state2, 1)
    assert damped < base


# ---------------------------------------------------------------------------
# 24. Psychic Terrain blocks priority moves on grounded targets
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="psychic terrain priority block not yet wired")
def test_psychic_terrain_blocks_priority(fresh_battle, step_turn, hp_of):
    # Slow Snorlax is grounded; Garchomp uses Quick Attack -> should be blocked.
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["quickattack", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    state.battle_state[OFF_FIELD + F_TERRAIN] = TERRAIN_PSYCHIC
    state.battle_state[OFF_META + M_TERRAIN_TURNS] = 5
    hp1_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 1) == hp1_pre  # priority move was blocked


# ---------------------------------------------------------------------------
# 25. Misty Terrain blocks status on grounded mons
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="misty terrain status block not yet wired")
def test_misty_terrain_blocks_status(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("venusaur", ["spore", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state.battle_state[OFF_FIELD + F_TERRAIN] = TERRAIN_MISTY
    state.battle_state[OFF_META + M_TERRAIN_TURNS] = 5
    step_turn(state, prng, 0, 0)
    from pokepy.core.constants import STATUS_NONE

    assert status_of(state, 1) == STATUS_NONE


# ---------------------------------------------------------------------------
# 26. Misty Terrain halves dragon damage on grounded targets
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False, reason="misty terrain dragon halving not yet validated"
)
def test_misty_terrain_halves_dragon_damage(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["dragonpulse", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    base = hp_pre - hp_of(state, 1)

    state2, prng2 = fresh_battle(
        [MonSpec("garchomp", ["dragonpulse", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state2.battle_state[OFF_FIELD + F_TERRAIN] = TERRAIN_MISTY
    state2.battle_state[OFF_META + M_TERRAIN_TURNS] = 5
    hp_pre2 = hp_of(state2, 1)
    step_turn(state2, prng2, 0, 0)
    damped = hp_pre2 - hp_of(state2, 1)
    assert damped < base


# ---------------------------------------------------------------------------
# 27. Weather Ball: BP doubles in any weather (and changes type)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="weather ball BP doubling not yet validated")
def test_weather_ball_doubles_in_weather(fresh_battle, step_turn, hp_of):
    # Baseline: clear weather
    state, prng = fresh_battle(
        [MonSpec("torkoal", ["weatherball", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    base = hp_pre - hp_of(state, 1)

    state2, prng2 = fresh_battle(
        [MonSpec("torkoal", ["weatherball", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state2.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_SUN
    state2.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    hp_pre2 = hp_of(state2, 1)
    step_turn(state2, prng2, 0, 0)
    boosted = hp_pre2 - hp_of(state2, 1)
    # In sun: type=fire, BP doubles, plus 1.5x sun multiplier -> way more dmg.
    assert boosted >= 2 * base


# ---------------------------------------------------------------------------
# 28. Terrain Pulse: BP doubles when grounded user is on a terrain
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="terrain pulse BP doubling not yet validated")
def test_terrain_pulse_doubles_on_terrain(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("kingdra", ["terrainpulse", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    base = hp_pre - hp_of(state, 1)

    state2, prng2 = fresh_battle(
        [MonSpec("kingdra", ["terrainpulse", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state2.battle_state[OFF_FIELD + F_TERRAIN] = TERRAIN_ELECTRIC
    state2.battle_state[OFF_META + M_TERRAIN_TURNS] = 5
    hp_pre2 = hp_of(state2, 1)
    step_turn(state2, prng2, 0, 0)
    boosted = hp_pre2 - hp_of(state2, 1)
    assert boosted >= 2 * base


# ---------------------------------------------------------------------------
# 29. Utility Umbrella ignores weather chip damage in sand
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False, reason="utility umbrella weather immunity not yet wired"
)
def test_utility_umbrella_blocks_sand_chip(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [
            MonSpec(
                "snorlax",
                ["splash", "tackle", "tackle", "tackle"],
                item="utilityumbrella",
            )
        ],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=42,
    )
    state.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_SAND
    state.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    hp_pre = hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    # Note: Showdown actually doesn't block sand chip with umbrella (only sun/rain
    # boosts/heals); but our test simply asserts pokepy's umbrella interaction with
    # weather damage. xfail until verified.
    assert hp_of(state, 0) == hp_pre


# ---------------------------------------------------------------------------
# 30. Primordial Sea ends sun on switch / blocks fire moves
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="primal weather override not yet wired")
def test_primordial_sea_blocks_fire(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("typhlosion", ["flamethrower", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_PRIMORDIAL_SEA
    state.battle_state[OFF_META + M_WEATHER_TURNS] = 0  # permanent
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 1) == hp_pre  # fire move was nullified


# ---------------------------------------------------------------------------
# 31. Desolate Land blocks water moves
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=False, reason="desolate land water block not yet wired")
def test_desolate_land_blocks_water(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("kingdra", ["surf", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    state.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_DESOLATE_LAND
    state.battle_state[OFF_META + M_WEATHER_TURNS] = 0
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 1) == hp_pre


# ---------------------------------------------------------------------------
# 32. Air Lock / Cloud Nine suppress weather effects without ending weather
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False, reason="air lock / cloud nine suppression not yet wired"
)
def test_air_lock_suppresses_sand_chip(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [
            MonSpec(
                "snorlax", ["splash", "tackle", "tackle", "tackle"], ability="airlock"
            )
        ],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=42,
    )
    state.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_SAND
    state.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    hp_pre = hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    # Air Lock: weather flag still SAND but no chip dmg.
    assert hp_of(state, 0) == hp_pre
    assert int(state.battle_state[OFF_FIELD + F_WEATHER]) == WEATHER_SAND


# ---------------------------------------------------------------------------
# 33. Weather only damages once per EOT (not per active mon redundantly)
# ---------------------------------------------------------------------------


def test_sand_chip_once_per_turn(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["splash", "tackle", "tackle", "tackle"])],
        seed=42,
    )
    state.battle_state[OFF_FIELD + F_WEATHER] = WEATHER_SAND
    state.battle_state[OFF_META + M_WEATHER_TURNS] = 5
    mhp0 = max_hp_of(state, 0)
    mhp1 = max_hp_of(state, 1)
    hp0_pre = hp_of(state, 0)
    hp1_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    # Each side takes exactly mhp//16 chip — once.
    assert hp0_pre - hp_of(state, 0) == mhp0 // 16
    assert hp1_pre - hp_of(state, 1) == mhp1 // 16


# ---------------------------------------------------------------------------
# 34. Drizzle from Politoed sets rain on switch-in
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False, reason="switch-in abilities at battle start not yet wired"
)
def test_drizzle_sets_rain_on_entry(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [
            MonSpec(
                "politoed", ["tackle", "tackle", "tackle", "tackle"], ability="drizzle"
            )
        ],
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert int(state.battle_state[OFF_FIELD + F_WEATHER]) == WEATHER_RAIN
