"""Smoke tests for the ported ability effect bodies (group C).

Verifies the real ports of:
- pokepy.effects.abilities.apply_speed_boost
- pokepy.effects.abilities.apply_shed_skin_hydration
- pokepy.effects.abilities.apply_switch_in_ability  (Intimidate, Drought, etc.)
- pokepy.effects.abilities.apply_regenerator_on_switch_out
- pokepy.effects.abilities.apply_natural_cure_on_switch_out
- pokepy.effects.abilities.apply_absorb_ability_healing  (Volt/Water Absorb,
  Sap Sipper, Storm Drain, Lightning Rod, Motor Drive, Flash Fire)
- pokepy.effects.abilities.apply_weakness_policy
- pokepy.effects.abilities.apply_ko_boost_ability  (Moxie / Beast Boost)
- pokepy.effects.abilities.apply_contact_status_ability  (Static / Flame Body /
  Poison Point / Effect Spore)
"""

from __future__ import annotations

import numpy as np
import pytest

from pokepy.core.state import MultiFormatState
from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_FIELD,
    OFF_META,
    OFF_MOVES,
    POKEMON_SIZE,
    M_ACTIVE0,
    M_ACTIVE1,
    M_ACTIVE_MOVE_ACTIONS_0,
    M_WEATHER_TURNS,
    M_TERRAIN_TURNS,
    F_WEATHER,
    F_TERRAIN,
    F_SUBSTITUTE_0,
    F_SUBSTITUTE_1,
    NEUTRAL_BOOSTS_13,
    NEUTRAL_BOOSTS_14,
    STATUS_NONE,
    STATUS_BURN,
    STATUS_PARALYSIS,
    STATUS_POISON,
    STATUS_SLEEP,
    WEATHER_NONE,
    WEATHER_SUN,
    WEATHER_RAIN,
    WEATHER_SAND,
    TERRAIN_NONE,
    TERRAIN_ELECTRIC,
    TYPE_NORMAL,
    TYPE_FIRE,
    TYPE_WATER,
    TYPE_GRASS,
    TYPE_ELECTRIC,
    TYPE_ICE,
    ABILITY_INTIMIDATE,
    ABILITY_DROUGHT,
    ABILITY_DRIZZLE,
    ABILITY_REGENERATOR,
    ABILITY_NATURAL_CURE,
    ABILITY_VOLT_ABSORB,
    ABILITY_WATER_ABSORB,
    ABILITY_SAP_SIPPER,
    ABILITY_LIGHTNING_ROD,
    ABILITY_MOTOR_DRIVE,
    ABILITY_FLASH_FIRE,
    ABILITY_MOXIE,
    ABILITY_BEAST_BOOST,
    ABILITY_SOUL_HEART,
    ABILITY_SPEED_BOOST,
    ABILITY_SHED_SKIN,
    ABILITY_HYDRATION,
    ABILITY_FLAME_BODY,
    ABILITY_STATIC,
    ABILITY_DEFIANT,
    ABILITY_CONTRARY,
    ABILITY_CLEAR_BODY,
    ABILITY_DOWNLOAD,
    ABILITY_INTREPID_SWORD,
    ABILITY_DAUNTLESS_SHIELD,
    ABILITY_ELECTRIC_SURGE,
    ITEM_WEAKNESS_POLICY,
)
from pokepy.core.bitpack import (
    extract_boost,
    get_status,
    set_status,
    apply_boost_to_packed,
)
from pokepy.data.loader import load_game_data
from pokepy.utils.gen5_prng import Gen5PRNG

from pokepy.effects.abilities import (
    apply_speed_boost,
    apply_shed_skin_hydration,
    apply_switch_in_ability,
    apply_regenerator_on_switch_out,
    apply_natural_cure_on_switch_out,
    apply_absorb_ability_healing,
    apply_weakness_policy,
    apply_ko_boost_ability,
    apply_contact_status_ability,
)


def _hand_state(
    side0_ability: int = 0,
    side1_ability: int = 0,
    side0_type: int = TYPE_NORMAL,
    side1_type: int = TYPE_NORMAL,
    side0_hp: int = 100,
    side1_hp: int = 100,
    side0_max: int = 100,
    side1_max: int = 100,
    side0_item: int = 0,
    side1_item: int = 0,
    side0_status: int = STATUS_NONE,
    side1_status: int = STATUS_NONE,
):
    """Build a minimal battle buffer with one active per side."""
    state = MultiFormatState.create_empty(format_id=1)
    bs = state.battle_state
    bs[OFF_META + M_ACTIVE0] = 0
    bs[OFF_META + M_ACTIVE1] = 0
    bs[OFF_FIELD + F_WEATHER] = WEATHER_NONE
    bs[OFF_FIELD + F_TERRAIN] = TERRAIN_NONE

    configs = [
        (
            OFF_SIDE0,
            side0_ability,
            side0_type,
            side0_hp,
            side0_max,
            side0_item,
            side0_status,
        ),
        (
            OFF_SIDE1,
            side1_ability,
            side1_type,
            side1_hp,
            side1_max,
            side1_item,
            side1_status,
        ),
    ]
    for base, ab, ty, hp, mx, item, st in configs:
        poff = base + 0 * POKEMON_SIZE
        bs[poff + 0] = 1  # species (any nonzero)
        bs[poff + 1] = hp
        bs[poff + 2] = mx
        bs[poff + 3] = 100  # level
        bs[poff + 4] = (ty & 0xFF) | ((ty & 0xFF) << 8)
        bs[poff + 5] = ab
        bs[poff + 6] = item
        bs[poff + 7] = 80  # base atk
        bs[poff + 8] = 80  # base def
        bs[poff + 9] = 80  # base spa
        bs[poff + 10] = 80  # base spd
        bs[poff + 11] = 80  # base spe
        bs[poff + 12] = set_status(st, 0)
        bs[poff + 13] = NEUTRAL_BOOSTS_13
        bs[poff + 14] = NEUTRAL_BOOSTS_14
        bs[poff + 15] = 0
    return state


def _p0_off() -> int:
    return OFF_SIDE0 + 0 * POKEMON_SIZE


def _p1_off() -> int:
    return OFF_SIDE1 + 0 * POKEMON_SIZE


# ============================================================================
# Speed Boost
# ============================================================================


def test_speed_boost_grants_plus_one_speed():
    state = _hand_state(side0_ability=ABILITY_SPEED_BOOST)
    bs = state.battle_state
    poff = _p0_off()
    bs[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0] = 1
    assert extract_boost(int(bs[poff + 14]), 0) == 0
    apply_speed_boost(bs, poff, game_data=None)
    assert extract_boost(int(bs[poff + 14]), 0) == 1


def test_speed_boost_skips_when_no_ability():
    state = _hand_state(side0_ability=0)
    bs = state.battle_state
    poff = _p0_off()
    apply_speed_boost(bs, poff, game_data=None)
    assert extract_boost(int(bs[poff + 14]), 0) == 0


def test_speed_boost_skips_when_fainted():
    state = _hand_state(side0_ability=ABILITY_SPEED_BOOST, side0_hp=0)
    bs = state.battle_state
    poff = _p0_off()
    apply_speed_boost(bs, poff, game_data=None)
    assert extract_boost(int(bs[poff + 14]), 0) == 0


# ============================================================================
# Shed Skin / Hydration
# ============================================================================


def test_hydration_cures_in_rain():
    state = _hand_state(side0_ability=ABILITY_HYDRATION, side0_status=STATUS_BURN)
    bs = state.battle_state
    bs[OFF_FIELD + F_WEATHER] = WEATHER_RAIN
    poff = _p0_off()
    assert get_status(int(bs[poff + 12])) == STATUS_BURN
    apply_shed_skin_hydration(bs, poff, game_data=None, gen5_prng=Gen5PRNG())
    assert get_status(int(bs[poff + 12])) == STATUS_NONE


def test_hydration_no_rain_no_cure():
    state = _hand_state(side0_ability=ABILITY_HYDRATION, side0_status=STATUS_BURN)
    bs = state.battle_state
    bs[OFF_FIELD + F_WEATHER] = WEATHER_NONE
    poff = _p0_off()
    apply_shed_skin_hydration(bs, poff, game_data=None, gen5_prng=Gen5PRNG())
    assert get_status(int(bs[poff + 12])) == STATUS_BURN


# ============================================================================
# Switch-in abilities
# ============================================================================


def test_intimidate_lowers_opponent_attack():
    state = _hand_state(side0_ability=ABILITY_INTIMIDATE)
    bs = state.battle_state
    p0 = _p0_off()
    p1 = _p1_off()

    assert extract_boost(int(bs[p1 + 13]), 0) == 0
    apply_switch_in_ability(bs, p0, p1, did_switch=True)
    assert extract_boost(int(bs[p1 + 13]), 0) == -1


def test_intimidate_skipped_when_no_switch():
    state = _hand_state(side0_ability=ABILITY_INTIMIDATE)
    bs = state.battle_state
    p0, p1 = _p0_off(), _p1_off()
    apply_switch_in_ability(bs, p0, p1, did_switch=False)
    assert extract_boost(int(bs[p1 + 13]), 0) == 0


def test_intimidate_blocked_by_clear_body():
    state = _hand_state(
        side0_ability=ABILITY_INTIMIDATE, side1_ability=ABILITY_CLEAR_BODY
    )
    bs = state.battle_state
    p0, p1 = _p0_off(), _p1_off()
    apply_switch_in_ability(bs, p0, p1, did_switch=True)
    assert extract_boost(int(bs[p1 + 13]), 0) == 0


def test_intimidate_reversed_by_contrary():
    state = _hand_state(
        side0_ability=ABILITY_INTIMIDATE, side1_ability=ABILITY_CONTRARY
    )
    bs = state.battle_state
    p0, p1 = _p0_off(), _p1_off()
    apply_switch_in_ability(bs, p0, p1, did_switch=True)
    assert extract_boost(int(bs[p1 + 13]), 0) == 1


def test_intimidate_triggers_defiant():
    state = _hand_state(side0_ability=ABILITY_INTIMIDATE, side1_ability=ABILITY_DEFIANT)
    bs = state.battle_state
    p0, p1 = _p0_off(), _p1_off()
    apply_switch_in_ability(bs, p0, p1, did_switch=True)
    # -1 from intimidate, +2 from defiant -> +1 net
    assert extract_boost(int(bs[p1 + 13]), 0) == 1


def test_intimidate_blocked_by_substitute():
    state = _hand_state(side0_ability=ABILITY_INTIMIDATE)
    bs = state.battle_state
    bs[OFF_FIELD + F_SUBSTITUTE_1] = 25  # opponent (side1) has substitute
    p0, p1 = _p0_off(), _p1_off()
    apply_switch_in_ability(bs, p0, p1, did_switch=True)
    assert extract_boost(int(bs[p1 + 13]), 0) == 0


def test_drought_sets_sun():
    state = _hand_state(side1_ability=ABILITY_DROUGHT)
    bs = state.battle_state
    p0, p1 = _p0_off(), _p1_off()
    apply_switch_in_ability(bs, p1, p0, did_switch=True)
    assert int(bs[OFF_FIELD + F_WEATHER]) == WEATHER_SUN
    assert int(bs[OFF_META + M_WEATHER_TURNS]) == 5


def test_drizzle_sets_rain():
    state = _hand_state(side1_ability=ABILITY_DRIZZLE)
    bs = state.battle_state
    p0, p1 = _p0_off(), _p1_off()
    apply_switch_in_ability(bs, p1, p0, did_switch=True)
    assert int(bs[OFF_FIELD + F_WEATHER]) == WEATHER_RAIN


def test_electric_surge_sets_terrain():
    state = _hand_state(side0_ability=ABILITY_ELECTRIC_SURGE)
    bs = state.battle_state
    p0, p1 = _p0_off(), _p1_off()
    apply_switch_in_ability(bs, p0, p1, did_switch=True)
    assert int(bs[OFF_FIELD + F_TERRAIN]) == TERRAIN_ELECTRIC
    assert int(bs[OFF_META + M_TERRAIN_TURNS]) == 5


def test_download_boosts_atk_when_def_lower():
    state = _hand_state(side0_ability=ABILITY_DOWNLOAD)
    bs = state.battle_state
    p0, p1 = _p0_off(), _p1_off()
    bs[p1 + 8] = 50  # opponent def
    bs[p1 + 10] = 100  # opponent spd
    apply_switch_in_ability(bs, p0, p1, did_switch=True)
    # Def < SpD -> +1 Atk
    assert extract_boost(int(bs[p0 + 13]), 0) == 1


def test_download_boosts_spa_when_spd_lower():
    state = _hand_state(side0_ability=ABILITY_DOWNLOAD)
    bs = state.battle_state
    p0, p1 = _p0_off(), _p1_off()
    bs[p1 + 8] = 100  # opponent def
    bs[p1 + 10] = 50  # opponent spd
    apply_switch_in_ability(bs, p0, p1, did_switch=True)
    # Def >= SpD -> +1 SpA
    assert extract_boost(int(bs[p0 + 13]), 8) == 1


def test_intrepid_sword_boosts_attack():
    state = _hand_state(side0_ability=ABILITY_INTREPID_SWORD)
    bs = state.battle_state
    p0, p1 = _p0_off(), _p1_off()
    apply_switch_in_ability(bs, p0, p1, did_switch=True)
    assert extract_boost(int(bs[p0 + 13]), 0) == 1


def test_dauntless_shield_boosts_defense_once():
    state = _hand_state(side0_ability=ABILITY_DAUNTLESS_SHIELD)
    bs = state.battle_state
    p0, p1 = _p0_off(), _p1_off()
    apply_switch_in_ability(bs, p0, p1, did_switch=True)
    assert extract_boost(int(bs[p0 + 13]), 4) == 1
    # Reset boost manually then call again — should NOT re-apply
    bs[p0 + 13] = NEUTRAL_BOOSTS_13
    apply_switch_in_ability(bs, p0, p1, did_switch=True)
    assert extract_boost(int(bs[p0 + 13]), 4) == 0


# ============================================================================
# Regenerator / Natural Cure on switch out
# ============================================================================


def test_regenerator_heals_one_third():
    state = _hand_state(side1_ability=ABILITY_REGENERATOR, side1_hp=50, side1_max=100)
    bs = state.battle_state
    p1 = _p1_off()
    apply_regenerator_on_switch_out(bs, p1, did_switch=True)
    # 100 / 3 = 33; 50 + 33 = 83
    assert int(bs[p1 + 1]) == 83


def test_regenerator_does_not_overheal():
    state = _hand_state(side1_ability=ABILITY_REGENERATOR, side1_hp=90, side1_max=100)
    bs = state.battle_state
    p1 = _p1_off()
    apply_regenerator_on_switch_out(bs, p1, did_switch=True)
    assert int(bs[p1 + 1]) == 100


def test_regenerator_skipped_without_switch():
    state = _hand_state(side1_ability=ABILITY_REGENERATOR, side1_hp=50, side1_max=100)
    bs = state.battle_state
    p1 = _p1_off()
    apply_regenerator_on_switch_out(bs, p1, did_switch=False)
    assert int(bs[p1 + 1]) == 50


def test_natural_cure_clears_status_on_switch():
    state = _hand_state(side1_ability=ABILITY_NATURAL_CURE, side1_status=STATUS_BURN)
    bs = state.battle_state
    p1 = _p1_off()
    apply_natural_cure_on_switch_out(bs, p1, did_switch=True)
    assert get_status(int(bs[p1 + 12])) == STATUS_NONE


# ============================================================================
# Absorb abilities (healing & stat boosts)
# ============================================================================


def test_water_absorb_heals_25_percent():
    state = _hand_state(side1_ability=ABILITY_WATER_ABSORB, side1_hp=50, side1_max=100)
    bs = state.battle_state
    p1 = _p1_off()
    apply_absorb_ability_healing(bs, p1, TYPE_WATER, hit=True)
    # 100 / 4 = 25; 50 + 25 = 75
    assert int(bs[p1 + 1]) == 75


def test_volt_absorb_heals_on_electric():
    state = _hand_state(side1_ability=ABILITY_VOLT_ABSORB, side1_hp=20, side1_max=100)
    bs = state.battle_state
    p1 = _p1_off()
    apply_absorb_ability_healing(bs, p1, TYPE_ELECTRIC, hit=True)
    assert int(bs[p1 + 1]) == 45


def test_water_absorb_no_heal_for_fire_move():
    state = _hand_state(side1_ability=ABILITY_WATER_ABSORB, side1_hp=50, side1_max=100)
    bs = state.battle_state
    p1 = _p1_off()
    apply_absorb_ability_healing(bs, p1, TYPE_FIRE, hit=True)
    assert int(bs[p1 + 1]) == 50


def test_sap_sipper_boosts_attack_on_grass_move():
    state = _hand_state(side1_ability=ABILITY_SAP_SIPPER)
    bs = state.battle_state
    p1 = _p1_off()
    apply_absorb_ability_healing(bs, p1, TYPE_GRASS, hit=True)
    assert extract_boost(int(bs[p1 + 13]), 0) == 1


def test_lightning_rod_boosts_spa_on_electric():
    state = _hand_state(side1_ability=ABILITY_LIGHTNING_ROD)
    bs = state.battle_state
    p1 = _p1_off()
    apply_absorb_ability_healing(bs, p1, TYPE_ELECTRIC, hit=True)
    assert extract_boost(int(bs[p1 + 13]), 8) == 1


def test_motor_drive_boosts_speed_on_electric():
    state = _hand_state(side1_ability=ABILITY_MOTOR_DRIVE)
    bs = state.battle_state
    p1 = _p1_off()
    apply_absorb_ability_healing(bs, p1, TYPE_ELECTRIC, hit=True)
    assert extract_boost(int(bs[p1 + 14]), 0) == 1


def test_flash_fire_sets_flag():
    state = _hand_state(side1_ability=ABILITY_FLASH_FIRE)
    bs = state.battle_state
    p1 = _p1_off()
    apply_absorb_ability_healing(bs, p1, TYPE_FIRE, hit=True)
    assert int(bs[p1 + 15]) & 0x200, "Flash Fire flag bit should be set"


# ============================================================================
# Weakness Policy
# ============================================================================


def test_weakness_policy_triggers_on_super_effective():
    state = _hand_state(side1_type=TYPE_GRASS, side1_item=ITEM_WEAKNESS_POLICY)
    bs = state.battle_state
    p1 = _p1_off()
    apply_weakness_policy(bs, p1, TYPE_FIRE, hit=True, damage_dealt=20)
    assert extract_boost(int(bs[p1 + 13]), 0) == 2  # +2 Atk
    assert extract_boost(int(bs[p1 + 13]), 8) == 2  # +2 SpA
    assert int(bs[p1 + 6]) == 0  # item consumed


def test_weakness_policy_skips_normal_effective():
    state = _hand_state(side1_type=TYPE_NORMAL, side1_item=ITEM_WEAKNESS_POLICY)
    bs = state.battle_state
    p1 = _p1_off()
    apply_weakness_policy(bs, p1, TYPE_FIRE, hit=True, damage_dealt=10)
    assert extract_boost(int(bs[p1 + 13]), 0) == 0
    assert int(bs[p1 + 6]) == ITEM_WEAKNESS_POLICY


# ============================================================================
# KO Boost abilities
# ============================================================================


def test_moxie_boosts_attack_on_ko():
    state = _hand_state(side0_ability=ABILITY_MOXIE)
    bs = state.battle_state
    p0 = _p0_off()
    apply_ko_boost_ability(bs, p0, target_fainted=True, hit=True)
    assert extract_boost(int(bs[p0 + 13]), 0) == 1


def test_moxie_no_boost_when_target_alive():
    state = _hand_state(side0_ability=ABILITY_MOXIE)
    bs = state.battle_state
    p0 = _p0_off()
    apply_ko_boost_ability(bs, p0, target_fainted=False, hit=True)
    assert extract_boost(int(bs[p0 + 13]), 0) == 0


def test_soul_heart_boosts_spa_on_ko():
    state = _hand_state(side0_ability=ABILITY_SOUL_HEART)
    bs = state.battle_state
    p0 = _p0_off()
    apply_ko_boost_ability(bs, p0, target_fainted=True, hit=True)
    assert extract_boost(int(bs[p0 + 13]), 8) == 1


def test_beast_boost_picks_highest_stat():
    state = _hand_state(side0_ability=ABILITY_BEAST_BOOST)
    bs = state.battle_state
    p0 = _p0_off()
    # Set base stats: spe is highest
    bs[p0 + 7] = 80
    bs[p0 + 8] = 80
    bs[p0 + 9] = 80
    bs[p0 + 10] = 80
    bs[p0 + 11] = 130  # spe
    apply_ko_boost_ability(bs, p0, target_fainted=True, hit=True)
    assert extract_boost(int(bs[p0 + 14]), 0) == 1  # +1 Spe


# ============================================================================
# Contact status abilities
# ============================================================================


def test_static_paralysis_can_trigger():
    """Roll a Gen5PRNG until we land < 30 then verify it paralyzes."""
    gd = load_game_data()
    # Find a contact move ID — Tackle is move 33 (id may vary). Use move
    # whose flags say contact. We don't actually care about the move; we just
    # need its move_flags to include FLAG_CONTACT. Tackle is canonical id 33
    # in Showdown so try 33.
    move_id = 33
    flags = int(gd.move_flags[move_id])
    assert flags & 0x1, f"move {move_id} should have CONTACT flag, got {flags}"

    state = _hand_state(side1_ability=ABILITY_STATIC)
    bs = state.battle_state
    p0, p1 = _p0_off(), _p1_off()

    # Use a deterministic seed; loop until status applied OR we fail
    # We just want to confirm SOMETHING can happen — not specific seed.
    fired = False
    for seed in range(1, 200):
        st = _hand_state(side1_ability=ABILITY_STATIC)
        bs2 = st.battle_state
        prng = Gen5PRNG(seed=(seed, seed, seed, seed))
        apply_contact_status_ability(
            bs2,
            move_id,
            p0,
            p1,
            hit=True,
            game_data=gd,
            gen5_prng=prng,
        )
        if get_status(int(bs2[p0 + 12])) == STATUS_PARALYSIS:
            fired = True
            break
    assert fired, "Static should eventually paralyze attacker over many seeds"


def test_contact_ability_skipped_for_long_reach():
    """Long Reach attacker -> contact ability never fires."""
    gd = load_game_data()
    move_id = 33
    state = _hand_state(
        side0_ability=235,  # arbitrary nonzero -> we'll set to LONG_REACH below
        side1_ability=ABILITY_STATIC,
    )
    bs = state.battle_state
    from pokepy.core.constants import ABILITY_LONG_REACH

    p0, p1 = _p0_off(), _p1_off()
    bs[p0 + 5] = ABILITY_LONG_REACH

    # Try many seeds — none should paralyze
    for seed in range(1, 50):
        bs[p0 + 12] = set_status(STATUS_NONE, 0)
        prng = Gen5PRNG(seed=(seed, seed, seed, seed))
        apply_contact_status_ability(
            bs,
            move_id,
            p0,
            p1,
            hit=True,
            game_data=gd,
            gen5_prng=prng,
        )
        assert get_status(int(bs[p0 + 12])) == STATUS_NONE
