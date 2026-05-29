"""Smoke tests: ensure each pokepy.effects module imports and the documented
public functions are present. No runtime checks here — bodies are stubs that
raise NotImplementedError. The integration step will replace those.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = {
    "pokepy.effects.status_apply": [
        "apply_status_from_move",
        "apply_end_of_turn_status",
        "apply_end_of_turn_status_effects",
    ],
    "pokepy.effects.stat_changes": ["apply_stat_changes_from_move"],
    "pokepy.effects.end_of_turn": [
        "apply_end_of_turn_effects",
        "decrement_terrain",
        "decrement_trick_room",
        "decrement_screens",
        "decrement_weather",
    ],
    "pokepy.effects.weather_terrain": [
        "apply_weather_from_move",
        "apply_terrain_from_move",
        "apply_trick_room_from_move",
        "get_weather_type_multiplier",
        "get_terrain_type_multiplier",
        "apply_weather_damage",
        "apply_grassy_terrain_healing",
        "apply_weather_healing",
    ],
    "pokepy.effects.abilities": [
        "apply_speed_boost",
        "apply_shed_skin_hydration",
        "apply_switch_in_ability",
        "apply_regenerator_on_switch_out",
        "apply_natural_cure_on_switch_out",
        "apply_absorb_ability_healing",
        "apply_weakness_policy",
        "apply_ko_boost_ability",
        "apply_contact_status_ability",
    ],
    "pokepy.effects.items": [
        "apply_leftovers_healing",
        "apply_black_sludge_effect",
        "apply_sticky_barb_residual",
        "apply_sitrus_berry",
        "apply_lum_berry",
        "apply_status_curing_berries",
        "apply_persim_berry",
        "apply_stat_boosting_berries",
        "apply_pinch_healing_berries",
        "apply_life_orb_recoil",
    ],
    "pokepy.effects.hazards": [
        "apply_hazard_from_move",
        "apply_hazard_damage_on_switch",
    ],
    "pokepy.effects.volatiles": [
        "apply_leech_seed_damage",
        "apply_leech_seed_from_move",
        "apply_substitute_from_move",
        "apply_damage_to_substitute",
        "apply_perish_song_from_move",
        "apply_destiny_bond_from_move",
        "apply_lock_on_from_move",
        "apply_ghost_curse_from_move",
        "apply_pain_split_from_move",
        "apply_confusion_from_move",
        "apply_taunt_from_move",
        "apply_encore_from_move",
        "apply_phazing_from_move",
        "apply_extended_volatile",
        "check_confusion_self_hit",
        "decrement_confusion",
        "decrement_taunt_encore",
        "process_perish_song",
        "apply_curse_damage",
        "apply_salt_cure_damage",
    ],
    "pokepy.effects.protect": [
        "apply_protect_from_move",
        "check_protected",
        "check_protected_with_type",
        "clear_protect_at_turn_end",
        "apply_protect_contact_effects",
        "reset_protect_if_not_used",
    ],
    "pokepy.effects.recovery": [
        "apply_recovery_from_move",
        "apply_team_heal_status",
    ],
    "pokepy.effects.damage_modifiers": [
        "apply_recoil_drain_from_move",
        "apply_contact_damage",
    ],
    "pokepy.effects.flinch": [
        "apply_flinch_from_move",
        "check_flinched",
        "clear_volatile_turn_effects",
    ],
    "pokepy.effects.misc": [
        "apply_knock_off_from_move",
        "apply_trick_from_move",
        "apply_rapid_spin_from_move",
        "apply_defog_from_move",
        "apply_haze_from_move",
        "apply_clear_smog_from_move",
        "apply_psych_up_from_move",
        "apply_screen_from_move",
    ],
    "pokepy.effects.auto_switch": ["auto_switch", "count_alive"],
}


@pytest.mark.parametrize("module_name,functions", list(MODULES.items()))
def test_module_imports_and_exports(module_name, functions):
    mod = importlib.import_module(module_name)
    for fn in functions:
        assert hasattr(mod, fn), f"{module_name} missing {fn}"
        assert callable(getattr(mod, fn)), f"{module_name}.{fn} not callable"


def test_package_reexports():
    pkg = importlib.import_module("pokepy.effects")
    for fns in MODULES.values():
        for fn in fns:
            assert hasattr(pkg, fn), f"pokepy.effects missing re-export {fn}"
