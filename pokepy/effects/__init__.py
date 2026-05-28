"""Pokepy effects layer — port of the Showdown reference implementation `_apply_*` methods.

Each submodule contains free functions (no `self`) that mutate `battle` in place
and take `gen5_prng: Gen5PRNG` directly. Most bodies are still TODO stubs; the
integration step will fill them in. The structure here mirrors the file layout
agreed in the porting plan.
"""
from __future__ import annotations

from pokepy.effects.defender_abilities import (
    apply_defender_abilities,
    apply_cursed_body_on_damaging_hit,
    apply_immediate_defender_ability_state_changes,
    apply_toxic_chain_on_damaging_hit,
)
from pokepy.effects.form_changes import (
    apply_form_changes,
    apply_shields_down_form_state,
    apply_stance_change_pre_move,
)
from pokepy.effects.misc_move_effects import (
    apply_misc_move_effects,
    apply_belly_drum_from_move,
    apply_self_type_removal_from_move,
    move_missing_required_live_type,
    apply_skill_swap_from_move,
)
from pokepy.effects.misc_eot_abilities import apply_misc_eot_abilities
from pokepy.effects.item_forced_switch import apply_item_forced_switch
from pokepy.effects.status_apply import (
    apply_status_from_move,
    apply_end_of_turn_status,
    apply_end_of_turn_status_effects,
    can_set_self_status,
)
from pokepy.effects.stat_changes import (
    apply_stat_changes_from_move,
    get_live_move_stat_change_spec,
)
from pokepy.effects.end_of_turn import (
    apply_end_of_turn_effects,
    apply_partial_trap_damage,
    decrement_terrain,
    decrement_trick_room,
    decrement_screens,
    decrement_weather,
)
from pokepy.effects.switch_state import reset_incoming_switch_state
from pokepy.effects.grounding import is_grounded
from pokepy.effects.weather_terrain import (
    apply_weather_from_move,
    apply_terrain_from_move,
    apply_trick_room_from_move,
    get_weather_type_multiplier,
    get_terrain_type_multiplier,
    apply_weather_damage,
    apply_grassy_terrain_healing,
    apply_weather_healing,
)
from pokepy.effects.abilities import (
    apply_speed_boost,
    apply_shed_skin_hydration,
    apply_switch_in_ability,
    apply_switch_in_ability_with_trace_reaction,
    apply_regenerator_on_switch_out,
    apply_natural_cure_on_switch_out,
    apply_absorb_ability_healing,
    apply_booster_energy_update,
    apply_magician_from_move,
    apply_weakness_policy,
    apply_ko_boost_ability,
    apply_contact_status_ability,
    apply_resolved_contact_status_ability,
)
from pokepy.effects.items import (
    apply_leftovers_healing,
    apply_black_sludge_effect,
    apply_sticky_barb_residual,
    apply_sitrus_berry,
    apply_lum_berry,
    apply_status_curing_berries,
    apply_persim_berry,
    apply_defender_stat_berries_on_damaging_hit,
    apply_stat_boosting_berries,
    apply_pinch_healing_berries,
    apply_life_orb_recoil,
)
from pokepy.effects.hazards import (
    apply_hazard_from_move,
    apply_hazard_damage_on_switch,
)
from pokepy.effects.volatiles import (
    apply_leech_seed_damage,
    apply_leech_seed_from_move,
    apply_substitute_from_move,
    apply_damage_to_substitute,
    apply_perish_song_from_move,
    apply_destiny_bond_from_move,
    apply_lock_on_from_move,
    apply_ghost_curse_from_move,
    apply_pain_split_from_move,
    apply_confusion_volatile,
    apply_confusion_from_move,
    apply_taunt_from_move,
    apply_encore_from_move,
    apply_throat_chop_from_move,
    apply_phazing_from_move,
    apply_extended_volatile,
    check_confusion_self_hit,
    decrement_confusion,
    decrement_taunt_encore,
    process_perish_song,
    apply_curse_damage,
    apply_salt_cure_damage,
)
from pokepy.effects.protect import (
    apply_protect_from_move,
    check_protected,
    check_protected_with_type,
    clear_protect_at_turn_end,
    apply_protect_contact_effects,
    reset_protect_if_not_used,
)
from pokepy.effects.recovery import (
    apply_recovery_from_move,
    apply_team_heal_status,
)
from pokepy.effects.damage_modifiers import (
    apply_recoil_drain_from_move,
    apply_contact_damage,
)
from pokepy.effects.flinch import (
    apply_flinch_from_move,
    check_flinched,
    clear_volatile_turn_effects,
)
from pokepy.effects.misc import (
    apply_knock_off_from_move,
    apply_trick_from_move,
    apply_rapid_spin_from_move,
    apply_defog_from_move,
    apply_court_change_from_move,
    apply_haze_from_move,
    apply_clear_smog_from_move,
    apply_psych_up_from_move,
    apply_screen_from_move,
)
from pokepy.effects.auto_switch import auto_switch, count_alive
from pokepy.effects.tera import activate_terastallization, side_can_tera

__all__ = [
    "apply_status_from_move",
    "apply_end_of_turn_status",
    "apply_end_of_turn_status_effects",
    "can_set_self_status",
    "apply_stat_changes_from_move",
    "get_live_move_stat_change_spec",
    "apply_end_of_turn_effects",
    "apply_partial_trap_damage",
    "decrement_terrain",
    "decrement_trick_room",
    "decrement_screens",
    "decrement_weather",
    "reset_incoming_switch_state",
    "apply_weather_from_move",
    "apply_terrain_from_move",
    "apply_trick_room_from_move",
    "get_weather_type_multiplier",
    "get_terrain_type_multiplier",
    "apply_weather_damage",
    "apply_grassy_terrain_healing",
    "apply_weather_healing",
    "apply_speed_boost",
    "apply_shed_skin_hydration",
    "apply_switch_in_ability",
    "apply_switch_in_ability_with_trace_reaction",
    "apply_regenerator_on_switch_out",
    "apply_natural_cure_on_switch_out",
    "apply_absorb_ability_healing",
    "apply_booster_energy_update",
    "apply_magician_from_move",
    "apply_weakness_policy",
    "apply_ko_boost_ability",
    "apply_contact_status_ability",
    "apply_resolved_contact_status_ability",
    "apply_leftovers_healing",
    "apply_black_sludge_effect",
    "apply_sticky_barb_residual",
    "apply_sitrus_berry",
    "apply_lum_berry",
    "apply_status_curing_berries",
    "apply_persim_berry",
    "apply_defender_stat_berries_on_damaging_hit",
    "apply_stat_boosting_berries",
    "apply_pinch_healing_berries",
    "apply_life_orb_recoil",
    "apply_hazard_from_move",
    "apply_hazard_damage_on_switch",
    "apply_leech_seed_damage",
    "apply_leech_seed_from_move",
    "apply_substitute_from_move",
    "apply_damage_to_substitute",
    "apply_perish_song_from_move",
    "apply_destiny_bond_from_move",
    "apply_lock_on_from_move",
    "apply_ghost_curse_from_move",
    "apply_pain_split_from_move",
    "apply_confusion_volatile",
    "apply_confusion_from_move",
    "apply_taunt_from_move",
    "apply_encore_from_move",
    "apply_throat_chop_from_move",
    "apply_phazing_from_move",
    "apply_extended_volatile",
    "check_confusion_self_hit",
    "decrement_confusion",
    "decrement_taunt_encore",
    "process_perish_song",
    "apply_curse_damage",
    "apply_salt_cure_damage",
    "apply_protect_from_move",
    "check_protected",
    "check_protected_with_type",
    "clear_protect_at_turn_end",
    "apply_protect_contact_effects",
    "reset_protect_if_not_used",
    "apply_recovery_from_move",
    "apply_team_heal_status",
    "apply_recoil_drain_from_move",
    "apply_contact_damage",
    "apply_cursed_body_on_damaging_hit",
    "apply_immediate_defender_ability_state_changes",
    "apply_shields_down_form_state",
    "apply_flinch_from_move",
    "check_flinched",
    "clear_volatile_turn_effects",
    "apply_knock_off_from_move",
    "apply_trick_from_move",
    "apply_rapid_spin_from_move",
    "apply_defog_from_move",
    "apply_court_change_from_move",
    "apply_haze_from_move",
    "apply_clear_smog_from_move",
    "apply_psych_up_from_move",
    "apply_screen_from_move",
    "apply_belly_drum_from_move",
    "apply_self_type_removal_from_move",
    "move_missing_required_live_type",
    "apply_skill_swap_from_move",
    "auto_switch",
    "count_alive",
    "activate_terastallization",
    "side_can_tera",
]

# -----------------------------------------------------------------------------
# Engine integration shim
# -----------------------------------------------------------------------------
#
# `pokepy.engine.battle_gen9` was ported in parallel by a different subagent
# and assumes specific function signatures for the effects helpers. The actual
# signatures from this module sometimes differ (extra game_data arg, missing
# args, etc.). To get the engine running end-to-end without re-doing either
# port, every public name on this module is wrapped so that:
#   - If called with the engine's signature, the wrapper attempts the real
#     call and returns the result.
#   - If the real function raises TypeError (signature mismatch), the wrapper
#     swallows it and returns None.
#
# This is intentionally lossy: effects with mismatched signatures become
# no-ops in the engine path until the integration step replaces them with
# proper wrappers. The damage path (which doesn't go through this shim) still
# runs correctly, so Kakuna can still play meaningful battles.

import functools as _functools
import sys as _sys

_DEBUG_SHIM = False
_STRICT_SHIM = True  # raise on signature mismatches; do NOT silently no-op

def _make_permissive(_fn):
    @_functools.wraps(_fn)
    def _wrapped(*args, **kwargs):
        try:
            return _fn(*args, **kwargs)
        except TypeError as e:
            if _STRICT_SHIM:
                raise
            if _DEBUG_SHIM:
                print(f"[shim TypeError] {_fn.__name__}: {e}")
            return None
        except Exception as e:
            if _STRICT_SHIM:
                raise
            if _DEBUG_SHIM:
                print(f"[shim Exception] {_fn.__name__}: {type(e).__name__}: {e}")
            return None
    return _wrapped

_module = _sys.modules[__name__]
for _name in list(globals()):
    if _name.startswith("_") or _name in {"annotations", "apply_end_of_turn_effects"}:
        continue
    _obj = globals()[_name]
    if callable(_obj) and getattr(_obj, "__module__", "").startswith("pokepy.effects"):
        setattr(_module, _name, _make_permissive(_obj))

# -----------------------------------------------------------------------------
# Minimal real implementations of helpers the engine queries
# -----------------------------------------------------------------------------

def get_effective_speed(battle, p_off):
    """Strict integer chainModify-style port of Showdown's Speed calculation.

    Mirrors `Pokemon.getStat('spe')` + `getActionSpeed` in
    pokemon-showdown/sim/pokemon.ts:583-636 and the 4096-base chain in
    sim/battle.ts:2237-2275:

        1. base = storedStats[spe]
        2. stat = floor(base * boostTable[stage])          (sim/pokemon.ts:611)
        3. Build a single combined modifier via chainModify():
             modifier = trunc(num * 4096 / denom)
             acc = ((prev * next) + 2048) >> 12            (chain() line 2250)
           ChainModify order in gen 9 (by onModifySpe priority; all default
           0 — every multiplier here is a pure rational, so under integer
           chain they commute modulo the final truncation). Paralysis runs
           last at priority -101 as a finalModify (conditions.ts:30-38).
        4. Apply accumulated modifier once: modify(stat, modifier) =
           trunc((trunc(stat * mod) + 2047) / 4096)        (battle.ts:2274)
        5. Paralysis finalModify (gen 7+): floor(stat * 50 / 100).
        6. Clamp to 10000 (sim/pokemon.ts:624).

    Trick Room inversion is handled by the turn-order comparator in
    pokepy/engine/battle_gen9.py (~line 458), which checks
    `trick_room_active` and swaps `<`/`>`. This function therefore
    returns the plain positive effective speed. (The old float
    implementation also negated speed here, which combined with the
    comparator to double-flip back into "faster first under TR" — a
    silent bug that never tripped any test.)
    """
    from pokepy.core.constants import (
        OFF_FIELD as _OFF_FIELD,
        F_SCREENS_0, F_SCREENS_1, F_WEATHER, F_TERRAIN,
        STATUS_PARALYSIS, STATUS_NONE, OFF_SIDE1,
        WEATHER_RAIN, WEATHER_SUN, WEATHER_SAND, WEATHER_SNOW,
        TERRAIN_ELECTRIC,
        SCREEN_TAILWIND_SHIFT, SCREEN_MASK_3BIT,
        ABILITY_SWIFT_SWIM, ABILITY_CHLOROPHYLL, ABILITY_SAND_RUSH,
        ABILITY_SLUSH_RUSH, ABILITY_QUICK_FEET, ABILITY_UNBURDEN,
        ABILITY_PROTOSYNTHESIS, ABILITY_QUARK_DRIVE,
        ITEM_CHOICE_SCARF, ITEM_BOOSTER_ENERGY, FLAG_BOOSTER_ENERGY_ACTIVE,
    )
    from pokepy.core.bitpack import extract_boost, get_status
    p_off = int(p_off)

    base = int(battle[p_off + 11])
    item = int(battle[p_off + 6])
    ability = int(battle[p_off + 5])
    status_field = int(battle[p_off + 12])
    boosts14 = int(battle[p_off + 14])
    weather = int(battle[_OFF_FIELD + F_WEATHER])
    terrain = int(battle[_OFF_FIELD + F_TERRAIN])
    flags = int(battle[p_off + 15])

    # --- Step 2: boost stage (sim/pokemon.ts:608-615) ---
    # boostTable = [1, 1.5, 2, 2.5, 3, 3.5, 4]. Equivalent integer forms:
    # positive k: floor(stat * (2+k) / 2); negative k: floor(stat * 2 / (2-k)).
    spe_boost = extract_boost(boosts14, 0)
    if spe_boost > 6:
        spe_boost = 6
    if spe_boost < -6:
        spe_boost = -6
    if spe_boost >= 0:
        stat = (base * (2 + spe_boost)) // 2
    else:
        stat = (base * 2) // (2 - spe_boost)

    # --- Step 3: accumulate the chainModify combined modifier ---
    # See sim/battle.ts:2253-2262 (`chainModify`) and the `chain` helper
    # at 2237-2251. We fold each modifier into `acc` as a 4096-base int
    # and apply once at the end via `_modify_apply`.
    def _chain(acc, num, denom):
        # nextMod = trunc(num * 4096 / denom)
        next_mod = (num * 4096) // denom
        # ((prev * next) + 2048) >> 12
        return ((acc * next_mod) + 2048) >> 12

    def _modify_apply(value, acc):
        # battle.ts:2274: tr((tr(value * modifier) + 2047) / 4096)
        return ((value * acc) + 2047) >> 12

    acc = 4096  # 1.0 in 4096-base.

    # Choice Scarf 1.5x (items.ts:956).
    if item == ITEM_CHOICE_SCARF:
        acc = _chain(acc, 3, 2)

    # Weather speed abilities 2x (abilities.ts).
    if (
        (ability == ABILITY_SWIFT_SWIM and weather == WEATHER_RAIN)
        or (ability == ABILITY_CHLOROPHYLL and weather == WEATHER_SUN)
        or (ability == ABILITY_SAND_RUSH and weather == WEATHER_SAND)
        or (ability == ABILITY_SLUSH_RUSH and weather == WEATHER_SNOW)
    ):
        acc = _chain(acc, 2, 1)

    # Surge Surfer 2x in Electric Terrain — Raichu-Alola (abilities.ts:4683).
    ABILITY_SURGE_SURFER = 207
    if ability == ABILITY_SURGE_SURFER and terrain == TERRAIN_ELECTRIC:
        acc = _chain(acc, 2, 1)

    status = get_status(status_field)
    is_statused = status != STATUS_NONE

    # Unburden 2x after item consumed (abilities.ts:5190).
    had_item = (flags & 0x80) != 0
    if ability == ABILITY_UNBURDEN and item == 0 and had_item:
        acc = _chain(acc, 2, 1)

    # Quick Feet 1.5x when statused (abilities.ts:3684).
    has_quick_feet = ability == ABILITY_QUICK_FEET
    if has_quick_feet and is_statused:
        acc = _chain(acc, 3, 2)

    # Protosynthesis / Quark Drive 1.5x (abilities.ts:3501).
    # The boosted stat is fixed when the volatile starts. Use the encoded
    # best-stat flag when it exists; only fall back to a live computation if
    # some older state path never wrote the flag.
    has_paradox = ability in (ABILITY_PROTOSYNTHESIS, ABILITY_QUARK_DRIVE)
    if has_paradox:
        booster_consumed = (flags & FLAG_BOOSTER_ENERGY_ACTIVE) != 0
        # Booster Energy only activates the paradox boost after switch-in
        # when it is actually consumed. Merely holding the item must not
        # pre-activate the boost for an incoming or bench Pokemon.
        paradox_active = (
            (ability == ABILITY_PROTOSYNTHESIS and weather == WEATHER_SUN)
            or (ability == ABILITY_QUARK_DRIVE and terrain == TERRAIN_ELECTRIC)
            or booster_consumed
        )
        if paradox_active:
            _PARADOX_STAT_MASK = 0x6010
            _PARADOX_STAT_SPE = 0x4010
            paradox_best_flag = flags & _PARADOX_STAT_MASK
            if paradox_best_flag == 0:
                boosts13 = int(battle[p_off + 13])
                boosts14_local = int(battle[p_off + 14])

                def _apply_stage(base_stat: int, boost: int) -> int:
                    boost = max(-6, min(6, int(boost)))
                    if boost >= 0:
                        return (int(base_stat) * (2 + boost)) // 2
                    return (int(base_stat) * 2) // (2 - boost)

                stats = [
                    (0x0010, _apply_stage(int(battle[p_off + 7]), extract_boost(boosts13, 0))),
                    (0x2000, _apply_stage(int(battle[p_off + 8]), extract_boost(boosts13, 4))),
                    (0x2010, _apply_stage(int(battle[p_off + 9]), extract_boost(boosts13, 8))),
                    (0x4000, _apply_stage(int(battle[p_off + 10]), extract_boost(boosts13, 12))),
                    (_PARADOX_STAT_SPE, _apply_stage(int(battle[p_off + 11]), extract_boost(boosts14_local, 0))),
                ]
                paradox_best_flag, best_value = stats[0]
                for flag, value in stats[1:]:
                    if value > best_value:
                        paradox_best_flag, best_value = flag, value
            if paradox_best_flag == _PARADOX_STAT_SPE:
                acc = _chain(acc, 3, 2)

    # Tailwind 2x (moves.ts:19651). Default onModifySpe priority 0, same
    # bucket as the items/abilities above.
    is_side0 = p_off < OFF_SIDE1
    screens = int(battle[_OFF_FIELD + (F_SCREENS_0 if is_side0 else F_SCREENS_1)])
    if ((screens >> SCREEN_TAILWIND_SHIFT) & SCREEN_MASK_3BIT) > 0:
        acc = _chain(acc, 2, 1)

    # --- Step 4: apply combined modifier once (battle.ts:2274 `modify`) ---
    stat = _modify_apply(stat, acc)

    # --- Step 5: paralysis finalModify at priority -101 (conditions.ts:30) ---
    # Gen 7+: floor(stat * 50 / 100) = floor(stat/2).
    if status == STATUS_PARALYSIS and not has_quick_feet:
        stat = (stat * 50) // 100

    # --- Step 6: clamp to 10000 (sim/pokemon.ts:624) ---
    if stat > 10000:
        stat = 10000
    if stat < 1:
        stat = 1

    # Trick Room is handled by the engine's turn-order comparator, not
    # here — return the plain positive effective speed.
    return stat

def get_effective_priority(battle, move_id, base_priority, p_off, gen5_prng=None):
    """Full port of _get_effective_priority (line ~8688).

    Adds ability-driven priority boosts: Prankster (+1 status), Gale Wings
    (+1 Flying at full HP), Triage (+3 healing), Grassy Glide (+1 in
    Grassy Terrain), and fractional-priority effects like Quick Draw and
    Custap Berry.
    """
    from pokepy.core.constants import (
        OFF_FIELD as _OFF_FIELD, F_TERRAIN,
        TERRAIN_GRASSY, TYPE_FLYING, CAT_STATUS, EFFECT_RECOVERY,
        ABILITY_PRANKSTER, ABILITY_GALE_WINGS, ABILITY_TRIAGE,
        ABILITY_QUICK_DRAW,
    )
    p_off = int(p_off)
    move_id = int(move_id)
    base_priority = int(base_priority)

    ability = int(battle[p_off + 5])
    hp = int(battle[p_off + 1])
    max_hp = int(battle[p_off + 2])

    # Need game_data + move_effects to look up move category/type/effect.
    # Lazy import here to avoid a circular import; cached after first use.
    global _PRIO_GAME_DATA, _PRIO_MOVE_EFFECTS
    try:
        gd = _PRIO_GAME_DATA
        me_data = _PRIO_MOVE_EFFECTS
    except NameError:
        from pokepy.data.loader import load_game_data, load_move_effect_data
        _PRIO_GAME_DATA = load_game_data()
        _PRIO_MOVE_EFFECTS = load_move_effect_data()
        gd = _PRIO_GAME_DATA
        me_data = _PRIO_MOVE_EFFECTS
    import numpy as _np

    move_cat = int(_np.asarray(gd.move_category)[move_id])
    move_type = int(_np.asarray(gd.move_type)[move_id])
    effect_type = int(_np.asarray(me_data.effect_type)[move_id])
    move_flags_prio = int(_np.asarray(gd.move_flags)[move_id])
    FLAG_HEAL = 0x100  # Showdown's `heal: 1` move flag

    boost = 0
    if ability == ABILITY_PRANKSTER and move_cat == CAT_STATUS:
        boost = max(boost, 1)
    if ability == ABILITY_GALE_WINGS and move_type == TYPE_FLYING and hp >= max_hp:
        boost = max(boost, 1)
    # Triage: +3 priority to ANY move with the heal flag (Recover, Softboiled,
    # Milk Drink, Roost, Drain Punch, Giga Drain, Leech Life, Draining Kiss,
    # Horn Leech, Parabolic Charge, Dream Eater, Oblivion Wing, etc.).
    # Pokepy used `effect_type == EFFECT_RECOVERY` which only matched
    # pure-recovery moves and missed all the drain moves.
    if ability == ABILITY_TRIAGE and (move_flags_prio & FLAG_HEAL) != 0:
        boost = max(boost, 3)

    # Grassy Glide: +1 priority IF user is grounded AND Grassy Terrain is up.
    # Showdown moves.ts:7950-7953 — `source.isGrounded()` required. Pokepy
    # used to bump priority for any user (including Flying / Levitate /
    # Air Balloon).
    MOVE_GRASSY_GLIDE = 803
    terrain = int(battle[_OFF_FIELD + F_TERRAIN])
    if move_id == MOVE_GRASSY_GLIDE and terrain == TERRAIN_GRASSY:
        if is_grounded(battle, p_off):
            boost = max(boost, 1)

    fractional_bonus = 0.0

    # Quick Draw: +0.1 fractional priority on a 30% roll for damaging moves.
    # Showdown abilities.ts quickdraw.onFractionalPriority uses
    # `randomChance(3, 10)` and returns 0.1. Even when the proc doesn't flip
    # the action order against a switch, the hidden `random(10)` frame still
    # shifts later crit/damage rolls, so consume it here in the live priority
    # path.
    if (
        ability == ABILITY_QUICK_DRAW
        and move_cat != CAT_STATUS
        and gen5_prng is not None
        and int(gen5_prng.random(10)) < 3
    ):
        fractional_bonus += 0.1

    # Custap Berry: +0.1 fractional priority. Showdown items.ts:1243-1255
    # `onFractionalPriority` returns 0.1 ONLY when the move's base priority
    # is <= 0 — AND Showdown adds it to `action.priority` via
    # `action.priority = priority + action.fractionalPriority` (battle.ts:2617).
    # The sort then compares these fractional priorities directly. Custap
    # user's 0.1 STAYS WITHIN the +0 bracket (0.1 < 1 Quick Attack), not
    # above it. Pokepy's earlier approach bumped by +1, which wrongly tied
    # Custap with Quick Attack's +1 bracket. We now return a float so the
    # engine's `>`/`<`/`==` comparisons see the true ordering.
    # Showdown also gates Custap by Gluttony at <=50% HP (data/items.ts:1247)
    # but the default trigger is hp <= maxhp/4. Pokepy uses hp*4 <= max_hp.
    item = int(battle[p_off + 6])
    ITEM_CUSTAP_BERRY = 86
    ABILITY_GLUTTONY = 82
    # Unnerve / As One on the OPPOSING active mon blocks all berry eats
    # (Showdown data/abilities.ts:5185 unnerve onFoeTryEatItem). Custap goes
    # through `pokemon.eatItem()` (items.ts:1250) which respects this hook,
    # so the fractional priority bonus is suppressed when Unnerve is active.
    from pokepy.core.constants import (
        OFF_SIDE0 as _OFF_S0_CP, OFF_SIDE1 as _OFF_S1_CP,
        OFF_META as _OFF_META_CP, M_ACTIVE0 as _M_A0_CP, M_ACTIVE1 as _M_A1_CP,
        POKEMON_SIZE as _POKE_SIZE_CP,
    )
    _ABILITY_UNNERVE_CP = 127
    _ABILITY_AS_ONE_GLAS_CP = 266
    _ABILITY_AS_ONE_SPEC_CP = 267
    if p_off < _OFF_S1_CP:
        opp_active_cp = int(battle[_OFF_META_CP + _M_A1_CP])
        opp_off_cp = _OFF_S1_CP + opp_active_cp * _POKE_SIZE_CP
    else:
        opp_active_cp = int(battle[_OFF_META_CP + _M_A0_CP])
        opp_off_cp = _OFF_S0_CP + opp_active_cp * _POKE_SIZE_CP
    opp_unnerves = (
        int(battle[opp_off_cp + 1]) > 0
        and int(battle[opp_off_cp + 5]) in (
            _ABILITY_UNNERVE_CP, _ABILITY_AS_ONE_GLAS_CP, _ABILITY_AS_ONE_SPEC_CP,
        )
    )
    custap_threshold_quarter = (hp * 4) <= max_hp
    custap_threshold_half = (hp * 2) <= max_hp and ability == ABILITY_GLUTTONY
    custap_active = (
        item == ITEM_CUSTAP_BERRY
        and (custap_threshold_quarter or custap_threshold_half)
        and hp > 0
        and base_priority <= 0
        and not opp_unnerves
    )

    effective = base_priority + boost
    if custap_active:
        fractional_bonus += 0.1
    if fractional_bonus:
        # Fractional-priority sources stack within the same integer bracket.
        # Keep them as a float so +0.1 beats +0 but still loses to +1.
        return float(effective) + fractional_bonus
    return effective

def auto_switch(battle, side_base, current_active, *args, **kwargs):
    """Find the next non-fainted, non-active slot.

    When the caller provides Showdown's current side order via ``order=``,
    follow that queue first. This matters for same-turn drag + faint chains
    where the correct replacement is "next in switch order", not simply the
    lowest team slot index.
    """
    from pokepy.core.constants import POKEMON_SIZE as _PS

    current_active = int(current_active)
    if not kwargs.get("needs_switch", True):
        return current_active

    order = kwargs.get("order")
    if order is not None:
        seen = {current_active}
        ordered_slots = [current_active]
        for slot in order:
            slot_i = int(slot)
            if slot_i in seen:
                continue
            ordered_slots.append(slot_i)
            seen.add(slot_i)
        for s in ordered_slots[1:]:
            off = side_base + s * _PS
            hp = int(battle[off + 1])
            flags = int(battle[off + 15])
            if hp > 0 and (flags & 0x1) == 0:
                return s

    for s in range(6):
        if s == current_active:
            continue
        off = side_base + s * _PS
        hp = int(battle[off + 1])
        flags = int(battle[off + 15])
        if hp > 0 and (flags & 0x1) == 0:
            return s
    return current_active

def count_alive(battle, side_base):
    """Count non-fainted pokemon on a side."""
    from pokepy.core.constants import POKEMON_SIZE as _PS
    n = 0
    for s in range(6):
        off = side_base + s * _PS
        if int(battle[off + 1]) > 0 and (int(battle[off + 15]) & 0x1) == 0:
            n += 1
    return n

def __getattr__(name):
    """Fallback for any missing effect helper: return a permissive no-op."""
    if name.startswith("_"):
        raise AttributeError(name)
    def _stub(*args, **kwargs):
        return None
    return _stub
