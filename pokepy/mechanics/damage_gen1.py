"""Gen 1 damage — Showdown data/mods/gen1/scripts.ts getDamage parity."""

from __future__ import annotations

import math

import numpy as np

from pokepy.core.bitpack import extract_boost, get_status
from pokepy.core.constants import (
    CAT_PHYSICAL,
    GEN1_PHYSICAL_TYPES,
    OFF_META,
    OFF_SIDE0,
    OFF_SIDE1,
    POKEMON_SIZE,
    STATUS_BURN,
    STATUS_PARALYSIS,
    TYPE_UNKNOWN,
)
from pokepy.core.gen_profile import GEN1_PROFILE, GenProfile
from pokepy.utils.gen5_prng import Gen5PRNG

_GEN1_POS_BOOST = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
_GEN1_NEG_NUM = (100, 66, 50, 40, 33, 28, 25)
_GEN1_ACC_TABLE = (25, 28, 33, 40, 50, 66, 100, 150, 200, 250, 300, 350, 400)


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _random_chance(prng: Gen5PRNG, num: int, denom: int) -> bool:
    return int(num) > 0 and int(prng.random(int(denom))) < int(num)


def _gen1_stat(
    base: int,
    boost: int,
    *,
    status: int,
    stat_name: str,
) -> int:
    boost = _clamp_int(boost, -6, 6)
    stat = int(base)
    if boost >= 0:
        stat = int(math.floor(stat * _GEN1_POS_BOOST[boost]))
    else:
        stat = int(math.floor(stat * _GEN1_NEG_NUM[-boost] / 100))
    if status == STATUS_PARALYSIS and stat_name == "spe":
        stat = math.floor(stat / 4)
    if status == STATUS_BURN and stat_name == "atk":
        stat = math.floor(stat / 2)
    return _clamp_int(stat, 1, 999)


def calc_damage_gen1(
    battle: np.ndarray,
    atk_side: int,
    move_idx: int,
    player_moves: np.ndarray,
    opp_moves: np.ndarray,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    is_moving_last: bool = False,
    override_move_id: int = -1,
    gen5_prng=None,
    out_meta: dict | None = None,
    target_hurt_this_turn: bool = False,
    target_newly_switched: bool = False,
    user_hurt_by_target_this_turn: bool = False,
    suppress_attacker_item: bool = False,
    suppress_attacker_boosts: bool = False,
    override_field_atk_ability: int = -1,
    override_field_def_ability: int = -1,
    profile: GenProfile = GEN1_PROFILE,
) -> int:
    """Returns damage as a Python int using gen1 PRNG and formula."""
    if gen5_prng is None:
        gen5_prng = Gen5PRNG((1, 1, 1, 1))

    atk_side = int(atk_side)
    move_idx = int(move_idx)
    def_side = 1 - atk_side

    atk_active = int(battle[OFF_META + (0 if atk_side == 0 else 1)])
    def_active = int(battle[OFF_META + (1 if atk_side == 0 else 0)])

    atk_base_off = OFF_SIDE0 if atk_side == 0 else OFF_SIDE1
    def_base_off = OFF_SIDE0 if def_side == 0 else OFF_SIDE1
    atk_offset = atk_base_off + atk_active * POKEMON_SIZE
    def_offset = def_base_off + def_active * POKEMON_SIZE

    if override_move_id is not None and override_move_id >= 0:
        move_id = int(override_move_id)
    else:
        moves = player_moves if atk_side == 0 else opp_moves
        move_id = int(moves[atk_active, move_idx])
    move_id = max(0, min(int(game_data.move_base_power.shape[0]) - 1, move_id))

    bp = int(game_data.move_base_power[move_id])
    if bp <= 0:
        if out_meta is not None:
            out_meta["num_hits"] = 1
        return 0

    move_type = int(game_data.move_type[move_id])
    move_cat = int(game_data.move_category[move_id])
    if profile.phys_spec_mode == "type":
        is_physical = bool(np.isin(move_type, GEN1_PHYSICAL_TYPES))
    else:
        is_physical = move_cat == CAT_PHYSICAL

    def_types = int(battle[def_offset + 4]) & 0xFFFF
    def_type1 = def_types & 0xFF
    def_type2 = (def_types >> 8) & 0xFF
    def_types_for_eff = [def_type1]
    if def_type2 not in (0, def_type1):
        def_types_for_eff.append(def_type2)

    for def_type in def_types_for_eff:
        if float(type_chart[def_type, move_type]) <= 0.0:
            if out_meta is not None:
                out_meta["num_hits"] = 1
            return 0

    accuracy = int(game_data.move_accuracy[move_id])
    if accuracy > 0 and accuracy < 999:
        acc_val = math.floor(accuracy * 255 / 100)
        atk_boosts = int(battle[atk_offset + 14])
        def_boosts = int(battle[def_offset + 14])
        acc_stage = _clamp_int(extract_boost(atk_boosts, 4), -6, 6)
        eva_stage = _clamp_int(extract_boost(def_boosts, 8), -6, 6)
        if not suppress_attacker_boosts:
            acc_val = math.floor(
                acc_val * _GEN1_ACC_TABLE[acc_stage + 6] / 100
            )
        acc_val = math.floor(
            acc_val * _GEN1_ACC_TABLE[-eva_stage + 6] / 100
        )
        acc_val = _clamp_int(acc_val, 1, 255)
        if not _random_chance(gen5_prng, acc_val, 256):
            if out_meta is not None:
                out_meta["num_hits"] = 1
            return 0

    species_id = int(battle[atk_offset + 0])
    base_spe = int(game_data.species_base_stats[species_id][5])
    crit_chance = math.floor(base_spe / 2)
    crit_chance = _clamp_int(crit_chance * 2, 1, 255)
    crit_ratio = int(game_data.move_crit_ratio[move_id])
    if crit_ratio == 1:
        crit_chance = math.floor(crit_chance / 2)
    elif crit_ratio == 2:
        crit_chance = _clamp_int(crit_chance * 4, 1, 255)
    is_crit = _random_chance(gen5_prng, crit_chance, 256)

    atk_status = get_status(int(battle[atk_offset + 12]))
    def_status = get_status(int(battle[def_offset + 12]))
    atk_boosts13 = int(battle[atk_offset + 13])
    def_boosts13 = int(battle[def_offset + 13])

    if is_physical:
        atk_stat_base = int(battle[atk_offset + 7])
        def_stat_base = int(battle[def_offset + 8])
        atk_boost = extract_boost(atk_boosts13, 0)
        def_boost = extract_boost(def_boosts13, 4)
        atk_name, def_name = "atk", "def"
    else:
        atk_stat_base = int(battle[atk_offset + 9])
        def_stat_base = int(battle[def_offset + 10])
        atk_boost = extract_boost(atk_boosts13, 8)
        def_boost = extract_boost(def_boosts13, 12)
        atk_name, def_name = "spa", "spd"

    if is_crit:
        attack = int(atk_stat_base)
        defense = int(def_stat_base)
    else:
        if suppress_attacker_boosts:
            atk_boost = 0
        attack = _gen1_stat(
            atk_stat_base, atk_boost, status=atk_status, stat_name=atk_name
        )
        defense = _gen1_stat(
            def_stat_base, def_boost, status=def_status, stat_name=def_name
        )

    if defense <= 0:
        defense = 1

    if attack >= 256 or defense >= 256:
        attack = _clamp_int(math.floor(attack / 4) % 256, 1, 255)
        defense = _clamp_int(math.floor(defense / 4) % 256, 1, 255)

    level = int(battle[atk_offset + 3])
    if is_crit:
        level *= 2

    damage = level * 2
    damage = math.floor(damage / 5)
    damage += 2
    damage *= bp
    damage *= attack
    damage = math.floor(damage / defense)
    damage = _clamp_int(math.floor(damage / 50), 0, 997)
    damage += 2

    atk_types = int(battle[atk_offset + 4]) & 0xFFFF
    atk_type1 = atk_types & 0xFF
    atk_type2 = (atk_types >> 8) & 0xFF
    has_stab = move_type != TYPE_UNKNOWN and (
        move_type == atk_type1 or move_type == atk_type2
    )
    if has_stab:
        damage += math.floor(damage / 2)

    for def_type in def_types_for_eff:
        eff = float(type_chart[def_type, move_type])
        if eff > 1.0:
            damage *= 20
            damage = math.floor(damage / 10)
        elif eff < 1.0:
            damage *= 5
            damage = math.floor(damage / 10)

    if damage == 0:
        if out_meta is not None:
            out_meta["num_hits"] = 1
        return 0

    if damage > 1:
        damage *= int(gen5_prng.random(217, 256))
        damage = math.floor(damage / 255)

    if out_meta is not None:
        out_meta["num_hits"] = 1
    return int(damage)
