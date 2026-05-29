"""Niche special-move handlers.

Hosts handlers that don't fit the main dispatch tables:
  - Burning Jealousy (secondary burn if target has any positive boost)
  - Strength Sap (heal by target's atk, then -1 atk)
  - Belly Drum (fail if hp <= 50% or atk already +6, else -50% HP + set atk to +6)
  - Skill Swap (swap abilities between user and target)

Data for each handler cross-checked against data/moves.ts in the
vendored Pokemon Showdown tree.
"""

from __future__ import annotations

import numpy as np

from pokepy.core.constants import (
    STATUS_NONE,
    STATUS_BURN,
    TYPE_FIRE,
    TYPE_ELECTRIC,
    TYPE_UNKNOWN,
    ABILITY_WATER_VEIL,
    ABILITY_WATER_BUBBLE,
    ABILITY_THERMAL_EXCHANGE,
)
from pokepy.core.bitpack import (
    extract_boost,
    apply_boost_to_packed,
    get_status,
    set_status,
)
from pokepy.mechanics.stats import get_boost_multiplier

MOVE_BURNING_JEALOUSY = 807
MOVE_STRENGTH_SAP = 668
MOVE_BELLY_DRUM = 187
MOVE_FILLET_AWAY = 868  # also pays 50% HP but handled independently
MOVE_SKILL_SWAP = 285
MOVE_BURN_UP = 682
MOVE_DOUBLE_SHOCK = 892


def _to_int16(val: int) -> int:
    val = int(val) & 0xFFFF
    if val >= 0x8000:
        val -= 0x10000
    return val


def apply_misc_move_effects(
    battle: np.ndarray,
    move_id0: int,
    move_id1: int,
    user0_off: int,
    user1_off: int,
    target0_off: int,
    target1_off: int,
    hit0: bool,
    hit1: bool,
) -> None:
    pairs = [
        (int(move_id0), int(user0_off), int(target0_off), bool(hit0)),
        (int(move_id1), int(user1_off), int(target1_off), bool(hit1)),
    ]

    # ----- Burning Jealousy: burn target if they have any positive boost -----
    for mid, _user, target_off, did_hit in pairs:
        if mid != MOVE_BURNING_JEALOUSY or not did_hit:
            continue
        b13 = int(battle[target_off + 13])
        b14 = int(battle[target_off + 14])
        any_pos = (
            extract_boost(b13, 0) > 0
            or extract_boost(b13, 4) > 0
            or extract_boost(b13, 8) > 0
            or extract_boost(b13, 12) > 0
            or extract_boost(b14, 0) > 0
        )
        if not any_pos:
            continue
        cur_status = get_status(int(battle[target_off + 12]))
        target_types = int(battle[target_off + 4]) & 0xFFFF
        target_t1 = target_types & 0xFF
        target_t2 = (target_types >> 8) & 0xFF
        target_ability = int(battle[target_off + 5])
        burn_blocking_ability = target_ability in (
            ABILITY_WATER_VEIL,
            ABILITY_WATER_BUBBLE,
            ABILITY_THERMAL_EXCHANGE,
        )
        is_fire = target_t1 == TYPE_FIRE or target_t2 == TYPE_FIRE
        if cur_status == STATUS_NONE and not burn_blocking_ability and not is_fire:
            battle[target_off + 12] = _to_int16(set_status(STATUS_BURN, 0))

    # ----- Strength Sap: heal user by target's effective Atk, then -1 Atk -----
    for mid, user_off, target_off, did_hit in pairs:
        if mid != MOVE_STRENGTH_SAP or not did_hit:
            continue
        target_atk_base = int(battle[target_off + 7])
        target_atk_boost = extract_boost(int(battle[target_off + 13]), 0)
        target_atk_eff = int(
            target_atk_base * float(get_boost_multiplier(target_atk_boost))
        )
        user_hp = int(battle[user_off + 1])
        user_max = int(battle[user_off + 2])
        if user_hp > 0:
            new_hp = min(user_hp + target_atk_eff, user_max)
            battle[user_off + 1] = np.int16(new_hp)
        b13 = int(battle[target_off + 13])
        battle[target_off + 13] = _to_int16(apply_boost_to_packed(b13, 0, -1))


def move_missing_required_live_type(
    battle: np.ndarray,
    move_id: int,
    user_off: int,
) -> bool:
    """Showdown onTryMove gate for Burn Up / Double Shock."""
    required_type = None
    if int(move_id) == MOVE_BURN_UP:
        required_type = TYPE_FIRE
    elif int(move_id) == MOVE_DOUBLE_SHOCK:
        required_type = TYPE_ELECTRIC
    if required_type is None:
        return False

    types = int(battle[int(user_off) + 4]) & 0xFFFF
    type1 = types & 0xFF
    type2 = (types >> 8) & 0xFF
    return type1 != required_type and type2 != required_type


def apply_self_type_removal_from_move(
    battle: np.ndarray,
    move_id: int,
    user_off: int,
    hit: bool,
) -> bool:
    """Showdown self.onHit for Burn Up / Double Shock."""
    if not bool(hit):
        return False

    removed_type = None
    if int(move_id) == MOVE_BURN_UP:
        removed_type = TYPE_FIRE
    elif int(move_id) == MOVE_DOUBLE_SHOCK:
        removed_type = TYPE_ELECTRIC
    if removed_type is None:
        return False

    poff = int(user_off)
    types = int(battle[poff + 4]) & 0xFFFF
    type1 = types & 0xFF
    type2 = (types >> 8) & 0xFF
    if type1 != removed_type and type2 != removed_type:
        return False

    new_type1 = TYPE_UNKNOWN if type1 == removed_type else type1
    new_type2 = TYPE_UNKNOWN if type2 == removed_type else type2
    battle[poff + 4] = np.int16(_to_int16(new_type1 | (new_type2 << 8)))
    return True


def apply_belly_drum_from_move(
    battle: np.ndarray,
    move_id: int,
    user_off: int,
    hit: bool,
) -> bool:
    """Belly Drum — Showdown data/moves.ts:bellydrum onHit.

    Fail conditions:
      - User HP <= maxhp / 2 (can't pay cost)
      - User atk boost already >= +6
      - Shedinja clause (maxhp == 1)

    On success: user loses 50% max HP (directDamage), atk is set to +6.
    The upstream stat-change path tries to +12 the user's atk (caps to
    +6) regardless of fail conditions; this function PAYS the HP cost
    only when the move actually succeeds and returns True so the caller
    can undo the stat-change if it failed.

    Returns True if Belly Drum succeeded (and HP cost was paid), False
    otherwise. The caller must read the return to decide whether to
    revert the +12 atk boost that the generic stat-change pipeline
    already applied.
    """
    if not bool(hit) or int(move_id) != MOVE_BELLY_DRUM:
        return False
    uoff = int(user_off)
    hp = int(battle[uoff + 1])
    max_hp = int(battle[uoff + 2])
    if hp <= 0 or max_hp <= 1:
        return False
    # Fail if HP is at exactly 50% or less (Showdown: `hp <= maxhp / 2`).
    # Use exact arithmetic: hp * 2 <= max_hp.
    if hp * 2 <= max_hp:
        return False
    # Fail if atk boost already at +6.
    atk_boost = extract_boost(int(battle[uoff + 13]), 0)
    if atk_boost >= 6:
        return False
    cost = max_hp // 2
    battle[uoff + 1] = np.int16(max(0, hp - cost))
    return True


def apply_skill_swap_from_move(
    battle: np.ndarray,
    move_id: int,
    user_off: int,
    target_off: int,
    hit: bool,
) -> None:
    """Skill Swap — Showdown data/moves.ts:skillswap onHit.

    Simple ability swap. Showdown forbids swapping certain abilities
    (Wonder Guard, Multitype, RKS System, Stance Change, Schooling,
    Comatose, Shields Down, Disguise, Ice Face, Zen Mode, Power
    Construct, Protosynthesis, Quark Drive, As One, Battle Bond, Gulp
    Missile, Hunger Switch, Ice Face, Illusion, Neutralizing Gas,
    Trace). Pokepy uses a conservative list of the common-OU-relevant
    ones and otherwise swaps.
    """
    if not bool(hit) or int(move_id) != MOVE_SKILL_SWAP:
        return
    uoff = int(user_off)
    toff = int(target_off)
    # Skip if either is fainted
    if int(battle[uoff + 1]) <= 0 or int(battle[toff + 1]) <= 0:
        return
    # Abilities disallowed per Showdown sim/pokemon.ts flags + skillswap
    # onTryHit: Wonder Guard, Multitype, Stance Change, Schooling, Comatose,
    # Shields Down, Disguise, Power Construct, Ice Face, Zen Mode,
    # Illusion, Battle Bond, Gulp Missile, RKS System, Hunger Switch,
    # Neutralizing Gas, As One, Protosynthesis, Quark Drive.
    FORBIDDEN = frozenset(
        (
            25,  # wonder_guard
            121,  # multitype
            176,  # stance_change
            208,  # schooling
            197,  # shields_down
            209,  # disguise
            211,  # power_construct
            248,  # ice_face
            161,  # zen_mode
            149,  # illusion
            210,  # battle_bond
            241,  # gulp_missile
            225,  # rks_system
            258,  # hunger_switch
            259,  # neutralizing_gas
            266,  # as_one_glastrier
            267,  # as_one_spectrier
            281,  # protosynthesis (ABILITY_PROTOSYNTHESIS)
            282,  # quark_drive (ABILITY_QUARK_DRIVE)
        )
    )
    u_ab = int(battle[uoff + 5])
    t_ab = int(battle[toff + 5])
    if u_ab in FORBIDDEN or t_ab in FORBIDDEN:
        return
    # Can't swap if both sides have the same ability.
    if u_ab == t_ab:
        return
    # Ability Shield (item 746) blocks any ability change. Showdown source:
    # data/items.ts:2-20 abilityshield onSetAbility returns null. The shield
    # protects the holder on either side of the swap.
    _ITEM_ABILITY_SHIELD_SS = 746
    if (
        int(battle[uoff + 6]) == _ITEM_ABILITY_SHIELD_SS
        or int(battle[toff + 6]) == _ITEM_ABILITY_SHIELD_SS
    ):
        return
    battle[uoff + 5] = t_ab
    battle[toff + 5] = u_ab
