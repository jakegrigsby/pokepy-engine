"""Protect-family moves and contact effects.

Port of MultiFormatFastEnv._apply_protect_from_move and friends
(the Showdown reference implementation).
"""
from __future__ import annotations

from pokepy.effects._common import np, MultiFormatState, Gen5PRNG
from pokepy.core.bitpack import (
    apply_boost_to_packed,
    clear_protect_active,
    get_protect_active,
    get_protect_consecutive,
    get_protect_type,
    get_status,
    increment_protect_consecutive,
    reset_protect_state,
    set_protect_active,
    set_protect_type,
    set_status,
)
from pokepy.core.constants import (
    EFFECT_PROTECT,
    F_PROTECT_0,
    F_PROTECT_1,
    MOVE_BANEFUL_BUNKER,
    MOVE_BURNING_BULWARK,
    MOVE_KINGS_SHIELD,
    MOVE_OBSTRUCT,
    MOVE_QUICK_GUARD,
    MOVE_SILK_TRAP,
    MOVE_SPIKY_SHIELD,
    OFF_FIELD,
    PROTECT_BANEFUL_BUNKER,
    PROTECT_BASIC,
    PROTECT_BURNING_BULWARK,
    PROTECT_ENDURE,
    PROTECT_KINGS_SHIELD,
    PROTECT_OBSTRUCT,
    PROTECT_QUICK_GUARD,
    PROTECT_SILK_TRAP,
    PROTECT_SPIKY_SHIELD,
    STATUS_NONE,
    STATUS_BURN,
    STATUS_POISON,
    FLAG_CONTACT,
    TYPE_FIRE,
    ABILITY_WATER_VEIL,
    ABILITY_WATER_BUBBLE,
    ABILITY_THERMAL_EXCHANGE,
)

def _to_int16(val: int) -> int:
    val = int(val) & 0xFFFF
    if val >= 0x8000:
        val -= 0x10000
    return val

def _protect_type_for_move(move_id: int) -> int:
    if move_id == 203:  # Endure
        return PROTECT_ENDURE
    if move_id == MOVE_KINGS_SHIELD:
        return PROTECT_KINGS_SHIELD
    if move_id == MOVE_BANEFUL_BUNKER:
        return PROTECT_BANEFUL_BUNKER
    if move_id == MOVE_SILK_TRAP:
        return PROTECT_SILK_TRAP
    if move_id == MOVE_QUICK_GUARD:
        return PROTECT_QUICK_GUARD
    if move_id == MOVE_SPIKY_SHIELD:
        return PROTECT_SPIKY_SHIELD
    if move_id == MOVE_OBSTRUCT:
        return PROTECT_OBSTRUCT
    if move_id == MOVE_BURNING_BULWARK:
        return PROTECT_BURNING_BULWARK
    return PROTECT_BASIC

def apply_protect_from_move(
    battle: np.ndarray,
    move_id: int,
    user_side: int,
    move_effects,
    gen5_prng: Gen5PRNG,
) -> bool:
    """Port of _apply_protect_from_move (line ~9714).

    Returns True iff protect activated successfully. Mutates `battle`
    in place. Consecutive uses make protect progressively likely to fail
    (1 / 3^consecutive).
    """
    move_id = int(move_id)
    user_side = int(user_side)

    effect_type = int(move_effects.effect_type[move_id])
    is_protect = effect_type == EFFECT_PROTECT
    if not is_protect:
        return False

    protect_type = _protect_type_for_move(move_id)

    protect_offset = OFF_FIELD + (F_PROTECT_0 if user_side == 0 else F_PROTECT_1)
    protect_state = int(battle[protect_offset])

    consecutive = get_protect_consecutive(protect_state)

    # Showdown: Protect's `onPrepareHit` calls `runEvent('StallMove')`. The
    # ONLY handler is `conditions.ts:stall.onStallMove`, which calls
    # `randomChance(1, counter)` and only exists while the stall volatile is
    # up. On the FIRST consecutive Protect (consecutive == 0) the stall
    # volatile doesn't exist yet, so runEvent returns true with NO random
    # frame consumed. Subsequent Protects run randomChance(1, 3^consecutive).
    # Skipping the first-use roll keeps the PRNG aligned with Showdown.
    if consecutive == 0:
        success = True
    else:
        denom = 1
        for _ in range(int(consecutive)):
            denom *= 3
        success = gen5_prng.random_chance(1, denom)

    if success:
        new_state = increment_protect_consecutive(protect_state)
        new_state = set_protect_active(new_state, True)
        new_state = set_protect_type(new_state, protect_type)
        battle[protect_offset] = _to_int16(new_state)

    return bool(success)

def check_protected(battle: np.ndarray, target_side: int) -> bool:
    """Port of _check_protected (line ~9776).

    Returns True iff target's protect is active and the protect type is
    neither Quick Guard nor Endure. Quick Guard blocks only priority moves,
    and Endure is a stalling-move volatile that preserves 1 HP without
    blocking the incoming hit.
    """
    target_side = int(target_side)
    protect_offset = OFF_FIELD + (F_PROTECT_0 if target_side == 0 else F_PROTECT_1)
    protect_state = int(battle[protect_offset])
    is_active = get_protect_active(protect_state) > 0
    protect_type = get_protect_type(protect_state)
    return bool(is_active and protect_type not in (PROTECT_QUICK_GUARD, PROTECT_ENDURE))

def check_protected_with_type(battle: np.ndarray, target_side: int):
    """Port of _check_protected_with_type (line ~9796).

    Returns (is_active, protect_type).
    """
    target_side = int(target_side)
    protect_offset = OFF_FIELD + (F_PROTECT_0 if target_side == 0 else F_PROTECT_1)
    protect_state = int(battle[protect_offset])
    is_active = get_protect_active(protect_state) > 0
    protect_type = get_protect_type(protect_state)
    return bool(is_active), int(protect_type)

def clear_protect_at_turn_end(battle: np.ndarray) -> None:
    """Port of _clear_protect_at_turn_end (line ~9812).

    Clears the active flag for both sides. The consecutive counter is
    only reset by `reset_protect_if_not_used` when protect was not used.
    """
    for off in (OFF_FIELD + F_PROTECT_0, OFF_FIELD + F_PROTECT_1):
        cur = int(battle[off])
        battle[off] = _to_int16(clear_protect_active(cur))

def apply_protect_contact_effects(
    battle: np.ndarray,
    move_id: int,
    attacker_offset: int,
    defender_side: int,
    was_blocked: bool,
    game_data,
    move_effects=None,
) -> None:
    """Port of _apply_protect_contact_effects (line ~9827).

    Applies the per-protect-variant on-contact side effects:
      King's Shield  -> -1 Atk
      Baneful Bunker -> Poison
      Silk Trap      -> -1 Spe
      Spiky Shield   -> 1/8 max HP damage
      Obstruct       -> -2 Def
    Effects only fire when (was_blocked and the blocked move makes contact).
    """
    if not bool(was_blocked) or game_data is None:
        return
    move_id = int(move_id)
    attacker_offset = int(attacker_offset)
    defender_side = int(defender_side)

    is_contact = (int(game_data.move_flags[move_id]) & FLAG_CONTACT) != 0
    if not is_contact:
        return

    # Protective Pads: makes checkMoveMakesContact() return false, so
    # all contact-triggered protect side effects (Baneful Bunker poison,
    # Spiky Shield damage, Obstruct def drop, King's Shield atk drop,
    # Silk Trap spe drop, Burning Bulwark burn) are blocked.
    # Showdown sim/battle.ts:1290 checkMoveMakesContact.
    # Long Reach (ability) has the same effect — it makes the move
    # non-contact from the attacker's perspective.
    # Punching Glove strips the contact flag from PUNCH moves
    # (data/items.ts:4604 `delete move.flags['contact']`).
    from pokepy.core.constants import ABILITY_LONG_REACH as _ABILITY_LONG_REACH, FLAG_PUNCH as _FLAG_PUNCH
    _ITEM_PROTECTIVE_PADS = 663
    _ITEM_PUNCHING_GLOVE = 749
    atk_item_pp = int(battle[int(attacker_offset) + 6])
    atk_ab_pp = int(battle[int(attacker_offset) + 5])
    is_punch = (int(game_data.move_flags[move_id]) & _FLAG_PUNCH) != 0
    if (
        atk_item_pp == _ITEM_PROTECTIVE_PADS
        or atk_ab_pp == _ABILITY_LONG_REACH
        or (is_punch and atk_item_pp == _ITEM_PUNCHING_GLOVE)
    ):
        return

    protect_offset = OFF_FIELD + (F_PROTECT_0 if defender_side == 0 else F_PROTECT_1)
    protect_state = int(battle[protect_offset])
    protect_type = get_protect_type(protect_state)

    # Stat-drop contact effects (King's Shield -1 atk, Silk Trap -1 spe,
    # Obstruct -2 def) must honor the attacker's boost immunity/reaction
    # abilities just like any other opponent-triggered stat drop. Showdown
    # routes these through `this.boost(...)` which calls the full onTryBoost
    # / onAfterBoost chain. Pokepy used to apply them directly, which meant
    # Clear Body / White Smoke / Full Metal Body didn't block them, Clear
    # Amulet didn't block them, Contrary didn't invert them, Simple didn't
    # double them, and Defiant / Competitive didn't counter-boost atk/spa.
    from pokepy.core.constants import (
        ABILITY_CLEAR_BODY as _AB_CB_P,
        ABILITY_WHITE_SMOKE as _AB_WS_P,
        ABILITY_FULL_METAL_BODY as _AB_FMB_P,
        ABILITY_CONTRARY as _AB_CO_P,
        ABILITY_SIMPLE as _AB_SI_P,
        ABILITY_DEFIANT as _AB_DE_P,
        ABILITY_COMPETITIVE as _AB_CM_P,
        ABILITY_MIRROR_ARMOR as _AB_MA_P,
    )
    _ITEM_CA_P = 747  # Clear Amulet (matches other effects modules)
    atk_ab_prt = int(battle[int(attacker_offset) + 5])
    atk_item_prt = int(battle[int(attacker_offset) + 6])
    atk_blocks_drops = (
        atk_ab_prt in (_AB_CB_P, _AB_WS_P, _AB_FMB_P)
        or atk_item_prt == _ITEM_CA_P
    )
    atk_has_contrary = atk_ab_prt == _AB_CO_P
    atk_has_simple = atk_ab_prt == _AB_SI_P
    atk_has_defiant = atk_ab_prt == _AB_DE_P
    atk_has_competitive = atk_ab_prt == _AB_CM_P
    # Hyper Cutter blocks Atk drops, Big Pecks blocks Def drops, Keen Eye
    # and Mind's Eye block Acc drops. The protect effects only drop atk/
    # def/spe so we only need Hyper Cutter + Big Pecks here.
    _AB_HC_P = 110
    _AB_BP_P = 145
    atk_has_hyper_cutter = atk_ab_prt == _AB_HC_P
    atk_has_big_pecks = atk_ab_prt == _AB_BP_P

    def _apply_drop(idx: int, shift_off: int, shift_bit: int, delta: int):
        # idx: 0=atk, 1=def, 2=spa, 3=spd, 4=spe
        # If blocked, don't apply. Track whether a drop actually took.
        if atk_blocks_drops:
            return False
        if idx == 0 and atk_has_hyper_cutter:
            return False
        if idx == 1 and atk_has_big_pecks:
            return False
        actual = delta
        if atk_has_contrary:
            actual = -actual
        if atk_has_simple:
            actual = actual * 2
        off = int(attacker_offset) + shift_off
        battle[off] = apply_boost_to_packed(int(battle[off]), shift_bit, actual)
        return actual < 0

    any_drop = False

    # King's Shield: -1 Atk on contact
    if protect_type == PROTECT_KINGS_SHIELD:
        if _apply_drop(0, 13, 0, -1):
            any_drop = True

    # Baneful Bunker: poison on contact
    if protect_type == PROTECT_BANEFUL_BUNKER:
        atk_status_off = attacker_offset + 12
        atk_status = int(battle[atk_status_off])
        cur_status = get_status(atk_status)
        if cur_status == STATUS_NONE:
            battle[atk_status_off] = set_status(STATUS_POISON, 0)

    # Silk Trap: -1 Spe on contact (Spe lives at offset+14, shift 0)
    if protect_type == PROTECT_SILK_TRAP:
        if _apply_drop(4, 14, 0, -1):
            any_drop = True

    # Spiky Shield: 1/8 max HP damage on contact (min 1)
    if protect_type == PROTECT_SPIKY_SHIELD:
        atk_hp_off = attacker_offset + 1
        atk_max_hp_off = attacker_offset + 2
        atk_hp = int(battle[atk_hp_off])
        atk_max_hp = int(battle[atk_max_hp_off])
        # Showdown damage(maxhp/8) floors.
        spiky_damage = max(atk_max_hp // 8, 1)
        battle[atk_hp_off] = max(0, atk_hp - spiky_damage)

    # Obstruct: -2 Def on contact (Def lives at offset+13, shift 4)
    if protect_type == PROTECT_OBSTRUCT:
        if _apply_drop(1, 13, 4, -2):
            any_drop = True

    # Defiant / Competitive counter-boost from a stat drop by the opponent's
    # protect effect.
    if any_drop:
        if atk_has_defiant:
            atk_b13_off = int(attacker_offset) + 13
            battle[atk_b13_off] = apply_boost_to_packed(int(battle[atk_b13_off]), 0, 2)
        if atk_has_competitive:
            atk_b13_off = int(attacker_offset) + 13
            battle[atk_b13_off] = apply_boost_to_packed(int(battle[atk_b13_off]), 8, 2)

    # Burning Bulwark: burn the attacker on contact. Showdown
    # data/moves.ts:burningbulwark `source.trySetStatus('brn', target)`.
    # trySetStatus respects type immunities (Fire types can't be burned).
    if protect_type == PROTECT_BURNING_BULWARK:
        atk_status_off = attacker_offset + 12
        atk_status = int(battle[atk_status_off])
        cur_status = get_status(atk_status)
        # Fire-type attackers are immune to burn
        atk_types_packed = int(battle[attacker_offset + 4])
        atk_t1 = atk_types_packed & 0xFF
        atk_t2 = (atk_types_packed >> 8) & 0xFF
        is_fire = (atk_t1 == TYPE_FIRE) or (atk_t2 == TYPE_FIRE)
        atk_ability = int(battle[attacker_offset + 5])
        burn_blocking_ability = atk_ability in (
            ABILITY_WATER_VEIL,
            ABILITY_WATER_BUBBLE,
            ABILITY_THERMAL_EXCHANGE,
        )
        if cur_status == STATUS_NONE and not is_fire and not burn_blocking_ability:
            battle[atk_status_off] = set_status(STATUS_BURN, 0)

def reset_protect_if_not_used(
    battle: np.ndarray,
    user_side: int,
    used_protect: bool,
) -> None:
    """Port of _reset_protect_if_not_used (line ~9916).

    Resets the consecutive counter (and active/type) if protect was NOT
    used this turn.
    """
    user_side = int(user_side)
    protect_offset = OFF_FIELD + (F_PROTECT_0 if user_side == 0 else F_PROTECT_1)
    protect_state = int(battle[protect_offset])
    if not bool(used_protect):
        battle[protect_offset] = _to_int16(reset_protect_state(protect_state))
