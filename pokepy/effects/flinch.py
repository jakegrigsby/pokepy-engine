"""Flinch handling and per-turn volatile clearing.

Port of MultiFormatFastEnv._apply_flinch_from_move / _check_flinched /
_clear_volatile_turn_effects (the Showdown reference implementation
lines 10148-10233).
"""
from __future__ import annotations

from pokepy.effects._common import np, MultiFormatState, Gen5PRNG
from pokepy.core.bitpack import (
    clear_volatile_turn_effects as _bitpack_clear_volatile_turn_effects,
    get_flinched,
    set_flinched,
)
from pokepy.core.constants import (
    ABILITY_SERENE_GRACE,
    ABILITY_SHEER_FORCE,
    ABILITY_INFILTRATOR,
    ABILITY_INNER_FOCUS,
    F_SUBSTITUTE_0,
    F_SUBSTITUTE_1,
    F_VOLATILE_0,
    F_VOLATILE_1,
    FLAG_SOUND,
    M_ACTIVE0,
    M_ACTIVE1,
    OFF_FIELD,
    OFF_META,
    OFF_SIDE0,
    OFF_SIDE1,
    POKEMON_SIZE,
    VOLATILE_FLINCH,
)

ABILITY_SHIELD_DUST = 19
ITEM_COVERT_CLOAK = 750

def _to_int16(val: int) -> int:
    val = int(val) & 0xFFFF
    if val >= 0x8000:
        val -= 0x10000
    return val

def apply_flinch_from_move(
    battle: np.ndarray,
    move_id: int,
    target_side: int,
    hit: bool,
    move_effects,
    gen5_prng: Gen5PRNG,
    game_data=None,
    num_hits: int = 1,
    prerolled_rolls: "list[int] | None" = None,
) -> None:
    """Port of _apply_flinch_from_move (line ~10148).

    Sets the target's flinch volatile bit if the move is a flinch move,
    hit, the volatile chance roll succeeds, and the attacker doesn't have
    Sheer Force (which suppresses secondary effects).

    Showdown also blocks flinch via Inner Focus, Shield Dust, Covert Cloak,
    and substitute (sub absorbs the secondary unless attacker is Sound /
    Infiltrator). The PRNG is only advanced when the move can actually
    flinch (matches Showdown's behavior of only calling random() when needed).
    """
    move_id = int(move_id)
    target_side = int(target_side)
    hit = bool(hit)

    volatile_type = int(move_effects.volatile[move_id])
    volatile_chance = int(move_effects.volatile_chance[move_id])
    is_flinch_move = volatile_type == VOLATILE_FLINCH

    # Determine attacker's offset (opposite side) for Sheer Force check.
    atk_side = 1 - target_side
    atk_active = int(
        battle[OFF_META + (M_ACTIVE0 if atk_side == 0 else M_ACTIVE1)]
    )
    atk_base = OFF_SIDE0 if atk_side == 0 else OFF_SIDE1
    atk_off = atk_base + atk_active * POKEMON_SIZE
    atk_ability = int(battle[atk_off + 5])
    has_sheer_force = atk_ability == ABILITY_SHEER_FORCE

    # Target side: Inner Focus / Shield Dust / Covert Cloak / Substitute block
    target_base = OFF_SIDE0 if target_side == 0 else OFF_SIDE1
    target_active = int(
        battle[OFF_META + (M_ACTIVE0 if target_side == 0 else M_ACTIVE1)]
    )
    target_off = target_base + target_active * POKEMON_SIZE
    target_ability = int(battle[target_off + 5])
    target_item = int(battle[target_off + 6])
    has_inner_focus = target_ability == ABILITY_INNER_FOCUS
    has_shield_dust = target_ability == ABILITY_SHIELD_DUST
    has_covert_cloak = target_item == ITEM_COVERT_CLOAK

    # Substitute on the target absorbs secondaries unless the move is sound
    # or attacker has Infiltrator.
    sub_offset = OFF_FIELD + (F_SUBSTITUTE_0 if target_side == 0 else F_SUBSTITUTE_1)
    target_has_sub = int(battle[sub_offset]) > 0
    is_infiltrator = atk_ability == ABILITY_INFILTRATOR
    is_sound_move = False
    if game_data is not None:
        try:
            is_sound_move = (int(game_data.move_flags[move_id]) & FLAG_SOUND) != 0
        except Exception:
            is_sound_move = False
    sub_blocks = target_has_sub and not is_infiltrator and not is_sound_move

    # Showdown sim/battle-actions.ts:1357 calls `secondaryRoll = random(100)`
    # for EVERY secondary that survives `runEvent('ModifySecondaries')`. Of
    # the blockers below, only Shield Dust and Covert Cloak filter via
    # onModifySecondaries (data/abilities.ts:4150, data/items.ts:1207), so
    # they suppress the PRNG call. Inner Focus blocks via onTryAddVolatile
    # INSIDE moveHit (after the random call), and Substitute marks the target
    # as null but the secondaries loop still rolls per surviving secondary —
    # so those still consume a frame. Sheer Force removes secondaries via
    # onModifyMove → no random call.
    secondary_filtered = has_sheer_force or has_shield_dust or has_covert_cloak
    if not (is_flinch_move and hit and (volatile_chance > 0)) or secondary_filtered:
        return

    # Serene Grace (Showdown data/abilities.ts:serenegrace) doubles secondary
    # effect chance (capped at 100). This applies to flinch secondaries too.
    effective_chance = volatile_chance
    if atk_ability == ABILITY_SERENE_GRACE:
        effective_chance = min(100, volatile_chance * 2)

    # Multi-hit moves (Double Iron Bash etc.) roll the secondary chance
    # per hit (Showdown sim/battle-actions.ts:1357 secondaryRoll inside
    # the per-hit moveHit loop). Each hit independently triggers the
    # flinch volatile; success on any hit flinches the target.
    #
    # If `prerolled_rolls` is provided, use those values instead of
    # consuming PRNG frames here (matches Showdown's per-move secondaryRoll
    # ordering — see battle_gen9.py preroll logic).
    n = max(1, int(num_hits))
    flinch_success = False
    for i in range(n):
        if prerolled_rolls is not None and i < len(prerolled_rolls):
            roll = int(prerolled_rolls[i])
        else:
            roll = gen5_prng.random(100)
        if roll < effective_chance:
            flinch_success = True
    # Inner Focus / Substitute block the volatile AFTER the roll fired.
    if has_inner_focus or sub_blocks:
        flinch_success = False

    volatile_offset = OFF_FIELD + (F_VOLATILE_0 if target_side == 0 else F_VOLATILE_1)
    current_volatile = int(battle[volatile_offset])
    new_volatile = set_flinched(current_volatile, flinch_success)
    battle[volatile_offset] = _to_int16(new_volatile)

def check_flinched(battle: np.ndarray, side: int) -> bool:
    """Port of _check_flinched (line ~10202)."""
    side = int(side)
    volatile_offset = OFF_FIELD + (F_VOLATILE_0 if side == 0 else F_VOLATILE_1)
    return bool(get_flinched(int(battle[volatile_offset])))

def clear_volatile_turn_effects(battle: np.ndarray) -> None:
    """Port of _clear_volatile_turn_effects (line ~10220).

    Clears per-turn volatile effects (flinch) at end of turn for both sides.
    """
    for off in (OFF_FIELD + F_VOLATILE_0, OFF_FIELD + F_VOLATILE_1):
        cur = int(battle[off])
        battle[off] = _to_int16(_bitpack_clear_volatile_turn_effects(cur))
