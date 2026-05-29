"""End-of-turn effect orchestration + decrement helpers.

Pokepy uses a single-battle scalar engine: every helper mutates `battle`
in place and takes a stateful `Gen5PRNG`. The orchestrator wires together
status, weather, salt cure, speed boost, weather healing, shed skin /
hydration, and the trick room / screens / weather / terrain decrements.
"""

from __future__ import annotations

from pokepy.effects._common import np, MultiFormatState, Gen5PRNG
from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_FIELD,
    OFF_META,
    OFF_MOVES,
    POKEMON_SIZE,
    F_TERRAIN,
    F_TRICK_ROOM,
    F_WEATHER,
    F_SCREENS_0,
    F_SCREENS_1,
    F_EXTENDED_VOLATILE_0,
    F_EXTENDED_VOLATILE_1,
    M_WEATHER_TURNS,
    M_TERRAIN_TURNS,
    WEATHER_NONE,
    SCREEN_REFLECT_SHIFT,
    SCREEN_LIGHTSCREEN_SHIFT,
    SCREEN_AURORAVEIL_SHIFT,
    SCREEN_TAILWIND_SHIFT,
    SCREEN_SAFEGUARD_SHIFT,
    SCREEN_MIST_SHIFT,
    SCREEN_MASK_2BIT,
    SCREEN_MASK_3BIT,
    screen_clear_mask,
)

# Pre-computed (shift, mask, clear_mask) tuples for screen decrement.
_SCREEN_FIELDS = [
    (
        SCREEN_REFLECT_SHIFT,
        SCREEN_MASK_3BIT,
        screen_clear_mask(SCREEN_MASK_3BIT, SCREEN_REFLECT_SHIFT),
    ),
    (
        SCREEN_LIGHTSCREEN_SHIFT,
        SCREEN_MASK_3BIT,
        screen_clear_mask(SCREEN_MASK_3BIT, SCREEN_LIGHTSCREEN_SHIFT),
    ),
    (
        SCREEN_AURORAVEIL_SHIFT,
        SCREEN_MASK_3BIT,
        screen_clear_mask(SCREEN_MASK_3BIT, SCREEN_AURORAVEIL_SHIFT),
    ),
    (
        SCREEN_TAILWIND_SHIFT,
        SCREEN_MASK_3BIT,
        screen_clear_mask(SCREEN_MASK_3BIT, SCREEN_TAILWIND_SHIFT),
    ),
    (
        SCREEN_SAFEGUARD_SHIFT,
        SCREEN_MASK_2BIT,
        screen_clear_mask(SCREEN_MASK_2BIT, SCREEN_SAFEGUARD_SHIFT),
    ),
    (
        SCREEN_MIST_SHIFT,
        SCREEN_MASK_2BIT,
        screen_clear_mask(SCREEN_MASK_2BIT, SCREEN_MIST_SHIFT),
    ),
]


def apply_end_of_turn_effects(
    battle: np.ndarray,
    active0: int,
    active1: int,
    game_data,
    move_effects,
    gen5_prng: Gen5PRNG,
) -> None:
    """Apply all end-of-turn effects for both active Pokemon.

    Port of MultiFormatFastEnv._apply_end_of_turn_effects (line ~7578).
    Mutates `battle` in place.
    """
    # Import from pokepy.effects (the package) so we go through the
    # permissive shim — TODO subagents that haven't ported their bodies
    # yet raise NotImplementedError, which the shim swallows. Local import
    # avoids the circular import (effects.__init__ imports us).
    import pokepy.effects as _fx

    pokemon0_offset = int(OFF_SIDE0 + int(active0) * POKEMON_SIZE)
    pokemon1_offset = int(OFF_SIDE1 + int(active1) * POKEMON_SIZE)

    # Apply status effects to both active Pokemon (only if alive).
    hp0 = int(battle[pokemon0_offset + 1])
    if hp0 > 0:
        _fx.apply_end_of_turn_status(
            battle, pokemon0_offset, game_data, move_effects, gen5_prng
        )
    hp1 = int(battle[pokemon1_offset + 1])
    if hp1 > 0:
        _fx.apply_end_of_turn_status(
            battle, pokemon1_offset, game_data, move_effects, gen5_prng
        )

    # Apply weather damage (sandstorm) — re-check HP after status damage.
    if int(battle[pokemon0_offset + 1]) > 0:
        _fx.apply_weather_damage(battle, pokemon0_offset, game_data)
    if int(battle[pokemon1_offset + 1]) > 0:
        _fx.apply_weather_damage(battle, pokemon1_offset, game_data)

    # Apply Salt Cure damage (1/8 max HP; 1/4 for Water/Steel types).
    if int(battle[pokemon0_offset + 1]) > 0:
        _fx.apply_salt_cure_damage(
            battle,
            pokemon0_offset,
            int(OFF_FIELD + F_EXTENDED_VOLATILE_0),
            game_data,
        )
    if int(battle[pokemon1_offset + 1]) > 0:
        _fx.apply_salt_cure_damage(
            battle,
            pokemon1_offset,
            int(OFF_FIELD + F_EXTENDED_VOLATILE_1),
            game_data,
        )

    # Partially-trapped residual damage (Wrap/Bind/Fire Spin/Whirlpool/
    # Sand Tomb/Magma Storm/Clamp/Snap Trap/Thunder Cage/Infestation).
    # Showdown conditions.ts partiallytrapped onResidualOrder 13:
    # `this.damage(pokemon.baseMaxhp / this.effectState.boundDivisor)`
    # where boundDivisor is 8 normally, 6 if the source holds Binding Band.
    # Pokepy doesn't snapshot the source's item at apply time, so we check
    # the currently-active opponent's item as a best-effort proxy.
    if int(battle[pokemon0_offset + 1]) > 0:
        apply_partial_trap_damage(battle, pokemon0_offset, side=0)
    if int(battle[pokemon1_offset + 1]) > 0:
        apply_partial_trap_damage(battle, pokemon1_offset, side=1)

    # Speed Boost ability: +1 Speed at end of turn.
    _fx.apply_speed_boost(battle, pokemon0_offset, game_data)
    _fx.apply_speed_boost(battle, pokemon1_offset, game_data)

    # Ice Body / Rain Dish: heal 1/16 HP in snow / rain.
    _fx.apply_weather_healing(battle, pokemon0_offset, game_data)
    _fx.apply_weather_healing(battle, pokemon1_offset, game_data)

    # Shed Skin (33% cure status) and Hydration (cure status in rain).
    _fx.apply_shed_skin_hydration(battle, pokemon0_offset, game_data, gen5_prng)
    _fx.apply_shed_skin_hydration(battle, pokemon1_offset, game_data, gen5_prng)

    # Aqua Ring / Ingrain: heal 1/16 max HP at EOT. Showdown residual
    # orders 6 and 7 respectively. Big Root boosts the heal by 1.3x.
    # Heal Block suppresses both.
    _apply_aqua_ring_ingrain_heal(battle, pokemon0_offset, side=0)
    _apply_aqua_ring_ingrain_heal(battle, pokemon1_offset, side=1)

    # Yawn: drowsy → sleep transition.
    _process_yawn(battle, pokemon0_offset, side=0, gen5_prng=gen5_prng)
    _process_yawn(battle, pokemon1_offset, side=1, gen5_prng=gen5_prng)

    # Decrement field-wide turn counters.
    decrement_trick_room(battle)
    decrement_screens(battle)
    decrement_weather(battle)
    decrement_terrain(battle)


def apply_partial_trap_damage(
    battle: np.ndarray,
    pokemon_offset: int,
    side: int,
) -> None:
    """Residual damage for Wrap/Bind/Fire Spin/etc. (conditions.ts
    partiallytrapped onResidual, order 13).

    Base damage is `max_hp / 8`; `max_hp / 6` if the opposing active
    mon is holding Binding Band. Magic Guard prevents the damage.
    The stored duration mirrors Showdown's partiallytrapped condition:
    the callback sets 5/6 turns (or 8 with Grip Claw), residual
    decrements first, and the last countdown step ends the condition
    without dealing damage.

    Note: battle_gen9.py must call this from its residual-order-13 slot
    to actually deal damage. The orchestrator in `apply_end_of_turn_effects`
    also invokes it but that function is not the engine's real EOT loop.
    """
    from pokepy.core.constants import (
        EXT_VOL_PARTIAL_TRAP,
        ITEM_BINDING_BAND,
        ABILITY_MAGIC_GUARD,
        POKEMON_SIZE,
        M_ACTIVE0,
        M_ACTIVE1,
        M_PARTIAL_TRAP_TURNS_0,
        M_PARTIAL_TRAP_TURNS_1,
    )

    poff = int(pokemon_offset)
    hp = int(battle[poff + 1])
    if hp <= 0:
        return

    ext_off = OFF_FIELD + (
        F_EXTENDED_VOLATILE_0 if side == 0 else F_EXTENDED_VOLATILE_1
    )
    ext_vol = int(battle[ext_off]) & 0xFFFF
    if (ext_vol & EXT_VOL_PARTIAL_TRAP) == 0:
        return

    def _wrap_i16(v: int) -> int:
        v = int(v) & 0xFFFF
        return v - 0x10000 if v >= 0x8000 else v

    turns_off = OFF_MOVES + (
        M_PARTIAL_TRAP_TURNS_0 if side == 0 else M_PARTIAL_TRAP_TURNS_1
    )
    turns = int(battle[turns_off])
    if turns <= 0:
        battle[ext_off] = np.int16(_wrap_i16(ext_vol & ~EXT_VOL_PARTIAL_TRAP))
        return

    new_turns = max(0, turns - 1)
    battle[turns_off] = np.int16(new_turns)
    if new_turns == 0:
        battle[ext_off] = np.int16(_wrap_i16(ext_vol & ~EXT_VOL_PARTIAL_TRAP))
        return

    # Magic Guard prevents all indirect damage.
    ability = int(battle[poff + 5])
    if ability == ABILITY_MAGIC_GUARD:
        return

    # Locate the opposing active mon to check for Binding Band. Showdown ends
    # standard partial trapping immediately if the trapping source is no
    # longer active or has fainted; pokepy does not yet track the original
    # source slot, but the common singles case is still recoverable here.
    opp_side = 1 - int(side)
    opp_base = OFF_SIDE0 if opp_side == 0 else OFF_SIDE1
    opp_active = int(battle[OFF_META + (M_ACTIVE0 if opp_side == 0 else M_ACTIVE1)])
    opp_off = opp_base + int(opp_active) * POKEMON_SIZE
    if int(battle[opp_off + 1]) <= 0:
        battle[ext_off] = np.int16(_wrap_i16(ext_vol & ~EXT_VOL_PARTIAL_TRAP))
        battle[turns_off] = 0
        return
    opp_item = int(battle[opp_off + 6])
    divisor = 6 if opp_item == ITEM_BINDING_BAND else 8

    max_hp = int(battle[poff + 2])
    dmg = max(1, max_hp // divisor)
    battle[poff + 1] = max(0, hp - dmg)


def _apply_aqua_ring_ingrain_heal(
    battle: np.ndarray,
    pokemon_offset: int,
    side: int,
) -> None:
    """Heal 1/16 max HP for each of Aqua Ring / Ingrain at EOT.

    Showdown: data/moves.ts aquaring condition.onResidual (order 6),
    ingrain condition.onResidual (order 7). Both call
    `this.heal(pokemon.baseMaxhp / 16)` which passes through the
    `TryHeal` event — Heal Block blocks it, Big Root multiplies it 1.3x.
    Pokepy stacks both heals if both volatiles are set.
    """
    from pokepy.core.constants import (
        EXT_VOL_AQUA_RING,
        EXT_VOL_INGRAIN,
        EXT_VOL_HEAL_BLOCK,
        ITEM_BIG_ROOT,
    )

    hp = int(battle[pokemon_offset + 1])
    if hp <= 0:
        return

    ext_off = OFF_FIELD + (
        F_EXTENDED_VOLATILE_0 if side == 0 else F_EXTENDED_VOLATILE_1
    )
    ext_vol = int(battle[ext_off]) & 0xFFFF

    has_aqua_ring = (ext_vol & EXT_VOL_AQUA_RING) != 0
    has_ingrain = (ext_vol & EXT_VOL_INGRAIN) != 0
    if not (has_aqua_ring or has_ingrain):
        return

    # Heal Block suppresses all healing on the affected mon.
    if (ext_vol & EXT_VOL_HEAL_BLOCK) != 0:
        return

    max_hp = int(battle[pokemon_offset + 2])
    item = int(battle[pokemon_offset + 6])
    has_big_root = item == ITEM_BIG_ROOT

    def _heal_amount() -> int:
        base = max(1, max_hp // 16)
        if has_big_root:
            # Showdown items.ts:bigroot onTryHeal chainModify([5324, 4096])
            # — note 5324, NOT 5325, the value is deliberately off-by-one.
            base = max(1, (base * 5324) // 4096)
        return base

    total = 0
    if has_aqua_ring:
        total += _heal_amount()
    if has_ingrain:
        total += _heal_amount()

    new_hp = min(max_hp, hp + total)
    battle[pokemon_offset + 1] = new_hp


def _process_yawn(
    battle: np.ndarray,
    pokemon_offset: int,
    side: int,
    gen5_prng: Gen5PRNG = None,
) -> None:
    """Decrement yawn counter; on transition 1 -> 0, apply sleep.

    Showdown's yawn condition has duration: 2. On apply, turn 1 sets it
    (drowsy). The condition's `onEnd` fires when duration is decremented
    to 0 (turn 2 EOT), which calls `trySetStatus('slp')`. Sleep apply
    respects terrain (Electric, Misty), status immunity abilities
    (Insomnia/Vital Spirit/Sweet Veil/Purifying Salt/Comatose),
    and existing status. We delegate to `can_set_self_status` so all
    of these are honoured.

    Showdown: data/moves.ts yawn -> condition.onEnd.
    """
    from pokepy.core.constants import (
        F_YAWN_TURNS_0,
        F_YAWN_TURNS_1,
        F_EXTENDED_VOLATILE_0,
        F_EXTENDED_VOLATILE_1,
        EXT_VOL_YAWN,
        STATUS_SLEEP,
    )
    from pokepy.effects.status_apply import can_set_self_status

    yawn_off = OFF_FIELD + (F_YAWN_TURNS_0 if side == 0 else F_YAWN_TURNS_1)
    ext_off = OFF_FIELD + (
        F_EXTENDED_VOLATILE_0 if side == 0 else F_EXTENDED_VOLATILE_1
    )
    turns = int(battle[yawn_off])
    if turns <= 0:
        return
    if int(battle[pokemon_offset + 1]) <= 0:
        battle[yawn_off] = 0
        battle[ext_off] = int(battle[ext_off]) & ~EXT_VOL_YAWN
        return

    new_turns = turns - 1
    if new_turns == 0:
        # Trigger sleep via the standard apply path so terrain/ability
        # immunities match Showdown's trySetStatus('slp').
        if can_set_self_status(battle, pokemon_offset, STATUS_SLEEP):
            # Sleep duration 1-3 turns (Showdown: this.random(2, 5) = 2/3/4;
            # decrement-first semantics → 1/2/3 actual missed moves, which
            # pokepy stores as 1/2/3).
            if gen5_prng is not None:
                sleep_turns = int(gen5_prng.random(3)) + 1
            else:
                sleep_turns = 2
            battle[pokemon_offset + 12] = (sleep_turns << 8) | STATUS_SLEEP
        battle[yawn_off] = 0
        battle[ext_off] = int(battle[ext_off]) & ~EXT_VOL_YAWN
    else:
        battle[yawn_off] = new_turns


def decrement_terrain(battle: np.ndarray) -> None:
    """Decrement terrain turns at end of turn. Terrain expires after 5 turns.

    Port of MultiFormatFastEnv._decrement_terrain (line ~7675).
    """
    turns = int(battle[OFF_META + M_TERRAIN_TURNS])
    new_turns = turns - 1 if turns > 0 else 0
    terrain_expired = (turns > 0) and (new_turns == 0)
    if terrain_expired:
        battle[OFF_FIELD + F_TERRAIN] = 0
    battle[OFF_META + M_TERRAIN_TURNS] = new_turns


def decrement_trick_room(battle: np.ndarray) -> None:
    """Decrement Trick Room turns at end of turn.

    Port of MultiFormatFastEnv._decrement_trick_room (line ~7685).
    """
    trick_room_turns = int(battle[OFF_FIELD + F_TRICK_ROOM])
    new_turns = trick_room_turns - 1 if trick_room_turns > 0 else 0
    battle[OFF_FIELD + F_TRICK_ROOM] = new_turns


def decrement_screens(battle: np.ndarray) -> None:
    """Decrement screen turns for both sides at end of turn.

    Port of MultiFormatFastEnv._decrement_screens (line ~7708).
    """
    for screens_offset in (OFF_FIELD + F_SCREENS_0, OFF_FIELD + F_SCREENS_1):
        screens = int(battle[screens_offset])
        new_screens = screens
        for shift, mask, clear_mask in _SCREEN_FIELDS:
            val = (screens >> shift) & mask
            new_val = val - 1 if val > 0 else 0
            new_screens = (new_screens & int(clear_mask)) | (new_val << shift)
        # Re-bias to int16 range like the canonical format.
        if new_screens >= 0x8000:
            new_screens -= 0x10000
        battle[screens_offset] = new_screens


def decrement_weather(battle: np.ndarray) -> None:
    """Decrement weather turns at end of turn.

    Weather set by moves lasts 5 turns (M_WEATHER_TURNS > 0). Weather set
    by abilities is permanent (M_WEATHER_TURNS = 0).

    Port of MultiFormatFastEnv._decrement_weather (line ~7720).
    """
    turns = int(battle[OFF_META + M_WEATHER_TURNS])
    new_turns = turns - 1 if turns > 0 else 0
    weather_expired = (turns > 0) and (new_turns == 0)
    if weather_expired:
        battle[OFF_FIELD + F_WEATHER] = WEATHER_NONE
    battle[OFF_META + M_WEATHER_TURNS] = new_turns
