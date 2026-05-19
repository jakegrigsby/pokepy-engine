"""Status application effects (port of _apply_status_from_move and friends).

Ports of:

The the Showdown reference versions are batched the reference functions that thread `gen5_seed` through
each call. The pokepy ports mutate `battle: np.ndarray[256]` in place and
take a stateful `Gen5PRNG`.
"""
from __future__ import annotations

from pokepy.effects._common import np, MultiFormatState, Gen5PRNG
from pokepy.effects.ability_suppression import effective_ability
from pokepy.effects.grounding import is_grounded
from pokepy.core.bitpack import (
    apply_boost_to_packed,
    extract_boost,
    get_status,
    get_status_turns,
    set_status,
)
from pokepy.core.constants import (
    OFF_FIELD,
    F_EXTENDED_VOLATILE_0,
    F_EXTENDED_VOLATILE_1,
    F_TERRAIN,
    F_WEATHER,
    EXT_VOL_HEAL_BLOCK,
    STATUS_NONE,
    STATUS_BURN,
    STATUS_PARALYSIS,
    STATUS_SLEEP,
    STATUS_FREEZE,
    STATUS_POISON,
    STATUS_TOXIC,
    TYPE_FIRE,
    TYPE_ELECTRIC,
    TYPE_ICE,
    TYPE_POISON,
    TYPE_STEEL,
    TYPE_GROUND,
    TYPE_FLYING,
    TYPE_GRASS,
    TERRAIN_MISTY,
    TERRAIN_ELECTRIC,
    WEATHER_SUN,
    WEATHER_DESOLATE_LAND,
    FLAG_POWDER,
    ABILITY_SERENE_GRACE,
    ABILITY_PURIFYING_SALT,
    ABILITY_WATER_VEIL,
    ABILITY_WATER_BUBBLE,
    ABILITY_THERMAL_EXCHANGE,
    ABILITY_LIMBER,
    ABILITY_INSOMNIA,
    ABILITY_VITAL_SPIRIT,
    ABILITY_IMMUNITY,
    ABILITY_MAGMA_ARMOR,
    ABILITY_LEVITATE,
    ABILITY_OVERCOAT,
    ABILITY_MAGIC_GUARD,
    ABILITY_POISON_HEAL,
    ABILITY_MOLD_BREAKER,
    ABILITY_TERAVOLT,
    ABILITY_TURBOBLAZE,
    ITEM_SAFETY_GOGGLES,
    OFF_SIDE1,
)

# Ability IDs not in constants.py — defined locally to mirror the Showdown reference usage
ABILITY_GOOD_AS_GOLD = 283
ABILITY_SHIELDS_DOWN_STATUS = 197
ABILITY_COMATOSE = 213
ABILITY_CORROSION = 212
ABILITY_SYNCHRONIZE = 28
ABILITY_SWEET_VEIL = 175
ABILITY_LEAF_GUARD = 102
MOVE_TRI_ATTACK = 161

def _has_heal_block(battle: np.ndarray, pokemon_offset: int) -> bool:
    """Return True iff the holder is currently under Heal Block."""
    poff = int(pokemon_offset)
    ext_off = OFF_FIELD + (F_EXTENDED_VOLATILE_0 if poff < OFF_SIDE1 else F_EXTENDED_VOLATILE_1)
    ext = int(battle[ext_off]) & 0xFFFF
    return (ext & EXT_VOL_HEAL_BLOCK) != 0

def _is_grounded(battle: np.ndarray, pokemon_offset: int) -> bool:
    """Check if a Pokemon is grounded for terrain/status interactions."""
    return is_grounded(battle, pokemon_offset)

def can_set_self_status(
    battle: np.ndarray,
    pokemon_offset: int,
    status: int,
    *,
    allow_existing_status: bool = False,
) -> bool:
    """Check whether `status` can be applied to the Pokemon at `pokemon_offset`
    as a SELF-inflicted status (e.g., Toxic Orb, Flame Orb, Rest).

    Mirrors Showdown's `trySetStatus` checks but skips the Safeguard / Corrosion
    source checks (not applicable for self-application). Returns True iff the
    status CAN be applied (no blocking existing status, no immunity blocks).

    Used by Toxic Orb / Flame Orb residual handling.
    """
    poff = int(pokemon_offset)

    # Self-status moves/items normally fail when any status is already present.
    # Rest is the exception: it can replace a non-sleep status with sleep, so
    # callers can opt into `allow_existing_status=True` to mirror that path.
    current_status = get_status(int(battle[poff + 12]))
    if allow_existing_status:
        if current_status == status:
            return False
    elif current_status != STATUS_NONE:
        return False

    ability = int(battle[poff + 5])

    # Blanket-immunity abilities
    if ability == ABILITY_PURIFYING_SALT:
        return False
    if ability == ABILITY_COMATOSE:
        return False
    if ability == ABILITY_GOOD_AS_GOLD:
        # Good as Gold blocks status *moves*, not items/passive. Skip here.
        pass
    if ability == ABILITY_SHIELDS_DOWN_STATUS:
        hp = int(battle[poff + 1])
        max_hp = int(battle[poff + 2])
        if hp * 2 > max_hp:
            return False

    # Ability-based status immunities
    if status == STATUS_BURN and ability in (
        ABILITY_WATER_VEIL,
        ABILITY_WATER_BUBBLE,
        ABILITY_THERMAL_EXCHANGE,
    ):
        return False
    if status == STATUS_PARALYSIS and ability == ABILITY_LIMBER:
        return False
    if status == STATUS_SLEEP and ability in (
        ABILITY_INSOMNIA, ABILITY_VITAL_SPIRIT, ABILITY_SWEET_VEIL,
    ):
        return False
    if status in (STATUS_POISON, STATUS_TOXIC) and ability == ABILITY_IMMUNITY:
        return False
    if status == STATUS_FREEZE and ability == ABILITY_MAGMA_ARMOR:
        return False

    # Type-based status immunities (Gen 6+)
    types_packed = int(battle[poff + 4]) & 0xFFFF
    t1 = types_packed & 0xFF
    t2 = (types_packed >> 8) & 0xFF

    if status == STATUS_BURN and (t1 == TYPE_FIRE or t2 == TYPE_FIRE):
        return False
    if status == STATUS_PARALYSIS and (t1 == TYPE_ELECTRIC or t2 == TYPE_ELECTRIC):
        return False
    if status in (STATUS_POISON, STATUS_TOXIC):
        # Corrosion (source = self in self-application) bypasses Poison/Steel
        # type immunity. Source: sim/pokemon.ts:1661 — setStatus skips the
        # runStatusImmunity chain when source has Corrosion and status is
        # psn/tox. Self-application passes `source = this`.
        has_corrosion = ability == ABILITY_CORROSION
        if not has_corrosion:
            if (t1 == TYPE_POISON or t2 == TYPE_POISON or
                t1 == TYPE_STEEL or t2 == TYPE_STEEL):
                return False

    # Terrain immunities for grounded Pokemon
    current_terrain = int(battle[OFF_FIELD + F_TERRAIN])
    current_weather = int(battle[OFF_FIELD + F_WEATHER])
    if ability == ABILITY_LEAF_GUARD and current_weather in (
        WEATHER_SUN,
        WEATHER_DESOLATE_LAND,
    ):
        return False
    if _is_grounded(battle, poff):
        if current_terrain == TERRAIN_MISTY:
            return False
        if status == STATUS_SLEEP and current_terrain == TERRAIN_ELECTRIC:
            return False

    if status == STATUS_FREEZE and current_weather in (WEATHER_SUN, WEATHER_DESOLATE_LAND):
        return False

    return True

def _can_apply_status(
    battle: np.ndarray,
    move_id: int | None,
    target_offset: int,
    status: int,
    game_data,
    user_offset: int = None,
    *,
    is_status_move: bool,
) -> bool:
    """Return True iff the target can receive `status` from this source."""
    target_offset = int(target_offset)
    status = int(status)

    if int(battle[target_offset + 1]) <= 0:
        return False

    # Get target's ability for immunity checks
    target_ability = int(battle[target_offset + 5])
    user_ability = int(battle[int(user_offset) + 5]) if user_offset is not None else 0
    user_ignores_target_ability = user_ability in (
        ABILITY_MOLD_BREAKER,
        ABILITY_TERAVOLT,
        ABILITY_TURBOBLAZE,
    )

    # Good as Gold blocks status moves, not damaging-move secondaries.
    if (
        is_status_move
        and target_ability == ABILITY_GOOD_AS_GOLD
        and not user_ignores_target_ability
    ):
        return False

    # Purifying Salt: immune to all status conditions
    if target_ability == ABILITY_PURIFYING_SALT:
        return False

    # Shields Down (Minior): immune to status when HP > 50% (Meteor Form)
    if target_ability == ABILITY_SHIELDS_DOWN_STATUS:
        target_hp_sd = int(battle[target_offset + 1])
        target_maxhp_sd = int(battle[target_offset + 2])
        if target_hp_sd * 2 > target_maxhp_sd:
            return False

    # Comatose (Komala): permanently asleep, immune to other status
    if target_ability == ABILITY_COMATOSE:
        return False

    # Ability-based status immunities
    burn_immune_ability = (
        target_ability in (
            ABILITY_WATER_VEIL,
            ABILITY_WATER_BUBBLE,
            ABILITY_THERMAL_EXCHANGE,
        )
    ) and (status == STATUS_BURN)
    para_immune_ability = (target_ability == ABILITY_LIMBER) and (status == STATUS_PARALYSIS)
    sleep_immune = (
        target_ability == ABILITY_INSOMNIA
        or target_ability == ABILITY_VITAL_SPIRIT
        or target_ability == ABILITY_SWEET_VEIL
    ) and (status == STATUS_SLEEP)
    poison_immune_ability = (target_ability == ABILITY_IMMUNITY) and (
        status == STATUS_POISON or status == STATUS_TOXIC
    )
    freeze_immune = (target_ability == ABILITY_MAGMA_ARMOR) and (status == STATUS_FREEZE)
    current_weather = int(battle[OFF_FIELD + F_WEATHER])
    freeze_immune_weather = (status == STATUS_FREEZE) and (
        current_weather == WEATHER_SUN or current_weather == WEATHER_DESOLATE_LAND
    )

    # Get target's types for type-based immunities
    target_types = int(battle[target_offset + 4]) & 0xFFFF
    target_type1 = target_types & 0xFF
    target_type2 = (target_types >> 8) & 0xFF

    is_fire_type = (target_type1 == TYPE_FIRE) or (target_type2 == TYPE_FIRE)
    burn_immune_type = is_fire_type and (status == STATUS_BURN)

    is_electric_type = (target_type1 == TYPE_ELECTRIC) or (target_type2 == TYPE_ELECTRIC)
    para_immune_type = is_electric_type and (status == STATUS_PARALYSIS)

    is_ice_type = (target_type1 == TYPE_ICE) or (target_type2 == TYPE_ICE)
    freeze_immune_type = is_ice_type and (status == STATUS_FREEZE)

    user_has_corrosion = False
    if user_offset is not None:
        user_ab_corr = int(battle[user_offset + 5])
        user_has_corrosion = user_ab_corr == ABILITY_CORROSION
    is_poison_type = (target_type1 == TYPE_POISON) or (target_type2 == TYPE_POISON)
    is_steel_type = (target_type1 == TYPE_STEEL) or (target_type2 == TYPE_STEEL)
    poison_immune_type = (
        (is_poison_type or is_steel_type)
        and (status == STATUS_POISON or status == STATUS_TOXIC)
        and (not user_has_corrosion)
    )

    # Ground types are only immune to Electric-type status moves.
    move_type = int(game_data.move_type[move_id]) if move_id is not None else -1
    is_ground_type = (target_type1 == TYPE_GROUND) or (target_type2 == TYPE_GROUND)
    ground_immune_elec = (
        is_ground_type and (move_type == TYPE_ELECTRIC) and (status == STATUS_PARALYSIS)
    )
    para_immune_type = para_immune_type or ground_immune_elec

    burn_immune = burn_immune_ability or burn_immune_type
    para_immune = para_immune_ability or para_immune_type
    poison_immune = poison_immune_ability or poison_immune_type

    # Terrain status immunity for grounded Pokemon
    current_terrain = int(battle[OFF_FIELD + F_TERRAIN])
    is_grounded = _is_grounded(battle, target_offset)
    is_misty = current_terrain == TERRAIN_MISTY
    misty_terrain_immune = is_misty and is_grounded
    is_electric_terrain = current_terrain == TERRAIN_ELECTRIC
    electric_terrain_sleep_immune = (
        is_electric_terrain and is_grounded and (status == STATUS_SLEEP)
    )

    # Powder/spore immunities
    is_grass_type = (target_type1 == TYPE_GRASS) or (target_type2 == TYPE_GRASS)
    move_flags_status = int(game_data.move_flags[move_id]) if move_id is not None else 0
    is_powder_move = (move_flags_status & FLAG_POWDER) != 0
    target_item = int(battle[target_offset + 6])
    has_overcoat_status = target_ability == ABILITY_OVERCOAT
    has_goggles_status = target_item == ITEM_SAFETY_GOGGLES
    grass_powder_immune = (
        (is_grass_type or has_overcoat_status or has_goggles_status) and is_powder_move
    )

    # Safeguard blocks all non-volatile status from opponents.
    target_is_side0 = target_offset < OFF_SIDE1
    from pokepy.core.constants import (
        F_SCREENS_0 as _FS0_SA, F_SCREENS_1 as _FS1_SA,
        SCREEN_SAFEGUARD_SHIFT as _SAFEGUARD_SHIFT,
    )
    screens_target = int(battle[OFF_FIELD + (_FS0_SA if target_is_side0 else _FS1_SA)])
    safeguard_active = ((screens_target >> _SAFEGUARD_SHIFT) & 0x3) > 0
    user_has_infiltrator = False
    if user_offset is not None:
        user_ab_inf = int(battle[user_offset + 5])
        ABILITY_INFILTRATOR_SA = 151
        user_has_infiltrator = user_ab_inf == ABILITY_INFILTRATOR_SA
    safeguard_immune = safeguard_active and not user_has_infiltrator

    status_blocked = (
        burn_immune
        or para_immune
        or sleep_immune
        or poison_immune
        or freeze_immune_type
        or freeze_immune
        or freeze_immune_weather
        or misty_terrain_immune
        or grass_powder_immune
        or electric_terrain_sleep_immune
        or safeguard_immune
    )
    if status_blocked:
        return False

    # Only apply if target has no status (can't be statused twice)
    status_offset = target_offset + 12
    current_status = get_status(int(battle[status_offset]))
    if current_status != STATUS_NONE:
        return False

    return True

def _try_apply_status(
    battle: np.ndarray,
    move_id: int | None,
    target_offset: int,
    status: int,
    game_data,
    gen5_prng: Gen5PRNG,
    user_offset: int = None,
    *,
    is_status_move: bool,
    prerolled_sleep_turns: int | None = None,
) -> None:
    """Apply an already-selected status if the target can receive it.

    Shared by regular status secondaries, Tri Attack's callback secondary,
    and passive status sources that do not come from a move.
    """
    target_offset = int(target_offset)
    status = int(status)

    if not _can_apply_status(
        battle,
        move_id,
        target_offset,
        status,
        game_data,
        user_offset=user_offset,
        is_status_move=is_status_move,
    ):
        return

    target_ability = int(battle[target_offset + 5])
    status_offset = target_offset + 12

    # Sleep is the only case that still needs a PRNG frame here.
    # Showdown conditions.ts:slp.onStart uses random(2, 5), i.e. stored
    # duration values 2..4 rather than 1..3.
    if status == STATUS_SLEEP:
        initial_turns = (
            int(prerolled_sleep_turns)
            if prerolled_sleep_turns is not None
            else gen5_prng.random(2, 5)
        )
    else:
        initial_turns = 0

    battle[status_offset] = set_status(status, initial_turns)

    # Synchronize reflects burn / para / poison / toxic back at the source.
    if user_offset is not None:
        has_synchronize = target_ability == ABILITY_SYNCHRONIZE
        sync_valid = status in (STATUS_BURN, STATUS_PARALYSIS, STATUS_POISON, STATUS_TOXIC)
        if has_synchronize and sync_valid:
            user_status_offset = user_offset + 12
            user_current_status = get_status(int(battle[user_status_offset]))
            if user_current_status == STATUS_NONE:
                user_ability = int(battle[user_offset + 5])
                blocked = False
                if user_ability in (ABILITY_PURIFYING_SALT, ABILITY_COMATOSE):
                    blocked = True
                if status == STATUS_BURN and user_ability in (
                    ABILITY_WATER_VEIL, ABILITY_THERMAL_EXCHANGE,
                ):
                    blocked = True
                if status == STATUS_PARALYSIS and user_ability == ABILITY_LIMBER:
                    blocked = True
                if status in (STATUS_POISON, STATUS_TOXIC) and user_ability == ABILITY_IMMUNITY:
                    blocked = True
                if user_ability == ABILITY_SHIELDS_DOWN_STATUS:
                    u_hp = int(battle[user_offset + 1])
                    u_max = int(battle[user_offset + 2])
                    if u_hp * 2 > u_max:
                        blocked = True
                if not blocked:
                    battle[user_status_offset] = set_status(status, 0)

def apply_status_from_move(
    battle: np.ndarray,
    move_id: int,
    target_offset: int,
    hit: bool,
    game_data,
    move_effects,
    gen5_prng: Gen5PRNG,
    user_offset: int = None,
    num_hits: int = 1,
    prerolled_rolls: "list[int] | None" = None,
) -> None:
    """Port of MultiFormatFastEnv._apply_status_from_move (line ~7045).

    Mutates `battle` in place. Advances `gen5_prng` only when the move actually
    has a status effect (matches Showdown's behavior of only calling random()
    when needed).

    For multi-hit moves (Twineedle 20% poison per hit), Showdown rolls the
    secondary chance per hit inside the moveHit loop
    (sim/battle-actions.ts:1357 secondaryRoll). Each hit independently
    triggers; success on any hit applies the status.
    """
    move_id = int(move_id)
    target_offset = int(target_offset)
    hit = bool(hit)

    status = int(move_effects.status[move_id])
    status_chance = int(move_effects.status_chance[move_id])
    move_effect_type = int(move_effects.effect_type[move_id])

    # Rest (EFFECT_RECOVERY with self-inflicted sleep) is handled entirely
    # inside apply_recovery_from_move. The move_effects table stores
    # status=SLEEP, status_chance=100 on Rest, but that's a SELF status,
    # not a target-directed secondary. Skip here to avoid writing sleep
    # to the opponent.
    from pokepy.data.move_effects import EFFECT_RECOVERY as _EFFECT_RECOVERY_SA
    from pokepy.core.constants import MOVE_REST as _MOVE_REST_SA
    if move_id == _MOVE_REST_SA and move_effect_type == _EFFECT_RECOVERY_SA:
        return

    # Sheer Force (data/abilities.ts:4122) deletes `move.secondaries` in
    # onModifyMove BEFORE the move executes, so Showdown never rolls a random
    # for the secondary. Gate the PRNG advance on (not Sheer Force) to keep
    # frame counts in sync.
    user_has_sheer_force = False
    if user_offset is not None:
        user_offset = int(user_offset)
        user_ability = int(battle[user_offset + 5])
        if user_ability == ABILITY_SERENE_GRACE:
            # Serene Grace: double secondary effect chance
            status_chance = min(status_chance * 2, 100)
        from pokepy.core.constants import ABILITY_SHEER_FORCE as _ABILITY_SHEER_FORCE_SA
        user_has_sheer_force = user_ability == _ABILITY_SHEER_FORCE_SA

    # Shield Dust (data/abilities.ts:shielddust) and Covert Cloak
    # (data/items.ts:covertcloak) filter out target-affecting secondaries in
    # `onModifySecondaries` BEFORE the secondaryRoll — so either prevents
    # PRNG consumption on a secondary status. Only applies to secondaries,
    # i.e. when the move is a damaging / multi-hit move — primary status
    # moves (Will-O-Wisp, Toxic, Thunder Wave) have effect_type == EFFECT_STATUS
    # and are unaffected. Shield Dust is bypassed by Mold Breaker /
    # Turboblaze / Teravolt — pokepy doesn't track Mold Breaker here, but
    # those abilities are rare in OU.
    from pokepy.data.move_effects import EFFECT_STATUS as _EFFECT_STATUS_SD
    _ABILITY_SHIELD_DUST_SA = 19
    _ITEM_COVERT_CLOAK_SA = 750
    target_ability_sd = int(battle[target_offset + 5])
    target_item_sd = int(battle[target_offset + 6])
    is_secondary_status = (move_effect_type != _EFFECT_STATUS_SD)
    target_has_shield_dust = target_ability_sd == _ABILITY_SHIELD_DUST_SA
    target_has_covert_cloak = target_item_sd == _ITEM_COVERT_CLOAK_SA
    shield_dust_blocks = is_secondary_status and (
        target_has_shield_dust or target_has_covert_cloak
    )

    # CRITICAL: Only advance PRNG if move actually has a status effect AND
    # the secondary survives Sheer Force / Shield Dust suppression. Sheer
    # Force deletes `move.secondaries` outright, so it suppresses ALL
    # secondary status (including 100% chance ones like Zap Cannon, Nuzzle);
    # primary status moves (Will-O-Wisp etc.) have effect_type == EFFECT_STATUS
    # and are NOT secondaries, so Sheer Force ignores them.
    sheer_force_blocks = user_has_sheer_force and (move_effect_type != _EFFECT_STATUS_SD)
    has_status_effect = (
        hit and (status > 0) and (status_chance > 0)
        and not sheer_force_blocks
        and not shield_dust_blocks
    )
    if not has_status_effect:
        return

    # Primary-status moves (Toxic, Will-O-Wisp, Thunder Wave, Spore, ...) in
    # Showdown apply their `move.status` directly once the move has hit — no
    # `randomChance` call inside `moveHit`. Only secondary status (Fire Blast
    # 10% burn, Scald 30% burn, Ice Beam 10% freeze, ...) gets the roll.
    # We already know the move hit (the caller only calls us when `hit=True`),
    # so a primary status always applies without consuming a PRNG frame.
    if not is_secondary_status:
        should_apply = True
    else:
        # Multi-hit secondary rolls (Twineedle, etc.). Roll N times; apply if
        # any hit succeeds. Single-hit moves use n=1 — identical to the
        # previous one-roll behavior.
        #
        # If `prerolled_rolls` is provided, use those values instead of
        # consuming PRNG frames here (matches Showdown's per-move
        # secondaryRoll ordering — see battle_gen9.py preroll logic around
        # the _calc_pN block).
        n = max(1, int(num_hits))
        should_apply = False
        for i in range(n):
            if prerolled_rolls is not None and i < len(prerolled_rolls):
                roll = int(prerolled_rolls[i])
            else:
                roll = gen5_prng.random(100)
            if roll < status_chance:
                should_apply = True
    if not should_apply:
        return

    _try_apply_status(
        battle,
        move_id,
        target_offset,
        status,
        game_data,
        gen5_prng,
        user_offset=user_offset,
        is_status_move=not is_secondary_status,
    )

def apply_tri_attack_status_from_move(
    battle: np.ndarray,
    move_id: int,
    target_offset: int,
    hit: bool,
    game_data,
    gen5_prng: Gen5PRNG,
    user_offset: int = None,
    prerolled_roll: "int | None" = None,
    prerolled_status: "int | None" = None,
) -> None:
    """Apply Tri Attack's callback secondary.

    Showdown models this as `secondary.onHit` with its own `random(3)` status
    picker after the `20%` secondary roll. The picker still fires on KO turns
    once the chance roll lands; the later `trySetStatus` simply fails against a
    fainted target.
    """
    move_id = int(move_id)
    target_offset = int(target_offset)
    if move_id != MOVE_TRI_ATTACK or not bool(hit):
        return

    chance = 20
    if user_offset is not None and int(battle[int(user_offset) + 5]) == ABILITY_SERENE_GRACE:
        chance = min(100, chance * 2)

    roll = int(prerolled_roll) if prerolled_roll is not None else int(gen5_prng.random(100))
    if roll >= chance:
        return

    choice = int(prerolled_status) if prerolled_status is not None else int(gen5_prng.random(3))
    if choice == 0:
        status = STATUS_BURN
    elif choice == 1:
        status = STATUS_PARALYSIS
    else:
        status = STATUS_FREEZE

    _try_apply_status(
        battle,
        move_id,
        target_offset,
        status,
        game_data,
        gen5_prng,
        user_offset=user_offset,
        is_status_move=False,
    )

def apply_end_of_turn_status(
    battle: np.ndarray,
    pokemon_offset: int,
    game_data,
    move_effects,
    gen5_prng: Gen5PRNG,
    *,
    skip_sleep_decrement: bool = False,
) -> None:
    """Port of MultiFormatFastEnv._apply_end_of_turn_status (line ~7444).

    Applies burn/poison/toxic damage and decrements/cures sleep + freeze.
    Mutates `battle` in place. Advances `gen5_prng` only when the Pokemon
    is actually frozen (matches Showdown's behavior).
    """
    pokemon_offset = int(pokemon_offset)
    status_offset = pokemon_offset + 12
    hp_offset = pokemon_offset + 1
    max_hp_offset = pokemon_offset + 2

    status_field = int(battle[status_offset])
    status = get_status(status_field)
    status_turns = get_status_turns(status_field)

    current_hp = int(battle[hp_offset])
    max_hp = int(battle[max_hp_offset])

    # Calculate damage based on status.
    # Showdown sim/battle.ts:2029 and 2044 applies `clampIntRange(damage, 1)`
    # to every residual damage value, so burn / psn / tox deal at least 1 HP
    # when the raw divisor rounds to zero (e.g., very low max HP).
    burn_damage = max(1, max_hp // 16)
    poison_damage = max(1, max_hp // 8)
    # Toxic counter increments each turn: 1/16, 2/16, 3/16, ... up to 15/16.
    # On the turn toxic is applied, status_turns=0 and counter=1.
    # After EOT, status_turns becomes 1; next turn uses counter=2.
    # Showdown clamps the per-unit to min 1 then multiplies by stage
    # (conditions.ts line 159: clampIntRange(baseMaxhp/16, 1) * stage).
    toxic_counter = status_turns + 1
    toxic_unit = max(1, max_hp // 16)
    toxic_damage = toxic_unit * toxic_counter

    if status == STATUS_BURN:
        damage = burn_damage
    elif status == STATUS_POISON:
        damage = poison_damage
    elif status == STATUS_TOXIC:
        damage = toxic_damage
    else:
        damage = 0

    # Residual status hooks use the currently effective ability state, so
    # Neutralizing Gas / Ability Shield suppression applies to Magic Guard,
    # Poison Heal, Heatproof, and Early Bird exactly like Showdown.
    ability = effective_ability(battle, pokemon_offset)

    # Heatproof halves burn residual damage.
    # Showdown: data/abilities.ts heatproof `onDamage(damage, target, source, effect)
    #   { if (effect && effect.id === 'brn') return damage / 2; }`. Pokepy previously
    # only halved Fire-type *move* damage, never the residual burn tick.
    _ABILITY_HEATPROOF = 85
    if status == STATUS_BURN and ability == _ABILITY_HEATPROOF:
        damage = damage // 2

    # Magic Guard prevents indirect damage
    if ability == ABILITY_MAGIC_GUARD:
        damage = 0

    # Poison Heal: heal 1/8 HP instead of taking poison damage. Showdown's
    # heal() (sim/battle.ts:2200) clamps `0 < damage <= 1` up to 1, so a
    # fractional heal (e.g. maxhp=7 → 0.875) still heals at least 1 HP.
    has_poison_heal = ability == ABILITY_POISON_HEAL
    is_poisoned = status == STATUS_POISON or status == STATUS_TOXIC
    poison_heal_amount = max(1, max_hp // 8)
    poison_heal_blocked = _has_heal_block(battle, pokemon_offset)

    if has_poison_heal and is_poisoned:
        if poison_heal_blocked:
            new_hp = current_hp
        else:
            new_hp = min(max_hp, current_hp + poison_heal_amount)
    else:
        new_hp = max(0, current_hp - damage)
    battle[hp_offset] = new_hp

    # Update status turns
    new_toxic_turns = min(status_turns + 1, 15)
    # Early Bird halves sleep time. Showdown data/abilities.ts earlybird is
    # implemented in conditions.ts slp onBeforeMove: `if
    # (pokemon.hasAbility('earlybird')) pokemon.statusState.time--;` BEFORE
    # the normal `pokemon.statusState.time--` — so the counter drops by 2
    # per turn. Pokepy's EOT decrement model mirrors this by subtracting 2
    # instead of 1 when the sleeping mon has Early Bird.
    _ABILITY_EARLY_BIRD = 48
    sleep_decrement = 2 if ability == _ABILITY_EARLY_BIRD else 1
    new_sleep_turns = max(status_turns - sleep_decrement, 0)
    # Sleep curing is deferred to the NEXT turn's onBeforeMove gate so the
    # end-of-turn snapshot still reflects slp (matching Showdown's
    # per-mon onBeforeMove timing — Showdown decrements + cures at the
    # start of the next runMove, not at EOT).  See
    # `engine/battle_gen9.py` sleep_blocked block for the turn-start
    # wake check.  EOT just decrements.

    # Freeze thaw is NOT rolled at EOT — Showdown rolls 1/5 in
    # `onBeforeMove` (data/conditions.ts:99 frz onBeforeMove → randomChance(1, 5)).
    # See engine/battle_gen9.py status-immobilization block for the actual roll.
    # Doing it here would consume PRNG frames at the wrong time and would
    # also fire on switching mons that never attempt a move.
    if status == STATUS_TOXIC:
        new_turns = new_toxic_turns
    elif status == STATUS_SLEEP:
        new_turns = status_turns if skip_sleep_decrement else new_sleep_turns
    else:
        new_turns = status_turns

    new_status = status

    battle[status_offset] = set_status(new_status, new_turns)

def apply_end_of_turn_status_effects(
    battle: np.ndarray,
    pokemon0_offset: int,
    pokemon1_offset: int,
    game_data,
    move_effects,
    gen5_prng: Gen5PRNG,
    *,
    skip_sleep_decrement0: bool = False,
    skip_sleep_decrement1: bool = False,
) -> None:
    """Port of MultiFormatFastEnv._apply_end_of_turn_status_effects (line ~7540).

    Applies status damage (burn/poison/toxic) and freeze thaw checks for both
    active Pokemon, in turn order. Only processes Pokemon with HP > 0 (matches
    Showdown's order: status damage comes after weather/items/leech seed).
    """
    pokemon0_offset = int(pokemon0_offset)
    pokemon1_offset = int(pokemon1_offset)

    hp0 = int(battle[pokemon0_offset + 1])
    if hp0 > 0:
        apply_end_of_turn_status(
            battle,
            pokemon0_offset,
            game_data,
            move_effects,
            gen5_prng,
            skip_sleep_decrement=skip_sleep_decrement0,
        )

    hp1 = int(battle[pokemon1_offset + 1])
    if hp1 > 0:
        apply_end_of_turn_status(
            battle,
            pokemon1_offset,
            game_data,
            move_effects,
            gen5_prng,
            skip_sleep_decrement=skip_sleep_decrement1,
        )
