"""Entry hazards (Stealth Rock, Spikes, Toxic Spikes, Sticky Web).

Port of MultiFormatFastEnv._apply_hazard_from_move / _apply_hazard_damage_on_switch
(the Showdown reference implementation).
"""

from __future__ import annotations

from pokepy.effects._common import np, MultiFormatState, Gen5PRNG
from pokepy.core.bitpack import (
    get_spikes_layers,
    get_status,
    get_stealth_rock,
    get_sticky_web,
    get_toxic_spikes_layers,
    set_spikes,
    set_status,
    set_stealth_rock,
    set_sticky_web,
    set_toxic_spikes,
)
from pokepy.effects.stat_changes import apply_direct_stat_changes
from pokepy.core.constants import (
    ABILITY_LEVITATE,
    ABILITY_MAGIC_GUARD,
    F_HAZARDS_0,
    F_HAZARDS_1,
    HAZARD_SPIKES,
    HAZARD_STEALTH_ROCK,
    HAZARD_STICKY_WEB,
    HAZARD_TOXIC_SPIKES,
    ITEM_AIR_BALLOON,
    ITEM_HEAVY_DUTY_BOOTS,
    OFF_FIELD,
    OFF_META,
    OFF_SIDE0,
    OFF_SIDE1,
    M_ACTIVE0,
    M_ACTIVE1,
    POKEMON_SIZE,
    STATUS_NONE,
    STATUS_POISON,
    STATUS_TOXIC,
    TYPE_FLYING,
    TYPE_POISON,
    TYPE_ROCK,
    TYPE_STEEL,
)
from pokepy.data.type_charts import MODERN_TYPE_CHART

# Iron Ball grounds Flying-types; not in pokepy.core.constants yet.
# Showdown: data/items.ts ironball -> sim/pokemon.ts isGrounded().
_ITEM_IRON_BALL = 278


def apply_hazard_from_move(
    battle: np.ndarray,
    move_id: int,
    target_side: int,
    hit: bool,
    game_data=None,
    move_effects=None,
    user_ability: int = -1,
    user_offset: int | None = None,
    source_hp_override: int | None = None,
    enabled_hazards: frozenset | None = None,
) -> None:
    """Port of _apply_hazard_from_move (line ~7816).

    Hazards are placed on the *target's* side. Mutates `battle` in place.
    Magic Bounce on the target's active mon reflects the hazard back to the
    user's side instead — unless the user has Mold Breaker / Teravolt /
    Turboblaze, which suppress Magic Bounce.

    Stone Axe (830) and Ceaseless Edge (845) are damaging moves that set
    a hazard via onAfterHit (Showdown source: data/moves.ts). Their hazard
    application is gated by:
      - user.hp > 0   (Showdown `if (... && source.hp)` check)
      - !move.hasSheerForce  (Sheer Force suppresses the hazard side
        effect because the move's `secondary: {}` triggers the SF boost)
    They also are NOT reflectable (no `reflectable: 1` flag), so Magic
    Bounce must NOT swap sides for these moves.
    """
    if move_effects is None:
        return
    move_id = int(move_id)
    target_side = int(target_side)
    hazard_type = int(move_effects.hazard[move_id])
    if hazard_type == 0:
        return

    if enabled_hazards is not None:
        _hazard_names = {
            HAZARD_STEALTH_ROCK: "stealthrock",
            HAZARD_SPIKES: "spikes",
            HAZARD_TOXIC_SPIKES: "toxicspikes",
            HAZARD_STICKY_WEB: "stickyweb",
        }
        hname = _hazard_names.get(hazard_type)
        if hname is not None and hname not in enabled_hazards:
            return

    # Stone Axe / Ceaseless Edge gating: damaging hazard-setters need
    # source.hp > 0 and !Sheer Force, and they don't bounce.
    _MOVE_STONE_AXE = 830
    _MOVE_CEASELESS_EDGE = 845
    _ABILITY_SHEER_FORCE_HZ = 125
    is_damaging_hazard_move = move_id in (_MOVE_STONE_AXE, _MOVE_CEASELESS_EDGE)
    if is_damaging_hazard_move:
        if not bool(hit):
            return
        # Ceaseless Edge / Stone Axe fire during the user's own move
        # resolution. When ordered field effects are replayed after both moves
        # in step_battle_gen9, the live user HP may already include the
        # opposing move's later damage. Allow the caller to provide the user's
        # projected HP at its own onAfterHit timing so faster users that faint
        # later in the turn can still lay the hazard, matching Showdown.
        source_hp = None
        if source_hp_override is not None:
            source_hp = int(source_hp_override)
        elif user_offset is not None:
            source_hp = int(battle[int(user_offset) + 1])
        if source_hp is not None and source_hp <= 0:
            return
        if int(user_ability) == _ABILITY_SHEER_FORCE_HZ:
            return
        # Skip Magic Bounce reflection — these moves are not reflectable.
        hazard_offset_dh = OFF_FIELD + (
            F_HAZARDS_0 if target_side == 0 else F_HAZARDS_1
        )
        cur_dh = int(battle[hazard_offset_dh])
        if hazard_type == HAZARD_STEALTH_ROCK:
            new_dh = set_stealth_rock(cur_dh)
        else:  # HAZARD_SPIKES
            layers_dh = get_spikes_layers(cur_dh)
            new_dh = set_spikes(cur_dh, layers_dh + 1)
        val_dh = int(new_dh) & 0xFFFF
        if val_dh >= 0x8000:
            val_dh -= 0x10000
        battle[hazard_offset_dh] = val_dh
        return

    # Magic Bounce reflection: even if `hit` is False (because earlier code
    # blocked the status move), reflect the hazard back. We detect this by
    # checking the target's active mon ability — if Magic Bounce, swap sides.
    from pokepy.core.constants import (
        OFF_SIDE0,
        OFF_SIDE1,
        OFF_META,
        M_ACTIVE0,
        M_ACTIVE1,
        POKEMON_SIZE,
        ABILITY_MAGIC_BOUNCE,
    )

    target_base = OFF_SIDE0 if target_side == 0 else OFF_SIDE1
    target_active = int(
        battle[OFF_META + (M_ACTIVE0 if target_side == 0 else M_ACTIVE1)]
    )
    target_ability = int(battle[target_base + target_active * POKEMON_SIZE + 5])
    _MOLD_BREAKER_SET_HZ = (104, 163, 164)  # moldbreaker, turboblaze, teravolt
    user_ignores_ability = int(user_ability) in _MOLD_BREAKER_SET_HZ
    bounced = (target_ability == ABILITY_MAGIC_BOUNCE) and not user_ignores_ability
    if bounced:
        target_side = 1 - target_side
    elif not bool(hit):
        return

    hazard_offset = OFF_FIELD + (F_HAZARDS_0 if target_side == 0 else F_HAZARDS_1)
    current = int(battle[hazard_offset])
    new_hazards = current

    if hazard_type == HAZARD_STEALTH_ROCK:
        new_hazards = set_stealth_rock(new_hazards)
    elif hazard_type == HAZARD_SPIKES:
        layers = get_spikes_layers(new_hazards)
        new_hazards = set_spikes(new_hazards, layers + 1)
    elif hazard_type == HAZARD_TOXIC_SPIKES:
        layers = get_toxic_spikes_layers(new_hazards)
        new_hazards = set_toxic_spikes(new_hazards, layers + 1)
    elif hazard_type == HAZARD_STICKY_WEB:
        new_hazards = set_sticky_web(new_hazards)

    # Re-bias to int16 range.
    val = int(new_hazards) & 0xFFFF
    if val >= 0x8000:
        val -= 0x10000
    battle[hazard_offset] = val


def apply_hazard_damage_on_switch(
    battle: np.ndarray,
    pokemon_offset: int,
    hazard_offset: int,
    game_data=None,
) -> None:
    """Port of _apply_hazard_damage_on_switch (line ~7876).

    Applies stealth rock damage, spikes damage, toxic spikes status,
    and sticky web speed drop. Heavy-Duty Boots grants total immunity;
    Magic Guard prevents stealth rock and spikes damage but not status
    from toxic spikes.
    """
    pokemon_offset = int(pokemon_offset)
    hazard_offset = int(hazard_offset)

    item = int(battle[pokemon_offset + 6])
    has_boots = item == ITEM_HEAVY_DUTY_BOOTS
    ability = int(battle[pokemon_offset + 5])
    has_magic_guard = ability == ABILITY_MAGIC_GUARD

    hazards = int(battle[hazard_offset])
    hp = int(battle[pokemon_offset + 1])
    max_hp = int(battle[pokemon_offset + 2])
    types = int(battle[pokemon_offset + 4])
    type1 = types & 0xFF
    type2 = (types >> 8) & 0xFF

    # ----- Stealth Rock damage: maxhp * 2^typeMod / 8 -----
    # Showdown: data/moves.ts stealthrock condition.onSwitchIn uses
    # pokemon.runEffectiveness(stealthrock) -> damage(maxhp * 2^typeMod / 8).
    # We emulate runEffectiveness via MODERN_TYPE_CHART[def, atk=ROCK], and
    # guard against double-application for monotype mons where type2 == type1
    # (the damage calc and abilities.py both use this same guard).
    has_sr = get_stealth_rock(hazards)
    sr_damage = 0
    if has_sr > 0 and not has_boots and not has_magic_guard:
        sr_eff1 = float(MODERN_TYPE_CHART[type1, TYPE_ROCK])
        sr_eff2 = 1.0 if type2 == type1 else float(MODERN_TYPE_CHART[type2, TYPE_ROCK])
        sr_mult = sr_eff1 * sr_eff2
        sr_damage_raw = int(max_hp * sr_mult / 8)
        sr_damage = max(sr_damage_raw, 1)

    # ----- Grounding check (Showdown sim/pokemon.ts isGrounded) -----
    # Iron Ball forces Flying-types to be grounded; Air Balloon floats the
    # holder regardless of type/ability; Levitate and Flying-type ungrounds.
    # Magnet Rise / Telekinesis / Smackdown / Gravity / Ingrain are not yet
    # modeled in pokepy.
    is_flying = (type1 == TYPE_FLYING) or (type2 == TYPE_FLYING)
    has_levitate = ability == ABILITY_LEVITATE
    has_iron_ball = item == _ITEM_IRON_BALL
    has_air_balloon = item == ITEM_AIR_BALLOON
    if has_iron_ball:
        is_grounded = True
    elif has_air_balloon:
        is_grounded = False
    else:
        is_grounded = (not is_flying) and (not has_levitate)

    # ----- Spikes damage: 1/8, 1/6, 1/4 by layer count -----
    spikes_layers = get_spikes_layers(hazards)
    spikes_damage = 0
    if is_grounded and not has_boots and not has_magic_guard and spikes_layers > 0:
        if spikes_layers == 1:
            spikes_raw = max_hp // 8
        elif spikes_layers == 2:
            spikes_raw = max_hp // 6
        else:
            spikes_raw = max_hp // 4
        spikes_damage = max(spikes_raw, 1)

    # ----- Toxic Spikes: poison or toxic; absorbed by Poison types -----
    # Showdown: data/moves.ts toxicspikes.onSwitchIn -> `pokemon.trySetStatus`
    # which goes through `setStatus` -> `runStatusImmunity` AND the target's
    # side `SetStatus` event (which Safeguard intercepts). sim/pokemon.ts:1657
    # and data/moves.ts:safeguard:16171. Pokepy previously only checked
    # Poison/Steel/HDB — missing Immunity ability, Safeguard, Misty Terrain,
    # and Purifying Salt / Comatose blanket immunities.
    from pokepy.core.constants import (
        ABILITY_IMMUNITY as _ABI_IMMUN,
        ABILITY_PURIFYING_SALT as _ABI_PSALT,
        TERRAIN_MISTY as _TMISTY,
        F_SCREENS_0 as _FS0_TS,
        F_SCREENS_1 as _FS1_TS,
        F_TERRAIN as _FTER_TS,
        SCREEN_SAFEGUARD_SHIFT as _SAFE_TS,
        OFF_SIDE1 as _OFF_SIDE1_TS,
    )

    _ABILITY_COMATOSE_TS = 213  # see pokepy/effects/status_apply.py:63
    is_poison = (type1 == TYPE_POISON) or (type2 == TYPE_POISON)
    is_steel = (type1 == TYPE_STEEL) or (type2 == TYPE_STEEL)
    tspikes_layers = get_toxic_spikes_layers(hazards)

    # Side-condition / terrain / ability immunities (only gate the STATUS
    # application — Poison-type absorption still runs below).
    target_is_side0 = int(pokemon_offset) < _OFF_SIDE1_TS
    screens_word = int(battle[OFF_FIELD + (_FS0_TS if target_is_side0 else _FS1_TS)])
    safeguard_active = ((screens_word >> _SAFE_TS) & 0x3) > 0
    current_terrain_ts = int(battle[OFF_FIELD + _FTER_TS])
    misty_grounded_immune = is_grounded and (current_terrain_ts == _TMISTY)
    ability_immune = ability in (_ABI_IMMUN, _ABI_PSALT, _ABILITY_COMATOSE_TS)

    can_be_poisoned = (
        is_grounded
        and (not is_poison)
        and (not is_steel)
        and (not has_boots)
        and (not safeguard_active)
        and (not misty_grounded_immune)
        and (not ability_immune)
    )
    current_status = get_status(int(battle[pokemon_offset + 12]))
    has_no_status = current_status == STATUS_NONE

    if can_be_poisoned and has_no_status and tspikes_layers > 0:
        new_status_val = STATUS_POISON if tspikes_layers == 1 else STATUS_TOXIC
        battle[pokemon_offset + 12] = set_status(new_status_val, 0)

    # Poison-type absorb: in Showdown (data/moves.ts toxicspikes.onSwitchIn)
    # the `hasType('Poison')` check runs before the Heavy-Duty Boots guard,
    # so a grounded Poison-type with HDB still clears the layers.
    if is_poison and is_grounded and tspikes_layers > 0:
        new_hazards = set_toxic_spikes(hazards, 0)
        nh = int(new_hazards) & 0xFFFF
        if nh >= 0x8000:
            nh -= 0x10000
        battle[hazard_offset] = nh
        hazards = nh  # update local view

    # ----- Sticky Web: opponent-caused -1 Speed boost event -----
    # Showdown routes Sticky Web through `this.boost({spe: -1}, pokemon,
    # pokemon.side.foe.active[0], stickyweb)` on switch-in, so the holder's
    # full onTryBoost/onAfterEachBoost pipeline runs here too.
    has_web = get_sticky_web(hazards)
    if is_grounded and (not has_boots) and has_web > 0:
        target_is_side0 = int(pokemon_offset) < OFF_SIDE1
        source_slot = int(
            battle[OFF_META + (M_ACTIVE1 if target_is_side0 else M_ACTIVE0)]
        )
        source_base = OFF_SIDE1 if target_is_side0 else OFF_SIDE0
        source_offset = source_base + source_slot * POKEMON_SIZE
        apply_direct_stat_changes(
            battle,
            source_offset,
            int(pokemon_offset),
            [0, 0, 0, 0, -1, 0, 0],
            stat_target=1,
        )

    # ----- Apply combined SR + Spikes damage -----
    total_damage = sr_damage + spikes_damage
    new_hp = max(0, hp - total_damage)
    battle[pokemon_offset + 1] = new_hp
    if new_hp == 0:
        # Entry-hazard KOs immediately count as a fainted teammate for later
        # same-turn effects such as Supreme Overlord on the next replacement.
        flags = int(battle[pokemon_offset + 15]) | 0x1
        if flags >= 0x8000:
            flags -= 0x10000
        battle[pokemon_offset + 15] = flags
