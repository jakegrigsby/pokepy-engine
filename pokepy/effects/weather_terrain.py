"""Weather, terrain, trick room move effects + type multipliers / damage / healing.

(interleaved sections):
- _apply_weather_from_move        (8752)
- _apply_weather_damage           (8778)
- _get_weather_type_multiplier    (8869)
- _apply_terrain_from_move        (8906)
- _apply_trick_room_from_move     (8929)
- _get_terrain_type_multiplier    (8959)
- _apply_grassy_terrain_healing   (9033)
- _apply_weather_healing          (7763)
"""

from __future__ import annotations

from pokepy.effects._common import np, MultiFormatState, Gen5PRNG
from pokepy.effects.grounding import is_grounded
from pokepy.core.constants import (
    OFF_FIELD,
    OFF_META,
    OFF_MOVES,
    OFF_SIDE1,
    F_WEATHER,
    F_TERRAIN,
    F_TRICK_ROOM,
    F_VOLATILE_0,
    F_VOLATILE_1,
    F_EXTENDED_VOLATILE_0,
    F_EXTENDED_VOLATILE_1,
    M_ACTIVE_MOVE_ACTIONS_0,
    M_ACTIVE_MOVE_ACTIONS_1,
    M_WEATHER_TURNS,
    M_TERRAIN_TURNS,
    ACTIVE_MOVE_ACTIONS_SEMI_INVUL,
    WEATHER_NONE,
    WEATHER_SUN,
    WEATHER_RAIN,
    WEATHER_SAND,
    WEATHER_SNOW,
    TERRAIN_NONE,
    TERRAIN_ELECTRIC,
    TERRAIN_GRASSY,
    TERRAIN_PSYCHIC,
    TERRAIN_MISTY,
    TYPE_FIRE,
    TYPE_WATER,
    TYPE_ELECTRIC,
    TYPE_GRASS,
    TYPE_PSYCHIC,
    TYPE_DRAGON,
    TYPE_FLYING,
    TYPE_GROUND,
    TYPE_ROCK,
    TYPE_STEEL,
    ABILITY_DRY_SKIN,
    ABILITY_LEVITATE,
    ABILITY_MAGIC_GUARD,
    ABILITY_OVERCOAT,
    ABILITY_ICE_BODY,
    ABILITY_RAIN_DISH,
    ITEM_DAMP_ROCK,
    ITEM_HEAT_ROCK,
    ITEM_ICY_ROCK,
    ITEM_SMOOTH_ROCK,
    ITEM_TERRAIN_EXTENDER,
    ITEM_SAFETY_GOGGLES,
    ITEM_UTILITY_UMBRELLA,
    ITEM_AIR_BALLOON,
    EFFECT_TRICK_ROOM,
    EXT_VOL_HEAL_BLOCK,
)

# -----------------------------------------------------------------------------
# Move-triggered weather / terrain / trick room
# -----------------------------------------------------------------------------


def _has_heal_block(battle: np.ndarray, pokemon_offset: int) -> bool:
    poff = int(pokemon_offset)
    ext_off = OFF_FIELD + (
        F_EXTENDED_VOLATILE_0 if poff < OFF_SIDE1 else F_EXTENDED_VOLATILE_1
    )
    ext = int(battle[ext_off]) & 0xFFFF
    return (ext & EXT_VOL_HEAL_BLOCK) != 0


def _is_grounded_for_terrain_residual(battle: np.ndarray, pokemon_offset: int) -> bool:
    """Approximate Showdown's Pokemon.isGrounded() for terrain residuals.

    Pokepy currently models Flying typing, Levitate, Air Balloon, and Iron Ball
    for groundedness. Gravity, Ingrain, Smack Down, Magnet Rise, Telekinesis,
    and Roost's ???/Flying edge case are not modeled here yet.
    """
    return is_grounded(battle, pokemon_offset)


def _is_semi_invulnerable(battle: np.ndarray, pokemon_offset: int) -> bool:
    poff = int(pokemon_offset)
    actions_off = OFF_MOVES + (
        M_ACTIVE_MOVE_ACTIONS_0 if poff < OFF_SIDE1 else M_ACTIVE_MOVE_ACTIONS_1
    )
    return (int(battle[actions_off]) & ACTIVE_MOVE_ACTIONS_SEMI_INVUL) != 0


def apply_weather_from_move(
    battle: np.ndarray,
    move_id: int,
    hit: bool,
    game_data,
    move_effects,
    user_offset: int = None,
) -> None:
    """Apply weather from a move. Weather rocks extend duration to 8 turns.

    Port of _apply_weather_from_move (line ~8752). Also blocks move-set
    weather from overriding any currently-active primal weather (Desolate
    Land / Primordial Sea / Delta Stream). Showdown:
    data/abilities.ts:926 deltastream.onAnySetWeather,
    data/abilities.ts:950 desolateland.onAnySetWeather,
    data/abilities.ts:974 primordialsea.onAnySetWeather — all reject non-
    primal setWeather while the matching primal is up.
    """
    from pokepy.core.constants import (
        WEATHER_PRIMORDIAL_SEA as _WPS,
        WEATHER_DESOLATE_LAND as _WDL,
        WEATHER_DELTA_STREAM as _WDS,
    )

    weather = int(move_effects.weather[int(move_id)])
    should_set = bool(hit) and (weather > 0)
    if not should_set:
        return

    cur_weather = int(battle[OFF_FIELD + F_WEATHER])
    # Primal weathers reject any non-primal override. Move-set weathers
    # (Rain Dance / Sunny Day / Sandstorm / Snowscape) are always non-primal.
    primal_weathers = (_WPS, _WDL, _WDS)
    if cur_weather in primal_weathers and weather not in primal_weathers:
        return

    battle[OFF_FIELD + F_WEATHER] = weather

    base_turns = 5
    if user_offset is not None:
        user_item = int(battle[int(user_offset) + 6])
        has_rock = (
            (weather == WEATHER_RAIN and user_item == ITEM_DAMP_ROCK)
            or (weather == WEATHER_SUN and user_item == ITEM_HEAT_ROCK)
            or (weather == WEATHER_SNOW and user_item == ITEM_ICY_ROCK)
            or (weather == WEATHER_SAND and user_item == ITEM_SMOOTH_ROCK)
        )
        if has_rock:
            base_turns = 8
    battle[OFF_META + M_WEATHER_TURNS] = base_turns


def apply_terrain_from_move(
    battle: np.ndarray,
    move_id: int,
    hit: bool,
    game_data,
    move_effects,
    user_offset: int = None,
) -> None:
    """Apply terrain from a move. Terrain Extender extends to 8 turns.

    Port of _apply_terrain_from_move (line ~8906).
    """
    terrain = int(move_effects.terrain[int(move_id)])
    should_set = bool(hit) and (terrain > 0)
    if should_set:
        battle[OFF_FIELD + F_TERRAIN] = terrain

        base_turns = 5
        if user_offset is not None:
            user_item = int(battle[int(user_offset) + 6])
            if user_item == ITEM_TERRAIN_EXTENDER:
                base_turns = 8
        battle[OFF_META + M_TERRAIN_TURNS] = base_turns

        from pokepy.core.constants import (
            OFF_SIDE0,
            OFF_SIDE1,
            POKEMON_SIZE,
            M_ACTIVE0,
            M_ACTIVE1,
        )
        from pokepy.effects.abilities import apply_terrain_seed_item

        active0 = int(battle[OFF_META + M_ACTIVE0])
        active1 = int(battle[OFF_META + M_ACTIVE1])
        apply_terrain_seed_item(battle, OFF_SIDE0 + active0 * POKEMON_SIZE)
        apply_terrain_seed_item(battle, OFF_SIDE1 + active1 * POKEMON_SIZE)


def apply_trick_room_from_move(
    battle: np.ndarray,
    move_id: int,
    hit: bool,
    game_data,
    move_effects,
) -> None:
    """Apply Trick Room toggle from a move.

    Trick Room toggles - if already active, using it again ends it.
    Duration: 5 turns (including the turn it was used).

    Port of _apply_trick_room_from_move (line ~8929).
    """
    effect = int(move_effects.effect_type[int(move_id)])
    is_trick_room = effect == EFFECT_TRICK_ROOM
    if is_trick_room and bool(hit):
        current_turns = int(battle[OFF_FIELD + F_TRICK_ROOM])
        battle[OFF_FIELD + F_TRICK_ROOM] = 0 if current_turns > 0 else 5


# -----------------------------------------------------------------------------
# Type multipliers (consumed by damage calc)
# -----------------------------------------------------------------------------


def get_weather_type_multiplier(weather: int, move_type: int) -> float:
    """Return type multiplier based on current weather.

    Sun: Fire +50%, Water -50%
    Rain: Water +50%, Fire -50%
    Other / no weather: 1.0

    Port of _get_weather_type_multiplier (line ~8869). Pokepy's variant
    takes the scalar weather and move_type directly (no battle array).
    """
    weather = int(weather)
    move_type = int(move_type)
    if weather == WEATHER_SUN:
        if move_type == TYPE_FIRE:
            return 1.5
        if move_type == TYPE_WATER:
            return 0.5
        return 1.0
    if weather == WEATHER_RAIN:
        if move_type == TYPE_WATER:
            return 1.5
        if move_type == TYPE_FIRE:
            return 0.5
        return 1.0
    return 1.0


def get_terrain_type_multiplier(terrain: int, move_type: int, grounded: bool) -> float:
    """Return type multiplier for terrain effects.

    Pokepy's signature collapses the batched `(move_type, atk_offset,
    def_offset, move_id)` shape into the most common case: a single
    grounded flag for the relevant Pokemon. The damage path queries this
    helper twice — once for an attacker grounding (offensive boosts) and
    once for a defender grounding (Misty / Grassy halving). The simplified
    version returns the offensive boost when `grounded=True` for boost
    types, and the defensive halver when `grounded=True` for the halving
    cases. Earthquake / Bulldoze handling is left to damage_gen9 which
    multiplies in the 0.5 itself.

    Port of _get_terrain_type_multiplier (line ~8959).
    """
    terrain = int(terrain)
    move_type = int(move_type)
    if not grounded:
        return 1.0

    if terrain == TERRAIN_ELECTRIC and move_type == TYPE_ELECTRIC:
        return 1.3
    if terrain == TERRAIN_GRASSY and move_type == TYPE_GRASS:
        return 1.3
    if terrain == TERRAIN_PSYCHIC and move_type == TYPE_PSYCHIC:
        return 1.3
    if terrain == TERRAIN_MISTY and move_type == TYPE_DRAGON:
        return 0.5
    return 1.0


# -----------------------------------------------------------------------------
# End-of-turn weather damage / healing
# -----------------------------------------------------------------------------


def apply_weather_damage(
    battle: np.ndarray,
    pokemon_offset: int,
    game_data,
) -> None:
    """Apply end-of-turn weather damage to a Pokemon.

    Sandstorm: 1/16 max HP damage (immune: Rock, Ground, Steel)
    Snow: No damage in Gen 9 (was hail damage in older gens)

    Magic Guard / Overcoat / Safety Goggles all prevent weather damage.

    Port of _apply_weather_damage (line ~8778).
    """
    pokemon_offset = int(pokemon_offset)
    # Air Lock / Cloud Nine on either active mon suppresses weather effects.
    from pokepy.core.constants import (
        ABILITY_AIR_LOCK,
        ABILITY_CLOUD_NINE,
        OFF_SIDE0 as _OS0,
        OFF_SIDE1 as _OS1,
        OFF_META as _OM,
        M_ACTIVE0 as _MA0,
        M_ACTIVE1 as _MA1,
        POKEMON_SIZE as _PS,
    )

    a0 = int(battle[_OM + _MA0])
    a1 = int(battle[_OM + _MA1])
    ab0 = int(battle[_OS0 + a0 * _PS + 5])
    ab1 = int(battle[_OS1 + a1 * _PS + 5])
    if ab0 in (ABILITY_AIR_LOCK, ABILITY_CLOUD_NINE) or ab1 in (
        ABILITY_AIR_LOCK,
        ABILITY_CLOUD_NINE,
    ):
        return
    weather = int(battle[OFF_FIELD + F_WEATHER])
    if weather != WEATHER_SAND:
        return

    hp = int(battle[pokemon_offset + 1])
    if hp <= 0:
        return

    max_hp = int(battle[pokemon_offset + 2])
    types = int(battle[pokemon_offset + 4])
    type1 = types & 0xFF
    type2 = (types >> 8) & 0xFF

    is_rock = (type1 == TYPE_ROCK) or (type2 == TYPE_ROCK)
    is_ground = (type1 == TYPE_GROUND) or (type2 == TYPE_GROUND)
    is_steel = (type1 == TYPE_STEEL) or (type2 == TYPE_STEEL)
    if is_rock or is_ground or is_steel:
        return

    ability = int(battle[pokemon_offset + 5])
    item = int(battle[pokemon_offset + 6])
    # Utility Umbrella also blocks weather effects on the holder
    from pokepy.core.constants import (
        ITEM_UTILITY_UMBRELLA,
        ABILITY_SAND_VEIL,
        ABILITY_SAND_RUSH,
    )

    # Sand Force / Sand Rush / Sand Veil have `onImmunity('sandstorm')`
    # in Showdown, exempting their holders from sand damage. Sand Force
    # is local id 159 and isn't in constants.py.
    _ABILITY_SAND_FORCE_LOC = 159
    if (
        ability == ABILITY_MAGIC_GUARD
        or ability == ABILITY_OVERCOAT
        or ability == ABILITY_SAND_VEIL
        or ability == ABILITY_SAND_RUSH
        or ability == _ABILITY_SAND_FORCE_LOC
        or item == ITEM_SAFETY_GOGGLES
    ):
        return

    sand_damage = max_hp // 16
    new_hp = max(0, hp - sand_damage)
    battle[pokemon_offset + 1] = new_hp


def apply_grassy_terrain_healing(
    battle: np.ndarray,
    pokemon_offset: int,
    game_data,
) -> None:
    """Apply Grassy Terrain healing at end of turn (1/16 max HP).

    Only heals grounded, alive Pokemon while Grassy Terrain is active.
    Minimum heal of 1 HP.

    Port of _apply_grassy_terrain_healing (line ~9033).
    """
    pokemon_offset = int(pokemon_offset)
    terrain = int(battle[OFF_FIELD + F_TERRAIN])
    if terrain != TERRAIN_GRASSY:
        return

    current_hp = int(battle[pokemon_offset + 1])
    if current_hp <= 0:
        return
    if _has_heal_block(battle, pokemon_offset):
        return

    if not _is_grounded_for_terrain_residual(battle, pokemon_offset):
        return
    if _is_semi_invulnerable(battle, pokemon_offset):
        return

    max_hp = int(battle[pokemon_offset + 2])
    heal_amount = max(1, max_hp // 16)
    new_hp = min(max_hp, current_hp + heal_amount)
    battle[pokemon_offset + 1] = new_hp


def apply_weather_healing(
    battle: np.ndarray,
    pokemon_offset: int,
    game_data,
) -> None:
    """Apply end-of-turn weather ability healing / damage.

    Showdown abilities.ts onWeather hooks (gen 9):
    - Ice Body (hail/snowscape):      +1/16 max HP
    - Rain Dish (raindance):          +1/16 max HP, suppressed by Utility Umbrella
    - Dry Skin (raindance):           +1/8  max HP, suppressed by Utility Umbrella
    - Dry Skin (sunnyday):            -1/8  max HP, suppressed by Utility Umbrella
    Solar Power sun damage lives in misc_eot_abilities.py.

    Air Lock / Cloud Nine on either active mon suppresses ALL weather
    effects (sim/pokemon.ts effectiveWeather + battle `suppressingWeather`).
    Magic Guard blocks the Dry Skin sun damage (non-Move damage guard).

    Port of _apply_weather_healing (line ~7763).
    """
    pokemon_offset = int(pokemon_offset)
    ability = int(battle[pokemon_offset + 5])
    hp = int(battle[pokemon_offset + 1])
    if hp <= 0:
        return

    # Air Lock / Cloud Nine on either active mon suppresses weather effects.
    from pokepy.core.constants import (
        ABILITY_AIR_LOCK,
        ABILITY_CLOUD_NINE,
        OFF_SIDE0 as _OS0,
        OFF_SIDE1 as _OS1,
        OFF_META as _OM,
        M_ACTIVE0 as _MA0,
        M_ACTIVE1 as _MA1,
        POKEMON_SIZE as _PS,
    )

    a0 = int(battle[_OM + _MA0])
    a1 = int(battle[_OM + _MA1])
    ab0 = int(battle[_OS0 + a0 * _PS + 5])
    ab1 = int(battle[_OS1 + a1 * _PS + 5])
    if ab0 in (ABILITY_AIR_LOCK, ABILITY_CLOUD_NINE) or ab1 in (
        ABILITY_AIR_LOCK,
        ABILITY_CLOUD_NINE,
    ):
        return
    weather = int(battle[OFF_FIELD + F_WEATHER])
    item = int(battle[pokemon_offset + 6])
    has_umbrella = item == ITEM_UTILITY_UMBRELLA

    has_ice_body = ability == ABILITY_ICE_BODY
    has_rain_dish = ability == ABILITY_RAIN_DISH
    has_dry_skin = ability == ABILITY_DRY_SKIN

    max_hp = int(battle[pokemon_offset + 2])
    heal_blocked = _has_heal_block(battle, pokemon_offset)

    # Ice Body: heal 1/16 in snow. Showdown data/abilities.ts:1899 checks
    # `effect.id === 'hail' || 'snowscape'`. Showdown's `heal()` enforces
    # minimum 1 HP. Utility umbrella is irrelevant (snow not sun/rain).
    from pokepy.core.constants import (
        WEATHER_PRIMORDIAL_SEA as _WPS_WT,
        WEATHER_DESOLATE_LAND as _WDL_WT,
    )

    if has_ice_body and weather == WEATHER_SNOW and not heal_blocked:
        heal_amount = max(1, max_hp // 16)
        battle[pokemon_offset + 1] = min(max_hp, hp + heal_amount)
        return

    # Rain Dish: heal 1/16 in rain or primordial sea, suppressed by utility
    # umbrella. Showdown data/abilities.ts:3680.
    if (
        has_rain_dish
        and weather in (WEATHER_RAIN, _WPS_WT)
        and not has_umbrella
        and not heal_blocked
    ):
        heal_amount = max(1, max_hp // 16)
        battle[pokemon_offset + 1] = min(max_hp, hp + heal_amount)
        return

    # Dry Skin: +1/8 HP in rain/primordial, -1/8 HP in sun/desolate; umbrella
    # suppresses both. Showdown data/abilities.ts:1091-1098.
    if has_dry_skin and not has_umbrella:
        if weather in (WEATHER_RAIN, _WPS_WT) and not heal_blocked:
            heal_amount = max(1, max_hp // 8)
            battle[pokemon_offset + 1] = min(max_hp, hp + heal_amount)
            return
        if weather in (WEATHER_SUN, _WDL_WT):
            # Magic Guard blocks non-Move damage; Dry Skin sun damage is a
            # non-Move damage source.
            if ability == ABILITY_MAGIC_GUARD:
                return
            dmg = max(1, max_hp // 8)
            battle[pokemon_offset + 1] = max(0, hp - dmg)
            return


# -----------------------------------------------------------------------------
# Salt Cure (lives in volatiles.py too — keep this body in sync if either
# location is updated). Port of _apply_salt_cure_damage (line ~8824).
# -----------------------------------------------------------------------------


def apply_salt_cure_damage(
    battle: np.ndarray,
    pokemon_offset: int,
    ext_vol_offset: int,
    game_data,
) -> None:
    """Apply Salt Cure end-of-turn damage.

    Salt Cure deals 1/8 max HP per turn (1/4 for Water or Steel types).
    Magic Guard prevents it.

    Port of _apply_salt_cure_damage (line ~8824).
    """
    from pokepy.core.constants import EXT_VOL_SALT_CURE

    pokemon_offset = int(pokemon_offset)
    ext_vol_offset = int(ext_vol_offset)
    ext_vol = int(battle[ext_vol_offset])
    if (ext_vol & EXT_VOL_SALT_CURE) == 0:
        return

    hp = int(battle[pokemon_offset + 1])
    if hp <= 0:
        return

    ability = int(battle[pokemon_offset + 5])
    if ability == ABILITY_MAGIC_GUARD:
        return

    max_hp = int(battle[pokemon_offset + 2])
    types = int(battle[pokemon_offset + 4])
    type1 = types & 0xFF
    type2 = (types >> 8) & 0xFF

    is_water = (type1 == TYPE_WATER) or (type2 == TYPE_WATER)
    is_steel = (type1 == TYPE_STEEL) or (type2 == TYPE_STEEL)
    salt_damage = max_hp // 4 if (is_water or is_steel) else max_hp // 8
    salt_damage = max(1, salt_damage)

    battle[pokemon_offset + 1] = max(0, hp - salt_damage)
