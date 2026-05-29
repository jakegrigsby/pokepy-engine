"""Recovery and team-heal moves."""

from __future__ import annotations

from pokepy.effects._common import np, MultiFormatState, Gen5PRNG
from pokepy.core.bitpack import set_status
from pokepy.core.constants import (
    EXT_VOL_HEAL_BLOCK,
    F_EXTENDED_VOLATILE_0,
    F_EXTENDED_VOLATILE_1,
    POKEMON_SIZE,
    OFF_FIELD,
    OFF_META,
    OFF_SIDE1,
    F_WEATHER,
    M_WISH_TURNS_0,
    M_WISH_TURNS_1,
    M_WISH_HP_0,
    M_WISH_HP_1,
    EFFECT_RECOVERY,
    WEATHER_SUN,
    WEATHER_RAIN,
    WEATHER_SAND,
    WEATHER_SNOW,
    STATUS_NONE,
    STATUS_SLEEP,
    MOVE_REST,
    MOVE_HEAL_BELL,
    MOVE_AROMATHERAPY,
)

# Move IDs not exported by constants.py — declared locally to mirror the Showdown reference usage.
MOVE_MORNING_SUN = 234
MOVE_SYNTHESIS = 235
MOVE_MOONLIGHT = 236
MOVE_SHORE_UP = 659
MOVE_WISH = 273


def _has_heal_block(battle: np.ndarray, user_offset: int) -> bool:
    """Return True iff the active user is currently under Heal Block."""
    uoff = int(user_offset)
    ext_off = OFF_FIELD + (
        F_EXTENDED_VOLATILE_0 if uoff < OFF_SIDE1 else F_EXTENDED_VOLATILE_1
    )
    ext = int(battle[ext_off]) & 0xFFFF
    return (ext & EXT_VOL_HEAL_BLOCK) != 0


def can_rest_succeed(battle: np.ndarray, user_offset: int) -> bool:
    """Mirror Showdown Rest legality closely enough for move execution.

    Rest fails at full HP, and otherwise it succeeds only if the user can set
    itself to sleep. Unlike Toxic Orb / Flame Orb, Rest may replace a non-sleep
    existing status, so we allow that specific replacement path here.
    """
    uoff = int(user_offset)
    current_hp = int(battle[uoff + 1])
    max_hp = int(battle[uoff + 2])
    if current_hp >= max_hp:
        return False

    from pokepy.effects.status_apply import can_set_self_status

    return can_set_self_status(
        battle,
        uoff,
        STATUS_SLEEP,
        allow_existing_status=True,
    )


def apply_recovery_from_move(
    battle: np.ndarray,
    move_id: int,
    user_offset: int,
    hit: bool,
    game_data,
    move_effects,
    gen5_prng: Gen5PRNG | None = None,
) -> None:
    """Port of _apply_recovery_from_move (line ~9941).

    Heals the user by a percentage of max HP. Weather-dependent moves
    (Morning Sun, Synthesis, Moonlight, Shore Up) heal 66% in sun, 25% in
    rain/sand/snow, 50% otherwise. Wish queues a delayed heal. Rest sets
    sleep on the user.
    """
    move_id = int(move_id)
    uoff = int(user_offset)
    hit = bool(hit)

    effect_type = int(move_effects.effect_type[move_id])
    is_recovery = effect_type == EFFECT_RECOVERY

    heal_pct = float(int(move_effects.heal[move_id]))

    current_hp = int(battle[uoff + 1])
    max_hp = int(battle[uoff + 2])
    heal_blocked = _has_heal_block(battle, uoff)

    # Weather-dependent healing moves all use `pokemon.effectiveWeather()` /
    # `field.isWeather()` in Showdown, both of which return '' under Air Lock
    # / Cloud Nine. Synthesis / Morning Sun / Moonlight additionally use
    # `pokemon.effectiveWeather()` which also returns '' if the user holds
    # Utility Umbrella (sun/rain only). Shore Up uses `field.isWeather`
    # (no umbrella check), and only varies between sand and not-sand.
    # Sources: data/moves.ts:synthesis L19459 (`pokemon.effectiveWeather()`)
    # and data/moves.ts:shoreup L16977 (`this.field.isWeather('sandstorm')`).
    raw_weather = int(battle[OFF_FIELD + F_WEATHER])
    # Air Lock / Cloud Nine on either active mon → effective weather is none.
    from pokepy.core.constants import (
        ABILITY_AIR_LOCK as _AL_RC,
        ABILITY_CLOUD_NINE as _CN_RC,
        OFF_SIDE0 as _OS0_RC,
        OFF_SIDE1 as _OS1_RC,
        OFF_META as _OM_RC,
        M_ACTIVE0 as _MA0_RC,
        M_ACTIVE1 as _MA1_RC,
        POKEMON_SIZE as _PS_RC,
        ITEM_UTILITY_UMBRELLA as _UMB_RC,
        WEATHER_PRIMORDIAL_SEA as _WPS_RC,
        WEATHER_DESOLATE_LAND as _WDL_RC,
    )

    _a0_rc = int(battle[_OM_RC + _MA0_RC])
    _a1_rc = int(battle[_OM_RC + _MA1_RC])
    _ab0_rc = int(battle[_OS0_RC + _a0_rc * _PS_RC + 5])
    _ab1_rc = int(battle[_OS1_RC + _a1_rc * _PS_RC + 5])
    _weather_sup_rc = _ab0_rc in (_AL_RC, _CN_RC) or _ab1_rc in (_AL_RC, _CN_RC)
    weather = 0 if _weather_sup_rc else raw_weather

    is_shore_up = move_id == MOVE_SHORE_UP
    is_solar_heal = move_id in (MOVE_MORNING_SUN, MOVE_MOONLIGHT, MOVE_SYNTHESIS)
    user_item = int(battle[uoff + 6])
    has_umbrella = user_item == _UMB_RC

    if is_shore_up:
        # Shore Up: 66.7% in sandstorm, 50% otherwise. No umbrella check.
        if weather == WEATHER_SAND:
            weather_heal_pct = 66.0
        else:
            weather_heal_pct = 50.0
    elif is_solar_heal:
        # Synthesis / Morning Sun / Moonlight use `pokemon.effectiveWeather()`,
        # which also returns '' for the user under Utility Umbrella in sun/rain
        # (but NOT under primordial/desolate, which Showdown still suppresses
        # via the same umbrella branch — see sim/pokemon.ts:2149).
        eff_weather = weather
        if has_umbrella and eff_weather in (
            WEATHER_SUN,
            WEATHER_RAIN,
            _WPS_RC,
            _WDL_RC,
        ):
            eff_weather = 0
        if eff_weather in (WEATHER_SUN, _WDL_RC):
            weather_heal_pct = 66.0
        elif eff_weather in (WEATHER_RAIN, _WPS_RC, WEATHER_SAND, WEATHER_SNOW):
            weather_heal_pct = 25.0
        else:
            weather_heal_pct = 50.0
    else:
        weather_heal_pct = heal_pct

    is_weather_heal = is_shore_up or is_solar_heal
    final_heal_pct = weather_heal_pct if is_weather_heal else heal_pct

    # Showdown uses Math.round (not floor) for Gen 5+ healing moves
    # (sim/battle-actions.ts:1028: `(gen < 5 ? Math.floor : Math.round)(amount)`).
    # Python's round() uses banker's rounding (round-half-even), while JS
    # Math.round uses round-half-up. For the half case (e.g., 161.5 → 162
    # in JS), explicitly use floor(x + 0.5) to match JS Math.round.
    import math

    heal_amount = math.floor(max_hp * final_heal_pct / 100.0 + 0.5)
    new_hp = min(current_hp + heal_amount, max_hp)

    is_rest = move_id == MOVE_REST
    rest_will_succeed = (
        is_rest and hit and (not heal_blocked) and can_rest_succeed(battle, uoff)
    )
    is_wish = move_id == MOVE_WISH
    wish_hp = math.floor(max_hp / 2 + 0.5)  # Math.round parity for Gen 5+
    is_side0 = uoff < OFF_SIDE1
    wish_turns_off = OFF_META + (M_WISH_TURNS_0 if is_side0 else M_WISH_TURNS_1)
    wish_hp_off = OFF_META + (M_WISH_HP_0 if is_side0 else M_WISH_HP_1)

    # Showdown stores Wish as a slot condition. Reusing Wish while one is
    # already pending on that slot fails as a no-op and must not overwrite the
    # queued end-of-turn heal from the earlier Wish.
    wish_pending = int(battle[wish_turns_off]) > 0
    should_set_wish = (
        is_wish and is_recovery and hit and (not heal_blocked) and not wish_pending
    )
    if should_set_wish:
        # Showdown data/moves.ts:wish stores `slot: source.position` in the
        # side condition and heals THAT slot at EOT (not whoever is active
        # then). Pack the wish-setter's team slot into bits 12-14 of wish_hp
        # (3 bits, slot 0-5 fits) so we can recover it on resolution.
        # wish_hp itself uses ~10 bits (max ~350 in OU), so bits 12+ are free.
        from pokepy.core.constants import OFF_SIDE0 as _OFF0, POKEMON_SIZE as _PS

        side_base = _OFF0 if is_side0 else OFF_SIDE1
        slot = (uoff - side_base) // _PS
        wish_hp_packed = (int(wish_hp) & 0x0FFF) | ((int(slot) & 0x7) << 12)
        # Convert to signed int16 (since storage is int16)
        if wish_hp_packed >= 0x8000:
            wish_hp_packed -= 0x10000
        battle[wish_turns_off] = 2  # Triggers end of NEXT turn
        battle[wish_hp_off] = wish_hp_packed

    should_heal = (
        is_recovery
        and hit
        and (not heal_blocked)
        and (not is_wish)
        and ((not is_rest) or rest_will_succeed)
    )
    final_hp = new_hp if should_heal else current_hp
    battle[uoff + 1] = final_hp

    # Rest: Showdown calls setStatus('slp') first, which runs sleep.onStart
    # and consumes a hidden random(2, 5) frame, then overrides the stored
    # sleep timer back to 3 in data/moves.ts:rest onHit. Pokepy also stores 3
    # to preserve the same two forced-sleep turns under our EOT decrement
    # model, but we still need to consume the same hidden onStart frame to
    # keep the PRNG stream aligned.
    if rest_will_succeed:
        if gen5_prng is not None:
            gen5_prng.random(2, 5)
        battle[uoff + 12] = set_status(STATUS_SLEEP, 3)


def apply_team_heal_status(
    battle: np.ndarray,
    move_id: int,
    user_side_offset: int,
    hit: bool,
    game_data,
    move_effects,
) -> None:
    """Port of _apply_team_heal_status (line ~10035).

    Heal Bell / Aromatherapy clear non-volatile status from all 6 Pokemon
    on the user's team.
    """
    move_id = int(move_id)
    side_off = int(user_side_offset)
    hit = bool(hit)

    is_heal_bell = move_id == MOVE_HEAL_BELL
    is_aromatherapy = move_id == MOVE_AROMATHERAPY
    should_heal = (is_heal_bell or is_aromatherapy) and hit

    if not should_heal:
        return

    for slot_idx in range(6):
        pokemon_offset = side_off + slot_idx * POKEMON_SIZE
        status_offset = pokemon_offset + 12
        battle[status_offset] = set_status(STATUS_NONE, 0)
