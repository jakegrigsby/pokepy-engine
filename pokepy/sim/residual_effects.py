"""End-of-turn residual handlers — single owner for fieldEvent('Residual')."""

from __future__ import annotations

import pokepy.effects as fx
from pokepy.core.gen_profile import GenProfile
from pokepy.effects.end_of_turn import (
    _apply_aqua_ring_ingrain_heal,
    _process_yawn,
    apply_partial_trap_damage,
    decrement_screens,
    decrement_terrain,
    decrement_trick_room,
    decrement_weather,
)
from pokepy.sim import dispatch
from pokepy.sim.views import PokemonView


def _on_status_residual(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.hp <= 0:
        return relay
    fx.apply_end_of_turn_status(
        battle.battle,
        pokemon.offset(),
        battle.game_data,
        battle.move_effects,
        battle.prng,
    )
    if pokemon.hp <= 0:
        battle.faint(pokemon, source=source)
    return relay


def _on_salt_cure_residual(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.hp <= 0:
        return relay
    from pokepy.core.constants import (
        F_EXTENDED_VOLATILE_0,
        F_EXTENDED_VOLATILE_1,
        OFF_FIELD,
    )

    ext_off = OFF_FIELD + (
        F_EXTENDED_VOLATILE_0 if pokemon.side == 0 else F_EXTENDED_VOLATILE_1
    )
    fx.apply_salt_cure_damage(
        battle.battle, pokemon.offset(), ext_off, battle.game_data
    )
    if pokemon.hp <= 0:
        battle.faint(pokemon, source=source)
    return relay


def _on_partial_trap_residual(
    battle, pokemon: PokemonView, source, effect, relay, **kw
):
    if pokemon.hp <= 0:
        return relay
    apply_partial_trap_damage(battle.battle, pokemon.offset(), pokemon.side)
    if pokemon.hp <= 0:
        battle.faint(pokemon, source=source)
    return relay


def _on_weather_healing_residual(
    battle, pokemon: PokemonView, source, effect, relay, **kw
):
    if pokemon.hp <= 0:
        return relay
    fx.apply_weather_healing(battle.battle, pokemon.offset(), battle.game_data)
    return relay


def _on_aqua_ring_ingrain_residual(
    battle, pokemon: PokemonView, source, effect, relay, **kw
):
    if pokemon.hp <= 0:
        return relay
    _apply_aqua_ring_ingrain_heal(battle.battle, pokemon.offset(), pokemon.side)
    return relay


def _on_yawn_residual(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.hp <= 0:
        return relay
    _process_yawn(battle.battle, pokemon.offset(), pokemon.side, gen5_prng=battle.prng)
    return relay


def _on_field_decrement_residual(
    battle, pokemon: PokemonView, source, effect, relay, **kw
):
    """Run once per residual pass — attached to side-0 active as anchor."""
    if pokemon.side != 0:
        return relay
    decrement_trick_room(battle.battle)
    decrement_screens(battle.battle)
    decrement_weather(battle.battle)
    decrement_terrain(battle.battle)
    fx.clear_protect_at_turn_end(battle.battle)
    fx.clear_volatile_turn_effects(battle.battle)
    return relay


def register_all(profile: GenProfile) -> None:
    fmt = int(profile.format_id)
    dispatch.register(
        fmt, "status_eot", dispatch.Residual, _on_status_residual, order=10, priority=0
    )
    dispatch.register(
        fmt, "saltcure", dispatch.Residual, _on_salt_cure_residual, order=12, priority=0
    )
    dispatch.register(
        fmt,
        "partialtrap",
        dispatch.Residual,
        _on_partial_trap_residual,
        order=13,
        priority=0,
    )
    dispatch.register(
        fmt,
        "weatherheal",
        dispatch.Residual,
        _on_weather_healing_residual,
        order=6,
        priority=0,
    )
    dispatch.register(
        fmt,
        "aquaring",
        dispatch.Residual,
        _on_aqua_ring_ingrain_residual,
        order=7,
        priority=0,
    )
    dispatch.register(
        fmt, "yawn", dispatch.Residual, _on_yawn_residual, order=8, priority=0
    )
    dispatch.register(
        fmt,
        "fielddecrement",
        dispatch.Residual,
        _on_field_decrement_residual,
        order=300,
        priority=0,
    )
