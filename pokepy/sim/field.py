"""Weather, terrain, and hazard field handlers."""

from __future__ import annotations

import pokepy.effects as fx
from pokepy.core.constants import (
    F_HAZARDS_0,
    F_HAZARDS_1,
    F_WEATHER,
    OFF_FIELD,
    WEATHER_NONE,
)
from pokepy.core.gen_profile import GenProfile
from pokepy.sim import dispatch
from pokepy.sim.views import PokemonView


def _on_switch_in_hazards(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.hp <= 0:
        return relay
    side = pokemon.side
    hazards_off = F_HAZARDS_0 if side == 0 else F_HAZARDS_1
    fx.apply_hazard_damage_on_switch(
        battle.battle, pokemon.offset(), OFF_FIELD + hazards_off
    )
    return relay


def _on_weather_residual(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if int(battle.battle[OFF_FIELD + F_WEATHER]) == WEATHER_NONE:
        return relay
    if pokemon.hp <= 0:
        return relay
    fx.apply_weather_damage(battle.battle, pokemon.offset(), battle.game_data)
    return relay


def _on_fieldterrain_residual(
    battle, pokemon: PokemonView, source, effect, relay, **kw
):
    if pokemon.hp <= 0:
        return relay
    fx.apply_grassy_terrain_healing(battle.battle, pokemon.offset(), battle.game_data)
    return relay


def register_all(profile: GenProfile) -> None:
    fmt = int(profile.format_id)
    # Entry hazards fire on the switching mon's SwitchIn event.
    if profile.enabled_hazards:
        dispatch.register(
            fmt, "hazards", dispatch.SwitchIn, _on_switch_in_hazards, priority=0
        )
    # Weather residual damage (sand/hail) at residual order 1.
    if profile.has_terrain or profile.gen >= 2:
        dispatch.register(
            fmt, "weather", dispatch.Residual, _on_weather_residual, order=1, priority=0
        )
    # Grassy Terrain EOT heal (order 6 in Showdown terrain.ts).
    if profile.has_terrain:
        dispatch.register(
            fmt,
            "fieldterrain",
            dispatch.Residual,
            _on_fieldterrain_residual,
            order=6,
            priority=0,
        )
