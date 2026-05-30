"""Ability event handlers — wrap pokepy/effects/abilities.py.

Handlers are registered keyed by the ability's integer id (data-driven); the
Battle event collector looks them up via ``str(mon.ability)`` so any ability
with a registered handler participates with no hardcoded id->name table.
"""

from __future__ import annotations

import pokepy.effects as fx
from pokepy.core.constants import (
    ABILITY_HYDRATION,
    ABILITY_SHED_SKIN,
    ABILITY_SPEED_BOOST,
)
from pokepy.core.gen_profile import GenProfile
from pokepy.sim import dispatch
from pokepy.sim.views import PokemonView


def _on_speed_boost_residual(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.ability != ABILITY_SPEED_BOOST or pokemon.hp <= 0:
        return relay
    fx.apply_speed_boost(battle.battle, pokemon.offset(), battle.game_data)
    return relay


def _on_shed_skin_hydration_residual(
    battle, pokemon: PokemonView, source, effect, relay, **kw
):
    if pokemon.hp <= 0:
        return relay
    fx.apply_shed_skin_hydration(
        battle.battle, pokemon.offset(), battle.game_data, battle.prng
    )
    return relay


def register_all(profile: GenProfile) -> None:
    if not profile.has_abilities:
        return
    fmt = int(profile.format_id)
    # Speed Boost: +1 Spe at end of turn (residual order 28 in Showdown).
    dispatch.register(
        fmt,
        str(ABILITY_SPEED_BOOST),
        dispatch.Residual,
        _on_speed_boost_residual,
        order=28,
        priority=0,
    )
    # Shed Skin (33% cure) / Hydration (cure in rain) at residual order 5.
    dispatch.register(
        fmt,
        str(ABILITY_SHED_SKIN),
        dispatch.Residual,
        _on_shed_skin_hydration_residual,
        order=5,
        priority=0,
    )
    dispatch.register(
        fmt,
        str(ABILITY_HYDRATION),
        dispatch.Residual,
        _on_shed_skin_hydration_residual,
        order=5,
        priority=0,
    )
