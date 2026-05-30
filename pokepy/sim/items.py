"""Item event handlers — wrap pokepy/effects/items.py.

Items are registered keyed by integer id (data-driven). ``berries`` is a
bundle collected on the Update event for any held item.
"""

from __future__ import annotations

import pokepy.effects as fx
from pokepy.core.constants import (
    ITEM_BLACK_SLUDGE,
    ITEM_LEFTOVERS,
)
from pokepy.core.gen_profile import GenProfile
from pokepy.sim import dispatch
from pokepy.sim.views import PokemonView


def _on_update_berries(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.hp <= 0:
        return relay
    b = battle.battle
    poff = pokemon.offset()
    gd = battle.game_data
    fx.apply_sitrus_berry(b, poff, gd)
    fx.apply_lum_berry(b, poff, gd)
    fx.apply_status_curing_berries(b, poff, gd)
    fx.apply_persim_berry(b, poff, gd)
    fx.apply_stat_boosting_berries(b, poff, gd)
    fx.apply_pinch_healing_berries(b, poff, gd)
    return relay


def _on_leftovers_residual(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.hp <= 0:
        return relay
    fx.apply_leftovers_healing(battle.battle, pokemon.offset(), battle.game_data)
    return relay


def _on_black_sludge_residual(
    battle, pokemon: PokemonView, source, effect, relay, **kw
):
    if pokemon.hp <= 0:
        return relay
    fx.apply_black_sludge_effect(battle.battle, pokemon.offset(), battle.game_data)
    return relay


def register_all(profile: GenProfile) -> None:
    if not profile.has_items:
        return
    fmt = int(profile.format_id)
    dispatch.register(fmt, "berries", dispatch.Update, _on_update_berries, priority=0)
    # Leftovers / Black Sludge heal at residual order 5 (Showdown items.ts).
    dispatch.register(
        fmt,
        str(ITEM_LEFTOVERS),
        dispatch.Residual,
        _on_leftovers_residual,
        order=5,
        priority=0,
    )
    dispatch.register(
        fmt,
        str(ITEM_BLACK_SLUDGE),
        dispatch.Residual,
        _on_black_sludge_residual,
        order=5,
        priority=0,
    )
