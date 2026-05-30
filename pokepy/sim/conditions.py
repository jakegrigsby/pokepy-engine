"""Condition handlers — mirror data/conditions.ts (+ gen mods)."""

from __future__ import annotations

from pokepy.core.constants import (
    STATUS_BURN,
    STATUS_FREEZE,
    STATUS_NONE,
    STATUS_PARALYSIS,
    STATUS_POISON,
    STATUS_SLEEP,
    STATUS_TOXIC,
)
from pokepy.sim import dispatch
from pokepy.sim.views import PokemonView


def _register_base(format_id: int) -> None:
    dispatch.register(
        format_id, "slp", dispatch.BeforeMove, on_slp_before_move, priority=10
    )
    dispatch.register(
        format_id, "par", dispatch.BeforeMove, on_par_before_move, priority=1
    )
    dispatch.register(
        format_id, "frz", dispatch.BeforeMove, on_frz_before_move, priority=10
    )
    dispatch.register(
        format_id, "brn", dispatch.Residual, on_brn_residual, priority=10, order=10
    )
    dispatch.register(
        format_id, "psn", dispatch.Residual, on_psn_residual, priority=10, order=9
    )
    dispatch.register(
        format_id, "tox", dispatch.Residual, on_tox_residual, priority=10, order=9
    )


def register_gen1() -> None:
    from pokepy.core.constants import FORMAT_GEN1OU

    fmt = FORMAT_GEN1OU
    handlers = dispatch.get_handlers(fmt, "slp")
    handlers.pop(dispatch.BeforeMove, None)
    handlers.pop(dispatch.AfterMoveSelf, None)
    dispatch.register(
        fmt, "slp", dispatch.BeforeMove, on_gen1_slp_before_move, priority=10
    )
    dispatch.register(
        fmt, "slp", dispatch.AfterMoveSelf, on_gen1_slp_after_move_self, priority=3
    )
    par_handlers = dispatch.get_handlers(fmt, "par")
    par_handlers.pop(dispatch.BeforeMove, None)
    dispatch.register(
        fmt, "par", dispatch.BeforeMove, on_gen1_par_before_move, priority=2
    )


def register_all(profile) -> None:
    fmt = int(profile.format_id)
    _register_base(fmt)
    if profile.gen == 1:
        register_gen1()
    elif profile.gen in (3, 4):
        dispatch.register(
            fmt,
            "slp",
            dispatch.BeforeMove,
            on_modern_slp_before_move,
            priority=10,
        )
    if profile.gen == 4:
        dispatch.register(
            fmt,
            "slp",
            dispatch.SetStatus,
            on_early_bird_set_status,
            priority=-2,
        )


def on_slp_before_move(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.status != STATUS_SLEEP:
        return relay
    turns = pokemon.status_turns
    if battle.profile.gen >= 3:
        turns -= 1
        pokemon.status_turns = turns
        if turns <= 0:
            pokemon.status = STATUS_NONE
            return relay
    return False


def on_gen1_slp_before_move(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.status != STATUS_SLEEP:
        return relay
    turns = pokemon.status_turns - 1
    pokemon.status_turns = max(turns, 0)
    return False


def on_gen1_slp_after_move_self(
    battle, pokemon: PokemonView, source, effect, relay, **kw
):
    if pokemon.status != STATUS_SLEEP:
        return relay
    if pokemon.status_turns <= 0:
        pokemon.status = STATUS_NONE
    return relay


def on_modern_slp_before_move(
    battle, pokemon: PokemonView, source, effect, relay, **kw
):
    if pokemon.status != STATUS_SLEEP:
        return relay
    turns = pokemon.status_turns - 1
    pokemon.status_turns = max(turns, 0)
    if turns <= 0:
        pokemon.status = STATUS_NONE
        return relay
    return False


def on_early_bird_set_status(battle, pokemon: PokemonView, source, effect, relay, **kw):
    from pokepy.core.constants import ABILITY_EARLY_BIRD

    if pokemon.ability == ABILITY_EARLY_BIRD and pokemon.status == STATUS_SLEEP:
        pokemon.status_turns = max(1, pokemon.status_turns // 2)
    return relay


def on_par_before_move(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.status != STATUS_PARALYSIS:
        return relay
    if battle.random_chance(
        battle.profile.full_para_num, battle.profile.full_para_denom
    ):
        return False
    return relay


def on_gen1_par_before_move(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.status != STATUS_PARALYSIS:
        return relay
    if battle.random_chance(63, 256):
        return False
    return relay


def on_frz_before_move(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.status != STATUS_FREEZE:
        return relay
    if battle.random_chance(1, 5):
        pokemon.status = STATUS_NONE
        return relay
    return False


def on_brn_residual(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.status != STATUS_BURN or pokemon.hp <= 0:
        return relay
    import pokepy.effects as fx

    fx.apply_end_of_turn_status(
        battle.battle,
        pokemon.offset(),
        battle.game_data,
        battle.move_effects,
        battle.prng,
    )
    return relay


def on_psn_residual(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.status != STATUS_POISON or pokemon.hp <= 0:
        return relay
    import pokepy.effects as fx

    fx.apply_end_of_turn_status(
        battle.battle,
        pokemon.offset(),
        battle.game_data,
        battle.move_effects,
        battle.prng,
    )
    return relay


def on_tox_residual(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.status != STATUS_TOXIC or pokemon.hp <= 0:
        return relay
    import pokepy.effects as fx

    fx.apply_end_of_turn_status(
        battle.battle,
        pokemon.offset(),
        battle.game_data,
        battle.move_effects,
        battle.prng,
    )
    return relay
