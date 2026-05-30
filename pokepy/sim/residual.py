"""Per-mon end-of-turn residual bundle.

Holds the residual effects that don't (yet) have a dedicated effect-keyed
handler: weather-ability healing (Ice Body / Rain Dish / Dry Skin), Salt Cure,
partially-trapped damage (Wrap/Bind/etc.), Aqua Ring / Ingrain healing, and the
Yawn -> sleep transition. Each sub-effect self-gates on the relevant volatile
bit or ability, so the bundle is safe to collect for every active mon.

These delegate to the proven implementations in pokepy/effects so behavior
matches the legacy end-of-turn orchestrator; the difference is they now fire
through a single ``fieldEvent('Residual')`` pass instead of a duplicate path.
"""

from __future__ import annotations

import pokepy.effects as fx
from pokepy.core.constants import (
    F_EXTENDED_VOLATILE_0,
    F_EXTENDED_VOLATILE_1,
    OFF_FIELD,
)
from pokepy.core.gen_profile import GenProfile
from pokepy.effects.end_of_turn import (
    _apply_aqua_ring_ingrain_heal,
    _process_yawn,
    apply_partial_trap_damage,
)
from pokepy.sim import dispatch
from pokepy.sim.views import PokemonView


def _on_mon_residual(battle, pokemon: PokemonView, source, effect, relay, **kw):
    if pokemon.hp <= 0:
        return relay
    b = battle.battle
    poff = pokemon.offset()
    side = pokemon.side
    ext_off = OFF_FIELD + (
        F_EXTENDED_VOLATILE_0 if side == 0 else F_EXTENDED_VOLATILE_1
    )

    # Ice Body / Rain Dish / Dry Skin weather healing (self-gates on ability).
    fx.apply_weather_healing(b, poff, battle.game_data)
    # Salt Cure (self-gates on the salt-cure volatile bit).
    if pokemon.hp > 0:
        fx.apply_salt_cure_damage(b, poff, ext_off, battle.game_data)
    # Partial trap residual (Wrap/Bind/Fire Spin/etc.).
    if pokemon.hp > 0:
        apply_partial_trap_damage(b, poff, side)
    # Aqua Ring / Ingrain healing.
    if pokemon.hp > 0:
        _apply_aqua_ring_ingrain_heal(b, poff, side)
    # Yawn -> sleep transition.
    if pokemon.hp > 0:
        _process_yawn(b, poff, side, gen5_prng=battle.prng)
    return relay


def register_all(profile: GenProfile) -> None:
    # Residual order ~10 places this alongside status/leftovers in the EOT pass.
    dispatch.register(
        int(profile.format_id),
        "monresidual",
        dispatch.Residual,
        _on_mon_residual,
        order=10,
        priority=0,
    )
