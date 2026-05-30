"""Switch-in, faint, forced-switch, and request model."""

from __future__ import annotations

from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np

from pokepy.core.gen_profile import GenProfile, profile_for_gen
from pokepy.core.state import MultiFormatState
from pokepy.engine.dispatch import BitpackBattleContext, make_context
from pokepy.engine.switch_requests import SwitchRequest, resolve_switch_choices_sync

# Re-use the battle-tested switch implementations from the legacy module.
from pokepy.engine.battle_gen9 import (  # noqa: F401
    _inline_post_faint_switch_side1,
    step_forced_switch,
)


def run_switch_in_events(
    ctx: BitpackBattleContext,
    pokemon_offset: int,
    *,
    is_switch: bool = True,
) -> None:
    """Fire SwitchIn / AfterSwitchInSelf through dispatch + legacy effects."""
    from pokepy import effects as fx

    poff = int(pokemon_offset)
    ctx.single_event("SwitchIn", None, None, poff, relay_var=True)
    opp_active = ctx.battle[ctx.battle.shape[0] - 1]  # unused placeholder
    _ = opp_active
    if ctx.profile.has_abilities:
        from pokepy.core.constants import (
            OFF_META,
            M_ACTIVE0,
            M_ACTIVE1,
            OFF_SIDE0,
            OFF_SIDE1,
            POKEMON_SIZE,
        )

        if poff < OFF_SIDE1:
            opp_off = OFF_SIDE0 + int(ctx.battle[OFF_META + M_ACTIVE0]) * POKEMON_SIZE
            if poff >= OFF_SIDE0:
                opp_off = (
                    OFF_SIDE1 + int(ctx.battle[OFF_META + M_ACTIVE1]) * POKEMON_SIZE
                )
        else:
            opp_off = OFF_SIDE0 + int(ctx.battle[OFF_META + M_ACTIVE0]) * POKEMON_SIZE
        if int(ctx.battle[opp_off + 1]) > 0:
            fx.apply_switch_in_ability_with_trace_reaction(
                ctx.battle,
                poff,
                opp_off,
                is_switch,
                gen5_prng=ctx.gen5_prng,
                has_terrain=ctx.profile.has_terrain,
                ability_weather_limited=ctx.profile.ability_weather_limited,
            )
    from pokepy.core.constants import F_HAZARDS_0, F_HAZARDS_1, OFF_FIELD, OFF_SIDE1

    hazard_off = OFF_FIELD + (F_HAZARDS_1 if poff >= OFF_SIDE1 else F_HAZARDS_0)
    fx.apply_hazard_damage_on_switch(ctx.battle, poff, hazard_off)
    ctx.single_event("AfterSwitchInSelf", None, None, poff, relay_var=None)


def step_forced_switch_modular(
    state: MultiFormatState,
    action: int,
    side: int,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    gen5_prng,
    *,
    profile: Optional[GenProfile] = None,
) -> Tuple[np.float32, np.float32, bool]:
    """Forced switch entry point (delegates to legacy implementation)."""
    return step_forced_switch(
        state,
        action,
        side,
        game_data,
        move_effects,
        type_chart,
        gen5_prng,
        profile=profile or profile_for_gen(getattr(state, "format_gen", 9) or 9),
    )


__all__ = [
    "run_switch_in_events",
    "step_forced_switch",
    "step_forced_switch_modular",
    "_inline_post_faint_switch_side1",
    "SwitchRequest",
    "resolve_switch_choices_sync",
]
