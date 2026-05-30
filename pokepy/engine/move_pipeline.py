"""Move pipeline: runMove through damage, modifyDamage, secondaries, recoil.

Mirrors sim/battle-actions.ts structure. Delegates damage calculation to
generation-specific mechanics modules and routes effect hooks through dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from pokepy.core.constants import OFF_SIDE0, OFF_SIDE1, POKEMON_SIZE
from pokepy.engine.dispatch import BitpackBattleContext
from pokepy.engine.gen_mods import damage_fn_for_gen, modify_damage_order


@dataclass
class MoveContext:
    move_id: int
    user_offset: int
    target_offset: int
    hit: bool = True
    crit: bool = False
    damage: int = 0


def _side_base(offset: int) -> int:
    return OFF_SIDE0 if int(offset) < OFF_SIDE1 else OFF_SIDE1


def get_damage(
    ctx: BitpackBattleContext,
    move_id: int,
    user_offset: int,
    target_offset: int,
    *,
    crit: bool = False,
) -> int:
    calc = damage_fn_for_gen(ctx.gen)
    try:
        return int(
            calc(
                ctx.battle,
                int(move_id),
                int(user_offset),
                int(target_offset),
                ctx.game_data,
                ctx.move_effects,
                ctx.type_chart,
                ctx.gen5_prng,
                profile=ctx.profile,
                crit=bool(crit),
            )
        )
    except TypeError:
        return int(
            calc(
                ctx.battle,
                int(move_id),
                int(user_offset),
                int(target_offset),
                ctx.game_data,
                ctx.move_effects,
                ctx.type_chart,
                ctx.gen5_prng,
            )
        )


def modify_damage(
    ctx: BitpackBattleContext,
    base_damage: int,
    move_id: int,
    user_offset: int,
    target_offset: int,
) -> int:
    order = modify_damage_order(ctx.profile)
    dmg = int(base_damage)
    if order == "gen3":
        # Gen3: STAB/type before randomizer; randomizer last.
        dmg = ctx.single_event(
            "ModifyDamage",
            None,
            None,
            int(target_offset),
            int(user_offset),
            relay_var=dmg,
        )
        if dmg is False:
            return 0
        dmg = ctx.randomizer(int(dmg))
        return max(1, int(dmg))
    relay = ctx.single_event(
        "ModifyDamage",
        None,
        None,
        int(target_offset),
        int(user_offset),
        relay_var=int(dmg),
    )
    if relay is False:
        return 0
    dmg = int(relay)
    dmg = ctx.randomizer(dmg)
    return max(1, int(dmg))


def run_move_effects(
    ctx: BitpackBattleContext,
    mc: MoveContext,
) -> None:
    """Post-damage move effects via dispatch + legacy effect helpers."""
    from pokepy import effects as fx

    if not mc.hit or mc.damage <= 0:
        return
    fx.apply_recoil_drain_from_move(
        ctx.battle,
        mc.move_id,
        mc.user_offset,
        mc.damage,
        True,
        mc.target_offset,
        gen5_prng=ctx.gen5_prng,
        game_data=ctx.game_data,
    )
    fx.apply_life_orb_recoil(ctx.battle, mc.user_offset, mc.damage, True, mc.move_id)


def spread_move_hit(
    ctx: BitpackBattleContext,
    move_id: int,
    user_offset: int,
    target_offset: int,
    *,
    force_crit: bool = False,
) -> MoveContext:
    crit = bool(force_crit)
    base = get_damage(ctx, move_id, user_offset, target_offset, crit=crit)
    final = modify_damage(ctx, base, move_id, user_offset, target_offset)
    if final > 0:
        ctx.damage(target_offset, final)
    mc = MoveContext(
        move_id=int(move_id),
        user_offset=int(user_offset),
        target_offset=int(target_offset),
        hit=True,
        crit=crit,
        damage=int(final),
    )
    run_move_effects(ctx, mc)
    return mc


def run_move(
    ctx: BitpackBattleContext,
    move_id: int,
    user_offset: int,
    target_offset: int,
) -> MoveContext:
    """Top-level move entry (accuracy/protect checks live in turn_loop)."""
    return spread_move_hit(ctx, move_id, user_offset, target_offset)


__all__ = [
    "MoveContext",
    "get_damage",
    "modify_damage",
    "run_move",
    "run_move_effects",
    "spread_move_hit",
]
