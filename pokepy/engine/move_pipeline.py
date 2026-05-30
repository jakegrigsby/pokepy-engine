"""Move pipeline: runMove through damage, modifyDamage, secondaries, recoil.

Mirrors sim/battle-actions.ts structure. Delegates damage calculation to
generation-specific mechanics modules and routes effect hooks through dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from pokepy.core.constants import OFF_SIDE0, OFF_SIDE1, POKEMON_SIZE
from pokepy.engine.dispatch import BitpackBattleContext
from pokepy.engine.gen_mods import damage_fn_for_gen, hit_step_order, modify_damage_order


@dataclass
class MoveContext:
    move_id: int
    user_offset: int
    target_offset: int
    hit: bool = True
    crit: bool = False
    damage: int = 0


@dataclass
class HitResult:
    target_offset: int
    success: bool
    damage: int = 0
    failed_step: str = ""


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


def hit_step_invulnerability_event(
    ctx: BitpackBattleContext,
    targets: List[int],
    pokemon_offset: int,
    move_id: int,
) -> List[HitResult]:
    out: List[HitResult] = []
    for t in targets:
        relay = ctx.run_event("Invulnerability", int(t), int(pokemon_offset), relay_var=True)
        out.append(HitResult(target_offset=int(t), success=bool(relay), failed_step="invuln" if not relay else ""))
    return out


def hit_step_try_hit_event(
    ctx: BitpackBattleContext,
    targets: List[int],
    pokemon_offset: int,
    move_id: int,
) -> List[HitResult]:
    out: List[HitResult] = []
    for t in targets:
        relay = ctx.run_event("TryHit", int(t), int(pokemon_offset), relay_var=True)
        ok = relay is not False
        out.append(HitResult(target_offset=int(t), success=ok, failed_step="tryhit" if not ok else ""))
    return out


def hit_step_accuracy(
    ctx: BitpackBattleContext,
    targets: List[int],
    pokemon_offset: int,
    move_id: int,
    *,
    accuracy: int = 100,
) -> List[HitResult]:
    out: List[HitResult] = []
    for t in targets:
        roll_ok = bool(ctx.random_chance(int(accuracy), 100))
        out.append(HitResult(target_offset=int(t), success=roll_ok, failed_step="accuracy" if not roll_ok else ""))
    return out


def hit_step_move_hit_loop(
    ctx: BitpackBattleContext,
    targets: List[int],
    pokemon_offset: int,
    move_id: int,
) -> List[HitResult]:
    out: List[HitResult] = []
    for t in targets:
        mc = spread_move_hit(ctx, int(move_id), int(pokemon_offset), int(t))
        out.append(
            HitResult(
                target_offset=int(t),
                success=bool(mc.hit),
                damage=int(mc.damage),
                failed_step="" if mc.hit else "damage",
            )
        )
    return out


def try_spread_move_hit(
    ctx: BitpackBattleContext,
    targets: List[int],
    pokemon_offset: int,
    move_id: int,
) -> List[HitResult]:
    """Showdown-style hit-step dispatcher."""
    results: List[HitResult] = [HitResult(target_offset=int(t), success=True) for t in targets]
    step_names = hit_step_order(ctx.profile)
    for step in step_names:
        current_targets = [r.target_offset for r in results if r.success]
        if not current_targets:
            break
        if step == "try_primary_hit":
            try_hit_results = hit_step_try_hit_event(ctx, current_targets, pokemon_offset, move_id)
            by_t = {r.target_offset: r for r in try_hit_results}
            for r in results:
                if r.success and r.target_offset in by_t:
                    rr = by_t[r.target_offset]
                    r.success = rr.success
                    r.failed_step = rr.failed_step
        elif step == "secondary":
            acc_results = hit_step_accuracy(ctx, current_targets, pokemon_offset, move_id)
            by_t = {r.target_offset: r for r in acc_results}
            for r in results:
                if r.success and r.target_offset in by_t:
                    rr = by_t[r.target_offset]
                    r.success = rr.success
                    r.failed_step = rr.failed_step
        elif step == "spread":
            dmg_results = hit_step_move_hit_loop(ctx, current_targets, pokemon_offset, move_id)
            by_t = {r.target_offset: r for r in dmg_results}
            for r in results:
                if r.success and r.target_offset in by_t:
                    rr = by_t[r.target_offset]
                    r.success = rr.success
                    r.damage = rr.damage
                    r.failed_step = rr.failed_step
    return results


def run_move(
    ctx: BitpackBattleContext,
    move_id: int,
    user_offset: int,
    target_offset: int,
) -> MoveContext:
    """Top-level move entry with Showdown-like hit-step sequencing."""
    hit_results = try_spread_move_hit(ctx, [int(target_offset)], int(user_offset), int(move_id))
    if not hit_results or not hit_results[0].success:
        return MoveContext(
            move_id=int(move_id),
            user_offset=int(user_offset),
            target_offset=int(target_offset),
            hit=False,
            damage=0,
        )
    return MoveContext(
        move_id=int(move_id),
        user_offset=int(user_offset),
        target_offset=int(target_offset),
        hit=True,
        damage=int(hit_results[0].damage),
    )


__all__ = [
    "MoveContext",
    "HitResult",
    "get_damage",
    "hit_step_accuracy",
    "hit_step_invulnerability_event",
    "hit_step_move_hit_loop",
    "hit_step_try_hit_event",
    "modify_damage",
    "run_move",
    "run_move_effects",
    "spread_move_hit",
    "try_spread_move_hit",
]
