"""Move execution pipeline — mirrors sim/battle-actions.ts.

Staged to match Showdown's call tree so PRNG frames are consumed at the same
points as the reference engine:

    run_move            battle-actions.ts:210  (BeforeMove gates, AfterMove)
      use_move          battle-actions.ts:377  (TryMove / Protect)
        try_spread_move_hit   battle-actions.ts:545  (hitStep dispatch)
          spread_move_hit     battle-actions.ts:1044 (damage + runMoveEffects)

The accuracy roll, crit roll, and 85-100% damage randomizer all live inside
``calc_damage_gen*`` (in Showdown's internal order: accuracy -> crit ->
randomizer). The pipeline therefore calls the damage unit EXACTLY ONCE at the
spread_move_hit position and never pre-rolls accuracy separately — the previous
code rolled accuracy here AND inside calc_damage, drifting the LCG by one frame
on every damaging move.

Deferred to the hardening phase (do not block the end-to-end gate): full
two-turn/charge, recharge lock, locked-move (Outrage) continuation, and
pivot/self-switch control flow. The runtime MoveEffectData does not expose those
flags, so they are intentionally not half-implemented here (a partial version
risks stalls). Multi-hit, OHKO, and fixed-damage are already handled inside
``calc_damage_gen*`` (it sums per-hit damage and reports num_hits via out_meta).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pokepy.effects as fx
from pokepy.core.constants import CAT_STATUS
from pokepy.mechanics.damage_gen9 import calc_damage_gen9
from pokepy.sim import dispatch
from pokepy.sim.helpers import move_id_for_side


def run_move(
    battle_ctx,
    side: int,
    move_slot: int,
    *,
    is_second: bool = False,
) -> bool:
    """battle-actions.ts:210 runMove — outer gates then useMove + AfterMove."""
    state = battle_ctx.state
    battle = battle_ctx.battle
    move_id = move_id_for_side(state, side, move_slot)
    if move_id < 0:
        return False

    user = battle_ctx.field.active(int(side))
    user_off = user.offset()
    if user.hp <= 0 or user.fainted:
        return False

    # --- BeforeMove gates (flinch, status immobilization, confusion) --------
    # Flinch (set by a faster foe's move this turn).
    if fx.check_flinched(battle, int(side)):
        battle_ctx.run_event(dispatch.AfterMoveSelf, target=user, relay_var=True)
        return False

    # Status BeforeMove handlers (sleep / paralysis / freeze) via dispatch.
    before = battle_ctx.run_event(dispatch.BeforeMove, target=user, relay_var=True)
    if before is False:
        battle_ctx.run_event(dispatch.AfterMoveSelf, target=user, relay_var=True)
        return False

    # Confusion self-hit (applies damage internally; returns True if it hit).
    if fx.check_confusion_self_hit(battle, int(side), user_off, battle_ctx.prng):
        # A fainting self-hit is picked up by the runAction faint postamble.
        battle_ctx.run_event(dispatch.AfterMoveSelf, target=user, relay_var=True)
        return False

    # The move is now actually being used — reveal it to the observer/opponent.
    _reveal_move(state, int(side), int(move_slot))

    used = use_move(battle_ctx, int(side), int(move_slot), move_id, is_second=is_second)

    # --- AfterMove ----------------------------------------------------------
    battle_ctx.run_event(dispatch.AfterMoveSelf, target=user, relay_var=True)
    battle_ctx.each_event(dispatch.Update)
    return used


def use_move(
    battle_ctx,
    side: int,
    move_slot: int,
    move_id: int,
    *,
    is_second: bool = False,
) -> bool:
    """battle-actions.ts:377 useMoveInner — Protect gates then hit resolution."""
    battle = battle_ctx.battle
    foe_side = 1 - int(side)
    me = battle_ctx.move_effects

    # The user's own protect-family setup (Protect/Detect/etc.).
    if fx.apply_protect_from_move(battle, move_id, int(side), me, battle_ctx.prng):
        return True

    # Foe is behind Protect — the move is blocked.
    if fx.check_protected(battle, foe_side):
        return True

    return try_spread_move_hit(
        battle_ctx, int(side), int(move_slot), move_id, is_second=is_second
    )


def try_spread_move_hit(
    battle_ctx,
    side: int,
    move_slot: int,
    move_id: int,
    *,
    is_second: bool = False,
) -> bool:
    """battle-actions.ts:545 trySpreadMoveHit — split damaging vs status."""
    gd = battle_ctx.game_data
    cat = int(np.asarray(gd.move_category)[move_id])
    if cat != CAT_STATUS:
        return _spread_move_hit_damaging(
            battle_ctx, side, move_slot, move_id, is_second=is_second
        )
    return _status_move_hit(battle_ctx, side, move_slot, move_id)


def _spread_move_hit_damaging(
    battle_ctx,
    side: int,
    move_slot: int,
    move_id: int,
    *,
    is_second: bool,
) -> bool:
    """battle-actions.ts:1044 spreadMoveHit (damaging branch).

    getDamage (accuracy/crit/randomizer) -> spreadDamage (substitute + HP) ->
    DamagingHit / AfterHit cascade -> runMoveEffects (on-hit) -> PP.
    """
    state = battle_ctx.state
    battle = battle_ctx.battle
    gd = battle_ctx.game_data
    me = battle_ctx.move_effects
    foe_side = 1 - int(side)
    user = battle_ctx.field.active(int(side))
    target = battle_ctx.field.active(foe_side)
    user_off = user.offset()
    target_off = target.offset()
    cat = int(np.asarray(gd.move_category)[move_id])
    move_type = int(np.asarray(gd.move_type)[move_id])
    move_flags = int(np.asarray(gd.move_flags)[move_id])

    meta: Dict[str, Any] = {}
    # getDamage + modifyDamage: this single call rolls accuracy, crit, and the
    # 85-100% randomizer in Showdown's internal order and returns 0 on miss /
    # immunity. No separate accuracy pre-roll (that was a duplicate frame).
    damage = int(
        calc_damage_gen9(
            battle,
            int(side),
            int(move_slot),
            state.team_moves,
            state.opp_moves,
            gd,
            me,
            battle_ctx.type_chart,
            is_moving_last=is_second,
            gen5_prng=battle_ctx.prng,
            out_meta=meta,
            profile=battle_ctx.profile,
        )
    )
    hit = damage > 0

    if damage > 0:
        through = fx.apply_damage_to_substitute(battle, foe_side, damage, True)
        if through > 0:
            target.hp = max(0, target.hp - int(through))
            # A fainting target is picked up by the runAction faint postamble.
            if battle_ctx.profile.has_abilities:
                fx.apply_immediate_defender_ability_state_changes(
                    battle,
                    user_off,
                    target_off,
                    True,
                    int(through),
                    move_type,
                    cat,
                    move_flags,
                )
                fx.apply_cursed_body_on_damaging_hit(
                    battle,
                    user_off,
                    target_off,
                    move_id,
                    move_slot,
                    True,
                    int(through),
                    battle_ctx.prng,
                    gen=battle_ctx.profile.gen,
                )
                fx.apply_toxic_chain_on_damaging_hit(
                    battle,
                    user_off,
                    target_off,
                    True,
                    int(through),
                    gd,
                    battle_ctx.prng,
                )
            if battle_ctx.profile.has_items:
                fx.apply_defender_stat_berries_on_damaging_hit(
                    battle,
                    target_off,
                    cat,
                    True,
                    int(through),
                    gd,
                )
            if battle_ctx.profile.has_abilities:
                fx.apply_contact_damage(
                    battle, move_id, user_off, target_off, True, gd, me
                )
                fx.apply_contact_status_ability(
                    battle,
                    move_id,
                    user_off,
                    target_off,
                    True,
                    gd,
                    me,
                    battle_ctx.prng,
                )
            if battle_ctx.profile.has_items:
                fx.apply_knock_off_from_move(
                    battle,
                    move_id,
                    target_off,
                    True,
                    gd,
                    me,
                    user_offset=user_off,
                )
                fx.apply_life_orb_recoil(battle, user_off, int(through), True, move_id)
            fx.apply_recoil_drain_from_move(
                battle,
                move_id,
                user_off,
                int(through),
                True,
                gd,
                me,
                target_offset=target_off,
                gen=battle_ctx.profile.gen,
            )

    if hit:
        _run_move_effects(battle_ctx, int(side), move_slot, move_id, gd, me)

    _deduct_pp(state, int(side), move_slot, hit)
    return True


def _status_move_hit(battle_ctx, side: int, move_slot: int, move_id: int) -> bool:
    """battle-actions.ts:1044 spreadMoveHit (status branch).

    Status moves never run getDamage, so accuracy is rolled here. accuracy <= 0
    or >= 100 means "always hits" (Showdown `accuracy: true`)."""
    state = battle_ctx.state
    gd = battle_ctx.game_data
    me = battle_ctx.move_effects
    acc = int(np.asarray(gd.move_accuracy)[move_id])
    if 0 < acc < 100:
        hit = int(battle_ctx.random(100)) < acc
    else:
        hit = True
    if hit:
        _run_move_effects(battle_ctx, int(side), move_slot, move_id, gd, me)
    _deduct_pp(state, int(side), move_slot, hit)
    return True


def _run_move_effects(
    battle_ctx, side: int, move_slot: int, move_id: int, gd, me
) -> None:
    """battle-actions.ts:runMoveEffects — boosts, status, flinch, recovery,
    substitute, hazards, weather, terrain, Trick Room, screens (on hit)."""
    battle = battle_ctx.battle
    foe_side = 1 - int(side)
    user_off = battle_ctx.field.active(int(side)).offset()
    target_off = battle_ctx.field.active(foe_side).offset()

    fx.apply_stat_changes_from_move(
        battle,
        move_id,
        user_off,
        target_off,
        True,
        gd,
        me,
        battle_ctx.prng,
        gen=battle_ctx.profile.gen,
    )
    fx.apply_status_from_move(
        battle,
        move_id,
        target_off,
        True,
        gd,
        me,
        battle_ctx.prng,
        user_offset=user_off,
        set_status_speedsort=battle_ctx.maybe_set_status_speedsort,
    )
    fx.apply_flinch_from_move(
        battle, move_id, foe_side, True, me, battle_ctx.prng, game_data=gd
    )
    fx.apply_recovery_from_move(battle, move_id, user_off, True, gd, me)
    fx.apply_substitute_from_move(battle, move_id, int(side), user_off, gd, me)
    fx.apply_leech_seed_from_move(battle, move_id, foe_side, target_off, True, gd, me)
    fx.apply_perish_song_from_move(battle, move_id, True, me, user_side=int(side))
    fx.apply_destiny_bond_from_move(battle, move_id, int(side), me)
    fx.apply_lock_on_from_move(battle, move_id, int(side), True, me)
    fx.apply_ghost_curse_from_move(battle, move_id, user_off, foe_side, True, gd, me)
    fx.apply_pain_split_from_move(battle, move_id, user_off, target_off, True, gd, me)
    fx.apply_confusion_from_move(
        battle, move_id, foe_side, True, gd, me, battle_ctx.prng
    )
    fx.apply_taunt_from_move(battle, move_id, foe_side, True, me, battle_ctx.prng)
    fx.apply_encore_from_move(battle, move_id, foe_side, True, me, battle_ctx.prng)
    fx.apply_throat_chop_from_move(battle, move_id, foe_side, True)
    fx.apply_phazing_from_move(battle, move_id, foe_side, target_off, True, gd, me)
    fx.apply_extended_volatile(
        battle, move_id, foe_side, int(side), True, gd, me, battle_ctx.prng
    )
    if battle_ctx.profile.gen >= 2:
        fx.apply_hazard_from_move(
            battle, move_id, foe_side, True, gd, me, user_offset=user_off
        )
    if battle_ctx.profile.gen >= 3:
        fx.apply_weather_from_move(battle, move_id, True, gd, me, user_offset=user_off)
    if battle_ctx.profile.has_terrain:
        fx.apply_terrain_from_move(battle, move_id, True, gd, me, user_offset=user_off)
    fx.apply_trick_room_from_move(battle, move_id, True, gd, me)
    fx.apply_screen_from_move(battle, move_id, int(side), True, me)
    fx.apply_rapid_spin_from_move(
        battle, move_id, user_off, int(side), True, move_effects=me
    )
    fx.apply_defog_from_move(
        battle, move_id, True, move_effects=me, user_side=int(side)
    )
    fx.apply_court_change_from_move(battle, move_id, True)
    fx.apply_haze_from_move(battle, move_id, True, move_effects=me)
    fx.apply_clear_smog_from_move(battle, move_id, target_off, True, move_effects=me)
    fx.apply_psych_up_from_move(
        battle, move_id, user_off, target_off, True, move_effects=me
    )
    fx.apply_trick_from_move(battle, move_id, user_off, target_off, True, gd, me)
    fx.apply_belly_drum_from_move(battle, move_id, user_off, True)
    fx.apply_self_type_removal_from_move(battle, move_id, user_off, True)
    fx.apply_skill_swap_from_move(battle, move_id, user_off, target_off, True)


def _reveal_move(state, side: int, move_slot: int) -> None:
    """Mark a used move as revealed in the observer's view.

    team_moves_revealed is side 0's moves (seen by side 1); opp_moves_revealed
    is side 1's moves (seen by side 0). Indexed [active team slot, move slot].
    """
    from pokepy.sim.helpers import active_slot

    slot = int(move_slot)
    if slot < 0 or slot >= 4:
        return
    active = int(active_slot(state.battle_state, int(side)))
    arr = state.team_moves_revealed if int(side) == 0 else state.opp_moves_revealed
    if 0 <= active < arr.shape[0]:
        arr[active, slot] = True


def _deduct_pp(state, side: int, move_slot: int, hit: bool) -> None:
    if not hit:
        return
    from pokepy.sim.helpers import active_slot

    active = active_slot(state.battle_state, side)
    pp_arr = state.team_pp if side == 0 else state.opp_pp
    slot = int(move_slot)
    if slot < 0 or slot >= 4:
        return
    if int(pp_arr[active, slot]) <= 0:
        return
    pp_arr[active, slot] = max(0, int(pp_arr[active, slot]) - 1)
