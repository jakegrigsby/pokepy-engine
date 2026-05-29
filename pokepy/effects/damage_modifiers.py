"""Recoil/drain and contact damage modifiers.

10907-10974.
"""

from __future__ import annotations

from pokepy.effects._common import np, MultiFormatState, Gen5PRNG
from pokepy.effects.ability_suppression import effective_ability
from pokepy.core.constants import (
    MOVE_STRUGGLE,
    FLAG_CONTACT,
    ABILITY_ROCK_HEAD,
    ABILITY_MAGIC_GUARD,
    ABILITY_LONG_REACH,
    ABILITY_ROUGH_SKIN,
    ABILITY_IRON_BARBS,
    ITEM_ROCKY_HELMET,
    ITEM_PROTECTIVE_PADS,
)

# Liquid Ooze ability ID — not in constants.py.
ABILITY_LIQUID_OOZE = 64


def apply_recoil_drain_from_move(
    battle: np.ndarray,
    move_id: int,
    user_offset: int,
    damage_dealt: int,
    hit: bool,
    game_data,
    move_effects,
    target_offset: int = None,
    phase: str = "both",
    move_attempted: bool = False,
    gen: int = 9,
) -> None:
    """Port of _apply_recoil_drain_from_move (line ~10070).

    Recoil moves (positive recoil pct) damage the user; drain moves
    (negative pct) heal the user. Struggle uses 25% of max HP, while the
    fixed self-damage moves Mind Blown / Steel Beam / Chloroblast each use
    half the user's max HP. Mind Blown / Steel Beam recoil is a Showdown
    `mindBlownRecoil` after-move hook, so it still fires when the move was
    attempted but missed or hit an immunity. Chloroblast is ordinary recoil
    and only fires when it dealt damage. Magic Guard blocks recoil-style
    damage; Rock Head blocks ordinary move recoil but does NOT block Struggle
    recoil (Showdown routes that through `strugglerecoil`). Liquid Ooze on
    the target inverts drain into damage.

    `phase` selects which of the two sub-effects is applied:
      - "drain": only heal (or Liquid Ooze damage). Showdown applies this
        inside `spreadDamage`, before `DamagingHit` → contact abilities.
      - "recoil": only user-side recoil damage. Showdown applies this
        after every hit of the move is done but BEFORE
        `afterMoveSecondaryEvent` (Life Orb / Sticky Barb / Shell Bell).
      - "both": legacy behavior, applies whichever one is relevant.
    The engine calls this twice per move — once with "drain" right after
    damage, and once with "recoil" after the contact-ability cascade.
    """
    move_id = int(move_id)
    uoff = int(user_offset)
    damage_dealt = int(damage_dealt)
    hit = bool(hit)

    # High Jump Kick (136), Jump Kick (26), Supercell Slam (916), and Axe
    # Kick (853) have hasCrashDamage in Showdown — they apply 50% maxhp
    # damage on MISS, not on hit. Pokepy's move_effects table marks Axe
    # Kick with `recoil: 50` (matching the 50% value) but the recoil
    # pipeline below would apply it on HIT. Skip them entirely here; the
    # engine handles the on-miss crash separately. Showdown source:
    # data/moves.ts:axekick onMoveFail and hasCrashDamage flag.
    _MOVE_HIGH_JUMP_KICK = 136
    _MOVE_JUMP_KICK = 26
    _MOVE_SUPERCELL_SLAM = 916
    _MOVE_AXE_KICK = 853
    if move_id in (
        _MOVE_HIGH_JUMP_KICK,
        _MOVE_JUMP_KICK,
        _MOVE_SUPERCELL_SLAM,
        _MOVE_AXE_KICK,
    ):
        return

    _MIND_BLOWN_RECOIL_MOVES = {720, 796}  # Mind Blown, Steel Beam
    _FIXED_MAXHP_RECOIL_MOVES = _MIND_BLOWN_RECOIL_MOVES | {835}  # + Chloroblast

    recoil_pct = int(move_effects.recoil[move_id])

    current_hp = int(battle[uoff + 1])
    max_hp = int(battle[uoff + 2])

    is_struggle = move_id == MOVE_STRUGGLE
    # Gen 3 Struggle recoil is damageDealt * 1/4 (data/mods/gen3/moves.ts).
    # Modern gens route through the strugglerecoil condition (maxHP / 4).
    if is_struggle and gen <= 3:
        recoil_base = float(damage_dealt)
    elif is_struggle:
        recoil_base = float(max_hp)
    else:
        recoil_base = float(damage_dealt)
    # Showdown sim/battle-actions.ts:1396 — Math.round(damage * recoil[0] / recoil[1])
    # then clampIntRange(min=1). Pokepy used to truncate (int()) which can be
    # off by 1 vs Showdown for fractional cases. JS's Math.round rounds half
    # AWAY-FROM-ZERO for positive values; Python's round() is banker's
    # (half-to-even). Use floor(x + 0.5) to match JS.
    import math as _math

    recoil_amount = int(_math.floor(recoil_base * abs(recoil_pct) / 100.0 + 0.5))
    if move_id in _FIXED_MAXHP_RECOIL_MOVES:
        recoil_amount = int(_math.floor(float(max_hp) / 2.0 + 0.5))

    if (damage_dealt > 0) and (
        (recoil_pct != 0) or (move_id in _FIXED_MAXHP_RECOIL_MOVES)
    ):
        recoil_amount = max(recoil_amount, 1)

    is_mind_blown_recoil = move_id in _MIND_BLOWN_RECOIL_MOVES
    is_recoil = (recoil_pct > 0) or (move_id in _FIXED_MAXHP_RECOIL_MOVES)
    is_drain = recoil_pct < 0

    target_for_user_ability = target_offset
    if (
        target_for_user_ability is not None
        and int(battle[int(target_for_user_ability) + 1]) <= 0
    ):
        target_for_user_ability = None
    user_ability = effective_ability(battle, uoff, target_for_user_ability)
    has_rock_head = user_ability == ABILITY_ROCK_HEAD
    # Magic Guard blocks all non-move recoil damage, including Struggle's
    # dedicated `strugglerecoil` condition. Rock Head only blocks ordinary
    # move recoil; Showdown explicitly lets Struggle recoil through.
    has_magic_guard_rec = user_ability == ABILITY_MAGIC_GUARD
    if not is_mind_blown_recoil:
        if is_struggle:
            is_recoil = is_recoil and (not has_magic_guard_rec)
        else:
            is_recoil = is_recoil and (not has_rock_head) and (not has_magic_guard_rec)

    # Big Root (item 29) — drain heals 1.3x. Showdown items.ts:bigroot
    # `onTryHealPriority: 1, onTryHeal(damage)`. Pokepy used to ignore it.
    user_item = int(battle[uoff + 6])
    ITEM_BIG_ROOT = 296
    drain_heal_amount = recoil_amount
    if is_drain and user_item == ITEM_BIG_ROOT:
        drain_heal_amount = (recoil_amount * 5325) // 4096  # chainModify(1.3)
        drain_heal_amount = max(drain_heal_amount, 1)

    new_hp_recoil = max(0, current_hp - recoil_amount)

    has_liquid_ooze = False
    if target_offset is not None:
        toff = int(target_offset)
        has_liquid_ooze = effective_ability(battle, toff, uoff) == ABILITY_LIQUID_OOZE

    if has_liquid_ooze:
        new_hp_drain = max(0, current_hp - recoil_amount)
    else:
        new_hp_drain = min(max_hp, current_hp + drain_heal_amount)

    # Phase gating — see docstring.
    if phase == "drain" and not is_drain:
        return
    if phase == "recoil" and not is_recoil:
        return

    if is_recoil:
        new_hp = new_hp_recoil
    elif is_drain:
        new_hp = new_hp_drain
    else:
        new_hp = current_hp

    should_apply = (is_mind_blown_recoil and move_attempted) or (
        hit
        and (damage_dealt > 0)
        and ((recoil_pct != 0) or (move_id in _FIXED_MAXHP_RECOIL_MOVES))
    )
    final_hp = new_hp if should_apply else current_hp

    battle[uoff + 1] = final_hp


def apply_life_orb_recoil(
    battle: np.ndarray,
    user_offset: int,
    damage_dealt: int,
    hit: bool,
    game_data,
    move_id: int = None,
    move_effects=None,
) -> None:
    """Re-export of items.apply_life_orb_recoil for callers that import from
    damage_modifiers (mirrors the Showdown reference structure where it lives in this module
    in some call sites).
    """
    from pokepy.effects.items import apply_life_orb_recoil as _impl

    _impl(
        battle,
        user_offset,
        damage_dealt,
        hit,
        game_data,
        move_id=move_id,
        move_effects=move_effects,
    )


def apply_contact_damage(
    battle: np.ndarray,
    move_id: int,
    attacker_offset: int,
    defender_offset: int,
    hit: bool,
    game_data,
    move_effects,
    num_hits: int = 1,
) -> None:
    """Port of _apply_contact_damage (line ~10907).

    Rough Skin / Iron Barbs deal 1/8 max HP, Rocky Helmet 1/6 max HP, both
    can stack. Magic Guard, Long Reach, and Protective Pads block.

    Showdown fires `onDamagingHit` inside `spreadMoveHit` once per hit
    (sim/battle-actions.ts:1139), so multi-hit moves trigger Rocky Helmet /
    Rough Skin / Iron Barbs `num_hits` times. Pokepy applies aggregate
    damage so the engine passes the actual hit count here. The loop also
    short-circuits if the attacker faints mid-sequence (Showdown stops
    processing once damagedTargets is empty).
    """
    move_id = int(move_id)
    aoff = int(attacker_offset)
    doff = int(defender_offset)
    hit = bool(hit)
    num_hits = max(1, int(num_hits))

    move_flags = int(game_data.move_flags[move_id])
    is_contact = (move_flags & FLAG_CONTACT) != 0

    atk_ability = effective_ability(battle, aoff, doff)
    has_long_reach = atk_ability == ABILITY_LONG_REACH
    is_contact = is_contact and (not has_long_reach)

    def_ability = effective_ability(battle, doff, aoff)
    def_item = int(battle[doff + 6])

    has_rough_skin = (def_ability == ABILITY_ROUGH_SKIN) or (
        def_ability == ABILITY_IRON_BARBS
    )
    has_rocky_helmet = def_item == ITEM_ROCKY_HELMET

    atk_hp = int(battle[aoff + 1])
    atk_max_hp = int(battle[aoff + 2])

    has_magic_guard = atk_ability == ABILITY_MAGIC_GUARD
    atk_item = int(battle[aoff + 6])
    has_protective_pads = atk_item == ITEM_PROTECTIVE_PADS

    # Showdown damage() floors: attacker.baseMaxhp / 8 (rough skin / iron barbs)
    # and / 6 (rocky helmet). Use integer division to match exactly.
    rough_skin_dmg = max(atk_max_hp // 8, 1)
    rocky_helmet_dmg = max(atk_max_hp // 6, 1)

    per_hit_damage = (rough_skin_dmg if has_rough_skin else 0) + (
        rocky_helmet_dmg if has_rocky_helmet else 0
    )

    if not (
        is_contact
        and hit
        and (atk_hp > 0)
        and (not has_magic_guard)
        and (not has_protective_pads)
        and (per_hit_damage > 0)
    ):
        return

    # Fire per-hit, stopping when attacker faints (Showdown's loop ends
    # once damagedTargets is filtered out for fainted source).
    cur_hp = atk_hp
    for _ in range(num_hits):
        if cur_hp <= 0:
            break
        cur_hp = max(0, cur_hp - per_hit_damage)

    battle[aoff + 1] = cur_hp
