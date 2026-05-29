"""Defender ability cascade — port of multi_format_fast_env lines 4535-4765.

After damage is applied each turn, certain defender abilities trigger based on
type/category of incoming move and damage taken. This module ports all of:

- Toxic Chain (Pecharunt): 30% chance to badly poison on damaging hit
- Toxic Debris (Glimmora): scatter Toxic Spikes on opponent's side when hit by physical
- Anger Shell: +1 Atk/SpA/Spe -1 Def/SpD when HP crosses 50%
- Cotton Down: -1 Spe to attacker
- Thermal Exchange (Iron Moth): +1 Atk on Fire hit
- Justified: +1 Atk on Dark hit
- Water Compaction: +2 Def on Water hit
- Steam Engine: +6 Spe on Fire/Water hit
- Weak Armor: -1 Def +2 Spe on physical hit
- Mummy / Lingering Aroma: replace attacker ability on contact
- Wandering Spirit: swap abilities on contact

Eject Button / Red Card (item-triggered forced switches) live in
`pokepy/effects/eject_red_card.py` (or are integrated by the engine via
`fx.apply_item_forced_switch`).
"""

from __future__ import annotations

import numpy as np

from pokepy.effects.misc import is_take_item_blocked_by_item_rule
from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    POKEMON_SIZE,
    OFF_MOVES,
    OFF_FIELD,
    F_HAZARDS_0,
    F_HAZARDS_1,
    F_DISABLE_0,
    F_DISABLE_1,
    F_DISABLE_TURNS_0,
    F_DISABLE_TURNS_1,
    F_VOLATILE_0,
    F_VOLATILE_1,
    M_ACTIVE_MOVE_ACTIONS_0,
    M_ACTIVE_MOVE_ACTIONS_1,
    ACTIVE_MOVE_ACTIONS_SEMI_INVUL,
    FLAG_CHARGE,
    STATUS_NONE,
    STATUS_TOXIC,
    STATUS_PARALYSIS,
    TYPE_FIRE,
    TYPE_WATER,
    TYPE_DARK,
    TYPE_BUG,
    TYPE_GHOST,
    TYPE_POISON,
    TYPE_STEEL,
    FLAG_CONTACT,
)
from pokepy.core.bitpack import (
    apply_boost_to_packed,
    get_status,
    get_toxic_spikes_layers,
    set_toxic_spikes,
)
from pokepy.effects.form_changes import (
    GULP_MISSILE_GORGING,
    GULP_MISSILE_GULPING,
    GULP_MISSILE_NONE,
    clear_gulp_missile_state,
    get_gulp_missile_state,
)
from pokepy.effects.switch_slot_conditions import (
    apply_pending_wish_on_switch_in,
    is_pending_wish_sentinel,
)
from pokepy.effects.stat_changes import apply_direct_stat_changes
from pokepy.effects.status_apply import _try_apply_status
from pokepy.utils.gen5_prng import Gen5PRNG

# Local ability constants — match the Showdown reference source local definitions
ABILITY_TOXIC_CHAIN = 305
ABILITY_TOXIC_DEBRIS = 295
ABILITY_ANGER_SHELL = 271
ABILITY_COTTON_DOWN = 238
ABILITY_THERMAL_EXCHANGE = 270
ABILITY_JUSTIFIED = 154
ABILITY_WATER_COMPACTION = 195
ABILITY_STEAM_ENGINE = 243
ABILITY_WEAK_ARMOR = 133
ABILITY_MUMMY = 152
ABILITY_LINGERING_AROMA = 268
ABILITY_WANDERING_SPIRIT = 254
ABILITY_WIMP_OUT = 193
ABILITY_EMERGENCY_EXIT = 194
ABILITY_BERSERK = 201
ABILITY_LONG_REACH = 203
ABILITY_AFTERMATH = 106
ABILITY_INNARDS_OUT = 215
ABILITY_MAGIC_GUARD = 98
ABILITY_DAMP = 6
ABILITY_RATTLED = 155
# Added — contact/hit abilities
ABILITY_GOOEY = 183
ABILITY_TANGLING_HAIR = 221
ABILITY_CURSED_BODY = 130
ABILITY_COLOR_CHANGE = 16
ABILITY_ANGER_POINT = 83
ABILITY_ELECTROMORPHOSIS = 280
ABILITY_SEED_SOWER = 269  # already wired in engine but re-listed
ABILITY_GULP_MISSILE = 241
ABILITY_SHIELD_DUST = 19
ITEM_COVERT_CLOAK = 1885
ITEM_PROTECTIVE_PADS = 880


def _to_int16(val: int) -> int:
    val = int(val) & 0xFFFF
    if val >= 0x8000:
        val -= 0x10000
    return val


def apply_toxic_chain_on_damaging_hit(
    battle: np.ndarray,
    atk_off: int,
    def_off: int,
    did_hit: bool,
    dmg: int,
    game_data,
    gen5_prng: Gen5PRNG,
    *,
    prerolled_roll: int | None = None,
) -> bool:
    """Apply Toxic Chain at Showdown's `onSourceDamagingHit` timing.

    Returns True when this helper owned the Toxic Chain event for the hit, even
    if the effect was blocked or did not land. Callers can use that to suppress
    a later duplicate Toxic Chain pass for the same hit.
    """
    if not did_hit or dmg <= 0:
        return False

    atk_off = int(atk_off)
    def_off = int(def_off)
    if int(battle[atk_off + 5]) != ABILITY_TOXIC_CHAIN:
        return False

    tc_def_ab = int(battle[def_off + 5])
    tc_def_item = int(battle[def_off + 6])
    if tc_def_ab == ABILITY_SHIELD_DUST or tc_def_item == ITEM_COVERT_CLOAK:
        return True

    roll = (
        int(prerolled_roll) if prerolled_roll is not None else int(gen5_prng.random(10))
    )
    if int(battle[def_off + 1]) <= 0:
        return True

    # Showdown still spends Toxic Chain's chance frame even when the target is
    # status-immune. The actual blocker lives inside `trySetStatus`, so route
    # through the shared status gate instead of writing tox directly.
    if roll < 3:
        _try_apply_status(
            battle,
            None,
            def_off,
            STATUS_TOXIC,
            game_data,
            gen5_prng,
            user_offset=atk_off,
            is_status_move=False,
        )
    return True


def apply_immediate_defender_ability_state_changes(
    battle: np.ndarray,
    atk_off: int,
    def_off: int,
    did_hit: bool,
    dmg: int,
    move_type: int,
    move_cat: int,
    move_flags: int,
) -> None:
    """Pre-apply non-PRNG defender ability state changes before the slower move.

    Showdown resolves a large part of `onDamagingHit` during the first move's
    action, before the slower Pokemon executes its own `runMove`. When those
    effects change boosts, items, abilities, or typing, the slower move and its
    later Update speedSorts must see the live post-hit state on the same turn.

    This helper intentionally excludes PRNG branches (Cursed Body, Toxic Chain)
    and forced-switch/KO damage branches (Aftermath, Innards Out, Wimp Out) so
    the existing late shared cascade can continue to own those mechanics.
    Toxic Debris is included here because the freshly-laid Toxic Spikes can
    affect a same-turn forced switch target before the slower move resolves.
    """
    if not did_hit or dmg <= 0:
        return

    atk_off = int(atk_off)
    def_off = int(def_off)
    def_hp = int(battle[def_off + 1])
    # Showdown still runs `onDamagingHit` callbacks after a fatal hit. Effects
    # that need the defender alive already self-block downstream, while effects
    # that target the attacker or the field (Gooey, Cotton Down, Toxic Debris,
    # Pickpocket, etc.) must still resolve on the KO hit.
    def_ab = int(battle[def_off + 5])
    def_max_hp = int(battle[def_off + 2])
    is_contact = (int(move_flags) & FLAG_CONTACT) != 0

    atk_ab_check = int(battle[atk_off + 5])
    atk_item_check = int(battle[atk_off + 6])
    if atk_ab_check == ABILITY_LONG_REACH or atk_item_check == ITEM_PROTECTIVE_PADS:
        is_contact = False

    if def_ab == ABILITY_TOXIC_DEBRIS and move_cat == 1:
        atk_side = 0 if atk_off < OFF_SIDE1 else 1
        haz_off = OFF_FIELD + (F_HAZARDS_0 if atk_side == 0 else F_HAZARDS_1)
        haz = int(battle[haz_off])
        layers = get_toxic_spikes_layers(haz)
        if layers < 2:
            battle[haz_off] = _to_int16(set_toxic_spikes(haz, layers + 1))

    if def_ab == ABILITY_BERSERK:
        was_above_half = (def_hp + dmg) * 2 > def_max_hp
        now_at_half = def_hp * 2 <= def_max_hp
        if was_above_half and now_at_half:
            b13 = int(battle[def_off + 13])
            battle[def_off + 13] = _to_int16(apply_boost_to_packed(b13, 8, 1))

    if def_ab == ABILITY_ELECTROMORPHOSIS:
        battle[def_off + 15] = np.int16(int(battle[def_off + 15]) | FLAG_CHARGE)

    if def_ab == ABILITY_ANGER_SHELL:
        was_above_half = (def_hp + dmg) * 2 > def_max_hp
        now_at_half = def_hp * 2 <= def_max_hp
        if was_above_half and now_at_half:
            b13 = int(battle[def_off + 13])
            b14 = int(battle[def_off + 14])
            b13 = apply_boost_to_packed(b13, 0, 1)
            b13 = apply_boost_to_packed(b13, 4, -1)
            b13 = apply_boost_to_packed(b13, 8, 1)
            b13 = apply_boost_to_packed(b13, 12, -1)
            b14 = apply_boost_to_packed(b14, 0, 1)
            battle[def_off + 13] = _to_int16(b13)
            battle[def_off + 14] = _to_int16(b14)

    if def_ab == ABILITY_COTTON_DOWN:
        b14 = int(battle[atk_off + 14])
        battle[atk_off + 14] = _to_int16(apply_boost_to_packed(b14, 0, -1))

    if def_ab == ABILITY_THERMAL_EXCHANGE and move_type == TYPE_FIRE:
        b13 = int(battle[def_off + 13])
        battle[def_off + 13] = _to_int16(apply_boost_to_packed(b13, 0, 1))

    if def_ab == ABILITY_JUSTIFIED and move_type == TYPE_DARK:
        b13 = int(battle[def_off + 13])
        battle[def_off + 13] = _to_int16(apply_boost_to_packed(b13, 0, 1))

    if def_ab == ABILITY_RATTLED and move_type in (TYPE_BUG, TYPE_DARK, TYPE_GHOST):
        b14 = int(battle[def_off + 14])
        battle[def_off + 14] = _to_int16(apply_boost_to_packed(b14, 0, 1))

    if def_ab == ABILITY_WATER_COMPACTION and move_type == TYPE_WATER:
        b13 = int(battle[def_off + 13])
        battle[def_off + 13] = _to_int16(apply_boost_to_packed(b13, 4, 2))

    if def_ab == ABILITY_STEAM_ENGINE and move_type in (TYPE_FIRE, TYPE_WATER):
        b14 = int(battle[def_off + 14])
        battle[def_off + 14] = _to_int16(apply_boost_to_packed(b14, 0, 6))

    if def_ab == ABILITY_WEAK_ARMOR and move_cat == 1:
        b13 = int(battle[def_off + 13])
        b14 = int(battle[def_off + 14])
        battle[def_off + 13] = _to_int16(apply_boost_to_packed(b13, 4, -1))
        battle[def_off + 14] = _to_int16(apply_boost_to_packed(b14, 0, 2))

    _ITEM_ABILITY_SHIELD = 1881
    atk_hp_live = int(battle[atk_off + 1])
    if (
        def_ab in (ABILITY_MUMMY, ABILITY_LINGERING_AROMA)
        and is_contact
        and atk_hp_live > 0
    ):
        atk_item_live = int(battle[atk_off + 6])
        if atk_item_live != _ITEM_ABILITY_SHIELD:
            battle[atk_off + 5] = np.int16(def_ab)

    if def_ab == ABILITY_WANDERING_SPIRIT and is_contact and atk_hp_live > 0:
        atk_item_ws = int(battle[atk_off + 6])
        def_item_ws = int(battle[def_off + 6])
        if atk_item_ws != _ITEM_ABILITY_SHIELD and def_item_ws != _ITEM_ABILITY_SHIELD:
            atk_ab = int(battle[atk_off + 5])
            battle[def_off + 5] = np.int16(atk_ab)
            battle[atk_off + 5] = np.int16(ABILITY_WANDERING_SPIRIT)

    _ABILITY_PICKPOCKET = 124
    _ABILITY_STICKY_HOLD = 60
    if def_ab == _ABILITY_PICKPOCKET and is_contact:
        atk_ab_pp = int(battle[atk_off + 5])
        def_item_pp = int(battle[def_off + 6])
        atk_item_pp = int(battle[atk_off + 6])
        atk_species_pp = int(battle[atk_off + 0])
        if (
            def_item_pp == 0
            and atk_item_pp != 0
            and atk_ab_pp != _ABILITY_STICKY_HOLD
            and not is_take_item_blocked_by_item_rule(atk_item_pp, atk_species_pp)
        ):
            battle[def_off + 6] = np.int16(atk_item_pp)
            battle[atk_off + 6] = 0

    if def_ab in (ABILITY_GOOEY, ABILITY_TANGLING_HAIR) and is_contact:
        atk_ab_gt = int(battle[atk_off + 5])
        atk_item_gt = int(battle[atk_off + 6])
        _CLEAR_BODY_SET = (29, 73, 230)
        _CONTRARY = 126
        _SIMPLE = 86
        _ITEM_CLEAR_AMULET = 1882
        if atk_ab_gt not in _CLEAR_BODY_SET and atk_item_gt != _ITEM_CLEAR_AMULET:
            drop = 1
            if atk_ab_gt == _CONTRARY:
                drop = -1
            elif atk_ab_gt == _SIMPLE:
                drop = 2
            b14 = int(battle[atk_off + 14])
            battle[atk_off + 14] = _to_int16(apply_boost_to_packed(b14, 0, -drop))

    if def_ab == ABILITY_COLOR_CHANGE and move_cat != 0 and 0 <= move_type < 18:
        cur_types = int(battle[def_off + 4]) & 0xFFFF
        t1 = cur_types & 0xFF
        t2 = (cur_types >> 8) & 0xFF
        if not (t1 == move_type and t2 == move_type):
            battle[def_off + 4] = np.int16(move_type | (move_type << 8))


def apply_cursed_body_on_damaging_hit(
    battle: np.ndarray,
    atk_off: int,
    def_off: int,
    move_id: int,
    move_idx: int,
    did_hit: bool,
    dmg: int,
    gen5_prng: Gen5PRNG,
    *,
    prerolled_roll: int | None = None,
    gen: int = 9,
) -> bool:
    """Apply Cursed Body at Showdown's `onDamagingHit` timing.

    Returns True when the event was handled for the hit, even if it was
    blocked or the roll failed. Callers can use that to suppress a later
    duplicate pass.
    """
    if not did_hit or int(dmg) <= 0:
        return False

    atk_off = int(atk_off)
    def_off = int(def_off)
    move_id = int(move_id)
    move_idx = int(move_idx)

    if int(battle[def_off + 5]) != ABILITY_CURSED_BODY:
        return False
    if move_idx < 0 or move_id == 165:  # Struggle
        return False

    atk_item_cb = int(battle[atk_off + 6])
    _ITEM_COVERT_CLOAK_CB = 816
    if atk_item_cb == _ITEM_COVERT_CLOAK_CB:
        return True

    atk_side = 0 if atk_off < OFF_SIDE1 else 1
    cb_side_base_dis = F_DISABLE_0 if atk_side == 0 else F_DISABLE_1
    cb_side_base_dt = F_DISABLE_TURNS_0 if atk_side == 0 else F_DISABLE_TURNS_1
    cur_dis_slot = int(battle[OFF_FIELD + cb_side_base_dis])
    cur_dis_turns = int(battle[OFF_FIELD + cb_side_base_dt])
    if cur_dis_slot >= 0 and cur_dis_turns > 0:
        return True

    roll_cb = (
        int(prerolled_roll) if prerolled_roll is not None else int(gen5_prng.random(10))
    )
    if roll_cb < 3:
        battle[OFF_FIELD + cb_side_base_dis] = np.int16(move_idx)
        if gen <= 3:
            disable_turns = int(gen5_prng.random(2, 6))
        elif gen == 4:
            disable_turns = int(gen5_prng.random(4, 8))
        else:
            disable_turns = 4
        battle[OFF_FIELD + cb_side_base_dt] = np.int16(disable_turns)
    return True


def apply_defender_abilities(
    battle: np.ndarray,
    move_id0: int,
    move_id1: int,
    user0_off: int,
    user1_off: int,
    target0_off: int,
    target1_off: int,
    damage0: int,
    damage1: int,
    hit0: bool,
    hit1: bool,
    game_data,
    gen5_prng: Gen5PRNG,
    move_idx0: int = -1,
    move_idx1: int = -1,
    skip_toxic_chain0: bool = False,
    skip_toxic_chain1: bool = False,
    skip_immediate_stateful_move0: bool = False,
    skip_immediate_stateful_move1: bool = False,
    skip_cursed_body0: bool = False,
    skip_cursed_body1: bool = False,
    gen: int = 9,
) -> bool:
    """Apply all defender ability effects for one turn. Mutates `battle`.

    Returns True if side 0 needs to force-switch (Wimp Out / Emergency Exit
    triggered on the player's mon).
    """
    move_type0 = int(np.asarray(game_data.move_type)[int(move_id0)])
    move_type1 = int(np.asarray(game_data.move_type)[int(move_id1)])
    move_cat0 = int(np.asarray(game_data.move_category)[int(move_id0)])
    move_cat1 = int(np.asarray(game_data.move_category)[int(move_id1)])
    move_flags0 = int(np.asarray(game_data.move_flags)[int(move_id0)])
    move_flags1 = int(np.asarray(game_data.move_flags)[int(move_id1)])
    is_contact0 = (move_flags0 & FLAG_CONTACT) != 0
    is_contact1 = (move_flags1 & FLAG_CONTACT) != 0

    # Strip contact for Long Reach attacker / Protective Pads holder
    # (Showdown: checkMoveMakesContact returns false). Affects Mummy /
    # Lingering Aroma / Wandering Spirit / Iron Barbs / Rough Skin /
    # Static / Flame Body etc.
    p0_ab_check = int(battle[int(user0_off) + 5])
    p1_ab_check = int(battle[int(user1_off) + 5])
    p0_item_check = int(battle[int(user0_off) + 6])
    p1_item_check = int(battle[int(user1_off) + 6])
    if p0_ab_check == ABILITY_LONG_REACH or p0_item_check == ITEM_PROTECTIVE_PADS:
        is_contact0 = False
    if p1_ab_check == ABILITY_LONG_REACH or p1_item_check == ITEM_PROTECTIVE_PADS:
        is_contact1 = False

    pairs = [
        (
            int(user0_off),
            int(target0_off),
            bool(hit0),
            int(damage0),
            move_type0,
            move_cat0,
            is_contact0,
            0,
            int(move_id0),
            int(move_idx0),
        ),
        (
            int(user1_off),
            int(target1_off),
            bool(hit1),
            int(damage1),
            move_type1,
            move_cat1,
            is_contact1,
            1,
            int(move_id1),
            int(move_idx1),
        ),
    ]

    for (
        atk_off,
        def_off,
        did_hit,
        dmg,
        mtype,
        mcat,
        is_contact,
        atk_side,
        mid,
        midx,
    ) in pairs:
        if not did_hit or dmg <= 0:
            continue
        def_ab = int(battle[def_off + 5])
        def_hp = int(battle[def_off + 1])
        skip_immediate_stateful = (atk_side == 0 and skip_immediate_stateful_move0) or (
            atk_side == 1 and skip_immediate_stateful_move1
        )
        skip_cursed_body = (atk_side == 0 and skip_cursed_body0) or (
            atk_side == 1 and skip_cursed_body1
        )

        # Toxic Debris — physical hit lays Toxic Spikes on attacker's side
        # even if Glimmora faints from the hit. The shared live-defender
        # helper below already owns the surviving-target case; only handle
        # the defender-fainted path here so we don't lay Toxic Spikes twice.
        handled_fainted_toxic_debris = False
        if (
            not skip_immediate_stateful
            and def_ab == ABILITY_TOXIC_DEBRIS
            and mcat == 1
            and def_hp <= 0
        ):
            haz_off = OFF_FIELD + (F_HAZARDS_0 if atk_side == 0 else F_HAZARDS_1)
            haz = int(battle[haz_off])
            layers = get_toxic_spikes_layers(haz)
            if layers < 2:
                battle[haz_off] = _to_int16(set_toxic_spikes(haz, layers + 1))
            handled_fainted_toxic_debris = True

        # Aftermath / Innards Out fire when the defender was KO'd by this hit.
        # Showdown data/abilities.ts:
        #   aftermath: onDamagingHitOrder 1 — `if (!target.hp &&
        #     this.checkMoveMakesContact(...)) this.damage(source.baseMaxhp/4)`
        #   innardsout: onDamagingHitOrder 1 — `if (!target.hp)
        #     this.damage(target.getUndynamaxedHP(damage))`
        # Aftermath is blocked by Magic Guard / Protective Pads on attacker
        # AND by Damp on either active mon (`onAnyDamage` returns false when
        # `effect.name === 'Aftermath'`). Innards Out is only blocked by
        # Magic Guard on attacker (no contact, so Protective Pads doesn't
        # apply, and it's not an explosion-class effect so Damp ignores it).
        if def_hp <= 0 and dmg > 0:
            atk_ab_aft = int(battle[atk_off + 5])
            atk_item_aft = int(battle[atk_off + 6])
            atk_hp_aft = int(battle[atk_off + 1])
            atk_maxhp_aft = int(battle[atk_off + 2])
            # Damp on either active mon suppresses Aftermath. Showdown checks
            # `getAllActive` for the Damp holder via onAnyDamage.
            from pokepy.core.constants import (
                OFF_SIDE0 as _OS0_AF,
                OFF_SIDE1 as _OS1_AF,
                OFF_META as _OM_AF,
                M_ACTIVE0 as _MA0_AF,
                M_ACTIVE1 as _MA1_AF,
                POKEMON_SIZE as _PS_AF,
            )

            _a0_af = int(battle[_OM_AF + _MA0_AF])
            _a1_af = int(battle[_OM_AF + _MA1_AF])
            _ab0_af = int(battle[_OS0_AF + _a0_af * _PS_AF + 5])
            _ab1_af = int(battle[_OS1_AF + _a1_af * _PS_AF + 5])
            damp_active = (_ab0_af == ABILITY_DAMP) or (_ab1_af == ABILITY_DAMP)
            blocked_aft = (
                atk_ab_aft == ABILITY_MAGIC_GUARD
                or atk_item_aft == ITEM_PROTECTIVE_PADS
                or damp_active
            )
            if (
                def_ab == ABILITY_AFTERMATH
                and is_contact
                and atk_hp_aft > 0
                and not blocked_aft
            ):
                aftermath_dmg = max(int(atk_maxhp_aft / 4), 1)
                new_atk_hp = max(0, atk_hp_aft - aftermath_dmg)
                battle[atk_off + 1] = np.int16(new_atk_hp)
            elif (
                def_ab == ABILITY_INNARDS_OUT
                and atk_hp_aft > 0
                and atk_ab_aft != ABILITY_MAGIC_GUARD
            ):
                # Innards Out is NOT blocked by Protective Pads (it's not
                # contact damage — it returns the actual HP overflow).
                innards_dmg = max(int(dmg), 1)
                new_atk_hp = max(0, atk_hp_aft - innards_dmg)
                battle[atk_off + 1] = np.int16(new_atk_hp)

        # Gulp Missile — Showdown abilities.ts:1688-1708. The missile still
        # fires after the defender faints from the hit; the only hard guards
        # are that the attacker is still alive/active and the defender is not
        # semi-invulnerable. Pokepy tracks the loaded Gulping/Gorging state in
        # singles meta slots instead of the temporary Showdown forms.
        gulp_state = get_gulp_missile_state(battle, def_off)
        if def_ab == ABILITY_GULP_MISSILE and gulp_state != GULP_MISSILE_NONE:
            atk_hp_gm = int(battle[atk_off + 1])
            def_actions_off = OFF_MOVES + (
                M_ACTIVE_MOVE_ACTIONS_0
                if def_off < OFF_SIDE1
                else M_ACTIVE_MOVE_ACTIONS_1
            )
            def_is_semi_invul = (
                int(battle[def_actions_off]) & ACTIVE_MOVE_ACTIONS_SEMI_INVUL
            ) != 0
            if atk_hp_gm > 0 and not def_is_semi_invul:
                atk_maxhp_gm = int(battle[atk_off + 2])
                missile_damage = max(int(atk_maxhp_gm / 4), 1)
                battle[atk_off + 1] = np.int16(max(0, atk_hp_gm - missile_damage))
                if gulp_state == GULP_MISSILE_GULPING:
                    apply_direct_stat_changes(
                        battle,
                        def_off,
                        atk_off,
                        (0, -1, 0, 0, 0, 0, 0),
                        stat_target=1,
                    )
                elif gulp_state == GULP_MISSILE_GORGING:
                    _try_apply_status(
                        battle,
                        mid,
                        atk_off,
                        STATUS_PARALYSIS,
                        game_data,
                        gen5_prng,
                        user_offset=def_off,
                        is_status_move=False,
                    )
            clear_gulp_missile_state(battle, def_off)

        # Cursed Body has no `target.hp` guard in Showdown's onDamagingHit
        # handler, so it still spends its 30% roll even if the defender was
        # KO'd by the hit that triggered it.
        if not skip_cursed_body:
            apply_cursed_body_on_damaging_hit(
                battle,
                atk_off,
                def_off,
                mid,
                midx,
                did_hit,
                dmg,
                gen5_prng,
                gen=gen,
            )

        skip_toxic_chain = (atk_side == 0 and skip_toxic_chain0) or (
            atk_side == 1 and skip_toxic_chain1
        )
        if not skip_toxic_chain:
            apply_toxic_chain_on_damaging_hit(
                battle,
                atk_off,
                def_off,
                did_hit,
                dmg,
                game_data,
                gen5_prng,
            )

        def_max_hp = int(battle[def_off + 2])

        # Fatal-hit Toxic Debris already applied above for the late-only path.
        # Running the generic immediate-state helper again would add a second
        # Toxic Spikes layer and incorrectly upgrade later switch-ins from psn
        # to tox.
        if skip_immediate_stateful or handled_fainted_toxic_debris:
            continue

        apply_immediate_defender_ability_state_changes(
            battle,
            atk_off,
            def_off,
            did_hit,
            dmg,
            mtype,
            mcat,
            FLAG_CONTACT if is_contact else 0,
        )

        # Anger Point / Electromorphosis: require crit flag / charge volatile
        # tracking that pokepy doesn't yet expose. Left as TODO.

    # Wimp Out / Emergency Exit — defender HP crosses 50% → force switch.
    # Track separately because we need to know which side switched.
    p0_needs_switch = False
    for (
        atk_off,
        def_off,
        did_hit,
        dmg,
        mtype,
        mcat,
        is_contact,
        atk_side,
        mid,
        midx,
    ) in pairs:
        if not did_hit or dmg <= 0:
            continue
        def_ab = int(battle[def_off + 5])
        if def_ab not in (ABILITY_WIMP_OUT, ABILITY_EMERGENCY_EXIT):
            continue
        def_hp = int(battle[def_off + 1])
        if def_hp <= 0:
            continue
        def_max_hp = int(battle[def_off + 2])
        was_above_half = (def_hp + dmg) * 2 > def_max_hp
        now_at_half = def_hp * 2 <= def_max_hp
        if not (was_above_half and now_at_half):
            continue
        # Determine if there's a valid bench mon to switch to
        side_for_def = 1 - atk_side  # the defender's side
        side_base = OFF_SIDE0 if side_for_def == 0 else OFF_SIDE1
        bench_alive = 0
        for i in range(6):
            slot_off = side_base + i * POKEMON_SIZE
            if slot_off == def_off:
                continue
            if int(battle[slot_off + 1]) > 0 and (int(battle[slot_off + 15]) & 1) == 0:
                bench_alive += 1
        if bench_alive == 0:
            continue
        if side_for_def == 0:
            p0_needs_switch = True
        else:
            # Side 1 auto-switches now (Wimp Out / Emergency Exit). Showdown
            # fires BeforeSwitchOut which triggers Regenerator / Natural Cure,
            # then runs the standard switch-in pipeline (clear volatiles +
            # choicelock, hazard damage, switch-in abilities).
            from pokepy.effects.abilities import (
                apply_regenerator_on_switch_out as _regen_wo,
                apply_natural_cure_on_switch_out as _natcure_wo,
                apply_switch_in_ability as _switch_in_wo,
            )
            from pokepy.effects.switch_state import (
                reset_incoming_switch_state as _reset_switch_in_wo,
            )
            from pokepy.effects.hazards import (
                apply_hazard_damage_on_switch as _hazard_wo,
            )

            _regen_wo(battle, def_off, True)
            _natcure_wo(battle, def_off, True)
            from pokepy.effects import auto_switch as _auto_switch_fn
            from pokepy.core.constants import (
                OFF_META,
                M_ACTIVE0,
                M_ACTIVE1,
                F_CHOICE_LOCK_1,
                F_LAST_MOVE_1,
                F_VOLATILE_1,
                F_LEECH_SEED_1,
                F_EXTENDED_VOLATILE_1,
                F_DESTINY_BOND_1,
                F_SUBSTITUTE_1,
                F_YAWN_TURNS_1,
                F_PERISH_COUNT_1,
                NEUTRAL_BOOSTS_13,
                NEUTRAL_BOOSTS_14,
            )

            active1 = int(battle[OFF_META + M_ACTIVE1])
            new_active1 = _auto_switch_fn(battle, OFF_SIDE1, active1)
            if new_active1 != active1:
                _pending_switch_slot_condition1_wo = int(
                    battle[OFF_FIELD + F_DESTINY_BOND_1]
                )
                battle[OFF_META + M_ACTIVE1] = np.int16(new_active1)
                # Showdown switchIn clears volatiles and per-side state
                # (sim/pokemon.ts:clearVolatile + sim/side.ts switchIn).
                for off in (F_CHOICE_LOCK_1, F_LAST_MOVE_1, F_DISABLE_1):
                    battle[OFF_FIELD + off] = -1
                battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1] = 0
                for off in (
                    F_VOLATILE_1,
                    F_LEECH_SEED_1,
                    F_DISABLE_TURNS_1,
                    F_EXTENDED_VOLATILE_1,
                    F_DESTINY_BOND_1,
                    F_SUBSTITUTE_1,
                    F_YAWN_TURNS_1,
                    F_PERISH_COUNT_1,
                ):
                    battle[OFF_FIELD + off] = 0
                new_p1 = OFF_SIDE1 + new_active1 * POKEMON_SIZE
                _reset_switch_in_wo(battle, new_p1, game_data)
                _wo_consumed_pending1 = apply_pending_wish_on_switch_in(
                    battle,
                    1,
                    new_p1,
                    state,
                    game_data,
                    _pending_switch_slot_condition1_wo,
                )
                if (
                    is_pending_wish_sentinel(_pending_switch_slot_condition1_wo)
                    and not _wo_consumed_pending1
                ):
                    battle[OFF_FIELD + F_DESTINY_BOND_1] = np.int16(
                        _pending_switch_slot_condition1_wo
                    )
                _hazard_wo(battle, new_p1, OFF_FIELD + F_HAZARDS_1)
                active0_now = int(battle[OFF_META + M_ACTIVE0])
                _switch_in_wo(
                    battle, new_p1, OFF_SIDE0 + active0_now * POKEMON_SIZE, True
                )
    return p0_needs_switch
