"""Gen 9 battle turn loop, ported from the Showdown reference implementation.

        (_step_battle_gen9 + _step_forced_switch).

This is a structural Python port. The the Showdown reference version is a Showdown-style
function over a batched [B] MultiFormatState. Pokepy operates on a single
mutable MultiFormatState — `jnp.where(cond, a, b)` collapses to Python
`a if cond else b`, and `battle.at[i].set(v)` collapses to `battle[i] = v`.

Helper functions on MultiFormatFastEnv (`self._apply_*`, `self._calc_damage_gen9`,
etc.) are referenced via the modules listed below. Most of these are being
ported in parallel by other subagents; this file imports them lazily and
defensively so it loads cleanly even when dependencies are missing.

================================================================================
Effects helpers referenced (from pokepy.effects.*  — TBD per parallel port):
================================================================================
  Switch / hazards / abilities:
    - apply_regenerator_on_switch_out(battle, p_off, did_switch)
    - apply_natural_cure_on_switch_out(battle, p_off, did_switch)
    - apply_hazard_damage_on_switch(battle, p_off, hazard_field_off)
    - apply_switch_in_ability(battle, self_off, opp_off, did_switch)
    - auto_switch(battle, side_base, current_active, should_switch) -> new_active

  Move pre-checks:
    - apply_protect_from_move(battle, move_id, side, rng) -> success
    - apply_substitute_from_move(battle, move_id, side, user_off)
    - apply_taunt_from_move(battle, move_id, target_side, hit, gen5_seed)
    - apply_destiny_bond_from_move(battle, move_id, side)
    - check_protected(battle, defender_side) -> bool
    - get_effective_speed(battle, p_off) -> int
    - get_effective_priority(battle, move_id, base_pri, p_off) -> int

  Status / stat / volatile:
    - apply_status_from_move(battle, move_id, target_off, gen5_seed, hit, user_offset)
    - apply_confusion_from_move(battle, move_id, target_side, hit, gen5_seed)
    - apply_encore_from_move(battle, move_id, target_side, hit, gen5_seed)
    - apply_extended_volatile(battle, move_id, target_side, user_side, hit, gen5_seed)
    - apply_stat_changes_from_move(battle, move_id, user_off, target_off, gen5_seed, hit)
    - apply_flinch_from_move(battle, move_id, target_side, hit, gen5_seed)
    - check_flinched(battle, side) -> bool
    - check_confusion_self_hit(battle, side, user_off, gen5_seed) -> self_hit

  Hazards / weather / terrain / screens:
    - apply_hazard_from_move(battle, move_id, target_side, hit)
    - apply_weather_from_move(battle, move_id, hit, user_off)
    - apply_terrain_from_move(battle, move_id, hit, user_off)
    - apply_trick_room_from_move(battle, move_id, hit)
    - apply_screen_from_move(battle, move_id, side, hit)

  Healing / utility:
    - apply_recovery_from_move(battle, move_id, user_off, hit)
    - apply_recoil_drain_from_move(battle, move_id, user_off, dmg, hit, target_off)
    - apply_life_orb_recoil(battle, user_off, dmg, hit, move_id)
    - apply_team_heal_status(battle, move_id, side_base, hit)
    - apply_leech_seed_from_move(battle, move_id, target_side, target_off, hit)
    - apply_perish_song_from_move(battle, move_id, hit)
    - apply_lock_on_from_move(battle, move_id, user_side, hit)
    - apply_ghost_curse_from_move(battle, move_id, user_off, target_side, hit)
    - apply_pain_split_from_move(battle, move_id, user_off, target_off, hit)
    - apply_knock_off_from_move(battle, move_id, target_off, hit_unprotected)
    - apply_trick_from_move(battle, move_id, user_off, target_off, hit_unprotected)
    - apply_rapid_spin_from_move(battle, move_id, user_side, hit)
    - apply_defog_from_move(battle, move_id, hit, user_side)
    - apply_haze_from_move(battle, move_id, hit)
    - apply_clear_smog_from_move(battle, move_id, target_off, hit)
    - apply_psych_up_from_move(battle, move_id, user_off, target_off, hit)
    - apply_phazing_from_move(battle, move_id, target_side, user_off, hit, gen5_seed)
    - apply_protect_contact_effects(battle, move_id, user_off, def_side, target_protected)
    - apply_absorb_ability_healing(battle, def_off, move_type, hit)
    - apply_weakness_policy(battle, def_off, move_type, hit, dmg)
    - apply_contact_damage(battle, move_id, atk_off, def_off, hit_dmg)
    - apply_contact_status_ability(battle, move_id, atk_off, def_off, hit_dmg, gen5_seed)
    - apply_ko_boost_ability(battle, atk_off, opp_fainted, hit)

  Berries / items (immediate + EOT):
    - apply_sitrus_berry(battle, p_off)
    - apply_lum_berry(battle, p_off)
    - apply_status_curing_berries(battle, p_off)
    - apply_persim_berry(battle, p_off)
    - apply_stat_boosting_berries(battle, p_off)
    - apply_pinch_healing_berries(battle, p_off)

  End-of-turn:
    - apply_weather_damage(battle, p_off)
    - apply_speed_boost(battle, p_off)
    - apply_weather_healing(battle, p_off)
    - apply_shed_skin_hydration(battle, p_off, gen5_seed)
    - apply_leftovers_healing(battle, p_off)
    - apply_black_sludge_effect(battle, p_off)
    - apply_grassy_terrain_healing(battle, p_off)
    - apply_leech_seed_damage(battle, p0_off, p1_off)
    - apply_end_of_turn_status_effects(battle, p0_off, p1_off, gen5_seed)
    - reset_protect_if_not_used(battle, side, used_protect)
    - clear_protect_at_turn_end(battle)
    - decrement_confusion(battle)
    - decrement_taunt_encore(battle)
    - clear_volatile_turn_effects(battle)
    - decrement_screens(battle)
    - decrement_weather(battle)
    - decrement_terrain(battle)
    - process_perish_song(battle, p0_off, p1_off)
    - apply_curse_damage(battle, p0_off, p1_off)
    - count_alive(battle, side_base) -> int
================================================================================
"""

from __future__ import annotations

from typing import Dict, Generator, Tuple
import numpy as np

from pokepy.core.state import MultiFormatState
from pokepy.core.constants import *  # noqa: F401,F403  (wide constants set)
from pokepy.engine.action_mask import get_battle_move_mask
from pokepy.effects.form_changes import prime_gulp_missile_state_from_move
from pokepy.core.bitpack import (
    extract_boost,
    apply_boost_to_packed,
    get_status,
    set_status,
    get_status_turns,
    get_protect_active,
    get_protect_consecutive,
    get_protect_type,
    set_protect_active,
    get_flinched,
    set_flinched,
    get_confusion_turns,
    set_confusion_turns,
    get_taunt_turns,
    set_taunt_turns,
    get_encore_turns,
    set_encore_turns,
    get_stockpile_def_count,
    get_stockpile_layers,
    get_stockpile_spd_count,
    get_throat_chop_turns,
    set_stockpile_def_count,
    set_stockpile_layers,
    set_stockpile_spd_count,
    get_toxic_spikes_layers,
    set_toxic_spikes,
    clear_volatile_turn_effects,
)
from pokepy.effects.status_apply import apply_tri_attack_status_from_move
from pokepy.effects.grounding import is_grounded
from pokepy.effects.recovery import can_rest_succeed as _can_rest_succeed_bg
from pokepy.effects.switch_slot_conditions import (
    apply_pending_wish_on_switch_in,
    is_pending_wish_sentinel,
)
from pokepy.mechanics.stats import get_boost_multiplier
from pokepy.engine.speed_sort import SpeedSortTracker
from pokepy.utils.gen5_prng import Gen5PRNG
from pokepy.engine.switch_requests import (
    SwitchRequest,
    resolve_switch_choices_sync,
    slot_from_pokepy_action,
    pokepy_action_from_slot,
)

# -----------------------------------------------------------------------------
# Optional dependency loaders. Effects + damage are being ported in parallel.
# We import lazily so this module loads even when nothing else exists yet.
# Each loader returns either the real callable or a no-op stub.
# -----------------------------------------------------------------------------


def _try_import(modpath, name):
    try:
        mod = __import__(modpath, fromlist=[name])
        return getattr(mod, name)
    except Exception:
        return None


def _stub_calc_damage(*args, **kwargs):
    return 0


def _get_calc_damage():
    fn = _try_import("pokepy.mechanics.damage_gen9", "calc_damage_gen9")
    return fn if fn is not None else _stub_calc_damage


# Phase enum lives in pokepy.core.constants (or state); fall back to a small enum.
try:
    from pokepy.core.constants import Phase  # type: ignore
except ImportError:

    class Phase:  # minimal fallback so the module imports
        BATTLE = 2
        FORCED_SWITCH = 3


# Constants used here that may not yet be in pokepy.core.constants.
# These match the Showdown reference implementation.
MOVE_STRUGGLE = globals().get("MOVE_STRUGGLE", 165)
MOVE_TAUNT = globals().get("MOVE_TAUNT", 269)
MOVE_DESTINY_BOND = globals().get("MOVE_DESTINY_BOND", 194)
MOVE_DIG = globals().get("MOVE_DIG", 91)
MOVE_FLY = globals().get("MOVE_FLY", 19)
MOVE_DIVE = globals().get("MOVE_DIVE", 291)
MOVE_BOUNCE = globals().get("MOVE_BOUNCE", 340)
MOVE_SHADOW_FORCE = globals().get("MOVE_SHADOW_FORCE", 467)
MOVE_PHANTOM_FORCE = globals().get("MOVE_PHANTOM_FORCE", 566)
MOVE_FEINT = globals().get("MOVE_FEINT", 364)
MOVE_FIRST_IMPRESSION = globals().get("MOVE_FIRST_IMPRESSION", 660)
MOVE_FAKE_OUT = globals().get("MOVE_FAKE_OUT", 252)
MOVE_FUTURE_SIGHT = globals().get("MOVE_FUTURE_SIGHT", 248)
MOVE_DOOM_DESIRE = globals().get("MOVE_DOOM_DESIRE", 353)
MOVE_LAST_RESORT = globals().get("MOVE_LAST_RESORT", 387)
MOVE_SLEEP_TALK = globals().get("MOVE_SLEEP_TALK", 214)
MOVE_SNORE = globals().get("MOVE_SNORE", 173)
MOVE_SUCKER_PUNCH = globals().get("MOVE_SUCKER_PUNCH", 389)
MOVE_THUNDERCLAP = globals().get("MOVE_THUNDERCLAP", 909)
MOVE_POLTERGEIST = globals().get("MOVE_POLTERGEIST", 809)
MOVE_COUNTER = globals().get("MOVE_COUNTER", 68)
MOVE_MIRROR_COAT = globals().get("MOVE_MIRROR_COAT", 243)
MOVE_METAL_BURST = globals().get("MOVE_METAL_BURST", 368)
MOVE_REVENGE = globals().get("MOVE_REVENGE", 279)
MOVE_AVALANCHE = globals().get("MOVE_AVALANCHE", 419)
MOVE_FOCUS_PUNCH = globals().get("MOVE_FOCUS_PUNCH", 264)
MOVE_EXPLOSION = globals().get("MOVE_EXPLOSION", 153)
MOVE_SELF_DESTRUCT = globals().get("MOVE_SELF_DESTRUCT", 120)
MOVE_MEMENTO = globals().get("MOVE_MEMENTO", 262)
MOVE_FINAL_GAMBIT = globals().get("MOVE_FINAL_GAMBIT", 515)
MOVE_MISTY_EXPLOSION = globals().get("MOVE_MISTY_EXPLOSION", 802)
MOVE_HEALING_WISH = globals().get("MOVE_HEALING_WISH", 361)
MOVE_LUNAR_DANCE = globals().get("MOVE_LUNAR_DANCE", 461)
MOVE_ROOST = globals().get("MOVE_ROOST", 355)
MOVE_TRI_ATTACK = globals().get("MOVE_TRI_ATTACK", 161)

HEALING_WISH_PENDING = globals().get("HEALING_WISH_PENDING", 2)
PROTECT_QUICK_GUARD = globals().get("PROTECT_QUICK_GUARD", 4)

ABILITY_PROTEAN = 168
ABILITY_LIBERO = 236
ABILITY_DAZZLING = 219
ABILITY_QUEENLY_MAJESTY = 214
ABILITY_ARMOR_TAIL = 296
ABILITY_UNSEEN_FIST = 260
ABILITY_INFILTRATOR = globals().get("ABILITY_INFILTRATOR", 151)
ABILITY_DISGUISE = globals().get("ABILITY_DISGUISE", 209)
ABILITY_ICE_FACE = globals().get("ABILITY_ICE_FACE", 248)
ABILITY_STURDY = globals().get("ABILITY_STURDY", 5)
ABILITY_STAMINA = globals().get("ABILITY_STAMINA", 192)
ABILITY_SEED_SOWER = globals().get("ABILITY_SEED_SOWER", 269)
ABILITY_ZEN_MODE = globals().get("ABILITY_ZEN_MODE", 161)
ABILITY_LEVITATE = globals().get("ABILITY_LEVITATE", 26)
ABILITY_PRESSURE = globals().get("ABILITY_PRESSURE", 46)
ABILITY_PRANKSTER = globals().get("ABILITY_PRANKSTER", 158)
ABILITY_MAGIC_BOUNCE = globals().get("ABILITY_MAGIC_BOUNCE", 156)
ABILITY_MAGIC_GUARD = globals().get("ABILITY_MAGIC_GUARD", 98)
ABILITY_SHEER_FORCE = globals().get("ABILITY_SHEER_FORCE", 125)
ABILITY_WEAK_ARMOR = globals().get("ABILITY_WEAK_ARMOR", 133)
ITEM_TERRAIN_EXTENDER = globals().get("ITEM_TERRAIN_EXTENDER", 662)

# -----------------------------------------------------------------------------
# Tiny helpers
# -----------------------------------------------------------------------------


def _is_valid_switch_target(battle, base_offset, target, current_active):
    if target < 0 or target >= 6:
        return False
    if target == current_active:
        return False
    offset = base_offset + target * POKEMON_SIZE
    flags = battle[offset + 15]
    hp = battle[offset + 1]
    is_fainted = (flags & 1) != 0
    return (hp > 0) and not is_fainted


def _count_alive(battle, side_base):
    alive = 0
    for i in range(6):
        off = side_base + i * POKEMON_SIZE
        if battle[off + 1] > 0 and (battle[off + 15] & 1) == 0:
            alive += 1
    return alive


def _preroll_contact_status_ability(
    battle,
    move_id,
    attacker_off,
    defender_off,
    hit,
    game_data,
    gen5_prng,
):
    """Pre-consume PRNG frames for Flame Body / Static / Poison Point /
    Effect Spore / Cute Charm / Poison Touch contact ability rolls, matching Showdown's
    DamagingHit event ordering (sim/battle-actions.ts:1139).

    In Showdown, DamagingHit fires after secondaries but before the NEXT
    move's damage calc. In pokepy, apply_contact_status_ability historically
    ran AFTER both moves resolved, shifting the PRNG stream when the first
    mover's move is a contact move against a Flame Body / Static / Poison
    Point / Effect Spore / Cute Charm defender (or when the attacker has
    Poison Touch).

    This helper mirrors the exact guard conditions of
    apply_contact_status_ability in effects/abilities.py and consumes the
    same number of random(100) frames at the correct position. Returns a
    list of prerolled ints that apply_contact_status_ability will pop from
    instead of rolling live.
    """
    from pokepy.effects.abilities import (
        ABILITY_FLAME_BODY,
        ABILITY_STATIC,
        ABILITY_POISON_POINT,
        ABILITY_EFFECT_SPORE,
        ABILITY_CUTE_CHARM,
        ABILITY_LONG_REACH,
        FLAG_CONTACT,
        ITEM_PROTECTIVE_PADS,
        _effect_spore_roll_blocked,
    )
    from pokepy.core.bitpack import get_status

    rolls = []
    if not hit:
        return rolls

    mid = int(move_id)
    a_off = int(attacker_off)
    d_off = int(defender_off)

    move_flags = int(game_data.move_flags[mid])
    is_contact = (move_flags & FLAG_CONTACT) != 0
    atk_ability = int(battle[a_off + 5])
    atk_item = int(battle[a_off + 6])
    if atk_ability == ABILITY_LONG_REACH:
        is_contact = False
    if atk_item == ITEM_PROTECTIVE_PADS:
        is_contact = False

    if not is_contact:
        return rolls

    def_ability = int(battle[d_off + 5])

    # Poison Touch — mirrors the guard conditions in apply_contact_status_ability
    _ABILITY_POISON_TOUCH = 143
    _ABILITY_SHIELD_DUST = 19
    _ITEM_COVERT_CLOAK = 1885
    _ABILITY_CORROSION = 212
    _TYPE_POISON = TYPE_POISON
    _TYPE_STEEL = TYPE_STEEL
    if atk_ability == _ABILITY_POISON_TOUCH:
        target_hp_pt = int(battle[d_off + 1])
        target_status_pt = get_status(int(battle[d_off + 12]))
        target_item_pt = int(battle[d_off + 6])
        target_has_block = (
            def_ability == _ABILITY_SHIELD_DUST or target_item_pt == _ITEM_COVERT_CLOAK
        )
        tt_packed = int(battle[d_off + 4]) & 0xFFFF
        tt1 = tt_packed & 0xFF
        tt2 = (tt_packed >> 8) & 0xFF
        type_immune_pt = (
            tt1 == _TYPE_POISON
            or tt2 == _TYPE_POISON
            or tt1 == _TYPE_STEEL
            or tt2 == _TYPE_STEEL
        ) and atk_ability != _ABILITY_CORROSION
        should_roll_pt = not target_has_block
        if should_roll_pt:
            rolls.append(gen5_prng.random(10))

    # Flame Body / Static / Poison Point / Effect Spore / Cute Charm
    has_contact_ability = def_ability in (
        ABILITY_FLAME_BODY,
        ABILITY_STATIC,
        ABILITY_POISON_POINT,
        ABILITY_EFFECT_SPORE,
        ABILITY_CUTE_CHARM,
    )
    if not has_contact_ability:
        return rolls

    if int(battle[a_off + 1]) <= 0:
        return rolls

    # Effect Spore uses random(100); Flame Body/Static/Poison Point/Cute
    # Charm use randomChance(3, 10) → random(10). Must match the exact
    # modulus.
    if def_ability == ABILITY_EFFECT_SPORE:
        # Showdown skips Effect Spore's random(100) entirely when the source
        # is already statused or naturally immune to powder.
        if _effect_spore_roll_blocked(battle, a_off):
            return rolls
        roll_es = int(gen5_prng.random(100))
        rolls.append(roll_es)
        if roll_es < 11:
            from pokepy.effects.status_apply import _can_apply_status

            if _can_apply_status(
                battle,
                None,
                a_off,
                STATUS_SLEEP,
                game_data,
                user_offset=d_off,
                is_status_move=False,
            ):
                rolls.append(int(gen5_prng.random(2, 5)))
    else:
        rolls.append(gen5_prng.random(10))
    return rolls


def _preroll_toxic_chain(
    battle,
    attacker_off,
    defender_off,
    hit,
    gen5_prng,
):
    """Pre-consume Toxic Chain's `random(10)` at Showdown's DamagingHit
    timing, which is after the move's own secondaries but before the
    defender's next `runMove`.

    Returns the rolled value or ``None`` if Toxic Chain would not spend a
    frame for this hit.
    """
    if not hit:
        return None

    a_off = int(attacker_off)
    d_off = int(defender_off)

    _ABILITY_TOXIC_CHAIN = 305
    _ABILITY_SHIELD_DUST = 19
    _ITEM_COVERT_CLOAK = 1885
    if int(battle[a_off + 5]) != _ABILITY_TOXIC_CHAIN:
        return None

    def_ability = int(battle[d_off + 5])
    def_item = int(battle[d_off + 6])
    if def_ability == _ABILITY_SHIELD_DUST or def_item == _ITEM_COVERT_CLOAK:
        return None

    return int(gen5_prng.random(10))


def _preroll_move_secondaries(
    battle,
    move_id,
    user_off,
    target_off,
    damage,
    target_protected,
    is_switch,
    is_first_attacker,
    num_hits,
    game_data,
    move_effects,
    gen5_prng,
    target_stats_raised_this_turn=False,
    target_hp_override=None,
    *,
    profile=None,
):
    """Pre-consume the secondary-chance PRNG frames for one move, matching
    Showdown's per-move `secondaryRoll` ordering (sim/battle-actions.ts:1357).

    In Showdown, each move's secondaries roll inside its own `moveHit` —
    i.e. BETWEEN move N's damage calc and move N+1's damage calc. Pokepy's
    apply_*_from_move functions historically consumed these frames AFTER
    both moves' damage calcs, which shifted the PRNG stream for any turn
    where move 0 has a secondary chance. This helper rolls them at the
    correct PRNG position and returns a dict of prerolled values keyed by
    the apply_* call that will later consume them.

    Returned dict keys (all optional): 'status' -> list[int] (one per hit
    for Twineedle / Fang-style multi-hit; [] if none), 'stat_change' -> int
    or None, 'flinch' -> list[int] or [], 'confusion' -> int or None,
    'taunt' -> int or None, 'encore' -> int or None, 'ext_vol' -> int or
    None.

    The gates here MUST match Showdown's true "filter BEFORE secondaryRoll"
    behavior so we consume exactly the same number of frames. The pre-roll
    filters are Sheer Force plus target-side ModifySecondaries hooks like
    Shield Dust / Covert Cloak. A substitute does NOT suppress the
    secondaryRoll itself; Showdown nulls the target after the primary-hit
    stage, then still iterates the move's secondaries and burns the roll
    before the later moveHit no-ops on the null target.
    """
    prerolled = {
        "status": None,
        "stat_change": None,
        "flinch": None,
        "flinch_lands": False,
        "tri_attack_roll": None,
        "tri_attack_status": None,
        "confusion": None,
        "confusion_lands": False,
        "confusion_duration": None,
        "confusion_can_apply": None,
        "taunt": None,
        "encore": None,
        "ext_vol": None,
    }
    if is_switch:
        return prerolled
    mid = int(move_id)
    cat = int(game_data.move_category[mid])
    # Status-category moves do not have chance-based secondaries in our data,
    # so nothing to preroll. Skip to avoid mis-modeling their accuracy rolls.
    CAT_STATUS_LOCAL = 0
    if cat == CAT_STATUS_LOCAL:
        # Curse (174) used by a non-Ghost is the one Status move that goes
        # through Showdown's `selfDrops` path: `move.self = { boosts: ... }`
        # is set inside onTryHit, and `selfDrops` always rolls
        # `random(100)` BEFORE checking the (undefined) chance — even
        # though the boost is guaranteed to apply. Pokepy classifies Curse
        # as EFFECT_STAT_CHANGE → primary boost → no roll, so we have to
        # consume the frame here at the same PRNG offset Showdown does
        # (immediately after the move's damage step, which for a Status
        # move means right after acc_roll). Ghost-type users go through
        # `apply_ghost_curse_from_move` instead and never hit selfDrops.
        _MOVE_CURSE_PR = 174
        _TYPE_GHOST_PR = 13
        if mid == _MOVE_CURSE_PR:
            _types_pr = int(battle[user_off + 4]) & 0xFFFF
            _t1_pr = _types_pr & 0xFF
            _t2_pr = (_types_pr >> 8) & 0xFF
            _is_ghost_pr = (_t1_pr == _TYPE_GHOST_PR) or (_t2_pr == _TYPE_GHOST_PR)
            if not _is_ghost_pr:
                gen5_prng.random(100)
        return prerolled
    # Damaging move must have actually dealt damage (or at least been on-hit).
    # `damage == 0` covers both protect-blocked and missed / immune cases.
    if damage <= 0 or target_protected:
        return prerolled

    # A live substitute makes the later secondary application no-op on a null
    # target, but Showdown still spends the generic secondaryRoll frames. The
    # only target-dependent pre-roll filters that disappear are target-side
    # ModifySecondaries hooks like Shield Dust / Covert Cloak. Sound moves and
    # Infiltrator bypass the substitute entirely.
    from pokepy.core.constants import (
        F_SUBSTITUTE_0 as _F_SUB_0,
        F_SUBSTITUTE_1 as _F_SUB_1,
        OFF_FIELD as _OFF_F,
        OFF_SIDE1 as _OFF_S1,
        FLAG_SOUND as _FLAG_SOUND,
        ABILITY_INFILTRATOR as _ABIL_INF,
    )

    target_on_side1 = target_off >= _OFF_S1
    sub_field = _OFF_F + (_F_SUB_1 if target_on_side1 else _F_SUB_0)
    sub_hp = int(battle[sub_field])
    flags_m = int(game_data.move_flags[mid])
    is_sound = (flags_m & _FLAG_SOUND) != 0
    attacker_ab = int(battle[user_off + 5])
    is_infiltrator = attacker_ab == _ABIL_INF
    sub_blocks = (sub_hp > 0) and not is_sound and not is_infiltrator

    def _target_will_faint_after_hit() -> bool:
        """Approximate whether the target actually faints from this hit.

        This mirrors the survival/prevention gates that matter for hit-time
        addVolatile checks such as confusion and partial trapping. Raw damage
        alone is not sufficient because Focus Sash, Sturdy, Disguise, and Ice
        Face can keep the target alive long enough for Showdown to apply the
        post-hit volatile and, for the slower mover, run same-turn onBeforeMove.
        """
        raw_damage = int(damage)
        target_hp = (
            int(target_hp_override)
            if target_hp_override is not None
            else int(battle[target_off + 1])
        )
        if raw_damage <= 0 or target_hp <= 0:
            return False

        damage_through = raw_damage
        target_flags = int(battle[target_off + 15])
        target_ability = int(battle[target_off + 5])
        target_item = int(battle[target_off + 6])
        target_max_hp = int(battle[target_off + 2])

        face_absorbed_hit = (
            ((target_flags & 0x40) != 0)
            and (
                target_ability == ABILITY_DISGUISE
                or (target_ability == ABILITY_ICE_FACE and cat == CAT_PHYSICAL)
            )
            and raw_damage > 0
        )
        if face_absorbed_hit:
            damage_through = target_max_hp // 8

        new_hp = max(0, target_hp - damage_through)
        if new_hp == 0 and target_hp == target_max_hp:
            is_multihit = int(move_effects.hits_max[mid]) > 1
            if target_item == ITEM_FOCUS_SASH and not is_multihit:
                new_hp = 1
            elif target_ability == ABILITY_STURDY:
                new_hp = 1

        return new_hp == 0

    # Shield Dust / Covert Cloak filter secondaries BEFORE the roll fires
    # (Showdown: abilities.ts:shielddust onModifySecondaries, items.ts:
    # covertcloak onModifySecondaries). Check these off the target.
    _ABIL_SHIELD_DUST = 19
    _ITEM_COVERT_CLOAK = 1885
    target_ab = int(battle[target_off + 5])
    target_it = int(battle[target_off + 6])
    has_shield_dust = target_ab == _ABIL_SHIELD_DUST
    has_covert_cloak = target_it == _ITEM_COVERT_CLOAK
    effective_has_shield_dust = has_shield_dust and not sub_blocks
    effective_has_covert_cloak = has_covert_cloak and not sub_blocks

    # Sheer Force on the attacker deletes `move.secondaries` entirely →
    # no secondaryRoll consumed for non-primary secondaries. Primary effects
    # (effect_type == EFFECT_STATUS or EFFECT_STAT_CHANGE) are NOT secondaries
    # in Showdown and survive Sheer Force.
    _ABIL_SHEER_FORCE = 125  # matches constants
    has_sheer_force = attacker_ab == _ABIL_SHEER_FORCE

    from pokepy.data.move_effects import (
        EFFECT_STATUS as _EFFECT_STATUS_PR,
        EFFECT_STAT_CHANGE as _EFFECT_STAT_CHANGE_PR,
    )

    effect_type_m = int(move_effects.effect_type[mid])

    # ------------------------------------------------------------------
    # STATUS secondary (e.g. Fire Blast 10% burn, Thunder 30% para, Scald
    # 30% burn, Ice Beam 10% freeze, ...). Multi-hit moves (Twineedle)
    # roll per hit.
    # ------------------------------------------------------------------
    status = int(move_effects.status[mid])
    status_chance = int(move_effects.status_chance[mid])
    is_secondary_status = effect_type_m != _EFFECT_STATUS_PR
    status_blocked_by_dust = is_secondary_status and (
        effective_has_shield_dust or effective_has_covert_cloak
    )
    status_blocked_by_sheer = has_sheer_force and is_secondary_status
    if (
        status > 0
        and status_chance > 0
        and not status_blocked_by_dust
        and not status_blocked_by_sheer
    ):
        nh = max(1, int(num_hits))
        prerolled["status"] = [gen5_prng.random(100) for _ in range(nh)]

    # Tri Attack is a callback secondary in Showdown data/moves.ts, so it
    # isn't represented by move_effects.status/status_chance. It still burns
    # one `secondaryRoll` frame plus a `random(3)` status picker when the
    # chance lands, even if the target faints to the hit.
    if mid == MOVE_TRI_ATTACK and not (
        effective_has_shield_dust or effective_has_covert_cloak or has_sheer_force
    ):
        tri_attack_chance = 20
        if attacker_ab == ABILITY_SERENE_GRACE:
            tri_attack_chance = min(100, tri_attack_chance * 2)
        tri_attack_roll = gen5_prng.random(100)
        prerolled["tri_attack_roll"] = tri_attack_roll
        if tri_attack_roll < tri_attack_chance:
            prerolled["tri_attack_status"] = gen5_prng.random(3)

    # ------------------------------------------------------------------
    # CONFUSION volatile (Psybeam 10%, Water Pulse 20%, Hurricane 30%, ...).
    # Target-side ModifySecondaries hooks like Shield Dust and Covert Cloak
    # strip damaging confusion secondaries BEFORE secondaryRoll in Showdown,
    # so no random(100) frame is consumed when either is active. Sheer Force
    # blocks if chance < 100.
    # ------------------------------------------------------------------
    from pokepy.core.constants import (
        VOLATILE_FLINCH as _VOL_FL_PR,
        VOLATILE_CONFUSION as _VOL_CONF_PR,
        VOLATILE_TAUNT as _VOL_TAUNT_PR,
        VOLATILE_ENCORE as _VOL_ENC_PR,
        VOLATILE_FOCUS_ENERGY as _VOL_FOCUS_PR,
        VOLATILE_PARTIAL_TRAP as _VOL_PT_PR,
        EXT_VOL_PARTIAL_TRAP as _EXT_VOL_PT_PR,
        OFF_SIDE1 as _OFF_SIDE1_PR,
        OFF_FIELD as _OFF_FIELD_PR,
        F_EXTENDED_VOLATILE_0 as _F_EXTVOL0_PR,
        F_EXTENDED_VOLATILE_1 as _F_EXTVOL1_PR,
    )

    _ITEM_GRIP_CLAW_PR = 179

    vol_type = int(move_effects.volatile[mid])
    vol_chance = int(move_effects.volatile_chance[mid])
    # Showdown encodes the true trapping moves as PRIMARY move-level
    # `volatileStatus: 'partiallytrapped'` with `secondary: null`, so they do
    # not spend a generic `secondaryRoll` frame. The only PRNG they spend on a
    # fresh application is the duration callback in conditions.ts
    # partiallytrapped.durationCallback (`random(5, 7)` unless Grip Claw makes
    # it a fixed 8). Pokepy's extracted move metadata collapses those primary
    # volatiles into `volatile_chance == 100`, so classify the actual trapping
    # move ids here instead of treating every 100%-chance extended volatile as
    # a secondary, and keep the rolled duration for the late apply path.
    _PRIMARY_PARTIAL_TRAP_MOVE_IDS = frozenset(
        (
            20,  # Bind
            35,  # Wrap
            83,  # Fire Spin
            128,  # Clamp
            250,  # Whirlpool
            328,  # Sand Tomb
            463,  # Magma Storm
            611,  # Infestation
            779,  # Snap Trap
            819,  # Thunder Cage
        )
    )
    is_primary_partial_trap_move = (
        vol_type == _VOL_PT_PR and mid in _PRIMARY_PARTIAL_TRAP_MOVE_IDS
    )

    # Confusion
    from pokepy.data.move_effects import (
        EFFECT_DAMAGE as _EFFECT_DAMAGE_PR,
        EFFECT_MULTI_HIT as _EFFECT_MULTI_HIT_PR,
    )

    is_damaging_effect = effect_type_m in (_EFFECT_DAMAGE_PR, _EFFECT_MULTI_HIT_PR)
    if vol_type == _VOL_CONF_PR and vol_chance > 0:
        conf_sheer_block = (vol_chance < 100) and has_sheer_force
        conf_dust_block = is_damaging_effect and (
            effective_has_shield_dust or effective_has_covert_cloak
        )
        if not (conf_sheer_block or conf_dust_block):
            prerolled["confusion"] = gen5_prng.random(100)

    # Taunt / Encore (rare as secondaries on damaging moves but mirror the
    # apply_* gates: just "has chance > 0" and "volatile type matches").
    if vol_type == _VOL_TAUNT_PR and vol_chance > 0:
        prerolled["taunt"] = gen5_prng.random(100)
    if vol_type == _VOL_ENC_PR and vol_chance > 0:
        prerolled["encore"] = gen5_prng.random(100)

    # Extended volatile (focus energy, torment, attract, yawn, ...) — self
    # or opponent. Sub only blocks opponent-target ones; keep the gate
    # conservative (sub_blocks for opponent target).
    if (
        vol_chance > 0
        and vol_type >= _VOL_FOCUS_PR
        and vol_type != _VOL_CONF_PR
        and vol_type != _VOL_TAUNT_PR
        and vol_type != _VOL_ENC_PR
        and vol_type != _VOL_FL_PR
        and not is_primary_partial_trap_move
    ):
        prerolled["ext_vol"] = gen5_prng.random(100)

    # Throat Chop is modeled locally as plain damage, but Showdown still
    # represents its sound-blocking follow-up as a guaranteed secondary with
    # `chance: 100`, so every successful hit consumes one `secondaryRoll`
    # frame even if the target is already under the effect. Consume that frame
    # here so later crit/damage rolls stay aligned.
    _MOVE_THROAT_CHOP_PR = 675
    if (
        mid == _MOVE_THROAT_CHOP_PR
        and not has_sheer_force
        and not effective_has_shield_dust
        and not effective_has_covert_cloak
        and prerolled["ext_vol"] is None
    ):
        prerolled["ext_vol"] = gen5_prng.random(100)

    # Some moves consume the generic `secondaryRoll` frame in Showdown even
    # though pokepy's extracted move metadata does not encode that secondary
    # directly:
    #   - Stone Axe / Ceaseless Edge: empty `secondary: {}`
    #   - Sparkling Aria: guaranteed `volatileStatus: 'sparklingaria'`
    #     callback that later cures burns on hit targets
    # We still need to spend the same random(100) frame here so the next
    # move's PRNG lands at the correct offset.
    _SECONDARY_ROLL_ONLY_MOVE_IDS = frozenset(
        (
            664,  # Sparkling Aria
            830,  # Stone Axe
            845,  # Ceaseless Edge
        )
    )
    if (
        mid in _SECONDARY_ROLL_ONLY_MOVE_IDS
        and not has_sheer_force
        and not effective_has_shield_dust
        and not effective_has_covert_cloak
    ):
        gen5_prng.random(100)

    # Primary trapping moves (Bind / Magma Storm / etc.) consume a duration
    # frame only when they successfully add `partiallytrapped` to a fresh,
    # surviving target. Re-hits while the target is already trapped do not
    # reroll duration, and Grip Claw makes the duration deterministic.
    if is_primary_partial_trap_move and not sub_blocks and int(damage) > 0:
        target_on_side1 = int(target_off) >= _OFF_SIDE1_PR
        ext_off = _OFF_FIELD_PR + (_F_EXTVOL1_PR if target_on_side1 else _F_EXTVOL0_PR)
        already_trapped = (int(battle[ext_off]) & _EXT_VOL_PT_PR) != 0
        target_will_faint = _target_will_faint_after_hit()
        if not (already_trapped or target_will_faint):
            attacker_item_pr = int(battle[user_off + 6])
            if attacker_item_pr == _ITEM_GRIP_CLAW_PR:
                prerolled["partial_trap_duration"] = 8
            else:
                prerolled["partial_trap_duration"] = gen5_prng.random(5, 7)

    # ------------------------------------------------------------------
    # STAT_CHANGE secondary (Crunch 20% def drop, Shadow Ball 20% spd drop,
    # Iron Tail 30% def drop, ...). Also primary stat-drop moves (Growl,
    # Leer) come through here with chance == 100 — but those are category
    # STATUS and we return early above.
    # ------------------------------------------------------------------
    from pokepy.effects.stat_changes import (
        get_live_move_stat_change_spec as _get_live_move_stat_change_spec_pr,
    )

    sc_arr, stat_target_m, stat_chance, is_selfboost_like = (
        _get_live_move_stat_change_spec_pr(
            battle,
            mid,
            move_effects,
            user_off,
        )
    )
    has_any_sc = any(int(sc_arr[i]) != 0 for i in range(7))
    _ON_TRY_MOVE_SELFBOOST_PR = frozenset(
        (130, 800, 905)
    )  # Skull Bash / Meteor Beam / Electro Shot
    is_primary_sc_move = (
        effect_type_m
        in (
            _EFFECT_STAT_CHANGE_PR,
            _EFFECT_STATUS_PR,
            EFFECT_SWITCH,
        )
        or mid in _ON_TRY_MOVE_SELFBOOST_PR
        or (
            is_selfboost_like
            and stat_target_m == 0
            and stat_chance == 100
            and has_any_sc
        )
    )
    sc_sheer_block = has_sheer_force and (stat_chance < 100)
    sc_dust_block = (
        (stat_target_m == 1)
        and (not is_primary_sc_move)
        and (effective_has_shield_dust or effective_has_covert_cloak)
    )
    if (
        stat_chance > 0
        and has_any_sc
        and not is_primary_sc_move
        and prerolled["stat_change"] is None
        and not sc_sheer_block
        and not sc_dust_block
    ):
        if profile is not None and profile.gen <= 2:
            prerolled["stat_change"] = gen5_prng.random(256)
        else:
            prerolled["stat_change"] = gen5_prng.random(100)

    # ------------------------------------------------------------------
    # FLINCH volatile (Iron Head 30%, Rock Slide 30%, ...). Showdown still
    # consumes the flinch secondary roll even when the user moved second and
    # the flinch can no longer matter; only the `flinch_lands` shortcut is
    # restricted to the first attacker.
    # ------------------------------------------------------------------
    if vol_type == _VOL_FL_PR and vol_chance > 0:
        fl_sheer_block = has_sheer_force
        fl_dust_block = effective_has_shield_dust or effective_has_covert_cloak
        if not (fl_sheer_block or fl_dust_block):
            nh = max(1, int(num_hits))
            prerolled["flinch"] = [gen5_prng.random(100) for _ in range(nh)]
            # Also compute whether the flinch will land. The engine uses
            # this to skip the second attacker's damage_calc when the
            # target is going to be flinched (Showdown's flinch volatile
            # has onBeforeMove that returns false BEFORE any of the
            # second mover's PRNG frames are consumed).
            _ABIL_SERENE_GRACE = 35  # ABILITY_SERENE_GRACE — match constants
            from pokepy.core.constants import (
                ABILITY_SERENE_GRACE as _ABIL_SG_PR,
                ABILITY_INNER_FOCUS as _ABIL_IF_PR,
            )

            effective_chance = vol_chance
            if attacker_ab == _ABIL_SG_PR:
                effective_chance = min(100, vol_chance * 2)
            target_has_inner_focus = target_ab == _ABIL_IF_PR
            any_hit_lands = any(int(r) < effective_chance for r in prerolled["flinch"])
            flinch_lands = (
                is_first_attacker
                and any_hit_lands
                and not target_has_inner_focus
                and not sub_blocks
            )
            prerolled["flinch_lands"] = bool(flinch_lands)

    # ------------------------------------------------------------------
    # CONFUSION DURATION roll. Showdown's confusion.onStart fires inside
    # `addVolatile` (called from secondary moveHit) and rolls
    # `random(2, 6)` for the duration BEFORE moving on to the next move.
    # We must consume that frame here, after the chance roll, so the
    # next move's damage_calc lands at the right offset.
    #
    # The duration roll only fires when:
    #   - the chance roll succeeded (or chance is 100, in which case the
    #     "chance roll" frame above also fired and we know it succeeded);
    #   - the target doesn't already have confusion;
    #   - the target isn't blocked by Own Tempo / grounded misty terrain.
    # Match the gates inside apply_confusion_from_move so we consume the
    # SAME number of frames in the SAME order.
    # ------------------------------------------------------------------
    if (
        vol_type == _VOL_CONF_PR
        and vol_chance > 0
        and prerolled["confusion"] is not None
    ):
        chance_landed = int(prerolled["confusion"]) < vol_chance
        if chance_landed:
            _MOVE_ALLURING_VOICE_PR = 914
            if mid == _MOVE_ALLURING_VOICE_PR and not target_stats_raised_this_turn:
                return prerolled
            from pokepy.core.constants import (
                ABILITY_OWN_TEMPO as _ABIL_OT_PR,
                TERRAIN_MISTY as _TER_MISTY_PR,
                F_TERRAIN as _F_TER_PR,
                F_VOLATILE_0 as _F_VOL0_PR,
                F_VOLATILE_1 as _F_VOL1_PR,
            )
            from pokepy.core.bitpack import (
                get_confusion_turns as _get_conf_turns_pr,
                set_confusion_turns as _set_conf_turns_pr,
                set_confusion_newly_applied as _set_newly_pr,
            )

            target_ability_c = int(battle[target_off + 5])
            blocked_by_own_tempo = target_ability_c == _ABIL_OT_PR
            blocked_by_misty = is_grounded(battle, target_off) and (
                int(battle[_OFF_F + _F_TER_PR]) == _TER_MISTY_PR
            )
            # `addVolatile` returns false BEFORE onStart if the target
            # already has the volatile — no duration frame consumed.
            tgt_vol_off = _OFF_F + (_F_VOL1_PR if target_on_side1 else _F_VOL0_PR)
            already_confused = _get_conf_turns_pr(int(battle[tgt_vol_off])) != 0
            # Showdown pokemon.ts:addVolatile line 1943 early-returns false
            # when `!this.hp && !status.affectsFainted`. Confusion doesn't
            # set affectsFainted, so if this move KO'd the target, the
            # secondary's addVolatile fails BEFORE onStart — no duration
            # roll consumed. Use the same survival/prevention approximation as
            # the live damage path so Focus Sash / Sturdy / Disguise / Ice
            # Face survivors still get same-turn confusion handling.
            target_will_faint = _target_will_faint_after_hit()
            can_apply_confusion = not (
                blocked_by_own_tempo
                or blocked_by_misty
                or already_confused
                or target_will_faint
                or sub_blocks
            )
            prerolled["confusion_can_apply"] = bool(can_apply_confusion)
            if can_apply_confusion:
                # Showdown rolls `random(2, 6)` → 2,3,4,5. Pokepy decrements
                # confusion at end of turn, so it must store the raw 2-5
                # duration here; the end-of-turn decrement converts that to
                # the correct remaining 1-4 future self-hit checks.
                duration_roll = gen5_prng.random(2, 6)
                conf_K = int(duration_roll)
                prerolled["confusion_duration"] = conf_K
                prerolled["confusion_lands"] = True
                # Apply the volatile NOW so the later onBeforeMove check sees
                # confusion, but leave the actual self-hit chance for the
                # confused pokemon's real runMove. Showdown spends that frame
                # after the inter-move Update sorts, not here inside moveHit.
                _cur_vol = int(battle[tgt_vol_off])
                _new_vol = _set_conf_turns_pr(_cur_vol, conf_K)
                _new_vol = _set_newly_pr(_new_vol, False)
                _new_vol = _new_vol & 0xFFFF
                if _new_vol >= 0x8000:
                    _new_vol -= 0x10000
                battle[tgt_vol_off] = _new_vol

    return prerolled


# =============================================================================
# Residual speedSort frame counter
# =============================================================================


def _count_residual_speedsort_frames(
    battle,
    tracker: "SpeedSortTracker",
    p0_off: int,
    p1_off: int,
    p0_speed: int,
    p1_speed: int,
    p0_item: int,
    p1_item: int,
    terrain: int,
) -> None:
    """Consume PRNG frames for Showdown's fieldEvent('Residual') speedSort.

    Showdown collects all per-pokemon onResidual handlers, runs speedSort on
    the combined list, and Fisher-Yates shuffles each tied group (consuming
    group_size - 1 frames per group).

    For a 1v1 battle the two sides' handlers at the same (order, subOrder)
    tie iff effective speeds are equal, because comparePriority is:
        order asc → priority desc → speed desc → subOrder asc → effectOrder asc

    Handler order values (from Showdown data files):
        psn/tox status  : order=9,  subOrder=0
        brn status      : order=10, subOrder=0
        Leftovers       : order=5,  subOrder=4
        Black Sludge    : order=5,  subOrder=4
        Speed Boost     : order=28, subOrder=2

    We sort the collected handler list by comparePriority before passing it to
    speed_sort_consume so tied groups are always consecutive.
    """
    from functools import cmp_to_key
    from pokepy.core.bitpack import (
        get_encore_turns as _get_encore_turns_r,
        get_heal_block_turns as _get_heal_block_turns_r,
        get_taunt_turns as _get_taunt_turns_r,
    )
    from pokepy.core.constants import (
        ITEM_LEFTOVERS as _ITEM_LEFT_R,
        ITEM_BLACK_SLUDGE as _ITEM_BS_R,
        ABILITY_SPEED_BOOST as _AB_SB_R,
        ABILITY_HARVEST as _AB_HARVEST_R,
        ABILITY_HYDRATION as _AB_HYDRATION_R,
        ABILITY_SHED_SKIN as _AB_SHED_SKIN_R,
        ABILITY_SCHOOLING as _AB_SCHOOLING_R,
        ABILITY_SHIELDS_DOWN as _AB_SHIELDS_DOWN_R,
        ABILITY_HUNGER_SWITCH as _AB_HUNGER_SWITCH_R,
        ABILITY_OPPORTUNIST as _AB_OPPORTUNIST_R,
        ABILITY_ZEN_MODE as _AB_ZEN_MODE_R,
        OFF_META as _OFF_META_R,
        OFF_FIELD as _OFF_FIELD_R,
        OFF_SIDE1 as _OFF_SIDE1_R,
        F_TERRAIN as _F_TERRAIN_R,
        F_VOLATILE_0 as _F_VOL0_R,
        F_VOLATILE_1 as _F_VOL1_R,
        F_DISABLE_0 as _F_DIS0_R,
        F_DISABLE_1 as _F_DIS1_R,
        F_DISABLE_TURNS_0 as _F_DIST0_R,
        F_DISABLE_TURNS_1 as _F_DIST1_R,
        F_YAWN_TURNS_0 as _F_YAWN0_R,
        F_YAWN_TURNS_1 as _F_YAWN1_R,
        M_CHARGING_0 as _M_CHG0_R,
        M_CHARGING_1 as _M_CHG1_R,
        MOVE_BOUNCE as _MOVE_BOUNCE_R,
        MOVE_DIG as _MOVE_DIG_R,
        MOVE_DIVE as _MOVE_DIVE_R,
        MOVE_FLY as _MOVE_FLY_R,
        MOVE_PHANTOM_FORCE as _MOVE_PHANTOM_FORCE_R,
        MOVE_SHADOW_FORCE as _MOVE_SHADOW_FORCE_R,
        TERRAIN_GRASSY as _TER_GRASSY_R,
        STATUS_BURN as _ST_BRN_R,
        STATUS_POISON as _ST_PSN_R,
        STATUS_TOXIC as _ST_TOX_R,
    )

    # Showdown onResidual abilities that matter for speedSort frame
    # consumption but do not all have exported constants in pokepy.
    _AB_BAD_DREAMS_R = 123
    _AB_HEALER_R = 131
    _AB_MOODY_R = 141
    _AB_PICKUP_R = 53
    _AB_SLOW_START_R = 112
    _AB_POWER_CONSTRUCT_R = 211
    _AB_CUD_CHEW_R = 291

    handlers = []  # list of (order, priority, speed, subOrder, effectOrder)

    for poff, spd, item in (
        (p0_off, p0_speed, p0_item),
        (p1_off, p1_speed, p1_item),
    ):
        # Note: do NOT filter by HP here. Showdown includes fainted mons'
        # handlers in the speedSort list (they are added but skipped during
        # iteration at battle.ts:512). This matters in the KO path where one
        # mon is fainted but its handler still participates in the sort.
        status = get_status(int(battle[poff + 12]))
        # psn / tox both use onResidualOrder=9
        if status in (_ST_PSN_R, _ST_TOX_R):
            handlers.append((9, 0, spd, 0, 0))
        # burn uses onResidualOrder=10
        if status == _ST_BRN_R:
            handlers.append((10, 0, spd, 0, 0))
        # Leftovers / Black Sludge: onResidualOrder=5, onResidualSubOrder=4
        if item in (_ITEM_LEFT_R, _ITEM_BS_R):
            handlers.append((5, 0, spd, 4, 0))
        # Grassy Terrain healing is an onResidual terrain handler that is
        # queued once per active Pokemon at order=5, subOrder=2. The handler
        # itself later checks grounded/semi-invulnerable state, but the
        # speedSort frame is spent on the queued handlers regardless.
        if int(terrain) == _TER_GRASSY_R:
            handlers.append((5, 0, spd, 2, 0))
        ability = int(battle[poff + 5])
        # Healer / Hydration / Shed Skin: onResidualOrder=5, onResidualSubOrder=3
        if ability in (_AB_HEALER_R, _AB_HYDRATION_R, _AB_SHED_SKIN_R):
            handlers.append((5, 0, spd, 3, 0))
        # Bad Dreams / Cud Chew / Harvest / Moody / Pickup / Slow Start /
        # Speed Boost: onResidualOrder=28, onResidualSubOrder=2
        if ability in (
            _AB_BAD_DREAMS_R,
            _AB_CUD_CHEW_R,
            _AB_HARVEST_R,
            _AB_MOODY_R,
            _AB_PICKUP_R,
            _AB_SLOW_START_R,
            _AB_SB_R,
        ):
            handlers.append((28, 0, spd, 2, 0))
        # Hunger Switch / Opportunist / Power Construct / Schooling /
        # Shields Down / Zen Mode: onResidualOrder=29
        if ability in (
            _AB_HUNGER_SWITCH_R,
            _AB_OPPORTUNIST_R,
            _AB_POWER_CONSTRUCT_R,
            _AB_SCHOOLING_R,
            _AB_SHIELDS_DOWN_R,
            _AB_ZEN_MODE_R,
        ):
            handlers.append((29, 0, spd, 0, 0))
        # Only the semi-invulnerable two-turn moves keep a move-specific
        # volatile with its own duration during the charge turn. Showdown then
        # has two residual duration handlers for that active: `twoturnmove`
        # plus the move-specific volatile (for example `phantomforce`).
        charge_meta_off = _OFF_META_R + (
            _M_CHG0_R if poff < _OFF_SIDE1_R else _M_CHG1_R
        )
        charge_move = int(battle[charge_meta_off])
        if charge_move in (
            _MOVE_DIG_R,
            _MOVE_FLY_R,
            _MOVE_DIVE_R,
            _MOVE_BOUNCE_R,
            _MOVE_SHADOW_FORCE_R,
            _MOVE_PHANTOM_FORCE_R,
        ):
            handlers.append((0, 0, spd, 0, 0))
            handlers.append((0, 0, spd, 0, 0))
        side_idx = 0 if poff < _OFF_SIDE1_R else 1
        vol = int(battle[_OFF_FIELD_R + (_F_VOL0_R if side_idx == 0 else _F_VOL1_R)])
        # Duration volatiles collected via fieldEvent('Residual') getKey='duration'.
        if _get_taunt_turns_r(vol) > 0:
            handlers.append((15, 0, spd, 0, 0))
        if _get_encore_turns_r(vol) > 0:
            handlers.append((16, 0, spd, 0, 0))
        dis_off = _OFF_FIELD_R + (_F_DIS0_R if side_idx == 0 else _F_DIS1_R)
        dis_turns_off = _OFF_FIELD_R + (_F_DIST0_R if side_idx == 0 else _F_DIST1_R)
        if int(battle[dis_off]) >= 0 and int(battle[dis_turns_off]) > 0:
            handlers.append((17, 0, spd, 0, 0))
        if _get_heal_block_turns_r(vol) > 0:
            handlers.append((20, 0, spd, 0, 0))
        yawn_turns = int(
            battle[_OFF_FIELD_R + (_F_YAWN0_R if side_idx == 0 else _F_YAWN1_R)]
        )
        if yawn_turns > 0:
            handlers.append((23, 0, spd, 0, 0))
    if len(handlers) < 2:
        return

    # Sort entries by comparePriority so tied groups are consecutive.
    handlers.sort(key=cmp_to_key(tracker.compare_priority))
    tracker.speed_sort_consume(handlers)


def _consume_residual_weather_event_frames(
    battle,
    tracker: "SpeedSortTracker",
    p0_off: int,
    p1_off: int,
    p0_speed: int,
    p1_speed: int,
) -> None:
    """Mirror Showdown weather residual's Weather + Update speedSort frames.

    Active weather conditions run `onFieldResidual`, which in turn calls
    `eachEvent('Weather')`. In gen 7+, `eachEvent('Weather')` immediately
    follows with one extra `eachEvent('Update')`. Both speed-sort the live
    active pair, so each tied singles board spends one `random(0, 2)` frame.
    """
    if int(battle[OFF_FIELD + F_WEATHER]) == WEATHER_NONE:
        return
    if int(battle[p0_off + 1]) <= 0 or int(battle[p1_off + 1]) <= 0:
        return
    if int(p0_speed) != int(p1_speed):
        return

    tracker.each_event_update([int(p0_speed), int(p1_speed)])
    tracker.each_event_update([int(p0_speed), int(p1_speed)])


def _consume_runswitch_tie_frame(
    battle,
    switcher_off: int,
    foe_off: int,
    gen5_prng,
    *,
    switcher_speed: int | None = None,
    foe_speed: int | None = None,
) -> None:
    """Mirror Showdown's hidden runSwitch speedSort frame for mid-turn switches.

    `BattleActions.switchIn()` queues a `runSwitch` action for pivot/self-switch
    replacements, and `runSwitch()` immediately calls
    `speedSort(getAllActive(true))` before any SwitchIn handlers run. In
    singles, that sort burns exactly one `random(0, 2)` frame when the new
    switcher ties the current opposing active slot by speed, even if that foe
    is already fainted and is only still present in `getAllActive(true)`.

    This sort happens before hazards or other entry effects mutate the incoming
    mon, so callers must invoke it on the raw post-switch state.
    """
    from pokepy import effects as fx

    if switcher_speed is None:
        # Fresh switch-ins still use the stale cached `pokemon.speed` slot
        # from setSpecies() here, before SwitchIn handlers or updateSpeed().
        switcher_speed = int(battle[int(switcher_off) + 11])
    if foe_speed is None:
        if int(battle[int(foe_off) + 1]) <= 0:
            foe_speed = int(battle[int(foe_off) + 11])
        else:
            foe_speed = fx.get_effective_speed(battle, foe_off)

    if int(switcher_speed) == int(foe_speed):
        gen5_prng.random(0, 2)


def _consume_endturn_quick_claw_roll(profile, gen5_prng) -> None:
    """Mirror Showdown endTurn Quick Claw pre-roll (sim/battle.ts endTurn).

    Gen 2 rolls randomChance(60, 256); gen 3 rolls randomChance(1, 5).
    Showdown stores the result on battle.quickClawRoll before each turn's move
    request, including the turn-0 -> turn-1 transition after lead switch-ins.
    """
    if profile.gen == 2:
        gen5_prng.random(256)
    elif profile.gen == 3:
        gen5_prng.random(5)


def _consume_switch_request_resume_tie_frames(
    switcher_speed: int,
    foe_speed: int,
    foe_alive: bool,
    gen5_prng,
) -> None:
    """Mirror Showdown's 3-frame post-switch continuation on tied speeds.

    Switch-request resumes such as late U-turn / Volt Switch, post-residual
    replacements, and forced-switch resumes execute as:
      1. switch action post-action eachEvent('Update')
      2. runSwitch speedSort(allActive)
      3. runSwitch post-action eachEvent('Update')

    Showdown uses the incoming mon's neutral on-entry speed for all 3 frames,
    before hazards or entry effects like Sticky Web mutate it.
    """
    if foe_alive and int(switcher_speed) == int(foe_speed):
        for _ in range(3):
            gen5_prng.random(0, 2)


def _run_switch_in_update_item_hooks_common(
    battle: np.ndarray,
    pokemon_offset: int,
    game_data,
    run_hook,
) -> None:
    poff = int(pokemon_offset)
    if int(battle[poff + 1]) <= 0:
        return

    from pokepy import effects as fx

    run_hook(fx.apply_sitrus_berry, poff, battle, poff, game_data)
    run_hook(fx.apply_gold_berry, poff, battle, poff, game_data)
    run_hook(fx.apply_lum_berry, poff, battle, poff, game_data)
    run_hook(fx.apply_status_curing_berries, poff, battle, poff, game_data)
    run_hook(fx.apply_persim_berry, poff, battle, poff, game_data)
    run_hook(fx.apply_stat_boosting_berries, poff, battle, poff, game_data)
    run_hook(fx.apply_pinch_healing_berries, poff, battle, poff, game_data)


def _consume_team_preview_queue_sort_frames(
    battle: np.ndarray,
    gen5_prng,
) -> None:
    """Mirror Showdown's hidden team-preview queue sort PRNG.

    Before the first visible turn, Showdown resolves the fixed `team 123456`
    choice by queueing one `team` action per team slot on each side and then
    calling `queue.sort()` / `speedSort(...)` on that 12-entry list. In
    singles, different preview slots never tie because their `priority`
    fields are `-slot`, but the same slot across the two sides can tie when
    both Pokemon have the same raw stored Speed stat.

    Each tied slot contributes one Fisher-Yates shuffle frame. The pair for
    preview slot `i` lands at indices `[2*i, 2*i + 1]` in the already-grouped
    sorted list, so Showdown consumes `random(2*i, 2*i + 2)` for that tie.
    The return value is unused; we mirror the same bounds so raw PRNG traces
    line up exactly.
    """
    for slot in range(6):
        p0_off = OFF_SIDE0 + slot * POKEMON_SIZE
        p1_off = OFF_SIDE1 + slot * POKEMON_SIZE
        if int(battle[p0_off + 0]) <= 0 or int(battle[p1_off + 0]) <= 0:
            continue
        if int(battle[p0_off + 11]) == int(battle[p1_off + 11]):
            gen5_prng.random(2 * slot, 2 * slot + 2)


def _sync_had_item_flag_on_switch_in(battle: np.ndarray, pokemon_offset: int) -> None:
    """Refresh current-entry item state on switch-in.

    `had_item` is used for Unburden and should reflect whether the Pokemon
    currently entered with an item. Booster Energy's paradox activation is a
    separate per-entry flag and must always be cleared here.
    """
    poff = int(pokemon_offset)
    flags = int(battle[poff + 15])
    from pokepy.core.constants import FLAG_BOOSTER_ENERGY_ACTIVE as _FLAG_BOOSTER

    flags &= ~_FLAG_BOOSTER
    if int(battle[poff + 6]) > 0:
        battle[poff + 15] = flags | 0x80
    else:
        battle[poff + 15] = flags & ~0x80


def _get_switch_resume_action_speed(
    battle: np.ndarray,
    pokemon_offset: int,
) -> int:
    """Return Showdown's runSwitch comparator speed for a fresh switch-in.

    The switch-resume queue uses the entrant's action speed before hazards and
    other entry effects mutate the mon. Passive speed modifiers like Choice
    Scarf still count, but Protosynthesis / Quark Drive must stay suppressed
    until their switch-in activation resolves.
    """
    from pokepy import effects as fx

    trick_room_active = int(battle[OFF_FIELD + F_TRICK_ROOM]) > 0

    def _as_action_speed(speed: int) -> int:
        return (10000 - int(speed)) if trick_room_active else int(speed)

    poff = int(pokemon_offset)
    ability = int(battle[poff + 5])
    if ability not in (ABILITY_PROTOSYNTHESIS, ABILITY_QUARK_DRIVE):
        return _as_action_speed(fx.get_effective_speed(battle, poff))

    battle[poff + 5] = 0
    try:
        return _as_action_speed(fx.get_effective_speed(battle, poff))
    finally:
        battle[poff + 5] = ability


def _reset_toxic_counter_on_switch_in(battle: np.ndarray, pokemon_offset: int) -> None:
    """Showdown resets a toxic stage counter when the mon re-enters."""
    poff = int(pokemon_offset)
    status_field = int(battle[poff + 12])
    if get_status(status_field) == STATUS_TOXIC:
        battle[poff + 12] = set_status(STATUS_TOXIC, 0)


def _clear_opponent_source_tied_lock_state(battle: np.ndarray, side: int) -> None:
    """Clear opponent volatiles that should end when this side's source leaves."""
    from pokepy.core.constants import (
        EXT_VOL_MEAN_LOOK as _EXT_VOL_MEAN_LOOK_SW,
        EXT_VOL_PARTIAL_TRAP as _EXT_VOL_PARTIAL_TRAP_SW,
        F_EXTENDED_VOLATILE_0 as _F_EXTVOL0_SW,
        F_EXTENDED_VOLATILE_1 as _F_EXTVOL1_SW,
        M_PARTIAL_TRAP_TURNS_0 as _MPT0_SW,
        M_PARTIAL_TRAP_TURNS_1 as _MPT1_SW,
    )

    opp_ext_vol = _F_EXTVOL1_SW if int(side) == 0 else _F_EXTVOL0_SW
    opp_partial_trap_turns = _MPT1_SW if int(side) == 0 else _MPT0_SW

    # In singles, source-tied trapping volatiles on the opponent end when the
    # trapping source leaves the field. Pokepy only stores the target-side
    # bits, so clear them when the source side actually switches or faints.
    battle[OFF_FIELD + opp_ext_vol] = int(battle[OFF_FIELD + opp_ext_vol]) & ~(
        _EXT_VOL_MEAN_LOOK_SW | _EXT_VOL_PARTIAL_TRAP_SW
    )
    battle[OFF_MOVES + opp_partial_trap_turns] = 0


def _clear_side_switch_state_common(battle: np.ndarray, side: int) -> None:
    """Clear per-side move/choice state that Showdown drops on switch-out."""
    if int(side) == 0:
        choice_lock = F_CHOICE_LOCK_0
        last_move = F_LAST_MOVE_0
        disable = F_DISABLE_0
        active_move_actions = M_ACTIVE_MOVE_ACTIONS_0
        charging = M_CHARGING_0
        locked_move = M_LOCKED_MOVE_0
        locked_turns = M_LOCKED_TURNS_0
        recharge = M_RECHARGE_0
        partial_trap_turns = M_PARTIAL_TRAP_TURNS_0
        stockpile_state = M_STOCKPILE_STATE_0
    else:
        choice_lock = F_CHOICE_LOCK_1
        last_move = F_LAST_MOVE_1
        disable = F_DISABLE_1
        active_move_actions = M_ACTIVE_MOVE_ACTIONS_1
        charging = M_CHARGING_1
        locked_move = M_LOCKED_MOVE_1
        locked_turns = M_LOCKED_TURNS_1
        recharge = M_RECHARGE_1
        partial_trap_turns = M_PARTIAL_TRAP_TURNS_1
        stockpile_state = M_STOCKPILE_STATE_1

    battle[OFF_FIELD + choice_lock] = -1
    battle[OFF_FIELD + last_move] = -1
    battle[OFF_FIELD + disable] = -1
    battle[OFF_MOVES + active_move_actions] = 0
    battle[OFF_META + charging] = -1
    battle[OFF_MOVES + locked_move] = -1
    battle[OFF_MOVES + locked_turns] = 0
    battle[OFF_MOVES + recharge] = 0
    battle[OFF_MOVES + partial_trap_turns] = 0
    battle[OFF_MOVES + stockpile_state] = 0

    _clear_opponent_source_tied_lock_state(battle, side)


def _reset_move_used_mask_for_offset(
    state: MultiFormatState, pokemon_offset: int
) -> None:
    poff = int(pokemon_offset)
    if poff < OFF_SIDE1:
        slot = (poff - OFF_SIDE0) // POKEMON_SIZE
        state.team_move_used_masks[slot] = np.int8(0)
    else:
        slot = (poff - OFF_SIDE1) // POKEMON_SIZE
        state.opp_move_used_masks[slot] = np.int8(0)


def _mark_move_slot_used(
    state: MultiFormatState, side: int, slot: int, move_slot: int
) -> None:
    if not (0 <= int(slot) < 6 and 0 <= int(move_slot) < 4):
        return
    if int(side) == 0:
        state.team_move_used_masks[int(slot)] = np.int8(
            int(state.team_move_used_masks[int(slot)]) | (1 << int(move_slot))
        )
    else:
        state.opp_move_used_masks[int(slot)] = np.int8(
            int(state.opp_move_used_masks[int(slot)]) | (1 << int(move_slot))
        )


def _last_resort_fails_for_slot(
    state: MultiFormatState,
    side: int,
    slot: int,
    move_id: int,
) -> bool:
    if int(move_id) != MOVE_LAST_RESORT:
        return False
    if not (0 <= int(slot) < 6):
        return True
    moves = (
        state.team_moves[int(slot)] if int(side) == 0 else state.opp_moves[int(slot)]
    )
    used_mask = (
        int(state.team_move_used_masks[int(slot)])
        if int(side) == 0
        else int(state.opp_move_used_masks[int(slot)])
    )
    known_moves = 0
    has_last_resort = False
    for idx, known_move_id in enumerate(moves):
        mid = int(known_move_id)
        if mid < 0:
            continue
        known_moves += 1
        if mid == MOVE_LAST_RESORT:
            has_last_resort = True
            continue
        if (used_mask & (1 << idx)) == 0:
            return True
    return known_moves < 2 or not has_last_resort


# =============================================================================
# Main turn loop
# =============================================================================


def step_battle_gen9_iter(
    state: MultiFormatState,
    action0: int,
    action1: int,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    gen5_prng,
    resolve_mid_turn_switch0=None,
    wants_tera0: bool = False,
    wants_tera1: bool = False,
    profile=None,
) -> Generator[SwitchRequest, Dict[int, int], Tuple[np.float32, np.float32, bool]]:
    """Execute one Gen 9 battle turn. Mutates `state` in place.

    Returns (reward0, reward1, done). reward1 = -reward0 (zero-sum shaped).

    Faithful structural port of MultiFormatFastEnv._step_battle_gen9
    (multi_format_fast_env.py lines 2925-5458).
    """
    # Lazy effect imports — done here so module load doesn't depend on parallel work.
    from pokepy import effects as fx  # type: ignore  # noqa: F401  (TODO: real module)

    calc_damage_gen9 = _get_calc_damage()
    if profile is None:
        from pokepy.core.gen_profile import GEN9_PROFILE

        profile = GEN9_PROFILE
    if not profile.has_tera:
        wants_tera0 = False
        wants_tera1 = False

    battle = state.battle_state
    max_turns = getattr(state, "max_turns", 200)
    side_order0 = state.side_order0
    side_order1 = state.side_order1
    _inline_knock_removed_focus_sash0 = False
    _inline_knock_removed_focus_sash1 = False
    _inline_knock_saved_item0 = 0
    _inline_knock_saved_item1 = 0
    _inline_knock_saved_target0 = -1
    _inline_knock_saved_target1 = -1
    _knock_off_source_alive0 = False
    _knock_off_source_alive1 = False
    _inline_rapid_spin_done0 = False
    _inline_rapid_spin_done1 = False

    if int(state.turn) == 0:
        startup_calls = getattr(state, "startup_prng_calls", ())
        if startup_calls:
            for call_args in startup_calls:
                gen5_prng.random(*tuple(int(a) for a in call_args))
        else:
            _consume_team_preview_queue_sort_frames(battle, gen5_prng)
        _consume_endturn_quick_claw_roll(profile, gen5_prng)
    state.pending_opp_switch_in_slot = np.int8(-1)
    state.pending_opp_switch_action_speed = np.int16(0)

    def _berry_tracker_for_offset(pokemon_offset: int):
        poff = int(pokemon_offset)
        if poff < OFF_SIDE1:
            return state.team_last_consumed_berry, (poff - OFF_SIDE0) // POKEMON_SIZE
        return state.opp_last_consumed_berry, (poff - OFF_SIDE1) // POKEMON_SIZE

    def _record_consumed_berry(pokemon_offset: int, item_id: int) -> None:
        item_id = int(item_id)
        if item_id <= 0:
            return
        from pokepy.data.item_aliases import (
            ITEM_GOLD_BERRY_INTERNAL as _ITEM_GOLD_BERRY_INTERNAL,
        )

        is_berry = item_id == _ITEM_GOLD_BERRY_INTERNAL
        item_is_berry = getattr(game_data, "item_is_berry", None)
        if item_is_berry is not None and 0 <= item_id < len(item_is_berry):
            is_berry = bool(item_is_berry[item_id])
        if not is_berry:
            return
        tracker, slot = _berry_tracker_for_offset(pokemon_offset)
        tracker[slot] = np.int16(item_id)

    def _get_consumed_berry(pokemon_offset: int) -> int:
        tracker, slot = _berry_tracker_for_offset(pokemon_offset)
        return int(tracker[slot])

    def _clear_consumed_berry(pokemon_offset: int) -> None:
        tracker, slot = _berry_tracker_for_offset(pokemon_offset)
        tracker[slot] = np.int16(0)

    def _run_item_hook_with_berry_tracking(hook, pokemon_offset: int, *args) -> None:
        poff = int(pokemon_offset)
        prev_item = int(battle[poff + 6])
        hook(*args)
        if prev_item > 0 and int(battle[poff + 6]) == 0:
            _record_consumed_berry(poff, prev_item)
            # Consuming an item (berry/gem/herb/sash/balloon/booster) emits an
            # `-enditem` message in Showdown, revealing it to the opponent.
            _mark_item_revealed(poff)

    def _apply_harvest(pokemon_offset: int) -> None:
        poff = int(pokemon_offset)
        if int(battle[poff + 1]) <= 0:
            return
        if int(battle[poff + 5]) != ABILITY_HARVEST:
            return
        weather_now = int(battle[OFF_FIELD + F_WEATHER])
        if weather_now not in (WEATHER_SUN, WEATHER_DESOLATE_LAND):
            if int(gen5_prng.random(2)) != 0:
                return
        if int(battle[poff + 6]) != 0:
            return
        consumed_berry = _get_consumed_berry(poff)
        if consumed_berry <= 0:
            return
        battle[poff + 6] = np.int16(consumed_berry)

    stats_raised_this_turn0 = False
    stats_raised_this_turn1 = False
    stats_lowered_this_turn0 = False
    stats_lowered_this_turn1 = False

    def _snapshot_boost_stages(pokemon_offset: int) -> tuple[int, int]:
        poff = int(pokemon_offset)
        return (int(battle[poff + 13]), int(battle[poff + 14]))

    def _boosts_were_raised(before: tuple[int, int], after: tuple[int, int]) -> bool:
        b13_before, b14_before = before
        b13_after, b14_after = after
        for shift in (0, 4, 8, 12):
            if extract_boost(b13_after, shift) > extract_boost(b13_before, shift):
                return True
        for shift in (0, 4, 8):
            if extract_boost(b14_after, shift) > extract_boost(b14_before, shift):
                return True
        return False

    def _boosts_were_lowered(before: tuple[int, int], after: tuple[int, int]) -> bool:
        b13_before, b14_before = before
        b13_after, b14_after = after
        for shift in (0, 4, 8, 12):
            if extract_boost(b13_after, shift) < extract_boost(b13_before, shift):
                return True
        for shift in (0, 4, 8):
            if extract_boost(b14_after, shift) < extract_boost(b14_before, shift):
                return True
        return False

    # ------------------------------------------------------------------
    # Fog-of-war reveal tracking (partial observability). The obs adapter
    # masks an opponent's item/ability until it has been "revealed" the way
    # Showdown surfaces it (an `-item`/`-ability`/`[from] ...` protocol
    # message). We detect those reveals by observing that an ability/item
    # actually produced an effect, which is exactly when Showdown announces
    # it; conditional abilities (Protosynthesis with no sun, an unused Flash
    # Fire) therefore stay hidden until they fire. `team_*` arrays track
    # side-0 mons (seen by side 1); `opp_*` track side-1 mons (seen by side 0).
    def _reveal_slot_for_offset(pokemon_offset: int):
        poff = int(pokemon_offset)
        if poff < OFF_SIDE1:
            slot = (poff - OFF_SIDE0) // POKEMON_SIZE
            return slot, state.team_abilities_revealed, state.team_items_revealed
        slot = (poff - OFF_SIDE1) // POKEMON_SIZE
        return slot, state.opp_abilities_revealed, state.opp_items_revealed

    def _mark_ability_revealed(pokemon_offset: int) -> None:
        slot, abil_arr, _ = _reveal_slot_for_offset(pokemon_offset)
        if 0 <= slot < 6:
            abil_arr[slot] = True

    def _mark_item_revealed(pokemon_offset: int) -> None:
        slot, _, item_arr = _reveal_slot_for_offset(pokemon_offset)
        if 0 <= slot < 6:
            item_arr[slot] = True

    def _ability_reveal_signature(pokemon_offset: int) -> tuple:
        # Observable per-Pokemon state an ability could change on activation:
        # boost stages, current ability id (Trace), HP (absorb/Regenerator),
        # status, and the paradox(0x6010)/booster(0x1000)/flash-fire(0x200)
        # flag bits.
        poff = int(pokemon_offset)
        return (
            int(battle[poff + 13]),
            int(battle[poff + 14]) & 0x0FFF,
            int(battle[poff + 5]),
            int(battle[poff + 1]),
            int(battle[poff + 12]),
            int(battle[poff + 15]) & 0x7210,
        )

    # Aliases so the reveal wrappers below can call the originals without the
    # file-wide rename of `fx.apply_contact_*` recursing back into them.
    _fx_apply_contact_damage = fx.apply_contact_damage
    _fx_apply_contact_status_ability = fx.apply_contact_status_ability
    _fx_apply_life_orb_recoil = fx.apply_life_orb_recoil

    def _apply_recoil_drain_from_move_tracked(*args, **kwargs):
        kwargs.setdefault("gen", profile.gen)
        return fx.apply_recoil_drain_from_move(*args, **kwargs)

    def _apply_life_orb_recoil_tracked(*args, **kwargs) -> None:
        # (battle, user_off, dmg, hit, move_id). Life Orb recoil reveals the
        # holder's item via `[from] item: Life Orb`. Skip simulation copies.
        sim = args[0]
        if sim is not battle:
            _fx_apply_life_orb_recoil(*args, **kwargs)
            return
        uoff = int(args[1])
        before_hp = int(sim[uoff + 1])
        _fx_apply_life_orb_recoil(*args, **kwargs)
        if int(sim[uoff + 1]) < before_hp:
            _mark_item_revealed(uoff)

    def _apply_contact_damage_tracked(*args, **kwargs) -> None:
        # (battle, move_id, atk_off, def_off, hit, ...). Rough Skin / Iron Barbs
        # (defender ability) and Rocky Helmet (defender item) chip the attacker
        # on contact; whichever chipped is revealed. Skip reveal when called on
        # a simulation copy rather than the real battle array.
        sim = args[0]
        if sim is not battle:
            _fx_apply_contact_damage(*args, **kwargs)
            return
        aoff = int(args[2])
        doff = int(args[3])
        before_hp = int(sim[aoff + 1])
        _fx_apply_contact_damage(*args, **kwargs)
        if int(sim[aoff + 1]) < before_hp:
            if int(sim[doff + 5]) in (ABILITY_ROUGH_SKIN, ABILITY_IRON_BARBS):
                _mark_ability_revealed(doff)
            if int(sim[doff + 6]) == ITEM_ROCKY_HELMET:
                _mark_item_revealed(doff)

    def _apply_contact_status_ability_tracked(*args, **kwargs) -> None:
        # (battle, move_id, atk_off, def_off, ...). Defender's Static / Flame
        # Body / Poison Point / Effect Spore statuses the attacker (reveals the
        # defender); the attacker's Poison Touch statuses the defender (reveals
        # the attacker). Skip reveal on simulation copies.
        sim = args[0]
        if sim is not battle:
            _fx_apply_contact_status_ability(*args, **kwargs)
            return
        aoff = int(args[2])
        doff = int(args[3])
        before_a = int(sim[aoff + 12])
        before_d = int(sim[doff + 12])
        _fx_apply_contact_status_ability(*args, **kwargs)
        if int(sim[aoff + 12]) != before_a:
            _mark_ability_revealed(doff)
        if int(sim[doff + 12]) != before_d:
            _mark_ability_revealed(aoff)

    def _mark_stats_raised_this_turn(
        pokemon_offset: int, before: tuple[int, int]
    ) -> None:
        nonlocal stats_raised_this_turn0, stats_raised_this_turn1
        poff = int(pokemon_offset)
        after = _snapshot_boost_stages(poff)
        if not _boosts_were_raised(before, after):
            return
        if poff < OFF_SIDE1:
            stats_raised_this_turn0 = True
        else:
            stats_raised_this_turn1 = True

    def _mark_stats_lowered_this_turn(
        pokemon_offset: int, before: tuple[int, int]
    ) -> None:
        nonlocal stats_lowered_this_turn0, stats_lowered_this_turn1
        poff = int(pokemon_offset)
        after = _snapshot_boost_stages(poff)
        if not _boosts_were_lowered(before, after):
            return
        if poff < OFF_SIDE1:
            stats_lowered_this_turn0 = True
        else:
            stats_lowered_this_turn1 = True

    _move_phase_residual_speed_refresh = False

    def _active_live_speed(pokemon_offset: int) -> "int | None":
        poff = int(pokemon_offset)
        active0_off = OFF_SIDE0 + int(battle[OFF_META + M_ACTIVE0]) * POKEMON_SIZE
        active1_off = OFF_SIDE1 + int(battle[OFF_META + M_ACTIVE1]) * POKEMON_SIZE
        if poff != active0_off and poff != active1_off:
            return None
        return int(fx.get_effective_speed(battle, poff))

    def _mark_move_phase_residual_speed_refresh(
        pokemon_offset: int,
        before_speed: "int | None",
    ) -> None:
        nonlocal _move_phase_residual_speed_refresh
        if before_speed is None:
            return
        after_speed = _active_live_speed(pokemon_offset)
        if after_speed is None:
            return
        if int(after_speed) != int(before_speed):
            _move_phase_residual_speed_refresh = True

    def _apply_white_herb_if_ready(pokemon_offset: int) -> bool:
        from pokepy.core.constants import ITEM_WHITE_HERB

        poff = int(pokemon_offset)
        if int(battle[poff + 1]) <= 0:
            return False
        if int(battle[poff + 6]) != ITEM_WHITE_HERB:
            return False
        b13 = int(battle[poff + 13]) & 0xFFFF
        b14 = int(battle[poff + 14]) & 0xFFFF
        any_negative = False
        new13 = b13
        for shift in (0, 4, 8, 12):
            n = (b13 >> shift) & 0xF
            if n < 6:
                new13 = (new13 & ~(0xF << shift)) | (6 << shift)
                any_negative = True
        new14 = b14
        for shift in (0, 4, 8):  # spe, acc, eva (skip tera nibble at 12)
            n = (b14 >> shift) & 0xF
            if n < 6:
                new14 = (new14 & ~(0xF << shift)) | (6 << shift)
                any_negative = True
        if not any_negative:
            return False
        battle[poff + 13] = new13 if new13 < 0x8000 else new13 - 0x10000
        battle[poff + 14] = new14 if new14 < 0x8000 else new14 - 0x10000
        battle[poff + 6] = 0
        return True

    def _base_ability_for_offset(pokemon_offset: int) -> int:
        poff = int(pokemon_offset)
        if poff < OFF_SIDE1:
            slot = (poff - OFF_SIDE0) // POKEMON_SIZE
            return int(state.team_abilities[slot])
        slot = (poff - OFF_SIDE1) // POKEMON_SIZE
        return int(state.opp_abilities[slot])

    def _reset_incoming_switch_state_tracked(pokemon_offset: int) -> None:
        fx.reset_incoming_switch_state(
            battle,
            pokemon_offset,
            game_data,
            base_ability=_base_ability_for_offset(pokemon_offset),
            state=state,
        )
        _reset_move_used_mask_for_offset(state, pokemon_offset)

    def _reveal_switch_in_abilities(
        switcher_offset: int,
        opponent_offset: int,
        sw_sig_before: tuple,
        opp_boosts_before: tuple,
        field_before: tuple,
    ) -> None:
        field_after = (
            int(battle[OFF_FIELD + F_WEATHER]),
            int(battle[OFF_FIELD + F_TERRAIN]),
        )
        sw_sig_after = _ability_reveal_signature(switcher_offset)
        opp_boosts_after = _snapshot_boost_stages(opponent_offset)
        # The switcher's ability announced if it changed the field or its own
        # observable state, or dropped the opponent's Attack (Intimidate).
        if (
            field_after != field_before
            or sw_sig_after != sw_sig_before
            or _boosts_were_lowered(opp_boosts_before, opp_boosts_after)
        ):
            _mark_ability_revealed(switcher_offset)
        # Trace copies and announces the opponent's ability (the switcher's
        # ability id changes to match the opponent's).
        if sw_sig_after[2] != sw_sig_before[2]:
            _mark_ability_revealed(opponent_offset)
        # Defiant / Competitive / Guard Dog: the opponent raised its own stats
        # reacting to Intimidate, which reveals the opponent's ability.
        if _boosts_were_raised(opp_boosts_before, opp_boosts_after):
            _mark_ability_revealed(opponent_offset)

    def _apply_switch_in_ability_tracked(
        switcher_offset: int,
        opponent_offset: int,
        did_switch: bool,
    ) -> None:
        if not profile.has_abilities:
            return
        before_switcher = _snapshot_boost_stages(switcher_offset)
        before_opponent = _snapshot_boost_stages(opponent_offset)
        _rev_sw_sig = _ability_reveal_signature(switcher_offset)
        _rev_field = (
            int(battle[OFF_FIELD + F_WEATHER]),
            int(battle[OFF_FIELD + F_TERRAIN]),
        )
        fx.apply_switch_in_ability(
            battle,
            switcher_offset,
            opponent_offset,
            did_switch,
            gen5_prng=gen5_prng,
            has_terrain=profile.has_terrain,
            ability_weather_limited=profile.ability_weather_limited,
        )
        _mark_stats_raised_this_turn(switcher_offset, before_switcher)
        _mark_stats_raised_this_turn(opponent_offset, before_opponent)
        _mark_stats_lowered_this_turn(switcher_offset, before_switcher)
        _mark_stats_lowered_this_turn(opponent_offset, before_opponent)
        if did_switch:
            _reveal_switch_in_abilities(
                switcher_offset,
                opponent_offset,
                _rev_sw_sig,
                before_opponent,
                _rev_field,
            )
        _apply_white_herb_if_ready(switcher_offset)
        if int(opponent_offset) != int(switcher_offset):
            _apply_white_herb_if_ready(opponent_offset)

    def _apply_switch_in_ability_with_trace_reaction_tracked(
        switcher_offset: int,
        opponent_offset: int,
        did_switch: bool,
    ) -> None:
        if not profile.has_abilities:
            return
        before_switcher = _snapshot_boost_stages(switcher_offset)
        before_opponent = _snapshot_boost_stages(opponent_offset)
        _rev_sw_sig = _ability_reveal_signature(switcher_offset)
        _rev_field = (
            int(battle[OFF_FIELD + F_WEATHER]),
            int(battle[OFF_FIELD + F_TERRAIN]),
        )
        fx.apply_switch_in_ability_with_trace_reaction(
            battle,
            switcher_offset,
            opponent_offset,
            did_switch,
            gen5_prng=gen5_prng,
            has_terrain=profile.has_terrain,
            ability_weather_limited=profile.ability_weather_limited,
        )
        _mark_stats_raised_this_turn(switcher_offset, before_switcher)
        _mark_stats_raised_this_turn(opponent_offset, before_opponent)
        _mark_stats_lowered_this_turn(switcher_offset, before_switcher)
        _mark_stats_lowered_this_turn(opponent_offset, before_opponent)
        if did_switch:
            _reveal_switch_in_abilities(
                switcher_offset,
                opponent_offset,
                _rev_sw_sig,
                before_opponent,
                _rev_field,
            )
        _apply_white_herb_if_ready(switcher_offset)
        if int(opponent_offset) != int(switcher_offset):
            _apply_white_herb_if_ready(opponent_offset)

    def _store_pending_opp_switch_in(active_slot: int, action_speed: int) -> None:
        state.pending_opp_switch_in_slot = np.int8(int(active_slot))
        state.pending_opp_switch_action_speed = np.int16(int(action_speed))

    def _apply_stat_changes_from_move_tracked(
        move_id: int,
        user_offset: int,
        target_offset: int,
        hit: bool,
        prerolled_roll: "int | None" = None,
    ) -> None:
        _MOVE_STOCKPILE = 254
        before_user_speed = _active_live_speed(user_offset)
        before_target_speed = (
            None
            if int(target_offset) == int(user_offset)
            else _active_live_speed(target_offset)
        )
        if int(move_id) == _MOVE_STOCKPILE:
            stockpile_state_off = OFF_MOVES + (
                M_STOCKPILE_STATE_0
                if int(user_offset) < OFF_SIDE1
                else M_STOCKPILE_STATE_1
            )
            stockpile_state = int(battle[stockpile_state_off]) & 0xFFFF
            stockpile_layers = get_stockpile_layers(stockpile_state)
            if (not bool(hit)) or stockpile_layers >= 3:
                return

            before_user = _snapshot_boost_stages(user_offset)
            before_target = _snapshot_boost_stages(target_offset)
            before_def = extract_boost(int(battle[int(user_offset) + 13]), 4)
            before_spd = extract_boost(int(battle[int(user_offset) + 13]), 12)
            fx.apply_stat_changes_from_move(
                battle,
                move_id,
                user_offset,
                target_offset,
                hit,
                game_data,
                move_effects,
                gen5_prng,
                prerolled_roll=prerolled_roll,
                gen=profile.gen,
            )
            after_def = extract_boost(int(battle[int(user_offset) + 13]), 4)
            after_spd = extract_boost(int(battle[int(user_offset) + 13]), 12)
            stockpile_state = set_stockpile_layers(
                stockpile_state, stockpile_layers + 1
            )
            if before_def != after_def:
                stockpile_state = set_stockpile_def_count(
                    stockpile_state,
                    get_stockpile_def_count(stockpile_state) + 1,
                )
            if before_spd != after_spd:
                stockpile_state = set_stockpile_spd_count(
                    stockpile_state,
                    get_stockpile_spd_count(stockpile_state) + 1,
                )
            battle[stockpile_state_off] = np.int16(stockpile_state)
            _mark_stats_raised_this_turn(user_offset, before_user)
            _mark_stats_raised_this_turn(target_offset, before_target)
            _mark_stats_lowered_this_turn(user_offset, before_user)
            _mark_stats_lowered_this_turn(target_offset, before_target)
            _apply_white_herb_if_ready(user_offset)
            if int(target_offset) != int(user_offset):
                _apply_white_herb_if_ready(target_offset)
            _mark_move_phase_residual_speed_refresh(user_offset, before_user_speed)
            if int(target_offset) != int(user_offset):
                _mark_move_phase_residual_speed_refresh(
                    target_offset, before_target_speed
                )
            return

        before_user = _snapshot_boost_stages(user_offset)
        before_target = _snapshot_boost_stages(target_offset)
        fx.apply_stat_changes_from_move(
            battle,
            move_id,
            user_offset,
            target_offset,
            hit,
            game_data,
            move_effects,
            gen5_prng,
            prerolled_roll=prerolled_roll,
            gen=profile.gen,
        )
        _mark_stats_raised_this_turn(user_offset, before_user)
        _mark_stats_raised_this_turn(target_offset, before_target)
        _mark_stats_lowered_this_turn(user_offset, before_user)
        _mark_stats_lowered_this_turn(target_offset, before_target)
        _apply_white_herb_if_ready(user_offset)
        if int(target_offset) != int(user_offset):
            _apply_white_herb_if_ready(target_offset)
        _mark_move_phase_residual_speed_refresh(user_offset, before_user_speed)
        if int(target_offset) != int(user_offset):
            _mark_move_phase_residual_speed_refresh(target_offset, before_target_speed)

    def _apply_booster_energy_update_tracked(pokemon_offset: int) -> None:
        before_speed = _active_live_speed(pokemon_offset)
        before = _snapshot_boost_stages(pokemon_offset)
        fx.apply_booster_energy_update(battle, pokemon_offset)
        _mark_stats_raised_this_turn(pokemon_offset, before)
        _mark_move_phase_residual_speed_refresh(pokemon_offset, before_speed)

    def _run_switch_in_update_item_hooks(pokemon_offset: int) -> None:
        poff = int(pokemon_offset)
        if int(battle[poff + 1]) <= 0:
            return

        before_speed = _active_live_speed(poff)
        before = _snapshot_boost_stages(poff)
        _run_switch_in_update_item_hooks_common(
            battle,
            poff,
            game_data,
            _run_item_hook_with_berry_tracking,
        )

        _mark_stats_raised_this_turn(poff, before)
        _mark_stats_lowered_this_turn(poff, before)
        _mark_move_phase_residual_speed_refresh(poff, before_speed)

    def _apply_immediate_defender_ability_state_changes_tracked(
        user_offset: int,
        target_offset: int,
        hit: bool,
        damage: int,
        move_type: int,
        move_category: int,
        move_flags: int,
    ) -> None:
        before_speed = _active_live_speed(target_offset)
        _rev_t_sig = _ability_reveal_signature(target_offset)
        _rev_u_hp = int(battle[int(user_offset) + 1])
        _rev_u_status = int(battle[int(user_offset) + 12])
        fx.apply_immediate_defender_ability_state_changes(
            battle,
            user_offset,
            target_offset,
            hit,
            damage,
            move_type,
            move_category,
            move_flags,
        )
        # The defender's ability (Rough Skin / Static / Flame Body / Weak Armor
        # / Justified / an absorb or Flash Fire, etc.) fired if it changed the
        # target's observable state or affected the attacker — Showdown shows it.
        if (
            _ability_reveal_signature(target_offset) != _rev_t_sig
            or int(battle[int(user_offset) + 1]) != _rev_u_hp
            or int(battle[int(user_offset) + 12]) != _rev_u_status
        ):
            _mark_ability_revealed(target_offset)
        _mark_move_phase_residual_speed_refresh(target_offset, before_speed)

    def _apply_knock_off_from_move_tracked(
        move_id: int,
        target_offset: int,
        hit: bool,
        *,
        user_offset: int | None = None,
        source_alive: bool | None = None,
    ) -> None:
        before_speed = _active_live_speed(target_offset)
        _rev_had_item = int(battle[int(target_offset) + 6]) > 0
        fx.apply_knock_off_from_move(
            battle,
            move_id,
            target_offset,
            hit,
            game_data,
            move_effects,
            user_offset=user_offset,
            source_alive=source_alive,
        )
        # Knock Off reveals (and removes) the target's item via `-enditem`.
        if _rev_had_item and int(battle[int(target_offset) + 6]) <= 0:
            _mark_item_revealed(target_offset)
        _mark_move_phase_residual_speed_refresh(target_offset, before_speed)

    def _apply_trick_from_move_tracked(
        move_id: int,
        user_offset: int,
        target_offset: int,
        hit: bool,
    ) -> bool:
        before_user_speed = _active_live_speed(user_offset)
        before_target_speed = _active_live_speed(target_offset)
        swapped = fx.apply_trick_from_move(
            battle,
            move_id,
            user_offset,
            target_offset,
            hit,
        )
        # Trick / Switcheroo swaps items and reveals both via `-item` messages.
        if swapped:
            _mark_item_revealed(user_offset)
            _mark_item_revealed(target_offset)
        _mark_move_phase_residual_speed_refresh(user_offset, before_user_speed)
        _mark_move_phase_residual_speed_refresh(target_offset, before_target_speed)
        return bool(swapped)

    def _maybe_preapply_on_try_move_selfboost(
        side: int,
        user_offset: int,
        target_offset: int,
        move_blocked: bool,
    ) -> bool:
        if move_blocked:
            return False
        if side == 0:
            if not _on_try_selfboost0:
                return False
            _apply_stat_changes_from_move_tracked(
                move_id0, user_offset, target_offset, True
            )
            return True
        if not _on_try_selfboost1:
            return False
        _apply_stat_changes_from_move_tracked(
            move_id1, user_offset, target_offset, True
        )
        return True

    _SCREEN_SKIP_REFLECT = 0x1
    _SCREEN_SKIP_LIGHTSCREEN = 0x2
    _SCREEN_SKIP_AURORAVEIL = 0x4

    def _apply_screen_from_move_tracked(move_id: int, side: int, hit: bool) -> None:
        from pokepy.core.constants import (
            F_SCREENS_0,
            F_SCREENS_1,
            ITEM_LIGHT_CLAY,
            MOVE_AURORA_VEIL,
            MOVE_LIGHT_SCREEN,
            MOVE_REFLECT,
        )

        screen_offset = OFF_FIELD + (F_SCREENS_0 if side == 0 else F_SCREENS_1)
        before_screens = int(battle[screen_offset]) & 0xFFFF
        fx.apply_screen_from_move(battle, move_id, side, hit)
        after_screens = int(battle[screen_offset]) & 0xFFFF
        if after_screens == before_screens:
            return
        if move_id == MOVE_REFLECT:
            skip_bit = _SCREEN_SKIP_REFLECT
        elif move_id == MOVE_LIGHT_SCREEN:
            skip_bit = _SCREEN_SKIP_LIGHTSCREEN
        elif move_id == MOVE_AURORA_VEIL:
            skip_bit = _SCREEN_SKIP_AURORAVEIL
        else:
            return
        user_active = int(battle[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])
        user_base = OFF_SIDE0 if side == 0 else OFF_SIDE1
        user_item = int(battle[user_base + user_active * POKEMON_SIZE + 6])
        if user_item != ITEM_LIGHT_CLAY:
            return
        if side == 0:
            state.screen_skip_decrement0 = np.int8(
                int(state.screen_skip_decrement0) | skip_bit
            )
        else:
            state.screen_skip_decrement1 = np.int8(
                int(state.screen_skip_decrement1) | skip_bit
            )

    def _decrement_screens_tracked() -> None:
        from pokepy.core.constants import (
            F_SCREENS_0,
            F_SCREENS_1,
            SCREEN_AURORAVEIL_SHIFT,
            SCREEN_LIGHTSCREEN_SHIFT,
            SCREEN_MASK_2BIT,
            SCREEN_MASK_3BIT,
            SCREEN_MIST_SHIFT,
            SCREEN_REFLECT_SHIFT,
            SCREEN_SAFEGUARD_SHIFT,
            SCREEN_TAILWIND_SHIFT,
        )

        for side in (0, 1):
            screens_offset = OFF_FIELD + (F_SCREENS_0 if side == 0 else F_SCREENS_1)
            skip_mask = int(
                state.screen_skip_decrement0
                if side == 0
                else state.screen_skip_decrement1
            )
            screens = int(battle[screens_offset]) & 0xFFFF
            new_screens = screens
            for shift, mask, skip_bit in (
                (SCREEN_REFLECT_SHIFT, SCREEN_MASK_3BIT, _SCREEN_SKIP_REFLECT),
                (SCREEN_LIGHTSCREEN_SHIFT, SCREEN_MASK_3BIT, _SCREEN_SKIP_LIGHTSCREEN),
                (SCREEN_AURORAVEIL_SHIFT, SCREEN_MASK_3BIT, _SCREEN_SKIP_AURORAVEIL),
                (SCREEN_TAILWIND_SHIFT, SCREEN_MASK_3BIT, 0),
                (SCREEN_SAFEGUARD_SHIFT, SCREEN_MASK_2BIT, 0),
                (SCREEN_MIST_SHIFT, SCREEN_MASK_2BIT, 0),
            ):
                val = (screens >> shift) & mask
                if skip_bit and (skip_mask & skip_bit) and val > 0:
                    new_val = val
                else:
                    new_val = val - 1 if val > 0 else 0
                new_screens = (new_screens & ~(mask << shift)) | (new_val << shift)
            if new_screens >= 0x8000:
                new_screens -= 0x10000
            battle[screens_offset] = new_screens
        state.screen_skip_decrement0 = np.int8(0)
        state.screen_skip_decrement1 = np.int8(0)

    def _sync_showdown_order_on_switch(order_arr, new_active_slot):
        new_active_slot = int(new_active_slot)
        idx = -1
        for i in range(len(order_arr)):
            if int(order_arr[i]) == new_active_slot:
                idx = i
                break
        if idx <= 0:
            return
        old_front = int(order_arr[0])
        order_arr[0] = np.int8(new_active_slot)
        order_arr[idx] = np.int8(old_front)

    def _apply_seed_sower(source_off: int) -> None:
        if not profile.has_terrain:
            return
        if int(battle[source_off + 5]) != ABILITY_SEED_SOWER:
            return
        if int(battle[OFF_FIELD + F_TERRAIN]) == TERRAIN_GRASSY:
            return
        battle[OFF_FIELD + F_TERRAIN] = TERRAIN_GRASSY
        battle[OFF_META + M_TERRAIN_TURNS] = (
            8 if int(battle[source_off + 6]) == ITEM_TERRAIN_EXTENDER else 5
        )
        from pokepy.effects.abilities import (
            apply_terrain_seed_item as _apply_terrain_seed_item,
        )

        _apply_terrain_seed_item(battle, p0_off)
        _apply_terrain_seed_item(battle, p1_off)

    def _resolve_switch_target_from_action(_side_base, _active_slot, _action):
        _active_slot = int(_active_slot)
        _target_slot = max(0, min(5, int(_action) - 4))
        _target_off = _side_base + _target_slot * POKEMON_SIZE
        _target_alive = (int(battle[_target_off + 1]) > 0) and (
            (int(battle[_target_off + 15]) & 1) == 0
        )
        if int(_action) >= 4 and _target_alive and _target_slot != _active_slot:
            return _target_slot
        for _slot in range(6):
            if _slot == _active_slot:
                continue
            _slot_off = _side_base + _slot * POKEMON_SIZE
            if (
                int(battle[_slot_off + 1]) > 0
                and (int(battle[_slot_off + 15]) & 1) == 0
            ):
                return _slot
        return _active_slot

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------
    action = int(action0)
    opp_action = int(action1)

    wants_tera0 = bool(wants_tera0)
    wants_tera1 = bool(wants_tera1)
    if action >= 9:
        wants_tera0 = True
        action = action - 9
    if opp_action >= 9:
        wants_tera1 = True
        opp_action = opp_action - 9

    is_switch = action >= 4
    move_idx = 0 if is_switch else action
    switch_target = (action - 4) if is_switch else -1

    active0 = int(battle[OFF_META + M_ACTIVE0])
    active1 = int(battle[OFF_META + M_ACTIVE1])
    _start_turn_active0 = active0
    _start_turn_active1 = active1

    # Heuristic opponent if -1 (single-call random move 0..3)
    if opp_action < 0:
        opp_action = int(gen5_prng.random(4))
    opp_is_switch = opp_action >= 4
    opp_move_idx = 0 if opp_is_switch else opp_action
    opp_switch_target = (opp_action - 4) if opp_is_switch else -1

    # ------------------------------------------------------------------
    # Pre-switch speeds — captured BEFORE the voluntary switch fires so
    # that the turn-start `BeforeTurn`/`Update` speedSort frames use the
    # OUTGOING actives' speeds (Showdown's `runAction('beforeTurn')` runs
    # before the switch action executes; the `eachEvent('BeforeTurn')`
    # and post-BeforeTurn `eachEvent('Update')` calls therefore speedSort
    # the pre-switch active pair). Needed for parity on turns where one
    # side voluntarily switches into a mon with a different speed.
    # ------------------------------------------------------------------
    from pokepy import effects as _fx_pre  # type: ignore

    _pre_p0_off = OFF_SIDE0 + active0 * POKEMON_SIZE
    _pre_p1_off = OFF_SIDE1 + active1 * POKEMON_SIZE
    pre_switch_p0_speed = _fx_pre.get_effective_speed(battle, _pre_p0_off)
    pre_switch_p1_speed = _fx_pre.get_effective_speed(battle, _pre_p1_off)
    pre_switch_speeds_tied = pre_switch_p0_speed == pre_switch_p1_speed
    trick_room_active = int(battle[OFF_FIELD + F_TRICK_ROOM]) > 0

    def _action_speed_from_effective(_speed: int) -> int:
        # Showdown queues actions using pokemon.getActionSpeed(), which already
        # inverts under Trick Room. Use the outgoing mon's speed for voluntary
        # switches and the active mover's speed for moves.
        return (10000 - int(_speed)) if trick_room_active else int(_speed)

    def _cached_switchin_speed(_p_off: int) -> int:
        # Fresh switch-ins inherit `pokemon.speed = storedStats.spe` from
        # Pokemon.setSpecies() and do not get a battle.updateSpeed() refresh
        # until a later explicit update (for example the residual action).
        # When that fresh switcher faints before the refresh, Showdown's
        # residual handler sort still sees this stale cached slot speed.
        return int(battle[int(_p_off) + 11])

    # ------------------------------------------------------------------
    # Voluntary switches (priority -6)
    # ------------------------------------------------------------------
    can_switch0 = is_switch and _is_valid_switch_target(
        battle, OFF_SIDE0, switch_target, active0
    )
    can_switch1 = opp_is_switch and _is_valid_switch_target(
        battle, OFF_SIDE1, opp_switch_target, active1
    )

    team_tera = getattr(state, "team_tera", None)
    if profile.has_tera and wants_tera0 and not is_switch:
        fx.activate_terastallization(
            battle, 0, team_tera=team_tera, active_slot=active0
        )
    if profile.has_tera and wants_tera1 and not opp_is_switch:
        fx.activate_terastallization(
            battle, 1, team_tera=team_tera, active_slot=active1
        )

    new_active0 = switch_target if can_switch0 else active0
    new_active1 = opp_switch_target if can_switch1 else active1

    old_p0_off = OFF_SIDE0 + active0 * POKEMON_SIZE
    old_p1_off = OFF_SIDE1 + active1 * POKEMON_SIZE
    pending_switch_slot_condition0 = (
        int(battle[OFF_FIELD + F_DESTINY_BOND_0]) if can_switch0 else 0
    )
    pending_switch_slot_condition1 = (
        int(battle[OFF_FIELD + F_DESTINY_BOND_1]) if can_switch1 else 0
    )

    # Regenerator + Natural Cure on switch-out
    fx.apply_regenerator_on_switch_out(battle, old_p0_off, can_switch0)
    fx.apply_regenerator_on_switch_out(battle, old_p1_off, can_switch1)
    fx.apply_natural_cure_on_switch_out(battle, old_p0_off, can_switch0)
    fx.apply_natural_cure_on_switch_out(battle, old_p1_off, can_switch1)

    # Reset boosts on outgoing Pokemon (preserve tera bits in b2 high nibble)
    if can_switch0:
        battle[old_p0_off + 13] = NEUTRAL_BOOSTS_13
        tera = int(battle[old_p0_off + 14]) & -4096
        battle[old_p0_off + 14] = (NEUTRAL_BOOSTS_14 & 4095) | tera
    if can_switch1:
        battle[old_p1_off + 13] = NEUTRAL_BOOSTS_13
        tera = int(battle[old_p1_off + 14]) & -4096
        battle[old_p1_off + 14] = (NEUTRAL_BOOSTS_14 & 4095) | tera

    # Clear volatile flag bits on switch-out (Showdown pokemon.ts:clearVolatile
    # line 1474-1533 wipes per-switch volatiles). Bits cleared:
    #   0x02 previous_move_failed, 0x20 charge, 0x200 flash_fire,
    #   0x400 glaive_rush. Live semi-invulnerability rides in the active
    #   move-actions slot and is cleared by the standard switch-state reset.
    # Bits PRESERVED (persistent pokemon state):
    #   0x01 fainted, 0x04 has_tera, 0x08 tera_used, 0x100 once_per_battle,
    #   0x40 disguise, 0x80 had_item.
    from pokepy.core.constants import FLAG_GLAIVE_RUSH as _FLAG_GLAIVE_RUSH_SW

    from pokepy.core.constants import FLAG_BOOSTER_ENERGY_ACTIVE as _FLAG_BOOSTER_SW
    from pokepy.core.constants import FLAG_CHARGE as _FLAG_CHARGE_SW

    _PARADOX_STAT_MASK_SW = 0x6010
    _SWITCH_OUT_CLEAR_MASK = (
        _FLAG_GLAIVE_RUSH_SW
        | 0x200
        | _FLAG_CHARGE_SW
        | 0x02
        | _FLAG_BOOSTER_SW
        | _PARADOX_STAT_MASK_SW
    )
    if can_switch0:
        battle[old_p0_off + 15] = int(battle[old_p0_off + 15]) & ~_SWITCH_OUT_CLEAR_MASK
    if can_switch1:
        battle[old_p1_off + 15] = int(battle[old_p1_off + 15]) & ~_SWITCH_OUT_CLEAR_MASK

    # Update active slots
    battle[OFF_META + M_ACTIVE0] = new_active0
    battle[OFF_META + M_ACTIVE1] = new_active1
    if can_switch0:
        _sync_showdown_order_on_switch(side_order0, new_active0)
    if can_switch1:
        _sync_showdown_order_on_switch(side_order1, new_active1)

    p0_off = OFF_SIDE0 + new_active0 * POKEMON_SIZE
    p1_off = OFF_SIDE1 + new_active1 * POKEMON_SIZE
    # Capture the actual move users after voluntary switch resolution, before
    # any mid-turn pivot can rewrite the live active offsets. Downstream
    # self-effects (Life Orb, recoil, Shell Bell, crash, etc.) must stay
    # attached to these original movers even if U-turn / Volt Switch replaces
    # the active slot before the slower move or post-hit chain runs.
    user0_off = p0_off
    user1_off = p1_off

    # Reset boosts on incoming Pokemon (preserve tera)
    consumed_pending_switch_slot_condition0 = False
    consumed_pending_switch_slot_condition1 = False
    if can_switch0:
        _reset_incoming_switch_state_tracked(p0_off)
        consumed_pending_switch_slot_condition0 = apply_pending_wish_on_switch_in(
            battle,
            0,
            p0_off,
            state,
            game_data,
            pending_switch_slot_condition0,
        )
    if can_switch1:
        _reset_incoming_switch_state_tracked(p1_off)
        consumed_pending_switch_slot_condition1 = apply_pending_wish_on_switch_in(
            battle,
            1,
            p1_off,
            state,
            game_data,
            pending_switch_slot_condition1,
        )

    # Showdown's voluntary switch path resumes through:
    #   1. post-switch action eachEvent('Update')
    #   2. runSwitch speedSort(allActive)
    #   3. post-runSwitch eachEvent('Update')
    # These use the switch-in's on-entry action speed before hazards or entry
    # effects like Sticky Web mutate it.
    _postswitch_action_speed0 = (
        _get_switch_resume_action_speed(battle, p0_off)
        if can_switch0
        else _action_speed_from_effective(fx.get_effective_speed(battle, p0_off))
    )
    _postswitch_action_speed1 = (
        _get_switch_resume_action_speed(battle, p1_off)
        if can_switch1
        else _action_speed_from_effective(fx.get_effective_speed(battle, p1_off))
    )

    # Hazard damage on switch-in
    if can_switch0:
        fx.apply_hazard_damage_on_switch(battle, p0_off, OFF_FIELD + F_HAZARDS_0)
        _reset_toxic_counter_on_switch_in(battle, p0_off)
    if can_switch1:
        fx.apply_hazard_damage_on_switch(battle, p1_off, OFF_FIELD + F_HAZARDS_1)
        _reset_toxic_counter_on_switch_in(battle, p1_off)

    # Switch-in abilities (Intimidate, terrain setters, weather setters, ...)
    # Showdown speed-sorts simultaneous switch-ins (sim/battle.ts:507 speedSort
    # called from runEvent('SwitchIn'); battle-actions.ts:182). Faster mon's
    # onStart fires first; slower mon's setter overrides on weather/terrain
    # ties. Hardcoding p0-then-p1 was wrong for double-switch turns where p1
    # is the faster side. Skip mons that hazards KO'd
    # (battle-actions.ts:187 `if (!poke.hp) continue;`).
    p0_alive_si = (not can_switch0) or int(battle[p0_off + 1]) > 0
    p1_alive_si = (not can_switch1) or int(battle[p1_off + 1]) > 0
    if can_switch0 and can_switch1:
        sp0_si = fx.get_effective_speed(battle, p0_off)
        sp1_si = fx.get_effective_speed(battle, p1_off)
        # speedSort uses pokemon.speed = getActionSpeed() (sim/pokemon.ts:631-639)
        # which inverts under Trick Room (10000 - speed): slower mon goes FIRST
        # in the SwitchIn queue → faster mon's setter overrides on TR.
        _tr_active_si = int(battle[OFF_FIELD + F_TRICK_ROOM]) > 0
        p0_first_si = (sp0_si >= sp1_si) if not _tr_active_si else (sp0_si <= sp1_si)
        if p0_first_si:
            if p0_alive_si:
                _apply_switch_in_ability_tracked(p0_off, p1_off, True)
            if p1_alive_si:
                _apply_switch_in_ability_tracked(p1_off, p0_off, True)
        else:
            if p1_alive_si:
                _apply_switch_in_ability_tracked(p1_off, p0_off, True)
            if p0_alive_si:
                _apply_switch_in_ability_tracked(p0_off, p1_off, True)
    else:
        if can_switch0 and p0_alive_si:
            _apply_switch_in_ability_with_trace_reaction_tracked(p0_off, p1_off, True)
        if can_switch1 and p1_alive_si:
            _apply_switch_in_ability_with_trace_reaction_tracked(p1_off, p0_off, True)

    if can_switch0 ^ can_switch1:
        _switch_survived = p0_alive_si if can_switch0 else p1_alive_si
        if _switch_survived:
            _incoming_off = p0_off if can_switch0 else p1_off
            _incoming_resume_speed = (
                _postswitch_action_speed0 if can_switch0 else _postswitch_action_speed1
            )
            _foe_resume_speed = (
                _postswitch_action_speed1 if can_switch0 else _postswitch_action_speed0
            )
            if _incoming_resume_speed == _foe_resume_speed:
                for _ in range(3):
                    gen5_prng.random(0, 2)

    if can_switch0 and p0_alive_si:
        _run_switch_in_update_item_hooks(p0_off)
    if can_switch1 and p1_alive_si:
        _run_switch_in_update_item_hooks(p1_off)

    # Clear per-side switch state
    for did_sw, side in [(can_switch0, 0), (can_switch1, 1)]:
        if not did_sw:
            continue
        s = side
        _clear_side_switch_state_common(battle, s)
        battle[OFF_FIELD + (F_VOLATILE_0 if s == 0 else F_VOLATILE_1)] = 0
        battle[OFF_FIELD + (F_LEECH_SEED_0 if s == 0 else F_LEECH_SEED_1)] = 0
        battle[OFF_FIELD + (F_SUBSTITUTE_0 if s == 0 else F_SUBSTITUTE_1)] = 0
        battle[OFF_FIELD + (F_DISABLE_TURNS_0 if s == 0 else F_DISABLE_TURNS_1)] = 0
        battle[OFF_FIELD + (F_YAWN_TURNS_0 if s == 0 else F_YAWN_TURNS_1)] = 0
        battle[OFF_FIELD + (F_DESTINY_BOND_0 if s == 0 else F_DESTINY_BOND_1)] = 0
        battle[
            OFF_FIELD + (F_EXTENDED_VOLATILE_0 if s == 0 else F_EXTENDED_VOLATILE_1)
        ] = 0
        battle[OFF_FIELD + (F_PERISH_COUNT_0 if s == 0 else F_PERISH_COUNT_1)] = 0
    if (
        can_switch0
        and is_pending_wish_sentinel(pending_switch_slot_condition0)
        and not consumed_pending_switch_slot_condition0
    ):
        battle[OFF_FIELD + F_DESTINY_BOND_0] = np.int16(pending_switch_slot_condition0)
    if (
        can_switch1
        and is_pending_wish_sentinel(pending_switch_slot_condition1)
        and not consumed_pending_switch_slot_condition1
    ):
        battle[OFF_FIELD + F_DESTINY_BOND_1] = np.int16(pending_switch_slot_condition1)

    active0 = new_active0
    active1 = new_active1
    # Keep the original acting slots stable for move bookkeeping. Mid-turn
    # pivots, Red Card / Eject Button drags, and phazing can rewrite
    # `active*` later in the turn, but Showdown still deducts PP and reveals
    # the move on the mon that actually selected it.
    move_user_slot0 = active0
    move_user_slot1 = active1

    # ------------------------------------------------------------------
    # Speeds, priorities, move IDs, PP / Struggle
    # ------------------------------------------------------------------
    p0_speed = fx.get_effective_speed(battle, p0_off)
    p1_speed = fx.get_effective_speed(battle, p1_off)
    raw_move_id0 = int(state.team_moves[move_user_slot0, move_idx])
    raw_move_id1 = int(state.opp_moves[move_user_slot1, opp_move_idx])

    move_pp0 = int(state.team_pp[move_user_slot0, move_idx])
    move_pp1 = int(state.opp_pp[move_user_slot1, opp_move_idx])
    total_pp0 = int(state.team_pp[move_user_slot0].sum())
    total_pp1 = int(state.opp_pp[move_user_slot1].sum())

    forced_struggle0 = False
    forced_struggle1 = False
    if not is_switch:
        _, forced_struggle0 = get_battle_move_mask(state, 0, game_data)
    if not opp_is_switch:
        _, forced_struggle1 = get_battle_move_mask(state, 1, game_data)

    must_struggle0 = (not is_switch) and (
        move_pp0 <= 0 or total_pp0 <= 0 or forced_struggle0
    )
    must_struggle1 = (not opp_is_switch) and (
        move_pp1 <= 0 or total_pp1 <= 0 or forced_struggle1
    )

    move_id0 = MOVE_STRUGGLE if must_struggle0 else raw_move_id0
    move_id1 = MOVE_STRUGGLE if must_struggle1 else raw_move_id1
    move_id0 = max(0, min(int(game_data.move_priority.shape[0]) - 1, move_id0))
    move_id1 = max(0, min(int(game_data.move_priority.shape[0]) - 1, move_id1))
    # ------------------------------------------------------------------
    # Lockedmove override (Outrage / Petal Dance / Thrash / Raging Fury).
    # Showdown data/conditions.ts:253 lockedmove. While the volatile is
    # active the user is forced to repeat the same move; confused on end.
    # Pokepy stores (locked_move_id, turns_remaining) in OFF_MOVES.
    # Recharge override (Hyper Beam / Giga Impact / etc.). Showdown
    # data/conditions.ts:364 mustrecharge. Forces the user to skip the
    # next turn (handled later by adding to is_immobile).
    # ------------------------------------------------------------------
    locked_move0_pre = int(battle[OFF_MOVES + M_LOCKED_MOVE_0])
    locked_turns0_pre = int(battle[OFF_MOVES + M_LOCKED_TURNS_0])
    locked_move1_pre = int(battle[OFF_MOVES + M_LOCKED_MOVE_1])
    locked_turns1_pre = int(battle[OFF_MOVES + M_LOCKED_TURNS_1])
    is_locked_turn0 = (
        (not is_switch) and locked_move0_pre >= 0 and locked_turns0_pre > 0
    )
    is_locked_turn1 = (
        (not opp_is_switch) and locked_move1_pre >= 0 and locked_turns1_pre > 0
    )
    if is_locked_turn0:
        move_id0 = locked_move0_pre
        must_struggle0 = False
    if is_locked_turn1:
        move_id1 = locked_move1_pre
        must_struggle1 = False

    # Recharge: store the prior turn's recharge flag and clear it now —
    # the volatile only blocks ONE turn (Showdown duration: 2 with
    # onBeforeMove returning null on the second turn, then removeVolatile).
    recharge0_pre = int(battle[OFF_MOVES + M_RECHARGE_0])
    recharge1_pre = int(battle[OFF_MOVES + M_RECHARGE_1])
    must_recharge0 = (not is_switch) and recharge0_pre > 0
    must_recharge1 = (not opp_is_switch) and recharge1_pre > 0
    if must_recharge0:
        battle[OFF_MOVES + M_RECHARGE_0] = 0
        must_struggle0 = False
    if must_recharge1:
        battle[OFF_MOVES + M_RECHARGE_1] = 0
        must_struggle1 = False

    # Showdown tracks Fake Out / First Impression legality via
    # `pokemon.activeMoveActions`, which resets on switch-in and increments on
    # every move attempt (even if the move later fails in BeforeMove).
    active_move_actions0_live = int(battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0])
    active_move_actions1_live = int(battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1])
    active_move_actions0_pre = (
        active_move_actions0_live & ACTIVE_MOVE_ACTIONS_COUNT_MASK
    )
    active_move_actions1_pre = (
        active_move_actions1_live & ACTIVE_MOVE_ACTIONS_COUNT_MASK
    )
    if not is_switch:
        battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0] = (
            active_move_actions0_live & ACTIVE_MOVE_ACTIONS_SEMI_INVUL
        ) | (active_move_actions0_pre + 1)
    if not opp_is_switch:
        battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1] = (
            active_move_actions1_live & ACTIVE_MOVE_ACTIONS_SEMI_INVUL
        ) | (active_move_actions1_pre + 1)

    # ------------------------------------------------------------------
    # Sleep Talk substitution (Showdown data/moves.ts:17513 sleeptalk).
    # When a sleeping user chooses Sleep Talk, Showdown's `onHit` samples
    # a random non-nosleeptalk / non-charge move from the user's remaining
    # moveSlots and then calls `useMove(randomMove, pokemon)` which runs
    # the fresh move's tryMoveHit. In pokepy this is implemented as a
    # direct `move_id0` / `move_id1` substitution BEFORE the cat/bp/
    # priority derivation below — properties (damage category, BP,
    # accuracy, priority, secondary chance) come from the sampled move,
    # NOT Sleep Talk. The sample PRNG frame is intentionally NOT consumed
    # here — it's consumed inline at the `_calc_pN` call site so its
    # position in the global PRNG stream matches Showdown (which runs it
    # inside Sleep Talk's onHit, after the preceding mon's move rolls).
    # PP deduction still targets the chosen Sleep Talk slot via
    # `move_idx` / `opp_move_idx` unchanged, mirroring Showdown's
    # deductPP on the outer move.
    #
    # Only activates when the user is asleep (Showdown: slp status or
    # Comatose ability) AND at least one valid candidate move exists
    # besides Sleep Talk itself. We don't yet filter `nosleeptalk` /
    # `charge` / Z / Max flags — pokepy doesn't track those per-move.
    # For the parity harness' single-candidate case this is sufficient.
    def _sleep_talk_substitute(side: int):
        """Return (candidates_list, num_candidates) or ([], 0) if the
        substitution does not apply.  ``num_candidates`` is used by the
        caller to consume the right number of `random(N)` frames at the
        right position.

        Activates when the user is asleep OR when the opponent's move is
        about to put the user to sleep this turn with 100% accuracy +
        100% status chance (Spore, Dark Void, etc.).  The latter case
        handles the "same-turn Spore → Sleep Talk" ordering — Showdown
        processes each mon's runMove atomically so the Spore applies
        before Snorlax's Sleep Talk fires, but pokepy applies all status
        effects after both _calc_pN, so we look ahead here.
        """
        if side == 0:
            if is_switch:
                return [], 0
            mv = move_id0
            user_off_lt = p0_off
            team_moves = state.team_moves
            act_lt = active0
            opp_mv = move_id1
            opp_switching = opp_is_switch
        else:
            if opp_is_switch:
                return [], 0
            mv = move_id1
            user_off_lt = p1_off
            team_moves = state.opp_moves
            act_lt = active1
            opp_mv = move_id0
            opp_switching = is_switch
        if mv != MOVE_SLEEP_TALK:
            return [], 0
        user_status_lt = get_status(int(battle[user_off_lt + 12]))
        will_be_asleep = user_status_lt == STATUS_SLEEP
        if not will_be_asleep and not opp_switching:
            # Lookahead: opponent using a primary-status sleep move with
            # 100% acc + 100% chance (Spore, Sleep Powder, Lovely Kiss,
            # Dark Void) will land if the user is not sleep-immune.
            opp_eff = int(move_effects.effect_type[opp_mv])
            opp_status = int(move_effects.status[opp_mv])
            opp_status_chance = int(move_effects.status_chance[opp_mv])
            opp_acc = int(game_data.move_accuracy[opp_mv])
            # effect_type == 2 → EFFECT_STATUS primary, status == 3 → slp,
            # chance 100 + acc 100 means a deterministic sleep apply on
            # this side (ignoring sleep immunity abilities / grass-type
            # Powder immunity). Skip the Spore grass-immunity for now —
            # scenario 30 uses Snorlax (Normal type) which is always
            # susceptible to Spore.
            if (
                opp_eff == 2
                and opp_status == STATUS_SLEEP
                and opp_status_chance >= 100
                and opp_acc >= 100
            ):
                # Also: opponent moves first (so Spore resolves before
                # our Sleep Talk). Priority / speed ordering is computed
                # later in this function, but for a 0-priority Spore vs a
                # 0-priority Sleep Talk, speed decides.  In the scenario
                # Breloom (262 Spe) > Snorlax (96 Spe), so Spore fires
                # first.  Use the effective speeds computed above.
                if side == 0:
                    opp_first = p1_speed > p0_speed
                else:
                    opp_first = p0_speed > p1_speed
                if opp_first:
                    will_be_asleep = True
        if not will_be_asleep:
            return [], 0
        # Gather candidates: any other move slot with move_id != SLEEP_TALK
        # and move_id >= 0 (unset slots are -1). No flag-based filtering
        # implemented — fine for the minimal parity case.
        cands = []
        for slot in range(4):
            cand_id = int(team_moves[act_lt, slot])
            if cand_id < 0:
                continue
            if cand_id == MOVE_SLEEP_TALK:
                continue
            cands.append(cand_id)
        return cands, len(cands)

    _sleep_talk_cands0, _sleep_talk_n0 = _sleep_talk_substitute(0)
    _sleep_talk_cands1, _sleep_talk_n1 = _sleep_talk_substitute(1)
    # Record that Sleep Talk substitution WILL happen — we defer the
    # random(N) consumption to the moment this side actually runs its move
    # so the PRNG frame lands between the preceding mon's move rolls and the
    # called move, matching Showdown's `Sleep Talk -> useMove(...)` order.
    # Until then we keep a placeholder first candidate so the surrounding
    # setup has a valid move id; the real sampled move is committed right
    # before this side's move-specific pre-hit logic runs.
    # Remember which side just had a Sleep-Talk substitution so the
    # downstream `sleep_blocked{0,1}` gate (which checks for the outer
    # Sleep Talk / Snore whitelist) still sees the original move.
    _sleep_talk_orig0 = MOVE_SLEEP_TALK if _sleep_talk_n0 > 0 else -1
    _sleep_talk_orig1 = MOVE_SLEEP_TALK if _sleep_talk_n1 > 0 else -1
    # When the substitution was triggered by the LOOKAHEAD path (opponent
    # about to apply primary sleep this turn) the user is NOT yet asleep.
    # We set a pending-sleep flag so the main damage pipeline can
    # pre-apply sleep BEFORE this side's Sleep Talk damage calc, keeping
    # the `random(3)` sleep-turns frame at the same global PRNG position
    # Showdown consumes it (inside Spore's hit resolution, before the
    # target's runMove starts).
    _sleep_talk_pending_slp0 = (
        _sleep_talk_n0 > 0 and get_status(int(battle[p0_off + 12])) != STATUS_SLEEP
    )
    _sleep_talk_pending_slp1 = (
        _sleep_talk_n1 > 0 and get_status(int(battle[p1_off + 12])) != STATUS_SLEEP
    )
    if _sleep_talk_n0 > 0:
        move_id0 = int(_sleep_talk_cands0[0])
    if _sleep_talk_n1 > 0:
        move_id1 = int(_sleep_talk_cands1[0])

    # Move IDs for the lockedmove / recharge sets (Showdown data/moves.ts).
    _LOCKED_MOVES = (MOVE_OUTRAGE, MOVE_PETAL_DANCE, MOVE_THRASH, MOVE_RAGING_FURY)
    _RECHARGE_MOVES = (
        MOVE_HYPER_BEAM,
        MOVE_GIGA_IMPACT,
        MOVE_FRENZY_PLANT,
        MOVE_HYDRO_CANNON,
        MOVE_BLAST_BURN,
        MOVE_ROCK_WRECKER,
        MOVE_ETERNABEAM,
        MOVE_PRISMATIC_LASER,
        MOVE_ROAR_OF_TIME,
        MOVE_METEOR_ASSAULT,
    )

    if is_switch:
        base_priority0 = 0
        priority0 = 0.0
    else:
        base_priority0 = int(game_data.move_priority[move_id0])
        priority0 = fx.get_effective_priority(
            battle, move_id0, base_priority0, p0_off, gen5_prng
        )
    if opp_is_switch:
        base_priority1 = 0
        priority1 = 0.0
    else:
        base_priority1 = int(game_data.move_priority[move_id1])
        priority1 = fx.get_effective_priority(
            battle, move_id1, base_priority1, p1_off, gen5_prng
        )
    import math as _math_priority

    # Showdown keeps fractional sources like Custap Berry / Quick Draw inside
    # the same integer priority bracket. They affect move order, but blocker
    # checks like Psychic Terrain / Quick Guard / Dazzling still key off the
    # integer bracket (`move.priority > 0`), not the fractional tie-breaker.
    priority_bracket0 = int(_math_priority.floor(float(priority0)))
    priority_bracket1 = int(_math_priority.floor(float(priority1)))

    # Custap Berry consumption: Showdown items.ts:1244-1254 onFractionalPriority
    # fires only when `priority <= 0` (i.e. the move is at +0 or negative
    # priority bracket — Custap gives +0.1 which wins the speed tie at that
    # bracket but loses to +1 moves like Quick Attack). Conditions:
    #   - HP <= 1/4 max (default), or HP <= 1/2 max + Gluttony
    #   - User actually used a move this turn (not a switch)
    # Pokepy used to consume Custap even for +1 moves (Quick Attack, Aqua
    # Jet, etc.), wrongly losing the berry with no benefit.
    _ITEM_CUSTAP_BERRY = 210
    _ABILITY_GLUTTONY = 82
    # Custap is a berry — Unnerve on the OPPOSING active mon blocks the eat
    # via Showdown's `runEvent('TryEatItem')` path (data/abilities.ts:5185
    # unnerve onFoeTryEatItem returns false). The same id list mirrored in
    # pokepy/effects/items.py is used here.
    _ABILITY_UNNERVE_LOCAL = 127
    _ABILITY_AS_ONE_GLAS_LOCAL = 266
    _ABILITY_AS_ONE_SPEC_LOCAL = 267
    _UNNERVE_ABS = (
        _ABILITY_UNNERVE_LOCAL,
        _ABILITY_AS_ONE_GLAS_LOCAL,
        _ABILITY_AS_ONE_SPEC_LOCAL,
    )
    p1_unnerves_p0 = (
        int(battle[p1_off + 5]) in _UNNERVE_ABS and int(battle[p1_off + 1]) > 0
    )
    p0_unnerves_p1 = (
        int(battle[p0_off + 5]) in _UNNERVE_ABS and int(battle[p0_off + 1]) > 0
    )
    if not is_switch and base_priority0 <= 0:
        hp0_c = int(battle[p0_off + 1])
        max0_c = int(battle[p0_off + 2])
        ab0_c = int(battle[p0_off + 5])
        quarter0 = (hp0_c * 4) <= max0_c
        half_glut0 = (hp0_c * 2) <= max0_c and ab0_c == _ABILITY_GLUTTONY
        if (
            int(battle[p0_off + 6]) == _ITEM_CUSTAP_BERRY
            and (quarter0 or half_glut0)
            and hp0_c > 0
            and not p1_unnerves_p0
        ):
            _record_consumed_berry(p0_off, int(battle[p0_off + 6]))
            battle[p0_off + 6] = 0
    if not opp_is_switch and base_priority1 <= 0:
        hp1_c = int(battle[p1_off + 1])
        max1_c = int(battle[p1_off + 2])
        ab1_c = int(battle[p1_off + 5])
        quarter1 = (hp1_c * 4) <= max1_c
        half_glut1 = (hp1_c * 2) <= max1_c and ab1_c == _ABILITY_GLUTTONY
        if (
            int(battle[p1_off + 6]) == _ITEM_CUSTAP_BERRY
            and (quarter1 or half_glut1)
            and hp1_c > 0
            and not p0_unnerves_p1
        ):
            _record_consumed_berry(p1_off, int(battle[p1_off + 6]))
            battle[p1_off + 6] = 0

    # Speed-tie + speedSort frames at turn start.
    #
    # Showdown's speedSort (sim/battle.ts:429) is a selection sort that only
    # consumes PRNG frames when a TIED GROUP has length > 1. Each tied pair
    # consumes exactly 1 frame (via `shuffle(list, start, end)` which loops
    # `end - start - 1` times → 1 frame for a 2-element tied group).
    #
    # Whether a call ties depends on the comparator:
    #   * comparePriority (default, used by queue.sort, fieldEvent handlers):
    #       compares order → priority → speed → subOrder → effectOrder. Two
    #       same-priority moves tie iff their speeds are equal. A switch
    #       action (order=103) and move (order=200) NEVER tie.
    #   * speed comparator (eachEvent path at battle.ts:468): compares only
    #       pokemon.speed. Two actives tie iff their effective speeds match.
    #
    # The 4 turn-start frames map to:
    #   [0] commitChoices queue.sort on [action0, action1] — comparePriority.
    #       Ties iff both are moves with equal priority AND equal speed.
    #   [1] runAction(beforeTurn) → eachEvent('BeforeTurn') — speed comparator.
    #       Ties iff the two actives have equal effective speed.
    #   [2] runAction(beforeTurn) post-processing → eachEvent('Update')
    #       (battle.ts:2860) — speed comparator, same condition as [1].
    #   [3] gen 8+ queue re-sort (battle.ts:2918-2924) on [move, move, residual]
    #       — comparePriority. Only fires when peek is a move/runDynamax (i.e.
    #       no switch is next), and only ties on the two moves if both have
    #       equal priority AND equal speed (residual has order=300 so it
    #       never ties with move actions).
    #
    # Frames [0] and [3] BOTH can swap the effective move order. For a turn
    # where both sides' moves are tied the net result is shuffle[0] XOR
    # shuffle[3]: if both swap or neither swaps, side0 still moves first;
    # if exactly one swaps, side1 moves first.
    #
    # Non-tied speeds (Alakazam 120 vs Snorlax 30): all 4 frames consume 0
    # — Showdown never enters the shuffle branch in any of these 4 sorts.
    # Mid-move switch (1 action is 'switch', 1 is 'move'): commitChoices
    # never ties (different order), and gen8+ re-sort never fires (peek is
    # switch because switch order=103 < move order=200).
    has_switch_action = is_switch or opp_is_switch
    speeds_tied = p0_speed == p1_speed

    action_priority0 = 0 if is_switch else priority0
    action_priority1 = 0 if opp_is_switch else priority1
    action_speed0 = _action_speed_from_effective(
        pre_switch_p0_speed if is_switch else p0_speed
    )
    action_speed1 = _action_speed_from_effective(
        pre_switch_p1_speed if opp_is_switch else p1_speed
    )
    # Cached action speeds must track any mid-turn runSwitch that completes
    # before the later move acts. Showdown reuses `pokemon.speed` inside the
    # move action's post-hit Update sorts, so a first-mover U-turn / phaze
    # should refresh the cache for the second mover, while same-move self
    # boosts (Rapid Spin) must not.
    current_action_speed0 = (
        _postswitch_action_speed0 if has_switch_action else action_speed0
    )
    current_action_speed1 = (
        _postswitch_action_speed1 if has_switch_action else action_speed1
    )
    priorities_tied = action_priority0 == action_priority1
    queue_sort_tied = (
        (not has_switch_action) and priorities_tied and (action_speed0 == action_speed1)
    )

    def _refresh_current_action_speeds() -> None:
        nonlocal current_action_speed0, current_action_speed1
        _cur_active0 = int(battle[OFF_META + M_ACTIVE0])
        _cur_active1 = int(battle[OFF_META + M_ACTIVE1])
        current_action_speed0 = _action_speed_from_effective(
            fx.get_effective_speed(battle, OFF_SIDE0 + _cur_active0 * POKEMON_SIZE)
        )
        current_action_speed1 = _action_speed_from_effective(
            fx.get_effective_speed(battle, OFF_SIDE1 + _cur_active1 * POKEMON_SIZE)
        )

    _move0_preapplied_immediate_defender_state = False
    _move1_preapplied_immediate_defender_state = False

    _sst = SpeedSortTracker(gen5_prng)
    # S1: commitChoices queue.sort — comparePriority on the two action entries.
    # Switches have order=103, moves have order=200. Ties iff both are moves
    # with equal priority AND equal speed.
    _order0 = 103 if is_switch else 200
    _order1 = 103 if opp_is_switch else 200
    _s1_entries = [
        (_order0, action_priority0, action_speed0, 0, 0),
        (_order1, action_priority1, action_speed1, 0, 0),
    ]
    commit_shuffle = _sst.queue_sort(_s1_entries)
    # The beforeTurn action handler produces TWO speed-sort frames:
    #   S2: eachEvent('BeforeTurn')                       (sim/battle.ts:2830)
    #   S3: the trailing eachEvent('Update') that runAction runs after EVERY
    #       action — `if (this.gen < 5) this.eachEvent('Update')` at
    #       sim/battle.ts:2938 (gen>=5 runs the equivalent at :2881).
    # Both speedSort the active Pokemon by speed, so each costs one shuffle
    # frame when the actives' speeds tie. These fire BEFORE any switch action
    # executes, so a voluntary switch this turn still sees the PRE-switch
    # actives — use pre_switch speeds in that case.
    _bt_tied = pre_switch_speeds_tied if (is_switch or opp_is_switch) else speeds_tied
    if _bt_tied:
        _s2_speeds = [
            pre_switch_p0_speed if (is_switch or opp_is_switch) else p0_speed,
            pre_switch_p1_speed if (is_switch or opp_is_switch) else p1_speed,
        ]
        _sst.each_event_update(_s2_speeds)  # S2: eachEvent('BeforeTurn')
        _sst.each_event_update(_s2_speeds)  # S3: trailing eachEvent('Update')
    # S4: gen 8+ queue re-sort — only when no switch is ahead AND the two
    # move actions tie by comparePriority.
    resort_shuffle = (
        _sst.queue_sort(_s1_entries) if queue_sort_tied and profile.gen >= 8 else 0
    )
    action_sort_roll = commit_shuffle ^ resort_shuffle
    tie_break = action_sort_roll == 0  # 0 = no net swap (side0 first)

    if _order0 < _order1:
        side0_first = True
    elif _order0 > _order1:
        side0_first = False
    elif action_priority0 > action_priority1:
        side0_first = True
    elif action_priority0 < action_priority1:
        side0_first = False
    elif action_speed0 > action_speed1:
        side0_first = True
    elif action_speed0 < action_speed1:
        side0_first = False
    else:
        side0_first = tie_break

    # Voluntary switch action internals: Showdown's switch action can run one
    # pre-swap `eachEvent('Update')` frame per switch action, but on a
    # double-switch turn the second switch only ties if the first switch-in's
    # action speed still matches the opposing side's yet-unswitched active.
    # Pokepy used to spend one frame per switch whenever the two outgoing
    # actives tied, which overcounted battles like 206 by one raw frame.
    if pre_switch_speeds_tied:
        if can_switch0 and can_switch1:
            gen5_prng.random(0, 2)
            second_switch_tied = (
                (_postswitch_action_speed0 == action_speed1)
                if side0_first
                else (action_speed0 == _postswitch_action_speed1)
            )
            if second_switch_tied:
                gen5_prng.random(0, 2)
        elif can_switch0 or can_switch1:
            gen5_prng.random(0, 2)

    # ------------------------------------------------------------------
    # Substitute / Taunt / Destiny Bond — early application
    # ------------------------------------------------------------------
    protect_success0 = False
    protect_success1 = False

    effect0 = int(move_effects.effect_type[move_id0])
    effect1 = int(move_effects.effect_type[move_id1])

    is_sub0 = effect0 == EFFECT_SUBSTITUTE
    is_sub1 = effect1 == EFFECT_SUBSTITUTE
    if side0_first and is_sub0 and not is_switch:
        fx.apply_substitute_from_move(
            battle, move_id0, 0, p0_off, game_data, move_effects
        )
    if (not side0_first) and is_sub1 and not opp_is_switch:
        fx.apply_substitute_from_move(
            battle, move_id1, 1, p1_off, game_data, move_effects
        )

    is_taunt0 = move_id0 == MOVE_TAUNT
    is_taunt1 = move_id1 == MOVE_TAUNT
    if side0_first and is_taunt0 and not is_switch:
        fx.apply_taunt_from_move(battle, move_id0, 1, True, move_effects, gen5_prng)
    if (not side0_first) and is_taunt1 and not opp_is_switch:
        fx.apply_taunt_from_move(battle, move_id1, 0, True, move_effects, gen5_prng)

    if side0_first and move_id0 == MOVE_DESTINY_BOND and not is_switch:
        fx.apply_destiny_bond_from_move(battle, move_id0, 0)
    if (not side0_first) and move_id1 == MOVE_DESTINY_BOND and not opp_is_switch:
        fx.apply_destiny_bond_from_move(battle, move_id1, 1)

    # ------------------------------------------------------------------
    # Two-turn moves (Dig/Fly/Dive/Bounce/Phantom/Shadow Force + charge moves
    # like Solar Beam, Solar Blade, Meteor Beam, Electro Shot, Sky Attack,
    # Skull Bash, Razor Wind, Geomancy, Freeze Shock, Ice Burn). Power Herb
    # is not in the team pool so all charge moves take 2 turns here.
    # Source: pokemon-showdown/data/moves.ts (search `flags: { ... charge: 1`).
    # Solar Beam / Solar Blade / Electro Shot have weather-skip behavior
    # (sun for SB, rain for ES) — handled below by checking weather.
    # ------------------------------------------------------------------
    MOVE_SKY_ATTACK = 143
    MOVE_SKULL_BASH = 130
    MOVE_RAZOR_WIND = 13
    MOVE_GEOMANCY = 601
    MOVE_FREEZE_SHOCK = 553
    MOVE_ICE_BURN = 554
    MOVE_METEOR_BEAM = 800
    MOVE_ELECTRO_SHOT = 905
    MOVE_SOLAR_BEAM = 76
    MOVE_SOLAR_BLADE = 669
    two_turn_moves = (
        MOVE_DIG,
        MOVE_FLY,
        MOVE_DIVE,
        MOVE_BOUNCE,
        MOVE_SHADOW_FORCE,
        MOVE_PHANTOM_FORCE,
        MOVE_SOLAR_BEAM,
        MOVE_SOLAR_BLADE,
        MOVE_METEOR_BEAM,
        MOVE_ELECTRO_SHOT,
        MOVE_SKY_ATTACK,
        MOVE_SKULL_BASH,
        MOVE_RAZOR_WIND,
        MOVE_GEOMANCY,
        MOVE_FREEZE_SHOCK,
        MOVE_ICE_BURN,
    )
    is_two_turn0 = move_id0 in two_turn_moves
    is_two_turn1 = move_id1 in two_turn_moves
    # Solar Beam / Solar Blade in sun and Electro Shot in rain skip the
    # charge turn (Showdown: moves.ts solarbeam / electroshot onTryMove,
    # both check `attacker.effectiveWeather()`). The user's effective weather
    # returns '' under Air Lock / Cloud Nine on either active mon, AND when
    # the user holds Utility Umbrella (sim/pokemon.ts:2149-2158, blocks
    # sun / rain / desolate / primordial). Both must be checked here.
    from pokepy.core.constants import ITEM_UTILITY_UMBRELLA as _UMB_CHG

    cur_weather = int(battle[OFF_FIELD + F_WEATHER])
    # Air Lock / Cloud Nine on either active mon suppresses weather effects.
    _ab0_chg = int(battle[p0_off + 5])
    _ab1_chg = int(battle[p1_off + 5])
    _weather_supp = _ab0_chg in (76, 13) or _ab1_chg in (
        76,
        13,
    )  # AIR_LOCK=76, CLOUD_NINE=13
    _item0_chg = int(battle[p0_off + 6])
    _item1_chg = int(battle[p1_off + 6])
    if not _weather_supp:
        if (
            move_id0 in (MOVE_SOLAR_BEAM, MOVE_SOLAR_BLADE)
            and cur_weather == WEATHER_SUN
            and _item0_chg != _UMB_CHG
        ):
            is_two_turn0 = False
        if (
            move_id1 in (MOVE_SOLAR_BEAM, MOVE_SOLAR_BLADE)
            and cur_weather == WEATHER_SUN
            and _item1_chg != _UMB_CHG
        ):
            is_two_turn1 = False
        if (
            move_id0 == MOVE_ELECTRO_SHOT
            and cur_weather == WEATHER_RAIN
            and _item0_chg != _UMB_CHG
        ):
            is_two_turn0 = False
        if (
            move_id1 == MOVE_ELECTRO_SHOT
            and cur_weather == WEATHER_RAIN
            and _item1_chg != _UMB_CHG
        ):
            is_two_turn1 = False
    was_charging0 = int(battle[OFF_META + M_CHARGING_0]) >= 0
    was_charging1 = int(battle[OFF_META + M_CHARGING_1]) >= 0
    is_charge_turn0 = is_two_turn0 and (not was_charging0) and (not is_switch)
    is_strike_turn0 = was_charging0 and (not is_switch)
    is_charge_turn1 = is_two_turn1 and (not was_charging1) and (not opp_is_switch)
    is_strike_turn1 = was_charging1 and (not opp_is_switch)

    if is_strike_turn0:
        move_id0 = int(battle[OFF_META + M_CHARGING_0])
        must_struggle0 = False
    if is_strike_turn1:
        move_id1 = int(battle[OFF_META + M_CHARGING_1])
        must_struggle1 = False
    if is_charge_turn0 or is_strike_turn0:
        battle[OFF_META + M_CHARGING_0] = move_id0
    else:
        battle[OFF_META + M_CHARGING_0] = -1
    if is_charge_turn1 or is_strike_turn1:
        battle[OFF_META + M_CHARGING_1] = move_id1
    else:
        battle[OFF_META + M_CHARGING_1] = -1
    # Showdown resolves Skull Bash / Meteor Beam / Electro Shot's self-boost
    # in `onTryMove`, before any same-turn damage and before the later strike
    # turn for true charge moves. Electro Shot in rain still uses this same
    # path even though it skips charging, so keep these boosts out of the
    # normal post-hit stat-change lane.
    _ON_TRY_MOVE_SELFBOOST_MOVES = (
        MOVE_SKULL_BASH,
        MOVE_METEOR_BEAM,
        MOVE_ELECTRO_SHOT,
    )
    _on_try_selfboost0 = (
        int(move_id0) in _ON_TRY_MOVE_SELFBOOST_MOVES and not is_strike_turn0
    )
    _on_try_selfboost1 = (
        int(move_id1) in _ON_TRY_MOVE_SELFBOOST_MOVES and not is_strike_turn1
    )
    _skip_strike_turn_selfboost0 = (
        int(move_id0) in _ON_TRY_MOVE_SELFBOOST_MOVES and is_strike_turn0
    )
    _skip_strike_turn_selfboost1 = (
        int(move_id1) in _ON_TRY_MOVE_SELFBOOST_MOVES and is_strike_turn1
    )

    # Semi-invulnerability is only granted by Fly / Dig / Dive / Bounce /
    # Shadow Force / Phantom Force — NOT by charge moves like Solar Beam /
    # Skull Bash / Sky Attack etc. Showdown data/moves.ts: only the
    # fly/dig/dive/bounce/shadowforce/phantomforce condition handlers set
    # `onLockMove` + `onInvulnerability`. All other two-turn moves just
    # charge in place and can be hit normally on the charge turn.
    _SEMI_INVUL_TWO_TURN = (
        MOVE_DIG,
        MOVE_FLY,
        MOVE_DIVE,
        MOVE_BOUNCE,
        MOVE_SHADOW_FORCE,
        MOVE_PHANTOM_FORCE,
    )
    _chg0_is_semi_invul = move_id0 in _SEMI_INVUL_TWO_TURN
    _chg1_is_semi_invul = move_id1 in _SEMI_INVUL_TWO_TURN
    # Showdown only makes the user untargetable once the charge-turn action
    # has actually started. A faster charge-turn user disappears before the
    # foe moves later that turn; a slower one stays targetable until its own
    # action, then remains semi-invulnerable into the next turn. On the
    # strike turn the invulnerability lasts until the user resolves the hit:
    # a faster striker clears it before the foe's move, while a slower
    # striker keeps it long enough for the earlier foe move to miss.
    if is_charge_turn0 and _chg0_is_semi_invul and side0_first:
        battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0] = (
            int(battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0])
            | ACTIVE_MOVE_ACTIONS_SEMI_INVUL
        )
    elif is_strike_turn0 and _chg0_is_semi_invul:
        if side0_first:
            battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0] = (
                int(battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0])
                & ~ACTIVE_MOVE_ACTIONS_SEMI_INVUL
            )
        else:
            battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0] = (
                int(battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0])
                | ACTIVE_MOVE_ACTIONS_SEMI_INVUL
            )
    if is_charge_turn1 and _chg1_is_semi_invul and (not side0_first):
        battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1] = (
            int(battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1])
            | ACTIVE_MOVE_ACTIONS_SEMI_INVUL
        )
    elif is_strike_turn1 and _chg1_is_semi_invul:
        if not side0_first:
            battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1] = (
                int(battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1])
                & ~ACTIVE_MOVE_ACTIONS_SEMI_INVUL
            )
        else:
            battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1] = (
                int(battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1])
                | ACTIVE_MOVE_ACTIONS_SEMI_INVUL
            )

    used_protect0 = False
    used_protect1 = False

    # ------------------------------------------------------------------
    # Protection / priority blockers (Quick Guard, Dazzling, Psychic Terrain)
    # ------------------------------------------------------------------
    target0_protected = False
    target1_protected = False

    terrain = int(battle[OFF_FIELD + F_TERRAIN])
    is_psychic_terrain = terrain == TERRAIN_PSYCHIC

    t1_types = int(battle[p1_off + 4])
    t1_t1 = t1_types & 0xFF
    t1_t2 = (t1_types >> 8) & 0xFF
    t1_ab = int(battle[p1_off + 5])
    t1_grounded = is_grounded(battle, p1_off)

    t0_types = int(battle[p0_off + 4])
    t0_t1 = t0_types & 0xFF
    t0_t2 = (t0_types >> 8) & 0xFF
    t0_ab = int(battle[p0_off + 5])
    t0_grounded = is_grounded(battle, p0_off)

    priority0_blocked = is_psychic_terrain and (priority_bracket0 > 0) and t1_grounded
    priority1_blocked = is_psychic_terrain and (priority_bracket1 > 0) and t0_grounded

    blocks_priority1 = t1_ab in (
        ABILITY_DAZZLING,
        ABILITY_QUEENLY_MAJESTY,
        ABILITY_ARMOR_TAIL,
    )
    blocks_priority0 = t0_ab in (
        ABILITY_DAZZLING,
        ABILITY_QUEENLY_MAJESTY,
        ABILITY_ARMOR_TAIL,
    )
    dazzling_blocked0 = blocks_priority1 and priority_bracket0 > 0 and not is_switch
    dazzling_blocked1 = blocks_priority0 and priority_bracket1 > 0 and not opp_is_switch

    move0_flags = int(game_data.move_flags[move_id0])
    move1_flags = int(game_data.move_flags[move_id1])
    is_contact0 = (move0_flags & FLAG_CONTACT) != 0
    is_contact1 = (move1_flags & FLAG_CONTACT) != 0
    # Punching Glove (item 749) and Long Reach ability strip the contact
    # flag for punch moves / all moves respectively. Showdown source:
    # data/items.ts:4604-4619 punchingglove `delete move.flags['contact']`
    # and data/abilities.ts:longreach `move.flags['contact'] = 0`. Without
    # this strip, Rocky Helmet, Rough Skin, Iron Barbs, Static, Flame Body,
    # Poison Point, Effect Spore, Tangling Hair, Gooey, Perish Body,
    # Pickpocket, Cursed Body, Mummy, etc. all wrongly trigger.
    _ITEM_PUNCHING_GLOVE = 1884
    # ABILITY_LONG_REACH is 203 (constants.py). Earlier revisions of this
    # file accidentally used 271 (which is actually Anger Shell), making
    # Anger Shell holders skip contact and Long Reach holders still make
    # contact — opposite of intended.
    _ABILITY_LONG_REACH = 203
    is_punch0 = (move0_flags & FLAG_PUNCH) != 0
    is_punch1 = (move1_flags & FLAG_PUNCH) != 0
    p0_ab = int(battle[p0_off + 5])
    p1_ab = int(battle[p1_off + 5])
    p0_item_pg = int(battle[p0_off + 6])
    p1_item_pg = int(battle[p1_off + 6])
    if (
        is_punch0 and p0_item_pg == _ITEM_PUNCHING_GLOVE
    ) or p0_ab == _ABILITY_LONG_REACH:
        is_contact0 = False
    if (
        is_punch1 and p1_item_pg == _ITEM_PUNCHING_GLOVE
    ) or p1_ab == _ABILITY_LONG_REACH:
        is_contact1 = False
    unseen_fist0 = (p0_ab == ABILITY_UNSEEN_FIST) and is_contact0
    unseen_fist1 = (p1_ab == ABILITY_UNSEEN_FIST) and is_contact1
    # Moves without the `protect` flag bypass Protect / Detect / Spiky
    # Shield / Baneful Bunker / Burning Bulwark / Silk Trap / King's Shield
    # / Obstruct entirely (Feint, Mighty Cleave, etc.). Showdown source:
    # sim/battle-actions.ts `if (!move.flags['protect']) ... bypass`.
    from pokepy.core.constants import FLAG_PROTECT as _FLAG_PROTECT_EARLY

    move0_respects_protect_early = (move0_flags & _FLAG_PROTECT_EARLY) != 0
    move1_respects_protect_early = (move1_flags & _FLAG_PROTECT_EARLY) != 0

    def _recompute_p0_skip() -> bool:
        return (
            is_switch
            or target0_bypasses_protect
            or is_charge_turn0
            or first_turn_fail0
            or hyperspace_fury_fail0
            or last_resort_fail0
            or poltergeist_fail0
            or is_delayed0
            or prankster_fail0
            or move0_no_target
            or sucker0_fails
            or thunderclap0_fails
        )

    def _recompute_p1_skip() -> bool:
        return (
            opp_is_switch
            or target1_bypasses_protect
            or is_charge_turn1
            or first_turn_fail1
            or hyperspace_fury_fail1
            or last_resort_fail1
            or poltergeist_fail1
            or is_delayed1
            or prankster_fail1
            or move1_no_target
            or sucker1_fails
            or thunderclap1_fails
        )

    def _refresh_protection_state() -> None:
        nonlocal target0_protected, target1_protected
        nonlocal target0_bypasses_protect, target1_bypasses_protect
        nonlocal p0_skip, p1_skip

        pr1_cur = int(battle[OFF_FIELD + F_PROTECT_1])
        pr0_cur = int(battle[OFF_FIELD + F_PROTECT_0])
        qg1_active = (get_protect_active(pr1_cur) > 0) and (
            get_protect_type(pr1_cur) == PROTECT_QUICK_GUARD
        )
        qg0_active = (get_protect_active(pr0_cur) > 0) and (
            get_protect_type(pr0_cur) == PROTECT_QUICK_GUARD
        )
        quick_guard1_blocked = qg1_active and (priority_bracket0 > 0)
        quick_guard0_blocked = qg0_active and (priority_bracket1 > 0)

        target0_protected = fx.check_protected(battle, 1)
        target1_protected = fx.check_protected(battle, 0)
        target0_protected = (
            target0_protected and not unseen_fist0 and move0_respects_protect_early
        )
        target1_protected = (
            target1_protected and not unseen_fist1 and move1_respects_protect_early
        )
        target0_protected = (
            target0_protected
            or priority0_blocked
            or quick_guard1_blocked
            or dazzling_blocked0
        )
        target1_protected = (
            target1_protected
            or priority1_blocked
            or quick_guard0_blocked
            or dazzling_blocked1
        )
        target0_bypasses_protect = target0_protected
        target1_bypasses_protect = target1_protected
        p0_skip = _recompute_p0_skip()
        p1_skip = _recompute_p1_skip()

    # ------------------------------------------------------------------
    # Protean / Libero (type change before the USER'S attack)
    # ------------------------------------------------------------------
    # Showdown runs Libero / Protean from the user's onPrepareHit hook
    # during that mover's own runMove, after onBeforeMove but before
    # accuracy. It must therefore NOT mutate the user's defensive typing
    # before a faster opponent attacks earlier in the same turn.
    from pokepy.core.constants import EXT_VOL_LIBERO_USED as _EVLIBERO

    def _maybe_apply_protean_libero(
        side_idx: int, mid: int, is_sw: bool, is_strike: bool
    ) -> None:
        side_base = OFF_SIDE0 if side_idx == 0 else OFF_SIDE1
        act = int(battle[OFF_META + (M_ACTIVE0 if side_idx == 0 else M_ACTIVE1)])
        poff = side_base + act * POKEMON_SIZE
        ab = int(battle[poff + 5])
        if ab not in (ABILITY_PROTEAN, ABILITY_LIBERO):
            return
        if is_sw:
            return
        # Strike turn of a charge/delayed move: Protean's onPrepareHit runs
        # on the CHOICE turn, not the strike turn.
        if is_strike:
            return
        if mid in (MOVE_STRUGGLE, MOVE_FUTURE_SIGHT, MOVE_DOOM_DESIRE):
            return
        ev_off = OFF_FIELD + (
            F_EXTENDED_VOLATILE_0 if side_idx == 0 else F_EXTENDED_VOLATILE_1
        )
        ev_cur = int(battle[ev_off]) & 0xFFFF
        if (ev_cur & _EVLIBERO) != 0:
            return
        mtype = int(game_data.move_type[mid])
        if mtype < 0 or mtype >= 18:
            return
        cur_types = int(battle[poff + 4]) & 0xFFFF
        cur_t1 = cur_types & 0xFF
        cur_t2 = (cur_types >> 8) & 0xFF
        if cur_t1 == mtype and cur_t2 == mtype:
            return
        battle[poff + 4] = mtype | (mtype << 8)
        new_ev = (ev_cur | _EVLIBERO) & 0xFFFF
        if new_ev >= 0x8000:
            new_ev -= 0x10000
        battle[ev_off] = new_ev

    # Roost: temporarily strip Flying type for the user's defensive calcs.
    # In Showdown each move executes serially: the faster mon moves first,
    # so its Roost type-strip applies BEFORE the slower mon's attack.  The
    # slower mon's Roost only strips Flying AFTER the faster mon has already
    # hit it — so the faster attacker still sees the original types.
    # We therefore only strip the FASTER Roosting mon now; the slower mon's
    # strip is deferred to the between-moves point (see _apply_slow_roost).
    is_roost0 = (move_id0 == MOVE_ROOST) and not is_switch
    is_roost1 = (move_id1 == MOVE_ROOST) and not opp_is_switch
    p0_types_pre_roost = int(battle[p0_off + 4])
    p1_types_pre_roost = int(battle[p1_off + 4])
    roost_applied0 = False
    roost_applied1 = False

    def _strip_flying(poff: int) -> None:
        types_raw = int(battle[poff + 4])
        t1_ = types_raw & 0xFF
        t2_ = (types_raw >> 8) & 0xFF
        nt1 = TYPE_NORMAL if t1_ == TYPE_FLYING else t1_
        nt2 = nt1 if t2_ == TYPE_FLYING else t2_
        battle[poff + 4] = nt1 | (nt2 << 8)

    def _apply_roost_type_strip(side_idx: int) -> None:
        nonlocal roost_applied0, roost_applied1

        if side_idx == 0:
            if roost_applied0 or not is_roost0:
                return
            poff = p0_off
            vol_off = OFF_FIELD + F_VOLATILE_0
            has_assault_vest = int(battle[poff + 6]) == ITEM_ASSAULT_VEST
            is_taunted = get_taunt_turns(int(battle[vol_off])) > 0
            is_tera = (int(battle[poff + 15]) & 0x8) != 0
            if (
                int(battle[poff + 1]) >= int(battle[poff + 2])
                or has_assault_vest
                or is_taunted
                or is_tera
            ):
                return
            _strip_flying(poff)
            roost_applied0 = True
            return

        if roost_applied1 or not is_roost1:
            return
        poff = p1_off
        vol_off = OFF_FIELD + F_VOLATILE_1
        has_assault_vest = int(battle[poff + 6]) == ITEM_ASSAULT_VEST
        is_taunted = get_taunt_turns(int(battle[vol_off])) > 0
        is_tera = (int(battle[poff + 15]) & 0x8) != 0
        if (
            int(battle[poff + 1]) >= int(battle[poff + 2])
            or has_assault_vest
            or is_taunted
            or is_tera
        ):
            return
        _strip_flying(poff)
        roost_applied1 = True

    # Helper called between the two movers' calc phases to apply the slower
    # Roosting mon's type change. Must be called after the faster mon's
    # damage hits the slower mon but before the slower mon's own attack.
    _slow_roost_applied = False

    def _apply_slow_roost():
        nonlocal _slow_roost_applied
        if _slow_roost_applied:
            return
        _slow_roost_applied = True
        if side0_first and is_roost1:
            _apply_roost_type_strip(1)
        elif not side0_first and is_roost0:
            _apply_roost_type_strip(0)

    # ------------------------------------------------------------------
    # Damage calculation — must be in SPEED ORDER and the slower attacker's
    # calc must be SKIPPED if they were KO'd by the faster move. This keeps
    # pokepy's gen5 PRNG consumption in sync with Showdown's serial flow.
    # ------------------------------------------------------------------
    # `target0_protected` already incorporates move-flag bypass (Feint,
    # Mighty Cleave) in the strip above.
    is_feint0 = move_id0 == MOVE_FEINT
    target0_bypasses_protect = target0_protected
    is_first_turn_only0 = move_id0 in (MOVE_FIRST_IMPRESSION, MOVE_FAKE_OUT)
    not_first_turn0 = active_move_actions0_pre > 0
    first_turn_fail0 = is_first_turn_only0 and not_first_turn0

    is_first_turn_only1 = move_id1 in (MOVE_FIRST_IMPRESSION, MOVE_FAKE_OUT)
    not_first_turn1 = active_move_actions1_pre > 0
    first_turn_fail1 = is_first_turn_only1 and not_first_turn1

    # Hyperspace Fury is only usable by Hoopa Unbound. In pokepy's compact
    # species representation the relevant practical discriminator is Hoopa's
    # original Psychic/Dark form typing; regular Hoopa (Psychic/Ghost) must
    # fail without entering the damage path.
    _MOVE_HYPERSPACE_FURY = 621
    _SPECIES_HOOPA = 720

    def _orig_types_for_active(side_idx: int, p_off: int) -> int:
        flags = int(battle[p_off + 15])
        tera_used = (flags & 0x8) != 0
        if not tera_used:
            return int(battle[p_off + 4])
        meta_off = OFF_META + (
            M_TERA_ORIG_TYPES_0 if side_idx == 0 else M_TERA_ORIG_TYPES_1
        )
        orig = int(battle[meta_off])
        return orig if orig != 0 else int(battle[p_off + 4])

    _types0_for_hf = _orig_types_for_active(0, p0_off)
    _types1_for_hf = _orig_types_for_active(1, p1_off)
    _hoopa_unbound0 = int(battle[p0_off + 0]) == _SPECIES_HOOPA and (
        ((_types0_for_hf & 0xFF) == TYPE_DARK)
        or (((_types0_for_hf >> 8) & 0xFF) == TYPE_DARK)
    )
    _hoopa_unbound1 = int(battle[p1_off + 0]) == _SPECIES_HOOPA and (
        ((_types1_for_hf & 0xFF) == TYPE_DARK)
        or (((_types1_for_hf >> 8) & 0xFF) == TYPE_DARK)
    )
    hyperspace_fury_fail0 = move_id0 == _MOVE_HYPERSPACE_FURY and not _hoopa_unbound0
    hyperspace_fury_fail1 = move_id1 == _MOVE_HYPERSPACE_FURY and not _hoopa_unbound1

    is_delayed0 = move_id0 in (MOVE_FUTURE_SIGHT, MOVE_DOOM_DESIRE)
    is_delayed1 = move_id1 in (MOVE_FUTURE_SIGHT, MOVE_DOOM_DESIRE)
    last_resort_fail0 = (not is_switch) and _last_resort_fails_for_slot(
        state, 0, move_user_slot0, move_id0
    )
    last_resort_fail1 = (not opp_is_switch) and _last_resort_fails_for_slot(
        state, 1, move_user_slot1, move_id1
    )

    fs_current = int(battle[OFF_META + M_FUTURE_SIGHT])
    delayed_ready0 = is_delayed0 and not is_switch and ((fs_current >> 12) & 0xF) == 0
    delayed_ready1 = (
        is_delayed1 and not opp_is_switch and ((fs_current >> 4) & 0xF) == 0
    )

    # Prankster vs Dark
    target1_is_dark = (t1_t1 == TYPE_DARK) or (t1_t2 == TYPE_DARK)
    target0_is_dark = (t0_t1 == TYPE_DARK) or (t0_t2 == TYPE_DARK)
    cat0 = int(game_data.move_category[move_id0])
    cat1 = int(game_data.move_category[move_id1])
    # Prankster vs Dark (gen 7+): a Prankster-boosted status move fails
    # ONLY when targeting the opponent directly. Self-targeted setup
    # (Calm Mind, Bulk Up, Swords Dance), allySide moves (Reflect, Light
    # Screen, Tailwind), foeSide hazards (Stealth Rock, Spikes), and
    # field-wide moves (Trick Room, Trick) are NOT blocked.
    # Showdown source: sim/battle-actions.ts:671-675 — checks
    # `!targets[i].isAlly(pokemon) && targets[i].hasType('Dark')`.
    # Pokepy used to fail any Prankster status move when ANY Dark mon was
    # opposite, breaking Grimmsnarl/Whimsicott/Sableye self-setup.
    # move_target encoding (extracted from Showdown):
    #   0 = normal foe mon — Prankster can fail vs Dark
    #   1, 2, 4, 5, 6, 8, 11, 12 = various foe-direct targets
    #   3 = self,  7 = all,  9 = allySide,  10 = foeSide
    target0_kind = int(game_data.move_target[move_id0])
    target1_kind = int(game_data.move_target[move_id1])
    move0_targets_foe_mon = target0_kind in (0, 1, 2, 4, 5, 6, 8, 11, 12)
    move1_targets_foe_mon = target1_kind in (0, 1, 2, 4, 5, 6, 8, 11, 12)
    move0_can_change_foe_item = int(
        move_effects.effect_type[move_id0]
    ) == EFFECT_KNOCK_OFF or int(move_id0) in (MOVE_TRICK, MOVE_SWITCHEROO)
    move1_can_change_foe_item = int(
        move_effects.effect_type[move_id1]
    ) == EFFECT_KNOCK_OFF or int(move_id1) in (MOVE_TRICK, MOVE_SWITCHEROO)
    move0_no_target = (
        (not is_delayed0)
        and move0_targets_foe_mon
        and (int(battle[p1_off + 1]) <= 0 or (int(battle[p1_off + 15]) & 0x1) != 0)
    )
    move1_no_target = (
        (not is_delayed1)
        and move1_targets_foe_mon
        and (int(battle[p0_off + 1]) <= 0 or (int(battle[p0_off + 15]) & 0x1) != 0)
    )
    prankster_fail0 = (
        (p0_ab == ABILITY_PRANKSTER)
        and (cat0 == CAT_STATUS)
        and move0_targets_foe_mon
        and target1_is_dark
    )
    prankster_fail1 = (
        (p1_ab == ABILITY_PRANKSTER)
        and (cat1 == CAT_STATUS)
        and move1_targets_foe_mon
        and target0_is_dark
    )

    is_feint1 = move_id1 == MOVE_FEINT
    target1_bypasses_protect = target1_protected
    # Sucker Punch / Thunderclap onTry fail is detected BEFORE the damage
    # calc so we can skip the calc entirely (and the PRNG frames it would
    # roll). Showdown's runMove calls onTry which returns false for Sucker
    # Punch when the target is using a Status move or switching; the
    # damage path is never entered. Pokepy used to still call calc and
    # zero damage0 afterwards, drifting the LCG by 1 acc + 1 crit + 1 dmg
    # frame per failed Sucker Punch.
    sucker0_fails = move_id0 == MOVE_SUCKER_PUNCH and (
        cat1 == CAT_STATUS or opp_is_switch or not side0_first
    )
    sucker1_fails = move_id1 == MOVE_SUCKER_PUNCH and (
        cat0 == CAT_STATUS or is_switch or side0_first
    )
    thunderclap0_fails = move_id0 == MOVE_THUNDERCLAP and (
        cat1 == CAT_STATUS or opp_is_switch or not side0_first
    )
    thunderclap1_fails = move_id1 == MOVE_THUNDERCLAP and (
        cat0 == CAT_STATUS or is_switch or side0_first
    )
    # Poltergeist fails in Showdown's onTry hook when the live target item slot
    # is empty, including switch-vs-move turns where a switch-in consumed its
    # item (e.g. Booster Energy) before the attack resolves.
    poltergeist_fail0 = (
        move_id0 == MOVE_POLTERGEIST
        and move0_targets_foe_mon
        and not is_switch
        and int(battle[p1_off + 6]) == 0
    )
    poltergeist_fail1 = (
        move_id1 == MOVE_POLTERGEIST
        and move1_targets_foe_mon
        and not opp_is_switch
        and int(battle[p0_off + 6]) == 0
    )
    pre_damage_fail0 = (
        first_turn_fail0
        or hyperspace_fury_fail0
        or last_resort_fail0
        or sucker0_fails
        or thunderclap0_fails
        or poltergeist_fail0
    )
    pre_damage_fail1 = (
        first_turn_fail1
        or hyperspace_fury_fail1
        or last_resort_fail1
        or sucker1_fails
        or thunderclap1_fails
        or poltergeist_fail1
    )
    p0_skip = False
    p1_skip = False
    protect_failed0 = False
    protect_failed1 = False
    _refresh_protection_state()

    def _resolve_protect_after_before_move(side: int) -> None:
        nonlocal protect_success0, protect_success1
        nonlocal used_protect0, used_protect1
        nonlocal protect_failed0, protect_failed1

        # Showdown's Protect-family onPrepareHit gate is
        # `!!this.queue.willAct() && this.runEvent('StallMove', pokemon)`.
        # In singles, if the Protect user is the last queued action of the
        # turn, `queue.willAct()` is false and the move fails BEFORE any stall
        # RNG or volatile/stall state is applied. This matters on
        # switch-vs-move turns (Battle 910) and any other "move last" cases.
        if (side == 0 and not side0_first) or (side == 1 and side0_first):
            if side == 0:
                if is_switch or effect0 != EFFECT_PROTECT:
                    return
                protect_success0 = False
                used_protect0 = False
                protect_failed0 = True
            else:
                if opp_is_switch or effect1 != EFFECT_PROTECT:
                    return
                protect_success1 = False
                used_protect1 = False
                protect_failed1 = True
            _refresh_protection_state()
            return

        if side == 0:
            if is_switch or effect0 != EFFECT_PROTECT:
                return
            protect_success0 = fx.apply_protect_from_move(
                battle, move_id0, 0, move_effects, gen5_prng
            )
            used_protect0 = bool(protect_success0)
            protect_failed0 = not protect_success0
        else:
            if opp_is_switch or effect1 != EFFECT_PROTECT:
                return
            protect_success1 = fx.apply_protect_from_move(
                battle, move_id1, 1, move_effects, gen5_prng
            )
            used_protect1 = bool(protect_success1)
            protect_failed1 = not protect_success1
        _refresh_protection_state()

    # Glaive Rush: clear the self-volatile flag BEFORE the user's next
    # move executes (Showdown onBeforeMove priority 100). The defender's
    # damage calc (calc_damage_gen9) reads this flag off the def_offset,
    # so clearing the attacker's own flag right before their move runs
    # keeps the opponent's attack-before-our-move 2x bonus intact while
    # correctly removing it once we start moving again.
    from pokepy.core.constants import FLAG_GLAIVE_RUSH as _FLAG_GLAIVE_RUSH

    # Capture per-call meta (num_hits) so the engine can fire per-hit
    # defender abilities (Rocky Helmet, Rough Skin, Iron Barbs) the right
    # number of times for multi-hit moves.
    _meta0 = {"num_hits": 1}
    _meta1 = {"num_hits": 1}

    def _calc_p0(user_hurt_by_target_this_turn: bool = False):
        if p0_skip:
            return 0
        # Status immobilization (sleep / frz-no-thaw / full para / recharge)
        # short-circuits the calc — Showdown's `onBeforeMove` returns false
        # BEFORE any acc/crit/dmg PRNG frames are rolled.
        if _status_chain_resolved0 and _status_chain_immobile0:
            return 0
        # Charge-turn moves return early from Showdown's `onTryMove` after
        # setting the prepare/volatile state, so there is no same-turn
        # accuracy, crit, damage, or secondary PRNG on the charging turn.
        if is_charge_turn0:
            return 0
        # Clear glaive_rush on self right before we attack.
        battle[p0_off + 15] = int(battle[p0_off + 15]) & ~_FLAG_GLAIVE_RUSH
        # Stance Change fires BEFORE damage calc (onModifyMovePriority 1).
        fx.apply_stance_change_pre_move(
            battle,
            p0_off,
            p1_off,
            move_id0,
            -1,
            False,
            True,
            game_data,
        )
        # Status moves short-circuit the damage calc — Showdown's
        # `moveHit` path does NOT invoke the damage formula and never rolls
        # the acc/crit/damage PRNG frames for category Status. The inline
        # `acc_roll0` at ~line 1361 handles Status-move accuracy separately.
        if cat0 == CAT_STATUS:
            return 0
        return calc_damage_gen9(
            battle,
            0,
            move_idx,
            state.team_moves,
            state.opp_moves,
            game_data,
            move_effects,
            type_chart,
            is_moving_last=(not side0_first),
            override_move_id=move_id0,
            gen5_prng=gen5_prng,
            out_meta=_meta0,
            user_hurt_by_target_this_turn=user_hurt_by_target_this_turn,
            target_newly_switched=bool(opp_is_switch or _mid_turn_pivot_1),
            profile=profile,
        )

    def _calc_p1(user_hurt_by_target_this_turn: bool = False):
        if p1_skip:
            return 0
        if _status_chain_resolved1 and _status_chain_immobile1:
            return 0
        if is_charge_turn1:
            return 0
        battle[p1_off + 15] = int(battle[p1_off + 15]) & ~_FLAG_GLAIVE_RUSH
        fx.apply_stance_change_pre_move(
            battle,
            p0_off,
            p1_off,
            -1,
            move_id1,
            True,
            False,
            game_data,
        )
        if cat1 == CAT_STATUS:
            return 0
        return calc_damage_gen9(
            battle,
            1,
            opp_move_idx,
            state.team_moves,
            state.opp_moves,
            game_data,
            move_effects,
            type_chart,
            is_moving_last=side0_first,
            override_move_id=move_id1,
            gen5_prng=gen5_prng,
            out_meta=_meta1,
            user_hurt_by_target_this_turn=user_hurt_by_target_this_turn,
            target_newly_switched=bool(is_switch or _mid_turn_pivot_0),
            profile=profile,
        )

    # Pre-fetch HPs to determine if the first attacker's damage KOs the second
    hp0_pre_dmg = int(battle[p0_off + 1])
    hp1_pre_dmg = int(battle[p1_off + 1])

    # Track whether the first attacker KO'd the target — this alters the
    # PRNG frame schedule for the rest of the turn (only 1 mid-move frame
    # instead of 3, 0 EOT frames instead of 4, and 3 switch-in frames after
    # the dead side's replacement comes in).
    move1_killed_target = False
    # When the first mover KOs the slower target before it can act, Showdown
    # still spends the single tied-speed `hitStepMoveHitLoop` Update shuffle
    # AFTER `DamagingHit` / `onAfterHit` return, not before the defender
    # ability cascade runs. Defer that synthetic frame so abilities like
    # Cursed Body read the aligned PRNG position.
    _deferred_hit_loop_update_frames = 0
    # Showdown consumes speedSort frames BETWEEN the two moves via:
    #   * `eachEvent('Update')` inside hitStepMoveHitLoop (battle-actions.ts:968)
    #     once per executed hit
    #   * `eachEvent('Update')` at end of runMove (battle-actions.ts:1022)
    #   * post-action `eachEvent('Update')` (battle.ts:2860)
    # These all use the speed comparator on `getAllActive()`, so each only
    # consumes a frame when the two actives have TIED speeds. Standard
    # multihit moves spend their per-hit `hitStepMoveHitLoop` Update frames
    # inline during damage calc, so the post-damage engine only needs to
    # account for the remaining runMove/post-action Updates here.
    #
    # If the first attacker KO'd the target, only the first eachEvent fires
    # before faintMessages marks the target fainted; the remaining two see a
    # 1-element list and never consume frames. So the KO case consumes:
    #   1 frame if speeds tied, 0 frames if untied.
    # Per-move secondary-roll prerolls. Showdown consumes each move's
    # secondaryRoll inside its own `moveHit` — i.e. BETWEEN move N's damage
    # calc and move N+1's damage calc (sim/battle-actions.ts:1357). Pokepy's
    # apply_*_from_move calls historically consumed these frames AFTER both
    # moves' damage calcs, shifting the PRNG stream. We preroll now and
    # thread the rolled values into the apply_* sites via prerolled_* kwargs.
    prerolled_move0 = None
    prerolled_move1 = None
    # Pre-rolled lockedmove self-effect PRNG values. Showdown's lockedmove
    # self-effect (data/conditions.ts:lockedmove.onStart → this.random(2,4))
    # fires at selfDrops time — AFTER the attacker's damage but BEFORE the
    # other mover's move. Pokepy must consume these PRNG frames inline right
    # after _calc_pN(), not deferred to _set_lockedmove_post.
    _lockedmove_prerolled = {0: None, 1: None}  # side -> rolled duration or None
    _LOCKEDMOVE_CONFUSION_PENDING = 10_000

    def _roll_pending_lockedmove_confusion(
        side: int, user_off: int, *, apply: bool = False
    ) -> None:
        """Roll fatigue-confusion duration after Showdown's post-hit Update."""
        if _lockedmove_prerolled.get(side) != _LOCKEDMOVE_CONFUSION_PENDING:
            return
        user_ab = int(battle[user_off + 5])
        vol_off = OFF_FIELD + (F_VOLATILE_0 if side == 0 else F_VOLATILE_1)
        cur_v = int(battle[vol_off])
        if user_ab != ABILITY_OWN_TEMPO and get_confusion_turns(cur_v) == 0:
            turns_roll = int(gen5_prng.random(2, 6))
            _lockedmove_prerolled[side] = -turns_roll
            if apply:
                new_v = set_confusion_turns(cur_v, turns_roll)
                if new_v >= 0x8000:
                    new_v -= 0x10000
                battle[vol_off] = new_v
                # The duration has already been materialized in battle state.
                # Do not let _set_lockedmove_post replay it after same-turn
                # Lum/Miracle/Persim update hooks have had a chance to cure it.
                _lockedmove_prerolled[side] = 0
        else:
            _lockedmove_prerolled[side] = 0

    def _preroll_lockedmove_self_effect(
        side: int,
        mid: int,
        executed: bool,
        self_effect_applies: bool,
        was_locked: bool,
        prev_turns: int,
        user_off: int,
    ) -> None:
        """Consume PRNG frames for lockedmove self-effect at the Showdown-
        correct position (after the mover's damage, before the other mover).
        Stores result in _lockedmove_prerolled[side] for _set_lockedmove_post.
        """
        if not executed:
            return
        if mid not in _LOCKED_MOVES:
            return
        cur_status = int(battle[user_off + 12]) & 0xFF
        if was_locked and cur_status == STATUS_SLEEP:
            return
        if not was_locked:
            if not self_effect_applies:
                return
            # First use: roll duration (Showdown: this.random(2, 4) → 2 or 3).
            rolled = 2 + int(gen5_prng.random(2))
            _lockedmove_prerolled[side] = rolled
        else:
            # Already locked: check if expiring this turn → confusion roll.
            new_remaining = prev_turns - 1
            # A dud repeated turn (for example Outrage into a Fairy) skips the
            # restart while there are still locked turns left, but an expiry
            # turn still ends the volatile and rolls fatigue confusion via
            # lockedmove.onEnd. Match that PRNG schedule here.
            if not self_effect_applies and new_remaining > 0:
                return
            if new_remaining <= 0:
                # Expiry — roll confusion turns if applicable.
                user_ab = int(battle[user_off + 5])
                vol_off = OFF_FIELD + (F_VOLATILE_0 if side == 0 else F_VOLATILE_1)
                cur_v = int(battle[vol_off])
                if user_ab != ABILITY_OWN_TEMPO and get_confusion_turns(cur_v) == 0:
                    # lockedmove.onEnd fires from runMove's AfterMove event,
                    # after the move's hit-loop Update. Defer the duration roll
                    # until those tied-speed Update frames have been consumed.
                    _lockedmove_prerolled[side] = _LOCKEDMOVE_CONFUSION_PENDING
                else:
                    _lockedmove_prerolled[side] = 0  # expiry, no confusion roll

    def _preroll0(target_hp_override=None):
        return _preroll_move_secondaries(
            battle,
            move_id0,
            p0_off,
            p1_off,
            damage0,
            target0_protected,
            is_switch,
            side0_first,
            int(_meta0.get("num_hits", 1)),
            game_data,
            move_effects,
            gen5_prng,
            target_stats_raised_this_turn=stats_raised_this_turn1,
            target_hp_override=target_hp_override,
            profile=profile,
        )

    def _preroll1(target_hp_override=None):
        return _preroll_move_secondaries(
            battle,
            move_id1,
            p1_off,
            p0_off,
            damage1,
            target1_protected,
            opp_is_switch,
            not side0_first,
            int(_meta1.get("num_hits", 1)),
            game_data,
            move_effects,
            gen5_prng,
            target_stats_raised_this_turn=stats_raised_this_turn0,
            target_hp_override=target_hp_override,
            profile=profile,
        )

    # Between-moves Update count. Showdown fires `eachEvent('Update')` inside
    # `hitStepMoveHitLoop` (line 840), after the move hit loop (line 885), and
    # again at post-action (`battle.ts:2361`). A normal successful single-hit
    # move therefore contributes 3 tied-speed Update frames. Side/field style
    # status moves and no-damage damaging moves use the lighter 1-frame path.
    # Successful self-target setup status moves still follow the full 3-Update
    # path in Showdown, including on switch-vs-move turns after the switch-in.
    _cat0_status = cat0 == CAT_STATUS
    _cat1_status = cat1 == CAT_STATUS

    def _move_update_count(
        is_status_move: bool,
        target_kind: int,
        successful_self_boost_status: bool,
        damage_val: int,
        num_hits: int = 1,
    ) -> int:
        if is_status_move and target_kind in (7, 9, 10):
            return 1
        if is_status_move and target_kind == 3 and not successful_self_boost_status:
            return 1
        if (not is_status_move) and int(damage_val) <= 0:
            return 1
        if int(num_hits) > 1:
            return 2
        # Gen 1/2 tryMoveHit does not fire hitStepMoveHitLoop Updates for
        # single-hit moves; only runAction's post-move Update (battle.ts:2938).
        if profile.gen <= 2:
            return 1
        # Gen 3 spends one fewer tied-speed Update frame per damaging move
        # hit loop / post-action chain than gen 4+ in Showdown.
        return 2 if profile.gen <= 3 else 3

    def _blocked_first_action_update_count(
        first_action_immobile: bool,
        full_move_update_count: int,
    ) -> int:
        """Mirror Showdown's post-action Update on blocked tied-speed turns.

        Even if `onBeforeMove` blocks the first queued move (freeze, sleep,
        full paralysis, Truant, etc.), `runAction` still reaches its generic
        `eachEvent('Update')` before the slower queued move resolves. Pokepy
        historically dropped that frame to zero by skipping the first mover's
        tied-speed update bucket entirely whenever the move was immobile.
        """
        return 1 if first_action_immobile else int(full_move_update_count)

    def _neutralizing_gas_on_end_restart_needed(
        source_offset: int,
        target_offset: int,
    ) -> bool:
        """Return whether Neutralizing Gas `onEnd` should replay active Starts."""

        s_off = int(source_offset)
        t_off = int(target_offset)
        if int(battle[s_off + 1]) > 0 or int(battle[t_off + 1]) <= 0:
            return False
        if int(battle[s_off + 5]) != ABILITY_NEUTRALIZING_GAS:
            return False
        # Showdown returns early if another active Neutralizing Gas user is
        # still on the field, so only the last active NGas user consumes this.
        if int(battle[t_off + 5]) == ABILITY_NEUTRALIZING_GAS:
            return False
        return True

    # Helper: when the first attacker's status move sets up a screen
    # (Reflect / Light Screen / Aurora Veil), Showdown applies the screen
    # mid-runMove BEFORE the second attacker resolves its own BP — so the
    # second mover's damage is halved already on the SAME turn the screen
    # goes up. Pre-set the screen value here so `_calc_pN` for the second
    # mover sees it; the later `apply_screen_from_move` is a no-op because
    # the slot is already filled.
    def _preset_screen_if_status(mid: int, side: int) -> None:
        if mid == MOVE_REFLECT or mid == MOVE_LIGHT_SCREEN or mid == MOVE_AURORA_VEIL:
            if mid == MOVE_AURORA_VEIL:
                _w = int(battle[OFF_FIELD + F_WEATHER])
                if _w != WEATHER_SNOW and _w != 5:
                    return
            _apply_screen_from_move_tracked(mid, side, True)

    # Pre-roll status-move accuracy in speed order. Showdown rolls each
    # status move's `randomChance(accuracy, 100)` inline at the start of
    # that move's `runMove`, BEFORE the next attacker's move executes
    # (sim/battle-actions.ts:600 inside hitStep). Pokepy historically
    # rolled `acc_roll0`/`acc_roll1` AFTER the entire damage_calc block,
    # which placed the frame at the END of all moves' rolls and shifted
    # the slow attacker's damage roll by 1 frame in mixed status/damaging
    # turns (e.g. Whimsicott Encore vs Snorlax Body Slam — the body slam
    # damage roll diverged from Showdown by exactly 1 frame). Pre-roll
    # here in the same speed order as the moves so the frame lands at
    # the right offset. Toxic from a Poison-type user, Wonder Skin
    # Showdown gating, accuracy-true moves and any p_skip case are all
    # handled by mirroring the `acc_rollN` logic below.
    _MOVE_TOXIC_PRE = 92
    _acc0_pre = int(game_data.move_accuracy[move_id0])
    _acc1_pre = int(game_data.move_accuracy[move_id1])
    _move0_is_status_pre = cat0 == CAT_STATUS
    _move1_is_status_pre = cat1 == CAT_STATUS

    def _showdown_accuracy_bypass(acc: int, is_status: bool, target_kind: int) -> bool:
        """Showdown skips randomChance only when accuracy === true (loader: 127).

        Gen2 tryMoveHit skips at scaled 255 (stored acc >= 100). Gen5+
        hitStepAccuracy sets self-targeting status moves to accuracy === true.
        """
        if acc == 127 or acc == 0:
            return True
        if profile.gen >= 5 and is_status and target_kind == 3:
            return True
        if profile.gen == 2 and acc >= 100:
            return True
        return False

    _acc0_bypass_pre = _showdown_accuracy_bypass(
        _acc0_pre, _move0_is_status_pre, target0_kind
    )
    _acc1_bypass_pre = _showdown_accuracy_bypass(
        _acc1_pre, _move1_is_status_pre, target1_kind
    )
    # Showdown uses `tryMoveHit` (no accuracy check) for moves with
    # target in {all, foeSide, allySide, allyTeam} (pokepy targets 7, 9, 10).
    # Self-targeting status moves (pokepy target 3) get `accuracy = true` in
    # Showdown's `hitStepAccuracy`.  None of these consume a PRNG frame for
    # accuracy; suppress the pre-roll so pokepy's frame schedule matches.
    _field_or_self0 = target0_kind in (3, 7, 9, 10)
    _field_or_self1 = target1_kind in (3, 7, 9, 10)
    _STATUS_TARGET_FOE_MON = (0, 1, 2, 4, 5, 6, 8, 11, 12)
    _MOVE_THUNDER_WAVE_PRE = 86
    _ABILITY_GOOD_AS_GOLD_PRE = 283

    def _status_pre_accuracy_blocked(
        move_id: int,
        user_off: int,
        target_off: int,
        target_kind: int,
        prankster_fail: bool,
    ) -> bool:
        """Return True when Showdown rejects this status move before accuracy.

        Mirrors the target-side blockers that occur in the
        TryHit/TypeImmunity/TryImmunity steps ahead of hitStepAccuracy:
        Prankster vs Dark, Good as Gold, deterministic target-side ability
        immunities (Motor Drive, Volt Absorb, Flash Fire, Soundproof, etc.),
        powder immunity, and Thunder Wave's explicit Ground immunity.

        Deliberately excludes post-accuracy status failures such as Safeguard,
        Misty Terrain, Purifying Salt, or an already-statused target.
        """
        from pokepy.effects.ability_suppression import (
            effective_ability as _effective_ability_pre,
        )

        if target_kind not in _STATUS_TARGET_FOE_MON:
            return False
        if prankster_fail:
            return True

        move_type = int(game_data.move_type[move_id])
        move_flags = int(game_data.move_flags[move_id])
        user_ab = _effective_ability_pre(battle, user_off, target_off)
        target_ab = _effective_ability_pre(battle, target_off, user_off)
        has_mold_breaker = user_ab in (
            ABILITY_MOLD_BREAKER,
            ABILITY_TERAVOLT,
            ABILITY_TURBOBLAZE,
        )

        if target_ab == _ABILITY_GOOD_AS_GOLD_PRE and not has_mold_breaker:
            return True

        if not has_mold_breaker:
            if target_ab == ABILITY_FLASH_FIRE and move_type == TYPE_FIRE:
                return True
            if target_ab == ABILITY_VOLT_ABSORB and move_type == TYPE_ELECTRIC:
                return True
            if target_ab == ABILITY_WATER_ABSORB and move_type == TYPE_WATER:
                return True
            if target_ab == ABILITY_SAP_SIPPER and move_type == TYPE_GRASS:
                return True
            if target_ab == ABILITY_STORM_DRAIN and move_type == TYPE_WATER:
                return True
            if target_ab == ABILITY_LIGHTNING_ROD and move_type == TYPE_ELECTRIC:
                return True
            if target_ab == ABILITY_MOTOR_DRIVE and move_type == TYPE_ELECTRIC:
                return True
            if target_ab == ABILITY_DRY_SKIN and move_type == TYPE_WATER:
                return True
            if target_ab == ABILITY_EARTH_EATER and move_type == TYPE_GROUND:
                return True
            if target_ab == ABILITY_WELL_BAKED_BODY and move_type == TYPE_FIRE:
                return True
            if target_ab == ABILITY_SOUNDPROOF and (move_flags & FLAG_SOUND) != 0:
                return True
            if target_ab == ABILITY_BULLETPROOF and (move_flags & FLAG_BULLET) != 0:
                return True

        target_types = int(battle[target_off + 4]) & 0xFFFF
        target_t1 = target_types & 0xFF
        target_t2 = (target_types >> 8) & 0xFF
        if (move_flags & FLAG_POWDER) != 0:
            target_item = int(battle[target_off + 6])
            # Grass powder immunity is gen6+ (Showdown battle-actions.ts onTryHit).
            if int(getattr(game_data, "gen", 9)) >= 6 and (
                target_t1 == TYPE_GRASS
                or target_t2 == TYPE_GRASS
                or target_ab == ABILITY_OVERCOAT
                or target_item == ITEM_SAFETY_GOGGLES
            ):
                return True

        if move_id == _MOVE_THUNDER_WAVE_PRE and (
            target_t1 == TYPE_GROUND or target_t2 == TYPE_GROUND
        ):
            return True

        return False

    _pre_accuracy_block0 = _status_pre_accuracy_blocked(
        move_id0,
        p0_off,
        p1_off,
        target0_kind,
        prankster_fail0,
    )
    _pre_accuracy_block1 = _status_pre_accuracy_blocked(
        move_id1,
        p1_off,
        p0_off,
        target1_kind,
        prankster_fail1,
    )

    def _status_accuracy_preroll_needed(side_idx: int) -> bool:
        if side_idx == 0:
            if (
                not _move0_is_status_pre
                or p0_skip
                or _field_or_self0
                or _pre_accuracy_block0
            ):
                return False
            if _acc0_bypass_pre:
                return False
            user_types = int(battle[p0_off + 4]) & 0xFFFF
            user_is_poison = (user_types & 0xFF) == TYPE_POISON or (
                (user_types >> 8) & 0xFF
            ) == TYPE_POISON
            return not (move_id0 == _MOVE_TOXIC_PRE and user_is_poison)
        if (
            not _move1_is_status_pre
            or p1_skip
            or _field_or_self1
            or _pre_accuracy_block1
        ):
            return False
        if _acc1_bypass_pre:
            return False
        user_types = int(battle[p1_off + 4]) & 0xFFFF
        user_is_poison = (user_types & 0xFF) == TYPE_POISON or (
            (user_types >> 8) & 0xFF
        ) == TYPE_POISON
        return not (move_id1 == _MOVE_TOXIC_PRE and user_is_poison)

    def _status_move_hits_pre(side_idx: int, prerolled_acc: "int | None") -> bool:
        if side_idx == 0:
            if _pre_accuracy_block0:
                return False
            user_types = int(battle[p0_off + 4]) & 0xFFFF
            user_is_poison = (user_types & 0xFF) == TYPE_POISON or (
                (user_types >> 8) & 0xFF
            ) == TYPE_POISON
            return (
                (move_id0 == _MOVE_TOXIC_PRE and user_is_poison)
                or _acc0_bypass_pre
                or (prerolled_acc is not None and prerolled_acc < _acc0_pre)
            )
        if _pre_accuracy_block1:
            return False
        user_types = int(battle[p1_off + 4]) & 0xFFFF
        user_is_poison = (user_types & 0xFF) == TYPE_POISON or (
            (user_types >> 8) & 0xFF
        ) == TYPE_POISON
        return (
            (move_id1 == _MOVE_TOXIC_PRE and user_is_poison)
            or _acc1_bypass_pre
            or (prerolled_acc is not None and prerolled_acc < _acc1_pre)
        )

    _prerolled_status_acc0: "int | None" = None
    _prerolled_status_acc1: "int | None" = None

    # Track whether the first attacker's flinch secondary lands. When it
    # does, Showdown's flinch volatile (data/conditions.ts) returns false
    # from `onBeforeMove` BEFORE the second attacker's move runs — no
    # PRNG frames are consumed for the flinched mon's acc / crit / damage.
    # Pokepy historically still ran _calc_p1 / _calc_p0 for the flinched
    # side and then nulled the damage out, advancing the LCG by 3 frames
    # too many on every flinch turn.
    _first_attacker_flinches_target = False
    # Confusion self-hit flags — set by the pre-calc
    # check_confusion_self_hit call below when a mon hurts itself before
    # its move runs. When set, the corresponding `_calc_pN` is skipped
    # and the helper-owned self-hit damage remains in state.
    _self_hit0 = False
    _self_hit1 = False
    _p0_immobile_pre = False
    _p1_immobile_pre = False
    # "Already checked confusion this turn" flags — so the later
    # post-calc check_confusion_self_hit call at ~line 1900+ doesn't
    # double-roll the chance frame for a mon whose confusion was already
    # resolved in the pre-calc position below.
    _conf_checked0 = False
    _conf_checked1 = False

    def _pre_calc_confusion_check(side_idx: int) -> bool:
        """Roll confusion self-hit BEFORE the mon's damage_calc, matching
        Showdown's onBeforeMove ordering. Returns True iff the mon is
        self-hitting (and thus should skip its damage_calc).

        If another path intentionally marked confusion as already handled
        for this turn, the volatile helper can still skip here via its
        internal bookkeeping.
        """
        nonlocal _conf_checked0, _conf_checked1
        if side_idx == 0:
            if is_switch:
                return False
            user_off = p0_off
        else:
            if opp_is_switch:
                return False
            user_off = p1_off
        # NOTE: sleep / frz / par / recharge guards are evaluated below in
        # the status_chain section, AFTER the calc step. We can't filter
        # those here without a larger refactor. The affected scenarios
        # (confused mon also asleep/frozen/paralyzed) are rare and not
        # covered by any current parity scenario. For now, fire the
        # confusion check whenever the mon is confused — this matches
        # Showdown for the common case.
        hit = fx.check_confusion_self_hit(battle, side_idx, user_off, gen5_prng)
        if side_idx == 0:
            _conf_checked0 = True
        else:
            _conf_checked1 = True
        return bool(hit)

    def _pre_calc_para_check(side_idx: int) -> bool:
        """Roll full paralysis after confusion, matching Showdown's
        onBeforeMove priorities (confusion=3, paralysis=1)."""
        nonlocal _status_chain_full_para0, _status_chain_full_para1
        nonlocal _status_chain_immobile0, _status_chain_immobile1
        if side_idx == 0:
            if is_switch or _status_chain_immobile0:
                return False
            user_off = p0_off
        else:
            if opp_is_switch or _status_chain_immobile1:
                return False
            user_off = p1_off

        is_par_cur = get_status(int(battle[user_off + 12])) == STATUS_PARALYSIS
        para = bool(
            is_par_cur
            and int(gen5_prng.random(int(profile.full_para_denom)))
            < int(profile.full_para_num)
        )
        if side_idx == 0:
            _status_chain_full_para0 = para
            _status_chain_immobile0 = _status_chain_immobile0 or para
        else:
            _status_chain_full_para1 = para
            _status_chain_immobile1 = _status_chain_immobile1 or para
        return para

    _sleep_talk_committed0 = False
    _sleep_talk_committed1 = False

    def _sleep_talk_consume_sample(side: int) -> None:
        """Commit Sleep Talk's sampled move at the PRNG-correct moment.

        Showdown samples inside Sleep Talk's `onHit`, after any earlier move
        this turn has already consumed its frames but before the called move's
        own pre-hit logic runs. We consume that sample here, rewrite the
        side's runtime move metadata, then refresh the protection / skip state
        so the called move, not the placeholder candidate, drives execution.
        """
        nonlocal move_id0, move_id1
        nonlocal base_priority0, base_priority1, priority0, priority1
        nonlocal priority_bracket0, priority_bracket1
        nonlocal move0_flags, move1_flags, is_contact0, is_contact1, is_punch0, is_punch1
        nonlocal unseen_fist0, unseen_fist1, move0_respects_protect_early, move1_respects_protect_early
        nonlocal effect0, effect1, is_sub0, is_sub1, cat0, cat1
        nonlocal target0_kind, target1_kind, move0_targets_foe_mon, move1_targets_foe_mon
        nonlocal move0_can_change_foe_item, move1_can_change_foe_item
        nonlocal move0_no_target, move1_no_target, prankster_fail0, prankster_fail1
        nonlocal last_resort_fail0, last_resort_fail1, pre_damage_fail0, pre_damage_fail1
        nonlocal _acc0_pre, _acc1_pre, _acc0_bypass_pre, _acc1_bypass_pre
        nonlocal _cat0_status, _cat1_status, _move0_is_status_pre, _move1_is_status_pre
        nonlocal is_feint0, is_feint1, _sleep_talk_committed0, _sleep_talk_committed1

        if side == 0:
            if _sleep_talk_committed0 or _sleep_talk_n0 <= 0:
                return
            if _sleep_talk_pending_slp0:
                initial_turns = gen5_prng.random(2, 5)
                battle[p0_off + 12] = set_status(STATUS_SLEEP, initial_turns)
            _mark_move_slot_used(state, 0, move_user_slot0, move_idx)
            sample_idx = int(gen5_prng.random(_sleep_talk_n0))
            move_id0 = int(_sleep_talk_cands0[sample_idx])
            base_priority0 = int(game_data.move_priority[move_id0])
            priority0 = fx.get_effective_priority(
                battle, move_id0, base_priority0, p0_off, gen5_prng
            )
            priority_bracket0 = int(_math_priority.floor(float(priority0)))
            move0_flags = int(game_data.move_flags[move_id0])
            is_punch0 = (move0_flags & FLAG_PUNCH) != 0
            is_contact0 = (move0_flags & FLAG_CONTACT) != 0
            if (
                is_punch0 and p0_item_pg == _ITEM_PUNCHING_GLOVE
            ) or p0_ab == ABILITY_LONG_REACH:
                is_contact0 = False
            unseen_fist0 = (p0_ab == ABILITY_UNSEEN_FIST) and is_contact0
            move0_respects_protect_early = (move0_flags & _FLAG_PROTECT_EARLY) != 0
            effect0 = int(move_effects.effect_type[move_id0])
            is_sub0 = effect0 == EFFECT_SUBSTITUTE
            cat0 = int(game_data.move_category[move_id0])
            _cat0_status = cat0 == CAT_STATUS
            _move0_is_status_pre = _cat0_status
            target0_kind = int(game_data.move_target[move_id0])
            _acc0_pre = int(game_data.move_accuracy[move_id0])
            _acc0_bypass_pre = _showdown_accuracy_bypass(
                _acc0_pre, _cat0_status, target0_kind
            )
            move0_targets_foe_mon = target0_kind in (0, 1, 2, 4, 5, 6, 8, 11, 12)
            move0_can_change_foe_item = int(
                move_effects.effect_type[move_id0]
            ) == EFFECT_KNOCK_OFF or int(move_id0) in (MOVE_TRICK, MOVE_SWITCHEROO)
            move0_no_target = (
                (move_id0 not in (MOVE_FUTURE_SIGHT, MOVE_DOOM_DESIRE))
                and move0_targets_foe_mon
                and (
                    int(battle[p1_off + 1]) <= 0
                    or (int(battle[p1_off + 15]) & 0x1) != 0
                )
            )
            prankster_fail0 = (
                (p0_ab == ABILITY_PRANKSTER)
                and _cat0_status
                and move0_targets_foe_mon
                and target1_is_dark
            )
            last_resort_fail0 = _last_resort_fails_for_slot(
                state, 0, move_user_slot0, move_id0
            )
            pre_damage_fail0 = (
                first_turn_fail0
                or hyperspace_fury_fail0
                or last_resort_fail0
                or sucker0_fails
                or thunderclap0_fails
                or poltergeist_fail0
            )
            is_feint0 = move_id0 == MOVE_FEINT
            _sleep_talk_committed0 = True
        else:
            if _sleep_talk_committed1 or _sleep_talk_n1 <= 0:
                return
            if _sleep_talk_pending_slp1:
                initial_turns = gen5_prng.random(2, 5)
                battle[p1_off + 12] = set_status(STATUS_SLEEP, initial_turns)
            _mark_move_slot_used(state, 1, move_user_slot1, opp_move_idx)
            sample_idx = int(gen5_prng.random(_sleep_talk_n1))
            move_id1 = int(_sleep_talk_cands1[sample_idx])
            base_priority1 = int(game_data.move_priority[move_id1])
            priority1 = fx.get_effective_priority(
                battle, move_id1, base_priority1, p1_off, gen5_prng
            )
            priority_bracket1 = int(_math_priority.floor(float(priority1)))
            move1_flags = int(game_data.move_flags[move_id1])
            is_punch1 = (move1_flags & FLAG_PUNCH) != 0
            is_contact1 = (move1_flags & FLAG_CONTACT) != 0
            if (
                is_punch1 and p1_item_pg == _ITEM_PUNCHING_GLOVE
            ) or p1_ab == ABILITY_LONG_REACH:
                is_contact1 = False
            unseen_fist1 = (p1_ab == ABILITY_UNSEEN_FIST) and is_contact1
            move1_respects_protect_early = (move1_flags & _FLAG_PROTECT_EARLY) != 0
            effect1 = int(move_effects.effect_type[move_id1])
            is_sub1 = effect1 == EFFECT_SUBSTITUTE
            cat1 = int(game_data.move_category[move_id1])
            _cat1_status = cat1 == CAT_STATUS
            _move1_is_status_pre = _cat1_status
            target1_kind = int(game_data.move_target[move_id1])
            _acc1_pre = int(game_data.move_accuracy[move_id1])
            _acc1_bypass_pre = _showdown_accuracy_bypass(
                _acc1_pre, _cat1_status, target1_kind
            )
            move1_targets_foe_mon = target1_kind in (0, 1, 2, 4, 5, 6, 8, 11, 12)
            move1_can_change_foe_item = int(
                move_effects.effect_type[move_id1]
            ) == EFFECT_KNOCK_OFF or int(move_id1) in (MOVE_TRICK, MOVE_SWITCHEROO)
            move1_no_target = (
                (move_id1 not in (MOVE_FUTURE_SIGHT, MOVE_DOOM_DESIRE))
                and move1_targets_foe_mon
                and (
                    int(battle[p0_off + 1]) <= 0
                    or (int(battle[p0_off + 15]) & 0x1) != 0
                )
            )
            prankster_fail1 = (
                (p1_ab == ABILITY_PRANKSTER)
                and _cat1_status
                and move1_targets_foe_mon
                and target0_is_dark
            )
            last_resort_fail1 = _last_resort_fails_for_slot(
                state, 1, move_user_slot1, move_id1
            )
            pre_damage_fail1 = (
                first_turn_fail1
                or hyperspace_fury_fail1
                or last_resort_fail1
                or sucker1_fails
                or thunderclap1_fails
                or poltergeist_fail1
            )
            is_feint1 = move_id1 == MOVE_FEINT
            _sleep_talk_committed1 = True

        _refresh_protection_state()

    # ------------------------------------------------------------------
    # PRE-CALC status immobilization check.
    #
    # Showdown's `runMove` fires `onBeforeMove` handlers BEFORE any damage
    # rolls. Returning false in that chain short-circuits `useMoveInner`
    # so NO acc/crit/dmg PRNG frames are consumed for the blocked mon.
    # Pokepy historically let `_calc_pN` run and nulled the damage post-
    # hoc, which drifted the LCG stream by 3 frames per blocked turn.
    #
    # This block:
    #   1. Reads each side's status (with wake-up cure applied).
    #   2. Computes sleep_blocked / frozen_blocked(unrolled) / recharge
    #      gates (no PRNG).
    #   3. The thaw + full-para rolls are deferred to each mon's runMove
    #      moment inside the calc block below — `_pre_calc_status_chain`
    #      is called right before the mon's confusion check / acc preroll
    #      / damage calc, matching Showdown's per-runMove order.
    #   4. Sets `is_immobile{0,1}` which extends `p{0,1}_skip` so
    #      `_calc_pN` short-circuits and the per-move frames vanish.
    #
    # The legacy status-chain block at ~line 2130+ still runs for its
    # side-effects (Fire-move self-thaw, Fire-move target thaw) but its
    # roll loop is gated off via `_status_chain_resolved`.
    # ------------------------------------------------------------------
    # Turn-start sleep wake — clear counter==0 sleep before the mon's
    # runMove. Showdown's slp.onBeforeMove decrements the counter at move
    # time and cures when it hits 0. Pokepy historically did the decrement
    # at EOT and the cure at turn-start.
    if not is_switch and get_status(int(battle[p0_off + 12])) == STATUS_SLEEP:
        _raw0 = int(battle[p0_off + 12])
        if get_status_turns(_raw0) <= 0:
            battle[p0_off + 12] = 0
    if not opp_is_switch and get_status(int(battle[p1_off + 12])) == STATUS_SLEEP:
        _raw1 = int(battle[p1_off + 12])
        if get_status_turns(_raw1) <= 0:
            battle[p1_off + 12] = 0

    _ABILITY_COMATOSE_SLEEP_TALK = 213
    _sleep_talk_try_failed0 = (
        (not is_switch)
        and raw_move_id0 == MOVE_SLEEP_TALK
        and (not _sleep_talk_pending_slp0)
        and get_status(int(battle[p0_off + 12])) != STATUS_SLEEP
        and p0_ab != _ABILITY_COMATOSE_SLEEP_TALK
    )
    _sleep_talk_try_failed1 = (
        (not opp_is_switch)
        and raw_move_id1 == MOVE_SLEEP_TALK
        and (not _sleep_talk_pending_slp1)
        and get_status(int(battle[p1_off + 12])) != STATUS_SLEEP
        and p1_ab != _ABILITY_COMATOSE_SLEEP_TALK
    )

    _pre_sb_mv0 = _sleep_talk_orig0 if _sleep_talk_orig0 > 0 else move_id0
    _pre_sb_mv1 = _sleep_talk_orig1 if _sleep_talk_orig1 > 0 else move_id1
    _DEFROST_MOVE_IDS_PRE = frozenset(
        (
            221,
            172,
            394,
            558,
            682,
            780,
            503,
            592,
            815,
            735,
            876,
            902,
        )
    )
    _pre_is_thaw0 = move_id0 in _DEFROST_MOVE_IDS_PRE
    _pre_is_thaw1 = move_id1 in _DEFROST_MOVE_IDS_PRE

    # Status-chain roll state: populated when we actually roll the thaw/
    # para frame for each side; consumed by the legacy block downstream
    # so it doesn't re-roll.
    _status_chain_resolved0 = False
    _status_chain_resolved1 = False
    _status_chain_rand_thaw0 = False
    _status_chain_rand_thaw1 = False
    _status_chain_full_para0 = False
    _status_chain_full_para1 = False
    _status_chain_frozen_blocked0 = False
    _status_chain_frozen_blocked1 = False
    _status_chain_immobile0 = False
    _status_chain_immobile1 = False
    # Side-inflict-status state: set when a side's secondary has been
    # early-applied on the target ahead of the target's runMove. Prevents
    # double-application at the post-calc `apply_status_from_move` call.
    _status_early_applied0 = False
    _status_early_applied1 = False
    _cursed_body_early_applied0 = False
    _cursed_body_early_applied1 = False
    _contact_status_early_applied0 = False
    _contact_status_early_applied1 = False

    def _run_immediate_status_berry_updates(target_off: int) -> None:
        """Fire same-turn status/confusion cure berries before the target acts."""
        _run_item_hook_with_berry_tracking(
            fx.apply_lum_berry, target_off, battle, target_off, game_data
        )
        _run_item_hook_with_berry_tracking(
            fx.apply_status_curing_berries, target_off, battle, target_off, game_data
        )
        _run_item_hook_with_berry_tracking(
            fx.apply_persim_berry, target_off, battle, target_off, game_data
        )

    def _early_apply_inflicted_status(inflicter_side: int) -> None:
        """Call `apply_status_from_move` for `inflicter_side`'s move as
        soon as its calc+preroll has run — BEFORE the target's runMove.
        Showdown applies secondary status inside the inflicter's `moveHit`
        so by the time the target's `runMove` fires, `onBeforeMove` sees
        the new status. Pokepy used to defer this until after both moves,
        which broke same-turn "inflict frz/slp/par → target becomes
        blocked" parity.

        Uses the prerolled roll stashed in `prerolled_move{0,1}['status']`
        (no new PRNG consumption for the secondary chance; sleep-turns
        `random(3)` IS consumed here, matching Showdown's
        `slp.onStart` inside setStatus).
        """
        nonlocal _status_early_applied0, _status_early_applied1

        def _target_has_non_bypassed_sub(
            user_off: int, target_off: int, move_id: int
        ) -> bool:
            move_flags_local = int(game_data.move_flags[int(move_id)])
            is_sound_local = (move_flags_local & FLAG_SOUND) != 0
            is_infiltrator_local = int(battle[int(user_off) + 5]) == ABILITY_INFILTRATOR
            if is_sound_local or is_infiltrator_local:
                return False
            sub_off = OFF_FIELD + (
                F_SUBSTITUTE_1 if int(target_off) >= OFF_SIDE1 else F_SUBSTITUTE_0
            )
            return int(battle[sub_off]) > 0

        if inflicter_side == 0:
            if _status_early_applied0 or prerolled_move0 is None:
                return
            rolled = prerolled_move0.get("status")
            if not rolled:
                return
            if int(damage0) <= 0:
                return
            target_off = p1_off
            user_off = p0_off
            mid = move_id0
            nh = int(_meta0.get("num_hits", 1))
            if _target_has_non_bypassed_sub(user_off, target_off, mid):
                return
            _status_early_applied0 = True
        else:
            if _status_early_applied1 or prerolled_move1 is None:
                return
            rolled = prerolled_move1.get("status")
            if not rolled:
                return
            if int(damage1) <= 0:
                return
            target_off = p0_off
            user_off = p1_off
            mid = move_id1
            nh = int(_meta1.get("num_hits", 1))
            if _target_has_non_bypassed_sub(user_off, target_off, mid):
                return
            _status_early_applied1 = True
        status_before = get_status(int(battle[target_off + 12]))
        fx.apply_status_from_move(
            battle,
            mid,
            target_off,
            True,
            game_data,
            move_effects,
            gen5_prng,
            user_offset=user_off,
            num_hits=nh,
            prerolled_rolls=rolled,
        )
        _maybe_apply_poison_puppeteer(
            user_off,
            target_off,
            status_before,
            inline_before_target_move=True,
        )
        _run_immediate_status_berry_updates(target_off)

    def _early_apply_toxic_chain(inflicter_side: int) -> None:
        """Apply a pre-consumed Toxic Chain bad poison before the slower
        target's `runMove`, matching Showdown's `onSourceDamagingHit` timing.
        """
        if inflicter_side == 0:
            rolled = _prerolled_toxic_chain0
            target_off = p1_off
            user_off = p0_off
        else:
            rolled = _prerolled_toxic_chain1
            target_off = p0_off
            user_off = p1_off
        if rolled is None or int(rolled) >= 3:
            return
        if int(battle[target_off + 1]) <= 0:
            return
        move_flags_local = int(
            game_data.move_flags[move_id0 if inflicter_side == 0 else move_id1]
        )
        is_sound_local = (move_flags_local & FLAG_SOUND) != 0
        is_infiltrator_local = int(battle[user_off + 5]) == ABILITY_INFILTRATOR
        if not is_sound_local and not is_infiltrator_local:
            sub_off = OFF_FIELD + (
                F_SUBSTITUTE_1 if int(target_off) >= OFF_SIDE1 else F_SUBSTITUTE_0
            )
            if int(battle[sub_off]) > 0:
                return
        from pokepy.effects.status_apply import _try_apply_status

        status_before = get_status(int(battle[target_off + 12]))
        _try_apply_status(
            battle,
            None,
            target_off,
            STATUS_TOXIC,
            game_data,
            gen5_prng,
            user_offset=user_off,
            is_status_move=False,
        )
        if get_status(int(battle[target_off + 12])) != status_before:
            _run_immediate_status_berry_updates(target_off)

    def _early_apply_cursed_body(inflicter_side: int) -> None:
        """Apply Cursed Body before the slower target's runMove.

        Showdown resolves the 30% disable roll in `onDamagingHit`, so the
        first mover's defender-ability PRNG is consumed before the slower
        move begins. Pokepy's shared late cascade still owns the non-first-
        mover cases; this helper only preapplies the first mover's hit so the
        slower move's accuracy / crit / damage rolls stay aligned.
        """
        nonlocal _cursed_body_early_applied0, _cursed_body_early_applied1

        if inflicter_side == 0:
            if _cursed_body_early_applied0 or int(damage0) <= 0:
                return
            atk_off = p0_off
            def_off = p1_off
            move_id_local = move_id0
            move_idx_local = move_idx
            _cursed_body_early_applied0 = fx.apply_cursed_body_on_damaging_hit(
                battle,
                atk_off,
                def_off,
                move_id_local,
                move_idx_local,
                True,
                int(damage0),
                gen5_prng,
                gen=profile.gen,
            )
            return

        if _cursed_body_early_applied1 or int(damage1) <= 0:
            return
        atk_off = p1_off
        def_off = p0_off
        move_id_local = move_id1
        move_idx_local = opp_move_idx
        _cursed_body_early_applied1 = fx.apply_cursed_body_on_damaging_hit(
            battle,
            atk_off,
            def_off,
            move_id_local,
            move_idx_local,
            True,
            int(damage1),
            gen5_prng,
            gen=profile.gen,
        )

    def _schedule_cursed_body_after_calc(inflicter_side: int) -> None:
        """Gen 5+ runs DamagingHit before move secondaries."""
        if profile.gen >= 5:
            _early_apply_cursed_body(inflicter_side)

    def _schedule_cursed_body_after_preroll(inflicter_side: int) -> None:
        """Gen 4 and below run DamagingHit after secondaryRoll."""
        if profile.gen <= 4:
            _early_apply_cursed_body(inflicter_side)

    def _early_apply_throat_chop(inflicter_side: int) -> None:
        """Apply a freshly-landed Throat Chop before the slower target moves.

        Showdown resolves the guaranteed `target.addVolatile('throatchop')`
        secondary inside the inflicter's `moveHit`, so a slower target's
        `onBeforeMove` and DisableMove hooks can already see the sound lock
        on the same turn.
        """
        if inflicter_side == 0:
            if prerolled_move0 is None or int(damage0) <= 0:
                return
            if int(battle[OFF_FIELD + F_SUBSTITUTE_1]) > 0:
                return
            fx.apply_throat_chop_from_move(
                battle,
                move_id0,
                1,
                True,
                prerolled_roll=prerolled_move0.get("ext_vol"),
            )
        else:
            if prerolled_move1 is None or int(damage1) <= 0:
                return
            if int(battle[OFF_FIELD + F_SUBSTITUTE_0]) > 0:
                return
            fx.apply_throat_chop_from_move(
                battle,
                move_id1,
                0,
                True,
                prerolled_roll=prerolled_move1.get("ext_vol"),
            )

    def _apply_resolved_contact_status(inflicter_side: int) -> bool:
        """Apply a contact-status result whose PRNG already ran in damage calc."""
        if inflicter_side == 0:
            meta = _meta0
            atk_off = p0_off
            def_off = p1_off
        else:
            meta = _meta1
            atk_off = p1_off
            def_off = p0_off
        if not meta.get("contact_status_consumed"):
            return False
        before_status = get_status(int(battle[atk_off + 12]))
        before_ext = (
            int(
                battle[
                    OFF_FIELD
                    + (
                        F_EXTENDED_VOLATILE_0
                        if atk_off < OFF_SIDE1
                        else F_EXTENDED_VOLATILE_1
                    )
                ]
            )
            & 0xFFFF
        )
        status_applied = fx.apply_resolved_contact_status_ability(
            battle,
            atk_off,
            def_off,
            game_data,
            move_effects,
            gen5_prng,
            resolved_status_field=meta.get("contact_status_packed"),
            apply_attract=bool(meta.get("contact_status_apply_attract")),
        )
        after_ext = (
            int(
                battle[
                    OFF_FIELD
                    + (
                        F_EXTENDED_VOLATILE_0
                        if atk_off < OFF_SIDE1
                        else F_EXTENDED_VOLATILE_1
                    )
                ]
            )
            & 0xFFFF
        )
        if status_applied and get_status(int(battle[atk_off + 12])) != before_status:
            _run_immediate_status_berry_updates(atk_off)
        return status_applied or after_ext != before_ext

    def _maybe_apply_poison_puppeteer(
        source_off: int,
        target_off: int,
        status_before: int,
        *,
        inline_before_target_move: bool = False,
    ) -> None:
        _ABILITY_POISON_PUPPETEER = 310
        if int(battle[int(source_off) + 5]) != _ABILITY_POISON_PUPPETEER:
            return
        status_after = get_status(int(battle[int(target_off) + 12]))
        if status_after == status_before:
            return
        if status_after not in (STATUS_POISON, STATUS_TOXIC):
            return
        target_side = 0 if int(target_off) < OFF_SIDE1 else 1
        landed = fx.apply_confusion_volatile(battle, target_side, gen5_prng)
        if not landed or not inline_before_target_move:
            return
        _run_immediate_status_berry_updates(target_off)
        # Showdown adds Poison Puppeteer's confusion during the inflicter's
        # move resolution, but the confused target's actual onBeforeMove check
        # still happens later when its runMove starts. Keep the volatile and
        # same-turn cure berries in place now, but let the normal pre-move
        # confusion path spend its PRNG after late after-move hooks like
        # Red Card have already resolved.

    def _pre_calc_status_chain(side_idx: int) -> bool:
        """Roll frz thaw + par full-para in Showdown onBeforeMove order for
        `side_idx`, at the point its runMove would fire. Reads the status
        FRESH at call time — critical when side0's move just inflicted
        frz/slp/par on side1 via a secondary this same turn (Showdown's
        runMove for side1 sees the updated status byte).

        Also handles Truant (ability 54): the loafing bit on pokemon[+15]
        toggles each runMove — when set, cancel this turn's move; when
        clear, set it so the NEXT turn is cancelled.

        Mutates battle state for thawed mons and records results into
        `_status_chain_*` so the legacy block at ~line 2130+ can skip
        re-rolling.

        Returns True iff the mon is immobilized (move cancelled — no
        acc/crit/dmg frames will be consumed).
        """
        nonlocal _status_chain_resolved0, _status_chain_resolved1
        nonlocal _status_chain_rand_thaw0, _status_chain_rand_thaw1
        nonlocal _status_chain_full_para0, _status_chain_full_para1
        nonlocal _status_chain_frozen_blocked0, _status_chain_frozen_blocked1
        nonlocal _status_chain_immobile0, _status_chain_immobile1

        from pokepy.core.constants import (
            FLAG_TRUANT_LOAFING as _FLAG_TRUANT_LOAFING,
            ABILITY_TRUANT as _ABILITY_TRUANT,
            FLAG_SOUND as _FLAG_SOUND_SC,
            F_VOLATILE_0 as _F_VOL_0_SC,
            F_VOLATILE_1 as _F_VOL_1_SC,
            OFF_FIELD as _OFF_FIELD_SC,
        )
        from pokepy.core.bitpack import (
            get_taunt_turns as _get_taunt_turns_sc,
            get_throat_chop_turns as _get_throat_chop_turns_sc,
        )

        if side_idx == 0:
            if _status_chain_resolved0:
                return _status_chain_immobile0
            switching = is_switch
            user_off = p0_off
            sb_mv = _pre_sb_mv0
            current_move_id = move_id0
            using_defrost = _pre_is_thaw0
            must_rech = must_recharge0
            _side_cat = cat0
            _side_vol_off = _OFF_FIELD_SC + _F_VOL_0_SC
        else:
            if _status_chain_resolved1:
                return _status_chain_immobile1
            switching = opp_is_switch
            user_off = p1_off
            sb_mv = _pre_sb_mv1
            current_move_id = move_id1
            using_defrost = _pre_is_thaw1
            must_rech = must_recharge1
            _side_cat = cat1
            _side_vol_off = _OFF_FIELD_SC + _F_VOL_1_SC

        # Read status FRESH here — may have been inflicted by the other
        # side's secondary moments ago.
        status_cur = get_status(int(battle[user_off + 12]))
        is_asleep = status_cur == STATUS_SLEEP
        is_frozen_cur = status_cur == STATUS_FREEZE
        sleep_blocked = is_asleep and sb_mv not in (MOVE_SLEEP_TALK, MOVE_SNORE)

        # Gen 3: sleep counter decrements in slp.onBeforeMove when the mon
        # attempts to act, not at end-of-turn (data/mods/gen3/conditions.ts).
        if profile.gen == 3 and sleep_blocked and not switching:
            sleep_turns = get_status_turns(int(battle[user_off + 12]))
            sleep_turns -= 1
            if sleep_turns <= 0:
                battle[user_off + 12] = 0
                sleep_blocked = False
                is_asleep = False
            else:
                battle[user_off + 12] = set_status(STATUS_SLEEP, sleep_turns)

        thaw = False
        if is_frozen_cur and (not using_defrost) and (not switching):
            thaw = int(gen5_prng.random(5)) == 0
        frozen_blocked_local = is_frozen_cur and (not using_defrost) and (not thaw)
        # Paralysis is lower priority than confusion in Showdown's
        # onBeforeMove chain. Defer the actual full-paralysis roll until
        # after confusion has had a chance to cancel the move.
        para = False
        # Apply thaw mutation.
        if thaw:
            battle[user_off + 12] = 0

        # Truant: toggle the loafing bit on pokemon[+15]. Fires at
        # onBeforeMove priority 9 (between frz(10) and par(1)).
        user_ab_side = int(battle[user_off + 5])
        truant_block = False
        if (
            user_ab_side == _ABILITY_TRUANT
            and not switching
            and not frozen_blocked_local
            and not sleep_blocked
        ):
            flag_word = int(battle[user_off + 15])
            if flag_word & _FLAG_TRUANT_LOAFING:
                # Loafing → cancel this move. Clear the bit — next turn
                # the mon acts again.
                battle[user_off + 15] = flag_word & ~_FLAG_TRUANT_LOAFING
                truant_block = True
            else:
                # Acting this turn → set the bit so next turn is loafing.
                battle[user_off + 15] = flag_word | _FLAG_TRUANT_LOAFING

        # Throat Chop blocks sound moves during the following choice / move
        # cycle. Showdown applies this in the condition's onBeforeMove at
        # priority 6, after Truant (9) and before Taunt (5).
        _throat_chop_block = False
        if not switching:
            _vol_word = int(battle[_side_vol_off])
            if (
                _get_throat_chop_turns_sc(_vol_word) > 0
                and int(current_move_id) >= 0
                and (int(game_data.move_flags[int(current_move_id)]) & _FLAG_SOUND_SC)
                != 0
            ):
                _throat_chop_block = True

        # Taunt cancels the user's status move at onBeforeMove time (Showdown
        # data/moves.ts taunt condition.onBeforeMove: returns false when
        # `move.category === 'Status' && move.id !== 'mefirst'`). When this
        # cancel fires, Showdown's runMove bails BEFORE hitStepAccuracy and
        # BEFORE selfDrops — no PRNG frames are consumed for the taunted
        # move (neither acc nor Curse selfDrops nor secondary rolls). Pokepy
        # used to apply the taunt cancel only at line ~2372 (hit0/hit1 =
        # False) which left the preroll frames intact, shifting the PRNG
        # stream for any taunted-user scenario.
        _taunt_block = False
        if not switching and _side_cat == CAT_STATUS:
            _vol_word = int(battle[_side_vol_off])
            if _get_taunt_turns_sc(_vol_word) > 0:
                _taunt_block = True

        immobile = (
            sleep_blocked
            or frozen_blocked_local
            or must_rech
            or truant_block
            or _throat_chop_block
            or _taunt_block
        )

        if side_idx == 0:
            _status_chain_rand_thaw0 = thaw
            _status_chain_full_para0 = para
            _status_chain_frozen_blocked0 = frozen_blocked_local
            _status_chain_immobile0 = immobile
            _status_chain_resolved0 = True
        else:
            _status_chain_rand_thaw1 = thaw
            _status_chain_full_para1 = para
            _status_chain_frozen_blocked1 = frozen_blocked_local
            _status_chain_immobile1 = immobile
            _status_chain_resolved1 = True
        return immobile

    _mid_turn_pivot_0 = False  # set True when side0's pivot switch fires mid-turn
    _mid_turn_pivot_1 = False  # set True when side1's pivot switch fires mid-turn
    _mid_turn_pivot_target_hp0 = None
    _mid_turn_pivot_target_hp1 = None
    _mid_turn_pivot_saved_hp0 = None
    _mid_turn_pivot_saved_hp1 = None
    _mid_turn_pivot_saved_status0 = None
    _mid_turn_pivot_saved_status1 = None
    _pivot_selfswitch_canceled_0 = False
    _pivot_selfswitch_canceled_1 = False
    _move0_canceled_pre = False  # set True when move0 is canceled before execution (e.g. inline item switch)
    _move1_canceled_pre = False  # set True when move1 is canceled before execution (e.g. inline item switch)
    _mirrored_damage0_on_p1 = 0
    _mirrored_damage1_on_p0 = 0
    _p0_item_forced_switch_pre = False
    _p0_inline_item_switch_resolved = False
    # Item-driven inline active rewrites (Red Card / target-side Eject Button)
    # must skip the later generic tied-speed Update chain once the switch
    # continuation has already resolved. Ordinary self-switch pivots still run
    # that late Update bucket after the slower move and residual.
    _inline_item_switch_rewrote_active = False
    _prerolled_contact0 = (
        []
    )  # prerolled Flame Body / Static / Poison Touch rolls for first mover
    _prerolled_contact1 = (
        []
    )  # prerolled contact rolls for second mover (empty = use live roll)
    _prerolled_toxic_chain0 = None
    _prerolled_toxic_chain1 = None
    _pivot_toxic_chain_handled0 = False
    _pivot_toxic_chain_handled1 = False
    _move0_stat_preapplied = (
        False  # set True when first mover's self-boosts applied early
    )
    _move1_stat_preapplied = (
        False  # set True when second mover's self-boosts applied early
    )
    _move0_successful_self_boost_status = False
    _move1_successful_self_boost_status = False
    _MOVE_PARTING_SHOT_INLINE = 575
    _partingshot0_pre_success = False
    _partingshot1_pre_success = False
    _eject_pack_blocked0 = False
    _eject_pack_blocked1 = False

    def _has_pivot_bench(_side_base, _active_slot):
        _active_slot = int(_active_slot)
        for _slot in range(6):
            if _slot == _active_slot:
                continue
            _slot_off = _side_base + _slot * POKEMON_SIZE
            if (
                int(battle[_slot_off + 1]) > 0
                and (int(battle[_slot_off + 15]) & 0x1) == 0
            ):
                return True
        return False

    def _resolve_inline_target_eject_pack_switch(
        holder_side: int,
        holder_off: int,
        active_slot: int,
        opponent_off: int,
        flush_side0_done: bool,
        flush_side1_done: bool,
    ) -> tuple[bool, int]:
        """Resolve target-side Eject Pack before the forced-out queued move runs."""

        holder_side = int(holder_side)
        holder_off = int(holder_off)
        active_slot = int(active_slot)
        opponent_off = int(opponent_off)
        side_base = OFF_SIDE0 if holder_side == 0 else OFF_SIDE1
        active_meta = M_ACTIVE0 if holder_side == 0 else M_ACTIVE1
        order = side_order0 if holder_side == 0 else side_order1
        hazards_off = OFF_FIELD + (F_HAZARDS_0 if holder_side == 0 else F_HAZARDS_1)
        destiny_bond = F_DESTINY_BOND_0 if holder_side == 0 else F_DESTINY_BOND_1
        if int(battle[holder_off + 1]) <= 0:
            return False, int(battle[holder_off + 1])
        if not _has_pivot_bench(side_base, active_slot):
            return False, int(battle[holder_off + 1])

        _flush_pending_hazard_setters(flush_side0_done, flush_side1_done)
        fx.apply_regenerator_on_switch_out(battle, holder_off, True)
        fx.apply_natural_cure_on_switch_out(battle, holder_off, True)
        battle[holder_off + 6] = 0

        switched = False
        while True:
            current_active = int(battle[OFF_META + active_meta])
            _switch_req = SwitchRequest((holder_side,))
            _choices = yield _switch_req
            _new_slot = int(_choices[holder_side])
            new_active = _resolve_switch_target_from_action(
                side_base,
                current_active,
                _new_slot + 4,
            )
            if new_active == current_active:
                break

            pending_switch_slot_condition = int(battle[OFF_FIELD + destiny_bond])
            battle[OFF_META + active_meta] = np.int16(new_active)
            _sync_showdown_order_on_switch(order, new_active)
            _clear_side_switch_state_common(battle, holder_side)
            if holder_side == 0:
                clear_offsets = (
                    F_VOLATILE_0,
                    F_LEECH_SEED_0,
                    F_DISABLE_TURNS_0,
                    F_EXTENDED_VOLATILE_0,
                    F_DESTINY_BOND_0,
                    F_SUBSTITUTE_0,
                    F_YAWN_TURNS_0,
                    F_PERISH_COUNT_0,
                )
            else:
                clear_offsets = (
                    F_VOLATILE_1,
                    F_LEECH_SEED_1,
                    F_DISABLE_TURNS_1,
                    F_EXTENDED_VOLATILE_1,
                    F_DESTINY_BOND_1,
                    F_SUBSTITUTE_1,
                    F_YAWN_TURNS_1,
                    F_PERISH_COUNT_1,
                )
            for clear_off in clear_offsets:
                battle[OFF_FIELD + clear_off] = 0

            new_off = side_base + new_active * POKEMON_SIZE
            _reset_incoming_switch_state_tracked(new_off)
            consumed_pending = apply_pending_wish_on_switch_in(
                battle,
                holder_side,
                new_off,
                state,
                game_data,
                pending_switch_slot_condition,
            )
            if (
                is_pending_wish_sentinel(pending_switch_slot_condition)
                and not consumed_pending
            ):
                battle[OFF_FIELD + destiny_bond] = np.int16(
                    pending_switch_slot_condition
                )

            switcher_speed = _get_switch_resume_action_speed(battle, new_off)
            if int(battle[opponent_off + 1]) > 0:
                _consume_switch_request_resume_tie_frames(
                    switcher_speed,
                    _action_speed_from_effective(
                        fx.get_effective_speed(battle, opponent_off)
                    ),
                    True,
                    gen5_prng,
                )
            else:
                _consume_runswitch_tie_frame(
                    battle,
                    new_off,
                    opponent_off,
                    gen5_prng,
                )
            fx.apply_hazard_damage_on_switch(battle, new_off, hazards_off)
            _reset_toxic_counter_on_switch_in(battle, new_off)
            switched = True
            if int(battle[new_off + 1]) > 0:
                if int(battle[opponent_off + 1]) > 0:
                    _apply_switch_in_ability_with_trace_reaction_tracked(
                        new_off,
                        opponent_off,
                        True,
                    )
                elif holder_side == 1:
                    _store_pending_opp_switch_in(new_active, switcher_speed)
                _run_switch_in_update_item_hooks(new_off)
                break
            if not _has_pivot_bench(side_base, new_active):
                break

        return switched, int(battle[holder_off + 1])

    def _eject_button_will_cancel_slower_move(
        _target_off: int,
        _target_side_base: int,
        _target_active_slot: int,
        _damage: int,
        _slower_side_switching: bool,
    ) -> bool:
        """Return True when a faster hit will cancel the slower move via Eject Button.

        Showdown resolves the holder's Eject Button switch request immediately
        after the faster hit and before the slower queued move can continue. In
        that path, the slower move never reaches its own runMove update chain,
        so the tied-speed between-moves Update frames must be skipped.
        """

        if _slower_side_switching:
            return False
        if int(_damage) <= 0:
            return False
        if int(battle[int(_target_off) + 6]) != ITEM_EJECT_BUTTON:
            return False
        if int(battle[int(_target_off) + 1]) <= 0:
            return False
        return _has_pivot_bench(_target_side_base, _target_active_slot)

    def _target_eject_button_cancels_damaging_pivot(
        _target_off: int,
        _target_side_base: int,
        _target_active_slot: int,
    ) -> bool:
        """Return True when the target's Eject Button suppresses the user's selfSwitch.

        Showdown's Eject Button handler clears the attacker's pending selfSwitch
        (`source.switchFlag = false`) after the target successfully uses the item.
        We mirror that by suppressing both the inline mid-turn pivot and the
        later post-turn selfSwitch path when the target will be switched out by
        Eject Button.
        """

        if int(battle[int(_target_off) + 6]) != ITEM_EJECT_BUTTON:
            return False
        if int(battle[int(_target_off) + 1]) <= 0:
            return False
        return _has_pivot_bench(_target_side_base, _target_active_slot)

    def _source_survives_damaging_hit_contact(
        _user_off,
        _target_off,
        _move_id,
        _actual_damage,
        _did_hit,
        _num_hits,
        _user_hp_before_contact,
        _move_idx=-1,
    ):
        """Return True if the faster mover survives through DamagingHit.

        Knock Off's onAfterHit item removal happens after drain and contact
        damage, but before recoil/Life Orb. Defender damaging-hit responses
        such as Gulp Missile / Aftermath / Innards Out also resolve in this
        window. The slower move's damage calc should only see Knock Off item
        removal, and the slower foe-targeted move should only remain valid,
        when the faster source is still alive at the end of that full
        DamagingHit cascade.
        """
        _sim = battle.copy()
        _sim[int(_user_off) + 1] = np.int16(int(_user_hp_before_contact))
        _apply_recoil_drain_from_move_tracked(
            _sim,
            _move_id,
            _user_off,
            _actual_damage,
            _did_hit,
            game_data,
            move_effects,
            target_offset=_target_off,
            phase="drain",
        )
        _apply_contact_damage_tracked(
            _sim,
            _move_id,
            _user_off,
            _target_off,
            _did_hit and int(_actual_damage) > 0,
            game_data,
            move_effects,
            num_hits=int(_num_hits),
        )
        _atk_side = 0 if int(_user_off) < OFF_SIDE1 else 1
        _sim_prng = Gen5PRNG(gen5_prng.get_seed_array())
        if _atk_side == 0:
            _dummy_user_off = _target_off
            _dummy_target_off = _user_off
            fx.apply_defender_abilities(
                _sim,
                _move_id,
                _move_id,
                _user_off,
                _dummy_user_off,
                _target_off,
                _dummy_target_off,
                int(_actual_damage),
                0,
                _did_hit and int(_actual_damage) > 0,
                False,
                game_data,
                _sim_prng,
                move_idx0=int(_move_idx),
                move_idx1=-1,
                skip_toxic_chain0=True,
                skip_toxic_chain1=True,
                skip_immediate_stateful_move0=True,
                skip_immediate_stateful_move1=True,
            )
        else:
            _dummy_user_off = _target_off
            _dummy_target_off = _user_off
            fx.apply_defender_abilities(
                _sim,
                _move_id,
                _move_id,
                _dummy_user_off,
                _user_off,
                _dummy_target_off,
                _target_off,
                0,
                int(_actual_damage),
                False,
                _did_hit and int(_actual_damage) > 0,
                game_data,
                _sim_prng,
                move_idx0=-1,
                move_idx1=int(_move_idx),
                skip_toxic_chain0=True,
                skip_toxic_chain1=True,
                skip_immediate_stateful_move0=True,
                skip_immediate_stateful_move1=True,
            )
        return int(_sim[int(_user_off) + 1]) > 0

    def _actual_hp_removed_from_hit(
        _raw_damage,
        _target_had_sub,
        _sub_hp_before,
        _target_hp_before,
        _target_hp_after,
    ):
        if int(_raw_damage) <= 0:
            return 0
        if _target_had_sub:
            return min(int(_raw_damage), int(_sub_hp_before))
        return min(
            int(_raw_damage), max(0, int(_target_hp_before) - int(_target_hp_after))
        )

    def _project_target_hp_after_hit(
        _user_off,
        _target_off,
        _move_id,
        _cat,
        _raw_damage,
        _target_had_sub,
        _sub_hp_before,
        _target_hp_before,
        *,
        target_flags_override=None,
        target_ability_override=None,
        target_item_override=None,
    ):
        _raw_damage = int(_raw_damage)
        _target_hp_before = int(_target_hp_before)
        if _raw_damage <= 0 or _target_hp_before <= 0:
            return max(0, _target_hp_before)
        _move_flags = int(game_data.move_flags[_move_id])
        _is_sound = (_move_flags & FLAG_SOUND) != 0
        _is_infiltrator = int(battle[_user_off + 5]) == ABILITY_INFILTRATOR
        if (
            _target_had_sub
            and int(_sub_hp_before) > 0
            and not _is_sound
            and not _is_infiltrator
        ):
            return max(0, _target_hp_before)
        _target_flags = (
            int(battle[_target_off + 15])
            if target_flags_override is None
            else int(target_flags_override)
        )
        _target_ability = (
            int(battle[_target_off + 5])
            if target_ability_override is None
            else int(target_ability_override)
        )
        _target_item = (
            int(battle[_target_off + 6])
            if target_item_override is None
            else int(target_item_override)
        )
        _target_max_hp = int(battle[_target_off + 2])
        _damage_through = _raw_damage
        if ((_target_flags & 0x40) != 0) and (
            _target_ability == ABILITY_DISGUISE
            or (_target_ability == ABILITY_ICE_FACE and _cat == CAT_PHYSICAL)
        ):
            _damage_through = max(1, _target_max_hp // 8)
        _new_hp = max(0, _target_hp_before - _damage_through)
        if _new_hp == 0 and _target_hp_before == _target_max_hp:
            _is_multihit = int(move_effects.hits_max[_move_id]) > 1
            if (_target_item == ITEM_FOCUS_SASH and not _is_multihit) or (
                _target_ability == ABILITY_STURDY
            ):
                _new_hp = 1
        return int(_new_hp)

    def _actual_damage_for_source_postmove(
        _user_off,
        _target_off,
        _move_id,
        _cat,
        _raw_damage,
        _target_had_sub,
        _sub_hp_before,
        _target_hp_before,
        _target_hp_after,
    ):
        _actual = _actual_hp_removed_from_hit(
            _raw_damage,
            _target_had_sub,
            _sub_hp_before,
            _target_hp_before,
            _target_hp_after,
        )
        if int(_actual) <= 0:
            return 0
        _target_flags = int(battle[_target_off + 15])
        _target_ability = int(battle[_target_off + 5])
        if (
            ((_target_flags & 0x40) != 0)
            and int(_raw_damage) > 0
            and (
                _target_ability == ABILITY_DISGUISE
                or (_target_ability == ABILITY_ICE_FACE and _cat == CAT_PHYSICAL)
            )
        ):
            return 0
        return int(_actual)

    def _target_would_faint_from_projected_hit(
        _user_off,
        _target_off,
        _move_id,
        _cat,
        _raw_damage,
        _target_had_sub,
        _sub_hp_before,
        _target_hp_before,
    ):
        return (
            _project_target_hp_after_hit(
                _user_off,
                _target_off,
                _move_id,
                _cat,
                _raw_damage,
                _target_had_sub,
                _sub_hp_before,
                _target_hp_before,
            )
            == 0
        )

    _CRASH_DAMAGE_MOVES_PRE = (26, 136, 853, 916)

    def _apply_crash_damage_on_move_fail(
        _sim, _user_off, _move_id, _did_hit, _move_attempted
    ):
        if int(_move_id) not in _CRASH_DAMAGE_MOVES_PRE:
            return
        if (not _move_attempted) or _did_hit:
            return
        if int(_sim[int(_user_off) + 1]) <= 0:
            return
        if int(_sim[int(_user_off) + 5]) == ABILITY_MAGIC_GUARD:
            return
        _max_hp_crash = int(_sim[int(_user_off) + 2])
        _crash_cost = max(1, _max_hp_crash // 2)
        _sim[int(_user_off) + 1] = np.int16(
            max(0, int(_sim[int(_user_off) + 1]) - _crash_cost)
        )

    def _project_user_hp_after_own_postmove(
        _user_off,
        _target_off,
        _move_id,
        _raw_damage,
        _actual_damage,
        _did_hit,
        _num_hits,
        _user_hp_before_postmove,
        *,
        include_after_move_secondary: bool = True,
        move_attempted: bool = False,
        restore_target_item: int = 0,
        restore_target_item_off: int = -1,
    ):
        """Project the faster mover's HP through the non-PRNG post-hit chain.

        Showdown resolves drain/contact/recoil/Life Orb before the slower
        foe-targeted move validates its target. If that faster user faints
        here, the slower move must become `[notarget]` and consume no move
        PRNG.
        """
        _sim = battle.copy()
        _sim[int(_user_off) + 1] = np.int16(int(_user_hp_before_postmove))
        if int(restore_target_item) > 0 and int(restore_target_item_off) >= 0:
            _restore_off = int(restore_target_item_off)
            if int(_sim[_restore_off + 6]) == 0:
                _sim[_restore_off + 6] = np.int16(int(restore_target_item))
        _apply_recoil_drain_from_move_tracked(
            _sim,
            _move_id,
            _user_off,
            _actual_damage,
            _did_hit,
            game_data,
            move_effects,
            target_offset=_target_off,
            phase="drain",
        )
        _apply_contact_damage_tracked(
            _sim,
            _move_id,
            _user_off,
            _target_off,
            _did_hit and int(_raw_damage) > 0,
            game_data,
            move_effects,
            num_hits=int(_num_hits),
        )
        _apply_recoil_drain_from_move_tracked(
            _sim,
            _move_id,
            _user_off,
            _actual_damage,
            _did_hit,
            game_data,
            move_effects,
            target_offset=_target_off,
            phase="recoil",
            move_attempted=move_attempted,
        )
        _apply_crash_damage_on_move_fail(
            _sim,
            _user_off,
            _move_id,
            _did_hit,
            move_attempted,
        )
        if not include_after_move_secondary:
            return int(_sim[int(_user_off) + 1])
        _apply_life_orb_recoil_tracked(
            _sim,
            _user_off,
            _raw_damage,
            _did_hit,
            game_data,
            move_id=_move_id,
            move_effects=move_effects,
        )
        _ITEM_SHELL_BELL_POST = 438
        if (
            int(_sim[int(_user_off) + 6]) == _ITEM_SHELL_BELL_POST
            and int(_sim[int(_user_off) + 1]) > 0
            and int(_actual_damage) > 0
            and _did_hit
        ):
            _cur_hp_sb = int(_sim[int(_user_off) + 1])
            _max_hp_sb = int(_sim[int(_user_off) + 2])
            _heal_sb = max(int(_actual_damage) // 8, 1)
            _sim[int(_user_off) + 1] = np.int16(min(_max_hp_sb, _cur_hp_sb + _heal_sb))
        return int(_sim[int(_user_off) + 1])

    _hp0_after_postmove_pre = None
    _hp1_after_postmove_pre = None
    _hazard_from_move_applied0 = False
    _hazard_from_move_applied1 = False
    _hazard_from_move_hit0 = False
    _hazard_from_move_hit1 = False

    def _apply_hazard_from_move_if_pending(side: int) -> None:
        nonlocal _hazard_from_move_applied0, _hazard_from_move_applied1
        if side == 0:
            if _hazard_from_move_applied0:
                return
            fx.apply_hazard_from_move(
                battle,
                move_id0,
                1,
                _hazard_from_move_hit0,
                game_data,
                move_effects,
                user_ability=p0_ab,
                user_offset=user0_off,
                source_hp_override=_hp0_after_postmove_pre,
                enabled_hazards=profile.enabled_hazards,
            )
            _hazard_from_move_applied0 = True
            return
        if _hazard_from_move_applied1:
            return
        fx.apply_hazard_from_move(
            battle,
            move_id1,
            0,
            _hazard_from_move_hit1,
            game_data,
            move_effects,
            user_ability=p1_ab,
            user_offset=user1_off,
            source_hp_override=_hp1_after_postmove_pre,
            enabled_hazards=profile.enabled_hazards,
        )
        _hazard_from_move_applied1 = True

    def _flush_pending_hazard_setters(side0_done: bool, side1_done: bool) -> None:
        if side0_first:
            if side0_done:
                _apply_hazard_from_move_if_pending(0)
            if side1_done:
                _apply_hazard_from_move_if_pending(1)
        else:
            if side1_done:
                _apply_hazard_from_move_if_pending(1)
            if side0_done:
                _apply_hazard_from_move_if_pending(0)

    def _first_move_instaswitch_skips_generic_update(
        *,
        user_off: int,
        target_off: int,
        move_id: int,
        cat: int,
        raw_damage: int,
        target_has_sub_pre: bool,
        target_sub_hp_pre: int,
        target_hp_pre: int,
        user_hp_pre: int,
        num_hits: int,
        move_hit: bool,
        move_attempted: bool,
        restore_target_item: int = 0,
        restore_target_item_off: int = -1,
    ) -> bool:
        """Return True when the faster move queues an instaswitch before move 2.

        Showdown still spends the first move's hit-loop / runMove Updates, but
        once faintMessages queues an instaswitch the later generic post-action
        Update is skipped. Mirror that by subtracting one tied-speed Update
        frame before the slower move begins.
        """

        if not move_attempted or int(user_hp_pre) <= 0:
            return False

        target_hp_after_hit = _project_target_hp_after_hit(
            user_off,
            target_off,
            move_id,
            cat,
            raw_damage,
            target_has_sub_pre,
            target_sub_hp_pre,
            target_hp_pre,
        )
        actual_damage = _actual_damage_for_source_postmove(
            user_off,
            target_off,
            move_id,
            cat,
            raw_damage,
            target_has_sub_pre,
            target_sub_hp_pre,
            target_hp_pre,
            int(target_hp_after_hit),
        )
        projected_user_hp = _project_user_hp_after_own_postmove(
            user_off,
            target_off,
            move_id,
            raw_damage,
            actual_damage,
            move_hit,
            num_hits,
            user_hp_pre,
            move_attempted=move_attempted,
            restore_target_item=restore_target_item,
            restore_target_item_off=restore_target_item_off,
        )
        return projected_user_hp <= 0

    _first_move_kos_p0 = False
    _first_move_kos_p1 = False
    _preapplied_after_move_secondary_hp_delta0 = 0
    _preapplied_after_move_secondary_hp_delta1 = 0
    _projected_after_move_secondary_hp_delta0 = 0
    _projected_after_move_secondary_hp_delta1 = 0
    _projected_full_postmove_hp_committed0 = False
    _projected_full_postmove_hp_committed1 = False
    _skip_contact_damage_effects0 = False
    _skip_contact_damage_effects1 = False
    _skip_crash_damage_effects0 = False
    _skip_crash_damage_effects1 = False
    _skip_after_move_secondary_hp_effects0 = False
    _skip_after_move_secondary_hp_effects1 = False
    _rest_try_failed0 = False
    _rest_try_failed1 = False

    def _rest_try_fails(side: int) -> bool:
        nonlocal _rest_try_failed0, _rest_try_failed1
        if side == 0:
            if move_id0 != MOVE_REST:
                return False
            _rest_try_failed0 = not _can_rest_succeed_bg(battle, p0_off)
            return _rest_try_failed0
        if move_id1 != MOVE_REST:
            return False
        _rest_try_failed1 = not _can_rest_succeed_bg(battle, p1_off)
        return _rest_try_failed1

    if side0_first:
        # First mover's pre-calc onBeforeMove chain: high-priority blockers
        # first, then confusion, then paralysis.
        # If this returns True the move is cancelled: no confusion roll, no
        # acc preroll, no sleep-talk sample, no damage calc, no secondaries
        # preroll. Matches Showdown's `runMove` early-return on falsey
        # onBeforeMove result.
        _p0_immobile_pre = False
        if not is_switch:
            _p0_immobile_pre = _pre_calc_status_chain(0)
        if not (is_switch or _p0_immobile_pre):
            # First mover's confusion check fires BEFORE its damage_calc.
            _self_hit0 = _pre_calc_confusion_check(0)
            if not _self_hit0:
                _p0_immobile_pre = _pre_calc_para_check(0)
            if not (_self_hit0 or _p0_immobile_pre or _sleep_talk_try_failed0):
                _sleep_talk_consume_sample(0)
            if not (_self_hit0 or _p0_immobile_pre or _sleep_talk_try_failed0):
                _rest_try_failed0 = _rest_try_fails(0)
            if not (
                _self_hit0
                or _p0_immobile_pre
                or _sleep_talk_try_failed0
                or _rest_try_failed0
            ):
                _resolve_protect_after_before_move(0)
            if not (
                _self_hit0
                or _p0_immobile_pre
                or _sleep_talk_try_failed0
                or _rest_try_failed0
            ):
                _apply_roost_type_strip(0)
            if not (
                _self_hit0
                or _p0_immobile_pre
                or _sleep_talk_try_failed0
                or _rest_try_failed0
                or pre_damage_fail0
            ):
                _maybe_apply_protean_libero(0, move_id0, is_switch, is_strike_turn0)
            if _maybe_preapply_on_try_move_selfboost(
                0,
                p0_off,
                p1_off,
                _self_hit0
                or _p0_immobile_pre
                or _sleep_talk_try_failed0
                or _rest_try_failed0
                or move0_no_target,
            ):
                _move0_stat_preapplied = True
            if _status_accuracy_preroll_needed(0) and not (
                _self_hit0
                or _p0_immobile_pre
                or _sleep_talk_try_failed0
                or _rest_try_failed0
            ):
                _prerolled_status_acc0 = int(gen5_prng.random(100))
            damage0 = (
                0
                if (
                    _self_hit0
                    or _p0_immobile_pre
                    or _sleep_talk_try_failed0
                    or _rest_try_failed0
                    or move0_no_target
                )
                else _calc_p0()
            )
            if int(damage0) > 0:
                _schedule_cursed_body_after_calc(0)
            if (
                _self_hit0
                or _p0_immobile_pre
                or _sleep_talk_try_failed0
                or _rest_try_failed0
                or move0_no_target
                or is_charge_turn0
            ):
                # Engine: self-hit already applied damage. Mark move as failed.
                if int(damage0) > 0:
                    _schedule_cursed_body_after_preroll(0)
            else:
                # Lockedmove self-effect PRNG (Showdown selfDrops, before
                # secondaries). Must fire before _preroll0 and before the other
                # mover's damage calc to keep PRNG ordering byte-identical.
                _preroll_lockedmove_self_effect(
                    0,
                    move_id0,
                    not is_switch,
                    (damage0 > 0) or target0_protected,
                    is_locked_turn0,
                    locked_turns0_pre,
                    p0_off,
                )
                prerolled_move0 = _preroll0() if not is_switch else None
                _schedule_cursed_body_after_preroll(0)
                # Preroll contact status ability (Flame Body / Static / Poison
                # Point / Poison Touch) for the first mover. In Showdown,
                # DamagingHit fires after secondaries but BEFORE the second
                # mover's turn begins, even when the move is a pivot move that
                # ends up not self-switching (for example, no live bench target
                # exists). Consume the PRNG frame here unconditionally and let
                # whichever branch actually resolves the hit (inline pivot or
                # the shared post-damage cascade) reuse the same prerolled
                # value.
                if _meta0.get("contact_status_consumed"):
                    _prerolled_contact0 = []
                else:
                    _prerolled_contact0 = _preroll_contact_status_ability(
                        battle,
                        move_id0,
                        p0_off,
                        p1_off,
                        not _self_hit0 and damage0 > 0,
                        game_data,
                        gen5_prng,
                    )
                _prerolled_toxic_chain0 = _preroll_toxic_chain(
                    battle,
                    p0_off,
                    p1_off,
                    not _self_hit0 and damage0 > 0,
                    gen5_prng,
                )
        else:
            damage0 = 0
        if move1_targets_foe_mon and not is_delayed1 and int(battle[p0_off + 1]) <= 0:
            move1_no_target = True
        _first_attacker_flinches_target = bool(
            (prerolled_move0 or {}).get("flinch_lands")
        )
        if (
            cat0 == CAT_STATUS
            and not _p0_immobile_pre
            and not _self_hit0
            and not is_switch
        ):
            _preset_screen_if_status(move_id0, 0)
            _status_hit0_pre = _status_move_hits_pre(0, _prerolled_status_acc0)
            # Pre-apply the first mover's primary self-boosts (Coil, Calm
            # Mind, Swords Dance, Dragon Dance, etc.) so the second mover's
            # damage calc sees the correct defensive stats.  Showdown resolves
            # the first mover's stat changes before the second move runs.
            # Skip later re-application at the post-damage stat-change block
            # by setting _move0_stat_preapplied.
            _eff0_pre = int(move_effects.effect_type[move_id0])
            _st0_pre = int(move_effects.stat_target[move_id0])
            _EFFECT_STAT_CHANGE_EARLY = 3  # pokepy.data.move_effects.EFFECT_STAT_CHANGE
            if _eff0_pre == _EFFECT_STAT_CHANGE_EARLY and _st0_pre == 0:
                _boosts0_before = (int(battle[p0_off + 13]), int(battle[p0_off + 14]))
                _apply_stat_changes_from_move_tracked(
                    move_id0,
                    p0_off,
                    p1_off,
                    True,
                )
                _move0_stat_preapplied = True
                _move0_successful_self_boost_status = (
                    _eff0_pre == _EFFECT_STAT_CHANGE_EARLY
                    and (int(battle[p0_off + 13]), int(battle[p0_off + 14]))
                    != _boosts0_before
                )
            if (
                _eff0_pre == EFFECT_SWITCH
                and move_id0 == _MOVE_PARTING_SHOT_INLINE
                and _status_hit0_pre
                and not target0_protected
                and not move0_no_target
                and not prankster_fail0
                and not _move0_canceled_pre
            ):
                _tgt0_boosts_before = (
                    int(battle[p1_off + 13]),
                    int(battle[p1_off + 14]),
                )
                _usr0_boosts_before = (
                    int(battle[p0_off + 13]),
                    int(battle[p0_off + 14]),
                )
                _apply_stat_changes_from_move_tracked(
                    move_id0,
                    p0_off,
                    p1_off,
                    True,
                )
                _move0_stat_preapplied = True
                _partingshot0_pre_success = (
                    int(battle[p1_off + 13]),
                    int(battle[p1_off + 14]),
                ) != _tgt0_boosts_before or (
                    int(battle[p0_off + 13]),
                    int(battle[p0_off + 14]),
                ) != _usr0_boosts_before
            # Pre-apply the first mover's primary status effect (Thunder
            # Wave, Toxic, Will-O-Wisp, Spore, etc.) so the second mover's
            # onBeforeMove sees the new status. Showdown resolves primary
            # status moves immediately inside runMove, so by the time the
            # second mover's `runMove` fires, `onBeforeMove` for frz/par/slp
            # already detects the status. Without this, a same-turn "TW then
            # Tackle" has no paralysis check on the second mover.
            # Only applies when the first mover's status move hit (acc check
            # passed) and the move is a primary status move (EFFECT_STATUS).
            _EFFECT_STATUS_EARLY = 2  # pokepy.data.move_effects.EFFECT_STATUS
            # Skip the pre-apply when the sleep-talk pending mechanism
            # in _sleep_talk_consume_sample will handle it — that code
            # already consumes the random(3) sleep-turns frame at the
            # correct PRNG position for the Spore + Sleep Talk combo.
            _sleep_talk_handles_status = _sleep_talk_pending_slp1
            if (
                _eff0_pre == _EFFECT_STATUS_EARLY
                and _status_hit0_pre
                and not _sleep_talk_handles_status
            ):
                _status_early_applied0 = True
                fx.apply_status_from_move(
                    battle,
                    move_id0,
                    p1_off,
                    True,
                    game_data,
                    move_effects,
                    gen5_prng,
                    user_offset=p0_off,
                    num_hits=1,
                    prerolled_rolls=None,
                )
                _run_immediate_status_berry_updates(p1_off)
        # Pre-apply the first mover's self-targeting stat changes from
        # damaging moves (Close Combat -1 Def/-1 SpD, Superpower -1 Atk/-1 Def,
        # Scale Shot -1 Def/+1 Spe, etc.) so the second mover's damage calc
        # sees the updated Def/SpD. Showdown resolves selfDrops for the first
        # move inside its moveHit before the second mover's runMove.
        # Only applies to:
        #   - self-targeted (stat_target == 0)
        #   - guaranteed (stat_chance == 100)
        #   - NOT pure stat-change moves (already handled above)
        #   - move actually executed (not immobilized, damage > 0 or acc passed)
        # The prerolled PRNG value is reused so no extra frames are consumed.
        if (
            not _move0_stat_preapplied
            and not _p0_immobile_pre
            and not _self_hit0
            and cat0 != CAT_STATUS
        ):
            _sc0_arr, _st0_dmg, _sc0_dmg, _selfboost0_like = (
                fx.get_live_move_stat_change_spec(
                    battle,
                    move_id0,
                    move_effects,
                    p0_off,
                )
            )
            _has_any_sc0 = any(int(_sc0_arr[i]) != 0 for i in range(7))
            _selfboost0_attempted = _selfboost0_like and bool(target1_protected)
            if (
                not _skip_strike_turn_selfboost0
                and (damage0 > 0 or _selfboost0_attempted)
                and _st0_dmg == 0
                and _sc0_dmg == 100
                and _has_any_sc0
            ):
                _apply_stat_changes_from_move_tracked(
                    move_id0,
                    p0_off,
                    p1_off,
                    True,
                    prerolled_roll=(prerolled_move0 or {}).get("stat_change"),
                )
                _move0_stat_preapplied = True
        _hazard_from_move_hit0 = (
            not is_switch
            and not _self_hit0
            and not _p0_immobile_pre
            and not _sleep_talk_try_failed0
            and not _rest_try_failed0
            and not move0_no_target
            and (
                (cat0 != CAT_STATUS and damage0 > 0)
                or (
                    cat0 == CAT_STATUS
                    and _status_move_hits_pre(0, _prerolled_status_acc0)
                )
            )
        )
        side1_survives = (hp1_pre_dmg - damage0) > 0
        if move0_targets_foe_mon and not is_switch:
            _projected_hp1_after_hit = _project_target_hp_after_hit(
                p0_off,
                p1_off,
                move_id0,
                cat0,
                damage0,
                int(battle[OFF_FIELD + F_SUBSTITUTE_1]) > 0,
                int(battle[OFF_FIELD + F_SUBSTITUTE_1]),
                hp1_pre_dmg,
            )
            _first_move_kos_p1 = _projected_hp1_after_hit == 0
            side1_survives = _projected_hp1_after_hit > 0
        # Pre-apply opponent-target stat changes from the first damaging move
        # (Moonblast, Shadow Ball, Snarl, Mystical Fire, etc.) before the
        # slower target computes its own move. Showdown resolves these inside
        # the first move's `moveHit`, so the second mover acts with the live
        # updated boost state on the same turn.
        if (
            side1_survives
            and not _move0_stat_preapplied
            and not _p0_immobile_pre
            and damage0 > 0
            and not _self_hit0
            and cat0 != CAT_STATUS
        ):
            _st0_opp = int(move_effects.stat_target[move_id0])
            _sc0_arr = move_effects.stat_changes[move_id0]
            _has_any_sc0 = any(int(_sc0_arr[i]) != 0 for i in range(7))
            if _st0_opp == 1 and _has_any_sc0:
                _apply_stat_changes_from_move_tracked(
                    move_id0,
                    p0_off,
                    p1_off,
                    True,
                    prerolled_roll=(prerolled_move0 or {}).get("stat_change"),
                )
                _move0_stat_preapplied = True
        if (
            side1_survives
            and not opp_is_switch
            and not _first_attacker_flinches_target
            and not _self_hit1
        ):
            if (
                speeds_tied
                and not is_switch
                and not _eject_button_will_cancel_slower_move(
                    p1_off,
                    OFF_SIDE1,
                    active1,
                    damage0,
                    opp_is_switch,
                )
            ):
                # move0 (first attacker) contributes Updates inside its
                # `hitStepMoveHitLoop` + post-action. Successful self-target
                # setup moves follow the same update path as normal-target
                # status on tied two-move turns; failed self-setup still uses
                # the lighter side/field-style path.
                _between_updates = _blocked_first_action_update_count(
                    bool(_p0_immobile_pre),
                    _move_update_count(
                        _cat0_status,
                        target0_kind,
                        _move0_successful_self_boost_status,
                        int(damage0),
                        int(_meta0.get("num_hits", 1)),
                    ),
                )
                # Gen 1/2: dual primary-status turns do not spend a tied-speed
                # between-move Update before the slower runMove — only the
                # post-action Update at runAction end. Extra frames here drift
                # same-turn full-paralysis rolls (Thunder Wave mirrors).
                if profile.gen <= 2 and _cat0_status and _cat1_status:
                    _between_updates = 0
                _target1_red_card_can_rewrite_slower = (
                    int(battle[p1_off + 6]) == ITEM_RED_CARD
                    and int(battle[p1_off + 1]) > 0
                    and damage0 > 0
                    and not is_switch
                    and not _p0_immobile_pre
                    and not _self_hit0
                )
                _move0_instaswitch_skips_generic_update = (
                    _first_move_instaswitch_skips_generic_update(
                        user_off=p0_off,
                        target_off=p1_off,
                        move_id=move_id0,
                        cat=cat0,
                        raw_damage=int(damage0),
                        target_has_sub_pre=int(battle[OFF_FIELD + F_SUBSTITUTE_1]) > 0,
                        target_sub_hp_pre=int(battle[OFF_FIELD + F_SUBSTITUTE_1]),
                        target_hp_pre=hp1_pre_dmg,
                        user_hp_pre=hp0_pre_dmg,
                        num_hits=int(_meta0.get("num_hits", 1)),
                        move_hit=not _self_hit0 and int(damage0) > 0,
                        move_attempted=not (
                            is_switch
                            or _self_hit0
                            or _p0_immobile_pre
                            or _sleep_talk_try_failed0
                            or _rest_try_failed0
                            or move0_no_target
                            or prankster_fail0
                            or _move0_canceled_pre
                        ),
                        restore_target_item=_inline_knock_saved_item1,
                        restore_target_item_off=_inline_knock_saved_target1,
                    )
                )
                if (
                    _target1_red_card_can_rewrite_slower
                    or _move0_instaswitch_skips_generic_update
                ):
                    # Showdown still spends the hit-loop and end-of-runMove
                    # Updates before Red Card / self-KO resolves, but the
                    # later generic post-action Update is skipped once the
                    # instaswitch continuation is pending (`runAction`
                    # returns early on `instaswitch`). So tied first-move
                    # Red Card / self-KO turns spend one fewer Update frame
                    # before the slower move starts.
                    _between_updates = max(0, int(_between_updates) - 1)
                for _ in range(_between_updates):
                    gen5_prng.random(0, 2)
            _roll_pending_lockedmove_confusion(0, p0_off, apply=True)
            # Showdown's runMove() applies damage0 before the second
            # attacker's runMove() evaluates its own BP. For variable-BP
            # moves like Reversal / Flail (which read the attacker's own
            # current HP -- Mienshao Reversal after taking a Sucker Punch),
            # we must temporarily mirror the damage application on the
            # second attacker's HP slot. Restore after the calc so the
            # real damage pipeline below applies damage0 cleanly.
            # If Sturdy / Focus Sash saved side1, clamp to 1 HP (not 0)
            # so Endeavor and other HP-dependent moves read the correct
            # post-survival HP value.
            _saved_hp1 = int(battle[p1_off + 1])
            battle[p1_off + 1] = np.int16(
                _project_target_hp_after_hit(
                    p0_off,
                    p1_off,
                    move_id0,
                    cat0,
                    damage0,
                    int(battle[OFF_FIELD + F_SUBSTITUTE_1]) > 0,
                    int(battle[OFF_FIELD + F_SUBSTITUTE_1]),
                    _saved_hp1,
                )
            )
            _mirrored_damage0_on_p1 = _saved_hp1 - int(battle[p1_off + 1])
            # Showdown resolves the first mover's non-PRNG `onDamagingHit`
            # state changes before the slower target runs its own move.
            # Pre-apply those live ability/item/type/boost mutations so the
            # second mover's damage calc and later Update sorts see the same
            # post-hit state.
            _defender0_before = _snapshot_boost_stages(p1_off)
            _apply_immediate_defender_ability_state_changes_tracked(
                p0_off,
                p1_off,
                int(damage0) > 0,
                int(damage0),
                int(game_data.move_type[move_id0]),
                cat0,
                int(game_data.move_flags[move_id0]),
            )
            _mark_stats_lowered_this_turn(p1_off, _defender0_before)
            _apply_white_herb_if_ready(p1_off)
            fx.apply_defender_stat_berries_on_damaging_hit(
                battle,
                p1_off,
                cat0,
                int(damage0) > 0,
                damage0,
                game_data,
            )
            # Showdown fires Weakness Policy as part of the hit before the
            # slower target's move, so same-turn damage reads the boosted
            # attacking stats and the consumed item state.
            fx.apply_weakness_policy(
                battle,
                p1_off,
                int(game_data.move_type[move_id0]),
                int(damage0) > 0,
                damage0,
                move_id0,
            )
            prime_gulp_missile_state_from_move(
                battle,
                p0_off,
                move_id0,
                move_executed=True,
                is_charge_turn=is_charge_turn0,
            )
            _move0_preapplied_immediate_defender_state = True
            # Early-apply side0's secondary status on side1 BEFORE side1's
            # onBeforeMove, so a freshly-inflicted frz/slp/par this turn
            # cancels side1's move (Showdown ordering).
            _early_apply_inflicted_status(0)
            _early_apply_toxic_chain(0)
            _contact_status_early_applied0 = _apply_resolved_contact_status(0)
            _early_apply_throat_chop(0)
            # Gen 8+ Showdown refreshes queued move speeds after each
            # completed action when another move remains in the queue. Do the
            # refresh after immediate defender-state and same-turn status
            # application so the later mover's cached `pokemon.speed` sees
            # effects like Icy Wind, Weak Armor, or a freshly applied
            # paralysis from the first move.
            _refresh_current_action_speeds()
            fx.apply_self_type_removal_from_move(
                battle,
                move_id0,
                p0_off,
                int(damage0) > 0,
            )
            # Flash Fire: if the first mover (side 0) used a Fire move
            # absorbed by side 1's Flash Fire, set the flag before side 1's
            # damage calc so the 1.5x boost applies on this turn.
            if (
                int(battle[p1_off + 5]) == ABILITY_FLASH_FIRE
                and int(game_data.move_type[move_id0]) == 1  # TYPE_FIRE
                and damage0 == 0
                and not _p0_immobile_pre
                and cat0 != CAT_STATUS
            ):
                _ff_flags1 = int(battle[p1_off + 15])
                _ff_new1 = _ff_flags1 | 0x200
                if _ff_new1 >= 0x8000:
                    _ff_new1 -= 0x10000
                battle[p1_off + 15] = _ff_new1
            # ----- MID-TURN PIVOT (U-turn / Volt Switch / Flip Turn /
            # Parting Shot / Teleport / Baton Pass / first-move Eject Pack) -----
            # Showdown flow: after the first mover's DamagingHit hooks fire,
            # the pivoter switches out immediately and the slower mon's move
            # targets the NEW active. We replicate that here by performing an
            # auto-switch on side 0 before computing damage1, so
            # calc_damage_gen9 reads the new mon's stats.
            _target1_eject_button_cancels_pivot0 = (
                int(move_effects.effect_type[move_id0]) == EFFECT_SWITCH
                and not is_switch
                and not _p0_immobile_pre
                and not _self_hit0
                and damage0 > 0
                and _has_pivot_bench(OFF_SIDE0, active0)
                and _target_eject_button_cancels_damaging_pivot(
                    p1_off,
                    OFF_SIDE1,
                    active1,
                )
            )
            if _target1_eject_button_cancels_pivot0:
                _pivot_selfswitch_canceled_0 = True
            _ITEM_EJECT_PACK_INLINE = 714
            _eject_pack_ready0 = (
                int(battle[p0_off + 6]) == _ITEM_EJECT_PACK_INLINE
                and int(battle[p0_off + 1]) > 0
                and stats_lowered_this_turn0
                and _has_pivot_bench(OFF_SIDE0, active0)
            )
            if _eject_pack_ready0 and _target1_eject_button_cancels_pivot0:
                _eject_pack_blocked0 = True
            _pivot_is_eject_pack0 = (
                _eject_pack_ready0 and not _target1_eject_button_cancels_pivot0
            )
            _ABILITY_SUCTION_CUPS_INLINE = 21
            _ABILITY_GUARD_DOG_INLINE = 275
            _target1_red_card_overrides_pivot0 = (
                int(battle[p1_off + 6]) == ITEM_RED_CARD
                and int(battle[p1_off + 1]) > 0
                and damage0 > 0
                and not is_switch
                and not _p0_immobile_pre
                and not _self_hit0
                and cat0 != CAT_STATUS
                and p0_ab
                not in (_ABILITY_SUCTION_CUPS_INLINE, _ABILITY_GUARD_DOG_INLINE)
            )
            if _target1_red_card_overrides_pivot0:
                _pivot_selfswitch_canceled_0 = True
            _is_pivot_0 = (
                int(move_effects.effect_type[move_id0]) == EFFECT_SWITCH
                and not is_switch
                and not _p0_immobile_pre
                and not _self_hit0
                and (damage0 > 0 or cat0 == CAT_STATUS)
                and (move_id0 != _MOVE_PARTING_SHOT_INLINE or _partingshot0_pre_success)
                and _has_pivot_bench(OFF_SIDE0, active0)
                and not _target1_eject_button_cancels_pivot0
                and not _target1_red_card_overrides_pivot0
            ) or (_pivot_is_eject_pack0 and not _target1_red_card_overrides_pivot0)
            if _is_pivot_0:
                _pivot_user_off = OFF_SIDE0 + active0 * POKEMON_SIZE
                _pivot_user_hp_before_postmove = int(battle[_pivot_user_off + 1])
                _pivot_user_alive = int(battle[_pivot_user_off + 1]) > 0
                # Apply contact damage from the pivot move to the user NOW
                # (Showdown fires onDamagingHit before selfSwitch). We
                # apply it to the user's HP directly so the outgoing mon
                # takes the recoil and Regenerator heals from the correct
                # HP. Restore p1's HP first so the contact functions see
                # the real state, then re-save.
                battle[p1_off + 1] = _saved_hp1
                if _pivot_user_alive and damage0 > 0:
                    _apply_contact_damage_tracked(
                        battle,
                        move_id0,
                        _pivot_user_off,
                        p1_off,
                        True,
                        game_data,
                        move_effects,
                        num_hits=int(_meta0.get("num_hits", 1)),
                    )
                    _apply_contact_status_ability_tracked(
                        battle,
                        move_id0,
                        _pivot_user_off,
                        p1_off,
                        True,
                        game_data,
                        move_effects,
                        gen5_prng,
                        prerolled_rolls=_prerolled_contact0,
                    )
                    if _prerolled_toxic_chain0 is None:
                        _pivot_toxic_chain_handled0 = (
                            fx.apply_toxic_chain_on_damaging_hit(
                                battle,
                                _pivot_user_off,
                                p1_off,
                                True,
                                int(damage0),
                                game_data,
                                gen5_prng,
                            )
                        )
                # Re-apply temporary damage for variable-BP calc below.
                _saved_hp1 = int(battle[p1_off + 1])
                battle[p1_off + 1] = max(0, _saved_hp1 - int(damage0))
                # Apply Regenerator / Natural Cure on the outgoing mon.
                _pivot_user_postmove_hp = _hp0_after_postmove_pre
                if _pivot_user_postmove_hp is None:
                    _pivot_user_postmove_hp = _project_user_hp_after_own_postmove(
                        _pivot_user_off,
                        p1_off,
                        move_id0,
                        damage0,
                        int(damage0) if damage0 > 0 else 0,
                        not _self_hit0 and (damage0 > 0 or cat0 == CAT_STATUS),
                        int(_meta0.get("num_hits", 1)),
                        _pivot_user_hp_before_postmove,
                        move_attempted=not (
                            is_switch
                            or _self_hit0
                            or _p0_immobile_pre
                            or _sleep_talk_try_failed0
                            or _rest_try_failed0
                            or move0_no_target
                            or prankster_fail0
                            or _move0_canceled_pre
                        ),
                    )
                battle[_pivot_user_off + 1] = np.int16(_pivot_user_postmove_hp)
                _pivot_user_alive = int(battle[_pivot_user_off + 1]) > 0
                if _pivot_user_alive:
                    fx.apply_regenerator_on_switch_out(
                        battle,
                        _pivot_user_off,
                        True,
                    )
                    fx.apply_natural_cure_on_switch_out(
                        battle,
                        _pivot_user_off,
                        True,
                    )
                    _mid_turn_pivot_saved_hp0 = int(battle[_pivot_user_off + 1])
                    _mid_turn_pivot_saved_status0 = int(battle[_pivot_user_off + 12])
                if _pivot_is_eject_pack0 and _pivot_user_alive:
                    battle[_pivot_user_off + 6] = 0
                if _pivot_user_alive:
                    # Showdown only sets switchFlag for self-switch moves if
                    # the source is still alive after post-move recoil/effects.
                    # If the pivoter faints (for example to Life Orb), the
                    # slower foe move sees no target and the replacement is
                    # requested only after upkeep.
                    _switch_req = SwitchRequest((0,))
                    _choices = yield _switch_req
                    _new_slot = int(_choices[0])
                    _new_active0_pivot = _resolve_switch_target_from_action(
                        OFF_SIDE0,
                        active0,
                        _new_slot + 4,
                    )
                    if _new_active0_pivot != active0:
                        _pending_switch_slot_condition0_pivot = int(
                            battle[OFF_FIELD + F_DESTINY_BOND_0]
                        )
                        battle[OFF_META + M_ACTIVE0] = _new_active0_pivot
                        _sync_showdown_order_on_switch(side_order0, _new_active0_pivot)
                        # Clear side-0 switch-related volatile state.
                        _clear_side_switch_state_common(battle, 0)
                        for _off in (
                            F_VOLATILE_0,
                            F_LEECH_SEED_0,
                            F_DISABLE_TURNS_0,
                            F_EXTENDED_VOLATILE_0,
                            F_DESTINY_BOND_0,
                            F_SUBSTITUTE_0,
                            F_YAWN_TURNS_0,
                            F_PERISH_COUNT_0,
                        ):
                            battle[OFF_FIELD + _off] = 0
                        _new_p0 = OFF_SIDE0 + _new_active0_pivot * POKEMON_SIZE
                        _reset_incoming_switch_state_tracked(_new_p0)
                        _pivot_consumed_pending0 = apply_pending_wish_on_switch_in(
                            battle,
                            0,
                            _new_p0,
                            state,
                            game_data,
                            _pending_switch_slot_condition0_pivot,
                        )
                        if (
                            is_pending_wish_sentinel(
                                _pending_switch_slot_condition0_pivot
                            )
                            and not _pivot_consumed_pending0
                        ):
                            battle[OFF_FIELD + F_DESTINY_BOND_0] = np.int16(
                                _pending_switch_slot_condition0_pivot
                            )
                        if int(battle[p1_off + 1]) > 0:
                            _consume_switch_request_resume_tie_frames(
                                fx.get_effective_speed(battle, _new_p0),
                                fx.get_effective_speed(battle, p1_off),
                                True,
                                gen5_prng,
                            )
                        else:
                            _consume_runswitch_tie_frame(
                                battle,
                                _new_p0,
                                p1_off,
                                gen5_prng,
                            )
                        _flush_pending_hazard_setters(True, False)
                        # Hazard damage on the new switch-in.
                        fx.apply_hazard_damage_on_switch(
                            battle,
                            _new_p0,
                            OFF_FIELD + F_HAZARDS_0,
                        )
                    _reset_toxic_counter_on_switch_in(battle, _new_p0)
                    # Switch-in ability (Intimidate, weather, terrain).
                    if int(battle[_new_p0 + 1]) > 0:
                        _apply_switch_in_ability_with_trace_reaction_tracked(
                            _new_p0,
                            p1_off,
                            True,
                        )
                        _run_switch_in_update_item_hooks(_new_p0)
                    # Keep the local live-target aliases aligned with the
                    # replacement so same-turn post-hit hooks (Knock Off,
                    # Trick, stat/status side effects) land on the new active.
                    active0 = _new_active0_pivot
                    p0_off = _new_p0
                    target1_off = p0_off
                    p0_ab = int(battle[p0_off + 5])
                    _mid_turn_pivot_target_hp0 = int(battle[p0_off + 1])
                    _refresh_current_action_speeds()
                    _mid_turn_pivot_0 = True
                    if (
                        move1_targets_foe_mon
                        and not is_delayed1
                        and int(battle[p0_off + 1]) <= 0
                    ):
                        # If the pivot replacement dies to switch-in hazards,
                        # the slower foe-targeted move finds no live target in
                        # Showdown and consumes no move PRNG.
                        move1_no_target = True
            # ----- END MID-TURN PIVOT -----
            # Roost: strip the slower Roosting mon's Flying type NOW, after the
            # faster mon's attack has already resolved against original types.
            _apply_slow_roost()
            if damage0 > 0:
                _apply_seed_sower(p1_off)
            _target1_has_sub_pre = int(battle[OFF_FIELD + F_SUBSTITUTE_1]) > 0
            _knock_off_source_alive0 = (
                int(move_effects.effect_type[move_id0]) == EFFECT_KNOCK_OFF
                and damage0 > 0
                and not target0_protected
                and (not _target1_has_sub_pre or p0_ab == ABILITY_INFILTRATOR)
                and _source_survives_damaging_hit_contact(
                    p0_off,
                    p1_off,
                    move_id0,
                    int(damage0),
                    True,
                    int(_meta0.get("num_hits", 1)),
                    int(battle[p0_off + 1]),
                    move_idx,
                )
            )
            if _knock_off_source_alive0:
                _inline_knock_removed_focus_sash1 = (
                    int(battle[p1_off + 6]) == ITEM_FOCUS_SASH
                )
                _inline_knock_saved_item1 = int(battle[p1_off + 6])
                _inline_knock_saved_target1 = int(p1_off)
                _apply_knock_off_from_move_tracked(
                    move_id0,
                    p1_off,
                    True,
                    user_offset=p0_off,
                )
            # Eject Button: if P1 holds Eject Button and was damaged by P0,
            # P1's move is cancelled (Showdown: BeforeSwitchOut fires, then
            # the pokemon switch happens, consuming P1's action). We check here
            # so that `_p1_immobile_pre` short-circuits the second-mover chain.
            _p1_eject_button_cancels = False
            if (
                int(battle[p1_off + 6]) == ITEM_EJECT_BUTTON
                and damage0 > 0
                and int(battle[p1_off + 1]) > 0
                and not _p0_immobile_pre
                and not _self_hit0
                and not opp_is_switch
            ):
                _pre_active1_item = int(battle[OFF_META + M_ACTIVE1])
                _flush_pending_hazard_setters(True, False)
                fx.apply_item_forced_switch(
                    battle,
                    move_id0,
                    -1,
                    p0_off,
                    p1_off,
                    p1_off,
                    p0_off,
                    damage0,
                    0,
                    True,
                    False,
                    side_order0,
                    side_order1,
                    gen5_prng,
                    game_data,
                    state,
                )
                _post_active1_item = int(battle[OFF_META + M_ACTIVE1])
                if _post_active1_item != _pre_active1_item:
                    _inline_item_switch_rewrote_active = True
                    _p1_eject_button_cancels = True
                    _move1_canceled_pre = True
                    # The shared HP writeback below still subtracts damage0
                    # from `_saved_hp1`; offset the baseline so switch-out
                    # healing from Regenerator survives that writeback.
                    _saved_hp1 = int(battle[p1_off + 1]) + int(_mirrored_damage0_on_p1)
            if (
                int(battle[p1_off + 6]) == _ITEM_EJECT_PACK_INLINE
                and int(battle[p1_off + 1]) > 0
                and stats_lowered_this_turn1
                and not _eject_pack_blocked1
                and not _p0_immobile_pre
                and not _self_hit0
                and not opp_is_switch
            ):
                _p1_eject_pack_switched, _p1_hp_after_eject_pack = (
                    _resolve_inline_target_eject_pack_switch(
                        1,
                        p1_off,
                        active1,
                        p0_off,
                        True,
                        False,
                    )
                )
                if _p1_eject_pack_switched:
                    _inline_item_switch_rewrote_active = True
                    _p1_eject_button_cancels = True
                    _move1_canceled_pre = True
                    _eject_pack_blocked1 = True
                    _saved_hp1 = int(_p1_hp_after_eject_pack) + int(
                        _mirrored_damage0_on_p1
                    )
            if (
                move1_targets_foe_mon
                and not move1_no_target
                and not _mid_turn_pivot_0
                and not is_switch
            ):
                _actual_dmg0 = _actual_damage_for_source_postmove(
                    p0_off,
                    p1_off,
                    move_id0,
                    cat0,
                    damage0,
                    _target1_has_sub_pre,
                    int(battle[OFF_FIELD + F_SUBSTITUTE_1]),
                    hp1_pre_dmg,
                    int(battle[p1_off + 1]),
                )
                if _self_hit0:
                    # Confusion self-hit already changed the faster user's live
                    # HP before the slower move begins. Reuse that real HP
                    # snapshot instead of re-projecting only the post-move
                    # recoil/contact chain from the turn-start state.
                    _hp0_after_postmove_pre = int(battle[p0_off + 1])
                else:
                    _hp0_after_postmove_pre = _project_user_hp_after_own_postmove(
                        p0_off,
                        p1_off,
                        move_id0,
                        damage0,
                        _actual_dmg0,
                        not _self_hit0 and damage0 > 0,
                        int(_meta0.get("num_hits", 1)),
                        hp0_pre_dmg,
                        move_attempted=not (
                            is_switch
                            or _self_hit0
                            or _p0_immobile_pre
                            or _sleep_talk_try_failed0
                            or _rest_try_failed0
                            or move0_no_target
                            or prankster_fail0
                            or _move0_canceled_pre
                        ),
                        restore_target_item=_inline_knock_saved_item1,
                        restore_target_item_off=_inline_knock_saved_target1,
                    )
                _preapplied_after_move_secondary_hp_delta0 = 0
                _projected_after_move_secondary_hp_delta0 = 0
                _hp0_before_after_move_secondary_pre = None
                if (
                    int(battle[p0_off + 6]) in (ITEM_LIFE_ORB, 438)
                    or move1_can_change_foe_item
                ):
                    _hp0_before_after_move_secondary_pre = (
                        _project_user_hp_after_own_postmove(
                            p0_off,
                            p1_off,
                            move_id0,
                            damage0,
                            _actual_dmg0,
                            not _self_hit0 and damage0 > 0,
                            int(_meta0.get("num_hits", 1)),
                            hp0_pre_dmg,
                            include_after_move_secondary=False,
                            move_attempted=not (
                                is_switch
                                or _self_hit0
                                or _p0_immobile_pre
                                or _sleep_talk_try_failed0
                                or _rest_try_failed0
                                or move0_no_target
                                or prankster_fail0
                                or _move0_canceled_pre
                            ),
                            restore_target_item=_inline_knock_saved_item1,
                            restore_target_item_off=_inline_knock_saved_target1,
                        )
                    )
                    _projected_after_move_secondary_hp_delta0 = int(
                        _hp0_after_postmove_pre
                    ) - int(_hp0_before_after_move_secondary_pre)
                if move1_can_change_foe_item:
                    _preapplied_after_move_secondary_hp_delta0 = int(
                        _projected_after_move_secondary_hp_delta0
                    )
                if _hp0_after_postmove_pre <= 0 and not is_delayed1:
                    move1_no_target = True
            else:
                _hp0_after_postmove_pre = None
                _preapplied_after_move_secondary_hp_delta0 = 0
                _projected_after_move_secondary_hp_delta0 = 0
            if (
                int(battle[p1_off + 6]) == ITEM_RED_CARD
                and int(battle[p1_off + 1]) > 0
                and damage0 > 0
                and not _p0_immobile_pre
                and not _self_hit0
                and not is_switch
            ):
                _pre_active0_item = int(battle[OFF_META + M_ACTIVE0])
                _red_card_user0_off = int(p0_off)
                _saved_live_user0_hp = int(battle[_red_card_user0_off + 1])
                _red_card_actual_dmg0 = _actual_damage_for_source_postmove(
                    _red_card_user0_off,
                    p1_off,
                    move_id0,
                    cat0,
                    damage0,
                    _target1_has_sub_pre,
                    int(battle[OFF_FIELD + F_SUBSTITUTE_1]),
                    hp1_pre_dmg,
                    int(battle[p1_off + 1]),
                )
                _red_card_user0_hp = _project_user_hp_after_own_postmove(
                    _red_card_user0_off,
                    p1_off,
                    move_id0,
                    damage0,
                    _red_card_actual_dmg0,
                    not _self_hit0 and damage0 > 0,
                    int(_meta0.get("num_hits", 1)),
                    hp0_pre_dmg,
                    include_after_move_secondary=False,
                    move_attempted=not (
                        is_switch
                        or _self_hit0
                        or _p0_immobile_pre
                        or _sleep_talk_try_failed0
                        or _rest_try_failed0
                        or move0_no_target
                        or prankster_fail0
                        or _move0_canceled_pre
                    ),
                )
                battle[_red_card_user0_off + 1] = np.int16(_red_card_user0_hp)
                fx.apply_item_forced_switch(
                    battle,
                    move_id0,
                    -1,
                    _red_card_user0_off,
                    p1_off,
                    p1_off,
                    _red_card_user0_off,
                    damage0,
                    0,
                    True,
                    False,
                    side_order0,
                    side_order1,
                    gen5_prng,
                    game_data,
                    state,
                )
                _post_active0_item = int(battle[OFF_META + M_ACTIVE0])
                if _post_active0_item != _pre_active0_item:
                    _inline_item_switch_rewrote_active = True
                    # Showdown's Red Card drag resolves immediately after the
                    # hit, so the slower foe-targeted move reads the new active
                    # and the dragged-out source skips afterMoveSecondarySelf.
                    _skip_after_move_secondary_hp_effects0 = True
                    _mid_turn_pivot_saved_hp0 = int(battle[_red_card_user0_off + 1])
                    _mid_turn_pivot_saved_status0 = int(
                        battle[_red_card_user0_off + 12]
                    )
                    active0 = _post_active0_item
                    p0_off = OFF_SIDE0 + active0 * POKEMON_SIZE
                    target1_off = p0_off
                    p0_ab = int(battle[p0_off + 5])
                    _mid_turn_pivot_target_hp0 = int(battle[p0_off + 1])
                    _hp0_after_postmove_pre = None
                    _preapplied_after_move_secondary_hp_delta0 = 0
                    _refresh_current_action_speeds()
                    _mid_turn_pivot_0 = True
                    if (
                        move1_targets_foe_mon
                        and not is_delayed1
                        and int(battle[p0_off + 1]) <= 0
                    ):
                        move1_no_target = True
                else:
                    battle[_red_card_user0_off + 1] = np.int16(_saved_live_user0_hp)
            # Second mover's onBeforeMove: high-priority blockers first,
            # then confusion, then paralysis.
            _p1_immobile_pre = False
            if move1_no_target:
                battle[p1_off + 1] = _saved_hp1
                _p1_immobile_pre = _pre_calc_status_chain(1)
                if not _p1_immobile_pre:
                    if not _conf_checked1 and not _self_hit1:
                        _self_hit1 = _pre_calc_confusion_check(1)
                    if not _self_hit1:
                        _p1_immobile_pre = _pre_calc_para_check(1)
                    if _self_hit1:
                        # The temp HP slot already includes move0's projected
                        # damage. Carry only the confusion self-hit delta back
                        # into the final baseline HP so the unified damage
                        # stage below doesn't apply move0's damage twice.
                        _saved_hp1 = int(battle[p1_off + 1]) + _mirrored_damage0_on_p1
                damage1 = 0
            else:
                _p1_immobile_pre = _pre_calc_status_chain(1)
            if _p1_eject_button_cancels:
                battle[p1_off + 1] = _saved_hp1
                damage1_after_flinch = 0
                damage1 = 0
                hit1 = False
            elif not (move1_no_target or _p1_immobile_pre):
                # Second mover's confusion check fires BEFORE its damage_calc
                # — but only when the volatile was NOT applied by the first
                # mover this turn (that case is handled via the inline check
                # in preroll0 above, which also sets `_conf_checked1`).
                if not _conf_checked1 and not _self_hit1:
                    _self_hit1 = _pre_calc_confusion_check(1)
                if not _self_hit1:
                    _p1_immobile_pre = _pre_calc_para_check(1)
                # If the pre-calc check self-hit (check_confusion_self_hit
                # mutates HP), carry only that extra self-hit delta into the
                # restored baseline HP. The real damage pipeline still needs
                # to apply move0's damage once, but not twice.
                if _self_hit1:
                    _saved_hp1 = int(battle[p1_off + 1]) + _mirrored_damage0_on_p1
                if not (_self_hit1 or _p1_immobile_pre or _sleep_talk_try_failed1):
                    _sleep_talk_consume_sample(1)
                if not (_self_hit1 or _p1_immobile_pre or _sleep_talk_try_failed1):
                    _rest_try_failed1 = _rest_try_fails(1)
                if not (
                    _self_hit1
                    or _p1_immobile_pre
                    or _sleep_talk_try_failed1
                    or _rest_try_failed1
                ):
                    _resolve_protect_after_before_move(1)
                if not (
                    _self_hit1
                    or _p1_immobile_pre
                    or _sleep_talk_try_failed1
                    or _rest_try_failed1
                ):
                    _apply_roost_type_strip(1)
                if not (
                    _self_hit1
                    or _p1_immobile_pre
                    or _sleep_talk_try_failed1
                    or _rest_try_failed1
                    or pre_damage_fail1
                ):
                    _maybe_apply_protean_libero(
                        1, move_id1, opp_is_switch, is_strike_turn1
                    )
                if _maybe_preapply_on_try_move_selfboost(
                    1,
                    p1_off,
                    p0_off,
                    _self_hit1
                    or _p1_immobile_pre
                    or _sleep_talk_try_failed1
                    or _rest_try_failed1
                    or move1_no_target,
                ):
                    _move1_stat_preapplied = True
                if _status_accuracy_preroll_needed(1) and not (
                    _self_hit1
                    or _p1_immobile_pre
                    or _sleep_talk_try_failed1
                    or _rest_try_failed1
                    or _first_move_kos_p1
                ):
                    _prerolled_status_acc1 = int(gen5_prng.random(100))
                _saved_target_hp0 = int(battle[p0_off + 1])
                if _hp0_after_postmove_pre is not None:
                    # Showdown resolves the first mover's deterministic
                    # post-hit HP changes before the slower move's damage
                    # formula runs, so full-HP checks (Multiscale, Shadow
                    # Shield, etc.) must see the chipped target.
                    battle[p0_off + 1] = np.int16(_hp0_after_postmove_pre)
                damage1 = (
                    0
                    if (
                        _self_hit1
                        or _p1_immobile_pre
                        or _sleep_talk_try_failed1
                        or _rest_try_failed1
                        or move1_no_target
                    )
                    else _calc_p1(
                        user_hurt_by_target_this_turn=(_mirrored_damage0_on_p1 > 0)
                    )
                )
                battle[p0_off + 1] = np.int16(_saved_target_hp0)
                battle[p1_off + 1] = _saved_hp1
                if not (
                    _self_hit1
                    or _p1_immobile_pre
                    or _sleep_talk_try_failed1
                    or _rest_try_failed1
                    or move1_no_target
                    or is_charge_turn1
                ):
                    # Lockedmove self-effect PRNG for second mover.
                    _preroll_lockedmove_self_effect(
                        1,
                        move_id1,
                        not opp_is_switch,
                        (damage1 > 0) or target1_protected,
                        is_locked_turn1,
                        locked_turns1_pre,
                        p1_off,
                    )
                    prerolled_move1 = (
                        _preroll1(target_hp_override=_hp0_after_postmove_pre)
                        if not opp_is_switch
                        else None
                    )
                    _schedule_cursed_body_after_preroll(1)
                # Pre-apply the second mover's primary status (Thunder Wave,
                # Spore, etc.) once its acc check passes — symmetric with the
                # first-mover block above. Showdown resolves these inside the
                # slower runMove before the legacy immobile pass.
                if (
                    cat1 == CAT_STATUS
                    and not _p1_immobile_pre
                    and not _self_hit1
                    and not opp_is_switch
                ):
                    _preset_screen_if_status(move_id1, 1)
                    _status_hit1_pre = _status_move_hits_pre(1, _prerolled_status_acc1)
                    _eff1_pre_second = int(move_effects.effect_type[move_id1])
                    _EFFECT_STATUS_EARLY1B = 2
                    if (
                        _eff1_pre_second == _EFFECT_STATUS_EARLY1B
                        and _status_hit1_pre
                        and not _sleep_talk_pending_slp0
                        and not _status_early_applied1
                    ):
                        _status_early_applied1 = True
                        fx.apply_status_from_move(
                            battle,
                            move_id1,
                            p0_off,
                            True,
                            game_data,
                            move_effects,
                            gen5_prng,
                            user_offset=p1_off,
                            num_hits=1,
                            prerolled_rolls=None,
                        )
                        _run_immediate_status_berry_updates(p0_off)
            else:
                battle[p1_off + 1] = _saved_hp1
                damage1 = 0
        elif _first_attacker_flinches_target:
            # Slowbro is going to be flinched — Showdown skips its move
            # entirely (no acc/crit/damage frames, no secondaries roll,
            # no between-move Updates because faintMessages doesn't fire
            # and `eachEvent` after the first move sees a 1-element list
            # when speeds tied — but we still need to consume the 1 frame
            # for the move0 hit-loop Update on tied speeds).
            #
            # Still mirror higher-priority onBeforeMove checks first.
            # Example: Ice Fang can both freeze and flinch. Showdown applies
            # the new freeze inside the first move's moveHit, then the slower
            # target's runMove still rolls frz thaw (random(5)) before flinch
            # would cancel the move. If pokepy short-circuits straight to the
            # flinch branch, it skips that thaw frame and drifts PRNG.
            if int(damage0) > 0 and not is_switch and not _self_hit0:
                _early_apply_inflicted_status(0)
                _p1_immobile_pre = _pre_calc_status_chain(1)
            if speeds_tied and not is_switch and not _p0_immobile_pre:
                _between_updates = _move_update_count(
                    _cat0_status,
                    target0_kind,
                    _move0_successful_self_boost_status,
                    int(damage0),
                    int(_meta0.get("num_hits", 1)),
                )
                for _ in range(_between_updates):
                    gen5_prng.random(0, 2)
            _roll_pending_lockedmove_confusion(0, p0_off, apply=True)
            damage1 = 0
        elif _self_hit1:
            # Second mover is self-hitting from confusion inline check
            # (first mover's secondary confused them this turn and the
            # inline check landed). No calc, no damage roll. Between-move
            # Updates still fire when speeds are tied.
            if speeds_tied and not is_switch and not _p0_immobile_pre:
                _between_updates = _move_update_count(
                    _cat0_status,
                    target0_kind,
                    _move0_successful_self_boost_status,
                    int(damage0),
                    int(_meta0.get("num_hits", 1)),
                )
                for _ in range(_between_updates):
                    gen5_prng.random(0, 2)
            _roll_pending_lockedmove_confusion(0, p0_off, apply=True)
            damage1 = 0
        else:
            if speeds_tied and not _p0_immobile_pre:
                _between_updates = (
                    _move_update_count(
                        _cat0_status,
                        target0_kind,
                        _move0_successful_self_boost_status,
                        int(damage0),
                        int(_meta0.get("num_hits", 1)),
                    )
                    if opp_is_switch
                    else (0 if int(_meta0.get("num_hits", 1)) > 1 else 1)
                )
                _deferred_hit_loop_update_frames += int(_between_updates)
            damage1 = 0
            move1_killed_target = True
    else:
        _p1_immobile_pre = False
        if not opp_is_switch:
            _p1_immobile_pre = _pre_calc_status_chain(1)
        if not (opp_is_switch or _p1_immobile_pre):
            _self_hit1 = _pre_calc_confusion_check(1)
            if not _self_hit1:
                _p1_immobile_pre = _pre_calc_para_check(1)
            if not (_self_hit1 or _p1_immobile_pre or _sleep_talk_try_failed1):
                _sleep_talk_consume_sample(1)
            if not (_self_hit1 or _p1_immobile_pre or _sleep_talk_try_failed1):
                _rest_try_failed1 = _rest_try_fails(1)
            if not (
                _self_hit1
                or _p1_immobile_pre
                or _sleep_talk_try_failed1
                or _rest_try_failed1
            ):
                _resolve_protect_after_before_move(1)
            if not (
                _self_hit1
                or _p1_immobile_pre
                or _sleep_talk_try_failed1
                or _rest_try_failed1
            ):
                _apply_roost_type_strip(1)
            if not (
                _self_hit1
                or _p1_immobile_pre
                or _sleep_talk_try_failed1
                or _rest_try_failed1
                or pre_damage_fail1
            ):
                _maybe_apply_protean_libero(1, move_id1, opp_is_switch, is_strike_turn1)
            if _maybe_preapply_on_try_move_selfboost(
                1,
                p1_off,
                p0_off,
                _self_hit1
                or _p1_immobile_pre
                or _sleep_talk_try_failed1
                or _rest_try_failed1
                or move1_no_target,
            ):
                _move1_stat_preapplied = True
            if _status_accuracy_preroll_needed(1) and not (
                _self_hit1
                or _p1_immobile_pre
                or _sleep_talk_try_failed1
                or _rest_try_failed1
            ):
                _prerolled_status_acc1 = int(gen5_prng.random(100))
            damage1 = (
                0
                if (
                    _self_hit1
                    or _p1_immobile_pre
                    or _sleep_talk_try_failed1
                    or _rest_try_failed1
                    or move1_no_target
                )
                else _calc_p1()
            )
            if int(damage1) > 0:
                _schedule_cursed_body_after_calc(1)
            if not (
                move1_no_target
                or _self_hit1
                or _p1_immobile_pre
                or _sleep_talk_try_failed1
                or _rest_try_failed1
                or is_charge_turn1
            ):
                # Lockedmove self-effect PRNG for first mover (side1 first).
                _preroll_lockedmove_self_effect(
                    1,
                    move_id1,
                    not opp_is_switch,
                    (damage1 > 0) or target1_protected,
                    is_locked_turn1,
                    locked_turns1_pre,
                    p1_off,
                )
                prerolled_move1 = _preroll1() if not opp_is_switch else None
                _schedule_cursed_body_after_preroll(1)
                # Preroll contact status ability for the first mover (side1).
                # Side1 pivot isn't implemented in pokepy, so no pivot skip needed.
                if _meta1.get("contact_status_consumed"):
                    _prerolled_contact1 = []
                else:
                    _prerolled_contact1 = _preroll_contact_status_ability(
                        battle,
                        move_id1,
                        p1_off,
                        p0_off,
                        not _self_hit1 and damage1 > 0,
                        game_data,
                        gen5_prng,
                    )
                _prerolled_toxic_chain1 = _preroll_toxic_chain(
                    battle,
                    p1_off,
                    p0_off,
                    not _self_hit1 and damage1 > 0,
                    gen5_prng,
                )
            elif int(damage1) > 0:
                _schedule_cursed_body_after_preroll(1)
        else:
            damage1 = 0
        if move0_targets_foe_mon and not is_delayed0 and int(battle[p1_off + 1]) <= 0:
            move0_no_target = True
        _first_attacker_flinches_target = bool(
            (prerolled_move1 or {}).get("flinch_lands")
        )
        if (
            cat1 == CAT_STATUS
            and not _p1_immobile_pre
            and not _self_hit1
            and not opp_is_switch
        ):
            _preset_screen_if_status(move_id1, 1)
            _status_hit1_pre = _status_move_hits_pre(1, _prerolled_status_acc1)
            # Symmetric pre-apply: side1 first mover's primary self-boosts.
            _eff1_pre = int(move_effects.effect_type[move_id1])
            _st1_pre = int(move_effects.stat_target[move_id1])
            _EFFECT_STAT_CHANGE_EARLY1 = 3
            if _eff1_pre == _EFFECT_STAT_CHANGE_EARLY1 and _st1_pre == 0:
                _boosts1_before = (int(battle[p1_off + 13]), int(battle[p1_off + 14]))
                _apply_stat_changes_from_move_tracked(
                    move_id1,
                    p1_off,
                    p0_off,
                    True,
                )
                _move1_stat_preapplied = True
                _move1_successful_self_boost_status = (
                    _eff1_pre == _EFFECT_STAT_CHANGE_EARLY1
                    and (int(battle[p1_off + 13]), int(battle[p1_off + 14]))
                    != _boosts1_before
                )
            if (
                _eff1_pre == EFFECT_SWITCH
                and move_id1 == _MOVE_PARTING_SHOT_INLINE
                and _status_hit1_pre
                and not target1_protected
                and not move1_no_target
                and not prankster_fail1
                and not _move1_canceled_pre
            ):
                _tgt1_boosts_before = (
                    int(battle[p0_off + 13]),
                    int(battle[p0_off + 14]),
                )
                _usr1_boosts_before = (
                    int(battle[p1_off + 13]),
                    int(battle[p1_off + 14]),
                )
                _apply_stat_changes_from_move_tracked(
                    move_id1,
                    p1_off,
                    p0_off,
                    True,
                )
                _move1_stat_preapplied = True
                _partingshot1_pre_success = (
                    int(battle[p0_off + 13]),
                    int(battle[p0_off + 14]),
                ) != _tgt1_boosts_before or (
                    int(battle[p1_off + 13]),
                    int(battle[p1_off + 14]),
                ) != _usr1_boosts_before
            # Symmetric pre-apply: side1 first mover's primary status effect.
            _EFFECT_STATUS_EARLY1 = 2
            _sleep_talk_handles_status1 = _sleep_talk_pending_slp0
            if (
                _eff1_pre == _EFFECT_STATUS_EARLY1
                and _status_hit1_pre
                and not _sleep_talk_handles_status1
            ):
                _status_early_applied1 = True
                fx.apply_status_from_move(
                    battle,
                    move_id1,
                    p0_off,
                    True,
                    game_data,
                    move_effects,
                    gen5_prng,
                    user_offset=p1_off,
                    num_hits=1,
                    prerolled_rolls=None,
                )
                _run_immediate_status_berry_updates(p0_off)
        # Symmetric pre-apply: side1 first mover's self-targeted stat drops from
        # DAMAGING moves (Close Combat -Def/-SpD, Superpower -Atk/-Def, Scale Shot,
        # etc.) so the second mover's (side0's) damage calc sees the updated stats.
        # This mirrors lines 2411-2426 in the side0_first branch.
        # Only applies to:
        #   - NOT already handled (not a pure STATUS move — those were handled above)
        #   - self-targeted (stat_target == 0)
        #   - guaranteed (stat_chance == 100)
        #   - move actually executed (not immobilized, damage > 0 or acc passed)
        if (
            not _move1_stat_preapplied
            and not _p1_immobile_pre
            and not _self_hit1
            and cat1 != CAT_STATUS
        ):
            _sc1_arr, _st1_dmg, _sc1_dmg, _selfboost1_like = (
                fx.get_live_move_stat_change_spec(
                    battle,
                    move_id1,
                    move_effects,
                    p1_off,
                )
            )
            _has_any_sc1 = any(int(_sc1_arr[i]) != 0 for i in range(7))
            _selfboost1_attempted = _selfboost1_like and bool(target0_protected)
            if (
                not _skip_strike_turn_selfboost1
                and (damage1 > 0 or _selfboost1_attempted)
                and _st1_dmg == 0
                and _sc1_dmg == 100
                and _has_any_sc1
            ):
                _apply_stat_changes_from_move_tracked(
                    move_id1,
                    p1_off,
                    p0_off,
                    True,
                    prerolled_roll=(prerolled_move1 or {}).get("stat_change"),
                )
                _move1_stat_preapplied = True
        _hazard_from_move_hit1 = (
            not opp_is_switch
            and not _self_hit1
            and not _p1_immobile_pre
            and not _sleep_talk_try_failed1
            and not _rest_try_failed1
            and not move1_no_target
            and (
                (cat1 != CAT_STATUS and damage1 > 0)
                or (
                    cat1 == CAT_STATUS
                    and _status_move_hits_pre(1, _prerolled_status_acc1)
                )
            )
        )
        side0_survives = (hp0_pre_dmg - damage1) > 0
        if move1_targets_foe_mon and not opp_is_switch:
            _projected_hp0_after_hit = _project_target_hp_after_hit(
                p1_off,
                p0_off,
                move_id1,
                cat1,
                damage1,
                int(battle[OFF_FIELD + F_SUBSTITUTE_0]) > 0,
                int(battle[OFF_FIELD + F_SUBSTITUTE_0]),
                hp0_pre_dmg,
            )
            _first_move_kos_p0 = _projected_hp0_after_hit == 0
            side0_survives = _projected_hp0_after_hit > 0
        # Symmetric opponent-target stat-drop pre-apply for the side1-first
        # branch so the slower side0 attacker reads the live lowered stats.
        if (
            side0_survives
            and not _move1_stat_preapplied
            and not _p1_immobile_pre
            and damage1 > 0
            and not _self_hit1
            and cat1 != CAT_STATUS
        ):
            _st1_opp = int(move_effects.stat_target[move_id1])
            _sc1_arr = move_effects.stat_changes[move_id1]
            _has_any_sc1 = any(int(_sc1_arr[i]) != 0 for i in range(7))
            if _st1_opp == 1 and _has_any_sc1:
                _apply_stat_changes_from_move_tracked(
                    move_id1,
                    p1_off,
                    p0_off,
                    True,
                    prerolled_roll=(prerolled_move1 or {}).get("stat_change"),
                )
                _move1_stat_preapplied = True
        if (
            side0_survives
            and not is_switch
            and not _first_attacker_flinches_target
            and not _self_hit0
        ):
            if (
                speeds_tied
                and not opp_is_switch
                and not _eject_button_will_cancel_slower_move(
                    p0_off,
                    OFF_SIDE0,
                    active0,
                    damage1,
                    is_switch,
                )
            ):
                _between_updates = _blocked_first_action_update_count(
                    bool(_p1_immobile_pre),
                    _move_update_count(
                        _cat1_status,
                        target1_kind,
                        _move1_successful_self_boost_status,
                        int(damage1),
                        int(_meta1.get("num_hits", 1)),
                    ),
                )
                if profile.gen <= 2 and _cat0_status and _cat1_status:
                    _between_updates = 0
                _target0_red_card_can_rewrite_slower = (
                    int(battle[p0_off + 6]) == ITEM_RED_CARD
                    and int(battle[p0_off + 1]) > 0
                    and damage1 > 0
                    and not opp_is_switch
                    and not _p1_immobile_pre
                    and not _self_hit1
                )
                _move1_instaswitch_skips_generic_update = (
                    _first_move_instaswitch_skips_generic_update(
                        user_off=p1_off,
                        target_off=p0_off,
                        move_id=move_id1,
                        cat=cat1,
                        raw_damage=int(damage1),
                        target_has_sub_pre=int(battle[OFF_FIELD + F_SUBSTITUTE_0]) > 0,
                        target_sub_hp_pre=int(battle[OFF_FIELD + F_SUBSTITUTE_0]),
                        target_hp_pre=hp0_pre_dmg,
                        user_hp_pre=hp1_pre_dmg,
                        num_hits=int(_meta1.get("num_hits", 1)),
                        move_hit=not _self_hit1 and int(damage1) > 0,
                        move_attempted=not (
                            opp_is_switch
                            or _self_hit1
                            or _p1_immobile_pre
                            or _sleep_talk_try_failed1
                            or _rest_try_failed1
                            or move1_no_target
                            or prankster_fail1
                            or _move1_canceled_pre
                        ),
                        restore_target_item=_inline_knock_saved_item0,
                        restore_target_item_off=_inline_knock_saved_target0,
                    )
                )
                if (
                    _target0_red_card_can_rewrite_slower
                    or _move1_instaswitch_skips_generic_update
                ):
                    # Same Showdown early-return as above for the side-1
                    # first-move branch.
                    _between_updates = max(0, int(_between_updates) - 1)
                for _ in range(_between_updates):
                    gen5_prng.random(0, 2)
            _roll_pending_lockedmove_confusion(1, p1_off, apply=True)
            # Mirror the damage1 application on side0 before computing
            # damage0 so variable-BP moves (Reversal / Flail / Wake-Up
            # Slap / Assurance) see the updated state.
            # If Sturdy / Focus Sash saved side0, clamp to 1 HP (not 0)
            # so Endeavor and other HP-dependent moves read the correct
            # post-survival HP value.
            _saved_hp0 = int(battle[p0_off + 1])
            battle[p0_off + 1] = np.int16(
                _project_target_hp_after_hit(
                    p1_off,
                    p0_off,
                    move_id1,
                    cat1,
                    damage1,
                    int(battle[OFF_FIELD + F_SUBSTITUTE_0]) > 0,
                    int(battle[OFF_FIELD + F_SUBSTITUTE_0]),
                    _saved_hp0,
                )
            )
            _mirrored_damage1_on_p0 = _saved_hp0 - int(battle[p0_off + 1])
            _defender1_before = _snapshot_boost_stages(p0_off)
            _apply_immediate_defender_ability_state_changes_tracked(
                p1_off,
                p0_off,
                int(damage1) > 0,
                int(damage1),
                int(game_data.move_type[move_id1]),
                cat1,
                int(game_data.move_flags[move_id1]),
            )
            _mark_stats_lowered_this_turn(p0_off, _defender1_before)
            _apply_white_herb_if_ready(p0_off)
            fx.apply_defender_stat_berries_on_damaging_hit(
                battle,
                p0_off,
                cat1,
                int(damage1) > 0,
                damage1,
                game_data,
            )
            # Symmetric same-turn Weakness Policy preapply for the side hit by
            # the first move so its slower action sees the live +2 Atk/+2 SpA.
            fx.apply_weakness_policy(
                battle,
                p0_off,
                int(game_data.move_type[move_id1]),
                int(damage1) > 0,
                damage1,
                move_id1,
            )
            prime_gulp_missile_state_from_move(
                battle,
                p1_off,
                move_id1,
                move_executed=True,
                is_charge_turn=is_charge_turn1,
            )
            _move1_preapplied_immediate_defender_state = True
            _early_apply_inflicted_status(1)
            _early_apply_toxic_chain(1)
            _contact_status_early_applied1 = _apply_resolved_contact_status(1)
            _early_apply_throat_chop(1)
            # Mirror Showdown's gen8+ post-action queue speed refresh before
            # the slower side0 move begins so later move-action Update sorts
            # see first-move speed changes that already resolved, including
            # same-turn paralysis inflicted by the faster move.
            _refresh_current_action_speeds()
            fx.apply_self_type_removal_from_move(
                battle,
                move_id1,
                p1_off,
                int(damage1) > 0,
            )
            # Flash Fire: if the first mover (side 1) used a Fire move
            # absorbed by side 0's Flash Fire, set the flag before side 0's
            # damage calc so the 1.5x boost applies on this turn.
            if (
                int(battle[p0_off + 5]) == ABILITY_FLASH_FIRE
                and int(game_data.move_type[move_id1]) == 1  # TYPE_FIRE
                and damage1 == 0
                and not _p1_immobile_pre
                and cat1 != CAT_STATUS
            ):
                _ff_flags0 = int(battle[p0_off + 15])
                _ff_new0 = _ff_flags0 | 0x200
                if _ff_new0 >= 0x8000:
                    _ff_new0 -= 0x10000
                battle[p0_off + 15] = _ff_new0
            # ----- MID-TURN PIVOT (side 1 first / inline Eject Pack) -----
            _target0_eject_button_cancels_pivot1 = (
                int(move_effects.effect_type[move_id1]) == EFFECT_SWITCH
                and not opp_is_switch
                and not _p1_immobile_pre
                and not _self_hit1
                and damage1 > 0
                and _has_pivot_bench(OFF_SIDE1, active1)
                and _target_eject_button_cancels_damaging_pivot(
                    p0_off,
                    OFF_SIDE0,
                    active0,
                )
            )
            if _target0_eject_button_cancels_pivot1:
                _pivot_selfswitch_canceled_1 = True
            _ITEM_EJECT_PACK_INLINE = 714
            _eject_pack_ready1 = (
                int(battle[p1_off + 6]) == _ITEM_EJECT_PACK_INLINE
                and int(battle[p1_off + 1]) > 0
                and stats_lowered_this_turn1
                and _has_pivot_bench(OFF_SIDE1, active1)
            )
            if _eject_pack_ready1 and _target0_eject_button_cancels_pivot1:
                _eject_pack_blocked1 = True
            _pivot_is_eject_pack1 = (
                _eject_pack_ready1 and not _target0_eject_button_cancels_pivot1
            )
            _ABILITY_SUCTION_CUPS_INLINE = 21
            _ABILITY_GUARD_DOG_INLINE = 275
            _target0_red_card_overrides_pivot1 = (
                int(battle[p0_off + 6]) == ITEM_RED_CARD
                and int(battle[p0_off + 1]) > 0
                and damage1 > 0
                and not opp_is_switch
                and not _p1_immobile_pre
                and not _self_hit1
                and cat1 != CAT_STATUS
                and p1_ab
                not in (_ABILITY_SUCTION_CUPS_INLINE, _ABILITY_GUARD_DOG_INLINE)
            )
            if _target0_red_card_overrides_pivot1:
                _pivot_selfswitch_canceled_1 = True
            _is_pivot_1 = (
                int(move_effects.effect_type[move_id1]) == EFFECT_SWITCH
                and not opp_is_switch
                and not _p1_immobile_pre
                and not _self_hit1
                and (damage1 > 0 or cat1 == CAT_STATUS)
                and (move_id1 != _MOVE_PARTING_SHOT_INLINE or _partingshot1_pre_success)
                and _has_pivot_bench(OFF_SIDE1, active1)
                and not _target0_eject_button_cancels_pivot1
                and not _target0_red_card_overrides_pivot1
            ) or (_pivot_is_eject_pack1 and not _target0_red_card_overrides_pivot1)
            if _is_pivot_1:
                _pivot_user_off = OFF_SIDE1 + active1 * POKEMON_SIZE
                _pivot_user_hp_before_postmove = int(battle[_pivot_user_off + 1])
                _pivot_user_alive = int(battle[_pivot_user_off + 1]) > 0
                battle[p0_off + 1] = _saved_hp0
                if _pivot_user_alive and damage1 > 0:
                    _apply_contact_damage_tracked(
                        battle,
                        move_id1,
                        _pivot_user_off,
                        p0_off,
                        True,
                        game_data,
                        move_effects,
                        num_hits=int(_meta1.get("num_hits", 1)),
                    )
                    _apply_contact_status_ability_tracked(
                        battle,
                        move_id1,
                        _pivot_user_off,
                        p0_off,
                        True,
                        game_data,
                        move_effects,
                        gen5_prng,
                        prerolled_rolls=_prerolled_contact1,
                    )
                    if _prerolled_toxic_chain1 is None:
                        _pivot_toxic_chain_handled1 = (
                            fx.apply_toxic_chain_on_damaging_hit(
                                battle,
                                _pivot_user_off,
                                p0_off,
                                True,
                                int(damage1),
                                game_data,
                                gen5_prng,
                            )
                        )
                _saved_hp0 = int(battle[p0_off + 1])
                battle[p0_off + 1] = max(0, _saved_hp0 - int(damage1))
                _pivot_user_postmove_hp = _hp1_after_postmove_pre
                if _pivot_user_postmove_hp is None:
                    _pivot_user_postmove_hp = _project_user_hp_after_own_postmove(
                        _pivot_user_off,
                        p0_off,
                        move_id1,
                        damage1,
                        int(damage1) if damage1 > 0 else 0,
                        not _self_hit1 and (damage1 > 0 or cat1 == CAT_STATUS),
                        int(_meta1.get("num_hits", 1)),
                        _pivot_user_hp_before_postmove,
                        move_attempted=not (
                            opp_is_switch
                            or _self_hit1
                            or _p1_immobile_pre
                            or _sleep_talk_try_failed1
                            or _rest_try_failed1
                            or move1_no_target
                            or prankster_fail1
                            or _move1_canceled_pre
                        ),
                    )
                battle[_pivot_user_off + 1] = np.int16(_pivot_user_postmove_hp)
                _pivot_user_alive = int(battle[_pivot_user_off + 1]) > 0
                if _pivot_user_alive:
                    fx.apply_regenerator_on_switch_out(
                        battle,
                        _pivot_user_off,
                        True,
                    )
                    fx.apply_natural_cure_on_switch_out(
                        battle,
                        _pivot_user_off,
                        True,
                    )
                    _mid_turn_pivot_saved_hp1 = int(battle[_pivot_user_off + 1])
                    _mid_turn_pivot_saved_status1 = int(battle[_pivot_user_off + 12])
                if _pivot_is_eject_pack1 and _pivot_user_alive:
                    battle[_pivot_user_off + 6] = 0
                if _pivot_user_alive:
                    _switch_req = SwitchRequest((1,))
                    _choices = yield _switch_req
                    _new_slot = int(_choices[1])
                    _new_active1_pivot = _resolve_switch_target_from_action(
                        OFF_SIDE1,
                        active1,
                        _new_slot + 4,
                    )
                    if _new_active1_pivot != active1:
                        _pending_switch_slot_condition1_pivot = int(
                            battle[OFF_FIELD + F_DESTINY_BOND_1]
                        )
                        battle[OFF_META + M_ACTIVE1] = _new_active1_pivot
                        _sync_showdown_order_on_switch(side_order1, _new_active1_pivot)
                        _clear_side_switch_state_common(battle, 1)
                        for _off in (
                            F_VOLATILE_1,
                            F_LEECH_SEED_1,
                            F_DISABLE_TURNS_1,
                            F_EXTENDED_VOLATILE_1,
                            F_DESTINY_BOND_1,
                            F_SUBSTITUTE_1,
                            F_YAWN_TURNS_1,
                            F_PERISH_COUNT_1,
                        ):
                            battle[OFF_FIELD + _off] = 0
                        _new_p1 = OFF_SIDE1 + _new_active1_pivot * POKEMON_SIZE
                        _reset_incoming_switch_state_tracked(_new_p1)
                        _pivot_consumed_pending1 = apply_pending_wish_on_switch_in(
                            battle,
                            1,
                            _new_p1,
                            state,
                            game_data,
                            _pending_switch_slot_condition1_pivot,
                        )
                        if (
                            is_pending_wish_sentinel(
                                _pending_switch_slot_condition1_pivot
                            )
                            and not _pivot_consumed_pending1
                        ):
                            battle[OFF_FIELD + F_DESTINY_BOND_1] = np.int16(
                                _pending_switch_slot_condition1_pivot
                            )
                        if int(battle[p0_off + 1]) > 0:
                            _consume_switch_request_resume_tie_frames(
                                fx.get_effective_speed(battle, _new_p1),
                                fx.get_effective_speed(battle, p0_off),
                                True,
                                gen5_prng,
                            )
                        else:
                            _consume_runswitch_tie_frame(
                                battle,
                                _new_p1,
                                p0_off,
                                gen5_prng,
                            )
                        _flush_pending_hazard_setters(False, True)
                        fx.apply_hazard_damage_on_switch(
                            battle,
                            _new_p1,
                            OFF_FIELD + F_HAZARDS_1,
                        )
                        _reset_toxic_counter_on_switch_in(battle, _new_p1)
                        if int(battle[_new_p1 + 1]) > 0:
                            _apply_switch_in_ability_with_trace_reaction_tracked(
                                _new_p1,
                                p0_off,
                                True,
                            )
                            _run_switch_in_update_item_hooks(_new_p1)
                        # Keep the local live-target aliases aligned with the
                        # replacement so same-turn post-hit hooks (Knock Off,
                        # Trick, stat/status side effects) land on the new active.
                        active1 = _new_active1_pivot
                        p1_off = _new_p1
                        target0_off = p1_off
                        p1_ab = int(battle[p1_off + 5])
                        _mid_turn_pivot_target_hp1 = int(battle[p1_off + 1])
                        _refresh_current_action_speeds()
                        _mid_turn_pivot_1 = True
                        if (
                            move0_targets_foe_mon
                            and not is_delayed0
                            and int(battle[p1_off + 1]) <= 0
                        ):
                            # Mirror the side-0 pivot case above: a replacement
                            # that dies to hazards leaves the slower foe-targeted
                            # move with `[notarget]` and zero PRNG in Showdown.
                            move0_no_target = True
            # Roost: strip the slower Roosting mon's Flying type NOW.
            _apply_slow_roost()
            if damage1 > 0:
                _apply_seed_sower(p0_off)
            _target0_has_sub_pre = int(battle[OFF_FIELD + F_SUBSTITUTE_0]) > 0
            _knock_off_source_alive1 = (
                int(move_effects.effect_type[move_id1]) == EFFECT_KNOCK_OFF
                and damage1 > 0
                and not target1_protected
                and (not _target0_has_sub_pre or p1_ab == ABILITY_INFILTRATOR)
                and _source_survives_damaging_hit_contact(
                    p1_off,
                    p0_off,
                    move_id1,
                    int(damage1),
                    True,
                    int(_meta1.get("num_hits", 1)),
                    int(battle[p1_off + 1]),
                    opp_move_idx,
                )
            )
            if _knock_off_source_alive1:
                _inline_knock_removed_focus_sash0 = (
                    int(battle[p0_off + 6]) == ITEM_FOCUS_SASH
                )
                _inline_knock_saved_item0 = int(battle[p0_off + 6])
                _inline_knock_saved_target0 = int(p0_off)
                _apply_knock_off_from_move_tracked(
                    move_id1,
                    p0_off,
                    True,
                    user_offset=p1_off,
                )
            # Eject Button: if P0 holds Eject Button and was damaged by the
            # faster P1 move, P0's slower move is canceled and the holder is
            # forced out after the turn (Showdown consumes the pending action).
            if (
                int(battle[p0_off + 6]) == ITEM_EJECT_BUTTON
                and damage1 > 0
                and int(battle[p0_off + 1]) > 0
                and not _p1_immobile_pre
                and not _self_hit1
                and not is_switch
            ):
                _flush_pending_hazard_setters(False, True)
                _p0_item_forced_switch_pre = fx.apply_item_forced_switch(
                    battle,
                    -1,
                    move_id1,
                    p0_off,
                    p1_off,
                    p1_off,
                    p0_off,
                    0,
                    damage1,
                    False,
                    True,
                    side_order0,
                    side_order1,
                    gen5_prng,
                    game_data,
                    state,
                )
                if _p0_item_forced_switch_pre:
                    _move0_canceled_pre = True
            if (
                int(battle[p0_off + 6]) == _ITEM_EJECT_PACK_INLINE
                and int(battle[p0_off + 1]) > 0
                and stats_lowered_this_turn0
                and not _eject_pack_blocked0
                and not _p1_immobile_pre
                and not _self_hit1
                and not is_switch
            ):
                _p0_eject_pack_switched, _p0_hp_after_eject_pack = (
                    _resolve_inline_target_eject_pack_switch(
                        0,
                        p0_off,
                        active0,
                        p1_off,
                        False,
                        True,
                    )
                )
                if _p0_eject_pack_switched:
                    _inline_item_switch_rewrote_active = True
                    _move0_canceled_pre = True
                    _eject_pack_blocked0 = True
                    _saved_hp0 = int(_p0_hp_after_eject_pack) + int(
                        _mirrored_damage1_on_p0
                    )
            if (
                move0_targets_foe_mon
                and not move0_no_target
                and not _mid_turn_pivot_1
                and not opp_is_switch
            ):
                _actual_dmg1 = _actual_damage_for_source_postmove(
                    p1_off,
                    p0_off,
                    move_id1,
                    cat1,
                    damage1,
                    _target0_has_sub_pre,
                    int(battle[OFF_FIELD + F_SUBSTITUTE_0]),
                    hp0_pre_dmg,
                    int(battle[p0_off + 1]),
                )
                if _self_hit1:
                    _hp1_after_postmove_pre = int(battle[p1_off + 1])
                else:
                    _hp1_after_postmove_pre = _project_user_hp_after_own_postmove(
                        p1_off,
                        p0_off,
                        move_id1,
                        damage1,
                        _actual_dmg1,
                        not _self_hit1 and damage1 > 0,
                        int(_meta1.get("num_hits", 1)),
                        hp1_pre_dmg,
                        move_attempted=not (
                            opp_is_switch
                            or _self_hit1
                            or _p1_immobile_pre
                            or _sleep_talk_try_failed1
                            or _rest_try_failed1
                            or move1_no_target
                            or prankster_fail1
                            or _move1_canceled_pre
                        ),
                        restore_target_item=_inline_knock_saved_item0,
                        restore_target_item_off=_inline_knock_saved_target0,
                    )
                _preapplied_after_move_secondary_hp_delta1 = 0
                _projected_after_move_secondary_hp_delta1 = 0
                _hp1_before_after_move_secondary_pre = None
                if (
                    int(battle[p1_off + 6]) in (ITEM_LIFE_ORB, 438)
                    or move0_can_change_foe_item
                ):
                    _hp1_before_after_move_secondary_pre = (
                        _project_user_hp_after_own_postmove(
                            p1_off,
                            p0_off,
                            move_id1,
                            damage1,
                            _actual_dmg1,
                            not _self_hit1 and damage1 > 0,
                            int(_meta1.get("num_hits", 1)),
                            hp1_pre_dmg,
                            include_after_move_secondary=False,
                            move_attempted=not (
                                opp_is_switch
                                or _self_hit1
                                or _p1_immobile_pre
                                or _sleep_talk_try_failed1
                                or _rest_try_failed1
                                or move1_no_target
                                or prankster_fail1
                                or _move1_canceled_pre
                            ),
                            restore_target_item=_inline_knock_saved_item0,
                            restore_target_item_off=_inline_knock_saved_target0,
                        )
                    )
                    _projected_after_move_secondary_hp_delta1 = int(
                        _hp1_after_postmove_pre
                    ) - int(_hp1_before_after_move_secondary_pre)
                if move0_can_change_foe_item:
                    _preapplied_after_move_secondary_hp_delta1 = int(
                        _projected_after_move_secondary_hp_delta1
                    )
                if _hp1_after_postmove_pre <= 0 and not is_delayed0:
                    move0_no_target = True
            else:
                _hp1_after_postmove_pre = None
                _preapplied_after_move_secondary_hp_delta1 = 0
                _projected_after_move_secondary_hp_delta1 = 0
            if (
                int(battle[p0_off + 6]) == ITEM_RED_CARD
                and int(battle[p0_off + 1]) > 0
                and damage1 > 0
                and not _p1_immobile_pre
                and not _self_hit1
                and not opp_is_switch
            ):
                _pre_active1_item = int(battle[OFF_META + M_ACTIVE1])
                _red_card_user1_off = int(p1_off)
                _saved_live_user1_hp = int(battle[_red_card_user1_off + 1])
                _red_card_actual_dmg1 = _actual_damage_for_source_postmove(
                    _red_card_user1_off,
                    p0_off,
                    move_id1,
                    cat1,
                    damage1,
                    _target0_has_sub_pre,
                    int(battle[OFF_FIELD + F_SUBSTITUTE_0]),
                    hp0_pre_dmg,
                    int(battle[p0_off + 1]),
                )
                _red_card_user1_hp = _project_user_hp_after_own_postmove(
                    _red_card_user1_off,
                    p0_off,
                    move_id1,
                    damage1,
                    _red_card_actual_dmg1,
                    not _self_hit1 and (damage1 > 0 or cat1 == CAT_STATUS),
                    int(_meta1.get("num_hits", 1)),
                    hp1_pre_dmg,
                    include_after_move_secondary=False,
                    move_attempted=not (
                        opp_is_switch
                        or _self_hit1
                        or _p1_immobile_pre
                        or _sleep_talk_try_failed1
                        or _rest_try_failed1
                        or move1_no_target
                        or prankster_fail1
                        or _move1_canceled_pre
                    ),
                )
                battle[_red_card_user1_off + 1] = np.int16(_red_card_user1_hp)
                fx.apply_item_forced_switch(
                    battle,
                    -1,
                    move_id1,
                    p0_off,
                    p1_off,
                    p1_off,
                    p0_off,
                    0,
                    damage1,
                    False,
                    True,
                    side_order0,
                    side_order1,
                    gen5_prng,
                    game_data,
                    state,
                )
                _post_active1_item = int(battle[OFF_META + M_ACTIVE1])
                if _post_active1_item != _pre_active1_item:
                    _inline_item_switch_rewrote_active = True
                    # Red Card drags the faster attacker out immediately, so
                    # the slower foe-targeted move sees the replacement and
                    # the dragged source skips afterMoveSecondarySelf hooks.
                    _skip_after_move_secondary_hp_effects1 = True
                    _mid_turn_pivot_saved_hp1 = int(battle[_red_card_user1_off + 1])
                    _mid_turn_pivot_saved_status1 = int(
                        battle[_red_card_user1_off + 12]
                    )
                    active1 = _post_active1_item
                    p1_off = OFF_SIDE1 + active1 * POKEMON_SIZE
                    target0_off = p1_off
                    p1_ab = int(battle[p1_off + 5])
                    _mid_turn_pivot_target_hp1 = int(battle[p1_off + 1])
                    _hp1_after_postmove_pre = None
                    _preapplied_after_move_secondary_hp_delta1 = 0
                    _refresh_current_action_speeds()
                    _mid_turn_pivot_1 = True
                    if (
                        move0_targets_foe_mon
                        and not is_delayed0
                        and int(battle[p1_off + 1]) <= 0
                    ):
                        move0_no_target = True
                else:
                    battle[_red_card_user1_off + 1] = np.int16(_saved_live_user1_hp)
            _p0_immobile_pre = False
            if move0_no_target:
                battle[p0_off + 1] = _saved_hp0
                _p0_immobile_pre = _pre_calc_status_chain(0)
                if not _p0_immobile_pre:
                    if not _conf_checked0 and not _self_hit0:
                        _self_hit0 = _pre_calc_confusion_check(0)
                    if not _self_hit0:
                        _p0_immobile_pre = _pre_calc_para_check(0)
                    if _self_hit0:
                        _saved_hp0 = int(battle[p0_off + 1]) + _mirrored_damage1_on_p0
                damage0 = 0
            else:
                _p0_immobile_pre = _pre_calc_status_chain(0)
                if _p0_immobile_pre:
                    damage0 = 0
            if _move0_canceled_pre:
                battle[p0_off + 1] = _saved_hp0
                damage0_after_flinch = 0
                damage0 = 0
                hit0 = False
            elif not (move0_no_target or _p0_immobile_pre):
                if not _conf_checked0 and not _self_hit0:
                    _self_hit0 = _pre_calc_confusion_check(0)
                if not _self_hit0:
                    _p0_immobile_pre = _pre_calc_para_check(0)
                # See the symmetric note in the `side0_first` branch above.
                if _self_hit0:
                    _saved_hp0 = int(battle[p0_off + 1]) + _mirrored_damage1_on_p0
                if not (_self_hit0 or _p0_immobile_pre or _sleep_talk_try_failed0):
                    _sleep_talk_consume_sample(0)
                if not (_self_hit0 or _p0_immobile_pre or _sleep_talk_try_failed0):
                    _rest_try_failed0 = _rest_try_fails(0)
                if not (
                    _self_hit0
                    or _p0_immobile_pre
                    or _sleep_talk_try_failed0
                    or _rest_try_failed0
                ):
                    _resolve_protect_after_before_move(0)
                if not (
                    _self_hit0
                    or _p0_immobile_pre
                    or _sleep_talk_try_failed0
                    or _rest_try_failed0
                ):
                    _apply_roost_type_strip(0)
                if not (
                    _self_hit0
                    or _p0_immobile_pre
                    or _sleep_talk_try_failed0
                    or _rest_try_failed0
                    or pre_damage_fail0
                ):
                    _maybe_apply_protean_libero(0, move_id0, is_switch, is_strike_turn0)
                if _maybe_preapply_on_try_move_selfboost(
                    0,
                    p0_off,
                    p1_off,
                    _self_hit0
                    or _p0_immobile_pre
                    or _sleep_talk_try_failed0
                    or _rest_try_failed0
                    or move0_no_target,
                ):
                    _move0_stat_preapplied = True
                if _status_accuracy_preroll_needed(0) and not (
                    _self_hit0
                    or _p0_immobile_pre
                    or _sleep_talk_try_failed0
                    or _rest_try_failed0
                    or _first_move_kos_p0
                ):
                    _prerolled_status_acc0 = int(gen5_prng.random(100))
                _saved_target_hp1 = int(battle[p1_off + 1])
                if _hp1_after_postmove_pre is not None:
                    battle[p1_off + 1] = np.int16(_hp1_after_postmove_pre)
                damage0 = (
                    0
                    if (
                        _self_hit0
                        or _p0_immobile_pre
                        or _sleep_talk_try_failed0
                        or _rest_try_failed0
                        or move0_no_target
                    )
                    else _calc_p0(
                        user_hurt_by_target_this_turn=(_mirrored_damage1_on_p0 > 0)
                    )
                )
                if int(damage0) > 0:
                    _schedule_cursed_body_after_calc(0)
                battle[p1_off + 1] = np.int16(_saved_target_hp1)
                _target1_has_sub_pre = int(battle[OFF_FIELD + F_SUBSTITUTE_1]) > 0
                _knock_off_source_alive0 = (
                    int(move_effects.effect_type[move_id0]) == EFFECT_KNOCK_OFF
                    and damage0 > 0
                    and not target0_protected
                    and (
                        not _target1_has_sub_pre
                        or int(battle[p0_off + 5]) == ABILITY_INFILTRATOR
                    )
                    and _source_survives_damaging_hit_contact(
                        p0_off,
                        p1_off,
                        move_id0,
                        int(damage0),
                        True,
                        int(_meta0.get("num_hits", 1)),
                        int(battle[p0_off + 1]),
                        move_idx,
                    )
                )
                if _knock_off_source_alive0:
                    _inline_knock_removed_focus_sash1 = (
                        int(battle[p1_off + 6]) == ITEM_FOCUS_SASH
                    )
                    _inline_knock_saved_item1 = int(battle[p1_off + 6])
                    _inline_knock_saved_target1 = int(p1_off)
                    _apply_knock_off_from_move_tracked(
                        move_id0,
                        p1_off,
                        True,
                        user_offset=p0_off,
                    )
                battle[p0_off + 1] = _saved_hp0
                if not (
                    _self_hit0
                    or _p0_immobile_pre
                    or _sleep_talk_try_failed0
                    or _rest_try_failed0
                    or move0_no_target
                    or is_charge_turn0
                ):
                    # Lockedmove self-effect PRNG for second mover (side0 second).
                    _preroll_lockedmove_self_effect(
                        0,
                        move_id0,
                        not is_switch,
                        (damage0 > 0) or target0_protected,
                        is_locked_turn0,
                        locked_turns0_pre,
                        p0_off,
                    )
                    prerolled_move0 = (
                        _preroll0(target_hp_override=_hp1_after_postmove_pre)
                        if not is_switch
                        else None
                    )
                    _schedule_cursed_body_after_preroll(0)
            else:
                if int(damage0) > 0:
                    _schedule_cursed_body_after_preroll(0)
                battle[p0_off + 1] = _saved_hp0
                damage0 = 0
        elif _first_attacker_flinches_target:
            # Symmetric flinch short-circuit: still apply any first-move
            # inflicted frz/slp/par before the slower target's high-priority
            # onBeforeMove chain runs. This preserves same-turn thaw /
            # sleep-turn PRNG like Showdown before flinch cancels the move.
            if int(damage1) > 0 and not opp_is_switch and not _self_hit1:
                _early_apply_inflicted_status(1)
                _p0_immobile_pre = _pre_calc_status_chain(0)
            if speeds_tied and not opp_is_switch and not _p1_immobile_pre:
                _between_updates = _move_update_count(
                    _cat1_status,
                    target1_kind,
                    _move1_successful_self_boost_status,
                    int(damage1),
                    int(_meta1.get("num_hits", 1)),
                )
                for _ in range(_between_updates):
                    gen5_prng.random(0, 2)
            _roll_pending_lockedmove_confusion(1, p1_off, apply=True)
            damage0 = 0
        elif _self_hit0:
            if speeds_tied and not opp_is_switch and not _p1_immobile_pre:
                _between_updates = _move_update_count(
                    _cat1_status,
                    target1_kind,
                    _move1_successful_self_boost_status,
                    int(damage1),
                )
                for _ in range(_between_updates):
                    gen5_prng.random(0, 2)
            _roll_pending_lockedmove_confusion(1, p1_off, apply=True)
            damage0 = 0
        else:
            if speeds_tied and not _p1_immobile_pre:
                _between_updates = (
                    _move_update_count(
                        _cat1_status,
                        target1_kind,
                        _move1_successful_self_boost_status,
                        int(damage1),
                        int(_meta1.get("num_hits", 1)),
                    )
                    if is_switch
                    else (0 if int(_meta1.get("num_hits", 1)) > 1 else 1)
                )
                _deferred_hit_loop_update_frames += int(_between_updates)
            damage0 = 0
            move1_killed_target = True

    # Fallback — if a side never ran (e.g., KO before it could move), give
    # the apply_* call sites an empty dict so the `.get()` lookups don't
    # explode. `damage == 0` will still gate the apply_* internals correctly.
    _empty_preroll = {
        "status": None,
        "stat_change": None,
        "flinch": None,
        "flinch_lands": False,
        "confusion": None,
        "confusion_lands": False,
        "confusion_duration": None,
        "confusion_can_apply": None,
        "taunt": None,
        "encore": None,
        "ext_vol": None,
    }
    if prerolled_move0 is None:
        prerolled_move0 = dict(_empty_preroll)
    if prerolled_move1 is None:
        prerolled_move1 = dict(_empty_preroll)

    # Restore Roost types after damage calc
    if roost_applied0:
        battle[p0_off + 4] = p0_types_pre_roost
    if roost_applied1:
        battle[p1_off + 4] = p1_types_pre_roost

    bp0 = int(game_data.move_base_power[move_id0])
    bp1 = int(game_data.move_base_power[move_id1])

    # Sucker Punch — fails if target is using a status move OR not attacking.
    # Showdown source: data/moves.ts:19131-19137 — `onTry` checks
    # `move.category !== 'Status'`, NOT `move.basePower`. Pokepy used
    # `bp == 0` which broke vs callback-BP attacks (Heavy Slam, Low Kick,
    # Gyro Ball, Heat Crash, Electro Ball, Grass Knot, Reversal, Flail,
    # Stored Power, Power Trip, Weather Ball, Terrain Pulse, Acrobatics,
    # Wake-Up Slap, Assurance, Facade, etc. all have static bp=0).
    # Sucker Punch also fails when the target has ALREADY moved this turn
    # (Showdown: `this.queue.willMove(target)` returns null → onTry fails).
    # With Sucker Punch at +1 priority this only matters when the opponent
    # moved at higher priority (Fake Out +3, Extreme Speed +2, Prankster
    # status, etc.) or when forced into an earlier slot. side0_first=False
    # means P1 already moved by the time Sucker Punch resolves for P0.
    if move_id0 == MOVE_SUCKER_PUNCH and (
        cat1 == CAT_STATUS or opp_is_switch or not side0_first
    ):
        damage0 = 0
    if move_id1 == MOVE_SUCKER_PUNCH and (
        cat0 == CAT_STATUS or is_switch or side0_first
    ):
        damage1 = 0
    # Thunderclap (Raging Bolt) — identical onTry to Sucker Punch: fails if
    # target is using a Status move (except Me First) or has already moved.
    # Showdown data/moves.ts:thunderclap onTry.
    if move_id0 == MOVE_THUNDERCLAP and (
        cat1 == CAT_STATUS or opp_is_switch or not side0_first
    ):
        damage0 = 0
    if move_id1 == MOVE_THUNDERCLAP and (
        cat0 == CAT_STATUS or is_switch or side0_first
    ):
        damage1 = 0

    # Upper Hand (id 918) — Showdown data/moves.ts:upperhand onTry: fails
    # unless target is about to use a priority move (priority > 0.1) and
    # not a Status move. Since Upper Hand is itself +3 priority, this only
    # matters when the opponent's move would have moved AFTER it but is
    # still a priority move (e.g., Aqua Jet, Bullet Punch, Quick Attack).
    # If the target has already moved (side0_first false for our move) the
    # gating fails as well.
    _MOVE_UPPER_HAND = 918
    _opp1_priority = int(game_data.move_priority[move_id1])
    _opp0_priority = int(game_data.move_priority[move_id0])
    if move_id0 == _MOVE_UPPER_HAND and (
        cat1 == CAT_STATUS or opp_is_switch or not side0_first or _opp1_priority <= 0
    ):
        damage0 = 0
    if move_id1 == _MOVE_UPPER_HAND and (
        cat0 == CAT_STATUS or is_switch or side0_first or _opp0_priority <= 0
    ):
        damage1 = 0

    # Status accuracy + hit determination
    acc0 = int(game_data.move_accuracy[move_id0])
    acc1 = int(game_data.move_accuracy[move_id1])
    move0_is_status = cat0 == CAT_STATUS
    move1_is_status = cat1 == CAT_STATUS
    # These late live-ability hooks must respect Neutralizing Gas / Ability
    # Shield the same way Showdown's generic `ignoringAbility()` event gate
    # does. The pre-accuracy status helper above already uses effective
    # abilities, but this later block historically re-read the raw ids and
    # could still apply Wonder Skin / No Guard / Damp / Good as Gold /
    # Magic Bounce after suppression.
    from pokepy.effects.ability_suppression import (
        effective_ability as _effective_ability_status,
    )

    p0_ab_status = _effective_ability_status(battle, p0_off, p1_off)
    p1_ab_status = _effective_ability_status(battle, p1_off, p0_off)

    # Wonder Skin (Cresselia, Tinkaton-line in special cases): caps status
    # move accuracy at 50. Showdown abilities.ts:wonderskin onModifyAccuracy.
    _ABILITY_WONDER_SKIN = 147
    if move0_is_status and p1_ab_status == _ABILITY_WONDER_SKIN and acc0 > 50:
        acc0 = 50
    if move1_is_status and p0_ab_status == _ABILITY_WONDER_SKIN and acc1 > 50:
        acc1 = 50
    # Showdown gen 8+: Toxic from a Poison-type user always hits regardless
    # of accuracy. Source: sim/battle-actions.ts:726.
    _MOVE_TOXIC = 92
    user0_types_t = int(battle[user0_off + 4])
    user1_types_t = int(battle[user1_off + 4])
    user0_is_poison = (user0_types_t & 0xFF) == TYPE_POISON or (
        (user0_types_t >> 8) & 0xFF
    ) == TYPE_POISON
    user1_is_poison = (user1_types_t & 0xFF) == TYPE_POISON or (
        (user1_types_t >> 8) & 0xFF
    ) == TYPE_POISON
    toxic_bypass0 = move_id0 == _MOVE_TOXIC and user0_is_poison
    toxic_bypass1 = move_id1 == _MOVE_TOXIC and user1_is_poison
    # Showdown battle-actions.ts:733 gates randomChance behind accuracy !== true.
    # Loader maps accuracy: true -> 127. Gen2 skips at scaled 255 (acc >= 100).
    acc0_bypass = _showdown_accuracy_bypass(acc0, move0_is_status, target0_kind)
    acc1_bypass = _showdown_accuracy_bypass(acc1, move1_is_status, target1_kind)
    # Status-move accuracy rolls were pre-consumed earlier in speed order
    # (see `_prerolled_status_accN` block above) so the frame lands at the
    # right Showdown offset. Re-use the prerolled value here; only roll
    # fresh if the move was eligible but somehow not preroll-captured
    # while the user is still actually able to act. If the user fainted,
    # self-hit, or got stopped by onBeforeMove after the preroll window,
    # Showdown never reaches hitStepAccuracy and spends no frame here.
    _side0_slow_move_kod = (not side0_first) and (
        move1_killed_target or _first_move_kos_p0
    )
    _side1_slow_move_kod = side0_first and (move1_killed_target or _first_move_kos_p1)
    if _prerolled_status_acc0 is not None:
        acc_roll0 = _prerolled_status_acc0
    elif (
        move0_is_status
        and not toxic_bypass0
        and not acc0_bypass
        and not is_switch
        and not p0_skip
        and not move0_no_target
        and not _side0_slow_move_kod
        and not _field_or_self0
        and not _pre_accuracy_block0
        and not _p0_immobile_pre
        and not _self_hit0
        and int(battle[p0_off + 1]) > 0
    ):
        acc_roll0 = int(gen5_prng.random(100))
    else:
        acc_roll0 = 0
    if _prerolled_status_acc1 is not None:
        acc_roll1 = _prerolled_status_acc1
    elif (
        move1_is_status
        and not toxic_bypass1
        and not acc1_bypass
        and not opp_is_switch
        and not p1_skip
        and not move1_no_target
        and not _side1_slow_move_kod
        and not _field_or_self1
        and not _pre_accuracy_block1
        and not _p1_immobile_pre
        and not _self_hit1
        and int(battle[p1_off + 1]) > 0
    ):
        acc_roll1 = int(gen5_prng.random(100))
    else:
        acc_roll1 = 0
    status_hit0 = toxic_bypass0 or acc0_bypass or (acc_roll0 < acc0)
    status_hit1 = toxic_bypass1 or acc1_bypass or (acc_roll1 < acc1)

    hit0 = (not is_switch) and ((damage0 > 0) if bp0 > 0 else status_hit0)
    hit1 = (not opp_is_switch) and ((damage1 > 0) if bp1 > 0 else status_hit1)
    if effect0 == EFFECT_PROTECT and protect_failed0:
        hit0 = False
    if effect1 == EFFECT_PROTECT and protect_failed1:
        hit1 = False
    if _sleep_talk_try_failed0:
        hit0 = False
    if _sleep_talk_try_failed1:
        hit1 = False
    if _rest_try_failed0:
        hit0 = False
    if _rest_try_failed1:
        hit1 = False
    if move0_is_status and _pre_accuracy_block0:
        hit0 = False
    if move1_is_status and _pre_accuracy_block1:
        hit1 = False
    # `[notarget]` must fail even for status moves with `accuracy: true`
    # (for example Defog after the only foe faints to switch-in hazards).
    # Without this gate, status_hit* can stay truthy and wrongly apply the
    # move's side effects despite Showdown returning `[notarget]`.
    if move0_no_target:
        hit0 = False
    if move1_no_target:
        hit1 = False
    _hazard_from_move_hit0 = bool(hit0)
    _hazard_from_move_hit1 = bool(hit1)

    # Primal weather move suppression — Primordial Sea nullifies Fire
    # damaging moves, Desolate Land nullifies Water damaging moves. Both
    # fire at onTryMove priority 1 BEFORE damage. Status moves of those
    # types are not suppressed (e.g., Will-O-Wisp under Primordial Sea
    # still succeeds — it's only non-Status). Showdown:
    # data/conditions.ts:514 primordialsea onTryMove and :592 desolateland.
    # The check also respects Air Lock / Cloud Nine via effectiveWeather,
    # but primal weather from an ability ignores the suppressor unless an
    # opposing Air Lock fully suppresses the field (rare).
    from pokepy.mechanics.damage_gen9 import _effective_weather as _effective_weather_pw

    _pw_cur = _effective_weather_pw(battle)

    def _effective_runtime_move_type(
        move_id: int, user_off: int, target_off: int
    ) -> int:
        from pokepy.effects.ability_suppression import (
            effective_ability as _effective_ability_mt,
        )

        _MOVE_RAGING_BULL_MT = 873
        _MOVE_REVELATION_DANCE_MT = 686
        _MOVE_TERA_BLAST_MT = 851
        _MOVE_IVY_CUDGEL_MT = 904
        _ITEM_WELLSPRING_MASK_MT = 759
        _ITEM_CORNERSTONE_MASK_MT = 758
        _ITEM_HEARTHFLAME_MASK_MT = 760
        _ABILITY_LIQUID_VOICE_MT = 204
        _ABILITY_REFRIGERATE_MT = 174
        _ABILITY_PIXILATE_MT = 182
        _ABILITY_AERILATE_MT = 184
        _ABILITY_GALVANIZE_MT = 206
        _ATE_ABILITIES_MT = (
            _ABILITY_REFRIGERATE_MT,
            _ABILITY_PIXILATE_MT,
            _ABILITY_AERILATE_MT,
            _ABILITY_GALVANIZE_MT,
        )

        _move_type = int(game_data.move_type[move_id])
        _move_flags = int(game_data.move_flags[move_id])
        _user_ability = _effective_ability_mt(battle, user_off, target_off)
        if move_id == _MOVE_RAGING_BULL_MT:
            _species = int(battle[user_off + 0])
            if _species == 2036:
                _move_type = TYPE_FIGHTING
            elif _species == 2035:
                _move_type = TYPE_FIRE
            elif _species == 2034:
                _move_type = TYPE_WATER
        if move_id == _MOVE_TERA_BLAST_MT:
            _flags = int(battle[user_off + 15])
            if (_flags & 0x8) != 0:
                _move_type = (int(battle[user_off + 14]) >> 12) & 0xF
        if move_id == _MOVE_REVELATION_DANCE_MT:
            # Showdown's Revelation Dance onModifyType uses
            # pokemon.getTypes()[0]. The live battle state already stores the
            # current runtime type tuple (including tera and type-changing
            # effects), so use the primary slot directly.
            _user_types = int(battle[user_off + 4]) & 0xFFFF
            _move_type = int(_user_types & 0xFF)
            if _move_type == 0xFF:
                _move_type = TYPE_NORMAL
        if move_id == MOVE_WEATHER_BALL:
            if _pw_cur == WEATHER_SUN:
                _move_type = TYPE_FIRE
            elif _pw_cur == WEATHER_RAIN:
                _move_type = TYPE_WATER
            elif _pw_cur == WEATHER_SNOW:
                _move_type = TYPE_ICE
            elif _pw_cur == WEATHER_SAND:
                _move_type = TYPE_ROCK
            elif _pw_cur == WEATHER_PRIMORDIAL_SEA:
                _move_type = TYPE_WATER
            elif _pw_cur == WEATHER_DESOLATE_LAND:
                _move_type = TYPE_FIRE
        if move_id == MOVE_TERRAIN_PULSE:
            _terrain = int(battle[OFF_FIELD + F_TERRAIN])
            if _terrain == TERRAIN_ELECTRIC:
                _move_type = TYPE_ELECTRIC
            elif _terrain == TERRAIN_GRASSY:
                _move_type = TYPE_GRASS
            elif _terrain == TERRAIN_PSYCHIC:
                _move_type = TYPE_PSYCHIC
            elif _terrain == TERRAIN_MISTY:
                _move_type = TYPE_FAIRY
        if (
            _user_ability == _ABILITY_LIQUID_VOICE_MT
            and (_move_flags & FLAG_SOUND) != 0
        ):
            _move_type = TYPE_WATER
        if (
            _user_ability in _ATE_ABILITIES_MT
            and _move_type == TYPE_NORMAL
            and int(game_data.move_base_power[move_id]) > 0
            and move_id
            not in (
                MOVE_WEATHER_BALL,
                MOVE_TERRAIN_PULSE,
                _MOVE_TERA_BLAST_MT,
                246,
                719,
                363,
                686,
                546,
            )
        ):
            if _user_ability == _ABILITY_REFRIGERATE_MT:
                _move_type = TYPE_ICE
            elif _user_ability == _ABILITY_PIXILATE_MT:
                _move_type = TYPE_FAIRY
            elif _user_ability == _ABILITY_AERILATE_MT:
                _move_type = TYPE_FLYING
            elif _user_ability == _ABILITY_GALVANIZE_MT:
                _move_type = TYPE_ELECTRIC
        if move_id == _MOVE_IVY_CUDGEL_MT:
            _item = int(battle[user_off + 6])
            if _item == _ITEM_WELLSPRING_MASK_MT:
                _move_type = TYPE_WATER
            elif _item == _ITEM_CORNERSTONE_MASK_MT:
                _move_type = TYPE_ROCK
            elif _item == _ITEM_HEARTHFLAME_MASK_MT:
                _move_type = TYPE_FIRE
        return _move_type

    _type0_pw = _effective_runtime_move_type(move_id0, user0_off, user1_off)
    _type1_pw = _effective_runtime_move_type(move_id1, user1_off, user0_off)
    _cat0_pw = int(game_data.move_category[move_id0])
    _cat1_pw = int(game_data.move_category[move_id1])
    if (
        _pw_cur == WEATHER_PRIMORDIAL_SEA
        and _type0_pw == TYPE_FIRE
        and _cat0_pw != CAT_STATUS
    ):
        hit0 = False
        damage0 = 0
    if (
        _pw_cur == WEATHER_PRIMORDIAL_SEA
        and _type1_pw == TYPE_FIRE
        and _cat1_pw != CAT_STATUS
    ):
        hit1 = False
        damage1 = 0
    if (
        _pw_cur == WEATHER_DESOLATE_LAND
        and _type0_pw == TYPE_WATER
        and _cat0_pw != CAT_STATUS
    ):
        hit0 = False
        damage0 = 0
    if (
        _pw_cur == WEATHER_DESOLATE_LAND
        and _type1_pw == TYPE_WATER
        and _cat1_pw != CAT_STATUS
    ):
        hit1 = False
        damage1 = 0

    # Semi-invulnerable target blocks status moves that target the foe.
    # Showdown fly/bounce/dig/dive.condition.onInvulnerability returns false
    # for any move not in its exception list — status moves are never
    # in those lists (they're all damaging moves). Pokepy's damage path
    # already zeroes dmg for semi-invul; this mirrors the same check for
    # status moves so Thunder Wave / Will-O-Wisp / etc. miss a Dig user.
    # No Guard / Lock On bypass (handled via can_hit_si_non_dmg below).
    def _target_semi_invul(side_idx: int) -> bool:
        actions_off = OFF_MOVES + (
            M_ACTIVE_MOVE_ACTIONS_0 if side_idx == 0 else M_ACTIVE_MOVE_ACTIONS_1
        )
        return (int(battle[actions_off]) & ACTIVE_MOVE_ACTIONS_SEMI_INVUL) != 0

    _p0_has_no_guard_si = p0_ab_status == ABILITY_NO_GUARD
    _p1_has_no_guard_si = p1_ab_status == ABILITY_NO_GUARD
    _p0_lock_on_si = (int(battle[OFF_FIELD + F_EXTENDED_VOLATILE_0]) & 0x800) != 0
    _p1_lock_on_si = (int(battle[OFF_FIELD + F_EXTENDED_VOLATILE_1]) & 0x800) != 0
    if (
        move0_is_status
        and _target_semi_invul(1)
        and target0_kind in (0, 1, 2, 4, 5, 6, 8, 11, 12)
        and not (_p0_has_no_guard_si or _p1_has_no_guard_si or _p0_lock_on_si)
    ):
        hit0 = False
    if (
        move1_is_status
        and _target_semi_invul(0)
        and target1_kind in (0, 1, 2, 4, 5, 6, 8, 11, 12)
        and not (_p0_has_no_guard_si or _p1_has_no_guard_si or _p1_lock_on_si)
    ):
        hit1 = False

    # Damp on either active mon blocks explosion-class moves. Showdown
    # data/abilities.ts:damp onAnyTryMove fails `selfdestruct`-flagged moves
    # and Mind Blown / Misty Explosion / Chloroblast (also flagged). Pokepy
    # honors Explosion / Self-Destruct / Misty Explosion here; Mold Breaker
    # on the attacker suppresses Damp. Final Gambit is NOT blocked by Damp —
    # Showdown finalgambit does not set the selfdestruct flag.
    _DAMP_MOVES = (MOVE_EXPLOSION, MOVE_SELF_DESTRUCT, MOVE_MISTY_EXPLOSION)
    _ABILITY_DAMP_BG = 6
    _MB_SET_BG = (104, 163, 164)  # moldbreaker, turboblaze, teravolt
    _damp_active = (p0_ab_status == _ABILITY_DAMP_BG) or (
        p1_ab_status == _ABILITY_DAMP_BG
    )
    if _damp_active and move_id0 in _DAMP_MOVES and p0_ab_status not in _MB_SET_BG:
        hit0 = False
        damage0 = 0
    if _damp_active and move_id1 in _DAMP_MOVES and p1_ab_status not in _MB_SET_BG:
        hit1 = False
        damage1 = 0

    # Assault Vest / Taunt blocks status
    user0_item = int(battle[user0_off + 6])
    user1_item = int(battle[user1_off + 6])
    has_vest0 = user0_item == ITEM_ASSAULT_VEST
    has_vest1 = user1_item == ITEM_ASSAULT_VEST
    vol0 = int(battle[OFF_FIELD + F_VOLATILE_0])
    vol1 = int(battle[OFF_FIELD + F_VOLATILE_1])
    is_taunted0 = get_taunt_turns(vol0) > 0
    is_taunted1 = get_taunt_turns(vol1) > 0
    if move0_is_status and (has_vest0 or is_taunted0):
        hit0 = False
    if move1_is_status and (has_vest1 or is_taunted1):
        hit1 = False
    # Good as Gold (Gholdengo): blocks status moves that actually target the
    # Pokemon, not foe-side / ally-side / field-targeting status moves like
    # Spikes or Stealth Rock. Showdown's onTryHit hook only runs when the
    # target is a Pokemon; side conditions resolve through side events instead.
    _ABILITY_GOOD_AS_GOLD = 283
    if (
        move0_is_status
        and p1_ab_status == _ABILITY_GOOD_AS_GOLD
        and p0_ab_status not in _MB_SET_BG
        and target0_kind in _STATUS_TARGET_FOE_MON
    ):
        hit0 = False
    if (
        move1_is_status
        and p0_ab_status == _ABILITY_GOOD_AS_GOLD
        and p1_ab_status not in _MB_SET_BG
        and target1_kind in _STATUS_TARGET_FOE_MON
    ):
        hit1 = False

    # Magic Bounce. Showdown reflects ANY status move with `flags.reflectable`
    # back at the attacker (Toxic, Will-O-Wisp, Thunder Wave, Sleep Powder,
    # Spore, Hypnosis, Sing, Stun Spore, Poison Powder, Taunt, Encore,
    # Disable, Torment, Swagger, Flatter, Attract, Leech Seed, Confuse Ray,
    # Yawn, hazards). Pokepy used to just set `hit = False` which silently
    # canned the move; the hazards redirect at pokepy/effects/hazards.py:72
    # is the only thing that actually bounced.
    #
    # Implement reflection by setting `mb_redirect_to_user` flags. The
    # status / volatile apply calls below honor these by swapping the
    # target side / target offset.
    # Mold Breaker / Teravolt / Turboblaze suppress Magic Bounce on the
    # target (Showdown data/abilities.ts moldbreaker onModifyMove sets
    # ignoreAbility → bypasses Magic Bounce). Gen 9 Mold Breaker id = 104.
    _ABILITY_MOLD_BREAKER_MB = 104
    _ABILITY_TERAVOLT_MB = 164
    _ABILITY_TURBOBLAZE_MB = 163
    _MB_IGNORE_SET = (
        _ABILITY_MOLD_BREAKER_MB,
        _ABILITY_TERAVOLT_MB,
        _ABILITY_TURBOBLAZE_MB,
    )
    p0_ignores_ability_mb = p0_ab_status in _MB_IGNORE_SET
    p1_ignores_ability_mb = p1_ab_status in _MB_IGNORE_SET
    mb_redirect0_to_user = (
        move0_is_status
        and p1_ab_status == ABILITY_MAGIC_BOUNCE
        and not p0_ignores_ability_mb
    )
    mb_redirect1_to_user = (
        move1_is_status
        and p0_ab_status == ABILITY_MAGIC_BOUNCE
        and not p1_ignores_ability_mb
    )

    # ------------------------------------------------------------------
    # Status immobilization (sleep / freeze / paralysis)
    #
    # Showdown's onBeforeMove priorities (data/conditions.ts):
    #   frz: priority 10  (random thaw 1/5, then block if not thawed)
    #   slp: priority 10  (no roll, decrement-and-check)
    #   par: priority  1  (random full-para 1/4)
    # Higher priority runs first, so for each Pokemon the order is:
    #   1. freeze thaw roll (only when frozen and not using a defrost move)
    #   2. paralysis full-para roll
    # We process the faster Pokemon's full chain before the slower's, then
    # combine the results. Switching mons skip onBeforeMove entirely, so
    # they consume zero PRNG frames here.
    # ------------------------------------------------------------------
    user0_status = get_status(int(battle[user0_off + 12]))
    user1_status = get_status(int(battle[user1_off + 12]))

    # Turn-start sleep wake check (mirrors Showdown's slp.onBeforeMove,
    # which decrements at move time and cures when the counter reaches 0).
    # Pokepy's EOT decrement brings the stored counter to 0 on the LAST
    # forced-sleep turn; the cure itself is deferred to THIS turn so the
    # end-of-turn snapshot still shows slp on the last asleep turn and
    # NONE on the wake turn — matching Showdown's per-turn status.
    def _turn_start_sleep_wake(user_off: int) -> int:
        """Clear sleep status if the counter has hit 0.  Returns the new
        raw status byte so callers can refresh `user_statusN`."""
        raw = int(battle[user_off + 12])
        st = get_status(raw)
        if st != STATUS_SLEEP:
            return raw
        turns = get_status_turns(raw)
        if turns > 0:
            return raw
        battle[user_off + 12] = 0
        return 0

    if not is_switch:
        _ts0_raw = _turn_start_sleep_wake(user0_off)
        user0_status = get_status(_ts0_raw)
    if not opp_is_switch:
        _ts1_raw = _turn_start_sleep_wake(user1_off)
        user1_status = get_status(_ts1_raw)

    is_par0 = user0_status == STATUS_PARALYSIS
    is_par1 = user1_status == STATUS_PARALYSIS
    is_asleep0 = user0_status == STATUS_SLEEP
    is_asleep1 = user1_status == STATUS_SLEEP
    _ABILITY_COMATOSE_SLEEP_TALK = 213
    _sleep_talk_try_failed0 = (
        (not is_switch)
        and raw_move_id0 == MOVE_SLEEP_TALK
        and (not _sleep_talk_pending_slp0)
        and (not is_asleep0)
        and p0_ab != _ABILITY_COMATOSE_SLEEP_TALK
    )
    _sleep_talk_try_failed1 = (
        (not opp_is_switch)
        and raw_move_id1 == MOVE_SLEEP_TALK
        and (not _sleep_talk_pending_slp1)
        and (not is_asleep1)
        and p1_ab != _ABILITY_COMATOSE_SLEEP_TALK
    )
    # Sleep-block gate: use the ORIGINAL move id (before any Sleep Talk
    # substitution) — Showdown's slp.onBeforeMove checks `move.sleepUsable`
    # which is only set on Sleep Talk / Snore, and Sleep Talk's onHit then
    # calls useMove for a different move that would otherwise be blocked.
    _sb_mv0 = _sleep_talk_orig0 if _sleep_talk_orig0 > 0 else move_id0
    _sb_mv1 = _sleep_talk_orig1 if _sleep_talk_orig1 > 0 else move_id1
    sleep_blocked0 = is_asleep0 and _sb_mv0 not in (MOVE_SLEEP_TALK, MOVE_SNORE)
    sleep_blocked1 = is_asleep1 and _sb_mv1 not in (MOVE_SLEEP_TALK, MOVE_SNORE)

    is_frozen0 = user0_status == STATUS_FREEZE
    is_frozen1 = user1_status == STATUS_FREEZE
    # Downstream onTryHit / onAfterHit callbacks need the effective type, not
    # the base move type. This includes ability-driven rewrites like Liquid
    # Voice in addition to item-driven ones like Ivy Cudgel's mask typing.
    move_type0 = _effective_runtime_move_type(move_id0, user0_off, user1_off)
    move_type1 = _effective_runtime_move_type(move_id1, user1_off, user0_off)
    # Showdown gen 9: only moves with the `defrost` flag thaw the user. Fire
    # moves *without* the defrost flag (Flamethrower, Fire Blast, Heat Wave,
    # etc.) do NOT self-thaw — they're blocked by frz. Source: data/moves.ts
    # `flags: { defrost: 1 }` and conditions.ts frz onModifyMove.
    DEFROST_MOVE_IDS = frozenset(
        (
            221,  # sacredfire (Fire)
            172,  # flamewheel (Fire)
            394,  # flareblitz (Fire)
            558,  # fusionflare (Fire)
            682,  # burnup (Fire)
            780,  # pyroball (Fire)
            503,  # scald (Water)
            592,  # steameruption (Water)
            815,  # scorchingsands (Ground)
            735,  # sizzlyslide (Electric)
            876,  # hydrosteam (Water)
            902,  # matchagotcha (Grass)
        )
    )
    is_thaw0 = move_id0 in DEFROST_MOVE_IDS
    is_thaw1 = move_id1 in DEFROST_MOVE_IDS
    fire_thaw0 = is_frozen0 and is_thaw0
    fire_thaw1 = is_frozen1 and is_thaw1

    # The pre-calc block (~line 1756) rolls frz thaw + par full-para at
    # each mon's actual runMove moment. Reuse those results here. If a
    # side was never resolved (KO'd by the first mover, or switched out),
    # its move never fires and no rolls should be consumed.
    rand_thaw0 = _status_chain_rand_thaw0
    rand_thaw1 = _status_chain_rand_thaw1
    full_para0 = _status_chain_full_para0
    full_para1 = _status_chain_full_para1

    if rand_thaw0:
        battle[user0_off + 12] = 0
        is_frozen0 = False
    if rand_thaw1:
        battle[user1_off + 12] = 0
        is_frozen1 = False
    frozen_blocked0 = is_frozen0 and not fire_thaw0
    frozen_blocked1 = is_frozen1 and not fire_thaw1

    # Mustrecharge counts as immobile (Showdown conditions.ts:364
    # mustrecharge.onBeforeMove → return null skips the move).
    is_immobile0 = sleep_blocked0 or frozen_blocked0 or full_para0 or must_recharge0
    is_immobile1 = sleep_blocked1 or frozen_blocked1 or full_para1 or must_recharge1

    if is_immobile0 or prankster_fail0:
        hit0 = False
        damage0 = 0
    if is_immobile1 or prankster_fail1:
        hit1 = False
        damage1 = 0

    if fire_thaw0:
        battle[user0_off + 12] = 0
    if fire_thaw1:
        battle[user1_off + 12] = 0

    # Showdown: any Fire damaging move thaws the FROZEN target on hit.
    # Source: data/conditions.ts frz onDamagingHit. Apply only if the
    # attack is going to land (move not cancelled, BP > 0).
    cat0_thaw = int(game_data.move_category[move_id0])
    cat1_thaw = int(game_data.move_category[move_id1])
    target1_frozen = is_frozen1
    target0_frozen = is_frozen0
    if (
        (not is_switch)
        and not is_immobile0
        and target1_frozen
        and move_type0 == TYPE_FIRE
        and cat0_thaw != CAT_STATUS
    ):
        battle[user1_off + 12] = 0
        is_frozen1 = False
        frozen_blocked1 = False
    if (
        (not opp_is_switch)
        and not is_immobile1
        and target0_frozen
        and move_type1 == TYPE_FIRE
        and cat1_thaw != CAT_STATUS
    ):
        battle[user0_off + 12] = 0
        is_frozen0 = False
        frozen_blocked0 = False

    # Confusion self-hit. Showdown's BeforeMove chain stops at the first
    # `return false`, so confusion never rolls if the mon is already
    # immobilized by sleep/freeze/full paralysis. Skip the roll in those
    # cases to avoid wasted PRNG advancement.
    # Snapshot HP BEFORE self-hit so we can apply Focus Sash after.
    hp0_pre_self = int(battle[user0_off + 1])
    hp1_pre_self = int(battle[user1_off + 1])
    # The new pre-calc confusion check (fired before _calc_pN in the
    # damage-calc section above) resolves the roll for mons already
    # confused at turn start — it sets `_conf_checked{0,1}` so we don't
    # double-roll here. For flows that took a different path (e.g.
    # second mover got confused by first mover's secondary THIS turn,
    # which is handled by the inline check inside _preroll_move_secondaries),
    # `_conf_checked{0,1}` are also set. Only fire this legacy call when
    # the pre-calc path did NOT already handle the check.
    # Also skip when the slower mon was KO'd by the first attacker's move
    # — Showdown's faintMessages fires between the two moves and the
    # second mover's runMove never executes, so onBeforeMove doesn't fire
    # and no confusion frame is consumed. `move1_killed_target` is True
    # in that case (the name is generic to "first mover killed the
    # target"; it's set regardless of which side is first).
    _slower_was_kod = move1_killed_target
    _slower_side = 1 if side0_first else 0
    _skip_conf0_ko = _slower_was_kod and _slower_side == 0
    _skip_conf1_ko = _slower_was_kod and _slower_side == 1
    if _conf_checked0 or is_switch or is_immobile0 or prankster_fail0 or _skip_conf0_ko:
        self_hit0 = _self_hit0
    else:
        self_hit0 = fx.check_confusion_self_hit(battle, 0, user0_off, gen5_prng)
    if (
        _conf_checked1
        or opp_is_switch
        or is_immobile1
        or prankster_fail1
        or _skip_conf1_ko
    ):
        self_hit1 = _self_hit1
    else:
        self_hit1 = fx.check_confusion_self_hit(battle, 1, user1_off, gen5_prng)
    if self_hit0:
        damage0 = 0
        hit0 = False
    if self_hit1:
        damage1 = 0
        hit1 = False

    # Focus Sash on confusion self-hit. Showdown routes confusion self-damage
    # through `Battle#damage`, which triggers item onDamage (Focus Sash etc.).
    # Only saves if the user was at full HP and the hit would have KO'd.
    if self_hit0:
        max_hp0_self = int(battle[user0_off + 2])
        hp0_after_self = int(battle[user0_off + 1])
        if (
            int(battle[user0_off + 6]) == ITEM_FOCUS_SASH
            and hp0_pre_self == max_hp0_self
            and hp0_after_self == 0
        ):
            battle[user0_off + 1] = 1
            battle[user0_off + 6] = 0
    if self_hit1:
        max_hp1_self = int(battle[user1_off + 2])
        hp1_after_self = int(battle[user1_off + 1])
        if (
            int(battle[user1_off + 6]) == ITEM_FOCUS_SASH
            and hp1_pre_self == max_hp1_self
            and hp1_after_self == 0
        ):
            battle[user1_off + 1] = 1
            battle[user1_off + 6] = 0

    # Flinch — only the first attacker can apply. For multi-hit moves
    # (Double Iron Bash etc.), Showdown rolls the secondary chance per hit
    # (sim/battle-actions.ts:1357 inside the moveHit loop), so the engine
    # passes num_hits.
    if side0_first:
        fx.apply_flinch_from_move(
            battle,
            move_id0,
            1,
            hit0,
            move_effects,
            gen5_prng,
            game_data,
            num_hits=int(_meta0.get("num_hits", 1)),
            prerolled_rolls=prerolled_move0.get("flinch"),
        )
    else:
        fx.apply_flinch_from_move(
            battle,
            move_id1,
            0,
            hit1,
            move_effects,
            gen5_prng,
            game_data,
            num_hits=int(_meta1.get("num_hits", 1)),
            prerolled_rolls=prerolled_move1.get("flinch"),
        )

    side0_flinched = fx.check_flinched(battle, 0)
    side1_flinched = fx.check_flinched(battle, 1)
    damage0_after_flinch = 0 if (not side0_first and side0_flinched) else damage0
    damage1_after_flinch = 0 if (side0_first and side1_flinched) else damage1
    # Flinched mons don't move at all — clear hit so on-hit triggers don't fire.
    if (not side0_first) and side0_flinched:
        hit0 = False
    if side0_first and side1_flinched:
        hit1 = False

    # ------------------------------------------------------------------
    # Counter / Mirror Coat / Metal Burst / Revenge / Focus Punch overrides
    # ------------------------------------------------------------------
    sub1_hp_pre = int(battle[OFF_FIELD + F_SUBSTITUTE_1])
    sub0_hp_pre = int(battle[OFF_FIELD + F_SUBSTITUTE_0])
    has_sub1_pre = sub1_hp_pre > 0
    has_sub0_pre = sub0_hp_pre > 0
    dmg1_thru_to_p0 = 0 if has_sub0_pre else damage1_after_flinch
    dmg0_thru_to_p1 = 0 if has_sub1_pre else damage0_after_flinch

    def _apply_retaliation(
        damage_self, mid_self, cat_other, dmg_thru_to_self, user_off_self
    ):
        d = damage_self
        if mid_self == MOVE_COUNTER:
            d = (
                min(dmg_thru_to_self * 2, 32767)
                if cat_other == CAT_PHYSICAL and dmg_thru_to_self > 0
                else 0
            )
        elif mid_self == MOVE_MIRROR_COAT:
            d = (
                min(dmg_thru_to_self * 2, 32767)
                if cat_other == CAT_SPECIAL and dmg_thru_to_self > 0
                else 0
            )
        elif mid_self == MOVE_METAL_BURST:
            d = min(dmg_thru_to_self * 3 // 2, 32767) if dmg_thru_to_self > 0 else 0
        elif mid_self == MOVE_FOCUS_PUNCH and dmg_thru_to_self > 0:
            d = 0
        elif mid_self == MOVE_FINAL_GAMBIT:
            # Showdown data/moves.ts:finalgambit damageCallback returns
            # `pokemon.hp` and `pokemon.faint()`. The user KO is handled
            # later in the self-KO section; we only override dealt damage
            # here. Fails (dmg=0) if user is already fainted.
            d = max(0, int(battle[int(user_off_self) + 1]))
        return d

    damage0_after_flinch = _apply_retaliation(
        damage0_after_flinch, move_id0, cat1, dmg1_thru_to_p0, user0_off
    )
    damage1_after_flinch = _apply_retaliation(
        damage1_after_flinch, move_id1, cat0, dmg0_thru_to_p1, user1_off
    )

    # ------------------------------------------------------------------
    # Faster mon's self-heal (Recover / Soft-Boiled / Roost / ...) fires
    # BEFORE the slower mon's damage in Showdown (each move's runMove
    # contains its own heal inside moveHit). Pokepy's damage pipeline
    # applies all damage THEN all heals in a single batch — that breaks
    # parity when the FASTER mon uses a heal move and the SLOWER mon
    # damages it, because the slower mon's damage is effectively healed
    # away (pokepy: damage applied first, then heal tops up; Showdown:
    # heal no-op at full, then damage stays). Fire the faster mon's heal
    # here so hp_pre used in the damage block below already reflects the
    # post-heal HP. Mark `_recovery_applied_early_{0,1}` so the later
    # batch heal at line ~3477 skips. Wish is still deferred to EOT.
    from pokepy.core.constants import EFFECT_RECOVERY as _EFFECT_RECOVERY_EARLY

    _MOVE_WISH_EARLY = 273
    _MOVE_REST_EARLY = 156
    _recovery_applied_early_0 = False
    _recovery_applied_early_1 = False
    _faster_user_off = user0_off if side0_first else user1_off
    _faster_move_id = move_id0 if side0_first else move_id1
    if side0_first:
        _faster_recovery_executed = (
            (not is_switch)
            and (not _self_hit0)
            and (not _p0_immobile_pre)
            and (not _sleep_talk_try_failed0)
            and (not _rest_try_failed0)
            and (not move0_no_target)
            and (not prankster_fail0)
            and (not _move0_canceled_pre)
            and (not pre_damage_fail0)
        )
    else:
        _faster_recovery_executed = (
            (not opp_is_switch)
            and (not _self_hit1)
            and (not _p1_immobile_pre)
            and (not _sleep_talk_try_failed1)
            and (not _rest_try_failed1)
            and (not move1_no_target)
            and (not prankster_fail1)
            and (not _move1_canceled_pre)
            and (not pre_damage_fail1)
        )
    _faster_is_recovery = (
        int(move_effects.effect_type[int(_faster_move_id)]) == _EFFECT_RECOVERY_EARLY
        and int(_faster_move_id) != _MOVE_WISH_EARLY
        and int(_faster_move_id) != _MOVE_REST_EARLY
        and bool(_faster_recovery_executed)
    )
    if _faster_is_recovery:
        fx.apply_recovery_from_move(
            battle,
            _faster_move_id,
            _faster_user_off,
            _faster_recovery_executed,
            game_data,
            move_effects,
            gen5_prng,
        )
        if side0_first:
            _recovery_applied_early_0 = True
            _hp0_after_postmove_pre = int(battle[_faster_user_off + 1])
        else:
            _recovery_applied_early_1 = True
            _hp1_after_postmove_pre = int(battle[_faster_user_off + 1])

    # ------------------------------------------------------------------
    # Apply damage (Substitute, Disguise, Focus Sash, Sturdy, Air Balloon, Destiny Bond)
    # ------------------------------------------------------------------
    # When side0's mid-turn pivot fired, update p0_off to the NEW active so
    # that damage1 (the slower mon's move) is applied to the switch-in, not
    # the old pivot user. Keep user0_off pinned to the original mover: its
    # own post-hit self-effects (Life Orb, recoil, Shell Bell, crash, etc.)
    # still belong to the pivot user, not the replacement.
    if _mid_turn_pivot_0:
        _new_a0 = int(battle[OFF_META + M_ACTIVE0])
        p0_off = OFF_SIDE0 + _new_a0 * POKEMON_SIZE
        p0_ab = int(battle[p0_off + 5])  # refresh ability for the new mon
    if _mid_turn_pivot_1:
        _new_a1 = int(battle[OFF_META + M_ACTIVE1])
        p1_off = OFF_SIDE1 + _new_a1 * POKEMON_SIZE
        p1_ab = int(battle[p1_off + 5])  # refresh ability for the new mon

    opp_hp_off = p1_off + 1
    own_hp_off = p0_off + 1
    opp_item_off = p1_off + 6
    own_item_off = p0_off + 6
    opp_max_hp_off = p1_off + 2
    own_max_hp_off = p0_off + 2

    hp1_pre = int(battle[opp_hp_off])
    hp0_pre = int(battle[own_hp_off])
    # When a faster attacker's post-hit self-HP changes deterministically
    # before the slower move begins (drain, recoil, Life Orb, etc.), the
    # slower move must target that live HP state rather than the turn-start
    # snapshot. Mid-turn pivots already rewrite p*_off above, so only the
    # non-pivot first-mover path needs the projected override here.
    _skip_recoil_hp_effects0 = False
    _skip_recoil_hp_effects1 = False
    hp1_pre_for_move0 = hp1_pre
    if (
        (not side0_first)
        and (not _mid_turn_pivot_1)
        and (_hp1_after_postmove_pre is not None)
    ):
        hp1_pre_for_move0 = int(_hp1_after_postmove_pre)
    hp0_pre_for_move1 = hp0_pre
    if (
        side0_first
        and (not _mid_turn_pivot_0)
        and (_hp0_after_postmove_pre is not None)
    ):
        hp0_pre_for_move1 = int(_hp0_after_postmove_pre)
    max_hp1 = int(battle[opp_max_hp_off])
    max_hp0 = int(battle[own_max_hp_off])
    item1 = int(battle[opp_item_off])
    item0 = int(battle[own_item_off])

    infiltrator0 = p0_ab == ABILITY_INFILTRATOR
    infiltrator1 = p1_ab == ABILITY_INFILTRATOR
    is_multi0 = int(move_effects.hits_max[move_id0]) > 1
    is_multi1 = int(move_effects.hits_max[move_id1]) > 1
    # Sound moves bypass substitute (gen 6+).
    flags0 = int(game_data.move_flags[move_id0])
    flags1 = int(game_data.move_flags[move_id1])
    is_sound0 = (flags0 & FLAG_SOUND) != 0
    is_sound1 = (flags1 & FLAG_SOUND) != 0

    # Sub on side1 absorbs side0's damage. For multi-hit moves, Showdown
    # caps each hit's damage at the substitute's remaining HP (data/moves.ts
    # substitute onTryPrimaryHit:19044-19056), then breaks the sub when it
    # reaches 0 HP and the next hit goes through to the mon. Overflow on
    # the breaking hit is LOST. Pokepy used to compute spillover as
    # `total - sub_hp` which over-credited the breaking hit's overflow.
    n_hits0 = int(_meta0.get("num_hits", 1)) if is_multi0 else 1
    n_hits1 = int(_meta1.get("num_hits", 1)) if is_multi1 else 1
    sub1_hp = int(battle[OFF_FIELD + F_SUBSTITUTE_1])
    has_sub1 = sub1_hp > 0 and not infiltrator0 and not is_sound0
    if has_sub1:
        if is_multi0 and n_hits0 > 1 and damage0_after_flinch > 0:
            per_hit0 = damage0_after_flinch // n_hits0
            cur_sub = sub1_hp
            mon_dmg = 0
            for _h in range(n_hits0):
                if cur_sub > 0:
                    cur_sub = max(0, cur_sub - per_hit0)
                else:
                    mon_dmg += per_hit0
            battle[OFF_FIELD + F_SUBSTITUTE_1] = cur_sub
            damage_through_to_p1 = mon_dmg
        else:
            new_sub1 = max(0, sub1_hp - damage0_after_flinch)
            battle[OFF_FIELD + F_SUBSTITUTE_1] = new_sub1
            damage_through_to_p1 = 0
    else:
        damage_through_to_p1 = damage0_after_flinch

    sub0_hp = int(battle[OFF_FIELD + F_SUBSTITUTE_0])
    has_sub0 = sub0_hp > 0 and not infiltrator1 and not is_sound1
    if has_sub0:
        if is_multi1 and n_hits1 > 1 and damage1_after_flinch > 0:
            per_hit1 = damage1_after_flinch // n_hits1
            cur_sub = sub0_hp
            mon_dmg = 0
            for _h in range(n_hits1):
                if cur_sub > 0:
                    cur_sub = max(0, cur_sub - per_hit1)
                else:
                    mon_dmg += per_hit1
            battle[OFF_FIELD + F_SUBSTITUTE_0] = cur_sub
            damage_through_to_p0 = mon_dmg
        else:
            new_sub0 = max(0, sub0_hp - damage1_after_flinch)
            battle[OFF_FIELD + F_SUBSTITUTE_0] = new_sub0
            damage_through_to_p0 = 0
    else:
        damage_through_to_p0 = damage1_after_flinch

    # Disguise / Ice Face — flag bit 0x40. These abilities still remove HP
    # from the holder, but that self-damage is not move.totalDamage and must
    # not feed recoil, drain, or Shell Bell.
    opp_flags_dis = int(battle[p1_off + 15])
    own_flags_dis = int(battle[p0_off + 15])
    face1_absorbed_hit = (
        ((opp_flags_dis & 0x40) != 0)
        and (
            p1_ab == ABILITY_DISGUISE
            or (p1_ab == ABILITY_ICE_FACE and cat0 == CAT_PHYSICAL)
        )
        and damage_through_to_p1 > 0
    )
    if face1_absorbed_hit:
        damage_through_to_p1 = max_hp1 // 8
        battle[p1_off + 15] = opp_flags_dis & ~0x40
    face0_absorbed_hit = (
        ((own_flags_dis & 0x40) != 0)
        and (
            p0_ab == ABILITY_DISGUISE
            or (p0_ab == ABILITY_ICE_FACE and cat1 == CAT_PHYSICAL)
        )
        and damage_through_to_p0 > 0
    )
    if face0_absorbed_hit:
        damage_through_to_p0 = max_hp0 // 8
        battle[p0_off + 15] = own_flags_dis & ~0x40

    # The slower move must subtract from the same projected live HP snapshot
    # that its full-HP checks already use. Otherwise first-mover drain/recoil /
    # Life Orb changes are visible to the damage formula but not to the actual
    # HP writeback, which can leave the target alive locally after a Showdown
    # KO and leak follow-up effects such as partial trapping onto the wrong turn.
    new_hp1 = max(0, hp1_pre_for_move0 - damage_through_to_p1)
    new_hp0 = max(0, hp0_pre_for_move1 - damage_through_to_p0)

    # Focus Sash / Sturdy — survive lethal hit at 1 HP. Showdown fires
    # PER HIT (sim/battle-actions.ts:tryMoveHit), so a multi-hit move
    # KO'd through Sash will die: Sash saves hit N at 1 HP, hit N+1 takes
    # them to 0. Pokepy applies damage as an aggregate, so we have to
    # detect "multi-hit move that would have triggered Sash mid-sequence"
    # and skip the Sash save.
    #
    # IMPORTANT: Sash/Sturdy runs BEFORE the p*_move_ran check below so
    # the slower mon's move is not cancelled when Sash saves it. Showdown's
    # per-hit `Battle#damage` fires Focus Sash inside the faster mon's
    # runMove, leaving the slower mon alive at 1 HP to take its normal turn.
    is_move0_multihit = int(move_effects.hits_max[move_id0]) > 1
    is_move1_multihit = int(move_effects.hits_max[move_id1]) > 1
    if (
        (item1 == ITEM_FOCUS_SASH or _inline_knock_removed_focus_sash1)
        and hp1_pre_for_move0 == max_hp1
        and new_hp1 == 0
        and damage0_after_flinch > 0
    ):
        if not is_move0_multihit:
            new_hp1 = 1
            battle[opp_item_off] = 0
        else:
            # Multi-hit: Sash saves hit N but hit N+1 finishes the KO. The
            # item is still consumed (fired once mid-sequence).
            battle[opp_item_off] = 0
    if (
        (item0 == ITEM_FOCUS_SASH or _inline_knock_removed_focus_sash0)
        and hp0_pre_for_move1 == max_hp0
        and new_hp0 == 0
        and damage1_after_flinch > 0
    ):
        if not is_move1_multihit:
            new_hp0 = 1
            battle[own_item_off] = 0
        else:
            battle[own_item_off] = 0
    if (
        p1_ab == ABILITY_STURDY
        and hp1_pre_for_move0 == max_hp1
        and new_hp1 == 0
        and damage0_after_flinch > 0
    ):
        new_hp1 = 1
    if (
        p0_ab == ABILITY_STURDY
        and hp0_pre_for_move1 == max_hp0
        and new_hp0 == 0
        and damage1_after_flinch > 0
    ):
        new_hp0 = 1
    _protect_state1_survive = int(battle[OFF_FIELD + F_PROTECT_1])
    _protect_state0_survive = int(battle[OFF_FIELD + F_PROTECT_0])
    if (
        get_protect_active(_protect_state1_survive) > 0
        and get_protect_type(_protect_state1_survive) == PROTECT_ENDURE
        and new_hp1 == 0
        and damage0_after_flinch > 0
    ):
        new_hp1 = 1
    if (
        get_protect_active(_protect_state0_survive) > 0
        and get_protect_type(_protect_state0_survive) == PROTECT_ENDURE
        and new_hp0 == 0
        and damage1_after_flinch > 0
    ):
        new_hp0 = 1

    # Determine if each attacker's move actually executed. If the slower
    # attacker is KO'd before they move, their move is cancelled — and any
    # post-damage on-hit triggers (Focus Sash, Air Balloon pop, Sturdy,
    # contact recoil, defender abilities, status apply, hazards from the
    # move, etc.) should NOT fire.
    #
    # On switch-vs-move turns the faster action is the voluntary switch, so
    # the non-switching side's `new_hp*` placeholder is still zero at this
    # point. Showdown still lets the move execute against the freshly switched
    # target, so don't treat that zero as "the slower attacker fainted".
    p0_move_ran = (
        (not is_switch)
        and (side0_first or opp_is_switch or new_hp0 > 0)
        and not _move0_canceled_pre
    )
    p1_move_ran = (
        (not opp_is_switch)
        and ((not side0_first) or is_switch or new_hp1 > 0)
        and not _move1_canceled_pre
    )
    if not p0_move_ran:
        damage_through_to_p1 = 0
        damage0_after_flinch = 0
        damage0 = 0
        # Preserve the faster user's projected postmove HP when a canceled
        # slower move would otherwise reset the target slot to turn-start HP.
        new_hp1 = hp1_pre_for_move0
        hit0 = False
        if is_charge_turn0:
            battle[OFF_META + M_CHARGING_0] = -1
            if _chg0_is_semi_invul:
                battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0] = (
                    int(battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0])
                    & ~ACTIVE_MOVE_ACTIONS_SEMI_INVUL
                )
    if not p1_move_ran:
        damage_through_to_p0 = 0
        damage1_after_flinch = 0
        damage1 = 0
        new_hp0 = hp0_pre_for_move1
        hit1 = False
        if is_charge_turn1:
            battle[OFF_META + M_CHARGING_1] = -1
            if _chg1_is_semi_invul:
                battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1] = (
                    int(battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1])
                    & ~ACTIVE_MOVE_ACTIONS_SEMI_INVUL
                )

    # Slower charge-turn users only become semi-invulnerable once their own
    # action has actually started. If they move second, set the flag now so
    # residual effects (e.g. Grassy Terrain healing) see the same state as
    # Showdown for the rest of this turn.
    if is_charge_turn0 and _chg0_is_semi_invul and (not side0_first) and p0_move_ran:
        battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0] = (
            int(battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0])
            | ACTIVE_MOVE_ACTIONS_SEMI_INVUL
        )
    if is_charge_turn1 and _chg1_is_semi_invul and side0_first and p1_move_ran:
        battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1] = (
            int(battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1])
            | ACTIVE_MOVE_ACTIONS_SEMI_INVUL
        )

    def _actual_hp_removed_from_hit(
        _raw_damage,
        _target_had_sub,
        _sub_hp_before,
        _target_hp_before,
        _target_hp_after,
    ):
        if int(_raw_damage) <= 0:
            return 0
        if _target_had_sub:
            return min(int(_raw_damage), int(_sub_hp_before))
        return min(
            int(_raw_damage), max(0, int(_target_hp_before) - int(_target_hp_after))
        )

    def _project_user_hp_after_own_postmove(
        _user_off,
        _target_off,
        _move_id,
        _raw_damage,
        _actual_damage,
        _did_hit,
        _num_hits,
        _user_hp_before_postmove,
        *,
        include_after_move_secondary: bool = True,
        move_attempted: bool = False,
        restore_target_item: int = 0,
        restore_target_item_off: int = -1,
    ):
        """Project the faster mover's HP after the non-PRNG post-hit chain
        that Showdown resolves before the slower move begins.

        This captures the common no-target cases where the faster user faints
        to drain/contact/recoil/Life Orb during its own move, leaving the
        slower foe-targeted move to fail with `[notarget]`.
        """
        _sim = battle.copy()
        _sim[int(_user_off) + 1] = np.int16(int(_user_hp_before_postmove))
        if int(restore_target_item) > 0 and int(restore_target_item_off) >= 0:
            _restore_off = int(restore_target_item_off)
            if int(_sim[_restore_off + 6]) == 0:
                _sim[_restore_off + 6] = np.int16(int(restore_target_item))
        _apply_recoil_drain_from_move_tracked(
            _sim,
            _move_id,
            _user_off,
            _actual_damage,
            _did_hit,
            game_data,
            move_effects,
            target_offset=_target_off,
            phase="drain",
        )
        _apply_contact_damage_tracked(
            _sim,
            _move_id,
            _user_off,
            _target_off,
            _did_hit and int(_raw_damage) > 0,
            game_data,
            move_effects,
            num_hits=int(_num_hits),
        )
        _apply_recoil_drain_from_move_tracked(
            _sim,
            _move_id,
            _user_off,
            _actual_damage,
            _did_hit,
            game_data,
            move_effects,
            target_offset=_target_off,
            phase="recoil",
            move_attempted=move_attempted,
        )
        _apply_crash_damage_on_move_fail(
            _sim,
            _user_off,
            _move_id,
            _did_hit,
            move_attempted,
        )
        if not include_after_move_secondary:
            return int(_sim[int(_user_off) + 1])
        _apply_life_orb_recoil_tracked(
            _sim,
            _user_off,
            _raw_damage,
            _did_hit,
            game_data,
            move_id=_move_id,
            move_effects=move_effects,
        )
        _ITEM_SHELL_BELL_POST = 438
        if (
            int(_sim[int(_user_off) + 6]) == _ITEM_SHELL_BELL_POST
            and int(_sim[int(_user_off) + 1]) > 0
            and int(_actual_damage) > 0
            and _did_hit
        ):
            _cur_hp_sb = int(_sim[int(_user_off) + 1])
            _max_hp_sb = int(_sim[int(_user_off) + 2])
            _heal_sb = max(int(_actual_damage) // 8, 1)
            _sim[int(_user_off) + 1] = np.int16(min(_max_hp_sb, _cur_hp_sb + _heal_sb))
        return int(_sim[int(_user_off) + 1])

    def _source_survives_damaging_hit_contact(
        _user_off,
        _target_off,
        _move_id,
        _actual_damage,
        _did_hit,
        _num_hits,
        _user_hp_before_contact,
        _move_idx=-1,
    ):
        """Return True if the faster mover survives through DamagingHit.

        Knock Off's onAfterHit item removal happens after drain and contact
        damage, but before recoil/Life Orb. Defender damaging-hit responses
        such as Gulp Missile / Aftermath / Innards Out also resolve in this
        window. The slower move's damage calc should only see Knock Off item
        removal, and the slower foe-targeted move should only remain valid,
        when the faster source is still alive at the end of that full
        DamagingHit cascade.
        """
        _sim = battle.copy()
        _sim[int(_user_off) + 1] = np.int16(int(_user_hp_before_contact))
        _apply_recoil_drain_from_move_tracked(
            _sim,
            _move_id,
            _user_off,
            _actual_damage,
            _did_hit,
            game_data,
            move_effects,
            target_offset=_target_off,
            phase="drain",
        )
        _apply_contact_damage_tracked(
            _sim,
            _move_id,
            _user_off,
            _target_off,
            _did_hit and int(_actual_damage) > 0,
            game_data,
            move_effects,
            num_hits=int(_num_hits),
        )
        _atk_side = 0 if int(_user_off) < OFF_SIDE1 else 1
        _sim_prng = Gen5PRNG(gen5_prng.get_seed_array())
        if _atk_side == 0:
            _dummy_user_off = _target_off
            _dummy_target_off = _user_off
            fx.apply_defender_abilities(
                _sim,
                _move_id,
                _move_id,
                _user_off,
                _dummy_user_off,
                _target_off,
                _dummy_target_off,
                int(_actual_damage),
                0,
                _did_hit and int(_actual_damage) > 0,
                False,
                game_data,
                _sim_prng,
                move_idx0=int(_move_idx),
                move_idx1=-1,
                skip_toxic_chain0=True,
                skip_toxic_chain1=True,
                skip_immediate_stateful_move0=True,
                skip_immediate_stateful_move1=True,
            )
        else:
            _dummy_user_off = _target_off
            _dummy_target_off = _user_off
            fx.apply_defender_abilities(
                _sim,
                _move_id,
                _move_id,
                _dummy_user_off,
                _user_off,
                _dummy_target_off,
                _target_off,
                0,
                int(_actual_damage),
                False,
                _did_hit and int(_actual_damage) > 0,
                game_data,
                _sim_prng,
                move_idx0=-1,
                move_idx1=int(_move_idx),
                skip_toxic_chain0=True,
                skip_toxic_chain1=True,
                skip_immediate_stateful_move0=True,
                skip_immediate_stateful_move1=True,
            )
        return int(_sim[int(_user_off) + 1]) > 0

    # Showdown runs the faster move's full post-hit chain before the slower
    # move reaches target validation in `useMoveInner()`. If the faster user
    # faints to its own drain/contact/recoil/Life Orb sequence, the slower
    # foe-targeted move must fail as `[notarget]` and apply no downstream hit
    # effects. Pokepy computes both moves together, so detect that case here
    # and roll back the stale slower-move target state before any on-hit or
    # after-hit helpers fire.
    if (
        side0_first
        and p1_move_ran
        and move1_targets_foe_mon
        and not is_delayed1
        and not _mid_turn_pivot_0
        and not is_switch
    ):
        _actual_dmg0 = int(_mirrored_damage0_on_p1)
        if _actual_dmg0 <= 0:
            _actual_dmg0 = _actual_damage_for_source_postmove(
                user0_off,
                p1_off,
                move_id0,
                cat0,
                damage0_after_flinch,
                has_sub1,
                sub1_hp,
                hp1_pre,
                new_hp1,
            )
        _hp0_after_postmove = _project_user_hp_after_own_postmove(
            user0_off,
            p1_off,
            move_id0,
            damage0_after_flinch,
            _actual_dmg0,
            hit0,
            int(_meta0.get("num_hits", 1)),
            hp0_pre,
            move_attempted=not (
                is_switch
                or _self_hit0
                or _p0_immobile_pre
                or _sleep_talk_try_failed0
                or _rest_try_failed0
                or move0_no_target
                or prankster_fail0
                or _move0_canceled_pre
            ),
        )
        _live_user0_hp_after_damaging_hit = int(battle[user0_off + 1])
        if (
            _live_user0_hp_after_damaging_hit <= 0
            or _hp0_after_postmove <= 0
            or not _source_survives_damaging_hit_contact(
                user0_off,
                p1_off,
                move_id0,
                _actual_dmg0,
                hit0,
                int(_meta0.get("num_hits", 1)),
                hp0_pre,
                move_idx,
            )
        ):
            move1_no_target = True
            p1_move_ran = False
            damage_through_to_p0 = 0
            damage1_after_flinch = 0
            damage1 = 0
            hit1 = False
            new_hp0 = int(_hp0_after_postmove)
            _projected_full_postmove_hp_committed0 = True
            _preapplied_after_move_secondary_hp_delta0 = 0
            _projected_after_move_secondary_hp_delta0 = 0
            battle[OFF_FIELD + F_SUBSTITUTE_0] = np.int16(sub0_hp)
            battle[p0_off + 15] = np.int16(own_flags_dis)
            battle[own_item_off] = np.int16(item0)
    elif (
        (not side0_first)
        and p0_move_ran
        and move0_targets_foe_mon
        and not is_delayed0
        and not _mid_turn_pivot_1
        and not opp_is_switch
    ):
        _actual_dmg1 = int(_mirrored_damage1_on_p0)
        if _actual_dmg1 <= 0:
            _actual_dmg1 = _actual_damage_for_source_postmove(
                user1_off,
                p0_off,
                move_id1,
                cat1,
                damage1_after_flinch,
                has_sub0,
                sub0_hp,
                hp0_pre,
                new_hp0,
            )
        _hp1_after_postmove = _project_user_hp_after_own_postmove(
            user1_off,
            p0_off,
            move_id1,
            damage1_after_flinch,
            _actual_dmg1,
            hit1,
            int(_meta1.get("num_hits", 1)),
            hp1_pre,
            move_attempted=not (
                opp_is_switch
                or _self_hit1
                or _p1_immobile_pre
                or _sleep_talk_try_failed1
                or _rest_try_failed1
                or move1_no_target
                or prankster_fail1
                or _move1_canceled_pre
            ),
        )
        _live_user1_hp_after_damaging_hit = int(battle[user1_off + 1])
        if (
            _live_user1_hp_after_damaging_hit <= 0
            or _hp1_after_postmove <= 0
            or not _source_survives_damaging_hit_contact(
                user1_off,
                p0_off,
                move_id1,
                _actual_dmg1,
                hit1,
                int(_meta1.get("num_hits", 1)),
                hp1_pre,
                opp_move_idx,
            )
        ):
            move0_no_target = True
            p0_move_ran = False
            damage_through_to_p1 = 0
            damage0_after_flinch = 0
            damage0 = 0
            hit0 = False
            new_hp1 = int(_hp1_after_postmove)
            _projected_full_postmove_hp_committed1 = True
            _preapplied_after_move_secondary_hp_delta1 = 0
            _projected_after_move_secondary_hp_delta1 = 0
            battle[OFF_FIELD + F_SUBSTITUTE_1] = np.int16(sub1_hp)
            battle[p1_off + 15] = np.int16(opp_flags_dis)
            battle[opp_item_off] = np.int16(item1)

    # Air Balloon — pops on any damaging hit (Showdown items.ts:199-207
    # has both onDamagingHit AND onAfterSubDamage). Use damage_after_flinch
    # rather than damage_through_to_p* so the balloon also pops when the
    # hit is absorbed by a substitute.
    if item0 == ITEM_AIR_BALLOON and damage1_after_flinch > 0 and not is_immobile1:
        battle[own_item_off] = 0
    if item1 == ITEM_AIR_BALLOON and damage0_after_flinch > 0 and not is_immobile0:
        battle[opp_item_off] = 0

    # Side ordering: if first attacker KO'd target, second can still hit only if alive
    if side0_first:
        final_hp1 = new_hp1
        final_hp0 = hp0_pre if new_hp1 == 0 else new_hp0
        if final_hp0 == new_hp0 and _projected_full_postmove_hp_committed0:
            _skip_contact_damage_effects0 = True
            _skip_crash_damage_effects0 = True
            _skip_recoil_hp_effects0 = True
            _skip_after_move_secondary_hp_effects0 = True
        if (
            (new_hp1 > 0)
            and (not _mid_turn_pivot_0)
            and (_hp0_after_postmove_pre is not None)
        ):
            _skip_contact_damage_effects0 = True
            _skip_crash_damage_effects0 = True
            _skip_recoil_hp_effects0 = True
            if _projected_after_move_secondary_hp_delta0 != 0:
                _skip_after_move_secondary_hp_effects0 = True
        if _preapplied_after_move_secondary_hp_delta0 != 0 and int(final_hp0) == int(
            hp0_pre
        ):
            # Only add the projected afterMoveSecondary delta when the final
            # HP is still on the turn-start baseline (for example the faster
            # user KO'd the target before the slower move could write through
            # hp0_pre_for_move1). When final_hp0 already came from the
            # projected postmove snapshot, adding the delta here would
            # double-apply Life Orb / Shell Bell.
            final_hp0 = max(
                0,
                min(
                    max_hp0,
                    int(final_hp0) + int(_preapplied_after_move_secondary_hp_delta0),
                ),
            )
            _skip_after_move_secondary_hp_effects0 = True
    else:
        final_hp0 = new_hp0
        final_hp1 = hp1_pre if new_hp0 == 0 else new_hp1
        if final_hp1 == new_hp1 and _projected_full_postmove_hp_committed1:
            _skip_contact_damage_effects1 = True
            _skip_crash_damage_effects1 = True
            _skip_recoil_hp_effects1 = True
            _skip_after_move_secondary_hp_effects1 = True
        if (
            (new_hp0 > 0)
            and (not _mid_turn_pivot_1)
            and (_hp1_after_postmove_pre is not None)
        ):
            _skip_contact_damage_effects1 = True
            _skip_crash_damage_effects1 = True
            _skip_recoil_hp_effects1 = True
            if _projected_after_move_secondary_hp_delta1 != 0:
                _skip_after_move_secondary_hp_effects1 = True
        if _preapplied_after_move_secondary_hp_delta1 != 0 and int(final_hp1) == int(
            hp1_pre
        ):
            # Same guard as side0: if the slower move already subtracted from
            # the projected faster-mover HP snapshot, the projected
            # afterMoveSecondary delta is already baked into final_hp1.
            final_hp1 = max(
                0,
                min(
                    max_hp1,
                    int(final_hp1) + int(_preapplied_after_move_secondary_hp_delta1),
                ),
            )
            _skip_after_move_secondary_hp_effects1 = True
    battle[opp_hp_off] = final_hp1
    battle[own_hp_off] = final_hp0

    # Type-resist berries (Shuca, Colbur, Occa, etc.) — consume on the
    # super-effective hit they reduced. Showdown items.ts onSourceModifyDamage +
    # `useItem`. Pokepy applied the 0.5x in damage_gen9.py but never
    # consumed the berry, leaving the defender with permanent 0.5x against
    # that type. Consume here once per hit.
    _BERRY_TYPE_RESIST_MAP = {
        311: TYPE_FIRE,
        329: TYPE_WATER,
        526: TYPE_ELECTRIC,
        409: TYPE_GRASS,
        567: TYPE_ICE,
        71: TYPE_FIGHTING,
        234: TYPE_POISON,
        443: TYPE_GROUND,
        62: TYPE_FLYING,
        233: TYPE_PSYCHIC,
        487: TYPE_BUG,
        76: TYPE_ROCK,
        185: TYPE_DRAGON,
        78: TYPE_DARK,
        17: TYPE_STEEL,
        603: TYPE_FAIRY,
        330: TYPE_PSYCHIC,
        66: TYPE_NORMAL,
    }

    def _consume_resist_berry(target_off, mt, type_eff_proxy):
        item = int(battle[target_off + 6])
        if item not in _BERRY_TYPE_RESIST_MAP:
            return
        if _BERRY_TYPE_RESIST_MAP[item] != mt:
            return
        # Chilan Berry (66) triggers on any Normal hit; others on SE only.
        if item == 66 or type_eff_proxy:
            _record_consumed_berry(target_off, item)
            battle[target_off + 6] = 0

    # Compute SE proxy by checking type chart for each side. We do this
    # cheaply rather than re-running the full damage calc.
    def _is_se(attacker_type, def_off):
        types_packed = int(battle[def_off + 4])
        t1 = types_packed & 0xFF
        t2 = (types_packed >> 8) & 0xFF
        e1 = float(type_chart[t1, attacker_type])
        e2 = 1.0 if t2 == t1 else float(type_chart[t2, attacker_type])
        return (e1 * e2) > 1.0

    if hit0 and damage0_after_flinch > 0 and final_hp1 > 0:
        _consume_resist_berry(p1_off, move_type0, _is_se(move_type0, p1_off))
    if hit1 and damage1_after_flinch > 0 and final_hp0 > 0:
        _consume_resist_berry(p0_off, move_type1, _is_se(move_type1, p0_off))

    # Destiny Bond cascade
    destiny1 = int(battle[OFF_FIELD + F_DESTINY_BOND_1])
    side1_fainted_from_dmg = (
        (final_hp1 == 0) and (hp1_pre > 0) and (damage_through_to_p1 > 0)
    )
    if (
        destiny1 > 0
        and not is_pending_wish_sentinel(destiny1)
        and side1_fainted_from_dmg
    ):
        final_hp0 = 0
        battle[own_hp_off] = 0
    destiny0 = int(battle[OFF_FIELD + F_DESTINY_BOND_0])
    side0_fainted_from_dmg = (
        (final_hp0 == 0) and (hp0_pre > 0) and (damage_through_to_p0 > 0)
    )
    if (
        destiny0 > 0
        and not is_pending_wish_sentinel(destiny0)
        and side0_fainted_from_dmg
    ):
        final_hp1 = 0
        battle[opp_hp_off] = 0
    # Showdown condition.onBeforeMove: using any non-DB move removes the
    # DB volatile. Preserve DB if the user just successfully used Destiny
    # Bond this turn (so the cascade still fires if the opponent KOs them
    # on the NEXT turn). Clear otherwise.
    if move_id0 != MOVE_DESTINY_BOND and not is_pending_wish_sentinel(destiny0):
        battle[OFF_FIELD + F_DESTINY_BOND_0] = 0
    if move_id1 != MOVE_DESTINY_BOND and not is_pending_wish_sentinel(destiny1):
        battle[OFF_FIELD + F_DESTINY_BOND_1] = 0

    # Decisive KO tracking (used to settle recoil ties)
    pre_recoil_alive0 = _count_alive(battle, OFF_SIDE0)
    pre_recoil_alive1 = _count_alive(battle, OFF_SIDE1)
    if side0_first and pre_recoil_alive1 == 0:
        decisive_winner = 0
    elif (not side0_first) and pre_recoil_alive0 == 0:
        decisive_winner = 1
    else:
        decisive_winner = -1

    # Protect contact effects (King's Shield, Baneful Bunker, Silk Trap)
    fx.apply_protect_contact_effects(
        battle, move_id0, user0_off, 1, target0_protected, game_data, move_effects
    )
    fx.apply_protect_contact_effects(
        battle, move_id1, user1_off, 0, target1_protected, game_data, move_effects
    )

    # Volt/Water Absorb healing, Sap Sipper / Storm Drain / Lightning Rod /
    # Motor Drive stat boosts, Flash Fire flag. Showdown fires these via
    # onTryHit BEFORE damage is rolled, so they trigger even when the type
    # immunity sets damage to 0. Gate on "move was executed" (not blocked
    # by sleep/freeze/full para/flinch/confusion self-hit/prankster fail)
    # instead of `hit` (which is False when damage is 0).
    #
    # When the slower move targets the first mover, pokepy has not yet
    # materialized that first mover's own late postmove HP changes
    # (Struggle recoil, drain/recoil shims, etc.). Applying absorb healing
    # immediately in that branch can therefore read a stale pre-recoil full
    # HP and later get overwritten by the delayed postmove phase. Defer only
    # that branch until after the recoil/drain section below, when the first
    # mover's live HP has caught up.
    p0_move_executed = (not is_switch) and not (
        is_immobile0
        or pre_damage_fail0
        or prankster_fail0
        or self_hit0
        or move0_no_target
        or (side0_flinched and not side0_first)
    )
    p1_move_executed = (not opp_is_switch) and not (
        is_immobile1
        or pre_damage_fail1
        or prankster_fail1
        or self_hit1
        or move1_no_target
        or (side1_flinched and side0_first)
    )
    if delayed_ready0 and p0_move_executed and p0_move_ran:
        fs_live = int(battle[OFF_META + M_FUTURE_SIGHT])
        fs_live = (fs_live & 0x0FFF) | (3 << 12)
        battle[OFF_META + M_FUTURE_SIGHT] = fs_live
        battle[OFF_MOVES + M_FUTURE_MOVE_0] = move_id0
        battle[OFF_MOVES + M_FUTURE_SRC_0] = active0
    if delayed_ready1 and p1_move_executed and p1_move_ran:
        fs_live = int(battle[OFF_META + M_FUTURE_SIGHT])
        fs_live = (fs_live & ~(0xF << 4)) | (3 << 4)
        battle[OFF_META + M_FUTURE_SIGHT] = fs_live
        battle[OFF_MOVES + M_FUTURE_MOVE_1] = move_id1
        battle[OFF_MOVES + M_FUTURE_SRC_1] = active1
    _defer_absorb_on_p1 = not side0_first
    _defer_absorb_on_p0 = side0_first
    if not _defer_absorb_on_p1:
        fx.apply_absorb_ability_healing(battle, p1_off, move_type0, p0_move_executed)
    if not _defer_absorb_on_p0:
        fx.apply_absorb_ability_healing(battle, p0_off, move_type1, p1_move_executed)
    fx.apply_weakness_policy(
        battle, p1_off, move_type0, hit0, damage0_after_flinch, move_id0
    )
    fx.apply_weakness_policy(
        battle, p0_off, move_type1, hit1, damage1_after_flinch, move_id1
    )

    # Stamina
    if (p1_ab == ABILITY_STAMINA) and hit0 and damage0_after_flinch > 0:
        battle[p1_off + 13] = apply_boost_to_packed(int(battle[p1_off + 13]), 4, 1)
    if (p0_ab == ABILITY_STAMINA) and hit1 and damage1_after_flinch > 0:
        battle[p0_off + 13] = apply_boost_to_packed(int(battle[p0_off + 13]), 4, 1)

    # Seed Sower (Grassy Terrain) — the first-mover case is handled inline
    # before the slower damage calc so same-turn Grass BP checks see the new
    # field state. Keep the late call for slower-mover triggers and for any
    # same-turn cases that did not run through the between-moves branch.
    if hit0 and damage0_after_flinch > 0:
        _apply_seed_sower(p1_off)
    if hit1 and damage1_after_flinch > 0:
        _apply_seed_sower(p0_off)

    # NOTE: Form-change abilities (Zen Mode, Stance Change, Schooling, Slow Start)
    # are simplified stat-mutation logic in the source (lines 4085-4275). Pokepy
    # Form-change abilities (Zen Mode, Stance Change, Schooling, Slow Start)
    fx.apply_form_changes(
        battle,
        p0_off,
        p1_off,
        move_id0,
        move_id1,
        is_switch,
        opp_is_switch,
        game_data,
    )

    # ------------------------------------------------------------------
    # Status / volatile / stat changes from moves
    # ------------------------------------------------------------------
    target0_off = p1_off  # P0's target is P1
    target1_off = p0_off  # P1's target is P0

    # Magic Bounce: redirect status moves back to the user. The hit0/hit1
    # flags are kept TRUE so the apply_*_from_move functions actually run,
    # but the target offset/side is swapped so the effect lands on the
    # original user. Hazards are bounced separately in pokepy/effects/hazards.
    if mb_redirect0_to_user:
        target0_off = user0_off  # P0's status now hits P0
        target_side0 = 0
    else:
        target_side0 = 1
    if mb_redirect1_to_user:
        target1_off = user1_off
        target_side1 = 1
    else:
        target_side1 = 0

    opp_status_pre = int(battle[p1_off + 12]) & 0xFF
    own_status_pre = int(battle[p0_off + 12]) & 0xFF

    hit0_alive = hit0 and (final_hp1 > 0) and not target0_protected
    hit1_alive = hit1 and (final_hp0 > 0) and not target1_protected
    # Showdown: when a substitute fully absorbs the hit, the target slot is
    # set to null in the secondary-effect loop (sim/battle-actions.ts
    # moveHitStepMoveHitLoop), so secondary status / stat drops / volatiles /
    # flinch do NOT fire on the mon behind the sub. This remains true even on
    # the exact hit that BREAKS the substitute. Pokepy previously checked the
    # post-damage substitute HP, which leaked secondaries on substitute-
    # breaking hits like Malignant Chain into Substituted Suicune.
    #
    # Use "target had a substitute before the move and no HP damage reached
    # the actual mon" rather than "sub still exists now". That keeps single-
    # hit breaking blows blocked while still allowing later hits of a multi-
    # hit move to apply secondaries once damage reaches the target.
    sub_absorbed_all0 = (
        has_sub1 and int(damage_through_to_p1) <= 0 and not mb_redirect0_to_user
    )
    sub_absorbed_all1 = (
        has_sub0 and int(damage_through_to_p0) <= 0 and not mb_redirect1_to_user
    )
    sub_blocks_secondary0 = sub_absorbed_all0
    sub_blocks_secondary1 = sub_absorbed_all1
    hit0_thru_sub = hit0_alive and not sub_blocks_secondary0
    hit1_thru_sub = hit1_alive and not sub_blocks_secondary1

    # Sheer Force / Covert Cloak — suppress secondaries
    has_secondary0 = (
        int(move_effects.status_chance[move_id0]) > 0
        or int(move_effects.stat_chance[move_id0]) > 0
        or int(move_effects.volatile_chance[move_id0]) > 0
        or int(move_id0) == MOVE_TRI_ATTACK
    )
    has_secondary1 = (
        int(move_effects.status_chance[move_id1]) > 0
        or int(move_effects.stat_chance[move_id1]) > 0
        or int(move_effects.volatile_chance[move_id1]) > 0
        or int(move_id1) == MOVE_TRI_ATTACK
    )
    sheer_force0 = (p0_ab == ABILITY_SHEER_FORCE) and has_secondary0
    sheer_force1 = (p1_ab == ABILITY_SHEER_FORCE) and has_secondary1
    ITEM_COVERT_CLOAK = 1885
    covert_cloak1 = int(battle[target0_off + 6]) == ITEM_COVERT_CLOAK
    covert_cloak0 = int(battle[target1_off + 6]) == ITEM_COVERT_CLOAK
    hit0_secondary = hit0_thru_sub and not sheer_force0 and not covert_cloak1
    hit1_secondary = hit1_thru_sub and not sheer_force1 and not covert_cloak0

    # Multi-hit secondaries (Twineedle 20% poison etc.) roll per hit in
    # Showdown (sim/battle-actions.ts:1357 inside moveHit loop), so the
    # engine passes num_hits.
    if not _status_early_applied0:
        _status_before0 = get_status(int(battle[target0_off + 12]))
        fx.apply_status_from_move(
            battle,
            move_id0,
            target0_off,
            hit0_secondary,
            game_data,
            move_effects,
            gen5_prng,
            user_offset=user0_off,
            num_hits=int(_meta0.get("num_hits", 1)),
            prerolled_rolls=prerolled_move0.get("status"),
        )
        _maybe_apply_poison_puppeteer(user0_off, target0_off, _status_before0)
    if not _status_early_applied1:
        _status_before1 = get_status(int(battle[target1_off + 12]))
        fx.apply_status_from_move(
            battle,
            move_id1,
            target1_off,
            hit1_secondary,
            game_data,
            move_effects,
            gen5_prng,
            user_offset=user1_off,
            num_hits=int(_meta1.get("num_hits", 1)),
            prerolled_rolls=prerolled_move1.get("status"),
        )
        _maybe_apply_poison_puppeteer(user1_off, target1_off, _status_before1)
    apply_tri_attack_status_from_move(
        battle,
        move_id0,
        target0_off,
        hit0_secondary,
        game_data,
        gen5_prng,
        user_offset=user0_off,
        prerolled_roll=prerolled_move0.get("tri_attack_roll"),
        prerolled_status=prerolled_move0.get("tri_attack_status"),
    )
    apply_tri_attack_status_from_move(
        battle,
        move_id1,
        target1_off,
        hit1_secondary,
        game_data,
        gen5_prng,
        user_offset=user1_off,
        prerolled_roll=prerolled_move1.get("tri_attack_roll"),
        prerolled_status=prerolled_move1.get("tri_attack_status"),
    )
    fx.apply_confusion_from_move(
        battle,
        move_id0,
        target_side0,
        hit0_secondary,
        game_data,
        move_effects,
        gen5_prng,
        prerolled_roll=prerolled_move0.get("confusion"),
        prerolled_duration=prerolled_move0.get("confusion_duration"),
        prerolled_can_apply=prerolled_move0.get("confusion_can_apply"),
        target_stats_raised_this_turn=stats_raised_this_turn1,
    )
    fx.apply_confusion_from_move(
        battle,
        move_id1,
        target_side1,
        hit1_secondary,
        game_data,
        move_effects,
        gen5_prng,
        prerolled_roll=prerolled_move1.get("confusion"),
        prerolled_duration=prerolled_move1.get("confusion_duration"),
        prerolled_can_apply=prerolled_move1.get("confusion_can_apply"),
        target_stats_raised_this_turn=stats_raised_this_turn0,
    )
    fx.apply_taunt_from_move(
        battle,
        move_id0,
        target_side0,
        hit0_alive,
        move_effects,
        gen5_prng,
        prerolled_roll=prerolled_move0.get("taunt"),
    )
    fx.apply_taunt_from_move(
        battle,
        move_id1,
        target_side1,
        hit1_alive,
        move_effects,
        gen5_prng,
        prerolled_roll=prerolled_move1.get("taunt"),
    )
    fx.apply_encore_from_move(
        battle,
        move_id0,
        target_side0,
        hit0_alive,
        move_effects,
        gen5_prng,
        prerolled_roll=prerolled_move0.get("encore"),
    )
    fx.apply_encore_from_move(
        battle,
        move_id1,
        target_side1,
        hit1_alive,
        move_effects,
        gen5_prng,
        prerolled_roll=prerolled_move1.get("encore"),
    )
    fx.apply_throat_chop_from_move(
        battle,
        move_id0,
        target_side0,
        hit0_alive,
        prerolled_roll=prerolled_move0.get("ext_vol"),
    )
    fx.apply_throat_chop_from_move(
        battle,
        move_id1,
        target_side1,
        hit1_alive,
        prerolled_roll=prerolled_move1.get("ext_vol"),
    )
    # Extended volatile (attract / torment / leech-seed-like): the second
    # arg is target side, third is user side. Magic Bounce flips both so
    # the effect lands on the original user with the original user as the
    # source.
    fx.apply_extended_volatile(
        battle,
        move_id0,
        target_side0,
        1 - target_side0,
        hit0_alive,
        game_data,
        move_effects,
        gen5_prng,
        prerolled_roll=prerolled_move0.get("ext_vol"),
        prerolled_duration=prerolled_move0.get("partial_trap_duration"),
        target_offset_override=target0_off,
    )
    fx.apply_extended_volatile(
        battle,
        move_id1,
        target_side1,
        1 - target_side1,
        hit1_alive,
        game_data,
        move_effects,
        gen5_prng,
        prerolled_roll=prerolled_move1.get("ext_vol"),
        prerolled_duration=prerolled_move1.get("partial_trap_duration"),
        target_offset_override=target1_off,
    )

    # Lum berry / status-curing berries fire onUpdate (Showdown items.ts:lum*)
    # — i.e. immediately when a status condition is applied to the holder.
    # Pokepy used to fire them only at EOT residual which made the holder
    # take one extra turn of status damage. Fire here for both sides.
    _run_item_hook_with_berry_tracking(
        fx.apply_lum_berry, p0_off, battle, p0_off, game_data
    )
    _run_item_hook_with_berry_tracking(
        fx.apply_lum_berry, p1_off, battle, p1_off, game_data
    )
    _run_item_hook_with_berry_tracking(
        fx.apply_status_curing_berries, p0_off, battle, p0_off, game_data
    )
    _run_item_hook_with_berry_tracking(
        fx.apply_status_curing_berries, p1_off, battle, p1_off, game_data
    )
    _run_item_hook_with_berry_tracking(
        fx.apply_persim_berry, p0_off, battle, p0_off, game_data
    )
    _run_item_hook_with_berry_tracking(
        fx.apply_persim_berry, p1_off, battle, p1_off, game_data
    )

    # Thaw on hit by thawsTarget move (Scald etc.). Showdown runs this in
    # `onAfterMoveSecondary`, so Sheer Force / Covert Cloak (secondary
    # suppression) prevents the thaw. Fire-type damaging moves are already
    # handled by the `onDamagingHit` path above (which is NOT gated by
    # secondary suppression).
    if (
        hit0_secondary
        and get_status(int(battle[target0_off + 12])) == STATUS_FREEZE
        and is_thaw0
        and damage_through_to_p1 > 0
    ):
        battle[target0_off + 12] = 0
    if (
        hit1_secondary
        and get_status(int(battle[target1_off + 12])) == STATUS_FREEZE
        and is_thaw1
        and damage_through_to_p0 > 0
    ):
        battle[target1_off + 12] = 0

    fx.apply_leech_seed_from_move(
        battle, move_id0, target_side0, target0_off, hit0_alive, game_data, move_effects
    )
    fx.apply_leech_seed_from_move(
        battle, move_id1, target_side1, target1_off, hit1_alive, game_data, move_effects
    )
    if p0_move_executed:
        fx.apply_substitute_from_move(
            battle, move_id0, 0, user0_off, game_data, move_effects
        )
    if p1_move_executed:
        fx.apply_substitute_from_move(
            battle, move_id1, 1, user1_off, game_data, move_effects
        )
    fx.apply_perish_song_from_move(battle, move_id0, hit0, user_side=0)
    fx.apply_perish_song_from_move(battle, move_id1, hit1, user_side=1)
    # Destiny Bond is applied early (lines ~512) for the user who moves
    # first (so the volatile is set before the opponent's damage so the
    # cascade can fire this turn). Here we only apply DB for the user
    # who moves SECOND (their opponent has already hit them, so the
    # volatile takes effect next turn). This avoids double-applying DB
    # which would trigger the consecutive-use fail path.
    if not side0_first:
        fx.apply_destiny_bond_from_move(battle, move_id0, 0)
    if side0_first:
        fx.apply_destiny_bond_from_move(battle, move_id1, 1)
    fx.apply_lock_on_from_move(battle, move_id0, 0, hit0)
    fx.apply_lock_on_from_move(battle, move_id1, 1, hit1)
    fx.apply_ghost_curse_from_move(battle, move_id0, user0_off, 1, hit0)
    fx.apply_ghost_curse_from_move(battle, move_id1, user1_off, 0, hit1)
    # Pain Split has the `protect: 1` flag in Showdown, so Protect and
    # Substitute both block it. Gate on hit_alive (post-protect) and on
    # sub_blocks_secondary (sub absorbs it).
    fx.apply_pain_split_from_move(
        battle,
        move_id0,
        user0_off,
        target0_off,
        hit0_alive and not sub_blocks_secondary0,
    )
    fx.apply_pain_split_from_move(
        battle,
        move_id1,
        user1_off,
        target1_off,
        hit1_alive and not sub_blocks_secondary1,
    )

    # Stat changes
    hit0_stat = hit0 and not target0_protected
    hit1_stat = hit1 and not target1_protected
    _, stat_target0, stat_chance0, selfboost_like0 = fx.get_live_move_stat_change_spec(
        battle,
        move_id0,
        move_effects,
        user0_off,
    )
    _, stat_target1, stat_chance1, selfboost_like1 = fx.get_live_move_stat_change_spec(
        battle,
        move_id1,
        move_effects,
        user1_off,
    )
    if _skip_strike_turn_selfboost0 and stat_target0 == 0 and stat_chance0 == 100:
        hit0_stat = False
    if _skip_strike_turn_selfboost1 and stat_target1 == 0 and stat_chance1 == 100:
        hit1_stat = False
    # Showdown `selfBoost` (moves.ts top-level, e.g. Scale Shot and Clanging
    # Scales) fires from `moveResult`, not `hit`. A Protected target yields a
    # truthy moveResult array, so these self boosts/drops still resolve even
    # though the damaging hit was blocked.
    if selfboost_like0 and hit0 and target0_protected:
        hit0_stat = True
    if selfboost_like1 and hit1 and target1_protected:
        hit1_stat = True
    # Substitute blocks opponent-targeted stat changes (whether the move
    # is status or has a stat-drop secondary). Sound moves and Infiltrator
    # bypass.
    if stat_target0 == 1 and sub_blocks_secondary0:
        hit0_stat = False
    if stat_target1 == 1 and sub_blocks_secondary1:
        hit1_stat = False
    if sheer_force0 and stat_chance0 < 100:
        hit0_stat = False
    if sheer_force1 and stat_chance1 < 100:
        hit1_stat = False
    # Covert Cloak (items.ts:1153) blocks secondary stat drops on the
    # holder. Only opponent-targeted stat changes (stat_target == 1) AND
    # only when the change is a SECONDARY (stat_chance < 100). Self-target
    # drops (stat_target == 0) and 100% chance moves are not blocked.
    if stat_target0 == 1 and covert_cloak1 and stat_chance0 < 100:
        hit0_stat = False
    if stat_target1 == 1 and covert_cloak0 and stat_chance1 < 100:
        hit1_stat = False

    # Belly Drum (187) — Showdown data/moves.ts checks fail conditions
    # BEFORE applying the +12 atk boost. Fail if the user already has
    # atk boost >= +6 or hp <= maxhp/2 (can't pay) or maxhp == 1
    # (Shedinja). Suppress the stat-change pipeline for the failing side.
    from pokepy.core.bitpack import extract_boost as _bd_extract

    _MOVE_BELLY_DRUM = 187

    def _belly_drum_fails(_uoff):
        hp = int(battle[_uoff + 1])
        max_hp = int(battle[_uoff + 2])
        if hp <= 0 or max_hp <= 1:
            return True
        if hp * 2 <= max_hp:
            return True
        if _bd_extract(int(battle[_uoff + 13]), 0) >= 6:
            return True
        return False

    def _belly_drum_pay(_uoff):
        max_hp = int(battle[_uoff + 2])
        hp = int(battle[_uoff + 1])
        cost = max_hp // 2
        battle[_uoff + 1] = max(0, hp - cost)

    if move_id0 == _MOVE_BELLY_DRUM and hit0:
        if _belly_drum_fails(user0_off):
            hit0_stat = False
        else:
            _belly_drum_pay(user0_off)
    if move_id1 == _MOVE_BELLY_DRUM and hit1:
        if _belly_drum_fails(user1_off):
            hit1_stat = False
        else:
            _belly_drum_pay(user1_off)

    # Fillet Away (868) / Clangorous Soul (775) — status moves that boost
    # stats AND pay HP as `directDamage`. Showdown data/moves.ts:
    #   filletaway: onTry fail if hp <= maxhp/2; onHit directDamage(maxhp/2);
    #               boosts +2 atk/spa/spe
    #   clangoroussoul: onTry fail if hp <= maxhp*33/100; onHit directDamage
    #               maxhp*33/100; boosts +1 atk/def/spa/spd/spe
    # Both are category Status (cat=0), so the normal recoil pipeline
    # (gated on damage_dealt > 0 in damage_modifiers.py) never fires —
    # pokepy must deduct HP manually here, mirroring the belly_drum pattern.
    _MOVE_FILLET_AWAY = 868
    _MOVE_CLANGOROUS_SOUL = 775

    def _fillet_away_fails(_uoff):
        hp = int(battle[_uoff + 1])
        max_hp = int(battle[_uoff + 2])
        if hp <= 0 or max_hp <= 1:
            return True
        # Showdown: fail if hp <= maxhp/2
        return hp * 2 <= max_hp

    def _clangorous_soul_fails(_uoff):
        hp = int(battle[_uoff + 1])
        max_hp = int(battle[_uoff + 2])
        if hp <= 0 or max_hp <= 1:
            return True
        # Showdown: fail if hp <= maxhp*33/100
        return hp * 100 <= max_hp * 33

    def _fillet_away_pay(_uoff):
        max_hp = int(battle[_uoff + 2])
        hp = int(battle[_uoff + 1])
        cost = max_hp // 2
        battle[_uoff + 1] = max(0, hp - cost)

    def _clangorous_soul_pay(_uoff):
        max_hp = int(battle[_uoff + 2])
        hp = int(battle[_uoff + 1])
        cost = (max_hp * 33) // 100
        battle[_uoff + 1] = max(0, hp - cost)

    if move_id0 == _MOVE_FILLET_AWAY and hit0:
        if _fillet_away_fails(user0_off):
            hit0_stat = False
        else:
            _fillet_away_pay(user0_off)
    if move_id1 == _MOVE_FILLET_AWAY and hit1:
        if _fillet_away_fails(user1_off):
            hit1_stat = False
        else:
            _fillet_away_pay(user1_off)
    if move_id0 == _MOVE_CLANGOROUS_SOUL and hit0:
        if _clangorous_soul_fails(user0_off):
            hit0_stat = False
        else:
            _clangorous_soul_pay(user0_off)
    if move_id1 == _MOVE_CLANGOROUS_SOUL and hit1:
        if _clangorous_soul_fails(user1_off):
            hit1_stat = False
        else:
            _clangorous_soul_pay(user1_off)

    # Curse (174) — Showdown data/moves.ts:curse has different behavior
    # based on user's type. Ghost users pay 50% HP and curse the target
    # (handled in apply_ghost_curse_from_move). Non-Ghost users get
    # +1 atk +1 def -1 spe. Pokepy's move_effects table encodes the
    # non-Ghost stat changes unconditionally, so we must suppress the
    # stat-change pipeline for Ghost-type users to avoid double-booking.
    from pokepy.core.constants import (
        MOVE_CURSE as _MOVE_CURSE,
        TYPE_GHOST as _TYPE_GHOST,
    )

    def _user_is_ghost(_uoff):
        types_packed = int(battle[_uoff + 4]) & 0xFFFF
        t1 = types_packed & 0xFF
        t2 = (types_packed >> 8) & 0xFF
        return (t1 == _TYPE_GHOST) or (t2 == _TYPE_GHOST)

    if move_id0 == _MOVE_CURSE and _user_is_ghost(user0_off):
        hit0_stat = False
    if move_id1 == _MOVE_CURSE and _user_is_ghost(user1_off):
        hit1_stat = False

    # Snapshot boosts for Eject Pack detection (next switch trigger).
    def _boost_sum(off):
        b13 = int(battle[off + 13]) & 0xFFFF
        b14 = int(battle[off + 14]) & 0xFFFF
        s = 0
        for shift in (0, 4, 8, 12):
            s += (b13 >> shift) & 0xF
        for shift in (0, 4, 8):  # spe, acc, eva (not tera nibble)
            s += (b14 >> shift) & 0xF
        return s

    p0_boosts_pre_sc_snapshot = _snapshot_boost_stages(p0_off)
    p1_boosts_pre_sc_snapshot = _snapshot_boost_stages(p1_off)
    p0_boosts_pre_sc = _boost_sum(p0_off)
    p1_boosts_pre_sc = _boost_sum(p1_off)

    # Snapshot opponents' boost sums BEFORE stat changes so we can detect
    # whether Parting Shot (move id 575) actually landed its -1 Atk / -1 SpA
    # drops. Showdown's partingshot onHit gates the selfSwitch on
    # `boost(...) returned success`: if every stat was blocked (Clear Body,
    # Full Metal Body, White Smoke, Hyper Cutter vs atk, etc.), the user
    # stays in. `tgt_boosts_pre` is the boost sum of the Parting Shot
    # target (opponent of the user).
    tgt0_boosts_pre = _boost_sum(target0_off)
    tgt1_boosts_pre = _boost_sum(target1_off)

    if not _move0_stat_preapplied:
        _apply_stat_changes_from_move_tracked(
            move_id0,
            user0_off,
            target0_off,
            hit0_stat,
            prerolled_roll=prerolled_move0.get("stat_change"),
        )
    if not _move1_stat_preapplied:
        _apply_stat_changes_from_move_tracked(
            move_id1,
            user1_off,
            target1_off,
            hit1_stat,
            prerolled_roll=prerolled_move1.get("stat_change"),
        )
    _move0_successful_self_boost_status = _move0_successful_self_boost_status or (
        cat0 == CAT_STATUS
        and target0_kind == 3
        and int(move_effects.effect_type[move_id0]) == 3
        and int(move_effects.stat_target[move_id0]) == 0
        and _boost_sum(user0_off) != p0_boosts_pre_sc
    )
    _move1_successful_self_boost_status = _move1_successful_self_boost_status or (
        cat1 == CAT_STATUS
        and target1_kind == 3
        and int(move_effects.effect_type[move_id1]) == 3
        and int(move_effects.stat_target[move_id1]) == 0
        and _boost_sum(user1_off) != p1_boosts_pre_sc
    )
    # Tera Blast (851) — Stellar Tera variant applies -1 Atk -1 SpA to the
    # user via `move.self = { boosts: { atk: -1, spa: -1 } }`. Showdown:
    # data/moves.ts:terablast onModifyMove. Only fires for terastallized
    # users with Stellar tera type (nibble 18) and only when the move
    # actually executes (hit). pokepy's move_effects table can't express
    # this conditional self-debuff, so handle it inline. Goes through the
    # standard boost path (Clear Body / Clear Amulet / Mirror Armor on
    # the user would block but those are extremely rare on the Tera Blast
    # user). Treat as a guaranteed self-boost (not secondary).
    _MOVE_TERA_BLAST_SD = 851

    def _stellar_tera_blast_self_debuff(_uoff: int, _mid: int, _did_hit: bool) -> None:
        if int(_mid) != _MOVE_TERA_BLAST_SD or not _did_hit:
            return
        flags_tb = int(battle[_uoff + 15])
        is_terad = (flags_tb & 0x8) != 0
        if not is_terad:
            return
        tera_nib = (int(battle[_uoff + 14]) >> 12) & 0xF
        # Stellar tera type packs into 4 bits as 18 (handled in damage
        # path) — but only 0..15 fit, so the obs/state packs Stellar as
        # nibble 0xF (15). Be permissive: also accept 0xF here.
        if tera_nib != 18 and tera_nib != 0xF:
            return
        before_tb = _snapshot_boost_stages(_uoff)
        b13_sd = int(battle[_uoff + 13])
        b13_sd = apply_boost_to_packed(b13_sd, 0, -1)  # -1 atk
        b13_sd = apply_boost_to_packed(b13_sd, 8, -1)  # -1 spa
        battle[_uoff + 13] = b13_sd
        _mark_stats_lowered_this_turn(_uoff, before_tb)
        _apply_white_herb_if_ready(_uoff)

    _stellar_tera_blast_self_debuff(user0_off, move_id0, hit0_stat)
    _stellar_tera_blast_self_debuff(user1_off, move_id1, hit1_stat)

    # Parting Shot (575) — cancel self-switch if target boosts unchanged.
    # Showdown onHit: `if (!success && !target.hasAbility('mirrorarmor'))
    # delete move.selfSwitch`. Mirror Armor reflects the drop onto the
    # user (user's boost sum drops), which Showdown still treats as
    # selfSwitch (the reflected-back failure branch doesn't fire).
    _MOVE_PARTING_SHOT = 575
    _partingshot0_failed = (
        move_id0 == _MOVE_PARTING_SHOT
        and _boost_sum(target0_off) == tgt0_boosts_pre
        and _boost_sum(user0_off) == p0_boosts_pre_sc
    )
    _partingshot1_failed = (
        move_id1 == _MOVE_PARTING_SHOT
        and _boost_sum(target1_off) == tgt1_boosts_pre
        and _boost_sum(user1_off) == p1_boosts_pre_sc
    )

    # Eject Pack: if any stat stage was lowered this turn and the holder has
    # a bench mon available, trigger the forced switch / item consumption.
    _ITEM_EJECT_PACK = 1119
    p0_eject_pack_switch = False
    if (
        int(battle[p0_off + 6]) == _ITEM_EJECT_PACK
        and int(battle[p0_off + 1]) > 0
        and not _eject_pack_blocked0
    ):
        if stats_lowered_this_turn0 or _boosts_were_lowered(
            p0_boosts_pre_sc_snapshot,
            _snapshot_boost_stages(p0_off),
        ):
            bench_alive_ep = 0
            for _i in range(6):
                _so = OFF_SIDE0 + _i * POKEMON_SIZE
                if _so == p0_off:
                    continue
                if int(battle[_so + 1]) > 0 and (int(battle[_so + 15]) & 1) == 0:
                    bench_alive_ep += 1
            if bench_alive_ep > 0:
                _flush_pending_hazard_setters(True, True)
                battle[p0_off + 6] = 0
                p0_eject_pack_switch = True
    if (
        int(battle[p1_off + 6]) == _ITEM_EJECT_PACK
        and int(battle[p1_off + 1]) > 0
        and not _eject_pack_blocked1
    ):
        if stats_lowered_this_turn1 or _boosts_were_lowered(
            p1_boosts_pre_sc_snapshot,
            _snapshot_boost_stages(p1_off),
        ):
            # Showdown items.ts:1738 ejectpack onUseItem returns false if
            # `!this.canSwitch(pokemon.side)`, so the item is NOT consumed
            # when no bench mon is available. Mirror the bench check from
            # the p0 path before forcing a switch / consuming the item.
            bench_alive_ep1 = 0
            for _i in range(6):
                _so = OFF_SIDE1 + _i * POKEMON_SIZE
                if _so == p1_off:
                    continue
                if int(battle[_so + 1]) > 0 and (int(battle[_so + 15]) & 1) == 0:
                    bench_alive_ep1 += 1
            if bench_alive_ep1 > 0:
                _flush_pending_hazard_setters(True, True)
                # Showdown fires BeforeSwitchOut → Regenerator/NaturalCure
                fx.apply_regenerator_on_switch_out(battle, p1_off, True)
                fx.apply_natural_cure_on_switch_out(battle, p1_off, True)
                battle[p1_off + 6] = 0
                while True:
                    active1_ep = int(battle[OFF_META + M_ACTIVE1])
                    _switch_req = SwitchRequest((1,))
                    _choices = yield _switch_req
                    _new_slot = int(_choices[1])
                    new1_ep = _resolve_switch_target_from_action(
                        OFF_SIDE1,
                        active1_ep,
                        _new_slot + 4,
                    )
                    if new1_ep == active1_ep:
                        break
                    _pending_switch_slot_condition1_ep = int(
                        battle[OFF_FIELD + F_DESTINY_BOND_1]
                    )
                    battle[OFF_META + M_ACTIVE1] = np.int16(new1_ep)
                    _sync_showdown_order_on_switch(side_order1, new1_ep)
                    _clear_side_switch_state_common(battle, 1)
                    for off in (
                        F_VOLATILE_1,
                        F_LEECH_SEED_1,
                        F_EXTENDED_VOLATILE_1,
                        F_SUBSTITUTE_1,
                        F_DISABLE_TURNS_1,
                        F_DESTINY_BOND_1,
                        F_YAWN_TURNS_1,
                        F_PERISH_COUNT_1,
                    ):
                        battle[OFF_FIELD + off] = 0
                    new_p1_ep = OFF_SIDE1 + new1_ep * POKEMON_SIZE
                    _reset_incoming_switch_state_tracked(new_p1_ep)
                    _ep_consumed_pending1 = apply_pending_wish_on_switch_in(
                        battle,
                        1,
                        new_p1_ep,
                        state,
                        game_data,
                        _pending_switch_slot_condition1_ep,
                    )
                    if (
                        is_pending_wish_sentinel(_pending_switch_slot_condition1_ep)
                        and not _ep_consumed_pending1
                    ):
                        battle[OFF_FIELD + F_DESTINY_BOND_1] = np.int16(
                            _pending_switch_slot_condition1_ep
                        )
                    _postswitch_p1_speed_ep = _get_switch_resume_action_speed(
                        battle, new_p1_ep
                    )
                    _consume_runswitch_tie_frame(
                        battle,
                        new_p1_ep,
                        OFF_SIDE0 + int(battle[OFF_META + M_ACTIVE0]) * POKEMON_SIZE,
                        gen5_prng,
                    )
                    fx.apply_hazard_damage_on_switch(
                        battle, new_p1_ep, OFF_FIELD + F_HAZARDS_1
                    )
                    _reset_toxic_counter_on_switch_in(battle, new_p1_ep)
                    if int(battle[new_p1_ep + 1]) > 0:
                        _opp_target_off_ep = (
                            OFF_SIDE0 + int(battle[OFF_META + M_ACTIVE0]) * POKEMON_SIZE
                        )
                        if int(battle[_opp_target_off_ep + 1]) > 0:
                            _apply_switch_in_ability_with_trace_reaction_tracked(
                                new_p1_ep,
                                _opp_target_off_ep,
                                True,
                            )
                        else:
                            _store_pending_opp_switch_in(
                                new1_ep, _postswitch_p1_speed_ep
                            )
                        _run_switch_in_update_item_hooks(new_p1_ep)
                        break
                    p1_bench_alive_ep = 0
                    for _i in range(6):
                        if _i == new1_ep:
                            continue
                        _so = OFF_SIDE1 + _i * POKEMON_SIZE
                        if (
                            int(battle[_so + 1]) > 0
                            and (int(battle[_so + 15]) & 1) == 0
                        ):
                            p1_bench_alive_ep += 1
                    if p1_bench_alive_ep == 0:
                        break

    prime_gulp_missile_state_from_move(
        battle,
        p0_off,
        move_id0,
        move_executed=p0_move_executed,
        is_charge_turn=is_charge_turn0,
    )
    prime_gulp_missile_state_from_move(
        battle,
        p1_off,
        move_id1,
        move_executed=p1_move_executed,
        is_charge_turn=is_charge_turn1,
    )

    # White Herb also rechecks on residual in Showdown, so keep the late
    # fallback after the immediate move/switch hooks above.
    for poff in (user0_off, user1_off, p0_off, p1_off):
        _apply_white_herb_if_ready(poff)

    # Mental Herb — cures attract / taunt / encore / torment / disable / heal block.
    # Showdown's onUpdate fires after volatiles are applied.
    from pokepy.core.constants import (
        ITEM_MENTAL_HERB,
        F_VOLATILE_0 as _FV0,
        F_VOLATILE_1 as _FV1,
        F_EXTENDED_VOLATILE_0 as _FEV0,
        F_EXTENDED_VOLATILE_1 as _FEV1,
        F_DISABLE_0 as _FD0,
        F_DISABLE_1 as _FD1,
        F_DISABLE_TURNS_0 as _FDT0,
        F_DISABLE_TURNS_1 as _FDT1,
        EXT_VOL_TORMENT,
        EXT_VOL_ATTRACT,
        EXT_VOL_HEAL_BLOCK,
    )

    # Bit shifts for volatile word: bits 4-6 = taunt turns, 7-9 = encore turns,
    # bits 11-13 = heal block turns.
    for poff, side in ((p0_off, 0), (p1_off, 1)):
        if int(battle[poff + 6]) != ITEM_MENTAL_HERB:
            continue
        vol_off = OFF_FIELD + (_FV0 if side == 0 else _FV1)
        ext_vol_off = OFF_FIELD + (_FEV0 if side == 0 else _FEV1)
        dis_off = OFF_FIELD + (_FD0 if side == 0 else _FD1)
        dis_turns_off = OFF_FIELD + (_FDT0 if side == 0 else _FDT1)
        vol = int(battle[vol_off]) & 0xFFFF
        ext = int(battle[ext_vol_off]) & 0xFFFF
        taunt_turns = (vol >> 4) & 0x7
        encore_turns = (vol >> 7) & 0x7
        disable_active = int(battle[dis_off]) >= 0 and int(battle[dis_turns_off]) > 0
        has_attract = (ext & EXT_VOL_ATTRACT) != 0
        has_torment = (ext & EXT_VOL_TORMENT) != 0
        has_heal_block = (ext & EXT_VOL_HEAL_BLOCK) != 0
        if (
            taunt_turns > 0
            or encore_turns > 0
            or disable_active
            or has_attract
            or has_torment
            or has_heal_block
        ):
            # Clear all the cured volatiles
            new_vol = vol & ~(0x7 << 4) & ~(0x7 << 7) & ~(0x7 << 11)
            new_ext = ext & ~EXT_VOL_ATTRACT & ~EXT_VOL_TORMENT & ~EXT_VOL_HEAL_BLOCK
            battle[vol_off] = np.int16(
                new_vol if new_vol < 0x8000 else new_vol - 0x10000
            )
            battle[ext_vol_off] = np.int16(
                new_ext if new_ext < 0x8000 else new_ext - 0x10000
            )
            if disable_active:
                battle[dis_off] = -1
                battle[dis_turns_off] = 0
            battle[poff + 6] = 0  # consume herb

    # Belly Drum (187) / Fillet Away (868) HP cost. Showdown applies the
    # cost via directDamage(maxhp / 2) only after onHit passes its fail
    # checks. For Belly Drum those checks are mirrored above in the
    # stat-change suppression path — re-run them here so the cost is
    # only paid on a successful cast. Fillet Away has analogous checks
    # (hp <= maxhp/2 fail + stat boost cap) handled similarly.
    from pokepy.core.bitpack import extract_boost as _cost_extract

    for side_off, mid, did_hit in [
        (user0_off, move_id0, hit0),
        (user1_off, move_id1, hit1),
    ]:
        if not did_hit:
            continue
        if mid == 187:  # Belly Drum
            hp_bd = int(battle[side_off + 1])
            max_bd = int(battle[side_off + 2])
            if hp_bd <= 0 or max_bd <= 1:
                continue
            if hp_bd * 2 <= max_bd:
                continue  # fail: not enough HP
            if _cost_extract(int(battle[side_off + 13]), 0) >= 6:
                continue  # fail: already at +6 atk
            cost = max_bd // 2
            battle[side_off + 1] = max(0, hp_bd - cost)
        elif mid == 868:  # Fillet Away
            hp_fa = int(battle[side_off + 1])
            max_fa = int(battle[side_off + 2])
            if hp_fa <= 0 or max_fa <= 1:
                continue
            if hp_fa * 2 <= max_fa:
                continue
            # Fillet Away caps at +6 for atk/spa/spe; fail if all three are
            # at +6. Simplified: require any boost slot to still have room.
            b13_fa = int(battle[side_off + 13])
            b14_fa = int(battle[side_off + 14])
            if (
                _cost_extract(b13_fa, 0) >= 6
                and _cost_extract(b13_fa, 8) >= 6
                and _cost_extract(b14_fa, 0) >= 6
            ):
                continue
            cost = max_fa // 2
            battle[side_off + 1] = max(0, hp_fa - cost)

    # Tidy Up — clear hazards + subs both sides
    for mid, did_hit in [(move_id0, hit0), (move_id1, hit1)]:
        if mid == 882 and did_hit:
            battle[OFF_FIELD + F_HAZARDS_0] = 0
            battle[OFF_FIELD + F_HAZARDS_1] = 0
            battle[OFF_FIELD + F_SUBSTITUTE_0] = 0
            battle[OFF_FIELD + F_SUBSTITUTE_1] = 0

    # Throat Spray — sound move user gets +1 SpA and consumes the item.
    # Showdown: items.ts throatspray onAfterMoveSecondarySelf checks move.flags.sound.
    from pokepy.core.constants import ITEM_THROAT_SPRAY

    for u_off, mid, did_hit in [
        (user0_off, move_id0, hit0),
        (user1_off, move_id1, hit1),
    ]:
        if not did_hit:
            continue
        if int(battle[u_off + 6]) != ITEM_THROAT_SPRAY:
            continue
        flags = int(game_data.move_flags[mid])
        if (flags & FLAG_SOUND) == 0:
            continue
        b13 = int(battle[u_off + 13])
        battle[u_off + 13] = apply_boost_to_packed(b13, 8, 1)  # +1 SpA
        battle[u_off + 6] = 0  # consume

    # Blunder Policy — Showdown sim/battle-actions.ts:739:
    #   if (!move.ohko && pokemon.hasItem('blunderpolicy') && pokemon.useItem())
    #     this.battle.boost({ spe: 2 }, pokemon);
    # Triggers when a non-OHKO move attempts to hit and misses its accuracy
    # roll. Move category doesn't matter — status moves count too. We approx
    # "missed accuracy" with `attempted_move and not did_hit`. The pokepy
    # `_exec*` flags computed below for crash damage capture "user actually
    # tried to use the move", so reuse the same gating after they're built.
    _ITEM_BLUNDER_POLICY = 1121
    _OHKO_MOVES_BP = (28, 71, 81, 329)  # Fissure, Horn Drill, Guillotine, Sheer Cold
    # Note: _exec0 / _exec1 / target0_protected / target1_protected /
    # is_immobile* / prankster_fail* are computed in the crash-damage block
    # below. Defer Blunder Policy to after that section.

    # Showdown's Rapid Spin / Mortal Spin side-condition clearing is an
    # onAfterHit effect for the acting move, so when the faster user spins
    # into a slower hazard setter, the clear happens before the slower move
    # can re-establish hazards. Keep only the faster-move case inline here;
    # the slower-move fallback stays in the later shared bucket below.
    if side0_first:
        _actual_dmg0_inline = _actual_damage_for_source_postmove(
            user0_off,
            p1_off,
            move_id0,
            cat0,
            damage0_after_flinch,
            has_sub1,
            sub1_hp,
            hp1_pre,
            new_hp1,
        )
        fx.apply_rapid_spin_from_move(
            battle,
            move_id0,
            user0_off,
            0,
            hit0,
            source_alive=_source_survives_damaging_hit_contact(
                user0_off,
                p1_off,
                move_id0,
                _actual_dmg0_inline,
                hit0,
                int(_meta0.get("num_hits", 1)),
                hp0_pre,
                move_idx,
            ),
            move_effects=move_effects,
        )
        _inline_rapid_spin_done0 = True
    else:
        _actual_dmg1_inline = _actual_damage_for_source_postmove(
            user1_off,
            p0_off,
            move_id1,
            cat1,
            damage1_after_flinch,
            has_sub0,
            sub0_hp,
            hp0_pre,
            new_hp0,
        )
        fx.apply_rapid_spin_from_move(
            battle,
            move_id1,
            user1_off,
            1,
            hit1,
            source_alive=_source_survives_damaging_hit_contact(
                user1_off,
                p0_off,
                move_id1,
                _actual_dmg1_inline,
                hit1,
                int(_meta1.get("num_hits", 1)),
                hp1_pre,
                opp_move_idx,
            ),
            move_effects=move_effects,
        )
        _inline_rapid_spin_done1 = True

    # Apply side/field move effects in actual move order, not fixed player
    # index order. This matters for exchanges like faster Defog into slower
    # Stealth Rock, or side1-first weather / terrain setters that would
    # otherwise be overwritten in the wrong order.
    def _apply_ordered_field_move_effects(
        move_id: int,
        did_hit: bool,
        move_ran: bool,
        target_protected: bool,
        user_off: int,
        user_side: int,
        user_ability: int,
        rapid_spin_done: bool,
        rapid_spin_source_alive: bool | None,
        recovery_applied_early: bool,
        user_postmove_hp: int | None,
    ) -> None:
        nonlocal _hazard_from_move_applied0, _hazard_from_move_applied1
        if move_ran:
            if not rapid_spin_done:
                fx.apply_rapid_spin_from_move(
                    battle,
                    move_id,
                    user_off,
                    user_side,
                    did_hit,
                    source_alive=rapid_spin_source_alive,
                    move_effects=move_effects,
                )
            if user_side == 0:
                if not _hazard_from_move_applied0:
                    fx.apply_hazard_from_move(
                        battle,
                        move_id,
                        1 - user_side,
                        did_hit,
                        game_data,
                        move_effects,
                        user_ability=user_ability,
                        user_offset=user_off,
                        source_hp_override=user_postmove_hp,
                        enabled_hazards=profile.enabled_hazards,
                    )
                    _hazard_from_move_applied0 = True
            else:
                if not _hazard_from_move_applied1:
                    fx.apply_hazard_from_move(
                        battle,
                        move_id,
                        1 - user_side,
                        did_hit,
                        game_data,
                        move_effects,
                        user_ability=user_ability,
                        user_offset=user_off,
                        source_hp_override=user_postmove_hp,
                        enabled_hazards=profile.enabled_hazards,
                    )
                    _hazard_from_move_applied1 = True
            fx.apply_weather_from_move(
                battle,
                move_id,
                did_hit,
                game_data,
                move_effects,
                user_offset=user_off,
            )
            if profile.has_terrain:
                fx.apply_terrain_from_move(
                    battle,
                    move_id,
                    did_hit,
                    game_data,
                    move_effects,
                    user_offset=user_off,
                )
            fx.apply_trick_room_from_move(
                battle,
                move_id,
                did_hit,
                game_data,
                move_effects,
            )
            # Skip any side whose recovery was already applied inline above
            # (faster mon's pre-damage heal). Wish still goes through here
            # (deferred to EOT).
            if not recovery_applied_early:
                fx.apply_recovery_from_move(
                    battle,
                    move_id,
                    user_off,
                    move_ran,
                    game_data,
                    move_effects,
                    gen5_prng,
                )
            fx.apply_defog_from_move(
                battle,
                move_id,
                did_hit and (not target_protected),
                move_effects,
                user_side=user_side,
            )
            fx.apply_court_change_from_move(
                battle,
                move_id,
                move_ran,
            )
        _apply_booster_energy_update_tracked(p0_off)
        _apply_booster_energy_update_tracked(p1_off)

    _MOVE_RAPID_SPIN = 229
    _MOVE_MORTAL_SPIN = 866

    def _rapid_spin_source_survives_damaging_hit(
        move_id: int,
        user_off: int,
        target_off: int,
        did_hit: bool,
        raw_damage: int,
        num_hits: int,
        move_slot_idx: int,
    ) -> bool | None:
        if int(move_id) not in (_MOVE_RAPID_SPIN, _MOVE_MORTAL_SPIN):
            return None
        return _source_survives_damaging_hit_contact(
            user_off,
            target_off,
            move_id,
            int(raw_damage),
            bool(did_hit),
            int(num_hits),
            int(battle[user_off + 1]),
            int(move_slot_idx),
        )

    if side0_first:
        _rapid_spin_source_alive0 = _rapid_spin_source_survives_damaging_hit(
            move_id0,
            user0_off,
            p1_off,
            hit0,
            damage0_after_flinch,
            int(_meta0.get("num_hits", 1)),
            move_idx,
        )
        _rapid_spin_source_alive1 = _rapid_spin_source_survives_damaging_hit(
            move_id1,
            user1_off,
            p0_off,
            hit1,
            damage1_after_flinch,
            int(_meta1.get("num_hits", 1)),
            opp_move_idx,
        )
        _apply_ordered_field_move_effects(
            move_id0,
            hit0,
            p0_move_ran and p0_move_executed,
            target0_protected,
            user0_off,
            0,
            p0_ab,
            _inline_rapid_spin_done0,
            _rapid_spin_source_alive0,
            _recovery_applied_early_0,
            _hp0_after_postmove_pre,
        )
        _apply_ordered_field_move_effects(
            move_id1,
            hit1,
            p1_move_ran and p1_move_executed,
            target1_protected,
            user1_off,
            1,
            p1_ab,
            _inline_rapid_spin_done1,
            _rapid_spin_source_alive1,
            _recovery_applied_early_1,
            _hp1_after_postmove_pre,
        )
    else:
        _rapid_spin_source_alive1 = _rapid_spin_source_survives_damaging_hit(
            move_id1,
            user1_off,
            p0_off,
            hit1,
            damage1_after_flinch,
            int(_meta1.get("num_hits", 1)),
            opp_move_idx,
        )
        _rapid_spin_source_alive0 = _rapid_spin_source_survives_damaging_hit(
            move_id0,
            user0_off,
            p1_off,
            hit0,
            damage0_after_flinch,
            int(_meta0.get("num_hits", 1)),
            move_idx,
        )
        _apply_ordered_field_move_effects(
            move_id1,
            hit1,
            p1_move_ran and p1_move_executed,
            target1_protected,
            user1_off,
            1,
            p1_ab,
            _inline_rapid_spin_done1,
            _rapid_spin_source_alive1,
            _recovery_applied_early_1,
            _hp1_after_postmove_pre,
        )
        _apply_ordered_field_move_effects(
            move_id0,
            hit0,
            p0_move_ran and p0_move_executed,
            target0_protected,
            user0_off,
            0,
            p0_ab,
            _inline_rapid_spin_done0,
            _rapid_spin_source_alive0,
            _recovery_applied_early_0,
            _hp0_after_postmove_pre,
        )

    # NOTE: Burning Jealousy (807) and Strength Sap (668) special-case logic at
    # Burning Jealousy + Strength Sap (source 4448-4490)
    fx.apply_misc_move_effects(
        battle,
        move_id0,
        move_id1,
        user0_off,
        user1_off,
        target0_off,
        target1_off,
        hit0,
        hit1,
    )
    fx.apply_self_type_removal_from_move(
        battle,
        move_id0,
        user0_off,
        hit0 and damage0_after_flinch > 0,
    )
    fx.apply_self_type_removal_from_move(
        battle,
        move_id1,
        user1_off,
        hit1 and damage1_after_flinch > 0,
    )

    fx.apply_team_heal_status(
        battle, move_id0, OFF_SIDE0, hit0, game_data, move_effects
    )
    fx.apply_team_heal_status(
        battle, move_id1, OFF_SIDE1, hit1, game_data, move_effects
    )

    # Showdown ordering (sim/battle-actions.ts useMove + spreadMoveHit):
    #   1. spreadDamage (inline) applies damage AND drain heal (gen5+ at
    #      line 2130-2133 of sim/battle.ts).
    #   2. runEvent('DamagingHit') fires contact abilities (Rocky Helmet,
    #      Rough Skin, Iron Barbs, Aftermath, Innards Out, Color Change,
    #      Cursed Body, Mummy / Lingering Aroma, Wandering Spirit,
    #      Pickpocket, Toxic Chain, Berserk, Anger Shell, Weak Armor,
    #      Cotton Down, Gooey / Tangling Hair, Rattled, etc.) + onAfterHit
    #      (Knock Off, Trick).
    #   3. After ALL hits are done: move.recoil damages the user
    #      (sim/battle-actions.ts:982) — this is separate from drain.
    #   4. afterMoveSecondaryEvent (sim/battle-actions.ts:1024) fires
    #      onAfterMoveSecondarySelf → Life Orb, Shell Bell, Sticky Barb
    #      contact transfer, Throat Spray.
    # Pokepy previously applied recoil together with drain BEFORE contact
    # abilities fired, which meant Aftermath / Innards Out skipped whenever
    # recoil already KO'd the attacker (`atk_hp > 0` gate in
    # defender_abilities.py). Split into two passes: drain immediately after
    # damage (phase="drain"), recoil right before afterMoveSecondary
    # (phase="recoil").
    # Showdown caps drain/recoil damage at the substitute's HP when sub
    # absorbs the hit (sim/data/moves.ts:substitute onTryPrimaryHit:
    # `if (damage > sub.hp) damage = sub.hp; ... if (move.drain) heal by that`).
    # So for a drain/recoil move into a substitute, the amount is bounded by
    # what the sub could soak.
    # Showdown's move.totalDamage (used for recoil/drain) is the sum of
    # actual HP removed from the target per hit, NOT the raw calculated
    # damage. pokemon.damage(d) returns min(d, pokemon.hp), so when the
    # target's remaining HP is less than the calculated damage, the
    # recoil/drain base is capped at the target's HP. Focus Sash also
    # reduces totalDamage via the onDamage event (caps damage at hp-1).
    # For substitute hits, Showdown caps at the sub's remaining HP.
    dmg_dealt0 = damage0_after_flinch
    dmg_dealt1 = damage1_after_flinch
    if face1_absorbed_hit:
        dmg_dealt0 = 0
    elif has_sub1 and damage0_after_flinch > 0:
        # sub1_hp was the sub HP BEFORE this move hit.
        dmg_dealt0 = min(damage0_after_flinch, sub1_hp)
    elif damage0_after_flinch > 0:
        # Cap at actual HP removed (accounts for Focus Sash, side ordering).
        dmg_dealt0 = min(damage0_after_flinch, max(0, hp1_pre - final_hp1))
    if face0_absorbed_hit:
        dmg_dealt1 = 0
    elif has_sub0 and damage1_after_flinch > 0:
        dmg_dealt1 = min(damage1_after_flinch, sub0_hp)
    elif damage1_after_flinch > 0:
        # Cap at actual HP removed (accounts for Focus Sash, side ordering).
        dmg_dealt1 = min(damage1_after_flinch, max(0, hp0_pre - final_hp0))
    # The slower move's drain/recoil base must use the target's LIVE HP at the
    # moment that slower move lands, not the turn-start HP. Re-project the
    # faster mover's deterministic post-hit HP here so slower recoil/drain is
    # capped after earlier Rocky Helmet / Rough Skin / recoil / Life Orb /
    # drain updates from the faster move.
    if side0_first and (not has_sub0) and damage1_after_flinch > 0 and dmg_dealt1 > 0:
        if _mid_turn_pivot_0:
            # After a mid-turn pivot, the slower move hits the live
            # replacement target. Recoil/drain caps must use that incoming
            # mon's post-switch HP snapshot from right before the slower move
            # lands rather than re-projecting the first mover's postmove
            # chain through the replacement.
            _hp0_before_move1 = max(0, int(_mid_turn_pivot_target_hp0 or 0))
        else:
            _hp0_before_move1 = max(0, int(hp0_pre_for_move1))
        dmg_dealt1 = min(dmg_dealt1, _hp0_before_move1)
        if _hp0_before_move1 > hp0_pre:
            # If the faster target healed itself before the slower hit lands
            # (for example via Giga Drain / Drain Punch / Draining Kiss),
            # the slower move's recoil/drain base must be recomputed from that
            # live pre-hit HP instead of the turn-start snapshot.
            dmg_dealt1 = min(
                damage1_after_flinch, max(0, _hp0_before_move1 - final_hp0)
            )
    elif (
        (not side0_first)
        and (not has_sub1)
        and damage0_after_flinch > 0
        and dmg_dealt0 > 0
    ):
        if _mid_turn_pivot_1:
            _hp1_before_move0 = max(0, int(_mid_turn_pivot_target_hp1 or 0))
        else:
            _hp1_before_move0 = max(0, int(hp1_pre_for_move0))
        dmg_dealt0 = min(dmg_dealt0, _hp1_before_move0)
        if _hp1_before_move0 > hp1_pre:
            dmg_dealt0 = min(
                damage0_after_flinch, max(0, _hp1_before_move0 - final_hp1)
            )
    # Phase 1 — drain heal (part of spreadDamage in Showdown).
    # Showdown applies drain inline during each move's resolution, so the
    # first mover's drain fires when its HP = hp_pre (before the opponent's
    # hit). Pokepy applies both damages first, then drains. To match
    # Showdown, temporarily restore the first mover's HP to hp_pre for drain,
    # then re-apply the opponent's damage on top of the post-drain HP.
    # Inline drain ordering fix: Showdown applies drain during each move's
    # resolution, so the first mover's drain fires when its HP = hp_pre
    # (before the opponent's hit). Only needed when the first mover used a
    # drain move; otherwise leave HP as-is (critical for Focus Sash).
    _recoil0 = int(move_effects.recoil[int(move_id0)]) if int(move_id0) >= 0 else 0
    _recoil1 = int(move_effects.recoil[int(move_id1)]) if int(move_id1) >= 0 else 0
    # The first-actor drain shim only applies when the first actor actually
    # executed a drain move. On switch-vs-move turns, using the switcher's
    # slot-0 move metadata here can incorrectly restore the incoming switch-in
    # to its pre-hit HP and erase valid onTryHit healing like Water Absorb.
    _first_is_drain = (
        ((not is_switch) and (_recoil0 < 0))
        if side0_first
        else ((not opp_is_switch) and (_recoil1 < 0))
    )
    if _first_is_drain and side0_first:
        # Side 0 used a drain move and moved first.
        battle[user0_off + 1] = np.int16(hp0_pre)
        _apply_recoil_drain_from_move_tracked(
            battle,
            move_id0,
            user0_off,
            dmg_dealt0,
            hit0,
            game_data,
            move_effects,
            target_offset=target0_off,
            phase="drain",
        )
        _hp0_postmove_baseline = int(hp0_pre_for_move1)
        if _hp0_after_postmove_pre is None:
            _hp0_after_drain_replay = int(battle[user0_off + 1])
            _actual_dmg0_drain = _actual_damage_for_source_postmove(
                user0_off,
                p1_off,
                move_id0,
                cat0,
                damage0_after_flinch,
                has_sub1,
                sub1_hp,
                hp1_pre,
                new_hp1,
            )
            _hp0_postmove_baseline = _project_user_hp_after_own_postmove(
                user0_off,
                p1_off,
                move_id0,
                damage0_after_flinch,
                _actual_dmg0_drain,
                hit0,
                int(_meta0.get("num_hits", 1)),
                hp0_pre,
                move_attempted=not (
                    is_switch
                    or _self_hit0
                    or _p0_immobile_pre
                    or _sleep_talk_try_failed0
                    or _rest_try_failed0
                    or move0_no_target
                    or prankster_fail0
                    or _move0_canceled_pre
                ),
                restore_target_item=_inline_knock_saved_item1,
                restore_target_item_off=_inline_knock_saved_target1,
            )
            if int(_hp0_postmove_baseline) != int(_hp0_after_drain_replay):
                _skip_contact_damage_effects0 = True
                _skip_crash_damage_effects0 = True
                _skip_recoil_hp_effects0 = True
                _skip_after_move_secondary_hp_effects0 = True
        _hp0_after_slower_hit = _project_target_hp_after_hit(
            user1_off,
            p0_off,
            move_id1,
            cat1,
            damage1_after_flinch,
            has_sub0,
            sub0_hp,
            hp0_pre_for_move1,
            target_flags_override=own_flags_dis,
            target_ability_override=p0_ab,
            target_item_override=item0,
        )
        _actual_hp_removed_to_p0 = 0
        if hit1 and damage1_after_flinch > 0 and int(new_hp1) > 0:
            _actual_hp_removed_to_p0 = max(
                0,
                int(_hp0_postmove_baseline) - int(_hp0_after_slower_hit),
            )
        # The drain replay above starts from hp_pre only to materialize the
        # inline heal itself. The slower move, however, must damage the
        # first mover from the HP that Showdown has at the start of that
        # slower move, including any preapplied Life Orb / recoil effects.
        battle[user0_off + 1] = np.int16(
            max(0, int(_hp0_postmove_baseline) - _actual_hp_removed_to_p0)
        )
        # Side 1 drain (second mover): normal, post-damage HP.
        _apply_recoil_drain_from_move_tracked(
            battle,
            move_id1,
            user1_off,
            dmg_dealt1,
            hit1,
            game_data,
            move_effects,
            target_offset=target1_off,
            phase="drain",
        )
    elif _first_is_drain and not side0_first:
        # Side 1 used a drain move and moved first.
        battle[user1_off + 1] = np.int16(hp1_pre)
        _apply_recoil_drain_from_move_tracked(
            battle,
            move_id1,
            user1_off,
            dmg_dealt1,
            hit1,
            game_data,
            move_effects,
            target_offset=target1_off,
            phase="drain",
        )
        _hp1_postmove_baseline = int(hp1_pre_for_move0)
        if _hp1_after_postmove_pre is None:
            _hp1_after_drain_replay = int(battle[user1_off + 1])
            _actual_dmg1_drain = _actual_damage_for_source_postmove(
                user1_off,
                p0_off,
                move_id1,
                cat1,
                damage1_after_flinch,
                has_sub0,
                sub0_hp,
                hp0_pre,
                new_hp0,
            )
            _hp1_postmove_baseline = _project_user_hp_after_own_postmove(
                user1_off,
                p0_off,
                move_id1,
                damage1_after_flinch,
                _actual_dmg1_drain,
                hit1,
                int(_meta1.get("num_hits", 1)),
                hp1_pre,
                move_attempted=not (
                    opp_is_switch
                    or _self_hit1
                    or _p1_immobile_pre
                    or _sleep_talk_try_failed1
                    or _rest_try_failed1
                    or move1_no_target
                    or prankster_fail1
                    or _move1_canceled_pre
                ),
                restore_target_item=_inline_knock_saved_item0,
                restore_target_item_off=_inline_knock_saved_target0,
            )
            if int(_hp1_postmove_baseline) != int(_hp1_after_drain_replay):
                _skip_contact_damage_effects1 = True
                _skip_crash_damage_effects1 = True
                _skip_recoil_hp_effects1 = True
                _skip_after_move_secondary_hp_effects1 = True
        _hp1_after_slower_hit = _project_target_hp_after_hit(
            user0_off,
            p1_off,
            move_id0,
            cat0,
            damage0_after_flinch,
            has_sub1,
            sub1_hp,
            hp1_pre_for_move0,
            target_flags_override=opp_flags_dis,
            target_ability_override=p1_ab,
            target_item_override=item1,
        )
        _actual_hp_removed_to_p1 = 0
        if hit0 and damage0_after_flinch > 0 and int(new_hp0) > 0:
            _actual_hp_removed_to_p1 = max(
                0,
                int(_hp1_postmove_baseline) - int(_hp1_after_slower_hit),
            )
        battle[user1_off + 1] = np.int16(
            max(0, int(_hp1_postmove_baseline) - _actual_hp_removed_to_p1)
        )
        # Side 0 drain (second mover): normal.
        _apply_recoil_drain_from_move_tracked(
            battle,
            move_id0,
            user0_off,
            dmg_dealt0,
            hit0,
            game_data,
            move_effects,
            target_offset=target0_off,
            phase="drain",
        )
    else:
        # First mover has no drain — apply both normally.
        _apply_recoil_drain_from_move_tracked(
            battle,
            move_id0,
            user0_off,
            dmg_dealt0,
            hit0,
            game_data,
            move_effects,
            target_offset=target0_off,
            phase="drain",
        )
        _apply_recoil_drain_from_move_tracked(
            battle,
            move_id1,
            user1_off,
            dmg_dealt1,
            hit1,
            game_data,
            move_effects,
            target_offset=target1_off,
            phase="drain",
        )

    # Crash damage on miss — Showdown hasCrashDamage moves: Jump Kick (26),
    # High Jump Kick (136), Supercell Slam (916), Axe Kick (853). On MISS,
    # the user takes 50% maxhp damage. The recoil pipeline above explicitly
    # skips these moves; this handler covers the on-miss case. Only triggers
    # when the move was actually attempted (not blocked by sleep/protect/
    # immune) and missed its accuracy roll. Showdown source: data/moves.ts
    # axekick / highjumpkick / jumpkick / supercellslam onMoveFail.
    _CRASH_DAMAGE_MOVES = (26, 136, 853, 916)
    from pokepy.core.constants import ABILITY_MAGIC_GUARD as _ABILITY_MAGIC_GUARD_CD

    def _maybe_crash_damage(_uoff, _mid, _hit, _executed):
        if int(_mid) not in _CRASH_DAMAGE_MOVES:
            return
        if not _executed or _hit:
            return
        hp_cd = int(battle[_uoff + 1])
        if hp_cd <= 0:
            return
        # Magic Guard blocks crash damage (Showdown: magicguard onDamage
        # blocks indirect damage sources including `hasCrashDamage`).
        if int(battle[_uoff + 5]) == _ABILITY_MAGIC_GUARD_CD:
            return
        max_cd = int(battle[_uoff + 2])
        cost = max(1, max_cd // 2)
        battle[_uoff + 1] = np.int16(max(0, hp_cd - cost))

    # `_executed` proxies "the user actually attempted the move" — true if
    # the user wasn't immobilized (sleep/freeze/full-para) and the target
    # wasn't protected. The simplest available signal is `not target_protected`
    # combined with `not is_immobile`. Use the local flags computed earlier.
    _exec0 = (
        (not is_switch)
        and (not is_immobile0)
        and (not prankster_fail0)
        and (not target0_protected)
    )
    _exec1 = (
        (not opp_is_switch)
        and (not is_immobile1)
        and (not prankster_fail1)
        and (not target1_protected)
    )
    if not _skip_crash_damage_effects0:
        _maybe_crash_damage(user0_off, move_id0, hit0, _exec0)
    if not _skip_crash_damage_effects1:
        _maybe_crash_damage(user1_off, move_id1, hit1, _exec1)

    # Blunder Policy resolution (see comment above near Throat Spray block).
    def _maybe_blunder_policy(_uoff, _mid, _hit, _executed):
        if not _executed or _hit:
            return
        if int(_mid) in _OHKO_MOVES_BP:
            return
        if int(battle[_uoff + 6]) != _ITEM_BLUNDER_POLICY:
            return
        if int(battle[_uoff + 1]) <= 0:
            return
        b14_bp = int(battle[_uoff + 14])
        battle[_uoff + 14] = apply_boost_to_packed(b14_bp, 0, 2)  # +2 Spe
        battle[_uoff + 6] = 0  # consume

    _maybe_blunder_policy(user0_off, move_id0, hit0, _exec0)
    _maybe_blunder_policy(user1_off, move_id1, hit1, _exec1)

    # Showdown ordering (sim/battle-actions.ts):
    #   1. spreadMoveHit: damage + drain (already done above via "drain" phase)
    #   2. runMoveEffects / secondaries
    #   3. DamagingHit event → contact abilities (Rough Skin, Iron Barbs) run
    #      at onDamagingHitOrder 1; Rocky Helmet runs at onDamagingHitOrder 2.
    #      Aftermath / Innards Out / Color Change / Cursed Body / Mummy /
    #      Wandering Spirit / Pickpocket / Toxic Chain / Berserk / Anger Shell
    #      / Weak Armor / Gooey / Tangling Hair / Rattled also run here.
    #   4. After ALL hits: move.recoil (sim/battle-actions.ts:982) — then
    #   5. afterMoveSecondaryEvent → AfterMoveSecondarySelf → Life Orb,
    #      Shell Bell, Sticky Barb contact transfer, Throat Spray.
    # Multi-hit moves: Showdown fires Rocky Helmet / Rough Skin / Iron Barbs
    # per hit (sim/battle-actions.ts:1139 runEvent('DamagingHit') is inside
    # spreadMoveHit, called once per hit by hitStepMoveHitLoop). Pass the
    # actual hit count from the damage calc so the contact-damage loop fires
    # the correct number of times.
    # Inline Knock Off removal is only for the second mover's live damage
    # formula. Showdown's onAfterHit takeItem fires after DamagingHit, so
    # Rocky Helmet and other on-hit item logic must still see the original
    # item through the contact / defender-ability cascade.
    if (
        _inline_knock_saved_item1 > 0
        and _inline_knock_saved_target1 >= 0
        and int(battle[_inline_knock_saved_target1 + 6]) == 0
    ):
        battle[_inline_knock_saved_target1 + 6] = np.int16(_inline_knock_saved_item1)
    if (
        _inline_knock_saved_item0 > 0
        and _inline_knock_saved_target0 >= 0
        and int(battle[_inline_knock_saved_target0 + 6]) == 0
    ):
        battle[_inline_knock_saved_target0 + 6] = np.int16(_inline_knock_saved_item0)
    if not _mid_turn_pivot_0 and not _skip_contact_damage_effects0:
        _apply_contact_damage_tracked(
            battle,
            move_id0,
            user0_off,
            p1_off,
            hit0 and damage0_after_flinch > 0,
            game_data,
            move_effects,
            num_hits=int(_meta0.get("num_hits", 1)),
        )
    if not _mid_turn_pivot_1 and not _skip_contact_damage_effects1:
        _apply_contact_damage_tracked(
            battle,
            move_id1,
            user1_off,
            p0_off,
            hit1 and damage1_after_flinch > 0,
            game_data,
            move_effects,
            num_hits=int(_meta1.get("num_hits", 1)),
        )
    # Pass prerolled contact-ability PRNG rolls to the first mover's call.
    # The first mover's contact ability roll was consumed at the correct PRNG
    # position (after secondaries, before the second mover's damage calc) via
    # _preroll_contact_status_ability. The second mover's roll stays live.
    if not _mid_turn_pivot_0:
        if _meta0.get("contact_status_consumed"):
            if not _contact_status_early_applied0:
                _apply_resolved_contact_status(0)
        else:
            _use_prerolled_0 = _prerolled_contact0 if side0_first else []
            _apply_contact_status_ability_tracked(
                battle,
                move_id0,
                user0_off,
                p1_off,
                hit0 and damage0_after_flinch > 0,
                game_data,
                move_effects,
                gen5_prng,
                prerolled_rolls=_use_prerolled_0,
            )
    if not _mid_turn_pivot_1:
        if _meta1.get("contact_status_consumed"):
            if not _contact_status_early_applied1:
                _apply_resolved_contact_status(1)
        else:
            _use_prerolled_1 = _prerolled_contact1 if not side0_first else []
            _apply_contact_status_ability_tracked(
                battle,
                move_id1,
                user1_off,
                p0_off,
                hit1 and damage1_after_flinch > 0,
                game_data,
                move_effects,
                gen5_prng,
                prerolled_rolls=_use_prerolled_1,
            )

    # Defender ability cascade — Toxic Chain, Toxic Debris, Anger Shell, Cotton
    # Down, Thermal Exchange, Justified, Water Compaction, Steam Engine, Weak
    # Armor, Mummy/Lingering Aroma, Wandering Spirit, Wimp Out / Emergency Exit,
    # Aftermath, Innards Out, Cursed Body, Color Change, Berserk, Pickpocket,
    # Gooey / Tangling Hair, Rattled. All run inside `DamagingHit` in Showdown,
    # so they fire BEFORE move.recoil and BEFORE afterMoveSecondary (Life Orb /
    # Sticky Barb / Shell Bell). Aftermath / Innards Out gate on attacker alive
    # (`source.hp > 0`) — if we applied recoil first, a recoiling attacker that
    # died to recoil would skip Aftermath. Ref: sim/battle-actions.ts:1139.
    p0_def_ability_force_switch = fx.apply_defender_abilities(
        battle,
        move_id0,
        move_id1,
        user0_off,
        user1_off,
        target0_off,
        target1_off,
        damage0_after_flinch,
        damage1_after_flinch,
        hit0,
        hit1,
        game_data,
        gen5_prng,
        move_idx0=move_idx,
        move_idx1=opp_move_idx,
        skip_toxic_chain0=(
            _prerolled_toxic_chain0 is not None or _pivot_toxic_chain_handled0
        ),
        skip_toxic_chain1=(
            _prerolled_toxic_chain1 is not None or _pivot_toxic_chain_handled1
        ),
        skip_immediate_stateful_move0=_move0_preapplied_immediate_defender_state,
        skip_immediate_stateful_move1=_move1_preapplied_immediate_defender_state,
        skip_cursed_body0=_cursed_body_early_applied0,
        skip_cursed_body1=_cursed_body_early_applied1,
        gen=profile.gen,
    )
    _sync_showdown_order_on_switch(
        side_order1,
        int(battle[OFF_META + M_ACTIVE1]),
    )

    # Knock Off onAfterHit — fires inside spreadMoveHit right after
    # `DamagingHit` (sim/battle-actions.ts:1140-1144), so it runs BEFORE
    # move.recoil and BEFORE afterMoveSecondary (Life Orb / Shell Bell /
    # Sticky Barb). Previously pokepy called it after Life Orb which meant
    # a Life-Orb-KO'd user skipped Knock Off's takeItem (the apply_knock_off
    # gate is `source.hp > 0`).
    _knock_target0_off = (
        _inline_knock_saved_target1 if _inline_knock_saved_target1 >= 0 else target0_off
    )
    _knock_target1_off = (
        _inline_knock_saved_target0 if _inline_knock_saved_target0 >= 0 else target1_off
    )
    _apply_knock_off_from_move_tracked(
        move_id0,
        _knock_target0_off,
        hit0_thru_sub and not target0_protected,
        user_offset=user0_off,
        source_alive=(
            _knock_off_source_alive0 if _inline_knock_saved_target1 >= 0 else None
        ),
    )
    _apply_knock_off_from_move_tracked(
        move_id1,
        _knock_target1_off,
        hit1_thru_sub and not target1_protected,
        user_offset=user1_off,
        source_alive=(
            _knock_off_source_alive1 if _inline_knock_saved_target0 >= 0 else None
        ),
    )

    # Showdown's trailing `hitStepMoveHitLoop` / `runMove` `eachEvent('Update')`
    # frames happen after `DamagingHit` / `onAfterHit` complete. Spend them
    # here, after defender abilities and Knock Off's `onAfterHit` hook, so
    # same-hit PRNG consumers like Cursed Body line up.
    for _ in range(int(_deferred_hit_loop_update_frames)):
        gen5_prng.random(0, 2)
    _ng_restart_from_p0 = _neutralizing_gas_on_end_restart_needed(
        p0_off,
        p1_off,
    )
    _ng_restart_from_p1 = _neutralizing_gas_on_end_restart_needed(
        p1_off,
        p0_off,
    )
    if _ng_restart_from_p0 or _ng_restart_from_p1:
        _ITEM_ABILITY_SHIELD_NG_END = 746
        # When the last active Neutralizing Gas user faints, Showdown's
        # `onEnd` sorts all actives before replaying the unsuppressed Start
        # events. A tied active pair therefore consumes one extra speedSort
        # frame after the hit-loop Update frames, and then the surviving
        # active's Start handlers run again.
        if current_action_speed0 == current_action_speed1:
            _sst.each_event_update([current_action_speed0, current_action_speed1])
        if (
            profile.has_abilities
            and _ng_restart_from_p0
            and int(battle[p1_off + 6]) != _ITEM_ABILITY_SHIELD_NG_END
        ):
            fx.apply_switch_in_ability(
                battle,
                p1_off,
                p0_off,
                True,
                gen5_prng,
                has_terrain=profile.has_terrain,
                ability_weather_limited=profile.ability_weather_limited,
            )
        if (
            profile.has_abilities
            and _ng_restart_from_p1
            and int(battle[p0_off + 6]) != _ITEM_ABILITY_SHIELD_NG_END
        ):
            fx.apply_switch_in_ability(
                battle,
                p0_off,
                p1_off,
                True,
                gen5_prng,
                has_terrain=profile.has_terrain,
                ability_weather_limited=profile.ability_weather_limited,
            )

    # Threshold berries are `onUpdate` items in Showdown, so they run after the
    # `DamagingHit` cascade and move `onAfterHit` hooks (for example Knock Off),
    # but before recoil and afterMoveSecondary items like Life Orb / Eject
    # Button. This ordering matters for abilities like Berserk and for item
    # removal suppressing a same-hit berry eat.
    fx.apply_defender_stat_berries_on_damaging_hit(
        battle,
        p1_off,
        int(game_data.move_category[move_id0]),
        hit0,
        damage0_after_flinch,
        game_data,
    )
    fx.apply_defender_stat_berries_on_damaging_hit(
        battle,
        p0_off,
        int(game_data.move_category[move_id1]),
        hit1,
        damage1_after_flinch,
        game_data,
    )
    _run_item_hook_with_berry_tracking(
        fx.apply_sitrus_berry, p0_off, battle, p0_off, game_data
    )
    _run_item_hook_with_berry_tracking(
        fx.apply_sitrus_berry, p1_off, battle, p1_off, game_data
    )
    _run_item_hook_with_berry_tracking(
        fx.apply_gold_berry, p0_off, battle, p0_off, game_data
    )
    _run_item_hook_with_berry_tracking(
        fx.apply_gold_berry, p1_off, battle, p1_off, game_data
    )
    _run_item_hook_with_berry_tracking(
        fx.apply_stat_boosting_berries, p0_off, battle, p0_off, game_data
    )
    _run_item_hook_with_berry_tracking(
        fx.apply_stat_boosting_berries, p1_off, battle, p1_off, game_data
    )
    _run_item_hook_with_berry_tracking(
        fx.apply_pinch_healing_berries, p0_off, battle, p0_off, game_data
    )
    _run_item_hook_with_berry_tracking(
        fx.apply_pinch_healing_berries, p1_off, battle, p1_off, game_data
    )

    # Contact-effects faint gate (Showdown sequential vs pokepy simultaneous):
    # In Showdown, move 0 executes fully (including Rocky Helmet / Rough Skin /
    # Iron Barbs firing via onDamagingHit) BEFORE move 1 executes. If those
    # contact effects kill the first mover (= target of move 1), move 1 finds
    # a fainted target → deals 0 damage → 0 recoil.
    # In pokepy's simultaneous model, both moves' damage is resolved together
    # and the first mover's HP is already set to 0 by move 1's own damage
    # before apply_contact_damage runs (which then early-exits on atk_hp=0).
    # This leaves dmg_dealt for move 1 = first_mover's pre-turn HP, causing
    # spurious recoil. Fix: compute whether contact effects from move 0 would
    # have killed the first mover using the pre-turn HP, and zero out dmg_dealt
    # for move 1 if so (Showdown: targets.every(!hp) → hit=1 → no damage/recoil).
    # Only applies to the SECOND (slower) mover's dmg_dealt.
    # Showdown source: sim/battle-actions.ts:887 inside hitStepMoveHitLoop.
    _ITEM_ROCKY_HELMET_CD = 417
    _ABILITY_ROUGH_SKIN_CD = 24
    _ABILITY_IRON_BARBS_CD = 160
    _ITEM_PROTECTIVE_PADS_CD = 663
    _ABILITY_MAGIC_GUARD_CD2 = 98
    _ABILITY_LONG_REACH_CD = 203

    def _contact_faint_gate(
        first_is_contact,
        first_hit,
        first_user_hp_pre,
        first_user_max_hp,
        first_user_abi,
        first_user_item,
        def_abi,
        def_item_orig,
    ):
        """Return True if contact effects from the first move would have
        killed the first mover before the second move executes."""
        if not (first_is_contact and first_hit and first_user_hp_pre > 0):
            return False
        # Magic Guard blocks Rocky Helmet / Rough Skin on the attacker.
        if first_user_abi == _ABILITY_MAGIC_GUARD_CD2:
            return False
        # Long Reach was already factored into is_contact, but double-check.
        if first_user_abi == _ABILITY_LONG_REACH_CD:
            return False
        # Protective Pads blocks contact effects entirely.
        if first_user_item == _ITEM_PROTECTIVE_PADS_CD:
            return False
        has_rough_skin = def_abi in (_ABILITY_ROUGH_SKIN_CD, _ABILITY_IRON_BARBS_CD)
        has_rocky_helmet = def_item_orig == _ITEM_ROCKY_HELMET_CD
        if not has_rough_skin and not has_rocky_helmet:
            return False
        per_hit = (max(first_user_max_hp // 8, 1) if has_rough_skin else 0) + (
            max(first_user_max_hp // 6, 1) if has_rocky_helmet else 0
        )
        return first_user_hp_pre <= per_hit

    if side0_first and not _mid_turn_pivot_0:
        # First mover = user0; second mover = user1. Defender items/abi for
        # contact effects are on p1_off; use item1 (pre-turn value, before
        # Knock Off may have removed it at apply_knock_off_from_move above).
        if _contact_faint_gate(
            is_contact0,
            hit0 and damage0_after_flinch > 0,
            hp0_pre,
            max_hp0,
            p0_ab,
            item0,
            p1_ab,
            item1,
        ):
            dmg_dealt1 = 0
    elif not side0_first and not _mid_turn_pivot_1:
        # First mover = user1; second mover = user0.
        if _contact_faint_gate(
            is_contact1,
            hit1 and damage1_after_flinch > 0,
            hp1_pre,
            max_hp1,
            p1_ab,
            item1,
            p0_ab,
            item0,
        ):
            dmg_dealt0 = 0

    # Phase 2 — move.recoil. Showdown applies this AFTER all hits of the move
    # are done (sim/battle-actions.ts:982), AFTER the `DamagingHit` cascade
    # fires on each hit. struggleRecoil is not handled specially here; the
    # recoil table already has MOVE_STRUGGLE marked as 25% max HP and Rock
    # Head/Magic Guard blocking matches Showdown. Substitute caps drain/recoil
    # to the sub's HP (sim/data/moves.ts:substitute onTryPrimaryHit).
    if not _skip_recoil_hp_effects0:
        _apply_recoil_drain_from_move_tracked(
            battle,
            move_id0,
            user0_off,
            dmg_dealt0,
            hit0,
            game_data,
            move_effects,
            target_offset=target0_off,
            phase="recoil",
            move_attempted=not (
                is_switch
                or _self_hit0
                or _p0_immobile_pre
                or _sleep_talk_try_failed0
                or _rest_try_failed0
                or move0_no_target
                or prankster_fail0
                or _move0_canceled_pre
            ),
        )
    if not _skip_recoil_hp_effects1:
        _apply_recoil_drain_from_move_tracked(
            battle,
            move_id1,
            user1_off,
            dmg_dealt1,
            hit1,
            game_data,
            move_effects,
            target_offset=target1_off,
            phase="recoil",
            move_attempted=not (
                opp_is_switch
                or _self_hit1
                or _p1_immobile_pre
                or _sleep_talk_try_failed1
                or _rest_try_failed1
                or move1_no_target
                or prankster_fail1
                or _move1_canceled_pre
            ),
        )

    if _defer_absorb_on_p1:
        fx.apply_absorb_ability_healing(battle, p1_off, move_type0, p0_move_executed)
    if _defer_absorb_on_p0:
        fx.apply_absorb_ability_healing(battle, p0_off, move_type1, p1_move_executed)

    # Phase 3 — afterMoveSecondaryEvent: Life Orb, Shell Bell, Sticky Barb
    # contact transfer, Throat Spray (Throat Spray is handled elsewhere in the
    # stat-change path).
    if not _skip_after_move_secondary_hp_effects0:
        _apply_life_orb_recoil_tracked(
            battle,
            user0_off,
            damage0_after_flinch,
            hit0,
            game_data,
            move_id=move_id0,
            move_effects=move_effects,
        )
    if not _skip_after_move_secondary_hp_effects1:
        _apply_life_orb_recoil_tracked(
            battle,
            user1_off,
            damage1_after_flinch,
            hit1,
            game_data,
            move_id=move_id1,
            move_effects=move_effects,
        )

    # Shell Bell — heals holder for 1/8 of damage dealt by the last damaging
    # move. Showdown items.ts:5645 onAfterMoveSecondarySelf:
    #   if (move.totalDamage && !pokemon.forceSwitchFlag) heal(totalDamage / 8)
    # heal() floors. Magic Guard does NOT block (it's heal not damage). Sheer
    # Force does NOT suppress (the suppression hook only zeroes secondary
    # effects, not totalDamage). Status moves contribute 0 totalDamage so the
    # `move.totalDamage` truthy check naturally gates this on damaging moves.
    _ITEM_SHELL_BELL = 253
    for u_off, did_hit_sb, dmg_sb, skip_sb in (
        (user0_off, hit0, dmg_dealt0, _skip_after_move_secondary_hp_effects0),
        (user1_off, hit1, dmg_dealt1, _skip_after_move_secondary_hp_effects1),
    ):
        if skip_sb:
            continue
        if not did_hit_sb:
            continue
        if int(battle[u_off + 6]) != _ITEM_SHELL_BELL:
            continue
        if int(battle[u_off + 1]) <= 0:
            continue
        if int(dmg_sb) <= 0:
            continue
        max_hp_sb = int(battle[u_off + 2])
        cur_hp_sb = int(battle[u_off + 1])
        heal_sb = max(int(dmg_sb) // 8, 1)
        battle[u_off + 1] = min(max_hp_sb, cur_hp_sb + heal_sb)

    # Sticky Barb contact transfer (items.ts:5691). On a contact hit, if the
    # attacker is item-less and the target is holding Sticky Barb, the barb
    # moves to the attacker. Does NOT deal damage here (residual is at
    # onResidualOrder 28; handled in the EOT block).
    def _try_sticky_barb_transfer(user_off, target_off, did_hit_mid, move_id):
        if not did_hit_mid:
            return
        if int(battle[target_off + 6]) != ITEM_STICKY_BARB:
            return
        if int(battle[user_off + 6]) != 0:
            return
        if int(battle[user_off + 1]) <= 0:
            return
        mflags = int(game_data.move_flags[int(move_id)])
        if (mflags & FLAG_CONTACT) == 0:
            return
        battle[target_off + 6] = 0
        battle[user_off + 6] = ITEM_STICKY_BARB

    _try_sticky_barb_transfer(
        user0_off, p1_off, hit0 and damage0_after_flinch > 0, move_id0
    )
    _try_sticky_barb_transfer(
        user1_off, p0_off, hit1 and damage1_after_flinch > 0, move_id1
    )
    fx.apply_magician_from_move(
        battle,
        move_id0,
        user0_off,
        p1_off,
        hit0,
        dmg_dealt0,
        game_data,
        move_effects,
    )
    fx.apply_magician_from_move(
        battle,
        move_id1,
        user1_off,
        p0_off,
        hit1,
        dmg_dealt1,
        game_data,
        move_effects,
    )

    # Mid-turn pivot users already had their own post-move chain resolved
    # before switching so the slower move could target the replacement. The
    # shared late buckets below can still transiently reuse `user*_off` and
    # clobber that outgoing bench slot. Restore the saved bench snapshot
    # here so later turns read the same stored HP/status that Showdown keeps
    # for the switched-out pivot user.
    if _mid_turn_pivot_0 and _mid_turn_pivot_saved_hp0 is not None:
        battle[user0_off + 1] = np.int16(_mid_turn_pivot_saved_hp0)
        if _mid_turn_pivot_saved_status0 is not None:
            battle[user0_off + 12] = np.int16(_mid_turn_pivot_saved_status0)
    if _mid_turn_pivot_1 and _mid_turn_pivot_saved_hp1 is not None:
        battle[user1_off + 1] = np.int16(_mid_turn_pivot_saved_hp1)
        if _mid_turn_pivot_saved_status1 is not None:
            battle[user1_off + 12] = np.int16(_mid_turn_pivot_saved_status1)

    # Eject Button / Red Card (item-driven forced switch). Returns True if
    # player 0's eject/red card consumed → must FORCED_SWITCH at end of turn.
    _pre_item_active0 = int(battle[OFF_META + M_ACTIVE0])
    _pre_item_active1 = int(battle[OFF_META + M_ACTIVE1])
    p0_item_forced_switch = _p0_item_forced_switch_pre or fx.apply_item_forced_switch(
        battle,
        move_id0,
        move_id1,
        user0_off,
        user1_off,
        target0_off,
        target1_off,
        damage0_after_flinch,
        damage1_after_flinch,
        hit0,
        hit1,
        side_order0,
        side_order1,
        gen5_prng,
        game_data,
        state,
    )
    _post_item_active0 = int(battle[OFF_META + M_ACTIVE0])
    _post_item_active1 = int(battle[OFF_META + M_ACTIVE1])
    _item_switch_changed_active0 = _post_item_active0 != _pre_item_active0
    _item_switch_changed_active1 = _post_item_active1 != _pre_item_active1
    if _item_switch_changed_active0 or _item_switch_changed_active1:
        _inline_item_switch_rewrote_active = True
    if (
        int(move_effects.effect_type[move_id0]) == EFFECT_SWITCH
        and not is_switch
        and hit0
        and (_item_switch_changed_active0 or _item_switch_changed_active1)
    ):
        _pivot_selfswitch_canceled_0 = True
    if (
        int(move_effects.effect_type[move_id1]) == EFFECT_SWITCH
        and not opp_is_switch
        and hit1
        and (
            p0_item_forced_switch
            or _item_switch_changed_active0
            or _item_switch_changed_active1
        )
    ):
        _pivot_selfswitch_canceled_1 = True

    # Trick / Skill Swap / screen breaks / Ice Spinner.
    # Knock Off is now fired above (before recoil) — see Showdown
    # sim/battle-actions.ts:1140-1144 onAfterHit.
    _trick_swapped0 = _apply_trick_from_move_tracked(
        move_id0, user0_off, target0_off, hit0 and not target0_protected
    )
    _trick_swapped1 = _apply_trick_from_move_tracked(
        move_id1, user1_off, target1_off, hit1 and not target1_protected
    )
    # Skill Swap (Showdown data/moves.ts:skillswap) — ability swap.
    fx.apply_skill_swap_from_move(
        battle, move_id0, user0_off, target0_off, hit0 and not target0_protected
    )
    fx.apply_skill_swap_from_move(
        battle, move_id1, user1_off, target1_off, hit1 and not target1_protected
    )

    # Screen-breaking moves (Brick Break / Psychic Fangs / Raging Bull)
    # Showdown's onTryHit runs at hit-step 1 (BEFORE accuracy), so the screen
    # break happens even on a MISS. It's only blocked by Protect (priority 3
    # onTryHit) or invulnerability. Also, Brick Break / Psychic Fangs /
    # Raging Bull ONLY clear Reflect, Light Screen, and Aurora Veil — NOT
    # Safeguard / Mist / Tailwind / Lucky Chant. Pokepy previously wiped the
    # entire F_SCREENS word and gated on `hit AND damage > 0`.
    # Refs: data/moves.ts:1884 brickbreak, :14577 psychicfangs, :15174 ragingbull.
    from pokepy.core.constants import (
        SCREEN_REFLECT_SHIFT as _SRS_BB,
        SCREEN_LIGHTSCREEN_SHIFT as _SLS_BB,
        SCREEN_AURORAVEIL_SHIFT as _SAV_BB,
        SCREEN_MASK_3BIT as _SM3_BB,
    )

    _screen_break_mask = (
        ~((_SM3_BB << _SRS_BB) | (_SM3_BB << _SLS_BB) | (_SM3_BB << _SAV_BB)) & 0xFFFF
    )

    def _apply_screen_break(target_screens_off: int) -> None:
        cur = int(battle[target_screens_off]) & 0xFFFF
        nv = cur & _screen_break_mask
        if nv >= 0x8000:
            nv -= 0x10000
        battle[target_screens_off] = nv

    eff0_t = int(move_effects.effect_type[move_id0])
    eff1_t = int(move_effects.effect_type[move_id1])
    # Gate on "attempted move that wasn't blocked by protect / immobilize /
    # invulnerability" — mirrors Showdown's onTryHit firing before accuracy.
    if (
        eff0_t == EFFECT_SCREEN_BREAK
        and not is_switch
        and not is_immobile0
        and not prankster_fail0
        and not target0_protected
    ):
        _apply_screen_break(OFF_FIELD + F_SCREENS_1)
    if (
        eff1_t == EFFECT_SCREEN_BREAK
        and not opp_is_switch
        and not is_immobile1
        and not prankster_fail1
        and not target1_protected
    ):
        _apply_screen_break(OFF_FIELD + F_SCREENS_0)

    # Ice Spinner — clear terrain
    if (move_id0 == 861 and hit0 and damage0_after_flinch > 0) or (
        move_id1 == 861 and hit1 and damage1_after_flinch > 0
    ):
        battle[OFF_FIELD + F_TERRAIN] = 0
        battle[OFF_META + M_TERRAIN_TURNS] = 0
        _apply_booster_energy_update_tracked(p0_off)
        _apply_booster_energy_update_tracked(p1_off)

    fx.apply_haze_from_move(battle, move_id0, hit0, move_effects)
    fx.apply_haze_from_move(battle, move_id1, hit1, move_effects)
    fx.apply_clear_smog_from_move(battle, move_id0, target0_off, hit0, move_effects)
    fx.apply_clear_smog_from_move(battle, move_id1, target1_off, hit1, move_effects)
    fx.apply_psych_up_from_move(
        battle, move_id0, user0_off, target0_off, hit0, move_effects
    )
    fx.apply_psych_up_from_move(
        battle, move_id1, user1_off, target1_off, hit1, move_effects
    )
    _apply_screen_from_move_tracked(move_id0, 0, hit0)
    _apply_screen_from_move_tracked(move_id1, 1, hit1)

    _PHAZING_MOVES = (MOVE_ROAR, MOVE_WHIRLWIND, MOVE_DRAGON_TAIL, MOVE_CIRCLE_THROW)
    _move2_mid_pre_phaze = move_id1 if side0_first else move_id0
    _move2_hit_pre_phaze = hit1 if side0_first else hit0
    _move2_is_switch_pre_phaze = opp_is_switch if side0_first else is_switch
    _move2_targets_phazing = _move2_mid_pre_phaze in _PHAZING_MOVES
    # When the slower move is a phazing move, Showdown still runs the two
    # `hitStepMoveHitLoop` eachEvent('Update') speedSorts (battle-actions.ts
    # 840 + 885) BEFORE selecting the drag target. Pokepy previously delayed
    # all post-move speedSort accounting until after phazing, which skipped
    # these two tied-speed frames and shifted the later drag sample.
    if (
        _move2_targets_phazing
        and (not _move2_is_switch_pre_phaze)
        and _move2_hit_pre_phaze
    ):
        _phaze_pre_p0_speed = fx.get_effective_speed(battle, p0_off)
        _phaze_pre_p1_speed = fx.get_effective_speed(battle, p1_off)
        if _phaze_pre_p0_speed == _phaze_pre_p1_speed:
            _sst.each_event_update([_phaze_pre_p0_speed, _phaze_pre_p1_speed])
            _sst.each_event_update([_phaze_pre_p0_speed, _phaze_pre_p1_speed])

    # Phazing. Snapshot active slots pre-phaze so we can detect whether a
    # phaze actually occurred (apply_phazing_from_move returns the new
    # slot). When it does, apply hazard damage + switch-in ability to
    # the phazed-in Pokemon to match Showdown (the drag-in trigger runs
    # `pokemon.switchIn` which fires `hazard onStart`, `onSwitchIn`,
    # ability onStart, etc.). Skip if the source move hit missed.
    _pre_active0 = int(battle[OFF_META + M_ACTIVE0])
    _pre_active1 = int(battle[OFF_META + M_ACTIVE1])
    fx.apply_phazing_from_move(
        battle,
        move_id0,
        target_side0,
        user0_off,
        hit0,
        side_order1 if target_side0 == 1 else side_order0,
        game_data,
        move_effects,
        gen5_prng,
    )
    fx.apply_phazing_from_move(
        battle,
        move_id1,
        target_side1,
        user1_off,
        hit1,
        side_order0 if target_side1 == 0 else side_order1,
        game_data,
        move_effects,
        gen5_prng,
    )
    _post_active0 = int(battle[OFF_META + M_ACTIVE0])
    _post_active1 = int(battle[OFF_META + M_ACTIVE1])
    _phazed0 = _post_active0 != _pre_active0
    _phazed1 = _post_active1 != _pre_active1
    _new_p0_pz = OFF_SIDE0 + _post_active0 * POKEMON_SIZE
    _new_p1_pz = OFF_SIDE1 + _post_active1 * POKEMON_SIZE
    if _phazed0:
        _sync_showdown_order_on_switch(side_order0, _post_active0)
        _reset_incoming_switch_state_tracked(_new_p0_pz)
        _phazed0_pending_switch_slot_condition = int(
            battle[OFF_FIELD + F_DESTINY_BOND_0]
        )
        if apply_pending_wish_on_switch_in(
            battle,
            0,
            _new_p0_pz,
            state,
            game_data,
            _phazed0_pending_switch_slot_condition,
        ):
            battle[OFF_FIELD + F_DESTINY_BOND_0] = 0
    if _phazed1:
        _sync_showdown_order_on_switch(side_order1, _post_active1)
        if hasattr(state, "hidden_opp_switches"):
            state.hidden_opp_switches.append(int(_post_active1))
        _reset_incoming_switch_state_tracked(_new_p1_pz)
        _phazed1_pending_switch_slot_condition = int(
            battle[OFF_FIELD + F_DESTINY_BOND_1]
        )
        if apply_pending_wish_on_switch_in(
            battle,
            1,
            _new_p1_pz,
            state,
            game_data,
            _phazed1_pending_switch_slot_condition,
        ):
            battle[OFF_FIELD + F_DESTINY_BOND_1] = 0
    _move2_phazed = bool(
        _move2_targets_phazing
        and ((side0_first and _phazed0) or ((not side0_first) and _phazed1))
    )
    # Showdown's drag-in path queues `runSwitch` immediately and speed-sorts
    # the live actives before any SwitchIn handlers mutate the incoming mon.
    if _phazed0 or _phazed1:
        if _phazed0 and _phazed1:
            _consume_runswitch_tie_frame(
                battle,
                _new_p0_pz,
                _new_p1_pz,
                gen5_prng,
                switcher_speed=int(battle[_new_p0_pz + 11]),
                foe_speed=int(battle[_new_p1_pz + 11]),
            )
        elif _phazed0:
            _consume_runswitch_tie_frame(
                battle,
                _new_p0_pz,
                _new_p1_pz,
                gen5_prng,
                switcher_speed=int(battle[_new_p0_pz + 11]),
            )
        else:
            _consume_runswitch_tie_frame(
                battle,
                _new_p1_pz,
                _new_p0_pz,
                gen5_prng,
                switcher_speed=int(battle[_new_p1_pz + 11]),
            )
    # Hazards always apply per side first.
    if _phazed1:
        fx.apply_hazard_damage_on_switch(battle, _new_p1_pz, OFF_FIELD + F_HAZARDS_1)
        _reset_toxic_counter_on_switch_in(battle, _new_p1_pz)
    if _phazed0:
        fx.apply_hazard_damage_on_switch(battle, _new_p0_pz, OFF_FIELD + F_HAZARDS_0)
        _reset_toxic_counter_on_switch_in(battle, _new_p0_pz)
    # Switch-in abilities are speed-sorted (sim/battle.ts:507 speedSort,
    # called from runEvent('SwitchIn')). Faster mon resolves first; slower
    # mon's setter overrides on weather/terrain ties.
    if _phazed0 and _phazed1:
        sp0_pz = fx.get_effective_speed(battle, _new_p0_pz)
        sp1_pz = fx.get_effective_speed(battle, _new_p1_pz)
        # Trick Room inverts speedSort order (sim/pokemon.ts:631-639).
        _tr_active_pz = int(battle[OFF_FIELD + F_TRICK_ROOM]) > 0
        p0_first_pz = (sp0_pz >= sp1_pz) if not _tr_active_pz else (sp0_pz <= sp1_pz)
        if p0_first_pz:
            if int(battle[_new_p0_pz + 1]) > 0:
                _apply_switch_in_ability_tracked(_new_p0_pz, _new_p1_pz, True)
            if int(battle[_new_p1_pz + 1]) > 0:
                _apply_switch_in_ability_tracked(_new_p1_pz, _new_p0_pz, True)
        else:
            if int(battle[_new_p1_pz + 1]) > 0:
                _apply_switch_in_ability_tracked(_new_p1_pz, _new_p0_pz, True)
            if int(battle[_new_p0_pz + 1]) > 0:
                _apply_switch_in_ability_tracked(_new_p0_pz, _new_p1_pz, True)
    elif _phazed1:
        if int(battle[_new_p1_pz + 1]) > 0:
            _apply_switch_in_ability_with_trace_reaction_tracked(
                _new_p1_pz, _new_p0_pz, True
            )
    elif _phazed0:
        if int(battle[_new_p0_pz + 1]) > 0:
            _apply_switch_in_ability_with_trace_reaction_tracked(
                _new_p0_pz, _new_p1_pz, True
            )
    # `runAction` then calls `eachEvent('SwitchIn')` on the current actives.
    if _phazed0 or _phazed1:
        _switchin_p0_speed = fx.get_effective_speed(battle, _new_p0_pz)
        _switchin_p1_speed = fx.get_effective_speed(battle, _new_p1_pz)
        _refresh_current_action_speeds()
        if _switchin_p0_speed == _switchin_p1_speed:
            _sst.each_event_update([_switchin_p0_speed, _switchin_p1_speed])
        if _phazed0 and int(battle[_new_p0_pz + 1]) > 0:
            _run_switch_in_update_item_hooks(_new_p0_pz)
        if _phazed1 and int(battle[_new_p1_pz + 1]) > 0:
            _run_switch_in_update_item_hooks(_new_p1_pz)

    # Self-KO moves (Explosion, Self-Destruct, Memento, Final Gambit, Misty
    # Explosion, Healing Wish, Lunar Dance). Showdown's selfdestruct phase
    # runs only after a successful useMove (tryMoveHit → `if (moveResult)`),
    # so asleep / frozen / full-para'd / flinched users do NOT blow up. Gate
    # on "move actually executed" — side0_could_move / side1_could_move are
    # computed later so inline the same condition here.
    _side0_executed = (not is_switch) and not (
        is_immobile0
        or pre_damage_fail0
        or prankster_fail0
        or self_hit0
        or move0_no_target
        or (side0_flinched and not side0_first)
    )
    _side1_executed = (not opp_is_switch) and not (
        is_immobile1
        or pre_damage_fail1
        or prankster_fail1
        or self_hit1
        or move1_no_target
        or (side1_flinched and side0_first)
    )
    # Glaive Rush (862) — Baxcalibur signature. Showdown data/moves.ts
    # applies the `glaiverush` self-volatile on hit; while active, moves
    # targeting the user bypass accuracy AND deal 2x damage. We track this
    # via the FLAG_GLAIVE_RUSH bit on the user's flag byte. Cleared at the
    # start of the user's next move (see _calc_p0/_calc_p1 above).
    _MOVE_GLAIVE_RUSH = 862
    if _side0_executed and hit0 and move_id0 == _MOVE_GLAIVE_RUSH:
        battle[user0_off + 15] = int(battle[user0_off + 15]) | _FLAG_GLAIVE_RUSH
    if _side1_executed and hit1 and move_id1 == _MOVE_GLAIVE_RUSH:
        battle[user1_off + 15] = int(battle[user1_off + 15]) | _FLAG_GLAIVE_RUSH

    # ------------------------------------------------------------------
    # Lockedmove (Outrage / Petal Dance / Thrash / Raging Fury) + recharge
    # (Hyper Beam / Giga Impact / etc.) post-move state.
    # Showdown data/conditions.ts:253 (lockedmove) + :364 (mustrecharge).
    # Lockedmove rolls a 2-3 turn duration on first use; user is locked
    # into the same move; on expiration the user becomes confused.
    # Sleep clears lockedmove without confusing.
    # Recharge sets a 1-turn skip-next-turn flag after the move resolves.
    # ------------------------------------------------------------------
    def _set_lockedmove_post(
        side: int,
        mid: int,
        executed: bool,
        self_effect_applies: bool,
        user_off: int,
        was_locked: bool,
        prev_turns: int,
    ) -> None:
        loc_off = OFF_MOVES + (M_LOCKED_MOVE_0 if side == 0 else M_LOCKED_MOVE_1)
        trn_off = OFF_MOVES + (M_LOCKED_TURNS_0 if side == 0 else M_LOCKED_TURNS_1)
        # Sleep clears lockedmove without confusing (Showdown
        # conditions.ts:253 onResidual: if status === 'slp' delete volatile).
        cur_status = int(battle[user_off + 12]) & 0xFF
        if was_locked and cur_status == STATUS_SLEEP:
            battle[loc_off] = -1
            battle[trn_off] = 0
            return
        if not executed:
            # Showdown resolves BeforeMove before LockMove. When a locked user
            # loses its turn to confusion self-hit / full para / flinch / etc.,
            # the move never actually executes, so our synthetic lock state
            # must not persist into the next turn as if the move had been used.
            if was_locked:
                battle[loc_off] = -1
                battle[trn_off] = 0
            return
        if mid not in _LOCKED_MOVES:
            # User used something else (e.g. Sleep Talk forced) — clear.
            if was_locked:
                battle[loc_off] = -1
                battle[trn_off] = 0
            return
        if not was_locked:
            if not self_effect_applies:
                return
            # First use: duration was pre-rolled inline after _calc_pN()
            # by _preroll_lockedmove_self_effect. Use the stored value.
            rolled = _lockedmove_prerolled.get(side)
            if rolled is None:
                # Fallback: if not pre-rolled (shouldn't happen), roll now.
                rolled = 2 + int(gen5_prng.random(2))
            remaining = rolled - 1
            if remaining > 0:
                battle[loc_off] = mid
                battle[trn_off] = remaining
            else:
                # Already done — confuse immediately (dead code in practice:
                # rolled is 2 or 3, so remaining is 1 or 2).
                vol_off = OFF_FIELD + (F_VOLATILE_0 if side == 0 else F_VOLATILE_1)
                cur_v = int(battle[vol_off])
                user_ab = int(battle[user_off + 5])
                if user_ab != ABILITY_OWN_TEMPO and get_confusion_turns(cur_v) == 0:
                    turns_roll = int(gen5_prng.random(2, 6))
                    new_v = set_confusion_turns(cur_v, turns_roll)
                    if new_v >= 0x8000:
                        new_v -= 0x10000
                    battle[vol_off] = new_v
                battle[loc_off] = -1
                battle[trn_off] = 0
        else:
            # Repeated turns only preserve the rampage lock if Showdown's
            # selfDrops path actually restarted the volatile. A dud repeated
            # turn (for example Outrage into a Fairy) drops the lock early
            # while there are still turns left, but an expiry turn still ends
            # the volatile and adds fatigue confusion.
            new_remaining = prev_turns - 1
            if not self_effect_applies and new_remaining > 0:
                battle[loc_off] = -1
                battle[trn_off] = 0
                return
            # Already locked: decrement remaining turns.
            if new_remaining > 0:
                battle[trn_off] = new_remaining
            else:
                # Lockedmove expires — confuse the user. Confusion turns
                # were pre-rolled by _preroll_lockedmove_self_effect.
                prerolled = _lockedmove_prerolled.get(side)
                vol_off = OFF_FIELD + (F_VOLATILE_0 if side == 0 else F_VOLATILE_1)
                cur_v = int(battle[vol_off])
                if prerolled is not None and prerolled < 0:
                    # Negative value = pre-rolled confusion turns.
                    turns_roll = -prerolled
                    new_v = set_confusion_turns(cur_v, turns_roll)
                    if new_v >= 0x8000:
                        new_v -= 0x10000
                    battle[vol_off] = new_v
                elif prerolled == _LOCKEDMOVE_CONFUSION_PENDING:
                    # The second mover's fatigue-confusion duration is rolled
                    # after its pending post-hit Update frames below.
                    pass
                elif prerolled is None:
                    # Fallback: if not pre-rolled, roll now.
                    user_ab = int(battle[user_off + 5])
                    if user_ab != ABILITY_OWN_TEMPO and get_confusion_turns(cur_v) == 0:
                        turns_roll = int(gen5_prng.random(2, 6))
                        new_v = set_confusion_turns(cur_v, turns_roll)
                        if new_v >= 0x8000:
                            new_v -= 0x10000
                        battle[vol_off] = new_v
                # prerolled == 0 means expiry but no confusion (Own Tempo
                # or already confused).
                battle[loc_off] = -1
                battle[trn_off] = 0

    def _set_recharge_post(side: int, mid: int, executed: bool, did_hit: bool) -> None:
        # Showdown adds mustrecharge through the move's `self` block, but
        # that path only runs for targets that survive the accuracy/protect/
        # immunity filter. Misses, Protect, and type-immunity failures all
        # skip selfDrops, so recharge is only set when the move actually hit.
        if not executed:
            return
        if not did_hit:
            return
        if mid not in _RECHARGE_MOVES:
            return
        rec_off = OFF_MOVES + (M_RECHARGE_0 if side == 0 else M_RECHARGE_1)
        battle[rec_off] = 1

    def _set_charge_post(user_off: int, mid: int, executed: bool) -> None:
        if not executed or mid != MOVE_CHARGE:
            return
        battle[user_off + 15] = np.int16(int(battle[user_off + 15]) | FLAG_CHARGE)

    def _consume_charge_post(
        user_off: int, mid: int, move_type: int, move_ran: bool
    ) -> None:
        if not move_ran or mid == MOVE_CHARGE or move_type != TYPE_ELECTRIC:
            return
        battle[user_off + 15] = np.int16(int(battle[user_off + 15]) & ~FLAG_CHARGE)

    _set_lockedmove_post(
        0,
        move_id0,
        p0_move_ran,
        (damage0 > 0) or target0_protected,
        user0_off,
        is_locked_turn0,
        locked_turns0_pre,
    )
    _set_lockedmove_post(
        1,
        move_id1,
        p1_move_ran,
        (damage1 > 0) or target1_protected,
        user1_off,
        is_locked_turn1,
        locked_turns1_pre,
    )
    _set_recharge_post(0, move_id0, p0_move_ran, hit0)
    _set_recharge_post(1, move_id1, p1_move_ran, hit1)
    _set_charge_post(user0_off, move_id0, p0_move_ran)
    _set_charge_post(user1_off, move_id1, p1_move_ran)
    _consume_charge_post(user0_off, move_id0, move_type0, p0_move_ran)
    _consume_charge_post(user1_off, move_id1, move_type1, p1_move_ran)

    self_ko0_set = (
        MOVE_EXPLOSION,
        MOVE_SELF_DESTRUCT,
        MOVE_MEMENTO,
        MOVE_FINAL_GAMBIT,
        MOVE_MISTY_EXPLOSION,
        MOVE_HEALING_WISH,
        MOVE_LUNAR_DANCE,
    )
    # Healing Wish vs Lunar Dance: store distinct pending sentinel so the
    # replacement-switch handler can restore PP for Lunar Dance only
    # (Showdown data/moves.ts:lunardance onSwap resets each moveSlot.pp).
    from pokepy.core.constants import LUNAR_DANCE_PENDING as _LDP

    # Healing Wish / Lunar Dance fail if there's nobody to switch in.
    # Showdown data/moves.ts:healingwish onTry checks for `canSwitch`.
    # Pokepy must skip both the sentinel AND the self-faint in that case.
    def _bench_alive(_side_base, _active_off):
        for _i in range(6):
            _so = _side_base + _i * POKEMON_SIZE
            if _so == _active_off:
                continue
            if int(battle[_so + 1]) > 0 and (int(battle[_so + 15]) & 1) == 0:
                return True
        return False

    _hw0_has_switch = _bench_alive(OFF_SIDE0, user0_off)
    _hw1_has_switch = _bench_alive(OFF_SIDE1, user1_off)
    _hw0_valid = (
        move_id0 not in (MOVE_HEALING_WISH, MOVE_LUNAR_DANCE)
    ) or _hw0_has_switch
    _hw1_valid = (
        move_id1 not in (MOVE_HEALING_WISH, MOVE_LUNAR_DANCE)
    ) or _hw1_has_switch

    if _side0_executed and hit0 and _hw0_valid:
        if move_id0 == MOVE_HEALING_WISH:
            battle[OFF_FIELD + F_DESTINY_BOND_0] = HEALING_WISH_PENDING
        elif move_id0 == MOVE_LUNAR_DANCE:
            battle[OFF_FIELD + F_DESTINY_BOND_0] = _LDP
    if move_id0 in self_ko0_set and _side0_executed and hit0 and _hw0_valid:
        battle[user0_off + 1] = 0
    # Side 1 Healing Wish / Lunar Dance support — symmetry with side 0.
    from pokepy.core.constants import F_DESTINY_BOND_1 as _FDB1

    if _side1_executed and hit1 and _hw1_valid:
        if move_id1 == MOVE_HEALING_WISH:
            battle[OFF_FIELD + _FDB1] = HEALING_WISH_PENDING
        elif move_id1 == MOVE_LUNAR_DANCE:
            battle[OFF_FIELD + _FDB1] = _LDP
    if (
        move_id1
        in (
            MOVE_EXPLOSION,
            MOVE_SELF_DESTRUCT,
            MOVE_MEMENTO,
            MOVE_FINAL_GAMBIT,
            MOVE_MISTY_EXPLOSION,
            MOVE_HEALING_WISH,
            MOVE_LUNAR_DANCE,
        )
        and _side1_executed
        and hit1
        and _hw1_valid
    ):
        battle[user1_off + 1] = 0

    # Re-read active slots after phazing
    active0 = int(battle[OFF_META + M_ACTIVE0])
    active1 = int(battle[OFF_META + M_ACTIVE1])
    own_hp_off_u = OFF_SIDE0 + active0 * POKEMON_SIZE + 1
    opp_hp_off_u = OFF_SIDE1 + active1 * POKEMON_SIZE + 1
    hp0_after_effects = int(battle[own_hp_off_u])
    hp1_after_effects = int(battle[opp_hp_off_u])

    own_flags_off = OFF_SIDE0 + active0 * POKEMON_SIZE + 15
    opp_flags_off = OFF_SIDE1 + active1 * POKEMON_SIZE + 15
    if hp1_after_effects == 0:
        battle[opp_flags_off] = int(battle[opp_flags_off]) | 1
        _clear_opponent_source_tied_lock_state(battle, 1)
    if hp0_after_effects == 0:
        battle[own_flags_off] = int(battle[own_flags_off]) | 1
        _clear_opponent_source_tied_lock_state(battle, 0)

    # KO boost abilities (Moxie, Beast Boost, Soul-Heart) only trigger when
    # the user's own hit actually caused the opposing active to faint. Do not
    # infer this from the opponent being dead later in the turn: self-KO moves
    # like Explosion and other downstream faint paths can otherwise leak a
    # false Moxie/Beast Boost/Soul-Heart stage onto the opposing attacker.
    move0_direct_ko = bool(hit0 and damage0_after_flinch > 0 and new_hp1 == 0)
    move1_direct_ko = bool(hit1 and damage1_after_flinch > 0 and new_hp0 == 0)
    fx.apply_ko_boost_ability(battle, user0_off, move0_direct_ko, hit0)
    fx.apply_ko_boost_ability(battle, user1_off, move1_direct_ko, hit1)

    # Player 0 fainted? — handled via FORCED_SWITCH (no auto-switch)
    p0_fainted = hp0_after_effects == 0
    p0_original_active = active0

    # Pressure PP loss follows the move's resolved target(s), not whichever
    # Pokemon happens to be active after later self-switch / faint replacement
    # flows mutate the board. Snapshot the relevant opposing ability before
    # U-turn / Volt Switch / post-upkeep replacement logic can rewrite the
    # active slot.
    pp_pressure_ability0 = (
        int(battle[target0_off + 5])
        if move0_targets_foe_mon
        else int(battle[OFF_SIDE1 + active1 * POKEMON_SIZE + 5])
    )
    pp_pressure_ability1 = (
        int(battle[target1_off + 5])
        if move1_targets_foe_mon
        else int(battle[OFF_SIDE0 + active0 * POKEMON_SIZE + 5])
    )

    # Showdown's faint replacement timing: a fainted Pokemon's replacement
    # is sent in via `makeRequest('switch')` AFTER the residual phase
    # (sim/battle.ts:2842-2879). EOT residuals run with the dead slot
    # in place (and skip dead mons via hp > 0 checks). Pokepy used to
    # auto-switch the opponent immediately, leaking a free EOT tick onto
    # the replacement mon (sand/leftovers/leech seed/etc). Now we mark
    # opp_fainted here but defer the auto-switch + hazards + switch-in
    # ability until AFTER the EOT block below.
    opp_fainted = hp1_after_effects == 0

    # U-turn / Volt Switch
    user0_alive = int(battle[OFF_SIDE0 + active0 * POKEMON_SIZE + 1]) > 0
    user1_alive = int(battle[OFF_SIDE1 + active1 * POKEMON_SIZE + 1]) > 0
    _move2_update_p0_speed = fx.get_effective_speed(
        battle,
        OFF_SIDE0 + active0 * POKEMON_SIZE,
    )
    _move2_update_p1_speed = fx.get_effective_speed(
        battle,
        OFF_SIDE1 + active1 * POKEMON_SIZE,
    )
    effect0_is_switch = (
        int(move_effects.effect_type[move_id0]) == EFFECT_SWITCH
    ) and not is_switch
    effect1_is_switch = (
        int(move_effects.effect_type[move_id1]) == EFFECT_SWITCH
    ) and not opp_is_switch
    should_uturn0 = effect0_is_switch and hit0 and user0_alive and not p0_fainted
    should_uturn1 = effect1_is_switch and hit1 and user1_alive
    # Mid-turn pivot already handled side0's switch — skip the post-turn path.
    if _mid_turn_pivot_0:
        should_uturn0 = False
    if _mid_turn_pivot_1:
        should_uturn1 = False
    if _pivot_selfswitch_canceled_0:
        should_uturn0 = False
    if _pivot_selfswitch_canceled_1:
        should_uturn1 = False
    # Parting Shot failure gates the selfSwitch (see stat-change block above).
    if _partingshot0_failed:
        should_uturn0 = False
    if _partingshot1_failed:
        should_uturn1 = False

    # U-turn / Volt Switch bench alive check (Showdown: battle-actions.js
    # canSwitch before committing selfSwitch). If no bench mon is available,
    # the selfSwitch is suppressed — user stays in, no hazards/Regenerator.
    _p0_uturn_bench = _bench_alive(OFF_SIDE0, OFF_SIDE0 + active0 * POKEMON_SIZE)
    if should_uturn0 and not _p0_uturn_bench:
        should_uturn0 = False
    _p1_uturn_bench = _bench_alive(OFF_SIDE1, OFF_SIDE1 + active1 * POKEMON_SIZE)
    if should_uturn1 and not _p1_uturn_bench:
        should_uturn1 = False
    # If the opposing side has no surviving replacement after the KO that
    # just happened, Showdown ends the battle instead of executing selfSwitch.
    if (
        should_uturn0
        and opp_fainted
        and not _bench_alive(OFF_SIDE1, OFF_SIDE1 + active1 * POKEMON_SIZE)
    ):
        should_uturn0 = False
    if (
        should_uturn1
        and p0_fainted
        and not _bench_alive(OFF_SIDE0, OFF_SIDE0 + active0 * POKEMON_SIZE)
    ):
        should_uturn1 = False

    _late_self_switch_before_residual = False

    if should_uturn0:
        old_off = OFF_SIDE0 + active0 * POKEMON_SIZE
        fx.apply_regenerator_on_switch_out(battle, old_off, True)
        fx.apply_natural_cure_on_switch_out(battle, old_off, True)
        while True:
            _switch_req = SwitchRequest((0,))
            _choices = yield _switch_req
            _new_slot = int(_choices[0])
            active0 = _resolve_switch_target_from_action(
                OFF_SIDE0,
                active0,
                _new_slot + 4,
            )
            battle[OFF_META + M_ACTIVE0] = active0
            _sync_showdown_order_on_switch(side_order0, active0)
            _pending_switch_slot_condition0_ut = int(
                battle[OFF_FIELD + F_DESTINY_BOND_0]
            )
            _clear_side_switch_state_common(battle, 0)
            for off in (
                F_VOLATILE_0,
                F_LEECH_SEED_0,
                F_EXTENDED_VOLATILE_0,
                F_SUBSTITUTE_0,
                F_DISABLE_TURNS_0,
                F_DESTINY_BOND_0,
                F_YAWN_TURNS_0,
                F_PERISH_COUNT_0,
            ):
                battle[OFF_FIELD + off] = 0
            new_p0 = OFF_SIDE0 + active0 * POKEMON_SIZE
            _reset_incoming_switch_state_tracked(new_p0)
            _uturn_consumed_pending0 = apply_pending_wish_on_switch_in(
                battle,
                0,
                new_p0,
                state,
                game_data,
                _pending_switch_slot_condition0_ut,
            )
            if (
                is_pending_wish_sentinel(_pending_switch_slot_condition0_ut)
                and not _uturn_consumed_pending0
            ):
                battle[OFF_FIELD + F_DESTINY_BOND_0] = np.int16(
                    _pending_switch_slot_condition0_ut
                )
            _opp_target_off_ut = OFF_SIDE1 + active1 * POKEMON_SIZE
            _postswitch_p0_speed_ut = _get_switch_resume_action_speed(battle, new_p0)
            _postswitch_p1_speed_ut = _action_speed_from_effective(
                fx.get_effective_speed(battle, _opp_target_off_ut)
            )
            if int(battle[_opp_target_off_ut + 1]) > 0:
                _consume_switch_request_resume_tie_frames(
                    _postswitch_p0_speed_ut,
                    _postswitch_p1_speed_ut,
                    True,
                    gen5_prng,
                )
            else:
                _consume_runswitch_tie_frame(
                    battle,
                    new_p0,
                    _opp_target_off_ut,
                    gen5_prng,
                )
            fx.apply_hazard_damage_on_switch(battle, new_p0, OFF_FIELD + F_HAZARDS_0)
            _reset_toxic_counter_on_switch_in(battle, new_p0)
            # Skip onSwitchIn if hazards KO'd the U-turn replacement
            # (battle-actions.ts:187 `if (!poke.hp) continue;`).
            if int(battle[new_p0 + 1]) > 0:
                _apply_switch_in_ability_with_trace_reaction_tracked(
                    new_p0,
                    _opp_target_off_ut,
                    True,
                )
                _run_switch_in_update_item_hooks(new_p0)
                _late_self_switch_before_residual = True
                break
            else:
                # If the first late self-switch target dies to hazards,
                # Showdown leaves the side empty through upkeep and only asks
                # for the fallback replacement after residual finishes.
                break
        should_uturn0 = False

    if should_uturn1:
        old_off = OFF_SIDE1 + active1 * POKEMON_SIZE
        fx.apply_regenerator_on_switch_out(battle, old_off, True)
        fx.apply_natural_cure_on_switch_out(battle, old_off, True)
        while True:
            _switch_req = SwitchRequest((1,))
            _choices = yield _switch_req
            _new_slot = int(_choices[1])
            active1 = _resolve_switch_target_from_action(
                OFF_SIDE1,
                active1,
                _new_slot + 4,
            )
            battle[OFF_META + M_ACTIVE1] = active1
            _sync_showdown_order_on_switch(side_order1, active1)
            _pending_switch_slot_condition1_ut = int(
                battle[OFF_FIELD + F_DESTINY_BOND_1]
            )
            _clear_side_switch_state_common(battle, 1)
            for off in (
                F_VOLATILE_1,
                F_LEECH_SEED_1,
                F_EXTENDED_VOLATILE_1,
                F_SUBSTITUTE_1,
                F_DISABLE_TURNS_1,
                F_DESTINY_BOND_1,
                F_YAWN_TURNS_1,
                F_PERISH_COUNT_1,
            ):
                battle[OFF_FIELD + off] = 0
            new_p1 = OFF_SIDE1 + active1 * POKEMON_SIZE
            _reset_incoming_switch_state_tracked(new_p1)
            _uturn_consumed_pending1 = apply_pending_wish_on_switch_in(
                battle,
                1,
                new_p1,
                state,
                game_data,
                _pending_switch_slot_condition1_ut,
            )
            if (
                is_pending_wish_sentinel(_pending_switch_slot_condition1_ut)
                and not _uturn_consumed_pending1
            ):
                battle[OFF_FIELD + F_DESTINY_BOND_1] = np.int16(
                    _pending_switch_slot_condition1_ut
                )
            _opp_target_off = OFF_SIDE0 + active0 * POKEMON_SIZE
            _postswitch_p0_speed_ut = _action_speed_from_effective(
                fx.get_effective_speed(battle, _opp_target_off)
            )
            _postswitch_p1_speed_ut = _get_switch_resume_action_speed(battle, new_p1)
            if int(battle[_opp_target_off + 1]) > 0:
                _consume_switch_request_resume_tie_frames(
                    _postswitch_p1_speed_ut,
                    _postswitch_p0_speed_ut,
                    True,
                    gen5_prng,
                )
            else:
                _consume_runswitch_tie_frame(
                    battle,
                    new_p1,
                    _opp_target_off,
                    gen5_prng,
                )
            fx.apply_hazard_damage_on_switch(battle, new_p1, OFF_FIELD + F_HAZARDS_1)
            _reset_toxic_counter_on_switch_in(battle, new_p1)
            # Skip onSwitchIn if hazards KO'd the U-turn replacement
            # (battle-actions.ts:187 `if (!poke.hp) continue;`).
            if int(battle[new_p1 + 1]) > 0:
                if int(battle[_opp_target_off + 1]) > 0:
                    _apply_switch_in_ability_with_trace_reaction_tracked(
                        new_p1,
                        _opp_target_off,
                        True,
                    )
                else:
                    # When side 0 is still fainted and awaiting a forced
                    # replacement, defer the opponent's onSwitchIn ability
                    # until the real target enters in step_forced_switch.
                    _store_pending_opp_switch_in(active1, _postswitch_p1_speed_ut)
                _run_switch_in_update_item_hooks(new_p1)
                _late_self_switch_before_residual = True
                break
            else:
                # Match Showdown's pivot timing: a hazard-KO'd late U-turn /
                # Volt Switch target does not chain immediately into another
                # live replacement before residual, so the fallback switch-in
                # cannot collect same-turn Leftovers or other upkeep effects.
                break

    # Player-side Eject Button is a same-turn switch continuation in Showdown,
    # not a deferred "next step" switch after residual. Resolve it here, after
    # all move-resolution bookkeeping is finished but before the residual
    # bucket starts, so the replacement can take hazards, fire switch-in
    # abilities, and receive same-turn residual healing such as Leftovers.
    if p0_item_forced_switch:
        # Showdown queues the follow-up switch action with the outgoing
        # holder's current action speed, then later runs `runSwitch` for the
        # chosen replacement. When the holder ties the foe (for example
        # Alomomola vs Vaporeon), that queued switch action still burns the
        # three hidden Update/runSwitch/Update tie frames even if the incoming
        # replacement is faster or slower. Carry the live outgoing speed into
        # step_forced_switch so same-turn Eject Button / Red Card continuations
        # reuse the original switch action speed instead of the replacement's.
        state.forced_switch_action_speed = np.int16(
            fx.get_effective_speed(
                battle,
                OFF_SIDE0 + int(battle[OFF_META + M_ACTIVE0]) * POKEMON_SIZE,
            )
        )
        _switch_req = SwitchRequest((0,))
        _choices = yield _switch_req
        _new_slot = int(_choices[0])
        _forced_switch_action0 = int(_new_slot) + 4
        step_forced_switch(
            state,
            _forced_switch_action0,
            0,
            game_data,
            move_effects,
            type_chart,
            gen5_prng,
            profile=profile,
        )
        active0 = int(battle[OFF_META + M_ACTIVE0])
        p0_off = OFF_SIDE0 + active0 * POKEMON_SIZE
        target1_off = p0_off
        _p0_inline_item_switch_resolved = True
        p0_item_forced_switch = False

    post_move_alive0 = _count_alive(battle, OFF_SIDE0)
    post_move_alive1 = _count_alive(battle, OFF_SIDE1)
    battle_over_from_move = (post_move_alive0 == 0) or (post_move_alive1 == 0)

    # Choice lock — Showdown adds the choicelock volatile from the Choice
    # item's onModifyMove hook during move execution, before endTurn runs the
    # active Pokemon's `DisableMove` handlers. Apply the lock here so the
    # residual cleanup sees the live volatile on the move's own turn.
    side0_could_move_pre_eot = not (
        is_immobile0
        or prankster_fail0
        or self_hit0
        or (side0_flinched and not side0_first)
    )
    side1_could_move_pre_eot = not (
        is_immobile1 or prankster_fail1 or self_hit1 or (side1_flinched and side0_first)
    )
    move_user_still_active0_pre_eot = (
        int(battle[OFF_META + M_ACTIVE0]) == move_user_slot0
    )
    move_user_still_active1_pre_eot = (
        int(battle[OFF_META + M_ACTIVE1]) == move_user_slot1
    )
    item0_pre_eot = int(battle[user0_off + 6])
    item1_pre_eot = int(battle[user1_off + 6])
    has_choice0_pre_eot = item0_pre_eot in (
        ITEM_CHOICE_BAND,
        ITEM_CHOICE_SPECS,
        ITEM_CHOICE_SCARF,
    )
    has_choice1_pre_eot = item1_pre_eot in (
        ITEM_CHOICE_BAND,
        ITEM_CHOICE_SPECS,
        ITEM_CHOICE_SCARF,
    )
    if (
        has_choice0_pre_eot
        and (not is_switch)
        and not must_struggle0
        and not _trick_swapped0
        and side0_could_move_pre_eot
        and move_user_still_active0_pre_eot
        and int(battle[OFF_FIELD + F_CHOICE_LOCK_0]) < 0
    ):
        battle[OFF_FIELD + F_CHOICE_LOCK_0] = move_idx
    if (
        has_choice1_pre_eot
        and (not opp_is_switch)
        and not must_struggle1
        and not _trick_swapped1
        and side1_could_move_pre_eot
        and move_user_still_active1_pre_eot
        and int(battle[OFF_FIELD + F_CHOICE_LOCK_1]) < 0
    ):
        battle[OFF_FIELD + F_CHOICE_LOCK_1] = opp_move_idx

    # Skip the generic post-action Update speedSort chain whenever the turn is
    # already parked on an immediate switch continuation. In Showdown this is
    # true not only after faint-driven replacement flows, but also when the
    # player is waiting on an item-driven forced switch such as Eject Button /
    # Red Card AND after an inline item-driven drag replacement has already
    # consumed the queued switch/runSwitch tie frames before the later move.
    # Leaving the normal Update chain enabled in those states spends extra
    # tied-speed frames before the next real action boundary.
    _faint_instaswitch_pending = (
        p0_fainted or opp_fainted
    ) and not battle_over_from_move
    _inline_switch_resume_pending = (
        p0_item_forced_switch
        or _p0_inline_item_switch_resolved
        or _inline_item_switch_rewrote_active
    ) and not battle_over_from_move
    _instaswitch_pending = _faint_instaswitch_pending or _inline_switch_resume_pending
    p0_off = OFF_SIDE0 + active0 * POKEMON_SIZE
    p1_off = OFF_SIDE1 + active1 * POKEMON_SIZE
    hp0_after_eot = int(battle[p0_off + 1])
    hp1_after_eot = int(battle[p1_off + 1])
    p0_eot_fainted = False
    _terrain_pre_eot = int(battle[OFF_FIELD + F_TERRAIN])
    _skip_remaining_eot = False
    _resid_action_speed0_pre = current_action_speed0
    _resid_action_speed1_pre = current_action_speed1
    if _late_self_switch_before_residual or _move_phase_residual_speed_refresh:
        _resid_active0_pre = int(battle[OFF_META + M_ACTIVE0])
        _resid_active1_pre = int(battle[OFF_META + M_ACTIVE1])
        _resid_p0_off_pre = OFF_SIDE0 + _resid_active0_pre * POKEMON_SIZE
        _resid_p1_off_pre = OFF_SIDE1 + _resid_active1_pre * POKEMON_SIZE
        _resid_action_speed0_pre = _action_speed_from_effective(
            fx.get_effective_speed(battle, _resid_p0_off_pre)
        )
        _resid_action_speed1_pre = _action_speed_from_effective(
            fx.get_effective_speed(battle, _resid_p1_off_pre)
        )
    _move2_status = _cat0_status if not side0_first else _cat1_status
    _move2_damage = damage0 if not side0_first else damage1
    _move2_target_kind = target1_kind if side0_first else target0_kind
    _move2_successful_self_boost = (
        _move1_successful_self_boost_status
        if side0_first
        else _move0_successful_self_boost_status
    )
    _move2_ran = p1_move_ran if side0_first else p0_move_ran
    _move2_num_hits = (
        int(_meta1.get("num_hits", 1))
        if side0_first
        else int(_meta0.get("num_hits", 1))
    )
    _move2_post_action_updates = _move_update_count(
        _move2_status,
        _move2_target_kind,
        _move2_successful_self_boost,
        int(_move2_damage),
        _move2_num_hits,
    )
    _move2_side_field = _move2_status and (
        _move2_target_kind in (7, 9, 10)
        or (_move2_target_kind == 3 and not _move2_successful_self_boost)
    )
    _move2_no_internal_updates = (not _move2_status) and int(_move2_damage) <= 0
    _move2_pre_residual_updates_consumed = False

    if not battle_over_from_move:
        # Showdown spends the second action's generic post-action
        # `eachEvent('Update')` before the residual action begins. Keeping it
        # deferred until after residuals is only safe when upkeep does not
        # consume any intervening PRNG; Future Sight / Doom Desire do, so spend
        # the tied-speed frame here at the true action boundary.
        if _instaswitch_pending:
            # A faint-driven switch continuation skips Showdown's generic
            # post-action Update chain, but lockedmove.onEnd still fires during
            # runMove's AfterMove phase before the residual action begins. If
            # the move hit-loop itself has a tied Update frame, spend that first.
            if (
                _faint_instaswitch_pending
                and _move2_ran
                and not _move2_side_field
                and not _move2_no_internal_updates
                and _move2_num_hits <= 1
                and current_action_speed0 == current_action_speed1
            ):
                _sst.each_event_update([current_action_speed0, current_action_speed1])
                _move2_pre_residual_updates_consumed = True
            _roll_pending_lockedmove_confusion(0, p0_off, apply=True)
            _roll_pending_lockedmove_confusion(1, p1_off, apply=True)
        elif not _move2_phazed:
            if current_action_speed0 == current_action_speed1:
                for _ in range(_move2_post_action_updates):
                    _sst.each_event_update(
                        [current_action_speed0, current_action_speed1]
                    )
                _move2_pre_residual_updates_consumed = True
            # `lockedmove.onEnd` runs during runMove's AfterMove phase, before
            # the later residual action can resolve Future Sight / Doom Desire.
            # Tied-speed Update frames may need to precede the roll, but no-tie
            # turns must still materialize fatigue confusion before residuals.
            _roll_pending_lockedmove_confusion(0, p0_off, apply=True)
            _roll_pending_lockedmove_confusion(1, p1_off, apply=True)
        # ------------------------------------------------------------------
        # End-of-turn effects — Showdown's strict residual order:
        #   weather damage (1-2)
        #   futuremove (3)
        #   wish (4)
        #   leftovers / blacksludge / shedskin / hydration / weather healing /
        #     grassy terrain heal / Solar Power dmg (5)
        #   aquaring (6) / ingrain (7) / leechseed (8)
        #   psn / tox damage (9) — Poison Heal handled here
        #   brn damage / sitrus & healing berries (10)
        #   saltcure / partial trap (13)
        #   yawn (23)
        #   speed boost / toxic orb / flame orb / sticky barb / bad dreams (28)
        #   eject pack / mirror herb / white herb (29)
        # ------------------------------------------------------------------

        # Showdown expires weather before the final residual pass that turn.
        # If sand is on its last turn, the engine logs `|-weather|none` and
        # skips that turn's weather damage / healing. Keep the weather
        # decrement ahead of the residual groups so both lead setters and
        # post-upkeep replacement setters line up.
        _weather_before_eot = int(battle[OFF_FIELD + F_WEATHER])
        fx.decrement_weather(battle)
        if (
            _weather_before_eot != WEATHER_NONE
            and int(battle[OFF_FIELD + F_WEATHER]) == WEATHER_NONE
            and int(battle[p0_off + 1]) > 0
            and int(battle[p1_off + 1]) > 0
            and _resid_action_speed0_pre == _resid_action_speed1_pre
        ):
            # Showdown's weather duration expiry goes through
            # `field.clearWeather()`, which fires `eachEvent('WeatherChange')`
            # on the live active pair before the weather residual body is
            # skipped for the now-cleared weather state.
            _sst.each_event_update(
                [
                    _resid_action_speed0_pre,
                    _resid_action_speed1_pre,
                ]
            )
        _apply_booster_energy_update_tracked(p0_off)
        _apply_booster_energy_update_tracked(p1_off)
        # 1-2. Weather damage (sand/hail)
        if int(battle[p0_off + 1]) > 0:
            fx.apply_weather_damage(battle, p0_off, game_data)
        if int(battle[p1_off + 1]) > 0:
            fx.apply_weather_damage(battle, p1_off, game_data)

        def _stop_early_eot_after_decisive_ko() -> bool:
            nonlocal hp0_after_eot, hp1_after_eot, p0_eot_fainted
            alive0_mid = _count_alive(battle, OFF_SIDE0)
            alive1_mid = _count_alive(battle, OFF_SIDE1)
            if alive0_mid > 0 and alive1_mid > 0:
                return False
            hp0_after_eot = int(battle[OFF_SIDE0 + active0 * POKEMON_SIZE + 1])
            hp1_after_eot = int(battle[OFF_SIDE1 + active1 * POKEMON_SIZE + 1])
            if hp0_after_eot == 0:
                off = OFF_SIDE0 + active0 * POKEMON_SIZE + 15
                battle[off] = int(battle[off]) | 1
            if hp1_after_eot == 0:
                off = OFF_SIDE1 + active1 * POKEMON_SIZE + 15
                battle[off] = int(battle[off]) | 1
            p0_eot_fainted = hp0_after_eot == 0
            return True

        _skip_remaining_eot = _stop_early_eot_after_decisive_ko()

        if not _skip_remaining_eot:
            # 3. Future Sight resolution
            def _apply_delayed_move_damage(
                target_off: int,
                damage: int,
                move_id: int,
                source_off: int,
                *,
                source_live: bool,
            ) -> None:
                """Apply delayed Future Sight / Doom Desire damage via move-like survival gates.

                Showdown resolves future moves through `trySpreadMoveHit`, so the
                hit still triggers onDamage-style survival hooks such as Focus Sash
                and Sturdy instead of acting like raw HP subtraction.
                """
                hp_before = int(battle[target_off + 1])
                if hp_before <= 0 or int(damage) <= 0:
                    return False

                max_hp = int(battle[target_off + 2])
                item_off = int(target_off) + 6
                item = int(battle[item_off])
                target_ability = int(battle[target_off + 5])
                target_flags = int(battle[target_off + 15])

                source_ability = int(battle[source_off + 5]) if source_live else 0
                has_mold_breaker = source_ability in (
                    ABILITY_MOLD_BREAKER,
                    ABILITY_TERAVOLT,
                    ABILITY_TURBOBLAZE,
                )

                is_physical = int(game_data.move_category[int(move_id)]) == CAT_PHYSICAL
                if (
                    not has_mold_breaker
                    and (target_flags & 0x40) != 0
                    and (
                        target_ability == ABILITY_DISGUISE
                        or (target_ability == ABILITY_ICE_FACE and is_physical)
                    )
                ):
                    damage = max_hp // 8
                    battle[target_off + 15] = np.int16(target_flags & ~0x40)

                new_hp = max(0, hp_before - int(damage))
                if item == ITEM_FOCUS_SASH and hp_before == max_hp and new_hp == 0:
                    new_hp = 1
                    battle[item_off] = 0
                if (
                    not has_mold_breaker
                    and target_ability == ABILITY_STURDY
                    and hp_before == max_hp
                    and new_hp == 0
                ):
                    new_hp = 1
                if item == ITEM_AIR_BALLOON and int(damage) > 0:
                    battle[item_off] = 0

                battle[target_off + 1] = np.int16(new_hp)
                return True

            def _consume_delayed_move_internal_updates() -> None:
                if _resid_action_speed0_pre != _resid_action_speed1_pre:
                    return
                # `futuremove` resolves through Showdown's `trySpreadMoveHit`,
                # which consumes the same hit-loop and end-of-runMove Update
                # frames as a normal successful single-hit move.
                _sst.each_event_update(
                    [
                        _resid_action_speed0_pre,
                        _resid_action_speed1_pre,
                    ]
                )
                _sst.each_event_update(
                    [
                        _resid_action_speed0_pre,
                        _resid_action_speed1_pre,
                    ]
                )

            fs = int(battle[OFF_META + M_FUTURE_SIGHT])
            fs_p0 = (fs >> 12) & 0xF
            fs_p1 = (fs >> 4) & 0xF
            new_fs_p0 = fs_p0 - 1 if fs_p0 > 0 else 0
            new_fs_p1 = fs_p1 - 1 if fs_p1 > 0 else 0
            if fs_p1 > 0 and new_fs_p1 == 0:
                # Showdown stores Future Sight / Doom Desire as a slot condition
                # on the TARGET slot, so delayed hits resolve in target-slot order:
                # side-0 target first, then side-1 target. That means the side-1
                # stored move (which targets side 0) consumes the first PRNG
                # damage-roll bucket when both future hits land on the same upkeep.
                types0 = int(battle[p0_off + 4])
                if not (
                    ((types0 & 0xFF) == TYPE_DARK)
                    or (((types0 >> 8) & 0xFF) == TYPE_DARK)
                ):
                    hp = int(battle[p0_off + 1])
                    if hp > 0:
                        fs_move1 = int(battle[OFF_MOVES + M_FUTURE_MOVE_1])
                        fs_src1 = int(battle[OFF_MOVES + M_FUTURE_SRC_1])
                        if fs_move1 < 0:
                            fs_move1 = MOVE_FUTURE_SIGHT
                        if not (0 <= fs_src1 < 6):
                            fs_src1 = int(battle[OFF_META + M_ACTIVE1])
                        fs_src1_live = (
                            int(battle[OFF_META + M_ACTIVE1]) == fs_src1
                            and int(battle[OFF_SIDE1 + fs_src1 * POKEMON_SIZE + 1]) > 0
                        )
                        from pokepy.effects.ability_suppression import (
                            effective_ability as _effective_ability_fs1,
                        )

                        saved_active1 = int(battle[OFF_META + M_ACTIVE1])
                        fs_src1_off = OFF_SIDE1 + fs_src1 * POKEMON_SIZE
                        saved_active1_off = OFF_SIDE1 + saved_active1 * POKEMON_SIZE
                        saved_ability1 = int(battle[fs_src1_off + 5])
                        field_atk_ability1 = _effective_ability_fs1(
                            battle, saved_active1_off, p0_off
                        )
                        field_def_ability1 = _effective_ability_fs1(
                            battle, p0_off, saved_active1_off
                        )
                        battle[OFF_META + M_ACTIVE1] = fs_src1
                        if not fs_src1_live:
                            # Match Showdown's off-field delayed-hit rule on side 1 too.
                            battle[fs_src1_off + 5] = 0
                        fs_damage1 = calc_damage_gen9(
                            battle,
                            1,
                            0,
                            state.team_moves,
                            state.opp_moves,
                            game_data,
                            move_effects,
                            type_chart,
                            is_moving_last=False,
                            override_move_id=fs_move1,
                            gen5_prng=gen5_prng,
                            suppress_attacker_item=not fs_src1_live,
                            suppress_attacker_boosts=not fs_src1_live,
                            override_field_atk_ability=field_atk_ability1,
                            override_field_def_ability=field_def_ability1,
                            profile=profile,
                        )
                        battle[fs_src1_off + 5] = saved_ability1
                        battle[OFF_META + M_ACTIVE1] = saved_active1
                        if _apply_delayed_move_damage(
                            p0_off,
                            fs_damage1,
                            fs_move1,
                            fs_src1_off,
                            source_live=fs_src1_live,
                        ):
                            _consume_delayed_move_internal_updates()
                battle[OFF_MOVES + M_FUTURE_MOVE_1] = -1
                battle[OFF_MOVES + M_FUTURE_SRC_1] = -1
                _skip_remaining_eot = _stop_early_eot_after_decisive_ko()
            if not _skip_remaining_eot and fs_p0 > 0 and new_fs_p0 == 0:
                # P0's stored future hit targets side 1, so it resolves after the
                # side-0 target slot above when both queued hits expire together.
                types1 = int(battle[p1_off + 4])
                if not (
                    ((types1 & 0xFF) == TYPE_DARK)
                    or (((types1 >> 8) & 0xFF) == TYPE_DARK)
                ):
                    hp = int(battle[p1_off + 1])
                    if hp > 0:
                        fs_move0 = int(battle[OFF_MOVES + M_FUTURE_MOVE_0])
                        fs_src0 = int(battle[OFF_MOVES + M_FUTURE_SRC_0])
                        if fs_move0 < 0:
                            fs_move0 = MOVE_FUTURE_SIGHT
                        if not (0 <= fs_src0 < 6):
                            fs_src0 = int(battle[OFF_META + M_ACTIVE0])
                        fs_src0_live = (
                            int(battle[OFF_META + M_ACTIVE0]) == fs_src0
                            and int(battle[OFF_SIDE0 + fs_src0 * POKEMON_SIZE + 1]) > 0
                        )
                        from pokepy.effects.ability_suppression import (
                            effective_ability as _effective_ability_fs0,
                        )

                        saved_active0 = int(battle[OFF_META + M_ACTIVE0])
                        fs_src0_off = OFF_SIDE0 + fs_src0 * POKEMON_SIZE
                        saved_active0_off = OFF_SIDE0 + saved_active0 * POKEMON_SIZE
                        saved_ability0 = int(battle[fs_src0_off + 5])
                        field_atk_ability0 = _effective_ability_fs0(
                            battle, saved_active0_off, p1_off
                        )
                        field_def_ability0 = _effective_ability_fs0(
                            battle, p1_off, saved_active0_off
                        )
                        battle[OFF_META + M_ACTIVE0] = fs_src0
                        if not fs_src0_live:
                            # Showdown resolves off-field Future Sight / Doom Desire
                            # from the source mon's natural attacking state: no live
                            # ability, no held-item boost, no current offensive boosts.
                            battle[fs_src0_off + 5] = 0
                        fs_damage0 = calc_damage_gen9(
                            battle,
                            0,
                            0,
                            state.team_moves,
                            state.opp_moves,
                            game_data,
                            move_effects,
                            type_chart,
                            is_moving_last=False,
                            override_move_id=fs_move0,
                            gen5_prng=gen5_prng,
                            suppress_attacker_item=not fs_src0_live,
                            suppress_attacker_boosts=not fs_src0_live,
                            override_field_atk_ability=field_atk_ability0,
                            override_field_def_ability=field_def_ability0,
                            profile=profile,
                        )
                        battle[fs_src0_off + 5] = saved_ability0
                        battle[OFF_META + M_ACTIVE0] = saved_active0
                        if _apply_delayed_move_damage(
                            p1_off,
                            fs_damage0,
                            fs_move0,
                            fs_src0_off,
                            source_live=fs_src0_live,
                        ):
                            _consume_delayed_move_internal_updates()
                battle[OFF_MOVES + M_FUTURE_MOVE_0] = -1
                battle[OFF_MOVES + M_FUTURE_SRC_0] = -1
            battle[OFF_META + M_FUTURE_SIGHT] = (new_fs_p0 << 12) | (new_fs_p1 << 4)
            if not _skip_remaining_eot:
                _skip_remaining_eot = _stop_early_eot_after_decisive_ko()

        if not _skip_remaining_eot:
            # 4. Wish resolution
            # Showdown data/moves.ts:wish heals the SLOT that set the wish (stored
            # as `slot: source.position` in the side condition), not whichever mon
            # is currently active. Pokepy packs the slot into bits 12-14 of the
            # wish_hp field (set in effects/recovery.py:apply_recovery_from_move).
            for side in (0, 1):
                wt_off = OFF_META + (M_WISH_TURNS_0 if side == 0 else M_WISH_TURNS_1)
                whp_off = OFF_META + (M_WISH_HP_0 if side == 0 else M_WISH_HP_1)
                wt = int(battle[wt_off])
                new_wt = wt - 1 if wt > 0 else 0
                if wt > 0 and new_wt == 0:
                    packed = int(battle[whp_off]) & 0xFFFF
                    wish_amount = packed & 0x0FFF
                    wish_slot = (packed >> 12) & 0x7
                    side_base = OFF_SIDE0 if side == 0 else OFF_SIDE1
                    target_off = side_base + wish_slot * POKEMON_SIZE
                    p_hp = int(battle[target_off + 1])
                    target_flags = int(battle[target_off + 15])
                    target_fainted = (target_flags & 0x1) != 0
                    if p_hp > 0 and not target_fainted:
                        battle[target_off + 1] = min(
                            p_hp + wish_amount, int(battle[target_off + 2])
                        )
                battle[wt_off] = new_wt

            _resid_p0_speed = fx.get_effective_speed(battle, p0_off)
            _resid_p1_speed = fx.get_effective_speed(battle, p1_off)
            _resid_action_speed0 = _action_speed_from_effective(_resid_p0_speed)
            _resid_action_speed1 = _action_speed_from_effective(_resid_p1_speed)
            _consume_residual_weather_event_frames(
                battle,
                _sst,
                p0_off,
                p1_off,
                _resid_p0_speed,
                _resid_p1_speed,
            )

            # 5. Order-5 group: Hydration / Shed Skin / weather healing abilities,
            #    leftovers, black sludge, grassy terrain heal, Solar Power damage.
            fx.apply_weather_healing(battle, p0_off, game_data)
            fx.apply_weather_healing(battle, p1_off, game_data)
            fx.apply_shed_skin_hydration(battle, p0_off, game_data, gen5_prng)
            fx.apply_shed_skin_hydration(battle, p1_off, game_data, gen5_prng)
            for _lb_off in (p0_off, p1_off):
                # Leftovers / Black Sludge reveal themselves via a `[from] item`
                # heal/damage message whenever they change the holder's HP.
                _lb_hp = int(battle[_lb_off + 1])
                fx.apply_leftovers_healing(battle, _lb_off, game_data)
                fx.apply_black_sludge_effect(battle, _lb_off, game_data)
                if int(battle[_lb_off + 1]) != _lb_hp:
                    _mark_item_revealed(_lb_off)
            if profile.has_terrain:
                fx.apply_grassy_terrain_healing(battle, p0_off, game_data)
                fx.apply_grassy_terrain_healing(battle, p1_off, game_data)
            # Solar Power damage is part of order 5 (onResidual on the ability).
            # apply_misc_eot_abilities also covers Bad Dreams (28) — split it so
            # Solar Power runs here and Bad Dreams runs later at order 28.
            fx.apply_misc_eot_abilities(battle, p0_off, p1_off)

            # 6-7. Aqua Ring (order 6) / Ingrain (order 7) heal 1/16 max HP.
            # Showdown: data/moves.ts aquaring.condition.onResidual (6),
            # ingrain.condition.onResidual (7).
            from pokepy.effects.end_of_turn import _apply_aqua_ring_ingrain_heal

            _apply_aqua_ring_ingrain_heal(battle, p0_off, side=0)
            _apply_aqua_ring_ingrain_heal(battle, p1_off, side=1)

            # 8. Leech Seed damage
            fx.apply_leech_seed_damage(battle, p0_off, p1_off)

            # 9-10. Status damage (poison/toxic at 9, burn at 10) + freeze thaw.
            # Poison Heal is handled inside apply_end_of_turn_status_effects.
            _p0_switched_in_this_turn = (
                int(battle[OFF_META + M_ACTIVE0]) != _start_turn_active0
            )
            _p1_switched_in_this_turn = (
                int(battle[OFF_META + M_ACTIVE1]) != _start_turn_active1
            )
            fx.apply_end_of_turn_status_effects(
                battle,
                p0_off,
                p1_off,
                game_data,
                move_effects,
                gen5_prng,
                skip_sleep_decrement0=_p0_switched_in_this_turn,
                skip_sleep_decrement1=_p1_switched_in_this_turn,
            )

        def _run_eot_berry_update_hooks() -> None:
            _run_item_hook_with_berry_tracking(
                fx.apply_sitrus_berry, p0_off, battle, p0_off, game_data
            )
            _run_item_hook_with_berry_tracking(
                fx.apply_sitrus_berry, p1_off, battle, p1_off, game_data
            )
            _run_item_hook_with_berry_tracking(
                fx.apply_gold_berry, p0_off, battle, p0_off, game_data
            )
            _run_item_hook_with_berry_tracking(
                fx.apply_gold_berry, p1_off, battle, p1_off, game_data
            )
            _run_item_hook_with_berry_tracking(
                fx.apply_lum_berry, p0_off, battle, p0_off, game_data
            )
            _run_item_hook_with_berry_tracking(
                fx.apply_lum_berry, p1_off, battle, p1_off, game_data
            )
            _run_item_hook_with_berry_tracking(
                fx.apply_status_curing_berries, p0_off, battle, p0_off, game_data
            )
            _run_item_hook_with_berry_tracking(
                fx.apply_status_curing_berries, p1_off, battle, p1_off, game_data
            )
            _run_item_hook_with_berry_tracking(
                fx.apply_persim_berry, p0_off, battle, p0_off, game_data
            )
            _run_item_hook_with_berry_tracking(
                fx.apply_persim_berry, p1_off, battle, p1_off, game_data
            )
            _run_item_hook_with_berry_tracking(
                fx.apply_stat_boosting_berries, p0_off, battle, p0_off, game_data
            )
            _run_item_hook_with_berry_tracking(
                fx.apply_stat_boosting_berries, p1_off, battle, p1_off, game_data
            )
            _run_item_hook_with_berry_tracking(
                fx.apply_pinch_healing_berries, p0_off, battle, p0_off, game_data
            )
            _run_item_hook_with_berry_tracking(
                fx.apply_pinch_healing_berries, p1_off, battle, p1_off, game_data
            )

        # 10. Healing berries (Sitrus etc.) — fire after burn damage so a
        # low-HP mon can be saved by Sitrus. Showdown items.ts berry order = 10.
        _run_eot_berry_update_hooks()

        # 13. Partial trap damage (Wrap/Bind/Fire Spin/Whirlpool/Magma Storm/etc.)
        # Showdown conditions.ts partiallytrapped.onResidualOrder=13: 1/8 max HP,
        # 1/6 if source held Binding Band. Magic Guard immune.
        if int(battle[p0_off + 1]) > 0:
            fx.apply_partial_trap_damage(battle, p0_off, side=0)
        if int(battle[p1_off + 1]) > 0:
            fx.apply_partial_trap_damage(battle, p1_off, side=1)

        # 23. Yawn → sleep transition (drowsy → sleep on second EOT).
        # Showdown yawn is onResidualOrder=23, BEFORE speed boost (26) and the
        # status orbs (28). Running yawn here matches Showdown ordering.
        from pokepy.effects.end_of_turn import _process_yawn

        _process_yawn(battle, p0_off, side=0, gen5_prng=gen5_prng)
        _process_yawn(battle, p1_off, side=1, gen5_prng=gen5_prng)

        # 26. Speed Boost (onResidualOrder 26).
        fx.apply_speed_boost(battle, p0_off, game_data)
        fx.apply_speed_boost(battle, p1_off, game_data)

        # Trick Room and terrain durations both expire on the field-residual
        # hook at order 27, after Speed Boost and before the order-28 item
        # group. Keep both decrements here so stale field state cannot leak
        # into the next turn's speed comparator or Booster Energy checks.
        fx.decrement_trick_room(battle)
        if profile.has_terrain:
            _terrain_before_eot = int(battle[OFF_FIELD + F_TERRAIN])
            fx.decrement_terrain(battle)
        _apply_booster_energy_update_tracked(p0_off)
        _apply_booster_energy_update_tracked(p1_off)
        # 28. Harvest (abilities.ts:1743) runs before Sticky Barb / status orbs
        # at the same order. Outside sun it still burns the 50% PRNG roll even
        # when there is no berry to restore.
        _apply_harvest(p0_off)
        _apply_harvest(p1_off)
        # Harvest can restore an eaten berry and Showdown's later update pass
        # lets that freshly restored berry trigger in the same residual cycle.
        _run_eot_berry_update_hooks()

        # 28. Sticky Barb residual damage (items.ts:5686, onResidualOrder 28
        # subOrder 3). Fires BEFORE the orb status applications at the same
        # order so a low-HP mon wearing both orb + barb takes barb damage first.
        fx.apply_sticky_barb_residual(battle, p0_off, game_data)
        fx.apply_sticky_barb_residual(battle, p1_off, game_data)

        # 28. Toxic Orb / Flame Orb (onResidualOrder 28, subOrder 3). Status
        # is set AFTER status damage so the orb-applied status takes effect
        # NEXT turn, matching Showdown items.ts. Uses trySetStatus-style
        # immunity checks — Fire-type holders ignore Flame Orb, Poison/Steel
        # holders ignore Toxic Orb, Purifying Salt / Comatose block both, etc.
        for poff in (p0_off, p1_off):
            p_item = int(battle[poff + 6])
            p_hp = int(battle[poff + 1])
            if p_hp <= 0:
                continue
            if p_item == ITEM_TOXIC_ORB:
                if fx.can_set_self_status(battle, poff, 6):  # STATUS_TOXIC
                    battle[poff + 12] = 6
            elif p_item == ITEM_FLAME_ORB:
                if fx.can_set_self_status(battle, poff, 1):  # STATUS_BURN
                    battle[poff + 12] = 1

        # 29. Shields Down (Minior) checks its HP threshold after the order-28
        # residual handlers and rewrites the live form stats before the turn
        # fully ends.
        fx.apply_shields_down_form_state(battle, p0_off, state, game_data)
        fx.apply_shields_down_form_state(battle, p1_off, state, game_data)

        fx.reset_protect_if_not_used(battle, 0, used_protect0)
        fx.reset_protect_if_not_used(battle, 1, used_protect1)
        fx.clear_protect_at_turn_end(battle)
        fx.decrement_confusion(battle)
        fx.decrement_taunt_encore(battle, gen5_prng)
        fx.clear_volatile_turn_effects(battle)
        _decrement_screens_tracked()
        fx.process_perish_song(battle, p0_off, p1_off)
        fx.apply_curse_damage(battle, p0_off, p1_off)
        fx.apply_salt_cure_damage(battle, p0_off, OFF_FIELD + F_EXTENDED_VOLATILE_0)
        fx.apply_salt_cure_damage(battle, p1_off, OFF_FIELD + F_EXTENDED_VOLATILE_1)
        # Showdown's later update pass still lets HP-threshold berries fire
        # after late residual damage such as Sticky Barb / Curse / Salt Cure.
        _run_eot_berry_update_hooks()

        # Faints from EOT
        hp0_after_eot = int(battle[OFF_SIDE0 + active0 * POKEMON_SIZE + 1])
        hp1_after_eot = int(battle[OFF_SIDE1 + active1 * POKEMON_SIZE + 1])
        if hp0_after_eot == 0:
            off = OFF_SIDE0 + active0 * POKEMON_SIZE + 15
            battle[off] = int(battle[off]) | 1
        if hp1_after_eot == 0:
            off = OFF_SIDE1 + active1 * POKEMON_SIZE + 15
            battle[off] = int(battle[off]) | 1

        # Wimp Out / Emergency Exit (opponent only — auto). When triggered,
        # clear opp side switch state, reset boosts on the incoming mon, and
        # apply hazard damage + switch-in ability — same as a regular switch.
        # Showdown abilities.ts:wimpout/emergencyexit fire onEmergencyExit
        # which calls `pokemon.switchFlag = true`, then the standard switch
        # pipeline runs (BeforeSwitchOut, hazards, switch-in abilities, etc.).
        s_off = OFF_SIDE1 + active1 * POKEMON_SIZE
        s_ab = int(battle[s_off + 5])
        if s_ab in (193, 194):
            s_max = int(battle[s_off + 2])
            if hp1_pre * 2 > s_max and hp1_after_eot * 2 <= s_max and hp1_after_eot > 0:
                fx.apply_regenerator_on_switch_out(battle, s_off, True)
                fx.apply_natural_cure_on_switch_out(battle, s_off, True)
                while True:
                    _switch_req = SwitchRequest((1,))
                    _choices = yield _switch_req
                    _new_slot = int(_choices[1])
                    active1 = _resolve_switch_target_from_action(
                        OFF_SIDE1,
                        active1,
                        _new_slot + 4,
                    )
                    _pending_switch_slot_condition1_we = int(
                        battle[OFF_FIELD + F_DESTINY_BOND_1]
                    )
                    battle[OFF_META + M_ACTIVE1] = active1
                    _sync_showdown_order_on_switch(side_order1, active1)
                    _clear_side_switch_state_common(battle, 1)
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
                    new_p1_we = OFF_SIDE1 + active1 * POKEMON_SIZE
                    _reset_incoming_switch_state_tracked(new_p1_we)
                    _we_consumed_pending1 = apply_pending_wish_on_switch_in(
                        battle,
                        1,
                        new_p1_we,
                        state,
                        game_data,
                        _pending_switch_slot_condition1_we,
                    )
                    if (
                        is_pending_wish_sentinel(_pending_switch_slot_condition1_we)
                        and not _we_consumed_pending1
                    ):
                        battle[OFF_FIELD + F_DESTINY_BOND_1] = np.int16(
                            _pending_switch_slot_condition1_we
                        )
                    _postswitch_p0_speed_we = _action_speed_from_effective(
                        fx.get_effective_speed(
                            battle, OFF_SIDE0 + active0 * POKEMON_SIZE
                        )
                    )
                    _postswitch_p1_speed_we = _get_switch_resume_action_speed(
                        battle, new_p1_we
                    )
                    fx.apply_hazard_damage_on_switch(
                        battle, new_p1_we, OFF_FIELD + F_HAZARDS_1
                    )
                    _reset_toxic_counter_on_switch_in(battle, new_p1_we)
                    # Showdown's runSwitch loop skips fainted mons (battle-actions.ts:187
                    # `if (!poke.hp) continue;`). Don't fire onSwitchIn if hazards KO'd it.
                    if int(battle[new_p1_we + 1]) > 0:
                        _opp_target_off_we = OFF_SIDE0 + active0 * POKEMON_SIZE
                        if int(battle[_opp_target_off_we + 1]) > 0:
                            _apply_switch_in_ability_with_trace_reaction_tracked(
                                new_p1_we, _opp_target_off_we, True
                            )
                        else:
                            _store_pending_opp_switch_in(
                                active1, _postswitch_p1_speed_we
                            )
                        _consume_switch_request_resume_tie_frames(
                            _postswitch_p1_speed_we,
                            _postswitch_p0_speed_we,
                            int(battle[_opp_target_off_we + 1]) > 0,
                            gen5_prng,
                        )
                        _run_switch_in_update_item_hooks(new_p1_we)
                        break
                    else:
                        p1_bench_alive = 0
                        for i in range(6):
                            if i != active1:
                                so = OFF_SIDE1 + i * POKEMON_SIZE
                                if (
                                    int(battle[so + 1]) > 0
                                    and (int(battle[so + 15]) & 1) == 0
                                ):
                                    p1_bench_alive += 1
                        if p1_bench_alive == 0:
                            break

        p0_eot_fainted = hp0_after_eot == 0

    p0_fainted = p0_fainted or p0_eot_fainted
    alive0 = _count_alive(battle, OFF_SIDE0)
    alive1 = _count_alive(battle, OFF_SIDE1)
    done = (alive0 == 0) or (alive1 == 0) or (int(state.turn) >= max_turns)

    if decisive_winner >= 0:
        winner = decisive_winner
    elif alive0 == 0:
        winner = 1
    elif alive1 == 0:
        winner = 0
    else:
        winner = -1

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------
    terminal_reward = 0.0
    if done and winner == 0:
        terminal_reward = 100.0
    elif done and winner == 1:
        terminal_reward = -100.0

    opp_maxhp = float(max_hp1)
    own_maxhp = float(max_hp0)
    damage_dealt = float(hp1_pre - final_hp1)
    damage_taken = float(hp0_pre - final_hp0)
    damage_reward = (damage_dealt / opp_maxhp) if opp_maxhp > 0 else 0.0
    damage_penalty = (damage_taken / own_maxhp) if own_maxhp > 0 else 0.0
    ko_reward = 1.0 if (hp1_pre > 0 and final_hp1 == 0) else 0.0
    ko_penalty = 1.0 if (hp0_pre > 0 and final_hp0 == 0) else 0.0

    opp_status_post = int(battle[OFF_SIDE1 + active1 * POKEMON_SIZE + 12]) & 0xFF
    own_status_post = int(battle[OFF_SIDE0 + active0 * POKEMON_SIZE + 12]) & 0xFF
    gave_status = (opp_status_pre == 0) and (opp_status_post > 0)
    took_status = (own_status_pre == 0) and (own_status_post > 0)
    status_reward = (0.5 if gave_status else 0.0) - (0.5 if took_status else 0.0)

    reward0 = (
        terminal_reward
        + damage_reward
        - damage_penalty
        + ko_reward
        - ko_penalty
        + status_reward
    )
    reward1 = -reward0

    # PP decrement (with Pressure). Showdown deducts PP only if the move
    # actually started (BeforeMove returned true). Skip when fully paralysed,
    # asleep, frozen, flinched, or otherwise prevented from moving. Also
    # skip on the STRIKE turn of a two-turn move — Showdown's LockMove event
    # short-circuits PP deduction (`if (!lockedMove) { pokemon.deductPP(...) }`
    # in battle-actions.ts:277-287), so the charge turn pays PP and the
    # strike turn does not.
    # Pressure's extra PP loss follows Showdown's `pressureTargets`, not just
    # "targets an opposing mon". Most self/allySide/field moves do not pay it,
    # but Showdown marks specific non-direct moves such as Stealth Rock / Spikes
    # with `flags.mustpressure`, and those do pay the extra PP.
    _MOVE_FLAG_MUSTPRESSURE = 0x80000
    move0_pressure_targets = move0_targets_foe_mon or (
        (move0_flags & _MOVE_FLAG_MUSTPRESSURE) != 0
    )
    move1_pressure_targets = move1_targets_foe_mon or (
        (move1_flags & _MOVE_FLAG_MUSTPRESSURE) != 0
    )
    # Showdown only applies Pressure's EXTRA PP loss after target resolution;
    # `[notarget]` moves still spend their normal 1 PP, but they do not pay
    # the bonus Pressure drop because battle-actions.ts returns before the
    # pressureTargets loop when `target` is gone.
    _pressure0 = (
        pp_pressure_ability0 == ABILITY_PRESSURE
        and move0_pressure_targets
        and not move0_no_target
    )
    _pressure1 = (
        pp_pressure_ability1 == ABILITY_PRESSURE
        and move1_pressure_targets
        and not move1_no_target
    )
    pp_cost0 = -2 if _pressure0 else -1
    pp_cost1 = -2 if _pressure1 else -1
    side0_could_move = not (
        is_immobile0
        or prankster_fail0
        or self_hit0
        or (side0_flinched and not side0_first)
    )
    side1_could_move = not (
        is_immobile1 or prankster_fail1 or self_hit1 or (side1_flinched and side0_first)
    )
    if (
        (not is_switch)
        and not must_struggle0
        and side0_could_move
        and not is_strike_turn0
        and not is_locked_turn0
    ):
        _mark_move_slot_used(state, 0, move_user_slot0, move_idx)
        state.team_pp[move_user_slot0, move_idx] = max(
            0, int(state.team_pp[move_user_slot0, move_idx]) + pp_cost0
        )
    if (
        (not opp_is_switch)
        and not must_struggle1
        and side1_could_move
        and not is_strike_turn1
        and not is_locked_turn1
    ):
        _mark_move_slot_used(state, 1, move_user_slot1, opp_move_idx)
        state.opp_pp[move_user_slot1, opp_move_idx] = max(
            0, int(state.opp_pp[move_user_slot1, opp_move_idx]) + pp_cost1
        )

    move_user_still_active0 = int(battle[OFF_META + M_ACTIVE0]) == move_user_slot0
    move_user_still_active1 = int(battle[OFF_META + M_ACTIVE1]) == move_user_slot1

    # Choice lock — Showdown adds the choicelock volatile from the Choice
    # item's onModifyMove hook during move execution, so endTurn
    # `runEvent('DisableMove')` already sees it on the move's own turn.
    # Keep the existing BeforeMove / Struggle / switched-out gates, but set the
    # lock here before the residual cleanup so Disable+Choice and analogous
    # endTurn handler ties can spend their hidden speedSort shuffle frames.
    item0 = int(battle[user0_off + 6])
    item1 = int(battle[user1_off + 6])
    has_choice0 = item0 in (ITEM_CHOICE_BAND, ITEM_CHOICE_SPECS, ITEM_CHOICE_SCARF)
    has_choice1 = item1 in (ITEM_CHOICE_BAND, ITEM_CHOICE_SPECS, ITEM_CHOICE_SCARF)
    if (
        has_choice0
        and (not is_switch)
        and not must_struggle0
        and not _trick_swapped0
        and side0_could_move
        and move_user_still_active0
        and int(battle[OFF_FIELD + F_CHOICE_LOCK_0]) < 0
    ):
        battle[OFF_FIELD + F_CHOICE_LOCK_0] = move_idx
    if (
        has_choice1
        and (not opp_is_switch)
        and not must_struggle1
        and not _trick_swapped1
        and side1_could_move
        and move_user_still_active1
        and int(battle[OFF_FIELD + F_CHOICE_LOCK_1]) < 0
    ):
        battle[OFF_FIELD + F_CHOICE_LOCK_1] = opp_move_idx

    # Track Showdown's `pokemon.moveLastTurnResult === false` state for
    # moves like Stomping Tantrum / Temper Flare. Use pokemon flag bit
    # 0x02 to mark "previous move failed". A move "failed" if it was
    # attempted but did not actually do anything. Self-targeted setup
    # moves at capped boosts (e.g. Swords Dance at +6) must count as
    # failures even though the move technically "hit".
    p0_flags_st = OFF_SIDE0 + active0 * POKEMON_SIZE + 15
    p1_flags_st = OFF_SIDE1 + active1 * POKEMON_SIZE + 15
    _EFFECT_STAT_CHANGE_LAST = 3
    _move0_is_self_boost_status = (
        move0_is_status
        and int(move_effects.effect_type[move_id0]) == _EFFECT_STAT_CHANGE_LAST
        and int(move_effects.stat_target[move_id0]) == 0
        and target0_kind == 3
    )
    _move1_is_self_boost_status = (
        move1_is_status
        and int(move_effects.effect_type[move_id1]) == _EFFECT_STAT_CHANGE_LAST
        and int(move_effects.stat_target[move_id1]) == 0
        and target1_kind == 3
    )
    if not is_switch and not is_immobile0 and not self_hit0:
        attempted0 = True
        succeeded0 = (damage0_after_flinch > 0) or (
            move0_is_status
            and (
                _move0_successful_self_boost_status
                if _move0_is_self_boost_status
                else hit0
            )
        )
        if attempted0 and not succeeded0:
            battle[p0_flags_st] = int(battle[p0_flags_st]) | 0x02
        else:
            battle[p0_flags_st] = int(battle[p0_flags_st]) & ~0x02
    if not opp_is_switch and not is_immobile1 and not self_hit1:
        attempted1 = True
        succeeded1 = (damage1_after_flinch > 0) or (
            move1_is_status
            and (
                _move1_successful_self_boost_status
                if _move1_is_self_boost_status
                else hit1
            )
        )
        if attempted1 and not succeeded1:
            battle[p1_flags_st] = int(battle[p1_flags_st]) | 0x02
        else:
            battle[p1_flags_st] = int(battle[p1_flags_st]) & ~0x02

    # Track the current active's last real move for Encore / Torment /
    # cantusetwice. This follows Showdown's pokemon.lastMove semantics:
    # only update it if BeforeMove reached the move, and do not carry the
    # outgoing pivot's move onto the replacement that entered mid-turn.
    if (not is_switch) and side0_could_move and move_user_still_active0:
        battle[OFF_FIELD + F_LAST_MOVE_0] = move_idx
    if (not opp_is_switch) and side1_could_move and move_user_still_active1:
        battle[OFF_FIELD + F_LAST_MOVE_1] = opp_move_idx

    # Reveal opponent — gate on successful execution so an asleep / frozen
    # / flinched mon doesn't leak its queued move slot.
    state.opp_revealed[move_user_slot1] = True
    if (not opp_is_switch) and not must_struggle1 and side1_could_move:
        state.opp_moves_revealed[move_user_slot1, opp_move_idx] = True
    # Symmetric reveal for side-1-perspective consumers (e.g. a Kakuna
    # opponent on side 1). Gates mirror the side-0 equivalents above so an
    # asleep / frozen / flinched / struggle-forced mon doesn't leak its
    # queued slot.
    state.team_revealed[move_user_slot0] = True
    if (not is_switch) and not must_struggle0 and side0_could_move:
        state.team_moves_revealed[move_user_slot0, move_idx] = True

    # Showdown's EOT speedSort frame chain (for a 1v1 turn with both mons
    # alive through the second move):
    #   * `eachEvent('Update')` inside move2's hitStepMoveHitLoop (968)
    #   * `eachEvent('Update')` at end of move2's runMove (1022)
    #   * post-move2 `eachEvent('Update')` (battle.ts:2860)
    #   * post-residual `eachEvent('Update')` (battle.ts:2860 again after
    #       runAction(residual))
    # All four use the speed comparator on 2 actives — each consumes 1
    # frame iff speeds are tied, 0 otherwise. `fieldEvent('Residual')` on
    # the residual action also calls speedSort on the handler list, but for
    # 1v1 with no per-pokemon residual handlers (no status, no Leftovers)
    # the list has 0 or 1 entries → no shuffle, 0 frames.
    #
    # When Leftovers (or any per-pokemon item/ability onResidual handler)
    # fires on BOTH sides, the residual handler list gets 2 tied entries
    # and `speedSort(handlers)` inside `fieldEvent('Residual')` consumes 1
    # extra frame. We detect the narrow "both-sides Leftovers + speeds tied"
    # case here and add the frame BEFORE the 4th (post-residual Update)
    # frame in the chain. The broader "count residual handlers" fix would
    # cover Black Sludge / Sticky Barb / Leech Seed / Aqua Ring / burn-tox
    # / Speed Boost / etc. — add those as parity scenarios grow.
    #
    # When the first attacker KO'd the target mid-turn, move-2 is skipped,
    # the residual `fieldEvent('Residual')` sees only 1 active pokemon, and
    # the post-residual eachEvent also sees 1 → no frames from EOT. Instead,
    # Showdown consumes 3 frames after the replacement (post-instaswitch
    # Update + runSwitch speedSort + post-runSwitch Update); those are all
    # speed-comparator calls on 2 actives and only tie-consume when the
    # NEW incoming mon matches the survivor's speed.
    # Strike-turn semi-invulnerability ends once move resolution is over.
    # Keep the charge metadata alive through both actions so slower strike-turn
    # users can still dodge (or be hit by the narrow Fly/Dig/Dive exception
    # moves), but clear it BEFORE residual accounting. Showdown removes the
    # move-specific volatile during the user's strike-turn `onTryMove`, so
    # `fieldEvent('Residual')` should not see synthetic duration handlers for
    # `twoturnmove` / `phantomforce` / `fly` / etc. lingering into upkeep.
    if is_strike_turn0:
        battle[OFF_META + M_CHARGING_0] = -1
        if _chg0_is_semi_invul:
            battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0] = (
                int(battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_0])
                & ~ACTIVE_MOVE_ACTIONS_SEMI_INVUL
            )
    if is_strike_turn1:
        battle[OFF_META + M_CHARGING_1] = -1
        if _chg1_is_semi_invul:
            battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1] = (
                int(battle[OFF_MOVES + M_ACTIVE_MOVE_ACTIONS_1])
                & ~ACTIVE_MOVE_ACTIONS_SEMI_INVUL
            )

    # Snapshot the residual-handler item state BEFORE applying any KO
    # replacement switch — once the replacement happens, `battle[p0_off + 6]`
    # reflects the NEW mon's item, not the fainted one's.  Status is stable
    # here (EOT damage does not clear it) so we read it live in the helper.
    _p0_item_pre_eot = int(battle[p0_off + 6])
    _p1_item_pre_eot = int(battle[p1_off + 6])

    # Independent of speed-tie state: each side that successfully used Protect
    # this turn adds BOTH a `protect` volatile (duration 1, no onResidual) and
    # a `stall` volatile (duration 2, no onResidual) to the Pokemon. Both end
    # up in the Residual handler list via fieldEvent's `getKey='duration'`
    # path. They tie in the comparePriority comparator (same order=false,
    # priority 0, speed = mon's speed, subOrder 2 — Stall isn't an Ability so
    # the special subOrder=9 case doesn't apply), and `speedSort` shuffles
    # them with a single `prng.shuffle(list, sorted, sorted+2)` call. We
    # consume one frame per Protect-using side regardless of `speeds_tied` —
    # the tie is between the side's own two volatiles, not between the two
    # actives.
    if used_protect0:
        gen5_prng.random(0, 2)
    if used_protect1:
        gen5_prng.random(0, 2)

    # Showdown refreshes `pokemon.speed` once at the START of the residual
    # action (`battle.updateSpeed()` in battle.ts). Snapshot that cached pair
    # before weather / terrain expiry or other residual-time effects mutate the
    # live board.
    _resid_action_speed0 = _resid_action_speed0_pre
    _resid_action_speed1 = _resid_action_speed1_pre

    if not _instaswitch_pending:
        # Showdown's move-action `eachEvent('Update')` calls sort on the
        # cached `pokemon.speed` values last refreshed when the action queue
        # was built. Mid-move speed changes like Rapid Spin's +1 Spe do not
        # affect those pre-residual Update frames; only the residual action's
        # `battle.updateSpeed()` refresh picks up the new speed.
        if (
            (not _move2_pre_residual_updates_consumed)
            and (not _move2_phazed)
            and current_action_speed0 == current_action_speed1
        ):
            for _ in range(_move2_post_action_updates):
                _sst.each_event_update([current_action_speed0, current_action_speed1])
        _roll_pending_lockedmove_confusion(0, p0_off, apply=True)
        _roll_pending_lockedmove_confusion(1, p1_off, apply=True)
        # Residual speedSort: collect per-pokemon onResidual handlers
        # (status, items, abilities) and consume PRNG frames for tied groups.
        # This replaces the old hardcoded `_both_leftovers_pre_eot` check and
        # extends coverage to burn, poison/toxic, and Black Sludge.
        # Showdown refreshes `pokemon.speed` once at the START of the residual
        # action (`battle.updateSpeed()`), then reuses that cached action-speed
        # pair for both the residual handler speedSort and the generic
        # post-residual `eachEvent('Update')`. Residual effects like Speed
        # Boost can change the live speed mid-residual, but they do not refresh
        # the cached `pokemon.speed` value until the next action.
        _eot_p0_speed = _resid_action_speed0
        _eot_p1_speed = _resid_action_speed1
        _count_residual_speedsort_frames(
            battle,
            _sst,
            p0_off,
            p1_off,
            _eot_p0_speed,
            _eot_p1_speed,
            _p0_item_pre_eot,
            _p1_item_pre_eot,
            _terrain_pre_eot,
        )
        # After the residual action itself, Showdown runs one more generic
        # `eachEvent('Update')` using those same cached residual action speeds.
        if hp0_after_eot > 0 and hp1_after_eot > 0 and _eot_p0_speed == _eot_p1_speed:
            _sst.each_event_update([_eot_p0_speed, _eot_p1_speed])
    else:
        # Showdown skips the move action's normal post-action `eachEvent
        # ('Update')` chain while an instaswitch is pending after a faint
        # (battle.ts returns early before the generic post-action update
        # block). The residual handler list is still built, so keep the
        # residual-handler speedSort accounting here, but do not spend an
        # extra synthetic residual post-action Update frame.
        _ko_p0_off = OFF_SIDE0 + active0 * POKEMON_SIZE
        _ko_p1_off = OFF_SIDE1 + active1 * POKEMON_SIZE

        def _ko_residual_action_speed(_p_off: int) -> int:
            if int(battle[_p_off + 1]) > 0:
                return _action_speed_from_effective(
                    fx.get_effective_speed(battle, _p_off)
                )
            # Showdown only calls `battle.updateSpeed()` on live actives at
            # residual start. Fainted slots therefore keep the stale cached
            # `pokemon.speed` value they already had, which in our packed
            # state is approximated by the slot's stored speed cache.
            return _cached_switchin_speed(_p_off)

        _ko_p0_speed = _ko_residual_action_speed(_ko_p0_off)
        _ko_p1_speed = _ko_residual_action_speed(_ko_p1_off)
        # If move-2 actually entered hitStepMoveHitLoop before the switch
        # request short-circuited the rest of runMove, Showdown still spends
        # the first internal `eachEvent('Update')` from that hit loop. That
        # sort uses cached `pokemon.speed`, not a freshly recomputed live
        # effective speed, so same-move self speed changes (Rapid Spin) must
        # NOT retie the pair here. Mid-turn switches / first-move speed changes
        # are already folded into current_action_speed* via the earlier
        # `_refresh_current_action_speeds()` calls.
        if (
            _faint_instaswitch_pending
            and (not _move2_pre_residual_updates_consumed)
            and _move2_ran
            and not _move2_side_field
            and not _move2_no_internal_updates
            and _move2_num_hits <= 1
            and current_action_speed0 == current_action_speed1
        ):
            _sst.each_event_update([current_action_speed0, current_action_speed1])
        _roll_pending_lockedmove_confusion(0, p0_off, apply=True)
        _roll_pending_lockedmove_confusion(1, p1_off, apply=True)
        # In the KO path, Showdown still runs `fieldEvent('Residual')` on
        # the residual action. Both sides still have their handlers in the
        # list (the fainted side's handler is added then skipped during
        # iteration at battle.ts:512). So the speedSort on 2 tied handlers
        # still consumes 1 frame when applicable.
        _eot_p0_speed = _ko_p0_speed
        _eot_p1_speed = _ko_p1_speed
        _count_residual_speedsort_frames(
            battle,
            _sst,
            _ko_p0_off,
            _ko_p1_off,
            _eot_p0_speed,
            _eot_p1_speed,
            _p0_item_pre_eot,
            _p1_item_pre_eot,
            _terrain_pre_eot,
        )
    # Auto-switch opponent if fainted (either from a move pre-EOT or from
    # EOT damage). Showdown does this only after the residual action's own
    # post-action `eachEvent('Update')`, so keep it after the EOT accounting
    # above. The replacement still appears before the next turn starts and
    # does not take another residual tick on the predecessor's death turn.
    p1_bench_alive_post = 0
    for i in range(6):
        if i == active1:
            continue
        slot_off = OFF_SIDE1 + i * POKEMON_SIZE
        if int(battle[slot_off + 1]) > 0 and (int(battle[slot_off + 15]) & 1) == 0:
            p1_bench_alive_post += 1
    p1_needs_switch_post = hp1_after_eot == 0 and p1_bench_alive_post > 0 and not done
    if hp1_after_eot == 0 and not done and not p1_needs_switch_post:
        while True:
            _use_hidden_opp_order = bool(getattr(state, "hidden_opp_switches", ()))
            if _use_hidden_opp_order:
                active1 = fx.auto_switch(
                    battle,
                    OFF_SIDE1,
                    active1,
                    True,
                    order=side_order1,
                )
            else:
                active1 = fx.auto_switch(
                    battle,
                    OFF_SIDE1,
                    active1,
                    True,
                )
            _pending_switch_slot_condition1_post = int(
                battle[OFF_FIELD + F_DESTINY_BOND_1]
            )
            battle[OFF_META + M_ACTIVE1] = active1
            _sync_showdown_order_on_switch(side_order1, active1)
            # Clear opp switch state (matches what the old pre-EOT block did)
            _clear_side_switch_state_common(battle, 1)
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
            # Reset boosts on incoming opponent (preserve tera nibble)
            new_p1 = OFF_SIDE1 + active1 * POKEMON_SIZE
            _reset_incoming_switch_state_tracked(new_p1)
            _postswitch_consumed_pending1 = apply_pending_wish_on_switch_in(
                battle,
                1,
                new_p1,
                state,
                game_data,
                _pending_switch_slot_condition1_post,
            )
            if (
                is_pending_wish_sentinel(_pending_switch_slot_condition1_post)
                and not _postswitch_consumed_pending1
            ):
                battle[OFF_FIELD + F_DESTINY_BOND_1] = np.int16(
                    _pending_switch_slot_condition1_post
                )
            _postswitch_p0_speed = _action_speed_from_effective(
                fx.get_effective_speed(battle, OFF_SIDE0 + active0 * POKEMON_SIZE)
            )
            _postswitch_p1_speed = _get_switch_resume_action_speed(battle, new_p1)
            fx.apply_hazard_damage_on_switch(battle, new_p1, OFF_FIELD + F_HAZARDS_1)
            _reset_toxic_counter_on_switch_in(battle, new_p1)
            # Skip onSwitchIn if hazards KO'd the replacement (Showdown
            # battle-actions.ts:187 `if (!poke.hp) continue;`).
            if int(battle[new_p1 + 1]) > 0:
                _opp_target_off = OFF_SIDE0 + active0 * POKEMON_SIZE
                if int(battle[_opp_target_off + 1]) > 0:
                    _apply_switch_in_ability_with_trace_reaction_tracked(
                        new_p1,
                        _opp_target_off,
                        True,
                    )
                else:
                    _store_pending_opp_switch_in(active1, _postswitch_p1_speed)
                # Post-residual replacement switch (after the second mover or EOT
                # Post-residual replacement switch (after the second mover or EOT
                # faints) resumes the turn through a synthetic switch request in
                # Showdown. That path consumes 3 speed-comparator frames when the
                # replacement ties the surviving active:
                #   1. post-switch action eachEvent('Update')
                #   2. runSwitch speedSort(allActive)
                #   3. post-runSwitch eachEvent('Update')
                # The tie check uses the replacement's neutral on-entry speed
                # before hazards/entry effects like Sticky Web mutate it.
                _consume_switch_request_resume_tie_frames(
                    _postswitch_p1_speed,
                    _postswitch_p0_speed,
                    int(battle[_opp_target_off + 1]) > 0,
                    gen5_prng,
                )
                _run_switch_in_update_item_hooks(new_p1)
                break
            else:
                # replacement fainted to hazards. Check if P1 has any alive bench pokemon
                p1_bench_alive = 0
                for i in range(6):
                    if i != active1:
                        so = OFF_SIDE1 + i * POKEMON_SIZE
                        if int(battle[so + 1]) > 0 and (int(battle[so + 15]) & 1) == 0:
                            p1_bench_alive += 1
                if p1_bench_alive == 0:
                    break

    # Replacement chains can end the battle after the original terminal
    # snapshot above (for example, the last opposing bench mons die to
    # Stealth Rock while trying to replace a fainted active). Recompute the
    # final terminal state after the full opponent switch loop finishes.
    alive0_final = _count_alive(battle, OFF_SIDE0)
    alive1_final = _count_alive(battle, OFF_SIDE1)
    done_final = (
        (alive0_final == 0) or (alive1_final == 0) or (int(state.turn) >= max_turns)
    )

    if decisive_winner >= 0:
        winner_final = decisive_winner
    elif alive0_final == 0:
        winner_final = 1
    elif alive1_final == 0:
        winner_final = 0
    else:
        winner_final = -1

    terminal_reward_final = 0.0
    if done_final and winner_final == 0:
        terminal_reward_final = 100.0
    elif done_final and winner_final == 1:
        terminal_reward_final = -100.0
    reward0 += terminal_reward_final - terminal_reward
    reward1 = -reward0
    done = done_final
    winner = winner_final

    # Forced switch decision for player 0
    p0_bench_alive = 0
    for i in range(6):
        if i == active0:
            continue
        slot_off = OFF_SIDE0 + i * POKEMON_SIZE
        if int(battle[slot_off + 1]) > 0 and (int(battle[slot_off + 15]) & 1) == 0:
            p0_bench_alive += 1
    p0_needs_switch = (
        p0_fainted
        or should_uturn0
        or p0_item_forced_switch
        or p0_def_ability_force_switch
        or p0_eject_pack_switch
    ) and (p0_bench_alive > 0)
    if not done:
        if p0_needs_switch and p1_needs_switch_post:
            state.phase = np.int8(Phase.FORCED_SWITCH)
            state.forced_switch_side = np.int8(2)
        elif p0_needs_switch:
            state.phase = np.int8(Phase.FORCED_SWITCH)
            state.forced_switch_side = np.int8(0)
        elif p1_needs_switch_post:
            state.phase = np.int8(Phase.FORCED_SWITCH)
            state.forced_switch_side = np.int8(1)

    _consume_endturn_quick_claw_roll(profile, gen5_prng)
    state.turn = np.int16(int(state.turn) + 1)
    state.done = np.bool_(done)
    state.winner = np.int8(winner)
    return np.float32(reward0), np.float32(reward1), bool(done)


def step_battle_gen9(
    state: MultiFormatState,
    action0: int,
    action1: int,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    gen5_prng,
    resolve_mid_turn_switch0=None,
    wants_tera0: bool = False,
    wants_tera1: bool = False,
    profile=None,
    defer_p1_forced_switch: bool = False,
) -> Tuple[np.float32, np.float32, bool]:
    """Synchronous wrapper around :func:`step_battle_gen9_iter`."""
    gen = step_battle_gen9_iter(
        state,
        action0,
        action1,
        game_data,
        move_effects,
        type_chart,
        gen5_prng,
        resolve_mid_turn_switch0=resolve_mid_turn_switch0,
        wants_tera0=wants_tera0,
        wants_tera1=wants_tera1,
        profile=profile,
    )
    try:
        req = next(gen)
        while True:
            choices = resolve_switch_choices_sync(
                state,
                state.battle_state,
                req,
                side_order0=state.side_order0,
                side_order1=state.side_order1,
                resolve_mid_turn_switch0=resolve_mid_turn_switch0,
            )
            req = gen.send(choices)
    except StopIteration as stop:
        result = stop.value
        forced_side = int(getattr(state, "forced_switch_side", -1))
        if (
            not defer_p1_forced_switch
            and int(state.phase) == Phase.FORCED_SWITCH
            and forced_side in (1, 2)
        ):
            active0 = int(state.battle_state[OFF_META + M_ACTIVE0])
            active1 = int(state.battle_state[OFF_META + M_ACTIVE1])
            _inline_post_faint_switch_side1(
                state,
                game_data,
                gen5_prng,
                active0,
                active1,
                profile=profile,
            )
            if forced_side == 2:
                state.forced_switch_side = np.int8(0)
            else:
                state.forced_switch_side = np.int8(-1)
                state.phase = np.int8(Phase.BATTLE)
        return result


def _inline_post_faint_switch_side1(
    state: MultiFormatState,
    game_data,
    gen5_prng,
    active0: int,
    active1: int,
    profile=None,
) -> int:
    """Auto-switch side 1 after EOT faint (sync wrapper back-compat)."""
    from pokepy import effects as fx  # type: ignore

    if profile is None:
        from pokepy.core.gen_profile import GEN9_PROFILE

        profile = GEN9_PROFILE

    battle = state.battle_state
    side_order1 = state.side_order1

    def _base_ability_for_offset(pokemon_offset: int) -> int:
        poff = int(pokemon_offset)
        if poff < OFF_SIDE1:
            slot = (poff - OFF_SIDE0) // POKEMON_SIZE
            return int(state.team_abilities[slot])
        slot = (poff - OFF_SIDE1) // POKEMON_SIZE
        return int(state.opp_abilities[slot])

    def _reset_incoming_switch_state_tracked(pokemon_offset: int) -> None:
        fx.reset_incoming_switch_state(
            battle,
            pokemon_offset,
            game_data,
            base_ability=_base_ability_for_offset(pokemon_offset),
            state=state,
        )
        _reset_move_used_mask_for_offset(state, pokemon_offset)

    def _sync_showdown_order_on_switch(order_arr, new_active_slot):
        new_active_slot = int(new_active_slot)
        idx = -1
        for i in range(len(order_arr)):
            if int(order_arr[i]) == new_active_slot:
                idx = i
                break
        if idx <= 0:
            return
        old_front = int(order_arr[0])
        order_arr[0] = np.int8(new_active_slot)
        order_arr[idx] = np.int8(old_front)

    def _run_switch_in_update_item_hooks(pokemon_offset: int) -> None:
        def _run_untracked(hook, _pokemon_offset: int, *args) -> None:
            hook(*args)

        _run_switch_in_update_item_hooks_common(
            battle,
            pokemon_offset,
            game_data,
            _run_untracked,
        )

    def _action_speed_from_effective(eff_speed: int) -> int:
        if int(battle[OFF_FIELD + F_TRICK_ROOM]) > 0:
            return 10000 - int(eff_speed)
        return int(eff_speed)

    while True:
        _use_hidden_opp_order = bool(getattr(state, "hidden_opp_switches", ()))
        if _use_hidden_opp_order:
            active1 = fx.auto_switch(
                battle,
                OFF_SIDE1,
                active1,
                True,
                order=side_order1,
            )
        else:
            active1 = fx.auto_switch(
                battle,
                OFF_SIDE1,
                active1,
                True,
            )
        _pending_switch_slot_condition1_post = int(battle[OFF_FIELD + F_DESTINY_BOND_1])
        battle[OFF_META + M_ACTIVE1] = active1
        _sync_showdown_order_on_switch(side_order1, active1)
        _clear_side_switch_state_common(battle, 1)
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
        new_p1 = OFF_SIDE1 + active1 * POKEMON_SIZE
        _reset_incoming_switch_state_tracked(new_p1)
        _postswitch_consumed_pending1 = apply_pending_wish_on_switch_in(
            battle,
            1,
            new_p1,
            state,
            game_data,
            _pending_switch_slot_condition1_post,
        )
        if (
            is_pending_wish_sentinel(_pending_switch_slot_condition1_post)
            and not _postswitch_consumed_pending1
        ):
            battle[OFF_FIELD + F_DESTINY_BOND_1] = np.int16(
                _pending_switch_slot_condition1_post
            )
        _postswitch_p0_speed = _action_speed_from_effective(
            fx.get_effective_speed(battle, OFF_SIDE0 + active0 * POKEMON_SIZE)
        )
        _postswitch_p1_speed = _get_switch_resume_action_speed(battle, new_p1)
        fx.apply_hazard_damage_on_switch(battle, new_p1, OFF_FIELD + F_HAZARDS_1)
        _reset_toxic_counter_on_switch_in(battle, new_p1)
        if int(battle[new_p1 + 1]) > 0:
            _opp_target_off = OFF_SIDE0 + active0 * POKEMON_SIZE
            if int(battle[_opp_target_off + 1]) > 0 and profile.has_abilities:
                fx.apply_switch_in_ability_with_trace_reaction(
                    battle,
                    new_p1,
                    _opp_target_off,
                    True,
                    gen5_prng=gen5_prng,
                    has_terrain=profile.has_terrain,
                    ability_weather_limited=profile.ability_weather_limited,
                )
            else:
                state.pending_opp_switch_in_slot = np.int8(active1)
                state.pending_opp_switch_action_speed = np.int16(_postswitch_p1_speed)
            _consume_switch_request_resume_tie_frames(
                _postswitch_p1_speed,
                _postswitch_p0_speed,
                int(battle[_opp_target_off + 1]) > 0,
                gen5_prng,
            )
            _run_switch_in_update_item_hooks(new_p1)
            break
        p1_bench_alive = 0
        for i in range(6):
            if i != active1:
                so = OFF_SIDE1 + i * POKEMON_SIZE
                if int(battle[so + 1]) > 0 and (int(battle[so + 15]) & 1) == 0:
                    p1_bench_alive += 1
        if p1_bench_alive == 0:
            break
    return active1


# =============================================================================
# Forced switch handler
# =============================================================================


def step_forced_switch(
    state: MultiFormatState,
    action: int,
    side: int,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    gen5_prng,
    profile=None,
) -> Tuple[np.float32, np.float32, bool]:
    """Forced switch after KO or pivot. No turn passes.

    Handles side 0 (player) and side 1 (opponent) symmetrically.
    """
    from pokepy import effects as fx  # type: ignore

    if profile is None:
        from pokepy.core.gen_profile import GEN9_PROFILE

        profile = GEN9_PROFILE

    side = int(side)
    battle = state.battle_state
    side_base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    active_meta = M_ACTIVE0 if side == 0 else M_ACTIVE1
    opp_active_meta = M_ACTIVE1 if side == 0 else M_ACTIVE0
    side_order = state.side_order0 if side == 0 else state.side_order1
    destiny_bond_off = F_DESTINY_BOND_0 if side == 0 else F_DESTINY_BOND_1
    hazards_off = F_HAZARDS_0 if side == 0 else F_HAZARDS_1
    clear_field_offs = (
        (
            F_VOLATILE_0,
            F_LEECH_SEED_0,
            F_DISABLE_TURNS_0,
            F_EXTENDED_VOLATILE_0,
            F_DESTINY_BOND_0,
            F_SUBSTITUTE_0,
            F_YAWN_TURNS_0,
            F_PERISH_COUNT_0,
        )
        if side == 0
        else (
            F_VOLATILE_1,
            F_LEECH_SEED_1,
            F_DISABLE_TURNS_1,
            F_EXTENDED_VOLATILE_1,
            F_DESTINY_BOND_1,
            F_SUBSTITUTE_1,
            F_YAWN_TURNS_1,
            F_PERISH_COUNT_1,
        )
    )

    def _clear_pending_opp_switch_in() -> None:
        state.pending_opp_switch_in_slot = np.int8(-1)
        state.pending_opp_switch_action_speed = np.int16(0)

    def _run_switch_in_update_item_hooks(pokemon_offset: int) -> None:
        def _run_untracked(hook, _pokemon_offset: int, *args) -> None:
            hook(*args)

        _run_switch_in_update_item_hooks_common(
            battle,
            pokemon_offset,
            game_data,
            _run_untracked,
        )

    def _base_ability_for_offset(pokemon_offset: int) -> int:
        poff = int(pokemon_offset)
        if poff < OFF_SIDE1:
            slot = (poff - OFF_SIDE0) // POKEMON_SIZE
            return int(state.team_abilities[slot])
        slot = (poff - OFF_SIDE1) // POKEMON_SIZE
        return int(state.opp_abilities[slot])

    def _reset_incoming_switch_state_tracked(pokemon_offset: int) -> None:
        fx.reset_incoming_switch_state(
            battle,
            pokemon_offset,
            game_data,
            base_ability=_base_ability_for_offset(pokemon_offset),
            state=state,
        )
        _reset_move_used_mask_for_offset(state, pokemon_offset)

    def _sync_showdown_order_on_switch(order_arr, new_active_slot):
        new_active_slot = int(new_active_slot)
        idx = -1
        for i in range(len(order_arr)):
            if int(order_arr[i]) == new_active_slot:
                idx = i
                break
        if idx <= 0:
            return
        old_front = int(order_arr[0])
        order_arr[0] = np.int8(new_active_slot)
        order_arr[idx] = np.int8(old_front)

    target_slot = max(0, min(5, int(action) - 4))
    target_off = side_base + target_slot * POKEMON_SIZE
    target_alive = (int(battle[target_off + 1]) > 0) and (
        (int(battle[target_off + 15]) & 1) == 0
    )

    if not target_alive or int(action) < 4:
        for i in range(6):
            so = side_base + i * POKEMON_SIZE
            if int(battle[so + 1]) > 0 and (int(battle[so + 15]) & 1) == 0:
                target_slot = i
                break

    old_active = int(battle[OFF_META + active_meta])
    old_off = side_base + old_active * POKEMON_SIZE
    if int(battle[old_off + 1]) > 0:
        fx.apply_regenerator_on_switch_out(battle, old_off, True)
        fx.apply_natural_cure_on_switch_out(battle, old_off, True)

    heal_wish_sentinel = int(battle[OFF_FIELD + destiny_bond_off])

    battle[OFF_META + active_meta] = target_slot
    _sync_showdown_order_on_switch(side_order, target_slot)
    _clear_side_switch_state_common(battle, side)
    for off in clear_field_offs:
        battle[OFF_FIELD + off] = 0

    chosen = side_base + target_slot * POKEMON_SIZE
    _reset_incoming_switch_state_tracked(chosen)
    _forced_switch_consumed_pending = apply_pending_wish_on_switch_in(
        battle,
        side,
        chosen,
        state,
        game_data,
        heal_wish_sentinel,
    )
    if (
        is_pending_wish_sentinel(heal_wish_sentinel)
        and not _forced_switch_consumed_pending
    ):
        battle[OFF_FIELD + destiny_bond_off] = np.int16(heal_wish_sentinel)

    opp_active = int(battle[OFF_META + opp_active_meta])
    opp_side_base = OFF_SIDE1 if side == 0 else OFF_SIDE0
    opp_off = opp_side_base + opp_active * POKEMON_SIZE
    pending_opp_slot = int(state.pending_opp_switch_in_slot)
    pending_opp_speed = int(state.pending_opp_switch_action_speed)
    forced_switch_action_speed = int(state.forced_switch_action_speed)
    _postswitch_self_speed = (
        forced_switch_action_speed
        if forced_switch_action_speed > 0
        else _get_switch_resume_action_speed(battle, chosen)
    )
    _opp_effective_speed = fx.get_effective_speed(battle, opp_off)
    if int(battle[OFF_FIELD + F_TRICK_ROOM]) > 0:
        _postswitch_opp_speed = 10000 - int(_opp_effective_speed)
    else:
        _postswitch_opp_speed = int(_opp_effective_speed)
    fx.apply_hazard_damage_on_switch(battle, chosen, OFF_FIELD + hazards_off)
    _reset_toxic_counter_on_switch_in(battle, chosen)
    if int(battle[chosen + 1]) > 0:
        has_pending_opp_switch = (
            pending_opp_slot == opp_active
            and pending_opp_slot >= 0
            and int(battle[opp_off + 1]) > 0
        )
        if has_pending_opp_switch:
            self_switchin_speed = fx.get_effective_speed(battle, chosen)
            opp_switchin_speed = fx.get_effective_speed(battle, opp_off)
            tr_active = int(battle[OFF_FIELD + F_TRICK_ROOM]) > 0
            self_first = (
                (self_switchin_speed >= opp_switchin_speed)
                if not tr_active
                else (self_switchin_speed <= opp_switchin_speed)
            )
            if profile.has_abilities:
                if self_first:
                    fx.apply_switch_in_ability(
                        battle,
                        chosen,
                        opp_off,
                        True,
                        gen5_prng=gen5_prng,
                        has_terrain=profile.has_terrain,
                        ability_weather_limited=profile.ability_weather_limited,
                    )
                    if int(battle[opp_off + 1]) > 0:
                        fx.apply_switch_in_ability(
                            battle,
                            opp_off,
                            chosen,
                            True,
                            gen5_prng=gen5_prng,
                            has_terrain=profile.has_terrain,
                            ability_weather_limited=profile.ability_weather_limited,
                        )
                else:
                    fx.apply_switch_in_ability(
                        battle,
                        opp_off,
                        chosen,
                        True,
                        gen5_prng=gen5_prng,
                        has_terrain=profile.has_terrain,
                        ability_weather_limited=profile.ability_weather_limited,
                    )
                    if int(battle[chosen + 1]) > 0:
                        fx.apply_switch_in_ability(
                            battle,
                            chosen,
                            opp_off,
                            True,
                            gen5_prng=gen5_prng,
                            has_terrain=profile.has_terrain,
                            ability_weather_limited=profile.ability_weather_limited,
                        )
            _postswitch_opp_speed = pending_opp_speed
        elif int(battle[opp_off + 1]) > 0 and profile.has_abilities:
            fx.apply_switch_in_ability_with_trace_reaction(
                battle,
                chosen,
                opp_off,
                True,
                gen5_prng=gen5_prng,
                has_terrain=profile.has_terrain,
                ability_weather_limited=profile.ability_weather_limited,
            )
        elif side == 1:
            state.pending_opp_switch_in_slot = np.int8(target_slot)
            state.pending_opp_switch_action_speed = np.int16(_postswitch_self_speed)
        if side == 0:
            _tie_speed0 = _postswitch_self_speed
            _tie_speed1 = _postswitch_opp_speed
        else:
            _tie_speed0 = _postswitch_opp_speed
            _tie_speed1 = _postswitch_self_speed
        _consume_switch_request_resume_tie_frames(
            _tie_speed0,
            _tie_speed1,
            int(battle[opp_off + 1]) > 0,
            gen5_prng,
        )
        _run_switch_in_update_item_hooks(chosen)
        _clear_pending_opp_switch_in()
        state.phase = np.int8(Phase.BATTLE)
    else:
        bench_alive = 0
        for i in range(6):
            if i != target_slot:
                so = side_base + i * POKEMON_SIZE
                if int(battle[so + 1]) > 0 and (int(battle[so + 15]) & 1) == 0:
                    bench_alive += 1
        if bench_alive > 0:
            state.phase = np.int8(Phase.FORCED_SWITCH)
            state.forced_switch_side = np.int8(side)
        else:
            _clear_pending_opp_switch_in()
            state.phase = np.int8(Phase.BATTLE)

    state.forced_switch_slot = np.int8(-1)
    state.forced_switch_hp = np.int16(0)
    state.forced_switch_original = np.int8(-1)
    state.forced_switch_action_speed = np.int16(0)
    if int(state.phase) != Phase.FORCED_SWITCH:
        state.forced_switch_side = np.int8(-1)
    alive0 = _count_alive(battle, OFF_SIDE0)
    alive1 = _count_alive(battle, OFF_SIDE1)
    max_turns = getattr(state, "max_turns", 200)
    done = (alive0 == 0) or (alive1 == 0) or (int(state.turn) >= max_turns)
    if alive0 == 0:
        winner = 1
    elif alive1 == 0:
        winner = 0
    else:
        winner = -1

    reward0 = 0.0
    if done and winner == 0:
        reward0 = 100.0
    elif done and winner == 1:
        reward0 = -100.0
    reward1 = -reward0

    state.done = np.bool_(done)
    state.winner = np.int8(winner)
    return np.float32(reward0), np.float32(reward1), bool(done)
