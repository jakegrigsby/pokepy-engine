"""Volatile-status effects (leech seed, substitute, perish song, taunt, encore, ...).

Real ports of the Showdown reference implementation methods. All functions
mutate `battle: np.ndarray` in place. Functions that need RNG accept a stateful
`Gen5PRNG`. The the Showdown reference versions are batched array-style — these are scalar
numpy ports using ordinary if/else, since pokepy v0 runs one battle at a time.

Source line ranges (multi_format_fast_env.py):
- _apply_leech_seed_damage:        9411
- _apply_leech_seed_from_move:     9472
- _apply_substitute_from_move:     9518
- _apply_damage_to_substitute:     9571
- _apply_perish_song_from_move:    9602
- _apply_destiny_bond_from_move:   9625
- _apply_lock_on_from_move:        9640
- _apply_ghost_curse_from_move:    9654
- _apply_pain_split_from_move:     9687
- _apply_confusion_from_move:     10234
- _apply_taunt_from_move:         10289
- _apply_encore_from_move:        10333
- _apply_phazing_from_move:       10390
- _apply_extended_volatile:       10595
- _check_confusion_self_hit:      10705
- _decrement_confusion:           10758
- _decrement_taunt_encore:        10778
- _process_perish_song:           10804
- _apply_curse_damage:            10838
- _apply_salt_cure_damage:         8824
"""
from __future__ import annotations

from pokepy.effects._common import np, MultiFormatState, Gen5PRNG
from pokepy.effects.grounding import is_grounded
from pokepy.core.bitpack import (
    get_confusion_turns,
    set_confusion_turns,
    get_confusion_newly_applied,
    set_confusion_newly_applied,
    get_taunt_turns,
    set_taunt_turns,
    get_encore_turns,
    set_encore_turns,
    get_heal_block_turns,
    set_heal_block_turns,
    get_throat_chop_turns,
    set_throat_chop_turns,
)
from pokepy.core.constants import (
    OFF_FIELD,
    OFF_META,
    OFF_SIDE0,
    OFF_SIDE1,
    POKEMON_SIZE,
    M_ACTIVE0,
    M_ACTIVE1,
    F_LEECH_SEED_0,
    F_LEECH_SEED_1,
    F_SUBSTITUTE_0,
    F_SUBSTITUTE_1,
    F_PERISH_COUNT_0,
    F_PERISH_COUNT_1,
    F_DESTINY_BOND_0,
    F_DESTINY_BOND_1,
    F_VOLATILE_0,
    F_VOLATILE_1,
    F_EXTENDED_VOLATILE_0,
    F_EXTENDED_VOLATILE_1,
    F_LAST_MOVE_0,
    F_LAST_MOVE_1,
    F_CHOICE_LOCK_0,
    F_CHOICE_LOCK_1,
    OFF_MOVES,
    M_PARTIAL_TRAP_TURNS_0,
    M_PARTIAL_TRAP_TURNS_1,
    NEUTRAL_BOOSTS_13,
    NEUTRAL_BOOSTS_14,
    EFFECT_LEECH_SEED,
    EFFECT_SUBSTITUTE,
    EXT_VOL_FOCUS_ENERGY,
    EXT_VOL_TORMENT,
    EXT_VOL_ATTRACT,
    EXT_VOL_YAWN,
    EXT_VOL_EMBARGO,
    EXT_VOL_HEAL_BLOCK,
    EXT_VOL_IMPRISON,
    EXT_VOL_INGRAIN,
    EXT_VOL_AQUA_RING,
    EXT_VOL_CURSE,
    EXT_VOL_MEAN_LOOK,
    EXT_VOL_LOCK_ON,
    EXT_VOL_PARTIAL_TRAP,
    EXT_VOL_SALT_CURE,
    EXT_VOL_FORESIGHT,
    VOLATILE_LEECH_SEED,
    VOLATILE_CONFUSION,
    VOLATILE_TAUNT,
    VOLATILE_ENCORE,
    VOLATILE_FOCUS_ENERGY,
    VOLATILE_TORMENT,
    VOLATILE_ATTRACT,
    VOLATILE_YAWN,
    VOLATILE_EMBARGO,
    VOLATILE_HEAL_BLOCK,
    VOLATILE_IMPRISON,
    VOLATILE_INGRAIN,
    VOLATILE_AQUA_RING,
    VOLATILE_CURSE,
    VOLATILE_MEAN_LOOK,
    VOLATILE_LOCK_ON,
    VOLATILE_PARTIAL_TRAP,
    VOLATILE_SALT_CURE,
    VOLATILE_FORESIGHT,
    MOVE_PERISH_SONG,
    MOVE_DESTINY_BOND,
    MOVE_LOCK_ON,
    MOVE_MIND_READER,
    MOVE_CURSE,
    MOVE_PAIN_SPLIT,
    MOVE_ROAR,
    MOVE_WHIRLWIND,
    MOVE_DRAGON_TAIL,
    MOVE_CIRCLE_THROW,
    MOVE_STRUGGLE,
    MOVE_ENCORE,
    MOVE_MIMIC,
    MOVE_TRANSFORM,
    MOVE_SKETCH,
    TYPE_GRASS,
    TYPE_GHOST,
    TYPE_WATER,
    TYPE_STEEL,
    GENDER_MALE,
    GENDER_FEMALE,
    ABILITY_MAGIC_GUARD,
    ABILITY_SUCTION_CUPS,
    ABILITY_GUARD_DOG,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _side_field(side: int, off_a: int, off_b: int) -> int:
    """Pick OFF_FIELD-relative offset based on side (0/1)."""
    return OFF_FIELD + (off_a if int(side) == 0 else off_b)

def _active_offset(battle: np.ndarray, side: int) -> int:
    if int(side) == 0:
        active = int(battle[OFF_META + M_ACTIVE0])
        return OFF_SIDE0 + active * POKEMON_SIZE
    else:
        active = int(battle[OFF_META + M_ACTIVE1])
        return OFF_SIDE1 + active * POKEMON_SIZE

def _to_int16(value: int) -> int:
    """Wrap an unsigned 16-bit value into the signed int16 range."""
    value = int(value) & 0xFFFF
    if value >= 0x8000:
        value -= 0x10000
    return value

# -----------------------------------------------------------------------------
# Leech Seed
# -----------------------------------------------------------------------------

def apply_leech_seed_damage(
    battle: np.ndarray,
    pokemon0_offset: int,
    pokemon1_offset: int,
    game_data=None,
) -> None:
    """Port of _apply_leech_seed_damage (line ~9411).

    Drains 1/8 max HP from each seeded Pokemon and heals the opposing active.
    Magic Guard prevents the drain. Mutates `battle` in place.
    """
    p0 = int(pokemon0_offset)
    p1 = int(pokemon1_offset)
    ITEM_BIG_ROOT = 29

    player_seeded = int(battle[OFF_FIELD + F_LEECH_SEED_0]) > 0
    opp_seeded = int(battle[OFF_FIELD + F_LEECH_SEED_1]) > 0

    hp0 = int(battle[p0 + 1])
    max_hp0 = int(battle[p0 + 2])
    has_magic_guard0 = int(battle[p0 + 5]) == ABILITY_MAGIC_GUARD
    item0 = int(battle[p0 + 6])
    ext_vol0 = int(battle[OFF_FIELD + F_EXTENDED_VOLATILE_0]) & 0xFFFF

    hp1 = int(battle[p1 + 1])
    max_hp1 = int(battle[p1 + 2])
    has_magic_guard1 = int(battle[p1 + 5]) == ABILITY_MAGIC_GUARD
    item1 = int(battle[p1 + 6])
    ext_vol1 = int(battle[OFF_FIELD + F_EXTENDED_VOLATILE_1]) & 0xFFFF

    # Heal Block on the seed owner (the one who would heal) blocks the heal
    # but the drain still happens. Showdown: healblock condition.onTryHeal.
    from pokepy.core.constants import EXT_VOL_HEAL_BLOCK
    heal_block0 = (ext_vol0 & EXT_VOL_HEAL_BLOCK) != 0
    heal_block1 = (ext_vol1 & EXT_VOL_HEAL_BLOCK) != 0

    drain0 = max(1, min(hp0, max_hp0 // 8))
    drain1 = max(1, min(hp1, max_hp1 // 8))

    def _boost_heal(amount: int, item: int) -> int:
        # Big Root: 1.3x heal (Showdown items.ts bigroot onTryHealPriority: 1)
        if item == ITEM_BIG_ROOT:
            return max(1, (amount * 5325) // 4096)
        return amount

    # Player (side 0) is seeded by opponent → drain p0, heal p1.
    # Showdown: leechseed.onResidual short-circuits with "Nothing to leech
    # into" when the source mon (the seeder, the opposing side's active in
    # 1v1) is fainted/hp<=0 — no drain happens. The seeder for side 0 is
    # side 1's active mon (p1).
    should_drain0 = (player_seeded and (hp0 > 0)
                     and (hp1 > 0) and (not has_magic_guard0))
    if should_drain0:
        hp0 = hp0 - drain0
        if hp1 > 0 and not heal_block1:
            hp1 = min(max_hp1, hp1 + _boost_heal(drain0, item1))

    # Opponent (side 1) is seeded by player → drain p1, heal p0. Same
    # short-circuit: skip if the seeder (side 0's active = p0) is fainted.
    should_drain1 = (opp_seeded and (hp1 > 0)
                     and (hp0 > 0) and (not has_magic_guard1))
    if should_drain1:
        hp1 = hp1 - drain1
        if hp0 > 0 and not heal_block0:
            hp0 = min(max_hp0, hp0 + _boost_heal(drain1, item0))

    battle[p0 + 1] = max(0, hp0)
    battle[p1 + 1] = max(0, hp1)

def apply_leech_seed_from_move(
    battle: np.ndarray,
    move_id: int,
    target_side: int,
    target_offset: int,
    hit: bool,
    game_data=None,
    move_effects=None,
) -> None:
    """Port of _apply_leech_seed_from_move (line ~9472).

    Sets the F_LEECH_SEED flag on `target_side` if the move applies leech seed
    and the target isn't Grass-typed.
    """
    if not bool(hit) or move_effects is None:
        return
    move_id = int(move_id)
    target_offset = int(target_offset)

    move_effect = int(move_effects.effect_type[move_id])
    is_leech_seed = move_effect == EFFECT_LEECH_SEED

    volatile_type = int(move_effects.volatile[move_id])
    is_leech_seed = is_leech_seed or (volatile_type == VOLATILE_LEECH_SEED)
    if not is_leech_seed:
        return

    target_types = int(battle[target_offset + 4]) & 0xFFFF
    target_type1 = target_types & 0xFF
    target_type2 = (target_types >> 8) & 0xFF
    is_grass = (target_type1 == TYPE_GRASS) or (target_type2 == TYPE_GRASS)
    if is_grass:
        return

    target_hp = int(battle[target_offset + 1])
    if target_hp <= 0:
        return

    # Substitute on the target blocks status moves like leech seed (gen 6+).
    target_sub_off = _side_field(target_side, F_SUBSTITUTE_0, F_SUBSTITUTE_1)
    if int(battle[target_sub_off]) > 0:
        return

    leech_offset = _side_field(target_side, F_LEECH_SEED_0, F_LEECH_SEED_1)
    if int(battle[leech_offset]) == 0:
        battle[leech_offset] = 1

# -----------------------------------------------------------------------------
# Substitute
# -----------------------------------------------------------------------------

def apply_substitute_from_move(
    battle: np.ndarray,
    move_id: int,
    user_side: int,
    user_offset: int,
    game_data=None,
    move_effects=None,
) -> None:
    """Port of _apply_substitute_from_move (line ~9518).

    Pays 1/4 max HP and creates a substitute. Fails if HP <= 1/4 max or
    a substitute already exists.
    """
    if move_effects is None:
        return
    move_id = int(move_id)
    user_offset = int(user_offset)

    move_effect = int(move_effects.effect_type[move_id])
    if move_effect != EFFECT_SUBSTITUTE:
        return

    current_hp = int(battle[user_offset + 1])
    max_hp = int(battle[user_offset + 2])
    sub_cost = max(1, max_hp // 4)

    sub_offset = _side_field(user_side, F_SUBSTITUTE_0, F_SUBSTITUTE_1)
    current_sub = int(battle[sub_offset])

    has_enough_hp = current_hp > sub_cost
    no_sub = current_sub == 0
    if not (has_enough_hp and no_sub):
        return

    battle[user_offset + 1] = current_hp - sub_cost
    battle[sub_offset] = sub_cost

def apply_damage_to_substitute(
    battle: np.ndarray,
    target_side: int,
    damage: int,
    hit: bool,
) -> int:
    """Port of _apply_damage_to_substitute (line ~9571).

    If the target has a substitute and was hit, soak damage into the sub
    (capped at sub HP) and return 0 damage through. Otherwise return `damage`
    unchanged. Mutates `battle` in place.

    Returns the damage that gets through to the Pokemon (after sub absorbs).
    """
    sub_offset = _side_field(target_side, F_SUBSTITUTE_0, F_SUBSTITUTE_1)
    sub_hp = int(battle[sub_offset])

    has_sub = sub_hp > 0
    if has_sub and bool(hit):
        damage_to_sub = min(int(damage), sub_hp)
        battle[sub_offset] = sub_hp - damage_to_sub
        return 0
    return int(damage)

# -----------------------------------------------------------------------------
# Perish Song / Destiny Bond / Lock-On
# -----------------------------------------------------------------------------

def apply_perish_song_from_move(
    battle: np.ndarray,
    move_id: int,
    hit: bool,
    move_effects=None,
    user_side: int = -1,
) -> None:
    """Port of _apply_perish_song_from_move (line ~9602).

    Sets perish counter to 4 on both sides if not already set. Soundproof
    makes the affected mon immune UNLESS it is the user of the Perish
    Song (Showdown abilities.ts soundproof onTryHit: `target !== source`).
    """
    if not bool(hit):
        return
    if int(move_id) != MOVE_PERISH_SONG:
        return

    # Soundproof: perish song is a sound move (Showdown data/moves.ts
    # perishsong flags: { sound: 1 }). In Gen 8+ Soundproof only blocks
    # sound moves from OTHERS (`target !== source`); the user's own
    # Perish Song still affects itself. `user_side` is the side that
    # actually used the move; pass -1 when caller doesn't know.
    ABILITY_SOUNDPROOF = 43
    p0_active_off = _active_offset(battle, 0)
    p1_active_off = _active_offset(battle, 1)
    p0_soundproof = int(battle[p0_active_off + 5]) == ABILITY_SOUNDPROOF
    p1_soundproof = int(battle[p1_active_off + 5]) == ABILITY_SOUNDPROOF

    perish0 = int(battle[OFF_FIELD + F_PERISH_COUNT_0])
    perish1 = int(battle[OFF_FIELD + F_PERISH_COUNT_1])
    # Block only if Soundproof AND not the user themselves.
    block0 = p0_soundproof and (int(user_side) != 0)
    block1 = p1_soundproof and (int(user_side) != 1)
    if perish0 == 0 and not block0:
        battle[OFF_FIELD + F_PERISH_COUNT_0] = 4
    if perish1 == 0 and not block1:
        battle[OFF_FIELD + F_PERISH_COUNT_1] = 4

def apply_destiny_bond_from_move(
    battle: np.ndarray,
    move_id: int,
    user_side: int,
    move_effects=None,
) -> None:
    """Port of _apply_destiny_bond_from_move (line ~9625).

    Gen 7+ Destiny Bond onPrepareHit (Showdown data/moves.ts:destinybond):
      return !pokemon.removeVolatile('destinybond');
    In words: if the user already had the DB volatile set (from last turn),
    remove it and FAIL the current use — two DBs in a row cancel out. Only
    set the volatile on first use.
    """
    if int(move_id) != MOVE_DESTINY_BOND:
        return
    destiny_offset = _side_field(user_side, F_DESTINY_BOND_0, F_DESTINY_BOND_1)
    cur = int(battle[destiny_offset])
    if cur > 0:
        # Consecutive use — fail and clear. User is now vulnerable.
        battle[destiny_offset] = 0
        return
    battle[destiny_offset] = 1

def apply_lock_on_from_move(
    battle: np.ndarray,
    move_id: int,
    user_side: int,
    hit: bool,
    move_effects=None,
) -> None:
    """Port of _apply_lock_on_from_move (line ~9640).

    Sets EXT_VOL_LOCK_ON on the user.
    """
    if not bool(hit):
        return
    mid = int(move_id)
    if mid != MOVE_LOCK_ON and mid != MOVE_MIND_READER:
        return

    ext_vol_offset = _side_field(user_side, F_EXTENDED_VOLATILE_0, F_EXTENDED_VOLATILE_1)
    cur = int(battle[ext_vol_offset]) & 0xFFFF
    battle[ext_vol_offset] = _to_int16(cur | EXT_VOL_LOCK_ON)

# -----------------------------------------------------------------------------
# Ghost Curse / Pain Split
# -----------------------------------------------------------------------------

def apply_ghost_curse_from_move(
    battle: np.ndarray,
    move_id: int,
    user_offset: int,
    target_side: int,
    hit: bool,
    game_data=None,
    move_effects=None,
) -> None:
    """Port of _apply_ghost_curse_from_move (line ~9654).

    Ghost-type Curse: user pays 50% max HP, target gets EXT_VOL_CURSE.
    """
    if not bool(hit):
        return
    if int(move_id) != MOVE_CURSE:
        return

    user_offset = int(user_offset)
    user_types = int(battle[user_offset + 4]) & 0xFFFF
    user_type1 = user_types & 0xFF
    user_type2 = (user_types >> 8) & 0xFF
    user_is_ghost = (user_type1 == TYPE_GHOST) or (user_type2 == TYPE_GHOST)
    if not user_is_ghost:
        return

    max_hp = int(battle[user_offset + 2])
    current_hp = int(battle[user_offset + 1])
    hp_cost = max(1, max_hp // 2)
    battle[user_offset + 1] = max(0, current_hp - hp_cost)

    ext_vol_offset = _side_field(target_side, F_EXTENDED_VOLATILE_0, F_EXTENDED_VOLATILE_1)
    cur = int(battle[ext_vol_offset]) & 0xFFFF
    battle[ext_vol_offset] = _to_int16(cur | EXT_VOL_CURSE)

def apply_pain_split_from_move(
    battle: np.ndarray,
    move_id: int,
    user_offset: int,
    target_offset: int,
    hit: bool,
    game_data=None,
    move_effects=None,
) -> None:
    """Port of _apply_pain_split_from_move (line ~9687).

    Averages HP between user and target (capped at each Pokemon's max HP).
    """
    if not bool(hit):
        return
    if int(move_id) != MOVE_PAIN_SPLIT:
        return

    user_offset = int(user_offset)
    target_offset = int(target_offset)

    user_hp = int(battle[user_offset + 1])
    target_hp = int(battle[target_offset + 1])
    avg_hp = (user_hp + target_hp) // 2

    user_max = int(battle[user_offset + 2])
    target_max = int(battle[target_offset + 2])
    battle[user_offset + 1] = min(avg_hp, user_max)
    battle[target_offset + 1] = min(avg_hp, target_max)

# -----------------------------------------------------------------------------
# Confusion / Taunt / Encore
# -----------------------------------------------------------------------------

def apply_confusion_volatile(
    battle: np.ndarray,
    target_side: int,
    gen5_prng: Gen5PRNG,
    prerolled_duration: "int | None" = None,
) -> bool:
    """Apply a plain confusion volatile with Showdown's addVolatile timing.

    This is used for non-move sources such as Poison Puppeteer, which do not
    spend a secondary chance roll but still use the normal confusion start
    checks and duration roll.
    """
    from pokepy.core.constants import (
        ABILITY_OWN_TEMPO, TERRAIN_MISTY, F_TERRAIN,
    )

    target_offset = _active_offset(battle, int(target_side))
    target_ability = int(battle[target_offset + 5])
    if target_ability == ABILITY_OWN_TEMPO:
        return False

    if is_grounded(battle, target_offset) and int(battle[OFF_FIELD + F_TERRAIN]) == TERRAIN_MISTY:
        return False

    volatile_offset = _side_field(target_side, F_VOLATILE_0, F_VOLATILE_1)
    current_volatile = int(battle[volatile_offset])
    current_turns = get_confusion_turns(current_volatile)
    if current_turns != 0:
        return False

    if prerolled_duration is not None:
        confusion_turns = int(prerolled_duration)
    else:
        confusion_turns = int(gen5_prng.random(2, 6))

    battle[volatile_offset] = _to_int16(set_confusion_turns(current_volatile, confusion_turns))
    return True

def apply_confusion_from_move(
    battle: np.ndarray,
    move_id: int,
    target_side: int,
    hit: bool,
    game_data=None,
    move_effects=None,
    gen5_prng: Gen5PRNG = None,
    prerolled_roll: "int | None" = None,
    prerolled_duration: "int | None" = None,
    prerolled_can_apply: "bool | None" = None,
    target_stats_raised_this_turn: bool = False,
) -> None:
    """Port of _apply_confusion_from_move (line ~10234).

    Rolls confusion via Gen5 PRNG; lasts 2-5 turns. Only advances PRNG when
    move actually has a confusion effect (matches Showdown).
    """
    if move_effects is None or gen5_prng is None:
        return
    move_id = int(move_id)
    volatile_type = int(move_effects.volatile[move_id])
    volatile_chance = int(move_effects.volatile_chance[move_id])

    is_confuse_move = volatile_type == VOLATILE_CONFUSION
    may_confuse = is_confuse_move and bool(hit) and (volatile_chance > 0)
    if not may_confuse:
        return

    # Sheer Force suppresses ALL secondary confusion (chance < 100). The
    # attacker is on the opposite side from the target. Source:
    # data/abilities.ts:4122 sheerforce onModifyMove deletes move.secondaries.
    if volatile_chance < 100:
        atk_offset = _active_offset(battle, 1 - int(target_side))
        from pokepy.core.constants import ABILITY_SHEER_FORCE as _ABILITY_SHEER_FORCE_VC
        if int(battle[atk_offset + 5]) == _ABILITY_SHEER_FORCE_VC:
            return

    # Shield Dust (data/abilities.ts:shielddust) filters out non-self
    # secondaries BEFORE the secondaryRoll — target-side Shield Dust blocks
    # confusion secondaries with NO PRNG consumption. Covert Cloak is
    # already gated at the engine call site. Confuse Ray / Supersonic /
    # Sweet Kiss / Teeter Dance / Swagger / Flatter are PRIMARY confuse
    # moves (effect_type != EFFECT_DAMAGE), while Dynamic Punch (100%),
    # Psybeam (10%), Water Pulse (20%), Hurricane (30%), Strange Steam (20%)
    # etc. are damaging moves with secondary.volatileStatus confusion —
    # Shield Dust blocks the latter at any chance including 100%.
    from pokepy.data.move_effects import (
        EFFECT_DAMAGE as _EFFECT_DAMAGE_VC,
        EFFECT_MULTI_HIT as _EFFECT_MULTI_HIT_VC,
    )
    _move_effect_vc = int(move_effects.effect_type[move_id])
    _is_damaging_vc = _move_effect_vc in (_EFFECT_DAMAGE_VC, _EFFECT_MULTI_HIT_VC)
    if _is_damaging_vc:
        _ABILITY_SHIELD_DUST_VC = 19
        target_offset_sd = _active_offset(battle, int(target_side))
        if int(battle[target_offset_sd + 5]) == _ABILITY_SHIELD_DUST_VC:
            return

    if prerolled_roll is None:
        roll = gen5_prng.random(100)
    else:
        roll = int(prerolled_roll)
    if roll >= volatile_chance:
        return

    # Showdown's Alluring Voice always burns secondaryRoll, but the callback
    # only adds confusion if the target gained a positive stat stage earlier
    # in the same turn.
    _MOVE_ALLURING_VOICE = 914
    if move_id == _MOVE_ALLURING_VOICE and not target_stats_raised_this_turn:
        return

    # When the engine already evaluated addVolatile's hit-time gates during
    # preroll (Own Tempo, Misty Terrain, already confused, target will faint),
    # preserve that result here instead of re-checking after later same-turn
    # state changes. This prevents expired confusion from being re-applied at
    # end of turn after Showdown already rejected it at hit time.
    if prerolled_can_apply is False:
        return

    apply_confusion_volatile(
        battle,
        int(target_side),
        gen5_prng,
        prerolled_duration=prerolled_duration,
    )

def apply_taunt_from_move(
    battle: np.ndarray,
    move_id: int,
    target_side: int,
    hit: bool,
    move_effects=None,
    gen5_prng: Gen5PRNG = None,
    prerolled_roll: "int | None" = None,
) -> None:
    """Port of _apply_taunt_from_move (line ~10289).

    Taunt lasts 3 turns. Only advances PRNG if the move can taunt.
    """
    if move_effects is None or gen5_prng is None:
        return
    move_id = int(move_id)
    volatile_type = int(move_effects.volatile[move_id])
    volatile_chance = int(move_effects.volatile_chance[move_id])

    is_taunt_move = volatile_type == VOLATILE_TAUNT
    may_taunt = is_taunt_move and bool(hit) and (volatile_chance > 0)
    if not may_taunt:
        return

    # Showdown only consumes a `secondaryRoll` random(100) frame for
    # secondaries on damaging moves (sim/battle-actions.ts:1357). Taunt is
    # encoded as a STATUS move with `volatileStatus: 'taunt'` at the move
    # level — that's a PRIMARY effect, not a secondary, and Showdown's
    # moveHit path takes it without rolling random(100). Pokepy's
    # `_preroll_move_secondaries` already short-circuits for status moves
    # (no preroll), so primary taunt arrives here with `prerolled_roll
    # is None`. We must NOT roll random(100) in that case — doing so
    # causes a 1-frame PRNG drift on every status-move Taunt use, and this
    # function is called twice per turn (early path + late path), so the
    # drift would be 2 frames per Taunt. Only roll when we have a
    # prerolled value (i.e. the rare case of a taunt secondary on a
    # damaging move).
    if prerolled_roll is None:
        roll = 0  # short-circuit: chance==100 for primary taunt
    else:
        roll = int(prerolled_roll)
    if roll >= volatile_chance:
        return

    # Oblivious blocks Taunt (+ Attract / Captivate). Showdown: abilities.ts
    # oblivious onTryHit -> return null when move is taunt.
    from pokepy.core.constants import ABILITY_OBLIVIOUS
    target_offset_t = _active_offset(battle, int(target_side))
    if int(battle[target_offset_t + 5]) == ABILITY_OBLIVIOUS:
        return
    # Aroma Veil blocks Attract / Disable / Encore / Heal Block / Taunt /
    # Torment. Showdown data/abilities.ts aromaveil onAllyTryAddVolatile.
    _ABILITY_AROMA_VEIL = 165
    if int(battle[target_offset_t + 5]) == _ABILITY_AROMA_VEIL:
        return

    volatile_offset = _side_field(target_side, F_VOLATILE_0, F_VOLATILE_1)
    current_volatile = int(battle[volatile_offset])
    current_turns = get_taunt_turns(current_volatile)
    if current_turns != 0:
        return

    battle[volatile_offset] = _to_int16(set_taunt_turns(current_volatile, 3))

def apply_encore_from_move(
    battle: np.ndarray,
    move_id: int,
    target_side: int,
    hit: bool,
    move_effects=None,
    gen5_prng: Gen5PRNG = None,
    prerolled_roll: "int | None" = None,
) -> None:
    """Port of _apply_encore_from_move (line ~10333).

    Encore lasts 3 turns and only succeeds when target has a valid last move.
    Only advances PRNG if the move can encore.
    """
    if move_effects is None or gen5_prng is None:
        return
    move_id = int(move_id)
    volatile_type = int(move_effects.volatile[move_id])
    volatile_chance = int(move_effects.volatile_chance[move_id])

    is_encore_move = volatile_type == VOLATILE_ENCORE
    may_encore = is_encore_move and bool(hit) and (volatile_chance > 0)
    if not may_encore:
        return

    # Showdown only consumes a `secondaryRoll` random(100) frame for
    # secondaries on damaging moves (sim/battle-actions.ts:1357). Encore is
    # encoded as a STATUS move with `volatileStatus: 'encore'` at the move
    # level — that's a PRIMARY effect, not a secondary, and Showdown's
    # moveHit path takes it without rolling random(100). Pokepy's
    # `_preroll_move_secondaries` already short-circuits for status moves
    # (no preroll), so primary encore arrives here with `prerolled_roll
    # is None`. We must NOT roll random(100) in that case — doing so
    # causes a 1-frame PRNG drift on every status-move Encore use,
    # breaking parity. Only roll when we have a prerolled value (i.e. the
    # rare case of an encore secondary on a damaging move).
    if prerolled_roll is None:
        roll = 0  # short-circuit: chance==100 for primary encore
    else:
        roll = int(prerolled_roll)
    if roll >= volatile_chance:
        return

    last_move_offset = _side_field(target_side, F_LAST_MOVE_0, F_LAST_MOVE_1)
    last_move = int(battle[last_move_offset])
    if last_move < 0:
        return

    # Aroma Veil blocks Encore. Showdown data/abilities.ts aromaveil
    # onAllyTryAddVolatile (encore is in the blocklist).
    _ABILITY_AROMA_VEIL_E = 165
    target_offset_e = _active_offset(battle, int(target_side))
    if int(battle[target_offset_e + 5]) == _ABILITY_AROMA_VEIL_E:
        return

    # Showdown: data/moves.ts encore condition.onStart fails when the last
    # move has `failencore: 1` (Struggle / Encore / Mimic / Sketch /
    # Transform / Mirror Move / Assist / Copycat / Metronome / Nature Power /
    # Me First / Sleep Talk / etc). Pokepy only has IDs for the most common
    # ones — gate those explicitly and let the others fall through.
    uncancellable = (
        MOVE_STRUGGLE, MOVE_ENCORE, MOVE_MIMIC, MOVE_SKETCH, MOVE_TRANSFORM,
    )
    if last_move in uncancellable:
        return

    volatile_offset = _side_field(target_side, F_VOLATILE_0, F_VOLATILE_1)
    current_volatile = int(battle[volatile_offset])
    current_turns = get_encore_turns(current_volatile)
    if current_turns != 0:
        return

    battle[volatile_offset] = _to_int16(set_encore_turns(current_volatile, 3))

def apply_throat_chop_from_move(
    battle: np.ndarray,
    move_id: int,
    target_side: int,
    hit: bool,
    prerolled_roll: "int | None" = None,
) -> bool:
    """Apply Throat Chop's sound-lock volatile when its guaranteed secondary lands.

    Showdown adds a 2-turn `throatchop` volatile, but the effect only needs to
    survive through the next choice / move cycle. Pokepy stores that as a
    2-step countdown in the shared volatile word:
    - `2` on the hit turn, so end-of-turn DisableMove processing can still see
      the fresh application
    - `1` on the following turn, when sound moves remain disabled
    - `0` after that turn's end-of-turn cleanup
    """
    _MOVE_THROAT_CHOP = 675
    if int(move_id) != _MOVE_THROAT_CHOP or not bool(hit):
        return False
    if prerolled_roll is None or int(prerolled_roll) >= 100:
        return False

    target_offset = _active_offset(battle, int(target_side))
    if int(battle[target_offset + 1]) <= 0:
        return False

    volatile_offset = _side_field(target_side, F_VOLATILE_0, F_VOLATILE_1)
    current_volatile = int(battle[volatile_offset])
    if get_throat_chop_turns(current_volatile) > 0:
        return False

    battle[volatile_offset] = _to_int16(set_throat_chop_turns(current_volatile, 2))
    return True

# -----------------------------------------------------------------------------
# Phazing (Roar / Whirlwind / Dragon Tail / Circle Throw)
# -----------------------------------------------------------------------------

def apply_phazing_from_move(
    battle: np.ndarray,
    move_id: int,
    target_side: int,
    user_offset: int,
    hit: bool,
    order: np.ndarray | None = None,
    game_data=None,
    move_effects=None,
    gen5_prng: Gen5PRNG = None,
) -> int:
    """Port of _apply_phazing_from_move (line ~10390).

    Forces target side to switch to a random non-fainted slot. Resets boosts
    and clears volatiles on the new active. Returns the new active slot index
    (or current active if no switch occurred).

    NOTE: Showdown also re-applies hazard damage and switch-in abilities here.
    The pokepy port leaves those to the engine pipeline (the corresponding
    helpers are still stubs in `pokepy.effects.hazards/abilities`), so this
    function only handles the slot swap + boost/volatile reset.
    """
    is_side0 = int(target_side) == 0
    active_meta = OFF_META + (M_ACTIVE0 if is_side0 else M_ACTIVE1)
    current_active = int(battle[active_meta])

    mid = int(move_id)
    is_phazing = mid in (MOVE_ROAR, MOVE_WHIRLWIND, MOVE_DRAGON_TAIL, MOVE_CIRCLE_THROW)
    if not (is_phazing and bool(hit)):
        return current_active

    # DragOut blockers: Suction Cups / Guard Dog abilities, Ingrain volatile.
    # Showdown: abilities.ts suctioncups/guarddog onDragOut, moves.ts ingrain
    # condition.onDragOut -> return null.
    target_active_off = OFF_SIDE0 + current_active * POKEMON_SIZE if is_side0 \
        else OFF_SIDE1 + current_active * POKEMON_SIZE
    # Dragon Tail / Circle Throw only drag a replacement when the target
    # survives the hit. If the target fainted to the move's damage, Showdown
    # skips the phazing effect and lets the normal faint replacement happen.
    if int(battle[target_active_off + 1]) <= 0:
        return current_active
    target_ability = int(battle[target_active_off + 5])
    if target_ability in (ABILITY_SUCTION_CUPS, ABILITY_GUARD_DOG):
        return current_active

    # Soundproof blocks sound moves entirely (Showdown: abilities.ts soundproof
    # onTryHit). For phazing this means Roar fails against Soundproof targets
    # (Whirlwind / Dragon Tail / Circle Throw are not sound moves so they
    # still work). The damage path already gates Soundproof for damaging sound
    # moves but status sound moves like Roar route through here.
    from pokepy.core.constants import ABILITY_SOUNDPROOF as _ABILITY_SOUNDPROOF, FLAG_SOUND as _FLAG_SOUND
    if target_ability == _ABILITY_SOUNDPROOF and game_data is not None:
        flags = int(game_data.move_flags[mid])
        if (flags & _FLAG_SOUND) != 0:
            return current_active
    target_ext_vol = int(battle[OFF_FIELD + (
        F_EXTENDED_VOLATILE_0 if is_side0 else F_EXTENDED_VOLATILE_1
    )]) & 0xFFFF
    if (target_ext_vol & EXT_VOL_INGRAIN) != 0:
        return current_active

    target_base = OFF_SIDE0 if is_side0 else OFF_SIDE1

    order_arr = order if order is not None else np.arange(6, dtype=np.int8)

    # Showdown phazing uses Battle.getRandomSwitchable(), which samples
    # uniformly from switchable bench mons in the side's live team order.
    available = []
    for i in range(1, len(order_arr)):
        slot = int(order_arr[i])
        off = target_base + slot * POKEMON_SIZE
        hp = int(battle[off + 1])
        flags = int(battle[off + 15])
        if hp > 0 and ((flags & 0x1) == 0):
            available.append(slot)

    if not available:
        return current_active
    # Showdown still routes single-candidate drag-outs through random(1),
    # so consume the frame whenever a PRNG is available instead of
    # short-circuiting len(available) == 1.
    if gen5_prng is None:
        new_active = int(available[0])
    else:
        new_active = int(available[int(gen5_prng.random(len(available)))])

    # Dragging a target out is still a real switch-out for the outgoing active.
    # Showdown runs switch-out ability hooks such as Regenerator / Natural Cure
    # on the phazed target before the replacement finishes entering.
    from pokepy.effects.abilities import (
        apply_natural_cure_on_switch_out as _apply_natural_cure_on_switch_out,
        apply_regenerator_on_switch_out as _apply_regenerator_on_switch_out,
    )
    _apply_regenerator_on_switch_out(battle, target_active_off, True)
    _apply_natural_cure_on_switch_out(battle, target_active_off, True)

    battle[active_meta] = new_active

    # Reset boosts on the phazed-in Pokemon (preserve tera_type in top 4 bits of slot 14)
    phazed_off = target_base + new_active * POKEMON_SIZE
    old_b2 = int(battle[phazed_off + 14]) & 0xFFFF
    tera_keep = old_b2 & 0xF000
    reset_b2 = (NEUTRAL_BOOSTS_14 & 0x0FFF) | tera_keep
    battle[phazed_off + 13] = _to_int16(NEUTRAL_BOOSTS_13)
    battle[phazed_off + 14] = _to_int16(reset_b2)

    # Clear volatiles for the phazed side (switching clears most volatiles)
    if is_side0:
        battle[OFF_FIELD + F_VOLATILE_0] = 0
        battle[OFF_FIELD + F_EXTENDED_VOLATILE_0] = 0
    else:
        battle[OFF_FIELD + F_VOLATILE_1] = 0
        battle[OFF_FIELD + F_EXTENDED_VOLATILE_1] = 0

    return new_active

# -----------------------------------------------------------------------------
# Extended volatile bag (torment, attract, yawn, embargo, ...)
# -----------------------------------------------------------------------------

def apply_extended_volatile(
    battle: np.ndarray,
    move_id: int,
    target_side: int,
    attacker_side: int,
    hit: bool,
    game_data=None,
    move_effects=None,
    gen5_prng: Gen5PRNG = None,
    prerolled_roll: "int | None" = None,
    prerolled_duration: "int | None" = None,
    target_offset_override: "int | None" = None,
) -> None:
    """Port of _apply_extended_volatile (line ~10595).

    Sets a single bit in F_EXTENDED_VOLATILE_x for one of the bag-of-flags
    volatiles (focus energy, torment, attract, yawn, embargo, heal block,
    imprison, ingrain, aqua ring, curse, mean look, lock-on, partial trap,
    salt cure, foresight). Self-targeted ones land on attacker_side.
    """
    if move_effects is None or gen5_prng is None:
        return
    move_id = int(move_id)
    volatile_type = int(move_effects.volatile[move_id])
    volatile_chance = int(move_effects.volatile_chance[move_id])

    has_extended_volatile = (
        bool(hit)
        and (volatile_chance > 0)
        and (volatile_type >= VOLATILE_FOCUS_ENERGY)
    )
    if not has_extended_volatile:
        return

    # Showdown only consumes a `secondaryRoll` random(100) frame for
    # secondaries on damaging moves (sim/battle-actions.ts:1357). Primary
    # volatiles set by status moves (Aqua Ring, Yawn, Leech Seed-like
    # bag-of-flags volatiles, Curse, Attract, Foresight, Miracle Eye, Laser
    # Focus, Helping Hand, Ingrain, Magnet Rise, Telekinesis, Embargo, Heal
    # Block, Salt Cure, Focus Energy, Torment, Imprison, Mean Look, Lock-On,
    # Partial Trap) are encoded at the move level as `volatileStatus: '...'`
    # — these are PRIMARY effects, not secondaries, and Showdown's moveHit
    # path takes them without rolling random(100). Pokepy's
    # `_preroll_move_secondaries` already short-circuits for status moves
    # (no preroll), so primary extended volatiles arrive here with
    # `prerolled_roll is None`. We must NOT roll random(100) in that case —
    # doing so causes a 1-frame PRNG drift on every primary-volatile status
    # move (e.g. Aqua Ring, Yawn), and this function is called twice per
    # turn (both move slots), so the drift compounds. Only roll when we
    # have a prerolled value (i.e. the rare case of an extended-volatile
    # secondary on a damaging move, in which case the preroll pipeline
    # has already consumed the frame at the correct PRNG position).
    if prerolled_roll is None:
        roll = 0  # short-circuit: primary extended volatile, no random(100)
    else:
        roll = int(prerolled_roll)
    if roll >= volatile_chance:
        return

    is_focus_energy = volatile_type == VOLATILE_FOCUS_ENERGY
    is_torment = volatile_type == VOLATILE_TORMENT
    is_attract = volatile_type == VOLATILE_ATTRACT
    is_yawn = volatile_type == VOLATILE_YAWN
    # Yawn fails if target already has a non-volatile status OR is immune
    # to sleep (ability / type / terrain). Showdown: data/moves.ts yawn
    # onTryHit + Electric Terrain onTryAddVolatile. Safeguard also blocks
    # the yawn volatile when target is not the source (opposing yawn) —
    # moves.ts safeguard.onTryAddVolatile line 16199.
    if is_yawn:
        _target_side_yawn = int(target_side)
        _target_offset_yawn = _active_offset(battle, _target_side_yawn)
        from pokepy.effects.status_apply import can_set_self_status
        from pokepy.core.constants import (
            STATUS_SLEEP, F_SCREENS_0 as _FS0_YA, F_SCREENS_1 as _FS1_YA,
            SCREEN_SAFEGUARD_SHIFT as _SG_SHIFT_YA,
        )
        _cur_status_yawn = int(battle[_target_offset_yawn + 12]) & 0xFF
        if _cur_status_yawn != 0:
            return
        if not can_set_self_status(battle, _target_offset_yawn, STATUS_SLEEP):
            return
        # Safeguard on the target's side blocks opposing yawn.
        if int(attacker_side) != _target_side_yawn:
            _screens_yawn = int(battle[OFF_FIELD + (_FS0_YA if _target_side_yawn == 0 else _FS1_YA)])
            if ((_screens_yawn >> _SG_SHIFT_YA) & 0x3) > 0:
                return
    is_embargo = volatile_type == VOLATILE_EMBARGO
    is_heal_block = volatile_type == VOLATILE_HEAL_BLOCK
    is_imprison = volatile_type == VOLATILE_IMPRISON
    is_ingrain = volatile_type == VOLATILE_INGRAIN
    is_aqua_ring = volatile_type == VOLATILE_AQUA_RING
    is_curse = volatile_type == VOLATILE_CURSE
    is_mean_look = volatile_type == VOLATILE_MEAN_LOOK
    is_lock_on = volatile_type == VOLATILE_LOCK_ON
    is_partial_trap = volatile_type == VOLATILE_PARTIAL_TRAP
    is_salt_cure = volatile_type == VOLATILE_SALT_CURE
    is_foresight = volatile_type == VOLATILE_FORESIGHT

    is_self_target = (
        is_focus_energy or is_imprison or is_ingrain or is_aqua_ring or is_lock_on
    )
    vol_target_side = int(attacker_side) if is_self_target else int(target_side)

    # Showdown's target.addVolatile(...) is a no-op on fainted foes. This is
    # especially important for trapping moves: partiallytrapped.durationCallback
    # only runs when the target survives the hit, so rolling the duration here
    # after a KO drifts every later PRNG frame.
    target_offset = (
        int(target_offset_override)
        if target_offset_override is not None
        else _active_offset(battle, vol_target_side)
    )
    if not is_self_target:
        target_hp_live = int(battle[target_offset + 1])
        target_flags_live = int(battle[target_offset + 15])
        if target_hp_live <= 0 or (target_flags_live & 0x1) != 0:
            return

    ext_vol_offset = _side_field(vol_target_side, F_EXTENDED_VOLATILE_0, F_EXTENDED_VOLATILE_1)
    current_ext_vol = int(battle[ext_vol_offset]) & 0xFFFF
    partial_trap_turns_offset = OFF_MOVES + (
        M_PARTIAL_TRAP_TURNS_0 if vol_target_side == 0 else M_PARTIAL_TRAP_TURNS_1
    )

    target_types = int(battle[target_offset + 4]) & 0xFFFF
    target_type1 = target_types & 0xFF
    target_type2 = (target_types >> 8) & 0xFF
    target_is_ghost = (target_type1 == TYPE_GHOST) or (target_type2 == TYPE_GHOST)
    mean_look_succeeds = is_mean_look and (not target_is_ghost)

    # Attract: opposite genders only. Oblivious makes target immune.
    # Showdown: abilities.ts oblivious onTryHit (attract/captivate/taunt)
    # and onImmunity(type='attract'). Genderless mons (0 or 3) can't be
    # attracted either way.
    from pokepy.core.constants import ABILITY_OBLIVIOUS as _ABIL_OBLIV
    target_ability_xv = int(battle[target_offset + 5])
    atk_offset = _active_offset(battle, int(attacker_side))
    atk_flags = int(battle[atk_offset + 15])
    atk_gender = (atk_flags >> 4) & 0x3
    target_flags = int(battle[target_offset + 15])
    target_gender = (target_flags >> 4) & 0x3
    opposite_genders = (
        (atk_gender == GENDER_MALE and target_gender == GENDER_FEMALE)
        or (atk_gender == GENDER_FEMALE and target_gender == GENDER_MALE)
    )
    # Aroma Veil blocks Attract / Disable / Encore / Heal Block / Taunt /
    # Torment from opposing moves only. Showdown data/abilities.ts aromaveil
    # onAllyTryAddVolatile (singles: ally == self).
    _ABILITY_AROMA_VEIL_X = 165
    has_aroma_veil_xv = target_ability_xv == _ABILITY_AROMA_VEIL_X
    is_aroma_blocked_vol = (
        is_attract or is_torment or is_heal_block
    )
    if has_aroma_veil_xv and is_aroma_blocked_vol and not is_self_target:
        return
    attract_succeeds = (
        is_attract and opposite_genders and (target_ability_xv != _ABIL_OBLIV)
    )

    bit_to_set = 0
    if is_focus_energy:
        bit_to_set = EXT_VOL_FOCUS_ENERGY
    if is_torment:
        bit_to_set = EXT_VOL_TORMENT
    if attract_succeeds:
        bit_to_set = EXT_VOL_ATTRACT
    if is_yawn:
        bit_to_set = EXT_VOL_YAWN
    if is_embargo:
        bit_to_set = EXT_VOL_EMBARGO
    if is_heal_block:
        bit_to_set = EXT_VOL_HEAL_BLOCK
    if is_imprison:
        bit_to_set = EXT_VOL_IMPRISON
    if is_ingrain:
        bit_to_set = EXT_VOL_INGRAIN
    if is_aqua_ring:
        bit_to_set = EXT_VOL_AQUA_RING
    if is_curse:
        bit_to_set = EXT_VOL_CURSE
    if mean_look_succeeds:
        bit_to_set = EXT_VOL_MEAN_LOOK
    if is_lock_on:
        bit_to_set = EXT_VOL_LOCK_ON
    if is_partial_trap:
        if (current_ext_vol & EXT_VOL_PARTIAL_TRAP) != 0:
            return
        bit_to_set = EXT_VOL_PARTIAL_TRAP
    if is_salt_cure:
        bit_to_set = EXT_VOL_SALT_CURE
    if is_foresight:
        bit_to_set = EXT_VOL_FORESIGHT

    if bit_to_set == 0:
        return
    if is_heal_block and (current_ext_vol & EXT_VOL_HEAL_BLOCK) != 0:
        return
    battle[ext_vol_offset] = _to_int16(current_ext_vol | bit_to_set)
    if is_partial_trap:
        _ITEM_GRIP_CLAW = 179
        if prerolled_duration is not None:
            trap_turns = int(prerolled_duration)
        else:
            attacker_item = int(battle[atk_offset + 6])
            trap_turns = 8 if attacker_item == _ITEM_GRIP_CLAW else int(gen5_prng.random(5, 7))
        battle[partial_trap_turns_offset] = _to_int16(trap_turns)
    if is_heal_block:
        volatile_offset = _side_field(vol_target_side, F_VOLATILE_0, F_VOLATILE_1)
        current_volatile = int(battle[volatile_offset]) & 0xFFFF
        # Showdown moves.ts:healblock durationCallback returns 2 when the
        # volatile came from Psychic Noise, otherwise 5.
        _MOVE_PSYCHIC_NOISE_HB = 917
        hb_turns = 2 if move_id == _MOVE_PSYCHIC_NOISE_HB else 5
        battle[volatile_offset] = _to_int16(set_heal_block_turns(current_volatile, hb_turns))

    # Yawn: also set the F_YAWN_TURNS counter so EOT can put the target to
    # sleep on the second turn. Showdown duration=2 → EOT decrements to 1
    # (drowsy), next EOT → trySetStatus 'slp'. We store 2 here.
    if is_yawn:
        from pokepy.core.constants import F_YAWN_TURNS_0, F_YAWN_TURNS_1
        yawn_turns_off = OFF_FIELD + (F_YAWN_TURNS_0 if vol_target_side == 0 else F_YAWN_TURNS_1)
        # Only set if not already yawning
        if int(battle[yawn_turns_off]) == 0:
            battle[yawn_turns_off] = 2

# -----------------------------------------------------------------------------
# Confusion turn handling
# -----------------------------------------------------------------------------

def check_confusion_self_hit(
    battle: np.ndarray,
    side: int,
    pokemon_offset: int,
    gen5_prng: Gen5PRNG = None,
) -> bool:
    """Port of _check_confusion_self_hit (line ~10705).

    Returns True iff the user hits itself in confusion (mutating HP in
    place). Engine consumes only the bool — internal damage application
    matches the source.
    """
    pokemon_offset = int(pokemon_offset)
    volatile_offset = _side_field(side, F_VOLATILE_0, F_VOLATILE_1)
    volatile = int(battle[volatile_offset])
    confusion_turns = get_confusion_turns(volatile)

    is_confused = confusion_turns > 0
    if not is_confused or gen5_prng is None:
        return False

    # "Newly applied this turn" means the move-side inline confusion path has
    # already handled the SAME-TURN onBeforeMove check. Mirror Showdown's
    # pre-check decrement once here, then skip a duplicate self-hit roll.
    if get_confusion_newly_applied(volatile):
        volatile = set_confusion_newly_applied(volatile, False)
        volatile = set_confusion_turns(volatile, max(0, confusion_turns - 1))
        battle[volatile_offset] = _to_int16(volatile)
        return False

    # Showdown decrements confusion before testing expiry or self-hit.
    confusion_turns = max(0, confusion_turns - 1)
    volatile = set_confusion_turns(volatile, confusion_turns)
    battle[volatile_offset] = _to_int16(volatile)
    if confusion_turns == 0:
        return False

    # Showdown: randomChance(33, 100) — exactly 33%, NOT 1/3 (33.33%).
    # Source: data/conditions.ts confusion onBeforeMove.
    conf_roll = gen5_prng.random(100)
    self_hit = conf_roll < 33
    if not self_hit:
        return False

    # Showdown confusion self-hit damage (sim/battle-actions.ts:1843-1854):
    #     baseDamage = floor(floor(floor(floor(2L/5 + 2) * BP * Atk) / Def) / 50) + 2
    #     damage = tr(baseDamage, 16)       # 16-bit truncation
    #     damage = randomizer(damage)       # 85-100% damage roll
    #     return max(1, damage)
    # BP is 40, Atk/Def use calculateStat('atk', boosts['atk']) — base stats
    # with stage BOOSTS applied. No burn halving, no STAB/crit.
    from pokepy.core.bitpack import extract_boost
    level = int(battle[pokemon_offset + 3])
    atk_base = int(battle[pokemon_offset + 7])
    def_base = max(1, int(battle[pokemon_offset + 8]))
    boosts13 = int(battle[pokemon_offset + 13])
    atk_stage = max(-6, min(6, extract_boost(boosts13, 0)))
    def_stage = max(-6, min(6, extract_boost(boosts13, 4)))
    if atk_stage >= 0:
        atk = (atk_base * (2 + atk_stage)) // 2
    else:
        atk = (atk_base * 2) // (2 - atk_stage)
    if def_stage >= 0:
        def_stat = (def_base * (2 + def_stage)) // 2
    else:
        def_stat = (def_base * 2) // (2 - def_stage)
    def_stat = max(1, def_stat)
    current_hp = int(battle[pokemon_offset + 1])

    base_step1 = (2 * level) // 5 + 2
    base_step2 = base_step1 * 40 * atk
    base_step3 = base_step2 // def_stat
    base_step4 = base_step3 // 50
    base_damage = base_step4 + 2
    # 16-bit truncation (battle-actions.ts:1852 `tr(baseDamage, 16)`).
    base_damage = base_damage & 0xFFFF
    # randomizer: tr(tr(baseDamage * (100 - random(16))) / 100)
    rand_roll = gen5_prng.random(16)
    damage = (base_damage * (100 - rand_roll)) // 100
    damage = max(1, damage)

    new_hp = max(0, current_hp - damage)
    battle[pokemon_offset + 1] = new_hp
    return True

def decrement_confusion(battle: np.ndarray) -> None:
    """Port of _decrement_confusion (line ~10758).

    Showdown does not tick confusion on residual; the counter decrements on
    confusion.onBeforeMove. Residual only clears the one-turn "newly applied"
    marker for cases where a same-turn confusion never reached onBeforeMove.
    """
    for vol_off in (OFF_FIELD + F_VOLATILE_0, OFF_FIELD + F_VOLATILE_1):
        v = int(battle[vol_off])
        v = set_confusion_newly_applied(v, False)
        battle[vol_off] = _to_int16(v)

def decrement_taunt_encore(battle: np.ndarray, gen5_prng: Gen5PRNG | None = None) -> None:
    """Port of _decrement_taunt_encore (line ~10778).

    Decrements taunt, encore, heal block, Throat Chop and disable turns on
    both sides at end of turn.
    """
    from pokepy.core.constants import (
        F_DISABLE_0 as _FD0_DI,
        F_DISABLE_1 as _FD1_DI,
        F_DISABLE_TURNS_0 as _FDT0_DI,
        F_DISABLE_TURNS_1 as _FDT1_DI,
    )

    def _consume_disablemove_shuffle_frames(
        side: int,
        vol_off: int,
        ext_off: int,
        choice_off: int,
        dis_off: int,
        dis_turns_off: int,
    ) -> None:
        """Mirror Showdown's next-request `runEvent('DisableMove', pokemon)` speedSort.

        Equal-priority `onDisableMove` handlers on the same active Pokemon are
        shuffled via `Battle.speedSort -> PRNG.shuffle`, which burns hidden PRNG
        frames even when the resulting disabled move set is unchanged. Pokepy
        applies the disable sources deterministically, so consume the matching
        shuffle frames here for the DisableMove handlers we model explicitly:
        Choice lock, Disable, Taunt, Encore, Heal Block, Throat Chop, Torment,
        and Assault Vest.
        """
        if gen5_prng is None:
            return

        from pokepy.core.constants import (
            ITEM_ASSAULT_VEST,
        )

        active_slot = int(
            battle[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)]
        )
        active_off = (OFF_SIDE0 if side == 0 else OFF_SIDE1) + active_slot * POKEMON_SIZE
        active_hp = int(battle[active_off + 1])
        active_flags = int(battle[active_off + 15])
        # Showdown rebuilds DisableMove handlers on the next move request. If
        # the current active fainted this turn, the next request is a forced
        # switch rather than an active move, so there is no DisableMove
        # handler shuffle to emulate.
        if active_hp <= 0 or (active_flags & 0x1) != 0:
            return
        vol = int(battle[vol_off]) & 0xFFFF
        ext = int(battle[ext_off]) & 0xFFFF

        # Showdown only shuffles tied DisableMove handlers. The handlers we
        # model here do not all share the same tie group:
        #   - Conditions / volatiles (Choice lock, Disable, Taunt, Encore,
        #     Heal Block, Throat Chop, Torment) use subOrder 2
        #   - Items (Assault Vest) use subOrder 8
        # Different subOrders are ordered deterministically, so only consume
        # shuffle frames within each surviving tie group.
        condition_handler_count = 0
        item_handler_count = 0
        if int(battle[choice_off]) >= 0:
            condition_handler_count += 1
        # Pokepy calls this helper before the residual countdowns below, but
        # Showdown rebuilds disabled moves on the next move request after those
        # countdowns have already ticked. Only count timed handlers that will
        # still exist at the next request.
        if int(battle[dis_off]) >= 0 and int(battle[dis_turns_off]) > 1:
            condition_handler_count += 1
        if get_taunt_turns(vol) > 1:
            condition_handler_count += 1
        if get_encore_turns(vol) > 1:
            condition_handler_count += 1
        if get_heal_block_turns(vol) > 1:
            condition_handler_count += 1
        # Pokepy's packed Throat Chop countdown stores `2` on the hit turn and
        # `1` on the following lingering lock turn. Showdown's hidden
        # DisableMove shuffle frame only occurs on the fresh-application turn.
        if get_throat_chop_turns(vol) > 1:
            condition_handler_count += 1
        if (ext & EXT_VOL_TORMENT) != 0:
            condition_handler_count += 1
        if int(battle[active_off + 6]) == ITEM_ASSAULT_VEST:
            item_handler_count += 1

        # Showdown's PRNG.shuffle(items, 0, n) uses random(0, n),
        # random(1, n), ..., random(n - 2, n).
        for group_size in (condition_handler_count, item_handler_count):
            for start in range(group_size - 1):
                gen5_prng.random(start, group_size)

    for vol_off, ext_off, choice_off in (
        (OFF_FIELD + F_VOLATILE_0, OFF_FIELD + F_EXTENDED_VOLATILE_0, OFF_FIELD + F_CHOICE_LOCK_0),
        (OFF_FIELD + F_VOLATILE_1, OFF_FIELD + F_EXTENDED_VOLATILE_1, OFF_FIELD + F_CHOICE_LOCK_1),
    ):
        v = int(battle[vol_off])
        taunt = get_taunt_turns(v)
        encore = get_encore_turns(v)
        heal_block = get_heal_block_turns(v)
        throat_chop = get_throat_chop_turns(v)
        # Showdown rebuilds active `onDisableMove` handlers on the next move
        # request, after the residual countdown decrements below. Pokepy
        # applies those hooks deterministically, so emulate the hidden
        # speedSort shuffle frames here using only handlers that survive the
        # pending countdown step.
        if vol_off == OFF_FIELD + F_VOLATILE_0:
            _consume_disablemove_shuffle_frames(
                0,
                vol_off,
                ext_off,
                choice_off,
                OFF_FIELD + _FD0_DI,
                OFF_FIELD + _FDT0_DI,
            )
        else:
            _consume_disablemove_shuffle_frames(
                1,
                vol_off,
                ext_off,
                choice_off,
                OFF_FIELD + _FD1_DI,
                OFF_FIELD + _FDT1_DI,
            )
        v = set_taunt_turns(v, max(0, taunt - 1))
        v = set_encore_turns(v, max(0, encore - 1))
        new_heal_block = max(0, heal_block - 1)
        v = set_heal_block_turns(v, new_heal_block)
        v = set_throat_chop_turns(v, max(0, throat_chop - 1))
        battle[vol_off] = _to_int16(v)
        if heal_block > 0 and new_heal_block == 0:
            ext = int(battle[ext_off]) & 0xFFFF
            battle[ext_off] = _to_int16(ext & ~EXT_VOL_HEAL_BLOCK)

    # Disable: residual order 17. Decrement turn counter; clear slot when
    # it hits zero. Showdown data/moves.ts:disable condition onResidualOrder.
    for dis_off, dis_turns_off in (
        (OFF_FIELD + _FD0_DI, OFF_FIELD + _FDT0_DI),
        (OFF_FIELD + _FD1_DI, OFF_FIELD + _FDT1_DI),
    ):
        dt = int(battle[dis_turns_off])
        if dt > 0:
            new_dt = dt - 1
            battle[dis_turns_off] = _to_int16(new_dt)
            if new_dt == 0:
                battle[dis_off] = -1

# -----------------------------------------------------------------------------
# Perish Song / Curse end-of-turn
# -----------------------------------------------------------------------------

def process_perish_song(
    battle: np.ndarray,
    pokemon0_offset: int,
    pokemon1_offset: int,
) -> None:
    """Port of _process_perish_song (line ~10804).

    Decrements perish counters; faints the Pokemon when its counter hits 0.
    """
    p0 = int(pokemon0_offset)
    p1 = int(pokemon1_offset)

    perish0 = int(battle[OFF_FIELD + F_PERISH_COUNT_0])
    perish1 = int(battle[OFF_FIELD + F_PERISH_COUNT_1])

    has_perish0 = perish0 > 0
    has_perish1 = perish1 > 0

    new_perish0 = (perish0 - 1) if has_perish0 else perish0
    new_perish1 = (perish1 - 1) if has_perish1 else perish1

    faint0 = has_perish0 and (new_perish0 == 0)
    faint1 = has_perish1 and (new_perish1 == 0)

    if faint0:
        battle[p0 + 1] = 0
    if faint1:
        battle[p1 + 1] = 0

    battle[OFF_FIELD + F_PERISH_COUNT_0] = new_perish0
    battle[OFF_FIELD + F_PERISH_COUNT_1] = new_perish1

def apply_curse_damage(
    battle: np.ndarray,
    pokemon0_offset: int,
    pokemon1_offset: int,
    game_data=None,
) -> None:
    """Port of _apply_curse_damage (line ~10838).

    Ghost Curse deals 1/4 max HP per turn to the cursed Pokemon. Showdown
    calls `this.damage(pokemon.baseMaxhp / 4)` (moves.ts:3421), which runs
    through `onDamage`, so Magic Guard (abilities.ts:2418) blocks curse
    damage — "prevents indirect damage" includes curse.
    """
    p0 = int(pokemon0_offset)
    p1 = int(pokemon1_offset)

    ext_vol0 = int(battle[OFF_FIELD + F_EXTENDED_VOLATILE_0]) & 0xFFFF
    ext_vol1 = int(battle[OFF_FIELD + F_EXTENDED_VOLATILE_1]) & 0xFFFF

    has_curse0 = (ext_vol0 & EXT_VOL_CURSE) != 0
    has_curse1 = (ext_vol1 & EXT_VOL_CURSE) != 0

    if has_curse0 and int(battle[p0 + 5]) != ABILITY_MAGIC_GUARD:
        max_hp0 = int(battle[p0 + 2])
        cur_hp0 = int(battle[p0 + 1])
        dmg0 = max(1, max_hp0 // 4)
        battle[p0 + 1] = max(0, cur_hp0 - dmg0)

    if has_curse1 and int(battle[p1 + 5]) != ABILITY_MAGIC_GUARD:
        max_hp1 = int(battle[p1 + 2])
        cur_hp1 = int(battle[p1 + 1])
        dmg1 = max(1, max_hp1 // 4)
        battle[p1 + 1] = max(0, cur_hp1 - dmg1)

# -----------------------------------------------------------------------------
# Salt Cure
# -----------------------------------------------------------------------------

def apply_salt_cure_damage(
    battle: np.ndarray,
    pokemon_offset: int,
    ext_vol_offset: int,
    game_data=None,
) -> None:
    """Port of _apply_salt_cure_damage (line ~8824).

    Salt Cure deals 1/8 max HP (1/4 for Water/Steel types). Magic Guard
    prevents the damage.
    """
    pokemon_offset = int(pokemon_offset)
    ext_vol_offset = int(ext_vol_offset)

    ext_vol = int(battle[ext_vol_offset]) & 0xFFFF
    has_salt_cure = (ext_vol & EXT_VOL_SALT_CURE) != 0
    if not has_salt_cure:
        return

    hp = int(battle[pokemon_offset + 1])
    if hp <= 0:
        return

    ability = int(battle[pokemon_offset + 5])
    if ability == ABILITY_MAGIC_GUARD:
        return

    max_hp = int(battle[pokemon_offset + 2])
    types = int(battle[pokemon_offset + 4]) & 0xFFFF
    type1 = types & 0xFF
    type2 = (types >> 8) & 0xFF
    is_water = (type1 == TYPE_WATER) or (type2 == TYPE_WATER)
    is_steel = (type1 == TYPE_STEEL) or (type2 == TYPE_STEEL)

    if is_water or is_steel:
        salt_damage = max_hp // 4
    else:
        salt_damage = max_hp // 8
    salt_damage = max(1, salt_damage)

    battle[pokemon_offset + 1] = max(0, hp - salt_damage)
