"""Item effects (berries, leftovers, life orb, ...).

"""
from __future__ import annotations

from pokepy.effects._common import np, MultiFormatState, Gen5PRNG
from pokepy.core.bitpack import (
    apply_boost_to_packed,
    get_status,
    set_confusion_turns,
)
from pokepy.core.constants import (
    OFF_FIELD,
    OFF_META,
    OFF_SIDE0,
    OFF_SIDE1,
    M_ACTIVE0,
    M_ACTIVE1,
    POKEMON_SIZE,
    F_VOLATILE_0,
    F_VOLATILE_1,
    F_EXTENDED_VOLATILE_0,
    F_EXTENDED_VOLATILE_1,
    EXT_VOL_HEAL_BLOCK,
    STATUS_NONE,
    STATUS_BURN,
    STATUS_PARALYSIS,
    STATUS_SLEEP,
    STATUS_FREEZE,
    STATUS_POISON,
    STATUS_TOXIC,
    TYPE_POISON,
    CAT_PHYSICAL,
    CAT_SPECIAL,
    ABILITY_GLUTTONY,
    ABILITY_CHEEK_POUCH,
    ABILITY_RIPEN,
    ABILITY_MAGIC_GUARD,
    ABILITY_SHEER_FORCE,
    ITEM_LEFTOVERS,
    ITEM_BLACK_SLUDGE,
    ITEM_SITRUS_BERRY,
    ITEM_LUM_BERRY,
    ITEM_MIRACLE_BERRY,
    ITEM_CHERI_BERRY,
    ITEM_RAWST_BERRY,
    ITEM_PECHA_BERRY,
    ITEM_CHESTO_BERRY,
    ITEM_ASPEAR_BERRY,
    ITEM_PERSIM_BERRY,
    ITEM_LIECHI_BERRY,
    ITEM_PETAYA_BERRY,
    ITEM_SALAC_BERRY,
    ITEM_GANLON_BERRY,
    ITEM_APICOT_BERRY,
    ITEM_KEE_BERRY,
    ITEM_MARANGA_BERRY,
    ITEM_STARF_BERRY,
    ITEM_FIGY_BERRY,
    ITEM_WIKI_BERRY,
    ITEM_MAGO_BERRY,
    ITEM_AGUAV_BERRY,
    ITEM_IAPAPA_BERRY,
    ITEM_ORAN_BERRY,
    ITEM_STICKY_BARB,
    ITEM_LIFE_ORB,
)
from pokepy.data.item_aliases import ITEM_GOLD_BERRY_INTERNAL

# Unnerve (ability id 127, num 127 in showdown). Not in core/constants.py.
# data/abilities.ts:5175 unnerve onFoeTryEatItem returns false → opponent
# cannot eat any berry while a foe with Unnerve is active. As One (Spectrier
# / Glastrier) and Unnerve all share the same Unnerve eat-block. We treat
# only the canonical Unnerve id here; the As One ids aren't tracked yet.
ABILITY_UNNERVE = 127
ABILITY_AS_ONE_GLASTRIER = 266
ABILITY_AS_ONE_SPECTRIER = 267
MOVE_TRI_ATTACK = 161
ITEM_GOLD_BERRY = ITEM_GOLD_BERRY_INTERNAL

def _has_heal_block(battle: np.ndarray, poff: int) -> bool:
    """Return True iff the holder is under Heal Block."""
    p = int(poff)
    ext_off = OFF_FIELD + (F_EXTENDED_VOLATILE_0 if p < OFF_SIDE1 else F_EXTENDED_VOLATILE_1)
    ext = int(battle[ext_off]) & 0xFFFF
    return (ext & EXT_VOL_HEAL_BLOCK) != 0

def _opponent_has_unnerve(battle: np.ndarray, poff: int) -> bool:
    """Return True iff the OPPONENT's active mon has Unnerve / As One.

    Showdown semantics: berries are blocked from being eaten while any foe
    with Unnerve is on the field. In singles that is just the one active
    opposing mon. The active mon offset is read from OFF_META M_ACTIVE0/1.
    """
    p = int(poff)
    holder_is_side0 = p < OFF_SIDE1
    if holder_is_side0:
        opp_active_idx = int(battle[OFF_META + M_ACTIVE1])
        opp_off = OFF_SIDE1 + opp_active_idx * POKEMON_SIZE
    else:
        opp_active_idx = int(battle[OFF_META + M_ACTIVE0])
        opp_off = OFF_SIDE0 + opp_active_idx * POKEMON_SIZE
    if int(battle[opp_off + 1]) <= 0:
        return False
    opp_ability = int(battle[opp_off + 5])
    return opp_ability in (
        ABILITY_UNNERVE, ABILITY_AS_ONE_GLASTRIER, ABILITY_AS_ONE_SPECTRIER,
    )

def _apply_cheek_pouch_heal(battle: np.ndarray, poff: int) -> None:
    """Showdown abilities.ts cheekpouch onEatItem: heal 1/3 max HP when the
    holder eats any berry. Called right after any berry consumption.

    Gated on ability == Cheek Pouch and holder not KO'd. Does nothing if
    already at full HP (min clip). Heal Block suppresses the heal because
    Showdown routes it through `this.heal(...)`.
    """
    p = int(poff)
    ability = int(battle[p + 5])
    if ability != ABILITY_CHEEK_POUCH:
        return
    hp = int(battle[p + 1])
    if hp <= 0:
        return
    if _has_heal_block(battle, p):
        return
    max_hp = int(battle[p + 2])
    heal = max(int(max_hp / 3), 1)
    battle[p + 1] = min(max_hp, hp + heal)

def apply_sticky_barb_residual(
    battle: np.ndarray, pokemon_offset: int, game_data
) -> None:
    """Sticky Barb onResidual (items.ts:5686).

    Deals 1/8 max HP at end of turn. Magic Guard blocks. Does not consume
    the item. Contact-transfer logic lives in the damage/contact path
    (pokepy handles this in battle_gen9 when a contact move hits a barbed
    target with an empty-handed attacker — not yet implemented).
    """
    poff = int(pokemon_offset)
    hp = int(battle[poff + 1])
    if hp <= 0:
        return
    item = int(battle[poff + 6])
    if item != ITEM_STICKY_BARB:
        return
    ability = int(battle[poff + 5])
    if ability == ABILITY_MAGIC_GUARD:
        return
    max_hp = int(battle[poff + 2])
    dmg = max(int(max_hp / 8), 1)
    battle[poff + 1] = max(0, hp - dmg)

def apply_leftovers_healing(battle: np.ndarray, pokemon_offset: int, game_data) -> None:
    """Port of _apply_leftovers_healing (line ~9070).

    Heals 1/16 max HP at end of turn if the Pokemon is holding Leftovers and
    is alive. Mutates `battle` in place.
    """
    poff = int(pokemon_offset)
    current_hp = int(battle[poff + 1])
    max_hp = int(battle[poff + 2])
    item = int(battle[poff + 6])

    has_leftovers = item == ITEM_LEFTOVERS

    heal_amount = max(int(max_hp / 16), 1)
    new_hp = min(max_hp, current_hp + heal_amount)

    should_heal = has_leftovers and (current_hp > 0) and (not _has_heal_block(battle, poff))
    final_hp = new_hp if should_heal else current_hp

    battle[poff + 1] = final_hp

def apply_black_sludge_effect(battle: np.ndarray, pokemon_offset: int, game_data) -> None:
    """Port of _apply_black_sludge_effect (line ~9100).

    Heals Poison-type holder 1/16 max HP, damages non-Poison holders 1/8 max HP.
    """
    poff = int(pokemon_offset)
    current_hp = int(battle[poff + 1])
    max_hp = int(battle[poff + 2])
    item = int(battle[poff + 6])
    types_packed = int(battle[poff + 4])
    type1 = types_packed & 0xFF
    type2 = (types_packed >> 8) & 0xFF

    has_black_sludge = item == ITEM_BLACK_SLUDGE
    is_poison_type = (type1 == TYPE_POISON) or (type2 == TYPE_POISON)

    heal_amount = max(int(max_hp / 16), 1)
    healed_hp = min(max_hp, current_hp + heal_amount)

    damage_amount = max(int(max_hp / 8), 1)
    damaged_hp = max(0, current_hp - damage_amount)

    heal_blocked = _has_heal_block(battle, poff)
    effect_hp = current_hp if (is_poison_type and heal_blocked) else (healed_hp if is_poison_type else damaged_hp)

    should_apply = has_black_sludge and (current_hp > 0)
    final_hp = effect_hp if should_apply else current_hp

    battle[poff + 1] = final_hp

def apply_sitrus_berry(battle: np.ndarray, pokemon_offset: int, game_data) -> None:
    """Port of _apply_sitrus_berry (line ~9144).

    Heals 25% max HP and consumes the berry when the holder drops below 50% HP.
    """
    poff = int(pokemon_offset)
    current_hp = int(battle[poff + 1])
    max_hp = int(battle[poff + 2])
    item = int(battle[poff + 6])

    has_sitrus = item == ITEM_SITRUS_BERRY

    # Showdown items.ts:5353 — `pokemon.hp <= pokemon.maxhp / 2` (inclusive).
    # Pokepy used strict `<` so a mon at exactly 50% HP never triggered.
    hp_threshold = int(max_hp / 2)
    below_half = current_hp <= hp_threshold

    heal_amount = max(int(max_hp / 4), 1)
    new_hp = min(max_hp, current_hp + heal_amount)

    # Unnerve on the opposing active mon blocks all berry eats.
    # Showdown data/abilities.ts:5185 unnerve onFoeTryEatItem.
    if _opponent_has_unnerve(battle, poff):
        return

    should_heal = has_sitrus and below_half and (current_hp > 0)
    final_hp = new_hp if should_heal else current_hp

    battle[poff + 1] = final_hp
    battle[poff + 6] = 0 if should_heal else item

    if should_heal:
        _apply_cheek_pouch_heal(battle, poff)

def apply_gold_berry(battle: np.ndarray, pokemon_offset: int, game_data) -> None:
    """Showdown items.ts goldberry: heal 30 HP at <= 1/2 max HP."""
    poff = int(pokemon_offset)
    current_hp = int(battle[poff + 1])
    max_hp = int(battle[poff + 2])
    item = int(battle[poff + 6])

    if item != ITEM_GOLD_BERRY:
        return
    if _opponent_has_unnerve(battle, poff):
        return
    if current_hp <= 0 or current_hp > int(max_hp / 2):
        return

    battle[poff + 1] = min(max_hp, current_hp + 30)
    battle[poff + 6] = 0
    _apply_cheek_pouch_heal(battle, poff)

def apply_lum_berry(battle: np.ndarray, pokemon_offset: int, game_data) -> None:
    """Apply all-status berries such as Lum Berry and Miracle Berry.

    Showdown items.ts lumberry: onUpdate fires if the holder has a non-volatile
    status OR the confusion volatile; onEat calls cureStatus AND
    removeVolatile('confusion'). Pokepy used to only clear the non-volatile
    status, which left Dynamic Punch confusion through a Lum Berry.
    Gen 2's Miracle Berry has the same status/confusion cure behavior and can
    appear in custom Gen 9 teams used by the parity harness.
    """
    poff = int(pokemon_offset)
    status_field = int(battle[poff + 12])
    item = int(battle[poff + 6])

    if item not in (ITEM_LUM_BERRY, ITEM_MIRACLE_BERRY):
        return

    # Unnerve blocks the eat (Showdown abilities.ts:5185).
    if _opponent_has_unnerve(battle, poff):
        return

    status = get_status(status_field)
    has_status = status != STATUS_NONE

    # Confusion lives on the side's volatile field — check both slots.
    from pokepy.core.bitpack import get_confusion_turns as _get_conf_turns
    p_is_side0 = poff < OFF_SIDE1
    vol_off = OFF_FIELD + (F_VOLATILE_0 if p_is_side0 else F_VOLATILE_1)
    vol_val = int(battle[vol_off])
    conf_turns = _get_conf_turns(vol_val)
    has_confusion = conf_turns > 0

    if not (has_status or has_confusion):
        return

    # Cure status.
    if has_status:
        battle[poff + 12] = 0
    # Cure confusion (zero out the turns counter in the volatile field).
    if has_confusion:
        battle[vol_off] = np.int16(set_confusion_turns(vol_val, 0))
    battle[poff + 6] = 0
    _apply_cheek_pouch_heal(battle, poff)

def apply_status_curing_berries(battle: np.ndarray, pokemon_offset: int, game_data) -> None:
    """Port of _apply_status_curing_berries (line ~9223).

    Cheri/Rawst/Pecha/Chesto/Aspear cure Para/Burn/Poison/Sleep/Freeze.
    """
    poff = int(pokemon_offset)
    status_field = int(battle[poff + 12])
    item = int(battle[poff + 6])
    status = get_status(status_field)

    # Unnerve blocks all berry eats (Showdown abilities.ts:5185).
    if _opponent_has_unnerve(battle, poff):
        return

    has_cheri = item == ITEM_CHERI_BERRY
    is_paralyzed = status == STATUS_PARALYSIS
    cheri_triggers = has_cheri and is_paralyzed

    has_rawst = item == ITEM_RAWST_BERRY
    is_burned = status == STATUS_BURN
    rawst_triggers = has_rawst and is_burned

    has_pecha = item == ITEM_PECHA_BERRY
    is_poisoned = (status == STATUS_POISON) or (status == STATUS_TOXIC)
    pecha_triggers = has_pecha and is_poisoned

    has_chesto = item == ITEM_CHESTO_BERRY
    is_asleep = status == STATUS_SLEEP
    chesto_triggers = has_chesto and is_asleep

    has_aspear = item == ITEM_ASPEAR_BERRY
    is_frozen = status == STATUS_FREEZE
    aspear_triggers = has_aspear and is_frozen

    any_triggers = (
        cheri_triggers
        or rawst_triggers
        or pecha_triggers
        or chesto_triggers
        or aspear_triggers
    )

    new_status_field = 0 if any_triggers else status_field
    new_item = 0 if any_triggers else item

    battle[poff + 12] = new_status_field
    battle[poff + 6] = new_item

    if any_triggers:
        _apply_cheek_pouch_heal(battle, poff)

def apply_persim_berry(battle: np.ndarray, pokemon_offset: int, game_data) -> None:
    """Port of _apply_persim_berry (line ~9273).

    Cures Confusion (volatile) and consumes the berry.
    """
    poff = int(pokemon_offset)
    item = int(battle[poff + 6])

    # Unnerve blocks the eat (Showdown abilities.ts:5185).
    if _opponent_has_unnerve(battle, poff):
        return

    # Determine which side's volatile field to use. In pokepy effects modules,
    # active Pokemon for player 0 are at OFF_SIDE0 and player 1 at OFF_SIDE1.
    # Party slots map back to side 0/1 by which half they fall in.
    if poff < OFF_SIDE1:
        volatile_offset = OFF_FIELD + F_VOLATILE_0
    else:
        volatile_offset = OFF_FIELD + F_VOLATILE_1

    volatile = int(battle[volatile_offset])
    from pokepy.core.bitpack import get_confusion_turns

    confusion_turns = get_confusion_turns(volatile)

    has_persim = item == ITEM_PERSIM_BERRY
    is_confused = confusion_turns > 0
    persim_triggers = has_persim and is_confused

    new_volatile = set_confusion_turns(volatile, 0) if persim_triggers else volatile
    new_item = 0 if persim_triggers else item

    battle[volatile_offset] = new_volatile
    battle[poff + 6] = new_item

    if persim_triggers:
        _apply_cheek_pouch_heal(battle, poff)

def apply_defender_stat_berries_on_damaging_hit(
    battle: np.ndarray,
    pokemon_offset: int,
    move_category: int,
    hit: bool,
    damage_dealt: int,
    game_data,
) -> None:
    """Showdown keeberry / marangaberry `onDamagingHit` item hooks."""
    if not hit or int(damage_dealt) <= 0:
        return

    poff = int(pokemon_offset)
    if int(battle[poff + 1]) <= 0:
        return

    item = int(battle[poff + 6])
    if item not in (ITEM_KEE_BERRY, ITEM_MARANGA_BERRY):
        return
    if _opponent_has_unnerve(battle, poff):
        return

    move_cat = int(move_category)
    boosts1 = int(battle[poff + 13])
    boost_amount = 2 if int(battle[poff + 5]) == ABILITY_RIPEN else 1
    triggered = False

    if item == ITEM_KEE_BERRY and move_cat == CAT_PHYSICAL:
        boosts1 = apply_boost_to_packed(boosts1, 4, boost_amount)
        triggered = True
    elif item == ITEM_MARANGA_BERRY and move_cat == CAT_SPECIAL:
        boosts1 = apply_boost_to_packed(boosts1, 12, boost_amount)
        triggered = True

    if not triggered:
        return

    battle[poff + 13] = boosts1
    battle[poff + 6] = 0
    _apply_cheek_pouch_heal(battle, poff)

def apply_stat_boosting_berries(battle: np.ndarray, pokemon_offset: int, game_data) -> None:
    """Port of _apply_stat_boosting_berries (line ~9315).

    Pinch berries (HP <= 1/4 max, or 1/2 with Gluttony):
      Liechi  → +1 Atk  (items.ts:3107)
      Ganlon  → +1 Def  (items.ts:2174)
      Petaya  → +1 SpA  (items.ts:4208)
      Apicot  → +1 SpD  (items.ts:264)
      Salac   → +1 Spe  (items.ts:5118)
      Starf   → +2 random stat (items.ts; uses prng)

    All consume on trigger. Cheek Pouch adds +1/3 max HP heal on eat.
    """
    poff = int(pokemon_offset)
    hp = int(battle[poff + 1])
    max_hp = int(battle[poff + 2])
    ability = int(battle[poff + 5])
    item = int(battle[poff + 6])
    boosts1 = int(battle[poff + 13])  # Atk/Def/SpA/SpD packed
    boosts2 = int(battle[poff + 14])  # Spe/Acc/Eva packed

    # Unnerve blocks all berry eats (Showdown abilities.ts:5185).
    if _opponent_has_unnerve(battle, poff):
        return

    has_gluttony = ability == ABILITY_GLUTTONY
    hp_threshold_normal = int(max_hp / 4)
    hp_threshold_gluttony = int(max_hp / 2)
    hp_threshold = hp_threshold_gluttony if has_gluttony else hp_threshold_normal
    below_threshold = (hp <= hp_threshold) and (hp > 0)

    has_liechi = item == ITEM_LIECHI_BERRY
    liechi_triggers = has_liechi and below_threshold
    if liechi_triggers:
        boosts1 = apply_boost_to_packed(boosts1, 0, 1)  # Atk shift 0

    has_ganlon = item == ITEM_GANLON_BERRY
    ganlon_triggers = has_ganlon and below_threshold
    if ganlon_triggers:
        boosts1 = apply_boost_to_packed(boosts1, 4, 1)  # Def shift 4

    has_petaya = item == ITEM_PETAYA_BERRY
    petaya_triggers = has_petaya and below_threshold
    if petaya_triggers:
        boosts1 = apply_boost_to_packed(boosts1, 8, 1)  # SpA shift 8

    has_apicot = item == ITEM_APICOT_BERRY
    apicot_triggers = has_apicot and below_threshold
    if apicot_triggers:
        boosts1 = apply_boost_to_packed(boosts1, 12, 1)  # SpD shift 12

    has_salac = item == ITEM_SALAC_BERRY
    salac_triggers = has_salac and below_threshold
    if salac_triggers:
        boosts2 = apply_boost_to_packed(boosts2, 0, 1)  # Spe shift 0

    # Starf Berry: +2 random non-evasion stat. Showdown picks uniformly
    # from Atk/Def/SpA/SpD/Spe (accuracy is excluded in items.ts:5531).
    # Pokepy uses a deterministic pick (mod with poff) to stay JIT-friendly
    # since we can't call the Gen5PRNG from here without threading it through.
    # This is a pragmatic approximation matching prior niche-item style.
    has_starf = item == ITEM_STARF_BERRY
    starf_triggers = has_starf and below_threshold
    if starf_triggers:
        pick = (int(hp) + int(max_hp) + int(poff)) % 5
        if pick == 0:
            boosts1 = apply_boost_to_packed(boosts1, 0, 2)   # Atk
        elif pick == 1:
            boosts1 = apply_boost_to_packed(boosts1, 4, 2)   # Def
        elif pick == 2:
            boosts1 = apply_boost_to_packed(boosts1, 8, 2)   # SpA
        elif pick == 3:
            boosts1 = apply_boost_to_packed(boosts1, 12, 2)  # SpD
        else:
            boosts2 = apply_boost_to_packed(boosts2, 0, 2)   # Spe

    any_triggers = (
        liechi_triggers or ganlon_triggers or petaya_triggers
        or apicot_triggers or salac_triggers or starf_triggers
    )
    new_item = 0 if any_triggers else item

    battle[poff + 13] = boosts1
    battle[poff + 14] = boosts2
    battle[poff + 6] = new_item

    if any_triggers:
        _apply_cheek_pouch_heal(battle, poff)

def apply_pinch_healing_berries(battle: np.ndarray, pokemon_offset: int, game_data) -> None:
    """Port of _apply_pinch_healing_berries (line ~9376).

    Figy/Wiki/Mago/Aguav/Iapapa berries heal 33% max HP at <=25% HP
    (50% with Gluttony). Showdown items.ts:1856/7273/3403/155/2630 —
    same heal amount, only difference is which nature flaw triggers
    confusion. Gen 9 still keeps the nature confusion mechanic
    (`pokemon.getNature().minus === 'atk' → addVolatile('confusion')`),
    but pokepy doesn't track per-mon natures so we skip the confusion
    side effect. Also fires Oran Berry (10 HP flat heal at <=50% HP).
    """
    poff = int(pokemon_offset)
    hp = int(battle[poff + 1])
    max_hp = int(battle[poff + 2])
    ability = int(battle[poff + 5])
    item = int(battle[poff + 6])

    # Unnerve blocks all berry eats (Showdown abilities.ts:5185).
    if _opponent_has_unnerve(battle, poff):
        return

    has_gluttony = ability == ABILITY_GLUTTONY
    hp_threshold_normal = int(max_hp / 4)
    hp_threshold_gluttony = int(max_hp / 2)
    hp_threshold = hp_threshold_gluttony if has_gluttony else hp_threshold_normal
    below_threshold = (hp <= hp_threshold) and (hp > 0)

    is_figy_family = item in (
        ITEM_FIGY_BERRY,
        ITEM_WIKI_BERRY,
        ITEM_MAGO_BERRY,
        ITEM_AGUAV_BERRY,
        ITEM_IAPAPA_BERRY,
    )
    figy_triggers = is_figy_family and below_threshold

    figy_heal = max(int(max_hp / 3), 1)
    figy_hp = min(max_hp, hp + figy_heal)

    # Oran Berry (items.ts, oranberry): heals 10 flat at <= 1/2 max HP.
    has_oran = item == ITEM_ORAN_BERRY
    oran_trigger = has_oran and (hp <= int(max_hp / 2)) and (hp > 0)
    oran_hp = min(max_hp, hp + 10)

    if figy_triggers:
        battle[poff + 1] = figy_hp
        battle[poff + 6] = 0
        _apply_cheek_pouch_heal(battle, poff)
    elif oran_trigger:
        battle[poff + 1] = oran_hp
        battle[poff + 6] = 0
        _apply_cheek_pouch_heal(battle, poff)

def apply_life_orb_recoil(
    battle: np.ndarray,
    user_offset: int,
    damage_dealt: int,
    hit: bool,
    game_data,
    move_id: int = None,
    move_effects=None,
) -> None:
    """Port of _apply_life_orb_recoil (line ~10870).

    Life Orb deals 1/10 max HP recoil after dealing damage. Suppressed by
    Magic Guard, and by Sheer Force when the move has secondary effects.
    """
    uoff = int(user_offset)
    damage_dealt = int(damage_dealt)
    hit = bool(hit)

    item = int(battle[uoff + 6])
    has_life_orb = item == ITEM_LIFE_ORB

    ability = int(battle[uoff + 5])
    has_magic_guard = ability == ABILITY_MAGIC_GUARD

    has_sheer_force = ability == ABILITY_SHEER_FORCE
    sf_suppresses = False
    if move_id is not None and move_effects is not None:
        mid = int(move_id)
        has_secondary = (
            int(move_effects.status_chance[mid]) > 0
            or int(move_effects.stat_chance[mid]) > 0
            or int(move_effects.volatile_chance[mid]) > 0
            or mid == MOVE_TRI_ATTACK
        )
        sf_suppresses = has_sheer_force and has_secondary

    max_hp = int(battle[uoff + 2])
    current_hp = int(battle[uoff + 1])

    # Showdown items.ts lifeorb: this.damage(source.baseMaxhp / 10).
    # damage() floors via calculateDamage -> Math.floor. Use integer division
    # to match exactly (avoids float rounding at maxhp edges like 353).
    recoil = max(max_hp // 10, 1)
    new_hp = max(0, current_hp - recoil)

    should_recoil = (
        has_life_orb
        and hit
        and (damage_dealt > 0)
        and (not has_magic_guard)
        and (not sf_suppresses)
    )
    final_hp = new_hp if should_recoil else current_hp

    battle[uoff + 1] = final_hp
