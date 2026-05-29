"""Volatile-status mechanics in pokepy.

Each test sets up a tiny one-mon-vs-one-mon Gen 9 OU singles scenario, runs
one or more turns of `step_battle_gen9`, and asserts on the resulting flat
battle buffer. Volatile flags live in OFF_FIELD slots:
    F_LEECH_SEED_0/1, F_SUBSTITUTE_0/1, F_DISABLE_0/1, F_DISABLE_TURNS_0/1,
    F_VOLATILE_0/1 (bit-packed: confusion/taunt/encore turns),
    F_EXTENDED_VOLATILE_0/1 (bit-packed bag: torment, attract, ingrain, ...),
    F_PERISH_COUNT_0/1, F_DESTINY_BOND_0/1, F_YAWN_TURNS_0/1.

Showdown source of truth:
    pokemon-showdown/data/moves.ts (search each move's lowercase id)
    pokemon-showdown/data/conditions.ts (search "substitute", "leechseed",
        "confusion", "taunt", "encore", "perishsong", "destinybond",
        "yawn", "saltcure", "attract", "torment", "healblock", "embargo",
        "imprison", "ingrain", "aquaring", "curse", "lockon", "foresight")
"""

from __future__ import annotations

import pytest

from tests.conftest import MonSpec
from pokepy.core.constants import (
    OFF_FIELD,
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_META,
    POKEMON_SIZE,
    M_ACTIVE0,
    M_ACTIVE1,
    F_CHOICE_LOCK_0,
    F_LEECH_SEED_0,
    F_LEECH_SEED_1,
    F_SUBSTITUTE_0,
    F_SUBSTITUTE_1,
    F_DISABLE_0,
    F_DISABLE_1,
    F_DISABLE_TURNS_0,
    F_DISABLE_TURNS_1,
    F_VOLATILE_0,
    F_VOLATILE_1,
    F_EXTENDED_VOLATILE_0,
    F_EXTENDED_VOLATILE_1,
    F_PERISH_COUNT_0,
    F_PERISH_COUNT_1,
    F_DESTINY_BOND_0,
    F_DESTINY_BOND_1,
    F_YAWN_TURNS_0,
    F_YAWN_TURNS_1,
    EXT_VOL_TORMENT,
    EXT_VOL_ATTRACT,
    EXT_VOL_HEAL_BLOCK,
    EXT_VOL_EMBARGO,
    EXT_VOL_IMPRISON,
    EXT_VOL_INGRAIN,
    EXT_VOL_AQUA_RING,
    EXT_VOL_CURSE,
    EXT_VOL_SALT_CURE,
    EXT_VOL_FORESIGHT,
    EXT_VOL_LOCK_ON,
    EXT_VOL_MEAN_LOOK,
    STATUS_SLEEP,
    GENDER_MALE,
    GENDER_FEMALE,
)
from pokepy.core.bitpack import (
    get_confusion_turns,
    get_taunt_turns,
    get_encore_turns,
)
from pokepy.effects.volatiles import decrement_taunt_encore
from pokepy.utils.gen5_prng import Gen5PRNG

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vol(state, side):
    off = F_VOLATILE_0 if side == 0 else F_VOLATILE_1
    return int(state.battle_state[OFF_FIELD + off]) & 0xFFFF


def _ext_vol(state, side):
    off = F_EXTENDED_VOLATILE_0 if side == 0 else F_EXTENDED_VOLATILE_1
    return int(state.battle_state[OFF_FIELD + off]) & 0xFFFF


def _set_gender(state, side, gender):
    base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    active = int(state.battle_state[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])
    p = base + active * POKEMON_SIZE + 15
    flags = int(state.battle_state[p]) & 0xFFFF
    flags = (flags & ~(0x3 << 4)) | ((int(gender) & 0x3) << 4)
    state.battle_state[p] = flags


class _LoggingPRNG:
    def __init__(self, seed=(1, 2, 3, 4)):
        self._prng = Gen5PRNG(seed)
        self.calls = []

    def random(self, *args):
        result = self._prng.random(*args)
        self.calls.append((args, int(result)))
        return result

    def __getattr__(self, name):
        return getattr(self._prng, name)


# ===========================================================================
# Substitute
# ===========================================================================


def test_substitute_costs_one_quarter_max_hp(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["substitute", "tackle", "tackle", "tackle"])],
        [MonSpec("garchomp", ["tackle", "tackle", "tackle", "tackle"])],
        seed=11,
    )
    max_hp = max_hp_of(state, 0)
    hp_pre = hp_of(state, 0)
    step_turn(state, prng, 0, 0)  # blissey uses substitute
    sub = int(state.battle_state[OFF_FIELD + F_SUBSTITUTE_0])
    assert sub == max(1, max_hp // 4)
    # HP dropped by ~max/4 (plus tackle damage from garchomp)
    assert hp_of(state, 0) <= hp_pre - max(1, max_hp // 4)


def test_substitute_blocks_damage(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["substitute", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=12,
    )
    step_turn(state, prng, 0, 0)
    hp_after_sub = hp_of(state, 0)
    sub_after = int(state.battle_state[OFF_FIELD + F_SUBSTITUTE_0])
    # Sub still up; further tackles must be soaked by sub, not the mon
    step_turn(state, prng, 1, 0)  # blissey switches to tackle (slot 1)
    assert hp_of(state, 0) == hp_after_sub
    assert int(state.battle_state[OFF_FIELD + F_SUBSTITUTE_0]) <= sub_after


def test_substitute_breaks_at_zero_hp(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["substitute", "tackle", "tackle", "tackle"])],
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        seed=13,
    )
    step_turn(state, prng, 0, 0)  # sub goes up
    # Force the sub HP to 1, then a strong move should break it
    state.battle_state[OFF_FIELD + F_SUBSTITUTE_0] = 1
    step_turn(state, prng, 1, 0)
    assert int(state.battle_state[OFF_FIELD + F_SUBSTITUTE_0]) == 0


def test_substitute_fails_if_hp_below_quarter(
    fresh_battle, step_turn, hp_of, max_hp_of
):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["substitute", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=14,
    )
    # Drop HP below 25% manually
    base = OFF_SIDE0 + int(state.battle_state[OFF_META + M_ACTIVE0]) * POKEMON_SIZE
    state.battle_state[base + 1] = max(1, max_hp_of(state, 0) // 5)
    step_turn(state, prng, 0, 0)
    assert int(state.battle_state[OFF_FIELD + F_SUBSTITUTE_0]) == 0


def test_substitute_blocks_leech_seed(fresh_battle, step_turn):
    """Pre-set substitute then verify leech seed is blocked."""
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["splash"] * 4)],
        [MonSpec("venusaur", ["leechseed", "tackle", "tackle", "tackle"])],
        seed=15,
    )
    # Pre-set sub on side 0 (any positive value works as the flag)
    state.battle_state[OFF_FIELD + F_SUBSTITUTE_0] = 100
    step_turn(state, prng, 0, 0)  # venusaur leech seed
    assert int(state.battle_state[OFF_FIELD + F_LEECH_SEED_0]) == 0


@pytest.mark.xfail(
    strict=False, reason="sound move bypass of substitute not implemented"
)
def test_substitute_pierced_by_sound_move(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["substitute", "tackle", "tackle", "tackle"])],
        [MonSpec("primarina", ["hypervoice", "tackle", "tackle", "tackle"])],
        seed=16,
    )
    step_turn(state, prng, 0, 0)
    hp_pre = hp_of(state, 0)
    step_turn(state, prng, 1, 0)
    # Hyper Voice should pierce the sub and damage blissey directly
    assert hp_of(state, 0) < hp_pre


# ===========================================================================
# Leech Seed
# ===========================================================================


def test_leech_seed_sets_flag(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("venusaur", ["leechseed", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=2,
    )
    step_turn(state, prng, 0, 0)
    assert int(state.battle_state[OFF_FIELD + F_LEECH_SEED_1]) > 0


def test_leech_seed_drains_each_turn(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [MonSpec("venusaur", ["leechseed", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=22,
    )
    step_turn(state, prng, 0, 0)  # apply seed
    snorlax_hp_pre = hp_of(state, 1)
    venu_hp_pre = hp_of(state, 0)
    snorlax_max = max_hp_of(state, 1)
    step_turn(state, prng, 1, 0)  # both tackle, then EOT drain
    drain = max(1, snorlax_max // 8)
    # Snorlax should have lost at least the leech seed drain
    assert hp_of(state, 1) <= snorlax_hp_pre - drain
    # Venusaur should have been healed by drain (if not full)
    assert hp_of(state, 0) >= venu_hp_pre - 50  # tackle dmg minus leech heal


def test_leech_seed_fails_on_grass_type(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("venusaur", ["leechseed", "tackle", "tackle", "tackle"])],
        [MonSpec("ferrothorn", ["tackle", "tackle", "tackle", "tackle"])],
        seed=23,
    )
    step_turn(state, prng, 0, 0)
    assert int(state.battle_state[OFF_FIELD + F_LEECH_SEED_1]) == 0


# ===========================================================================
# Confusion
# ===========================================================================


def test_confusion_sets_volatile_turns(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("gardevoir", ["confuseray", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=31,
    )
    step_turn(state, prng, 0, 0)
    # apply sets 1-4 turns (matching Showdown's 1-4 actual confusion checks
    # — Showdown rolls 2-5 but decrements before the check, leaving 1-4
    # rolls). Then EOT decrement → 0-3.
    turns = get_confusion_turns(_vol(state, 1))
    assert 0 <= turns <= 3


def test_confusion_decrements_each_turn(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("gardevoir", ["confuseray", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=32,
    )
    step_turn(state, prng, 0, 0)
    t0 = get_confusion_turns(_vol(state, 1))
    step_turn(state, prng, 1, 0)
    t1 = get_confusion_turns(_vol(state, 1))
    # Turns either decremented or hit-self-and-cleared (NOT incremented)
    assert t1 <= t0


# ===========================================================================
# Taunt
# ===========================================================================


def test_taunt_sets_three_turns(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["taunt", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=41,
    )
    step_turn(state, prng, 0, 0)
    # set to 3, then decremented at EOT → 2
    assert get_taunt_turns(_vol(state, 1)) == 2


def test_taunt_decrements_each_turn(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["taunt", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=42,
    )
    step_turn(state, prng, 0, 0)
    pre = get_taunt_turns(_vol(state, 1))  # 2 (after EOT decrement)
    step_turn(state, prng, 1, 0)
    post = get_taunt_turns(_vol(state, 1))
    assert post == max(0, pre - 1)


# ===========================================================================
# Encore
# ===========================================================================


def test_encore_sets_three_turns(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("clefable", ["encore", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=51,
    )
    # Snorlax must use a move first so encore has a "last move" to lock
    step_turn(state, prng, 1, 0)  # clefable tackle, snorlax tackle
    step_turn(state, prng, 0, 0)  # clefable encore
    # set to 3, then decremented at EOT → 2
    assert get_encore_turns(_vol(state, 1)) == 2


# ===========================================================================
# Disable
# ===========================================================================


@pytest.mark.xfail(strict=False, reason="disable move not wired into engine pipeline")
def test_disable_sets_disable_field(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["disable", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=61,
    )
    step_turn(state, prng, 1, 0)  # snorlax tackles so it has a last move
    step_turn(state, prng, 0, 0)  # gengar disables
    assert int(state.battle_state[OFF_FIELD + F_DISABLE_TURNS_1]) > 0


def test_disablemove_shuffle_consumes_prng_for_disable_plus_choice_lock(fresh_battle):
    state, _ = fresh_battle(
        [
            MonSpec(
                "latios",
                ["dracometeor", "recover", "flipturn", "icebeam"],
                item="choicescarf",
            )
        ],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=62,
    )

    state.battle_state[OFF_FIELD + F_CHOICE_LOCK_0] = 0
    state.battle_state[OFF_FIELD + F_DISABLE_0] = 0
    state.battle_state[OFF_FIELD + F_DISABLE_TURNS_0] = 4

    prng = _LoggingPRNG()
    decrement_taunt_encore(state.battle_state, prng)

    assert len(prng.calls) == 1
    assert prng.calls[0][0] == (0, 2)
    assert int(state.battle_state[OFF_FIELD + F_DISABLE_TURNS_0]) == 3


# ===========================================================================
# Yawn
# ===========================================================================


@pytest.mark.xfail(
    strict=False, reason="yawn move not routed via apply_extended_volatile"
)
def test_yawn_sleeps_on_next_turn_eot(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("slowbro", ["yawn", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=71,
    )
    step_turn(state, prng, 0, 0)
    step_turn(state, prng, 1, 0)
    assert status_of(state, 1) == STATUS_SLEEP


# ===========================================================================
# Perish Song
# ===========================================================================


def test_perish_song_sets_counters(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["perishsong", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=81,
    )
    step_turn(state, prng, 0, 0)
    p0 = int(state.battle_state[OFF_FIELD + F_PERISH_COUNT_0])
    p1 = int(state.battle_state[OFF_FIELD + F_PERISH_COUNT_1])
    # Set to 4, then decremented at EOT to 3
    assert p0 == 3 and p1 == 3


def test_perish_song_counters_decrement(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["perishsong", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=82,
    )
    step_turn(state, prng, 0, 0)
    pre = int(state.battle_state[OFF_FIELD + F_PERISH_COUNT_0])
    step_turn(state, prng, 1, 0)
    post = int(state.battle_state[OFF_FIELD + F_PERISH_COUNT_0])
    assert post == pre - 1


# ===========================================================================
# Destiny Bond
# ===========================================================================


def test_destiny_bond_sets_flag(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["destinybond", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=91,
    )
    step_turn(state, prng, 0, 0)
    assert int(state.battle_state[OFF_FIELD + F_DESTINY_BOND_0]) == 1


# ===========================================================================
# Curse (ghost vs non-ghost)
# ===========================================================================


def test_curse_ghost_sets_curse_bit(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["curse", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=101,
    )
    step_turn(state, prng, 0, 0)
    assert (_ext_vol(state, 1) & EXT_VOL_CURSE) != 0


def test_curse_non_ghost_boosts_atk_def(fresh_battle, step_turn, boost_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["curse", "tackle", "tackle", "tackle"])],
        [MonSpec("garchomp", ["tackle", "tackle", "tackle", "tackle"])],
        seed=102,
    )
    step_turn(state, prng, 0, 0)
    assert boost_of(state, 0, "atk") == 1
    assert boost_of(state, 0, "def") == 1
    assert boost_of(state, 0, "spe") == -1


# ===========================================================================
# Salt Cure
# ===========================================================================


def test_salt_cure_sets_volatile_bit(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("garganacl", ["saltcure", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=111,
    )
    step_turn(state, prng, 0, 0)
    assert (_ext_vol(state, 1) & EXT_VOL_SALT_CURE) != 0


def test_salt_cure_drains_one_eighth_each_turn(
    fresh_battle, step_turn, hp_of, max_hp_of
):
    state, prng = fresh_battle(
        [MonSpec("garganacl", ["saltcure", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=112,
    )
    step_turn(state, prng, 0, 0)  # apply salt cure (+ tackles)
    hp_pre = hp_of(state, 1)
    max_hp = max_hp_of(state, 1)
    step_turn(state, prng, 1, 0)  # garganacl tackles, EOT salt cure drains
    drain = max(1, max_hp // 8)
    assert hp_of(state, 1) <= hp_pre - drain


# ===========================================================================
# Attract
# ===========================================================================


def test_attract_sets_attract_bit(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("clefable", ["attract", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=121,
    )
    _set_gender(state, 0, GENDER_MALE)
    _set_gender(state, 1, GENDER_FEMALE)
    step_turn(state, prng, 0, 0)
    assert (_ext_vol(state, 1) & EXT_VOL_ATTRACT) != 0


def test_attract_fails_same_gender(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("clefable", ["attract", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=122,
    )
    _set_gender(state, 0, GENDER_MALE)
    _set_gender(state, 1, GENDER_MALE)
    step_turn(state, prng, 0, 0)
    assert (_ext_vol(state, 1) & EXT_VOL_ATTRACT) == 0


# ===========================================================================
# Torment
# ===========================================================================


def test_torment_sets_torment_bit(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("sableye", ["torment", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=131,
    )
    step_turn(state, prng, 0, 0)
    assert (_ext_vol(state, 1) & EXT_VOL_TORMENT) != 0


# ===========================================================================
# Heal Block
# ===========================================================================


def test_heal_block_sets_bit(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("dragapult", ["healblock", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=141,
    )
    step_turn(state, prng, 0, 0)
    assert (_ext_vol(state, 1) & EXT_VOL_HEAL_BLOCK) != 0


# ===========================================================================
# Embargo
# ===========================================================================


def test_embargo_sets_bit(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("sableye", ["embargo", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=151,
    )
    step_turn(state, prng, 0, 0)
    assert (_ext_vol(state, 1) & EXT_VOL_EMBARGO) != 0


# ===========================================================================
# Imprison (self-target)
# ===========================================================================


def test_imprison_sets_bit_on_user(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["imprison", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=161,
    )
    step_turn(state, prng, 0, 0)
    assert (_ext_vol(state, 0) & EXT_VOL_IMPRISON) != 0


# ===========================================================================
# Ingrain (self-target)
# ===========================================================================


def test_ingrain_sets_bit_on_user(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("ferrothorn", ["ingrain", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=171,
    )
    step_turn(state, prng, 0, 0)
    assert (_ext_vol(state, 0) & EXT_VOL_INGRAIN) != 0


# ===========================================================================
# Aqua Ring (self-target)
# ===========================================================================


def test_aqua_ring_sets_bit_on_user(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("toxapex", ["aquaring", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=181,
    )
    step_turn(state, prng, 0, 0)
    assert (_ext_vol(state, 0) & EXT_VOL_AQUA_RING) != 0


# ===========================================================================
# Mean Look
# ===========================================================================


def test_mean_look_sets_bit_on_target(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("crobat", ["meanlook", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=191,
    )
    step_turn(state, prng, 0, 0)
    assert (_ext_vol(state, 1) & EXT_VOL_MEAN_LOOK) != 0


# ===========================================================================
# Lock-On (self-target)
# ===========================================================================


def test_lock_on_sets_bit_on_user(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("magnezone", ["lockon", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=201,
    )
    step_turn(state, prng, 0, 0)
    assert (_ext_vol(state, 0) & EXT_VOL_LOCK_ON) != 0


# ===========================================================================
# Foresight
# ===========================================================================


def test_foresight_sets_bit_on_target(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("ursaring", ["foresight", "tackle", "tackle", "tackle"])],
        [MonSpec("gengar", ["tackle", "tackle", "tackle", "tackle"])],
        seed=211,
    )
    step_turn(state, prng, 0, 0)
    assert (_ext_vol(state, 1) & EXT_VOL_FORESIGHT) != 0
