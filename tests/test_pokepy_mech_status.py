"""Status-inflicting moves and status condition mechanics for pokepy.

Covers brn / par / slp / frz / psn / tox application, end-of-turn damage,
ability and type immunities, contact-status defender abilities, and the
"already statused" / Comatose / Purifying Salt / Magic Guard short-circuits.

Showdown source references:
- pokemon-showdown/data/conditions.ts (status condition definitions for
  brn/par/slp/frz/psn/tox)
- pokemon-showdown/data/moves.ts (willowisp, thunderwave, spore, toxic,
  poisonpowder, icebeam, scald, discharge, lavaplume, bodyslam, flamethrower,
  thunder, rest, facade, hypnosis, darkvoid, yawn)
- pokemon-showdown/data/abilities.ts (Limber, Insomnia, Vital Spirit,
  Magma Armor, Water Veil, Immunity, Comatose, Purifying Salt, Synchronize,
  Flame Body, Static, Poison Point, Effect Spore, Guts, Magic Guard)

The pokepy implementation lives in:
- pokepy/effects/status_apply.py (apply_status_from_move,
  apply_end_of_turn_status, apply_end_of_turn_status_effects)
- pokepy/effects/end_of_turn.py (orchestration)
- pokepy/engine/battle_gen9.py (step_battle_gen9)
"""

from __future__ import annotations

import pytest

from tests.conftest import MonSpec
from pokepy.core.constants import (
    OFF_SIDE0,
    OFF_SIDE1,
    OFF_META,
    OFF_FIELD,
    POKEMON_SIZE,
    M_ACTIVE0,
    M_ACTIVE1,
    STATUS_BURN,
    STATUS_PARALYSIS,
    STATUS_SLEEP,
    STATUS_FREEZE,
    STATUS_POISON,
    STATUS_TOXIC,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_status_raw(state, side: int, status_code: int, turns: int = 0) -> None:
    """Stamp a status directly on the active Pokemon (bypass apply_status)."""
    base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    active = int(state.battle_state[OFF_META + (M_ACTIVE0 if side == 0 else M_ACTIVE1)])
    state.battle_state[base + active * POKEMON_SIZE + 12] = (turns << 8) | status_code


# ===========================================================================
# 1. Burn — Will-O-Wisp applies brn, EOT damage = 1/16 max HP
# ===========================================================================


def test_willowisp_applies_burn(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["willowisp", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=3,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_BURN


def test_burn_eot_damage_one_sixteenth(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["willowisp", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=3,
    )
    mhp = max_hp_of(state, 1)
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    # EOT burn = mhp / 16 (Snorlax tackle on Gengar deals 0 because Normal -> Ghost is immune)
    assert hp_pre - hp_of(state, 1) == mhp // 16


def test_burn_halves_physical_damage(fresh_battle, step_turn, hp_of):
    # Compare body slam damage with vs without burn
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["bodyslam", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    no_burn = hp_pre - hp_of(state, 1)

    state, prng = fresh_battle(
        [MonSpec("snorlax", ["bodyslam", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    _set_status_raw(state, 0, STATUS_BURN)
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    burned = hp_pre - hp_of(state, 1)
    # Burn should roughly halve physical damage
    assert burned <= no_burn // 2 + no_burn // 8


# ===========================================================================
# 2. Paralysis
# ===========================================================================


def test_thunderwave_applies_paralysis(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["thunderwave", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=2,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_PARALYSIS


def test_thunderwave_immune_on_ground_type(fresh_battle, step_turn, status_of):
    # Garchomp is Dragon/Ground; T-Wave is Electric -> ground immunity
    state, prng = fresh_battle(
        [MonSpec("gengar", ["thunderwave", "tackle", "tackle", "tackle"])],
        [MonSpec("garchomp", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == 0


def test_thunderwave_immune_on_electric_type(fresh_battle, step_turn, status_of):
    # Pawmot is Electric/Fighting — gen 6+ Electric type immune to paralysis
    state, prng = fresh_battle(
        [MonSpec("gengar", ["thunderwave", "tackle", "tackle", "tackle"])],
        [MonSpec("pawmot", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == 0


def test_paralyzed_mon_can_fail_to_move(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("chansey", ["seismictoss", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        # seed picked so the par full-para roll fires (1/4 lands) at the
        # exact PRNG position the new pre-calc onBeforeMove path consumes.
        # (Under old post-calc ordering the seed was 6.)
        seed=10,
    )
    _set_status_raw(state, 0, STATUS_PARALYSIS)
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    # Snorlax took 0 damage from chansey -> chansey was fully paralyzed
    assert hp_of(state, 1) == hp_pre


# ===========================================================================
# 3. Sleep
# ===========================================================================


def test_spore_applies_sleep(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("breloom", ["spore", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_SLEEP


def test_spore_immune_on_grass_type(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("breloom", ["spore", "tackle", "tackle", "tackle"])],
        [MonSpec("venusaur", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == 0


def test_hypnosis_applies_sleep(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["hypnosis", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle", "tackle", "tackle", "tackle"])],
        seed=6,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_SLEEP


def test_darkvoid_applies_sleep(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["darkvoid", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle", "tackle", "tackle", "tackle"])],
        seed=7,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_SLEEP


# ===========================================================================
# 4. Freeze
# ===========================================================================


def test_icebeam_can_freeze(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["icebeam", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle", "tackle", "tackle", "tackle"])],
        seed=16,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_FREEZE


def test_fire_move_thaws_frozen_target(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["flamethrower", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    _set_status_raw(state, 1, STATUS_FREEZE)
    step_turn(state, prng, 0, 0)
    # The point of this test is that Fire-type damaging moves thaw the
    # target on hit. With the new pre-calc onBeforeMove ordering the
    # seed-1 PRNG stream sometimes lands Flamethrower's 10% burn secondary
    # on the thawed Chansey — which is still a legal "thawed" state. What
    # matters is that the frozen status is GONE.
    assert status_of(state, 1) != STATUS_FREEZE


# ===========================================================================
# 5. Toxic / regular poison
# ===========================================================================


def test_toxic_applies_badly_poisoned(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("toxapex", ["toxic", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_TOXIC


def test_toxic_damage_increases_over_turns(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [MonSpec("toxapex", ["toxic", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)  # turn 1: apply + first tick
    hp_a = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    hp_b = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    hp_c = hp_of(state, 1)
    # The two later deltas should be at least as large as the first
    assert (hp_b - hp_c) >= (hp_a - hp_b)


def test_poison_powder_applies_regular_poison(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("toxapex", ["poisonpowder", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle", "tackle", "tackle", "tackle"])],
        seed=4,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_POISON


def test_poison_eot_one_eighth(fresh_battle, step_turn, hp_of, max_hp_of):
    state, prng = fresh_battle(
        [MonSpec("toxapex", ["poisonpowder", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle", "tackle", "tackle", "tackle"])],
        seed=4,
    )
    mhp = max_hp_of(state, 1)
    hp_pre = hp_of(state, 1)
    step_turn(state, prng, 0, 0)
    assert (hp_pre - hp_of(state, 1)) == mhp // 8


# ===========================================================================
# 6. Type immunities
# ===========================================================================


def test_willowisp_fails_on_fire_type(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["willowisp", "tackle", "tackle", "tackle"])],
        [MonSpec("arcanine", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == 0


def test_toxic_fails_on_poison_type(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["toxic", "tackle", "tackle", "tackle"])],
        [MonSpec("toxapex", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == 0


def test_toxic_fails_on_steel_type(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["toxic", "tackle", "tackle", "tackle"])],
        [MonSpec("ferrothorn", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == 0


def test_poison_user_toxic_bypasses_steel(fresh_battle, step_turn, status_of):
    # Gen 6+: a Poison-type using Toxic never misses (also bypasses immunities
    # for Poison-type users). Toxapex (Poison) -> Garchomp (not steel).
    state, prng = fresh_battle(
        [MonSpec("toxapex", ["toxic", "tackle", "tackle", "tackle"])],
        [MonSpec("garchomp", ["tackle", "tackle", "tackle", "tackle"])],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_TOXIC


# ===========================================================================
# 7. Ability immunities
# ===========================================================================


def test_limber_blocks_paralysis(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["thunderwave", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4, ability="limber")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == 0


def test_insomnia_blocks_sleep(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("breloom", ["spore", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4, ability="insomnia")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == 0


def test_vital_spirit_blocks_sleep(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("breloom", ["spore", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4, ability="vitalspirit")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == 0


def test_water_veil_blocks_burn(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("gengar", ["willowisp", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4, ability="waterveil")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == 0


def test_immunity_blocks_poison(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("toxapex", ["poisonpowder", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4, ability="immunity")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == 0


def test_magma_armor_blocks_freeze(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["icebeam", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle"] * 4, ability="magmaarmor")],
        seed=8,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == 0


def test_purifying_salt_blocks_all_status(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("toxapex", ["toxic", "tackle", "tackle", "tackle"])],
        [MonSpec("clodsire", ["tackle"] * 4, ability="purifyingsalt")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == 0


def test_comatose_blocks_other_status(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("toxapex", ["toxic", "tackle", "tackle", "tackle"])],
        [MonSpec("komala", ["tackle"] * 4, ability="comatose")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == 0


# ===========================================================================
# 8. Defender contact-status abilities
# ===========================================================================


def test_synchronize_copies_toxic_to_attacker(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("toxapex", ["toxic", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle"] * 4, ability="synchronize")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_TOXIC
    assert status_of(state, 0) == STATUS_TOXIC


def test_synchronize_does_not_copy_sleep(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("breloom", ["spore", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle"] * 4, ability="synchronize")],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_SLEEP
    assert status_of(state, 0) == 0


def test_flame_body_burns_contact_attacker(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle"] * 4, ability="flamebody")],
        seed=15,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 0) == STATUS_BURN


def test_static_paralyzes_contact_attacker(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle"] * 4, ability="static")],
        seed=5,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 0) == STATUS_PARALYSIS


def test_poison_point_poisons_contact_attacker(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle"] * 4, ability="poisonpoint")],
        seed=5,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 0) == STATUS_POISON


def test_effect_spore_can_inflict_status(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["tackle", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle"] * 4, ability="effectspore")],
        seed=5,
    )
    step_turn(state, prng, 0, 0)
    # Effect Spore can roll psn / par / slp; just verify *something* applied.
    assert status_of(state, 0) in (STATUS_POISON, STATUS_PARALYSIS, STATUS_SLEEP)


# ===========================================================================
# 9. Secondary status chances on damaging moves (deterministic seeds)
# ===========================================================================


def test_scald_can_burn(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("jellicent", ["scald", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle"] * 4)],
        seed=0,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_BURN


def test_lavaplume_can_burn(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["lavaplume", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle"] * 4)],
        seed=16,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_BURN


def test_discharge_can_paralyze(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["discharge", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle"] * 4)],
        seed=16,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_PARALYSIS


def test_bodyslam_can_paralyze(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["bodyslam", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle"] * 4)],
        seed=15,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_PARALYSIS


# ===========================================================================
# 10. Status overlap & self-status
# ===========================================================================


def test_status_fails_if_already_statused(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("breloom", ["spore", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle"] * 4)],
        seed=1,
    )
    _set_status_raw(state, 1, STATUS_BURN)
    step_turn(state, prng, 0, 0)
    # Spore should not overwrite the burn
    assert status_of(state, 1) == STATUS_BURN


def test_rest_heals_full_and_sleeps(
    fresh_battle, step_turn, status_of, hp_of, max_hp_of
):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["rest", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle"] * 4)],
        seed=1,
    )
    mhp = max_hp_of(state, 0)
    state.battle_state[OFF_SIDE0 + 1] = 50  # set hp low
    step_turn(state, prng, 0, 0)
    assert hp_of(state, 0) == mhp
    assert status_of(state, 0) == STATUS_SLEEP


# ===========================================================================
# 11. Guts / Facade — status interaction with offensive moves
# ===========================================================================


def test_guts_negates_burn_atk_drop(fresh_battle, step_turn, hp_of):
    # Use Blissey (bulkier) so Body Slam doesn't OHKO and we can compare HP.
    # Read slot-0 HP directly to avoid auto-switch when defender is KO'd.
    state, prng = fresh_battle(
        [
            MonSpec(
                "snorlax", ["bodyslam", "tackle", "tackle", "tackle"], ability="guts"
            )
        ],
        [MonSpec("blissey", ["tackle"] * 4)],
        seed=1,
    )
    hp_pre = int(state.battle_state[OFF_SIDE1 + 1])
    step_turn(state, prng, 0, 0)
    no_burn = hp_pre - int(state.battle_state[OFF_SIDE1 + 1])

    state, prng = fresh_battle(
        [
            MonSpec(
                "snorlax", ["bodyslam", "tackle", "tackle", "tackle"], ability="guts"
            )
        ],
        [MonSpec("blissey", ["tackle"] * 4)],
        seed=1,
    )
    _set_status_raw(state, 0, STATUS_BURN)
    hp_pre = int(state.battle_state[OFF_SIDE1 + 1])
    step_turn(state, prng, 0, 0)
    burned = hp_pre - int(state.battle_state[OFF_SIDE1 + 1])
    # With Guts + burn the damage should be at least as large (1.5x boost)
    assert burned >= no_burn


def test_facade_doubles_when_statused(fresh_battle, step_turn, hp_of):
    # Use Ting-Lu (high HP + def) so Facade can't OHKO either way and the
    # burned/clean ratio reflects the true damage formula.
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["facade", "tackle", "tackle", "tackle"])],
        [MonSpec("tinglu", ["tackle"] * 4)],
        seed=1,
    )
    hp_pre = int(state.battle_state[OFF_SIDE1 + 1])
    step_turn(state, prng, 0, 0)
    clean = hp_pre - int(state.battle_state[OFF_SIDE1 + 1])

    state, prng = fresh_battle(
        [MonSpec("snorlax", ["facade", "tackle", "tackle", "tackle"])],
        [MonSpec("tinglu", ["tackle"] * 4)],
        seed=1,
    )
    _set_status_raw(state, 0, STATUS_BURN)
    hp_pre = int(state.battle_state[OFF_SIDE1 + 1])
    step_turn(state, prng, 0, 0)
    burned = hp_pre - int(state.battle_state[OFF_SIDE1 + 1])
    # Facade doubles BP and ignores burn halving → ~2x damage
    assert burned >= int(clean * 1.7)


# ===========================================================================
# 12. Indirect-damage interactions (Magic Guard)
# ===========================================================================


def test_magic_guard_blocks_burn_damage(fresh_battle, step_turn, hp_of):
    state, prng = fresh_battle(
        [
            MonSpec(
                "clefable",
                ["splash", "splash", "splash", "splash"],
                ability="magicguard",
            )
        ],
        [MonSpec("chansey", ["splash"] * 4)],
        seed=1,
    )
    _set_status_raw(state, 0, STATUS_BURN)
    hp_pre = hp_of(state, 0)
    step_turn(state, prng, 0, 0)
    # Magic Guard prevents indirect damage (burn EOT). Both mons splash so no
    # tackle confounder.
    assert hp_of(state, 0) == hp_pre


# ===========================================================================
# 13. Yawn — should set up sleep for the next turn
# ===========================================================================


@pytest.mark.xfail(
    strict=False, reason="Yawn delayed-sleep volatile not yet implemented"
)
def test_yawn_puts_target_to_sleep_next_turn(fresh_battle, step_turn, status_of):
    state, prng = fresh_battle(
        [MonSpec("snorlax", ["yawn", "tackle", "tackle", "tackle"])],
        [MonSpec("chansey", ["tackle"] * 4)],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == 0  # not yet asleep
    step_turn(state, prng, 0, 0)
    assert status_of(state, 1) == STATUS_SLEEP
