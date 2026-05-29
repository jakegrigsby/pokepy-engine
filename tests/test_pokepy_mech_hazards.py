"""Entry hazards (Stealth Rock, Spikes, Toxic Spikes, Sticky Web) and removal.

Tests cover hazard setters (SR/Spikes/TSpikes/Web), per-type effectiveness for
Stealth Rock damage, layer-count damage scaling for Spikes, immunity from Heavy
Duty Boots / Magic Guard / Levitate / Flying / Steel (for Toxic Spikes), and
hazard removal via Rapid Spin / Defog / Court Change / Tidy Up.

Switch-in semantics: in conftest's `step_turn`, action 4..9 = switch to roster
slot 0..5. Slot 0 is the starting active mon, so switching to slot 1 (the
second mon in the MonSpec list) uses action 5.

Showdown source references:
- pokemon-showdown/data/moves.ts: search "stealthrock", "spikes", "toxicspikes",
  "stickyweb", "rapidspin", "defog", "courtchange", "tidyup", "mortalspin"
- pokemon-showdown/data/items.ts: "heavydutyboots"
- pokemon-showdown/data/abilities.ts: "magicguard", "levitate", "magicbounce"
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
    F_HAZARDS_0,
    F_HAZARDS_1,
    F_SCREENS_0,
    F_SCREENS_1,
    STATUS_POISON,
    STATUS_TOXIC,
)
from pokepy.core.bitpack import (
    get_spikes_layers,
    get_stealth_rock,
    get_sticky_web,
    get_toxic_spikes_layers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Switch-action constants for clarity. action 4 = switch to roster slot 0,
# action 5 = switch to roster slot 1, etc. Slot 0 is the starting active mon
# so the smallest meaningful switch is action 5.
SWITCH_TO_SLOT1 = 5
SWITCH_TO_SLOT2 = 6


def _hazards_word(state, side: int) -> int:
    return int(
        state.battle_state[OFF_FIELD + (F_HAZARDS_0 if side == 0 else F_HAZARDS_1)]
    )


def _hp_slot(state, side: int, slot: int) -> int:
    base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    return int(state.battle_state[base + slot * POKEMON_SIZE + 1])


def _max_hp_slot(state, side: int, slot: int) -> int:
    base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    return int(state.battle_state[base + slot * POKEMON_SIZE + 2])


def _status_slot(state, side: int, slot: int) -> int:
    base = OFF_SIDE0 if side == 0 else OFF_SIDE1
    return int(state.battle_state[base + slot * POKEMON_SIZE + 12]) & 0xFF


# ---------------------------------------------------------------------------
# 1. Stealth Rock sets the hazard layer
# ---------------------------------------------------------------------------


def test_stealth_rock_sets_layer(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["stealthrock", "tackle", "tackle", "tackle"])],
        [MonSpec("blissey", ["tackle"] * 4)],
        seed=1,
    )
    step_turn(state, prng, 0, 0)
    assert get_stealth_rock(_hazards_word(state, 1)) == 1


# ---------------------------------------------------------------------------
# 2. SR damage scales with rock-type effectiveness
# ---------------------------------------------------------------------------


def test_sr_damage_neutral_one_eighth(fresh_battle, step_turn):
    # Snorlax (Normal) takes 1/8 from Stealth Rock.
    state, prng = fresh_battle(
        [MonSpec("garchomp", ["stealthrock", "tackle", "tackle", "tackle"])],
        [MonSpec("blissey", ["tackle"] * 4), MonSpec("snorlax", ["tackle"] * 4)],
        seed=2,
    )
    step_turn(state, prng, 0, 0)
    max_hp = _max_hp_slot(state, 1, 1)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    hp_after = _hp_slot(state, 1, 1)
    assert hp_after == max_hp - max(max_hp // 8, 1)


def test_sr_damage_resist_one_sixteenth(fresh_battle, step_turn):
    # Garchomp (Dragon/Ground) → 0.5x rock → 1/16.
    state, prng = fresh_battle(
        [MonSpec("blissey", ["stealthrock", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4), MonSpec("garchomp", ["tackle"] * 4)],
        seed=3,
    )
    step_turn(state, prng, 0, 0)
    max_hp = _max_hp_slot(state, 1, 1)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    hp_after = _hp_slot(state, 1, 1)
    expected = max(int(max_hp * 0.5 / 8), 1)
    assert hp_after == max_hp - expected


def test_sr_damage_4x_weak_one_half(fresh_battle, step_turn):
    # Volcarona (Bug/Fire) → 4x rock → 1/2.
    state, prng = fresh_battle(
        [MonSpec("blissey", ["stealthrock", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4), MonSpec("volcarona", ["tackle"] * 4)],
        seed=4,
    )
    step_turn(state, prng, 0, 0)
    max_hp = _max_hp_slot(state, 1, 1)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    hp_after = _hp_slot(state, 1, 1)
    expected = max(int(max_hp * 4.0 / 8), 1)
    assert hp_after == max_hp - expected


# ---------------------------------------------------------------------------
# 3. SR vs Heavy Duty Boots — no damage
# ---------------------------------------------------------------------------


def test_sr_blocked_by_heavy_duty_boots(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["stealthrock", "tackle", "tackle", "tackle"])],
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("volcarona", ["tackle"] * 4, item="heavydutyboots"),
        ],
        seed=5,
    )
    step_turn(state, prng, 0, 0)
    max_hp = _max_hp_slot(state, 1, 1)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    assert _hp_slot(state, 1, 1) == max_hp


# ---------------------------------------------------------------------------
# 4. SR vs Magic Guard — no damage
# ---------------------------------------------------------------------------


def test_sr_blocked_by_magic_guard(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["stealthrock", "tackle", "tackle", "tackle"])],
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("clefable", ["tackle"] * 4, ability="magicguard"),
        ],
        seed=6,
    )
    step_turn(state, prng, 0, 0)
    max_hp = _max_hp_slot(state, 1, 1)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    assert _hp_slot(state, 1, 1) == max_hp


# ---------------------------------------------------------------------------
# 5-7. Spikes layer-count damage
# ---------------------------------------------------------------------------


def _setup_spikes_then_switch(fresh_battle, step_turn, layers: int, seed: int):
    moves = ["spikes", "tackle", "tackle", "tackle"]
    state, prng = fresh_battle(
        [MonSpec("blissey", moves)],
        [MonSpec("snorlax", ["tackle"] * 4), MonSpec("garchomp", ["tackle"] * 4)],
        seed=seed,
    )
    for _ in range(layers):
        step_turn(state, prng, 0, 0)
    max_hp = _max_hp_slot(state, 1, 1)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    return max_hp, _hp_slot(state, 1, 1)


def test_spikes_one_layer(fresh_battle, step_turn):
    max_hp, hp = _setup_spikes_then_switch(fresh_battle, step_turn, 1, seed=10)
    assert hp == max_hp - max(max_hp // 8, 1)


def test_spikes_two_layers(fresh_battle, step_turn):
    max_hp, hp = _setup_spikes_then_switch(fresh_battle, step_turn, 2, seed=11)
    assert hp == max_hp - max(max_hp // 6, 1)


def test_spikes_three_layers(fresh_battle, step_turn):
    max_hp, hp = _setup_spikes_then_switch(fresh_battle, step_turn, 3, seed=12)
    assert hp == max_hp - max(max_hp // 4, 1)


# ---------------------------------------------------------------------------
# 8-10. Spikes immunities (Flying, Levitate, Heavy Duty Boots)
# ---------------------------------------------------------------------------


def test_spikes_ignores_flying_type(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["spikes", "tackle", "tackle", "tackle"])],
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("dragonite", ["tackle"] * 4),
        ],
        seed=20,
    )
    step_turn(state, prng, 0, 0)
    max_hp = _max_hp_slot(state, 1, 1)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    assert _hp_slot(state, 1, 1) == max_hp


def test_spikes_ignores_levitate(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["spikes", "tackle", "tackle", "tackle"])],
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("rotomwash", ["tackle"] * 4, ability="levitate"),
        ],
        seed=21,
    )
    step_turn(state, prng, 0, 0)
    max_hp = _max_hp_slot(state, 1, 1)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    assert _hp_slot(state, 1, 1) == max_hp


def test_spikes_blocked_by_heavy_duty_boots(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["spikes", "tackle", "tackle", "tackle"])],
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("garchomp", ["tackle"] * 4, item="heavydutyboots"),
        ],
        seed=22,
    )
    step_turn(state, prng, 0, 0)
    max_hp = _max_hp_slot(state, 1, 1)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    assert _hp_slot(state, 1, 1) == max_hp


# ---------------------------------------------------------------------------
# 11-12. Toxic Spikes layer count → poison vs toxic
# ---------------------------------------------------------------------------


def test_toxic_spikes_one_layer_poisons(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["toxicspikes", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4), MonSpec("garchomp", ["tackle"] * 4)],
        seed=30,
    )
    step_turn(state, prng, 0, 0)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    assert _status_slot(state, 1, 1) == STATUS_POISON


def test_toxic_spikes_two_layers_badly_poisons(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["toxicspikes", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4), MonSpec("garchomp", ["tackle"] * 4)],
        seed=31,
    )
    step_turn(state, prng, 0, 0)
    step_turn(state, prng, 0, 0)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    assert _status_slot(state, 1, 1) == STATUS_TOXIC


# ---------------------------------------------------------------------------
# 13. Toxic Spikes absorbed by grounded Poison-type switch-in
# ---------------------------------------------------------------------------


def test_toxic_spikes_absorbed_by_grounded_poison(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["toxicspikes", "splash", "splash", "splash"])],
        [MonSpec("snorlax", ["splash"] * 4), MonSpec("toxapex", ["splash"] * 4)],
        seed=32,
    )
    # T1: blissey lays toxic spikes
    step_turn(state, prng, 0, 0)
    # T2: blissey idles (splash, action 1), toxapex switches in and absorbs the layer
    step_turn(state, prng, 1, SWITCH_TO_SLOT1)
    assert _status_slot(state, 1, 1) != STATUS_POISON
    assert _status_slot(state, 1, 1) != STATUS_TOXIC
    assert get_toxic_spikes_layers(_hazards_word(state, 1)) == 0


# ---------------------------------------------------------------------------
# 14. Toxic Spikes ignores Flying / Levitate / Steel
# ---------------------------------------------------------------------------


def test_toxic_spikes_ignores_steel(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["toxicspikes", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4), MonSpec("heatran", ["tackle"] * 4)],
        seed=33,
    )
    step_turn(state, prng, 0, 0)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    assert _status_slot(state, 1, 1) not in (STATUS_POISON, STATUS_TOXIC)


def test_toxic_spikes_ignores_flying(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["toxicspikes", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4), MonSpec("dragonite", ["tackle"] * 4)],
        seed=34,
    )
    step_turn(state, prng, 0, 0)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    assert _status_slot(state, 1, 1) not in (STATUS_POISON, STATUS_TOXIC)


# ---------------------------------------------------------------------------
# 15-16. Sticky Web -1 Speed; ignores Flying / HDB
# ---------------------------------------------------------------------------


def test_sticky_web_drops_speed(fresh_battle, step_turn, boost_of):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["stickyweb", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4), MonSpec("garchomp", ["tackle"] * 4)],
        seed=40,
    )
    step_turn(state, prng, 0, 0)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    assert boost_of(state, 1, "spe") == -1


def test_sticky_web_ignores_flying(fresh_battle, step_turn, boost_of):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["stickyweb", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4), MonSpec("dragonite", ["tackle"] * 4)],
        seed=41,
    )
    step_turn(state, prng, 0, 0)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    assert boost_of(state, 1, "spe") == 0


def test_sticky_web_blocked_by_heavy_duty_boots(fresh_battle, step_turn, boost_of):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["stickyweb", "tackle", "tackle", "tackle"])],
        [
            MonSpec("snorlax", ["tackle"] * 4),
            MonSpec("garchomp", ["tackle"] * 4, item="heavydutyboots"),
        ],
        seed=42,
    )
    step_turn(state, prng, 0, 0)
    step_turn(state, prng, 0, SWITCH_TO_SLOT1)
    assert boost_of(state, 1, "spe") == 0


# ---------------------------------------------------------------------------
# 17. Rapid Spin removes user-side hazards
# ---------------------------------------------------------------------------


def test_rapid_spin_removes_user_side_hazards(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["spikes", "stealthrock", "tackle", "tackle"])],
        [MonSpec("garchomp", ["tackle", "rapidspin", "tackle", "tackle"])],
        seed=50,
    )
    # Turn 1: side 0 sets spikes (move 0), side 1 tackles (move 0 = tackle)
    step_turn(state, prng, 0, 0)
    assert get_spikes_layers(_hazards_word(state, 1)) == 1
    # Turn 2: side 0 tackles (move 2), side 1 rapid spins (move 1)
    step_turn(state, prng, 2, 1)
    assert get_spikes_layers(_hazards_word(state, 1)) == 0


# ---------------------------------------------------------------------------
# 18. Defog removes hazards on BOTH sides
# ---------------------------------------------------------------------------


def test_defog_clears_both_sides(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["stealthrock", "defog", "tackle", "tackle"])],
        [MonSpec("garchomp", ["stealthrock", "tackle", "tackle", "tackle"])],
        seed=51,
    )
    # Both sides set SR on the opposing side
    step_turn(state, prng, 0, 0)
    assert get_stealth_rock(_hazards_word(state, 0)) == 1
    assert get_stealth_rock(_hazards_word(state, 1)) == 1
    # Side 0 defogs (move index 1)
    step_turn(state, prng, 1, 1)
    assert get_stealth_rock(_hazards_word(state, 0)) == 0
    assert get_stealth_rock(_hazards_word(state, 1)) == 0


# ---------------------------------------------------------------------------
# 19. Court Change swaps hazards across sides
# ---------------------------------------------------------------------------


def test_court_change_swaps_hazards(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["stealthrock", "courtchange", "tackle", "tackle"])],
        [MonSpec("garchomp", ["spikes", "tackle", "tackle", "tackle"])],
        seed=52,
    )
    # Turn 1: side 0 SR -> side 1, side 1 spikes -> side 0
    step_turn(state, prng, 0, 0)
    assert get_stealth_rock(_hazards_word(state, 1)) == 1
    assert get_spikes_layers(_hazards_word(state, 0)) == 1
    # Turn 2: side 0 court change. After swap, side 0 should have SR and
    # side 1 should have spikes.
    step_turn(state, prng, 1, 1)
    assert get_stealth_rock(_hazards_word(state, 0)) == 1
    assert get_spikes_layers(_hazards_word(state, 1)) == 1


# ---------------------------------------------------------------------------
# 20. Tidy Up removes all hazards on user's side AND opponent's
# ---------------------------------------------------------------------------


def test_tidy_up_clears_hazards(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["stealthrock", "tackle", "tackle", "tackle"])],
        [MonSpec("garchomp", ["tidyup", "tackle", "tackle", "tackle"])],
        seed=53,
    )
    step_turn(state, prng, 0, 0)
    assert get_stealth_rock(_hazards_word(state, 1)) == 1
    step_turn(state, prng, 1, 0)  # garchomp tidy up
    assert get_stealth_rock(_hazards_word(state, 1)) == 0
    assert get_stealth_rock(_hazards_word(state, 0)) == 0


# ---------------------------------------------------------------------------
# 21. Magic Bounce reflects hazard moves back at the setter
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False, reason="magic bounce blocks but does not yet reflect hazards"
)
def test_magic_bounce_reflects_stealth_rock(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["stealthrock", "tackle", "tackle", "tackle"])],
        [MonSpec("espeon", ["tackle"] * 4, ability="magicbounce")],
        seed=60,
    )
    step_turn(state, prng, 0, 0)
    # SR should not land on side 1 (blocked) but should reflect onto side 0.
    assert get_stealth_rock(_hazards_word(state, 1)) == 0
    assert get_stealth_rock(_hazards_word(state, 0)) == 1


# ---------------------------------------------------------------------------
# 22. Hazard layer caps (3 spikes max, 2 toxic spikes max)
# ---------------------------------------------------------------------------


def test_spikes_caps_at_three_layers(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["spikes", "tackle", "tackle", "tackle"])],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=70,
    )
    for _ in range(5):
        step_turn(state, prng, 0, 0)
    assert get_spikes_layers(_hazards_word(state, 1)) == 3


def test_toxic_spikes_caps_at_two_layers(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["toxicspikes", "tackle", "tackle", "tackle"])],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=71,
    )
    for _ in range(4):
        step_turn(state, prng, 0, 0)
    assert get_toxic_spikes_layers(_hazards_word(state, 1)) == 2


def test_stealth_rock_single_layer_only(fresh_battle, step_turn):
    state, prng = fresh_battle(
        [MonSpec("blissey", ["stealthrock", "tackle", "tackle", "tackle"])],
        [MonSpec("garchomp", ["tackle"] * 4)],
        seed=72,
    )
    for _ in range(3):
        step_turn(state, prng, 0, 0)
    assert get_stealth_rock(_hazards_word(state, 1)) == 1
