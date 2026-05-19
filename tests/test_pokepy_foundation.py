"""Foundation tests for pokepy.

Covers state schema, bit-pack helpers, type chart, data loader, Gen5 PRNG.
Also asserts that pokepy stays free of heavy ML dependencies (jax/torch).
"""

from __future__ import annotations

import numpy as np
import pytest

from pokepy.core.state import MultiFormatState
from pokepy.core import constants as C
from pokepy.core import bitpack as bp
from pokepy.data.type_charts import MODERN_TYPE_CHART
from pokepy.data.loader import load_game_data
from pokepy.utils.gen5_prng import Gen5PRNG

def test_state_create_empty():
    s = MultiFormatState.create_empty(format_id=1)
    assert int(s.format_id) == 1
    assert s.battle_state.shape == (C.STATE_SIZE,)
    assert s.battle_state.dtype == np.int16
    assert s.team_species.shape == (6,)
    assert (s.team_species == -1).all()
    assert s.team_moves.shape == (6, 4)
    assert s.opp_pp.shape == (6, 4)
    assert s.opp_revealed.dtype == np.bool_
    assert int(s.turn) == 0
    assert int(s.winner) == -1

def test_state_byte_layout_consistent():
    """The flat buffer accommodates 6+6 pokemon + field + meta + reserved."""
    needed = C.OFF_SIDE0 + 0  # start
    needed = max(needed, C.OFF_SIDE1 + 6 * C.POKEMON_SIZE)
    needed = max(needed, C.OFF_FIELD + 32)
    needed = max(needed, C.OFF_META + 16)
    needed = max(needed, C.OFF_MOVES + 16)
    assert needed <= C.STATE_SIZE

# -----------------------------------------------------------------------------
# Bitpack: boost packing
# -----------------------------------------------------------------------------

def test_boost_pack_roundtrip():
    """Pack -6..+6 in each of 4 slots, extract, assert equal."""
    packed = C.NEUTRAL_BOOSTS_13
    for shift, delta in [(0, 3), (4, -2), (8, 6), (12, -6)]:
        packed = bp.apply_boost_to_packed(packed, shift, delta)
    assert bp.extract_boost(packed, 0) == 3
    assert bp.extract_boost(packed, 4) == -2
    assert bp.extract_boost(packed, 8) == 6
    assert bp.extract_boost(packed, 12) == -6

def test_boost_clamps_to_range():
    packed = C.NEUTRAL_BOOSTS_13
    packed = bp.apply_boost_to_packed(packed, 0, 100)
    assert bp.extract_boost(packed, 0) == 6
    packed = bp.apply_boost_to_packed(packed, 0, -100)
    assert bp.extract_boost(packed, 0) == -6

def test_boost_independent_slots():
    packed = C.NEUTRAL_BOOSTS_13
    packed = bp.apply_boost_to_packed(packed, 0, 3)
    # Other slots remain neutral
    assert bp.extract_boost(packed, 4) == 0
    assert bp.extract_boost(packed, 8) == 0
    assert bp.extract_boost(packed, 12) == 0

# -----------------------------------------------------------------------------
# Bitpack: status, hazards, protect, volatiles
# -----------------------------------------------------------------------------

def test_status_pack_roundtrip():
    field = bp.set_status(C.STATUS_TOXIC, turns=5)
    assert bp.get_status(field) == C.STATUS_TOXIC
    assert bp.get_status_turns(field) == 5

def test_hazards_pack_roundtrip():
    h = 0
    h = bp.set_spikes(h, 3)
    h = bp.set_stealth_rock(h)
    h = bp.set_toxic_spikes(h, 2)
    h = bp.set_sticky_web(h)
    assert bp.get_spikes_layers(h) == 3
    assert bp.get_stealth_rock(h) == 1
    assert bp.get_toxic_spikes_layers(h) == 2
    assert bp.get_sticky_web(h) == 1
    assert bp.clear_hazards(h) == 0

def test_hazards_layer_caps():
    assert bp.get_spikes_layers(bp.set_spikes(0, 99)) == 3
    assert bp.get_toxic_spikes_layers(bp.set_toxic_spikes(0, 99)) == 2

def test_protect_pack_roundtrip():
    p = 0
    p = bp.set_protect_active(p, True)
    p = bp.set_protect_consecutive(p, 4)
    p = bp.set_protect_type(p, C.PROTECT_KINGS_SHIELD)
    assert bp.get_protect_active(p) == 1
    assert bp.get_protect_consecutive(p) == 4
    assert bp.get_protect_type(p) == C.PROTECT_KINGS_SHIELD
    p2 = bp.clear_protect_active(p)
    assert bp.get_protect_active(p2) == 0
    assert bp.get_protect_consecutive(p2) == 4  # unchanged

def test_volatile_pack_roundtrip():
    v = 0
    v = bp.set_flinched(v, True)
    v = bp.set_confusion_turns(v, 3)
    v = bp.set_taunt_turns(v, 4)
    v = bp.set_encore_turns(v, 2)
    assert bp.get_flinched(v) is True
    assert bp.get_confusion_turns(v) == 3
    assert bp.get_taunt_turns(v) == 4
    assert bp.get_encore_turns(v) == 2
    cleared = bp.clear_volatile_turn_effects(v)
    assert bp.get_flinched(cleared) is False
    # Other counters preserved
    assert bp.get_confusion_turns(cleared) == 3

# -----------------------------------------------------------------------------
# Type chart
# -----------------------------------------------------------------------------

def test_modern_type_chart_shape_and_neutrals():
    assert MODERN_TYPE_CHART.shape == (19, 19)
    assert MODERN_TYPE_CHART.dtype == np.float32
    # Spot checks
    assert MODERN_TYPE_CHART[C.TYPE_GRASS, C.TYPE_FIRE] == 2.0
    assert MODERN_TYPE_CHART[C.TYPE_NORMAL, C.TYPE_GHOST] == 0.0
    assert MODERN_TYPE_CHART[C.TYPE_FAIRY, C.TYPE_DRAGON] == 0.0
    assert MODERN_TYPE_CHART[C.TYPE_FIRE, C.TYPE_WATER] == 2.0
    assert MODERN_TYPE_CHART[C.TYPE_DRAGON, C.TYPE_FIRE] == 0.5
    assert MODERN_TYPE_CHART[C.TYPE_NORMAL, C.TYPE_NORMAL] == 1.0

# -----------------------------------------------------------------------------
# Data loader
# -----------------------------------------------------------------------------

def test_data_loader_loads():
    gd = load_game_data()
    assert gd.type_chart.shape == (19, 19)
    assert gd.species_base_stats.ndim == 2
    assert gd.species_base_stats.shape[1] == 6  # 6 stats
    assert gd.move_base_power.ndim == 1
    assert gd.move_priority.ndim == 1
    # Move 0 (silent) typically has 0 base power; just verify dtype is small int
    assert gd.move_base_power.dtype.kind in ("i", "u")

# -----------------------------------------------------------------------------
# Gen 5 PRNG
# -----------------------------------------------------------------------------

def test_gen5_prng_deterministic():
    a = Gen5PRNG((1, 2, 3, 4))
    b = Gen5PRNG((1, 2, 3, 4))
    for _ in range(100):
        assert a.next() == b.next()

def test_gen5_prng_damage_roll_in_range():
    p = Gen5PRNG((1, 2, 3, 4))
    for _ in range(1000):
        r = p.damage_roll()
        assert 0.85 <= r < 1.0

def test_pokepy_has_no_jax_or_torch_dependency():
    """pokepy must remain pure-Python/numpy — no jax, torch, or tensorflow."""
    import sys
    BANNED = ("jax", "torch", "tensorflow", "flax")
    for k in list(sys.modules):
        if any(k == b or k.startswith(b + ".") for b in BANNED):
            del sys.modules[k]
    for k in list(sys.modules):
        if k == "pokepy" or k.startswith("pokepy."):
            del sys.modules[k]
    import pokepy  # noqa: F401
    import pokepy.core.state  # noqa: F401
    import pokepy.core.bitpack  # noqa: F401
    import pokepy.data.loader  # noqa: F401
    import pokepy.data.type_charts  # noqa: F401
    import pokepy.utils.gen5_prng  # noqa: F401
    leaked = [k for k in sys.modules if any(k == b or k.startswith(b + ".") for b in BANNED)]
    assert not leaked, f"pokepy leaked imports of a heavy dependency: {leaked}"
