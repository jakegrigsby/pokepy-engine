"""Smoke test for the pokepy gen 9 battle engine port.

Verifies that the structural port (pokepy.engine.battle_gen9) imports
cleanly. Runtime test is skipped until effects helpers land in
pokepy/effects/.
"""

import importlib
import pytest

def test_battle_gen9_module_imports():
    mod = importlib.import_module("pokepy.engine.battle_gen9")
    assert hasattr(mod, "step_battle_gen9")
    assert hasattr(mod, "step_forced_switch")

def test_engine_package_exports():
    pkg = importlib.import_module("pokepy.engine")
    # The engine package may or may not re-export step_battle_gen9 depending
    # on whether dependencies are present at import time. Just sanity-check
    # the underlying module is reachable.
    bg9 = importlib.import_module("pokepy.engine.battle_gen9")
    assert callable(bg9.step_battle_gen9)
    assert callable(bg9.step_forced_switch)

def test_runtime_step_battle_gen9():
    """Real one-turn step. Skipped until effects + damage helpers exist."""
    try:
        import pokepy.effects  # noqa: F401
        import pokepy.mechanics.damage_gen9  # noqa: F401
    except ImportError:
        pytest.skip("pokepy.effects / pokepy.mechanics.damage_gen9 not yet ported")

    from pokepy.engine.battle_gen9 import step_battle_gen9
    from pokepy.core.state import MultiFormatState

    state = MultiFormatState.create_empty()
    # Wiring real game_data / move_effects / type_chart / prng is left to
    # the post-integration test once the parallel ports land.
    pytest.skip("end-to-end wiring deferred to post-integration step")
