"""Smoke test for the pokepy event battle engine."""

import importlib


def test_battle_gen9_module_imports():
    mod = importlib.import_module("pokepy.engine.battle_gen9")
    assert hasattr(mod, "step_battle_gen9")
    assert hasattr(mod, "step_battle_gen9_iter")
    assert hasattr(mod, "step_forced_switch")


def test_engine_package_exports():
    pkg = importlib.import_module("pokepy.engine")
    bg9 = importlib.import_module("pokepy.engine.battle_gen9")
    assert callable(bg9.step_battle_gen9)
    assert callable(bg9.step_forced_switch)
    assert hasattr(pkg, "step_battle_event")
