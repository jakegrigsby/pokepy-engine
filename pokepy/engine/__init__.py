"""Pokepy battle engine — Gen 9 turn loop, forced switch, action mask.

Public API mirrors the Showdown reference implementation free-function form.
"""

from pokepy.engine.action_mask import get_battle_action_mask, get_action_mask

# Optional: battle_gen9 may not be present in early phases of the port
try:
    from pokepy.engine.battle_gen9 import step_battle_gen9, step_forced_switch

    __all__ = [
        "step_battle_gen9",
        "step_forced_switch",
        "get_battle_action_mask",
        "get_action_mask",
    ]
except ImportError:
    __all__ = ["get_battle_action_mask", "get_action_mask"]
