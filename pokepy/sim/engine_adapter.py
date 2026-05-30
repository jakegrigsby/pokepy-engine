"""Engine adapter — single sim/ entry for all generations."""

from __future__ import annotations

from pokepy.sim.event_turn import (
    step_forced_switch_event,
    step_turn_event,
    step_turn_event_iter,
)

# Preserved names for engine/__init__.py and external callers.
step_battle_event = step_turn_event
step_battle_event_iter = step_turn_event_iter
step_battle_gen9 = step_turn_event
step_battle_gen9_iter = step_turn_event_iter
step_forced_switch = step_forced_switch_event

__all__ = [
    "step_battle_event",
    "step_battle_event_iter",
    "step_battle_gen9",
    "step_battle_gen9_iter",
    "step_forced_switch",
    "step_forced_switch_event",
    "step_turn_event",
    "step_turn_event_iter",
]
