"""Back-compat shim — all gens use the event engine in pokepy/sim/."""

from pokepy.sim.event_turn import (
    step_forced_switch_event as step_forced_switch,
    step_turn_event as step_battle_gen9,
    step_turn_event_iter as step_battle_gen9_iter,
)

__all__ = [
    "step_battle_gen9",
    "step_battle_gen9_iter",
    "step_forced_switch",
]
