"""Re-export speedSort from battle.py (single implementation)."""

from pokepy.sim.battle import (
    PriorityEntry,
    SpeedSortTracker,
    compare_priority_entries,
    consume_two_way_speed_tie,
    shuffle_in_place,
    speed_sort_in_place,
)

__all__ = [
    "PriorityEntry",
    "SpeedSortTracker",
    "compare_priority_entries",
    "consume_two_way_speed_tie",
    "shuffle_in_place",
    "speed_sort_in_place",
]
