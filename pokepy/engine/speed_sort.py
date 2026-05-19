"""SpeedSortTracker — simulates Showdown's speedSort PRNG frame consumption.

Showdown's speedSort (sim/battle.ts:429-460) is a selection sort that calls
prng.shuffle(list, start, end) on each tied group. shuffle() is a Fisher-Yates
loop that consumes (end - start - 1) PRNG frames, so a 2-element tied group
consumes exactly 1 frame.

comparePriority (battle.ts:404-426) compares:
    order      — lower is better (switch=103, move=200, residual=300)
    priority   — higher is better (Quick Attack=+1, Trick Room=-7)
    speed      — higher is better (faster mon goes first)
    subOrder   — lower is better
    effectOrder — lower is better (for abilities/items)

Returns negative if a goes before b, positive if b goes before a, 0 if tied.
"""

from __future__ import annotations

from typing import List, Tuple

# Entry type: (order, priority, speed, subOrder, effectOrder)
Entry = Tuple[int, int, int, int, int]

class SpeedSortTracker:
    """Simulates Showdown's speedSort PRNG frame consumption."""

    def __init__(self, prng):
        self.prng = prng

    def compare_priority(self, a: Entry, b: Entry) -> int:
        """Match Showdown's comparePriority (battle.ts:404-426).

        Compare by: order (lower first), priority (higher first),
        speed (higher first), subOrder (lower first), effectOrder (lower first).

        Returns negative if a goes first, positive if b goes first, 0 if tied.
        """
        # order: lower is better
        if a[0] != b[0]:
            return a[0] - b[0]
        # priority: higher is better
        if a[1] != b[1]:
            return b[1] - a[1]
        # speed: higher is better
        if a[2] != b[2]:
            return b[2] - a[2]
        # subOrder: lower is better
        if a[3] != b[3]:
            return a[3] - b[3]
        # effectOrder: lower is better
        if a[4] != b[4]:
            return a[4] - b[4]
        return 0

    def speed_sort_consume(self, entries: List[Entry]) -> int:
        """Simulate speedSort on a list of entry tuples.

        Uses selection sort matching Showdown's battle.ts:429-460.
        Consumes PRNG frames for tied groups via Fisher-Yates shuffle.
        A tied group of size N consumes N-1 PRNG frames.

        For a 2-element list, returns the PRNG value consumed when the pair
        ties (0 = no swap / first entry stays first, 1 = swapped), or 0 if
        no frame was consumed (entries not tied).  For longer lists returns 0.
        """
        n = len(entries)
        if n < 2:
            return 0

        i = 0
        result = 0
        while i < n - 1:
            # Find the end of the current tied group starting at i.
            j = i + 1
            while j < n and self.compare_priority(entries[i], entries[j]) == 0:
                j += 1
            # Tied group is entries[i:j]
            group_size = j - i
            if group_size > 1:
                # Fisher-Yates shuffle consumes (group_size - 1) PRNG frames.
                # For exactly 2 entries capture the single frame value so the
                # caller can derive the effective ordering.
                for k in range(group_size - 1):
                    v = int(self.prng.random(0, 2))
                    if group_size == 2 and k == 0:
                        result = v
            i = j

        return result

    def each_event_update(self, active_speeds: List[int]) -> None:
        """Simulate eachEvent('Update') — speedSort actives by speed only.

        Used for BeforeTurn eachEvent and post-BeforeTurn Update eachEvent.
        The speed comparator (battle.ts:468) only compares pokemon.speed —
        two actives tie iff their effective speeds match.
        Frame is consumed but return value is discarded (ordering only).
        """
        if len(active_speeds) < 2:
            return
        entries = [(0, 0, s, 0, 0) for s in active_speeds]
        self.speed_sort_consume(entries)

    def queue_sort(self, actions: List[Entry]) -> int:
        """Simulate queue.sort() — comparePriority on action entries.

        Used for commitChoices sort and gen8+ queue re-sort.
        Returns the shuffle value (0 or 1) if a 2-entry tied pair was sorted,
        0 otherwise.  The caller uses this to derive tie_break ordering.
        """
        return self.speed_sort_consume(actions)
