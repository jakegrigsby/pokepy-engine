"""Startup PRNG consumption before turn 1."""

from __future__ import annotations

import numpy as np

from pokepy.core.constants import OFF_SIDE0, OFF_SIDE1, POKEMON_SIZE


def consume_team_preview_queue_sort_frames(battle: np.ndarray, gen5_prng) -> None:
    """Mirror Showdown team-preview queue sort PRNG (battle.ts commitChoices)."""
    for slot in range(6):
        p0_off = OFF_SIDE0 + slot * POKEMON_SIZE
        p1_off = OFF_SIDE1 + slot * POKEMON_SIZE
        if int(battle[p0_off + 0]) <= 0 or int(battle[p1_off + 0]) <= 0:
            continue
        if int(battle[p0_off + 11]) == int(battle[p1_off + 11]):
            gen5_prng.random(2 * slot, 2 * slot + 2)
