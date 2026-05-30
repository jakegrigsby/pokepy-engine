"""Modular turn loop entry points.

Provides the Showdown-shaped turn driver (``TurnDriver``) while delegating the
full battle semantics to the ported ``battle_gen9`` implementation. New code
paths use ``BitpackBattleContext`` + ``BattleQueue`` + ``move_pipeline``;
the legacy monolith remains the authoritative executor until parity work
retires the remaining inline frame hooks.
"""

from __future__ import annotations

from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np

from pokepy.core.constants import Phase
from pokepy.core.gen_profile import GenProfile, profile_for_gen
from pokepy.core.state import MultiFormatState
from pokepy.engine.battle_gen9 import step_battle_gen9, step_battle_gen9_iter
from pokepy.engine.dispatch import BitpackBattleContext, make_context
from pokepy.engine.gen_mods import apply_gen_end_turn_rolls, profile_for_state
from pokepy.engine.queue import BattleQueue
from pokepy.engine.switch import resolve_switch_choices_sync
from pokepy.engine.switch_requests import SwitchRequest


class TurnDriver:
    """One-turn driver with optional modular pre/post hooks."""

    def __init__(
        self,
        state: MultiFormatState,
        game_data,
        move_effects,
        type_chart: np.ndarray,
        gen5_prng,
        *,
        profile: Optional[GenProfile] = None,
    ):
        self.state = state
        self.game_data = game_data
        self.move_effects = move_effects
        self.type_chart = type_chart
        self.gen5_prng = gen5_prng
        self.profile = profile or profile_for_state(state)
        self.ctx = make_context(
            state,
            game_data,
            move_effects,
            type_chart,
            gen5_prng,
            profile=self.profile,
        )
        self.queue = BattleQueue(self.ctx)

    def begin_turn(self) -> None:
        """Turn-start hooks (team preview / Quick Claw rolls handled in legacy path)."""
        apply_gen_end_turn_rolls(self.ctx.battle, self.profile, self.gen5_prng)

    def commit_choices(self, action0: int, action1: int) -> None:
        """Record player choices into the action queue (metadata for modular path)."""
        from pokepy.core.constants import OFF_META, M_ACTIVE0, M_ACTIVE1, OFF_SIDE0, OFF_SIDE1, POKEMON_SIZE

        a0 = int(action0)
        a1 = int(action1)
        p0 = OFF_SIDE0 + int(self.state.battle_state[OFF_META + M_ACTIVE0]) * POKEMON_SIZE
        p1 = OFF_SIDE1 + int(self.state.battle_state[OFF_META + M_ACTIVE1]) * POKEMON_SIZE
        if a0 < 4:
            self.queue.insert_choice(
                choice="move",
                side=0,
                action_index=a0,
                pokemon_offset=p0,
            )
        else:
            self.queue.insert_choice(
                choice="switch",
                side=0,
                action_index=a0,
                pokemon_offset=p0,
            )
        if a1 < 4:
            self.queue.insert_choice(
                choice="move",
                side=1,
                action_index=a1,
                pokemon_offset=p1,
            )
        else:
            self.queue.insert_choice(
                choice="switch",
                side=1,
                action_index=a1,
                pokemon_offset=p1,
            )
        self.queue.sort()

    def run_turn_legacy(
        self,
        action0: int,
        action1: int,
        *,
        wants_tera0: bool = False,
        wants_tera1: bool = False,
        resolve_mid_turn_switch0=None,
        defer_p1_forced_switch: bool = False,
    ) -> Tuple[np.float32, np.float32, bool]:
        """Execute one full turn via the legacy generator wrapper."""
        return step_battle_gen9(
            self.state,
            int(action0),
            int(action1),
            self.game_data,
            self.move_effects,
            self.type_chart,
            self.gen5_prng,
            resolve_mid_turn_switch0=resolve_mid_turn_switch0,
            wants_tera0=wants_tera0,
            wants_tera1=wants_tera1,
            profile=self.profile,
            defer_p1_forced_switch=defer_p1_forced_switch,
        )

    def iter_turn(
        self,
        action0: int,
        action1: int,
        *,
        wants_tera0: bool = False,
        wants_tera1: bool = False,
        resolve_mid_turn_switch0=None,
    ) -> Generator[SwitchRequest, Dict[int, int], Tuple[np.float32, np.float32, bool]]:
        """Mid-turn switch generator (same contract as ``step_battle_gen9_iter``)."""
        return step_battle_gen9_iter(
            self.state,
            int(action0),
            int(action1),
            self.game_data,
            self.move_effects,
            self.type_chart,
            self.gen5_prng,
            resolve_mid_turn_switch0=resolve_mid_turn_switch0,
            wants_tera0=wants_tera0,
            wants_tera1=wants_tera1,
            profile=self.profile,
        )


def run_turn(
    state: MultiFormatState,
    action0: int,
    action1: int,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    gen5_prng,
    *,
    profile: Optional[GenProfile] = None,
    wants_tera0: bool = False,
    wants_tera1: bool = False,
    resolve_mid_turn_switch0=None,
    defer_p1_forced_switch: bool = False,
) -> Tuple[np.float32, np.float32, bool]:
    """Canonical modular one-turn entry (delegates to legacy executor)."""
    driver = TurnDriver(
        state,
        game_data,
        move_effects,
        type_chart,
        gen5_prng,
        profile=profile,
    )
    driver.begin_turn()
    driver.commit_choices(action0, action1)
    return driver.run_turn_legacy(
        action0,
        action1,
        wants_tera0=wants_tera0,
        wants_tera1=wants_tera1,
        resolve_mid_turn_switch0=resolve_mid_turn_switch0,
        defer_p1_forced_switch=defer_p1_forced_switch,
    )


__all__ = [
    "TurnDriver",
    "run_turn",
    "step_battle_gen9",
    "step_battle_gen9_iter",
    "resolve_switch_choices_sync",
]
