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
from pokepy.engine.move_pipeline import run_move
from pokepy.engine.switch import resolve_switch_choices_sync, step_forced_switch_modular
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
        from pokepy.core.constants import (
            OFF_META,
            M_ACTIVE0,
            M_ACTIVE1,
            OFF_SIDE0,
            OFF_SIDE1,
            POKEMON_SIZE,
        )

        a0 = int(action0)
        a1 = int(action1)
        p0 = (
            OFF_SIDE0
            + int(self.state.battle_state[OFF_META + M_ACTIVE0]) * POKEMON_SIZE
        )
        p1 = (
            OFF_SIDE1
            + int(self.state.battle_state[OFF_META + M_ACTIVE1]) * POKEMON_SIZE
        )
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

    def run_action_modular(self, action) -> None:
        """Execute one queued action via modular pipeline."""
        from pokepy.core.constants import (
            OFF_META,
            M_ACTIVE0,
            M_ACTIVE1,
            OFF_SIDE0,
            OFF_SIDE1,
            POKEMON_SIZE,
        )

        choice = action.get("choice")
        side = int(action.get("side", 0))
        active0 = int(self.state.battle_state[OFF_META + M_ACTIVE0])
        active1 = int(self.state.battle_state[OFF_META + M_ACTIVE1])
        p0 = OFF_SIDE0 + active0 * POKEMON_SIZE
        p1 = OFF_SIDE1 + active1 * POKEMON_SIZE
        if choice == "move":
            slot = int(action.get("action_index", 0))
            user_off = p0 if side == 0 else p1
            target_off = p1 if side == 0 else p0
            # team_moves / opp_moves are indexed by active slot.
            if side == 0:
                move_id = int(self.state.team_moves[active0, slot])
            else:
                move_id = int(self.state.opp_moves[active1, slot])
            if move_id >= 0:
                run_move(self.ctx, move_id, user_off, target_off)
        elif choice in ("switch", "instaswitch"):
            target = int(action.get("action_index", 0)) - 4
            if target >= 0:
                step_forced_switch_modular(
                    self.state,
                    target,
                    side,
                    self.game_data,
                    self.move_effects,
                    self.type_chart,
                    self.gen5_prng,
                    profile=self.profile,
                )

    def residual(self) -> None:
        """Residual step hook."""
        self.ctx.each_event("Residual", lambda off: self.ctx.run_event("Residual", off))

    def end_turn(self) -> None:
        """End-turn hook for modular path."""
        self.residual()
        self.state.turn = np.int16(int(self.state.turn) + 1)

    def run_turn_modular(self, action0: int, action1: int) -> Tuple[np.float32, np.float32, bool]:
        """Execute one turn via modular queue/action flow."""
        self.begin_turn()
        self.commit_choices(action0, action1)
        while self.queue:
            action = self.queue.shift()
            if action is None:
                break
            self.run_action_modular(action)
        self.end_turn()
        done = bool(self.state.done)
        return np.float32(0.0), np.float32(0.0), done

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
    import os

    driver = TurnDriver(
        state,
        game_data,
        move_effects,
        type_chart,
        gen5_prng,
        profile=profile,
    )
    if os.environ.get("POKEPY_MODULAR_TURN_LOOP") == "1":
        return driver.run_turn_modular(action0, action1)
    # While the legacy monolith remains the authoritative executor, invoking
    # modular pre-turn hooks here double-consumes PRNG frames (quick-claw roll
    # + queue tie shuffle) before `step_battle_gen9` runs its own draws.
    # Keep this wrapper frame-neutral until modular turn execution fully takes over.
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
