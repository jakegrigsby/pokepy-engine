"""Pokepy battle engine — gen-keyed turn loop, forced switch, action mask."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from pokepy.core.gen_profile import (
    GEN9_PROFILE,
    GenProfile,
    profile_for_gen,
    registered_gens,
)
from pokepy.engine.action_mask import get_action_mask, get_battle_action_mask

try:
    from pokepy.engine.battle_gen9 import step_battle_gen9, step_forced_switch
except ImportError:
    step_battle_gen9 = None  # type: ignore
    step_forced_switch = None  # type: ignore

try:
    from pokepy.engine.battle_gen2 import step_battle_gen2
except ImportError:
    step_battle_gen2 = None  # type: ignore

try:
    from pokepy.engine.battle_gen1 import step_battle_gen1
except ImportError:
    step_battle_gen1 = None  # type: ignore


StepFn = Callable[..., Tuple[np.float32, np.float32, bool]]
ForcedSwitchFn = Callable[..., Tuple[np.float32, np.float32, bool]]


@dataclass(frozen=True)
class EngineEntry:
    step_fn: StepFn
    forced_switch_fn: ForcedSwitchFn
    profile: GenProfile


def _wrap_modern_step(profile: GenProfile) -> StepFn:
    def _step(
        state,
        action0,
        action1,
        game_data,
        move_effects,
        type_chart,
        gen5_prng,
        **kwargs,
    ):
        return step_battle_gen9(
            state,
            action0,
            action1,
            game_data,
            move_effects,
            type_chart,
            gen5_prng,
            profile=profile,
            **kwargs,
        )

    return _step


def _build_registry() -> Dict[int, EngineEntry]:
    reg: Dict[int, EngineEntry] = {}
    if step_battle_gen9 is not None and step_forced_switch is not None:
        for gen in (4, 3, 9):
            prof = profile_for_gen(gen)
            reg[gen] = EngineEntry(
                step_fn=_wrap_modern_step(prof),
                forced_switch_fn=step_forced_switch,
                profile=prof,
            )
    if step_battle_gen2 is not None and step_forced_switch is not None:
        reg[2] = EngineEntry(
            step_fn=step_battle_gen2,
            forced_switch_fn=step_forced_switch,
            profile=profile_for_gen(2),
        )
    if step_battle_gen1 is not None and step_forced_switch is not None:
        reg[1] = EngineEntry(
            step_fn=step_battle_gen1,
            profile=profile_for_gen(1),
            forced_switch_fn=step_forced_switch,
        )
    return reg


ENGINE_REGISTRY: Dict[int, EngineEntry] = _build_registry()


def get_engine(gen: int) -> EngineEntry:
    try:
        return ENGINE_REGISTRY[int(gen)]
    except KeyError as exc:
        raise KeyError(
            f"No pokepy engine registered for gen {gen}. "
            f"Registered: {sorted(ENGINE_REGISTRY)}"
        ) from exc


def step_battle(
    gen: int,
    state,
    action0: int,
    action1: int,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    gen5_prng,
    **kwargs,
) -> Tuple[np.float32, np.float32, bool]:
    entry = get_engine(gen)
    return entry.step_fn(
        state,
        action0,
        action1,
        game_data,
        move_effects,
        type_chart,
        gen5_prng,
        **kwargs,
    )


def step_forced_switch_for_gen(
    gen: int,
    state,
    action: int,
    side: int,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    gen5_prng,
) -> Tuple[np.float32, np.float32, bool]:
    entry = get_engine(gen)
    return entry.forced_switch_fn(
        state,
        action,
        side,
        game_data,
        move_effects,
        type_chart,
        gen5_prng,
        profile=entry.profile,
    )


__all__ = [
    "ENGINE_REGISTRY",
    "EngineEntry",
    "GEN9_PROFILE",
    "get_action_mask",
    "get_battle_action_mask",
    "get_engine",
    "registered_gens",
    "step_battle",
    "step_battle_gen9",
    "step_forced_switch",
    "step_forced_switch_for_gen",
]
