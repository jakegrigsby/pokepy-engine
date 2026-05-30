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

StepFn = Callable[..., Tuple[np.float32, np.float32, bool]]
ForcedSwitchFn = Callable[..., Tuple[np.float32, np.float32, bool]]


@dataclass(frozen=True)
class EngineEntry:
    step_fn: StepFn
    forced_switch_fn: ForcedSwitchFn
    profile: GenProfile


def _wrap_event_step(profile: GenProfile) -> StepFn:
    from pokepy.sim.engine_adapter import step_battle_event

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
        return step_battle_event(
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
    from pokepy.sim.engine_adapter import step_forced_switch_event

    reg: Dict[int, EngineEntry] = {}
    for gen in (1, 2, 3, 4, 9):
        prof = profile_for_gen(gen)
        reg[gen] = EngineEntry(
            step_fn=_wrap_event_step(prof),
            forced_switch_fn=step_forced_switch_event,
            profile=prof,
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


def __getattr__(name: str):
    if name == "battle_gen9":
        import pokepy.engine.battle_gen9 as mod

        return mod
    if name in (
        "step_battle_gen9",
        "step_battle_gen9_iter",
        "step_forced_switch",
        "step_battle_event",
        "step_battle_event_iter",
    ):
        from pokepy.sim import engine_adapter

        return getattr(engine_adapter, name)
    raise AttributeError(name)


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
    "step_battle_event",
    "step_battle_event_iter",
    "step_forced_switch",
    "step_forced_switch_for_gen",
]
