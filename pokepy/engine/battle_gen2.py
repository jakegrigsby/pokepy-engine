"""Gen 2 OU battle step — reuses modern core with GEN2_PROFILE."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from pokepy.core.gen_profile import GEN2_PROFILE
from pokepy.core.state import MultiFormatState
from pokepy.engine.battle_gen9 import step_battle_gen9


def step_battle_gen2(
    state: MultiFormatState,
    action0: int,
    action1: int,
    game_data,
    move_effects,
    type_chart: np.ndarray,
    gen5_prng,
    resolve_mid_turn_switch0=None,
    wants_tera0: bool = False,
    wants_tera1: bool = False,
    profile=GEN2_PROFILE,
) -> Tuple[np.float32, np.float32, bool]:
    return step_battle_gen9(
        state,
        action0,
        action1,
        game_data,
        move_effects,
        type_chart,
        gen5_prng,
        resolve_mid_turn_switch0=resolve_mid_turn_switch0,
        wants_tera0=False,
        wants_tera1=False,
        profile=profile or GEN2_PROFILE,
    )
