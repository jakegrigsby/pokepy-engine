"""Top-level Kakuna observation builder.

Public API:
    build_kakuna_obs(universal_state) -> dict
        Returns {'text_tokens': int32[N], 'numbers': float32[55], 'illegal_actions': bool[13]}
        where N is the number of whitespace-separated words in the obs (~106 for
        OpponentMoveObservationSpace + Expanded + TeamPreview extras).

    build_rl2_features(prev_reward, prev_action) -> float32[14]
        [prev_reward, one_hot(prev_action, 13)]

The bridge from pokepy's `MultiFormatState` to a `UniversalState` is in
`state_to_universal.py` so this file stays a thin orchestrator.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from pokepy.obs.universal import UniversalState
from pokepy.obs.observation_space import state_to_obs
from pokepy.obs.tokenizer import load_default_tokenizer

def build_kakuna_obs(
    universal_state: UniversalState,
    illegal_actions: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Produce the obs dict Kakuna's encoder consumes.

    Args:
        universal_state: A constructed UniversalState (use
            pokepy.obs.state_to_universal.state_to_universal_state to build one
            from pokepy MultiFormatState).
        illegal_actions: Optional bool[13] mask of illegal action indices. If
            None, returns all-False (all actions legal).

    Returns:
        {
            'text_tokens': np.int32[N],
            'numbers': np.float32[55],
            'illegal_actions': np.bool_[13],
        }
    """
    text, numbers = state_to_obs(universal_state)
    tok = load_default_tokenizer()
    text_tokens = tok.tokenize(text)
    if illegal_actions is None:
        illegal_actions = np.zeros(13, dtype=np.bool_)
    return {
        "text_tokens": text_tokens,
        "numbers": numbers,
        "illegal_actions": illegal_actions.astype(np.bool_),
    }

def build_rl2_features(prev_reward: float, prev_action: int) -> np.ndarray:
    """RL2 history features for Kakuna's trajectory encoder.

    Returns float32[14] = [prev_reward, one_hot(prev_action, 13)].
    """
    rl2 = np.zeros(14, dtype=np.float32)
    rl2[0] = float(prev_reward)
    if 0 <= int(prev_action) < 13:
        rl2[1 + int(prev_action)] = 1.0
    return rl2
