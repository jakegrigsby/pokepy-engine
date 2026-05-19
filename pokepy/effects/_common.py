"""Common imports/utilities for pokepy effects modules.

Effects functions mutate `battle: np.ndarray` in place. They take an explicit
`gen5_prng: Gen5PRNG` (stateful) instead of returning rng. State for top-level
fields is passed as `state: MultiFormatState`.

NOTE: These ports are scaffolds — function signatures and module layout are
in place but most bodies are TODO. Integration step will fill in real logic.
"""
from __future__ import annotations

import numpy as np

from pokepy.core.state import MultiFormatState
from pokepy.utils.gen5_prng import Gen5PRNG

__all__ = ["np", "MultiFormatState", "Gen5PRNG"]
