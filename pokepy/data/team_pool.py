"""Real Gen 9 OU team pool — 50k teams from the metamon HuggingFace dataset.

Loaded once from `pokepy/data/teams_gen9ou.npz` (built via
`scripts/build_pokepy_team_pool.py`). Each call to `sample_team()` returns a
team dict in the format `init_battle_state` consumes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

_NPZ_PATH = Path(__file__).parent / "teams_gen9ou.npz"
_cache: Optional[Dict[str, np.ndarray]] = None


def _load() -> Dict[str, np.ndarray]:
    global _cache
    if _cache is not None:
        return _cache
    if not _NPZ_PATH.exists():
        raise FileNotFoundError(
            f"{_NPZ_PATH} not found. Build it with:\n"
            "  .venv-metamon/bin/python scripts/build_pokepy_team_pool.py"
        )
    data = np.load(_NPZ_PATH)
    _cache = {k: data[k] for k in data.files}
    return _cache


def num_teams() -> int:
    return int(_load()["species"].shape[0])


def get_team(idx: int) -> Dict[str, Any]:
    """Return team `idx` as a dict in the format `init_battle_state` expects."""
    d = _load()
    n = num_teams()
    if not 0 <= idx < n:
        raise IndexError(f"team idx {idx} out of range [0, {n})")
    species = d["species"][idx]
    # Filter out empty slots (species == -1)
    valid = species >= 0
    return dict(
        species=[int(s) for s in species[valid]],
        moves=[[int(m) for m in row] for row in d["moves"][idx][valid]],
        items=[int(s) for s in d["items"][idx][valid]],
        abilities=[int(s) for s in d["abilities"][idx][valid]],
        tera_types=[int(s) for s in d["tera_types"][idx][valid]],
        levels=[int(s) for s in d["levels"][idx][valid]],
        evs=[[int(e) for e in row] for row in d["evs"][idx][valid]],
        ivs=[[int(i) for i in row] for row in d["ivs"][idx][valid]],
    )


def sample_team(rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """Random team from the pool."""
    if rng is None:
        rng = np.random.default_rng()
    return get_team(int(rng.integers(0, num_teams())))


def sample_team_pair(rng: Optional[np.random.Generator] = None):
    """Two random teams (one for each side)."""
    if rng is None:
        rng = np.random.default_rng()
    n = num_teams()
    i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
    return get_team(i), get_team(j)
