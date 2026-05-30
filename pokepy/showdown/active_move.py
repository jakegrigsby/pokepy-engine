"""ActiveMove: a mutable per-use copy of a move (Showdown Dex.getActiveMove).

The pipeline mutates per-use fields (hit, totalDamage, spreadHit, type, ...)
without touching the shared Dex entry. Data fields are copied from the entry;
unknown attribute reads return None (JS ``undefined`` semantics).
"""

from __future__ import annotations

from typing import Any, Dict


class ActiveMove:
    def __init__(self, entry):
        # Copy data fields off the immutable Dex entry.
        d: Dict[str, Any] = dict(getattr(entry, "raw", {}) or {})
        object.__setattr__(self, "_d", d)
        object.__setattr__(self, "handlers", getattr(entry, "handlers", {}) or {})
        object.__setattr__(self, "exists", bool(d))
        # Common per-use runtime fields.
        d.setdefault("hit", 0)
        # Mirror sim/dex-moves.ts: single secondary becomes a one-element list.
        secondaries = d.get("secondaries")
        secondary = d.get("secondary")
        if secondaries:
            d["secondaries"] = secondaries
        elif secondary:
            d["secondaries"] = [secondary]
        self.totalDamage = 0
        self.spreadHit = False
        self.hitTargets = None
        self.lastHit = False

    def __getattr__(self, name: str) -> Any:
        try:
            return self._d[name]
        except KeyError:
            return None

    def __setattr__(self, name: str, value: Any):
        self._d[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        return self._d.get(name, default)

    @property
    def id(self) -> str:
        return self._d.get("id", "")

    @property
    def name(self) -> str:
        return self._d.get("name", "")

    @property
    def effectType(self) -> str:  # noqa: N802
        return "Move"

    def __repr__(self) -> str:
        return f"<ActiveMove {self.id!r}>"
