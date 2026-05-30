"""Dex: per-gen data access for the ported sim.

Loads the JSON tables dumped from Showdown's own Dex
(``scripts/extract_dex_json.js`` -> ``pokepy/data/showdown/gen{N}/*.json``) and
exposes them through an object surface that mirrors Showdown's runtime Dex, so
translated callbacks can read near-verbatim:

    this.dex.moves.get('psychic').basePower   ->   battle.dex.moves.get('psychic').basePower

Data *fields* come from JSON (gen-correct, complete). Effect *callbacks*
(onHit, basePowerCallback, ...) are translated to Python and registered in
``pokepy.showdown.effects``; the Dex merges any registered handlers onto the
returned entry as ``entry.handlers`` (an event_name -> callable mapping).

Field surface used by the runtime (non-exhaustive):
  - moves:   num, name, id, type, category, basePower, accuracy, pp, priority,
             critRatio, target, flags, secondary/secondaries, boosts, status,
             volatileStatus, recoil, drain, heal, selfdestruct, multihit,
             ignoreImmunity, willCrit, ohko, ...
  - species: num, name, id, baseStats{hp,atk,def,spa,spd,spe}, types,
             abilities{0,1,H,S}, weightkg, ...
  - abilities/items: num, name, id, (+ translated handlers)
  - types:   damageTaken{Type: code}, HPivs/HPdvs
  - natures: plus/minus
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "showdown"

_ID_RE = re.compile(r"[^a-z0-9]+")

# Showdown's BASE_MOD. Gens we support; everything inherits behavior down to base.
SUPPORTED_GENS = (1, 2, 3, 4, 9)

_TABLE_NAMES = ("moves", "species", "abilities", "items", "typechart", "natures")


def to_id(text: Any) -> str:
    """Port of Showdown's ``toID``: lowercase, strip non-alphanumerics.

    Accepts strings, dicts with id/name, or entry objects.
    """
    if text is None:
        return ""
    if isinstance(text, DexEntry):
        return text.id
    if isinstance(text, dict):
        text = text.get("id") or text.get("name") or ""
    elif not isinstance(text, str):
        text = getattr(text, "id", None) or getattr(text, "name", None) or str(text)
    return _ID_RE.sub("", text.lower())


class DexEntry:
    """Attribute + dict view over a Showdown data record.

    Missing fields return ``None`` (mirrors JS ``undefined``) so translated
    callbacks like ``if move.secondary:`` read naturally. Registered Python
    effect handlers (if any) are attached as ``self.handlers``.
    """

    __slots__ = ("_d", "handlers", "exists")

    def __init__(self, data: Optional[Dict[str, Any]], handlers: Optional[Dict[str, Any]] = None):
        object.__setattr__(self, "_d", data or {})
        object.__setattr__(self, "handlers", handlers or {})
        object.__setattr__(self, "exists", bool(data))

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails (i.e. not a slot).
        try:
            return self._d[name]
        except KeyError:
            return None

    def get(self, name: str, default: Any = None) -> Any:
        return self._d.get(name, default)

    def has(self, name: str) -> bool:
        return name in self._d

    @property
    def raw(self) -> Dict[str, Any]:
        return self._d

    @property
    def id(self) -> str:
        return self._d.get("id", "")

    @property
    def name(self) -> str:
        return self._d.get("name", "")

    def __bool__(self) -> bool:
        return self.exists

    def __repr__(self) -> str:
        return f"<DexEntry {self._d.get('id', '?')!r}>"


# A non-existent sentinel entry (mirrors Showdown returning an object with
# exists === false rather than null).
_NULL_ENTRY = DexEntry(None)


class DexTable:
    """One category (moves/species/...) for one gen, with id-keyed lookup."""

    def __init__(self, name: str, raw: Dict[str, Dict[str, Any]], dex: "Dex"):
        self._name = name
        self._raw = raw
        self._dex = dex
        self._cache: Dict[str, DexEntry] = {}

    def get(self, name: Any) -> DexEntry:
        if isinstance(name, DexEntry):
            return name
        eid = to_id(name)
        cached = self._cache.get(eid)
        if cached is not None:
            return cached
        data = self._raw.get(eid)
        if data is None:
            return _NULL_ENTRY
        handlers = self._dex._lookup_handlers(self._name, eid)
        entry = DexEntry(data, handlers)
        self._cache[eid] = entry
        return entry

    def all(self) -> List[DexEntry]:
        return [self.get(eid) for eid in self._raw]

    def __contains__(self, name: Any) -> bool:
        return to_id(name) in self._raw

    def __iter__(self) -> Iterable[DexEntry]:
        return iter(self.all())


class ConditionsTable:
    """``dex.conditions.get(id)`` mirroring Showdown's condition resolution.

    Resolves statuses/weather/terrain/volatiles from the conditions table, then
    falls back to move/ability/item conditions (a move id like 'leechseed' that
    attaches a volatile resolves to the move's own effect handlers).
    """

    def __init__(self, raw: Dict[str, Dict[str, Any]], dex: "Dex"):
        self._raw = raw
        self._dex = dex
        self._cache: Dict[str, DexEntry] = {}

    def get_by_id(self, eid: Any) -> DexEntry:
        if isinstance(eid, DexEntry):
            return eid
        cid = to_id(eid)
        if not cid:
            return _NULL_ENTRY
        cached = self._cache.get(cid)
        if cached is not None:
            return cached
        data = self._raw.get(cid)
        if data is not None:
            handlers = self._dex._lookup_handlers("conditions", cid)
            entry = DexEntry(data, handlers)
            self._cache[cid] = entry
            return entry
        # Fallback: move / ability / item conditions.
        for table in (self._dex.moves, self._dex.items, self._dex.abilities):
            entry = table.get(cid)
            if entry:
                self._cache[cid] = entry
                return entry
        return _NULL_ENTRY

    # Showdown exposes both get and getByID; callbacks use both spellings.
    def get(self, eid: Any) -> DexEntry:
        return self.get_by_id(eid)

    getByID = get_by_id  # camelCase alias for verbatim callbacks

    def __contains__(self, eid: Any) -> bool:
        return to_id(eid) in self._raw


def _load_table(gen: int, name: str) -> Dict[str, Dict[str, Any]]:
    path = _DATA_ROOT / f"gen{gen}" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Showdown Dex JSON missing: {path}. Run "
            f"`node scripts/extract_dex_json.js` to (re)generate."
        )
    with open(path) as f:
        return json.load(f)


class Dex:
    """Per-gen data Dex mirroring Showdown's runtime Dex surface."""

    _by_gen: Dict[int, "Dex"] = {}

    def __init__(self, gen: int):
        self.gen = int(gen)
        self.moves = DexTable("moves", _load_table(self.gen, "moves"), self)
        self.species = DexTable("species", _load_table(self.gen, "species"), self)
        self.abilities = DexTable("abilities", _load_table(self.gen, "abilities"), self)
        self.items = DexTable("items", _load_table(self.gen, "items"), self)
        self.types = DexTable("typechart", _load_table(self.gen, "typechart"), self)
        self.natures = DexTable("natures", _load_table(self.gen, "natures"), self)
        self.conditions = ConditionsTable(_load_table(self.gen, "conditions"), self)

    # -- effect handler registry hook -------------------------------------
    def _lookup_handlers(self, table: str, eid: str) -> Dict[str, Any]:
        """Return translated Python effect handlers for (table, id), if any.

        Wired in A4 once ``pokepy.showdown.effects`` exists; until then every
        entry is purely data-driven (no callbacks).
        """
        try:
            from pokepy.showdown import effects
        except Exception:
            return {}
        return effects.get_handlers(self.gen, table, eid)

    def get_active_move(self, move) -> "ActiveMove":
        """Return a fresh mutable ActiveMove (Showdown Dex.getActiveMove)."""
        from pokepy.showdown.active_move import ActiveMove

        if isinstance(move, ActiveMove):
            return move
        entry = self.moves.get(move)
        return ActiveMove(entry)

    getActiveMove = get_active_move

    # -- type effectiveness ------------------------------------------------
    def get_effectiveness(self, source_type: str, target_type: str) -> int:
        """Showdown Dex.getEffectiveness: returns 1 (weak), -1 (resist), 0.

        ``damageTaken`` codes: 0 neutral, 1 weak (2x), 2 resist (0.5x),
        3 immune. Immunity is handled separately via ``get_immunity``.
        """
        ti = self.types.get(target_type)
        if not ti:
            return 0
        code = (ti.damageTaken or {}).get(_type_name(source_type), 0)
        if code == 1:
            return 1
        if code == 2:
            return -1
        return 0

    def get_immunity(self, source_type: str, target_types) -> bool:
        """True if NOT immune (i.e. damage can be dealt). Mirrors Dex.getImmunity."""
        if isinstance(target_types, str):
            target_types = [target_types]
        for ttype in target_types:
            ti = self.types.get(ttype)
            if ti and (ti.damageTaken or {}).get(_type_name(source_type), 0) == 3:
                return False
        return True


def _type_name(t: str) -> str:
    """Normalize a type id/name to the capitalized form used as damageTaken keys."""
    return t[:1].upper() + t[1:].lower() if t else t


def get_dex(gen: int) -> Dex:
    gen = int(gen)
    d = Dex._by_gen.get(gen)
    if d is None:
        d = Dex(gen)
        Dex._by_gen[gen] = d
    return d
