"""Translated effect-callback registry (the Phase B work surface).

Showdown data entries (moves/abilities/items/conditions) carry callback
functions (onHit, onModifyMove, basePowerCallback, onResidual, ...). Those are
dropped from the JSON dump, so each one that matters is hand-translated to
Python and registered here, keyed by (table, id) with optional per-gen
overrides. The Dex merges the registered handlers onto the entry it returns as
``entry.handlers`` (event_name -> callable).

Data-driven fields (secondary, boosts, status, recoil, drain, ...) need NO
registered callback - the move pipeline reads them directly. Only genuinely
custom logic needs a translation here.

Gen inheritance: a gen looks up its own override first, then falls back to base
(gen9 BASE_MOD). See TRANSLATION_GUIDE.md for the worked examples and the
event-name vocabulary.

Usage (Phase B):

    from pokepy.showdown.effects import register

    @register("moves", "brickbreak")          # base (all gens)
    class BrickBreak:
        def on_try_hit(battle, target, source, move): ...

    @register("conditions", "brn", gen=1)      # gen1-specific override
    class Gen1Burn:
        def on_residual(battle, pokemon): ...

The method-name -> event-name mapping is snake_case of ``on<Event>``:
``on_residual`` -> ``onResidual``, ``on_modify_atk`` -> ``onModifyAtk``.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional, Tuple

# (table, id) -> {event_name: callable}  (base / all-gen)
_BASE: Dict[Tuple[str, str], Dict[str, Callable]] = {}
# (gen, table, id) -> {event_name: callable}
_GEN: Dict[Tuple[int, str, str], Dict[str, Callable]] = {}

_METHOD_RE = re.compile(r"_([a-z0-9])")


def _method_to_event(method_name: str) -> Optional[str]:
    """``on_modify_atk`` -> ``onModifyAtk``; non-handlers return None."""
    if not method_name.startswith("on_"):
        return None
    # on_modify_atk -> on + ModifyAtk
    rest = method_name[3:]
    camel = _METHOD_RE.sub(lambda m: m.group(1).upper(), "_" + rest)
    return "on" + camel[0].upper() + camel[1:]


def _collect_handlers(obj: Any) -> Dict[str, Callable]:
    handlers: Dict[str, Callable] = {}
    for attr in dir(obj):
        if not attr.startswith("on_"):
            continue
        event = _method_to_event(attr)
        if event is None:
            continue
        handlers[event] = getattr(obj, attr)
    return handlers


def register(table: str, eid: str, gen: Optional[int] = None):
    """Decorator: register a class/object's ``on_*`` methods as effect handlers."""

    def deco(obj):
        handlers = _collect_handlers(obj)
        if gen is None:
            _BASE.setdefault((table, eid), {}).update(handlers)
        else:
            _GEN.setdefault((int(gen), table, eid), {}).update(handlers)
        return obj

    return deco


def get_handlers(gen: int, table: str, eid: str) -> Dict[str, Callable]:
    """Resolve handlers for (gen, table, id): gen override merged over base."""
    merged: Dict[str, Callable] = {}
    base = _BASE.get((table, eid))
    if base:
        merged.update(base)
    gen_specific = _GEN.get((int(gen), table, eid))
    if gen_specific:
        merged.update(gen_specific)
    return merged


def unported(table: str, eid: str, event: str, gen: Optional[int] = None) -> None:
    """Raise a loud, grep-friendly failure for a missing callback translation.

    Use this inside ported code when you know Showdown invokes a callback that
    is not yet registered — never silently skip custom logic.
    """
    suffix = f" (gen{gen})" if gen is not None else ""
    raise NotImplementedError(f"unported effect callback: {table}/{eid} on{event}{suffix}")


def _load_registered_modules() -> None:
    """Import all sibling ``effects/*.py`` modules so ``@register`` runs."""
    import importlib
    import pkgutil
    from pathlib import Path

    pkg_dir = Path(__file__).resolve().parent
    for info in pkgutil.iter_modules([str(pkg_dir)]):
        if info.name.startswith("_") or info.name == "__init__":
            continue
        importlib.import_module(f"{__name__}.{info.name}")


_load_registered_modules()
