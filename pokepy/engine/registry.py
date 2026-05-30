"""Effect registry: (table, id, event) -> pokepy.effects handler.

Binds the existing scalar bitpack effect library to Showdown-style dispatch
events without rewriting effect bodies.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

RegistryKey = Tuple[str, int, str]
Handler = Callable[..., Any]

# Showdown camelCase event -> snake_case effect function prefix/suffix.
_EVENT_ALIASES: Dict[str, str] = {
    "ModifySpe": "modify_spe",
    "ModifyAtk": "modify_atk",
    "ModifyDef": "modify_def",
    "ModifySpA": "modify_spa",
    "ModifySpD": "modify_spd",
    "ModifyAccuracy": "modify_accuracy",
    "ModifyEvasion": "modify_evasion",
    "ModifyDamage": "modify_damage",
    "BasePower": "base_power",
    "TryHit": "try_hit",
    "TryMove": "try_move",
    "SwitchIn": "switch_in",
    "SwitchOut": "switch_out",
    "AfterSwitchInSelf": "after_switch_in_self",
    "Residual": "residual",
    "DamagingHit": "damaging_hit",
    "Hit": "on_hit",
}


def _snake_event(event: str) -> str:
    if event in _EVENT_ALIASES:
        return _EVENT_ALIASES[event]
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", event)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


@dataclass
class EffectRegistry:
    """Maps Showdown dispatch keys to pokepy.effects callables."""

    handlers: Dict[RegistryKey, Handler] = field(default_factory=dict)
    by_event: Dict[str, list] = field(default_factory=dict)

    def register(
        self,
        table: str,
        effect_id: int,
        event: str,
        fn: Handler,
        *,
        priority: int = 0,
        order: bool = False,
    ) -> None:
        key = (table, int(effect_id), event)
        self.handlers[key] = fn
        self.by_event.setdefault(event, []).append(
            {
                "table": table,
                "id": int(effect_id),
                "fn": fn,
                "priority": priority,
                "order": order,
            }
        )

    def lookup(self, table: str, effect_id: int, event: str) -> Optional[Handler]:
        return self.handlers.get((table, int(effect_id), event))

    def handlers_for_event(self, event: str) -> list:
        return list(self.by_event.get(event, []))


def _discover_effect_module_handlers() -> Dict[str, Handler]:
    """Collect public callables from pokepy.effects submodules."""
    import pokepy.effects as fx_pkg

    out: Dict[str, Handler] = {}
    for name in dir(fx_pkg):
        if name.startswith("_"):
            continue
        obj = getattr(fx_pkg, name)
        if callable(obj) and getattr(obj, "__module__", "").startswith(
            "pokepy.effects"
        ):
            out[name] = obj
    return out


def _infer_table_and_event(fn_name: str) -> Optional[Tuple[str, str]]:
    """Best-effort mapping from effect helper name -> (table, event)."""
    name = fn_name
    if name.startswith("apply_"):
        body = name[6:]
        if "switch_in" in body or body == "switch_in_ability":
            return "ability", "SwitchIn"
        if "switch_out" in body or "on_switch_out" in body:
            return "ability", "SwitchOut"
        if "from_move" in body or "on_damaging_hit" in body:
            return "move", "Hit"
        if "end_of_turn" in body or "residual" in body:
            return "condition", "Residual"
        if "berry" in body or "leftovers" in body or "sludge" in body:
            return "item", "Residual"
        if "weather" in body or "terrain" in body:
            return "condition", "Residual"
        if "hazard" in body:
            return "condition", "SwitchIn"
        return "move", "Hit"
    if name.startswith("check_"):
        return "volatile", "TryHit"
    if name.startswith("get_effective_"):
        return "ability", "ModifySpe" if "speed" in name else "ModifyDamage"
    if name.startswith("decrement_"):
        return "volatile", "Residual"
    return None


def build_default_registry() -> EffectRegistry:
    """Auto-bind pokepy.effects public functions to dispatch events."""
    reg = EffectRegistry()
    fns = _discover_effect_module_handlers()
    for fn_name, fn in fns.items():
        inferred = _infer_table_and_event(fn_name)
        if inferred is None:
            continue
        table, event = inferred
        # Use fn_name hash bucket as pseudo-id for generic handlers.
        pseudo_id = abs(hash(fn_name)) % 100000
        reg.register(table, pseudo_id, event, fn)
    return reg


DEFAULT_REGISTRY = build_default_registry()

__all__ = [
    "EffectRegistry",
    "RegistryKey",
    "Handler",
    "DEFAULT_REGISTRY",
    "build_default_registry",
]
