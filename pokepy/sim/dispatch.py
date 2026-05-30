"""Event handler registration — mirrors Showdown effect dispatch tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# Event ids (string names map to Showdown onX suffix)
BeforeMove = "BeforeMove"
BeforeTurn = "BeforeTurn"
ModifyPriority = "ModifyPriority"
# Move pipeline events (battle-actions.ts runMove/useMove/spreadMoveHit).
TryMove = "TryMove"
ModifyMove = "ModifyMove"
TryHit = "TryHit"
Accuracy = "Accuracy"
ModifyDamage = "ModifyDamage"
CriticalHit = "CriticalHit"
Damage = "Damage"
Hit = "Hit"
DamagingHit = "DamagingHit"
AfterHit = "AfterHit"
AfterMoveSecondary = "AfterMoveSecondary"
AfterMoveSecondarySelf = "AfterMoveSecondarySelf"
Update = "Update"
Residual = "Residual"
SwitchIn = "SwitchIn"
SetStatus = "SetStatus"
AfterSetStatus = "AfterSetStatus"
AfterMoveSelf = "AfterMoveSelf"
EndTurn = "EndTurn"

Handler = Callable[..., Any]


@dataclass
class HandlerEntry:
    handler: Handler
    priority: int = 0
    order: Optional[int] = None
    effect_id: str = ""


class DispatchRegistry:
    """Per-format handler tables keyed by effect id then event id."""

    def __init__(self) -> None:
        self._tables: Dict[int, Dict[str, Dict[str, HandlerEntry]]] = {}

    def register(
        self,
        format_id: int,
        effect_id: str,
        event_id: str,
        handler: Handler,
        *,
        priority: int = 0,
        order: Optional[int] = None,
    ) -> None:
        fmt = self._tables.setdefault(int(format_id), {})
        eff = fmt.setdefault(str(effect_id), {})
        eff[str(event_id)] = HandlerEntry(
            handler=handler,
            priority=priority,
            order=order,
            effect_id=str(effect_id),
        )

    def get_handlers(self, format_id: int, effect_id: str) -> Dict[str, HandlerEntry]:
        return self._tables.get(int(format_id), {}).get(str(effect_id), {})

    def clear_format(self, format_id: int) -> None:
        self._tables.pop(int(format_id), None)


GLOBAL_REGISTRY = DispatchRegistry()


def register(
    format_id: int,
    effect_id: str,
    event_id: str,
    handler: Handler,
    *,
    priority: int = 0,
    order: Optional[int] = None,
) -> None:
    GLOBAL_REGISTRY.register(
        format_id, effect_id, event_id, handler, priority=priority, order=order
    )


def get_handlers(format_id: int, effect_id: str) -> Dict[str, HandlerEntry]:
    return GLOBAL_REGISTRY.get_handlers(format_id, effect_id)
