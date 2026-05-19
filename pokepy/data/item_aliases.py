"""Internal item aliases needed for strict Showdown parity.

The extracted id mappings currently collapse `sitrusberry` and `goldberry`
onto the same numeric item id. Showdown does not: Gold Berry heals a flat 30
HP, while Sitrus Berry heals 25% max HP. We keep Sitrus on the extracted id
and route Gold Berry through a synthetic internal id so battle-state logic can
distinguish them.
"""
from __future__ import annotations

from typing import Dict

ITEM_GOLD_BERRY_INTERNAL = 3001
ITEM_SITRUS_BERRY_COLLAPSED = 448

def _normalize(name: str) -> str:
    return "".join(c for c in str(name) if c.isalnum()).lower()

def encode_item_id(name: str, item_to_idx: Dict[str, int]) -> int:
    """Map a raw item name to the internal battle-state item id."""
    key = _normalize(name)
    if key == "goldberry":
        return ITEM_GOLD_BERRY_INTERNAL
    if key in item_to_idx:
        return int(item_to_idx[key])
    if str(name) in item_to_idx:
        return int(item_to_idx[str(name)])
    return -1

def pack_item_name(item_id: int, item_names: Dict[int, str]) -> str:
    """Map an internal item id back to the Showdown item name."""
    if int(item_id) == ITEM_GOLD_BERRY_INTERNAL:
        return "Gold Berry"
    if int(item_id) == ITEM_SITRUS_BERRY_COLLAPSED:
        # The extracted mappings collapse `sitrusberry` and `goldberry` onto
        # 448, but the current Gen 9 OU source pool only uses Sitrus Berry.
        # Pack the live battle-state id back to the source Showdown item.
        return "Sitrus Berry"
    return item_names.get(int(item_id), "")
