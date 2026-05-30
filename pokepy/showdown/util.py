"""Low-level numeric primitives ported verbatim from Showdown.

Kept as free functions (also exposed as Battle methods) so translated callbacks
can call ``self.battle.trunc(...)`` etc. exactly like Showdown's ``this.trunc``.
"""

from __future__ import annotations

from typing import Sequence, Union

Number = Union[int, float]


def trunc(num: Number, bits: int = 0) -> int:
    """Port of Dex.trunc (sim/dex.ts:365).

    ``trunc(num)``       -> ToUint32(num)              (floor for non-negative)
    ``trunc(num, bits)`` -> ToUint32(num) % 2**bits
    Showdown only ever truncates non-negative values here, so int() (toward
    zero) matches JS ``>>> 0`` for the values that occur.
    """
    n = int(num) & 0xFFFFFFFF
    if bits:
        return n % (1 << bits)
    return n


def clamp_int_range(num: Number, min_val: int = None, max_val: int = None) -> int:
    """Port of Battle.clampIntRange (sim/battle.ts)."""
    num = int(num)
    if min_val is not None and num < min_val:
        num = min_val
    if max_val is not None and num > max_val:
        num = max_val
    return num


def modify(value: Number, numerator, denominator: int = 1) -> int:
    """Port of Battle.modify (sim/battle.ts:2345).

    ``modify(value, [num, denom])`` or ``modify(value, fraction)``.
    Returns ``tr((tr(value * modifier) + 2048 - 1) / 4096)`` where
    ``modifier = tr(numerator * 4096 / denominator)``.
    """
    if isinstance(numerator, (list, tuple)):
        denominator = numerator[1]
        numerator = numerator[0]
    modifier = trunc(numerator * 4096 / denominator)
    return trunc((trunc(value * modifier) + 2048 - 1) / 4096)


def chain(previous_mod, next_mod) -> float:
    """Port of Battle.chain (sim/battle.ts:2318). Returns a 1.0-based float."""
    if isinstance(previous_mod, (list, tuple)):
        previous_mod = trunc(previous_mod[0] * 4096 / previous_mod[1])
    else:
        previous_mod = trunc(previous_mod * 4096)
    if isinstance(next_mod, (list, tuple)):
        next_mod = trunc(next_mod[0] * 4096 / next_mod[1])
    else:
        next_mod = trunc(next_mod * 4096)
    return ((previous_mod * next_mod + 2048) >> 12) / 4096
