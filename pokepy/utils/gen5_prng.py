"""Gen 5 LCG PRNG, matches the Gen 5 LCG used by Pokemon Showdown.

LCG: x_{n+1} = (A * x_n + C) % 2^64, output = upper 32 bits.
A = 0x5D588B656C078965, C = 0x269EC3.

This is the canonical Pokemon Showdown PRNG. Used by pokepy's engine for
damage rolls and accuracy checks. Seeded with the same initial state, two
runs advance bit-for-bit alongside each other.
"""

from __future__ import annotations

from typing import Tuple

A = 0x5D588B656C078965
C = 0x269EC3
M = 2**64


def gen5_seed_from_array(seed_array: Tuple[int, int, int, int]) -> int:
    return (
        (seed_array[0] << 48)
        | (seed_array[1] << 32)
        | (seed_array[2] << 16)
        | seed_array[3]
    )


def gen5_seed_to_array(seed: int) -> Tuple[int, int, int, int]:
    return (
        (seed >> 48) & 0xFFFF,
        (seed >> 32) & 0xFFFF,
        (seed >> 16) & 0xFFFF,
        seed & 0xFFFF,
    )


class Gen5PRNG:
    """Showdown-compatible Gen 5 LCG."""

    __slots__ = ("seed",)

    def __init__(self, seed: Tuple[int, int, int, int] = (12345, 12345, 12345, 12345)):
        self.seed = gen5_seed_from_array(seed)

    def next(self) -> int:
        self.seed = (A * self.seed + C) % M
        return (self.seed >> 32) & 0xFFFFFFFF

    def random(self, from_val=None, to_val=None) -> float:
        """Showdown-compatible random() (sim/prng.ts:91-103).

        - random() → float in [0, 1)
        - random(n) → int in [0, n)
        - random(m, n) → int in [m, n)

        All three forms consume exactly ONE PRNG frame.
        """
        value = self.next()
        if from_val is None:
            return value / (2**32)
        from_val = int(from_val)
        if to_val is None or to_val == 0:
            return int(value * from_val / (2**32))
        to_val = int(to_val)
        return int(value * (to_val - from_val) / (2**32)) + from_val

    def random_chance(self, numerator: int, denominator: int) -> bool:
        """Showdown PRNG.randomChance: returns True with probability num/denom.

        Equivalent to `random(denominator) < numerator`.
        """
        return self.random(denominator) < numerator

    def random_range(self, min_val: float, max_val: float) -> float:
        """Real-valued range — pokepy-only convenience, NOT a Showdown call.

        Consumes one PRNG frame and returns a real number in [min_val, max_val).
        """
        return min_val + self.random() * (max_val - min_val)

    def damage_roll(self) -> float:
        """Convenience for the damage roll.

        Showdown formula: `tr(tr(baseDamage * (100 - random(16))) / 100)`,
        i.e. one `random(16)` call and integer arithmetic. Use `random(16)`
        directly in damage code; this helper exists only for callers that
        want a [0.85, 1.0) float (and is NOT bit-identical to Showdown).
        """
        return self.random_range(0.85, 1.0)

    def get_seed_array(self) -> Tuple[int, int, int, int]:
        return gen5_seed_to_array(self.seed)
