"""Prime-field arithmetic shared by secret sharing, MPC and VDAF.

All multi-party protocols in this package operate over the Mersenne prime field
F_p with p = 2^61 - 1, which keeps reductions cheap in Python while supporting
statistically-secret additive sharing for the cohort sizes involved.
"""
from __future__ import annotations

MERSENNE_61 = (1 << 61) - 1


class PrimeField:
    """Modular arithmetic helper over a fixed prime modulus."""

    def __init__(self, modulus: int = MERSENNE_61) -> None:
        if modulus < 3 or modulus % 2 == 0:
            raise ValueError("modulus must be an odd prime")
        self.modulus = int(modulus)

    def add(self, a: int, b: int) -> int:
        return (a + b) % self.modulus

    def sub(self, a: int, b: int) -> int:
        return (a - b) % self.modulus

    def mul(self, a: int, b: int) -> int:
        return (a * b) % self.modulus

    def neg(self, a: int) -> int:
        return (-a) % self.modulus

    def inv(self, a: int) -> int:
        if a % self.modulus == 0:
            raise ZeroDivisionError("no inverse of zero in the field")
        return pow(a, -1, self.modulus)

    def reduce(self, a: int) -> int:
        return a % self.modulus

    def random(self, rng) -> int:
        """Uniform field element from a numpy Generator or secrets-style source."""
        return int(rng.integers(0, self.modulus)) if hasattr(rng, "integers") else int(rng.randbelow(self.modulus))

    def batch_add(self, left: list[int], right: list[int]) -> list[int]:
        if len(left) != len(right):
            raise ValueError("batch operands must have equal length")
        return [self.add(a, b) for a, b in zip(left, right)]

    def batch_sub(self, left: list[int], right: list[int]) -> list[int]:
        if len(left) != len(right):
            raise ValueError("batch operands must have equal length")
        return [self.sub(a, b) for a, b in zip(left, right)]
