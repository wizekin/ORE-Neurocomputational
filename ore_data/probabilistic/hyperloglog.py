"""HyperLogLog cardinality estimation with compatible merging (Phase E).

Steps 26-31: HyperLogLog estimates distinct counts where approximate
cardinality suffices and exact sets would be too expensive. The precision
parameter p gives m = 2^p registers (Step 27); a stable hash selects the
register and the remaining bits determine the rank statistic (Step 28);
registers keep the maximum rank (Step 29); the harmonic-mean estimator with
small- and large-range corrections estimates cardinality (Step 30); merging
is register-wise maximum and only allowed between compatible sketches
(Step 31). Relative standard error is approximately 1.04/sqrt(m)
(source 4.3).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable


class HyperLogLogError(ValueError):
    """Raised for unusable or incompatible HyperLogLog parameters."""


#: Minimum and maximum precision parameters (register counts 16 .. 65536).
MIN_PRECISION = 4
MAX_PRECISION = 16

#: Hash algorithm identity used by every compatible producer (Step 28).
HLL_HASH_ALGORITHM = "sha256-64"

#: Canonicalization identity used by every compatible producer (Step 28).
HLL_CANONICALIZATION = "ore.data.canonical/0.1.0"


@dataclass(frozen=True, slots=True)
class CardinalityEstimate:
    """Typed cardinality estimate with precision and algorithm metadata (Step 30)."""

    estimate: float
    precision: int
    relative_standard_error: float
    hash_algorithm: str
    canonicalization: str
    correction_applied: str
    algorithm: str = "hyperloglog"
    algorithm_version: str = "0.1.0"
    timestamp: str = ""
    source_watermark: int = 0

    def __post_init__(self) -> None:
        """Attach the estimate timestamp."""
        if not self.timestamp:
            object.__setattr__(self, "timestamp", _now())

    @property
    def exact(self) -> bool:
        """Cardinality estimates are approximate; they do not enumerate elements."""
        return False

    def to_dict(self) -> dict[str, object]:
        """Return the estimate with its full metadata (section 5.1 requirement)."""
        return {
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "estimate": self.estimate,
            "precision": self.precision,
            "registers": 1 << self.precision,
            "relative_standard_error": self.relative_standard_error,
            "hash_algorithm": self.hash_algorithm,
            "canonicalization": self.canonicalization,
            "correction_applied": self.correction_applied,
            "timestamp": self.timestamp,
            "source_watermark": self.source_watermark,
            "exact": self.exact,
        }


def _now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _hash64(item: str) -> int:
    """Stable 64-bit hash of canonical item identity (Step 28)."""
    return int.from_bytes(hashlib.sha256(item.encode("utf-8")).digest()[:8], "big")


def _rank(suffix: int, width: int) -> int:
    """Rank rho(w): position of the leftmost 1-bit in the remaining hash suffix."""
    if suffix == 0:
        return width + 1
    return width - suffix.bit_length() + 1


class HyperLogLog:
    """HyperLogLog cardinality estimator with corrections and compatible merging."""

    def __init__(
        self,
        precision: int,
        *,
        hash_algorithm: str = HLL_HASH_ALGORITHM,
        canonicalization: str = HLL_CANONICALIZATION,
    ) -> None:
        """Create one empty sketch with m = 2^p registers (Step 27)."""
        if not MIN_PRECISION <= precision <= MAX_PRECISION:
            raise HyperLogLogError(
                f"precision must be within [{MIN_PRECISION}, {MAX_PRECISION}]"
            )
        self.precision = precision
        self.m = 1 << precision
        self._registers = [0] * self.m
        self._hash_algorithm = hash_algorithm
        self._canonicalization = canonicalization
        self._watermark = 0

    # -- update -------------------------------------------------------------

    def add(self, item: str, watermark: int | None = None) -> None:
        """Update the selected register with the new rank (Steps 28-29)."""
        hashed = _hash64(str(item))
        index = hashed >> (64 - self.precision)
        suffix = hashed & ((1 << (64 - self.precision)) - 1)
        rank = _rank(suffix, 64 - self.precision)
        if rank > self._registers[index]:
            self._registers[index] = rank
        if watermark is not None:
            self._watermark = max(self._watermark, int(watermark))

    def add_all(self, items: Iterable[str]) -> None:
        """Add many canonical items."""
        for item in items:
            self.add(item)

    # -- estimation -----------------------------------------------------------

    def estimate(self) -> CardinalityEstimate:
        """Estimate cardinality with the corrected harmonic-mean estimator (Step 30)."""
        register_sum = sum(2.0 ** (-value) for value in self._registers)
        alpha = 0.7213 / (1.0 + 1.079 / self.m)
        raw = alpha * self.m * self.m / register_sum
        zeros = sum(1 for value in self._registers if value == 0)
        correction = "none"
        value = raw
        if raw <= 2.5 * self.m and zeros > 0:
            # Small-range correction: linear counting (Step 30).
            value = self.m * math.log(self.m / zeros)
            correction = "small_range_linear_counting"
        elif raw > (1 << 32) / 30.0:
            # Large-range correction kept for estimator completeness; with
            # 64-bit hashes this branch is effectively unreachable.
            value = -(1 << 32) * math.log(1.0 - raw / float(1 << 32))
            correction = "large_range"
        return CardinalityEstimate(
            estimate=value,
            precision=self.precision,
            relative_standard_error=1.04 / math.sqrt(self.m),
            hash_algorithm=self._hash_algorithm,
            canonicalization=self._canonicalization,
            correction_applied=correction,
            timestamp=_now(),
            source_watermark=self._watermark,
        )

    # -- merging --------------------------------------------------------------

    def compatible_with(self, other: "HyperLogLog") -> bool:
        """Return whether two sketches share hash, canonicalization and precision (Step 31)."""
        return (
            other.precision == self.precision
            and other._hash_algorithm == self._hash_algorithm
            and other._canonicalization == self._canonicalization
        )

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        """Merge compatible sketches by register-wise maximum (Step 31)."""
        if not self.compatible_with(other):
            raise HyperLogLogError(
                "cannot merge incompatible sketches: hash algorithm, "
                "canonicalization and precision must match (Step 31)"
            )
        merged = HyperLogLog(self.precision)
        merged._hash_algorithm = self._hash_algorithm
        merged._canonicalization = self._canonicalization
        merged._registers = [
            max(a, b) for a, b in zip(self._registers, other._registers)
        ]
        merged._watermark = max(self._watermark, other._watermark)
        return merged

    # -- introspection -----------------------------------------------------------

    @property
    def watermark(self) -> int:
        """Return the source watermark of this sketch's updates."""
        return self._watermark

    def memory_bytes(self) -> int:
        """Return the register array allocation size (bytes per sketch)."""
        return self.m * 1  # one byte per register at p <= 16

    def describe(self) -> dict[str, object]:
        """Return the sketch metadata for observability."""
        return {
            "algorithm": "hyperloglog",
            "version": "0.1.0",
            "precision": self.precision,
            "registers": self.m,
            "hash_algorithm": self._hash_algorithm,
            "canonicalization": self._canonicalization,
            "relative_standard_error": 1.04 / math.sqrt(self.m),
            "memory_bytes": self.memory_bytes(),
        }


def required_precision_for(target_relative_error: float) -> int:
    """Return the smallest precision achieving a target relative standard error (Step 27)."""
    if not 0.0 < target_relative_error < 1.0:
        raise HyperLogLogError("target_relative_error must be within (0, 1)")
    for precision in range(MIN_PRECISION, MAX_PRECISION + 1):
        if 1.04 / math.sqrt(1 << precision) <= target_relative_error:
            return precision
    raise HyperLogLogError(
        f"no precision <= {MAX_PRECISION} achieves {target_relative_error}"
    )


__all__ = [
    "HyperLogLogError",
    "MIN_PRECISION",
    "MAX_PRECISION",
    "HLL_HASH_ALGORITHM",
    "HLL_CANONICALIZATION",
    "CardinalityEstimate",
    "HyperLogLog",
    "required_precision_for",
]
