"""Bloom filter membership acceleration with watermark safety (Phase C).

Steps 13-19: Bloom filters accelerate high-volume membership prechecks. The
bit array is sized from expected cardinality and target error
(``m = ceil(-n ln p / (ln 2)^2)``), probes use double hashing
(``k = round((m/n) ln 2)``), and every query goes through the safe two-stage
path: a negative may skip the exact lookup only when the filter is complete
for the queried authoritative watermark, and a positive always performs the
exact lookup (Step 17). Standard filters never delete (Step 18); a counting
filter with underflow guards exists separately. Reconciliation samples
absent keys to measure the observed false-positive rate and rebuilds beyond
policy (Step 19).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

#: Number of bits per addressable unit: one bit array.
BYTE_BITS = 8


class MembershipVerdict(str, Enum):
    """Safe two-stage membership verdicts (Step 17)."""

    DEFINITELY_ABSENT = "definitely_absent"
    POSSIBLY_PRESENT = "possibly_present"
    UNKNOWN_INCOMPLETE = "unknown_incomplete"


class BloomSizingError(ValueError):
    """Raised for unusable Bloom filter parameters."""


@dataclass(frozen=True, slots=True)
class BloomParameters:
    """Sizing parameters derived from expected cardinality and target error (Steps 14-15)."""

    expected_keys: int
    target_false_positive_rate: float
    bits: int
    hash_probes: int

    def __post_init__(self) -> None:
        """Validate derived parameters."""
        if self.expected_keys <= 0:
            raise BloomSizingError("expected_keys must be positive")
        if not 0.0 < self.target_false_positive_rate < 1.0:
            raise BloomSizingError("target_false_positive_rate must be within (0, 1)")
        if self.bits <= 0 or self.hash_probes <= 0:
            raise BloomSizingError("derived bits and hash_probes must be positive")

    @classmethod
    def from_cardinality_and_error(
        cls,
        expected_keys: int,
        target_false_positive_rate: float,
    ) -> "BloomParameters":
        """Derive optimal sizing: m = ceil(-n ln p / (ln 2)^2), k = round(m/n ln 2)."""
        if expected_keys <= 0:
            raise BloomSizingError("expected_keys must be positive")
        if not 0.0 < target_false_positive_rate < 1.0:
            raise BloomSizingError("target_false_positive_rate must be within (0, 1)")
        numerator = -float(expected_keys) * math.log(target_false_positive_rate)
        bits = int(math.ceil(numerator / (math.log(2.0) ** 2)))
        # Round up to whole bytes for practical allocation (Step 14).
        bits = max(8, (bits + BYTE_BITS - 1) // BYTE_BITS * BYTE_BITS)
        probes = max(1, int(round((bits / expected_keys) * math.log(2.0))))
        return cls(
            expected_keys=expected_keys,
            target_false_positive_rate=target_false_positive_rate,
            bits=bits,
            hash_probes=probes,
        )

    @classmethod
    def from_memory_budget(
        cls,
        memory_bytes: int,
        expected_keys: int,
    ) -> "BloomParameters":
        """Derive parameters from a hard memory budget (Step 6)."""
        if memory_bytes <= 0:
            raise BloomSizingError("memory_bytes must be positive")
        bits = memory_bytes * BYTE_BITS
        probes = max(1, int(round((bits / expected_keys) * math.log(2.0))))
        target = theoretical_false_positive_rate(bits, expected_keys, probes)
        return cls(
            expected_keys=expected_keys,
            target_false_positive_rate=target,
            bits=bits,
            hash_probes=probes,
        )

    def bytes_required(self) -> int:
        """Return the allocated byte count of the bit array."""
        return (self.bits + BYTE_BITS - 1) // BYTE_BITS

    def to_dict(self) -> dict[str, float | int]:
        """Return the parameters as a plain dictionary."""
        return {
            "expected_keys": self.expected_keys,
            "target_false_positive_rate": self.target_false_positive_rate,
            "bits": self.bits,
            "hash_probes": self.hash_probes,
        }


def theoretical_false_positive_rate(bits: int, keys: int, probes: int) -> float:
    """Return p ~ (1 - e^(-kn/m))^k (source section 4.1)."""
    if bits <= 0 or keys <= 0 or probes <= 0:
        raise ValueError("bits, keys and probes must be positive")
    fill = 1.0 - math.exp(-float(probes * keys) / bits)
    return fill**probes


def _double_hash_positions(key: str, bits: int, probes: int, domain: str) -> tuple[int, ...]:
    """Derive k probe positions with double hashing (Step 15).

    Two 64-bit halves of one SHA-256 digest act as h1 and h2; probe i is
    ``(h1 + i * h2) mod m``. Double hashing avoids k independent expensive
    hashes while keeping the empirical behavior close to optimal.
    """
    digest = hashlib.sha256(f"{domain}|{key}".encode("utf-8")).digest()
    h1 = int.from_bytes(digest[:8], "big")
    h2 = int.from_bytes(digest[8:16], "big") | 1  # odd, so it is invertible mod 2^k
    return tuple(
        (h1 + i * h2) % bits for i in range(probes)
    )


@dataclass(frozen=True, slots=True)
class MembershipHint:
    """Result of one safe two-stage membership query (Step 17)."""

    verdict: MembershipVerdict
    key: str
    filter_watermark: int
    filter_complete: bool
    queried_watermark: int

    @property
    def safe_to_skip_exact_lookup(self) -> bool:
        """A negative shortcut is usable only for a complete filter at a covering watermark."""
        return self.verdict is MembershipVerdict.DEFINITELY_ABSENT

    def to_dict(self) -> dict[str, object]:
        """Return the hint as a plain dictionary."""
        return {
            "verdict": self.verdict.value,
            "key": self.key,
            "filter_watermark": self.filter_watermark,
            "filter_complete": self.filter_complete,
            "queried_watermark": self.queried_watermark,
            "safe_to_skip_exact_lookup": self.safe_to_skip_exact_lookup,
        }


class BloomFilter:
    """Standard Bloom filter with insertion watermarks (Steps 14-17, 19).

    The filter records the authoritative sequence through which every
    inserted key was applied. A negative is authoritative only when the
    filter is complete and its watermark covers the queried version;
    otherwise queries return ``UNKNOWN_INCOMPLETE`` and the caller must fall
    back to the authoritative index (Step 19 safeguard).
    """

    def __init__(
        self,
        parameters: BloomParameters,
        *,
        domain: str = "ore.data.bloom",
    ) -> None:
        """Create one empty filter with given parameters and hash namespace."""
        if not domain.strip():
            raise ValueError("domain must be non-empty")
        self.parameters = parameters
        self.domain = domain
        self._bits = bytearray(parameters.bytes_required())
        self._insertions = 0
        self._insertion_watermark = 0
        self._complete = False
        self.version = "0.1.0"

    # -- mutation ----------------------------------------------------------

    def add(self, key: str, watermark: int | None = None) -> None:
        """Insert one key (Step 16).

        ``watermark`` is the authoritative sequence at which this key became
        eligible; the insertion watermark advances monotonically.
        """
        for position in _double_hash_positions(
            key, self.parameters.bits, self.parameters.hash_probes, self.domain
        ):
            self._bits[position // BYTE_BITS] |= 1 << (position % BYTE_BITS)
        self._insertions += 1
        if watermark is not None:
            self._insertion_watermark = max(self._insertion_watermark, int(watermark))

    def mark_complete(self, watermark: int) -> None:
        """Mark the filter complete for one authoritative watermark (Step 17)."""
        if watermark < self._insertion_watermark:
            raise ValueError("completeness watermark cannot precede the insertion watermark")
        self._insertion_watermark = watermark
        self._complete = True

    def rebuild(self, keys: Iterable[str], watermark: int) -> "BloomFilter":
        """Rebuild as an immutable generation from one key snapshot (Step 18)."""
        rebuilt = BloomFilter(self.parameters, domain=self.domain)
        for key in keys:
            rebuilt.add(key)
        rebuilt.mark_complete(watermark)
        return rebuilt

    # -- query -------------------------------------------------------------

    def might_contain(self, key: str) -> bool:
        """Return the raw filter answer: possibly-present bits (Step 17)."""
        for position in _double_hash_positions(
            key, self.parameters.bits, self.parameters.hash_probes, self.domain
        ):
            if not (self._bits[position // BYTE_BITS] & (1 << (position % BYTE_BITS))):
                return False
        return True

    def membership_hint(
        self,
        key: str,
        authoritative_watermark: int,
    ) -> MembershipHint:
        """Return the safe two-stage membership verdict (Step 17).

        ``DEFINITELY_ABSENT`` only when the filter is known complete for the
        authoritative version being queried; ``POSSIBLY_PRESENT`` always
        proceeds to the exact lookup.
        """
        if not self._complete or authoritative_watermark > self._insertion_watermark:
            return MembershipHint(
                MembershipVerdict.UNKNOWN_INCOMPLETE,
                key,
                self._insertion_watermark,
                self._complete,
                authoritative_watermark,
            )
        if self.might_contain(key):
            return MembershipHint(
                MembershipVerdict.POSSIBLY_PRESENT,
                key,
                self._insertion_watermark,
                self._complete,
                authoritative_watermark,
            )
        return MembershipHint(
            MembershipVerdict.DEFINITELY_ABSENT,
            key,
            self._insertion_watermark,
            self._complete,
            authoritative_watermark,
        )

    def delete(self, key: str) -> None:  # noqa: ARG002 - deliberate refusal
        """Refuse deletion: standard Bloom filters cannot safely delete (Step 18)."""
        raise NotImplementedError(
            "a standard Bloom filter does not support safe arbitrary deletion; "
            "use rebuild(), immutable generations, or CountingBloomFilter (Step 18)"
        )

    # -- introspection -----------------------------------------------------

    @property
    def insertion_watermark(self) -> int:
        """Return the authoritative sequence covered by this filter."""
        return self._insertion_watermark

    @property
    def complete(self) -> bool:
        """Return whether the filter is known complete for its watermark."""
        return self._complete

    @property
    def insertions(self) -> int:
        """Return the number of keys inserted."""
        return self._insertions

    def memory_bytes(self) -> int:
        """Return the bit-array allocation size (metric: bytes per indexed key)."""
        return len(self._bits)

    def bytes_per_key(self) -> float:
        """Return allocated bytes per indexed key."""
        return self.memory_bytes() / max(1, self._insertions)

    def describe(self) -> dict[str, object]:
        """Return the filter's metadata for observability."""
        return {
            "algorithm": "bloom_filter",
            "version": self.version,
            "domain": self.domain,
            "parameters": self.parameters.to_dict(),
            "theoretical_fp_rate": theoretical_false_positive_rate(
                self.parameters.bits,
                max(1, self._insertions),
                self.parameters.hash_probes,
            ),
            "insertions": self._insertions,
            "insertion_watermark": self._insertion_watermark,
            "complete": self._complete,
        }


class CountingBloomFilter:
    """Counting Bloom filter for deletions, with underflow guards (Step 18).

    Counting filters introduce counter-overflow, concurrency and memory
    trade-offs and are therefore separately validated: counters saturate at
    15 (4-bit nibbles), removal refuses to underflow, and a rebuilt
    generation remains available for cases the counting filter cannot
    handle.
    """

    MAX_COUNT = 15

    def __init__(self, parameters: BloomParameters, *, domain: str = "ore.data.counting_bloom") -> None:
        """Create one empty counting filter."""
        self.parameters = parameters
        self.domain = domain
        # One 4-bit nibble per filter position: bits nibbles = ceil(bits/2) bytes.
        self._counters = bytearray((parameters.bits + 1) // 2)
        self._keys: dict[str, int] = {}

    def _positions(self, key: str) -> tuple[int, ...]:
        return _double_hash_positions(
            key, self.parameters.bits, self.parameters.hash_probes, self.domain
        )

    def _read(self, position: int) -> int:
        byte = self._counters[position // 2]
        return (byte >> 4) if (position % 2 == 0) else (byte & 0x0F)

    def _write(self, position: int, value: int) -> None:
        index, shift = position // 2, (position % 2 == 0)
        current = self._counters[index]
        if shift:
            self._counters[index] = (current & 0x0F) | ((value & 0x0F) << 4)
        else:
            self._counters[index] = (current & 0xF0) | (value & 0x0F)

    def add(self, key: str) -> None:
        """Insert one key, saturating counters at the maximum (Step 18 trade-off)."""
        for position in self._positions(key):
            current = self._read(position)
            if current < self.MAX_COUNT:
                self._write(position, current + 1)
        self._keys[key] = self._keys.get(key, 0) + 1

    def remove(self, key: str) -> None:
        """Remove one key; refuse underflow instead of corrupting membership."""
        if self._keys.get(key, 0) <= 0:
            raise ValueError(
                f"cannot remove {key!r}: counter underflow would corrupt "
                "membership behavior (Step 18)"
            )
        for position in self._positions(key):
            current = self._read(position)
            if current > 0:
                self._write(position, current - 1)
        self._keys[key] -= 1

    def might_contain(self, key: str) -> bool:
        """Return possibly-present membership under counting semantics."""
        return all(self._read(position) > 0 for position in self._positions(key))


@dataclass(frozen=True, slots=True)
class BloomReconciliationReport:
    """Step 19 reconciliation outcome for one filter."""

    sampled_absent: int
    observed_false_positives: int
    observed_false_positive_rate: float
    theoretical_rate: float
    target_rate: float
    watermark_verified: bool
    recommendation: str

    def to_dict(self) -> dict[str, object]:
        """Return the report as a plain dictionary."""
        return {
            "sampled_absent": self.sampled_absent,
            "observed_false_positives": self.observed_false_positives,
            "observed_false_positive_rate": self.observed_false_positive_rate,
            "theoretical_rate": self.theoretical_rate,
            "target_rate": self.target_rate,
            "watermark_verified": self.watermark_verified,
            "recommendation": self.recommendation,
        }


def estimate_false_positive_rate(
    filter_: BloomFilter,
    absent_keys: Sequence[str],
) -> float:
    """Measure the observed false-positive rate on an absent-key sample (Step 19)."""
    if not absent_keys:
        raise ValueError("absent_keys sample must be non-empty")
    hits = sum(1 for key in absent_keys if filter_.might_contain(key))
    return hits / len(absent_keys)


def reconcile_bloom_filter(
    filter_: BloomFilter,
    *,
    present_keys: Sequence[str],
    absent_keys: Sequence[str],
    authoritative_watermark: int,
) -> BloomReconciliationReport:
    """Reconcile one filter against authoritative data (Step 19).

    Samples exact present records (false negatives are impossible by
    construction and still verified) and absent keys (false positives are
    measured), verifies the completeness watermark, and recommends a
    rebuild when the observed error or staleness exceeds policy.
    """
    # False negatives are impossible for a complete filter; verify anyway.
    negatives = sum(1 for key in present_keys if not filter_.might_contain(key))
    observed_fp = estimate_false_positive_rate(filter_, absent_keys) if absent_keys else 0.0
    theoretical = theoretical_false_positive_rate(
        filter_.parameters.bits,
        max(1, filter_.insertions),
        filter_.parameters.hash_probes,
    )
    watermark_verified = (
        filter_.complete and filter_.insertion_watermark >= authoritative_watermark
    )
    target = filter_.parameters.target_false_positive_rate
    exceeds_error = observed_fp > target
    stale = not watermark_verified
    if stale:
        recommendation = (
            "bypass_negative_shortcut_and_query_authoritative_index; rebuild at watermark "
            f"{authoritative_watermark}"
        )
    elif exceeds_error:
        recommendation = "rebuild: observed false-positive rate exceeds target policy"
    else:
        recommendation = "healthy"
    return BloomReconciliationReport(
        sampled_absent=len(absent_keys),
        observed_false_positives=int(round(observed_fp * len(absent_keys))),
        observed_false_positive_rate=observed_fp,
        theoretical_rate=theoretical,
        target_rate=target,
        watermark_verified=watermark_verified,
        recommendation=recommendation,
    ) if not negatives else BloomReconciliationReport(
        sampled_absent=len(absent_keys),
        observed_false_positives=int(round(observed_fp * len(absent_keys))),
        observed_false_positive_rate=observed_fp,
        theoretical_rate=theoretical,
        target_rate=target,
        watermark_verified=False,
        recommendation="rebuild: membership regression detected (impossible for a "
        "correctly maintained filter)",
    )


class BloomPrecheckRouter:
    """Two-stage membership path in front of one exact lookup (Steps 13, 17).

    The router counts avoided exact lookups (metric: Bloom prechecks reduce
    expensive lookups) and never skips the exact lookup unless the verdict
    is ``DEFINITELY_ABSENT`` under a complete, covering watermark.
    """

    def __init__(
        self,
        filter_: BloomFilter,
        exact_lookup: Mapping[str, object],
    ) -> None:
        """Wrap one exact authoritative mapping with one bloom precheck."""
        self.filter = filter_
        self._exact = exact_lookup
        self.prechecks = 0
        self.avoided_exact_lookups = 0
        self.bypassed_stale_filters = 0

    def lookup(self, key: str, authoritative_watermark: int) -> object | None:
        """Return the exact record, using the filter only as a safe accelerator."""
        self.prechecks += 1
        hint = self.filter.membership_hint(key, authoritative_watermark)
        if hint.verdict is MembershipVerdict.UNKNOWN_INCOMPLETE:
            self.bypassed_stale_filters += 1
            return self._exact.get(key)
        if hint.verdict is MembershipVerdict.DEFINITELY_ABSENT:
            self.avoided_exact_lookups += 1
            return None
        return self._exact.get(key)

    def stats(self) -> dict[str, int]:
        """Return precheck statistics."""
        return {
            "prechecks": self.prechecks,
            "avoided_exact_lookups": self.avoided_exact_lookups,
            "bypassed_stale_filters": self.bypassed_stale_filters,
        }


def bloom_use_case_justified(
    *,
    exact_lookup_rate_per_s: float,
    exact_lookup_cost_ms: float,
    target_fp_rate: float,
) -> bool:
    """Return whether a Bloom precheck is justified for one workload (Step 13).

    Avoid filters where exact indexes are already fast enough or where
    false positives would trigger expensive behavior; the gate requires a
    high-volume path and a bounded target error.
    """
    if exact_lookup_rate_per_s <= 0 or exact_lookup_cost_ms <= 0:
        return False
    if not 0.0 < target_fp_rate < 0.05:
        return False
    return exact_lookup_rate_per_s * exact_lookup_cost_ms > 1.0


__all__ = [
    "MembershipVerdict",
    "BloomSizingError",
    "BloomParameters",
    "theoretical_false_positive_rate",
    "MembershipHint",
    "BloomFilter",
    "CountingBloomFilter",
    "BloomReconciliationReport",
    "estimate_false_positive_rate",
    "reconcile_bloom_filter",
    "BloomPrecheckRouter",
    "bloom_use_case_justified",
]
