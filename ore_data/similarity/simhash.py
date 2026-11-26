"""SimHash fingerprinting and Hamming-distance candidate detection (Phase G, Step 41).

For a weighted feature vector, signed feature contributions accumulate into
bit dimensions and each output bit follows the sign of its accumulator;
fingerprints compare by Hamming distance (source 4.6). A random-hyperplane
bit differs with probability theta/pi for angular separation theta, so
Hamming distance estimates angular distance and is not a universal
semantic-equivalence test. Candidate pairs found via bit-chunk bucketing
still require exact or semantic verification (Step 42).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping, Tuple


#: Default fingerprint width in bits.
DEFAULT_BITS = 64


@dataclass(frozen=True, slots=True)
class SimHashFingerprint:
    """One SimHash fingerprint with its feature context (Step 41)."""

    record_id: str
    bits: int
    value: int
    feature_count: int
    total_weight: float
    algorithm: str = "simhash"
    algorithm_version: str = "0.1.0"

    def __post_init__(self) -> None:
        """Validate the fingerprint."""
        if not self.record_id.strip():
            raise ValueError("record_id must be non-empty")
        if self.bits <= 0 or self.bits % 8:
            raise ValueError("bits must be a positive multiple of 8")
        if not 0 <= self.value < (1 << self.bits):
            raise ValueError("value must fit the fingerprint width")
        if self.feature_count <= 0:
            raise ValueError("feature_count must be positive")

    def to_dict(self) -> dict[str, object]:
        """Return the fingerprint metadata as a plain dictionary."""
        return {
            "record_id": self.record_id,
            "bits": self.bits,
            "value": self.value,
            "feature_count": self.feature_count,
            "total_weight": self.total_weight,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
        }


def _feature_hash(feature: str, bits: int) -> int:
    """Stable n-bit hash of one feature identifier."""
    digest = hashlib.sha256(f"simhash|{feature}".encode("utf-8")).digest()
    return int.from_bytes(digest[: bits // 8], "big")


def simhash(
    record_id: str,
    features: Mapping[str, float],
    *,
    bits: int = DEFAULT_BITS,
) -> SimHashFingerprint:
    """Generate one SimHash fingerprint from weighted features (Step 41).

    Each feature contributes its weight positively or negatively to every
    bit dimension according to its hashed bit pattern; the fingerprint bit
    is the sign of the accumulated contribution.
    """
    if bits <= 0 or bits % 8:
        raise ValueError("bits must be a positive multiple of 8")
    if not features:
        raise ValueError("features must be non-empty")
    for feature, weight in features.items():
        if not isinstance(weight, (int, float)):
            raise TypeError("feature weights must be numbers")
    accumulators = [0.0] * bits
    total_weight = 0.0
    for feature, weight in features.items():
        weight = float(weight)
        if weight == 0.0:
            continue
        total_weight += abs(weight)
        hashed = _feature_hash(str(feature), bits)
        for dimension in range(bits):
            if (hashed >> dimension) & 1:
                accumulators[dimension] += weight
            else:
                accumulators[dimension] -= weight
    value = 0
    for dimension in range(bits):
        if accumulators[dimension] >= 0:
            value |= 1 << dimension
    return SimHashFingerprint(
        record_id=record_id,
        bits=bits,
        value=value,
        feature_count=len(features),
        total_weight=total_weight,
    )


def hamming_distance(first: SimHashFingerprint, second: SimHashFingerprint) -> int:
    """Return the Hamming distance between two fingerprints (Step 41)."""
    if first.bits != second.bits:
        raise ValueError("cannot compare fingerprints of different widths")
    return (first.value ^ second.value).bit_count()


def estimated_angular_distance(first: SimHashFingerprint, second: SimHashFingerprint) -> float:
    """Return the angular estimate: theta/pi ~ Hamming_distance/bits (source 4.6)."""
    return hamming_distance(first, second) / first.bits


class SimHashIndex:
    """Candidate index over SimHash fingerprints using bit-chunk bucketing (Step 41).

    Fingerprints split into C chunks; by the pigeonhole principle any pair whose
    Hamming distance is at most C-1 must share at least one identical chunk,
    so chunk buckets nominate near-duplicate candidates cheaply. Candidates
    still require exact verification (Step 42).
    """

    def __init__(self, *, bits: int = DEFAULT_BITS, chunk_bits: int = 8) -> None:
        """Create one empty chunk-bucket index."""
        if bits <= 0 or bits % chunk_bits or bits % 8:
            raise ValueError("bits must be a multiple of chunk_bits")
        self.bits = bits
        self.chunk_bits = chunk_bits
        self._buckets: dict[Tuple[int, int], set[str]] = {}
        self._records: dict[str, SimHashFingerprint] = {}

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks; pairs within distance C-1 always match."""
        return self.bits // self.chunk_bits

    def add(self, fingerprint: SimHashFingerprint) -> None:
        """Index one fingerprint into every chunk bucket."""
        if fingerprint.bits != self.bits:
            raise ValueError("fingerprint width does not match the index")
        chunks = self.bits // self.chunk_bits
        for chunk in range(chunks):
            shift = chunk * self.chunk_bits
            mask = (1 << self.chunk_bits) - 1
            key = (chunk, (fingerprint.value >> shift) & mask)
            self._buckets.setdefault(key, set()).add(fingerprint.record_id)
        self._records[fingerprint.record_id] = fingerprint

    def add_all(self, fingerprints: Iterable[SimHashFingerprint]) -> None:
        """Index many fingerprints."""
        for fingerprint in fingerprints:
            self.add(fingerprint)

    def candidates(self, fingerprint: SimHashFingerprint, *, max_distance: int | None = None) -> set[str]:
        """Return records sharing at least one chunk bucket (Step 41)."""
        if fingerprint.bits != self.bits:
            raise ValueError("fingerprint width does not match the index")
        found: set[str] = set()
        chunks = self.bits // self.chunk_bits
        for chunk in range(chunks):
            shift = chunk * self.chunk_bits
            mask = (1 << self.chunk_bits) - 1
            key = (chunk, (fingerprint.value >> shift) & mask)
            found.update(self._buckets.get(key, set()))
        found.discard(fingerprint.record_id)
        if max_distance is not None:
            found = {
                record_id
                for record_id in found
                if hamming_distance(fingerprint, self._records[record_id]) <= max_distance
            }
        return found

    def records(self) -> int:
        """Return the number of indexed records."""
        return len(self._records)


def validate_distance_threshold(
    distance: int,
    bits: int = DEFAULT_BITS,
) -> bool:
    """Validate a Hamming-distance threshold on representative data (Step 41)."""
    if distance < 0 or distance > bits:
        raise ValueError("distance must be within [0, bits]")
    # Angular interpretation: theta/pi = distance/bits; thresholds above
    # half the bits compare almost everything and are not selective.
    return distance <= bits // 2


@dataclass(frozen=True, slots=True)
class SimHashCandidate:
    """One SimHash candidate pair with its Hamming distance (Step 41)."""

    pair: Tuple[str, str]
    hamming_distance: int
    bits: int
    angular_estimate: float
    algorithm: str = "simhash"
    algorithm_version: str = "0.1.0"
    timestamp: str = ""

    def __post_init__(self) -> None:
        """Stamp the candidate."""
        if not self.timestamp:
            object.__setattr__(self, "timestamp", _now())

    def to_dict(self) -> dict[str, object]:
        """Return the candidate as a plain dictionary."""
        return {
            "pair": list(self.pair),
            "hamming_distance": self.hamming_distance,
            "bits": self.bits,
            "angular_estimate": self.angular_estimate,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "timestamp": self.timestamp,
        }


def _now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


__all__ = [
    "DEFAULT_BITS",
    "SimHashFingerprint",
    "simhash",
    "hamming_distance",
    "estimated_angular_distance",
    "SimHashIndex",
    "validate_distance_threshold",
    "SimHashCandidate",
]
