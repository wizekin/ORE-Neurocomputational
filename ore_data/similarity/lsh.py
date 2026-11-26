"""LSH candidate bucketing with banding thresholds (Phase G, Step 40).

Signatures are partitioned into b bands of r rows; each band hashes to a
bucket and any shared bucket nominates a candidate pair. The probability a
pair with similarity s becomes a candidate is ``1 - (1 - s^r)^b`` (source
4.5), so bands and rows tune the trade-off between candidate volume,
recall and verification cost. LSH intentionally trades recall and precision
for lower search cost: every candidate still requires exact or semantic
verification (Step 42).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Set, Tuple

from ore_data.similarity.minhash import MinHashSignature


@dataclass(frozen=True, slots=True)
class LshBandConfig:
    """Banding configuration tuned to one similarity threshold (Step 40)."""

    bands: int
    rows_per_band: int

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.bands <= 0 or self.rows_per_band <= 0:
            raise ValueError("bands and rows_per_band must be positive")

    @property
    def signature_length(self) -> int:
        """Return the signature length this configuration requires."""
        return self.bands * self.rows_per_band

    def threshold(self) -> float:
        """Return the approximate similarity threshold of this configuration."""
        return (1.0 / self.bands) ** (1.0 / self.rows_per_band)


def candidate_probability(similarity: float, config: LshBandConfig) -> float:
    """Return P(candidate) = 1 - (1 - s^r)^b (source 4.5)."""
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("similarity must be within [0, 1]")
    return 1.0 - (1.0 - similarity**config.rows_per_band) ** config.bands


def choose_bands(
    signature_length: int,
    target_similarity: float,
) -> LshBandConfig:
    """Choose a banding configuration for one target threshold (Step 40).

    Among the divisor pairs (b, r) of the signature length, this picks the
    configuration whose crossover threshold is closest to the target.
    """
    if signature_length < 2:
        raise ValueError("signature_length must be at least 2")
    if not 0.0 < target_similarity < 1.0:
        raise ValueError("target_similarity must be within (0, 1)")
    best: LshBandConfig | None = None
    best_distance = None
    for bands in range(1, signature_length + 1):
        if signature_length % bands:
            continue
        rows = signature_length // bands
        config = LshBandConfig(bands=bands, rows_per_band=rows)
        distance = abs(config.threshold() - target_similarity)
        if best is None or distance < best_distance:
            best, best_distance = config, distance
    if best is None:  # pragma: no cover - signature_length >= 2 always yields bands=1
        raise ValueError("no divisor configuration found")
    return best


def _bucket_key(band: int, values: Sequence[int]) -> tuple[int, str]:
    """Hash one band slice to a bucket key."""
    payload = "|".join(str(value) for value in values)
    digest = hashlib.sha256(f"lsh|{band}|{payload}".encode("utf-8")).hexdigest()[:16]
    return band, digest


class LshBandIndex:
    """Candidate bucket index over MinHash signatures (Step 40)."""

    def __init__(self, config: LshBandConfig) -> None:
        """Create one empty bucket index for one banding configuration."""
        self.config = config
        self._buckets: dict[tuple[int, str], set[str]] = {}
        self._records: dict[str, MinHashSignature] = {}
        self._candidate_sets_generated = 0

    def add(self, signature: MinHashSignature) -> None:
        """Index one signature into every band bucket (Step 40)."""
        if len(signature.values) != self.config.signature_length:
            raise ValueError(
                f"signature length {len(signature.values)} does not match the "
                f"banding configuration {self.config.signature_length}"
            )
        for band_index in range(self.config.bands):
            key = _bucket_key(band_index, signature.band(band_index, self.config.rows_per_band))
            self._buckets.setdefault(key, set()).add(signature.record_id)
        self._records[signature.record_id] = signature

    def add_all(self, signatures: Iterable[MinHashSignature]) -> None:
        """Index many signatures."""
        for signature in signatures:
            self.add(signature)

    def candidates(self, signature: MinHashSignature) -> Set[str]:
        """Return the record IDs sharing at least one band bucket (Step 40)."""
        self._candidate_sets_generated += 1
        found: set[str] = set()
        for band_index in range(self.config.bands):
            key = _bucket_key(
                band_index, signature.band(band_index, self.config.rows_per_band)
            )
            found.update(self._buckets.get(key, set()))
        found.discard(signature.record_id)
        return found

    def candidate_pairs(self) -> Set[Tuple[str, str]]:
        """Return every indexed candidate pair (for recall measurement)."""
        pairs: set[Tuple[str, str]] = set()
        for members in self._buckets.values():
            ordered = sorted(members)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    pairs.add((ordered[i], ordered[j]))
        return pairs

    def bucket_count(self) -> int:
        """Return the number of occupied buckets (candidate volume metric)."""
        return len(self._buckets)

    def records(self) -> int:
        """Return the number of indexed records."""
        return len(self._records)


@dataclass(frozen=True, slots=True)
class LshRecallReport:
    """Measured recall, precision and candidate volume of one LSH index (Step 40)."""

    exact_pairs: int
    candidate_pairs_found: int
    recall: float
    precision: float
    candidate_volume: int
    config: LshBandConfig

    def to_dict(self) -> dict[str, object]:
        """Return the report as a plain dictionary."""
        return {
            "exact_pairs": self.exact_pairs,
            "candidate_pairs_found": self.candidate_pairs_found,
            "recall": self.recall,
            "precision": self.precision,
            "candidate_volume": self.candidate_volume,
            "bands": self.config.bands,
            "rows_per_band": self.config.rows_per_band,
            "threshold": self.config.threshold(),
        }


def measure_recall(
    index: LshBandIndex,
    exact_similar_pairs: Set[Tuple[str, str]],
    *,
    similarity_threshold: float = 0.8,
) -> LshRecallReport:
    """Measure LSH candidate recall and precision against exact pairs (source 8.2)."""
    found = index.candidate_pairs()
    relevant = {
        pair for pair in exact_similar_pairs
    }
    matched = len(found & relevant)
    recall = matched / len(relevant) if relevant else 1.0
    precision = matched / len(found) if found else 1.0
    return LshRecallReport(
        exact_pairs=len(relevant),
        candidate_pairs_found=len(found),
        recall=recall,
        precision=precision,
        candidate_volume=index.bucket_count(),
        config=index.config,
    )


def threshold_config_for(target_similarity: float) -> LshBandConfig:
    """Return a canonical banding configuration for one threshold (Step 40)."""
    if not 0.0 < target_similarity < 1.0:
        raise ValueError("target_similarity must be within (0, 1)")
    signature_length = 128
    return choose_bands(signature_length, target_similarity)


def expected_candidate_volume(n_records: int, config: LshBandConfig) -> float:
    """Return the expected candidate volume for uniformly random signatures.

    For unrelated records (s ~ 0), P(candidate) ~ 1 - (1 - 0)^b = 0; the
    formula below bounds the collision-driven volume with an assumed
    background similarity s0 (uniform random sign collisions give a tiny
    but non-zero band-match probability).
    """
    if n_records <= 1:
        return 0.0
    pairs = n_records * (n_records - 1) / 2
    background = 1.0 / (1 << config.rows_per_band * 8)
    probability = 1.0 - (1.0 - background) ** config.bands
    return pairs * probability


__all__ = [
    "LshBandConfig",
    "candidate_probability",
    "choose_bands",
    "LshBandIndex",
    "LshRecallReport",
    "measure_recall",
    "threshold_config_for",
    "expected_candidate_volume",
]
