"""Count-Min Sketch frequency estimation with namespace separation (Phase D).

Steps 20-25: Count-Min sketches track approximate frequencies for semantic
concept retrieval, module activation, task patterns, error categories and
repeated queries. Each metric namespace is hashed separately so unrelated
keys never contaminate one another (Step 20). Dimensions follow
``w = ceil(e/epsilon)`` and ``d = ceil(ln(1/delta))`` (Step 21, source
4.2); the minimum over rows estimates frequency as an upper-biased
approximation that is never presented as an exact count (Steps 23-24);
epoch rotation separates lifetime from recent frequency (Step 25).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping


class CountMinSizingError(ValueError):
    """Raised for unusable Count-Min parameters."""


@dataclass(frozen=True, slots=True)
class CountMinParameters:
    """Sketch dimensions derived from additive error and failure probability (Step 21)."""

    epsilon: float
    delta: float
    width: int
    depth: int

    def __post_init__(self) -> None:
        """Validate derived parameters."""
        if self.epsilon <= 0:
            raise CountMinSizingError("epsilon must be positive")
        if not 0.0 < self.delta < 1.0:
            raise CountMinSizingError("delta must be within (0, 1)")
        if self.width <= 0 or self.depth <= 0:
            raise CountMinSizingError("width and depth must be positive")

    @classmethod
    def from_error_bounds(cls, epsilon: float, delta: float) -> "CountMinParameters":
        """Derive w = ceil(e/epsilon), d = ceil(ln(1/delta)) (source 4.2)."""
        if epsilon <= 0:
            raise CountMinSizingError("epsilon must be positive")
        if not 0.0 < delta < 1.0:
            raise CountMinSizingError("delta must be within (0, 1)")
        width = int(math.ceil(math.e / epsilon))
        depth = int(math.ceil(math.log(1.0 / delta)))
        return cls(epsilon=epsilon, delta=delta, width=width, depth=depth)

    def additive_error_bound(self, total_weight: float) -> float:
        """Return the additive overestimate bound epsilon * N (Step 21)."""
        return self.epsilon * total_weight

    def to_dict(self) -> dict[str, float | int]:
        """Return the parameters as a plain dictionary."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "width": self.width,
            "depth": self.depth,
        }


@dataclass(frozen=True, slots=True)
class FrequencyEstimate:
    """Typed frequency estimate: an upper-biased approximation, never exact (Step 23)."""

    namespace: str
    key: str
    estimate: float
    additive_error_bound: float
    total_stream_weight: float
    confidence_description: str
    epoch: int
    source_watermark: int
    timestamp: str
    algorithm: str = "count_min_sketch"
    algorithm_version: str = "0.1.0"

    @property
    def exact(self) -> bool:
        """Frequency estimates are never exact occurrence counts."""
        return False

    @property
    def is_upper_estimate(self) -> bool:
        """The estimate upper-bounds the true frequency under standard assumptions."""
        return True

    def to_dict(self) -> dict[str, object]:
        """Return the estimate with its full metadata (section 5.1 requirement)."""
        return {
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "namespace": self.namespace,
            "key": self.key,
            "estimate": self.estimate,
            "additive_error_bound": self.additive_error_bound,
            "total_stream_weight": self.total_stream_weight,
            "confidence_description": self.confidence_description,
            "epoch": self.epoch,
            "source_watermark": self.source_watermark,
            "timestamp": self.timestamp,
            "exact": self.exact,
            "upper_estimate": self.is_upper_estimate,
        }


def _now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _row_columns(
    key: str,
    namespace: str,
    width: int,
    depth: int,
) -> tuple[int, ...]:
    """Hash one namespaced key into one column per row (Steps 20, 22)."""
    digest = hashlib.sha256(f"cms|{namespace}|{key}".encode("utf-8")).digest()
    h1 = int.from_bytes(digest[:8], "big")
    h2 = int.from_bytes(digest[8:16], "big") | 1
    return tuple((h1 + row * h2) % width for row in range(depth))


class CountMinSketch:
    """Namespaced Count-Min sketch with epoch rotation (Steps 20-25)."""

    def __init__(
        self,
        parameters: CountMinParameters,
        *,
        namespace: str = "ore.data.frequency",
        epochs_retained: int = 8,
    ) -> None:
        """Create one empty sketch for one metric namespace."""
        if not namespace.strip():
            raise ValueError("namespace must be non-empty")
        if epochs_retained <= 0:
            raise ValueError("epochs_retained must be positive")
        self.parameters = parameters
        self.namespace = namespace
        self._counters = [
            [0] * parameters.width for _ in range(parameters.depth)
        ]
        self._total_weight = 0.0
        self._epoch = 0
        self._history: list[CountMinSketch] = []
        self._epochs_retained = epochs_retained
        self._watermark = 0

    # -- update path -------------------------------------------------------

    def update(self, key: str, weight: float = 1.0, watermark: int | None = None) -> None:
        """Update one key's counters with a validated non-negative weight (Step 22)."""
        if not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
            raise ValueError("weight must be a finite number")
        if weight < 0:
            raise ValueError("weights must be non-negative; negative weights are rejected (Step 22)")
        for row, column in enumerate(
            _row_columns(key, self.namespace, self.parameters.width, self.parameters.depth)
        ):
            self._counters[row][column] += weight
        self._total_weight += weight
        if watermark is not None:
            self._watermark = max(self._watermark, int(watermark))

    # -- estimation ---------------------------------------------------------

    def estimate(self, key: str) -> FrequencyEstimate:
        """Estimate frequency as the minimum over rows (Step 23)."""
        columns = _row_columns(
            key, self.namespace, self.parameters.width, self.parameters.depth
        )
        value = min(self._counters[row][column] for row, column in enumerate(columns))
        return FrequencyEstimate(
            namespace=self.namespace,
            key=key,
            estimate=value,
            additive_error_bound=self.parameters.additive_error_bound(self._total_weight),
            total_stream_weight=self._total_weight,
            confidence_description=(
                "estimate exceeds the true frequency by at most epsilon*N with "
                "probability at least 1-delta under the algorithm's assumptions"
            ),
            epoch=self._epoch,
            source_watermark=self._watermark,
            timestamp=_now(),
        )

    def estimates_prioritize_only(self, key: str) -> str:
        """Document that estimates drive prioritization, not truth (Step 24)."""
        estimate = self.estimate(key)
        return (
            f"prioritization_hint:{key}:{estimate.estimate:.0f}; exact counts "
            "require the authoritative event store"
        )

    # -- aging --------------------------------------------------------------

    def rotate(self) -> int:
        """Rotate the current sketch into an epoch bucket and start a fresh one (Step 25)."""
        archived = CountMinSketch(
            self.parameters,
            namespace=self.namespace,
            epochs_retained=self._epochs_retained,
        )
        archived._counters = [row[:] for row in self._counters]
        archived._total_weight = self._total_weight
        archived._epoch = self._epoch
        archived._watermark = self._watermark
        self._history.append(archived)
        if len(self._history) > self._epochs_retained:
            self._history = self._history[-self._epochs_retained :]
        self._counters = [[0] * self.parameters.width for _ in range(self.parameters.depth)]
        self._total_weight = 0.0
        self._epoch += 1
        self._watermark = 0
        return self._epoch

    def recent_estimate(self, key: str, epochs: int = 1) -> FrequencyEstimate:
        """Estimate recent frequency over the last ``epochs`` epochs (Step 25).

        ``epochs=1`` covers the current epoch; larger values include the
        archived epoch buckets, keeping recent frequency distinct from
        lifetime frequency.
        """
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        included = self._history[-(epochs - 1):] if epochs > 1 else []
        columns = _row_columns(
            key, self.namespace, self.parameters.width, self.parameters.depth
        )
        total = min(
            self._counters[row][column] for row, column in enumerate(columns)
        )
        weight = self._total_weight
        for sketch in included:
            sketch_columns = _row_columns(
                key, sketch.namespace, sketch.parameters.width, sketch.parameters.depth
            )
            total += min(
                sketch._counters[row][column]
                for row, column in enumerate(sketch_columns)
            )
            weight += sketch._total_weight
        return FrequencyEstimate(
            namespace=self.namespace,
            key=key,
            estimate=total,
            additive_error_bound=self.parameters.additive_error_bound(weight),
            total_stream_weight=weight,
            confidence_description=(
                "summed minimum-row estimate over epoch buckets; recent "
                "frequency stays distinct from lifetime frequency (Step 25)"
            ),
            epoch=self._epoch,
            source_watermark=self._watermark,
            timestamp=_now(),
        )

    def lifetime_estimate(self, key: str) -> FrequencyEstimate:
        """Estimate lifetime frequency across every retained epoch (Step 25)."""
        epochs = len(self._history) + 1
        return self.recent_estimate(key, epochs)

    # -- introspection -----------------------------------------------------

    @property
    def total_weight(self) -> float:
        """Return the current epoch's total stream weight N (Step 22)."""
        return self._total_weight

    @property
    def epoch(self) -> int:
        """Return the current epoch number."""
        return self._epoch

    @property
    def watermark(self) -> int:
        """Return the source watermark of the current epoch's updates."""
        return self._watermark

    def memory_bytes(self) -> int:
        """Return the counter array allocation size (bytes per sketch metric)."""
        return self.parameters.width * self.parameters.depth * 8

    def merge(self, other: "CountMinSketch") -> None:
        """Merge a compatible sketch of the same namespace and parameters."""
        if other.namespace != self.namespace:
            raise ValueError("cannot merge sketches of different namespaces (Step 20)")
        if other.parameters != self.parameters:
            raise ValueError("cannot merge sketches with different parameters")
        for row in range(self.parameters.depth):
            for column in range(self.parameters.width):
                self._counters[row][column] += other._counters[row][column]
        self._total_weight += other._total_weight

    def describe(self) -> dict[str, object]:
        """Return the sketch metadata for observability."""
        return {
            "algorithm": "count_min_sketch",
            "version": "0.1.0",
            "namespace": self.namespace,
            "parameters": self.parameters.to_dict(),
            "epoch": self._epoch,
            "total_weight": self._total_weight,
            "watermark": self._watermark,
            "memory_bytes": self.memory_bytes(),
        }


__all__ = [
    "CountMinSizingError",
    "CountMinParameters",
    "FrequencyEstimate",
    "CountMinSketch",
]
