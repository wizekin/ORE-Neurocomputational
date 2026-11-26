"""Streaming quantile estimation with a KLL sketch (Phase F, Steps 32-34).

A KLL (Kane, Nelson, Yu) sketch monitors response-time, queue-delay, memory
and resource-consumption distributions without storing every sample
(Step 32). Each validated observation carries timestamp, unit and source,
and approximate summary values stay explicitly distinct from exact
observations (Step 33). Percentile queries return typed estimates that feed
CROS capacity analysis and coordinator observability, never correctness
claims about an individual execution (Step 34).

Implementation notes: the sketch keeps levels of sorted items whose weight
doubles per level; when a level exceeds its capacity it is compacted by
sorting, randomly keeping one item of each consecutive pair and promoting
the survivors to the next level. The compaction randomness is a seeded
``random.Random`` so tests are deterministic.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class QuantileEstimate:
    """Typed quantile estimate: approximate, never a correctness claim (Step 34)."""

    percentile: float
    value: float
    unit: str
    source: str
    nominal_error_bound: float
    algorithm: str = "kll"
    algorithm_version: str = "0.1.0"
    timestamp: str = ""
    source_watermark: int = 0
    observations: int = 0
    available: bool = True

    def __post_init__(self) -> None:
        """Validate and stamp the estimate."""
        if not 0.0 <= self.percentile <= 1.0:
            raise ValueError("percentile must be within [0, 1]")
        if not self.timestamp:
            object.__setattr__(self, "timestamp", _now())

    @property
    def exact(self) -> bool:
        """Quantile estimates are approximate summary values (Step 33)."""
        return False

    def to_dict(self) -> dict[str, object]:
        """Return the estimate with its full metadata (section 5.1 requirement)."""
        return {
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "percentile": self.percentile,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "nominal_error_bound": self.nominal_error_bound,
            "timestamp": self.timestamp,
            "source_watermark": self.source_watermark,
            "observations": self.observations,
            "available": self.available,
            "exact": self.exact,
        }


def _now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class KLLQuantileSketch:
    """KLL quantile sketch over a continuous observation stream (Steps 32-34)."""

    #: Geometric level-capacity ratio from the KLL analysis.
    CAPACITY_RATIO = 2.0 / 3.0
    MIN_LEVEL_CAPACITY = 8
    MAX_LEVELS = 64

    def __init__(
        self,
        *,
        k: int = 128,
        unit: str = "ms",
        source: str = "ore.data.telemetry",
        seed: int = 7,
        retain_samples: int = 0,
    ) -> None:
        """Create one sketch with capacity k, stable units and a seeded compaction RNG."""
        if k < 8:
            raise ValueError("k must be at least 8")
        if not unit.strip():
            raise ValueError("unit must be non-empty")
        if retain_samples < 0:
            raise ValueError("retain_samples must be non-negative")
        self.k = k
        self.unit = unit
        self.source = source
        self._rng = random.Random(seed)
        self._levels: list[list[float]] = [[]]
        self._count = 0
        self._watermark = 0
        self._retained: list[float] = [] if retain_samples else []
        self._retain_limit = retain_samples

    # -- update path -------------------------------------------------------

    def update(
        self,
        observation: float,
        *,
        timestamp: str | None = None,
        source: str | None = None,
        watermark: int | None = None,
    ) -> None:
        """Feed one validated observation into the sketch (Step 33).

        Observations are validated (finite numbers); missing or outlier
        values are the caller's explicit decision: this method raises on
        non-finite input rather than silently distorting the summary. Units
        stay stable per sketch instance.
        """
        value = float(observation)
        if not math.isfinite(value):
            raise ValueError("observations must be finite; handle outliers explicitly (Step 33)")
        self._levels[0].append(value)
        self._count += 1
        self._compact_if_needed(0)
        if self._retain_limit:
            self._retained.append(value)
            if len(self._retained) > self._retain_limit:
                self._retained = self._retained[-self._retain_limit :]
        if watermark is not None:
            self._watermark = max(self._watermark, int(watermark))
        del timestamp, source  # recorded by the caller's telemetry envelope

    def update_all(self, observations: Iterable[float]) -> None:
        """Feed many observations."""
        for value in observations:
            self.update(value)

    # -- compaction ----------------------------------------------------------

    def _level_capacity(self, level: int) -> int:
        return max(self.MIN_LEVEL_CAPACITY, math.ceil(self.k * (self.CAPACITY_RATIO**level)))

    def _compact_if_needed(self, level: int) -> None:
        """Compact one level once it exceeds capacity, cascading upward."""
        while level < len(self._levels) and len(self._levels[level]) > self._level_capacity(level):
            self._compact_level(level)
            level += 1

    def _compact_level(self, level: int) -> None:
        """Sort, keep one item of each consecutive pair, promote survivors (KLL)."""
        items = sorted(self._levels[level])
        survivors = []
        for index in range(0, len(items) - 1, 2):
            survivors.append(items[index + 1] if self._rng.random() < 0.5 else items[index])
        if len(items) % 2 == 1:
            # The unpaired item is retained with certainty.
            survivors.append(items[-1])
        self._levels[level] = []
        if level + 1 == len(self._levels):
            if level + 1 >= self.MAX_LEVELS:  # pragma: no cover - weight guard
                raise RuntimeError("KLL level guard exceeded")
            self._levels.append([])
        self._levels[level + 1].extend(survivors)

    # -- query -----------------------------------------------------------------

    def query(self, percentile: float) -> QuantileEstimate:
        """Estimate one percentile from the weighted sketch state (Step 34).

        An empty sketch degrades gracefully (source 8.4): the estimate is
        returned typed, explicitly marked unavailable, with a NaN value so
        no approximate ranking can mistake it for a real measurement.
        """
        if not 0.0 <= percentile <= 1.0:
            raise ValueError("percentile must be within [0, 1]")
        if self._count == 0:
            return QuantileEstimate(
                percentile=percentile,
                value=float("nan"),
                unit=self.unit,
                source=self.source,
                nominal_error_bound=self.nominal_error_bound(),
                timestamp=_now(),
                source_watermark=self._watermark,
                observations=0,
                available=False,
            )
        weighted: list[tuple[float, int]] = []
        for level, items in enumerate(self._levels):
            weight = 1 << level
            for value in items:
                weighted.append((value, weight))
        weighted.sort(key=lambda pair: pair[0])
        total = sum(weight for _, weight in weighted)
        target = percentile * total
        cumulative = 0
        selected = weighted[0][0]
        for value, weight in weighted:
            cumulative += weight
            if cumulative >= target:
                selected = value
                break
        return QuantileEstimate(
            percentile=percentile,
            value=selected,
            unit=self.unit,
            source=self.source,
            nominal_error_bound=self.nominal_error_bound(),
            timestamp=_now(),
            source_watermark=self._watermark,
            observations=self._count,
        )

    def query_percentiles(self, percentiles: Sequence[float]) -> tuple[QuantileEstimate, ...]:
        """Estimate several operational percentiles (Step 34: p50, p90, p95, p99)."""
        return tuple(self.query(percentile) for percentile in percentiles)

    # -- validation --------------------------------------------------------------

    def exact_percentile(self, percentile: float) -> float:
        """Return the exact percentile over the retained observation window (Step 36 aid).

        The retained window exists only for validation against exact
        samples (source 8.2); the sketch itself never claims exactness.
        """
        if not self._retained:
            raise ValueError("no retained exact samples for validation")
        if not 0.0 <= percentile <= 1.0:
            raise ValueError("percentile must be within [0, 1]")
        ordered = sorted(self._retained)
        index = min(len(ordered) - 1, int(percentile * len(ordered)))
        return ordered[index]

    # -- introspection -------------------------------------------------------------

    def nominal_error_bound(self) -> float:
        """Return the documented nominal rank-error bound of this KLL configuration."""
        return 2.0 / math.sqrt(self.k)

    @property
    def observation_count(self) -> int:
        """Return the number of observations fed into the sketch."""
        return self._count

    @property
    def watermark(self) -> int:
        """Return the source watermark of this sketch's updates."""
        return self._watermark

    def memory_bytes(self) -> int:
        """Return the current sketch allocation size."""
        return sum(len(level) * 8 for level in self._levels) + 8 * len(self._levels)

    def describe(self) -> dict[str, object]:
        """Return the sketch metadata for observability."""
        return {
            "algorithm": "kll",
            "version": "0.1.0",
            "k": self.k,
            "unit": self.unit,
            "source": self.source,
            "observations": self._count,
            "levels": len(self._levels),
            "nominal_error_bound": self.nominal_error_bound(),
            "watermark": self._watermark,
            "available": self._count > 0,
            "memory_bytes": self.memory_bytes(),
        }


__all__ = [
    "QuantileEstimate",
    "KLLQuantileSketch",
]
