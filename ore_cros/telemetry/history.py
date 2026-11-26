"""Time-windowed resource history with bounded retention (ORE Point 23, Step 7).

Stores short windows for interactive response, medium windows for active
workloads and long windows for trends. Retention is bounded per series and
records older than the long window are compacted into aggregate summaries
instead of accumulating forever.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

from ..config import WindowConfig
from ..resource_model import ResourceKind
from .schemas import Measurement, MetricKind


def quantile(values: Sequence[float], q: float) -> float:
    """Return the linearly interpolated q-quantile of a value sequence."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be within [0, 1]")
    if not values:
        raise ValueError("quantile of an empty sequence is undefined")
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def ewma(values: Sequence[float], alpha: float = 0.5) -> float:
    """Return the exponentially weighted moving average of a sequence."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be within (0, 1]")
    if not values:
        raise ValueError("EWMA of an empty sequence is undefined")
    accumulator = float(values[0])
    for value in values[1:]:
        accumulator = alpha * float(value) + (1.0 - alpha) * accumulator
    return accumulator


@dataclass(frozen=True)
class SeriesStatistics:
    """Summary statistics over one history window."""

    count: int
    mean: float
    variance: float
    p50: float
    p95: float
    p99: float
    ewma: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a detached, JSON-compatible snapshot of the statistics."""
        return {
            "count": self.count, "mean": self.mean, "variance": self.variance,
            "p50": self.p50, "p95": self.p95, "p99": self.p99,
            "ewma": self.ewma, "minimum": self.minimum, "maximum": self.maximum,
        }


@dataclass(frozen=True)
class CompactedAggregate:
    """Compact record retained for expired history samples."""

    first_timestamp: float
    last_timestamp: float
    count: int
    mean: float
    minimum: float
    maximum: float


@dataclass
class TimeWindowedResourceHistory:
    """Bounded multi-window history for resource telemetry series."""

    windows: WindowConfig = field(default_factory=WindowConfig)
    _samples: dict[str, deque[tuple[float, float]]] = field(default_factory=dict)
    _compacted: dict[str, CompactedAggregate] = field(default_factory=dict)

    @staticmethod
    def series_key(measurement: Measurement) -> str:
        """Return the stable series key for one measurement."""
        return (
            f"{measurement.scope.level.value}:{measurement.scope.scope_id}/"
            f"{measurement.resource.value}/{measurement.metric.value}"
        )

    def record(self, measurements: Iterable[Measurement]) -> None:
        """Append measurements to their bounded per-series buffers."""
        for measurement in measurements:
            key = self.series_key(measurement)
            buffer = self._samples.setdefault(key, deque())
            buffer.append((measurement.timestamp, measurement.value))
            while len(buffer) > self.windows.max_samples_per_series:
                buffer.popleft()

    def prune(self, *, now: float) -> int:
        """Drop samples older than the long window, compacting them first.

        Returns the number of samples removed; total memory stays bounded by
        ``series_count * max_samples_per_series``.
        """
        removed = 0
        cutoff = now - self.windows.long_window_seconds
        for key, buffer in list(self._samples.items()):
            expired: list[tuple[float, float]] = []
            while buffer and buffer[0][0] < cutoff:
                expired.append(buffer.popleft())
                removed += 1
            if expired:
                self._compact(key, expired)
        return removed

    def _compact(self, key: str, expired: Sequence[tuple[float, float]]) -> None:
        """Fold expired samples into the compacted aggregate for one series."""
        values = [value for _, value in expired]
        existing = self._compacted.get(key)
        if existing is None:
            self._compacted[key] = CompactedAggregate(
                first_timestamp=expired[0][0], last_timestamp=expired[-1][0],
                count=len(values), mean=statistics.fmean(values),
                minimum=min(values), maximum=max(values),
            )
            return
        combined_count = existing.count + len(values)
        merged_mean = (
            existing.mean * existing.count + statistics.fmean(values) * len(values)
        ) / combined_count
        self._compacted[key] = CompactedAggregate(
            first_timestamp=existing.first_timestamp,
            last_timestamp=expired[-1][0], count=combined_count, mean=merged_mean,
            minimum=min(existing.minimum, min(values)),
            maximum=max(existing.maximum, max(values)),
        )

    def statistics(
        self, key: str, *, window_seconds: float | None = None, now: float | None = None,
    ) -> SeriesStatistics | None:
        """Return summary statistics for one series within an observation window."""
        buffer = self._samples.get(key)
        if not buffer:
            return None
        window = window_seconds if window_seconds is not None else self.windows.long_window_seconds
        cutoff = None if now is None else now - window
        values = [
            value for timestamp, value in buffer
            if cutoff is None or timestamp >= cutoff
        ]
        if not values:
            return None
        variance = statistics.pvariance(values) if len(values) > 1 else 0.0
        return SeriesStatistics(
            count=len(values), mean=statistics.fmean(values), variance=variance,
            p50=quantile(values, 0.50), p95=quantile(values, 0.95),
            p99=quantile(values, 0.99), ewma=ewma(values),
            minimum=min(values), maximum=max(values),
        )

    def series_keys(self) -> tuple[str, ...]:
        """Return all currently known series keys."""
        return tuple(sorted(self._samples))

    def compacted(self) -> Mapping[str, CompactedAggregate]:
        """Return the read-only compacted aggregates for expired samples."""
        return dict(self._compacted)

    def total_samples(self) -> int:
        """Return the total number of retained samples across all series."""
        return sum(len(buffer) for buffer in self._samples.values())

    def memory_bound(self) -> int:
        """Return the hard upper bound on retained samples (acceptance test)."""
        return max(1, len(self._samples)) * self.windows.max_samples_per_series

    def kind_statistics(
        self, resource: ResourceKind, metric: MetricKind, *,
        window_seconds: float | None = None, now: float | None = None,
    ) -> dict[str, SeriesStatistics]:
        """Return statistics for every series of one (resource, metric) pair."""
        suffix = f"/{resource.value}/{metric.value}"
        result: dict[str, SeriesStatistics] = {}
        for key in self.series_keys():
            if key.endswith(suffix):
                stats = self.statistics(key, window_seconds=window_seconds, now=now)
                if stats is not None:
                    result[key] = stats
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of retained history."""
        return {
            "series": {
                key: [
                    {"timestamp": timestamp, "value": value}
                    for timestamp, value in buffer
                ]
                for key, buffer in sorted(self._samples.items())
            },
            "compacted": {
                key: {
                    "first_timestamp": item.first_timestamp,
                    "last_timestamp": item.last_timestamp, "count": item.count,
                    "mean": item.mean, "minimum": item.minimum, "maximum": item.maximum,
                }
                for key, item in sorted(self._compacted.items())
            },
        }


__all__ = [
    "TimeWindowedResourceHistory", "SeriesStatistics", "CompactedAggregate",
    "quantile", "ewma",
]
