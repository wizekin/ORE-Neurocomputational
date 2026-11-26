"""Telemetry quality assessment (ORE Point 23, Step 8).

Tracks freshness, missingness, source reliability, granularity and counter
resets. Low-confidence observations cannot trigger high-impact actions such
as migration; that gate is exposed as an explicit predicate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from ..config import QualityThresholds
from ..resource_model import ResourceKind
from .schemas import Measurement, MetricKind, detect_counter_reset


#: Prior reliability of known telemetry sources; unknown sources get a neutral prior.
SOURCE_RELIABILITY: Mapping[str, float] = {
    "/proc": 1.0,
    "cgroup.v2": 0.95,
    "os.times": 0.85,
    "simulated": 0.75,
}

#: Default expected (resource, metric) series for completeness accounting.
EXPECTED_SERIES: tuple[tuple[ResourceKind, MetricKind], ...] = (
    (ResourceKind.CPU, MetricKind.UTILIZATION),
    (ResourceKind.RAM, MetricKind.ALLOCATED),
    (ResourceKind.IO, MetricKind.UTILIZATION),
    (ResourceKind.NET, MetricKind.UTILIZATION),
    (ResourceKind.QUEUE, MetricKind.QUEUE_DEPTH),
)


@dataclass(frozen=True)
class SeriesQuality:
    """Quality assessment of one telemetry series."""

    resource: ResourceKind
    metric: MetricKind
    source: str
    age_seconds: float
    confidence: float
    stale: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot."""
        return {
            "resource": self.resource.value, "metric": self.metric.value,
            "source": self.source, "age_seconds": self.age_seconds,
            "confidence": self.confidence, "stale": self.stale,
        }


@dataclass(frozen=True)
class QualityReport:
    """Aggregated quality of one observation batch (spec Step 8)."""

    series: Mapping[str, SeriesQuality]
    missing_sources: tuple[str, ...]
    counter_resets: tuple[str, ...]
    missing_series: tuple[str, ...]
    high_impact_confidence: float

    def confidence(self, resource: ResourceKind) -> float:
        """Return the best available series confidence for one resource kind."""
        scores = [
            item.confidence for item in self.series.values()
            if item.resource == resource and not item.stale
        ]
        return max(scores) if scores else 0.0

    def allows_high_impact(self, resource: ResourceKind) -> bool:
        """Return True only when confidence clears the high-impact gate."""
        return self.confidence(resource) >= self.high_impact_confidence

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the report."""
        return {
            "series": {key: value.to_dict() for key, value in self.series.items()},
            "missing_sources": list(self.missing_sources),
            "counter_resets": list(self.counter_resets),
            "missing_series": list(self.missing_series),
            "high_impact_confidence": self.high_impact_confidence,
        }


class TelemetryQualityAssessor:
    """Compute freshness, missingness and confidence for telemetry batches."""

    def __init__(
        self, thresholds: QualityThresholds | None = None,
        *,
        expected_series: Sequence[tuple[ResourceKind, MetricKind]] = EXPECTED_SERIES,
    ) -> None:
        """Configure quality thresholds and the expected completeness profile."""
        self.thresholds = thresholds or QualityThresholds()
        self.expected_series = tuple(expected_series)

    def assess(
        self,
        measurements: Sequence[Measurement],
        *,
        now: float,
        failed_sources: Sequence[str] = (),
        previous: Sequence[Measurement] | None = None,
    ) -> QualityReport:
        """Assess one batch of measurements collected at time ``now``."""
        if not math.isfinite(now) or now < 0.0:
            raise ValueError("now must be a finite, non-negative timestamp")
        series: dict[str, SeriesQuality] = {}
        resets: list[str] = []
        for measurement in measurements:
            key = f"{measurement.resource.value}/{measurement.metric.value}"
            age = max(0.0, now - measurement.timestamp)
            stale = age > self.thresholds.stale_after_seconds
            reliability = SOURCE_RELIABILITY.get(measurement.source, 0.8)
            confidence = measurement.quality * reliability * (0.0 if stale else 1.0)
            candidate = SeriesQuality(
                resource=measurement.resource, metric=measurement.metric,
                source=measurement.source, age_seconds=age,
                confidence=round(min(1.0, confidence), 6), stale=stale,
            )
            existing = series.get(key)
            if existing is None or candidate.confidence > existing.confidence:
                series[key] = candidate
        if previous:
            resets = self._detect_resets(measurements, previous)
        expected_keys = {
            f"{resource.value}/{metric.value}" for resource, metric in self.expected_series
        }
        missing_series = tuple(sorted(expected_keys - set(series)))
        return QualityReport(
            series=series,
            missing_sources=tuple(sorted(set(str(item) for item in failed_sources))),
            counter_resets=tuple(sorted(set(resets))),
            missing_series=missing_series,
            high_impact_confidence=self.thresholds.high_impact_confidence,
        )

    @staticmethod
    def _detect_resets(
        current: Sequence[Measurement], previous: Sequence[Measurement],
    ) -> list[str]:
        """Flag counter resets between consecutive batches of the same series."""
        previous_by_key: dict[str, Measurement] = {}
        for measurement in previous:
            key = (
                f"{measurement.scope.level.value}:{measurement.scope.scope_id}/"
                f"{measurement.resource.value}/{measurement.metric.value}"
            )
            previous_by_key[key] = measurement
        resets: list[str] = []
        for measurement in current:
            key = (
                f"{measurement.scope.level.value}:{measurement.scope.scope_id}/"
                f"{measurement.resource.value}/{measurement.metric.value}"
            )
            prior = previous_by_key.get(key)
            if prior is not None and detect_counter_reset(prior, measurement):
                resets.append(key)
        return resets


__all__ = [
    "TelemetryQualityAssessor", "QualityReport", "SeriesQuality",
    "SOURCE_RELIABILITY", "EXPECTED_SERIES",
]
