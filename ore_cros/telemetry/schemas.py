"""Unit-safe typed telemetry schemas (ORE Point 23, Step 3).

Every measurement carries its resource, metric, canonical unit, timestamp,
scope, source, quality and sampling window. Records that disagree with the
canonical unit table, carry impossible values, or carry malformed timestamps
are rejected at construction time so no downstream policy can silently
combine incompatible measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time as _time
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ..resource_model import ResourceKind, ResourceScope


class TelemetrySchemaError(ValueError):
    """Raised when a measurement violates the typed telemetry schema."""


class MetricKind(str, Enum):
    """Telemetry metric families with fixed canonical units."""

    CAPACITY = "CAPACITY"
    ALLOCATED = "ALLOCATED"
    RESERVED = "RESERVED"
    UTILIZATION = "UTILIZATION"
    PRESSURE = "PRESSURE"
    QUEUE_DEPTH = "QUEUE_DEPTH"
    ARRIVAL_RATE = "ARRIVAL_RATE"
    WAIT_SECONDS = "WAIT_SECONDS"
    LATENCY_SECONDS = "LATENCY_SECONDS"
    ENERGY_JOULES = "ENERGY_JOULES"


#: Canonical units per resource kind for quantity metrics (spec section 2.3).
CAPACITY_UNITS: Mapping[ResourceKind, str] = {
    ResourceKind.CPU: "cores",
    ResourceKind.RAM: "MiB",
    ResourceKind.IO: "ops/s",
    ResourceKind.NET: "Mbit/s",
    ResourceKind.GPU: "MiB",
    ResourceKind.STORAGE: "MiB",
    ResourceKind.QUEUE: "jobs",
}

#: Canonical units per metric kind.
METRIC_UNITS: Mapping[MetricKind, str] = {
    MetricKind.CAPACITY: "capacity-units",
    MetricKind.ALLOCATED: "capacity-units",
    MetricKind.RESERVED: "capacity-units",
    MetricKind.UTILIZATION: "ratio",
    MetricKind.PRESSURE: "ratio",
    MetricKind.QUEUE_DEPTH: "jobs",
    MetricKind.ARRIVAL_RATE: "jobs/s",
    MetricKind.WAIT_SECONDS: "s",
    MetricKind.LATENCY_SECONDS: "s",
    MetricKind.ENERGY_JOULES: "J",
}

#: Metrics that must be whole numbers (impossible fractional values rejected).
INTEGER_METRICS: frozenset[MetricKind] = frozenset({
    MetricKind.QUEUE_DEPTH,
})

#: Metrics restricted to particular resource kinds.
METRIC_KIND_CONSTRAINTS: Mapping[MetricKind, frozenset[ResourceKind]] = {
    MetricKind.QUEUE_DEPTH: frozenset({ResourceKind.QUEUE}),
    MetricKind.ARRIVAL_RATE: frozenset({ResourceKind.QUEUE}),
    MetricKind.WAIT_SECONDS: frozenset({ResourceKind.QUEUE}),
    MetricKind.ENERGY_JOULES: frozenset({ResourceKind.CPU}),
}

#: Plausible upper bound used to reject impossible ratio measurements.
MAX_RATIO = 1000.0

#: Tolerance for timestamps slightly ahead of the clock (clock skew allowance).
CLOCK_SKEW_SECONDS = 300.0

#: Timestamps further than this from the epoch origin are treated as malformed.
MAX_TIMESTAMP = 4102444800.0  # 2100-01-01T00:00:00Z


def canonical_unit(resource: ResourceKind, metric: MetricKind) -> str:
    """Return the canonical unit string for one (resource, metric) pair."""
    if metric in (MetricKind.CAPACITY, MetricKind.ALLOCATED, MetricKind.RESERVED):
        return CAPACITY_UNITS[resource]
    return METRIC_UNITS[metric]


@dataclass(frozen=True)
class Measurement:
    """One unit-safe telemetry record (spec section 2.3)."""

    resource: ResourceKind
    metric: MetricKind
    value: float
    unit: str
    timestamp: float
    scope: ResourceScope
    source: str
    quality: float = 1.0
    window_seconds: float = 1.0

    def __post_init__(self) -> None:
        """Reject invalid units, impossible values and malformed timestamps."""
        expected = canonical_unit(self.resource, self.metric)
        if self.unit != expected:
            raise TelemetrySchemaError(
                f"unit mismatch for {self.resource.value}/{self.metric.value}: "
                f"expected {expected!r}, got {self.unit!r}"
            )
        value = float(self.value)
        if not math.isfinite(value) or value < 0.0:
            raise TelemetrySchemaError(
                f"{self.metric.value} for {self.resource.value} must be finite and "
                f"non-negative, got {value}"
            )
        if self.metric in (
            MetricKind.UTILIZATION, MetricKind.PRESSURE,
        ) and value > MAX_RATIO:
            raise TelemetrySchemaError(
                f"{self.metric.value} of {value} exceeds the plausible bound {MAX_RATIO}"
            )
        if self.metric in INTEGER_METRICS and value != int(value):
            raise TelemetrySchemaError(
                f"{self.metric.value} must be a whole number of jobs, got {value}"
            )
        if self.metric in METRIC_KIND_CONSTRAINTS:
            allowed = METRIC_KIND_CONSTRAINTS[self.metric]
            if self.resource not in allowed:
                raise TelemetrySchemaError(
                    f"{self.metric.value} is only defined for "
                    f"{sorted(item.value for item in allowed)}"
                )
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise TelemetrySchemaError("timestamp must be a finite number")
        if self.timestamp < 0.0 or self.timestamp > MAX_TIMESTAMP:
            raise TelemetrySchemaError(f"timestamp {self.timestamp} is out of range")
        if self.timestamp > _time.time() + CLOCK_SKEW_SECONDS:
            raise TelemetrySchemaError("timestamp is unreasonably in the future")
        if not isinstance(self.source, str) or not self.source.strip():
            raise TelemetrySchemaError("source must be a non-empty string")
        quality = float(self.quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise TelemetrySchemaError("quality must be within [0, 1]")
        window = float(self.window_seconds)
        if not math.isfinite(window) or window <= 0.0:
            raise TelemetrySchemaError("window_seconds must be finite and positive")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "window_seconds", window)

    def with_source_time_override(self, timestamp: float) -> "Measurement":
        """Return a copy with a corrected timestamp for replayed traces."""
        return Measurement(
            resource=self.resource, metric=self.metric, value=self.value,
            unit=self.unit, timestamp=float(timestamp), scope=self.scope,
            source=self.source, quality=self.quality,
            window_seconds=self.window_seconds,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the measurement."""
        return {
            "resource": self.resource.value, "metric": self.metric.value,
            "value": self.value, "unit": self.unit, "timestamp": self.timestamp,
            "scope": {"level": self.scope.level.value, "scope_id": self.scope.scope_id},
            "source": self.source, "quality": self.quality,
            "window_seconds": self.window_seconds,
        }


def require_single_scope(
    measurements: Iterable[Measurement], *, what: str = "aggregate",
) -> ResourceScope:
    """Return the shared scope or raise when scopes would be silently mixed.

    Machine, process, task and module metrics cannot be combined into one
    aggregate without an explicit scope decision (spec Step 3).
    """
    seen: dict[tuple[str, str], ResourceScope] = {}
    for measurement in measurements:
        key = (measurement.scope.level.value, measurement.scope.scope_id)
        seen.setdefault(key, measurement.scope)
    if not seen:
        raise TelemetrySchemaError(f"cannot validate scope of an empty {what}")
    if len(seen) > 1:
        keys = sorted(f"{level}:{identifier}" for level, identifier in seen)
        raise TelemetrySchemaError(
            f"refusing to mix {what} measurements across scopes {keys}"
        )
    return next(iter(seen.values()))


def join_by_scope(
    measurements: Sequence[Measurement],
) -> dict[tuple[str, str], list[Measurement]]:
    """Group measurements by scope identity without merging scopes."""
    grouped: dict[tuple[str, str], list[Measurement]] = {}
    for measurement in measurements:
        key = (measurement.scope.level.value, measurement.scope.scope_id)
        grouped.setdefault(key, []).append(measurement)
    return grouped


def detect_counter_reset(previous: Measurement, current: Measurement) -> bool:
    """Return True when a monotonically increasing counter appears to have reset."""
    if previous.resource != current.resource or previous.metric != current.metric:
        raise TelemetrySchemaError("counter reset checks require matching series")
    if previous.scope != current.scope:
        raise TelemetrySchemaError("counter reset checks require the same scope")
    return current.value < previous.value
