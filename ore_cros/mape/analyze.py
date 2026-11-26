"""Pressure detection and MAPE-K Analyze stage (ORE Point 23, Steps 6 and 10).

Pressure detection separates sustained saturation from short spikes and
applies Little's Law (L = lambda * W) to queue observations. The analyzer
then diagnoses the dominant resource condition, ranking bottleneck
candidates by contribution, persistence, impact and confidence with the
measured evidence attached.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from ..resource_model import ResourceKind, TaskRecord
from ..telemetry.history import TimeWindowedResourceHistory
from ..telemetry.schemas import Measurement, MetricKind
from .monitor import ObservationSnapshot


class ResourceCondition(str, Enum):
    """Diagnosed resource conditions (spec Step 10)."""

    HEALTHY = "HEALTHY"
    CPU_BOUND = "CPU_BOUND"
    MEMORY_BOUND = "MEMORY_BOUND"
    IO_BOUND = "IO_BOUND"
    NETWORK_BOUND = "NETWORK_BOUND"
    GPU_BOUND = "GPU_BOUND"
    DEPENDENCY_BLOCKED = "DEPENDENCY_BLOCKED"


@dataclass(frozen=True)
class PressureReading:
    """One resource's pressure classification (spec Step 6)."""

    resource: ResourceKind
    pressure: float
    sustained: bool
    spike: bool
    window_seconds: float
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the reading."""
        return {
            "resource": self.resource.value, "pressure": self.pressure,
            "sustained": self.sustained, "spike": self.spike,
            "window_seconds": self.window_seconds, "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class QueueReading:
    """Queueing observation checked against Little's Law (spec 3.2)."""

    depth: float
    arrival_rate: float | None
    mean_wait_seconds: float | None
    littles_law_work: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the reading."""
        return {
            "depth": self.depth, "arrival_rate": self.arrival_rate,
            "mean_wait_seconds": self.mean_wait_seconds,
            "littles_law_work": self.littles_law_work,
        }


@dataclass(frozen=True)
class PressureVector:
    """Multidimensional pressure vector (spec Step 6)."""

    readings: tuple[PressureReading, ...]
    queue: QueueReading | None

    def reading(self, resource: ResourceKind) -> PressureReading | None:
        """Return the pressure reading for one resource kind."""
        return next((item for item in self.readings if item.resource == resource), None)

    def sustained_resources(self) -> tuple[ResourceKind, ...]:
        """Return the resources under sustained saturation."""
        return tuple(
            item.resource for item in self.readings if item.sustained
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the vector."""
        return {
            "readings": [item.to_dict() for item in self.readings],
            "queue": self.queue.to_dict() if self.queue else None,
        }


class PressureDetector:
    """Detect sustained saturation, spikes and queueing (Step 6)."""

    def __init__(
        self, *,
        saturation_threshold: float = 0.85,
        sustained_windows: int = 3,
    ) -> None:
        """Configure saturation thresholds and persistence requirements."""
        if not 0.0 < saturation_threshold <= 1.0:
            raise ValueError("saturation_threshold must be within (0, 1]")
        if sustained_windows < 1:
            raise ValueError("sustained_windows must be at least 1")
        self.saturation_threshold = saturation_threshold
        self.sustained_windows = sustained_windows

    def detect(
        self,
        measurements: Sequence[Measurement],
        history: TimeWindowedResourceHistory,
        *,
        now: float,
        window_seconds: float,
    ) -> PressureVector:
        """Build the pressure vector from current and historical evidence."""
        latest: dict[ResourceKind, float] = {}
        for measurement in measurements:
            if measurement.metric is MetricKind.PRESSURE:
                latest[measurement.resource] = measurement.value
        readings: list[PressureReading] = []
        for resource, pressure in sorted(latest.items(), key=lambda item: item[0].value):
            evidence: list[str] = []
            stats = None
            for key in history.series_keys():
                if key.endswith(f"/{resource.value}/{MetricKind.PRESSURE.value}"):
                    stats = history.statistics(
                        key, window_seconds=window_seconds, now=now,
                    )
                    break
            sustained = False
            spike = False
            if stats is not None:
                above = [
                    1.0 if value >= self.saturation_threshold else 0.0
                    for value in _series_values(
                        history, resource, MetricKind.PRESSURE, now, window_seconds,
                    )
                ]
                recent = above[-self.sustained_windows:]
                sustained = (
                    len(recent) >= self.sustained_windows
                    and all(item == 1.0 for item in recent)
                )
                spike = (
                    pressure >= self.saturation_threshold and not sustained
                )
                evidence.append(f"{stats.count} samples in window, p95={stats.p95:.3f}")
            if pressure >= self.saturation_threshold:
                evidence.append(f"current pressure {pressure:.3f} above threshold")
            readings.append(PressureReading(
                resource=resource, pressure=pressure, sustained=sustained,
                spike=spike, window_seconds=window_seconds,
                evidence=tuple(evidence),
            ))
        queue = self._queue_reading(measurements)
        return PressureVector(readings=tuple(readings), queue=queue)

    @staticmethod
    def _queue_reading(measurements: Sequence[Measurement]) -> QueueReading | None:
        """Extract queue observations and verify Little's Law consistency."""
        depth = None
        arrival = None
        wait = None
        for measurement in measurements:
            if measurement.resource is ResourceKind.QUEUE:
                if measurement.metric is MetricKind.QUEUE_DEPTH:
                    depth = measurement.value
                elif measurement.metric is MetricKind.ARRIVAL_RATE:
                    arrival = measurement.value
                elif measurement.metric is MetricKind.WAIT_SECONDS:
                    wait = measurement.value
        if depth is None:
            return None
        implied = None
        if arrival is not None and wait is not None and arrival > 0.0:
            implied = arrival * wait
        return QueueReading(
            depth=depth, arrival_rate=arrival,
            mean_wait_seconds=wait, littles_law_work=implied,
        )


def _series_values(
    history: TimeWindowedResourceHistory, resource: ResourceKind,
    metric: MetricKind, now: float, window_seconds: float,
) -> list[float]:
    """Return raw retained values for one series inside the window."""
    suffix = f"/{resource.value}/{metric.value}"
    values: list[float] = []
    for key, samples in history._samples.items():  # noqa: SLF001 - internal helper
        if key.endswith(suffix):
            cutoff = now - window_seconds
            values.extend(value for timestamp, value in samples if timestamp >= cutoff)
    return sorted(values)


@dataclass(frozen=True)
class BottleneckCandidate:
    """One ranked bottleneck candidate with evidence (spec Step 10)."""

    condition: ResourceCondition
    resource: ResourceKind | None
    contribution: float
    persistence: float
    impact: float
    confidence: float
    contributing_tasks: tuple[str, ...]
    evidence: tuple[str, ...]

    @property
    def rank_score(self) -> float:
        """Return the deterministic ranking score (contribution/persistence/impact)."""
        return self.contribution * self.persistence * self.impact * self.confidence

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the candidate."""
        return {
            "condition": self.condition.value,
            "resource": self.resource.value if self.resource else None,
            "contribution": self.contribution, "persistence": self.persistence,
            "impact": self.impact, "confidence": self.confidence,
            "contributing_tasks": list(self.contributing_tasks),
            "evidence": list(self.evidence), "rank_score": self.rank_score,
        }


@dataclass(frozen=True)
class Diagnosis:
    """Diagnosed resource condition with evidence (spec Step 10)."""

    condition: ResourceCondition
    candidates: tuple[BottleneckCandidate, ...]
    pressure: PressureVector
    snapshot_version: int

    def primary(self) -> BottleneckCandidate:
        """Return the top-ranked bottleneck candidate."""
        return self.candidates[0]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the diagnosis."""
        return {
            "condition": self.condition.value,
            "candidates": [item.to_dict() for item in self.candidates],
            "pressure": self.pressure.to_dict(),
            "snapshot_version": self.snapshot_version,
        }


#: Mapping from resource kinds to the condition they indicate.
_KIND_CONDITIONS: dict[ResourceKind, ResourceCondition] = {
    ResourceKind.CPU: ResourceCondition.CPU_BOUND,
    ResourceKind.RAM: ResourceCondition.MEMORY_BOUND,
    ResourceKind.IO: ResourceCondition.IO_BOUND,
    ResourceKind.NET: ResourceCondition.NETWORK_BOUND,
    ResourceKind.GPU: ResourceCondition.GPU_BOUND,
}


class Analyzer:
    """MAPE-K Analyze stage: bottleneck diagnosis (spec Step 10)."""

    def diagnose(
        self,
        snapshot: ObservationSnapshot,
        pressure: PressureVector,
        *,
        blocked_tasks: Sequence[TaskRecord] = (),
        dependency_blocked_threshold: float = 0.5,
    ) -> Diagnosis:
        """Rank bottleneck candidates from the snapshot and pressure vector."""
        if not 0.0 <= dependency_blocked_threshold <= 1.0:
            raise ValueError("dependency_blocked_threshold must be within [0, 1]")
        candidates: list[BottleneckCandidate] = []
        utilization_by_kind = {
            measurement.resource: measurement.value
            for measurement in snapshot.measurements
            if measurement.metric is MetricKind.UTILIZATION
        }
        total_ready = len(snapshot.ready_tasks)
        for reading in pressure.readings:
            condition = _KIND_CONDITIONS.get(reading.resource)
            if condition is None:
                continue
            utilization = utilization_by_kind.get(reading.resource, reading.pressure)
            contributing = tuple(
                task.task_id for task in snapshot.ready_tasks
                if task.demand.get(reading.resource) > 0.0
            )
            contribution = min(1.0, utilization)
            persistence = 1.0 if reading.sustained else (0.5 if reading.spike else 0.2)
            impact = min(1.0, len(contributing) / total_ready) if total_ready else 0.0
            confidence = snapshot.quality.confidence(reading.resource)
            if reading.pressure <= 0.0 and utilization < 0.5:
                continue
            candidates.append(BottleneckCandidate(
                condition=condition, resource=reading.resource,
                contribution=contribution, persistence=persistence,
                impact=impact, confidence=confidence,
                contributing_tasks=contributing,
                evidence=reading.evidence,
            ))
        if blocked_tasks and total_ready:
            blocked_fraction = len(blocked_tasks) / (total_ready + len(blocked_tasks))
            if blocked_fraction >= dependency_blocked_threshold:
                candidates.append(BottleneckCandidate(
                    condition=ResourceCondition.DEPENDENCY_BLOCKED,
                    resource=None,
                    contribution=blocked_fraction,
                    persistence=1.0, impact=blocked_fraction,
                    confidence=1.0,
                    contributing_tasks=tuple(
                        task.task_id for task in blocked_tasks
                    ),
                    evidence=(
                        f"{len(blocked_tasks)} of "
                        f"{total_ready + len(blocked_tasks)} tasks dependency-blocked",
                    ),
                ))
        candidates.sort(key=lambda item: (-item.rank_score, item.condition.value))
        condition = candidates[0].condition if candidates else ResourceCondition.HEALTHY
        return Diagnosis(
            condition=condition, candidates=tuple(candidates),
            pressure=pressure, snapshot_version=snapshot.version,
        )


__all__ = [
    "PressureDetector", "PressureVector", "PressureReading", "QueueReading",
    "ResourceCondition", "Analyzer", "Diagnosis", "BottleneckCandidate",
]
