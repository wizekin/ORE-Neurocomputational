"""MAPE-K Monitor stage (ORE Point 23, Step 9).

Collects telemetry, task events, dependency changes, deadlines, queue
state and health into one coherent, versioned observation snapshot. The
snapshot is the only input the Analyze stage accepts.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from ..accounting.task_costs import TaskUsageAccountant
from ..config import CROSConfig
from ..resource_model import ResourceState, TaskRecord, TaskState
from ..telemetry.collectors import CollectorSet
from ..telemetry.history import TimeWindowedResourceHistory
from ..telemetry.quality import QualityReport, TelemetryQualityAssessor
from ..telemetry.schemas import Measurement


@dataclass(frozen=True)
class ObservationSnapshot:
    """Coherent, versioned observation snapshot (spec Step 9)."""

    version: int
    timestamp: float
    measurements: tuple[Measurement, ...]
    quality: QualityReport
    tasks: tuple[TaskRecord, ...]
    collector_failures: tuple[str, ...]

    @property
    def ready_tasks(self) -> tuple[TaskRecord, ...]:
        """Return tasks currently ready or running."""
        return tuple(
            task for task in self.tasks
            if task.state in (TaskState.READY, TaskState.RUNNING)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the observation."""
        return {
            "version": self.version, "timestamp": self.timestamp,
            "measurements": [item.to_dict() for item in self.measurements],
            "quality": self.quality.to_dict(),
            "tasks": [task.to_dict() for task in self.tasks],
            "collector_failures": list(self.collector_failures),
        }


class Monitor:
    """Aggregate collectors, tasks and history into snapshots (Step 9)."""

    def __init__(
        self,
        collector_set: CollectorSet,
        assessor: TelemetryQualityAssessor,
        history: TimeWindowedResourceHistory,
        *,
        config: CROSConfig | None = None,
    ) -> None:
        """Wire the monitor to its collectors, assessor and shared history."""
        self.collector_set = collector_set
        self.assessor = assessor
        self.history = history
        self.config = config or CROSConfig()
        self._version = 0
        self._previous_measurements: list[Measurement] = []

    def collect(
        self,
        tasks: Sequence[TaskRecord],
        *,
        now: float | None = None,
    ) -> ObservationSnapshot:
        """Publish one coherent, versioned observation snapshot."""
        import time as _time

        moment = _time.time() if now is None else float(now)
        if not math.isfinite(moment) or moment < 0.0:
            raise ValueError("now must be a finite, non-negative timestamp")
        measurements, statuses = self.collector_set.collect(now=moment)
        failed = tuple(
            status.name for status in statuses if not status.ok
        )
        quality = self.assessor.assess(
            measurements, now=moment, failed_sources=failed,
            previous=self._previous_measurements or None,
        )
        self.history.record(measurements)
        self.history.prune(now=moment)
        self._previous_measurements = list(measurements)
        self._version += 1
        return ObservationSnapshot(
            version=self._version, timestamp=moment,
            measurements=tuple(measurements), quality=quality,
            tasks=tuple(tasks), collector_failures=failed,
        )

    @property
    def version(self) -> int:
        """Return the current snapshot version counter."""
        return self._version


class TaskEventCollector:
    """Collect ready and running ORE tasks for the cycle (spec section 5)."""

    @staticmethod
    def collect_ready_and_running(
        tasks: Sequence[TaskRecord], *, completed: Sequence[str] = (),
    ) -> tuple[TaskRecord, ...]:
        """Return tasks that are ready or running, dependency-filtered."""
        completed_set = set(completed)
        ready: list[TaskRecord] = []
        for task in tasks:
            if task.state not in (TaskState.READY, TaskState.RUNNING, TaskState.PENDING):
                continue
            if task.state is TaskState.PENDING:
                continue
            if not all(item in completed_set for item in task.dependencies):
                continue
            ready.append(task)
        return tuple(sorted(ready, key=lambda item: item.task_id))

    @staticmethod
    def blocked_tasks(
        tasks: Sequence[TaskRecord], *, completed: Sequence[str] = (),
    ) -> tuple[TaskRecord, ...]:
        """Return tasks blocked by unfinished dependencies (evidence for Step 10)."""
        completed_set = set(completed)
        blocked = [
            task for task in tasks
            if task.state in (TaskState.READY, TaskState.PENDING, TaskState.RUNNING)
            and any(item not in completed_set for item in task.dependencies)
        ]
        return tuple(sorted(blocked, key=lambda item: item.task_id))


__all__ = ["Monitor", "ObservationSnapshot", "TaskEventCollector"]
