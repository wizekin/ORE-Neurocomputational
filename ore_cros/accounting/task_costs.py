"""Task and module resource attribution (ORE Point 23, Step 5).

Associates measured usage with tasks, workers, modules and parent goals.
When operating-system attribution is unavailable, usage is distributed as a
bounded, demand-proportional estimate and explicitly labelled as estimated
(spec Step 5: "use bounded estimates and label them as estimates").
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

from ..resource_model import (
    DEMAND_KINDS, ResourceDemand, ResourceKind, ResourceScope, TaskRecord,
)
from ..telemetry.schemas import Measurement, MetricKind


#: Metrics that measure rates (integrated over time) rather than levels.
RATE_METRICS: frozenset[tuple[ResourceKind, MetricKind]] = frozenset({
    (ResourceKind.CPU, MetricKind.ALLOCATED),
    (ResourceKind.IO, MetricKind.UTILIZATION),
    (ResourceKind.NET, MetricKind.UTILIZATION),
    (ResourceKind.IO, MetricKind.ALLOCATED),
    (ResourceKind.NET, MetricKind.ALLOCATED),
})


@dataclass(frozen=True)
class UsageSample:
    """One attributed usage observation for a task or module."""

    task_id: str
    module_id: str
    resource: ResourceKind
    value: float
    timestamp: float
    estimated: bool = False

    def __post_init__(self) -> None:
        """Reject malformed samples."""
        for name, value in (("task_id", self.task_id), ("module_id", self.module_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not math.isfinite(self.value) or self.value < 0.0:
            raise ValueError("value must be finite and non-negative")
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")


@dataclass(frozen=True)
class TaskCostRecord:
    """One row of the task-to-resource accounting table (spec Step 5)."""

    task_id: str
    module_id: str
    parent_goal_id: str | None
    workload_class: str
    usage: ResourceDemand
    started_at: float
    ended_at: float
    estimated: bool
    sample_count: int
    unattributed_kinds: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the row."""
        return {
            "task_id": self.task_id, "module_id": self.module_id,
            "parent_goal_id": self.parent_goal_id,
            "workload_class": self.workload_class,
            "usage": self.usage.to_dict(), "started_at": self.started_at,
            "ended_at": self.ended_at, "estimated": self.estimated,
            "sample_count": self.sample_count,
            "unattributed_kinds": list(self.unattributed_kinds),
        }


class TaskUsageAccountant:
    """Attribute resource usage to tasks and modules (spec Step 5).

    Math: rate metrics are integrated (mean value times interval) over each
    task's execution interval; level metrics contribute their peak observed
    value. Machine-scope measurements without direct attribution are split
    across concurrently running tasks proportionally to declared demand and
    the resulting rows are marked ``estimated=True``.
    """

    def __init__(self) -> None:
        """Initialize empty attribution registries."""
        self._intervals: dict[str, tuple[str, str, str | None, float, float]] = {}
        self._samples: dict[str, list[UsageSample]] = {}
        self._closed: dict[str, list[UsageSample]] = {}

    def open_task(self, task: TaskRecord, *, started_at: float) -> None:
        """Register the accounting interval for one running task."""
        if task.task_id in self._intervals:
            raise ValueError(f"task {task.task_id!r} already has an open interval")
        if not math.isfinite(started_at):
            raise ValueError("started_at must be finite")
        self._intervals[task.task_id] = (
            task.module_id, task.workload_class.value, task.parent_goal_id,
            started_at, math.inf,
        )
        self._samples[task.task_id] = []

    def close_task(self, task_id: str, *, ended_at: float) -> None:
        """Close the accounting interval for one finished task."""
        interval = self._intervals.get(task_id)
        if interval is None:
            raise ValueError(f"task {task_id!r} has no open interval")
        if not math.isfinite(ended_at):
            raise ValueError("ended_at must be finite")
        module_id, workload_class, parent_goal_id, started_at, _ = interval
        if ended_at < started_at:
            raise ValueError("ended_at cannot precede started_at")
        self._intervals[task_id] = (
            module_id, workload_class, parent_goal_id, started_at, ended_at,
        )
        self._closed[task_id] = self._samples.pop(task_id, [])

    def record_sample(
        self, task_id: str, module_id: str, resource: ResourceKind,
        value: float, *, timestamp: float, estimated: bool = False,
    ) -> None:
        """Record one directly attributed usage sample."""
        if task_id not in self._intervals:
            raise ValueError(f"task {task_id!r} has no open interval")
        self._samples[task_id].append(UsageSample(
            task_id=task_id, module_id=module_id, resource=resource,
            value=float(value), timestamp=float(timestamp), estimated=estimated,
        ))

    def attribute_machine_measurements(
        self, measurements: Sequence[Measurement], running_tasks: Sequence[TaskRecord],
        *, now: float,
    ) -> int:
        """Distribute machine-scope samples across running tasks by demand share.

        Returns the number of estimated samples created. Every created sample
        is labelled ``estimated=True``; direct samples are never overwritten.
        """
        if not running_tasks:
            return 0
        totals: dict[ResourceKind, float] = {}
        for measurement in measurements:
            if measurement.metric not in (MetricKind.ALLOCATED, MetricKind.UTILIZATION):
                continue
            if measurement.resource == ResourceKind.QUEUE:
                continue
            totals[measurement.resource] = max(
                totals.get(measurement.resource, 0.0), measurement.value,
            )
        demand_totals = {
            kind: sum(task.demand.get(kind) for task in running_tasks)
            for kind in DEMAND_KINDS
        }
        created = 0
        for task in running_tasks:
            if task.task_id not in self._intervals:
                continue
            for kind in DEMAND_KINDS:
                total_usage = totals.get(kind)
                if total_usage is None or total_usage <= 0.0:
                    continue
                share_denominator = demand_totals.get(kind, 0.0)
                if share_denominator <= 0.0:
                    continue
                share = task.demand.get(kind) / share_denominator
                if share <= 0.0:
                    continue
                self.record_sample(
                    task.task_id, task.module_id, kind, total_usage * share,
                    timestamp=now, estimated=True,
                )
                created += 1
        return created

    def accounting_table(self) -> tuple[TaskCostRecord, ...]:
        """Build the task-to-resource accounting table from recorded samples."""
        rows: list[TaskCostRecord] = []
        merged: dict[str, list[UsageSample]] = {**self._closed}
        for task_id, samples in self._samples.items():
            merged.setdefault(task_id, []).extend(samples)
        for task_id in sorted(merged):
            interval = self._intervals.get(task_id)
            samples = merged[task_id]
            if interval is None:
                continue
            module_id, workload_class, parent_goal_id, started_at, ended_at = interval
            duration = (ended_at - started_at) if math.isfinite(ended_at) else 0.0
            duration = max(0.0, duration)
            by_kind: dict[ResourceKind, list[UsageSample]] = {}
            for sample in samples:
                by_kind.setdefault(sample.resource, []).append(sample)
            usage_components: dict[str, float] = {}
            estimated = False
            unattributed: list[str] = []
            for kind in DEMAND_KINDS:
                kind_samples = by_kind.get(kind, [])
                field_name = {
                    ResourceKind.CPU: "cpu", ResourceKind.RAM: "ram",
                    ResourceKind.IO: "io", ResourceKind.NET: "net",
                    ResourceKind.GPU: "gpu", ResourceKind.STORAGE: "storage",
                }[kind]
                if not kind_samples:
                    usage_components[field_name] = 0.0
                    unattributed.append(kind.value)
                    continue
                if any(sample.estimated for sample in kind_samples):
                    estimated = True
                if (kind, MetricKind.ALLOCATED) in RATE_METRICS or (
                    kind, MetricKind.UTILIZATION
                ) in RATE_METRICS:
                    mean_rate = statistics.fmean(sample.value for sample in kind_samples)
                    usage_components[field_name] = mean_rate * duration
                else:
                    usage_components[field_name] = max(
                        sample.value for sample in kind_samples
                    )
            rows.append(TaskCostRecord(
                task_id=task_id, module_id=module_id, parent_goal_id=parent_goal_id,
                workload_class=workload_class,
                usage=ResourceDemand(**usage_components),
                started_at=started_at,
                ended_at=ended_at if math.isfinite(ended_at) else float("nan"),
                estimated=estimated, sample_count=len(samples),
                unattributed_kinds=tuple(unattributed),
            ))
        return tuple(rows)

    def module_totals(self) -> dict[str, ResourceDemand]:
        """Aggregate attributed usage per module across all tasks."""
        totals: dict[str, dict[str, float]] = {}
        for record in self.accounting_table():
            bucket = totals.setdefault(record.module_id, {
                "cpu": 0.0, "ram": 0.0, "io": 0.0, "net": 0.0,
                "gpu": 0.0, "storage": 0.0,
            })
            for key, value in record.usage.to_dict().items():
                if math.isfinite(value):
                    bucket[key] = bucket.get(key, 0.0) + value
        return {module: ResourceDemand(**components) for module, components in totals.items()}


def task_demand_to_scope(task_id: str, parent: ResourceScope) -> ResourceScope:
    """Return the TASK scope identifier for one task below a parent scope."""
    from ..resource_model import ScopeLevel

    return ResourceScope(ScopeLevel.TASK, task_id, parent)


__all__ = [
    "TaskUsageAccountant", "TaskCostRecord", "UsageSample", "RATE_METRICS",
    "task_demand_to_scope",
]
