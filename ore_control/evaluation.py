"""Benchmark harness for the control layer (Point 02 §7 reliability tests).

Compares control-layer-driven runs against the existing sequential
module-coordination baseline (a direct blackboard run per task) on a
representative task set, reporting goal completion, operator applications,
latency, retrieval cost, and conflict preservation with a paired bootstrap
confidence interval. The harness measures the coordination harness only; no
performance improvement is claimed from single demonstrations.
"""
from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .contracts import TriggerEnvelope, TriggerKind


@dataclass(frozen=True, slots=True)
class TaskRunMetrics:
    """Metrics recorded for one benchmark task run."""

    task_label: str
    path: str  # control | baseline
    goal_outcome: str
    goal_score: float
    status: str
    operator_applications: int
    module_calls: int
    latency_ms: float
    retrieval_cost: float
    conflicts_preserved: int
    success: bool


@dataclass
class BenchmarkReport:
    """Aggregate paired benchmark report (no improvement claims)."""

    metrics: tuple[TaskRunMetrics, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = (
        "Metrics measure coordination-harness behaviour on reference toy "
        "operators; no real-world task-quality claim is made.",
    )

    def by_path(self, path: str) -> tuple[TaskRunMetrics, ...]:
        """Return the metrics recorded for one execution path."""
        return tuple(item for item in self.metrics if item.path == path)

    def summary(self) -> dict[str, Mapping[str, Any]]:
        """Summarize both paths without claiming improvement."""
        summary: dict[str, Mapping[str, Any]] = {}
        for path in ("control", "baseline"):
            items = self.by_path(path)
            if not items:
                continue
            successes = sum(1 for item in items if item.success)
            summary[path] = {
                "tasks": len(items),
                "success_rate": successes / len(items),
                "mean_latency_ms": statistics.fmean(
                    item.latency_ms for item in items
                ),
                "median_latency_ms": statistics.median(
                    item.latency_ms for item in items
                ),
                "mean_operator_applications": statistics.fmean(
                    item.operator_applications for item in items
                ),
                "mean_module_calls": statistics.fmean(
                    item.module_calls for item in items
                ),
                "mean_retrieval_cost": statistics.fmean(
                    item.retrieval_cost for item in items
                ),
                "conflicts_preserved": sum(
                    item.conflicts_preserved for item in items
                ),
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable benchmark report."""
        return {
            "metrics": [
                {
                    "task_label": item.task_label,
                    "path": item.path,
                    "goal_outcome": item.goal_outcome,
                    "goal_score": item.goal_score,
                    "status": item.status,
                    "operator_applications": item.operator_applications,
                    "module_calls": item.module_calls,
                    "latency_ms": item.latency_ms,
                    "retrieval_cost": item.retrieval_cost,
                    "conflicts_preserved": item.conflicts_preserved,
                    "success": item.success,
                }
                for item in self.metrics
            ],
            "summary": self.summary(),
            "notes": list(self.notes),
        }


def paired_bootstrap_latency(
    report: BenchmarkReport,
    *,
    iterations: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap the paired latency difference control minus baseline.

    Returns (mean difference, low percentile, high percentile) in
    milliseconds. This is a harness statistic; it establishes no claim of
    improvement without repeated measurement.
    """
    control = {item.task_label: item for item in report.by_path("control")}
    baseline = {item.task_label: item for item in report.by_path("baseline")}
    shared = sorted(set(control) & set(baseline))
    if not shared:
        return 0.0, 0.0, 0.0
    differences = [
        control[label].latency_ms - baseline[label].latency_ms
        for label in shared
    ]
    generator = random.Random(seed)
    means: list[float] = []
    for _ in range(max(1, iterations)):
        sample = [
            differences[generator.randrange(len(differences))]
            for _ in differences
        ]
        means.append(statistics.fmean(sample))
    means.sort()
    low = means[int(0.025 * len(means))]
    high = means[int(0.975 * len(means) - 1)]
    return statistics.fmean(differences), low, high


def run_benchmark(
    coordinator: Any,
    tasks: Sequence[str],
    *,
    baseline: Any,
    task_labels: Sequence[str] | None = None,
) -> BenchmarkReport:
    """Run the paired control-versus-baseline benchmark (§7)."""
    from ore_blackboard.contracts import BlackboardTaskRequest

    labels = list(task_labels) if task_labels else [
        f"task_{index + 1}" for index in range(len(tasks))
    ]
    metrics: list[TaskRunMetrics] = []
    for label, text in zip(labels, tasks):
        started = time.perf_counter()
        result = coordinator.process_trigger(TriggerEnvelope(
            kind=TriggerKind.USER_TASK,
            idempotency_key=f"benchmark_control_{label}",
            event_reference=f"benchmark:{label}:control",
            payload={"text": text},
        ))
        control_latency = (time.perf_counter() - started) * 1000
        metrics.append(TaskRunMetrics(
            task_label=label,
            path="control",
            goal_outcome=result.goal_outcome,
            goal_score=result.goal_score,
            status=result.status,
            operator_applications=result.operator_applications,
            module_calls=0,
            latency_ms=control_latency,
            retrieval_cost=result.retrieval_cost,
            conflicts_preserved=result.conflicts_preserved,
            success=result.status == "SUCCEEDED",
        ))
        started = time.perf_counter()
        baseline_result = baseline.run(BlackboardTaskRequest(
            input=text,
            goal="understand_sentence",
            task_id=f"benchmark_baseline_{label}",
        ))
        baseline_latency = (time.perf_counter() - started) * 1000
        metrics.append(TaskRunMetrics(
            task_label=label,
            path="baseline",
            goal_outcome=baseline_result.status,
            goal_score=baseline_result.goal_score,
            status=baseline_result.status,
            operator_applications=0,
            module_calls=baseline_result.module_calls,
            latency_ms=baseline_latency,
            retrieval_cost=0.0,
            conflicts_preserved=baseline_result.conflict_count,
            success=baseline_result.status in {
                "completed", "completed_with_uncertainty",
            },
        ))
    return BenchmarkReport(metrics=tuple(metrics))


__all__ = [
    "BenchmarkReport",
    "TaskRunMetrics",
    "paired_bootstrap_latency",
    "run_benchmark",
]
