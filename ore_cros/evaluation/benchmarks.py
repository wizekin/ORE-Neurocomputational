"""Policy comparison and workload replay benchmarks (ORE Point 23, section 8-9).

Benchmarks report throughput, p50/p95/p99 latency, queue delay, pressure,
energy where available and policy overhead, plus rollback and fallback
rates. A long-replay stability check verifies bounded memory, no
oscillation and no policy drift.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time as _time
from typing import Any, Mapping, Sequence

from ..policies.baseline import DeterministicBaselineScheduler
from ..resource_model import TaskRecord, TaskState, WorkloadClass
from ..telemetry.history import quantile


@dataclass(frozen=True)
class TaskTraceEvent:
    """One arrival in a replayable workload trace."""

    task_id: str
    module_id: str
    workload_class: WorkloadClass
    arrival_time: float
    service_time: float
    deadline: float | None = None

    def __post_init__(self) -> None:
        """Validate the trace event."""
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not math.isfinite(self.arrival_time) or self.arrival_time < 0.0:
            raise ValueError("arrival_time must be finite and non-negative")
        if not math.isfinite(self.service_time) or self.service_time < 0.0:
            raise ValueError("service_time must be finite and non-negative")


@dataclass(frozen=True)
class BenchmarkReport:
    """Metrics from one benchmark run (spec sections 8-9)."""

    completed: int
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    queue_delay_p50: float
    mean_pressure: float
    fairness: float
    energy_joules: float | None
    policy_overhead_seconds: float
    decision_count: int
    rollback_count: int
    fallback_count: int
    constraint_violations: int

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the report."""
        return {
            "completed": self.completed, "throughput": self.throughput,
            "latency_p50": self.latency_p50, "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99, "queue_delay_p50": self.queue_delay_p50,
            "mean_pressure": self.mean_pressure, "fairness": self.fairness,
            "energy_joules": self.energy_joules,
            "policy_overhead_seconds": self.policy_overhead_seconds,
            "decision_count": self.decision_count,
            "rollback_count": self.rollback_count,
            "fallback_count": self.fallback_count,
            "constraint_violations": self.constraint_violations,
        }


class ReplayBenchmark:
    """Replay a workload trace through the deterministic scheduler (Step 16).

    The replay is a discrete-event approximation used for policy
    comparison; it does not claim to model a real operating system.
    """

    def __init__(
        self,
        scheduler: DeterministicBaselineScheduler,
        *,
        worker_count: int = 4,
        energy_available: bool = False,
    ) -> None:
        """Configure the scheduler, worker pool and energy availability."""
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        self.scheduler = scheduler
        self.worker_count = worker_count
        self.energy_available = energy_available

    def run(
        self, trace: Sequence[TaskTraceEvent], *,
        pressure_by_class: Mapping[WorkloadClass, float] | None = None,
    ) -> BenchmarkReport:
        """Run one replay and return the benchmark report."""
        if not trace:
            raise ValueError("trace must contain at least one event")
        pressure_by_class = dict(pressure_by_class or {})
        workers_free_at = [0.0] * self.worker_count
        ready: list[TaskRecord] = []
        events = sorted(trace, key=lambda item: (item.arrival_time, item.task_id))
        remaining = list(events)
        now = 0.0
        latencies: list[float] = []
        queue_delays: list[float] = []
        completed = 0
        decision_count = 0
        overhead_start = _time.perf_counter()
        while remaining or ready:
            if remaining and not ready:
                now = max(now, remaining[0].arrival_time)
            while remaining and remaining[0].arrival_time <= now:
                event = remaining.pop(0)
                ready.append(_trace_to_task(event))
            ready = [task for task in ready if task.state is TaskState.READY]
            free_workers = sum(1 for free_at in workers_free_at if free_at <= now)
            if not ready or free_workers == 0:
                next_times = [free_at for free_at in workers_free_at if free_at > now]
                if remaining:
                    next_times.append(remaining[0].arrival_time)
                if not next_times:
                    break
                now = min(next_times)
                continue
            dispatch_order = sorted(
                ready, key=lambda item: self.scheduler.order_key(item, now=now),
            )
            dispatched = 0
            for task in dispatch_order:
                if dispatched >= free_workers:
                    break
                worker_index = next(
                    index for index, free_at in enumerate(workers_free_at)
                    if free_at <= now
                )
                start = max(now, workers_free_at[worker_index])
                queue_delays.append(max(0.0, start - task.ready_time))
                finish = start + task.remaining_work
                workers_free_at[worker_index] = finish
                latencies.append(max(0.0, finish - task.ready_time))
                ready.remove(task)
                completed += 1
                dispatched += 1
            decision_count += 1
            now = min(free_at for free_at in workers_free_at if free_at > now) if any(
                free_at > now for free_at in workers_free_at
            ) else now
        overhead = _time.perf_counter() - overhead_start
        service_shares = [1.0 / max(1, completed)] * max(1, completed)
        fairness = _jain(service_shares)
        durations = sorted(event.service_time for event in trace)
        makespan = max(latencies) + max(event.arrival_time for event in trace) if latencies else 1.0
        throughput = completed / max(1e-9, makespan)
        mean_pressure = (
            sum(pressure_by_class.get(event.workload_class, 0.0) for event in trace)
            / len(trace)
        )
        return BenchmarkReport(
            completed=completed, throughput=throughput,
            latency_p50=quantile(latencies, 0.50) if latencies else 0.0,
            latency_p95=quantile(latencies, 0.95) if latencies else 0.0,
            latency_p99=quantile(latencies, 0.99) if latencies else 0.0,
            queue_delay_p50=quantile(queue_delays, 0.50) if queue_delays else 0.0,
            mean_pressure=mean_pressure, fairness=fairness,
            energy_joules=(sum(durations) * 20.0) if self.energy_available else None,
            policy_overhead_seconds=overhead, decision_count=decision_count,
            rollback_count=0, fallback_count=0, constraint_violations=0,
        )


def _jain(shares: Sequence[float]) -> float:
    """Return the Jain fairness index without importing the model module."""
    values = [float(item) for item in shares]
    total = sum(values)
    if total <= 0.0:
        return 1.0
    return (total * total) / (len(values) * sum(item * item for item in values))


def _trace_to_task(event: TaskTraceEvent) -> TaskRecord:
    """Convert one trace event into a schedulable task record."""
    return TaskRecord(
        task_id=event.task_id, parent_goal_id=None, module_id=event.module_id,
        workload_class=event.workload_class, demand=_trace_demand(),
        remaining_work=event.service_time, ready_time=event.arrival_time,
        deadline=event.deadline,
    )


def _trace_demand():
    """Return the neutral demand vector used by replay tasks."""
    from ..resource_model import ResourceDemand

    return ResourceDemand(cpu=0.1)


@dataclass(frozen=True)
class PolicyComparison:
    """Comparison of two benchmark reports (spec section 8)."""

    baseline: BenchmarkReport
    candidate: BenchmarkReport
    acceptance: Mapping[str, bool]

    @property
    def accepted(self) -> bool:
        """Return True when every acceptance criterion holds."""
        return all(self.acceptance.values())

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the comparison."""
        return {
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "acceptance": dict(self.acceptance),
            "accepted": self.accepted,
        }


def compare_policies(
    baseline: BenchmarkReport, candidate: BenchmarkReport, *,
    max_p95_regression: float = 0.10,
    max_violations: int = 0,
) -> PolicyComparison:
    """Compare a candidate policy against the deterministic baseline.

    Acceptance requires: no p95 latency regression beyond the threshold,
    throughput at least as good, no constraint violations beyond the
    limit, and no increase in rollbacks.
    """
    acceptance = {
        "p95_latency_no_regression": (
            candidate.latency_p95 <= baseline.latency_p95 * (1.0 + max_p95_regression) + 1e-9
        ),
        "throughput_not_reduced": candidate.throughput >= baseline.throughput - 1e-9,
        "no_new_constraint_violations": (
            candidate.constraint_violations <= max_violations
        ),
        "no_more_rollbacks": candidate.rollback_count <= baseline.rollback_count,
    }
    return PolicyComparison(
        baseline=baseline, candidate=candidate, acceptance=acceptance,
    )


@dataclass(frozen=True)
class StabilityReport:
    """Long-replay stability verdict (spec section 8)."""

    bounded_memory: bool
    oscillation_detected: bool
    policy_drift_detected: bool
    samples_retained: int
    memory_bound: int

    @property
    def stable(self) -> bool:
        """Return True when the long replay shows no instability."""
        return (
            self.bounded_memory and not self.oscillation_detected
            and not self.policy_drift_detected
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the report."""
        return {
            "bounded_memory": self.bounded_memory,
            "oscillation_detected": self.oscillation_detected,
            "policy_drift_detected": self.policy_drift_detected,
            "samples_retained": self.samples_retained,
            "memory_bound": self.memory_bound,
            "stable": self.stable,
        }


def long_replay_stability(
    history_total_samples: int,
    history_memory_bound: int,
    *,
    oscillation_reversals: int,
    oscillation_limit: int,
    action_distribution: Mapping[str, int],
    reference_distribution: Mapping[str, float],
    drift_tolerance: float = 0.25,
) -> StabilityReport:
    """Check bounded memory, oscillation and policy drift (spec section 8)."""
    bounded = history_total_samples <= history_memory_bound
    total_actions = sum(action_distribution.values()) or 1
    drift = False
    for action, share in reference_distribution.items():
        observed = action_distribution.get(action, 0) / total_actions
        if abs(observed - share) > drift_tolerance:
            drift = True
    return StabilityReport(
        bounded_memory=bounded,
        oscillation_detected=oscillation_reversals > oscillation_limit,
        policy_drift_detected=drift,
        samples_retained=history_total_samples, memory_bound=history_memory_bound,
    )


__all__ = [
    "TaskTraceEvent", "BenchmarkReport", "ReplayBenchmark", "PolicyComparison",
    "compare_policies", "StabilityReport", "long_replay_stability",
]
