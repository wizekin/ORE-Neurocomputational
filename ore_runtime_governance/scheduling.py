"""Deadline-aware and budget scheduling (integrated steps 43-44, section 5.6).

Classical real-time results are applied only under their own assumptions:
utilization U = sum_i(C_i/T_i); Earliest Deadline First orders by absolute
deadline and is uniprocessor-feasible for independent preemptible tasks when
U <= 1; Rate Monotonic assigns higher priority to shorter periods with the
Liu-Layland sufficient bound n(2^(1/n)-1). Cognitive workloads are usually
non-periodic and uncertain, so the honest path for them is budget
scheduling with class budgets, priorities, aging, and reserved capacity for
coordinator, safety, persistence, and recovery work (integrated step 44).
Budget and deadline accounting use monotonic clocks (section 5.6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

#: Classes that always keep reserved capacity (integrated step 44).
RESERVED_CLASSES = ("coordinator", "safety", "persistence", "recovery")


@dataclass(frozen=True)
class PeriodicSpec:
    """One periodic real-time task with execution time C and period T."""

    task_id: str
    execution_time_s: float
    period_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if self.execution_time_s <= 0 or self.period_s <= 0:
            raise ValueError("C_i and T_i must be positive")
        if self.execution_time_s > self.period_s:
            raise ValueError("C_i must fit inside the period T_i")


def utilization(specs: Iterable[PeriodicSpec]) -> float:
    """U = sum_i(C_i/T_i) (integrated step 43, section 5.2)."""
    return sum(spec.execution_time_s / spec.period_s for spec in specs)


def rms_order(specs: Iterable[PeriodicSpec]) -> tuple[PeriodicSpec, ...]:
    """Rate Monotonic: shorter periods get higher priority."""
    return tuple(sorted(specs, key=lambda spec: (spec.period_s, spec.task_id)))


def edf_order(specs: Iterable[PeriodicSpec], *, current_time: float = 0.0) -> tuple[PeriodicSpec, ...]:
    """Earliest Deadline First ordering by absolute deadline (step 43)."""
    return tuple(
        sorted(
            specs,
            key=lambda spec: (current_time + spec.period_s, spec.task_id),
        )
    )


def rms_sufficient_bound(task_count: int) -> float:
    """Liu-Layland sufficient utilization bound for Rate Monotonic."""
    if task_count < 1:
        raise ValueError("task count must be at least 1")
    if task_count == 1:
        return 1.0
    return task_count * (2 ** (1.0 / task_count) - 1.0)


@dataclass(frozen=True)
class SchedulabilityReport:
    """Honest feasibility report with the assumptions it rests on."""

    utilization: float
    edf_feasible: bool
    rms_sufficient: bool
    rms_bound: float
    assumptions: tuple[str, ...]
    caveat: str = (
        "classical bounds hold only for independent, preemptible, periodic tasks "
        "with known C_i; stochastic cognitive workloads must not use them blindly"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "utilization": self.utilization,
            "edf_feasible": self.edf_feasible,
            "rms_sufficient": self.rms_sufficient,
            "rms_bound": self.rms_bound,
            "assumptions": list(self.assumptions),
            "caveat": self.caveat,
        }


def schedulability_report(specs: Iterable[PeriodicSpec]) -> SchedulabilityReport:
    """EDF feasibility (U <= 1) and the RMS sufficient bound (step 43)."""
    specs = tuple(specs)
    total = utilization(specs)
    return SchedulabilityReport(
        utilization=total,
        edf_feasible=total <= 1.0,
        rms_sufficient=total <= rms_sufficient_bound(len(specs)) if specs else True,
        rms_bound=rms_sufficient_bound(len(specs)) if specs else 1.0,
        assumptions=(
            "independent tasks",
            "preemptible uniprocessor",
            "known worst-case execution times",
            "periodic release with deadline equal to the period",
        ),
    )


def deadline_remaining(deadline_monotonic: float, now: float) -> float:
    """D_remaining = deadline - current monotonic time (section 5.6)."""
    return deadline_monotonic - now


def budget_remaining(initial_budget_s: float, cpu_used_s: float, charged_overhead_s: float = 0.0) -> float:
    """B_remaining = B_initial - CPU_time_used - charged_overhead (section 5.6)."""
    if initial_budget_s < 0 or cpu_used_s < 0 or charged_overhead_s < 0:
        raise ValueError("budget terms must be non-negative")
    return initial_budget_s - cpu_used_s - charged_overhead_s


def safe_to_start(
    deadline_remaining_s: float,
    estimated_duration_s: float,
    *,
    validation_reserve_s: float = 0.1,
    persistence_reserve_s: float = 0.1,
) -> tuple[bool, float]:
    """A positive D_remaining alone does not make a task safe to start.

    The start decision reserves time for validation and persistence before
    comparing against the estimated duration (integrated step 6/section 5.6).
    """
    if estimated_duration_s < 0:
        raise ValueError("estimated duration must be non-negative")
    reserved = validation_reserve_s + persistence_reserve_s
    slack = deadline_remaining_s - estimated_duration_s - reserved
    return slack > 0, slack


@dataclass(frozen=True)
class SchedClass:
    """One workload class with its budget, priority, and aging policy (step 44)."""

    class_id: str
    cpu_budget_s: float
    wall_clock_budget_s: float
    priority: int
    aging_rate: float = 0.0
    reserved: bool = False

    _RESERVED_IDS = RESERVED_CLASSES

    def __post_init__(self) -> None:
        if not isinstance(self.class_id, str) or not self.class_id.strip():
            raise ValueError("class_id must be non-empty")
        if self.cpu_budget_s <= 0 or self.wall_clock_budget_s <= 0:
            raise ValueError("class budgets must be positive")
        if not 0 <= self.priority <= 10:
            raise ValueError("priority must be within 0..10")
        if self.aging_rate < 0:
            raise ValueError("aging_rate must be non-negative")
        if self.reserved and self.class_id not in self._RESERVED_IDS:
            raise ValueError(
                f"reserved classes are limited to {list(self._RESERVED_IDS)}"
            )

    def effective_priority(self, waited_s: float) -> int:
        """Aging raises effective priority with waiting time, capped at 10."""
        return min(10, self.priority + int(self.aging_rate * max(0.0, waited_s)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "cpu_budget_s": self.cpu_budget_s,
            "wall_clock_budget_s": self.wall_clock_budget_s,
            "priority": self.priority,
            "aging_rate": self.aging_rate,
            "reserved": self.reserved,
        }


@dataclass(frozen=True)
class SchedTask:
    """One schedulable task instance with remaining budget."""

    task_id: str
    class_id: str
    arrived_monotonic: float
    remaining_budget_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if self.remaining_budget_s < 0:
            raise ValueError("remaining budget must be non-negative")


@dataclass(frozen=True)
class PressureReport:
    """Graceful-degradation report under queue pressure (step 44)."""

    queue_pressure: float
    shed_task_ids: tuple[str, ...]
    reserved_starving: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_pressure": self.queue_pressure,
            "shed_task_ids": list(self.shed_task_ids),
            "reserved_starving": self.reserved_starving,
            "reason": self.reason,
        }


class BudgetScheduler:
    """Budget scheduler with priority aging and reserved capacity (step 44).

    Deterministic by design: every method takes ``now`` explicitly so tests
    and model checks replay scheduling decisions exactly.
    """

    def __init__(
        self,
        classes: Iterable[SchedClass],
        *,
        pressure_threshold: float = 0.75,
        starvation_seconds: float = 30.0,
    ) -> None:
        self.classes: dict[str, SchedClass] = {}
        for sched_class in classes:
            if sched_class.class_id in self.classes:
                raise ValueError(f"duplicate scheduling class {sched_class.class_id!r}")
            self.classes[sched_class.class_id] = sched_class
        if not self.classes:
            raise ValueError("at least one scheduling class is required")
        if not 0.0 <= pressure_threshold <= 1.0:
            raise ValueError("pressure_threshold must be within 0..1")
        if starvation_seconds <= 0:
            raise ValueError("starvation_seconds must be positive")
        self.pressure_threshold = pressure_threshold
        self.starvation_seconds = starvation_seconds
        self._last_selected: dict[str, float] = {}

    def _class_of(self, task: SchedTask) -> SchedClass:
        try:
            return self.classes[task.class_id]
        except KeyError as exc:
            raise KeyError(
                f"task {task.task_id} references unknown class {task.class_id!r}"
            ) from exc

    def effective_priority(self, task: SchedTask, now: float) -> int:
        return self._class_of(task).effective_priority(max(0.0, now - task.arrived_monotonic))

    def select_next(self, ready: Iterable[SchedTask], now: float) -> SchedTask | None:
        """Pick the highest effective-priority task with remaining budget."""
        candidates = [
            task
            for task in ready
            if task.remaining_budget_s > 0
        ]
        if not candidates:
            return None
        ordered = sorted(
            candidates,
            key=lambda task: (-self.effective_priority(task, now), task.arrived_monotonic, task.task_id),
        )
        selected = ordered[0]
        self._last_selected[selected.task_id] = now
        return selected

    def shed_under_pressure(
        self,
        ready: Iterable[SchedTask],
        *,
        queue_pressure: float,
    ) -> PressureReport:
        """Shed optional (non-reserved) low-priority work under pressure.

        Reserved classes (coordinator, safety, persistence, recovery) are
        never shed; low-priority exploration yields first (step 44).
        """
        tasks = tuple(ready)
        if queue_pressure < 0:
            raise ValueError("queue_pressure must be non-negative")
        if queue_pressure < self.pressure_threshold:
            return PressureReport(
                queue_pressure=queue_pressure,
                shed_task_ids=(),
                reserved_starving=False,
                reason="pressure below threshold; nothing shed",
            )
        shed: list[str] = []
        for task in tasks:
            sched_class = self._class_of(task)
            if sched_class.reserved:
                continue
            if sched_class.priority <= 3:
                shed.append(task.task_id)
        return PressureReport(
            queue_pressure=queue_pressure,
            shed_task_ids=tuple(sorted(shed)),
            reserved_starving=False,
            reason=(
                f"queue pressure {queue_pressure:.2f} >= "
                f"{self.pressure_threshold}; optional low-priority work yields, "
                "reserved classes keep their capacity"
            ),
        )

    def reserved_starvation_check(self, ready: Iterable[SchedTask], now: float) -> SchedTask | None:
        """Return a reserved task that has waited beyond the starvation bound."""
        for task in sorted(ready, key=lambda item: item.task_id):
            if self._class_of(task).reserved and now - task.arrived_monotonic > self.starvation_seconds:
                return task
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "classes": {
                class_id: sched_class.to_dict() for class_id, sched_class in sorted(self.classes.items())
            },
            "pressure_threshold": self.pressure_threshold,
            "starvation_seconds": self.starvation_seconds,
            "last_selected": dict(sorted(self._last_selected.items())),
        }
