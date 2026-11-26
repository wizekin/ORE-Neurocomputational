"""Deterministic baseline policy (ORE Point 23, Step 16).

The baseline scheduler is the known-good reference every adaptive policy is
compared against: explicit task classes, dependency readiness, capability
checks, bounded worker pools, priority weights, quotas and admission
control. It is a pure function of its inputs: identical task traces under
the deterministic policy produce equivalent allocation decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..actions import DEFERRABLE_CLASSES, RuntimeControlState
from ..resource_model import (
    CLASS_ORDER, DEFAULT_CLASS_WEIGHTS, DEMAND_KINDS, ResourceDemand,
    ResourceKind, ResourceState, TaskRecord, TaskState, WorkloadClass,
)


@dataclass(frozen=True)
class AdmissionDecision:
    """Outcome of one admission-control decision."""

    task_id: str
    admitted: bool
    deferred: bool
    reason: str
    deferral_class: WorkloadClass | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the decision."""
        return {
            "task_id": self.task_id, "admitted": self.admitted,
            "deferred": self.deferred, "reason": self.reason,
            "deferral_class": self.deferral_class.value if self.deferral_class else None,
        }


class AdmissionController:
    """Resource-aware admission with criticality-aware deferral (Step 16).

    Work is deferred or rejected before resource ceilings are exceeded.
    Under pressure, optional classes defer before core and integrity work.
    """

    def __init__(
        self,
        *,
        overcommit_policy: Mapping[ResourceKind, bool] | None = None,
    ) -> None:
        """Configure which resources permit documented overcommit, if any."""
        self.overcommit_policy = dict(overcommit_policy or {})

    def decide(
        self, task: TaskRecord, state: ResourceState, *,
        reserved: ResourceDemand = ResourceDemand(),
    ) -> AdmissionDecision:
        """Return the admission decision for one task against current state."""
        fits = True
        binding_kind: ResourceKind | None = None
        for kind in DEMAND_KINDS:
            headroom = state.headroom(kind)
            if headroom is None:
                continue
            demand = task.demand.get(kind) - reserved.get(kind) * (
                1.0 if task.workload_class is WorkloadClass.INTEGRITY_CRITICAL else 0.0
            )
            if demand > headroom + 1e-9:
                fits = False
                binding_kind = kind
                break
        if fits:
            return AdmissionDecision(
                task_id=task.task_id, admitted=True, deferred=False,
                reason="demand fits current headroom",
            )
        if task.workload_class in DEFERRABLE_CLASSES:
            return AdmissionDecision(
                task_id=task.task_id, admitted=False, deferred=True,
                reason=(
                    f"{binding_kind.value if binding_kind else 'resource'} headroom "
                    f"exhausted; optional work deferred"
                ),
                deferral_class=task.workload_class,
            )
        return AdmissionDecision(
            task_id=task.task_id, admitted=False, deferred=False,
            reason=(
                f"{binding_kind.value if binding_kind else 'resource'} headroom "
                f"exhausted for non-deferrable work"
            ),
        )


@dataclass(frozen=True)
class ScheduleDecision:
    """Deterministic scheduling decision for one scheduling round."""

    admitted_order: tuple[str, ...]
    deferred: tuple[str, ...]
    rejected: tuple[str, ...]
    decisions: tuple[AdmissionDecision, ...] = ()
    order_rationale: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the decision."""
        return {
            "admitted_order": list(self.admitted_order), "deferred": list(self.deferred),
            "rejected": list(self.rejected),
            "decisions": [item.to_dict() for item in self.decisions],
            "order_rationale": list(self.order_rationale),
        }


class DeterministicBaselineScheduler:
    """Known-good deterministic scheduler (spec Step 16).

    Ordering rule (stable and total): dependency readiness first, then
    criticality, then policy class order, then ready time, then task id.
    The worker pool bounds admitted concurrency; admission control runs
    before any task enters the pool.
    """

    def __init__(
        self,
        *,
        available_capabilities: Sequence[str] = (),
        class_weights: Mapping[WorkloadClass, float] | None = None,
        admission: AdmissionController | None = None,
    ) -> None:
        """Configure capabilities, class weights and admission controller."""
        self.available_capabilities = frozenset(available_capabilities)
        self.class_weights = dict(class_weights or DEFAULT_CLASS_WEIGHTS)
        self.admission = admission or AdmissionController()

    def dependencies_ready(self, task: TaskRecord, completed: Iterable[str]) -> bool:
        """Return True when every dependency of the task has completed."""
        completed_set = set(completed)
        return all(dependency in completed_set for dependency in task.dependencies)

    def capabilities_ok(self, task: TaskRecord) -> bool:
        """Return True when required capabilities are available."""
        return set(task.capabilities) <= self.available_capabilities

    def order_key(
        self, task: TaskRecord, *, now: float = 0.0,
    ) -> tuple[float, float, float, float, str]:
        """Return the deterministic, total ordering key for one ready task."""
        slack = task.slack(now)
        urgency = 0.0 if slack is None else max(0.0, 1.0 / (1.0 + slack))
        return (
            float(task.criticality.value),
            float(CLASS_ORDER[task.workload_class]),
            -urgency,
            task.ready_time,
            task.task_id,
        )

    def schedule(
        self,
        tasks: Sequence[TaskRecord],
        state: ResourceState,
        control: RuntimeControlState,
        *,
        completed: Iterable[str] = (),
        now: float = 0.0,
        reserved: ResourceDemand = ResourceDemand(),
    ) -> ScheduleDecision:
        """Produce the deterministic schedule for one round of tasks."""
        completed_set = set(completed)
        candidates: list[TaskRecord] = []
        rationale: list[str] = []
        for task in sorted(tasks, key=lambda item: item.task_id):
            if task.state not in (TaskState.READY, TaskState.PENDING):
                continue
            if not self.dependencies_ready(task, completed_set):
                rationale.append(f"{task.task_id}: blocked by dependencies")
                continue
            if not self.capabilities_ok(task):
                rationale.append(f"{task.task_id}: missing capabilities")
                continue
            candidates.append(task)
        ordered = sorted(candidates, key=lambda item: self.order_key(item, now=now))
        admitted: list[str] = []
        deferred: list[str] = []
        rejected: list[str] = []
        decisions: list[AdmissionDecision] = []
        for task in ordered:
            decision = self.admission.decide(task, state, reserved=reserved)
            decisions.append(decision)
            if decision.admitted:
                admitted.append(task.task_id)
            elif decision.deferred:
                deferred.append(task.task_id)
            else:
                rejected.append(task.task_id)
        if control.deferred_classes:
            filtered_admitted: list[str] = []
            task_by_id = {task.task_id: task for task in ordered}
            for task_id in admitted:
                task = task_by_id[task_id]
                if task.workload_class in control.deferred_classes:
                    deferred.append(task_id)
                    rationale.append(f"{task_id}: deferred by DEFER_BACKGROUND policy")
                else:
                    filtered_admitted.append(task_id)
            admitted = filtered_admitted
        bounded = admitted[: max(1, control.worker_count)]
        for task_id in admitted[max(1, control.worker_count):]:
            rationale.append(f"{task_id}: waiting for bounded worker pool")
        return ScheduleDecision(
            admitted_order=tuple(bounded),
            deferred=tuple(deferred),
            rejected=tuple(rejected),
            decisions=tuple(decisions),
            order_rationale=tuple(rationale),
        )


__all__ = [
    "AdmissionController", "AdmissionDecision", "DeterministicBaselineScheduler",
    "ScheduleDecision",
]
