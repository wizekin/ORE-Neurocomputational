"""Hard invariants, criticality and action filtering (ORE Point 23, Steps 13, 47-48).

Hard constraints are separate from the soft objective: any candidate plan
violating an invariant is rejected regardless of its predicted reward
(spec Step 13). Learning may never redefine these checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from ..actions import (
    DEFERRABLE_CLASSES, ActionType, PolicyOrigin, ResourceAction, RuntimeControlState,
)
from ..resource_model import (
    DEMAND_KINDS, ResourceDemand, ResourceKind, ResourceState, TaskRecord,
    WorkloadClass, WORKLOAD_CRITICALITY, CriticalityLevel,
)
from ..telemetry.quality import QualityReport


class RealTimeClaimError(ValueError):
    """Raised when a hard real-time claim lacks WCET or platform evidence."""


@dataclass(frozen=True)
class ConstraintVerdict:
    """Outcome of one hard-constraint evaluation."""

    allowed: bool
    violated: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the verdict."""
        return {
            "allowed": self.allowed, "violated": list(self.violated),
            "warnings": list(self.warnings),
        }


class CriticalityMatrix:
    """Enforceable criticality classification (spec Step 47).

    Criticality assignment is policy-owned: tasks cannot promote themselves
    (enforced in ``TaskRecord.reclassify``), and the matrix fixes which
    classes may be deferred and which hold reserved capacity.
    """

    def __init__(
        self,
        *,
        reserved_for_integrity: Mapping[ResourceKind, float] | None = None,
    ) -> None:
        """Configure the reservation floor protecting integrity-critical work."""
        self.reserved_for_integrity = dict(reserved_for_integrity or {})
        for kind, value in self.reserved_for_integrity.items():
            if not isinstance(kind, ResourceKind):
                raise TypeError("reservation keys must be ResourceKind members")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("reservation values must be finite and non-negative")

    def criticality(self, workload_class: WorkloadClass) -> CriticalityLevel:
        """Return the policy criticality of one workload class."""
        return WORKLOAD_CRITICALITY[workload_class]

    def may_defer(self, workload_class: WorkloadClass) -> bool:
        """Return True when a class may legally be deferred under pressure."""
        return workload_class in DEFERRABLE_CLASSES

    def deferral_order(self) -> tuple[WorkloadClass, ...]:
        """Return the order in which optional classes defer first (Step 47)."""
        order = {
            WorkloadClass.EXPLORATORY: 0,
            WorkloadClass.MAINTENANCE: 1,
            WorkloadClass.BATCH: 2,
            WorkloadClass.CORE_COGNITIVE: 3,
            WorkloadClass.INTERACTIVE: 4,
            WorkloadClass.INTEGRITY_CRITICAL: 5,
        }
        return tuple(sorted(WorkloadClass, key=lambda item: order[item]))

    def reservation_floor(self, state: ResourceState) -> Mapping[ResourceKind, float]:
        """Return the per-kind reservation floor implied by this matrix."""
        return dict(self.reserved_for_integrity)


class HardConstraintSet:
    """The invariants every candidate action must satisfy (spec Step 13)."""

    def __init__(
        self,
        *,
        min_workers: int = 1,
        max_workers: int = 64,
        fairness_floor: float = 0.90,
        integrity_weight_floor: float = 1.0,
        queue_limit: float | None = None,
        custom_validators: Sequence[Any] = (),
    ) -> None:
        """Configure budget, fairness and isolation invariants."""
        if min_workers <= 0 or max_workers < min_workers:
            raise ValueError("worker bounds must satisfy 0 < min_workers <= max_workers")
        if not 0.0 < fairness_floor <= 1.0:
            raise ValueError("fairness_floor must be within (0, 1]")
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.fairness_floor = fairness_floor
        self.integrity_weight_floor = integrity_weight_floor
        self.queue_limit = queue_limit
        self.custom_validators = tuple(custom_validators)

    def evaluate_action(
        self,
        action: ResourceAction,
        state: ResourceState,
        control: RuntimeControlState,
        *,
        quality: QualityReport | None = None,
    ) -> ConstraintVerdict:
        """Evaluate one action against every hard invariant."""
        violated: list[str] = []
        warnings: list[str] = []
        if action.state_version != state.version:
            violated.append(
                f"action targets state version {action.state_version} "
                f"but current is {state.version}"
            )
        if action.action_type is ActionType.ADJUST_WORKER_COUNT:
            target = action.parameters["target_worker_count"]
            if not self.min_workers <= target <= self.max_workers:
                violated.append(
                    f"worker count {target} outside [{self.min_workers}, {self.max_workers}]"
                )
        elif action.action_type is ActionType.ADJUST_CPU_WEIGHTS:
            proposed = {**control.cpu_weights, **action.parameters["cpu_weights"]}
            for module_id, weight in action.parameters["cpu_weights"].items():
                if weight < self.integrity_weight_floor and module_id.startswith("integrity"):
                    violated.append(
                        f"integrity module {module_id!r} weight below floor "
                        f"{self.integrity_weight_floor}"
                    )
            if proposed and min(proposed.values()) <= 0.0:
                violated.append("cpu weights must remain positive")
        elif action.action_type is ActionType.DEFER_BACKGROUND:
            defer = action.parameters.get("defer", True)
            if defer:
                for item in action.parameters["defer_classes"]:
                    if item not in DEFERRABLE_CLASSES:
                        violated.append(f"{item.value} cannot be deferred")
            else:
                warnings.append("resuming previously deferred classes")
        elif action.action_type is ActionType.ADJUST_ADMISSION:
            if action.parameters.get("allow_overcommit") and not control.allow_overcommit:
                if not action.parameters.get("overcommit_documented", False):
                    violated.append("overcommit must be documented to be enabled")
        elif action.action_type is ActionType.CHANGE_CACHE_TIER:
            sizes = action.parameters["tier_sizes"]
            if sum(sizes.values()) <= 0:
                violated.append("cache tier sizes must not all be zero")
            if sizes.get("hot", 0) < sizes.get("warm", 0):
                warnings.append("hot tier smaller than warm tier is unusual")
        if action.high_impact and quality is not None:
            for kind in (ResourceKind.RAM, ResourceKind.NET):
                if not quality.allows_high_impact(kind):
                    violated.append(
                        f"high-impact action requires {kind.value} confidence >= "
                        f"{quality.high_impact_confidence:.2f}"
                    )
        if self.queue_limit is not None:
            queue_dimension = state.dimension(ResourceKind.QUEUE)
            if queue_dimension.capacity is not None:
                if queue_dimension.allocated > self.queue_limit:
                    violated.append(
                        f"queue depth {queue_dimension.allocated} exceeds limit {self.queue_limit}"
                    )
        for validator in self.custom_validators:
            verdict = validator(action, state, control)
            if verdict:
                violated.extend(str(item) for item in verdict)
        return ConstraintVerdict(
            allowed=not violated, violated=tuple(violated), warnings=tuple(warnings),
        )

    def filter_safe_actions(
        self,
        candidates: Sequence[ResourceAction],
        state: ResourceState,
        control: RuntimeControlState,
        *,
        quality: QualityReport | None = None,
    ) -> tuple[tuple[ResourceAction, ...], dict[str, tuple[str, ...]]]:
        """Return safe actions and the rejection reasons for every candidate.

        Learned, predictive and deterministic actions are filtered by the
        same invariants; no predicted reward can bypass them (Step 13).
        """
        safe: list[ResourceAction] = []
        rejections: dict[str, tuple[str, ...]] = {}
        for action in candidates:
            verdict = self.evaluate_action(action, state, control, quality=quality)
            if verdict.allowed:
                safe.append(action)
            else:
                rejections[action.action_id] = verdict.violated
        return tuple(safe), rejections

    def verify_admission(
        self, demand: ResourceDemand, state: ResourceState,
        *, overcommit_allowed: bool = False,
    ) -> ConstraintVerdict:
        """Check whether one demand fits current headroom (spec section 2.1)."""
        violated: list[str] = []
        warnings: list[str] = []
        for kind in DEMAND_KINDS:
            headroom = state.headroom(kind)
            if headroom is None:
                warnings.append(f"{kind.value} capacity unknown; treating as unbounded")
                continue
            if demand.get(kind) > headroom + 1e-9:
                if overcommit_allowed and kind in (ResourceKind.RAM, ResourceKind.CPU):
                    violated.append(
                        f"documented overcommit: {kind.value} demand {demand.get(kind):.3f} "
                        f"exceeds headroom {headroom:.3f}"
                    )
                else:
                    violated.append(
                        f"{kind.value} demand {demand.get(kind):.3f} exceeds "
                        f"headroom {headroom:.3f}"
                    )
        return ConstraintVerdict(
            allowed=not violated, violated=tuple(violated), warnings=tuple(warnings),
        )


@dataclass(frozen=True)
class RealTimeTaskSpec:
    """One periodic task spec for worst-case analysis (spec Step 48)."""

    name: str
    period: float
    wcet: float
    deadline: float | None = None

    def __post_init__(self) -> None:
        """Reject non-positive periods and WCETs larger than the period."""
        for name, value in (("period", self.period), ("wcet", self.wcet)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        deadline = self.deadline if self.deadline is not None else self.period
        if deadline < self.period:
            raise ValueError("deadline cannot be shorter than the period")
        if self.wcet > self.period:
            raise ValueError(f"task {self.name!r} WCET exceeds its period")


@dataclass(frozen=True)
class WcetAnalysis:
    """Result of a worst-case execution-time feasibility analysis."""

    feasible: bool
    utilization: float
    bound: float
    binding_factor: str
    notes: tuple[str, ...] = ()


class WcetAnalyzer:
    """Bounded feasibility analysis for real-time claims (spec Step 48).

    Desktop Linux is not automatically hard real-time: a hard real-time
    claim requires bounded execution times, accounted scheduling overhead
    and platform evidence. This analyzer refuses unsupported claims.
    """

    #: Liu-Layland utilization bound n*(2^(1/n)-1) for Rate Monotonic sets.
    @staticmethod
    def rm_utilization_bound(task_count: int) -> float:
        """Return the Rate Monotonic utilization bound for n tasks."""
        if task_count <= 0:
            raise ValueError("task_count must be positive")
        if task_count == 1:
            return 1.0
        return task_count * (2.0 ** (1.0 / task_count) - 1.0)

    def analyze_rm(
        self, tasks: Sequence[RealTimeTaskSpec], *,
        scheduling_overhead: float = 0.0,
    ) -> WcetAnalysis:
        """Analyze a Rate Monotonic task set with explicit overhead accounting."""
        if not tasks:
            raise ValueError("at least one task spec is required")
        if not math.isfinite(scheduling_overhead) or scheduling_overhead < 0.0:
            raise ValueError("scheduling_overhead must be finite and non-negative")
        utilization = sum(
            (task.wcet + scheduling_overhead) / task.period for task in tasks
        )
        bound = self.rm_utilization_bound(len(tasks))
        notes = [
            f"utilization {utilization:.4f} vs RM bound {bound:.4f}",
            f"scheduling overhead per job: {scheduling_overhead:.6f}",
        ]
        return WcetAnalysis(
            feasible=utilization <= bound + 1e-12, utilization=utilization,
            bound=bound, binding_factor="liu-layland", notes=tuple(notes),
        )

    def hard_real_time_claim(
        self, tasks: Sequence[RealTimeTaskSpec], *,
        platform_evidence: Mapping[str, Any] | None = None,
    ) -> WcetAnalysis:
        """Validate a hard real-time claim, refusing it without full evidence.

        Raises ``RealTimeClaimError`` unless bounded WCETs, accounted
        overhead and explicit platform evidence are all supplied.
        """
        evidence = platform_evidence or {}
        missing = [
            name for name in
            ("wcet_bounds", "interrupt_bound", "blocking_bound", "platform")
            if not evidence.get(name)
        ]
        if missing:
            raise RealTimeClaimError(
                "no hard real-time claim without bounded execution and platform "
                f"evidence; missing: {missing}"
            )
        overhead = float(evidence.get("scheduling_overhead", 0.0))
        analysis = self.analyze_rm(tasks, scheduling_overhead=overhead)
        blocking = float(evidence.get("blocking_bound", 0.0))
        utilization = analysis.utilization + blocking / min(task.period for task in tasks)
        return WcetAnalysis(
            feasible=utilization <= analysis.bound + 1e-12, utilization=utilization,
            bound=analysis.bound, binding_factor="liu-layland-with-blocking",
            notes=(*analysis.notes, f"blocking bound {blocking:.6f}"),
        )


class QuotaLedger:
    """Per-module quota enforcement with recorded authorized exceptions (Step 23).

    A module cannot silently consume another module's reserved budget:
    borrowing requires an explicit, recorded authorization.
    """

    def __init__(self) -> None:
        """Initialize the empty ledger."""
        self._charges: dict[str, ResourceDemand] = {}
        self._exceptions: list[dict[str, Any]] = []

    def charge(
        self, module_id: str, usage: ResourceDemand, *,
        reserved: ResourceDemand, cap: ResourceDemand, borrowable: bool = False,
        authorized_exception: bool = False, reason: str = "",
    ) -> ConstraintVerdict:
        """Charge usage against one module's reserved+cap budget."""
        if not isinstance(module_id, str) or not module_id.strip():
            raise ValueError("module_id must be a non-empty string")
        current = self._charges.get(module_id, ResourceDemand())
        total = ResourceDemand(
            cpu=current.cpu + usage.cpu, ram=current.ram + usage.ram,
            io=current.io + usage.io, net=current.net + usage.net,
            gpu=current.gpu + usage.gpu, storage=current.storage + usage.storage,
        )
        violated: list[str] = []
        for kind in DEMAND_KINDS:
            limit = cap.get(kind)
            used = total.get(kind)
            if used > limit + 1e-9:
                if borrowable or authorized_exception:
                    self._exceptions.append({
                        "module_id": module_id, "resource": kind.value,
                        "used": used, "limit": limit,
                        "reason": reason or ("borrowing" if borrowable else "authorized exception"),
                    })
                else:
                    violated.append(
                        f"{module_id} exceeds {kind.value} budget: {used:.3f} > {limit:.3f}"
                    )
        if not violated:
            self._charges[module_id] = total
        return ConstraintVerdict(allowed=not violated, violated=tuple(violated))

    def exceptions(self) -> tuple[dict[str, Any], ...]:
        """Return every recorded budget exception."""
        return tuple(dict(item) for item in self._exceptions)

    def reset(self) -> None:
        """Clear accumulated charges and exception records."""
        self._charges.clear()
        self._exceptions.clear()


__all__ = [
    "CriticalityMatrix", "HardConstraintSet", "ConstraintVerdict", "QuotaLedger",
    "WcetAnalyzer", "WcetAnalysis", "RealTimeTaskSpec", "RealTimeClaimError",
]
