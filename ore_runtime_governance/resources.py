"""Resource plane: records, reservations, admission control (steps 18, 26, 36-39).

The resource model distinguishes requested demand d_ij, reserved capacity
r_ij, observed usage u_ij, and enforced ceilings L_ij (integrated step 37).
Admission is granted only when

    admit(i) = policy_ok(i) AND dependencies_ready(i)
               AND for_all_j(reserved_j + d_ij <= capacity_j)          (section 5.2)

with capacity c_j = host capacity - system reserve. Hierarchical ownership
(integrated step 18) enforces sum(child_allocations) <= parent_budget after
reserved overhead, and child components can never enlarge their own budgets.
Reservations are context-managed so cleanup runs on completion, exception,
cancellation, or process death, with leak detection for reservations that
outlive their owning task (step 26). In-process counters flag hard-limit
breaches; actual enforcement against hostile native code belongs to the host
backends (section 2, "Host-enforced boundaries").
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .contracts import ResourceVector


@dataclass(frozen=True)
class Reservation:
    """One granted resource reservation (section 6, ResourceManager.reserve)."""

    reservation_id: str
    task_id: str
    owner: str
    amounts: ResourceVector
    limits: ResourceVector
    created_monotonic: float
    deadline_monotonic: float | None
    generation: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "reservation_id": self.reservation_id,
            "task_id": self.task_id,
            "owner": self.owner,
            "amounts": self.amounts.to_dict(),
            "limits": self.limits.to_dict(),
            "created_monotonic": self.created_monotonic,
            "deadline_monotonic": self.deadline_monotonic,
            "generation": self.generation,
        }


@dataclass(frozen=True)
class ReservationResult:
    """Typed admission outcome (never an ambiguous bool)."""

    granted: bool
    reason: str
    reservation: Reservation | None = None
    requested: ResourceVector | None = None
    remaining: ResourceVector | None = None
    feasibility: Mapping[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "granted": self.granted,
            "reason": self.reason,
            "reservation": self.reservation.to_dict() if self.reservation else None,
            "requested": self.requested.to_dict() if self.requested else None,
            "remaining": self.remaining.to_dict() if self.remaining else None,
            "feasibility": dict(self.feasibility),
        }


@dataclass(frozen=True)
class ReleaseResult:
    """Typed release outcome; idempotent and leak-aware."""

    reservation_id: str
    released: bool
    reason: str
    leaked: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "reservation_id": self.reservation_id,
            "released": self.released,
            "reason": self.reason,
            "leaked": self.leaked,
        }


@dataclass(frozen=True)
class LimitBreachRecord:
    """Observed usage u_ij exceeded the enforced ceiling L_ij (step 37)."""

    reservation_id: str
    task_id: str
    kind: str
    observed: float
    limit: float
    enforcement: str
    action: str
    detected_monotonic: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "reservation_id": self.reservation_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "observed": self.observed,
            "limit": self.limit,
            "enforcement": self.enforcement,
            "action": self.action,
            "detected_monotonic": self.detected_monotonic,
        }


class AdmissionDeniedError(RuntimeError):
    """Raised by strict reservation when admission control denies the task."""


class ResourceManager:
    """Reservation ledger with hierarchical budgets and leak detection.

    Section 5.1: for tasks i=1..n and resource kinds j, admission requires
    x_ij >= a_ij for admitted tasks and sum_i x_ij <= c_j for every resource,
    where c_j is the allocatable capacity after the system reserve s_j.
    """

    def __init__(
        self,
        capacity: ResourceVector | None = None,
        *,
        label: str = "root",
        parent: "ResourceManager | None" = None,
        system_reserve: ResourceVector | None = None,
        clock=None,
    ) -> None:
        if capacity is None and parent is not None:
            raise ValueError("child budgets must declare their capacity")
        if capacity is None:
            capacity = ResourceVector()
        if not isinstance(capacity, ResourceVector):
            raise TypeError("capacity must be a ResourceVector")
        if system_reserve is not None:
            if not isinstance(system_reserve, ResourceVector):
                raise TypeError("system_reserve must be a ResourceVector")
            if not system_reserve.elementwise_le(capacity):
                raise ValueError("system reserve s_j must fit inside the capacity")
            capacity = capacity - system_reserve
        if parent is not None and not capacity.elementwise_le(parent._allocatable):
            raise ValueError(
                "child budget must be within the parent allocatable capacity "
                "(children cannot enlarge their own budgets)"
            )
        self.label = label
        self._capacity = capacity
        self._allocatable = capacity
        self._parent = parent
        self._clock = clock or time.monotonic
        self._reservations: dict[str, Reservation] = {}
        self._released: set[str] = set()
        self._observed: dict[str, ResourceVector] = {}
        self._breaches: list[LimitBreachRecord] = []
        self._children: list[ResourceManager] = []
        self._lock = threading.RLock()
        if parent is not None:
            parent._children.append(self)

    # -- accounting ------------------------------------------------------------

    def _reserved_direct(self) -> ResourceVector:
        total = ResourceVector()
        for reservation in self._reservations.values():
            if reservation.reservation_id not in self._released:
                total = total + reservation.amounts
        return total

    def reserved_total(self) -> ResourceVector:
        """Reserved amounts including every child budget (step 18)."""
        with self._lock:
            total = self._reserved_direct()
            for child in self._children:
                total = total + child.reserved_total()
            return total

    def available(self) -> ResourceVector:
        with self._lock:
            return self._allocatable - self.reserved_total()

    def capacity(self) -> ResourceVector:
        return self._allocatable

    # -- hierarchical budgets (integrated step 18) --------------------------------

    def child_budget(
        self,
        label: str,
        capacity: ResourceVector,
        *,
        system_reserve: ResourceVector | None = None,
    ) -> "ResourceManager":
        """Delegate a bounded budget to a child controller (step 18).

        For resource r the parent enforces sum(child_allocations) <=
        parent_budget after reserved overhead; the child can never enlarge
        its own budget because its capacity is fixed here.
        """
        child = ResourceManager(
            capacity,
            label=label,
            parent=self,
            system_reserve=system_reserve,
            clock=self._clock,
        )
        # The child constructor registers itself with the parent ledger.
        return child

    # -- feasibility -------------------------------------------------------------

    def _feasibility(self, request: ResourceVector) -> dict[str, bool]:
        """Per-kind check along the ownership chain (section 5.2, step 18).

        Every ancestor enforces sum(child_allocations) + request <= its own
        allocatable budget, so no subtree can over-allocate its parent.
        """
        chain = self._ancestry()
        feasibility: dict[str, bool] = {}
        for kind in ResourceVector.kinds():
            if request.amounts()[kind] <= 0:
                continue
            feasibility[kind] = all(
                ancestor.reserved_total().amounts()[kind] + request.amounts()[kind]
                <= ancestor._allocatable.amounts()[kind]
                for ancestor in chain
            )
        return feasibility

    def _ancestry(self) -> list["ResourceManager"]:
        """Return [self, parent, grandparent, ...] up to the root budget."""
        chain: list[ResourceManager] = [self]
        node: ResourceManager | None = self._parent
        while node is not None:
            chain.append(node)
            node = node._parent
        return chain

    def admission_report(
        self,
        task_id: str,
        request_vector: ResourceVector,
        *,
        policy_ok: bool = True,
        dependencies_ready: bool = True,
    ) -> ReservationResult:
        """Evaluate admission without creating a reservation (step 38)."""
        if not isinstance(request_vector, ResourceVector):
            raise TypeError("request_vector must be a ResourceVector")
        reasons: list[str] = []
        if not policy_ok:
            reasons.append("policy check failed for the task profile")
        if not dependencies_ready:
            reasons.append("declared dependencies are not ready yet")
        feasibility = self._feasibility(request_vector)
        exceeded = [kind for kind, ok in feasibility.items() if not ok]
        if exceeded:
            chain = self._ancestry()
            for kind in exceeded:
                binding = next(
                    (
                        ancestor
                        for ancestor in chain
                        if ancestor.reserved_total().amounts()[kind]
                        + request_vector.amounts()[kind]
                        > ancestor._allocatable.amounts()[kind]
                    ),
                    self,
                )
                reasons.append(
                    f"resource {kind}: budget '{binding.label}' reserved "
                    f"{binding.reserved_total().amounts()[kind]} + requested "
                    f"{request_vector.amounts()[kind]} exceeds capacity "
                    f"{binding._allocatable.amounts()[kind]}"
                )
        granted = not reasons
        return ReservationResult(
            granted=granted,
            reason="; ".join(reasons) if reasons else "admission feasible",
            requested=request_vector,
            remaining=self.available(),
            feasibility=feasibility,
        )

    # -- reserve / release ---------------------------------------------------------

    def reserve(
        self,
        task_id: str,
        request_vector: ResourceVector,
        *,
        owner: str = "",
        limits: ResourceVector | None = None,
        deadline_monotonic: float | None = None,
        policy_ok: bool = True,
        dependencies_ready: bool = True,
    ) -> ReservationResult:
        """Reserve resources for one task or return a typed denial (step 38)."""
        report = self.admission_report(
            task_id,
            request_vector,
            policy_ok=policy_ok,
            dependencies_ready=dependencies_ready,
        )
        with self._lock:
            if not report.granted:
                return report
            reservation = Reservation(
                reservation_id=f"res-{uuid.uuid4().hex}",
                task_id=task_id,
                owner=owner or task_id,
                amounts=request_vector,
                limits=limits if limits is not None else request_vector,
                created_monotonic=self._clock(),
                deadline_monotonic=deadline_monotonic,
                generation=len(self._reservations) + 1,
            )
            self._reservations[reservation.reservation_id] = reservation
            return ReservationResult(
                granted=True,
                reason="reserved",
                reservation=reservation,
                requested=request_vector,
                remaining=self.available(),
                feasibility=report.feasibility,
            )

    def _owning_manager(self, reservation_id: str) -> "ResourceManager | None":
        """Find the manager in this subtree holding a reservation id."""
        for manager in self._all_managers():
            if reservation_id in manager._reservations:
                return manager
        return None

    def release(self, reservation_id: str, reason: str = "") -> ReleaseResult:
        """Release a reservation anywhere in the budget subtree; idempotent."""
        with self._lock:
            manager = self._owning_manager(reservation_id)
            if manager is None:
                return ReleaseResult(
                    reservation_id=reservation_id,
                    released=False,
                    reason="unknown reservation id",
                    leaked=False,
                )
            return manager._release_local(reservation_id, reason)

    def _release_local(self, reservation_id: str, reason: str) -> ReleaseResult:
        with self._lock:
            if reservation_id in self._released:
                return ReleaseResult(
                    reservation_id=reservation_id,
                    released=False,
                    reason="already released (idempotent release)",
                    leaked=False,
                )
            self._released.add(reservation_id)
            return ReleaseResult(
                reservation_id=reservation_id,
                released=True,
                reason=reason or "released by owner",
                leaked=False,
            )

    def reservation(self, reservation_id: str) -> Reservation | None:
        with self._lock:
            manager = self._owning_manager(reservation_id)
            if manager is None:
                return None
            if reservation_id in manager._released:
                return None
            return manager._reservations[reservation_id]

    def active_reservations(self) -> tuple[Reservation, ...]:
        with self._lock:
            return tuple(
                reservation
                for reservation_id, reservation in sorted(self._reservations.items())
                if reservation_id not in self._released
            )

    # -- usage observation (step 37) ---------------------------------------------

    def observe(self, reservation_id: str, usage: ResourceVector) -> list[LimitBreachRecord]:
        """Record observed usage u_ij and flag breaches of the ceiling L_ij."""
        if not isinstance(usage, ResourceVector):
            raise TypeError("usage must be a ResourceVector")
        with self._lock:
            manager = self._owning_manager(reservation_id)
            if manager is None or reservation_id in manager._released:
                raise KeyError(f"no active reservation {reservation_id!r} to observe")
            return manager._observe_local(reservation_id, usage)

    def _observe_local(self, reservation_id: str, usage: ResourceVector) -> list[LimitBreachRecord]:
        with self._lock:
            found = self._reservations[reservation_id]
            previous = self._observed.get(reservation_id, ResourceVector())
            self._observed[reservation_id] = previous + usage
            breaches: list[LimitBreachRecord] = []
            for kind in ResourceVector.kinds():
                observed_value = self._observed[reservation_id].amounts()[kind]
                limit_value = found.limits.amounts()[kind]
                if limit_value > 0 and observed_value > limit_value:
                    record = LimitBreachRecord(
                        reservation_id=reservation_id,
                        task_id=found.task_id,
                        kind=kind,
                        observed=observed_value,
                        limit=limit_value,
                        enforcement="in_process_counter",
                        action=(
                            "flagged; host backend (rlimits/cgroups/container) "
                            "performs the actual enforcement or termination"
                        ),
                        detected_monotonic=self._clock(),
                    )
                    breaches.append(record)
                    self._breaches.append(record)
            return breaches

    def breaches(self) -> tuple[LimitBreachRecord, ...]:
        with self._lock:
            return tuple(
                breach
                for manager in self._all_managers()
                for breach in manager._breaches
            )

    # -- lifecycle hygiene (step 26) ----------------------------------------------

    def _all_managers(self) -> list["ResourceManager"]:
        """Every manager in this budget subtree (self plus descendants)."""
        managers: list[ResourceManager] = [self]
        for child in self._children:
            managers.extend(child._all_managers())
        return managers

    def leaked_reservations(self, active_task_ids: Iterable[str]) -> tuple[Reservation, ...]:
        """Reservations whose owning task no longer exists (leak detection)."""
        active = set(active_task_ids)
        return tuple(
            reservation
            for manager in self._all_managers()
            for reservation in manager.active_reservations()
            if reservation.task_id not in active
        )

    def reconcile(
        self, active_task_ids: Iterable[str], reason: str = "supervisor reconciliation"
    ) -> tuple[ReleaseResult, ...]:
        """Release ghost reservations left by crashed workers (step 35)."""
        active = set(active_task_ids)
        results: list[ReleaseResult] = []
        for manager in self._all_managers():
            for reservation in manager.active_reservations():
                if reservation.task_id not in active:
                    results.append(manager.release(reservation.reservation_id, reason))
        return tuple(results)

    # -- context-managed leases (step 26) -------------------------------------------

    def session(
        self,
        task_id: str,
        request_vector: ResourceVector,
        **kwargs: Any,
    ) -> "_ReservationSession":
        """Context manager guaranteeing release on completion or exception."""
        return _ReservationSession(self, task_id, request_vector, kwargs)

    # -- description (requested/reserved/observed/enforced) ------------------------

    def describe(self) -> dict[str, Any]:
        with self._lock:
            reserved = self.reserved_total()
            requested = ResourceVector()
            observed_peak = ResourceVector()
            for reservation in self.active_reservations():
                requested = requested + reservation.amounts
                observed_peak = observed_peak + self._observed.get(
                    reservation.reservation_id, ResourceVector()
                )
            return {
                "label": self.label,
                "kinds": {
                    kind: {
                        "requested": requested.amounts()[kind],
                        "reserved": reserved.amounts()[kind],
                        "observed": observed_peak.amounts()[kind],
                        "enforced_limit": self._allocatable.amounts()[kind],
                        "enforcement": "in_process_counter (host backends enforce hard limits)",
                    }
                    for kind in ResourceVector.kinds()
                },
                "children": [child.label for child in self._children],
                "active_reservations": len(self.active_reservations()),
                "breaches": [breach.to_dict() for breach in self._breaches],
            }


class _ReservationSession:
    """Context manager so reservations never outlive their scope (step 26)."""

    def __init__(
        self,
        manager: ResourceManager,
        task_id: str,
        request_vector: ResourceVector,
        kwargs: Mapping[str, Any],
    ) -> None:
        self.manager = manager
        self.task_id = task_id
        self.request_vector = request_vector
        self.kwargs = dict(kwargs)
        self.result: ReservationResult | None = None

    def __enter__(self) -> ReservationResult:
        self.result = self.manager.reserve(self.task_id, self.request_vector, **self.kwargs)
        if not self.result.granted:
            raise AdmissionDeniedError(self.result.reason)
        return self.result

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.result is not None and self.result.reservation is not None:
            self.manager.release(
                self.result.reservation.reservation_id,
                reason=f"context exit ({'exception' if exc_type else 'completion'})",
            )
        return False
