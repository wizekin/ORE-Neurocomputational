"""Task leases with fencing tokens and health tracking (Steps 12, 15).

Leases carry monotonically increasing epochs so stale workers cannot
overwrite newer results: a result is accepted only when its lease epoch
equals the current epoch for the task.  Expired leases trigger reconciliation
before reassignment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..contracts import ResourceVector, TaskSpec, utc_now

__all__ = (
    "LeaseExpiredError",
    "LeaseManager",
    "StaleLeaseError",
    "TaskLease",
)


class StaleLeaseError(RuntimeError):
    """A worker presented a result with a stale fencing token (Step 15)."""


class LeaseExpiredError(RuntimeError):
    """The lease TTL elapsed before completion (Step 15)."""


@dataclass(frozen=True, slots=True)
class TaskLease:
    """One bounded task assignment with a fencing epoch (Step 12)."""

    lease_id: str
    workflow_id: str
    task_id: str
    worker_id: str
    epoch: int
    issued_at: str = field(default_factory=utc_now)
    ttl_seconds: float = 60.0
    permitted_tools: tuple[str, ...] = ()
    budget: ResourceVector = ResourceVector()
    input_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the lease (no payloads, only references)."""
        return {
            "lease_id": self.lease_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "epoch": self.epoch,
            "issued_at": self.issued_at,
            "ttl_seconds": self.ttl_seconds,
            "permitted_tools": list(self.permitted_tools),
            "budget": self.budget.to_dict(),
            "input_refs": list(self.input_refs),
        }


class LeaseManager:
    """Issue leases and enforce fencing on acceptance (Step 15)."""

    def __init__(self, default_ttl_seconds: float = 60.0,
                 clock: Callable[[], float] | None = None) -> None:
        """Track per-task epochs starting at zero."""
        self._default_ttl = default_ttl_seconds
        self._clock = clock
        self._epochs: dict[str, int] = {}
        self._leases: dict[str, TaskLease] = {}
        self._lease_times: dict[str, float] = {}

    def _now(self) -> float:
        """Return a monotonic timestamp (injectable for tests)."""
        if self._clock is not None:
            return float(self._clock())
        import time
        return time.monotonic()

    def current_epoch(self, task_id: str) -> int:
        """Return the current fencing epoch for a task."""
        return self._epochs.get(task_id, 0)

    def issue(self, workflow_id: str, task: TaskSpec, worker_id: str,
              *, ttl_seconds: float | None = None,
              permitted_tools: tuple[str, ...] = (),
              budget: ResourceVector | None = None,
              input_refs: tuple[str, ...] = ()) -> TaskLease:
        """Issue a new lease with a strictly increasing epoch (Step 12)."""
        epoch = self._epochs.get(task.task_id, 0) + 1
        self._epochs[task.task_id] = epoch
        lease = TaskLease(
            lease_id=f"lease-{workflow_id}-{task.task_id}-{epoch}",
            workflow_id=workflow_id, task_id=task.task_id, worker_id=worker_id,
            epoch=epoch, ttl_seconds=ttl_seconds if ttl_seconds is not None
            else self._default_ttl,
            permitted_tools=permitted_tools or task.permitted_tools,
            budget=budget or task.cost, input_refs=input_refs)
        self._leases[lease.lease_id] = lease
        self._lease_times[lease.lease_id] = self._now()
        return lease

    def validate(self, lease: TaskLease, *, enforce_ttl: bool = True) -> None:
        """Raise when the lease is stale or expired (Step 15)."""
        current = self._epochs.get(lease.task_id, 0)
        if lease.epoch != current:
            raise StaleLeaseError(
                f"lease epoch {lease.epoch} is stale; current epoch is {current} "
                f"for task {lease.task_id}")
        if enforce_ttl:
            elapsed = self._now() - self._lease_times.get(lease.lease_id, self._now())
            if elapsed > lease.ttl_seconds:
                raise LeaseExpiredError(
                    f"lease {lease.lease_id} expired before completion")

    def heartbeat(self, lease: TaskLease) -> bool:
        """Record a worker heartbeat; False when the lease is stale (Step 15)."""
        try:
            self.validate(lease, enforce_ttl=False)
        except StaleLeaseError:
            return False
        self._lease_times[lease.lease_id] = self._now()
        return True

    def accept(self, lease: TaskLease) -> None:
        """Accept a completed result, enforcing fencing (Step 15)."""
        self.validate(lease, enforce_ttl=False)

    def fence(self, task_id: str) -> int:
        """Invalidate all outstanding leases for a task, returning the new epoch."""
        epoch = self._epochs.get(task_id, 0) + 1
        self._epochs[task_id] = epoch
        return epoch

    def lease(self, lease_id: str) -> TaskLease | None:
        """Return a recorded lease by ID."""
        return self._leases.get(lease_id)

    def epochs(self) -> Mapping[str, int]:
        """Return the current epochs for checkpointing (Step 20)."""
        return dict(self._epochs)
