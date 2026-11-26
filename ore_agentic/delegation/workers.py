"""Worker contracts, local workers and bounded map-reduce execution (Steps 13-14).

Workers satisfy a typed protocol (``capabilities`` / ``execute``), receive a
bounded task envelope with least-privilege references, and return typed
results with provenance.  The map-reduce executor dispatches independent
tasks to a bounded pool with per-workflow, per-worker and global concurrency
limits and activates backpressure at queue capacity.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Mapping, Protocol

from ..contracts import (
    CancellationState,
    ResourceVector,
    TaskOutcomeStatus,
    TaskResult,
    TaskSpec,
    ProvenanceRecord,
)
from ..security.sandbox import SandboxProfile, SandboxedExecutor
from .leases import TaskLease

__all__ = (
    "BackpressureError",
    "BoundedWorkQueue",
    "LocalWorker",
    "MapOutcome",
    "MapReduceExecutor",
    "PermanentTaskError",
    "PolicyTaskError",
    "ResourceTaskError",
    "TaskEnvelope",
    "TransientTaskError",
    "Worker",
)


class TransientTaskError(RuntimeError):
    """A transient failure that may be retried with backoff (Step 16)."""


class PermanentTaskError(RuntimeError):
    """A permanent failure that must not be retried (Step 16)."""


class PolicyTaskError(RuntimeError):
    """An authorization/policy failure; never retried as transient (Step 16)."""


class ResourceTaskError(RuntimeError):
    """A resource-exhaustion failure routed to admission control (Step 16)."""


class BackpressureError(RuntimeError):
    """The bounded queue is at capacity; work is deferred or rejected (Step 14)."""


@dataclass(slots=True)
class TaskEnvelope:
    """The least-privilege execution envelope handed to a worker (Step 12)."""

    task: TaskSpec
    lease: TaskLease
    inputs: Mapping[str, Any]
    effects: Any = None  # SideEffectHandle, duck-typed to avoid import cycles
    cancellation: CancellationState = field(default_factory=CancellationState)
    deadline_seconds: float | None = None

    def check_cancelled(self) -> None:
        """Raise CancellationError when the workflow was cancelled (Step 22)."""
        if self.cancellation.cancelled:
            raise CancelledError()


class CancelledError(RuntimeError):
    """The task observed cancellation before completing (Step 22)."""


class Worker(Protocol):
    """The typed worker contract (specification Section 4)."""

    def capabilities(self) -> tuple[str, ...]: ...

    def execute(self, task_envelope: TaskEnvelope) -> TaskResult: ...


class LocalWorker:
    """A deterministic in-process worker wrapping a callable (reference impl)."""

    def __init__(self, worker_id: str, version: str, capabilities: tuple[str, ...],
                 handler: Callable[[TaskEnvelope], Mapping[str, Any]],
                 *, sandbox: SandboxProfile | None = None) -> None:
        """Wrap a handler mapping envelopes to result payloads."""
        self.worker_id = worker_id
        self.version = version
        self._capabilities = tuple(capabilities)
        self._handler = handler
        self._executor = SandboxedExecutor(sandbox)

    def capabilities(self) -> tuple[str, ...]:
        """Return the declared capabilities of this worker."""
        return self._capabilities

    def execute(self, task_envelope: TaskEnvelope) -> TaskResult:
        """Run the handler under the sandbox profile and wrap its payload."""
        outcome = self._executor.run(lambda: self._handler(task_envelope))
        payload = outcome.value if isinstance(outcome.value, Mapping) else {
            "value": outcome.value}
        evidence = payload.get("evidence", ()) if isinstance(payload, Mapping) else ()
        evidence_refs = tuple(str(item) for item in evidence) if isinstance(
            evidence, (list, tuple)) else ()
        return TaskResult(
            task_id=task_envelope.task.task_id, worker_id=self.worker_id,
            status=TaskOutcomeStatus.SUCCEEDED, payload=dict(payload),
            evidence_refs=evidence_refs,
            provenance=ProvenanceRecord(
                request_id=task_envelope.lease.lease_id, worker_id=self.worker_id,
                worker_version=self.version,
                versions={"handler": self.version}),
            resource_use=ResourceVector(seconds=outcome.duration_ms / 1000.0))


@dataclass(frozen=True, slots=True)
class MapOutcome:
    """Aggregated outcome of one map-reduce level (Step 13)."""

    results: tuple[TaskResult, ...]
    failures: tuple[TaskResult, ...]
    deferred: tuple[str, ...]
    peak_concurrency: int
    queue_wait_ms: dict[str, float]

    @property
    def succeeded_ids(self) -> tuple[str, ...]:
        """Return the IDs of tasks that succeeded at this level."""
        return tuple(result.task_id for result in self.results)


class BoundedWorkQueue:
    """A bounded in-memory queue with backpressure signalling (Step 14)."""

    def __init__(self, capacity: int) -> None:
        """Fix the queue capacity."""
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._items: list[tuple[str, float]] = []
        self._lock = threading.Lock()
        self.deferred_count = 0

    @property
    def capacity(self) -> int:
        """Return the queue capacity."""
        return self._capacity

    @property
    def depth(self) -> int:
        """Return the current queue depth."""
        with self._lock:
            return len(self._items)

    def submit(self, task_id: str) -> float:
        """Enqueue a task; raise BackpressureError at capacity (Step 14).

        Returns the enqueue timestamp used to measure queue wait.
        """
        now = time.monotonic()
        with self._lock:
            if len(self._items) >= self._capacity:
                self.deferred_count += 1
                raise BackpressureError(
                    f"queue at capacity {self._capacity}; task {task_id} deferred")
            self._items.append((task_id, now))
            return now

    def pop_wait(self, task_id: str, submitted_at: float) -> float:
        """Pop a task and return its queue wait in seconds (0.0 when absent)."""
        with self._lock:
            for index, (candidate, stamp) in enumerate(self._items):
                if candidate == task_id:
                    del self._items[index]
                    return max(0.0, time.monotonic() - stamp)
        return max(0.0, time.monotonic() - submitted_at)


class MapReduceExecutor:
    """Bounded parallel execution of independent tasks (Steps 13-14).

    Limits are enforced per workflow, per worker and globally; a shared
    observed-concurrency counter records the peak; a bounded queue defers
    overflow work instead of unbounded queuing.
    """

    def __init__(self, *, pool_size: int = 4, per_workflow_concurrency: int = 4,
                 per_worker_concurrency: int = 2, queue_capacity: int = 32) -> None:
        """Configure the executor bounds from the governed configuration."""
        for name, value in (("pool_size", pool_size),
                            ("per_workflow_concurrency", per_workflow_concurrency),
                            ("per_worker_concurrency", per_worker_concurrency),
                            ("queue_capacity", queue_capacity)):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        self._pool_size = pool_size
        self._per_workflow = per_workflow_concurrency
        self._per_worker = per_worker_concurrency
        self._queue = BoundedWorkQueue(queue_capacity)
        self._active = 0
        self._peak = 0
        self._lock = threading.Lock()
        self._per_worker_active: dict[str, int] = {}

    @property
    def peak_concurrency(self) -> int:
        """Return the maximum observed concurrent executions."""
        return self._peak

    @property
    def queue(self) -> BoundedWorkQueue:
        """Return the bounded work queue for metrics (Step 14)."""
        return self._queue

    def map(self, assignments: tuple[tuple[TaskSpec, TaskLease, Worker], ...],
            run_task: Callable[[TaskSpec, TaskLease, Worker], TaskResult]
            ) -> MapOutcome:
        """Execute one level of independent assignments in parallel (Step 13).

        ``run_task`` performs the actual attempt/retry logic; this method owns
        only bounded dispatch, concurrency accounting and partial-result
        preservation.  Assignments beyond the bounds are deferred, never lost.
        """
        results: list[TaskResult] = []
        failures: list[TaskResult] = []
        deferred: list[str] = []
        queue_waits: dict[str, float] = {}
        allowed: list[tuple[TaskSpec, TaskLease, Worker]] = []
        worker_slots = dict(self._per_worker_active)
        workflow_slots = self._per_workflow
        for assignment in assignments:
            task, lease, _worker = assignment
            worker_key = lease.worker_id
            if workflow_slots <= 0 or worker_slots.get(worker_key, 0) >= self._per_worker:
                deferred.append(task.task_id)
                continue
            try:
                self._queue.submit(task.task_id)
            except BackpressureError:
                deferred.append(task.task_id)
                continue
            workflow_slots -= 1
            worker_slots[worker_key] = worker_slots.get(worker_key, 0) + 1
            allowed.append(assignment)
        if not allowed:
            return MapOutcome(tuple(results), tuple(failures), tuple(deferred),
                              self._peak, queue_waits)

        def _run(assignment: tuple[TaskSpec, TaskLease, Worker]) -> TaskResult:
            task, lease, _ = assignment
            wait = self._queue.pop_wait(task.task_id, time.monotonic())
            with self._lock:
                self._active += 1
                self._peak = max(self._peak, self._active)
            try:
                return run_task(task, lease, _)
            finally:
                with self._lock:
                    self._active -= 1
                    self._per_worker_active[lease.worker_id] = max(
                        0, self._per_worker_active.get(lease.worker_id, 1) - 1)
                queue_waits[task.task_id] = wait * 1000.0

        with ThreadPoolExecutor(max_workers=self._pool_size,
                                thread_name_prefix="ore-agentic") as pool:
            futures = [pool.submit(_run, assignment) for assignment in allowed]
            for future in futures:
                result = future.result()
                if result.status is TaskOutcomeStatus.SUCCEEDED:
                    results.append(result)
                else:
                    failures.append(result)
        return MapOutcome(tuple(results), tuple(failures), tuple(deferred),
                          self._peak, queue_waits)
