"""Bounded worker lifecycle, cancellation, cooperative checkpoints, and dispatch."""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..contracts import (
    EngineConfig, TaskAttemptState, TaskCompletion, TaskSpec, WorkerSpec, utc_now,
)
from ..exceptions import TaskCancelledError
from .scheduler import WorkerQueueState


class CancellationToken:
    """Propagate cooperative cancellation through one task execution."""

    def __init__(self) -> None:
        """Initialize the cancellation event and reason slot."""
        self._event = threading.Event()
        self._reason: str | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        """Request cancellation and record the reason for diagnostics."""
        self._reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        """Return the recorded cancellation reason."""
        return self._reason

    def check(self) -> None:
        """Raise TaskCancelledError when cancellation is observed at a safe boundary."""
        if self._event.is_set():
            raise TaskCancelledError(
                f"cancellation observed at cooperative checkpoint: {self._reason}"
            )


class TaskContext:
    """Expose runtime services to one executing task body."""

    def __init__(
        self, task_spec: TaskSpec, snapshot_data: Mapping[str, Any],
        cancellation: CancellationToken, checkpoint_store: "CheckpointStore",
        correlation_id: str, run_id: str,
    ) -> None:
        """Bind the immutable read view, cancellation, and checkpoint services."""
        import copy as _copy

        self.task_spec = task_spec
        self.snapshot = _copy.deepcopy(dict(snapshot_data))
        self.cancellation = cancellation
        self.checkpoints = checkpoint_store
        self.correlation_id = correlation_id
        self.run_id = run_id

    def read(self, key: str, default: Any = None) -> Any:
        """Read one input from the immutable snapshot the task was dispatched with."""
        return self.snapshot.get(key, default)

    def check_cancellation(self) -> None:
        """Cooperative checkpoint that long-running modules call at safe boundaries."""
        self.cancellation.check()

    def save_checkpoint(self, progress: Mapping[str, Any]) -> None:
        """Persist resumable progress for crash recovery."""
        self.checkpoints.save(self.task_spec.task_id, progress)


class CheckpointStore:
    """Persist resumable task progress for crash recovery and replay."""

    def __init__(self) -> None:
        """Initialize the thread-safe in-memory checkpoint table."""
        self._lock = threading.RLock()
        self._checkpoints: dict[str, dict[str, Any]] = {}

    def save(self, task_id: str, progress: Mapping[str, Any]) -> None:
        """Save the latest progress record for one task."""
        with self._lock:
            self._checkpoints[task_id] = {
                "task_id": task_id,
                "progress": dict(progress),
                "saved_at": utc_now(),
            }

    def load(self, task_id: str) -> Mapping[str, Any] | None:
        """Return the latest saved progress mapping for one task."""
        with self._lock:
            record = self._checkpoints.get(task_id)
            return dict(record["progress"]) if record else None

    def latest(self, task_id: str) -> dict[str, Any] | None:
        """Return the full checkpoint record with its timestamp."""
        with self._lock:
            record = self._checkpoints.get(task_id)
            return dict(record) if record else None

    def forget(self, task_id: str) -> None:
        """Drop checkpoints for tasks that reached a terminal state."""
        with self._lock:
            self._checkpoints.pop(task_id, None)

    def snapshot_all(self) -> dict[str, dict[str, Any]]:
        """Return every checkpoint for recovery planning."""
        with self._lock:
            return {key: dict(record) for key, record in self._checkpoints.items()}


@dataclass(slots=True)
class PendingExecution:
    """Track one dispatched task awaiting completion in live or manual mode."""

    task_id: str
    worker_id: str
    attempt: int
    read_version: int
    started_at: float
    started_at_iso: str
    cancellation: CancellationToken
    timeout_seconds: float
    payload: Mapping[str, Any]
    future: Future | None = None
    recovered_from: str | None = None


class BoundedWorkerPool:
    """Run tasks on capability-matched workers with bounded active-task limits.

    Task claiming and completion bookkeeping are single lock-protected
    transitions, which are the linearization points of the pool: a task can
    never be claimed by two workers or completed twice.
    """

    def __init__(
        self, worker_specs: Sequence[WorkerSpec], config: EngineConfig,
        *, checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        """Create the bounded pool with one local queue per declared worker."""
        if not worker_specs:
            raise ValueError("worker_specs must not be empty.")
        self._config = config
        self._lock = threading.RLock()
        self._checkpoint_store = checkpoint_store or CheckpointStore()
        self._states: dict[str, WorkerQueueState] = {}
        for spec in worker_specs:
            if spec.worker_id in self._states:
                raise ValueError(f"duplicate worker id '{spec.worker_id}'.")
            self._states[spec.worker_id] = WorkerQueueState(
                worker_id=spec.worker_id, capabilities=frozenset(spec.capabilities),
            )
        self._pending: dict[str, PendingExecution] = {}
        self._cancellations: dict[str, CancellationToken] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False
        if config.execution_mode == "live":
            self._executor = ThreadPoolExecutor(
                max_workers=config.max_workers,
                thread_name_prefix="ore-runtime-worker",
            )

    # ------------------------------------------------------------------ access

    @property
    def checkpoint_store(self) -> CheckpointStore:
        """Return the shared checkpoint store for crash recovery."""
        return self._checkpoint_store

    @property
    def closed(self) -> bool:
        """Return whether the pool has been shut down."""
        return self._closed

    def worker_states(self) -> tuple[WorkerQueueState, ...]:
        """Return the live queue state of every worker."""
        with self._lock:
            return tuple(self._states.values())

    def worker_capabilities(self) -> dict[str, frozenset[str]]:
        """Return each worker's capability set for plan validation."""
        with self._lock:
            return {wid: state.capabilities for wid, state in self._states.items()}

    def has_capability(self, capability: str) -> bool:
        """Return whether any worker can ever execute the given capability."""
        with self._lock:
            return any(capability in state.capabilities for state in self._states.values())

    def capable_workers(self, capability: str) -> tuple[str, ...]:
        """Return every worker identifier offering the given capability."""
        with self._lock:
            return tuple(
                wid for wid, state in sorted(self._states.items())
                if capability in state.capabilities
            )

    def active_task_count(self) -> int:
        """Return the number of currently dispatched but uncompleted tasks."""
        with self._lock:
            return len(self._pending)

    def pending_task_ids(self) -> tuple[str, ...]:
        """Return dispatched task identifiers awaiting completion."""
        with self._lock:
            return tuple(sorted(self._pending))

    def pending_of(self, task_id: str) -> PendingExecution | None:
        """Return one task's pending execution record for manual stepping."""
        with self._lock:
            return self._pending.get(task_id)

    def utilization(self) -> dict[str, dict[str, float]]:
        """Report per-worker active load and completed counts for metrics."""
        with self._lock:
            return {
                wid: {
                    "active": float(len(state.active_task_ids)),
                    "queued": float(len(state.local_deque)),
                    "completed": float(state.completed_count),
                    "estimated_load": state.estimated_load,
                }
                for wid, state in sorted(self._states.items())
            }

    def load_imbalance(self) -> float:
        """Return max load over mean load with a balanced-by-convention guard."""
        with self._lock:
            loads = [state.estimated_load for state in self._states.values()] or [0.0]
            mean = sum(loads) / len(loads)
            if mean <= 0.0:
                return 1.0
            return max(loads) / (mean + 1e-6)

    # ---------------------------------------------------------------- dispatch

    def _wrap_body(
        self, task_spec: TaskSpec, body: Callable[[TaskContext], Mapping[str, Any]],
        context: TaskContext,
    ) -> Callable[[], Mapping[str, Any]]:
        """Wrap a task body so cooperative checkpoints and context stay attached."""
        def run() -> Mapping[str, Any]:
            """Execute the task body through the shared task context."""
            return body(context)
        return run

    def claim_task(self, task_id: str) -> bool:
        """Atomically register a claim so one task is dispatched at most once."""
        with self._lock:
            if task_id in self._pending:
                return False
            return True

    def dispatch(
        self, task_spec: TaskSpec, worker_id: str, snapshot: Mapping[str, Any],
        body: Callable[[TaskContext], Mapping[str, Any]], *, attempt: int = 1,
        read_version: int = 0, timeout_seconds: float | None = None,
        correlation_id: str = "", run_id: str = "",
    ) -> PendingExecution:
        """Dispatch one task to a worker with immutable inputs and a cancellation token."""
        with self._lock:
            if self._closed:
                raise RuntimeError("worker pool is closed.")
            state = self._states.get(worker_id)
            if state is None:
                raise KeyError(f"unknown worker '{worker_id}'.")
            if task_spec.capability not in state.capabilities:
                raise ValueError(
                    f"worker '{worker_id}' lacks capability '{task_spec.capability}'."
                )
            if task_spec.task_id in self._pending:
                raise ValueError(f"task '{task_spec.task_id}' is already dispatched.")
            if len(self._pending) >= self._config.max_active_tasks:
                raise ValueError(
                    f"active task budget reached ({self._config.max_active_tasks}); "
                    "dispatch deferred by admission control"
                )
            cancellation = CancellationToken()
            context = TaskContext(
                task_spec=task_spec,
                snapshot_data=snapshot,
                cancellation=cancellation,
                checkpoint_store=self._checkpoint_store,
                correlation_id=correlation_id,
                run_id=run_id,
            )
            pending = PendingExecution(
                task_id=task_spec.task_id, worker_id=worker_id, attempt=attempt,
                read_version=read_version, started_at=time.monotonic(),
                started_at_iso=utc_now(), cancellation=cancellation,
                timeout_seconds=(
                    timeout_seconds
                    if timeout_seconds is not None
                    else self._config.default_task_timeout_seconds
                ),
                payload=task_spec.payload,
            )
            wrapped = self._wrap_body(task_spec, body, context)
            if self._executor is not None:
                pending.future = self._executor.submit(wrapped)
            self._pending[task_spec.task_id] = pending
            self._cancellations[task_spec.task_id] = cancellation
            return pending

    # --------------------------------------------------------------- completion

    def collect_completions(
        self, *, timeout: float | None = None,
        failure_hook: Callable[[PendingExecution], Mapping[str, Any] | None] | None = None,
    ) -> tuple[TaskCompletion, ...]:
        """Collect finished live-mode executions and convert them into completions.

        Waits up to ``timeout`` seconds for at least one completion so fast
        task bodies finish within the same engine round; interrupted by the
        timeout, the remaining pending executions complete in later rounds.
        """
        if self._executor is None:
            return ()
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            completions = self._collect_finished(deadline)
            if completions:
                return tuple(completions)
            if not self._pending:
                return ()
            if deadline is not None and time.monotonic() >= deadline:
                return ()
            time.sleep(0.002)

    def _collect_finished(
        self, deadline: float | None,
    ) -> list[TaskCompletion]:
        """Collect every currently finished execution into completions."""
        completions: list[TaskCompletion] = []
        for task_id in self.pending_task_ids():
            pending = self._pending.get(task_id)
            if pending is None or pending.future is None:
                continue
            elapsed = time.monotonic() - pending.started_at
            timed_out = elapsed > pending.timeout_seconds
            if not pending.future.done():
                if pending.cancellation.cancelled or timed_out:
                    pending.future.cancel()
                    outcome = (
                        TaskAttemptState.CANCELLED if pending.cancellation.cancelled
                        else TaskAttemptState.TIMED_OUT
                    )
                    self._finalize(pending, outcome)
                    completions.append(self._completion_for(
                        pending, outcome,
                        error=pending.cancellation.reason or "task timeout exceeded",
                    ))
                continue
            try:
                output = pending.future.result(timeout=None)
            except TaskCancelledError as exc:
                self._finalize(pending, TaskAttemptState.CANCELLED)
                completions.append(self._completion_for(
                    pending, TaskAttemptState.CANCELLED, error=str(exc),
                ))
                continue
            except Exception as exc:  # noqa: BLE001 - worker isolation per spec
                self._finalize(pending, TaskAttemptState.FAILED)
                completions.append(self._completion_for(
                    pending, TaskAttemptState.FAILED, error=f"{type(exc).__name__}: {exc}",
                ))
                continue
            self._finalize(pending, TaskAttemptState.SUCCEEDED)
            completions.append(self._completion_for(pending, TaskAttemptState.SUCCEEDED,
                                                    output=output or {}))
        return completions

    def complete_manual(self, task_id: str, outcome: TaskAttemptState | str,
                        output: Mapping[str, Any] | None = None,
                        error: str | None = None) -> TaskCompletion:
        """Complete one manual-mode execution, optionally injecting a fault outcome."""
        outcome = TaskAttemptState(outcome)
        with self._lock:
            pending = self._pending.get(task_id)
            if pending is None:
                raise KeyError(f"task '{task_id}' is not dispatched.")
            if outcome is not TaskAttemptState.SUCCEEDED and error is None:
                error = f"injected {outcome.value} outcome"
            if outcome is TaskAttemptState.SUCCEEDED:
                self._finalize(pending, TaskAttemptState.SUCCEEDED)
            else:
                self._finalize(pending, outcome)
            return self._completion_for(
                pending, outcome, output=output or {}, error=error,
            )

    def _completion_for(
        self, pending: PendingExecution, outcome: TaskAttemptState,
        *, output: Mapping[str, Any] | None = None, error: str | None = None,
    ) -> TaskCompletion:
        """Build the completion record for one finished pending execution."""
        return TaskCompletion(
            task_id=pending.task_id,
            worker_id=pending.worker_id,
            attempt=pending.attempt,
            outcome=outcome,
            output=dict(output or {}),
            error=error,
            read_version=pending.read_version,
            elapsed_seconds=max(0.0, time.monotonic() - pending.started_at),
            provenance={
                "correlation_id": pending.payload.get("correlation_id", ""),
                "started_at": pending.started_at_iso,
            },
        )

    def _finalize(self, pending: PendingExecution, outcome: TaskAttemptState) -> None:
        """Retire one pending execution and update worker bookkeeping."""
        with self._lock:
            self._pending.pop(pending.task_id, None)
            self._cancellations.pop(pending.task_id, None)
            state = self._states.get(pending.worker_id)
            if state is not None:
                state.active_task_ids.discard(pending.task_id)
                state.local_deque = deque(
                    item for item in state.local_deque if item != pending.task_id
                )
                if outcome is TaskAttemptState.SUCCEEDED:
                    state.completed_count += 1

    # -------------------------------------------------------------- steering

    def cancel(self, task_id: str, reason: str = "cancelled") -> bool:
        """Request cooperative cancellation of one dispatched task."""
        with self._lock:
            token = self._cancellations.get(task_id)
            if token is None:
                return False
            token.cancel(reason)
            return True

    def cancel_all(self, reason: str = "goal cancelled") -> tuple[str, ...]:
        """Propagate cancellation to every dispatched task and return their IDs."""
        with self._lock:
            task_ids = tuple(self._cancellations)
        for task_id in task_ids:
            self.cancel(task_id, reason)
        return task_ids

    def timeout_expired(self, task_id: str) -> bool:
        """Check whether one dispatched task exceeded its timeout budget."""
        with self._lock:
            pending = self._pending.get(task_id)
            if pending is None:
                return False
            return (time.monotonic() - pending.started_at) > pending.timeout_seconds

    # -------------------------------------------------------------- recovery

    def recover_pending(self) -> dict[str, Mapping[str, Any]]:
        """Return pending checkpoints so a restarted engine can resume or retry safely."""
        return self._checkpoint_store.snapshot_all()

    def abandon_all(self, reason: str = "run abandoned") -> tuple[str, ...]:
        """Drop every pending execution when a run is abandoned by its driver."""
        with self._lock:
            abandoned = tuple(sorted(self._pending))
            for task_id in abandoned:
                pending = self._pending.pop(task_id, None)
                self._cancellations.pop(task_id, None)
                if pending is None:
                    continue
                pending.cancellation.cancel(reason)
                state = self._states.get(pending.worker_id)
                if state is not None:
                    state.active_task_ids.discard(task_id)
                    state.local_deque = deque(
                        item for item in state.local_deque if item != task_id
                    )
            return abandoned

    # ------------------------------------------------------------------- close

    def close(self, *, wait: bool = True) -> None:
        """Shut down the executor while leaving completed bookkeeping intact."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> "BoundedWorkerPool":
        """Return the pool itself for context-manager use."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the pool on context exit."""
        self.close()
