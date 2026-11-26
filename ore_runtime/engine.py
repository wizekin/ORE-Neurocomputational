"""End-to-end concurrent and incremental cognitive execution engine (Point 19).

Implements the specification's EXECUTE_GOAL pseudocode: normalize a goal into
a typed request, compile it into a validated task DAG with capability and
side-effect policies, capture a versioned workspace snapshot, then loop over
event draining, idempotent workflow updates, admission-controlled dispatch,
result validation with optimistic commit plus transactional outbox, dependency
index updates, incremental invalidation, and deadline/cancellation health
checks, before validating final outputs and the causal audit trace.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .adapters import WorkspaceRuntimeAdapter
from .concurrency.scheduler import AdmissionController, Scheduler
from .concurrency.task_graph import IterativeRegionSpec, TaskGraph, TaskNode
from .concurrency.versioned_state import Snapshot, VersionedStore
from .concurrency.worker_pool import BoundedWorkerPool, TaskContext
from .contracts import (
    EngineConfig, ExecutionRequest, RetrySafety, SideEffectClass, TaskCompletion,
    TaskSpec, TaskState, WorkerSpec, new_id, utc_now,
)
from .events.bus import EventBus
from .events.idempotency import IdempotencyStore
from .events.outbox import OutboxStatus, TransactionalOutbox
from .events.sagas import SagaCoordinator, SagaStep
from .events.schemas import (
    ARTIFACT_RECOMPUTED, ARTIFACT_STALE, FINDING_PUBLISHED, GOAL_COMPLETED,
    GOAL_STARTED, TASK_CANCELLED, TASK_DISPATCHED, TASK_FAILED, TASK_RETRIED,
    TASK_SUCCEEDED, USG_RECORD_CHANGED, WORKSPACE_COMMITTED, EventEnvelope,
    SCHEMA_VERSION,
)
from .exceptions import CapabilityUnavailableError, PlanValidationError
from .history.event_store import EventStore
from .incremental.dependency_graph import (
    DependencyGraph, DependencyRef, SourceRecord,
)
from .incremental.invalidation import InvalidationEngine
from .metrics import RuntimeMetrics

TaskBody = Callable[[TaskContext], Mapping[str, Any]]


@dataclass(slots=True)
class ExecutionRun:
    """Hold one prepared execution's live state across engine steps."""

    run_id: str
    request: ExecutionRequest
    graph: TaskGraph
    executors: Mapping[str, TaskBody]
    correlation_id: str
    read_snapshot: Snapshot
    rounds: int = 0
    status: str = "running"
    error: str | None = None
    committed_revisions: list[int] = field(default_factory=list)
    task_outputs: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    stale_artifacts: list[str] = field(default_factory=list)
    recomputed_artifacts: list[str] = field(default_factory=list)
    events_emitted: int = 0
    cancelled: bool = False
    cancel_reason: str | None = None

    def record(self, kind: str, **payload: Any) -> None:
        """Append one structured trace record for the audit trail."""
        self.trace.append({"round": self.rounds, "kind": kind, **payload})


class CognitiveExecutionEngine:
    """Coordinate ORE cognitive work as one safe concurrent execution system."""

    def __init__(
        self, config: EngineConfig | None = None, *,
        worker_specs: Sequence[WorkerSpec] | None = None,
        workspace_adapter: WorkspaceRuntimeAdapter | None = None,
        initial_state: Mapping[str, Any] | None = None,
        idempotency: IdempotencyStore | None = None,
    ) -> None:
        """Assemble every runtime subsystem with validated bounded configuration."""
        self._config = config or EngineConfig()
        self._store = VersionedStore(
            dict(initial_state or {}),
            retained_revision_limit=self._config.retained_revision_limit,
        )
        self._workers = tuple(worker_specs or (
            WorkerSpec(worker_id=f"worker-{index}", capabilities=frozenset({"generic"}))
            for index in range(self._config.max_workers)
        ))
        self._pool = BoundedWorkerPool(self._workers, self._config)
        self._scheduler = Scheduler(self._config)
        self._admission = AdmissionController(
            high_watermark=self._config.queue_high_watermark,
            low_watermark=self._config.queue_low_watermark,
        )
        self._idempotency = idempotency or IdempotencyStore()
        self._bus = EventBus(idempotency=self._idempotency)
        self._outbox = TransactionalOutbox(
            max_attempts=self._config.outbox_max_attempts,
            backoff_seconds=self._config.outbox_retry_backoff_seconds,
        )
        self._event_store = EventStore()
        self._dependencies = DependencyGraph()
        self._invalidation = InvalidationEngine(self._dependencies)
        self._sagas = SagaCoordinator(
            on_compensation=self._record_saga_compensation,
        )
        self._metrics = RuntimeMetrics()
        self._workspace = workspace_adapter
        self._aggregate_versions: dict[str, int] = {}
        self._current_run: ExecutionRun | None = None
        self._run_lock = threading.RLock()
        self._artifact_executors: dict[str, TaskBody] = {}
        self._closed = False
        self._subscribe_runtime_consumers()

    # ------------------------------------------------------------ subsystems

    @property
    def config(self) -> EngineConfig:
        """Return the validated engine configuration."""
        return self._config

    @property
    def store(self) -> VersionedStore:
        """Return the authoritative versioned state store."""
        return self._store

    @property
    def graph_dependencies(self) -> DependencyGraph:
        """Return the artifact dependency graph with reverse indexes."""
        return self._dependencies

    @property
    def event_store(self) -> EventStore:
        """Return the durable event history."""
        return self._event_store

    @property
    def bus(self) -> EventBus:
        """Return the local event bus."""
        return self._bus

    @property
    def outbox(self) -> TransactionalOutbox:
        """Return the transactional outbox."""
        return self._outbox

    @property
    def pool(self) -> BoundedWorkerPool:
        """Return the bounded worker pool."""
        return self._pool

    @property
    def scheduler(self) -> Scheduler:
        """Return the scheduling policy component."""
        return self._scheduler

    @property
    def invalidation(self) -> InvalidationEngine:
        """Return the incremental invalidation engine."""
        return self._invalidation

    @property
    def sagas(self) -> SagaCoordinator:
        """Return the saga coordinator for multi-step side effects."""
        return self._sagas

    @property
    def workspace_adapter(self) -> WorkspaceRuntimeAdapter | None:
        """Return the optional shared-workspace integration adapter."""
        return self._workspace

    # ---------------------------------------------------------- consumers

    def _subscribe_runtime_consumers(self) -> None:
        """Subscribe the engine's own durable consumers on the local bus."""
        self._bus.subscribe(
            "ore_runtime.history",
            self._history_consumer,
            delivery_guarantee="at_least_once",
        )
        self._bus.subscribe(
            "ore_runtime.workflow",
            self._workflow_consumer,
            event_types=(
                TASK_SUCCEEDED, TASK_FAILED, TASK_RETRIED, TASK_CANCELLED,
            ),
            delivery_guarantee="effectively_once",
        )
        self._bus.subscribe(
            "ore_runtime.incremental",
            self._invalidation_consumer,
            event_types=(USG_RECORD_CHANGED,),
            delivery_guarantee="effectively_once",
        )

    def _history_consumer(self, envelope: EventEnvelope) -> None:
        """Append every delivered event to the durable event log."""
        self._event_store.append(envelope)

    def _workflow_consumer(self, envelope: EventEnvelope) -> None:
        """Apply committed task events to the workflow state idempotently."""
        run = self._current_run
        if run is None:
            return
        payload = envelope.payload
        task_id = str(payload.get("task_id", ""))
        node = run.graph._nodes.get(task_id) if task_id else None
        if node is None:
            return
        if envelope.event_type == TASK_SUCCEEDED:
            node.state = TaskState.SUCCEEDED
            node.commit_version = int(payload.get("committed_revision", 0) or 0)
            output = run.task_outputs.get(task_id)
            if output is not None:
                node.output = dict(output)
            self._update_dependency_indexes(run, node)
        elif envelope.event_type == TASK_FAILED:
            node.state = TaskState.FAILED
            node.last_error = str(payload.get("error", ""))
        elif envelope.event_type == TASK_RETRIED:
            if node.state not in {TaskState.RUNNING}:
                node.state = TaskState.WAITING
        elif envelope.event_type == TASK_CANCELLED:
            node.state = TaskState.CANCELLED
            node.last_error = str(payload.get("reason", ""))

    def _invalidation_consumer(self, envelope: EventEnvelope) -> None:
        """Invalidate downstream artifacts when a USG source record changes."""
        record_id = str(envelope.payload.get("record_id", ""))
        if not record_id:
            return
        self._invalidation.invalidate([record_id])
        self._metrics.record_stale_artifacts(
            len(self._invalidation.stale_artifacts())
        )

    def _record_saga_compensation(self, saga_id: str, step_name: str) -> None:
        """Record one compensation audit event on the bus."""
        envelope = self._make_envelope(
            "saga.compensated", aggregate_id=f"saga::{saga_id}",
            payload={"saga_id": saga_id, "step_name": step_name},
        )
        self._outbox.record(envelope)
        run = self._current_run
        if run is not None:
            run.events_emitted += 1

    # ------------------------------------------------------- event helpers

    def _next_aggregate_version(self, aggregate_id: str) -> int:
        """Allocate the next per-aggregate version at envelope creation time.

        The counter mirrors the event store's expected version for the
        aggregate: the first envelope for an aggregate carries version 0 and
        the store advances its expectation after each durable append.
        """
        current = self._aggregate_versions.get(
            aggregate_id, self._event_store.latest_aggregate_version(aggregate_id),
        )
        self._aggregate_versions[aggregate_id] = current + 1
        return current

    def _make_envelope(
        self, event_type: str, aggregate_id: str, payload: Mapping[str, Any], *,
        causation_id: str | None = None, correlation_id: str | None = None,
        producer: str = "ore_runtime.engine",
    ) -> EventEnvelope:
        """Create one schema-validated envelope with causal identifiers."""
        envelope = EventEnvelope.create(
            event_type=event_type,
            aggregate_id=aggregate_id,
            aggregate_version=self._next_aggregate_version(aggregate_id),
            payload=payload,
            producer=producer,
            causation_id=causation_id,
            correlation_id=correlation_id,
        )
        return envelope

    # ------------------------------------------------------------- planning

    def normalize_goal(
        self, goal: str | ExecutionRequest, *,
        correlation_id: str | None = None,
    ) -> ExecutionRequest:
        """Normalize a goal description or request into a validated typed request."""
        if isinstance(goal, ExecutionRequest):
            return goal
        if not isinstance(goal, str) or not goal.strip():
            raise PlanValidationError("goal must be a non-empty description or request")
        return ExecutionRequest(
            goal_id=new_id("goal"),
            description=goal.strip(),
            inputs={},
            correlation_id=correlation_id or "",
        )

    def build_plan(
        self, request: ExecutionRequest, task_specs: Sequence[TaskSpec], *,
        iterative_regions: Sequence[IterativeRegionSpec] = (),
    ) -> TaskGraph:
        """Compile a request and manifest into a validated concurrent plan."""
        from .contracts import validate_task_specs
        specs = validate_task_specs(task_specs)
        graph = TaskGraph(specs)
        graph.validate_plan()
        for region in iterative_regions:
            graph.declare_iterative_region(region)
        for spec in specs:
            if not self._pool.has_capability(spec.capability):
                raise CapabilityUnavailableError(
                    f"no worker offers capability '{spec.capability}' required by "
                    f"task '{spec.task_id}'",
                    diagnostic_path=(spec.task_id,),
                )
        return graph

    # ------------------------------------------------------------ execution

    def start(
        self, goal: str | ExecutionRequest, task_specs: Sequence[TaskSpec],
        executors: Mapping[str, TaskBody], *,
        iterative_regions: Sequence[IterativeRegionSpec] = (),
        correlation_id: str | None = None,
    ) -> ExecutionRun:
        """Prepare an execution run through planning phases A and snapshot capture."""
        with self._run_lock:
            if self._closed:
                raise RuntimeError("engine is closed.")
            request = self.normalize_goal(goal, correlation_id=correlation_id)
            graph = self.build_plan(
                request, task_specs, iterative_regions=iterative_regions,
            )
            read_snapshot = self._store.capture()
            run = ExecutionRun(
                run_id=new_id("run"),
                request=request,
                graph=graph,
                executors=dict(executors),
                correlation_id=request.correlation_id,
                read_snapshot=read_snapshot,
            )
            for node in graph.nodes:
                node.enqueued_at = time.monotonic()
            self._emit_plan_events(run)
            self._current_run = run
            return run

    def execute_goal(
        self, goal: str | ExecutionRequest, task_specs: Sequence[TaskSpec],
        executors: Mapping[str, TaskBody], *,
        iterative_regions: Sequence[IterativeRegionSpec] = (),
        correlation_id: str | None = None,
    ) -> "GoalExecutionReport":
        """Execute one goal end-to-end and return the audited report."""
        run = self.start(
            goal, task_specs, executors,
            iterative_regions=iterative_regions, correlation_id=correlation_id,
        )
        return self.run_loop(run)

    def run_loop(self, run: ExecutionRun) -> "GoalExecutionReport":
        """Drive the live execution loop until the plan reaches a terminal state."""
        deadline_rounds = self._config.max_rounds
        while not run.graph.is_complete() and run.rounds < deadline_rounds:
            run.rounds += 1
            self.drain_events(run)
            completions = self._pool.collect_completions(timeout=0.05)
            for completion in completions:
                self._process_completion(run, completion)
            if completions:
                # Apply freshly committed events before readiness and health checks
                # so dependents never observe provisional inputs.
                self.drain_events(run)
            self._dispatch_ready(run)
            self._check_health(run)
            if run.rounds % self._config.checkpoint_interval_rounds == 0:
                self._persist_checkpoint(run)
            if self._no_progress_possible(run):
                self._break_deadlock(run)
                break
        self._run_iterative_regions(run)
        if not run.graph.is_complete():
            run.status = "incomplete"
            run.error = (
                f"execution stopped after {run.rounds} rounds with unfinished tasks"
            )
        self.drain_events(run)
        return self.finalize(run)

    def _no_progress_possible(self, run: ExecutionRun) -> bool:
        """Detect a plan that cannot progress: nothing pending and nothing ready."""
        if self._pool.active_task_count() > 0:
            return False
        return not run.graph.ready_tasks() and not run.graph.is_complete()

    def _break_deadlock(self, run: ExecutionRun) -> None:
        """Mark blocked tasks cancelled with a diagnostic reason."""
        for node in run.graph.nodes:
            if node.unfinished and node.state in {
                TaskState.CREATED, TaskState.READY, TaskState.WAITING,
            }:
                blocked = [
                    dep for dep in node.spec.dependencies
                    if run.graph.node(dep).state is not TaskState.SUCCEEDED
                ]
                self._cancel_node(run, node, reason=f"blocked by unfinished deps: {blocked}")
        run.status = "blocked"
        run.error = "no ready tasks can proceed; plan is blocked"

    # ---------------------------------------------------------- event drain

    def drain_events(self, run: ExecutionRun) -> int:
        """Drain committed outbox rows into the bus and durable history."""
        delivered = self._drain_all()
        if delivered:
            run.record("events_drained", count=delivered)
        run.events_emitted += delivered
        return delivered

    def _drain_all(self) -> int:
        """Publish every pending outbox row through the bus in FIFO order."""
        before = self._outbox.stats()["delivered"]
        self._outbox.drain(lambda envelope: self._bus.publish(envelope))
        return self._outbox.stats()["delivered"] - before

    def _emit_plan_events(self, run: ExecutionRun) -> None:
        """Record the goal and task creation events for the audit history."""
        goal_envelope = self._make_envelope(
            GOAL_STARTED, aggregate_id=f"goal::{run.request.goal_id}",
            payload={"goal_id": run.request.goal_id, "description": run.request.description},
            correlation_id=run.correlation_id,
        )
        self._outbox.record(goal_envelope)
        run.events_emitted += 1
        for node in run.graph.nodes:
            envelope = self._make_envelope(
                "task.created", aggregate_id=f"task::{node.task_id}",
                payload={
                    "task_id": node.task_id, "name": node.spec.name,
                    "capability": node.spec.capability,
                },
                correlation_id=run.correlation_id,
            )
            self._outbox.record(envelope)
            run.events_emitted += 1

    # ----------------------------------------------------------- dispatching

    def _executor_for(self, run: ExecutionRun, node: TaskNode) -> TaskBody | None:
        """Resolve a task's executor by id, capability, or declared executor id."""
        candidates = (
            node.task_id, node.spec.capability,
            str(node.spec.metadata.get("executor_id", "")),
        )
        for key in candidates:
            if key and key in run.executors:
                return run.executors[key]
        return None

    def _dispatch_ready(self, run: ExecutionRun) -> tuple[str, ...]:
        """Promote ready tasks and dispatch them under admission control.

        Bounded execution is enforced by the active-task ceiling and the
        admission watermarks. Backpressure defers dispatch when the ready
        backlog is above the high watermark, but an idle pool always admits
        at least one task so bursts degrade throughput without deadlocking.
        """
        run.graph.mark_ready()
        now = time.monotonic()
        ready = run.graph.ready_tasks()
        if not ready:
            return ()
        dispatched: list[str] = []
        assignments = self._scheduler.plan_dispatch(
            run.graph, self._pool.worker_states(), now,
        )
        assignment_by_task = {choice.task_id: choice for choice in assignments}
        idle_bypass = self._pool.active_task_count() == 0
        for node in ready:
            if node.task_id not in assignment_by_task:
                continue
            if self._pool.active_task_count() >= self._config.max_active_tasks:
                run.record("dispatch_deferred", task_id=node.task_id,
                           reason="active task budget reached")
                self._metrics.record_admission_rejection()
                break
            if not idle_bypass:
                outcome = self._admission.decide(len(ready))
                if outcome.decision.value != "admitted":
                    self._metrics.record_admission_rejection()
                    run.record("admission_deferred", task_id=node.task_id,
                               reason=outcome.reason)
                    continue
            executor = self._executor_for(run, node)
            if executor is None:
                self._complete_failed(
                    run, node, error="no executor registered for task",
                )
                continue
            choice = assignment_by_task[node.task_id]
            snapshot = self._store.latest()
            self._pool.dispatch(
                node.spec, choice.worker_id, snapshot.data, executor,
                attempt=node.attempts + 1,
                read_version=snapshot.revision,
                timeout_seconds=(
                    float(node.spec.metadata.get("timeout_seconds",
                                                 self._config.default_task_timeout_seconds))
                ),
                correlation_id=run.correlation_id,
                run_id=run.run_id,
            )
            node.state = TaskState.RUNNING
            node.attempts += 1
            node.assigned_worker = choice.worker_id
            node.read_version = snapshot.revision
            if node.enqueued_at is not None:
                self._metrics.record_queue_wait(now - node.enqueued_at)
                self._scheduler.record_queue_wait(now - node.enqueued_at)
            self._scheduler.record_dispatch(choice)
            envelope = self._make_envelope(
                TASK_DISPATCHED, aggregate_id=f"task::{node.task_id}",
                payload={
                    "task_id": node.task_id, "worker_id": choice.worker_id,
                    "attempt": node.attempts,
                },
                correlation_id=run.correlation_id,
            )
            self._outbox.record(envelope)
            run.events_emitted += 1
            run.record("dispatched", task_id=node.task_id, worker_id=choice.worker_id,
                       attempt=node.attempts, read_version=snapshot.revision)
            dispatched.append(node.task_id)
            idle_bypass = False
        return tuple(dispatched)

    # --------------------------------------------------------- completions

    def _process_completion(self, run: ExecutionRun, completion: TaskCompletion) -> None:
        """Validate, commit, or retry one task completion per its policy class."""
        node = run.graph.node(completion.task_id)
        node.assigned_worker = completion.worker_id
        run.record(
            "completion", task_id=completion.task_id,
            outcome=completion.outcome.value, attempt=completion.attempt,
        )
        if completion.succeeded:
            self._commit_success(run, node, completion)
        elif completion.outcome.value == "cancelled":
            self._complete_cancelled(run, node, reason=completion.error or "cancelled")
        else:
            self._apply_retry_or_compensation(run, node, completion)

    def _validate_output(self, output: Mapping[str, Any]) -> tuple[str, ...]:
        """Validate one task output mapping is finite and JSON-serializable."""
        errors: list[str] = []
        if not isinstance(output, Mapping):
            return ("output must be a mapping",)
        for key, value in output.items():
            if not isinstance(key, str):
                errors.append(f"output keys must be strings, got {type(key).__name__}")
                break
        from .contracts import canonical_json
        try:
            canonical_json(dict(output))
        except (TypeError, ValueError) as exc:
            errors.append(f"output must be JSON-serializable: {exc}")
        return tuple(errors)

    def _commit_success(
        self, run: ExecutionRun, node: TaskNode, completion: TaskCompletion,
    ) -> None:
        """Validate a result and commit it with its outbox events atomically."""
        output = dict(completion.output)
        errors = self._validate_output(output)
        if errors:
            self._apply_retry_or_compensation(run, node, completion, validation=errors)
            return
        task_id = node.task_id
        read_version = node.read_version or run.read_snapshot.revision
        changes = {
            f"output::{task_id}": output,
            "last_committed_task": task_id,
        }
        prepared = self._store.prepare(read_version, changes)
        success_envelope = self._make_envelope(
            TASK_SUCCEEDED, aggregate_id=f"task::{task_id}",
            payload={"task_id": task_id, "committed_revision": read_version + 1},
            correlation_id=run.correlation_id,
        )
        commit_envelope = self._make_envelope(
            WORKSPACE_COMMITTED, aggregate_id="workspace",
            payload={
                "committed_revision": read_version + 1,
                "read_version": read_version,
                "change_keys": [f"output::{task_id}"],
            },
            correlation_id=run.correlation_id,
        )
        finding_envelopes: list[EventEnvelope] = []

        def transaction() -> list[Any]:
            """Pair the state change with its outbox rows in one commit."""
            rows: list[EventEnvelope] = []
            if self._workspace is not None:
                finding, decision = self._workspace.publish_task_result(
                    node, output, read_revision=read_version,
                    correlation_id=run.correlation_id,
                )
                if decision in {"accepted", "merged"} and finding is not None:
                    finding_envelopes.append(self._make_envelope(
                        FINDING_PUBLISHED, aggregate_id=f"workspace::{finding.finding_id}",
                        payload={
                            "finding_id": finding.finding_id,
                            "field": finding.target.field,
                            "workspace_revision": read_version + 1,
                            "producer_module": finding.producer_module,
                            "confidence": finding.confidence,
                        },
                        correlation_id=run.correlation_id,
                        causation_id=success_envelope.event_id,
                    ))
            rows.extend(finding_envelopes)
            rows.append(success_envelope)
            rows.append(commit_envelope)
            for row in rows:
                self._outbox.record(row)
            return rows

        result = self._store.commit(prepared, transaction=transaction)
        if result.committed:
            run.task_outputs[task_id] = output
            run.committed_revisions.append(result.committed_revision)
            run.record(
                "commit", task_id=task_id, revision=result.committed_revision,
                read_version=read_version,
            )
            self._metrics.record_completion()
        else:
            self._metrics.record_conflict_retry()
            run.record(
                "commit_conflict", task_id=task_id, read_version=read_version,
                reason=result.reason,
            )
            node.state = TaskState.WAITING
            node.read_version = self._store.latest().revision
            retry_envelope = self._make_envelope(
                TASK_RETRIED, aggregate_id=f"task::{task_id}",
                payload={"task_id": task_id, "attempt": node.attempts},
                correlation_id=run.correlation_id,
                causation_id=commit_envelope.event_id,
            )
            self._outbox.record(retry_envelope)
            run.events_emitted += 1

    def _apply_retry_or_compensation(
        self, run: ExecutionRun, node: TaskNode, completion: TaskCompletion,
        validation: Sequence[str] = (),
    ) -> None:
        """Apply the retry or compensation policy for one failed attempt."""
        spec = node.spec
        policy = spec.retry_policy
        safety = RetrySafety.evaluate(
            spec.side_effect_class, spec.idempotency_key, spec.compensation,
        )
        error = "; ".join(
            [completion.error or f"{completion.outcome.value}"] + list(validation)
        )
        retryable = (
            node.attempts < policy.max_attempts
            and safety.retriable
        )
        if retryable:
            node.state = TaskState.WAITING
            node.last_error = error
            retry_envelope = self._make_envelope(
                TASK_RETRIED, aggregate_id=f"task::{node.task_id}",
                payload={"task_id": node.task_id, "attempt": node.attempts},
                correlation_id=run.correlation_id,
            )
            self._outbox.record(retry_envelope)
            run.events_emitted += 1
            self._metrics.record_completion(retried=True)
            run.record("retry", task_id=node.task_id, attempt=node.attempts, error=error)
            return
        if safety.compensation_available or spec.compensation is not None:
            self._compensate_task(run, node, error)
            return
        self._complete_failed(run, node, error=error)

    def _compensate_task(self, run: ExecutionRun, node: TaskNode, error: str) -> None:
        """Run one task's compensating action as a new auditable saga step."""
        node.state = TaskState.COMPENSATING
        compensating_envelope = self._make_envelope(
            "task.compensating", aggregate_id=f"task::{node.task_id}",
            payload={"task_id": node.task_id, "compensation_step": "task_failure"},
            correlation_id=run.correlation_id,
        )
        self._outbox.record(compensating_envelope)
        run.events_emitted += 1
        compensation = node.spec.compensation
        if compensation is not None:
            try:
                compensation(node.spec.payload, error)
            except Exception as exc:  # noqa: BLE001 - compensation isolation
                error = f"{error}; compensation failed: {exc}"
        else:
            saga_steps = (
                SagaStep(name=node.task_id,
                         action=lambda context: context.get("payload", {})),
            )
            self._sagas.run(f"saga::{node.task_id}", saga_steps,
                            {"payload": node.spec.payload})
        self._complete_failed(run, node, error=error, compensated=True)

    def _complete_failed(
        self, run: ExecutionRun, node: TaskNode, *, error: str,
        compensated: bool = False,
    ) -> None:
        """Mark one task failed and emit its audit event."""
        node.state = TaskState.FAILED
        node.last_error = error
        failure_envelope = self._make_envelope(
            TASK_FAILED, aggregate_id=f"task::{node.task_id}",
            payload={"task_id": node.task_id, "error": error[:500]},
            correlation_id=run.correlation_id,
        )
        self._outbox.record(failure_envelope)
        run.events_emitted += 1
        self._metrics.record_completion(failed=True)
        run.record("task_failed", task_id=node.task_id, error=error,
                   compensated=compensated)

    def _complete_cancelled(
        self, run: ExecutionRun, node: TaskNode, *, reason: str,
    ) -> None:
        """Mark one task cancelled and emit its audit event."""
        node.state = TaskState.CANCELLED
        node.last_error = reason
        envelope = self._make_envelope(
            TASK_CANCELLED, aggregate_id=f"task::{node.task_id}",
            payload={"task_id": node.task_id, "reason": reason},
            correlation_id=run.correlation_id,
        )
        self._outbox.record(envelope)
        run.events_emitted += 1
        self._metrics.record_completion(cancelled=True)
        run.record("task_cancelled", task_id=node.task_id, reason=reason)

    def _cancel_node(self, run: ExecutionRun, node: TaskNode, *, reason: str) -> None:
        """Cancel one not-yet-run task, preventing unsafe new side effects."""
        if node.state is TaskState.RUNNING:
            self._pool.cancel(node.task_id, reason)
        self._complete_cancelled(run, node, reason=reason)

    # -------------------------------------------------- dependency indexes

    def _update_dependency_indexes(self, run: ExecutionRun, node: TaskNode) -> None:
        """Record the task's derived artifact and its exact dependency set."""
        if node.state is not TaskState.SUCCEEDED:
            return
        deps: list[DependencyRef] = []
        for dep_id in node.spec.dependencies:
            upstream = self._dependencies.artifact(f"artifact::{dep_id}")
            if upstream is not None:
                deps.append(DependencyRef(
                    record_id=upstream.artifact_id,
                    version=1, content_hash=upstream.artifact_hash,
                ))
        for record_id in node.spec.metadata.get("usg_deps", ()):
            source = self._dependencies.source(str(record_id))
            if source is not None:
                deps.append(source.as_dependency())
        artifact_id = f"artifact::{node.task_id}"
        content = dict(node.output or {})
        if self._dependencies.artifact(artifact_id) is None:
            self._dependencies.register_artifact(
                artifact_id, deps,
                rules=tuple(node.spec.metadata.get("rules_applied", ())),
                config={"capability": node.spec.capability},
                content=content,
            )
            self._dependencies.register_source(artifact_id, content)
            executor = self._executor_for(run, node)
            if executor is not None:
                self._artifact_executors[artifact_id] = executor

    # ------------------------------------------------------- USG integration

    def register_usg_source(
        self, record_id: str, payload: Mapping[str, Any], *,
        validator: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> SourceRecord:
        """Register one USG source record through its read-validated contract."""
        return self._dependencies.register_source(
            record_id, payload, validator=validator,
        )

    def declare_usg_change(
        self, record_id: str, payload: Mapping[str, Any], *,
        validator: Callable[[Mapping[str, Any]], bool] | None = None,
        correlation_id: str | None = None,
    ) -> tuple[SourceRecord, Any]:
        """Commit a versioned USG change and trigger incremental invalidation."""
        record = self._dependencies.register_source(
            record_id, payload, validator=validator,
        )
        read_version = self._store.latest().revision
        prepared = self._store.prepare(read_version, {
            f"usg::{record_id}": {
                "version": record.version, "content_hash": record.content_hash,
            },
        })
        change_envelope = self._make_envelope(
            USG_RECORD_CHANGED, aggregate_id=f"usg::{record_id}",
            payload={
                "record_id": record_id, "record_version": record.version,
            },
            correlation_id=correlation_id,
        )

        def transaction() -> list[Any]:
            """Pair the USG change commit with its outbox row."""
            self._outbox.record(change_envelope)
            return [change_envelope]

        result = self._store.commit(prepared, transaction=transaction)
        self._drain_all()
        report = self._invalidation.invalidate([record_id])
        self._metrics.record_recompute_volume(
            len(report.recomputed), len(report.marked_stale),
        )
        return record, report

    def recompute_stale(
        self, executors: Mapping[str, TaskBody] | None = None,
    ) -> Any:
        """Recompute stale artifacts in topological order, reusing fresh caches."""
        executors = dict(executors or self._artifact_executors)

        def recompute_fn(record: Any) -> Mapping[str, Any]:
            """Run the artifact's stored executor against the latest snapshot."""
            executor = executors.get(record.artifact_id) or executors.get(
                record.artifact_id.removeprefix("artifact::")
            )
            if executor is None:
                return dict(record.content)
            task_spec = TaskSpec(
                task_id=record.artifact_id.removeprefix("artifact::"),
                name=record.artifact_id,
                payload=dict(record.content),
            )
            context = TaskContext(
                task_spec=task_spec,
                snapshot_data=self._store.latest().data,
                cancellation=__import__(
                    "ore_runtime.concurrency.worker_pool", fromlist=["CancellationToken"],
                ).CancellationToken(),
                checkpoint_store=self._pool.checkpoint_store,
                correlation_id="",
                run_id="recompute",
            )
            return executor(context)

        report = self._invalidation.recompute(
            self._invalidation.stale_artifacts(), recompute_fn,
        )
        for artifact_id in report.recomputed:
            envelope = self._make_envelope(
                ARTIFACT_RECOMPUTED, aggregate_id=f"artifact::{artifact_id}",
                payload={
                    "artifact_id": artifact_id,
                    "artifact_hash": (
                        self._dependencies.artifact(artifact_id).artifact_hash
                        if self._dependencies.artifact(artifact_id) else ""
                    ),
                },
            )
            self._outbox.record(envelope)
        self._drain_all()
        run = self._current_run
        if run is not None:
            run.recomputed_artifacts.extend(report.recomputed)
            run.stale_artifacts.extend(report.still_stale)
        return report

    # ------------------------------------------------------------- health

    def _check_health(self, run: ExecutionRun) -> None:
        """Check deadlines, cancellations, and blocked dependents."""
        now = time.monotonic()
        for node in run.graph.nodes:
            if not node.unfinished:
                continue
            if node.state in {TaskState.CREATED, TaskState.READY, TaskState.WAITING}:
                blocked_reason: str | None = None
                for dep in node.spec.dependencies:
                    dep_state = run.graph.node(dep).state
                    if dep_state is TaskState.FAILED:
                        blocked_reason = f"upstream dependency failed: {dep}"
                        break
                    if dep_state is TaskState.CANCELLED:
                        blocked_reason = f"upstream dependency cancelled: {dep}"
                        break
                if blocked_reason is not None:
                    self._cancel_node(run, node, reason=blocked_reason)
                    continue
                if node.spec.deadline_seconds is not None and node.enqueued_at is not None:
                    waited = now - node.enqueued_at
                    if waited > node.spec.deadline_seconds:
                        node.deadline_exceeded = True
                        self._cancel_node(
                            run, node, reason="deadline expired before dispatch",
                        )
        if run.cancelled:
            self._pool.cancel_all(run.cancel_reason or "goal cancelled")
            for node in run.graph.nodes:
                if node.unfinished and node.state is not TaskState.RUNNING:
                    self._cancel_node(
                        run, node, reason=run.cancel_reason or "goal cancelled",
                    )

    def cancel_goal(self, run: ExecutionRun, reason: str = "goal cancelled") -> None:
        """Propagate goal cancellation through the task tree."""
        run.cancelled = True
        run.cancel_reason = reason
        run.record("goal_cancelled", reason=reason)

    # --------------------------------------------------------- checkpoints

    def _persist_checkpoint(self, run: ExecutionRun) -> None:
        """Persist one run checkpoint with task states and committed revisions."""
        checkpoint = {
            "run_id": run.run_id,
            "goal_id": run.request.goal_id,
            "correlation_id": run.correlation_id,
            "rounds": run.rounds,
            "committed_revisions": list(run.committed_revisions),
            "task_states": run.graph.terminal_states(),
            "task_outputs": {
                task_id: dict(output) for task_id, output in run.task_outputs.items()
            },
            "saved_at": utc_now(),
        }
        self._pool.checkpoint_store.save(run.run_id, checkpoint)
        envelope = self._make_envelope(
            "recovery.checkpointed", aggregate_id=f"run::{run.run_id}",
            payload={"task_id": run.run_id},
            correlation_id=run.correlation_id,
        )
        self._outbox.record(envelope)
        run.events_emitted += 1

    def recover_run(
        self, checkpoint: Mapping[str, Any], task_specs: Sequence[TaskSpec],
        executors: Mapping[str, TaskBody], *,
        correlation_id: str | None = None,
    ) -> "GoalExecutionReport":
        """Resume or safely retry unfinished tasks after a simulated crash."""
        recovery_started = time.monotonic()
        states = dict(checkpoint.get("task_states", {}))
        request = ExecutionRequest(
            goal_id=str(checkpoint.get("goal_id", new_id("goal"))),
            description=f"recovery of run {checkpoint.get('run_id', 'unknown')}",
            inputs={},
            correlation_id=correlation_id or str(
                checkpoint.get("correlation_id", "")
            ),
        )
        graph = self.build_plan(request, task_specs)
        for node in graph.nodes:
            saved_state = states.get(node.task_id)
            if saved_state in {
                TaskState.SUCCEEDED.value, TaskState.FAILED.value,
                TaskState.CANCELLED.value,
            }:
                node.state = TaskState(saved_state)
            else:
                safety = RetrySafety.evaluate(
                    node.spec.side_effect_class,
                    node.spec.idempotency_key,
                    node.spec.compensation,
                )
                if safety.retriable:
                    node.state = TaskState.WAITING
                else:
                    node.state = TaskState.CANCELLED
                    node.last_error = (
                        "not retried after crash: side-effect class requires "
                        "manual review"
                    )
            node.enqueued_at = time.monotonic()
        run = ExecutionRun(
            run_id=str(checkpoint.get("run_id", new_id("run"))),
            request=request, graph=graph, executors=dict(executors),
            correlation_id=request.correlation_id,
            read_snapshot=self._store.capture(),
            rounds=0,
        )
        run.record("recovery_started", from_run=checkpoint.get("run_id"))
        self._current_run = run
        report = self.run_loop(run)
        self._metrics.record_recovery_duration(time.monotonic() - recovery_started)
        return report

    # --------------------------------------------------- iterative regions

    def _run_iterative_regions(self, run: ExecutionRun) -> None:
        """Execute declared iterative regions with their bounded convergence rules."""
        for region in run.graph.iterative_regions:
            result = run.graph.run_iterative_region(region)
            prepared = self._store.prepare(
                self._store.latest().revision,
                {f"region::{region.region_id}": {
                    "converged": result.converged,
                    "iterations": result.iterations,
                    "final_state": dict(result.final_state),
                    "stopped_because": result.stopped_because,
                }},
            )
            self._store.commit(prepared)
            run.record(
                "iterative_region", region_id=region.region_id,
                converged=result.converged, iterations=result.iterations,
                stopped_because=result.stopped_because,
            )

    # ------------------------------------------------------------ finalize

    def finalize(self, run: ExecutionRun) -> "GoalExecutionReport":
        """Validate final outputs, emit the completion event, and build the report."""
        from .contracts import GoalExecutionReport
        final_outputs: dict[str, Any] = {}
        missing: list[str] = []
        latest = self._store.latest()
        for name in run.request.expected_outputs:
            value = latest.data.get(f"output::{name}")
            if value is None:
                missing.append(name)
            final_outputs[name] = value
        status = "succeeded"
        if run.cancelled:
            status = "cancelled"
        elif any(
            node.state in {TaskState.FAILED, TaskState.CANCELLED}
            for node in run.graph.nodes
        ) or missing or run.error:
            status = "failed" if missing or run.error else "partial"
        if not run.graph.is_complete():
            status = "incomplete"
        completed_envelope = self._make_envelope(
            GOAL_COMPLETED, aggregate_id=f"goal::{run.request.goal_id}",
            payload={"goal_id": run.request.goal_id, "status": status},
            correlation_id=run.correlation_id,
        )
        self._outbox.record(completed_envelope)
        run.events_emitted += 1
        self.drain_events(run)
        self._metrics.record_stale_artifacts(len(self._invalidation.stale_artifacts()))
        succeeded = tuple(
            node.task_id for node in run.graph.nodes
            if node.state is TaskState.SUCCEEDED
        )
        failed = tuple(
            node.task_id for node in run.graph.nodes if node.state is TaskState.FAILED
        )
        cancelled = tuple(
            node.task_id for node in run.graph.nodes
            if node.state is TaskState.CANCELLED
        )
        run.status = status
        report = GoalExecutionReport(
            goal_id=run.request.goal_id,
            correlation_id=run.correlation_id,
            status=status,
            succeeded_tasks=succeeded,
            failed_tasks=failed,
            cancelled_tasks=cancelled,
            committed_revisions=tuple(run.committed_revisions),
            events_emitted=run.events_emitted,
            stale_artifacts=tuple(dict.fromkeys(run.stale_artifacts)),
            recomputed_artifacts=tuple(dict.fromkeys(run.recomputed_artifacts)),
            rounds=run.rounds,
            final_outputs=final_outputs,
            trace=tuple(run.trace),
            error=(
                run.error
                or (f"missing expected outputs: {missing}" if missing else None)
            ),
        )
        run.record("finalized", status=status, missing=missing)
        self._current_run = None
        return report

    # ------------------------------------------------------- manual driving

    def manual_dispatch(self, run: ExecutionRun) -> tuple[str, ...]:
        """Dispatch ready tasks once for deterministic single-step control."""
        return self._dispatch_ready(run)

    def manual_execute(
        self, run: ExecutionRun, task_id: str,
    ) -> TaskCompletion:
        """Run one dispatched task's executor synchronously and process it."""
        executor = self._executor_for(run, run.graph.node(task_id))
        if executor is None:
            completion = self._pool.complete_manual(
                task_id, "failed", error="no executor registered for task",
            )
            self._process_completion(run, completion)
            return completion
        pending = self._pool.pending_of(task_id)
        if pending is None:
            raise KeyError(f"task '{task_id}' is not dispatched.")
        context = TaskContext(
            task_spec=run.graph.node(task_id).spec,
            snapshot_data=self._store.revision(pending.read_version).data,
            cancellation=pending.cancellation,
            checkpoint_store=self._pool.checkpoint_store,
            correlation_id=run.correlation_id,
            run_id=run.run_id,
        )
        try:
            output = executor(context)
        except Exception as exc:  # noqa: BLE001 - manual fault capture
            completion = self._pool.complete_manual(
                task_id, "failed", error=f"{type(exc).__name__}: {exc}",
            )
            self._process_completion(run, completion)
            return completion
        completion = self._pool.complete_manual(
            task_id, "succeeded", output=output or {},
        )
        self._process_completion(run, completion)
        return completion

    def manual_inject(
        self, run: ExecutionRun, task_id: str, outcome: str,
        error: str | None = None,
    ) -> TaskCompletion:
        """Inject a fault outcome for one dispatched task and process it."""
        completion = self._pool.complete_manual(
            task_id, outcome, output={}, error=error,
        )
        self._process_completion(run, completion)
        return completion

    def manual_check_health(self, run: ExecutionRun) -> None:
        """Run one health-check pass for deterministic step control."""
        self._check_health(run)

    def abandon_pending(self, reason: str = "run abandoned") -> tuple[str, ...]:
        """Abandon leftover dispatched executions from an interrupted run."""
        abandoned = self._pool.abandon_all(reason)
        if abandoned and self._current_run is not None:
            self._current_run.record("abandoned_pending", tasks=list(abandoned))
        return abandoned

    # -------------------------------------------------------------- reports

    def audit_trace(self, correlation_id: str) -> tuple[dict[str, Any], ...]:
        """Reconstruct the causal event chain for one correlation identifier."""
        return self._event_store.causal_chain(correlation_id)

    def metrics_report(self) -> dict[str, Any]:
        """Return the full operational metrics report for this engine."""
        report = self._metrics.merge({
            "store": self._store.stats(),
            "outbox": self._outbox.stats(),
            "bus": self._bus.stats(),
            "event_store": self._event_store.stats(),
            "scheduler": self._scheduler.stats(),
            "worker_pool": {
                "active": self._pool.active_task_count(),
                "utilization": self._pool.utilization(),
                "load_imbalance": self._pool.load_imbalance(),
            },
            "invalidation": self._invalidation.stats(),
            "sagas": self._sagas.stats(),
            "admission": self._admission.stats(),
        })
        return report

    def diagnostics(self) -> dict[str, Any]:
        """Return subsystem health snapshots for operations dashboards."""
        return {
            "config": {
                "max_workers": self._config.max_workers,
                "max_active_tasks": self._config.max_active_tasks,
                "queue_high_watermark": self._config.queue_high_watermark,
                "queue_low_watermark": self._config.queue_low_watermark,
                "execution_mode": self._config.execution_mode,
            },
            "workers": {
                worker_id: sorted(capabilities)
                for worker_id, capabilities in self._pool.worker_capabilities().items()
            },
            "outbox_pending": len(self._outbox.pending_rows()),
            "event_catalogue_events": len(self._bus.publish_records()),
            "invalidation": self._invalidation.stats(),
        }

    # ------------------------------------------------------------- closing

    def close(self) -> None:
        """Shut down the pool and clear bus subscriptions deterministically."""
        with self._run_lock:
            if self._closed:
                return
            self._closed = True
        self._pool.close()
        self._bus.close()

    def __enter__(self) -> "CognitiveExecutionEngine":
        """Return the engine for context-manager use."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the engine on context exit."""
        self.close()
