"""Concurrent and incremental cognitive execution engine for ORE (Point 19)."""
from __future__ import annotations

from .adapters import USGSourceAdapter, WorkspaceRuntimeAdapter, build_workspace_permissions, connect_workspace
from .concurrency.scheduler import (
    AdmissionController, AdmissionDecision, AdmissionOutcome, Scheduler, SchedulerChoice,
    WorkerQueueState,
)
from .concurrency.task_graph import (
    IterationResult, IterativeRegionSpec, TaskGraph, TaskNode,
)
from .concurrency.versioned_state import (
    CommitResult, PreparedCommit, ReaderTicket, Snapshot, VersionedStore,
)
from .concurrency.worker_pool import (
    BoundedWorkerPool, CancellationToken, CheckpointStore, PendingExecution,
    TaskContext,
)
from .contracts import (
    CommitDecision, DeliveryGuarantee, EngineConfig, ExecutionRequest,
    GoalExecutionReport, RetryPolicy, RetrySafety, SideEffectClass, TaskAttemptState,
    TaskCompletion, TaskSpec, TaskState, WorkerSpec,
)
from .engine import CognitiveExecutionEngine, ExecutionRun
from .events.bus import DeliveryOutcome, DeliveryStatus, EventBus, Subscription
from .events.idempotency import ApplyStatus, IdempotencyStore
from .events.outbox import OutboxRow, OutboxStatus, TransactionalOutbox
from .events.sagas import (
    SagaCoordinator, SagaOutcome, SagaStep, SagaStepOutcome, build_compensatable_workflow,
)
from .events.schemas import (
    EVENT_CATALOGUE, SCHEMA_VERSION, EventEnvelope, EventSchema, assert_valid_envelope,
    catalogue_as_dicts, event_validity, supported_versions, validate_envelope,
)
from .exceptions import (
    CapabilityUnavailableError, CommitConflictError, CycleRejectedError,
    DeterminismError, EventSchemaError, InvariantViolationError, OutboxPublishError,
    PlanValidationError, RecoveryError, RuntimeEngineError, SagaCompensationIncompleteError,
    TaskCancelledError,
)
from .history.event_store import (
    AggregateSnapshot, EventStore, Projection, QuarantinedEvent, StoredEvent,
)
from .incremental.dependency_graph import (
    ArtifactRecord, ArtifactState, DependencyGraph, DependencyRef, SourceRecord,
    compute_artifact_hash,
)
from .incremental.invalidation import InvalidationEngine, InvalidationReport
from .metrics import RuntimeMetrics
from .testing.deterministic_scheduler import (
    DeterministicScheduler, Invariant, ReplayManifest, ScheduleStep, StepTrace,
    default_invariants, invariant_bounded_execution, invariant_capability_safety,
    invariant_dependency_ordering, invariant_event_schema_validity,
    invariant_no_partial_publication,
)

__all__ = [
    # engine and contracts
    "AdmissionController", "AdmissionDecision", "AdmissionOutcome",
    "AggregateSnapshot", "ApplyStatus", "ArtifactRecord", "ArtifactState",
    "BoundedWorkerPool", "CancellationToken", "CheckpointStore",
    "CommitConflictError", "CommitDecision", "CommitResult",
    "CognitiveExecutionEngine", "CycleRejectedError", "DeliveryGuarantee",
    "DeliveryOutcome", "DeliveryStatus", "DependencyGraph", "DependencyRef",
    "DeterminismError", "DeterministicScheduler", "EngineConfig",
    "EVENT_CATALOGUE", "EventBus", "EventEnvelope", "EventSchema", "EventSchemaError",
    "EventStore", "ExecutionRequest", "ExecutionRun", "GoalExecutionReport",
    "IdempotencyStore", "Invariant", "InvariantViolationError",
    "InvalidationEngine", "InvalidationReport", "IterationResult",
    "IterativeRegionSpec", "OutboxPublishError", "OutboxRow", "OutboxStatus",
    "PendingExecution", "PlanValidationError", "PreparedCommit", "Projection",
    "QuarantinedEvent", "ReaderTicket", "RecoveryError", "RetryPolicy",
    "RetrySafety", "RuntimeEngineError", "RuntimeMetrics",
    "SCHEMA_VERSION", "SagaCompensationIncompleteError", "SagaCoordinator",
    "SagaOutcome", "SagaStep", "SagaStepOutcome", "ScheduleStep", "Scheduler",
    "SchedulerChoice", "SideEffectClass", "Snapshot", "SourceRecord",
    "StoredEvent", "Subscription", "TaskAttemptState", "TaskCancelledError",
    "TaskCompletion", "TaskContext", "TaskGraph", "TaskNode", "TaskSpec",
    "TaskState", "TransactionalOutbox", "USGSourceAdapter", "VersionedStore",
    "WorkerQueueState", "WorkerSpec", "WorkspaceRuntimeAdapter",
    "assert_valid_envelope", "build_compensatable_workflow",
    "build_workspace_permissions", "catalogue_as_dicts", "compute_artifact_hash",
    "connect_workspace", "default_invariants", "event_validity",
    "invariant_bounded_execution", "invariant_capability_safety",
    "invariant_dependency_ordering", "invariant_event_schema_validity",
    "invariant_no_partial_publication", "supported_versions", "validate_envelope",
]
