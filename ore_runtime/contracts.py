"""Typed records and validated configuration for the Point 19 runtime engine."""
from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 millisecond timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    """Return a new unique identifier with the given dotted prefix."""
    return f"{prefix}_{uuid.uuid4().hex}"


def canonical_json(value: Any) -> str:
    """Serialize a value to deterministic JSON for hashing and content IDs."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def content_fingerprint(value: Any) -> str:
    """Hash canonical JSON content with SHA-256 so equal payloads share identifiers."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class TaskState(str, Enum):
    """Represent the complete lifecycle of a runtime task node."""

    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    COMMITTING = "committing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPENSATING = "compensating"


class SideEffectClass(str, Enum):
    """Classify task isolation so retry and compensation policy stays explicit."""

    PURE = "pure"
    LOCAL_MUTATION = "local_mutation"
    EXTERNAL_EFFECT = "external_effect"
    COMPENSATABLE = "compensatable"


class CommitDecision(str, Enum):
    """Report the outcome of an optimistic versioned commit."""

    COMMITTED = "committed"
    CONFLICT = "conflict"


class DeliveryGuarantee(str, Enum):
    """Label the honest delivery semantics of an event channel per the specification."""

    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EFFECTIVELY_ONCE = "effectively_once"


class TaskAttemptState(str, Enum):
    """Describe why a task attempt ended the way it did."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bound retries by purity, idempotency, and compensation availability."""

    max_attempts: int = 3
    backoff_seconds: float = 0.0
    requires_idempotency: bool = False

    def __post_init__(self) -> None:
        """Reject invalid retry bounds that would loop without progress."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        if self.backoff_seconds < 0.0:
            raise ValueError("backoff_seconds cannot be negative.")

    @classmethod
    def for_side_effect_class(cls, side_effect_class: SideEffectClass) -> "RetryPolicy":
        """Derive the conservative default policy for a task's side-effect class."""
        defaults = {
            SideEffectClass.PURE: cls(max_attempts=3),
            SideEffectClass.LOCAL_MUTATION: cls(max_attempts=2),
            SideEffectClass.EXTERNAL_EFFECT: cls(max_attempts=1),
            SideEffectClass.COMPENSATABLE: cls(max_attempts=2),
        }
        return defaults[SideEffectClass(side_effect_class)]


@dataclass(frozen=True, slots=True)
class RetrySafety:
    """Score how safely a task can be retried without duplicating effects."""

    retriable: bool
    idempotent: bool
    compensation_available: bool
    reason: str

    @classmethod
    def evaluate(cls, side_effect_class: SideEffectClass, idempotency_key: str | None,
                 compensation: Callable[..., Any] | None) -> "RetrySafety":
        """Compute retry safety from purity, idempotency, and compensation availability."""
        kind = SideEffectClass(side_effect_class)
        idempotent = idempotency_key is not None
        compensation_available = compensation is not None
        if kind is SideEffectClass.PURE:
            retriable, reason = True, "pure tasks may be retried freely"
        elif kind is SideEffectClass.LOCAL_MUTATION:
            retriable, reason = True, "local mutations are retried against versioned state"
        elif kind is SideEffectClass.EXTERNAL_EFFECT:
            retriable, reason = idempotent, (
                "external effects are retried only with an idempotency key"
                if idempotent else "non-idempotent external effects are never blindly retried"
            )
        else:
            retriable, reason = True, "compensatable workflow steps retry within their saga"
        return cls(retriable=retriable, idempotent=idempotent,
                   compensation_available=compensation_available, reason=reason)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Declare one independently schedulable unit of cognitive work."""

    task_id: str
    name: str
    dependencies: tuple[str, ...] = ()
    capability: str = "generic"
    priority: float = 0.5
    deadline_seconds: float | None = None
    estimated_cost: float = 1.0
    side_effect_class: SideEffectClass = SideEffectClass.PURE
    payload: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    max_attempts: int | None = None
    backoff_seconds: float = 0.0
    compensation: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        """Validate identity, dependency uniqueness, and bounded numeric fields."""
        if not self.task_id.strip() or not self.name.strip():
            raise ValueError("task_id and name must be non-empty.")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("dependencies must not contain duplicates.")
        if not self.capability.strip():
            raise ValueError("capability must be non-empty.")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("priority must lie in [0, 1].")
        if self.deadline_seconds is not None and self.deadline_seconds <= 0.0:
            raise ValueError("deadline_seconds must be positive when provided.")
        if self.estimated_cost <= 0.0:
            raise ValueError("estimated_cost must be positive.")
        if not isinstance(self.side_effect_class, SideEffectClass):
            raise TypeError("side_effect_class must be a SideEffectClass.")
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def retry_policy(self) -> RetryPolicy:
        """Return the effective retry policy for this task."""
        if self.max_attempts is not None:
            return RetryPolicy(max_attempts=self.max_attempts,
                               backoff_seconds=self.backoff_seconds)
        return RetryPolicy.for_side_effect_class(self.side_effect_class)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable planning record without callables."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "dependencies": list(self.dependencies),
            "capability": self.capability,
            "priority": self.priority,
            "deadline_seconds": self.deadline_seconds,
            "estimated_cost": self.estimated_cost,
            "side_effect_class": self.side_effect_class.value,
            "payload": dict(self.payload),
            "idempotency_key": self.idempotency_key,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Declare one bounded worker with capabilities and a resource budget."""

    worker_id: str
    capabilities: frozenset[str] | tuple[str, ...]
    resource_capacity: float = 1.0

    def __post_init__(self) -> None:
        """Validate worker identity, capability set, and positive resource budget."""
        if not self.worker_id.strip():
            raise ValueError("worker_id must be non-empty.")
        if isinstance(self.capabilities, frozenset):
            object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        else:
            if not self.capabilities:
                raise ValueError("capabilities must be non-empty.")
            object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        if not self.capabilities:
            raise ValueError("capabilities must be non-empty.")
        if self.resource_capacity <= 0.0:
            raise ValueError("resource_capacity must be positive.")

    def supports(self, capability: str) -> bool:
        """Return whether this worker offers a required capability."""
        return capability in self.capabilities


@dataclass(frozen=True, slots=True)
class SchedulerWeights:
    """Weight the scheduling score between priority, deadline slack, cost, and aging."""

    alpha_priority: float = 1.0
    beta_deadline: float = 0.25
    gamma_cost: float = 0.1
    delta_aging: float = 0.2
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        """Reject negative weights that would invert the documented scoring policy."""
        for name in ("alpha_priority", "beta_deadline", "gamma_cost", "delta_aging"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Bound every concurrency dimension of the execution engine explicitly."""

    max_workers: int = 4
    max_active_tasks: int = 4
    queue_high_watermark: int = 16
    queue_low_watermark: int = 8
    max_ready_queue_size: int = 64
    module_queue_quota: int = 16
    aging_rate: float = 0.05
    max_priority: float = 1.0
    scheduler_weights: SchedulerWeights = field(default_factory=SchedulerWeights)
    default_task_timeout_seconds: float = 30.0
    max_rounds: int = 256
    max_iterations_per_region: int = 64
    convergence_epsilon: float = 1e-6
    checkpoint_interval_rounds: int = 1
    outbox_retry_backoff_seconds: float = 0.01
    outbox_max_attempts: int = 5
    retained_revision_limit: int = 8
    execution_mode: str = "live"

    def __post_init__(self) -> None:
        """Validate bounds so parallelism can never exhaust the machine."""
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        if self.max_active_tasks < 1:
            raise ValueError("max_active_tasks must be at least 1.")
        if self.queue_high_watermark < 1:
            raise ValueError("queue_high_watermark must be at least 1.")
        if not 0 <= self.queue_low_watermark < self.queue_high_watermark:
            raise ValueError("queue_low_watermark must lie in [0, queue_high_watermark).")
        if self.max_ready_queue_size < self.queue_high_watermark:
            raise ValueError("max_ready_queue_size must cover the high watermark.")
        if self.module_queue_quota < 1:
            raise ValueError("module_queue_quota must be at least 1.")
        if self.aging_rate < 0.0 or self.max_priority <= 0.0:
            raise ValueError("aging_rate must be non-negative and max_priority positive.")
        if self.default_task_timeout_seconds <= 0.0:
            raise ValueError("default_task_timeout_seconds must be positive.")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1.")
        if self.max_iterations_per_region < 1:
            raise ValueError("max_iterations_per_region must be at least 1.")
        if self.convergence_epsilon <= 0.0:
            raise ValueError("convergence_epsilon must be positive.")
        if self.outbox_max_attempts < 1:
            raise ValueError("outbox_max_attempts must be at least 1.")
        if self.retained_revision_limit < 1:
            raise ValueError("retained_revision_limit must be at least 1.")
        if self.execution_mode not in {"live", "manual"}:
            raise ValueError("execution_mode must be 'live' or 'manual'.")


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """Carry a validated cognitive goal with explicit inputs, constraints, and outputs."""

    goal_id: str
    description: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    correlation_id: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize collections, validate identifiers, and attach a correlation ID."""
        if not self.goal_id.strip() or not self.description.strip():
            raise ValueError("goal_id and description must be non-empty.")
        object.__setattr__(self, "inputs", dict(self.inputs))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if any(not constraint.strip() for constraint in self.constraints):
            raise ValueError("constraints must be non-empty statements.")
        if any(not name.strip() for name in self.expected_outputs):
            raise ValueError("expected_outputs must be non-empty names.")
        object.__setattr__(
            self, "correlation_id",
            self.correlation_id.strip() or new_id("correlation"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable request record for the audit trace."""
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "inputs": dict(self.inputs),
            "constraints": list(self.constraints),
            "expected_outputs": list(self.expected_outputs),
            "correlation_id": self.correlation_id,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class TaskAttempt:
    """Record one execution attempt of a task for the audit trail."""

    attempt: int
    worker_id: str | None
    started_at: str
    finished_at: str | None
    outcome: TaskAttemptState | None
    error: str | None


@dataclass(frozen=True, slots=True)
class TaskCompletion:
    """Carry a worker's task outcome back to the engine for validation and commit."""

    task_id: str
    worker_id: str | None
    attempt: int
    outcome: TaskAttemptState
    output: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    read_version: int = 0
    elapsed_seconds: float = 0.0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach caller mappings and validate the attempt outcome shape."""
        object.__setattr__(self, "output", dict(self.output))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if not isinstance(self.outcome, TaskAttemptState):
            raise TypeError("outcome must be a TaskAttemptState.")
        if self.outcome is TaskAttemptState.SUCCEEDED and self.error is not None:
            raise ValueError("a succeeded attempt cannot carry an error.")
        if self.outcome is not TaskAttemptState.SUCCEEDED and self.error is None:
            raise ValueError("a non-successful attempt must record its error.")

    @property
    def succeeded(self) -> bool:
        """Return whether this completion succeeded and awaits commit."""
        return self.outcome is TaskAttemptState.SUCCEEDED


@dataclass(frozen=True, slots=True)
class GoalExecutionReport:
    """Summarize one executed goal with state, events, and audit evidence."""

    goal_id: str
    correlation_id: str
    status: str
    succeeded_tasks: tuple[str, ...]
    failed_tasks: tuple[str, ...]
    cancelled_tasks: tuple[str, ...]
    committed_revisions: tuple[int, ...]
    events_emitted: int
    stale_artifacts: tuple[str, ...]
    recomputed_artifacts: tuple[str, ...]
    rounds: int
    final_outputs: Mapping[str, Any] = field(default_factory=dict)
    trace: tuple[Mapping[str, Any], ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        """Detach final outputs so the report stays immutable for callers."""
        object.__setattr__(self, "final_outputs", dict(self.final_outputs))
        object.__setattr__(self, "trace", tuple(dict(item) for item in self.trace))

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable report for logs and replay manifests."""
        return {
            "goal_id": self.goal_id,
            "correlation_id": self.correlation_id,
            "status": self.status,
            "succeeded_tasks": list(self.succeeded_tasks),
            "failed_tasks": list(self.failed_tasks),
            "cancelled_tasks": list(self.cancelled_tasks),
            "committed_revisions": list(self.committed_revisions),
            "events_emitted": self.events_emitted,
            "stale_artifacts": list(self.stale_artifacts),
            "recomputed_artifacts": list(self.recomputed_artifacts),
            "rounds": self.rounds,
            "final_outputs": dict(self.final_outputs),
            "error": self.error,
        }


def validate_task_specs(specs: Sequence[TaskSpec]) -> tuple[TaskSpec, ...]:
    """Validate a task manifest for unique IDs and known dependencies."""
    if not specs:
        raise ValueError("task manifest must not be empty.")
    frozen = tuple(specs)
    ids = [spec.task_id for spec in frozen]
    if len(set(ids)) != len(ids):
        duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
        raise ValueError(f"task ids must be unique; duplicates: {duplicates}")
    known = set(ids)
    for spec in frozen:
        unknown = [dep for dep in spec.dependencies if dep not in known]
        if unknown:
            raise ValueError(
                f"task '{spec.task_id}' declares unknown dependencies: {sorted(unknown)}"
            )
    return frozen
