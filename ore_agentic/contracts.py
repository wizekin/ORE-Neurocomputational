"""Typed contracts for the Point 15 agentic workflow and memory layer.

These contracts implement the goal, task-graph, lease, result, provenance, and
outcome interfaces of the Point 15 specification.  They are deliberately
explicit about side-effect class, risk, provenance, and state so that workflow
state can never silently compete with the authoritative Shared Cognitive
Workspace, and so hypotheses and observations can never silently become
authoritative USG facts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
import time
from typing import Any, Mapping

__all__ = (
    "ApprovalStatus",
    "CancellationState",
    "DefectSeverity",
    "DeliverableSpec",
    "ExecutionDecision",
    "FailureClass",
    "GoalInterpretation",
    "GoalRequest",
    "Hypothesis",
    "JoinPolicy",
    "JoinSemantics",
    "OutcomeReport",
    "OutcomeStatus",
    "PatternKind",
    "ProvenanceRecord",
    "RequestRejection",
    "ResourceVector",
    "RetryPolicy",
    "RiskAssessment",
    "SideEffectClass",
    "SubgoalSpec",
    "TaskOutcomeStatus",
    "TaskResult",
    "TaskSpec",
    "TaskState",
    "ToolCallRecord",
    "WorkflowStatus",
    "utc_now",
    "new_id",
    "canonical_json",
    "content_digest",
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id(prefix: str) -> str:
    """Return a collision-resistant identifier with the given prefix."""
    digest = hashlib.sha256(f"{prefix}:{time.time_ns()}:{id(object())}".encode("utf-8"))
    return f"{prefix}_{digest.hexdigest()[:12]}"


def canonical_json(value: Any) -> str:
    """Serialize deterministically (sorted keys, compact separators)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def content_digest(value: Any) -> str:
    """Return a stable SHA-256 content fingerprint."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:16]


def _required(name: str, value: str) -> str:
    """Require a meaningful non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _unit_interval(name: str, value: float) -> float:
    """Validate a finite scalar bounded to [0, 1]."""
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return numeric


class SideEffectClass(str, Enum):
    """Side-effect classification used for risk and idempotency (Step 3)."""

    READ_ONLY = "read_only"
    REVERSIBLE_WRITE = "reversible_write"
    EXTERNALLY_VISIBLE = "externally_visible"
    IRREVERSIBLE = "irreversible"

    @property
    def risk_floor(self) -> float:
        """Return the minimum risk contribution of this class."""
        return {
            SideEffectClass.READ_ONLY: 0.0,
            SideEffectClass.REVERSIBLE_WRITE: 0.25,
            SideEffectClass.EXTERNALLY_VISIBLE: 0.5,
            SideEffectClass.IRREVERSIBLE: 0.75,
        }[self]


class ExecutionDecision(str, Enum):
    """Whether execution is automatic, gated on approval, or prohibited (Step 3)."""

    AUTOMATIC = "automatic"
    REQUIRES_APPROVAL = "requires_approval"
    PROHIBITED = "prohibited"


class TaskState(str, Enum):
    """Workflow task states of the validated state machine (Step 17)."""

    PENDING = "pending"
    READY = "ready"
    LEASED = "leased"
    RUNNING = "running"
    WAITING = "waiting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPENSATED = "compensated"


class TaskOutcomeStatus(str, Enum):
    """Terminal outcome of one delegated task attempt aggregation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABSTAINED = "abstained"
    CANCELLED = "cancelled"


class WorkflowStatus(str, Enum):
    """Workflow-level statuses, reported distinctly at close (Step 50)."""

    COMPILED = "compiled"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_CLARIFICATION = "waiting_clarification"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPENSATED = "compensated"
    REJECTED = "rejected"
    REUSED = "reused"


class FailureClass(str, Enum):
    """Failure taxonomy deciding retry safety (Step 16)."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"
    POLICY = "policy"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


class JoinSemantics(str, Enum):
    """Join semantics for aggregation nodes (Step 19)."""

    ALL = "all"
    ANY = "any"
    QUORUM = "quorum"
    WEIGHTED_QUORUM = "weighted_quorum"


class PatternKind(str, Enum):
    """Workflow patterns; the simplest suitable one must be chosen (Step 6)."""

    SEQUENTIAL = "sequential"
    SUPERVISOR_WORKER = "supervisor_worker"
    MAP_REDUCE = "map_reduce"
    EVALUATOR_OPTIMIZER = "evaluator_optimizer"
    CONDITIONAL_GRAPH = "conditional_graph"
    APPROVAL_GATE = "approval_gate"
    HYBRID = "hybrid"


class DefectSeverity(str, Enum):
    """Severity of a critic defect (Step 28)."""

    INFO = "info"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class ApprovalStatus(str, Enum):
    """Approval-gate state for consequential actions (Step 30)."""

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class OutcomeStatus(str, Enum):
    """Close-out categories reported distinctly (Step 50)."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    UNVERIFIED = "unverified"


class CancellationState:
    """Cooperative cancellation token shared with running workers (Step 22)."""

    def __init__(self) -> None:
        """Start in the non-cancelled state."""
        self._cancelled = False
        self.reason = ""

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._cancelled

    def cancel(self, reason: str = "operator request") -> None:
        """Request cancellation; running workers observe it via the token."""
        self._cancelled = True
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        """Serialize the observable cancellation state."""
        return {"cancelled": self._cancelled, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class ResourceVector:
    """Task cost vector c_i = (cpu, mem, io, net, time) with additive budgets."""

    cpu_units: float = 1.0
    memory_mb: float = 64.0
    io_units: float = 0.0
    network_units: float = 0.0
    seconds: float = 1.0

    def __post_init__(self) -> None:
        """Require finite non-negative components."""
        for name in ("cpu_units", "memory_mb", "io_units", "network_units", "seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")

    def add(self, other: ResourceVector) -> ResourceVector:
        """Return the component-wise sum with another vector."""
        return ResourceVector(
            cpu_units=self.cpu_units + other.cpu_units,
            memory_mb=self.memory_mb + other.memory_mb,
            io_units=self.io_units + other.io_units,
            network_units=self.network_units + other.network_units,
            seconds=self.seconds + other.seconds,
        )

    def fits_within(self, budget: ResourceVector) -> bool:
        """Return True when every component is within the budget."""
        return all(getattr(self, name) <= getattr(budget, name) for name in
                   ("cpu_units", "memory_mb", "io_units", "network_units", "seconds"))

    def to_dict(self) -> dict[str, float]:
        """Serialize the vector."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """An inferred value that is explicitly not a USG fact (Step 2)."""

    statement: str
    basis: str
    confidence: float

    def __post_init__(self) -> None:
        """Validate the hypothesis envelope."""
        _required("statement", self.statement)
        _required("basis", self.basis)
        _unit_interval("confidence", self.confidence)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the hypothesis."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SubgoalSpec:
    """One declared subgoal of a goal request (Step 5 input)."""

    name: str
    capability: str
    description: str = ""
    inputs: Mapping[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    side_effect_class: SideEffectClass = SideEffectClass.READ_ONLY
    output_schema: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate subgoal identity and dependency list."""
        _required("name", self.name)
        _required("capability", self.capability)
        if not isinstance(self.inputs, Mapping):
            raise ValueError("inputs must be a mapping")
        if not isinstance(self.output_schema, Mapping):
            raise ValueError("output_schema must be a mapping")
        if self.name in self.depends_on:
            raise ValueError(f"subgoal {self.name} cannot depend on itself")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the subgoal."""
        return {
            "name": self.name,
            "capability": self.capability,
            "description": self.description,
            "inputs": dict(self.inputs),
            "depends_on": list(self.depends_on),
            "side_effect_class": self.side_effect_class.value,
            "output_schema": dict(self.output_schema),
        }


@dataclass(frozen=True, slots=True)
class DeliverableSpec:
    """One declared deliverable of G = (O, C, I, D) (Step 2)."""

    name: str
    description: str = ""

    def __post_init__(self) -> None:
        """Validate deliverable identity."""
        _required("name", self.name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the deliverable."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GoalRequest:
    """The immutable original request with identity and constraints (Step 1)."""

    objective: str
    requester: str
    subgoals: tuple[SubgoalSpec, ...] = ()
    deliverables: tuple[DeliverableSpec, ...] = ()
    inputs: Mapping[str, Any] = field(default_factory=dict)
    constraints: Mapping[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    deadline_seconds: float | None = None
    flagged_ambiguities: tuple[str, ...] = ()
    side_effect_class: SideEffectClass = SideEffectClass.READ_ONLY
    idempotency_key: str | None = None
    request_id: str = field(default_factory=lambda: new_id("request"))
    timestamp: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Reject malformed or empty requests (Step 1)."""
        _required("objective", self.objective)
        _required("requester", self.requester)
        if len(self.objective) > 4000:
            raise ValueError("objective exceeds the maximum accepted length")
        _unit_interval("priority", self.priority)
        if self.deadline_seconds is not None and float(self.deadline_seconds) <= 0.0:
            raise ValueError("deadline_seconds must be positive when provided")
        names = [subgoal.name for subgoal in self.subgoals]
        if len(set(names)) != len(names):
            raise ValueError("subgoal names must be unique")

    def fingerprint(self, fields: tuple[str, ...] = ("objective", "inputs", "deliverables")
                    ) -> str:
        """Return a stable digest of the request content used for reuse checks."""
        payload = {}
        for name in fields:
            if name == "objective":
                payload["objective"] = self.objective
            elif name == "inputs":
                payload["inputs"] = canonical_json(dict(self.inputs))
            elif name == "deliverables":
                payload["deliverables"] = [item.to_dict() for item in self.deliverables]
            elif name == "subgoals":
                payload["subgoals"] = [item.to_dict() for item in self.subgoals]
        return content_digest(payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the immutable request reference."""
        return {
            "request_id": self.request_id,
            "requester": self.requester,
            "objective": self.objective,
            "subgoals": [item.to_dict() for item in self.subgoals],
            "deliverables": [item.to_dict() for item in self.deliverables],
            "inputs": dict(self.inputs),
            "constraints": dict(self.constraints),
            "priority": self.priority,
            "deadline_seconds": self.deadline_seconds,
            "flagged_ambiguities": list(self.flagged_ambiguities),
            "side_effect_class": self.side_effect_class.value,
            "idempotency_key": self.idempotency_key,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GoalRequest:
        """Rebuild a validated immutable request from its serialized form (Step 1)."""
        return cls(
            objective=value["objective"],
            requester=value["requester"],
            subgoals=tuple(SubgoalSpec(
                name=item["name"], capability=item["capability"],
                description=item.get("description", ""),
                inputs=dict(item.get("inputs", {})),
                depends_on=tuple(item.get("depends_on", ())),
                side_effect_class=SideEffectClass(item.get(
                    "side_effect_class", "read_only")),
                output_schema=dict(item.get("output_schema", {})))
                for item in value.get("subgoals", ())),
            deliverables=tuple(DeliverableSpec(name=item["name"],
                                               description=item.get("description", ""))
                               for item in value.get("deliverables", ())),
            inputs=dict(value.get("inputs", {})),
            constraints=dict(value.get("constraints", {})),
            priority=float(value.get("priority", 0.5)),
            deadline_seconds=(float(value["deadline_seconds"])
                              if value.get("deadline_seconds") is not None else None),
            flagged_ambiguities=tuple(value.get("flagged_ambiguities", ())),
            side_effect_class=SideEffectClass(value.get("side_effect_class",
                                                        "read_only")),
            idempotency_key=value.get("idempotency_key"),
            request_id=value.get("request_id", new_id("request")),
            timestamp=value.get("timestamp", utc_now()))


class RequestRejection(Exception):
    """A rejected request with the policy reason (Steps 1, 3).

    Deliberately a plain exception subclass (not a frozen dataclass) so that
    interpreter machinery such as ``__traceback__`` and ``__cause__``
    assignment keeps working when the rejection propagates through context
    managers.
    """

    def __init__(self, reason: str, code: str = "rejected") -> None:
        """Store the human-readable reason and stable code."""
        super().__init__(reason)
        self.reason = reason
        self.code = code

    def __str__(self) -> str:
        return f"{self.code}: {self.reason}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the rejection."""
        return {"reason": self.reason, "code": self.code}


@dataclass(frozen=True, slots=True)
class GoalInterpretation:
    """Goal G = (O, C, I, D) with hypotheses and ambiguities (Step 2)."""

    objective: str
    constraints: Mapping[str, Any]
    inputs: Mapping[str, Any]
    deliverables: tuple[DeliverableSpec, ...]
    entities: tuple[str, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    ambiguities: tuple[str, ...] = ()
    interpretation_version: str = "goal-v1"

    @property
    def needs_clarification(self) -> bool:
        """Clarification is requested when ambiguities materially change execution."""
        return len(self.ambiguities) > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the interpretation."""
        return {
            "objective": self.objective,
            "constraints": dict(self.constraints),
            "inputs": dict(self.inputs),
            "deliverables": [item.to_dict() for item in self.deliverables],
            "entities": list(self.entities),
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "ambiguities": list(self.ambiguities),
            "interpretation_version": self.interpretation_version,
        }


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """Configurable risk score R = sum(w_i * r_i) with decision (Step 3)."""

    score: float
    factors: Mapping[str, float]
    weights: Mapping[str, float]
    side_effect_class: SideEffectClass
    decision: ExecutionDecision
    reason: str

    def __post_init__(self) -> None:
        """Validate the assessment envelope."""
        _unit_interval("score", self.score)
        if not self.reason:
            raise ValueError("reason must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the assessment."""
        return {
            "score": self.score,
            "factors": dict(self.factors),
            "weights": dict(self.weights),
            "side_effect_class": self.side_effect_class.value,
            "decision": self.decision.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded exponential backoff with jitter (Step 16)."""

    max_attempts: int = 3
    initial_delay_seconds: float = 0.05
    max_delay_seconds: float = 2.0
    jitter_seconds: float = 0.02
    total_time_budget_seconds: float = 8.0

    def __post_init__(self) -> None:
        """Validate the retry bounds."""
        if int(self.max_attempts) < 1:
            raise ValueError("max_attempts must be at least 1")
        for name in ("initial_delay_seconds", "max_delay_seconds", "jitter_seconds",
                     "total_time_budget_seconds"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class JoinPolicy:
    """Join semantics and parameters for aggregation tasks (Step 19)."""

    semantics: JoinSemantics = JoinSemantics.ALL
    quorum: float = 0.5
    weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the quorum fraction."""
        if self.semantics in (JoinSemantics.QUORUM, JoinSemantics.WEIGHTED_QUORUM):
            _unit_interval("quorum", self.quorum)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy."""
        return {"semantics": self.semantics.value, "quorum": self.quorum,
                "weights": dict(self.weights)}


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """A typed subtask of the workflow graph (Step 5)."""

    task_id: str
    name: str
    capability: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    cost: ResourceVector = ResourceVector()
    retry: RetryPolicy = RetryPolicy()
    side_effect_class: SideEffectClass = SideEffectClass.READ_ONLY
    join: JoinPolicy = JoinPolicy()
    permitted_tools: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate task identity and dependency list."""
        _required("task_id", self.task_id)
        _required("name", self.name)
        _required("capability", self.capability)
        if self.task_id in self.depends_on:
            raise ValueError(f"task {self.task_id} cannot depend on itself")
        if not isinstance(self.input_schema, Mapping) or not isinstance(
                self.output_schema, Mapping):
            raise ValueError("schemas must be mappings")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the task specification."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "capability": self.capability,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "depends_on": list(self.depends_on),
            "cost": self.cost.to_dict(),
            "retry": self.retry.to_dict(),
            "side_effect_class": self.side_effect_class.value,
            "join": self.join.to_dict(),
            "permitted_tools": list(self.permitted_tools),
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """One tool invocation retained for provenance (invariant 3)."""

    tool: str
    capability: str
    request_id: str
    outcome: str
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Full provenance of a result: request, inputs, evidence, worker, versions."""

    request_id: str
    input_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    worker_id: str = ""
    worker_version: str = ""
    tool_calls: tuple[ToolCallRecord, ...] = ()
    versions: Mapping[str, str] = field(default_factory=dict)
    verification: str = "unverified"
    workspace_revision: int | None = None
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the provenance record."""
        return {
            "request_id": self.request_id,
            "input_refs": list(self.input_refs),
            "evidence_refs": list(self.evidence_refs),
            "worker_id": self.worker_id,
            "worker_version": self.worker_version,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "versions": dict(self.versions),
            "verification": self.verification,
            "workspace_revision": self.workspace_revision,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class TaskResult:
    """A typed worker result with evidence, provenance and resource use."""

    task_id: str
    worker_id: str
    status: TaskOutcomeStatus
    payload: Mapping[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    provenance: ProvenanceRecord | None = None
    warnings: tuple[str, ...] = ()
    resource_use: ResourceVector = ResourceVector(0.0, 0.0, 0.0, 0.0, 0.0)
    attempts: int = 1
    failure_class: FailureClass | None = None
    error: str = ""
    cached: bool = False
    timestamp: str = field(default_factory=utc_now)

    @property
    def succeeded(self) -> bool:
        """Return whether the task outcome is a success."""
        return self.status is TaskOutcomeStatus.SUCCEEDED

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result."""
        return {
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "status": self.status.value,
            "payload": dict(self.payload),
            "evidence_refs": list(self.evidence_refs),
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "warnings": list(self.warnings),
            "resource_use": self.resource_use.to_dict(),
            "attempts": self.attempts,
            "failure_class": self.failure_class.value if self.failure_class else None,
            "error": self.error,
            "cached": self.cached,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class OutcomeReport:
    """Close-out report distinguishing completed, partial, failed, unverified (Step 50)."""

    completed_task_ids: tuple[str, ...] = ()
    failed_task_ids: tuple[str, ...] = ()
    cancelled_task_ids: tuple[str, ...] = ()
    unverified_task_ids: tuple[str, ...] = ()
    deliverables: Mapping[str, Any] = field(default_factory=dict)

    @property
    def overall(self) -> OutcomeStatus:
        """Return the distinct overall outcome category."""
        if self.unverified_task_ids and not self.failed_task_ids:
            return OutcomeStatus.UNVERIFIED
        if self.failed_task_ids and self.completed_task_ids:
            return OutcomeStatus.PARTIAL
        if self.failed_task_ids and not self.completed_task_ids:
            return OutcomeStatus.FAILED
        return OutcomeStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report."""
        return {
            "overall": self.overall.value,
            "completed_task_ids": list(self.completed_task_ids),
            "failed_task_ids": list(self.failed_task_ids),
            "cancelled_task_ids": list(self.cancelled_task_ids),
            "unverified_task_ids": list(self.unverified_task_ids),
            "deliverables": dict(self.deliverables),
        }
