"""Typed, serializable contracts for the ORE cognitive control layer (Point 02 §3).

The initial schema is deliberately small and versioned: a trigger envelope
(Step 1), the :class:`CognitiveContext`, :class:`OperatorSpec`,
:class:`RetrievalCandidate`, and :class:`DecisionTrace` records, plus the
final control-run result. Serialization is JSON-compatible and round-trips
across the declared schema version.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4

from ore_workspace.models import utc_now

CONTROL_PACKAGE_VERSION = "0.1.0"
COGNITIVE_CONTROL_SCHEMA_VERSION = "1.0.0"
COGNITIVE_CONTROL_PRODUCER = "cognitive_control"


# --------------------------------------------------------------------- helpers


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize a mapping with sorted keys and stable separators."""
    return json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    """Coerce uncommon types into JSON-compatible forms without inventing data."""
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of a canonical JSON payload."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def new_control_id(prefix: str) -> str:
    """Create a stable unique identifier with a descriptive prefix."""
    return f"{prefix}_{uuid4().hex}"


def _require_finite(value: float, name: str, *, minimum: float | None = None) -> float:
    """Validate that a numeric contract field is finite and bounded."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


# ------------------------------------------------------------------ Step 1 ---


class TriggerKind(str, Enum):
    """Classify the supported cognitive trigger sources (Step 1)."""

    USER_TASK = "user_task"
    WORKSPACE_EVENT = "workspace_event"
    SUBTASK_COMPLETED = "subtask_completed"
    CONFLICT_NOTIFICATION = "conflict_notification"
    RESOURCE_EVENT = "resource_event"
    SCHEDULED_CONTINUATION = "scheduled_continuation"


@dataclass(frozen=True, slots=True)
class TriggerEnvelope:
    """Normalize one cognitive trigger with its idempotency key (Step 1).

    Malformed triggers are rejected before any work is created: the payload must
    name its source kind, carry a non-empty idempotency key and event reference,
    and provide finite priority when supplied.
    """

    kind: TriggerKind
    idempotency_key: str
    event_reference: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    entity_references: tuple[str, ...] = ()
    priority: float = 0.5
    occurred_at: str = field(default_factory=utc_now)
    schema_version: str = COGNITIVE_CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the trigger before it can create work."""
        if not isinstance(self.kind, TriggerKind):
            raise TypeError("trigger kind must be a TriggerKind")
        if not isinstance(self.idempotency_key, str) or not self.idempotency_key.strip():
            raise ValueError("idempotency_key must be a non-empty string")
        if not isinstance(self.event_reference, str) or not self.event_reference.strip():
            raise ValueError("event_reference must be a non-empty string")
        if not isinstance(self.payload, Mapping):
            raise TypeError("trigger payload must be a mapping")
        _require_finite(self.priority, "trigger priority")
        if not 0.0 <= float(self.priority) <= 1.0:
            raise ValueError("trigger priority must lie in [0, 1]")
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "entity_references", tuple(self.entity_references))

    def fingerprint(self) -> str:
        """Return the stable content fingerprint used for idempotent intake."""
        return sha256_json({
            "kind": self.kind.value,
            "idempotency_key": self.idempotency_key,
            "event_reference": self.event_reference,
            "entity_references": sorted(self.entity_references),
            "payload": dict(self.payload),
            "schema_version": self.schema_version,
        })

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable trigger record."""
        return {
            "kind": self.kind.value,
            "idempotency_key": self.idempotency_key,
            "event_reference": self.event_reference,
            "payload": dict(self.payload),
            "entity_references": tuple(self.entity_references),
            "priority": float(self.priority),
            "occurred_at": self.occurred_at,
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint(),
        }


# ------------------------------------------------------------------- §3.1 ----


class ContextStatus(str, Enum):
    """Lifecycle states of one cognitive context."""

    READY = "READY"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


_NONTERMINAL_CONTEXT_STATUSES = frozenset({
    ContextStatus.READY, ContextStatus.RUNNING, ContextStatus.WAITING,
})


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Remaining bounded budget attached to one context."""

    time_seconds: float = 30.0
    operations: int = 40
    memory_mb: float = 256.0
    concurrency: int = 2

    def __post_init__(self) -> None:
        """Validate that budgets are finite and non-negative."""
        self._check("time_seconds", self.time_seconds, positive=True)
        self._check("operations", self.operations, positive=True, integer=True)
        self._check("memory_mb", self.memory_mb, positive=True)
        self._check("concurrency", self.concurrency, positive=True, integer=True)

    @staticmethod
    def _check(name: str, value: Any, *, positive: bool, integer: bool = False) -> None:
        if integer and not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        if positive and number <= 0:
            raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable budget record."""
        return {
            "time_seconds": float(self.time_seconds),
            "operations": int(self.operations),
            "memory_mb": float(self.memory_mb),
            "concurrency": int(self.concurrency),
        }


@dataclass(frozen=True, slots=True)
class OpenQuestion:
    """One unresolved question with priority and evidence requirements."""

    question_id: str
    text: str
    priority: float
    required_evidence: tuple[str, ...] = ()
    resolved: bool = False

    def __post_init__(self) -> None:
        """Validate the question record."""
        if not self.question_id.strip():
            raise ValueError("question_id must be non-empty")
        if not self.text.strip():
            raise ValueError("question text must be non-empty")
        _require_finite(self.priority, "question priority")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("question priority must lie in [0, 1]")
        object.__setattr__(self, "required_evidence", tuple(self.required_evidence))

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable question record."""
        return {
            "question_id": self.question_id,
            "text": self.text,
            "priority": float(self.priority),
            "required_evidence": tuple(self.required_evidence),
            "resolved": bool(self.resolved),
        }


@dataclass(frozen=True, slots=True)
class CognitiveContext:
    """One bounded problem-solving context (Point 02 §3.1).

    A context references - never copies - the authoritative workspace goal and
    hypothesis records. It tracks its own remaining budget, subproblem depth,
    and a stable cycle fingerprint used for stagnation detection.
    """

    context_id: str
    goal_id: str
    workspace_revision: int
    status: ContextStatus = ContextStatus.READY
    parent_context_id: str | None = None
    active_hypotheses: tuple[str, ...] = ()
    open_questions: tuple[OpenQuestion, ...] = ()
    budget: ContextBudget = field(default_factory=ContextBudget)
    depth: int = 0
    cycle_fingerprint: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_version: str = COGNITIVE_CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate context identity, depth, and hypothesis references."""
        if not isinstance(self.context_id, str) or not self.context_id.strip():
            raise ValueError("context_id must be a non-empty string")
        if not isinstance(self.goal_id, str) or not self.goal_id.strip():
            raise ValueError("goal_id must be a non-empty identifier")
        if not isinstance(self.workspace_revision, int) or self.workspace_revision < 0:
            raise ValueError("workspace_revision must be a non-negative integer")
        if not isinstance(self.status, ContextStatus):
            raise TypeError("context status must be a ContextStatus")
        if self.depth < 0:
            raise ValueError("context depth cannot be negative")
        if self.parent_context_id is not None and self.depth < 1:
            raise ValueError("a child context must have depth >= 1")
        object.__setattr__(self, "active_hypotheses", tuple(self.active_hypotheses))
        object.__setattr__(self, "open_questions", tuple(self.open_questions))

    # ------------------------------------------------------------ lifecycle

    @property
    def nonterminal(self) -> bool:
        """Return whether this context can still be resumed (Step 4)."""
        return self.status in _NONTERMINAL_CONTEXT_STATUSES

    def with_updates(
        self,
        *,
        status: ContextStatus | None = None,
        workspace_revision: int | None = None,
        active_hypotheses: tuple[str, ...] | None = None,
        open_questions: tuple[OpenQuestion, ...] | None = None,
        budget: ContextBudget | None = None,
        cycle_fingerprint: str | None = None,
    ) -> "CognitiveContext":
        """Return an updated copy without mutating the stored context (Step 4)."""
        return CognitiveContext(
            context_id=self.context_id,
            goal_id=self.goal_id,
            workspace_revision=(
                self.workspace_revision if workspace_revision is None else workspace_revision
            ),
            status=self.status if status is None else status,
            parent_context_id=self.parent_context_id,
            active_hypotheses=(
                self.active_hypotheses if active_hypotheses is None else active_hypotheses
            ),
            open_questions=self.open_questions if open_questions is None else open_questions,
            budget=self.budget if budget is None else budget,
            depth=self.depth,
            cycle_fingerprint=self.cycle_fingerprint if cycle_fingerprint is None
            else cycle_fingerprint,
            created_at=self.created_at,
            updated_at=utc_now(),
            schema_version=self.schema_version,
        )

    # ---------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable context record (schema-versioned)."""
        return {
            "context_id": self.context_id,
            "parent_context_id": self.parent_context_id,
            "goal_id": self.goal_id,
            "workspace_revision": self.workspace_revision,
            "status": self.status.value,
            "active_hypotheses": list(self.active_hypotheses),
            "open_questions": [item.to_dict() for item in self.open_questions],
            "budget": self.budget.to_dict(),
            "depth": self.depth,
            "cycle_fingerprint": self.cycle_fingerprint,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "CognitiveContext":
        """Rebuild a context from a serialized record of the same schema version."""
        version = str(record.get("schema_version", ""))
        if version != COGNITIVE_CONTROL_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported context schema version: {version!r}; expected "
                f"{COGNITIVE_CONTROL_SCHEMA_VERSION!r}"
            )
        return cls(
            context_id=str(record["context_id"]),
            goal_id=str(record["goal_id"]),
            workspace_revision=int(record["workspace_revision"]),
            status=ContextStatus(str(record["status"])),
            parent_context_id=record.get("parent_context_id"),
            active_hypotheses=tuple(record.get("active_hypotheses", ())),
            open_questions=tuple(
                OpenQuestion(
                    question_id=str(item["question_id"]),
                    text=str(item["text"]),
                    priority=float(item["priority"]),
                    required_evidence=tuple(item.get("required_evidence", ())),
                    resolved=bool(item.get("resolved", False)),
                )
                for item in record.get("open_questions", [])
            ),
            budget=ContextBudget(**{
                key: value for key, value in record.get("budget", {}).items()
            }),
            depth=int(record.get("depth", 0)),
            cycle_fingerprint=str(record.get("cycle_fingerprint", "")),
            created_at=str(record.get("created_at", utc_now())),
            updated_at=str(record.get("updated_at", utc_now())),
            schema_version=version,
        )


# ------------------------------------------------------------------- §3.2 ----


class RiskClass(str, Enum):
    """Operator risk classification used for constraints and approval policy."""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry, timeout, and idempotency policy for one operator."""

    timeout_seconds: float = 5.0
    max_retries: int = 2
    backoff_initial_seconds: float = 0.01
    backoff_cap_seconds: float = 0.5
    idempotent: bool = True

    def __post_init__(self) -> None:
        """Validate retry policy bounds."""
        _require_finite(self.timeout_seconds, "timeout_seconds", minimum=0.0)
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        _require_finite(
            self.backoff_initial_seconds, "backoff_initial_seconds", minimum=0.0,
        )
        _require_finite(self.backoff_cap_seconds, "backoff_cap_seconds", minimum=0.0)


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Declarative specification of one cognitive operator (Point 02 §3.2).

    An operator is a decision-level cognitive action; actual work is delegated
    to the existing coordinators through the referenced handler.
    """

    operator_id: str
    version: str
    purpose: str
    preconditions: tuple[str, ...] = ()
    input_schema_id: str = "control.operator.input/1.0"
    output_schema_id: str = "control.operator.output/1.0"
    required_capabilities: tuple[str, ...] = ()
    estimated_time_seconds: float = 0.05
    estimated_memory_mb: float = 4.0
    estimated_compute: float = 0.1
    risk_class: RiskClass = RiskClass.LOW
    reversible: bool = True
    expected_progress: float = 0.5
    expected_information_gain: float = 0.5
    writes_workspace: bool = True
    write_fields: tuple[str, ...] = ()
    read_fields: tuple[str, ...] = ()
    validation_policy: str = "deterministic_contract_checks"
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    handler_reference: str = ""
    processing_path: str = "EXPLICIT_SYMBOLIC"
    provenance_annotation: str = "ore.control.operator"
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate the operator declaration (Step 6)."""
        if not self.operator_id.strip():
            raise ValueError("operator_id must be non-empty")
        if not self.version.strip():
            raise ValueError("operator version must be non-empty")
        if not self.purpose.strip():
            raise ValueError("operator purpose must be non-empty")
        if not isinstance(self.risk_class, RiskClass):
            raise TypeError("risk_class must be a RiskClass")
        for name in ("estimated_time_seconds", "estimated_memory_mb", "estimated_compute"):
            _require_finite(getattr(self, name), name, minimum=0.0)
        for name in ("expected_progress", "expected_information_gain"):
            _require_finite(getattr(self, name), name, minimum=0.0)
        if not 0.0 <= self.expected_progress <= 1.0:
            raise ValueError("expected_progress must lie in [0, 1]")
        if not 0.0 <= self.expected_information_gain <= 1.0:
            raise ValueError("expected_information_gain must lie in [0, 1]")
        object.__setattr__(self, "preconditions", tuple(self.preconditions))
        object.__setattr__(self, "required_capabilities", tuple(self.required_capabilities))
        object.__setattr__(self, "write_fields", tuple(self.write_fields))
        object.__setattr__(self, "read_fields", tuple(self.read_fields))
        if self.writes_workspace and not self.write_fields:
            raise ValueError("a workspace-writing operator must declare its write fields")

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable operator declaration."""
        return {
            "operator_id": self.operator_id,
            "version": self.version,
            "purpose": self.purpose,
            "preconditions": list(self.preconditions),
            "input_schema_id": self.input_schema_id,
            "output_schema_id": self.output_schema_id,
            "required_capabilities": list(self.required_capabilities),
            "estimated_time_seconds": float(self.estimated_time_seconds),
            "estimated_memory_mb": float(self.estimated_memory_mb),
            "estimated_compute": float(self.estimated_compute),
            "risk_class": self.risk_class.value,
            "reversible": bool(self.reversible),
            "expected_progress": float(self.expected_progress),
            "expected_information_gain": float(self.expected_information_gain),
            "writes_workspace": bool(self.writes_workspace),
            "write_fields": list(self.write_fields),
            "read_fields": list(self.read_fields),
            "validation_policy": self.validation_policy,
            "retry_policy": {
                "timeout_seconds": self.retry_policy.timeout_seconds,
                "max_retries": self.retry_policy.max_retries,
                "backoff_initial_seconds": self.retry_policy.backoff_initial_seconds,
                "backoff_cap_seconds": self.retry_policy.backoff_cap_seconds,
                "idempotent": self.retry_policy.idempotent,
            },
            "handler_reference": self.handler_reference,
            "processing_path": self.processing_path,
            "provenance_annotation": self.provenance_annotation,
            "enabled": bool(self.enabled),
        }


# ------------------------------------------------------------------- §3.3 ----


class AccessState(str, Enum):
    """Accessibility state of one retrieval candidate."""

    AVAILABLE = "AVAILABLE"
    RESTRICTED = "RESTRICTED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """One candidate memory or knowledge item with scoring inputs (Point 02 §3.3)."""

    record_reference: str
    source_type: str
    cue_match_features: Mapping[str, float] = field(default_factory=dict)
    usage_timestamps: tuple[float, ...] | None = None
    semantic_relevance: float = 0.0
    confidence: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    retrieval_cost: float = 0.0
    access_state: AccessState = AccessState.AVAILABLE
    version_reference: str = ""
    inclusion_reasons: tuple[str, ...] = ()
    exclusion_reasons: tuple[str, ...] = ()
    base_level_activation: float | None = None
    activation_available: bool = True
    cue_association: float | None = None
    cue_contributions: Mapping[str, float] = field(default_factory=dict)
    retrieval_score: float | None = None
    dropped_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the candidate record structure (Step 12)."""
        if not self.record_reference.strip():
            raise ValueError("record_reference must be non-empty")
        if not self.source_type.strip():
            raise ValueError("source_type must be non-empty")
        if not isinstance(self.cue_match_features, Mapping):
            raise TypeError("cue_match_features must be a mapping")
        for cue, strength in self.cue_match_features.items():
            _require_finite(strength, f"cue match strength {cue!r}")
            if float(strength) < 0:
                raise ValueError("cue match strengths must be non-negative")
        if self.semantic_relevance < 0:
            raise ValueError("semantic_relevance must be non-negative")
        _require_finite(self.retrieval_cost, "retrieval_cost", minimum=0.0)
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("candidate confidence must lie in [0, 1]")
        if not isinstance(self.access_state, AccessState):
            raise TypeError("access_state must be an AccessState")
        object.__setattr__(self, "cue_match_features", dict(self.cue_match_features))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "usage_timestamps", (
            None if self.usage_timestamps is None else tuple(self.usage_timestamps)
        ))
        object.__setattr__(self, "inclusion_reasons", tuple(self.inclusion_reasons))
        object.__setattr__(self, "exclusion_reasons", tuple(self.exclusion_reasons))
        object.__setattr__(self, "cue_contributions", dict(self.cue_contributions))
        object.__setattr__(self, "dropped_terms", tuple(self.dropped_terms))
        if self.usage_timestamps is None:
            object.__setattr__(self, "activation_available", False)

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable candidate record with score provenance."""
        return {
            "record_reference": self.record_reference,
            "source_type": self.source_type,
            "cue_match_features": dict(self.cue_match_features),
            "usage_timestamps": (
                None if self.usage_timestamps is None else list(self.usage_timestamps)
            ),
            "semantic_relevance": float(self.semantic_relevance),
            "confidence": self.confidence,
            "provenance": dict(self.provenance),
            "retrieval_cost": float(self.retrieval_cost),
            "access_state": self.access_state.value,
            "version_reference": self.version_reference,
            "inclusion_reasons": list(self.inclusion_reasons),
            "exclusion_reasons": list(self.exclusion_reasons),
            "base_level_activation": self.base_level_activation,
            "activation_available": bool(self.activation_available),
            "cue_association": self.cue_association,
            "cue_contributions": dict(self.cue_contributions),
            "retrieval_score": self.retrieval_score,
            "dropped_terms": list(self.dropped_terms),
        }


# ------------------------------------------------------------------- §3.4 ----


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One considered operator candidate with its evaluation trail (Steps 8-9)."""

    operator_id: str
    operator_version: str
    normalized_arguments: Mapping[str, Any]
    arguments_fingerprint: str
    eligibility_reasons: tuple[str, ...] = ()
    precondition_failures: tuple[str, ...] = ()
    constraint_rejections: tuple[str, ...] = ()
    expected_utility: float | None = None
    normalized_utility: float | None = None

    def __post_init__(self) -> None:
        """Validate the candidate record."""
        if not self.operator_id.strip():
            raise ValueError("operator_id must be non-empty")
        object.__setattr__(self, "normalized_arguments", dict(self.normalized_arguments))
        object.__setattr__(self, "eligibility_reasons", tuple(self.eligibility_reasons))
        object.__setattr__(self, "precondition_failures", tuple(self.precondition_failures))
        object.__setattr__(self, "constraint_rejections", tuple(self.constraint_rejections))

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable candidate record."""
        return {
            "operator_id": self.operator_id,
            "operator_version": self.operator_version,
            "normalized_arguments": dict(self.normalized_arguments),
            "arguments_fingerprint": self.arguments_fingerprint,
            "eligibility_reasons": list(self.eligibility_reasons),
            "precondition_failures": list(self.precondition_failures),
            "constraint_rejections": list(self.constraint_rejections),
            "expected_utility": self.expected_utility,
            "normalized_utility": self.normalized_utility,
        }


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    """The selected action of one cycle with its rationale (Step 21)."""

    selected_operator_id: str
    selected_arguments_fingerprint: str
    rationale: str
    parallel_set: tuple[str, ...] = ()
    rejected_alternatives: tuple[str, ...] = ()
    utility_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable selection record."""
        return {
            "selected_operator_id": self.selected_operator_id,
            "selected_arguments_fingerprint": self.selected_arguments_fingerprint,
            "rationale": self.rationale,
            "parallel_set": list(self.parallel_set),
            "rejected_alternatives": list(self.rejected_alternatives),
            "utility_score": self.utility_score,
        }


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """One recorded cognitive cycle decision (Point 02 §3.4).

    The trace keeps the eligible and rejected operators with machine-readable
    reasons, candidate retrievals with scores, resource estimates, the selected
    action, execution and validation outcomes, and the next-cycle reason.
    Payloads stay compact: hashes, IDs, and reasons rather than large inputs.
    """

    trace_id: str
    cycle: int
    context_id: str
    workspace_revision: int
    goal_id: str
    open_goal_status: str
    open_question_ids: tuple[str, ...] = ()
    candidates: tuple[CandidateRecord, ...] = ()
    retrieval_candidates: tuple[Mapping[str, Any], ...] = ()
    retrieval_shortlist_ids: tuple[str, ...] = ()
    retrieval_ambiguities: tuple[Mapping[str, Any], ...] = ()
    resource_estimates: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    selection: SelectionRecord | None = None
    execution_result: Mapping[str, Any] = field(default_factory=dict)
    validation_result: Mapping[str, Any] = field(default_factory=dict)
    integration_decisions: tuple[Mapping[str, Any], ...] = ()
    state_transition: str = ""
    next_cycle_reason: str = ""
    timing_ms: float = 0.0
    cost_measurements: Mapping[str, float] = field(default_factory=dict)
    metacognitive_report: Mapping[str, Any] = field(default_factory=dict)
    operation_steps: tuple[Mapping[str, Any], ...] = ()
    created_at: str = field(default_factory=utc_now)
    schema_version: str = COGNITIVE_CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize the trace collections."""
        object.__setattr__(self, "open_question_ids", tuple(self.open_question_ids))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "retrieval_candidates", tuple(
            dict(item) for item in self.retrieval_candidates
        ))
        object.__setattr__(self, "retrieval_shortlist_ids", tuple(self.retrieval_shortlist_ids))
        object.__setattr__(self, "retrieval_ambiguities", tuple(
            dict(item) for item in self.retrieval_ambiguities
        ))
        object.__setattr__(self, "resource_estimates", {
            key: dict(value) for key, value in self.resource_estimates.items()
        })
        object.__setattr__(self, "execution_result", dict(self.execution_result))
        object.__setattr__(self, "validation_result", dict(self.validation_result))
        object.__setattr__(self, "integration_decisions", tuple(
            dict(item) for item in self.integration_decisions
        ))
        object.__setattr__(self, "cost_measurements", dict(self.cost_measurements))
        object.__setattr__(self, "metacognitive_report", dict(self.metacognitive_report))
        object.__setattr__(self, "operation_steps", tuple(
            dict(item) for item in self.operation_steps
        ))
        _require_finite(self.timing_ms, "trace timing_ms", minimum=0.0)

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable decision-trace record."""
        return {
            "trace_id": self.trace_id,
            "cycle": int(self.cycle),
            "context_id": self.context_id,
            "workspace_revision": int(self.workspace_revision),
            "goal_id": self.goal_id,
            "open_goal_status": self.open_goal_status,
            "open_question_ids": list(self.open_question_ids),
            "candidates": [item.to_dict() for item in self.candidates],
            "retrieval_candidates": [dict(item) for item in self.retrieval_candidates],
            "retrieval_shortlist_ids": list(self.retrieval_shortlist_ids),
            "retrieval_ambiguities": [dict(item) for item in self.retrieval_ambiguities],
            "resource_estimates": {
                key: dict(value) for key, value in self.resource_estimates.items()
            },
            "selection": self.selection.to_dict() if self.selection else None,
            "execution_result": dict(self.execution_result),
            "validation_result": dict(self.validation_result),
            "integration_decisions": [
                dict(item) for item in self.integration_decisions
            ],
            "state_transition": self.state_transition,
            "next_cycle_reason": self.next_cycle_reason,
            "timing_ms": float(self.timing_ms),
            "cost_measurements": dict(self.cost_measurements),
            "metacognitive_report": dict(self.metacognitive_report),
            "operation_steps": [dict(item) for item in self.operation_steps],
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }


# --------------------------------------------------------------- final result -


@dataclass(frozen=True, slots=True)
class ControlRunResult:
    """Final outcome of one control-layer run over a trigger."""

    trigger_fingerprint: str
    run_id: str
    context_id: str
    status: str
    status_reason: str
    cycles: int
    operator_applications: int
    retrievals: int
    retrieval_cost: float
    child_contexts: int
    conflicts_preserved: int
    learned_proposals_used: int
    escalations: tuple[str, ...] = ()
    goal_outcome: str = ""
    goal_score: float = 0.0
    goal_threshold: float = 0.0
    missing_requirements: tuple[str, ...] = ()
    uncertainty_reasons: tuple[str, ...] = ()
    decision_traces: tuple[Mapping[str, Any], ...] = ()
    procedure_events: tuple[Mapping[str, Any], ...] = ()
    workspace_id: str = ""
    workspace_revision: int = 0
    elapsed_ms: float = 0.0
    reproducibility: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable run result."""
        return {
            "trigger_fingerprint": self.trigger_fingerprint,
            "run_id": self.run_id,
            "context_id": self.context_id,
            "status": self.status,
            "status_reason": self.status_reason,
            "cycles": int(self.cycles),
            "operator_applications": int(self.operator_applications),
            "retrievals": int(self.retrievals),
            "retrieval_cost": float(self.retrieval_cost),
            "child_contexts": int(self.child_contexts),
            "conflicts_preserved": int(self.conflicts_preserved),
            "learned_proposals_used": int(self.learned_proposals_used),
            "escalations": list(self.escalations),
            "goal_outcome": self.goal_outcome,
            "goal_score": float(self.goal_score),
            "goal_threshold": float(self.goal_threshold),
            "missing_requirements": list(self.missing_requirements),
            "uncertainty_reasons": list(self.uncertainty_reasons),
            "decision_traces": [dict(item) for item in self.decision_traces],
            "procedure_events": [dict(item) for item in self.procedure_events],
            "workspace_id": self.workspace_id,
            "workspace_revision": int(self.workspace_revision),
            "elapsed_ms": float(self.elapsed_ms),
            "reproducibility": dict(self.reproducibility),
        }


__all__ = [
    "AccessState",
    "COGNITIVE_CONTROL_PRODUCER",
    "COGNITIVE_CONTROL_SCHEMA_VERSION",
    "CONTROL_PACKAGE_VERSION",
    "CandidateRecord",
    "CognitiveContext",
    "ContextBudget",
    "ContextStatus",
    "ControlRunResult",
    "DecisionTrace",
    "OpenQuestion",
    "RetrievalCandidate",
    "RetryPolicy",
    "RiskClass",
    "SelectionRecord",
    "TriggerEnvelope",
    "TriggerKind",
    "canonical_json",
    "new_control_id",
    "sha256_json",
]
