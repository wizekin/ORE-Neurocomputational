"""Core MACL data contracts: descriptors, tasks, messages, results, and lifecycle states.

These contracts implement Section 2 and Section 4A of the ranked Point 11 (source
Point 11, sections 11.1-11.5) multi-agent coordination specification. An ORE agent
is a bounded decision-making participant with an explicit role, capability contract,
local state, permitted observations and actions, communication rules, a measurable
result contract, resource/time/retry limits, and a provenance trail.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ore_workspace.models import new_id, utc_now


class AutonomyLevel(str, Enum):
    """Classify how much self-directed action a component is permitted (spec Step 1)."""

    DETERMINISTIC_SPECIALIST = "deterministic_specialist"
    TOOL_USING_WORKER = "tool_using_worker"
    BOUNDED_AUTONOMOUS_AGENT = "bounded_autonomous_agent"
    SUPERVISOR = "supervisor"


class AgentLifecycleState(str, Enum):
    """Track agent health states without erasing task history (spec Step 5)."""

    REGISTERED = "registered"
    READY = "ready"
    BUSY = "busy"
    DEGRADED = "degraded"
    QUARANTINED = "quarantined"
    DISABLED = "disabled"


class MessageKind(str, Enum):
    """Enumerate the typed agent-message envelope kinds (spec Section 2, AgentMessage)."""

    REQUEST = "request"
    OFFER = "offer"
    ACCEPT = "accept"
    REJECT = "reject"
    PROGRESS = "progress"
    EVIDENCE_PROPOSAL = "evidence_proposal"
    FINDING = "finding"
    CONFLICT = "conflict"
    CLARIFICATION = "clarification"
    CANCELLATION = "cancellation"
    COMPLETION = "completion"
    FAILURE = "failure"
    HEARTBEAT = "heartbeat"


class AgentTaskKind(str, Enum):
    """Separate bounded-agent work from deterministic module execution (spec Section 3)."""

    AUTONOMOUS = "autonomous"
    DETERMINISTIC_MODULE = "deterministic_module"


class TaskTopology(str, Enum):
    """Select a coordination topology with an explicit justification (spec Step 16)."""

    HIERARCHICAL = "hierarchical"
    MAP_REDUCE = "map_reduce"
    DEBATE = "debate"
    PEER_TO_PEER = "peer_to_peer"
    MARKET = "market"


class AdjudicationDecision(str, Enum):
    """Record the result adjudicator's decision separately from agent self-evaluation."""

    ACCEPTED = "accepted"
    NEEDS_REVISION = "needs_revision"
    DEFERRED = "deferred"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


class TaskOutcome(str, Enum):
    """Describe one attempt's terminal state, including unknown-after-timeout."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class DeliveryOutcome(str, Enum):
    """Distinguish duplicate, expired, and unknown-delivery message outcomes (spec Step 35)."""

    DELIVERED = "delivered"
    DUPLICATE = "duplicate"
    EXPIRED = "expired"
    QUOTA_EXCEEDED = "quota_exceeded"
    UNKNOWN = "unknown"


class PeerNegotiationOutcome(str, Enum):
    """Record bounded peer negotiation results and required escalation (spec Step 36)."""

    ASSISTED = "assisted"
    DECLINED = "declined"
    MAX_DEPTH_REACHED = "max_depth_reached"
    ESCALATED = "escalated"
    FAILED = "failed"


VALID_FIELD_TYPES = frozenset({
    "string", "integer", "number", "boolean", "list", "dict", "any",
})


def _validate_json_value(value: Any, label: str) -> None:
    """Reject payloads that cannot be serialized deterministically for fingerprints."""
    try:
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite JSON-compatible values") from exc


@dataclass(frozen=True, slots=True)
class SchemaSpec:
    """Define a machine-checkable structural schema for agent inputs and outputs (Step 3)."""

    fields: Mapping[str, str]
    optional_fields: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate declared field types and reject duplicate keys."""
        for label, mapping in (("fields", self.fields), ("optional_fields", self.optional_fields)):
            if not isinstance(mapping, Mapping):
                raise TypeError(f"{label} must be a mapping")
            for name, kind in mapping.items():
                if not str(name).strip():
                    raise ValueError(f"{label} keys must be non-empty")
                if kind not in VALID_FIELD_TYPES:
                    raise ValueError(f"field {name!r} declares unknown type {kind!r}")
        overlap = set(self.fields) & set(self.optional_fields)
        if overlap:
            raise ValueError(f"fields cannot be both required and optional: {sorted(overlap)}")

    def validate(self, payload: Mapping[str, Any]) -> tuple[str, ...]:
        """Return structured errors for missing keys or type mismatches."""
        if not isinstance(payload, Mapping):
            return ("payload must be a mapping",)
        errors: list[str] = []
        for name, kind in self.fields.items():
            if name not in payload:
                errors.append(f"missing required field: {name}")
                continue
            mismatch = self._type_error(payload[name], kind)
            if mismatch:
                errors.append(f"field {name}: {mismatch}")
        for name, kind in self.optional_fields.items():
            if name in payload:
                mismatch = self._type_error(payload[name], kind)
                if mismatch:
                    errors.append(f"field {name}: {mismatch}")
        return tuple(errors)

    @staticmethod
    def _type_error(value: Any, kind: str) -> str | None:
        """Compare one value against its declared structural type."""
        checks = {
            "string": lambda item: isinstance(item, str),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
            "boolean": lambda item: isinstance(item, bool),
            "list": lambda item: isinstance(item, (list, tuple)),
            "dict": lambda item: isinstance(item, Mapping),
            "any": lambda item: True,
        }
        return None if checks[kind](value) else f"expected {kind}, got {type(value).__name__}"

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible schema description."""
        return {"fields": dict(self.fields), "optional_fields": dict(self.optional_fields)}


@dataclass(frozen=True, slots=True)
class ResourceEnvelope:
    """Bound an agent's concurrency, time, retries, messages, and cost (spec Step 2)."""

    max_concurrent_tasks: int = 1
    timeout_seconds: float = 30.0
    max_retries: int = 1
    max_messages_per_window: int = 128
    cost_budget: float = 32.0

    def __post_init__(self) -> None:
        """Reject non-positive or negative envelope values."""
        if self.max_concurrent_tasks < 1:
            raise ValueError("max_concurrent_tasks must be at least one")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.max_messages_per_window < 1:
            raise ValueError("max_messages_per_window must be at least one")
        if self.cost_budget <= 0:
            raise ValueError("cost_budget must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached envelope description."""
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_messages_per_window": self.max_messages_per_window,
            "cost_budget": self.cost_budget,
        }


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry only transient, idempotent operations and reconcile before re-dispatch (Step 51)."""

    max_attempts: int = 2
    idempotent_only: bool = True

    def __post_init__(self) -> None:
        """Validate attempt bounds."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")

    def allows(self, attempt_count: int, *, idempotent: bool, transient: bool) -> bool:
        """Decide whether one more retry is permitted under the declared policy."""
        if not transient:
            return False
        if self.idempotent_only and not idempotent:
            return False
        return attempt_count < self.max_attempts


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Reference workspace evidence compactly instead of embedding payloads (Step 25)."""

    artifact_id: str
    payload_hash: str
    location: str

    def __post_init__(self) -> None:
        """Require non-empty reference fields."""
        for name in ("artifact_id", "payload_hash", "location"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached reference record."""
        return {"artifact_id": self.artifact_id, "payload_hash": self.payload_hash, "location": self.location}


@dataclass(frozen=True, slots=True)
class AgentDescriptor:
    """Register identity, role, capabilities, schemas, permissions, and limits (Step 2)."""

    agent_id: str
    agent_version: str
    role: str
    autonomy: AutonomyLevel
    capability_ids: tuple[str, ...]
    accepted_input_schema: SchemaSpec
    produced_output_schema: SchemaSpec
    permitted_workspace_fields: tuple[str, ...] = ()
    writable_workspace_fields: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    policy_version: str = "macl-policy-v1"
    resources: ResourceEnvelope = field(default_factory=ResourceEnvelope)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    requires_verifier: bool = False
    autonomy_justification: str = ""
    reliability_samples: int = 0
    accepted_result_rate: float | None = None

    def __post_init__(self) -> None:
        """Validate contract completeness and enforce justification for autonomy (Steps 1-3)."""
        for name in ("agent_id", "agent_version", "role", "policy_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.autonomy, AutonomyLevel):
            raise TypeError("autonomy must be an AutonomyLevel")
        if not self.capability_ids:
            raise ValueError("capability_ids must not be empty")
        if len(set(self.capability_ids)) != len(self.capability_ids):
            raise ValueError("capability_ids must not contain duplicates")
        if len(set(self.permitted_workspace_fields)) != len(self.permitted_workspace_fields):
            raise ValueError("permitted_workspace_fields must not contain duplicates")
        if len(set(self.writable_workspace_fields)) != len(self.writable_workspace_fields):
            raise ValueError("writable_workspace_fields must not contain duplicates")
        unsupported = set(self.writable_workspace_fields) - set(self.permitted_workspace_fields)
        if unsupported:
            raise ValueError(
                "least privilege violated: writable fields must be readable: "
                f"{sorted(unsupported)}"
            )
        if self.autonomy is AutonomyLevel.BOUNDED_AUTONOMOUS_AGENT and not self.autonomy_justification.strip():
            raise ValueError("bounded autonomy requires an explicit justification")
        if self.autonomy is AutonomyLevel.SUPERVISOR and not self.autonomy_justification.strip():
            raise ValueError("supervisor autonomy requires an explicit justification")
        if self.reliability_samples < 0:
            raise ValueError("reliability_samples cannot be negative")
        if self.accepted_result_rate is not None and not 0.0 <= self.accepted_result_rate <= 1.0:
            raise ValueError("accepted_result_rate must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible descriptor."""
        return {
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "role": self.role,
            "autonomy": self.autonomy.value,
            "capability_ids": list(self.capability_ids),
            "accepted_input_schema": self.accepted_input_schema.to_dict(),
            "produced_output_schema": self.produced_output_schema.to_dict(),
            "permitted_workspace_fields": list(self.permitted_workspace_fields),
            "writable_workspace_fields": list(self.writable_workspace_fields),
            "tools": list(self.tools),
            "policy_version": self.policy_version,
            "resources": self.resources.to_dict(),
            "retry": {"max_attempts": self.retry.max_attempts, "idempotent_only": self.retry.idempotent_only},
            "requires_verifier": self.requires_verifier,
            "autonomy_justification": self.autonomy_justification,
            "reliability_samples": self.reliability_samples,
            "accepted_result_rate": self.accepted_result_rate,
        }


@dataclass(frozen=True, slots=True)
class SuccessCriteria:
    """Declare objective, machine-checkable acceptance conditions (Steps 8 and 23)."""

    required_output_keys: tuple[str, ...] = ()
    minimum_confidence: float = 0.0
    must_cite_evidence: bool = False
    forbidden_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate criteria bounds and duplicates."""
        if len(set(self.required_output_keys)) != len(self.required_output_keys):
            raise ValueError("required_output_keys must not contain duplicates")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached criteria description."""
        return {
            "required_output_keys": list(self.required_output_keys),
            "minimum_confidence": self.minimum_confidence,
            "must_cite_evidence": self.must_cite_evidence,
            "forbidden_actions": list(self.forbidden_actions),
        }


@dataclass(frozen=True, slots=True)
class AgentTask:
    """Bind a bounded objective to output contracts, budgets, and dependencies (Step 8)."""

    task_id: str
    goal_id: str
    objective: str
    required_capabilities: tuple[str, ...]
    expected_result_schema: SchemaSpec
    workspace_revision: int = 0
    parent_task_id: str | None = None
    dependencies: tuple[str, ...] = ()
    kind: AgentTaskKind = AgentTaskKind.AUTONOMOUS
    topology: TaskTopology = TaskTopology.HIERARCHICAL
    topology_justification: str = ""
    deadline_seconds: float = 30.0
    cost_budget: float = 16.0
    priority: float = 0.5
    idempotency_key: str = ""
    success_criteria: SuccessCriteria = field(default_factory=SuccessCriteria)
    permitted_workspace_fields: tuple[str, ...] = ()
    requires_independent_verification: bool = False
    verifier_requirements: tuple[str, ...] = ()
    module_route: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the task envelope and assign a stable idempotency key."""
        for name in ("task_id", "goal_id", "objective"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not self.required_capabilities:
            raise ValueError("required_capabilities must not be empty")
        if len(set(self.required_capabilities)) != len(self.required_capabilities):
            raise ValueError("required_capabilities must not contain duplicates")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("dependencies must not contain duplicates")
        if self.task_id in self.dependencies:
            raise ValueError("a task cannot depend on itself")
        if self.workspace_revision < 0:
            raise ValueError("workspace_revision cannot be negative")
        if self.deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be positive")
        if self.cost_budget <= 0:
            raise ValueError("cost_budget must be positive")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("priority must be in [0, 1]")
        if len(set(self.permitted_workspace_fields)) != len(self.permitted_workspace_fields):
            raise ValueError("permitted_workspace_fields must not contain duplicates")
        if self.topology is not TaskTopology.HIERARCHICAL and not self.topology_justification.strip():
            raise ValueError(f"{self.topology.value} topology requires an explicit justification")
        if self.topology is TaskTopology.MARKET and "cost" not in self.topology_justification.casefold():
            raise ValueError("market topology requires a bidding-cost justification")
        object.__setattr__(self, "module_route", copy.deepcopy(dict(self.module_route)))
        if not self.idempotency_key.strip():
            object.__setattr__(self, "idempotency_key", f"{self.task_id}:{self.goal_id}")

    def dependency_complete(self, accepted_task_ids: frozenset[str]) -> bool:
        """Check readiness using accepted artifacts only (Step 49)."""
        return all(dependency in accepted_task_ids for dependency in self.dependencies)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible task envelope."""
        return {
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "parent_task_id": self.parent_task_id,
            "objective": self.objective,
            "required_capabilities": list(self.required_capabilities),
            "expected_result_schema": self.expected_result_schema.to_dict(),
            "workspace_revision": self.workspace_revision,
            "dependencies": list(self.dependencies),
            "kind": self.kind.value,
            "topology": self.topology.value,
            "topology_justification": self.topology_justification,
            "deadline_seconds": self.deadline_seconds,
            "cost_budget": self.cost_budget,
            "priority": self.priority,
            "idempotency_key": self.idempotency_key,
            "success_criteria": self.success_criteria.to_dict(),
            "permitted_workspace_fields": list(self.permitted_workspace_fields),
            "requires_independent_verification": self.requires_independent_verification,
            "verifier_requirements": list(self.verifier_requirements),
            "module_route": dict(self.module_route),
        }


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """Wrap typed agent communication in a versioned, idempotent envelope (Section 2)."""

    sender_id: str
    recipient_id: str
    kind: MessageKind
    task_id: str
    schema_version: str = "macl-message-v1"
    message_id: str = field(default_factory=lambda: new_id("macl_message"))
    goal_id: str = ""
    correlation_id: str = ""
    topic: str = ""
    workspace_revision: int = 0
    artifact_refs: tuple[ArtifactRef, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    expires_at: float | None = None
    causal_parent_id: str | None = None
    sequence_no: int = 0
    idempotency_key: str = ""

    def __post_init__(self) -> None:
        """Validate envelope identity and detach payload data."""
        for name in ("sender_id", "recipient_id", "message_id", "task_id", "schema_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.kind, MessageKind):
            raise TypeError("kind must be a MessageKind")
        if self.sender_id == self.recipient_id and self.kind is not MessageKind.HEARTBEAT:
            raise ValueError("an agent cannot address itself except with a heartbeat")
        if self.workspace_revision < 0:
            raise ValueError("workspace_revision cannot be negative")
        if self.expires_at is not None and self.expires_at < self.timestamp:
            raise ValueError("expires_at cannot precede the message timestamp")
        if len({item.artifact_id for item in self.artifact_refs}) != len(self.artifact_refs):
            raise ValueError("artifact_refs must not contain duplicate artifact IDs")
        object.__setattr__(self, "payload", copy.deepcopy(dict(self.payload)))
        _validate_json_value(dict(self.payload), "payload")
        object.__setattr__(
            self, "idempotency_key",
            self.idempotency_key.strip() or f"{self.sender_id}:{self.kind.value}:{self.task_id}",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible message envelope."""
        return {
            "message_id": self.message_id,
            "schema_version": self.schema_version,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "kind": self.kind.value,
            "topic": self.topic,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "correlation_id": self.correlation_id,
            "workspace_revision": self.workspace_revision,
            "artifact_refs": [item.to_dict() for item in self.artifact_refs],
            "payload": copy.deepcopy(dict(self.payload)),
            "timestamp": self.timestamp,
            "expires_at": self.expires_at,
            "causal_parent_id": self.causal_parent_id,
            "sequence_no": self.sequence_no,
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True, slots=True)
class ProvenanceTrail:
    """Attach source artifacts, agent, task, operation, revision, and history (Step 26)."""

    agent_id: str
    task_id: str
    operation: str
    workspace_revision: int
    source_artifacts: tuple[ArtifactRef, ...] = ()
    transformation_history: tuple[str, ...] = ()
    model_policy_version: str = "macl-policy-v1"

    def __post_init__(self) -> None:
        """Validate trail fields and reject duplicates."""
        if not self.agent_id.strip() or not self.task_id.strip() or not self.operation.strip():
            raise ValueError("agent_id, task_id, and operation must be non-empty")
        if self.workspace_revision < 0:
            raise ValueError("workspace_revision cannot be negative")
        if len({item.artifact_id for item in self.source_artifacts}) != len(self.source_artifacts):
            raise ValueError("source_artifacts must not contain duplicate IDs")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached provenance record."""
        return {
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "operation": self.operation,
            "workspace_revision": self.workspace_revision,
            "source_artifacts": [item.to_dict() for item in self.source_artifacts],
            "transformation_history": list(self.transformation_history),
            "model_policy_version": self.model_policy_version,
        }


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Submit typed outputs with evidence, assumptions, uncertainty, and metrics (Step 22)."""

    agent_id: str
    agent_version: str
    task_id: str
    goal_id: str
    source_revision: int
    outputs: Mapping[str, Any]
    provenance: ProvenanceTrail
    status: str = "submitted"
    input_refs: tuple[ArtifactRef, ...] = ()
    evidence_refs: tuple[ArtifactRef, ...] = ()
    assumptions: tuple[str, ...] = ()
    uncertainty: float = 0.0
    resource_usage: float = 1.0
    elapsed_seconds: float = 0.0
    self_evaluation: str = ""
    self_confidence: float = 0.5
    limitations: tuple[str, ...] = ()
    unresolved_conflicts: tuple[str, ...] = ()
    verifier_requirements: tuple[str, ...] = ()
    submitted_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate result identity and detach outputs; self-evaluation is not acceptance."""
        for name in ("agent_id", "agent_version", "task_id", "goal_id", "status"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.provenance, ProvenanceTrail):
            raise TypeError("provenance must be a ProvenanceTrail")
        if self.source_revision < 0:
            raise ValueError("source_revision cannot be negative")
        if not isinstance(self.outputs, Mapping):
            raise TypeError("outputs must be a mapping")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")
        if not 0.0 <= self.self_confidence <= 1.0:
            raise ValueError("self_confidence must be in [0, 1]")
        if self.resource_usage < 0 or self.elapsed_seconds < 0:
            raise ValueError("resource_usage and elapsed_seconds cannot be negative")
        object.__setattr__(self, "outputs", copy.deepcopy(dict(self.outputs)))
        _validate_json_value(dict(self.outputs), "outputs")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible result envelope."""
        return {
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "source_revision": self.source_revision,
            "outputs": copy.deepcopy(dict(self.outputs)),
            "provenance": self.provenance.to_dict(),
            "status": self.status,
            "input_refs": [item.to_dict() for item in self.input_refs],
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "assumptions": list(self.assumptions),
            "uncertainty": self.uncertainty,
            "resource_usage": self.resource_usage,
            "elapsed_seconds": self.elapsed_seconds,
            "self_evaluation": self.self_evaluation,
            "self_confidence": self.self_confidence,
            "limitations": list(self.limitations),
            "unresolved_conflicts": list(self.unresolved_conflicts),
            "verifier_requirements": list(self.verifier_requirements),
            "submitted_at": self.submitted_at,
        }
