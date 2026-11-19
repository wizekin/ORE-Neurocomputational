"""Typed records shared by the ORE shared cognitive workspace."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    """Create a readable, collision-resistant identifier with a domain prefix."""
    return f"{prefix}_{uuid4().hex}"


def canonical_json(value: Any) -> str:
    """Serialize nested workspace data deterministically for fingerprints and snapshots."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def content_fingerprint(value: Any) -> str:
    """Return a SHA-256 fingerprint of a JSON-compatible value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class UpdateDecision(str, Enum):
    """Describe the state manager's decision for a published finding."""

    ACCEPTED = "accepted"
    MERGED = "merged"
    CONFLICT = "conflict"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    DUPLICATE = "duplicate"


class GoalOutcome(str, Enum):
    """Describe the result of evaluating a workspace against its current goal."""

    COMPLETED = "completed"
    COMPLETED_WITH_UNCERTAINTY = "completed_with_uncertainty"
    NEEDS_MORE_PROCESSING = "needs_more_processing"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FindingTarget:
    """Identify one workspace field and, optionally, a keyed cognitive object."""

    field: str
    key: str | None = None

    def __post_init__(self) -> None:
        """Reject empty or malformed target identifiers."""
        if not self.field.strip():
            raise ValueError("FindingTarget.field must be non-empty")
        if self.key is not None and not self.key.strip():
            raise ValueError("FindingTarget.key must be non-empty when supplied")


@dataclass(frozen=True, slots=True)
class Provenance:
    """Record where a finding came from and which evidence or rules produced it."""

    source: str
    source_span: tuple[int, int] | None = None
    rules_applied: tuple[str, ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        """Validate provenance source spans and normalize nested evidence safely."""
        if not self.source.strip():
            raise ValueError("Provenance.source must be non-empty")
        if self.source_span is not None:
            start, end = self.source_span
            if start < 0 or end < start:
                raise ValueError("source_span must be an increasing non-negative pair")
        object.__setattr__(self, "rules_applied", tuple(self.rules_applied))
        object.__setattr__(self, "evidence", tuple(copy.deepcopy(dict(item)) for item in self.evidence))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible provenance representation."""
        return {
            "source": self.source,
            "source_span": list(self.source_span) if self.source_span is not None else None,
            "rules_applied": list(self.rules_applied),
            "evidence": copy.deepcopy(list(self.evidence)),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class Finding:
    """Represent a proposed module contribution using the universal publication envelope."""

    workspace_id: str
    producer_module: str
    finding_type: str
    target: FindingTarget
    payload: Mapping[str, Any]
    provenance: Provenance
    base_workspace_revision: int
    finding_id: str = field(default_factory=lambda: new_id("finding"))
    dependencies: tuple[str, ...] = ()
    timestamp: str = field(default_factory=utc_now)
    confidence: float | None = None

    def __post_init__(self) -> None:
        """Validate the envelope and detach caller-owned payload data."""
        for name in ("workspace_id", "producer_module", "finding_type", "finding_id", "timestamp"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.base_workspace_revision < 0:
            raise ValueError("base_workspace_revision cannot be negative")
        if not isinstance(self.target, FindingTarget):
            raise TypeError("target must be a FindingTarget")
        if not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be a Provenance")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        object.__setattr__(self, "payload", copy.deepcopy(dict(self.payload)))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("dependencies must not contain duplicates")
        if self.finding_id in self.dependencies:
            raise ValueError("a finding cannot depend on itself")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible finding envelope."""
        return {
            "finding_id": self.finding_id,
            "workspace_id": self.workspace_id,
            "producer_module": self.producer_module,
            "finding_type": self.finding_type,
            "target": {"field": self.target.field, "key": self.target.key},
            "payload": copy.deepcopy(dict(self.payload)),
            "provenance": self.provenance.to_dict(),
            "dependencies": list(self.dependencies),
            "base_workspace_revision": self.base_workspace_revision,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }

    def fingerprint(self) -> str:
        """Fingerprint the proposed content independently of its unique finding ID."""
        data = self.to_dict()
        data.pop("finding_id", None)
        data.pop("timestamp", None)
        data.pop("base_workspace_revision", None)
        return content_fingerprint(data)


@dataclass(slots=True)
class WorkspaceState:
    """Hold the current shared cognitive state for one input-processing session."""

    workspace_id: str
    current_input: str
    current_goal: str
    input_revision: int = 1
    workspace_revision: int = 0
    lexical_hypotheses: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    syntactic_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    semantic_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    discourse_context: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    working_memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    retrieved_memories: list[dict[str, Any]] = field(default_factory=list)
    unresolved_questions: list[dict[str, Any]] = field(default_factory=list)
    module_status: dict[str, str] = field(default_factory=dict)
    pending_updates: list[str] = field(default_factory=list)
    accepted_updates: list[str] = field(default_factory=list)
    rejected_updates: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def clone(self) -> "WorkspaceState":
        """Return a deep copy so readers cannot mutate the live shared state."""
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dictionary suitable for snapshots and revision history."""
        return copy.deepcopy({name: getattr(self, name) for name in self.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    """Expose a stable, detached view of a workspace revision to consuming modules."""

    workspace_id: str
    revision: int
    state: Mapping[str, Any]
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Detach the state mapping when a snapshot is created."""
        object.__setattr__(self, "state", copy.deepcopy(dict(self.state)))

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the snapshot state and its revision metadata."""
        return {
            "workspace_id": self.workspace_id,
            "revision": self.revision,
            "state": copy.deepcopy(dict(self.state)),
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Report one validation or publication decision without hiding its rationale."""

    decision: UpdateDecision
    finding_id: str
    message: str
    workspace_revision: int
    changed_fields: tuple[str, ...] = ()
    conflict_id: str | None = None
    rebased_from_revision: int | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceEvent:
    """Describe a committed change for subscription-based module notification."""

    workspace_id: str
    workspace_revision: int
    event_type: str
    changed_fields: tuple[str, ...]
    producer_module: str
    finding_ids: tuple[str, ...]
    event_id: str = field(default_factory=lambda: new_id("event"))
    timestamp: str = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class ConflictRecord:
    """Retain incompatible findings together without deleting either hypothesis."""

    workspace_id: str
    conflicting_findings: tuple[str, ...]
    conflict_type: str
    affected_fields: tuple[str, ...]
    conflict_id: str = field(default_factory=lambda: new_id("conflict"))
    status: str = "unresolved"
    resolution_candidates: tuple[str, ...] = ()
    preferred_finding_id: str | None = None
    created_at: str = field(default_factory=utc_now)
    resolved_at: str | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceRevision:
    """Store a reconstructable workspace snapshot and the change that produced it."""

    revision: int
    snapshot: Mapping[str, Any]
    changed_fields: tuple[str, ...]
    accepted_finding_ids: tuple[str, ...]
    change_origin: str
    dependencies: tuple[str, ...]
    validation_result: str
    timestamp: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Detach snapshot data from the mutable workspace state."""
        object.__setattr__(self, "snapshot", copy.deepcopy(dict(self.snapshot)))


@dataclass(frozen=True, slots=True)
class GoalEvaluation:
    """Report weighted goal coverage and explicit critical requirements separately."""

    outcome: GoalOutcome
    score: float
    threshold: float
    component_scores: Mapping[str, float]
    missing_requirements: tuple[str, ...] = ()
    uncertainty_reasons: tuple[str, ...] = ()
    message: str = ""


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    """Capture an individual subscriber delivery outcome without aborting other deliveries."""

    module_id: str
    event_id: str
    delivered: bool
    error: str | None = None
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class ModuleActivationResult:
    """Describe whether a module was skipped, queued, completed, or failed."""

    module_id: str
    event_id: str
    relevance_score: float
    threshold: float
    queued: bool
    reason: str = ""


WORKSPACE_FIELDS = (
    "current_input",
    "current_goal",
    "lexical_hypotheses",
    "syntactic_hypotheses",
    "semantic_hypotheses",
    "discourse_context",
    "working_memory",
    "retrieved_memories",
    "unresolved_questions",
    "module_status",
)

APPEND_ONLY_FIELDS = frozenset({
    "lexical_hypotheses", "syntactic_hypotheses", "semantic_hypotheses",
    "discourse_context", "retrieved_memories", "unresolved_questions",
})