"""Typed request, layer-taxonomy, envelope, and result records for ORE blackboard coordination.

These records implement the typed-object requirements of Algorithm 1 (Steps 1, 3,
4, 17, 26, 28) without duplicating shared workspace state. They deliberately keep
epistemic categories distinct: observations, imported knowledge references,
derived hypotheses, inferences, and accepted task results are never collapsed
into one undifferentiated field.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Mapping
from uuid import uuid4

from ore_workspace.models import WORKSPACE_FIELDS, new_id, utc_now


BLACKBOARD_PACKAGE_VERSION = "1.0.0"

COGNITIVE_CONTROL_PRODUCER = "cognitive_control"


def canonical_json(value: Any) -> str:
    """Serialize nested values deterministically for fingerprints and digests."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest of the canonical JSON encoding of a value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class BlackboardLayer(IntEnum):
    """Assign every published cognitive record to exactly one primary blackboard layer.

    Step 3 of Algorithm 1 requires eight logical layers over one controlled shared
    state. The layers are namespaces over the existing workspace fields; they do
    not create eight separate databases.
    """

    INPUT = 1
    LEXICAL = 2
    STRUCTURAL = 3
    SEMANTIC = 4
    HYPOTHESIS = 5
    MEMORY = 6
    PLANNING = 7
    VERIFICATION = 8


LAYER_COUNT = len(BlackboardLayer)

LAYER_FIELDS: Mapping[BlackboardLayer, frozenset[str]] = {
    BlackboardLayer.INPUT: frozenset({"current_input", "artifact_references"}),
    BlackboardLayer.LEXICAL: frozenset({"lexical_hypotheses"}),
    BlackboardLayer.STRUCTURAL: frozenset({"syntactic_hypotheses"}),
    BlackboardLayer.SEMANTIC: frozenset({"semantic_hypotheses"}),
    BlackboardLayer.HYPOTHESIS: frozenset({
        "lexical_hypotheses", "syntactic_hypotheses", "semantic_hypotheses",
        "discourse_context",
    }),
    BlackboardLayer.MEMORY: frozenset({"retrieved_memories", "working_memory"}),
    BlackboardLayer.PLANNING: frozenset({"current_goal"}),
    BlackboardLayer.VERIFICATION: frozenset({"unresolved_questions", "module_status"}),
}


def one_hot_layer(layer: BlackboardLayer) -> tuple[int, ...]:
    """Return the one-hot layer vector z(x) with sum_k z_k(x) = 1 (Step 3)."""
    vector = [0] * LAYER_COUNT
    vector[int(layer) - 1] = 1
    return tuple(vector)


def layer_for_field(field: str) -> BlackboardLayer:
    """Return the primary layer for one workspace field using an explicit priority order.

    Fields that participate in multiple layers (hypothesis-layer records also live
    in lexical/structural/semantic fields) resolve through the declared priority
    so every record still has exactly one primary layer and explicit links.
    """
    priority = (
        BlackboardLayer.INPUT,
        BlackboardLayer.PLANNING,
        BlackboardLayer.VERIFICATION,
        BlackboardLayer.MEMORY,
        BlackboardLayer.LEXICAL,
        BlackboardLayer.STRUCTURAL,
        BlackboardLayer.SEMANTIC,
        BlackboardLayer.HYPOTHESIS,
    )
    for layer in priority:
        if field in LAYER_FIELDS[layer]:
            return layer
    raise ValueError(f"workspace field is not assigned to any blackboard layer: {field!r}")


class LifecycleState(str, Enum):
    """Lifecycle partial order for published records (Step 3).

    Records are promoted only through explicit, evidence-backed transitions; a
    numeric score alone never promotes a record.
    """

    CANDIDATE = "candidate"
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    EXTERNALLY_SUPPLIED = "externally_supplied"

    def at_least(self, other: "LifecycleState") -> bool:
        """Return whether this state is at or beyond another state in the partial order."""
        order = {
            LifecycleState.EXTERNALLY_SUPPLIED: 0,
            LifecycleState.CANDIDATE: 1,
            LifecycleState.REVIEWED: 2,
            LifecycleState.ACCEPTED: 3,
        }
        return order[self] >= order[other]


class FailureClassification(str, Enum):
    """Classify execution failures without converting them into successful results (Step 26)."""

    INVALID_INPUT = "invalid_input"
    CONTRACT_MISMATCH = "contract_mismatch"
    PERMISSION_DENIED = "permission_denied"
    DEPENDENCY_MISSING = "dependency_missing"
    MODULE_EXCEPTION = "module_exception"
    TIMEOUT = "timeout"
    EVENT_DELIVERY_FAILURE = "event_delivery_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DOWNSTREAM_VALIDATION_REJECTED = "downstream_validation_rejected"


@dataclass(frozen=True, slots=True)
class InputSource:
    """Describe one immutable input source with its own identity and source group (Step 4)."""

    source_id: str
    text: str
    source_group: str = "default"
    modality: str = "text"

    def __post_init__(self) -> None:
        """Validate source identity fields."""
        if not self.source_id.strip():
            raise ValueError("InputSource.source_id must be non-empty")
        if not isinstance(self.text, str):
            raise ValueError("InputSource.text must be a string")
        if not self.source_group.strip():
            raise ValueError("InputSource.source_group must be non-empty")
        if not self.modality.strip():
            raise ValueError("InputSource.modality must be non-empty")


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """Record one token occurrence with validated character bounds (Step 4)."""

    token: str
    source_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        """Reject decreasing or negative spans."""
        if self.start < 0 or self.end < self.start:
            raise ValueError("TokenSpan must satisfy 0 <= start <= end")
        if not self.token:
            raise ValueError("TokenSpan.token must be non-empty")


@dataclass(frozen=True, slots=True)
class InputRoot:
    """Preserve the original input as the immutable evidence root (Step 4)."""

    sources: tuple[InputSource, ...]
    tokens: tuple[TokenSpan, ...]
    input_digest: str
    root_id: str = field(default_factory=lambda: new_id("input_root"))
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate source and span consistency across the collected input."""
        if not self.sources:
            raise ValueError("InputRoot requires at least one input source")
        lengths = {source.source_id: len(source.text) for source in self.sources}
        for span in self.tokens:
            if span.source_id not in lengths:
                raise ValueError(f"token references unknown source: {span.source_id!r}")
            if span.end > lengths[span.source_id]:
                raise ValueError(
                    f"token span ({span.start}, {span.end}) exceeds source {span.source_id!r}"
                )
        if not (len(self.input_digest) == 64 and all(
            character in "0123456789abcdef" for character in self.input_digest
        )):
            raise ValueError("InputRoot.input_digest must be a lowercase SHA-256 hex digest")

    def primary_text(self) -> str:
        """Return the first source's text, which the shared workspace treats as current input."""
        return self.sources[0].text

    def source_group_of(self, source_id: str) -> str:
        """Return the declared source-group label g(q) for one source index q."""
        for source in self.sources:
            if source.source_id == source_id:
                return source.source_group
        raise KeyError(source_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible evidence-root record."""
        return {
            "root_id": self.root_id,
            "created_at": self.created_at,
            "input_digest": self.input_digest,
            "sources": [
                {
                    "source_id": source.source_id,
                    "source_group": source.source_group,
                    "modality": source.modality,
                    "text_digest": sha256_json(source.text),
                    "length": len(source.text),
                }
                for source in self.sources
            ],
            "tokens": [
                {
                    "token": span.token,
                    "source_id": span.source_id,
                    "start": span.start,
                    "end": span.end,
                }
                for span in self.tokens
            ],
        }


def build_input_root(sources: Mapping[str, Any] | InputSource | str) -> InputRoot:
    """Build the immutable evidence root from one or more validated input sources.

    Multiple inputs keep separate source identities and a task-level collection
    instead of being concatenated (Step 4). Spans are computed with whitespace
    tokenization over each source independently.
    """
    if isinstance(sources, str):
        items = (InputSource(source_id="input:0", text=sources),)
    elif isinstance(sources, InputSource):
        items = (sources,)
    elif isinstance(sources, Mapping):
        items = tuple(
            InputSource(
                source_id=str(source_id),
                text=value if isinstance(value, str) else canonical_json(value),
                source_group="default",
            )
            for source_id, value in sorted(sources.items())
        )
    else:
        raise TypeError("sources must be a string, an InputSource, or a mapping")
    tokens: list[TokenSpan] = []
    for source in items:
        for match in re.finditer(r"\w+", source.text):
            tokens.append(TokenSpan(
                match.group(0), source.source_id, match.start(), match.end(),
            ))
    digest = sha256_json({
        "kind": "ore_blackboard_input_root",
        "sources": [
            {"source_id": item.source_id, "group": item.source_group, "text": item.text}
            for item in items
        ],
    })
    return InputRoot(sources=items, tokens=tuple(tokens), input_digest=digest)


@dataclass(frozen=True, slots=True)
class BlackboardTaskRequest:
    """Typed task request with goal, constraints, and unique task identity (Step 1)."""

    input: Any
    goal: str
    task_id: str = field(default_factory=lambda: f"blackboard_task_{uuid4().hex}")
    required_capabilities: tuple[str, ...] = (
        "language.lexical_access",
        "language.syntax_parsing",
        "language.semantic_composition",
    )
    optional_capabilities: tuple[str, ...] = (
        "memory.episodic_retrieval",
        "cognition.verification",
    )
    task_family: str = "sentence_understanding"
    priority: float = 0.5
    execution_mode: str = "FUNCTIONAL"
    usg_bundle: Mapping[str, Any] | None = None
    usg_expected_digest: str | None = None
    memory_store: tuple[Mapping[str, Any], ...] | None = None
    verification_rules: tuple[Mapping[str, Any], ...] | None = None
    retrieved_memories: tuple[Mapping[str, Any], ...] | None = None
    caller_evidence: tuple[Mapping[str, Any], ...] = ()
    prior_run_id: str | None = None
    resume_workspace_id: str | None = None
    deadline_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate typed request fields, finite scalars, and bounded priorities (Step 1)."""
        if not self.task_id.strip():
            raise ValueError("task_id must be a non-empty idempotency key")
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("goal must be a non-empty string")
        if not isinstance(self.input, (str, InputSource, Mapping)):
            raise TypeError("input must be a string, an InputSource, or a mapping of sources")
        required = tuple(self.required_capabilities)
        optional = tuple(self.optional_capabilities)
        if any(not item.strip() for item in required + optional):
            raise ValueError("capabilities must be explicit non-empty identifiers")
        if set(required) & set(optional):
            raise ValueError("a capability cannot be both required and optional")
        if not math.isfinite(self.priority) or not 0.0 <= self.priority <= 1.0:
            raise ValueError("priority must be finite and normalized to [0, 1]")
        mode = str(self.execution_mode).upper()
        if mode not in {"FUNCTIONAL", "BIOLOGICAL", "HYBRID", "TRAIN", "INFER", "VALIDATE"}:
            raise ValueError(f"unsupported execution mode: {self.execution_mode!r}")
        if self.deadline_seconds is not None and (
            not math.isfinite(self.deadline_seconds) or self.deadline_seconds <= 0
        ):
            raise ValueError("deadline_seconds must be finite and positive")
        if self.usg_bundle is not None and not isinstance(self.usg_bundle, Mapping):
            raise TypeError("usg_bundle must be a mapping (a USG-KNOWLEDGE bundle)")
        object.__setattr__(self, "required_capabilities", required)
        object.__setattr__(self, "optional_capabilities", optional)
        object.__setattr__(self, "execution_mode", mode)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "caller_evidence", tuple(self.caller_evidence))

    def request_fingerprint(self) -> str:
        """Fingerprint canonical task metadata with typed input hashes (Step 1)."""
        if isinstance(self.input, str):
            input_digest = sha256_json(self.input)
        elif isinstance(self.input, InputSource):
            input_digest = self.input.input_digest
        elif isinstance(self.input, Mapping):
            input_digest = build_input_root(self.input).input_digest
        else:  # pragma: no cover - validated in __post_init__
            raise TypeError("unsupported input type")
        payload = {
            "goal": self.goal,
            "input_digest": input_digest,
            "task_family": self.task_family,
            "required_capabilities": sorted(self.required_capabilities),
            "optional_capabilities": sorted(self.optional_capabilities),
            "execution_mode": self.execution_mode,
            "usg_bundle_digest": (
                sha256_json(dict(self.usg_bundle)) if self.usg_bundle is not None else None
            ),
            "memory_store": (
                [dict(item) for item in self.memory_store] if self.memory_store else None
            ),
            "verification_rules": (
                [dict(item) for item in self.verification_rules] if self.verification_rules else None
            ),
            "retrieved_memories": (
                [dict(item) for item in self.retrieved_memories] if self.retrieved_memories else None
            ),
            "policy_fingerprint": _POLICY_FINGERPRINT,
        }
        return sha256_json(payload)


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Separate run identity from workspace identity (Step 2)."""

    task_id: str
    run_id: str
    task_fingerprint: str
    created_at: str = field(default_factory=utc_now)
    prior_run_id: str | None = None
    resumed_workspace_id: str | None = None


@dataclass(frozen=True, slots=True)
class USGSnapshotReference:
    """Compact, read-only reference to an attached USG knowledge bundle (Step 5)."""

    namespace: str
    bundle_id: str
    bundle_digest: str
    adapter_version: str
    resource_categories: Mapping[str, int]
    read_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return the compact published reference payload."""
        return {
            "reference_type": "usg_knowledge_bundle",
            "namespace": self.namespace,
            "bundle_id": self.bundle_id,
            "bundle_digest": self.bundle_digest,
            "adapter_version": self.adapter_version,
            "resource_categories": dict(sorted(self.resource_categories.items())),
            "read_only": self.read_only,
        }


@dataclass(frozen=True, slots=True)
class FindingEnvelope:
    """Declare one module-produced finding inside a typed artifact payload (Step 17)."""

    field: str
    payload: Mapping[str, Any]
    key: str | None = None
    finding_type: str = "module_hypothesis"
    source: str = "module_inference"
    source_span: tuple[int, int] | None = None
    rules_applied: tuple[str, ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()
    dependencies: tuple[str, ...] = ()
    usg_references: tuple[Mapping[str, Any], ...] = ()
    confidence: float | None = None
    layer: BlackboardLayer | None = None
    lifecycle_state: LifecycleState = LifecycleState.CANDIDATE
    notes: str | None = None

    def __post_init__(self) -> None:
        """Validate the envelope structure without instantiating a workspace finding."""
        if self.field not in WORKSPACE_FIELDS:
            raise ValueError(f"finding envelope targets unknown workspace field: {self.field!r}")
        if not isinstance(self.payload, Mapping):
            raise TypeError("finding envelope payload must be a mapping")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.source_span is not None:
            start, end = self.source_span
            if start < 0 or end < start:
                raise ValueError("source_span must be an increasing non-negative pair")
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "rules_applied", tuple(self.rules_applied))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "usg_references", tuple(self.usg_references))


@dataclass(frozen=True, slots=True)
class SupportedHypothesis:
    """Rank one retained hypothesis with its transparent evidence score (Steps 21, 28)."""

    hypothesis_id: str
    label: str
    score: float
    used_evidence: tuple[Mapping[str, Any], ...]
    suppressed_sources: tuple[str, ...]
    score_definition: str
    calibrated_probability: bool


@dataclass(frozen=True, slots=True)
class BlackboardTaskResult:
    """Final task outcome with epistemic categories kept separate (Step 28)."""

    task_id: str
    run_id: str
    workspace_id: str
    status: str
    status_reason: str
    goal_score: float
    goal_threshold: float
    component_scores: Mapping[str, float]
    missing_requirements: tuple[str, ...]
    uncertainty_reasons: tuple[str, ...]
    observations: tuple[Mapping[str, Any], ...]
    imported_knowledge_references: tuple[Mapping[str, Any], ...]
    supported_hypotheses: tuple[SupportedHypothesis, ...]
    unresolved_alternatives: tuple[Mapping[str, Any], ...]
    verified_conclusions: tuple[Mapping[str, Any], ...]
    recommended_next_actions: tuple[str, ...]
    workspace_revision: int
    accepted_finding_count: int
    pending_finding_count: int
    rejected_finding_count: int
    conflict_count: int
    rounds: int
    activations: int
    module_calls: int
    retries: int
    suppressed_computations: int
    elapsed_ms: float
    trace_completeness: float
    integrity_ok: bool
    failure_summary: tuple[Mapping[str, Any], ...]
    reproducibility: Mapping[str, Any]
    plan: Mapping[str, Any]
    resource_plan: Mapping[str, Any]
    module_results: Mapping[str, Mapping[str, Any]]
    trace: tuple[Mapping[str, Any], ...]
    operation_coverage: tuple[str, ...]
    idempotent_replay: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible summary without large cognitive payloads."""
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "workspace_id": self.workspace_id,
            "status": self.status,
            "status_reason": self.status_reason,
            "goal_score": self.goal_score,
            "goal_threshold": self.goal_threshold,
            "component_scores": dict(self.component_scores),
            "missing_requirements": list(self.missing_requirements),
            "uncertainty_reasons": list(self.uncertainty_reasons),
            "observations": [dict(item) for item in self.observations],
            "imported_knowledge_references": [
                dict(item) for item in self.imported_knowledge_references
            ],
            "supported_hypotheses": [
                {
                    "hypothesis_id": item.hypothesis_id,
                    "label": item.label,
                    "score": item.score,
                    "used_evidence": [dict(row) for row in item.used_evidence],
                    "suppressed_sources": list(item.suppressed_sources),
                    "score_definition": item.score_definition,
                    "calibrated_probability": item.calibrated_probability,
                }
                for item in self.supported_hypotheses
            ],
            "unresolved_alternatives": [dict(item) for item in self.unresolved_alternatives],
            "verified_conclusions": [dict(item) for item in self.verified_conclusions],
            "recommended_next_actions": list(self.recommended_next_actions),
            "workspace_revision": self.workspace_revision,
            "accepted_finding_count": self.accepted_finding_count,
            "pending_finding_count": self.pending_finding_count,
            "rejected_finding_count": self.rejected_finding_count,
            "conflict_count": self.conflict_count,
            "rounds": self.rounds,
            "activations": self.activations,
            "module_calls": self.module_calls,
            "retries": self.retries,
            "suppressed_computations": self.suppressed_computations,
            "elapsed_ms": self.elapsed_ms,
            "trace_completeness": self.trace_completeness,
            "integrity_ok": self.integrity_ok,
            "failure_summary": [dict(item) for item in self.failure_summary],
            "reproducibility": dict(self.reproducibility),
            "plan": dict(self.plan),
            "resource_plan": dict(self.resource_plan),
            "module_results": {
                key: dict(value) for key, value in self.module_results.items()
            },
            "trace": [dict(item) for item in self.trace],
            "operation_coverage": list(self.operation_coverage),
            "idempotent_replay": self.idempotent_replay,
        }


_POLICY_FINGERPRINT_PLACEHOLDER: dict[str, Any] = {
    "package": "ore_blackboard",
    "version": BLACKBOARD_PACKAGE_VERSION,
    "policy_defaults": "see ore_blackboard.config",
}

_POLICY_FINGERPRINT = sha256_json(_POLICY_FINGERPRINT_PLACEHOLDER)


__all__ = [
    "BLACKBOARD_PACKAGE_VERSION",
    "BlackboardLayer",
    "BlackboardTaskRequest",
    "BlackboardTaskResult",
    "COGNITIVE_CONTROL_PRODUCER",
    "FailureClassification",
    "FindingEnvelope",
    "InputRoot",
    "InputSource",
    "LAYER_COUNT",
    "LAYER_FIELDS",
    "LifecycleState",
    "RunIdentity",
    "SupportedHypothesis",
    "TokenSpan",
    "USGSnapshotReference",
    "build_input_root",
    "canonical_json",
    "layer_for_field",
    "one_hot_layer",
    "sha256_json",
]
