"""Request and result records for the 45-step reasoning run (Steps A1, Section 4).

``ReasoningRequest`` implements Step A1's parsing requirements: goal predicate,
scope, entity references, inference mode, deadline and verification level;
invalid scope keys and absent budgets are rejected deterministically at
construction. ``ReasoningRunResult`` keeps epistemic categories separate:
derived conclusions, preserved conflicts, ambiguity alternatives, probabilistic
support, model proposals and metrics are distinct fields, never one blob.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import (
    ConfidenceVector,
    EpistemicStatus,
    EvidenceItem,
    InferenceMode,
    Literal,
    LogicalTerm,
    NEUROSYMBOLIC_SCHEMA_VERSION,
    NSRLConflictRecord,
    ProofClass,
    PublicationState,
    ReasoningFinding,
    RuleClass,
    RuleDefinition,
    SemanticFact,
    VerificationLevel,
    canonical_json,
    make_id,
    sha256_json,
)


@dataclass(frozen=True, slots=True)
class ReasoningRequest:
    """One bounded neuro-symbolic inference request (Step A1)."""

    goal_predicate: str
    goal_args: tuple[str, ...]
    scope: Mapping[str, Any] = field(default_factory=dict)
    entity_references: tuple[str, ...] = ()
    inference_mode: InferenceMode = InferenceMode.HYBRID
    deadline_seconds: float = 30.0
    verification_level: VerificationLevel = VerificationLevel.BASIC
    facts: tuple[SemanticFact, ...] = ()
    rules: tuple[RuleDefinition, ...] = ()
    ruleset_version: str = "1.0.0"
    evidence_items: tuple[EvidenceItem, ...] = ()
    ground_truth: Mapping[str, Any] | None = None
    usg_bundle: Mapping[str, Any] | None = None
    usg_expected_digest: str | None = None
    task_id: str = field(default_factory=lambda: make_id("nsr_task"))
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = NEUROSYMBOLIC_SCHEMA_VERSION

    ALLOWED_SCOPE_KEYS = frozenset({
        "domain", "temporal", "entities", "constraints", "max_candidates",
        "review_impact_threshold", "notes",
    })

    def __post_init__(self) -> None:
        """Reject invalid scope, absent budgets and malformed goal fields (A1)."""
        if not isinstance(self.goal_predicate, str) or not self.goal_predicate:
            raise ValueError("goal_predicate must be a non-empty string")
        if not isinstance(self.goal_args, tuple):
            raise TypeError("goal_args must be a tuple of entity strings")
        if not all(isinstance(item, str) and item for item in self.goal_args):
            raise ValueError("goal_args must contain non-empty strings")
        unknown = set(self.scope) - self.ALLOWED_SCOPE_KEYS
        if unknown:
            raise ValueError(f"invalid scope keys: {sorted(unknown)}")
        if not math_is_finite_positive(self.deadline_seconds):
            raise ValueError("deadline_seconds must be finite and positive")
        if not isinstance(self.verification_level, VerificationLevel):
            raise TypeError("verification_level must be a VerificationLevel")
        if not isinstance(self.inference_mode, InferenceMode):
            raise TypeError("inference_mode must be an InferenceMode")
        if not self.facts and self.usg_bundle is None:
            raise ValueError(
                "absent budgets/knowledge: supply facts, a USG bundle, or both",
            )
        if self.usg_bundle is not None and not isinstance(self.usg_bundle, Mapping):
            raise TypeError("usg_bundle must be a mapping bundle or None")

    @property
    def goal_literal(self) -> Literal:
        """Project the goal onto the typed goal literal (Step A1)."""
        return Literal(
            predicate=self.goal_predicate,
            args=tuple(
                LogicalTerm(item, "entity") for item in self.goal_args
            ),
        )

    def fingerprint(self) -> str:
        """Return the content fingerprint used for idempotent replay."""
        return sha256_json({
            "goal": [self.goal_predicate, list(self.goal_args)],
            "scope": dict(self.scope),
            "mode": self.inference_mode.value,
            "verification": self.verification_level.value,
            "fact_ids": [fact.fact_id for fact in self.facts],
            "rule_refs": [[rule.rule_id, rule.version] for rule in self.rules],
            "usg_digest": (
                self.usg_expected_digest
                or (sha256_json(dict(self.usg_bundle)) if self.usg_bundle else None)
            ),
        })


def math_is_finite_positive(value: Any) -> bool:
    """Return whether one value is a finite, positive number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number > 0 and number == number and number not in (float("inf"),)


@dataclass(frozen=True, slots=True)
class ReasoningRunResult:
    """One completed reasoning run with separated epistemic categories."""

    run_id: str
    task_id: str
    status: str  # completed | completed_with_uncertainty | partial | blocked | failed
    status_reason: str
    workspace_id: str
    workspace_revision: int
    ruleset_version: str
    conclusions: tuple[ReasoningFinding, ...] = ()
    conflicts: tuple[NSRLConflictRecord, ...] = ()
    ambiguity_groups: tuple[Mapping[str, Any], ...] = ()
    probabilistic_support: tuple[Mapping[str, Any], ...] = ()
    confidence_vectors: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    model_proposals: tuple[Mapping[str, Any], ...] = ()
    proof_summary: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    operation_coverage: tuple[str, ...] = ()
    trace: tuple[Mapping[str, Any], ...] = ()
    trace_completeness: float = 0.0
    checkpoint: Mapping[str, Any] | None = None
    reproducibility: Mapping[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    published_finding_ids: tuple[str, ...] = ()
    review_requested_ids: tuple[str, ...] = ()
    goal_satisfied: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the detached JSON-compatible result."""
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "status": self.status,
            "status_reason": self.status_reason,
            "workspace_id": self.workspace_id,
            "workspace_revision": self.workspace_revision,
            "ruleset_version": self.ruleset_version,
            "conclusions": [item.to_dict() for item in self.conclusions],
            "conflicts": [item.to_dict() for item in self.conflicts],
            "ambiguity_groups": [dict(item) for item in self.ambiguity_groups],
            "probabilistic_support": [dict(item) for item in self.probabilistic_support],
            "confidence_vectors": {
                key: dict(value) for key, value in sorted(self.confidence_vectors.items())
            },
            "model_proposals": [dict(item) for item in self.model_proposals],
            "proof_summary": dict(self.proof_summary),
            "metrics": dict(self.metrics),
            "operation_coverage": list(self.operation_coverage),
            "trace": [dict(item) for item in self.trace],
            "trace_completeness": self.trace_completeness,
            "checkpoint": dict(self.checkpoint) if self.checkpoint else None,
            "reproducibility": dict(self.reproducibility),
            "elapsed_ms": round(self.elapsed_ms, 3),
            "published_finding_ids": list(self.published_finding_ids),
            "review_requested_ids": list(self.review_requested_ids),
            "goal_satisfied": self.goal_satisfied,
        }

    def summary_line(self) -> str:
        """Return a compact human-readable status line."""
        return (
            f"{self.status}: run={self.run_id} conclusions={len(self.conclusions)} "
            f"conflicts={len(self.conflicts)} published={len(self.published_finding_ids)} "
            f"coverage={len(self.operation_coverage)}/45"
        )


__all__ = ["ReasoningRequest", "ReasoningRunResult", "math_is_finite_positive"]
