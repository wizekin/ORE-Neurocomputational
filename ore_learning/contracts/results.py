"""Result, proposal, and validation-report contracts (Point 14, Sections 2 and 5.1).

A raw model response is never a canonical knowledge record: every inference result
is normalized into an evidence-bearing :class:`ModelProposal`, and verification lives
in a separate :class:`ValidationReport` whose deterministic checks, learned PRM
diagnostics, and human reviews are distinct result classes.  The one-way authority
boundary into USG is implemented by :class:`AuthorityBoundary`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from .manifests import canonical_payload_digest

__all__ = (
    "AuthorityBoundary",
    "CheckOutcome",
    "DeterministicCheck",
    "ExecutionMetrics",
    "ExplanationReference",
    "HumanReview",
    "InferenceResult",
    "InputProvenance",
    "ModelProposal",
    "PRMDiagnostic",
    "ProposalStatus",
    "UncertaintyMetadata",
    "ValidationReport",
    "ValidatorKind",
)


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


class ProposalStatus(str, Enum):
    """Lifecycle of a model proposal inside the workspace."""

    PROPOSED = "proposed"
    VALIDATED = "validated"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class CheckOutcome(str, Enum):
    """Outcome of one deterministic check (Step 34)."""

    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class ValidatorKind(str, Enum):
    """Validator taxonomy; only DETERMINISTIC kinds can produce proof (Step 34)."""

    DETERMINISTIC = "deterministic"
    PRM = "prm"
    HUMAN = "human"


@dataclass(frozen=True, slots=True)
class ExecutionMetrics:
    """Measured execution cost of one inference call (Section 5.1)."""

    wall_clock_ms: float
    cpu_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    steps_executed: int = 0
    cancelled: bool = False

    def __post_init__(self) -> None:
        """Metrics must be finite and non-negative."""
        for name in ("wall_clock_ms", "cpu_time_ms", "peak_memory_mb"):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"wall_clock_ms": round(self.wall_clock_ms, 3),
                "cpu_time_ms": round(self.cpu_time_ms, 3),
                "peak_memory_mb": round(self.peak_memory_mb, 3),
                "steps_executed": self.steps_executed, "cancelled": self.cancelled}


@dataclass(frozen=True, slots=True)
class InputProvenance:
    """References, digests, and timestamps for every consumed input (Step 54)."""

    input_refs: tuple[str, ...]
    input_digests: tuple[str, ...]
    timestamps: tuple[float, ...] = ()
    preprocessing: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refs and digests must align."""
        if not self.input_refs or not self.input_digests:
            raise ValueError("input refs and digests are both required")
        if len(self.input_refs) != len(self.input_digests):
            raise ValueError("one digest is required per input reference")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"input_refs": list(self.input_refs),
                "input_digests": list(self.input_digests),
                "timestamps": list(self.timestamps),
                "preprocessing": list(self.preprocessing)}


@dataclass(frozen=True, slots=True)
class UncertaintyMetadata:
    """Confidence and uncertainty metadata carried by every result (Section 2.2)."""

    confidence: float
    calibration_method: str = "declared-by-manifest"
    brier_score: float | None = None
    abstained: bool = False
    notes: str = "confidence is a proposal signal, never proof"

    def __post_init__(self) -> None:
        """Confidence is a fraction in [0, 1]."""
        if not math.isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be a fraction in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"confidence": self.confidence,
                "calibration_method": self.calibration_method,
                "brier_score": self.brier_score,
                "abstained": self.abstained, "notes": self.notes}


@dataclass(frozen=True, slots=True)
class ExplanationReference:
    """A reference to an explanation diagnostic or native trace (Step 27)."""

    explanation_id: str
    channel: str  # "native_trace" or "model_diagnostics"
    method: str = ""
    summary: str = ""

    def __post_init__(self) -> None:
        """Channel vocabulary is fixed (two distinct channels, Step 27)."""
        if self.channel not in ("native_trace", "model_diagnostics"):
            raise ValueError("channel must be native_trace or model_diagnostics")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"explanation_id": self.explanation_id, "channel": self.channel,
                "method": self.method, "summary": self.summary}


@dataclass(frozen=True, slots=True)
class DeterministicCheck:
    """Step 34 -- one checkable transformation with its own validator identity."""

    check_id: str
    validator: str
    outcome: CheckOutcome
    detail: str = ""
    validator_version: str = "1.0.0"

    def __post_init__(self) -> None:
        """Identity is mandatory."""
        object.__setattr__(self, "check_id", _nonempty("check_id", self.check_id))
        object.__setattr__(self, "validator", _nonempty("validator", self.validator))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"check_id": self.check_id, "validator": self.validator,
                "outcome": self.outcome.value, "detail": self.detail,
                "validator_version": self.validator_version}


@dataclass(frozen=True, slots=True)
class PRMDiagnostic:
    """Step 33 -- a learned process-reward screening signal, never a final validator."""

    step_id: str
    score: float
    calibrated: bool = False
    prm_model_id: str = "prm.reference@1.0.0"
    note: str = "screening signal only; deterministic checks take precedence"

    def __post_init__(self) -> None:
        """Scores are fractions."""
        if not math.isfinite(float(self.score)) or not 0.0 <= self.score <= 1.0:
            raise ValueError("PRM score must be a fraction in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"step_id": self.step_id, "score": self.score,
                "calibrated": self.calibrated, "prm_model_id": self.prm_model_id,
                "note": self.note}


@dataclass(frozen=True, slots=True)
class HumanReview:
    """Step 55 -- a recorded human review with identity, scope, and time."""

    review_id: str
    reviewer: str
    decision: str
    reviewed_evidence: tuple[str, ...] = ()
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        """Decision vocabulary is fixed."""
        if self.decision not in ("approved", "rejected", "needs_changes"):
            raise ValueError("decision must be approved/rejected/needs_changes")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"review_id": self.review_id, "reviewer": self.reviewer,
                "decision": self.decision,
                "reviewed_evidence": list(self.reviewed_evidence),
                "timestamp": self.timestamp}


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Section 5.1 -- deterministic checks, PRM diagnostics, and reviews as separate
    result classes (Steps 33-34).  Only deterministic PASS results can mark an
    outcome ``verified``; PRM diagnostics never upgrade a failed check."""

    checks: tuple[DeterministicCheck, ...] = ()
    prm_diagnostics: tuple[PRMDiagnostic, ...] = ()
    human_reviews: tuple[HumanReview, ...] = ()

    @property
    def verified(self) -> bool:
        """True when every applicable deterministic check passed (Step 34)."""
        applicable = [check for check in self.checks
                      if check.outcome is not CheckOutcome.NOT_APPLICABLE]
        return bool(applicable) and all(
            check.outcome is CheckOutcome.PASS for check in applicable)

    @property
    def failed_checks(self) -> tuple[DeterministicCheck, ...]:
        """The deterministic checks that failed."""
        return tuple(check for check in self.checks
                     if check.outcome is CheckOutcome.FAIL)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"verified": self.verified,
                "checks": [check.to_dict() for check in self.checks],
                "prm_diagnostics": [item.to_dict() for item in self.prm_diagnostics],
                "human_reviews": [review.to_dict() for review in self.human_reviews],
                "note": ("verification requires deterministic checks; learned PRM "
                         "scores and human reviews are recorded separately")}


@dataclass(slots=True)
class ModelProposal:
    """Section 2 authority chain: model proposal -> typed evidence record.

    The proposal is the only shape in which model output travels through the
    workspace.  It carries identity, provenance, uncertainty, and validation state;
    it starts as ``proposed`` and can only reach ``approved`` through independent
    validation and coordinator policy.
    """

    proposal_id: str
    task_id: str
    model_id: str
    model_version: str
    artifact_digest: str
    payload: dict[str, Any]
    uncertainty: UncertaintyMetadata
    input_provenance: InputProvenance
    status: ProposalStatus = ProposalStatus.PROPOSED
    validation: ValidationReport = field(default_factory=ValidationReport)
    explanation_refs: tuple[ExplanationReference, ...] = ()
    created_at: float = 0.0

    def __post_init__(self) -> None:
        """Validate identity fields."""
        for name, value in (("proposal_id", self.proposal_id), ("task_id", self.task_id),
                            ("model_id", self.model_id),
                            ("model_version", self.model_version),
                            ("artifact_digest", self.artifact_digest)):
            object.__setattr__(self, name, _nonempty(name, value))
        if not isinstance(self.payload, Mapping) or not self.payload:
            raise ValueError("payload must be a non-empty mapping")

    @property
    def is_actionable(self) -> bool:
        """Only validated-and-approved proposals may drive state transitions."""
        return self.status is ProposalStatus.APPROVED and self.validation.verified

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"proposal_id": self.proposal_id, "task_id": self.task_id,
                "model_id": self.model_id, "model_version": self.model_version,
                "artifact_digest": self.artifact_digest, "payload": dict(self.payload),
                "uncertainty": self.uncertainty.to_dict(),
                "input_provenance": self.input_provenance.to_dict(),
                "status": self.status.value,
                "validation": self.validation.to_dict(),
                "explanation_refs": [ref.to_dict() for ref in self.explanation_refs],
                "is_actionable": self.is_actionable, "created_at": self.created_at}


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Section 5.1 -- the typed output of one executed inference call."""

    task_id: str
    model_id: str
    model_version: str
    artifact_digest: str
    status: str  # completed | failed | cancelled | abstained
    typed_output: dict[str, Any]
    uncertainty: UncertaintyMetadata
    execution: ExecutionMetrics
    input_provenance: InputProvenance
    validation: ValidationReport = field(default_factory=ValidationReport)
    explanation_refs: tuple[ExplanationReference, ...] = ()
    failure_category: str = ""

    def __post_init__(self) -> None:
        """Validate identity and status."""
        for name in ("task_id", "model_id", "model_version", "artifact_digest"):
            object.__setattr__(self, name,
                               _nonempty(name, getattr(self, name)))
        if self.status not in ("completed", "failed", "cancelled", "abstained"):
            raise ValueError("status must be completed/failed/cancelled/abstained")

    def to_proposal(self, proposal_id: str) -> ModelProposal:
        """Normalize into an evidence-bearing proposal (never a canonical record)."""
        return ModelProposal(
            proposal_id=proposal_id, task_id=self.task_id, model_id=self.model_id,
            model_version=self.model_version, artifact_digest=self.artifact_digest,
            payload=dict(self.typed_output), uncertainty=self.uncertainty,
            input_provenance=self.input_provenance,
            validation=self.validation, explanation_refs=self.explanation_refs)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"task_id": self.task_id, "model_id": self.model_id,
                "model_version": self.model_version,
                "artifact_digest": self.artifact_digest, "status": self.status,
                "typed_output": dict(self.typed_output),
                "uncertainty": self.uncertainty.to_dict(),
                "execution": self.execution.to_dict(),
                "input_provenance": self.input_provenance.to_dict(),
                "validation": self.validation.to_dict(),
                "explanation_refs": [ref.to_dict() for ref in self.explanation_refs],
                "failure_category": self.failure_category}


class AuthorityBoundary:
    """The one-way boundary between learned models and authoritative USG state.

    Implements the specification's authority chain: model proposal -> typed evidence
    record -> independent validation -> coordinator policy -> approved state
    transition.  Raw model output, unvalidated proposals, and proposals lacking the
    required approval are refused with explicit reasons; refusals are recorded for
    audit instead of being silently swallowed.
    """

    purpose = ("models propose; only independently validated and approved records "
               "may request canonical USG updates")

    def __init__(self) -> None:
        """Initialize the refusal and update-request ledgers."""
        self.refusals: list[dict[str, Any]] = []
        self.update_requests: list[dict[str, Any]] = []

    def request_canonical_update(self, proposal: ModelProposal,
                                 approval: Any = None) -> dict[str, Any]:
        """Try to move a proposal to an approved canonical update request.

        Returns a decision record; refusals are kept in ``refusals``.  A raw
        :class:`InferenceResult` is not accepted here at all -- it must be normalized
        into a proposal first, and even then validation and approval are required.
        """
        if not isinstance(proposal, ModelProposal):
            record = {"decision": "refused", "reason": "raw model output is not accepted",
                      "task_id": "", "model_id": type(proposal).__name__}
            self.refusals.append(record)
            return record
        if not proposal.validation.verified:
            record = {"decision": "refused",
                      "reason": "proposal lacks independent deterministic validation",
                      "task_id": proposal.task_id, "model_id": proposal.model_id}
            self.refusals.append(record)
            return record
        if proposal.status is not ProposalStatus.APPROVED:
            record = {"decision": "refused",
                      "reason": ("coordinator approval missing; the model cannot approve "
                                 "its own output"),
                      "task_id": proposal.task_id, "model_id": proposal.model_id}
            self.refusals.append(record)
            return record
        request = {"decision": "queued_for_usg_workflow",
                   "task_id": proposal.task_id, "model_id": proposal.model_id,
                   "proposal_id": proposal.proposal_id,
                   "payload_digest": canonical_payload_digest(proposal.payload),
                   "approval": getattr(approval, "to_dict", dict)()
                   if approval is not None else None}
        self.update_requests.append(request)
        return request

    def describe(self) -> dict[str, Any]:
        """Summarize boundary activity for diagnostics."""
        return {"purpose": self.purpose, "refusals": len(self.refusals),
                "update_requests": len(self.update_requests),
                "rule": ("model proposal -> typed evidence -> independent validation -> "
                         "coordinator policy -> approved state transition")}
