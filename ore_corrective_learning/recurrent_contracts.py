"""Typed contracts for bounded recurrent feedback and corrective learning."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import math
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .contracts import CorrectionRequest, CorrectiveResult


def _plain(value: Any) -> Any:
    """Convert dataclasses, arrays, enums, and nested containers to detached JSON-safe values."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, set):
        return sorted((_plain(item) for item in value), key=repr)
    if hasattr(value, "tolist") and callable(value.tolist):
        return _plain(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return _plain(value.item())
    if hasattr(value, "value") and isinstance(value.value, (str, int, float)):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _bounded(name: str, value: float) -> float:
    """Require a finite normalized score and return its float value."""
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return numeric


@dataclass(frozen=True, slots=True)
class ConflictDimension:
    """Represent one available semantic-compatibility signal; None means not measured."""

    dimension: str
    compatibility_score: float | None
    weight: float = 1.0
    source: str = "caller"
    affected_nodes: tuple[str, ...] = ()
    affected_modules: tuple[str, ...] = ()
    supporting_evidence: tuple[str, ...] = ()
    contradicting_evidence: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate signal identity and weight while preserving unavailable scores as missing."""
        if not self.dimension.strip() or not self.source.strip():
            raise ValueError("dimension and source must be non-empty")
        if not math.isfinite(float(self.weight)) or self.weight < 0.0:
            raise ValueError("dimension weight must be finite and non-negative")
        if self.compatibility_score is not None:
            object.__setattr__(self, "compatibility_score", _bounded("compatibility_score", self.compatibility_score))
        for name in (
            "affected_nodes", "affected_modules", "supporting_evidence",
            "contradicting_evidence", "missing_information",
        ):
            object.__setattr__(self, name, tuple(str(item) for item in getattr(self, name) if str(item).strip()))
        object.__setattr__(self, "details", deepcopy(dict(self.details)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConflictDimension":
        """Build a typed signal from an explicitly supplied adapter result."""
        return cls(
            dimension=str(value.get("dimension", value.get("name", "unspecified"))),
            compatibility_score=value.get("compatibility_score", value.get("compatibility")),
            weight=float(value.get("weight", 1.0)),
            source=str(value.get("source", "module_adapter")),
            affected_nodes=tuple(value.get("affected_nodes", value.get("affected_semantic_nodes", ()))),
            affected_modules=tuple(value.get("affected_modules", ())),
            supporting_evidence=tuple(value.get("supporting_evidence", ())),
            contradicting_evidence=tuple(value.get("contradicting_evidence", ())),
            missing_information=tuple(value.get("missing_information", ())),
            details=value.get("details", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-safe signal record."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class ConflictReport:
    """Describe a possible interpretation conflict without asserting that unusual input is wrong."""

    conflict_id: str
    interpretation_id: str
    conflict_type: str
    conflict_score: float
    severity: str
    affected_semantic_nodes: tuple[str, ...]
    affected_modules: tuple[str, ...]
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    missing_information: tuple[str, ...]
    suggested_feedback_actions: tuple[str, ...]
    dimensions: tuple[ConflictDimension, ...]
    detected: bool = True
    status: str = "requires_investigation"

    def __post_init__(self) -> None:
        """Validate report fields and retain the individual dimensions behind the aggregate score."""
        if not self.conflict_id.strip() or not self.interpretation_id.strip():
            raise ValueError("conflict_id and interpretation_id must be non-empty")
        if self.severity not in {"none", "low", "medium", "high"}:
            raise ValueError("severity must be none, low, medium, or high")
        object.__setattr__(self, "conflict_score", _bounded("conflict_score", self.conflict_score))
        for name in (
            "affected_semantic_nodes", "affected_modules", "supporting_evidence",
            "contradicting_evidence", "missing_information", "suggested_feedback_actions",
        ):
            object.__setattr__(self, name, tuple(dict.fromkeys(
                str(item) for item in getattr(self, name) if str(item).strip()
            )))
        object.__setattr__(self, "dimensions", tuple(self.dimensions))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the conflict report without discarding missing-measurement information."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class FeedbackAction:
    """Represent a bounded, costed request sent to one cognitive module."""

    action_id: str
    responsible_module: str
    action_type: str
    target: str
    purpose: str
    expected_information_gain: float
    estimated_cost: float
    input_fingerprint: str
    attempt_number: int
    result_status: str = "pending"
    action_fingerprint: str = ""

    def __post_init__(self) -> None:
        """Validate action identity, information gain, cost, and attempt numbering."""
        for name in ("action_id", "responsible_module", "action_type", "target", "purpose", "input_fingerprint"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not math.isfinite(float(self.expected_information_gain)) or self.expected_information_gain < 0.0:
            raise ValueError("expected_information_gain must be finite and non-negative")
        if not math.isfinite(float(self.estimated_cost)) or self.estimated_cost < 0.0:
            raise ValueError("estimated_cost must be finite and non-negative")
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached action record for the shared episode and SQLite audit."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class CorrectionRecord:
    """Preserve an interpretation transition without deleting the old hypothesis or updating a model."""

    episode_id: str
    previous_interpretation_id: str
    revised_interpretation_id: str
    evidence_ids: tuple[str, ...]
    explanation: Mapping[str, Any]
    previous_confidence: float
    revised_confidence: float
    correction_status: str
    verification_status: str
    affected_context: Mapping[str, Any] = field(default_factory=dict)
    model_updated: bool = False

    def __post_init__(self) -> None:
        """Validate both confidence estimates and forbid inference-time model mutation."""
        for name in ("episode_id", "previous_interpretation_id", "revised_interpretation_id", "correction_status", "verification_status"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        object.__setattr__(self, "previous_confidence", _bounded("previous_confidence", self.previous_confidence))
        object.__setattr__(self, "revised_confidence", _bounded("revised_confidence", self.revised_confidence))
        if self.model_updated:
            raise ValueError("ordinary recurrent inference cannot update model parameters")
        object.__setattr__(self, "evidence_ids", tuple(dict.fromkeys(self.evidence_ids)))
        object.__setattr__(self, "explanation", deepcopy(dict(self.explanation)))
        object.__setattr__(self, "affected_context", deepcopy(dict(self.affected_context)))

    def to_dict(self) -> dict[str, Any]:
        """Return the immutable transition and its provenance as plain data."""
        return _plain(asdict(self))


@dataclass(slots=True)
class FeedbackEpisode:
    """Maintain the complete bounded feedback attempt and preserve its initial interpretation."""

    episode_id: str
    input_snapshot: Mapping[str, Any]
    initial_interpretation: Mapping[str, Any]
    current_interpretation: Mapping[str, Any]
    alternative_hypotheses: Sequence[Mapping[str, Any]]
    conflict_history: list[Mapping[str, Any]]
    evidence_history: list[Mapping[str, Any]]
    action_history: list[Mapping[str, Any]]
    current_iteration: int
    maximum_iterations: int
    consecutive_no_progress: int
    initial_confidence: float
    current_confidence: float
    status: str = "initialized"
    unresolved_questions: list[str] = field(default_factory=list)
    correction_record: Mapping[str, Any] | None = None
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Detach episode snapshots and validate its shared iteration and confidence state."""
        if not self.episode_id.strip():
            raise ValueError("episode_id must be non-empty")
        if self.maximum_iterations < 1 or self.current_iteration < 0:
            raise ValueError("episode iteration values are invalid")
        if self.consecutive_no_progress < 0:
            raise ValueError("consecutive_no_progress cannot be negative")
        self.input_snapshot = deepcopy(dict(self.input_snapshot))
        self.initial_interpretation = deepcopy(dict(self.initial_interpretation))
        self.current_interpretation = deepcopy(dict(self.current_interpretation))
        self.alternative_hypotheses = tuple(deepcopy(dict(item)) for item in self.alternative_hypotheses)
        self.initial_confidence = _bounded("initial_confidence", self.initial_confidence)
        self.current_confidence = _bounded("current_confidence", self.current_confidence)

    def to_dict(self) -> dict[str, Any]:
        """Return a complete detached episode snapshot for audit and future retrieval."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class RecurrentFeedbackConfig:
    """Configure bounded semantic feedback; defaults are engineering values, not biological constants."""

    max_semantic_feedback_rounds: int = 4
    maximum_hard_limit: int = 8
    max_attempts_per_action_signature: int = 1
    max_total_actions: int = 4
    max_total_action_cost: float = 4.0
    no_progress_patience: int = 2
    convergence_threshold: float = 0.0001
    minimum_confidence: float = 0.80
    minimum_evidence_quality: float = 0.35
    conflict_threshold: float = 0.25
    action_cost_weight: float = 0.25
    simulate_biology: bool = False

    def __post_init__(self) -> None:
        """Validate hard iteration, action, cost, convergence, and support limits."""
        if not 1 <= self.max_semantic_feedback_rounds <= self.maximum_hard_limit <= 8:
            raise ValueError("semantic feedback rounds must be within the hard limit of eight")
        if self.max_attempts_per_action_signature < 1 or self.max_total_actions < 1:
            raise ValueError("action attempt limits must be positive")
        if self.no_progress_patience < 1:
            raise ValueError("no_progress_patience must be positive")
        if not math.isfinite(float(self.max_total_action_cost)) or self.max_total_action_cost < 0.0:
            raise ValueError("max_total_action_cost must be finite and non-negative")
        if not math.isfinite(float(self.convergence_threshold)) or self.convergence_threshold < 0.0:
            raise ValueError("convergence_threshold must be finite and non-negative")
        for name in ("minimum_confidence", "minimum_evidence_quality", "conflict_threshold", "action_cost_weight"):
            _bounded(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class RecurrentFeedbackRequest:
    """Bind one Point 24 correction request to its original input and optional typed module requests."""

    request_id: str
    correction_request: CorrectionRequest
    input_snapshot: Mapping[str, Any] = field(default_factory=dict)
    initial_interpretation: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    compatibility_dimensions: Sequence[ConflictDimension] = ()
    module_inputs: Mapping[str, Any] = field(default_factory=dict)
    force_review: bool = False

    def __post_init__(self) -> None:
        """Validate identity and freeze evidence dimensions while retaining typed module inputs."""
        if not self.request_id.strip():
            raise ValueError("request_id must be non-empty")
        if not isinstance(self.correction_request, CorrectionRequest):
            raise TypeError("correction_request must be a CorrectionRequest")
        object.__setattr__(self, "input_snapshot", deepcopy(dict(self.input_snapshot)))
        object.__setattr__(self, "context", deepcopy(dict(self.context)))
        object.__setattr__(self, "compatibility_dimensions", tuple(self.compatibility_dimensions))
        # Module inputs may include live processor/request objects; copy the mapping, not the objects.
        object.__setattr__(self, "module_inputs", dict(self.module_inputs))

    def to_dict(self) -> dict[str, Any]:
        """Return serializable request metadata without attempting to serialize live processors."""
        return {
            "request_id": self.request_id,
            "correction_request": _plain(self.correction_request),
            "input_snapshot": _plain(self.input_snapshot),
            "initial_interpretation": _plain(self.initial_interpretation),
            "context": _plain(self.context),
            "compatibility_dimensions": [item.to_dict() for item in self.compatibility_dimensions],
            "module_input_names": sorted(self.module_inputs),
            "force_review": self.force_review,
        }


@dataclass(frozen=True, slots=True)
class RecurrentFeedbackResult:
    """Expose the final status, retained episode, latest correction, and complete operation history."""

    request_id: str
    status: str
    episode: FeedbackEpisode
    latest_corrective_result: CorrectiveResult | None
    operation_trace: Sequence[Mapping[str, Any]]
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the integrated result without dropping any unresolved alternatives."""
        return {
            "request_id": self.request_id,
            "status": self.status,
            "episode": self.episode.to_dict(),
            "latest_corrective_result": _plain(self.latest_corrective_result),
            "operation_trace": _plain(tuple(self.operation_trace)),
            "diagnostics": _plain(self.diagnostics),
        }
