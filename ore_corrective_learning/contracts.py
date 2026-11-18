"""Validated data contracts for Point 24 corrective learning."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


def _probability(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")


@dataclass(frozen=True)
class KnowledgeHypothesis:
    """A candidate interpretation and the outcomes it predicts."""

    hypothesis_id: str
    statement: str
    prior: float = 0.5
    predictions: Mapping[str, float] = field(default_factory=dict)
    components: Mapping[str, Any] = field(default_factory=dict)
    source: str = "inferred"

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip() or not self.statement.strip():
            raise ValueError("hypothesis_id and statement must be non-empty")
        _probability("hypothesis prior", self.prior)
        if self.source not in {"observed", "reported", "inferred", "recalled", "assumed", "instruction"}:
            raise ValueError("unsupported hypothesis source")
        if any(not str(key).strip() for key in self.predictions):
            raise ValueError("prediction outcome labels must be non-empty")
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in self.predictions.values()):
            raise ValueError("prediction probabilities must be finite and non-negative")
        if self.predictions and sum(float(v) for v in self.predictions.values()) <= 0:
            raise ValueError("prediction probabilities must have a positive sum")


@dataclass(frozen=True)
class FeedbackEvidence:
    """One feedback item with explicit likelihoods and provenance."""

    evidence_id: str
    description: str
    likelihoods: Mapping[str, float]
    source_name: str = "unspecified"
    source_type: str = "reported"
    reliability: float = 0.5
    relevance: float = 1.0
    diagnosticity: float = 1.0
    directness: str = "indirect"
    independence_key: str = ""
    context: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_id.strip() or not self.description.strip():
            raise ValueError("evidence_id and description must be non-empty")
        if self.source_type not in {"measurement", "expert", "primary_document", "reported", "memory", "model", "other"}:
            raise ValueError("unsupported evidence source_type")
        if self.directness not in {"direct", "indirect", "hearsay"}:
            raise ValueError("directness must be direct, indirect, or hearsay")
        for name in ("reliability", "relevance", "diagnosticity"):
            _probability(name, getattr(self, name))
        for key, value in self.likelihoods.items():
            if not str(key).strip():
                raise ValueError("likelihood hypothesis identifiers must be non-empty")
            _probability(f"likelihoods[{key}]", value)


@dataclass(frozen=True)
class ClarificationOption:
    """A possible question, source check, or test with response models per hypothesis."""

    action_id: str
    kind: str
    description: str
    response_models: Mapping[str, Mapping[str, float]]
    cost: float = 0.0

    def __post_init__(self) -> None:
        if not self.action_id.strip() or not self.description.strip():
            raise ValueError("action_id and description must be non-empty")
        if self.kind not in {"infer", "clarify", "check", "test"}:
            raise ValueError("action kind must be infer, clarify, check, or test")
        _probability("action cost", self.cost)
        for hypothesis_id, outcomes in self.response_models.items():
            if not hypothesis_id.strip() or not outcomes:
                raise ValueError("each response model needs a hypothesis and outcomes")
            if any(float(value) < 0 or not math.isfinite(float(value)) for value in outcomes.values()):
                raise ValueError("response probabilities must be finite and non-negative")
            if sum(float(value) for value in outcomes.values()) <= 0:
                raise ValueError("response probabilities must have a positive sum")


@dataclass(frozen=True)
class RetrievalAttempt:
    """A learner's unaided reconstruction of a corrected interpretation."""

    hypothesis_id: str
    confidence: float
    delay_days: float = 0.0
    response_text: str = ""

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("retrieval hypothesis_id must be non-empty")
        _probability("retrieval confidence", self.confidence)
        if self.delay_days < 0 or not math.isfinite(float(self.delay_days)):
            raise ValueError("delay_days must be finite and non-negative")


@dataclass(frozen=True)
class TransferProbe:
    """A new case with a caller-supplied target and learner response."""

    probe_id: str
    context: str
    target_hypothesis_id: str
    response_hypothesis_id: str
    is_counterexample: bool = False

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (
            self.probe_id, self.context, self.target_hypothesis_id, self.response_hypothesis_id
        )):
            raise ValueError("transfer probe identifiers and context must be non-empty")


@dataclass(frozen=True)
class CorrectionRequest:
    """All inputs required to run one transparent corrective-learning episode."""

    request_id: str
    hypotheses: tuple[KnowledgeHypothesis, ...]
    current_hypothesis_id: str
    observed_outcome: str | None = None
    observed_value: float | None = None
    predicted_value_by_hypothesis: Mapping[str, float] = field(default_factory=dict)
    context: str = ""
    error_hints: Mapping[str, float] = field(default_factory=dict)
    evidence: tuple[FeedbackEvidence, ...] = ()
    clarification_options: tuple[ClarificationOption, ...] = ()
    verified_hypothesis_id: str | None = None
    affected_components: tuple[str, ...] = ()
    explanation: Mapping[str, Any] = field(default_factory=dict)
    retrieval_attempt: RetrievalAttempt | None = None
    retrieval_feedback_available: bool = True
    transfer_probes: tuple[TransferProbe, ...] = ()
    confidence_estimate: float | None = None
    task_stakes: str = "normal"
    context_relevance: float = 1.0

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.hypotheses:
            raise ValueError("request_id and at least one hypothesis are required")
        ids = [item.hypothesis_id for item in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("hypothesis identifiers must be unique")
        if self.current_hypothesis_id not in ids:
            raise ValueError("current_hypothesis_id must match a supplied hypothesis")
        if self.verified_hypothesis_id is not None and self.verified_hypothesis_id not in ids:
            raise ValueError("verified_hypothesis_id must match a supplied hypothesis")
        for item in self.evidence:
            unknown = set(item.likelihoods) - set(ids)
            if unknown:
                raise ValueError(f"evidence references unknown hypotheses: {sorted(unknown)}")
        if self.observed_value is not None and not math.isfinite(float(self.observed_value)):
            raise ValueError("observed_value must be finite")
        if self.predicted_value_by_hypothesis and set(self.predicted_value_by_hypothesis) - set(ids):
            raise ValueError("predicted numeric values reference unknown hypotheses")
        for name, value in self.error_hints.items():
            _probability(f"error_hints[{name}]", value)
        if self.confidence_estimate is not None:
            _probability("confidence_estimate", self.confidence_estimate)
        _probability("context_relevance", self.context_relevance)
        if self.task_stakes not in {"low", "normal", "high"}:
            raise ValueError("task_stakes must be low, normal, or high")
        if self.retrieval_attempt and self.retrieval_attempt.hypothesis_id not in ids:
            raise ValueError("retrieval attempt must name a supplied hypothesis")
        for probe in self.transfer_probes:
            if probe.target_hypothesis_id not in ids or probe.response_hypothesis_id not in ids:
                raise ValueError("transfer probes must refer to supplied hypotheses")


@dataclass(frozen=True)
class CorrectiveConfig:
    """Engineering thresholds; tune on behavioural data before empirical claims."""

    mismatch_threshold: float = 0.15
    minimum_evidence_quality: float = 0.35
    correction_posterior_threshold: float = 0.80
    uncertainty_threshold: float = 0.70
    retrieval_threshold: float = 0.70
    transfer_threshold: float = 0.75
    learning_rate: float = 0.25
    forgetting_rate: float = 0.035
    consolidation_rate: float = 0.20
    independence_floor: float = 0.05
    simulate_biology: bool = True

    def __post_init__(self) -> None:
        for name in (
            "mismatch_threshold", "minimum_evidence_quality", "correction_posterior_threshold",
            "uncertainty_threshold", "retrieval_threshold", "transfer_threshold",
            "learning_rate", "consolidation_rate", "independence_floor",
        ):
            _probability(name, getattr(self, name))
        if self.forgetting_rate < 0 or not math.isfinite(self.forgetting_rate):
            raise ValueError("forgetting_rate must be finite and non-negative")


@dataclass
class CorrectiveState:
    """Persistent, inspectable state for correction memory and calibration."""

    belief_accessibility: dict[str, float] = field(default_factory=dict)
    correction_accessibility: dict[str, float] = field(default_factory=dict)
    consolidation_strength: dict[str, float] = field(default_factory=dict)
    component_values: dict[str, Any] = field(default_factory=dict)
    correction_history: list[dict[str, Any]] = field(default_factory=list)
    calibration_observations: list[dict[str, float]] = field(default_factory=list)
    run_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief_accessibility": dict(self.belief_accessibility),
            "correction_accessibility": dict(self.correction_accessibility),
            "consolidation_strength": dict(self.consolidation_strength),
            "component_values": dict(self.component_values),
            "correction_history": [dict(item) for item in self.correction_history],
            "calibration_observations": [dict(item) for item in self.calibration_observations],
            "run_count": self.run_count,
        }


@dataclass(frozen=True)
class CorrectiveResult:
    """Final interpretation, evidence trail, tests, uncertainty, and biological activity."""

    request_id: str
    initial_hypothesis_id: str
    selected_hypothesis_id: str
    posterior_probabilities: Mapping[str, float]
    prediction: Mapping[str, Any]
    mismatch: Mapping[str, Any]
    error_classification: Mapping[str, float]
    information_action: Mapping[str, Any]
    evidence_assessment: Mapping[str, Any]
    prior_interpretation_record: Mapping[str, Any]
    correction: Mapping[str, Any]
    component_updates: Mapping[str, Any]
    explanation_model: Mapping[str, Any]
    retrieval: Mapping[str, Any]
    retrieval_feedback: Mapping[str, Any]
    transfer: Mapping[str, Any]
    calibration: Mapping[str, Any]
    consolidation: Mapping[str, Any]
    regression: Mapping[str, Any]
    stopping: Mapping[str, Any]
    unresolved_questions: tuple[str, ...]
    operation_trace: tuple[Mapping[str, Any], ...]
    biological: Mapping[str, Any]
    final_state: Mapping[str, Any]
    metadata: Mapping[str, Any]
