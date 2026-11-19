"""Typed input, configuration, state, and result contracts for Point 07 feedback seeking."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import math
from typing import Any, Mapping


def _validate_unit_interval(name: str, value: float) -> None:
    """Reject non-finite values and values outside the normalized zero-to-one range."""
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1].")


def _validate_nonempty(name: str, value: str) -> None:
    """Require identifiers and labels to contain meaningful text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class TargetCriterion:
    """Represent one measurable requirement of the learner's current comprehension goal."""

    criterion_id: str
    description: str
    weight: float = 1.0
    achievement: float = 1.0

    def __post_init__(self) -> None:
        """Validate the criterion identity, importance, and observed achievement."""
        _validate_nonempty("criterion_id", self.criterion_id)
        _validate_nonempty("description", self.description)
        if not math.isfinite(float(self.weight)) or self.weight < 0.0:
            raise ValueError("weight must be finite and non-negative.")
        _validate_unit_interval("achievement", self.achievement)


@dataclass(frozen=True, slots=True)
class ReasoningNode:
    """Describe one inspectable reasoning step and its local evidence-consistency estimate."""

    node_id: str
    content: str
    consistency: float = 1.0
    parent_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the reasoning-node identifier and local consistency signal."""
        _validate_nonempty("node_id", self.node_id)
        _validate_nonempty("content", self.content)
        _validate_unit_interval("consistency", self.consistency)


@dataclass(frozen=True, slots=True)
class FeedbackSource:
    """Describe a feedback source with domain-specific reliability and information properties."""

    source_id: str
    name: str
    domain: str = "general"
    reliability: float = 0.5
    relevance: float = 0.5
    specificity: float = 0.5
    evidence_quality: float = 0.5
    noise: float = 0.0
    access_cost: float = 0.0
    outcome_likelihoods: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate normalized source-quality signals and non-negative access cost."""
        _validate_nonempty("source_id", self.source_id)
        _validate_nonempty("name", self.name)
        _validate_nonempty("domain", self.domain)
        for name in ("reliability", "relevance", "specificity", "evidence_quality", "noise"):
            _validate_unit_interval(name, getattr(self, name))
        if not math.isfinite(float(self.access_cost)) or self.access_cost < 0.0:
            raise ValueError("access_cost must be finite and non-negative.")
        for outcome, likelihoods in self.outcome_likelihoods.items():
            _validate_nonempty("outcome", str(outcome))
            for hypothesis, likelihood in likelihoods.items():
                _validate_nonempty("hypothesis", str(hypothesis))
                _validate_unit_interval("outcome likelihood", likelihood)


@dataclass(frozen=True, slots=True)
class FeedbackEvidence:
    """Represent one traceable feedback item, including correction and Bayesian likelihood data."""

    evidence_id: str
    source_id: str
    statement: str
    correction: str | None = None
    hypothesis_likelihoods: Mapping[str, float] = field(default_factory=dict)
    independence_key: str | None = None
    relevance: float = 0.5
    actionability: float = 0.5
    evidence_quality: float = 0.5
    noise: float = 0.0
    affected_components: tuple[str, ...] = ()
    corrected_component_values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate evidence identifiers, quality scores, and likelihood values."""
        _validate_nonempty("evidence_id", self.evidence_id)
        _validate_nonempty("source_id", self.source_id)
        _validate_nonempty("statement", self.statement)
        for name in ("relevance", "actionability", "evidence_quality", "noise"):
            _validate_unit_interval(name, getattr(self, name))
        for hypothesis, likelihood in self.hypothesis_likelihoods.items():
            _validate_nonempty("hypothesis", str(hypothesis))
            _validate_unit_interval("hypothesis likelihood", likelihood)


@dataclass(frozen=True, slots=True)
class PerformanceContext:
    """Capture temporary performance constraints separately from durable knowledge strength."""

    fatigue: float = 0.0
    distraction: float = 0.0
    working_memory_load: float = 0.0
    time_pressure: float = 0.0

    def __post_init__(self) -> None:
        """Validate the bounded state variables used by the performance-error model."""
        for name in ("fatigue", "distraction", "working_memory_load", "time_pressure"):
            _validate_unit_interval(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class TransferProbe:
    """Describe a new-context test, retaining structural similarity and counterexample status."""

    probe_id: str
    context: str
    expected_relation: str
    is_counterexample: bool = False
    correctness: float | None = None
    structural_similarity: float = 0.5
    surface_similarity: float = 0.5

    def __post_init__(self) -> None:
        """Validate transfer-probe identity and optional observed correctness."""
        _validate_nonempty("probe_id", self.probe_id)
        _validate_nonempty("context", self.context)
        _validate_nonempty("expected_relation", self.expected_relation)
        if self.correctness is not None:
            _validate_unit_interval("correctness", self.correctness)
        _validate_unit_interval("structural_similarity", self.structural_similarity)
        _validate_unit_interval("surface_similarity", self.surface_similarity)


@dataclass(frozen=True, slots=True)
class FeedbackRequest:
    """Provide all caller-observable evidence required for a Point 07 feedback episode."""

    request_id: str
    learning_target: str
    candidate_response: str
    domain: str = "general"
    criteria: tuple[TargetCriterion, ...] = ()
    reasoning_chain: tuple[ReasoningNode, ...] = ()
    confidence: float = 0.5
    ambiguity: float = 0.0
    importance: float = 0.5
    stakes: float = 0.5
    feedback_sources: tuple[FeedbackSource, ...] = ()
    feedback_evidence: tuple[FeedbackEvidence, ...] = ()
    hypothesis_priors: Mapping[str, float] = field(default_factory=dict)
    error_hints: Mapping[str, float] = field(default_factory=dict)
    knowledge_components: Mapping[str, Any] = field(default_factory=dict)
    affected_components: tuple[str, ...] = ()
    corrected_components: Mapping[str, Any] = field(default_factory=dict)
    corrected_response: str | None = None
    supporting_concepts: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    counterexample: str | None = None
    positive_example: str | None = None
    boundary_examples: tuple[str, ...] = ()
    retrieval_response: str | None = None
    retrieval_correctness: float | None = None
    transfer_probes: tuple[TransferProbe, ...] = ()
    delayed_check_days: float = 1.0
    performance: PerformanceContext = field(default_factory=PerformanceContext)

    def __post_init__(self) -> None:
        """Validate request identity, normalized estimates, and evidence references."""
        _validate_nonempty("request_id", self.request_id)
        _validate_nonempty("learning_target", self.learning_target)
        _validate_nonempty("candidate_response", self.candidate_response)
        _validate_nonempty("domain", self.domain)
        for name in ("confidence", "ambiguity", "importance", "stakes"):
            _validate_unit_interval(name, getattr(self, name))
        if not math.isfinite(float(self.delayed_check_days)) or self.delayed_check_days < 0.0:
            raise ValueError("delayed_check_days must be finite and non-negative.")
        if self.retrieval_correctness is not None:
            _validate_unit_interval("retrieval_correctness", self.retrieval_correctness)
        source_ids = {source.source_id for source in self.feedback_sources}
        unknown_sources = {
            evidence.source_id for evidence in self.feedback_evidence
            if evidence.source_id not in source_ids
        }
        if unknown_sources:
            raise ValueError(f"Feedback evidence references unknown sources: {sorted(unknown_sources)}")
        for hypothesis, prior in self.hypothesis_priors.items():
            _validate_nonempty("hypothesis", str(hypothesis))
            if not math.isfinite(float(prior)) or prior < 0.0:
                raise ValueError("Hypothesis priors must be finite and non-negative.")
        for category, weight in self.error_hints.items():
            _validate_nonempty("error category", str(category))
            _validate_unit_interval("error hint", weight)


@dataclass(slots=True)
class FeedbackState:
    """Maintain cross-episode knowledge, recurring-error, confidence, and review information."""

    knowledge_components: dict[str, Any] = field(default_factory=dict)
    error_frequencies: dict[str, float] = field(default_factory=dict)
    confidence_history: list[dict[str, float]] = field(default_factory=list)
    correction_history: list[dict[str, Any]] = field(default_factory=list)
    review_queue: list[dict[str, Any]] = field(default_factory=list)
    strategy_outcomes: dict[str, dict[str, float]] = field(default_factory=dict)
    episode_count: int = 0

    def snapshot(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the current learner state."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeedbackConfig:
    """Configure thresholds and learning rates for the functional feedback approximation."""

    uncertainty_threshold: float = 0.35
    feedback_quality_threshold: float = 0.45
    goal_completeness_threshold: float = 0.80
    uncertainty_confidence_weight: float = 0.45
    uncertainty_inconsistency_weight: float = 0.35
    uncertainty_ambiguity_weight: float = 0.20
    source_cost_weight: float = 0.10
    correction_learning_rate: float = 0.50
    error_frequency_decay: float = 0.85
    retention_decay_rate: float = 0.12
    retention_target: float = 0.65
    feedback_load_threshold: float = 0.65

    def __post_init__(self) -> None:
        """Validate normalized thresholds and positive retention dynamics."""
        for name in (
            "uncertainty_threshold", "feedback_quality_threshold",
            "goal_completeness_threshold", "uncertainty_confidence_weight",
            "uncertainty_inconsistency_weight", "uncertainty_ambiguity_weight",
            "source_cost_weight", "correction_learning_rate",
            "error_frequency_decay", "retention_target", "feedback_load_threshold",
        ):
            _validate_unit_interval(name, getattr(self, name))
        if self.retention_decay_rate <= 0.0 or not math.isfinite(self.retention_decay_rate):
            raise ValueError("retention_decay_rate must be finite and positive.")
        weight_sum = (
            self.uncertainty_confidence_weight
            + self.uncertainty_inconsistency_weight
            + self.uncertainty_ambiguity_weight
        )
        if weight_sum <= 0.0:
            raise ValueError("At least one uncertainty weight must be positive.")


@dataclass(frozen=True, slots=True)
class FeedbackResult:
    """Return functional and biological traces, stage outputs, diagnostics, and final state."""

    request_id: str
    operation_trace: tuple[dict[str, Any], ...]
    functional_outputs: Mapping[str, Any]
    biological_outputs: tuple[dict[str, Any], ...]
    diagnostics: Mapping[str, Any]
    final_state: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the complete result into nested Python dictionaries and lists."""
        def convert(value: Any) -> Any:
            if is_dataclass(value):
                return asdict(value)
            if isinstance(value, Mapping):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            return value

        return convert(self)
