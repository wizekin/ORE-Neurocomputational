"""Typed contracts for curiosity-guided, motivated learning and memory consolidation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import math
from typing import Any, Mapping


def _unit(name: str, value: float) -> float:
    """Validate and return a finite normalized scalar."""
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return numeric


def _nonnegative(name: str, value: float) -> float:
    """Validate a finite non-negative model parameter."""
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return numeric


def _required(name: str, value: str) -> str:
    """Require a meaningful identifier or descriptive string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


@dataclass(frozen=True, slots=True)
class LearningUnit:
    """Represent one goal-linked concept, claim, example, question, or event from the learning material."""

    unit_id: str
    content: str
    relevance: float = 0.5
    novelty: float = 0.5
    uncertainty: float = 0.5
    utility: float = 0.5
    intrinsic_interest: float = 0.5
    transfer_value: float = 0.5
    personal_relevance: float = 0.5
    conceptual_distance: float = 0.5
    missing_prerequisites: float = 0.0
    available_support: float = 0.5
    expected_value: float | None = None
    expected_features: tuple[str, ...] = ()
    target: bool = True
    temporal_context_link: float = 0.5
    semantic_context_link: float = 0.5
    elaboration_hint: float = 0.5
    reliability: float = 0.5
    future_utility: float = 0.5

    def __post_init__(self) -> None:
        """Validate identity, content, and every bounded feature estimate."""
        _required("unit_id", self.unit_id)
        _required("content", self.content)
        for name in (
            "relevance", "novelty", "uncertainty", "utility", "intrinsic_interest",
            "transfer_value", "personal_relevance", "conceptual_distance",
            "missing_prerequisites", "available_support", "temporal_context_link",
            "semantic_context_link", "elaboration_hint", "reliability", "future_utility",
        ):
            _unit(name, getattr(self, name))
        if self.expected_value is not None:
            _unit("expected_value", self.expected_value)


@dataclass(frozen=True, slots=True)
class LearningAction:
    """Describe a possible learning action and its expected gain, information, interest, and costs."""

    action_id: str
    unit_id: str
    action_type: str
    expected_learning_gain: float = 0.5
    expected_information_gain: float = 0.5
    expected_interest_gain: float = 0.5
    effort_cost: float = 0.5
    time_cost: float = 0.5
    stress_cost: float = 0.0
    source_reliability: float = 0.5
    available: bool = True
    learner_choice: bool = False
    question: str | None = None
    evidence: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Validate action identity and normalized estimates used by action selection."""
        for name in ("action_id", "unit_id", "action_type"):
            _required(name, getattr(self, name))
        for name in (
            "expected_learning_gain", "expected_information_gain",
            "expected_interest_gain", "effort_cost", "time_cost",
            "stress_cost", "source_reliability",
        ):
            _unit(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class LearningOutcome:
    """Record an observed action outcome without manufacturing unobserved learning or reward labels."""

    action_id: str
    observed_learning: float | None = None
    observed_interest_gain: float | None = None
    completed: bool = True
    evidence_quality: float = 0.5
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate any caller-observed scores while preserving missing values as unknown."""
        _required("action_id", self.action_id)
        if self.observed_learning is not None:
            _unit("observed_learning", self.observed_learning)
        if self.observed_interest_gain is not None:
            _unit("observed_interest_gain", self.observed_interest_gain)
        _unit("evidence_quality", self.evidence_quality)


@dataclass(frozen=True, slots=True)
class RetrievalCheck:
    """Store delayed factual, relational, transfer, and confidence measurements separately."""

    check_id: str
    factual_accuracy: float | None = None
    relational_accuracy: float | None = None
    transfer_accuracy: float | None = None
    confidence: float | None = None
    delayed_days: float = 0.0
    target_unit_id: str | None = None

    def __post_init__(self) -> None:
        """Validate observed retrieval metrics and elapsed review time."""
        _required("check_id", self.check_id)
        for name in ("factual_accuracy", "relational_accuracy", "transfer_accuracy", "confidence"):
            value = getattr(self, name)
            if value is not None:
                _unit(name, value)
        _nonnegative("delayed_days", self.delayed_days)


@dataclass(frozen=True, slots=True)
class LearningRequest:
    """Supply learning material, goals, candidate actions, observed outcomes, and current learner context."""

    request_id: str
    goal: str
    units: tuple[LearningUnit, ...]
    actions: tuple[LearningAction, ...] = ()
    outcomes: tuple[LearningOutcome, ...] = ()
    retrieval_checks: tuple[RetrievalCheck, ...] = ()
    intrinsic_motivation: float = 0.5
    extrinsic_motivation: float = 0.5
    personal_value: float = 0.5
    stress: float = 0.0
    fatigue: float = 0.0
    boredom: float = 0.0
    frustration: float = 0.0
    distraction: float = 0.0
    attention_available: float = 1.0
    time_available: float = 1.0
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.2
    perceived_control: float = 0.7
    emotional_relevance: float = 0.5
    observed_error_rate: float | None = None
    task_switch_rate: float = 0.0
    observed_milestones: float = 0.0
    planned_milestones: float = 1.0
    target_success_rate: float = 0.75
    sleep_quality: float = 0.7
    time_since_learning_days: float = 0.0
    hypothesis_priors: Mapping[str, float] = field(default_factory=dict)
    expected_learning_baseline: float = 0.5

    def __post_init__(self) -> None:
        """Validate request shape, context variables, outcome references, and learning-state bounds."""
        _required("request_id", self.request_id)
        _required("goal", self.goal)
        if not self.units:
            raise ValueError("At least one LearningUnit is required.")
        unit_ids = {unit.unit_id for unit in self.units}
        if len(unit_ids) != len(self.units):
            raise ValueError("LearningUnit identifiers must be unique within a request.")
        for action in self.actions:
            if action.unit_id not in unit_ids:
                raise ValueError(f"Action {action.action_id} references unknown unit {action.unit_id}.")
        action_ids = {action.action_id for action in self.actions}
        if len(action_ids) != len(self.actions):
            raise ValueError("LearningAction identifiers must be unique within a request.")
        unknown_retrieval_targets = {item.target_unit_id for item in self.retrieval_checks
                                     if item.target_unit_id is not None and item.target_unit_id not in unit_ids}
        if unknown_retrieval_targets:
            raise ValueError(f"Retrieval checks reference unknown units: {sorted(unknown_retrieval_targets)}")
        unknown_outcomes = {item.action_id for item in self.outcomes if item.action_id not in action_ids}
        if unknown_outcomes:
            raise ValueError(f"Outcomes reference unknown actions: {sorted(unknown_outcomes)}")
        for name in (
            "intrinsic_motivation", "extrinsic_motivation", "personal_value",
            "stress", "fatigue", "boredom", "frustration", "distraction",
            "attention_available", "time_available", "emotional_arousal",
            "perceived_control", "emotional_relevance", "task_switch_rate",
            "target_success_rate", "sleep_quality", "expected_learning_baseline",
        ):
            _unit(name, getattr(self, name))
        if not math.isfinite(float(self.emotional_valence)) or not -1.0 <= self.emotional_valence <= 1.0:
            raise ValueError("emotional_valence must be finite and in [-1, 1].")
        if self.observed_error_rate is not None:
            _unit("observed_error_rate", self.observed_error_rate)
        _nonnegative("observed_milestones", self.observed_milestones)
        _nonnegative("planned_milestones", self.planned_milestones)
        _nonnegative("time_since_learning_days", self.time_since_learning_days)
        for key, value in self.hypothesis_priors.items():
            _required("hypothesis key", str(key))
            _nonnegative("hypothesis prior", value)


@dataclass(slots=True)
class LearningState:
    """Maintain knowledge, motivation, agency, memory candidates, and cross-episode learning history."""

    knowledge: dict[str, Any] = field(default_factory=dict)
    goal: str = ""
    curiosity: float = 0.0
    interest: float = 0.0
    intrinsic_value: float = 0.0
    extrinsic_value: float = 0.0
    emotional_salience: float = 0.0
    stress: float = 0.0
    fatigue: float = 0.0
    attention_available: float = 1.0
    confidence: dict[str, float] = field(default_factory=dict)
    unresolved_questions: list[str] = field(default_factory=list)
    memory_candidates: list[dict[str, Any]] = field(default_factory=list)
    incidental_memory_candidates: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    action_values: dict[str, float] = field(default_factory=dict)
    competence: float = 0.5
    topic_values: dict[str, float] = field(default_factory=dict)
    strategy_success: dict[str, dict[str, float]] = field(default_factory=dict)
    episode_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible state snapshot."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MotivationConfig:
    """Configure model weights, learning rates, thresholds, and time constants."""

    curiosity_bias: float = -0.5
    curiosity_gap_weight: float = 1.2
    curiosity_value_weight: float = 1.1
    curiosity_learnability_weight: float = 0.9
    curiosity_threat_weight: float = 1.0
    curiosity_fatigue_weight: float = 0.8
    softmax_temperature: float = 0.35
    learning_rate: float = 0.25
    value_learning_rate: float = 0.20
    consolidation_rate: float = 0.35
    incidental_rate: float = 0.15
    competence_rate: float = 0.15
    challenge_rate: float = 0.20
    retention_rate: float = 0.15
    disengagement_threshold: float = 0.65
    high_stress_threshold: float = 0.80
    high_fatigue_threshold: float = 0.85
    action_stop_margin: float = 0.0
    arousal_optimum: float = 0.45
    arousal_width: float = 0.30
    error_frequency_decay: float = 0.85

    def __post_init__(self) -> None:
        """Validate bounded configuration values and a positive softmax temperature."""
        for name in (
            "learning_rate",
            "value_learning_rate", "consolidation_rate", "incidental_rate",
            "competence_rate", "challenge_rate", "retention_rate",
            "disengagement_threshold", "high_stress_threshold", "high_fatigue_threshold",
            "arousal_optimum", "arousal_width", "error_frequency_decay",
        ):
            _unit(name, getattr(self, name))
        for name in ("curiosity_gap_weight", "curiosity_value_weight",
                     "curiosity_learnability_weight", "curiosity_threat_weight",
                     "curiosity_fatigue_weight"):
            _nonnegative(name, getattr(self, name))
        if self.arousal_width <= 0.0:
            raise ValueError("arousal_width must be positive.")
        if self.softmax_temperature <= 0.0 or not math.isfinite(self.softmax_temperature):
            raise ValueError("softmax_temperature must be finite and positive.")
        if not math.isfinite(self.curiosity_bias) or not math.isfinite(self.action_stop_margin):
            raise ValueError("Bias and stop margin must be finite.")


@dataclass(frozen=True, slots=True)
class LearningResult:
    """Return the ordered 24-stage trace, functional state, biological reference, and evaluation metrics."""

    request_id: str
    operation_trace: tuple[dict[str, Any], ...]
    functional_outputs: Mapping[str, Any]
    biological_outputs: tuple[dict[str, Any], ...]
    diagnostics: Mapping[str, Any]
    final_state: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert nested dataclasses and containers to plain JSON-compatible objects."""
        def convert(value: Any) -> Any:
            if is_dataclass(value):
                return asdict(value)
            if isinstance(value, Mapping):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            return value
        return convert(self)
