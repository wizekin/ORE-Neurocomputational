"""Typed contracts for Point 13 emotion, motivation, attention, and memory encoding.

All numeric variables are computational model states or task measurements. They are not direct
estimates of neurotransmitter concentrations, hormone levels, neuronal firing, or clinical states.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping


def _finite(name: str, value: float) -> float:
    """Convert a numeric value to a finite float or reject it."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _probability(name: str, value: float) -> float:
    """Validate a normalized scalar in the closed interval [0, 1]."""
    result = _finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(name + " must be in [0, 1]")
    return result


def _valence(name: str, value: float) -> float:
    """Validate a signed affective value in the interval [-1, 1]."""
    result = _finite(name, value)
    if not -1.0 <= result <= 1.0:
        raise ValueError(name + " must be in [-1, 1]")
    return result


def _text(name: str, value: str) -> str:
    """Validate a non-empty identifier and return its trimmed form."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a non-empty string")
    return value.strip()


def _plain(value: Any) -> Any:
    """Convert dataclasses and nested containers to JSON-compatible values."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, set):
        return sorted((_plain(item) for item in value), key=repr)
    if hasattr(value, "item") and callable(value.item):
        return _plain(value.item())
    return value


@dataclass(frozen=True, slots=True)
class EventFeature:
    """Represent one event feature with salience, task relevance, and centrality metadata."""

    feature_id: str
    value: float = 1.0
    salience: float = 0.5
    goal_relevance: float = 0.5
    centrality: float = 0.5
    valence: float = 0.0
    arousal: float = 0.0
    tags: tuple[str, ...] = ()
    role: str = "context"
    provenance: str = "current_event"

    def __post_init__(self) -> None:
        """Validate feature identity, activation, affect, and central/peripheral role."""
        _text("feature_id", self.feature_id)
        _finite("value", self.value)
        for name in ("salience", "goal_relevance", "centrality", "arousal"):
            _probability(name, getattr(self, name))
        _valence("valence", self.valence)
        if self.role not in {"central", "peripheral", "context"}:
            raise ValueError("role must be central, peripheral, or context")
        object.__setattr__(self, "tags", tuple(_text("tag", item).lower() for item in self.tags))
        _text("provenance", self.provenance)

    def to_dict(self) -> dict[str, Any]:
        """Return the feature with its evidence and attention-related attributes."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class LearningExperience:
    """Describe a new experience, its goals, context, body-state cues, and learning demands."""

    event_id: str
    features: tuple[EventFeature, ...]
    context: Mapping[str, Any] = field(default_factory=dict)
    goal: str = ""
    task_demands: float = 0.5
    novelty: float = 0.0
    uncertainty: float = 0.0
    threat: float = 0.0
    reward_relevance: float = 0.0
    valence_hint: float = 0.0
    arousal_hint: float = 0.0
    fatigue: float = 0.0
    sleep_replay_support: float = 0.0
    retrieval_cues: tuple[str, ...] = ()
    body_state: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate event state and detach mutable mappings."""
        _text("event_id", self.event_id)
        if not self.features:
            raise ValueError("features must contain at least one EventFeature")
        identifiers = [item.feature_id for item in self.features]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("feature_id values must be unique within an experience")
        for name in (
            "task_demands", "novelty", "uncertainty", "threat",
            "reward_relevance", "arousal_hint", "fatigue", "sleep_replay_support",
        ):
            _probability(name, getattr(self, name))
        _valence("valence_hint", self.valence_hint)
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "retrieval_cues", tuple(
            _text("retrieval_cue", cue).lower() for cue in self.retrieval_cues
        ))
        state = {str(key): _probability("body_state." + str(key), value)
                 for key, value in self.body_state.items()}
        object.__setattr__(self, "body_state", state)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return the experience while preserving the source of its features."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class AffectiveAppraisal:
    """Represent estimated valence, arousal, significance, and distinct appraisal dimensions."""

    valence: float
    arousal: float
    significance: float
    threat: float
    reward_relevance: float
    novelty: float
    uncertainty: float
    confidence: float = 0.5
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate signed valence and bounded appraisal dimensions."""
        _valence("valence", self.valence)
        for name in (
            "arousal", "significance", "threat", "reward_relevance",
            "novelty", "uncertainty", "confidence",
        ):
            _probability(name, getattr(self, name))
        object.__setattr__(self, "labels", tuple(str(item) for item in self.labels))

    def to_dict(self) -> dict[str, Any]:
        """Return a plain appraisal record."""
        return _plain(asdict(self))


@dataclass
class ArousalState:
    """Maintain separate fast arousal, slow stress-load, NE-gain, glucocorticoid, and fatigue proxies."""

    fast_arousal: float = 0.0
    slow_stress_load: float = 0.0
    norepinephrine_gain: float = 0.5
    glucocorticoid_load: float = 0.0
    fatigue: float = 0.0

    def __post_init__(self) -> None:
        """Validate each distinct physiological-state proxy."""
        for name in (
            "fast_arousal", "slow_stress_load", "norepinephrine_gain",
            "glucocorticoid_load", "fatigue",
        ):
            setattr(self, name, _probability(name, getattr(self, name)))

    def to_dict(self) -> dict[str, float]:
        """Return the state proxies without implying direct physiological measurements."""
        return asdict(self)


@dataclass
class MotivationState:
    """Maintain the active learning goal, expected value, effort tolerance, and progress."""

    goal: str = ""
    goal_strength: float = 0.5
    expected_reward: float = 0.0
    effort_tolerance: float = 0.5
    progress: float = 0.0
    active: bool = True
    action_values: dict[str, float] = field(default_factory=dict)
    state_values: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate motivation scalars and detach value-function dictionaries."""
        for name in ("goal_strength", "effort_tolerance", "progress"):
            setattr(self, name, _probability(name, getattr(self, name)))
        self.expected_reward = _finite("expected_reward", self.expected_reward)
        self.action_values = {str(k): _finite("action_value", v)
                              for k, v in self.action_values.items()}
        self.state_values = {str(k): _finite("state_value", v)
                             for k, v in self.state_values.items()}

    def to_dict(self) -> dict[str, Any]:
        """Return the goal and learned action/state values."""
        return _plain(asdict(self))


@dataclass
class LearningState:
    """Hold persistent arousal, motivation, memory traces, and learned associations."""

    arousal: ArousalState = field(default_factory=ArousalState)
    motivation: MotivationState = field(default_factory=MotivationState)
    memory_traces: list[dict[str, Any]] = field(default_factory=list)
    associative_weights: dict[str, float] = field(default_factory=dict)
    concept_strengths: dict[str, float] = field(default_factory=dict)
    appraisal_history: list[dict[str, Any]] = field(default_factory=list)
    event_count: int = 0

    def __post_init__(self) -> None:
        """Normalize state containers and validate bounded associative weights."""
        self.memory_traces = list(self.memory_traces)
        self.associative_weights = {
            str(k): _probability("associative_weight", v)
            for k, v in self.associative_weights.items()
        }
        self.concept_strengths = {
            str(k): _probability("concept_strength", v)
            for k, v in self.concept_strengths.items()
        }
        self.appraisal_history = list(self.appraisal_history)
        if self.event_count < 0:
            raise ValueError("event_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete persistent state in a serializable form."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class AttentionAllocation:
    """Store feature priorities and bounded attentional resource allocations."""

    priorities: Mapping[str, float]
    weights: Mapping[str, float]
    selected_feature_ids: tuple[str, ...]
    entropy: float
    capacity_budget: float

    def __post_init__(self) -> None:
        """Validate normalized attention weights and resource-budget constraints."""
        priorities = {str(k): _probability("priority", v) for k, v in self.priorities.items()}
        weights = {str(k): _probability("attention_weight", v) for k, v in self.weights.items()}
        if set(priorities) != set(weights):
            raise ValueError("priorities and weights must contain identical feature IDs")
        if sum(weights.values()) > _probability("capacity_budget", self.capacity_budget) + 1e-8:
            raise ValueError("attention allocations exceed the capacity budget")
        _probability("entropy", self.entropy)
        object.__setattr__(self, "priorities", priorities)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "selected_feature_ids", tuple(self.selected_feature_ids))

    def to_dict(self) -> dict[str, Any]:
        """Return the attention allocation and its resource budget."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class RewardFeedback:
    """Represent an observed action outcome for temporal-difference learning."""

    state_id: str
    action_id: str
    reward: float
    next_state_id: str = ""
    terminal: bool = False

    def __post_init__(self) -> None:
        """Validate state/action identifiers and finite observed reward."""
        _text("state_id", self.state_id)
        _text("action_id", self.action_id)
        _finite("reward", self.reward)

    def to_dict(self) -> dict[str, Any]:
        """Return the outcome feedback used for value updating."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class EffortDecision:
    """Record selected effort, expected progress, utility, and persistence decision."""

    effort: float
    expected_progress: float
    estimated_cost: float
    net_utility: float
    continue_task: bool
    recommended_strategy: str

    def __post_init__(self) -> None:
        """Validate effort and progress estimates while permitting signed net utility."""
        for name in ("effort", "expected_progress", "estimated_cost"):
            _probability(name, getattr(self, name))
        _finite("net_utility", self.net_utility)
        _text("recommended_strategy", self.recommended_strategy)

    def to_dict(self) -> dict[str, Any]:
        """Return the effort decision and whether engagement should continue."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class MemoryTrace:
    """Represent selectively encoded event features and their consolidation state."""

    event_id: str
    feature_strengths: Mapping[str, float]
    encoded_feature_ids: tuple[str, ...]
    central_feature_ids: tuple[str, ...]
    peripheral_feature_ids: tuple[str, ...]
    relation_links: tuple[tuple[str, str, float], ...]
    context_tags: tuple[str, ...]
    encoding_strength: float
    consolidation_strength: float
    amygdala_modulation: float
    elaboration_links: tuple[str, ...] = ()
    provenance: str = "simulated_event_encoding"

    def __post_init__(self) -> None:
        """Validate feature-strength maps and memory strength variables."""
        _text("event_id", self.event_id)
        strengths = {str(k): _probability("feature_strength", v)
                     for k, v in self.feature_strengths.items()}
        object.__setattr__(self, "feature_strengths", strengths)
        for name in ("encoding_strength", "consolidation_strength"):
            _probability(name, getattr(self, name))
        _probability("amygdala_modulation", self.amygdala_modulation)
        object.__setattr__(self, "encoded_feature_ids", tuple(self.encoded_feature_ids))
        object.__setattr__(self, "central_feature_ids", tuple(self.central_feature_ids))
        object.__setattr__(self, "peripheral_feature_ids", tuple(self.peripheral_feature_ids))
        object.__setattr__(self, "relation_links", tuple(self.relation_links))
        object.__setattr__(self, "context_tags", tuple(self.context_tags))
        object.__setattr__(self, "elaboration_links", tuple(self.elaboration_links))
        _text("provenance", self.provenance)

    def to_dict(self) -> dict[str, Any]:
        """Return an event trace without claiming complete or perfectly accurate recall."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    """Represent a cue-driven memory reconstruction with separate vividness and accuracy."""

    event_id: str
    similarity: float
    vividness: float
    confidence: float
    recalled_feature_ids: tuple[str, ...]
    recalled_central_ids: tuple[str, ...]
    recalled_peripheral_ids: tuple[str, ...]
    accuracy: float | None = None

    def __post_init__(self) -> None:
        """Validate retrieval probabilities while allowing accuracy to remain unknown."""
        _text("event_id", self.event_id)
        for name in ("similarity", "vividness", "confidence"):
            _probability(name, getattr(self, name))
        if self.accuracy is not None:
            _probability("accuracy", self.accuracy)

    def to_dict(self) -> dict[str, Any]:
        """Return reconstructed content and keep unverified accuracy explicitly nullable."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class EmotionLearningConfig:
    """Configure appraisal, attention, arousal modulation, reward learning, and consolidation."""

    fast_arousal_retention: float = 0.35
    slow_stress_retention: float = 0.97
    ne_gain_strength: float = 0.65
    arousal_optimum: float = 0.48
    arousal_sigma: float = 0.38
    attention_capacity: float = 0.65
    threat_capture_gain: float = 0.25
    encoding_learning_rate: float = 0.35
    consolidation_rate: float = 0.25
    consolidation_time_constant_hours: float = 8.0
    reward_learning_rate: float = 0.20
    discount_factor: float = 0.90
    effort_cost_weight: float = 0.45
    fatigue_cost_weight: float = 0.35
    elaboration_threshold: float = 0.35
    retrieval_temperature: float = 0.35
    minimum_retrieval_similarity: float = 0.05
    max_memory_traces: int = 500

    def __post_init__(self) -> None:
        """Validate bounded controls and positive time/temperature parameters."""
        for name in (
            "fast_arousal_retention", "slow_stress_retention", "ne_gain_strength",
            "arousal_optimum", "attention_capacity", "threat_capture_gain",
            "encoding_learning_rate", "consolidation_rate", "reward_learning_rate",
            "discount_factor", "effort_cost_weight", "fatigue_cost_weight",
            "elaboration_threshold", "minimum_retrieval_similarity",
        ):
            _probability(name, getattr(self, name))
        if _finite("arousal_sigma", self.arousal_sigma) <= 0:
            raise ValueError("arousal_sigma must be positive")
        if _finite("consolidation_time_constant_hours",
                   self.consolidation_time_constant_hours) <= 0:
            raise ValueError("consolidation_time_constant_hours must be positive")
        if _finite("retrieval_temperature", self.retrieval_temperature) <= 0:
            raise ValueError("retrieval_temperature must be positive")
        if self.max_memory_traces < 1:
            raise ValueError("max_memory_traces must be at least 1")


@dataclass(frozen=True, slots=True)
class EmotionLearningResult:
    """Capture one completed 16-stage experience-processing cycle."""

    event_id: str
    appraisal: AffectiveAppraisal
    arousal_state: ArousalState
    attention: AttentionAllocation
    effort: EffortDecision
    memory_trace: MemoryTrace
    retrieved_memories: tuple[RetrievedMemory, ...]
    reward_prediction_error: float
    prediction_values: Mapping[str, float]
    tradeoff_metrics: Mapping[str, float]
    operation_trace: tuple[Mapping[str, Any], ...]
    biological: Mapping[str, Any]
    final_state: Mapping[str, Any]
    output: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate result identity, error scalar, and trace length."""
        _text("event_id", self.event_id)
        _finite("reward_prediction_error", self.reward_prediction_error)
        if len(self.operation_trace) != 16:
            raise ValueError("Point 13 results must contain exactly 16 operation trace entries")

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable record of the processing cycle."""
        return _plain(asdict(self))
