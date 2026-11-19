"""Validated contracts for Point 05 varied-context and interleaved-learning processing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Mapping


def _text(name: str, value: str) -> str:
    """Normalize a required text field and reject empty values."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _unit(name: str, value: float) -> float:
    """Validate a finite normalized score in [0, 1]."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _feature_map(name: str, values: Mapping[str, float]) -> dict[str, float]:
    """Validate a finite-valued feature dictionary."""
    result: dict[str, float] = {}
    for key, value in values.items():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{key}] must be finite")
        result[_text(f"{name} key", str(key))] = number
    return result


def normalize_utc(value: str | datetime) -> str:
    """Normalize a timezone-aware timestamp to ISO-8601 UTC."""
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be valid ISO-8601") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ValueError("timestamp must be a datetime or ISO-8601 string")
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class ConceptDefinition:
    """Describe a generalizable target, its diagnostic features, prerequisites, and valid conditions."""

    concept_id: str
    domain: str
    defining_features: Mapping[str, float]
    discriminative_features: Mapping[str, float] = field(default_factory=dict)
    prerequisite_concepts: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    boundary_conditions: Mapping[str, str] = field(default_factory=dict)
    learning_goal: str = "conceptual_transfer"
    importance: float = 1.0
    initial_familiarity: float = 0.0

    def __post_init__(self) -> None:
        """Validate concept identity, feature definitions, prerequisites, and boundary rules."""
        object.__setattr__(self, "concept_id", _text("concept_id", self.concept_id))
        object.__setattr__(self, "domain", _text("domain", self.domain))
        features = _feature_map("defining_features", self.defining_features)
        if not features:
            raise ValueError("defining_features must not be empty")
        object.__setattr__(self, "defining_features", features)
        object.__setattr__(self, "discriminative_features",
                           _feature_map("discriminative_features", self.discriminative_features))
        object.__setattr__(self, "prerequisite_concepts",
                           tuple(_text("prerequisite", x) for x in self.prerequisite_concepts))
        object.__setattr__(self, "relations",
                           tuple(_text("relation", x) for x in self.relations))
        object.__setattr__(self, "boundary_conditions",
                           {_text("condition key", k): _text("condition value", v)
                            for k, v in self.boundary_conditions.items()})
        object.__setattr__(self, "learning_goal", _text("learning_goal", self.learning_goal))
        object.__setattr__(self, "importance", _unit("importance", self.importance))
        object.__setattr__(self, "initial_familiarity",
                           _unit("initial_familiarity", self.initial_familiarity))


@dataclass(frozen=True, slots=True)
class LearningExample:
    """Represent a labeled episode with separate concept features, context cues, modality, and relations."""

    example_id: str
    concept_id: str
    concept_features: Mapping[str, float]
    context_features: Mapping[str, str]
    modality: str = "text"
    representation: str = "textual_example"
    relations: tuple[str, ...] = ()
    is_anchor: bool = False
    is_novel: bool = False
    transfer_level: str = "training"
    session_id: str = "session-1"
    elapsed_since_previous_hours: float | None = None
    verified: bool = True

    def __post_init__(self) -> None:
        """Validate episode identity, feature maps, transfer designation, and temporal metadata."""
        object.__setattr__(self, "example_id", _text("example_id", self.example_id))
        object.__setattr__(self, "concept_id", _text("concept_id", self.concept_id))
        features = _feature_map("concept_features", self.concept_features)
        if not features:
            raise ValueError("concept_features must not be empty")
        object.__setattr__(self, "concept_features", features)
        object.__setattr__(self, "context_features",
                           {_text("context key", k): _text("context value", v)
                            for k, v in self.context_features.items()})
        object.__setattr__(self, "modality", _text("modality", self.modality).casefold())
        object.__setattr__(self, "representation",
                           _text("representation", self.representation))
        object.__setattr__(self, "relations",
                           tuple(_text("relation", x) for x in self.relations))
        if self.transfer_level not in {"training", "near", "far", "delayed"}:
            raise ValueError("transfer_level must be training, near, far, or delayed")
        if self.transfer_level != "training" and not self.is_novel:
            object.__setattr__(self, "is_novel", True)
        object.__setattr__(self, "session_id", _text("session_id", self.session_id))
        if self.elapsed_since_previous_hours is not None:
            value = float(self.elapsed_since_previous_hours)
            if not math.isfinite(value) or value < 0:
                raise ValueError("elapsed_since_previous_hours must be finite and non-negative")
            object.__setattr__(self, "elapsed_since_previous_hours", value)


@dataclass(frozen=True, slots=True)
class TrialObservation:
    """Record a learner's concept selection, confidence, effort, context judgment, and relation evidence."""

    example_id: str
    selected_concept_id: str
    confidence: float
    effort: float = 0.5
    context_judgment: Mapping[str, str] = field(default_factory=dict)
    cue_type: str = "original"
    explanation: str = ""
    preserved_relations: tuple[str, ...] = ()
    incompatible_relations: tuple[str, ...] = ()
    response_time_ms: float = 0.0
    aid_level: int = 0
    feedback_verified: bool = False
    occurred_at_utc: str | datetime | None = None

    def __post_init__(self) -> None:
        """Validate the response observation and preserve relation evidence."""
        object.__setattr__(self, "example_id", _text("example_id", self.example_id))
        object.__setattr__(self, "selected_concept_id",
                           _text("selected_concept_id", self.selected_concept_id))
        object.__setattr__(self, "confidence", _unit("confidence", self.confidence))
        object.__setattr__(self, "effort", _unit("effort", self.effort))
        object.__setattr__(self, "context_judgment",
                           {_text("context key", k): _text("context value", v)
                            for k, v in self.context_judgment.items()})
        object.__setattr__(self, "cue_type", _text("cue_type", self.cue_type))
        object.__setattr__(self, "explanation", str(self.explanation))
        object.__setattr__(self, "preserved_relations",
                           tuple(_text("preserved_relation", x) for x in self.preserved_relations))
        object.__setattr__(self, "incompatible_relations",
                           tuple(_text("incompatible_relation", x)
                                 for x in self.incompatible_relations))
        response_time = float(self.response_time_ms)
        if not math.isfinite(response_time) or response_time < 0:
            raise ValueError("response_time_ms must be finite and non-negative")
        object.__setattr__(self, "response_time_ms", response_time)
        if self.aid_level < 0:
            raise ValueError("aid_level must be non-negative")
        if self.occurred_at_utc is not None:
            object.__setattr__(self, "occurred_at_utc", normalize_utc(self.occurred_at_utc))


@dataclass(frozen=True, slots=True)
class VariedContextsConfig:
    """Configure context variation, interleaving, transfer, complexity, and plasticity policies."""

    feature_learning_rate: float = 0.12
    context_smoothing: float = 0.5
    prototype_distance_scale: float = 1.0
    context_uncertainty_threshold: float = 0.55
    interleaving_min_relatedness: float = 0.25
    interleaving_min_confusion: float = 0.10
    target_accuracy: float = 0.80
    maximum_switch_cost: float = 0.55
    max_context_changes_per_trial: int = 2
    max_complexity: float = 1.5
    minimum_variation: float = 0.05
    maximum_variation: float = 1.0
    plasticity_learning_rate: float = 0.04
    prototype_decay: float = 0.01
    probability_temperature: float = 1.0
    minimum_relation_coverage: float = 0.5
    random_seed: int = 505

    def __post_init__(self) -> None:
        """Reject invalid probabilities, learning rates, and schedule constraints."""
        for name in (
            "feature_learning_rate", "context_smoothing", "context_uncertainty_threshold",
            "interleaving_min_relatedness", "interleaving_min_confusion", "target_accuracy",
            "maximum_switch_cost", "minimum_variation", "maximum_variation",
            "plasticity_learning_rate", "prototype_decay", "minimum_relation_coverage",
        ):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        for name in ("prototype_distance_scale", "probability_temperature", "max_complexity"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_variation > self.maximum_variation:
            raise ValueError("minimum_variation must not exceed maximum_variation")
        if self.max_context_changes_per_trial < 0:
            raise ValueError("max_context_changes_per_trial must be non-negative")


@dataclass
class ConceptMemoryState:
    """Maintain concept prototypes, context distributions, episodes, and verified relations."""

    concept_id: str
    prototype: dict[str, float] = field(default_factory=dict)
    feature_variance: dict[str, float] = field(default_factory=dict)
    example_count: int = 0
    context_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    modality_counts: dict[str, int] = field(default_factory=dict)
    cue_counts: dict[str, int] = field(default_factory=dict)
    verified_relations: dict[str, float] = field(default_factory=dict)
    error_counts: dict[str, int] = field(default_factory=dict)
    observation_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly copy of one concept's accumulated representation."""
        return asdict(self)


@dataclass
class VariedContextsState:
    """Persist concept memories and schedules across varied-context learning sessions."""

    concepts: dict[str, ConceptMemoryState] = field(default_factory=dict)
    episode_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    interleaving_history: list[str] = field(default_factory=list)
    context_transition_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    variation_level: float = 0.35
    process_count: int = 0

    def __post_init__(self) -> None:
        """Initialize an explicit bounded variation level."""
        self.variation_level = _unit("variation_level", self.variation_level)

    def to_dict(self) -> dict[str, Any]:
        """Serialize concept, episode, and schedule memory without exposing mutable objects."""
        return {
            "concepts": {key: value.to_dict() for key, value in self.concepts.items()},
            "episode_records": dict(self.episode_records),
            "interleaving_history": list(self.interleaving_history),
            "context_transition_counts": dict(self.context_transition_counts),
            "variation_level": self.variation_level,
            "process_count": self.process_count,
        }


@dataclass(frozen=True, slots=True)
class VariedContextsRequest:
    """Supply concepts, varied examples, observations, readiness, and learning constraints."""

    concepts: tuple[ConceptDefinition, ...]
    examples: tuple[LearningExample, ...]
    observations: tuple[TrialObservation, ...] = ()
    prerequisite_readiness: Mapping[str, float] = field(default_factory=dict)
    current_context: Mapping[str, str] = field(default_factory=dict)
    fatigue: float = 0.0
    available_time_minutes: float = 30.0
    original_context_id: str | None = None
    current_time_utc: str | datetime = "2026-01-01T00:00:00Z"

    def __post_init__(self) -> None:
        """Validate target/example references, observations, and bounded learner constraints."""
        concepts = tuple(self.concepts)
        examples = tuple(self.examples)
        observations = tuple(self.observations)
        if not concepts or not all(isinstance(x, ConceptDefinition) for x in concepts):
            raise ValueError("concepts must contain ConceptDefinition values")
        if not examples or not all(isinstance(x, LearningExample) for x in examples):
            raise ValueError("examples must contain LearningExample values")
        concept_ids = [x.concept_id for x in concepts]
        if len(concept_ids) != len(set(concept_ids)):
            raise ValueError("concept_id values must be unique")
        known = set(concept_ids)
        unknown_examples = {x.concept_id for x in examples} - known
        if unknown_examples:
            raise ValueError(f"examples reference unknown concepts: {sorted(unknown_examples)}")
        example_ids = [x.example_id for x in examples]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("example_id values must be unique")
        if any(not isinstance(x, TrialObservation) for x in observations):
            raise ValueError("observations must contain TrialObservation values")
        unknown_observations = {x.example_id for x in observations} - set(example_ids)
        if unknown_observations:
            raise ValueError(f"observations reference unknown examples: {sorted(unknown_observations)}")
        if any(x.selected_concept_id not in known for x in observations):
            raise ValueError("observations select a concept not present in concepts")
        object.__setattr__(self, "concepts", concepts)
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "prerequisite_readiness",
                           {str(k): _unit(f"prerequisite_readiness[{k}]", v)
                            for k, v in self.prerequisite_readiness.items()})
        object.__setattr__(self, "current_context",
                           {_text("context key", k): _text("context value", v)
                            for k, v in self.current_context.items()})
        object.__setattr__(self, "fatigue", _unit("fatigue", self.fatigue))
        time = float(self.available_time_minutes)
        if not math.isfinite(time) or time <= 0:
            raise ValueError("available_time_minutes must be finite and positive")
        object.__setattr__(self, "available_time_minutes", time)
        if self.original_context_id is not None:
            object.__setattr__(self, "original_context_id",
                               _text("original_context_id", self.original_context_id))
        object.__setattr__(self, "current_time_utc", normalize_utc(self.current_time_utc))


@dataclass(frozen=True, slots=True)
class VariedContextsResult:
    """Capture all 23 source-ordered operations and the updated flexible-concept model."""

    operation_trace: tuple[Mapping[str, Any], ...]
    concept_profiles: Mapping[str, Any]
    context_inference: Mapping[str, Any]
    interleaving_decision: Mapping[str, Any]
    schedule: Mapping[str, Any]
    trial_results: tuple[Mapping[str, Any], ...]
    transfer_report: Mapping[str, Any]
    flexibility_report: Mapping[str, Any]
    biological: Mapping[str, Any]
    final_state: Mapping[str, Any]
    output: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Require all 23 Point 05 operations in the documented sequence."""
        expected = [f"P05-{index:02d}" for index in range(1, 24)]
        actual = [str(row.get("operation_id", "")) for row in self.operation_trace]
        if actual != expected:
            raise ValueError("Point 05 results must contain exactly 23 ordered operations")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete varied-context result as nested Python dictionaries."""
        return asdict(self)
