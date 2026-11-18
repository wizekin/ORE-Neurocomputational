"""Validated data contracts for prototype, exemplar, and hybrid category learning.

Feature values are software-level observations. They are not literal symbols or numerical
registers in the brain; unknown values remain distinct from observed zero or false values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence, TypeAlias

FeatureValue: TypeAlias = float | int | str | bool | None


def _unit(name: str, value: float) -> float:
    """Validate a finite value in the closed unit interval."""
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _jsonable(value: Any) -> Any:
    """Convert nested model values to JSON-compatible Python primitives."""
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(frozen=True, slots=True)
class EpisodeInput:
    """One distinct experience with features, context, sensory modalities, and relations.

    Numeric values may use any finite scale. Strings and booleans represent categorical features;
    None means unobserved. Reliability and per-feature confidence are normalized to [0, 1].
    """

    episode_id: str
    features: Mapping[str, FeatureValue]
    context: Mapping[str, Any] = field(default_factory=dict)
    relations: Sequence[tuple[str, str, str]] = ()
    feature_modalities: Mapping[str, str] = field(default_factory=dict)
    feature_reliability: Mapping[str, float] = field(default_factory=dict)
    feature_uncertainty: Mapping[str, float] = field(default_factory=dict)
    reliability: float = 1.0
    timestamp: float | None = None

    def __post_init__(self) -> None:
        """Normalize and validate episode identity, observed values, relations, and reliability."""
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        normalized: dict[str, FeatureValue] = {}
        for raw_name, value in self.features.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("feature names must be non-empty strings")
            if isinstance(value, bool) or value is None or isinstance(value, str):
                normalized[name] = value.strip() if isinstance(value, str) else value
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                normalized[name] = float(value)
            else:
                raise ValueError(f"features[{name}] must be a finite number, string, bool, or None")
        if not normalized:
            raise ValueError("an episode must contain at least one candidate feature")
        relations: list[tuple[str, str, str]] = []
        for relation in self.relations:
            if len(relation) != 3 or any(not str(part).strip() for part in relation):
                raise ValueError("relations must be non-empty (subject, predicate, object) triples")
            relations.append(tuple(str(part).strip() for part in relation))
        modalities = {str(k).strip(): str(v).strip() for k, v in self.feature_modalities.items()}
        if any(not k or not v for k, v in modalities.items()):
            raise ValueError("feature modality names and values must be non-empty")
        feature_reliability = {str(k).strip(): _unit(f"feature_reliability[{k}]", v)
                               for k, v in self.feature_reliability.items()}
        feature_uncertainty = {str(k).strip(): float(v)
                               for k, v in self.feature_uncertainty.items()}
        if any(not k or not math.isfinite(v) or v < 0.0
               for k, v in feature_uncertainty.items()):
            raise ValueError("feature uncertainty must be finite, non-negative, and named")
        timestamp = self.timestamp
        if timestamp is not None and (not math.isfinite(float(timestamp)) or timestamp < 0):
            raise ValueError("timestamp must be finite and non-negative")
        object.__setattr__(self, "episode_id", self.episode_id.strip())
        object.__setattr__(self, "features", normalized)
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "relations", tuple(relations))
        object.__setattr__(self, "feature_modalities", modalities)
        object.__setattr__(self, "feature_reliability", feature_reliability)
        object.__setattr__(self, "feature_uncertainty", feature_uncertainty)
        object.__setattr__(self, "reliability", _unit("reliability", self.reliability))
        object.__setattr__(self, "timestamp", None if timestamp is None else float(timestamp))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full episode without converting unknowns into observed absences."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class PrototypeExemplarConfig:
    """Numerical and memory controls for the hybrid model.

    Parameters are tunable engineering assumptions, not universal neural constants.
    """

    feature_aliases: Mapping[str, str] = field(default_factory=dict)
    learning_rate: float = 0.12
    attention_temperature: float = 1.0
    choice_temperature: float = 0.35
    similarity_sensitivity: float = 2.2
    distance_exponent: float = 2.0
    memory_decay_rate: float = 0.025
    rehearsal_gain: float = 0.12
    retrieval_limit: int = 64
    max_exemplars_per_category: int = 1000
    acceptance_threshold: float = 0.52
    outlier_threshold: float = 2.8
    exception_similarity_threshold: float = 0.48
    beta_prior: float = 1.0
    feedback_learning_rate: float = 0.08
    cortical_units: int = 48
    recurrent_iterations: int = 5
    random_seed: int = 707
    unknown_category: str = "__unknown__"

    def __post_init__(self) -> None:
        """Validate feature aliases, positive scales, bounded rates, and memory capacities."""
        aliases = {str(k).strip(): str(v).strip() for k, v in self.feature_aliases.items()}
        if any(not k or not v for k, v in aliases.items()):
            raise ValueError("feature aliases must map non-empty names to non-empty names")
        object.__setattr__(self, "feature_aliases", aliases)
        for name in ("learning_rate", "rehearsal_gain", "acceptance_threshold",
                     "exception_similarity_threshold", "feedback_learning_rate"):
            _unit(name, getattr(self, name))
        for name in ("attention_temperature", "choice_temperature", "similarity_sensitivity",
                     "memory_decay_rate", "beta_prior", "outlier_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.distance_exponent not in (1.0, 2.0):
            raise ValueError("distance_exponent must be 1.0 or 2.0")
        if self.retrieval_limit < 1 or self.max_exemplars_per_category < 2:
            raise ValueError("retrieval_limit must be positive and exemplar capacity at least two")
        if self.cortical_units < 4 or self.recurrent_iterations < 1:
            raise ValueError("cortical_units must be >=4 and recurrent_iterations >=1")
        if not self.unknown_category.strip():
            raise ValueError("unknown_category must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe copy of all computational parameters."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class LearningRequest:
    """A set of labelled training categories and held-out cases for classification.

    Feedback labels are incorporated only after the corresponding held-out prediction has been
    computed, preventing accidental label leakage into the classification step.
    """

    training_examples: Mapping[str, Sequence[EpisodeInput]]
    prediction_episodes: Sequence[EpisodeInput] = ()
    attention_weights: Mapping[str, float] = field(default_factory=dict)
    goal_weights: Mapping[str, float] = field(default_factory=dict)
    prior_weights: Mapping[str, float] = field(default_factory=dict)
    task_context: str = "general_category_learning"
    feedback_labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check category names, distinct episode IDs, control weights, and post-prediction feedback."""
        training = {str(k).strip(): tuple(v) for k, v in self.training_examples.items()}
        if not training or any(not k for k in training):
            raise ValueError("training_examples must contain at least one named category")
        if any(not rows for rows in training.values()):
            raise ValueError("every training category must contain at least one episode")
        predictions = tuple(self.prediction_episodes)
        all_ids = [episode.episode_id for rows in training.values() for episode in rows]
        all_ids.extend(episode.episode_id for episode in predictions)
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("episode IDs must be unique across training and prediction batches")
        weights: dict[str, dict[str, float]] = {}
        for name in ("attention_weights", "goal_weights", "prior_weights"):
            source = getattr(self, name)
            weights[name] = {str(k).strip(): _unit(f"{name}[{k}]", v) for k, v in source.items()}
            if any(not k for k in weights[name]):
                raise ValueError(f"{name} feature names must be non-empty")
        feedback = {str(k).strip(): str(v).strip() for k, v in self.feedback_labels.items()}
        prediction_ids = {episode.episode_id for episode in predictions}
        if set(feedback) - prediction_ids:
            raise ValueError("feedback labels may only refer to prediction episode IDs")
        if any(not category for category in feedback.values()):
            raise ValueError("feedback category labels must be non-empty")
        if not isinstance(self.task_context, str) or not self.task_context.strip():
            raise ValueError("task_context must be a non-empty string")
        object.__setattr__(self, "training_examples", training)
        object.__setattr__(self, "prediction_episodes", predictions)
        object.__setattr__(self, "attention_weights", weights["attention_weights"])
        object.__setattr__(self, "goal_weights", weights["goal_weights"])
        object.__setattr__(self, "prior_weights", weights["prior_weights"])
        object.__setattr__(self, "feedback_labels", feedback)


@dataclass(slots=True)
class PrototypeState:
    """Incremental summary statistics for a category's prototype representation."""

    numeric_weight: dict[str, float] = field(default_factory=dict)
    numeric_mean: dict[str, float] = field(default_factory=dict)
    numeric_m2: dict[str, float] = field(default_factory=dict)
    numeric_weight_sq: dict[str, float] = field(default_factory=dict)
    categorical_counts: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_observations: dict[str, int] = field(default_factory=dict)
    feature_weights: dict[str, float] = field(default_factory=dict)
    relation_counts: dict[str, float] = field(default_factory=dict)
    supporting_episode_ids: list[str] = field(default_factory=list)
    total_weight: float = 0.0
    update_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Export prototype means, variances, categorical distributions, and evidence counts."""
        variances = {
            name: self.numeric_m2.get(name, 0.0) / max(self.numeric_weight.get(name, 1.0), 1e-12)
            for name in self.numeric_mean
        }
        categorical = {
            feature: {value: weight / max(sum(counts.values()), 1e-12)
                      for value, weight in counts.items()}
            for feature, counts in self.categorical_counts.items()
        }
        return _jsonable({
            "numeric_mean": self.numeric_mean, "numeric_variance": variances,
            "categorical_distribution": categorical,
            "feature_observations": self.feature_observations,
            "feature_weights": self.feature_weights, "relation_counts": self.relation_counts,
            "supporting_episode_ids": self.supporting_episode_ids,
            "total_weight": self.total_weight, "update_count": self.update_count,
        })


@dataclass(slots=True)
class ExemplarTrace:
    """Episode-specific trace retaining a concrete feature/context/relation combination."""

    episode_id: str
    category_id: str
    features: dict[str, FeatureValue]
    context: dict[str, Any]
    relations: tuple[tuple[str, str, str], ...]
    reliability: float
    feature_reliability: dict[str, float]
    created_at: float
    base_strength: float = 1.0
    retrieval_count: int = 0
    rehearsal_events: list[tuple[float, float]] = field(default_factory=list)
    is_exception: bool = False
    diagnosticity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the exemplar without merging its identity into the category prototype."""
        return _jsonable(asdict(self))


@dataclass(slots=True)
class CategoryState:
    """Maintain one prototype and an independent set of retrievable exemplar traces."""

    category_id: str
    prototype: PrototypeState = field(default_factory=PrototypeState)
    exemplars: dict[str, ExemplarTrace] = field(default_factory=dict)
    prior_count: float = 1.0
    numeric_observations: list[dict[str, float]] = field(default_factory=list)
    update_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the category summary and selected exemplar metadata for inspection."""
        return _jsonable({
            "category_id": self.category_id, "prototype": self.prototype.to_dict(),
            "exemplar_count": len(self.exemplars),
            "exemplar_ids": sorted(self.exemplars), "prior_count": self.prior_count,
            "update_count": self.update_count,
        })


@dataclass(slots=True)
class PrototypeExemplarResult:
    """Inspectable outcome of the ordered 16-operation prototype/exemplar algorithm."""

    operation_trace: list[dict[str, Any]]
    categories: dict[str, dict[str, Any]]
    predictions: list[dict[str, Any]]
    validation: dict[str, Any]
    biological: dict[str, Any]
    logger: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize algorithm traces, learned category states, predictions, and validation."""
        return _jsonable(asdict(self))
