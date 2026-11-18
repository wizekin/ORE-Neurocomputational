"""Validated contracts for commonality detection and incremental concept formation.

Feature values are graded observations in [0, 1]. Missing keys and explicit None values mean
"not observed"; a numeric zero means the feature was observed and absent. These are software
representations of evidence, not claims that the brain stores symbolic feature tables.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _unit(name: str, value: float) -> float:
    """Validate and return a finite normalized value in the closed unit interval."""
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _jsonable(value: Any) -> Any:
    """Convert dataclasses and nested containers into JSON-compatible primitives."""
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return _jsonable(value.item())
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
    """One distinct experience with observed features, context, modalities, and relations.

    A feature value of None or an omitted key is unknown, not absent. Numeric zero is an
    observed absence. Relations are typed triples such as ("wing", "part_of", "bird").
    Reliability describes confidence in the supplied episode evidence, not a biological
    measurement. category_label is used only for explicitly labelled prediction feedback.
    """

    episode_id: str
    features: Mapping[str, float | None]
    context: Mapping[str, Any] = field(default_factory=dict)
    relations: Sequence[tuple[str, str, str]] = ()
    feature_modalities: Mapping[str, str] = field(default_factory=dict)
    reliability: float = 1.0
    category_label: bool | None = None

    def __post_init__(self) -> None:
        """Validate episode identity, feature values, relation triples, and evidence reliability."""
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        normalized: dict[str, float | None] = {}
        for name, value in self.features.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("feature names must be non-empty strings")
            normalized[name.strip()] = None if value is None else _unit(
                f"features[{name}]", value
            )
        if not normalized:
            raise ValueError("an episode must contain at least one candidate feature")
        relations: list[tuple[str, str, str]] = []
        for relation in self.relations:
            if len(relation) != 3 or any(
                not isinstance(part, str) or not part.strip() for part in relation
            ):
                raise ValueError("relations must be non-empty (subject, predicate, object) triples")
            relations.append(tuple(part.strip() for part in relation))
        modalities = {str(key).strip(): str(value).strip()
                      for key, value in self.feature_modalities.items()}
        if any(not key or not value for key, value in modalities.items()):
            raise ValueError("feature modality names and values must be non-empty")
        if self.category_label is not None and not isinstance(self.category_label, bool):
            raise ValueError("category_label must be True, False, or None")
        object.__setattr__(self, "features", normalized)
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "relations", tuple(relations))
        object.__setattr__(self, "feature_modalities", modalities)
        object.__setattr__(self, "reliability", _unit("reliability", self.reliability))

    def to_dict(self) -> dict[str, Any]:
        """Return the episode and its observed evidence as a JSON-compatible mapping."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class ConceptLearningConfig:
    """Reproducible parameters for commonality, relational, cortical, and revision models.

    Numerical thresholds and population sizes are engineering parameters. They are not
    universal neural constants. Commonality detection is kept distinct from later prototype,
    exemplar, hub-and-spoke, schema, and analogical-transfer models.
    """

    feature_aliases: Mapping[str, str] = field(default_factory=dict)
    beta_prior: float = 1.0
    variance_floor: float = 0.025
    minimum_shared_weight: float = 0.08
    minimum_relation_support: float = 0.20
    max_related_episodes: int = 20
    cortical_units: int = 32
    recurrent_iterations: int = 6
    learning_rate: float = 0.035
    recurrent_gain: float = 0.12
    weight_decay: float = 0.008
    prediction_prior: float = 0.5
    prediction_threshold: float = 0.5
    mismatch_threshold: float = 0.50
    feedback_learning_rate: float = 0.12
    random_seed: int = 67
    max_episodes_per_concept: int = 5000

    def __post_init__(self) -> None:
        """Validate aliases, statistical priors, population sizes, and bounded learning controls."""
        aliases = {str(key).strip(): str(value).strip()
                   for key, value in self.feature_aliases.items()}
        if any(not key or not value for key, value in aliases.items()):
            raise ValueError("feature aliases must map non-empty names to non-empty names")
        object.__setattr__(self, "feature_aliases", aliases)
        if self.beta_prior <= 0.0 or self.variance_floor <= 0.0:
            raise ValueError("beta_prior and variance_floor must be positive")
        if self.max_related_episodes < 1 or self.max_episodes_per_concept < 2:
            raise ValueError("episode limits must be positive and concept capacity at least two")
        if self.cortical_units < 2 or self.recurrent_iterations < 1:
            raise ValueError("cortical_units must be >=2 and recurrent_iterations >=1")
        for name in (
            "minimum_shared_weight", "minimum_relation_support", "learning_rate",
            "recurrent_gain", "weight_decay", "prediction_prior",
            "prediction_threshold", "mismatch_threshold", "feedback_learning_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.prediction_prior in {0.0, 1.0}:
            raise ValueError("prediction_prior must be strictly between zero and one")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete set of numerical and feature-alignment controls."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class ConceptLearningRequest:
    """One update to a named provisional concept with target, comparison, and test episodes.

    Comparison episodes provide contrastive evidence. Prediction episodes remain held out until
    prediction; when a category_label is supplied, the result can use it as feedback only after
    the prediction-error stage. Attention, goal, and prior weights are optional feature-level
    controls in [0, 1].
    """

    concept_id: str
    target_episodes: Sequence[EpisodeInput] = ()
    comparison_episodes: Sequence[EpisodeInput] = ()
    prediction_episodes: Sequence[EpisodeInput] = ()
    attention_weights: Mapping[str, float] = field(default_factory=dict)
    goal_weights: Mapping[str, float] = field(default_factory=dict)
    prior_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate concept identity, sample roles, unique batch IDs, and optional feature weights."""
        if not isinstance(self.concept_id, str) or not self.concept_id.strip():
            raise ValueError("concept_id must be a non-empty string")
        target = tuple(self.target_episodes)
        comparison = tuple(self.comparison_episodes)
        prediction = tuple(self.prediction_episodes)
        all_batches = target + comparison + prediction
        ids = [episode.episode_id for episode in all_batches]
        if len(ids) != len(set(ids)):
            raise ValueError("episode IDs must be unique across request batches")
        if any(item.category_label is False for item in target):
            raise ValueError("target_episodes cannot contain category_label=False")
        if any(item.category_label is True for item in comparison):
            raise ValueError("comparison_episodes cannot contain category_label=True")
        weights: dict[str, dict[str, float]] = {}
        for field_name in ("attention_weights", "goal_weights", "prior_weights"):
            raw = getattr(self, field_name)
            normalized = {}
            for feature, value in raw.items():
                if not isinstance(feature, str) or not feature.strip():
                    raise ValueError(f"{field_name} keys must be non-empty feature names")
                normalized[feature.strip()] = _unit(f"{field_name}[{feature}]", value)
            weights[field_name] = normalized
        object.__setattr__(self, "concept_id", self.concept_id.strip())
        object.__setattr__(self, "target_episodes", target)
        object.__setattr__(self, "comparison_episodes", comparison)
        object.__setattr__(self, "prediction_episodes", prediction)
        object.__setattr__(self, "attention_weights", weights["attention_weights"])
        object.__setattr__(self, "goal_weights", weights["goal_weights"])
        object.__setattr__(self, "prior_weights", weights["prior_weights"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize the learning request while preserving unknown and explicitly absent features."""
        return _jsonable(asdict(self))


@dataclass(slots=True)
class ConceptCandidate:
    """Revisable shared-feature and relation representation distilled from multiple episodes.

    It intentionally is not a prototype/exemplar store or a schema/semantic-hub model. Detailed
    supporting episodes remain indexed, while feature and relation statistics summarize regularity,
    diagnosticity, variability, confidence, and exceptions.
    """

    concept_id: str
    feature_prevalence: dict[str, float]
    feature_variance: dict[str, float]
    feature_observation_count: dict[str, int]
    feature_effective_sample_size: dict[str, float]
    comparison_prevalence: dict[str, float | None]
    diagnosticity: dict[str, float | None]
    attention_weights: dict[str, float]
    feature_reliability: dict[str, float]
    feature_weights: dict[str, float]
    cooccurrence_pmi: dict[str, float]
    relation_strength: dict[str, float]
    shared_features: list[str]
    relation_edges: list[dict[str, Any]]
    uncertainty: dict[str, float]
    exceptions: list[str]
    supporting_episode_ids: list[str]
    comparison_episode_ids: list[str]
    cortical_pattern: list[float] = field(default_factory=list)
    update_count: int = 1
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete revisable concept, its evidence links, and cortical activity state."""
        return _jsonable(asdict(self))


@dataclass(slots=True)
class ConceptLearningResult:
    """Inspectable result for the 16 source-ordered commonality-detection operations."""

    concept_id: str
    operation_trace: list[dict[str, Any]]
    concept: dict[str, Any]
    related_episode_retrieval: dict[str, Any]
    predictions: list[dict[str, Any]]
    mismatch_and_revision: dict[str, Any]
    associative_index: dict[str, Any]
    biological: dict[str, Any]
    logger: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize all 16 operations, learned concept information, predictions, and diagnostics."""
        return _jsonable(asdict(self))
