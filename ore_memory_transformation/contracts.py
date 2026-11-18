"""Validated contracts for parallel episodic and semantic memory transformation.

Event details, semantic regularities, source provenance, confidence, and retrieval accessibility
are represented separately. Missing features mean unknown unless a caller explicitly supplies
a zero-valued feature observation. Numerical values are engineering abstractions, not measured
human physiological constants.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


FEATURE_GROUPS = ("content", "spatial", "temporal", "sensory", "affective", "causal")
SOURCE_TYPES = {"direct_experience", "testimony", "inference", "retrieved", "other"}
RETRIEVAL_TYPES = {"episodic", "semantic", "mixed"}


def _unit_value(name: str, value: float) -> float:
    """Validate and return one finite feature, confidence, or importance value in [0, 1]."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _string_value(name: str, value: str) -> str:
    """Require a non-empty string and return its whitespace-trimmed form."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _normalise_group_maps(
    name: str, value: Mapping[str, Mapping[str, float]], allowed_groups: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Validate nested feature maps while preserving explicit zero-valued observations."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping of feature groups to feature maps")
    result: dict[str, dict[str, float]] = {}
    for group, features in value.items():
        if group not in allowed_groups:
            raise ValueError(f"{name} contains unsupported feature group {group!r}")
        if not isinstance(features, Mapping):
            raise TypeError(f"{name}[{group!r}] must be a mapping")
        clean: dict[str, float] = {}
        for feature, raw in features.items():
            key = _string_value(f"{name} feature key", feature)
            clean[key] = _unit_value(f"{name}[{group}][{key}]", raw)
        result[group] = clean
    return result


def _jsonable(value: Any) -> Any:
    """Convert dataclasses, mappings, sequences, and array-like values into JSON-compatible data."""
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
class RelationEvidence:
    """One typed relation explicitly supported by an event or another information source.

    A relation is kept distinct from co-occurrence. For example, agent-of, causes, before, and
    located-at have different meanings. The source and confidence describe evidence provenance;
    they are not claims that a particular biological circuit stores symbolic triples.
    """

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source_type: str = "direct_experience"

    def __post_init__(self) -> None:
        """Validate relation endpoints, predicate, confidence, and evidence source."""
        for name in ("subject", "predicate", "object"):
            object.__setattr__(self, name, _string_value(name, getattr(self, name)))
        confidence = _unit_value("relation confidence", self.confidence)
        source = _string_value("source_type", self.source_type).lower()
        if source not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source_type", source)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the relation and provenance without changing its semantic type."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpisodeInput:
    """An encoded event with content, context, sensory, affective, relational, and source evidence.

    Feature maps use values in [0, 1]. A supplied zero is an observed absence or zero-strength
    value; an omitted key is unknown and must not be counted as negative evidence. Feature
    confidence and context importance are keyed by feature group and feature. The complete
    episode record remains available even when its retrieval accessibility later decreases.
    """

    episode_id: str
    content_features: Mapping[str, float] = field(default_factory=dict)
    spatial_features: Mapping[str, float] = field(default_factory=dict)
    temporal_features: Mapping[str, float] = field(default_factory=dict)
    sensory_features: Mapping[str, float] = field(default_factory=dict)
    affective_features: Mapping[str, float] = field(default_factory=dict)
    causal_features: Mapping[str, float] = field(default_factory=dict)
    relations: Sequence[RelationEvidence | Mapping[str, Any]] = ()
    feature_confidence: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    context_importance: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    concept_hint: str | None = None
    source_type: str = "direct_experience"
    source_id: str | None = None
    source_confidence: float = 1.0
    overall_confidence: float = 1.0
    timestamp_ms: float = 0.0
    repetition_count: int = 1
    spacing_ms: float = 0.0

    def __post_init__(self) -> None:
        """Normalize all episode fields and reject malformed or unsupported evidence."""
        object.__setattr__(self, "episode_id", _string_value("episode_id", self.episode_id))
        for name in FEATURE_GROUPS:
            field_name = f"{name}_features"
            values = getattr(self, field_name)
            if not isinstance(values, Mapping):
                raise TypeError(f"{field_name} must be a mapping")
            clean: dict[str, float] = {}
            for feature, raw in values.items():
                key = _string_value(f"{field_name} feature", feature)
                clean[key] = _unit_value(f"{field_name}[{key}]", raw)
            object.__setattr__(self, field_name, clean)
        confidence = _normalise_group_maps(
            "feature_confidence", self.feature_confidence, FEATURE_GROUPS
        )
        importance = _normalise_group_maps(
            "context_importance", self.context_importance, ("spatial", "temporal")
        )
        for group in importance:
            if not set(importance[group]).issubset(set(getattr(self, f"{group}_features"))):
                raise ValueError("context_importance may only refer to observed spatial/temporal features")
        object.__setattr__(self, "feature_confidence", confidence)
        object.__setattr__(self, "context_importance", importance)
        relations: list[RelationEvidence] = []
        for relation in self.relations:
            relations.append(relation if isinstance(relation, RelationEvidence)
                             else RelationEvidence(**dict(relation)))
        object.__setattr__(self, "relations", tuple(relations))
        hint = self.concept_hint
        if hint is not None:
            object.__setattr__(self, "concept_hint", _string_value("concept_hint", hint))
        source = _string_value("source_type", self.source_type).lower()
        if source not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        object.__setattr__(self, "source_type", source)
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _string_value("source_id", self.source_id))
        object.__setattr__(self, "source_confidence",
                           _unit_value("source_confidence", self.source_confidence))
        object.__setattr__(self, "overall_confidence",
                           _unit_value("overall_confidence", self.overall_confidence))
        timestamp = float(self.timestamp_ms)
        spacing = float(self.spacing_ms)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_ms must be finite and non-negative")
        if not math.isfinite(spacing) or spacing < 0.0:
            raise ValueError("spacing_ms must be finite and non-negative")
        if isinstance(self.repetition_count, bool) or self.repetition_count < 1:
            raise ValueError("repetition_count must be a positive integer")
        if not isinstance(self.repetition_count, int):
            raise TypeError("repetition_count must be an integer")
        object.__setattr__(self, "timestamp_ms", timestamp)
        object.__setattr__(self, "spacing_ms", spacing)

    def feature_groups(self) -> dict[str, dict[str, float]]:
        """Return all six feature groups, including empty groups, without dropping observed zeroes."""
        return {group: dict(getattr(self, f"{group}_features")) for group in FEATURE_GROUPS}

    def to_dict(self) -> dict[str, Any]:
        """Return the immutable event record and provenance as JSON-compatible values."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class MemoryTransformationRequest:
    """One reactivation, abstraction, or retrieval operation over persistent memory stores.

    New episodes are added without replacing an existing event with the same ID. Candidate IDs
    limit cue competition; target_episode_id requests one particular episode. If neither is
    supplied, the cue is compared against all registered events. now_ms is an optional model-time
    coordinate used for accessibility decay; omitting it does not advance memory age.
    """

    request_id: str
    episodes: Sequence[EpisodeInput] = ()
    cue_features: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    cue_relations: Sequence[RelationEvidence | Mapping[str, Any]] = ()
    candidate_episode_ids: Sequence[str] = ()
    target_episode_id: str | None = None
    target_concept_id: str | None = None
    retrieval_type: str = "mixed"
    now_ms: float | None = None
    reactivation_gain: float = 0.15
    rehearsal_episode_ids: Sequence[str] = ()
    task_goal: str = ""

    def __post_init__(self) -> None:
        """Validate request identity, cue evidence, retrieval goal, and optional model time."""
        object.__setattr__(self, "request_id", _string_value("request_id", self.request_id))
        episodes = tuple(self.episodes)
        if any(not isinstance(item, EpisodeInput) for item in episodes):
            raise TypeError("episodes must contain EpisodeInput objects")
        if len({item.episode_id for item in episodes}) != len(episodes):
            raise ValueError("episodes in one request must have unique episode_id values")
        object.__setattr__(self, "episodes", episodes)
        cue = _normalise_group_maps("cue_features", self.cue_features, FEATURE_GROUPS)
        object.__setattr__(self, "cue_features", cue)
        relations: list[RelationEvidence] = []
        for relation in self.cue_relations:
            relations.append(relation if isinstance(relation, RelationEvidence)
                             else RelationEvidence(**dict(relation)))
        object.__setattr__(self, "cue_relations", tuple(relations))
        for name in ("candidate_episode_ids", "rehearsal_episode_ids"):
            values = tuple(_string_value(name, value) for value in getattr(self, name))
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, values)
        for name in ("target_episode_id", "target_concept_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _string_value(name, value))
        retrieval = _string_value("retrieval_type", self.retrieval_type).lower()
        if retrieval not in RETRIEVAL_TYPES:
            raise ValueError(f"retrieval_type must be one of {sorted(RETRIEVAL_TYPES)}")
        object.__setattr__(self, "retrieval_type", retrieval)
        if self.now_ms is not None:
            now = float(self.now_ms)
            if not math.isfinite(now) or now < 0.0:
                raise ValueError("now_ms must be finite and non-negative")
            object.__setattr__(self, "now_ms", now)
        object.__setattr__(self, "reactivation_gain",
                           _unit_value("reactivation_gain", self.reactivation_gain))
        object.__setattr__(self, "task_goal", str(self.task_goal))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the request while retaining explicit zeroes and all source metadata."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class MemoryTransformationConfig:
    """Reproducible model settings for episodic-semantic transformation.

    Similarity cutoffs, population sizes, decay rates, and learning rates are engineering
    parameters. The dual-store architecture and region labels are explanatory abstractions;
    the numerical operations are not assertions about literal neuronal arithmetic.
    """

    hippocampal_units: int = 48
    semantic_units: int = 64
    related_similarity_threshold: float = 0.58
    concept_assignment_threshold: float = 0.62
    concept_ambiguity_margin: float = 0.04
    cue_selection_temperature: float = 0.55
    maximum_reactivated_episodes: int = 32
    invariant_prevalence_threshold: float = 0.70
    conflict_lower_threshold: float = 0.35
    conflict_upper_threshold: float = 0.65
    minimum_effective_support: float = 1.5
    semantic_learning_rate: float = 0.08
    context_default_importance: float = 0.50
    context_logit_bias: float = -0.35
    context_prevalence_gain: float = 1.0
    context_utility_gain: float = 1.2
    context_meaning_gain: float = 1.2
    interference_rate_per_hour: float = 0.0003
    accessibility_reactivation_boost: float = 0.12
    sparse_activity_fraction: float = 0.20
    random_seed: int = 41
    max_episodes: int = 10000

    def __post_init__(self) -> None:
        """Validate dimensions, normalized thresholds, and non-negative update controls."""
        if min(self.hippocampal_units, self.semantic_units,
               self.maximum_reactivated_episodes, self.max_episodes) < 2:
            raise ValueError("population sizes and episode limits must be >= 2")
        for name in ("related_similarity_threshold", "concept_assignment_threshold",
                     "concept_ambiguity_margin", "invariant_prevalence_threshold",
                     "conflict_lower_threshold", "conflict_upper_threshold",
                     "context_default_importance", "sparse_activity_fraction"):
            _unit_value(name, getattr(self, name))
        if self.conflict_lower_threshold >= self.conflict_upper_threshold:
            raise ValueError("conflict thresholds must be increasing")
        if self.cue_selection_temperature <= 0.0 or self.minimum_effective_support < 1.0:
            raise ValueError("selection temperature and effective support must be positive")
        for name in ("semantic_learning_rate", "interference_rate_per_hour",
                     "accessibility_reactivation_boost"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return model settings for experiment reproducibility and log metadata."""
        return asdict(self)


@dataclass(slots=True)
class MemoryTransformationResult:
    """Inspectable output containing event memory, semantic gist, retrieval, and 16 stage records."""

    request_id: str
    operation_trace: list[dict[str, Any]]
    selected_episode_ids: list[str]
    related_episode_ids: list[str]
    concept_ids: list[str]
    episodic_representation: dict[str, Any]
    semantic_representations: list[dict[str, Any]]
    schema_links: list[dict[str, Any]]
    retrieved_result: dict[str, Any]
    uncertainty: dict[str, Any]
    accessibility: dict[str, Any]
    quality: dict[str, Any]
    biological: dict[str, Any]
    logger: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize both memory tracks and the evidence needed to inspect their transformation."""
        return _jsonable({"schema": "ore.episodic_semantic_transformation.1.0", **asdict(self)})
