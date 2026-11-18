"""Typed contracts for Point 20 schema-driven inference and reconstructive memory."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence


SOURCE_TYPES = {
    "original_perception", "episodic_retrieval", "schema_inference",
    "imagination", "post_event_information", "later_testimony", "unknown",
}
EVIDENCE_STATUSES = {
    "direct_evidence", "episodically_retrieved", "weakly_retrieved",
    "not_retrieved", "contradicted", "not_queried", "suggested_after_event",
    "source_uncertain",
}


def _text(name: str, value: str) -> str:
    """Validate and normalize a required non-empty text field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite(name: str, value: float) -> float:
    """Convert one numeric model parameter to a finite float."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _unit(name: str, value: float) -> float:
    """Validate a finite model parameter within the closed unit interval."""
    number = _finite(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _jsonable(value: Any) -> Any:
    """Convert nested contracts and containers to JSON-compatible plain data."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=repr)
    if hasattr(value, "item") and callable(value.item):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class DetailStatus(str, Enum):
    """Represent a detail's evidential status without collapsing inference into recollection."""

    DIRECT_EVIDENCE = "direct_evidence"
    EPISODICALLY_RETRIEVED = "episodically_retrieved"
    INFERRED = "inferred"
    SUGGESTED_AFTER_EVENT = "suggested_after_event"
    SOURCE_UNCERTAIN = "source_uncertain"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EpisodicFragment:
    """One event-specific fragment with strength, source, episode, and retrieval-cue features."""

    detail_id: str
    feature_type: str
    content: Any
    status: str = "episodically_retrieved"
    retrieval_strength: float = 0.70
    episodic_context: float = 0.70
    attention_encoding: float = 0.70
    centrality: float = 0.50
    familiarity: float = 0.20
    recollection: float = 0.70
    source_type: str = "episodic_retrieval"
    source_id: str = ""
    episode_id: str = ""
    cue_features: Mapping[str, Any] = field(default_factory=dict)
    source_features: Mapping[str, float] = field(default_factory=dict)
    timestamp: float = 0.0
    provenance_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate feature status, bounded strengths, source labels, and provenance identifiers."""
        object.__setattr__(self, "detail_id", _text("detail_id", self.detail_id))
        object.__setattr__(self, "feature_type", _text("feature_type", self.feature_type).casefold())
        status = _text("status", self.status).casefold()
        if status not in EVIDENCE_STATUSES:
            raise ValueError(f"unsupported evidence status: {status}")
        object.__setattr__(self, "status", status)
        source = _text("source_type", self.source_type).casefold()
        if source not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        object.__setattr__(self, "source_type", source)
        for name in ("retrieval_strength", "episodic_context", "attention_encoding",
                     "centrality", "familiarity", "recollection"):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        object.__setattr__(self, "timestamp", _finite("timestamp", self.timestamp))
        object.__setattr__(self, "cue_features", dict(self.cue_features))
        normalized_sources = {str(k): _unit(f"source_features[{k}]", v)
                              for k, v in self.source_features.items()}
        object.__setattr__(self, "source_features", normalized_sources)
        object.__setattr__(self, "provenance_ids", tuple(str(x) for x in self.provenance_ids))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached fragment record with its evidence and source provenance."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class SchemaProfile:
    """Describe one reusable schema prior without treating its typical features as event facts."""

    schema_id: str
    name: str
    feature_probabilities: Mapping[str, Mapping[str, float]]
    prior_probability: float = 0.50
    context_profile: Mapping[str, float] = field(default_factory=dict)
    goal_relevance: float = 0.50
    cultural_scope: str = "general"
    description: str = ""

    def __post_init__(self) -> None:
        """Validate schema identity, prior, context, and categorical feature distributions."""
        object.__setattr__(self, "schema_id", _text("schema_id", self.schema_id))
        object.__setattr__(self, "name", _text("name", self.name))
        object.__setattr__(self, "prior_probability", _unit("prior_probability", self.prior_probability))
        object.__setattr__(self, "goal_relevance", _unit("goal_relevance", self.goal_relevance))
        object.__setattr__(self, "cultural_scope", _text("cultural_scope", self.cultural_scope))
        object.__setattr__(self, "context_profile", {
            str(key): _unit(f"context_profile[{key}]", value)
            for key, value in self.context_profile.items()
        })
        distributions: dict[str, dict[str, float]] = {}
        for feature, raw in self.feature_probabilities.items():
            values = {str(value): _unit(f"{feature}[{value}]", probability)
                      for value, probability in raw.items()}
            if not values or sum(values.values()) <= 0.0:
                raise ValueError(f"schema feature {feature!r} needs a positive distribution")
            total = sum(values.values())
            distributions[str(feature).casefold()] = {
                (str(value).strip().casefold()): probability / total
                for value, probability in values.items()
            }
        object.__setattr__(self, "feature_probabilities", distributions)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached schema description for logs and transparent inspection."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Record where a detail entered the information stream and how it relates to earlier sources."""

    source_id: str
    source_type: str
    detail_id: str = ""
    content: Any = None
    reliability: float = 0.70
    timestamp: float = 0.0
    independence_group: str = ""
    parent_source_ids: tuple[str, ...] = ()
    source_features: Mapping[str, float] = field(default_factory=dict)
    operation: str = "reported"
    verified: bool = False

    def __post_init__(self) -> None:
        """Validate source identity, type, reliability, timeline, and parent references."""
        object.__setattr__(self, "source_id", _text("source_id", self.source_id))
        source = _text("source_type", self.source_type).casefold()
        if source not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        object.__setattr__(self, "source_type", source)
        object.__setattr__(self, "detail_id", self.detail_id.strip())
        object.__setattr__(self, "reliability", _unit("reliability", self.reliability))
        object.__setattr__(self, "timestamp", _finite("timestamp", self.timestamp))
        object.__setattr__(self, "independence_group", self.independence_group.strip())
        object.__setattr__(self, "parent_source_ids", tuple(str(x) for x in self.parent_source_ids))
        object.__setattr__(self, "source_features", {
            str(key): _unit(f"source_features[{key}]", value)
            for key, value in self.source_features.items()
        })
        object.__setattr__(self, "operation", _text("operation", self.operation))

    def to_dict(self) -> dict[str, Any]:
        """Return the source record with its original timestamp and parent chain."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class FeedbackRecord:
    """One correction or corroboration with source reliability and independence information."""

    feedback_id: str
    feature_type: str
    value: Any
    supports: bool
    source_id: str
    source_type: str = "later_testimony"
    reliability: float = 0.80
    independence_group: str = ""
    timestamp: float = 0.0
    candidate_id: str = ""
    schema_id: str = ""
    verified: bool = False
    use_for_schema_update: bool = False

    def __post_init__(self) -> None:
        """Validate feedback provenance and prevent unverified corrections from becoming schema data."""
        for name in ("feedback_id", "feature_type", "source_id"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        object.__setattr__(self, "feature_type", self.feature_type.casefold())
        source = _text("source_type", self.source_type).casefold()
        if source not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")
        object.__setattr__(self, "source_type", source)
        object.__setattr__(self, "reliability", _unit("reliability", self.reliability))
        object.__setattr__(self, "timestamp", _finite("timestamp", self.timestamp))
        object.__setattr__(self, "independence_group", self.independence_group.strip())
        object.__setattr__(self, "candidate_id", self.candidate_id.strip())
        object.__setattr__(self, "schema_id", self.schema_id.strip())
        if self.use_for_schema_update and not self.verified:
            raise ValueError("schema updates require verified feedback")

    def to_dict(self) -> dict[str, Any]:
        """Return feedback as a detached record for Bayesian update history."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class Point20Config:
    """Calibratable reconstruction settings; values are not physiological constants."""

    retrieval_temperature: float = 0.45
    schema_temperature: float = 0.50
    posterior_temperature: float = 0.55
    source_temperature: float = 0.55
    population_decay: float = 0.20
    max_active_schemas: int = 3
    max_completions_per_feature: int = 3
    source_assignment_threshold: float = 0.62
    retrieval_promotion_threshold: float = 0.74
    report_confidence_threshold: float = 0.76
    clarify_confidence_threshold: float = 0.48
    min_feedback_reliability: float = 0.70
    schema_learning_rate: float = 0.08
    calibration_bias: float = -0.80
    evidence_weight: float = 2.0
    recollection_weight: float = 1.7
    inference_penalty: float = 1.25
    contradiction_penalty: float = 2.1
    source_uncertainty_penalty: float = 1.35
    familiarity_weight: float = 0.20
    recollection_decision_weight: float = 0.65
    gist_detail_prior_weight: float = 0.25
    epsilon: float = 1e-9
    simulate_biology: bool = True

    def __post_init__(self) -> None:
        """Reject invalid probabilities, counts, temperatures, and calibration parameters."""
        for name in (
            "population_decay", "source_assignment_threshold", "retrieval_promotion_threshold",
            "report_confidence_threshold", "clarify_confidence_threshold",
            "min_feedback_reliability", "schema_learning_rate",
            "familiarity_weight", "recollection_decision_weight",
            "gist_detail_prior_weight",
        ):
            _unit(name, getattr(self, name))
        for name in (
            "retrieval_temperature", "schema_temperature",
            "posterior_temperature", "source_temperature", "epsilon",
        ):
            value = _finite(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.max_active_schemas < 1 or self.max_completions_per_feature < 1:
            raise ValueError("schema and completion limits must be positive")
        for name in (
            "calibration_bias", "evidence_weight", "recollection_weight",
            "inference_penalty", "contradiction_penalty", "source_uncertainty_penalty",
        ):
            _finite(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class Point20Request:
    """One reconstruction demand with event fragments, schema candidates, source history, and feedback."""

    event_id: str
    cue: str | Mapping[str, Any]
    target_features: Sequence[str]
    episodic_fragments: Sequence[EpisodicFragment | Mapping[str, Any]]
    schemas: Sequence[SchemaProfile | Mapping[str, Any]]
    context: Mapping[str, Any] = field(default_factory=dict)
    source_history: Sequence[SourceRecord | Mapping[str, Any]] = ()
    competing_episodes: Sequence[Mapping[str, Any]] = ()
    feedback: Sequence[FeedbackRecord | Mapping[str, Any]] = ()
    task_type: str = "past_event_reconstruction"
    precision: str = "balanced"
    target_episode_id: str = ""
    source_reference: str = ""

    def __post_init__(self) -> None:
        """Normalize all nested records and preserve task precision and provenance distinctions."""
        object.__setattr__(self, "event_id", _text("event_id", self.event_id))
        if isinstance(self.cue, str):
            object.__setattr__(self, "cue", _text("cue", self.cue))
        elif isinstance(self.cue, Mapping):
            object.__setattr__(self, "cue", dict(self.cue))
        else:
            raise TypeError("cue must be non-empty text or a feature mapping")
        targets = tuple(_text("target feature", x).casefold() for x in self.target_features)
        if len(targets) != len(set(targets)):
            raise ValueError("target_features must be unique")
        object.__setattr__(self, "target_features", targets)
        fragments = tuple(
            item if isinstance(item, EpisodicFragment) else EpisodicFragment(**dict(item))
            for item in self.episodic_fragments
        )
        schemas = tuple(
            item if isinstance(item, SchemaProfile) else SchemaProfile(**dict(item))
            for item in self.schemas
        )
        sources = tuple(
            item if isinstance(item, SourceRecord) else SourceRecord(**dict(item))
            for item in self.source_history
        )
        feedback = tuple(
            item if isinstance(item, FeedbackRecord) else FeedbackRecord(**dict(item))
            for item in self.feedback
        )
        if len({item.detail_id for item in fragments}) != len(fragments):
            raise ValueError("episodic fragment detail_id values must be unique")
        if len({item.schema_id for item in schemas}) != len(schemas):
            raise ValueError("schema_id values must be unique")
        if len({item.source_id for item in sources}) != len(sources):
            raise ValueError("source_id values must be unique")
        if len({item.feedback_id for item in feedback}) != len(feedback):
            raise ValueError("feedback_id values must be unique")
        object.__setattr__(self, "episodic_fragments", fragments)
        object.__setattr__(self, "schemas", schemas)
        object.__setattr__(self, "source_history", sources)
        object.__setattr__(self, "feedback", feedback)
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "competing_episodes", tuple(dict(x) for x in self.competing_episodes))
        task = _text("task_type", self.task_type).casefold()
        if task not in {"past_event_reconstruction", "incomplete_scene", "story_retelling",
                        "relation_completion", "gist_summary"}:
            raise ValueError("unsupported task_type")
        precision = _text("precision", self.precision).casefold()
        if precision not in {"gist", "balanced", "exact"}:
            raise ValueError("precision must be gist, balanced, or exact")
        object.__setattr__(self, "task_type", task)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "target_episode_id", self.target_episode_id.strip())
        object.__setattr__(self, "source_reference", self.source_reference.strip())
        if not self.target_features and not fragments:
            raise ValueError("supply target_features or episodic fragments")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached request snapshot for provenance and logging."""
        return {
            "event_id": self.event_id,
            "cue": _jsonable(self.cue),
            "target_features": list(self.target_features),
            "episodic_fragments": [item.to_dict() for item in self.episodic_fragments],
            "schemas": [item.to_dict() for item in self.schemas],
            "context": _jsonable(self.context),
            "source_history": [item.to_dict() for item in self.source_history],
            "competing_episodes": _jsonable(self.competing_episodes),
            "feedback": [item.to_dict() for item in self.feedback],
            "task_type": self.task_type,
            "precision": self.precision,
            "target_episode_id": self.target_episode_id,
            "source_reference": self.source_reference,
        }


@dataclass
class DetailCandidate:
    """Represent one direct, retrieved, inferred, or uncertain detail with a full confidence record."""

    candidate_id: str
    feature_type: str
    content: Any
    status: str = "inferred"
    schema_id: str = ""
    schema_support: float = 0.0
    schema_prior: float = 0.0
    evidence_likelihood: float = 1.0
    posterior: float = 0.0
    evidence_strength: float = 0.0
    episodic_context: float = 0.0
    source_specificity: float = 0.0
    source_uncertainty: float = 1.0
    contradiction: float = 0.0
    contextual_fit: float = 0.5
    familiarity: float = 0.0
    recollection: float = 0.0
    inference_proportion: float = 1.0
    confidence: float = 0.0
    source_probabilities: dict[str, float] = field(default_factory=dict)
    assigned_source: str = "unknown"
    evidence_ids: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    constraint_results: list[dict[str, Any]] = field(default_factory=list)
    competing_episode_ids: list[str] = field(default_factory=list)
    feedback_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return every evidence, source, inference, and confidence dimension."""
        return _jsonable(asdict(self))


@dataclass
class Point20State:
    """Maintain schema learning counts and reconstruction versions separately from episode evidence."""

    schema_beta_counts: dict[str, dict[str, float]] = field(default_factory=dict)
    source_versions: list[dict[str, Any]] = field(default_factory=list)
    feedback_groups_seen: list[str] = field(default_factory=list)
    schema_feedback_groups_seen: list[str] = field(default_factory=list)
    reconstruction_versions: list[dict[str, Any]] = field(default_factory=list)
    episode_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    run_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a detached state snapshot without merging schema knowledge into event records."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class Point20Result:
    """Collect all 16 operations, candidate reports, provenance graph, biology, and final state."""

    event_id: str
    reconstruction: Mapping[str, Any]
    retrieval: Mapping[str, Any]
    feature_ledger: Mapping[str, Any]
    schema_activation: Mapping[str, Any]
    candidate_evaluation: Mapping[str, Any]
    source_monitoring: Mapping[str, Any]
    confidence: Mapping[str, Any]
    response_policy: Mapping[str, Any]
    retelling_monitor: Mapping[str, Any]
    feedback_update: Mapping[str, Any]
    schema_update: Mapping[str, Any]
    provenance_graph: Mapping[str, Any]
    biological: Mapping[str, Any]
    operation_trace: tuple[Mapping[str, Any], ...]
    final_state: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report, operation trace, biological reference, and state."""
        return _jsonable(asdict(self))
