"""Typed records for Point 09 schema learning and relational-frame inference.

These records make role, slot, relation, event, and provenance information explicit.
They are computational abstractions, not claims that neural tissue stores Python objects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _text(name: str, value: str) -> str:
    """Validate a required text identifier and return its stripped form."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a non-empty string")
    return value.strip()


def _probability(name: str, value: float) -> float:
    """Validate a probability or normalized evidence-strength value."""
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(name + " must be finite and in [0, 1]")
    return result


def _jsonable(value: Any) -> Any:
    """Convert nested records and containers into JSON-compatible values."""
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        values = [_jsonable(item) for item in value]
        return sorted(values, key=repr) if isinstance(value, set) else values
    if hasattr(value, "item") and callable(value.item):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(frozen=True, slots=True)
class EntityInput:
    """One episode-specific participant or object with a type and observed attributes."""

    entity_id: str
    label: str
    entity_type: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate entity identity and normalize its descriptive type."""
        object.__setattr__(self, "entity_id", _text("entity_id", self.entity_id))
        object.__setattr__(self, "label", _text("label", self.label))
        object.__setattr__(self, "entity_type", _text("entity_type", self.entity_type).casefold())
        object.__setattr__(self, "attributes", dict(self.attributes))

    def to_dict(self) -> dict[str, Any]:
        """Return the entity description without losing its episode-specific identity."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class RelationInput:
    """An observed or cautiously interpreted directed relation between two episode entities."""

    subject_id: str
    predicate: str
    object_id: str
    evidential_status: str = "observed"
    confidence: float = 1.0
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate relation endpoints, evidence label, and confidence."""
        for name in ("subject_id", "predicate", "object_id"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        allowed = {"observed", "functional", "causal_hypothesis"}
        status = _text("evidential_status", self.evidential_status).casefold()
        if status not in allowed:
            raise ValueError("evidential_status must be observed, functional, or causal_hypothesis")
        object.__setattr__(self, "evidential_status", status)
        object.__setattr__(self, "confidence", _probability("confidence", self.confidence))
        object.__setattr__(self, "context", dict(self.context))

    def to_dict(self) -> dict[str, Any]:
        """Return the typed relation and its epistemic status."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class EventInput:
    """One ordered event with participant-role bindings and optional time/context evidence."""

    event_type: str
    participants: Mapping[str, str]
    event_id: str = ""
    timestamp: float | None = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate event label, role-to-entity bindings, and optional timestamp."""
        object.__setattr__(self, "event_type", _text("event_type", self.event_type).casefold())
        object.__setattr__(self, "event_id", self.event_id.strip() if isinstance(self.event_id, str) else "")
        participants = {
            _text("participant role", str(role)).casefold():
            _text("participant entity_id", str(entity_id))
            for role, entity_id in self.participants.items()
        }
        if not participants:
            raise ValueError("each event must contain at least one participant role")
        object.__setattr__(self, "participants", participants)
        if self.timestamp is not None:
            timestamp = float(self.timestamp)
            if not math.isfinite(timestamp):
                raise ValueError("event timestamp must be finite when supplied")
            object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "context", dict(self.context))

    def to_dict(self) -> dict[str, Any]:
        """Return the ordered event and its explicit participant-role bindings."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class EpisodeInput:
    """A distinct experience retaining entities, ordered events, typed relations, and context.

    The order of events is the supplied event order. A missing relation is not treated as
    evidence that the relation was absent unless the caller explicitly models that
    negative observation elsewhere.
    """

    episode_id: str
    entities: Mapping[str, EntityInput]
    events: Sequence[EventInput]
    relations: Sequence[RelationInput] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    goals: Sequence[str] = ()
    outcomes: Sequence[str] = ()
    reliability: float = 1.0
    source_reference: str = "unspecified"

    def __post_init__(self) -> None:
        """Validate referential integrity while preserving this episode as a unique memory."""
        object.__setattr__(self, "episode_id", _text("episode_id", self.episode_id))
        entities = dict(self.entities)
        if not entities:
            raise ValueError("an episode must contain at least one entity")
        for key, entity in entities.items():
            if not isinstance(entity, EntityInput):
                raise TypeError("entities must map IDs to EntityInput records")
            if str(key) != entity.entity_id:
                raise ValueError("entity mapping keys must equal EntityInput.entity_id")
        events = tuple(self.events)
        if not events or any(not isinstance(item, EventInput) for item in events):
            raise ValueError("events must contain at least one EventInput")
        event_ids = [item.event_id for item in events if item.event_id]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_id values must be unique within an episode")
        for event in events:
            missing = set(event.participants.values()) - set(entities)
            if missing:
                raise ValueError("event participants reference unknown entities: " + ", ".join(sorted(missing)))
        relations = tuple(self.relations)
        if any(not isinstance(item, RelationInput) for item in relations):
            raise TypeError("relations must contain RelationInput records")
        for relation in relations:
            missing = {relation.subject_id, relation.object_id} - set(entities)
            if missing:
                raise ValueError("relation endpoints reference unknown entities: " + ", ".join(sorted(missing)))
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "goals", tuple(_text("goal", item).casefold() for item in self.goals))
        object.__setattr__(self, "outcomes", tuple(_text("outcome", item).casefold() for item in self.outcomes))
        object.__setattr__(self, "reliability", _probability("reliability", self.reliability))
        object.__setattr__(self, "source_reference", str(self.source_reference or "unspecified"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize an episode while retaining entity identity and event order."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class SchemaConfig:
    """Engineering controls for schema induction, uncertainty, hierarchy, and revision."""

    role_aliases: Mapping[str, str] = field(default_factory=dict)
    similarity_threshold: float = 0.42
    relation_alpha: float = 0.5
    relation_beta: float = 0.5
    transition_alpha: float = 0.5
    core_coverage_threshold: float = 0.80
    typical_coverage_threshold: float = 0.50
    prediction_confidence_threshold: float = 0.20
    hierarchy_similarity_threshold: float = 0.50
    update_mismatch_threshold: float = 0.38
    specialization_mismatch_threshold: float = 0.72
    learning_rate: float = 0.08
    max_predictions: int = 8

    def __post_init__(self) -> None:
        """Validate model controls and normalize role aliases."""
        aliases = {
            _text("role alias", str(key)).casefold():
            _text("canonical role", str(value)).casefold()
            for key, value in self.role_aliases.items()
        }
        object.__setattr__(self, "role_aliases", aliases)
        for name in (
            "similarity_threshold", "core_coverage_threshold", "typical_coverage_threshold",
            "prediction_confidence_threshold", "hierarchy_similarity_threshold",
            "update_mismatch_threshold", "specialization_mismatch_threshold", "learning_rate",
        ):
            _probability(name, getattr(self, name))
        if self.typical_coverage_threshold > self.core_coverage_threshold:
            raise ValueError("typical_coverage_threshold cannot exceed core_coverage_threshold")
        if self.update_mismatch_threshold > self.specialization_mismatch_threshold:
            raise ValueError("update threshold cannot exceed specialization threshold")
        for name in ("relation_alpha", "relation_beta", "transition_alpha"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.max_predictions < 1:
            raise ValueError("max_predictions must be positive")


@dataclass(frozen=True, slots=True)
class RoleAssignment:
    """Evidence-weighted mapping from one episode entity to a candidate functional role."""

    episode_id: str
    entity_id: str
    role_scores: Mapping[str, float]
    selected_role: str | None
    ambiguous: bool

    def to_dict(self) -> dict[str, Any]:
        """Return role alternatives and whether mapping was intentionally left ambiguous."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class SchemaSlot:
    """Variable role placeholder with observed fillers, type distribution, and support."""

    slot_id: str
    name: str
    allowed_type_probabilities: Mapping[str, float]
    observed_fillers: tuple[str, ...]
    support_episodes: int
    coverage: float
    status: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Return the slot without replacing original episode-specific filler identities."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class SchemaRelation:
    """Typed directed relation between schema slots with explicit evidence status."""

    subject_slot: str
    predicate: str
    object_slot: str
    evidential_status: str
    support_count: int
    eligible_count: int
    confidence: float
    status: str
    supporting_episode_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return relation support denominator, uncertainty, and epistemic label."""
        return _jsonable(asdict(self))


@dataclass(slots=True)
class SchemaFrame:
    """Integrated probabilistic frame containing slots, relations, transitions, goals, and lineage."""

    schema_id: str
    name: str
    slots: dict[str, SchemaSlot]
    relations: tuple[SchemaRelation, ...]
    transitions: tuple[Mapping[str, Any], ...]
    goals: Mapping[str, float]
    outcomes: Mapping[str, float]
    context_profile: Mapping[str, float]
    event_types: tuple[str, ...]
    supporting_episode_ids: tuple[str, ...]
    confidence: float
    status: str = "learned"
    parent_schema_ids: tuple[str, ...] = ()
    child_schema_ids: tuple[str, ...] = ()
    reusable_components: tuple[str, ...] = ()
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return the full frame as a serializable, provenance-preserving structure."""
        return _jsonable({
            "schema_id": self.schema_id, "name": self.name,
            "slots": {key: value.to_dict() for key, value in self.slots.items()},
            "relations": [item.to_dict() for item in self.relations],
            "transitions": list(self.transitions), "goals": self.goals,
            "outcomes": self.outcomes, "context_profile": self.context_profile,
            "event_types": self.event_types, "supporting_episode_ids": self.supporting_episode_ids,
            "confidence": self.confidence, "status": self.status,
            "parent_schema_ids": self.parent_schema_ids, "child_schema_ids": self.child_schema_ids,
            "reusable_components": self.reusable_components, "version": self.version,
        })


@dataclass(frozen=True, slots=True)
class SchemaLearningResult:
    """Result of inducing schema frames from a set of distinct experiences."""

    schemas: tuple[SchemaFrame, ...]
    role_assignments: tuple[RoleAssignment, ...]
    episode_groups: tuple[tuple[str, ...], ...]
    operation_trace: tuple[Mapping[str, Any], ...]
    biological: Mapping[str, Any]
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Export schemas, role decisions, operation evidence, and diagnostics."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class SchemaUpdateResult:
    """Outcome of comparing a new episode with a schema and revising without erasing evidence."""

    action: str
    schema_id: str
    mismatch: Mapping[str, Any]
    updated_schema: SchemaFrame
    operation_trace: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return selected update action and resulting schema lineage."""
        return _jsonable({
            "action": self.action, "schema_id": self.schema_id, "mismatch": self.mismatch,
            "updated_schema": self.updated_schema.to_dict(), "operation_trace": self.operation_trace,
        })
