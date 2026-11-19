"""Validated data contracts for Point 02 prior-knowledge integration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _text(name: str, value: str) -> str:
    """Return non-empty normalized text or raise a contract error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _unit(name: str, value: float) -> float:
    """Validate a finite scalar in the closed unit interval."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


@dataclass(frozen=True, slots=True)
class RelationInput:
    """Represent an explicitly observed or reported directed relation in new material."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    evidential_status: str = "observed"

    def __post_init__(self) -> None:
        """Validate endpoints, relation label, confidence, and provenance class."""
        for name in ("subject", "predicate", "object"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        object.__setattr__(self, "confidence", _unit("confidence", self.confidence))
        if self.evidential_status not in {"observed", "reported", "inferred", "hypothesis"}:
            raise ValueError("unsupported evidential_status")


@dataclass(frozen=True, slots=True)
class LearningUnit:
    """Describe a new information unit with optional explicit entities, features, and relations."""

    unit_id: str
    text: str
    modality: str = "text"
    entities: tuple[str, ...] = ()
    features: Mapping[str, float] = field(default_factory=dict)
    relations: tuple[RelationInput, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    reliability: float = 1.0
    provenance: str = "user_supplied_learning_material"

    def __post_init__(self) -> None:
        """Normalize structured content and preserve supplied evidence status."""
        object.__setattr__(self, "unit_id", _text("unit_id", self.unit_id))
        object.__setattr__(self, "text", _text("text", self.text))
        if self.modality not in {"text", "speech_transcript", "image_caption", "diagram", "observation", "other"}:
            raise ValueError("unsupported modality")
        object.__setattr__(self, "entities", tuple(_text("entity", x) for x in self.entities))
        normalized_features = {_text("feature", k).casefold(): _unit("feature_weight", v)
                               for k, v in self.features.items()}
        object.__setattr__(self, "features", normalized_features)
        object.__setattr__(self, "relations", tuple(self.relations))
        if any(not isinstance(item, RelationInput) for item in self.relations):
            raise ValueError("relations must contain RelationInput values")
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "reliability", _unit("reliability", self.reliability))
        object.__setattr__(self, "provenance", _text("provenance", self.provenance))


@dataclass(frozen=True, slots=True)
class KnowledgeNode:
    """Represent one prior concept, fact, event, or memory with provenance and context."""

    node_id: str
    label: str
    kind: str = "concept"
    features: Mapping[str, float] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    reliability: float = 1.0
    memory_type: str = "semantic"
    provenance: str = "unspecified_prior_knowledge"

    def __post_init__(self) -> None:
        """Validate node identity, feature weights, memory class, and reliability."""
        object.__setattr__(self, "node_id", _text("node_id", self.node_id))
        object.__setattr__(self, "label", _text("label", self.label))
        if self.kind not in {"concept", "feature", "event", "episode", "schema", "context"}:
            raise ValueError("unsupported node kind")
        object.__setattr__(self, "features", {
            _text("feature", k).casefold(): _unit("feature_weight", v)
            for k, v in self.features.items()
        })
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "reliability", _unit("reliability", self.reliability))
        if self.memory_type not in {"semantic", "episodic", "schema", "procedural", "other"}:
            raise ValueError("unsupported memory_type")
        object.__setattr__(self, "provenance", _text("provenance", self.provenance))


@dataclass(frozen=True, slots=True)
class KnowledgeRelation:
    """Represent a weighted directed link while distinguishing evidence from inference."""

    source_id: str
    predicate: str
    target_id: str
    strength: float = 0.5
    confidence: float = 0.5
    evidential_status: str = "observed"
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate graph endpoints, relation, weights, and evidential status."""
        for name in ("source_id", "predicate", "target_id"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        object.__setattr__(self, "strength", _unit("strength", self.strength))
        object.__setattr__(self, "confidence", _unit("confidence", self.confidence))
        if self.evidential_status not in {"observed", "reported", "inferred", "hypothesis", "contradicted"}:
            raise ValueError("unsupported evidential_status")
        object.__setattr__(self, "context", dict(self.context))


@dataclass(frozen=True, slots=True)
class KnowledgeSchema:
    """Represent a flexible schema of roles, expected relations, and qualified exceptions."""

    schema_id: str
    name: str
    roles: tuple[str, ...] = ()
    relation_patterns: tuple[tuple[str, str, str], ...] = ()
    expectations: Mapping[str, float] = field(default_factory=dict)
    exceptions: tuple[str, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    version: int = 1
    supporting_episode_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate schema structure while permitting incomplete and optional role slots."""
        object.__setattr__(self, "schema_id", _text("schema_id", self.schema_id))
        object.__setattr__(self, "name", _text("name", self.name))
        object.__setattr__(self, "roles", tuple(_text("role", x).casefold() for x in self.roles))
        normalized_patterns = tuple(tuple(_text("relation_pattern_item", x).casefold() for x in row)
                                    for row in self.relation_patterns)
        if any(len(row) != 3 for row in normalized_patterns):
            raise ValueError("each relation_pattern must contain subject, predicate, object")
        object.__setattr__(self, "relation_patterns", normalized_patterns)
        object.__setattr__(self, "expectations", {
            _text("expectation", k).casefold(): _unit("expectation_weight", v)
            for k, v in self.expectations.items()
        })
        object.__setattr__(self, "exceptions", tuple(str(x) for x in self.exceptions))
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "confidence", _unit("confidence", self.confidence))
        if self.version < 1:
            raise ValueError("schema version must be at least 1")
        object.__setattr__(self, "supporting_episode_ids", tuple(self.supporting_episode_ids))


@dataclass(frozen=True, slots=True)
class PriorKnowledgeConfig:
    """Configure retrieval, comparison, fit, and cautious learning thresholds."""

    dimensions: int = 64
    retrieval_limit: int = 12
    retrieval_temperature: float = 0.75
    integration_threshold: float = 0.62
    conflict_threshold: float = 0.45
    novelty_threshold: float = 0.72
    schema_update_threshold: float = 0.78
    learning_rate: float = 0.08
    minimum_reliability: float = 0.55
    max_spread_hops: int = 2

    def __post_init__(self) -> None:
        """Reject invalid dimensions, thresholds, and learning parameters."""
        if not 8 <= self.dimensions <= 4096:
            raise ValueError("dimensions must be between 8 and 4096")
        if not 1 <= self.retrieval_limit <= 100:
            raise ValueError("retrieval_limit must be between 1 and 100")
        if not math.isfinite(self.retrieval_temperature) or self.retrieval_temperature <= 0:
            raise ValueError("retrieval_temperature must be finite and positive")
        for name in ("integration_threshold", "conflict_threshold", "novelty_threshold",
                     "schema_update_threshold", "learning_rate", "minimum_reliability"):
            _unit(name, getattr(self, name))
        if not 1 <= self.max_spread_hops <= 4:
            raise ValueError("max_spread_hops must be between 1 and 4")


@dataclass
class PriorKnowledgeState:
    """Maintain a learner's graph, schemas, associative weights, episodes, and retrieval history."""

    learner_id: str
    knowledge_nodes: dict[str, KnowledgeNode] = field(default_factory=dict)
    knowledge_relations: list[KnowledgeRelation] = field(default_factory=list)
    schemas: dict[str, KnowledgeSchema] = field(default_factory=dict)
    associative_weights: dict[str, float] = field(default_factory=dict)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    retrieval_history: list[dict[str, Any]] = field(default_factory=list)
    consolidation_count: int = 0
    process_count: int = 0

    def __post_init__(self) -> None:
        """Validate learner identity and retain only finite bounded associative weights."""
        self.learner_id = _text("learner_id", self.learner_id)
        self.associative_weights = {str(k): _unit("associative_weight", v)
                                    for k, v in self.associative_weights.items()}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly copy of the complete learner knowledge state."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PriorKnowledgeRequest:
    """Specify one prior-knowledge integration episode and optional seed knowledge graph."""

    learner_id: str
    new_input: str | LearningUnit
    current_context: Mapping[str, Any] = field(default_factory=dict)
    knowledge_nodes: tuple[KnowledgeNode, ...] = ()
    knowledge_relations: tuple[KnowledgeRelation, ...] = ()
    schemas: tuple[KnowledgeSchema, ...] = ()
    learner_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize convenience string input and verify graph referential integrity."""
        object.__setattr__(self, "learner_id", _text("learner_id", self.learner_id))
        if isinstance(self.new_input, str):
            object.__setattr__(self, "new_input", LearningUnit("new-unit-001", self.new_input))
        if not isinstance(self.new_input, LearningUnit):
            raise ValueError("new_input must be text or a LearningUnit")
        object.__setattr__(self, "current_context", dict(self.current_context))
        learner_state = dict(self.learner_state)
        for key in ("attention", "working_memory"):
            if key in learner_state:
                learner_state[key] = _unit(key, learner_state[key])
        object.__setattr__(self, "learner_state", learner_state)
        object.__setattr__(self, "knowledge_nodes", tuple(self.knowledge_nodes))
        object.__setattr__(self, "knowledge_relations", tuple(self.knowledge_relations))
        object.__setattr__(self, "schemas", tuple(self.schemas))
        ids = {node.node_id for node in self.knowledge_nodes}
        if len(ids) != len(self.knowledge_nodes):
            raise ValueError("knowledge node IDs must be unique")
        if any(edge.source_id not in ids or edge.target_id not in ids
               for edge in self.knowledge_relations):
            raise ValueError("every knowledge relation endpoint must exist in knowledge_nodes")
        if len({schema.schema_id for schema in self.schemas}) != len(self.schemas):
            raise ValueError("schema IDs must be unique")


@dataclass(frozen=True, slots=True)
class PriorKnowledgeResult:
    """Capture the connected representation, schema decisions, diagnostics, and 14-step trace."""

    learner_id: str
    operation_trace: tuple[Mapping[str, Any], ...]
    integrated_representation: Mapping[str, Any]
    new_episode: Mapping[str, Any] | None
    schema_updates: tuple[Mapping[str, Any], ...]
    analogy_maps: tuple[Mapping[str, Any], ...]
    retrieval_routes: tuple[Mapping[str, Any], ...]
    confidence_and_uncertainty: Mapping[str, Any]
    biological: Mapping[str, Any]
    final_state: Mapping[str, Any]
    output: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Require the exact ordered 14-step registry in every completed result."""
        expected = [f"P02-{index:02d}" for index in range(1, 15)]
        actual = [str(item.get("operation_id", "")) for item in self.operation_trace]
        if actual != expected:
            raise ValueError("Point 02 results must contain exactly 14 ordered operations")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete result as ordinary nested Python mappings."""
        return asdict(self)
