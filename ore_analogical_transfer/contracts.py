"""Typed contracts for Point 11 cross-domain analogical transfer.

Situations are represented as relational graphs. Observations, source memories, mappings,
hypotheses, disanalogies, and validated outcomes remain explicitly distinguishable.
These are engineering abstractions, not literal neural data structures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _text(name: str, value: str) -> str:
    """Validate a required identifier or label and return its stripped form."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a non-empty string")
    return value.strip()


def _finite(name: str, value: float) -> float:
    """Convert a numeric value to a finite float."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _probability(name: str, value: float) -> float:
    """Validate and return a probability in [0, 1]."""
    result = _finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(name + " must be in [0, 1]")
    return result


def _jsonable(value: Any) -> Any:
    """Convert nested dataclasses, mappings, and scalar-like objects into JSON-safe values."""
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
class AnalogicalEntity:
    """One domain-specific object or event participating in a situation graph."""

    entity_id: str
    label: str
    kind: str
    attributes: Mapping[str, Any] = field(default_factory=dict)
    role_hints: tuple[str, ...] = ()
    importance: float = 1.0
    provenance: str = "source_observation"

    def __post_init__(self) -> None:
        """Normalize identity, role hints, importance, and provenance."""
        for name in ("entity_id", "label", "kind", "provenance"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        object.__setattr__(self, "kind", self.kind.casefold())
        object.__setattr__(self, "attributes", dict(self.attributes))
        object.__setattr__(self, "role_hints", tuple(
            _text("role hint", item).casefold() for item in self.role_hints
        ))
        object.__setattr__(self, "importance", _probability("importance", self.importance))

    def to_dict(self) -> dict[str, Any]:
        """Export entity content and origin without privileging surface resemblance."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class AnalogicalRelation:
    """A typed relation with argument roles, direction, conditions, and causal status."""

    relation_id: str
    name: str
    arguments: tuple[str, ...]
    direction: str = "directed"
    conditions: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    importance: float = 1.0
    causal_status: str = "unspecified"
    quantitative_form: str = ""
    evidential_status: str = "observed"
    provenance: str = "source_observation"

    def __post_init__(self) -> None:
        """Validate relation arity, direction, confidence, and epistemic status."""
        for name in ("relation_id", "name"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        arguments = tuple(_text("relation argument", item) for item in self.arguments)
        if len(arguments) < 2 or len(arguments) != len(set(arguments)):
            raise ValueError("relations need at least two distinct entity arguments")
        object.__setattr__(self, "arguments", arguments)
        direction = _text("direction", self.direction).casefold()
        if direction not in {"directed", "undirected"}:
            raise ValueError("direction must be directed or undirected")
        object.__setattr__(self, "direction", direction)
        causal = _text("causal_status", self.causal_status).casefold()
        allowed_causal = {"unspecified", "causal", "correlational", "temporal", "functional", "counterfactual"}
        if causal not in allowed_causal:
            raise ValueError("unsupported causal_status")
        object.__setattr__(self, "causal_status", causal)
        evidence = _text("evidential_status", self.evidential_status).casefold()
        if evidence not in {"observed", "inferred", "hypothesized", "contradicted"}:
            raise ValueError("invalid relation evidential_status")
        object.__setattr__(self, "evidential_status", evidence)
        object.__setattr__(self, "confidence", _probability("confidence", self.confidence))
        object.__setattr__(self, "importance", _probability("importance", self.importance))
        object.__setattr__(self, "name", self.name.casefold())
        object.__setattr__(self, "conditions", dict(self.conditions))
        object.__setattr__(self, "quantitative_form", self.quantitative_form.strip())
        object.__setattr__(self, "provenance", _text("provenance", self.provenance))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the relation while preserving direction and causal evidence."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class HigherOrderRelation:
    """A relation between two first-order relations, such as one constraining another."""

    link_id: str
    subject_relation_id: str
    predicate: str
    object_relation_id: str
    confidence: float = 1.0
    importance: float = 1.0
    evidential_status: str = "observed"

    def __post_init__(self) -> None:
        """Validate higher-order relation endpoints and evidential status."""
        for name in ("link_id", "subject_relation_id", "predicate", "object_relation_id"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        object.__setattr__(self, "predicate", self.predicate.casefold())
        if self.subject_relation_id == self.object_relation_id:
            raise ValueError("higher-order relation endpoints must differ")
        object.__setattr__(self, "confidence", _probability("confidence", self.confidence))
        object.__setattr__(self, "importance", _probability("importance", self.importance))
        status = _text("evidential_status", self.evidential_status).casefold()
        if status not in {"observed", "inferred", "hypothesized", "contradicted"}:
            raise ValueError("invalid higher-order evidential_status")
        object.__setattr__(self, "evidential_status", status)

    def to_dict(self) -> dict[str, Any]:
        """Return the higher-order link and its certainty."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class TransferConstraint:
    """An explicit target-domain boundary condition for a candidate transfer."""

    constraint_id: str
    kind: str
    subject: str = ""
    predicate: str = ""
    object: str = ""
    key: str = ""
    expected_value: Any = None
    explanation: str = ""
    severity: str = "hard"

    def __post_init__(self) -> None:
        """Validate supported relation/condition constraint forms and severity."""
        object.__setattr__(self, "constraint_id", _text("constraint_id", self.constraint_id))
        kind = _text("kind", self.kind).casefold()
        if kind not in {"forbid_relation", "require_relation", "condition_equals",
                        "condition_not_equals", "unsupported_relation"}:
            raise ValueError("unsupported target constraint kind")
        object.__setattr__(self, "kind", kind)
        for name in ("subject", "predicate", "object"):
            object.__setattr__(self, name, getattr(self, name).casefold().strip())
        object.__setattr__(self, "key", self.key.strip())
        severity = _text("severity", self.severity).casefold()
        if severity not in {"hard", "soft", "informational"}:
            raise ValueError("severity must be hard, soft, or informational")
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "explanation", self.explanation.strip())

    def to_dict(self) -> dict[str, Any]:
        """Serialize a target constraint without converting it into source evidence."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class SituationGraph:
    """Typed relational graph for a source or target situation.

    Relation arguments reference entity IDs. Higher-order links reference relation IDs.
    Constraints express target-domain boundaries rather than source observations.
    """

    case_id: str
    domain: str
    entities: Mapping[str, AnalogicalEntity]
    relations: Sequence[AnalogicalRelation]
    higher_order_relations: Sequence[HigherOrderRelation] = ()
    constraints: Sequence[TransferConstraint] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    goals: tuple[str, ...] = ()
    source_reference: str = "unspecified"

    def __post_init__(self) -> None:
        """Validate graph endpoints and preserve evidence status."""
        object.__setattr__(self, "case_id", _text("case_id", self.case_id))
        object.__setattr__(self, "domain", _text("domain", self.domain).casefold())
        entities = dict(self.entities)
        if not entities:
            raise ValueError("a situation graph requires at least one entity")
        for key, entity in entities.items():
            if not isinstance(entity, AnalogicalEntity) or key != entity.entity_id:
                raise ValueError("entity mappings must use AnalogicalEntity IDs as keys")
        relations = tuple(self.relations)
        if any(not isinstance(item, AnalogicalRelation) for item in relations):
            raise TypeError("relations must contain AnalogicalRelation")
        relation_ids = [item.relation_id for item in relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("relation IDs must be unique")
        for relation in relations:
            missing = set(relation.arguments) - set(entities)
            if missing:
                raise ValueError("relation references unknown entities: " + ", ".join(sorted(missing)))
        higher = tuple(self.higher_order_relations)
        relation_id_set = set(relation_ids)
        if any(not isinstance(item, HigherOrderRelation) for item in higher):
            raise TypeError("higher_order_relations must contain HigherOrderRelation")
        for item in higher:
            if {item.subject_relation_id, item.object_relation_id} - relation_id_set:
                raise ValueError("higher-order link references unknown relation IDs")
        constraints = tuple(self.constraints)
        if any(not isinstance(item, TransferConstraint) for item in constraints):
            raise TypeError("constraints must contain TransferConstraint")
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "higher_order_relations", higher)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "goals", tuple(_text("goal", item).casefold() for item in self.goals))
        object.__setattr__(self, "source_reference", str(self.source_reference or "unspecified"))

    def to_dict(self) -> dict[str, Any]:
        """Return a provenance-preserving graph snapshot."""
        return _jsonable({
            "case_id": self.case_id, "domain": self.domain,
            "entities": {key: value.to_dict() for key, value in self.entities.items()},
            "relations": [item.to_dict() for item in self.relations],
            "higher_order_relations": [item.to_dict() for item in self.higher_order_relations],
            "constraints": [item.to_dict() for item in self.constraints],
            "context": self.context, "goals": self.goals,
            "source_reference": self.source_reference,
        })


@dataclass(frozen=True, slots=True)
class ParameterBridge:
    """Explicit source-to-target parameter conversion with dimensional safeguards."""

    source_parameter: str
    target_parameter: str
    scale: float = 1.0
    offset: float = 0.0
    source_unit: str = ""
    target_unit: str = ""
    source_dimension: str = ""
    target_dimension: str = ""

    def __post_init__(self) -> None:
        """Reject invalid conversion coefficients and incompatible physical dimensions."""
        for name in ("source_parameter", "target_parameter"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        scale, offset = _finite("scale", self.scale), _finite("offset", self.offset)
        if scale == 0.0:
            raise ValueError("parameter bridge scale cannot be zero")
        if self.source_dimension and self.target_dimension and self.source_dimension != self.target_dimension:
            raise ValueError("cannot translate between different physical dimensions")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)

    def translate(self, value: float) -> float:
        """Convert one source numeric value using the declared target bridge."""
        return _finite("parameter value", value) * self.scale + self.offset

    def to_dict(self) -> dict[str, Any]:
        """Export conversion assumptions for review."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class EntityCorrespondence:
    """One source-to-target entity-role mapping with decomposed similarity scores."""

    source_entity_id: str
    target_entity_id: str
    score: float
    role_similarity: float
    relational_similarity: float
    surface_similarity: float
    goal_similarity: float
    confidence: float

    def __post_init__(self) -> None:
        """Validate mapping identities and all component scores."""
        object.__setattr__(self, "source_entity_id", _text("source_entity_id", self.source_entity_id))
        object.__setattr__(self, "target_entity_id", _text("target_entity_id", self.target_entity_id))
        for name in ("score", "role_similarity", "relational_similarity",
                     "surface_similarity", "goal_similarity", "confidence"):
            _finite(name, getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the feature-versus-role evidence breakdown."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class MappingCandidate:
    """A partial entity/relation map scored for structure, coherence, and contradiction."""

    mapping_id: str
    correspondences: tuple[EntityCorrespondence, ...]
    relation_pairs: tuple[tuple[str, str], ...]
    preserved_relation_ids: tuple[str, ...]
    missing_relation_ids: tuple[str, ...]
    contradicted_relation_ids: tuple[str, ...]
    node_score: float
    relation_score: float
    structural_consistency: float
    connectedness: float
    coherence_score: float
    total_score: float
    confidence: float

    @property
    def entity_map(self) -> dict[str, str]:
        """Return the partial assignment used to substitute source relation arguments."""
        return {item.source_entity_id: item.target_entity_id for item in self.correspondences}

    def to_dict(self) -> dict[str, Any]:
        """Return the mapping and its structural diagnostics."""
        return _jsonable({
            "mapping_id": self.mapping_id,
            "correspondences": [item.to_dict() for item in self.correspondences],
            "entity_map": self.entity_map, "relation_pairs": self.relation_pairs,
            "preserved_relation_ids": self.preserved_relation_ids,
            "missing_relation_ids": self.missing_relation_ids,
            "contradicted_relation_ids": self.contradicted_relation_ids,
            "node_score": self.node_score, "relation_score": self.relation_score,
            "structural_consistency": self.structural_consistency,
            "connectedness": self.connectedness, "coherence_score": self.coherence_score,
            "total_score": self.total_score, "confidence": self.confidence,
        })


@dataclass(frozen=True, slots=True)
class TransferHypothesis:
    """A source relation projected into a target with explicit support and scope caveats."""

    hypothesis_id: str
    source_relation_id: str
    relation: AnalogicalRelation
    confidence: float
    support_chain: tuple[str, ...]
    caveats: tuple[str, ...] = ()
    target_status: str = "candidate"
    rejected_reason: str = ""

    def __post_init__(self) -> None:
        """Keep each transferred relation hypothetical until target evidence validates it."""
        object.__setattr__(self, "hypothesis_id", _text("hypothesis_id", self.hypothesis_id))
        object.__setattr__(self, "source_relation_id", _text("source_relation_id", self.source_relation_id))
        if not isinstance(self.relation, AnalogicalRelation):
            raise TypeError("relation must be an AnalogicalRelation")
        object.__setattr__(self, "confidence", _probability("confidence", self.confidence))
        object.__setattr__(self, "support_chain", tuple(self.support_chain))
        object.__setattr__(self, "caveats", tuple(self.caveats))
        status = _text("target_status", self.target_status).casefold()
        if status not in {"candidate", "accepted", "qualified", "rejected", "validated", "unvalidated"}:
            raise ValueError("invalid target_status")
        object.__setattr__(self, "target_status", status)

    def to_dict(self) -> dict[str, Any]:
        """Return the target inference with its source relation and justification."""
        return _jsonable({
            "hypothesis_id": self.hypothesis_id, "source_relation_id": self.source_relation_id,
            "relation": self.relation.to_dict(), "confidence": self.confidence,
            "support_chain": self.support_chain, "caveats": self.caveats,
            "target_status": self.target_status, "rejected_reason": self.rejected_reason,
        })


@dataclass(frozen=True, slots=True)
class Disanalogy:
    """A source-target difference that limits or rejects an analogical transfer."""

    difference_id: str
    category: str
    description: str
    affected_relation_ids: tuple[str, ...]
    severity: str = "warning"
    evidence_source: str = "target_domain_constraint"

    def __post_init__(self) -> None:
        """Validate difference severity and retain its affected hypotheses."""
        for name in ("difference_id", "category", "description"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        severity = _text("severity", self.severity).casefold()
        if severity not in {"information", "warning", "hard_stop"}:
            raise ValueError("severity must be information, warning, or hard_stop")
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "affected_relation_ids", tuple(self.affected_relation_ids))

    def to_dict(self) -> dict[str, Any]:
        """Serialize an analogy boundary and its evidence source."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class AnalogySchema:
    """Domain-independent relation pattern with reliability and explicit exceptions."""

    schema_id: str
    name: str
    relation_templates: tuple[Mapping[str, Any], ...]
    conditions: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()
    source_domains: tuple[str, ...] = ()
    target_domains: tuple[str, ...] = ()
    successful_transfers: int = 0
    failed_transfers: int = 0
    alpha: float = 1.0
    beta: float = 1.0
    version: int = 1

    def __post_init__(self) -> None:
        """Validate schema identity, transfer counts, and positive Beta prior parameters."""
        object.__setattr__(self, "schema_id", _text("schema_id", self.schema_id))
        object.__setattr__(self, "name", _text("name", self.name))
        object.__setattr__(self, "relation_templates", tuple(dict(item) for item in self.relation_templates))
        for name in ("conditions", "exceptions", "source_domains", "target_domains"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.successful_transfers < 0 or self.failed_transfers < 0 or self.version < 1:
            raise ValueError("transfer counts cannot be negative and version must be positive")
        alpha, beta = _finite("alpha", self.alpha), _finite("beta", self.beta)
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("Beta prior parameters must be positive")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)

    @property
    def reliability(self) -> float:
        """Return the Beta-smoothed success probability for prior transfer outcomes."""
        return (self.alpha + self.successful_transfers) / (
            self.alpha + self.beta + self.successful_transfers + self.failed_transfers
        )

    def to_dict(self) -> dict[str, Any]:
        """Export reusable relation patterns and counterexample-sensitive reliability."""
        return _jsonable({**asdict(self), "reliability": self.reliability})


@dataclass(frozen=True, slots=True)
class AnalogicalConfig:
    """Calibratable controls for retrieval, graph matching, transfer, and schema consolidation."""

    relation_aliases: Mapping[str, str] = field(default_factory=dict)
    surface_weight: float = 0.15
    role_weight: float = 0.30
    relation_weight: float = 0.55
    mapping_threshold: float = 0.35
    minimum_structural_consistency: float = 0.55
    minimum_mapping_confidence: float = 0.45
    transfer_confidence_threshold: float = 0.35
    contradiction_penalty: float = 1.5
    missing_core_penalty: float = 0.35
    retrieval_limit: int = 5
    mapping_limit: int = 8
    max_source_entities: int = 12
    working_memory_budget: float = 8.0
    interference_penalty: float = 0.8
    surface_distraction_penalty: float = 0.65
    schema_update_minimum_successes: int = 3
    schema_update_learning_rate: float = 1.0

    def __post_init__(self) -> None:
        """Validate scoring scales, probability thresholds, and bounded search settings."""
        object.__setattr__(self, "relation_aliases", {
            _text("relation alias", str(key)).casefold(): _text("canonical relation", str(value)).casefold()
            for key, value in self.relation_aliases.items()
        })
        for name in ("surface_weight", "role_weight", "relation_weight", "mapping_threshold",
                     "minimum_structural_consistency", "minimum_mapping_confidence",
                     "transfer_confidence_threshold"):
            _probability(name, getattr(self, name))
        for name in ("contradiction_penalty", "missing_core_penalty", "interference_penalty",
                     "surface_distraction_penalty", "schema_update_learning_rate"):
            if _finite(name, getattr(self, name)) < 0.0:
                raise ValueError(name + " cannot be negative")
        if _finite("working_memory_budget", self.working_memory_budget) <= 0.0:
            raise ValueError("working_memory_budget must be positive")
        if min(self.retrieval_limit, self.mapping_limit, self.max_source_entities,
               self.schema_update_minimum_successes) < 1:
            raise ValueError("retrieval, mapping, entity, and consolidation limits must be positive")


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    """A remembered source situation and its cue, goal, and relational retrieval scores."""

    case_id: str
    domain: str
    retrieval_score: float
    cue_similarity: float
    goal_similarity: float
    relational_relevance: float
    source_reference: str

    def to_dict(self) -> dict[str, Any]:
        """Return source retrieval metadata without merging memory into target observations."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class TransferValidation:
    """Comparison of transferred hypotheses with later target-domain evidence."""

    target_case_id: str
    tested_hypotheses: tuple[Mapping[str, Any], ...]
    validated_count: int
    contradicted_count: int
    untested_count: int
    mean_brier_score: float | None
    mean_surprise: float | None
    supported: bool
    schema_update_authorized: bool
    updated_schema_id: str | None
    update_status: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize validation evidence and the explicit schema-update authorization."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class AnalogicalTransferResult:
    """Complete mapping, qualified hypotheses, target explanation, trace, and diagnostics."""

    target_case_id: str
    status: str
    source_candidates: tuple[SourceCandidate, ...]
    selected_source_case_id: str | None
    mappings: tuple[MappingCandidate, ...]
    selected_mapping: MappingCandidate | None
    transferred_hypotheses: tuple[TransferHypothesis, ...]
    rejected_hypotheses: tuple[TransferHypothesis, ...]
    disanalogies: tuple[Disanalogy, ...]
    target_explanation: tuple[Mapping[str, Any], ...]
    target_predictions: tuple[Mapping[str, Any], ...]
    parameter_translations: tuple[Mapping[str, Any], ...]
    evidence_ledger: tuple[Mapping[str, Any], ...]
    operation_trace: tuple[Mapping[str, Any], ...]
    biological: Mapping[str, Any]
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return every transfer decision with sources, limits, and unvalidated status."""
        return _jsonable({
            "target_case_id": self.target_case_id, "status": self.status,
            "source_candidates": [item.to_dict() for item in self.source_candidates],
            "selected_source_case_id": self.selected_source_case_id,
            "mappings": [item.to_dict() for item in self.mappings],
            "selected_mapping": self.selected_mapping.to_dict() if self.selected_mapping else None,
            "transferred_hypotheses": [item.to_dict() for item in self.transferred_hypotheses],
            "rejected_hypotheses": [item.to_dict() for item in self.rejected_hypotheses],
            "disanalogies": [item.to_dict() for item in self.disanalogies],
            "target_explanation": self.target_explanation,
            "target_predictions": self.target_predictions,
            "parameter_translations": self.parameter_translations,
            "evidence_ledger": self.evidence_ledger, "operation_trace": self.operation_trace,
            "biological": self.biological, "diagnostics": self.diagnostics,
        })
