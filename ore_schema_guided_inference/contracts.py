"""Typed Point 10 records for evidence-aware schema-guided inference."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

from ore_schema_formation.contracts import SchemaFrame


def _text(name: str, value: str) -> str:
    """Validate and strip a required identifier or label."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a non-empty string")
    return value.strip()


def _finite(name: str, value: float) -> float:
    """Convert a numeric input to a finite float."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _probability(name: str, value: float) -> float:
    """Validate and return a probability in the closed unit interval."""
    result = _finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(name + " must be in [0, 1]")
    return result


def value_key(value: Any) -> str:
    """Convert a categorical value to a stable key with distinct boolean values."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("feature values cannot contain NaN or infinity")
        return format(value, ".12g")
    return str(value).strip().casefold()


def _jsonable(value: Any) -> Any:
    """Convert nested dataclasses and containers into JSON-compatible values."""
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
class EvidenceItem:
    """One current-case feature with explicit value, reliability, epistemic status, and source."""

    feature: str
    value: Any
    status: str = "observed"
    reliability: float = 1.0
    source: str = "current_sensory_input"
    evidence_id: str = ""

    def __post_init__(self) -> None:
        """Validate input evidence without converting schema predictions into observations."""
        object.__setattr__(self, "feature", _text("feature", self.feature).casefold())
        status = _text("status", self.status).casefold()
        if status not in {"observed", "contradicted", "not_observed", "unknown"}:
            raise ValueError("input status must be observed, contradicted, not_observed, or unknown")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reliability", _probability("reliability", self.reliability))
        object.__setattr__(self, "source", _text("source", self.source))
        object.__setattr__(self, "evidence_id", self.evidence_id.strip())

    def to_dict(self) -> dict[str, Any]:
        """Return the evidence with its provenance and observation status."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class RelationEvidence:
    """An explicit directed relation observation between two case roles or entities."""

    subject: str
    predicate: str
    object: str
    status: str = "observed"
    reliability: float = 1.0
    source: str = "current_sensory_input"
    evidence_id: str = ""

    def __post_init__(self) -> None:
        """Validate relation direction, provenance, and observation status."""
        for name in ("subject", "predicate", "object"):
            object.__setattr__(self, name, _text(name, getattr(self, name)).casefold())
        status = _text("status", self.status).casefold()
        if status not in {"observed", "contradicted", "unknown", "not_observed"}:
            raise ValueError("invalid relation status")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reliability", _probability("reliability", self.reliability))
        object.__setattr__(self, "source", _text("source", self.source))
        object.__setattr__(self, "evidence_id", self.evidence_id.strip())

    def to_dict(self) -> dict[str, Any]:
        """Return the directed relation and its evidential source."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class QueryOption:
    """A possible information-gathering query with cost, availability, and risk."""

    feature: str
    cost: float = 0.0
    available: bool = True
    risk: float = 0.0
    action_label: str = ""

    def __post_init__(self) -> None:
        """Validate query cost and risk before expected-information-gain selection."""
        object.__setattr__(self, "feature", _text("feature", self.feature).casefold())
        cost, risk = _finite("cost", self.cost), _probability("risk", self.risk)
        if cost < 0.0:
            raise ValueError("query cost cannot be negative")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "risk", risk)
        object.__setattr__(self, "action_label", self.action_label.strip() or self.feature)


@dataclass(frozen=True, slots=True)
class ActionOption:
    """A candidate response with state-conditioned utility, cost, and explicit risk."""

    action_id: str
    utility_by_schema: Mapping[str, float]
    cost: float = 0.0
    risk: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        """Validate action utility and keep safety risk separate from category confidence."""
        object.__setattr__(self, "action_id", _text("action_id", self.action_id))
        utilities = {str(key): _finite("action utility", value)
                     for key, value in self.utility_by_schema.items()}
        if not utilities:
            raise ValueError("utility_by_schema cannot be empty")
        object.__setattr__(self, "utility_by_schema", utilities)
        cost, risk = _finite("cost", self.cost), _probability("risk", self.risk)
        if cost < 0.0:
            raise ValueError("action cost cannot be negative")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "risk", risk)
        object.__setattr__(self, "description", self.description.strip())

    def to_dict(self) -> dict[str, Any]:
        """Return an action without selecting it prematurely."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class CaseEvidence:
    """A current case kept separate from retrieved and schema-predicted information."""

    case_id: str
    features: Mapping[str, EvidenceItem]
    relations: Sequence[RelationEvidence] = ()
    event_history: Sequence[str] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    goals: Sequence[str] = ()
    query_options: Sequence[QueryOption] = ()
    action_options: Sequence[ActionOption] = ()
    source_reference: str = "unspecified"

    def __post_init__(self) -> None:
        """Validate evidence references and preserve unknowns instead of treating them as absent."""
        object.__setattr__(self, "case_id", _text("case_id", self.case_id))
        features: dict[str, EvidenceItem] = {}
        for key, item in self.features.items():
            if not isinstance(item, EvidenceItem):
                raise TypeError("features must map names to EvidenceItem records")
            feature_key = _text("feature key", str(key)).casefold()
            if feature_key != item.feature:
                raise ValueError("feature mapping keys must equal EvidenceItem.feature")
            features[feature_key] = item
        relations = tuple(self.relations)
        queries = tuple(self.query_options)
        actions = tuple(self.action_options)
        if any(not isinstance(item, RelationEvidence) for item in relations):
            raise TypeError("relations must contain RelationEvidence")
        if any(not isinstance(item, QueryOption) for item in queries):
            raise TypeError("query_options must contain QueryOption")
        if any(not isinstance(item, ActionOption) for item in actions):
            raise TypeError("action_options must contain ActionOption")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "event_history", tuple(_text("event", x).casefold() for x in self.event_history))
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "goals", tuple(_text("goal", x).casefold() for x in self.goals))
        object.__setattr__(self, "query_options", queries)
        object.__setattr__(self, "action_options", actions)
        object.__setattr__(self, "source_reference", str(self.source_reference or "unspecified"))

    @property
    def usable_evidence(self) -> tuple[EvidenceItem, ...]:
        """Return only present or explicitly contradicted feature observations."""
        return tuple(item for item in self.features.values()
                     if item.status in {"observed", "contradicted"})

    def to_dict(self) -> dict[str, Any]:
        """Serialize the current sensory case without adding schema-derived predictions."""
        return _jsonable({
            "case_id": self.case_id,
            "features": {key: item.to_dict() for key, item in self.features.items()},
            "relations": [item.to_dict() for item in self.relations],
            "event_history": self.event_history, "context": self.context, "goals": self.goals,
            "query_options": [asdict(item) for item in self.query_options],
            "action_options": [asdict(item) for item in self.action_options],
            "source_reference": self.source_reference,
        })


@dataclass(slots=True)
class SchemaProfile:
    """A Point 09 frame plus optional categorical feature likelihoods and a category prior."""

    frame: SchemaFrame
    feature_probabilities: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    feature_diagnosticity: Mapping[str, float] = field(default_factory=dict)
    prior_probability: float | None = None
    feature_counts: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize feature distributions and validate priors and diagnosticity values."""
        if not isinstance(self.frame, SchemaFrame):
            raise TypeError("frame must be a Point 09 SchemaFrame")
        normalized: dict[str, dict[str, float]] = {}
        for feature, values in self.feature_probabilities.items():
            name = _text("feature", str(feature)).casefold()
            row = {value_key(key): _probability("feature probability", probability)
                   for key, probability in values.items()}
            total = sum(row.values())
            if not row or total <= 0.0:
                raise ValueError("each feature probability row must have positive mass")
            normalized[name] = {key: value / total for key, value in row.items()}
        diagnosticity = {
            _text("feature", str(key)).casefold(): _probability("feature diagnosticity", value)
            for key, value in self.feature_diagnosticity.items()
        }
        if self.prior_probability is not None:
            object.__setattr__(self, "prior_probability",
                               _probability("prior_probability", self.prior_probability))
        counts: dict[str, dict[str, float]] = {}
        for feature, values in self.feature_counts.items():
            name = _text("feature", str(feature)).casefold()
            counts[name] = {value_key(key): _finite("feature count", value)
                            for key, value in values.items()}
            if any(value < 0.0 for value in counts[name].values()):
                raise ValueError("feature counts cannot be negative")
        object.__setattr__(self, "feature_probabilities", normalized)
        object.__setattr__(self, "feature_diagnosticity", diagnosticity)
        object.__setattr__(self, "feature_counts", counts)

    @property
    def schema_id(self) -> str:
        """Return the stable identity supplied by the Point 09 schema frame."""
        return self.frame.schema_id

    def to_dict(self) -> dict[str, Any]:
        """Export the profile and its reusable Point 09 structured frame."""
        return _jsonable({
            "schema_id": self.schema_id, "frame": self.frame.to_dict(),
            "feature_probabilities": self.feature_probabilities,
            "feature_diagnosticity": self.feature_diagnosticity,
            "prior_probability": self.prior_probability, "feature_counts": self.feature_counts,
        })


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Calibratable controls for candidate retrieval, inference, active sensing, and revision."""

    softmax_temperature: float = 1.0
    probability_floor: float = 1e-6
    default_reliability: float = 0.5
    relation_weight: float = 1.0
    feature_weight: float = 1.0
    context_weight: float = 0.6
    goal_weight: float = 0.6
    event_weight: float = 0.5
    contradiction_penalty: float = 1.0
    posterior_threshold: float = 0.60
    candidate_margin: float = 0.12
    unknown_log_prior: float = -1.5
    unknown_confidence_threshold: float = 0.45
    max_candidates: int = 8
    max_predictions: int = 12
    query_cost_weight: float = 0.5
    query_risk_weight: float = 1.0
    action_risk_weight: float = 2.0
    minimum_update_cases: int = 3
    update_learning_rate: float = 0.15

    def __post_init__(self) -> None:
        """Validate positive scales and bounded probability thresholds."""
        for name in ("softmax_temperature", "probability_floor"):
            if _finite(name, getattr(self, name)) <= 0.0:
                raise ValueError(name + " must be positive")
        for name in (
            "default_reliability", "posterior_threshold", "candidate_margin",
            "unknown_confidence_threshold", "update_learning_rate",
        ):
            _probability(name, getattr(self, name))
        for name in (
            "feature_weight", "relation_weight", "context_weight", "goal_weight",
            "event_weight", "contradiction_penalty", "query_cost_weight",
            "query_risk_weight", "action_risk_weight",
        ):
            if _finite(name, getattr(self, name)) < 0.0:
                raise ValueError(name + " cannot be negative")
        if self.max_candidates < 1 or self.max_predictions < 1 or self.minimum_update_cases < 1:
            raise ValueError("candidate, prediction, and update limits must be positive")


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """A posterior-ranked schema candidate and inspectable score components."""

    schema_id: str
    category: str
    posterior: float
    log_score: float
    feature_score: float
    relation_score: float
    context_score: float
    goal_score: float
    event_score: float
    matched_evidence_ids: tuple[str, ...] = ()
    conflicting_evidence_ids: tuple[str, ...] = ()
    parent_schema_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the posterior and scoring contributions for model auditing."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class InferredProperty:
    """A schema-derived value explicitly marked inferred, never relabeled as observed."""

    kind: str
    name: str
    value: Any
    confidence: float
    source_schema_id: str
    provenance: str = "schema_prediction"
    status: str = "inferred"
    supporting_evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return inferred content with confidence and its source schema."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class QueryRecommendation:
    """An information query ranked by expected information gain minus cost and risk."""

    feature: str
    action_label: str
    expected_information_gain: float
    cost: float
    risk: float
    net_value: float
    predicted_answers: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return estimated query value without claiming the query was executed."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class ActionDecision:
    """A risk-adjusted expected-utility recommendation among caller-supplied actions."""

    action_id: str
    expected_utility: float
    expected_risk: float
    selected: bool
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return expected utility and risk terms for a proposed action."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class FeedbackResult:
    """Prediction/outcome comparison distinguishing tested, contradicted, and untested items."""

    case_id: str
    tested_predictions: tuple[Mapping[str, Any], ...]
    untested_prediction_count: int
    mean_brier_score: float | None
    mean_surprise: float | None
    schema_update_authorized: bool
    updated_schema_id: str | None
    update_status: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize errors and the explicit schema-update authorization outcome."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Complete interpretation, predictions, decisions, provenance, and diagnostics."""

    case_id: str
    candidate_scores: tuple[CandidateScore, ...]
    selected_schema_id: str | None
    selected_category: str
    confidence: float
    inferred_properties: tuple[InferredProperty, ...]
    inferred_relations: tuple[InferredProperty, ...]
    next_event_predictions: tuple[InferredProperty, ...]
    query_recommendation: QueryRecommendation | None
    action_decision: ActionDecision | None
    evidence_ledger: tuple[Mapping[str, Any], ...]
    operation_trace: tuple[Mapping[str, Any], ...]
    biological: Mapping[str, Any]
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a complete source-aware schema-guided inference report."""
        return _jsonable({
            "case_id": self.case_id,
            "candidate_scores": [item.to_dict() for item in self.candidate_scores],
            "selected_schema_id": self.selected_schema_id, "selected_category": self.selected_category,
            "confidence": self.confidence,
            "inferred_properties": [item.to_dict() for item in self.inferred_properties],
            "inferred_relations": [item.to_dict() for item in self.inferred_relations],
            "next_event_predictions": [item.to_dict() for item in self.next_event_predictions],
            "query_recommendation": self.query_recommendation.to_dict() if self.query_recommendation else None,
            "action_decision": self.action_decision.to_dict() if self.action_decision else None,
            "evidence_ledger": self.evidence_ledger, "operation_trace": self.operation_trace,
            "biological": self.biological, "diagnostics": self.diagnostics,
        })
