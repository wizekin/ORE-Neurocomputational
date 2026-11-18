"""Typed contracts for Point 12 context-dependent interpretation.

The records preserve source provenance, uncertainty, and the distinction between observed,
stated, retrieved, and inferred information. They are computational data structures, not
literal neural representations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _required_text(name: str, value: str) -> str:
    """Validate a required text identifier and return its trimmed value."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a non-empty string")
    return value.strip()


def _finite_number(name: str, value: float) -> float:
    """Convert a numeric input to a finite float."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _probability(name: str, value: float) -> float:
    """Validate a probability in the closed interval [0, 1]."""
    result = _finite_number(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(name + " must be in [0, 1]")
    return result


def _plain(value: Any) -> Any:
    """Convert nested records, mappings, and containers into JSON-compatible values."""
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
class ContextEvidence:
    """Represent one discourse, knowledge, scene, convention, or speaker-context record."""

    evidence_id: str
    source_type: str
    content: str
    confidence: float = 1.0
    reliability: float = 1.0
    concepts: tuple[str, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)
    goal_tags: tuple[str, ...] = ()
    topic_id: str = ""
    speaker_id: str = ""
    timestamp: str = ""
    provenance: str = "retrieved_context"

    def __post_init__(self) -> None:
        """Validate identity, source category, confidence, and reliability."""
        _required_text("evidence_id", self.evidence_id)
        _required_text("content", self.content)
        allowed = {"discourse", "knowledge", "scene", "convention", "speaker"}
        if self.source_type not in allowed:
            raise ValueError("source_type must be one of " + ", ".join(sorted(allowed)))
        _probability("confidence", self.confidence)
        _probability("reliability", self.reliability)
        object.__setattr__(self, "concepts", tuple(
            _required_text("concept", item).lower() for item in self.concepts
        ))
        object.__setattr__(self, "goal_tags", tuple(
            _required_text("goal_tag", item).lower() for item in self.goal_tags
        ))
        object.__setattr__(self, "facts", dict(self.facts))

    def to_dict(self) -> dict[str, Any]:
        """Return the evidence record with explicit provenance and reliability fields."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class CandidateHypothesis:
    """Describe a possible meaning together with cues, assumptions, and candidate facts."""

    candidate_id: str
    meaning: str
    cue_terms: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)
    prior_hint: float | None = None

    def __post_init__(self) -> None:
        """Validate a candidate hypothesis without assuming it is true."""
        _required_text("candidate_id", self.candidate_id)
        _required_text("meaning", self.meaning)
        object.__setattr__(self, "cue_terms", tuple(
            _required_text("cue_term", item).lower() for item in self.cue_terms
        ))
        object.__setattr__(self, "assumptions", tuple(str(item) for item in self.assumptions))
        object.__setattr__(self, "facts", dict(self.facts))
        if self.prior_hint is not None:
            _probability("prior_hint", self.prior_hint)

    def to_dict(self) -> dict[str, Any]:
        """Return the candidate and its explicitly marked assumptions."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class InputEvent:
    """Represent the newly received linguistic, perceptual, or environmental event."""

    event_id: str
    content: str
    modality: str = "text"
    candidates: tuple[CandidateHypothesis, ...] = ()
    explicit_facts: Mapping[str, Any] = field(default_factory=dict)
    features: Mapping[str, Any] = field(default_factory=dict)
    scene_cues: tuple[str, ...] = ()
    candidate_evidence: Mapping[str, float] = field(default_factory=dict)
    speaker_id: str = ""
    goal: str = ""
    reliability: float = 1.0
    request_action: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the event and normalize nested evidence collections."""
        _required_text("event_id", self.event_id)
        _required_text("content", self.content)
        _required_text("modality", self.modality)
        _probability("reliability", self.reliability)
        if len({item.candidate_id for item in self.candidates}) != len(self.candidates):
            raise ValueError("candidate_id values must be unique within one event")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "explicit_facts", dict(self.explicit_facts))
        object.__setattr__(self, "features", dict(self.features))
        object.__setattr__(self, "scene_cues", tuple(
            _required_text("scene_cue", item).lower() for item in self.scene_cues
        ))
        evidence = {str(key): _probability("candidate_evidence", value)
                    for key, value in self.candidate_evidence.items()}
        object.__setattr__(self, "candidate_evidence", evidence)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return the event while retaining the difference between claims and metadata."""
        return _plain(asdict(self))


@dataclass
class SituationModel:
    """Maintain the current entities, supported facts, events, conflicts, and open ambiguities."""

    entities: dict[str, Any] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    fact_provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    unresolved_items: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    revision_count: int = 0

    def __post_init__(self) -> None:
        """Validate confidence and detach mutable input mappings and lists."""
        self.confidence = _probability("confidence", self.confidence)
        self.entities = dict(self.entities)
        self.facts = dict(self.facts)
        self.fact_provenance = dict(self.fact_provenance)
        self.events = list(self.events)
        self.unresolved_items = list(self.unresolved_items)
        self.conflicts = list(self.conflicts)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of the evolving situation model."""
        return _plain({
            "entities": self.entities, "facts": self.facts,
            "fact_provenance": self.fact_provenance, "events": self.events,
            "unresolved_items": self.unresolved_items, "conflicts": self.conflicts,
            "confidence": self.confidence, "revision_count": self.revision_count,
        })


@dataclass
class ContextState:
    """Hold dynamic discourse, long-term knowledge, scene, convention, goal, and situation state."""

    discourse_history: list[ContextEvidence] = field(default_factory=list)
    long_term_knowledge: list[ContextEvidence] = field(default_factory=list)
    scene_evidence: list[ContextEvidence] = field(default_factory=list)
    conventions: list[ContextEvidence] = field(default_factory=list)
    speaker_models: list[ContextEvidence] = field(default_factory=list)
    current_goal: str = ""
    goal_source_preferences: Mapping[str, float] = field(default_factory=dict)
    current_topic_id: str = ""
    situation: SituationModel = field(default_factory=SituationModel)
    context_weights: dict[str, float] = field(default_factory=lambda: {
        "discourse": 1.0, "knowledge": 1.0, "scene": 1.0,
        "convention": 1.0, "speaker": 1.0,
    })

    def __post_init__(self) -> None:
        """Validate context-source preferences and preserve mutable state containers."""
        for name in ("discourse_history", "long_term_knowledge", "scene_evidence",
                     "conventions", "speaker_models"):
            setattr(self, name, list(getattr(self, name)))
        weights = {str(key): _probability("context_weight", value)
                   for key, value in self.context_weights.items()}
        self.context_weights = weights
        preferences = {str(key): _probability("goal_source_preference", value)
                       for key, value in self.goal_source_preferences.items()}
        self.goal_source_preferences = preferences

    def to_dict(self) -> dict[str, Any]:
        """Return context source counts, current goal, weights, and the situation snapshot."""
        return _plain({
            "discourse_history": [item.to_dict() for item in self.discourse_history],
            "long_term_knowledge": [item.to_dict() for item in self.long_term_knowledge],
            "scene_evidence": [item.to_dict() for item in self.scene_evidence],
            "conventions": [item.to_dict() for item in self.conventions],
            "speaker_models": [item.to_dict() for item in self.speaker_models],
            "current_goal": self.current_goal,
            "goal_source_preferences": self.goal_source_preferences,
            "current_topic_id": self.current_topic_id,
            "context_weights": self.context_weights,
            "situation": self.situation.to_dict(),
        })


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """Store a candidate's contextual prior, input likelihood, posterior, and evidence trail."""

    candidate: CandidateHypothesis
    context_prior: float
    input_likelihood: float
    posterior: float
    context_score: float
    input_score: float
    evidence_ids: tuple[str, ...] = ()
    conflict_penalty: float = 0.0

    def __post_init__(self) -> None:
        """Validate normalized probabilities and non-negative conflict penalties."""
        _probability("context_prior", self.context_prior)
        _probability("input_likelihood", self.input_likelihood)
        _probability("posterior", self.posterior)
        _finite_number("context_score", self.context_score)
        _finite_number("input_score", self.input_score)
        penalty = _finite_number("conflict_penalty", self.conflict_penalty)
        if penalty < 0:
            raise ValueError("conflict_penalty must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a complete score record with candidate assumptions and source identifiers."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class ContextInterpretationConfig:
    """Configure contextual retrieval, Bayesian combination, uncertainty, and state adaptation."""

    temperature: float = 1.0
    selection_threshold: float = 0.62
    minimum_posterior_margin: float = 0.12
    maximum_normalized_entropy: float = 0.82
    context_recency_half_life: float = 12.0
    context_match_threshold: float = 0.0
    conflict_penalty: float = 2.0
    unknown_candidate_id: str = "unknown"
    maximum_context_items_per_source: int = 12
    adaptation_rate: float = 0.08
    minimum_probability: float = 1e-6

    def __post_init__(self) -> None:
        """Validate all probability-like thresholds and positive numerical parameters."""
        if _finite_number("temperature", self.temperature) <= 0:
            raise ValueError("temperature must be positive")
        _probability("selection_threshold", self.selection_threshold)
        _probability("minimum_posterior_margin", self.minimum_posterior_margin)
        _probability("maximum_normalized_entropy", self.maximum_normalized_entropy)
        _probability("context_match_threshold", self.context_match_threshold)
        if _finite_number("context_recency_half_life", self.context_recency_half_life) <= 0:
            raise ValueError("context_recency_half_life must be positive")
        if _finite_number("conflict_penalty", self.conflict_penalty) < 0:
            raise ValueError("conflict_penalty must be non-negative")
        if self.maximum_context_items_per_source < 1:
            raise ValueError("maximum_context_items_per_source must be at least 1")
        _probability("adaptation_rate", self.adaptation_rate)
        if not 0 < self.minimum_probability < 0.5:
            raise ValueError("minimum_probability must be in (0, 0.5)")
        _required_text("unknown_candidate_id", self.unknown_candidate_id)


@dataclass(frozen=True, slots=True)
class ContextInterpretationResult:
    """Represent the selected or unresolved interpretation, its trace, and supporting state."""

    event_id: str
    status: str
    selected: ScoredCandidate | None
    candidates: tuple[ScoredCandidate, ...]
    uncertainty: float
    normalized_entropy: float
    posterior_margin: float
    conflicts: tuple[Mapping[str, Any], ...]
    evidence_ledger: tuple[Mapping[str, Any], ...]
    operation_trace: tuple[Mapping[str, Any], ...]
    biological: Mapping[str, Any]
    output: Mapping[str, Any]
    situation: Mapping[str, Any]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate the result status and probability-valued diagnostics."""
        _required_text("event_id", self.event_id)
        if self.status not in {"selected", "ambiguous", "unresolved", "revised"}:
            raise ValueError("Unsupported interpretation status: " + self.status)
        for name in ("uncertainty", "normalized_entropy", "posterior_margin"):
            value = _finite_number(name, getattr(self, name))
            if value < 0:
                raise ValueError(name + " must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete result in a JSON-compatible form."""
        return _plain(asdict(self))
