"""Typed contracts for incremental parsing and garden-path recovery (Point 23)."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


def _unit(name: str, value: float) -> None:
    """Validate a finite probability-like value."""
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")


@dataclass(frozen=True)
class LexicalCandidate:
    """One lexical/grammatical analysis available for a word form."""

    category: str
    lemma: str = ""
    sense: str = ""
    prior: float = 1.0
    features: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the grammatical category and lexical prior."""
        if not self.category.strip():
            raise ValueError("lexical category must be non-empty")
        _unit("lexical candidate prior", self.prior)


@dataclass(frozen=True)
class ParseDependency:
    """An indexed dependency or thematic-role relation inside a candidate parse."""

    head_index: int
    dependent_index: int
    relation: str
    thematic_role: str = ""

    def __post_init__(self) -> None:
        """Require valid token indices and a named relation."""
        if self.head_index < 0 or self.dependent_index < 0:
            raise ValueError("dependency token indices must be non-negative")
        if not self.relation.strip():
            raise ValueError("dependency relation must be non-empty")


@dataclass(frozen=True)
class ParseChoicePoint:
    """A structural decision that distinguishes one candidate parse from another."""

    decision_id: str
    token_index: int
    choice: str

    def __post_init__(self) -> None:
        """Validate a stable decision identifier and its source position."""
        if not self.decision_id.strip() or not self.choice.strip():
            raise ValueError("decision_id and choice must be non-empty")
        if self.token_index < 0:
            raise ValueError("choice-point token_index must be non-negative")


@dataclass(frozen=True)
class ParseHypothesis:
    """One grammar-generated parse candidate with explicit likelihoods and structure."""

    parse_id: str
    description: str
    prior: float = 0.5
    token_categories: Mapping[int, str] = field(default_factory=dict)
    structural_likelihoods: Mapping[int, float] = field(default_factory=dict)
    semantic_plausibility: float = 0.5
    cue_likelihoods: Mapping[str, float] = field(default_factory=dict)
    dependencies: tuple[ParseDependency, ...] = ()
    thematic_roles: Mapping[str, int] = field(default_factory=dict)
    choice_points: tuple[ParseChoicePoint, ...] = ()
    unresolved_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate parse priors, token positions, and likelihoods."""
        if not self.parse_id.strip() or not self.description.strip():
            raise ValueError("parse_id and description must be non-empty")
        _unit("parse prior", self.prior)
        _unit("semantic_plausibility", self.semantic_plausibility)
        for position, category in self.token_categories.items():
            if int(position) < 0 or not str(category).strip():
                raise ValueError("token_categories must use non-negative indices and named categories")
        for position, value in self.structural_likelihoods.items():
            if int(position) < 0:
                raise ValueError("structural likelihood indices must be non-negative")
            _unit(f"structural_likelihoods[{position}]", value)
        for name, value in self.cue_likelihoods.items():
            _unit(f"cue_likelihoods[{name}]", value)
        for role, index in self.thematic_roles.items():
            if not str(role).strip() or int(index) < 0:
                raise ValueError("thematic role entries need a name and non-negative token index")


@dataclass(frozen=True)
class SentenceRequest:
    """Describe a sentence, supplied grammar candidates, context, and task demands."""

    sentence_id: str
    text: str
    language: str = "en"
    modality: str = "written"
    lexicon: Mapping[str, tuple[LexicalCandidate, ...]] = field(default_factory=dict)
    parse_hypotheses: tuple[ParseHypothesis, ...] = ()
    context_likelihoods: Mapping[str, float] = field(default_factory=dict)
    prosody_likelihoods: Mapping[str, float] = field(default_factory=dict)
    punctuation_likelihoods: Mapping[str, float] = field(default_factory=dict)
    discourse_likelihoods: Mapping[str, float] = field(default_factory=dict)
    task_goal: str = "precise"
    construction_id: str = "unspecified-construction"
    feedback_parse_id: str | None = None
    working_memory_load: float = 0.0
    attention: float = 1.0

    def __post_init__(self) -> None:
        """Validate sentence metadata and ensure referenced parse candidates are unique."""
        if not self.sentence_id.strip() or not self.text.strip():
            raise ValueError("sentence_id and text must be non-empty")
        if self.modality not in {"written", "spoken"}:
            raise ValueError("modality must be 'written' or 'spoken'")
        if self.task_goal not in {"precise", "gist"}:
            raise ValueError("task_goal must be 'precise' or 'gist'")
        if not self.language.strip() or not self.construction_id.strip():
            raise ValueError("language and construction_id must be non-empty")
        ids = [item.parse_id for item in self.parse_hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("parse_hypothesis parse_id values must be unique")
        for name in ("working_memory_load", "attention"):
            _unit(name, getattr(self, name))
        for source in (self.context_likelihoods, self.prosody_likelihoods,
                       self.punctuation_likelihoods, self.discourse_likelihoods):
            for parse_id, value in source.items():
                _unit(f"cue likelihood for {parse_id}", value)
        if self.feedback_parse_id is not None and self.parse_hypotheses:
            if self.feedback_parse_id not in ids:
                raise ValueError("feedback_parse_id must match a supplied parse hypothesis")


@dataclass(frozen=True)
class Point23Config:
    """Control beam size, memory accessibility, revision thresholds, and adaptation."""

    beam_width: int = 8
    parse_temperature: float = 1.0
    minimum_viability: float = 0.015
    conflict_threshold: float = 0.20
    revision_margin: float = 0.10
    full_reanalysis_threshold: float = 0.55
    confidence_threshold: float = 0.72
    memory_decay: float = 0.04
    interference_weight: float = 0.15
    working_memory_capacity: float = 1.0
    learning_rate: float = 0.15
    prior_smoothing: float = 0.5
    simulate_biology: bool = True

    def __post_init__(self) -> None:
        """Validate capacities, rates, and thresholds."""
        if self.beam_width < 1:
            raise ValueError("beam_width must be at least one")
        if self.parse_temperature <= 0 or not math.isfinite(self.parse_temperature):
            raise ValueError("parse_temperature must be positive and finite")
        if self.memory_decay < 0 or not math.isfinite(self.memory_decay):
            raise ValueError("memory_decay must be finite and non-negative")
        if self.interference_weight < 0 or not math.isfinite(self.interference_weight):
            raise ValueError("interference_weight must be finite and non-negative")
        if self.prior_smoothing <= 0 or not math.isfinite(self.prior_smoothing):
            raise ValueError("prior_smoothing must be positive and finite")
        for name in ("minimum_viability", "conflict_threshold", "revision_margin",
                     "full_reanalysis_threshold", "confidence_threshold",
                     "working_memory_capacity", "learning_rate"):
            _unit(name, getattr(self, name))


@dataclass
class Point23State:
    """Retain construction exposure, parse feedback, and prior interpretation history."""

    construction_counts: dict[str, dict[str, float]] = field(default_factory=dict)
    construction_exposures: dict[str, int] = field(default_factory=dict)
    interpretation_history: list[dict[str, Any]] = field(default_factory=list)
    run_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return an independent snapshot suitable for storage and tests."""
        return {
            "construction_counts": {
                key: dict(value) for key, value in self.construction_counts.items()
            },
            "construction_exposures": dict(self.construction_exposures),
            "interpretation_history": [dict(item) for item in self.interpretation_history],
            "run_count": self.run_count,
        }


@dataclass(frozen=True)
class Point23Result:
    """Expose parse evolution, revision diagnostics, confidence, and both model tracks."""

    sentence_id: str
    tokens: tuple[str, ...]
    lexical_state: tuple[Mapping[str, Any], ...]
    initial_parse_id: str
    final_parse_id: str
    parse_probabilities: Mapping[str, float]
    thematic_roles: Mapping[str, Any]
    confidence: Mapping[str, float]
    interpretation_status: str
    unresolved_ambiguities: tuple[str, ...]
    revisions: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]
    behavioral_predictions: Mapping[str, Any]
    biological: Mapping[str, Any]
    operation_trace: tuple[Mapping[str, Any], ...]
    final_state: Mapping[str, Any]
    metadata: Mapping[str, Any]
