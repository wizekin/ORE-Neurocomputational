"""Typed contracts for nine-stage sentence-level semantic composition.

The contracts preserve ambiguity and provenance. They distinguish a computational
semantic representation from a claim about a measured human brain state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence
import math


def _validate_features(name: str, values: Mapping[str, float]) -> None:
    """Validate a sparse semantic feature map whose values are bounded activations."""
    for key, raw in values.items():
        value = float(raw)
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} feature names must be non-empty strings")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} feature values must be finite and lie in [0, 1]")


def _jsonable(value: Any) -> Any:
    """Convert nested dataclasses, mappings, sequences, and numeric scalars to JSON-safe data."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return repr(value)


@dataclass(frozen=True, slots=True)
class PredicateFrame:
    """A provenance-tracked predicate valency frame with expected thematic roles.

    The frame states which participant roles a predicate can license and which
    semantic feature profiles are typical for each role. These are explicit
    lexical hypotheses supplied by the caller, not universal linguistic facts.
    """

    predicate: str
    roles: tuple[str, ...] = ("Agent", "Patient")
    required_roles: tuple[str, ...] = ()
    selectional_preferences: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    predicate_concept_id: str | None = None
    source_reference: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.predicate.strip():
            raise ValueError("predicate must be non-empty")
        if not self.roles or any(not role.strip() for role in self.roles):
            raise ValueError("roles must contain non-empty role names")
        if not set(self.required_roles).issubset(set(self.roles)):
            raise ValueError("required_roles must be a subset of roles")
        for role, features in self.selectional_preferences.items():
            if role not in self.roles:
                raise ValueError(f"selectional preference references unknown role: {role}")
            _validate_features(f"selectional_preferences[{role}]", features)


@dataclass(frozen=True, slots=True)
class SemanticCompositionRequest:
    """Describe one sentence and its optional upstream syntax and lexical evidence.

    Pass exactly one of text or tokens. concept_hypotheses maps token indexes to
    explicit concept IDs when upstream lexical resolution should constrain search.
    syntax_result may be an ore_syntax SyntaxResult or its to_dict() mapping.
    """

    sequence_id: str
    text: str | None = None
    tokens: Sequence[str] | None = None
    language: str = "en"
    concept_hypotheses: Mapping[int, Sequence[str]] = field(default_factory=dict)
    syntax_result: Any | None = None
    context_features: Mapping[str, float] = field(default_factory=dict)
    source_reference: str | None = None

    def __post_init__(self) -> None:
        if not self.sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")
        if (self.text is None) == (self.tokens is None):
            raise ValueError("provide exactly one of text or tokens")
        if self.text is not None and not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.tokens is not None and (
            not self.tokens or any(not isinstance(token, str) or not token.strip() for token in self.tokens)
        ):
            raise ValueError("tokens must be a non-empty sequence of non-empty strings")
        if not self.language.strip():
            raise ValueError("language must be non-empty")
        for index, concept_ids in self.concept_hypotheses.items():
            if not isinstance(index, int) or index < 0:
                raise ValueError("concept_hypotheses keys must be non-negative token indexes")
            if any(not isinstance(item, str) or not item.strip() for item in concept_ids):
                raise ValueError("concept_hypotheses must contain non-empty concept IDs")
        _validate_features("context_features", self.context_features)


@dataclass(frozen=True, slots=True)
class SemanticCompositionConfig:
    """Numerical controls for evidence integration, confidence gating, and LIF reference.

    All values are engineering parameters for an inspectable prototype. They are
    not measured neural constants or empirically calibrated language probabilities.
    """

    max_concepts_per_token: int = 8
    lexical_similarity_floor: float = 0.45
    lexical_weight: float = 0.42
    context_weight: float = 0.20
    frequency_weight: float = 0.08
    syntax_weight: float = 0.18
    selectional_weight: float = 0.12
    softmax_temperature: float = 0.35
    confidence_threshold: float = 0.50
    ambiguity_margin: float = 0.08
    max_scope_readings: int = 4
    max_tokens: int = 100
    simulate_biology: bool = True
    population_size: int = 12
    dt_ms: float = 1.0
    steps_per_operation: int = 4
    membrane_tau_ms: float = 20.0
    refractory_ms: float = 2.0
    rest_mv: float = -70.0
    reset_mv: float = -68.0
    threshold_mv: float = -50.0
    membrane_resistance_mohm: float = 100.0
    current_gain_pa: float = 300.0
    recurrent_gain_pa: float = 18.0
    inhibition_gain_pa: float = 20.0

    def __post_init__(self) -> None:
        if self.max_concepts_per_token < 1 or self.max_scope_readings < 1 or self.max_tokens < 1:
            raise ValueError("candidate, scope, and token limits must be positive")
        if self.population_size < 2 or self.steps_per_operation < 1:
            raise ValueError("population_size must be >= 2 and steps_per_operation positive")
        for name in (
            "lexical_similarity_floor", "lexical_weight", "context_weight",
            "frequency_weight", "syntax_weight", "selectional_weight",
            "confidence_threshold", "ambiguity_margin",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        if self.softmax_temperature <= 0 or not math.isfinite(self.softmax_temperature):
            raise ValueError("softmax_temperature must be finite and positive")
        for name in ("dt_ms", "membrane_tau_ms", "membrane_resistance_mohm", "current_gain_pa"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("refractory_ms", "recurrent_gain_pa", "inhibition_gain_pa"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.rest_mv <= self.reset_mv < self.threshold_mv:
            raise ValueError("require rest_mv <= reset_mv < threshold_mv")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for metadata and JSON serialization."""
        return asdict(self)


@dataclass(slots=True)
class SemanticCompositionResult:
    """Expose every semantic layer, alternatives, diagnostics, and operation traces."""

    sequence_id: str
    original_text: str
    tokens: list[str]
    word_concepts: list[dict[str, Any]]
    combined_meaning: dict[str, Any]
    predicate_arguments: list[dict[str, Any]]
    thematic_roles: list[dict[str, Any]]
    propositions: list[dict[str, Any]]
    negation_modality: dict[str, Any]
    quantifier_scopes: list[dict[str, Any]]
    causal_relations: list[dict[str, Any]]
    sentence_representation: dict[str, Any]
    biological: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete sentence-level representation without losing alternatives."""
        return _jsonable({
            "schema": "ore.semantic_composition.9stage.v1",
            "sequence_id": self.sequence_id,
            "original_text": self.original_text,
            "tokens": self.tokens,
            "word_concepts": self.word_concepts,
            "combined_meaning": self.combined_meaning,
            "predicate_arguments": self.predicate_arguments,
            "thematic_roles": self.thematic_roles,
            "propositions": self.propositions,
            "negation_modality": self.negation_modality,
            "quantifier_scopes": self.quantifier_scopes,
            "causal_relations": self.causal_relations,
            "sentence_representation": self.sentence_representation,
            "biological": self.biological,
            "operation_trace": self.operation_trace,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        })
