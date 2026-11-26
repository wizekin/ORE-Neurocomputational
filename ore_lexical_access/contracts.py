"""Typed request, configuration and result contracts for seven-stage lexical access."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping
import math


def _unit_map(name: str, values: Mapping[str, float]) -> None:
    for key, raw in values.items():
        value = float(raw)
        if not isinstance(key, str) or not key.strip() or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must map non-empty names to finite values in [0, 1]")


@dataclass(frozen=True, slots=True)
class LexicalAccessRequest:
    """One lexical token and optional semantic/syntactic context evidence."""

    sequence_id: str
    word: str
    language: str = "und"
    context_features: Mapping[str, float] = field(default_factory=dict)
    contextual_category_priors: Mapping[str, float] = field(default_factory=dict)
    source_reference: str | None = None
    timestamp_ms: float | None = None
    contextual_records: tuple[Any, ...] = ()
    evidence_records: tuple[Any, ...] = ()
    source_provenance: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not self.sequence_id.strip() or not self.word.strip():
            raise ValueError("sequence_id and word must be non-empty")
        if any(char.isspace() for char in self.word.strip()):
            raise ValueError("word must contain one token, not a whitespace-separated phrase")
        _unit_map("context_features", self.context_features)
        _unit_map("contextual_category_priors", self.contextual_category_priors)
        if any(not isinstance(getattr(item, "context_id", None), str) for item in self.contextual_records):
            raise TypeError("contextual_records must contain typed context records")
        if any(not isinstance(getattr(item, "evidence_id", None), str) for item in self.evidence_records):
            raise TypeError("evidence_records must contain typed evidence records")
        if any(not isinstance(getattr(item, "source_id", None), str) for item in self.source_provenance):
            raise TypeError("source_provenance must contain typed source records")
        if self.timestamp_ms is not None and (
            not math.isfinite(self.timestamp_ms) or self.timestamp_ms < 0
        ):
            raise ValueError("timestamp_ms must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class LexicalAccessConfig:
    """Numerical controls; defaults are hypotheses, not fitted neural constants."""

    max_candidates: int = 30
    max_edit_distance: int = 3
    minimum_similarity: float = 0.50
    identity_temperature: float = 0.18
    semantic_temperature: float = 0.20
    category_temperature: float = 0.20
    morphology_temperature: float = 0.22
    context_semantic_gain: float = 0.65
    morphology_gain: float = 0.45
    whole_word_bias: float = 0.30
    stem_frequency_gain: float = 0.25
    affix_frequency_gain: float = 0.12
    affix_productivity_gain: float = 0.25
    category_consistency_gain: float = 0.20
    semantic_consistency_gain: float = 0.20
    confidence_threshold: float = 0.55
    ambiguity_margin: float = 0.08
    # Incremental interactive-activation network.
    activation_temperature: float = 0.30
    activation_damping: float = 0.55
    bottom_up_gain: float = 1.55
    frequency_prior_gain: float = 0.30
    semantic_feedback_gain: float = 0.14
    category_feedback_gain: float = 0.10
    morphology_feedback_gain: float = 0.14
    recurrent_excitation: float = 0.18
    lateral_inhibition: float = 0.24
    activation_bias: float = -1.0
    settle_iterations_per_character: int = 4
    # Neural reference track.
    simulate_biology: bool = True
    dt_ms: float = 1.0
    steps_per_character: int = 2
    membrane_tau_ms: float = 20.0
    refractory_ms: float = 2.0
    population_size: int = 12
    rest_mv: float = -70.0
    reset_mv: float = -68.0
    threshold_mv: float = -50.0
    membrane_resistance_mohm: float = 100.0
    current_gain_pa: float = 300.0
    recurrent_gain_pa: float = 20.0
    inhibition_gain_pa: float = 22.0
    adaptation_current_gain_pa: float = 4.0

    def __post_init__(self) -> None:
        if self.max_candidates < 1 or self.max_edit_distance < 0:
            raise ValueError("candidate and edit-distance limits are invalid")
        if self.settle_iterations_per_character < 1 or self.steps_per_character < 1:
            raise ValueError("iteration counts must be positive")
        if self.population_size < 2:
            raise ValueError("population_size must be at least two")
        bounded = (
            "minimum_similarity", "context_semantic_gain", "morphology_gain", "whole_word_bias",
            "stem_frequency_gain", "affix_frequency_gain", "affix_productivity_gain",
            "category_consistency_gain", "semantic_consistency_gain", "confidence_threshold",
            "ambiguity_margin", "activation_damping", "bottom_up_gain", "frequency_prior_gain",
            "semantic_feedback_gain", "category_feedback_gain", "morphology_feedback_gain",
            "recurrent_excitation", "lateral_inhibition",
        )
        for name in bounded:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 2.0:
                raise ValueError(f"{name} must be finite and in [0, 2]")
        positive = (
            "identity_temperature", "semantic_temperature", "category_temperature",
            "morphology_temperature", "activation_temperature", "dt_ms", "membrane_tau_ms",
            "membrane_resistance_mohm", "current_gain_pa",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.refractory_ms < 0 or self.recurrent_gain_pa < 0 or self.inhibition_gain_pa < 0:
            raise ValueError("neural current/refractory parameters are invalid")
        if not self.rest_mv <= self.reset_mv < self.threshold_mv:
            raise ValueError("require rest_mv <= reset_mv < threshold_mv")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LexicalAccessResult:
    """Inspectable output of all seven lexical-access stages."""

    sequence_id: str
    original_word: str
    normalized_word: str
    lexical_identity_candidates: list[dict[str, Any]]
    selected_lexical_identity: dict[str, Any] | None
    meaning_candidates: list[dict[str, Any]]
    selected_meaning: dict[str, Any] | None
    grammatical_categories: list[dict[str, Any]]
    selected_grammatical_category: dict[str, Any] | None
    morphological_alternatives: list[dict[str, Any]]
    stem_affix_retrieval: list[dict[str, Any]]
    preferred_morphology: dict[str, Any] | None
    incremental_activation: list[dict[str, Any]]
    biological: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        def clean(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {str(key): clean(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [clean(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return repr(value)

        return clean({
            "schema": "ore.lexical_access.7stage.v1",
            "sequence_id": self.sequence_id,
            "original_word": self.original_word,
            "normalized_word": self.normalized_word,
            "lexical_identity_candidates": self.lexical_identity_candidates,
            "selected_lexical_identity": self.selected_lexical_identity,
            "meaning_candidates": self.meaning_candidates,
            "selected_meaning": self.selected_meaning,
            "grammatical_categories": self.grammatical_categories,
            "selected_grammatical_category": self.selected_grammatical_category,
            "morphological_alternatives": self.morphological_alternatives,
            "stem_affix_retrieval": self.stem_affix_retrieval,
            "preferred_morphology": self.preferred_morphology,
            "incremental_activation": self.incremental_activation,
            "biological": self.biological,
            "operation_trace": self.operation_trace,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        })
