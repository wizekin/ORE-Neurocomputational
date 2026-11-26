"""Typed contracts for the 44-operation semantic-cognition subsystem.

The records are intentionally framework-neutral. They describe lexical/concept
evidence and do not claim that a software feature map is a measured brain state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence
import math


def _validate_feature_map(name: str, values: Mapping[str, float]) -> None:
    for key, raw in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} feature names must be non-empty strings")
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} feature values must be finite and lie in [0, 1]")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return repr(value)


@dataclass(frozen=True, slots=True)
class SemanticConcept:
    """A provenance-tracked concept and its distributed semantic features."""

    concept_id: str
    label: str
    aliases: tuple[str, ...] = ()
    semantic_features: Mapping[str, float] = field(default_factory=dict)
    modality_features: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    lexical_category: str = "unknown"
    frequency: float = 1.0
    prototype_id: str | None = None
    language: str = "und"
    verified_by: str = "unspecified"
    source_reference: str = "unspecified"
    definition: str | None = None

    def __post_init__(self) -> None:
        if not self.concept_id.strip() or not self.label.strip():
            raise ValueError("concept_id and label must be non-empty")
        if self.definition is not None and not isinstance(self.definition, str):
            raise TypeError("definition must be a string or None")
        if not math.isfinite(self.frequency) or self.frequency <= 0:
            raise ValueError("frequency must be finite and positive")
        if any(not isinstance(alias, str) or not alias.strip() for alias in self.aliases):
            raise ValueError("aliases must contain non-empty strings")
        _validate_feature_map("semantic", self.semantic_features)
        for modality, features in self.modality_features.items():
            if not isinstance(modality, str) or not modality.strip():
                raise ValueError("modality names must be non-empty strings")
            _validate_feature_map(f"modality {modality}", features)


@dataclass(frozen=True, slots=True)
class WordFormEpisode:
    """A recalled encounter linking a surface word-form to concept evidence."""

    episode_id: str
    orthography: str
    concept_id: str | None = None
    phonological_form: tuple[str, ...] = ()
    semantic_features: Mapping[str, float] = field(default_factory=dict)
    context_features: Mapping[str, float] = field(default_factory=dict)
    strength: float = 1.0
    recency: float = 1.0
    source_reference: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.episode_id.strip() or not self.orthography.strip():
            raise ValueError("episode_id and orthography must be non-empty")
        for name in ("strength", "recency"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if any(not isinstance(phone, str) or not phone.strip() for phone in self.phonological_form):
            raise ValueError("phonological_form entries must be non-empty strings")
        _validate_feature_map("episode semantic", self.semantic_features)
        _validate_feature_map("episode context", self.context_features)


@dataclass(frozen=True, slots=True)
class SemanticRequest:
    """One word or ordered sequence for semantic access and concept selection."""

    sequence_id: str
    text: str | None = None
    tokens: Sequence[str] | None = None
    language: str = "und"
    task: str = "comprehension"
    context: Mapping[str, float] = field(default_factory=dict)
    modality_evidence: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    episodes: Sequence[WordFormEpisode] = ()
    source_reference: str | None = None
    contextual_records: Sequence[Any] = ()
    evidence_records: Sequence[Any] = ()
    source_provenance: Sequence[Any] = ()

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
        if not self.task.strip() or not self.language.strip():
            raise ValueError("task and language must be non-empty")
        _validate_feature_map("context", self.context)
        for modality, features in self.modality_evidence.items():
            if not isinstance(modality, str) or not modality.strip():
                raise ValueError("modality evidence keys must be non-empty strings")
            _validate_feature_map(f"input modality {modality}", features)
        if any(not isinstance(item, WordFormEpisode) for item in self.episodes):
            raise TypeError("episodes must contain WordFormEpisode records")
        if any(not isinstance(getattr(item, "context_id", None), str) for item in self.contextual_records):
            raise TypeError("contextual_records must contain typed context records")
        if any(not isinstance(getattr(item, "evidence_id", None), str) for item in self.evidence_records):
            raise TypeError("evidence_records must contain typed evidence records")
        if any(not isinstance(getattr(item, "source_id", None), str) for item in self.source_provenance):
            raise TypeError("source_provenance must contain typed source records")


@dataclass(frozen=True, slots=True)
class SemanticConfig:
    """Inference and reference LIF controls; all values are explicit hypotheses."""

    max_candidates: int = 10
    candidate_similarity_floor: float = 0.35
    interactive_steps: int = 5
    feedback_gain: float = 0.25
    lateral_inhibition: float = 0.15
    context_gain: float = 0.30
    episode_gain: float = 0.35
    modality_gain: float = 0.45
    prototype_gain: float = 0.20
    control_gain: float = 0.40
    confidence_threshold: float = 0.55
    ambiguity_margin: float = 0.08
    simulate_biology: bool = True
    dt_ms: float = 1.0
    steps_per_operation: int = 2
    membrane_tau_ms: float = 20.0
    refractory_ms: float = 2.0
    population_size: int = 12
    membrane_resistance_mohm: float = 100.0
    rest_mv: float = -70.0
    reset_mv: float = -68.0
    threshold_mv: float = -50.0
    current_gain_pa: float = 320.0
    recurrent_gain_pa: float = 18.0
    inhibition_gain_pa: float = 22.0

    def __post_init__(self) -> None:
        if self.max_candidates < 1 or self.interactive_steps < 1:
            raise ValueError("max_candidates and interactive_steps must be positive")
        if self.steps_per_operation < 1 or self.population_size < 2:
            raise ValueError("steps_per_operation must be positive and population_size >= 2")
        for name in (
            "candidate_similarity_floor", "feedback_gain", "lateral_inhibition",
            "context_gain", "episode_gain", "modality_gain", "prototype_gain",
            "control_gain", "confidence_threshold", "ambiguity_margin",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in (
            "dt_ms", "membrane_tau_ms", "membrane_resistance_mohm", "current_gain_pa",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("refractory_ms", "recurrent_gain_pa", "inhibition_gain_pa"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.rest_mv < self.threshold_mv or not self.rest_mv <= self.reset_mv < self.threshold_mv:
            raise ValueError("require rest_mv <= reset_mv < threshold_mv")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SemanticResult:
    """Functional and optional neural-reference outputs for one request."""

    sequence_id: str
    original_text: str
    normalized_text: str
    tokens: list[dict[str, Any]]
    semantic_state: dict[str, float]
    selected_concepts: list[dict[str, Any]]
    episodic_retrieval: list[dict[str, Any]]
    abstract_prototypes: list[dict[str, Any]]
    biological: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "schema": "ore.semantic_cognition.v1",
            "sequence_id": self.sequence_id,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "tokens": self.tokens,
            "semantic_state": self.semantic_state,
            "selected_concepts": self.selected_concepts,
            "episodic_retrieval": self.episodic_retrieval,
            "abstract_prototypes": self.abstract_prototypes,
            "biological": self.biological,
            "operation_trace": self.operation_trace,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        })
