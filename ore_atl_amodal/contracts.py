"""Validated data contracts for the 19-stage amodal ATL semantic model.

The contracts distinguish current observations, stored semantic prototypes,
phrase inputs, and explicitly authorized learning experiences. Numerical
representations are computational abstractions, not measurements of neurons.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


MODALITIES = (
    "visual",
    "auditory",
    "tactile",
    "spatial_motor",
    "linguistic",
    "affective",
    "functional",
)

_MODALITY_ALIASES = {
    "visual": "visual", "vision": "visual", "image": "visual", "picture": "visual",
    "auditory": "auditory", "audio": "auditory", "sound": "auditory",
    "phonological": "auditory", "spoken": "auditory",
    "tactile": "tactile", "touch": "tactile", "somatosensory": "tactile",
    "motor": "spatial_motor", "action": "spatial_motor", "spatial": "spatial_motor",
    "spatial_motor": "spatial_motor",
    "language": "linguistic", "text": "linguistic", "word": "linguistic",
    "verbal": "linguistic", "linguistic": "linguistic",
    "emotion": "affective", "emotional": "affective", "affect": "affective",
    "affective": "affective",
    "functional": "functional", "use": "functional", "purpose": "functional",
}


def canonical_modality(name: str) -> str:
    """Map a caller-supplied modality name to one of the model's canonical spokes."""
    key = str(name).strip().casefold().replace("-", "_").replace(" ", "_")
    if key not in _MODALITY_ALIASES:
        raise ValueError(f"Unsupported modality: {name!r}")
    return _MODALITY_ALIASES[key]


def validate_feature_map(name: str, values: Mapping[str, float]) -> None:
    """Validate named feature activations as finite graded values in the interval [0, 1]."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping of feature names to activations")
    for feature, raw in values.items():
        value = float(raw)
        if not isinstance(feature, str) or not feature.strip():
            raise ValueError(f"{name} feature names must be non-empty strings")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} values must be finite and lie in [0, 1]")


def normalize_modalities(
    inputs: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Canonicalize modality aliases and merge duplicate channels without losing activation strength."""
    normalized: dict[str, dict[str, float]] = {}
    for modality, features in inputs.items():
        key = canonical_modality(modality)
        validate_feature_map(f"{key} input", features)
        target = normalized.setdefault(key, {})
        for feature, value in features.items():
            target[str(feature)] = max(target.get(str(feature), 0.0), float(value))
    return normalized


def _jsonable(value: Any) -> Any:
    """Convert nested dataclasses, arrays, mappings, and tuples to JSON-safe primitives."""
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(frozen=True, slots=True)
class AmodalConcept:
    """A concept prototype linking a shared semantic feature map to optional modality-specific spokes."""

    concept_id: str
    label: str
    semantic_features: Mapping[str, float]
    modality_features: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    category: str = ""
    source_reference: str = "unspecified"

    def __post_init__(self) -> None:
        """Reject incomplete identities, malformed features, and unsupported modality keys."""
        if not self.concept_id.strip() or not self.label.strip():
            raise ValueError("Concept identifiers and labels must be non-empty")
        if not self.semantic_features and not self.modality_features:
            raise ValueError("A concept needs semantic or modality-specific features")
        validate_feature_map("concept semantic", self.semantic_features)
        normalize_modalities(self.modality_features)
        if not self.source_reference.strip():
            raise ValueError("Concept source_reference must be non-empty")


@dataclass(frozen=True, slots=True)
class AmodalATLConfig:
    """Numerical controls for encoders, bilateral ATL recurrence, phrase composition, learning, and LIF simulation."""

    hidden_dim: int = 32
    recurrent_steps: int = 10
    convergence_epsilon: float = 1e-5
    modality_temperature: float = 0.65
    feature_threshold: float = 0.05
    feedback_gain: float = 0.20
    noun_retention: float = 0.90
    phrase_update_rate: float = 0.55
    learning_rate: float = 0.025
    weight_decay: float = 0.001
    learning_authorized: bool = False
    simulate_biology: bool = True
    population_size: int = 8
    steps_per_stage: int = 2
    dt_ms: float = 1.0
    membrane_tau_ms: float = 20.0
    membrane_resistance_mohm: float = 100.0
    rest_mv: float = -70.0
    threshold_mv: float = -50.0
    reset_mv: float = -70.0
    refractory_ms: float = 2.0
    input_current_pa: float = 180.0
    recurrent_current_pa: float = 16.0
    inhibitory_current_pa: float = 8.0
    random_seed: int = 19

    def __post_init__(self) -> None:
        """Validate positive dimensions, stable time constants, and bounded update gains."""
        if self.hidden_dim < 4 or self.recurrent_steps < 1:
            raise ValueError("hidden_dim must be at least four and recurrent_steps must be positive")
        if self.population_size < 1 or self.steps_per_stage < 1:
            raise ValueError("population_size and steps_per_stage must be positive")
        for name in ("convergence_epsilon", "modality_temperature", "membrane_tau_ms",
                     "membrane_resistance_mohm", "dt_ms"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("feedback_gain", "noun_retention", "phrase_update_rate",
                     "learning_rate", "weight_decay", "feature_threshold"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not self.rest_mv < self.threshold_mv:
            raise ValueError("threshold_mv must be greater than rest_mv")


@dataclass(frozen=True, slots=True)
class AmodalATLRequest:
    """One multimodal concept-access or adjective–noun composition request with optional sourced learning episodes."""

    sequence_id: str
    text: str | None = None
    modality_inputs: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    context_features: Mapping[str, float] = field(default_factory=dict)
    noun: str | None = None
    adjective: str | None = None
    noun_features: Mapping[str, float] = field(default_factory=dict)
    adjective_features: Mapping[str, float] = field(default_factory=dict)
    adjective_type: str = "subsective"
    learning_experiences: Sequence[Mapping[str, Any]] = ()
    learning_authorized: bool = False
    source_reference: str = "unspecified"

    def __post_init__(self) -> None:
        """Validate identity, optional phrase structure, modality evidence, and the denotational modifier class."""
        if not self.sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")
        if self.text is not None and not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.noun is not None and not self.noun.strip():
            raise ValueError("noun cannot be empty")
        if self.adjective is not None and not self.adjective.strip():
            raise ValueError("adjective cannot be empty")
        if self.text is None and self.noun is None and not self.modality_inputs:
            raise ValueError("Provide text, a noun, or at least one modality input")
        if self.adjective is not None and self.noun is None and self.text is None:
            raise ValueError("An adjective requires a noun or phrase text")
        normalize_modalities(self.modality_inputs)
        validate_feature_map("context", self.context_features)
        validate_feature_map("noun", self.noun_features)
        validate_feature_map("adjective", self.adjective_features)
        if self.adjective_type not in {"subsective", "privative"}:
            raise ValueError("adjective_type must be 'subsective' or 'privative'")
        if not self.source_reference.strip():
            raise ValueError("source_reference must be non-empty")


@dataclass(slots=True)
class AmodalATLResult:
    """Complete functional, biological, learning, and diagnostic record for one 19-stage ATL pass."""

    sequence_id: str
    output: Mapping[str, Any]
    stage_outputs: Mapping[str, Mapping[str, Any]]
    stage_trace: Sequence[Mapping[str, Any]]
    operation_trace: Sequence[Mapping[str, Any]]
    biological: Mapping[str, Any]
    learning: Mapping[str, Any]
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation suitable for SQLite metadata, fixtures, and downstream ORE modules."""
        return _jsonable({
            "schema": "ore.atl_amodal.19_stage.v1",
            "sequence_id": self.sequence_id,
            "output": self.output,
            "stage_outputs": self.stage_outputs,
            "stage_trace": self.stage_trace,
            "operation_trace": self.operation_trace,
            "biological": self.biological,
            "learning": self.learning,
            "diagnostics": self.diagnostics,
        })
