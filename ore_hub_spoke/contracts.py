"""Typed input/output contracts for the 21-stage Hub-and-Spoke model.

The contracts distinguish observations, retrieved knowledge, and learned updates.
The model is a computational neuroscience-inspired approximation, not a neural
recording or a claim that neurons literally execute its numerical equations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

from ore_semantic.contracts import SemanticConcept, WordFormEpisode

MODALITIES = (
    "visual",
    "auditory",
    "tactile",
    "olfactory_gustatory",
    "spatial_motor",
    "linguistic",
    "emotional_interoceptive",
)

_MODALITY_ALIASES = {
    "visual": "visual",
    "vision": "visual",
    "auditory": "auditory",
    "audio": "auditory",
    "phonological": "auditory",
    "sound": "auditory",
    "tactile": "tactile",
    "somatosensory": "tactile",
    "touch": "tactile",
    "olfactory": "olfactory_gustatory",
    "gustatory": "olfactory_gustatory",
    "smell": "olfactory_gustatory",
    "taste": "olfactory_gustatory",
    "olfactory_gustatory": "olfactory_gustatory",
    "spatial": "spatial_motor",
    "motor": "spatial_motor",
    "action": "spatial_motor",
    "spatial_motor": "spatial_motor",
    "linguistic": "linguistic",
    "language": "linguistic",
    "verbal": "linguistic",
    "emotional": "emotional_interoceptive",
    "affective": "emotional_interoceptive",
    "interoceptive": "emotional_interoceptive",
    "emotional_interoceptive": "emotional_interoceptive",
}


def canonical_modality(name: str) -> str:
    """Map a modality alias to one of the seven canonical semantic spokes."""
    normalized = str(name).strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized not in _MODALITY_ALIASES:
        raise ValueError(f"unsupported semantic modality: {name!r}")
    return _MODALITY_ALIASES[normalized]


def validate_feature_map(name: str, values: Mapping[str, float]) -> None:
    """Validate named graded feature activations in the closed interval [0, 1]."""
    for feature, raw in values.items():
        if not isinstance(feature, str) or not feature.strip():
            raise ValueError(f"{name} feature names must be non-empty strings")
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} feature values must be finite and lie in [0, 1]")


def _normalize_modalities(
    inputs: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Merge modality aliases without losing graded feature values."""
    normalized: dict[str, dict[str, float]] = {}
    for modality, features in inputs.items():
        key = canonical_modality(modality)
        validate_feature_map(f"{key} input", features)
        target = normalized.setdefault(key, {})
        for feature, value in features.items():
            target[str(feature)] = max(target.get(str(feature), 0.0), float(value))
    return normalized


def _jsonable(value: Any) -> Any:
    """Convert result values to JSON-compatible Python primitives."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return repr(value)


@dataclass(frozen=True, slots=True)
class LearningExperience:
    """One sourced episode linking a known concept to one or more modality features."""

    concept_id: str
    modality_features: Mapping[str, Mapping[str, float]]
    semantic_features: Mapping[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    verified_by: str = "unspecified"
    source_reference: str = "unspecified"
    episode_id: str = ""

    def __post_init__(self) -> None:
        """Reject ungrounded identifiers, invalid feature values, and invalid confidence."""
        if not self.concept_id.strip():
            raise ValueError("concept_id must be non-empty")
        if not self.modality_features and not self.semantic_features:
            raise ValueError("an experience must contain semantic or modality feature evidence")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and lie in [0, 1]")
        _normalize_modalities(self.modality_features)
        validate_feature_map("experience semantic", self.semantic_features)
        if not self.source_reference.strip():
            raise ValueError("source_reference must be non-empty")


@dataclass(frozen=True, slots=True)
class HubAndSpokeRequest:
    """A single concept-access request with optional simultaneous multimodal evidence."""

    sequence_id: str
    text: str | None = None
    tokens: Sequence[str] | None = None
    modality_inputs: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    context_features: Mapping[str, float] = field(default_factory=dict)
    task: str = "recognize"
    task_relevance: Mapping[str, float] = field(default_factory=dict)
    irrelevant_features: Mapping[str, float] = field(default_factory=dict)
    episodes: Sequence[WordFormEpisode] = ()
    source_reference: str = "unspecified"

    def __post_init__(self) -> None:
        """Validate the request while preserving original text and caller token boundaries."""
        if not self.sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")
        if self.text is not None and self.tokens is not None:
            raise ValueError("provide text or tokens, not both")
        if self.text is not None and not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.tokens is not None and (
            not self.tokens
            or any(not isinstance(token, str) or not token.strip() for token in self.tokens)
        ):
            raise ValueError("tokens must be a non-empty sequence of non-empty strings")
        if self.task.strip() == "":
            raise ValueError("task must be non-empty")
        if self.text is None and self.tokens is None and not self.modality_inputs:
            raise ValueError("provide at least one word-form or modality input")
        _normalize_modalities(self.modality_inputs)
        validate_feature_map("context", self.context_features)
        validate_feature_map("task relevance", self.task_relevance)
        validate_feature_map("irrelevance", self.irrelevant_features)
        if any(not isinstance(item, WordFormEpisode) for item in self.episodes):
            raise TypeError("episodes must contain WordFormEpisode records")

    @property
    def concept_cue(self) -> str:
        """Return the preserved cue string used for lexical candidate access."""
        if self.text is not None:
            return self.text.strip()
        return " ".join(str(token) for token in (self.tokens or ()))


@dataclass(frozen=True, slots=True)
class HubAndSpokeConfig:
    """Numerical controls for functional processing, plasticity, lesions, and LIF dynamics."""

    hidden_dim: int = 32
    max_candidates: int = 10
    candidate_similarity_floor: float = 0.20
    feature_threshold: float = 0.35
    modality_temperature: float = 0.75
    feedback_gain: float = 0.25
    recurrent_gain: float = 0.45
    recurrent_steps: int = 12
    convergence_epsilon: float = 1e-5
    confidence_threshold: float = 0.48
    ambiguity_margin: float = 0.05
    learning_rate: float = 0.04
    weight_decay: float = 0.002
    lesion_severity: float = 0.35
    simulate_biology: bool = True
    random_seed: int = 17
    dt_ms: float = 1.0
    steps_per_stage: int = 2
    membrane_tau_ms: float = 20.0
    refractory_ms: float = 2.0
    population_size: int = 12
    membrane_resistance_mohm: float = 100.0
    rest_mv: float = -70.0
    reset_mv: float = -68.0
    threshold_mv: float = -50.0
    current_gain_pa: float = 320.0
    recurrent_current_pa: float = 18.0
    inhibitory_current_pa: float = 22.0

    def __post_init__(self) -> None:
        """Enforce finite bounded parameters before numerical state is allocated."""
        if self.hidden_dim < 4 or self.max_candidates < 1:
            raise ValueError("hidden_dim must be >= 4 and max_candidates must be positive")
        if self.recurrent_steps < 1 or self.steps_per_stage < 1 or self.population_size < 2:
            raise ValueError("iteration and population counts must be positive")
        bounded = (
            "candidate_similarity_floor", "feature_threshold", "feedback_gain",
            "recurrent_gain", "confidence_threshold", "ambiguity_margin",
            "learning_rate", "weight_decay", "lesion_severity",
        )
        for name in bounded:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        if not math.isfinite(self.modality_temperature) or self.modality_temperature <= 0:
            raise ValueError("modality_temperature must be finite and positive")
        if not math.isfinite(self.convergence_epsilon) or self.convergence_epsilon <= 0:
            raise ValueError("convergence_epsilon must be finite and positive")
        positive = (
            "dt_ms", "membrane_tau_ms", "membrane_resistance_mohm", "current_gain_pa",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        nonnegative = ("refractory_ms", "recurrent_current_pa", "inhibitory_current_pa")
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.rest_mv < self.threshold_mv:
            raise ValueError("rest_mv must be below threshold_mv")
        if not self.rest_mv <= self.reset_mv < self.threshold_mv:
            raise ValueError("require rest_mv <= reset_mv < threshold_mv")


@dataclass(slots=True)
class HubAndSpokeResult:
    """Complete 21-stage functional trace, optional biological trace, and diagnostics."""

    sequence_id: str
    original_input: str
    output: dict[str, Any]
    stage_outputs: dict[str, Any]
    stage_trace: list[dict[str, Any]]
    operation_trace: list[dict[str, Any]]
    biological: dict[str, Any]
    learning: dict[str, Any]
    lesions: dict[str, Any]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize all retained intermediate evidence and model-scope statements."""
        return _jsonable({
            "schema": "ore.hub_spoke.21_stage.v1",
            "sequence_id": self.sequence_id,
            "original_input": self.original_input,
            "output": self.output,
            "stage_outputs": self.stage_outputs,
            "stage_trace": self.stage_trace,
            "operation_trace": self.operation_trace,
            "biological": self.biological,
            "learning": self.learning,
            "lesions": self.lesions,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        })
