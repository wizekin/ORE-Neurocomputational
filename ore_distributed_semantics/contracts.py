"""Typed contracts for the 36-stage distributed semantic architecture.

The contracts keep observed inputs, retrieved memories, model hypotheses, and
authorized learning updates distinguishable. Equations are computational
approximations; they are not claims that neurons execute software operations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


MODALITIES = (
    "visual", "auditory", "tactile", "olfactory_gustatory",
    "spatial_motor", "linguistic", "emotional_interoceptive",
)

_MODALITY_ALIASES = {
    "visual": "visual", "vision": "visual", "image": "visual",
    "auditory": "auditory", "audio": "auditory", "sound": "auditory",
    "phonological": "auditory", "tactile": "tactile", "touch": "tactile",
    "somatosensory": "tactile", "olfactory": "olfactory_gustatory",
    "gustatory": "olfactory_gustatory", "smell": "olfactory_gustatory",
    "taste": "olfactory_gustatory", "olfactory_gustatory": "olfactory_gustatory",
    "spatial": "spatial_motor", "motor": "spatial_motor",
    "action": "spatial_motor", "spatial_motor": "spatial_motor",
    "linguistic": "linguistic", "language": "linguistic",
    "verbal": "linguistic", "emotional": "emotional_interoceptive",
    "affective": "emotional_interoceptive",
    "interoceptive": "emotional_interoceptive",
    "emotional_interoceptive": "emotional_interoceptive",
}


def canonical_modality(name: str) -> str:
    """Normalize a modality label or alias to one of the seven supported spokes."""
    normalized = str(name).strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized not in _MODALITY_ALIASES:
        raise ValueError(f"Unsupported semantic modality: {name!r}")
    return _MODALITY_ALIASES[normalized]


def validate_feature_map(name: str, values: Mapping[str, float]) -> None:
    """Validate named graded feature activations as finite values between zero and one."""
    for feature, raw_value in values.items():
        if not isinstance(feature, str) or not feature.strip():
            raise ValueError(f"{name} feature names must be non-empty strings")
        value = float(raw_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}.{feature} must be finite and lie in [0, 1]")


def normalize_modalities(
    values: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Merge modality aliases while retaining the strongest supplied feature evidence."""
    result: dict[str, dict[str, float]] = {}
    for raw_name, features in values.items():
        modality = canonical_modality(raw_name)
        validate_feature_map(f"{modality} input", features)
        target = result.setdefault(modality, {})
        for feature, strength in features.items():
            target[str(feature)] = max(target.get(str(feature), 0.0), float(strength))
    return result


def json_safe(value: Any) -> Any:
    """Convert dataclasses, NumPy-like objects, and nested state into JSON-safe values."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(json_safe(item) for item in value)
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(frozen=True, slots=True)
class DistributedSemanticConfig:
    """Configure functional thresholds and the separate biological-reference simulation."""

    simulate_biology: bool = True
    random_seed: int = 23
    max_candidates: int = 8
    confidence_threshold: float = 0.35
    margin_threshold: float = 0.04
    association_threshold: float = 0.18
    learning_rate: float = 0.08
    recurrent_steps: int = 3
    steps_per_stage: int = 2
    population_size: int = 8
    dt_ms: float = 0.5
    membrane_tau_ms: float = 20.0
    membrane_resistance_mohm: float = 100.0
    current_gain_pa: float = 260.0
    recurrent_gain_pa: float = 18.0
    inhibitory_gain_pa: float = 14.0
    threshold_mv: float = -50.0
    rest_mv: float = -65.0
    reset_mv: float = -68.0
    refractory_ms: float = 2.0

    def __post_init__(self) -> None:
        """Reject invalid simulation dimensions, thresholds, and non-finite parameters."""
        if self.max_candidates < 1 or self.population_size < 1:
            raise ValueError("max_candidates and population_size must be positive")
        if self.steps_per_stage < 1 or self.recurrent_steps < 0:
            raise ValueError("simulation step counts are invalid")
        if self.dt_ms <= 0 or self.membrane_tau_ms <= 0:
            raise ValueError("time constants must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must lie in [0, 1]")
        if not 0.0 <= self.margin_threshold <= 1.0:
            raise ValueError("margin_threshold must lie in [0, 1]")
        if not 0.0 <= self.association_threshold <= 1.0:
            raise ValueError("association_threshold must lie in [0, 1]")
        if not 0.0 <= self.learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in [0, 1]")
        numeric = (
            self.dt_ms, self.membrane_tau_ms, self.membrane_resistance_mohm,
            self.current_gain_pa, self.recurrent_gain_pa, self.inhibitory_gain_pa,
            self.threshold_mv, self.rest_mv, self.reset_mv, self.refractory_ms,
        )
        if not all(math.isfinite(float(item)) for item in numeric):
            raise ValueError("simulation parameters must be finite")


@dataclass(frozen=True, slots=True)
class DistributedSemanticRequest:
    """Describe one semantic-processing episode and keep externally sourced evidence explicit."""

    sequence_id: str
    text: str = ""
    tokens: Sequence[str] = ()
    modality_inputs: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    context_features: Mapping[str, float] = field(default_factory=dict)
    task: str = "comprehension"
    language: str = "en"
    multilingual_lexicon: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    syntactic_relations: Sequence[Mapping[str, str]] = ()
    event_roles: Mapping[str, str] = field(default_factory=dict)
    associations: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    episodic_memory: Sequence[Mapping[str, Any]] = ()
    schemas: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    discourse_history: Sequence[Mapping[str, Any]] = ()
    education_inputs: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    learning_feedback: Mapping[str, float] = field(default_factory=dict)
    allow_learning: bool = False
    developmental_experience_count: int = 0
    aging_factor: float = 0.0
    intervention: Mapping[str, Any] = field(default_factory=dict)
    evaluation_targets: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate all caller-supplied evidence and reject implicit or malformed updates."""
        if not self.sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")
        if not self.language.strip():
            raise ValueError("language must be non-empty")
        if self.developmental_experience_count < 0:
            raise ValueError("developmental_experience_count cannot be negative")
        if not math.isfinite(float(self.aging_factor)) or not 0.0 <= self.aging_factor <= 1.0:
            raise ValueError("aging_factor must be finite and lie in [0, 1]")
        normalize_modalities(self.modality_inputs)
        normalize_modalities(self.education_inputs)
        validate_feature_map("context_features", self.context_features)
        validate_feature_map("learning_feedback", self.learning_feedback)
        for source, targets in self.associations.items():
            if not str(source).strip():
                raise ValueError("association source identifiers must be non-empty")
            for target, strength in targets.items():
                value = float(strength)
                if not str(target).strip() or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("association targets and strengths are invalid")
        for item in self.syntactic_relations:
            if not isinstance(item, Mapping) or not {"head", "dependent", "relation"} <= set(item):
                raise ValueError("syntactic_relations require head, dependent, and relation")
        for key, value in self.event_roles.items():
            if not str(key).strip() or not str(value).strip():
                raise ValueError("event_roles must contain non-empty keys and values")
        for language, mapping in self.multilingual_lexicon.items():
            if not str(language).strip() or not isinstance(mapping, Mapping):
                raise ValueError("multilingual_lexicon must map language codes to mappings")


@dataclass(frozen=True, slots=True)
class DistributedSemanticResult:
    """Return functional meanings, per-stage traces, LIF reference activity, and diagnostics."""

    sequence_id: str
    output: Mapping[str, Any]
    stage_outputs: Mapping[str, Mapping[str, Any]]
    stage_trace: Sequence[Mapping[str, Any]]
    operation_trace: Sequence[Mapping[str, Any]]
    biological_trace: Sequence[Mapping[str, Any]]
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete result without leaking non-JSON numerical objects."""
        return json_safe({
            "schema": "ore.distributed_semantics.36_stage.v1",
            "sequence_id": self.sequence_id,
            "output": self.output,
            "stage_outputs": self.stage_outputs,
            "stage_trace": self.stage_trace,
            "operation_trace": self.operation_trace,
            "biological_trace": self.biological_trace,
            "diagnostics": self.diagnostics,
        })
