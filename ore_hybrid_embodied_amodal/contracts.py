"""Typed inputs, configuration, and outputs for hybrid embodied–amodal concept processing."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


MODALITIES = (
    "visual", "auditory", "motor", "somatosensory", "olfactory", "gustatory",
    "emotional", "interoceptive", "linguistic", "internal",
)


def _validate_feature_map(name: str, values: Mapping[str, float]) -> None:
    """Require finite, named feature activations in the documented [0, 1] range."""
    for key, raw in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} feature names must be non-empty strings")
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} values must be finite and lie in [0, 1]")


def _jsonable(value: Any) -> Any:
    """Convert nested result values and NumPy scalars into JSON-compatible primitives."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(frozen=True, slots=True)
class HybridConcept:
    """A provenance-tracked concept prototype with shared and modality-specific feature evidence."""

    concept_id: str
    label: str
    features: Mapping[str, float] = field(default_factory=dict)
    modality_features: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    category: str = "unknown"
    aliases: tuple[str, ...] = ()
    frequency: float = 1.0
    verified_by: str = "unspecified"
    source_reference: str = "unspecified"

    def __post_init__(self) -> None:
        """Validate concept identity, frequency, and all shared and embodied feature maps."""
        if not self.concept_id.strip() or not self.label.strip():
            raise ValueError("concept_id and label must be non-empty")
        if not math.isfinite(float(self.frequency)) or self.frequency <= 0:
            raise ValueError("frequency must be finite and positive")
        if any(not isinstance(alias, str) or not alias.strip() for alias in self.aliases):
            raise ValueError("aliases must contain non-empty strings")
        _validate_feature_map("concept", self.features)
        for modality, feature_map in self.modality_features.items():
            if not isinstance(modality, str) or not modality.strip():
                raise ValueError("modality names must be non-empty")
            _validate_feature_map(f"concept {modality}", feature_map)


@dataclass(frozen=True, slots=True)
class HybridLearningExample:
    """One new cross-modal experience with explicit reviewer and source provenance."""

    concept_id: str
    modality_inputs: Mapping[str, Mapping[str, float]]
    shared_features: Mapping[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    verified_by: str = "unspecified"
    source_reference: str = "unspecified"

    def __post_init__(self) -> None:
        """Reject unsupported learning values and untraceable modality examples."""
        if not self.concept_id.strip():
            raise ValueError("learning concept_id must be non-empty")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("learning confidence must lie in [0, 1]")
        if not self.modality_inputs:
            raise ValueError("a learning example must contain at least one modality")
        for modality, features in self.modality_inputs.items():
            if not modality.strip():
                raise ValueError("learning modality names must be non-empty")
            _validate_feature_map(f"learning {modality}", features)
        _validate_feature_map("learning shared", self.shared_features)


@dataclass(frozen=True, slots=True)
class HybridRequest:
    """A single concept-processing trial with multimodal, temporal, spatial, and task evidence."""

    sequence_id: str
    modality_inputs: Mapping[str, Mapping[str, float]]
    task: str = "comprehension"
    context_features: Mapping[str, float] = field(default_factory=dict)
    task_features: Mapping[str, float] = field(default_factory=dict)
    irrelevant_features: Mapping[str, float] = field(default_factory=dict)
    modality_uncertainty: Mapping[str, float] = field(default_factory=dict)
    modality_times: Mapping[str, float] = field(default_factory=dict)
    spatial_features: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    binding_priors: Mapping[str, float] = field(default_factory=dict)
    task_modality_weights: Mapping[str, float] = field(default_factory=dict)
    target_features: Mapping[str, float] = field(default_factory=dict)
    expected_concept_id: str | None = None
    response_mode: str = "recognition"
    learning_examples: Sequence[HybridLearningExample] = ()
    learning_authorized: bool = False
    source_reference: str = "unspecified"

    def __post_init__(self) -> None:
        """Validate request identity, evidence bounds, time values, and explicit learning records."""
        if not self.sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")
        if not self.modality_inputs:
            raise ValueError("provide at least one modality input")
        if not self.task.strip() or not self.response_mode.strip():
            raise ValueError("task and response_mode must be non-empty")
        for modality, features in self.modality_inputs.items():
            if not isinstance(modality, str) or not modality.strip():
                raise ValueError("modality names must be non-empty strings")
            _validate_feature_map(f"input {modality}", features)
        for name, values in (
            ("context", self.context_features),
            ("task", self.task_features),
            ("irrelevant", self.irrelevant_features),
            ("target", self.target_features),
        ):
            _validate_feature_map(name, values)
        for modality, value in self.modality_uncertainty.items():
            if modality not in self.modality_inputs:
                raise ValueError(f"uncertainty supplied for absent modality: {modality}")
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError("modality uncertainty must be finite and non-negative")
        for modality, value in self.modality_times.items():
            if modality not in self.modality_inputs:
                raise ValueError(f"time supplied for absent modality: {modality}")
            if not math.isfinite(float(value)):
                raise ValueError("modality times must be finite")
        for modality, features in self.spatial_features.items():
            if modality not in self.modality_inputs:
                raise ValueError(f"spatial data supplied for absent modality: {modality}")
            _validate_feature_map(f"spatial {modality}", features)
        for key, value in self.binding_priors.items():
            if not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"binding prior {key!r} must lie in [0, 1]")
        for modality, value in self.task_modality_weights.items():
            if modality not in self.modality_inputs:
                raise ValueError(f"task gate supplied for absent modality: {modality}")
            if not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
                raise ValueError("task modality weights must lie in [0, 1]")
        if self.expected_concept_id is not None and not self.expected_concept_id.strip():
            raise ValueError("expected_concept_id cannot be empty")
        if any(not isinstance(item, HybridLearningExample) for item in self.learning_examples):
            raise TypeError("learning_examples must contain HybridLearningExample values")


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """Numerical and LIF-reference settings; parameters are hypotheses, not measured constants."""

    hub_dimensions: int = 32
    max_recurrent_steps: int = 8
    max_feedback_iterations: int = 2
    convergence_tolerance: float = 1e-5
    temporal_tau: float = 0.35
    temporal_weight: float = 0.45
    spatial_weight: float = 0.25
    learned_binding_weight: float = 0.30
    attention_gain: float = 2.0
    inhibition_strength: float = 0.35
    common_weight: float = 0.65
    embodied_weight: float = 0.35
    prediction_update_rate: float = 0.15
    learning_rate: float = 0.02
    weight_decay: float = 0.001
    confidence_threshold: float = 0.45
    selection_margin: float = 0.03
    simulate_biology: bool = True
    random_seed: int = 41
    dt_ms: float = 1.0
    steps_per_stage: int = 2
    population_size: int = 8
    membrane_tau_ms: float = 20.0
    refractory_ms: float = 2.0
    membrane_resistance_mohm: float = 100.0
    rest_mv: float = -70.0
    reset_mv: float = -68.0
    threshold_mv: float = -50.0
    current_gain_pa: float = 320.0
    recurrent_current_pa: float = 18.0
    inhibitory_current_pa: float = 22.0

    def __post_init__(self) -> None:
        """Validate dimensions, recurrence bounds, interpolation weights, and LIF units."""
        if self.hub_dimensions < 2 or self.max_recurrent_steps < 1:
            raise ValueError("hub_dimensions must be >= 2 and max_recurrent_steps positive")
        if self.max_feedback_iterations < 0 or self.steps_per_stage < 1 or self.population_size < 2:
            raise ValueError("feedback, stage-step, or population count is invalid")
        unit_interval = (
            "temporal_weight", "spatial_weight", "learned_binding_weight",
            "inhibition_strength", "common_weight", "embodied_weight",
            "prediction_update_rate", "learning_rate", "weight_decay",
            "confidence_threshold", "selection_margin",
        )
        for name in unit_interval:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        positive = (
            "convergence_tolerance", "temporal_tau", "attention_gain",
            "dt_ms", "membrane_tau_ms", "membrane_resistance_mohm", "current_gain_pa",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("refractory_ms", "recurrent_current_pa", "inhibitory_current_pa"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.rest_mv < self.threshold_mv:
            raise ValueError("rest_mv must be below threshold_mv")
        if not self.rest_mv <= self.reset_mv < self.threshold_mv:
            raise ValueError("require rest_mv <= reset_mv < threshold_mv")


@dataclass(slots=True)
class HybridResult:
    """Functional, embodied, amodal, biological, learning, trace, and validation outputs."""

    sequence_id: str
    selected_concept: dict[str, Any] | None
    candidate_scores: list[dict[str, Any]]
    common_representation: list[float]
    modality_representations: dict[str, list[float]]
    modality_residuals: dict[str, list[float]]
    reactivated_representations: dict[str, list[float]]
    stage_outputs: dict[str, dict[str, Any]]
    stage_trace: list[dict[str, Any]]
    operation_trace: list[dict[str, Any]]
    biological: dict[str, Any]
    learning: dict[str, Any]
    response: dict[str, Any]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result contract without NumPy objects or non-finite JSON numbers."""
        return _jsonable({
            "schema": "ore.hybrid_embodied_amodal.19_stage.v1",
            "sequence_id": self.sequence_id,
            "selected_concept": self.selected_concept,
            "candidate_scores": self.candidate_scores,
            "common_representation": self.common_representation,
            "modality_representations": self.modality_representations,
            "modality_residuals": self.modality_residuals,
            "reactivated_representations": self.reactivated_representations,
            "stage_outputs": self.stage_outputs,
            "stage_trace": self.stage_trace,
            "operation_trace": self.operation_trace,
            "biological": self.biological,
            "learning": self.learning,
            "response": self.response,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        })
    