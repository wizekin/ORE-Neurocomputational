"""Typed request, configuration, and result contracts for Controlled Semantic Cognition."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


def _validate_feature_map(name: str, values: Mapping[str, float]) -> None:
    """Require named, finite feature activations bounded to the unit interval."""
    for key, raw in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("{0} feature keys must be non-empty strings".format(name))
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("{0} values must be finite and lie in [0, 1]".format(name))


def _jsonable(value: Any) -> Any:
    """Convert nested mappings, sequences, and numerical values to JSON-safe primitives."""
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
class CSCRequest:
    """One semantic-control trial with caller-supplied task and contextual evidence."""

    sequence_id: str
    text: str | None = None
    tokens: Sequence[str] | None = None
    task: str = "comprehension"
    context_features: Mapping[str, float] = field(default_factory=dict)
    task_features: Mapping[str, float] = field(default_factory=dict)
    modality_inputs: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    signal_samples: Sequence[float] = ()
    signal_modality: str = "auditory"
    previous_context_features: Mapping[str, float] = field(default_factory=dict)
    previous_task_features: Mapping[str, float] = field(default_factory=dict)
    response_options: Sequence[str] = ()
    required_fit: float = 0.65
    feedback_concept_id: str | None = None
    source_reference: str = "unspecified"

    def __post_init__(self) -> None:
        """Validate request identity, mutually exclusive text/tokens, and evidence maps."""
        if not self.sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")
        if (self.text is None) == (self.tokens is None):
            raise ValueError("provide exactly one of text or tokens")
        if self.text is not None and not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.tokens is not None and (
            not self.tokens
            or any(not isinstance(token, str) or not token.strip() for token in self.tokens)
        ):
            raise ValueError("tokens must be a non-empty sequence of non-empty strings")
        if not self.task.strip():
            raise ValueError("task must be non-empty")
        if not math.isfinite(float(self.required_fit)) or not 0.0 <= self.required_fit <= 1.0:
            raise ValueError("required_fit must be finite and lie in [0, 1]")
        if any(not isinstance(option, str) or not option.strip() for option in self.response_options):
            raise ValueError("response_options must contain non-empty strings")
        if self.feedback_concept_id is not None and not self.feedback_concept_id.strip():
            raise ValueError("feedback_concept_id cannot be empty")
        if not self.signal_modality.strip():
            raise ValueError("signal_modality must be non-empty")
        if any(not math.isfinite(float(sample)) for sample in self.signal_samples):
            raise ValueError("signal_samples must contain finite numeric values")
        _validate_feature_map("context", self.context_features)
        _validate_feature_map("task", self.task_features)
        _validate_feature_map("previous context", self.previous_context_features)
        _validate_feature_map("previous task", self.previous_task_features)
        for modality, features in self.modality_inputs.items():
            if not isinstance(modality, str) or not modality.strip():
                raise ValueError("modality names must be non-empty")
            _validate_feature_map("modality " + modality, features)

    @property
    def input_text(self) -> str:
        """Return original text or join caller-preserved tokens for lexical processing."""
        if self.text is not None:
            return self.text
        return " ".join(self.tokens or ())


@dataclass(frozen=True, slots=True)
class CSCConfig:
    """Numerical settings for functional control, feedback, and optional LIF reference dynamics."""

    max_control_iterations: int = 3
    convergence_epsilon: float = 1e-5
    softmax_temperature: float = 0.28
    confidence_threshold: float = 0.42
    selection_margin_threshold: float = 0.035
    feature_threshold: float = 0.30
    context_ema_alpha: float = 0.65
    task_ema_alpha: float = 0.70
    gain_strength: float = 0.65
    inhibition_strength: float = 0.45
    recurrence_gain: float = 0.30
    activity_decay: float = 0.08
    learning_rate: float = 0.04
    simulate_biology: bool = True
    random_seed: int = 23
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
        """Reject invalid iteration counts, thresholds, time constants, and LIF parameters."""
        if self.max_control_iterations < 1 or self.steps_per_stage < 1 or self.population_size < 2:
            raise ValueError("iteration and population counts must be positive")
        for name in (
            "confidence_threshold", "selection_margin_threshold", "feature_threshold",
            "context_ema_alpha", "task_ema_alpha", "gain_strength", "inhibition_strength",
            "recurrence_gain", "activity_decay", "learning_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("{0} must be finite and lie in [0, 1]".format(name))
        positive = (
            "convergence_epsilon", "softmax_temperature", "dt_ms", "membrane_tau_ms",
            "membrane_resistance_mohm", "current_gain_pa",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("{0} must be finite and positive".format(name))
        for name in ("refractory_ms", "recurrent_current_pa", "inhibitory_current_pa"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("{0} must be finite and non-negative".format(name))
        if not self.rest_mv < self.threshold_mv:
            raise ValueError("rest_mv must be below threshold_mv")
        if not self.rest_mv <= self.reset_mv < self.threshold_mv:
            raise ValueError("require rest_mv <= reset_mv < threshold_mv")


@dataclass(slots=True)
class CSCResult:
    """Functional, biological-reference, selection, trace, learning, and diagnostic outputs."""

    sequence_id: str
    original_text: str
    tokens: list[str]
    selected_concept: dict[str, Any] | None
    candidate_scores: list[dict[str, Any]]
    semantic_state: dict[str, float]
    controlled_state: dict[str, float]
    integrated_state: dict[str, float]
    response: dict[str, Any]
    stage_outputs: dict[str, dict[str, Any]]
    stage_trace: list[dict[str, Any]]
    operation_trace: list[dict[str, Any]]
    biological: dict[str, Any]
    learning: dict[str, Any]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete result without exposing arrays or non-finite numbers."""
        return _jsonable({
            "schema": "ore.csc.18_stage.v1",
            "sequence_id": self.sequence_id,
            "original_text": self.original_text,
            "tokens": self.tokens,
            "selected_concept": self.selected_concept,
            "candidate_scores": self.candidate_scores,
            "semantic_state": self.semantic_state,
            "controlled_state": self.controlled_state,
            "integrated_state": self.integrated_state,
            "response": self.response,
            "stage_outputs": self.stage_outputs,
            "stage_trace": self.stage_trace,
            "operation_trace": self.operation_trace,
            "biological": self.biological,
            "learning": self.learning,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        })
