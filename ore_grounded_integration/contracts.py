"""Typed contracts for the 22-stage grounded semantic integration pipeline.

The contracts separate observations, derived representations, model state and learner evidence.
They do not imply that the software equations are literal descriptions of neural computation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

MODALITIES = ("linguistic", "visual", "auditory", "motor", "context")

_MODALITY_ALIASES = {
    "language": "linguistic",
    "linguistic": "linguistic",
    "text": "linguistic",
    "verbal": "linguistic",
    "vision": "visual",
    "visual": "visual",
    "image": "visual",
    "auditory": "auditory",
    "audio": "auditory",
    "sound": "auditory",
    "motor": "motor",
    "action": "motor",
    "movement": "motor",
    "context": "context",
    "situational": "context",
}


def canonical_modality(name: str) -> str:
    """Resolve a supported modality alias to one of the five canonical channels."""
    key = str(name).strip().casefold().replace("-", "_").replace(" ", "_")
    if key not in _MODALITY_ALIASES:
        raise ValueError(f"unsupported modality: {name!r}")
    return _MODALITY_ALIASES[key]


def _finite_mapping(name: str, values: Mapping[str, float], lower: float | None = None) -> None:
    """Validate a named scalar feature map and optionally enforce a lower bound."""
    for key, raw in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings")
        number = float(raw)
        if not math.isfinite(number) or (lower is not None and number < lower):
            raise ValueError(f"{name}[{key!r}] must be finite and >= {lower}")


def _canonicalize_modalities(values: Mapping[str, Any]) -> dict[str, Any]:
    """Merge modality aliases while preserving each caller-supplied observation."""
    normalized: dict[str, Any] = {}
    for name, value in values.items():
        key = canonical_modality(name)
        if key in normalized:
            raise ValueError(f"modality {key!r} was supplied more than once through aliases")
        normalized[key] = value
    return normalized


def _jsonable(value: Any) -> Any:
    """Convert nested results and NumPy values into JSON-compatible Python primitives."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(frozen=True, slots=True)
class GroundedSemanticConfig:
    """Numerical controls for representation size, integration, learning and neural reference dynamics.

    The defaults are deliberately small and deterministic so the model can run locally without
    pretrained encoders. Input images and waveforms should be converted to feature arrays by
    caller-supplied encoders when raw modality processing is required.
    """

    hidden_dim: int = 48
    temporal_window: float = 2.0
    temporal_tau: float = 0.75
    softmax_temperature: float = 0.8
    alignment_margin: float = 0.25
    attention_temperature: float = 0.75
    learning_rate: float = 0.08
    weight_decay: float = 0.002
    memory_decay: float = 0.01
    max_episodes: int = 256
    retrieval_limit: int = 5
    graph_steps: int = 2
    confidence_threshold: float = 0.45
    simulate_biology: bool = True
    random_seed: int = 17
    population_size: int = 12
    dt_ms: float = 1.0
    membrane_tau_ms: float = 20.0
    membrane_resistance_mohm: float = 100.0
    rest_mv: float = -70.0
    reset_mv: float = -68.0
    threshold_mv: float = -50.0
    current_gain_pa: float = 260.0
    recurrent_gain_pa: float = 16.0
    inhibitory_gain_pa: float = 10.0
    refractory_ms: float = 2.0

    def __post_init__(self) -> None:
        """Reject invalid dimensional, probabilistic, learning and LIF parameters before execution."""
        if self.hidden_dim < 8 or self.population_size < 2:
            raise ValueError("hidden_dim must be >= 8 and population_size must be >= 2")
        if self.max_episodes < 1 or self.retrieval_limit < 1 or self.graph_steps < 1:
            raise ValueError("episode, retrieval and graph limits must be positive")
        positive = (
            "temporal_window", "temporal_tau", "softmax_temperature",
            "attention_temperature", "dt_ms", "membrane_tau_ms",
            "membrane_resistance_mohm", "current_gain_pa",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        bounded = (
            "alignment_margin", "learning_rate", "weight_decay", "memory_decay",
            "confidence_threshold",
        )
        for name in bounded:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("recurrent_gain_pa", "inhibitory_gain_pa", "refractory_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.rest_mv < self.threshold_mv:
            raise ValueError("rest_mv must be below threshold_mv")
        if not self.rest_mv <= self.reset_mv < self.threshold_mv:
            raise ValueError("require rest_mv <= reset_mv < threshold_mv")


@dataclass(frozen=True, slots=True)
class GroundedSemanticRequest:
    """One text-led or multimodal semantic request with optional context, feedback and learner evidence.

    Modality values may be named feature maps, numeric vectors or arrays. Raw image/audio inputs
    need explicit encoders supplied to the processor. Candidate senses map a sense label to
    weighted context features; learner scores are optional values in [0, 1].
    """

    sample_id: str
    text: str | None = None
    tokens: Sequence[str] | None = None
    modality_inputs: Mapping[str, Any] = field(default_factory=dict)
    modality_timestamps: Mapping[str, float] = field(default_factory=dict)
    modality_confidence: Mapping[str, float] = field(default_factory=dict)
    context: Mapping[str, float] = field(default_factory=dict)
    task: str = "understand"
    candidate_senses: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    feedback: Mapping[str, float] = field(default_factory=dict)
    learner_response: Mapping[str, float] | None = None
    usage_examples: Sequence[str] = ()
    source_reference: str = "unspecified"

    def __post_init__(self) -> None:
        """Validate identifiers, modality inputs, candidate evidence and learner score ranges."""
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if self.text is not None and self.tokens is not None:
            raise ValueError("provide text or tokens, not both")
        if self.text is not None and not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.tokens is not None and (
            not self.tokens
            or any(not isinstance(token, str) or not token.strip() for token in self.tokens)
        ):
            raise ValueError("tokens must be a non-empty sequence of non-empty strings")
        normalized = _canonicalize_modalities(self.modality_inputs)
        object.__setattr__(self, "modality_inputs", normalized)
        if self.text is None and self.tokens is None and not normalized:
            raise ValueError("provide text, tokens or at least one modality input")
        if not self.task.strip() or not self.source_reference.strip():
            raise ValueError("task and source_reference must be non-empty")
        _finite_mapping("context", self.context)
        _finite_mapping("modality_confidence", self.modality_confidence, 0.0)
        if any(float(value) > 1.0 for value in self.modality_confidence.values()):
            raise ValueError("modality confidence values must lie in [0, 1]")
        for name, value in self.modality_timestamps.items():
            if not str(name).strip() or not math.isfinite(float(value)):
                raise ValueError("modality timestamps must have non-empty names and finite values")
        for sense, features in self.candidate_senses.items():
            if not sense.strip():
                raise ValueError("candidate sense names must be non-empty")
            _finite_mapping(f"candidate_senses.{sense}", features, 0.0)
        _finite_mapping("feedback", self.feedback)
        if self.learner_response is not None:
            _finite_mapping("learner_response", self.learner_response, 0.0)
            if any(float(value) > 1.0 for value in self.learner_response.values()):
                raise ValueError("learner response scores must lie in [0, 1]")
        if any(not isinstance(item, str) or not item.strip() for item in self.usage_examples):
            raise ValueError("usage examples must be non-empty strings")

    @property
    def normalized_tokens(self) -> tuple[str, ...]:
        """Return caller tokens or tokenize the original text without modifying the source string."""
        if self.tokens is not None:
            return tuple(str(token).casefold() for token in self.tokens)
        if self.text:
            import re
            return tuple(re.findall(r"[\w'-]+", self.text.casefold()))
        return ()


@dataclass(slots=True)
class GroundedSemanticResult:
    """Return all 22 stage outputs, operation metadata, biological snapshots and diagnostics."""

    sample_id: str
    original_text: str
    output: Mapping[str, Any]
    stage_outputs: Mapping[str, Any]
    stage_trace: Sequence[Mapping[str, Any]]
    operation_trace: Sequence[Mapping[str, Any]]
    biological_trace: Sequence[Mapping[str, Any]]
    diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete pipeline result while keeping the schema stable and JSON-safe."""
        return _jsonable({
            "schema": "ore.grounded_semantic_integration.22_stage.v1",
            "sample_id": self.sample_id,
            "original_text": self.original_text,
            "output": self.output,
            "stage_outputs": self.stage_outputs,
            "stage_trace": self.stage_trace,
            "operation_trace": self.operation_trace,
            "biological_trace": self.biological_trace,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        })
