"""Validated input, configuration, and output contracts for rapid episodic encoding.

All vectors are dimensionless model activations unless explicitly labelled otherwise.
The contracts deliberately distinguish caller-supplied observations from computed hypotheses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _validate_unit_map(name: str, values: Mapping[str, float]) -> None:
    """Reject empty keys and non-finite or out-of-range normalized channel values."""
    for key, raw in values.items():
        value = float(raw)
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}[{key!r}] must be finite and in [0, 1]")


def _jsonable(value: Any) -> Any:
    """Convert nested dataclasses, mappings, sequences, and NumPy scalars to JSON values."""
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(frozen=True, slots=True)
class EventInput:
    """One observed event with sensory channels and optional explicit relational context.

    ``sensory_signals`` maps modality/channel names to finite numeric feature vectors. Context,
    goals, salience, uncertainty, components, and relations are optional caller-supplied evidence;
    the model does not invent absent observations. Component dictionaries may include ``id``,
    ``label``, ``features`` and ``kind``. Relation dictionaries may include ``source``, ``type``
    and ``target``. Timestamp is a model time coordinate in milliseconds, not wall-clock time.
    """

    event_id: str
    sensory_signals: Mapping[str, Sequence[float]]
    context: Sequence[float] = ()
    timestamp_ms: float = 0.0
    components: Sequence[Mapping[str, Any]] = ()
    relations: Sequence[Mapping[str, Any]] = ()
    goals: Mapping[str, float] = field(default_factory=dict)
    salience: Mapping[str, float] = field(default_factory=dict)
    uncertainty: Mapping[str, float] = field(default_factory=dict)
    modulatory_state: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize immutable input collections and validate every observed numeric value."""
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        if not isinstance(self.sensory_signals, Mapping) or not self.sensory_signals:
            raise ValueError("sensory_signals must be a non-empty mapping")
        normalized: dict[str, tuple[float, ...]] = {}
        for channel, raw_vector in self.sensory_signals.items():
            if not isinstance(channel, str) or not channel.strip():
                raise ValueError("sensory channel names must be non-empty strings")
            vector = tuple(float(item) for item in raw_vector)
            if not vector or any(not math.isfinite(item) for item in vector):
                raise ValueError(f"sensory_signals[{channel!r}] must be a non-empty finite vector")
            normalized[channel] = vector
        context = tuple(float(item) for item in self.context)
        if any(not math.isfinite(item) for item in context):
            raise ValueError("context values must be finite")
        timestamp = float(self.timestamp_ms)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_ms must be finite and non-negative")
        for name in ("goals", "salience", "uncertainty", "modulatory_state"):
            values = getattr(self, name)
            if not isinstance(values, Mapping):
                raise TypeError(f"{name} must be a mapping")
            _validate_unit_map(name, values)
        object.__setattr__(self, "sensory_signals", normalized)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "timestamp_ms", timestamp)
        object.__setattr__(self, "components", tuple(dict(item) for item in self.components))
        object.__setattr__(self, "relations", tuple(dict(item) for item in self.relations))
        object.__setattr__(self, "goals", dict(self.goals))
        object.__setattr__(self, "salience", dict(self.salience))
        object.__setattr__(self, "uncertainty", dict(self.uncertainty))
        object.__setattr__(self, "modulatory_state", dict(self.modulatory_state))

    def to_dict(self) -> dict[str, Any]:
        """Return the validated event and its provenance as a JSON-compatible dictionary."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class EncodingConfig:
    """Transparent model controls; these are engineering parameters, not measured human constants.

    The DG/CA3/CA1 population sizes and thresholds define a reproducible simulation. Biological
    evidence labels are attached to operations separately so a numerical choice is not mistaken
    for a measured physiological quantity.
    """

    dentate_units: int = 96
    dentate_active_fraction: float = 0.08
    ca3_units: int = 64
    ca1_units: int = 64
    attention_temperature: float = 0.7
    event_boundary_threshold: float = 0.32
    recurrent_gain: float = 0.14
    plasticity_learning_rate: float = 0.025
    plasticity_ltp_threshold: float = 0.62
    plasticity_ltd_threshold: float = 0.24
    plasticity_weight_decay: float = 0.002
    max_memory_records: int = 2048
    temporal_frequencies: int = 6
    random_seed: int = 17

    def __post_init__(self) -> None:
        """Reject invalid population sizes, rates, thresholds, and memory limits."""
        if min(self.dentate_units, self.ca3_units, self.ca1_units,
               self.max_memory_records, self.temporal_frequencies) < 2:
            raise ValueError("population, memory, and temporal-frequency limits must be >= 2")
        if not math.isfinite(self.dentate_active_fraction) or not 0.0 < self.dentate_active_fraction <= 1.0:
            raise ValueError("dentate_active_fraction must lie in (0, 1]")
        if not math.isfinite(self.attention_temperature) or self.attention_temperature <= 0.0:
            raise ValueError("attention_temperature must be positive and finite")
        if not math.isfinite(self.event_boundary_threshold) or not 0.0 <= self.event_boundary_threshold <= 1.0:
            raise ValueError("event_boundary_threshold must lie in [0, 1]")
        if not math.isfinite(self.recurrent_gain) or self.recurrent_gain < 0.0:
            raise ValueError("recurrent_gain must be finite and non-negative")
        if not math.isfinite(self.plasticity_learning_rate) or self.plasticity_learning_rate < 0.0:
            raise ValueError("plasticity_learning_rate must be finite and non-negative")
        if not 0.0 <= self.plasticity_ltd_threshold < self.plasticity_ltp_threshold <= 1.0:
            raise ValueError("plasticity thresholds must satisfy 0 <= LTD < LTP <= 1")
        if not math.isfinite(self.plasticity_weight_decay) or self.plasticity_weight_decay < 0.0:
            raise ValueError("plasticity_weight_decay must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return the full model configuration for reproducibility and logger metadata."""
        return asdict(self)


@dataclass(slots=True)
class EncodingResult:
    """Inspectable output containing all intermediate codes, relations, and model diagnostics."""

    event_id: str
    event_boundary: dict[str, Any]
    sensory_features: dict[str, list[float]]
    attention_weights: dict[str, float]
    item_representation: list[float]
    context_representation: list[float]
    temporal_representation: list[float]
    entorhinal_representation: list[float]
    novelty: dict[str, Any]
    dentate_activity: list[float]
    ca3_activity: list[float]
    ca1_activity: list[float]
    ca1_prediction_error: float
    plasticity: dict[str, Any]
    event_index: dict[str, Any]
    cortical_feedback: list[float]
    trace_quality: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    biological: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result while keeping computed estimates distinct from observed inputs."""
        return _jsonable({"schema": "ore.rapid_hippocampal_encoding.1.0", **asdict(self)})
