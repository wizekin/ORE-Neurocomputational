"""Validated contracts for gradual cortical learning across repeated encounters.

All numerical arrays represent dimensionless software activations or model parameters unless
explicitly stated. Context labels, modulators, and observed targets are caller-supplied evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _validate_unit_mapping(name: str, values: Mapping[str, float]) -> None:
    """Reject empty labels and non-finite normalized values in modulation and metadata maps."""
    for key, raw in values.items():
        value = float(raw)
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}[{key!r}] must be finite and in [0, 1]")


def _jsonable(value: Any) -> Any:
    """Convert dataclasses, mappings, sequences, and NumPy scalars into JSON-safe values."""
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
class EncounterInput:
    """One experience supplied to the incremental cortical learner.

    ``sensory_activity`` is required. Optional action, context, goal, and episode-trace vectors
    are concatenated in a fixed order by the processor. ``observed_activity`` is an optional
    downstream target used to compute a prediction error; if absent, the learner does not invent
    one. ``context_id`` keeps statistically different situations in separate reliability branches.
    Modulator entries can include attention, novelty, reward, and arousal in the normalized range.
    A caller should set ``evaluation_only=True`` only for an explicitly held-out encounter.
    ``pre_post_interval_ms`` is post-minus-pre timing in milliseconds for an STDP-inspired rule.
    """

    encounter_id: str
    sensory_activity: Sequence[float]
    context_id: str = "unspecified"
    action_activity: Sequence[float] = ()
    context: Sequence[float] = ()
    goal_activity: Sequence[float] = ()
    episode_trace: Sequence[float] = ()
    observed_activity: Sequence[float] | None = None
    modulatory_state: Mapping[str, float] = field(default_factory=dict)
    pre_post_interval_ms: float = 0.0
    evaluation_only: bool = False

    def __post_init__(self) -> None:
        """Normalize input vectors and reject missing identifiers or invalid observed values."""
        if not isinstance(self.encounter_id, str) or not self.encounter_id.strip():
            raise ValueError("encounter_id must be a non-empty string")
        if not isinstance(self.context_id, str) or not self.context_id.strip():
            raise ValueError("context_id must be a non-empty string")
        for name in ("sensory_activity", "action_activity", "context", "goal_activity", "episode_trace"):
            vector = tuple(float(item) for item in getattr(self, name))
            if name == "sensory_activity" and not vector:
                raise ValueError("sensory_activity must not be empty")
            if any(not math.isfinite(item) for item in vector):
                raise ValueError(f"{name} values must be finite")
            object.__setattr__(self, name, vector)
        if self.observed_activity is not None:
            observed = tuple(float(item) for item in self.observed_activity)
            if not observed or any(not math.isfinite(item) for item in observed):
                raise ValueError("observed_activity must be a non-empty finite vector when supplied")
            object.__setattr__(self, "observed_activity", observed)
        if not isinstance(self.modulatory_state, Mapping):
            raise TypeError("modulatory_state must be a mapping")
        _validate_unit_mapping("modulatory_state", self.modulatory_state)
        interval = float(self.pre_post_interval_ms)
        if not math.isfinite(interval):
            raise ValueError("pre_post_interval_ms must be finite")
        object.__setattr__(self, "pre_post_interval_ms", interval)
        object.__setattr__(self, "modulatory_state", dict(self.modulatory_state))
        if not isinstance(self.evaluation_only, bool):
            raise TypeError("evaluation_only must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        """Return the encounter and all explicitly supplied evidence as a JSON-safe dictionary."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class CorticalLearningConfig:
    """Reproducible software parameters for cortical population and incremental learning models.

    Dimensions are explicit so all encounters in one processor share a stable representation.
    Learning rates, target activity, time constants, and weight bounds are engineering parameters,
    not universal human physiological constants or fitted biological measurements.
    """

    sensory_size: int = 12
    action_size: int = 4
    context_size: int = 4
    goal_size: int = 3
    episode_trace_size: int = 5
    cortical_units: int = 32
    output_units: int = 12
    learning_rate: float = 0.008
    statistics_decay: float = 0.10
    reliability_prior_alpha: float = 1.0
    reliability_prior_beta: float = 1.0
    similarity_threshold: float = 0.72
    recurrent_gain: float = 0.20
    inhibitory_gain: float = 0.12
    modulatory_floor: float = 0.10
    eligibility_tau_ms: float = 250.0
    max_weight_step: float = 0.025
    max_abs_weight: float = 2.5
    interference_strength: float = 0.015
    importance_decay: float = 0.04
    reference_update_rate: float = 0.02
    homeostatic_rate: float = 0.008
    target_activity: float = 0.28
    homeostatic_gain_min: float = 0.5
    homeostatic_gain_max: float = 1.8
    weight_decay: float = 0.0005
    random_seed: int = 23
    max_context_branches: int = 128
    max_encounter_history: int = 4096

    def __post_init__(self) -> None:
        """Validate dimensions, bounded learning controls, and stable numerical time constants."""
        dimensions = (self.sensory_size, self.cortical_units, self.output_units,
                      self.max_context_branches, self.max_encounter_history)
        if min(dimensions) < 2:
            raise ValueError("sensory, population, output, and history dimensions must be >= 2")
        if min(self.action_size, self.context_size, self.goal_size, self.episode_trace_size) < 0:
            raise ValueError("optional channel sizes must be non-negative")
        for name in ("learning_rate", "max_weight_step", "max_abs_weight", "eligibility_tau_ms",
                     "target_activity", "homeostatic_gain_min", "homeostatic_gain_max",
                     "reliability_prior_alpha", "reliability_prior_beta"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("statistics_decay", "similarity_threshold", "modulatory_floor",
                     "importance_decay", "reference_update_rate", "homeostatic_rate",
                     "weight_decay"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in ("recurrent_gain", "inhibitory_gain", "interference_strength"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.homeostatic_gain_min > self.homeostatic_gain_max:
            raise ValueError("homeostatic gain minimum cannot exceed its maximum")
        if self.target_activity >= 1.0:
            raise ValueError("target_activity must be less than 1")

    @property
    def input_size(self) -> int:
        """Return the total fixed-width size of all concatenated encounter channels."""
        return self.sensory_size + self.action_size + self.context_size + self.goal_size + self.episode_trace_size

    def to_dict(self) -> dict[str, Any]:
        """Return every model parameter for reproducibility and SQLite logging."""
        return asdict(self)


@dataclass(slots=True)
class CorticalLearningResult:
    """Inspectable output for one encounter, including activity, reliability, plasticity, and tests."""

    encounter_id: str
    context_id: str
    evaluation_only: bool
    cortical_activity: list[float]
    prediction: list[float]
    observed_activity: list[float] | None
    prediction_error: list[float] | None
    prediction_mse: float | None
    similarities: dict[str, Any]
    recurring_features: dict[str, Any]
    recurring_relationships: dict[str, Any]
    relationship_reliability: dict[str, Any]
    plasticity_eligibility: dict[str, Any]
    modulatory_gate: dict[str, Any]
    synaptic_update: dict[str, Any]
    interference_control: dict[str, Any]
    accumulated_statistics: dict[str, Any]
    homeostasis: dict[str, Any]
    generalization_evaluation: dict[str, Any]
    uncertainty: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    biological: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize all computed values while preserving missing targets and uncertainty labels."""
        return _jsonable({"schema": "ore.gradual_cortical_learning.1.0", **asdict(self)})
