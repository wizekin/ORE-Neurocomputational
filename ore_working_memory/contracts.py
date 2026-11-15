"""Typed contracts for ORE's working-memory and cognitive-control subsystem.

The contract preserves input evidence, unfinished structures, alternative readings, and task
sequences as separate data so the pipeline can report uncertainty without inventing completions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import math
from typing import Any, Mapping, Sequence


def _validate_feature_map(name: str, values: Mapping[str, Any]) -> None:
    """Validate named feature strengths as finite values within the normalized interval."""
    for key, raw in values.items():
        value = float(raw)
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} values must be finite and lie in [0, 1]")


def _jsonable(value: Any) -> Any:
    """Convert nested contracts and numeric objects into JSON-compatible Python values."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return repr(value)


@dataclass(frozen=True, slots=True)
class WorkingMemoryRequest:
    """One linguistic sample with optional syntax, interpretation, goal, and task evidence.

    Supply exactly one of text or tokens. Partial structures and interpretation hypotheses are
    explicit evidence supplied by upstream modules or a caller; absent evidence is not fabricated.
    """

    sequence_id: str
    text: str | None = None
    tokens: Sequence[str] | None = None
    syntax_result: Any | None = None
    partial_structures: Sequence[Mapping[str, Any]] = ()
    interpretations: Sequence[Mapping[str, Any]] = ()
    task_sequence: Sequence[str] = ()
    goals: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    language: str = "en"
    source_reference: str | None = None

    def __post_init__(self) -> None:
        """Reject malformed linguistic samples before buffers or neural-reference state are allocated."""
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be a non-empty string")
        if (self.text is None) == (self.tokens is None):
            raise ValueError("provide exactly one of text or tokens")
        if self.text is not None and not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.tokens is not None:
            if not self.tokens or any(not isinstance(item, str) or not item.strip() for item in self.tokens):
                raise ValueError("tokens must be a non-empty sequence of non-empty strings")
            object.__setattr__(self, "tokens", tuple(self.tokens))
        object.__setattr__(self, "partial_structures", tuple(dict(item) for item in self.partial_structures))
        normalized_interpretations = tuple(dict(item) for item in self.interpretations)
        for index, candidate in enumerate(normalized_interpretations):
            for field_name in (
                "prior", "lexical_score", "semantic_score", "syntactic_score",
                "discourse_score", "world_score", "goal_compatibility",
            ):
                if field_name in candidate:
                    value = float(candidate[field_name])
                    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise ValueError(
                            f"interpretations[{index}].{field_name} must be finite and in [0, 1]"
                        )
            features = candidate.get("features", {})
            if not isinstance(features, Mapping):
                raise TypeError(f"interpretations[{index}].features must be a mapping")
            _validate_feature_map(f"interpretations[{index}].features", features)
        object.__setattr__(self, "interpretations", normalized_interpretations)
        object.__setattr__(self, "task_sequence", tuple(self.task_sequence))
        if any(not isinstance(task, str) or not task.strip() for task in self.task_sequence):
            raise ValueError("task_sequence must contain non-empty task names")
        if not isinstance(self.goals, Mapping) or not isinstance(self.context, Mapping):
            raise TypeError("goals and context must be mappings")
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("language must be a non-empty identifier")


@dataclass(frozen=True, slots=True)
class WorkingMemoryConfig:
    """Numerical controls for bounded buffers, competition, control, task switching, and LIF reference.

    Values are transparent engineering parameters for a reproducible software prototype. They
    are not physiological constants, fitted human parameters, or universal memory limits.
    """

    max_tokens: int = 120
    buffer_capacity: int = 24
    max_syntactic_frames: int = 32
    max_interpretations: int = 8
    inference_iterations: int = 6
    activation_decay_per_token: float = 0.075
    rehearsal_gain: float = 0.16
    retrieval_threshold: float = 0.52
    semantic_temperature: float = 0.55
    confidence_threshold: float = 0.60
    ambiguity_margin: float = 0.12
    control_gain: float = 0.65
    lateral_inhibition: float = 0.28
    recurrent_damping: float = 0.45
    task_switch_cost_ms: float = 125.0
    task_inertia: float = 0.25
    simulate_biology: bool = True
    population_size: int = 10
    dt_ms: float = 1.0
    steps_per_operation: int = 2
    membrane_tau_ms: float = 20.0
    refractory_ms: float = 2.0
    rest_mv: float = -70.0
    reset_mv: float = -68.0
    threshold_mv: float = -50.0
    membrane_resistance_mohm: float = 100.0
    current_gain_pa: float = 300.0
    recurrent_gain_pa: float = 18.0
    inhibition_gain_pa: float = 20.0

    def __post_init__(self) -> None:
        """Validate resource limits, normalized gains, and stable LIF reference parameters."""
        if min(
            self.max_tokens, self.buffer_capacity, self.max_syntactic_frames,
            self.max_interpretations, self.inference_iterations,
        ) < 1:
            raise ValueError("token, buffer, frame, interpretation, and iteration limits must be positive")
        if self.population_size < 2 or self.steps_per_operation < 1:
            raise ValueError("population_size must be >= 2 and steps_per_operation positive")
        for name in (
            "activation_decay_per_token", "rehearsal_gain", "retrieval_threshold",
            "confidence_threshold", "ambiguity_margin", "control_gain", "recurrent_damping",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in ("semantic_temperature", "dt_ms", "membrane_tau_ms",
                     "membrane_resistance_mohm", "current_gain_pa"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("task_switch_cost_ms", "lateral_inhibition", "refractory_ms",
                     "recurrent_gain_pa", "inhibition_gain_pa"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.task_inertia <= 1.0:
            raise ValueError("task_inertia must lie in [0, 1]")
        if not self.rest_mv <= self.reset_mv < self.threshold_mv:
            raise ValueError("require rest_mv <= reset_mv < threshold_mv")

    def to_dict(self) -> dict[str, Any]:
        """Return all configuration values as a plain dictionary for logs and result metadata."""
        return asdict(self)


@dataclass(slots=True)
class WorkingMemoryResult:
    """Inspectable output for linguistic buffers, open syntax, control, meaning, and task state."""

    sequence_id: str
    original_text: str
    tokens: list[str]
    buffers: dict[str, Any]
    incomplete_structures: dict[str, Any]
    executive_control: dict[str, Any]
    semantic_resolution: dict[str, Any]
    task_switching: dict[str, Any]
    biological: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete result while retaining alternatives and unresolved structures."""
        return _jsonable({
            "schema": "ore.working_memory_cognitive_control.1.0",
            "sequence_id": self.sequence_id,
            "original_text": self.original_text,
            "tokens": self.tokens,
            "buffers": self.buffers,
            "incomplete_structures": self.incomplete_structures,
            "executive_control": self.executive_control,
            "semantic_resolution": self.semantic_resolution,
            "task_switching": self.task_switching,
            "biological": self.biological,
            "operation_trace": self.operation_trace,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        })
