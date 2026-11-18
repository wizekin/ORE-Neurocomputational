"""Typed data contracts for Point 19 working-memory overload processing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _validate_unit(name: str, value: float) -> float:
    """Validate and return a finite value in the normalized unit interval."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return number


@dataclass(frozen=True)
class Point19Config:
    """Calibratable model parameters, not physiological constants or human capacity limits."""

    retention_decay: float = 0.055
    refresh_gain: float = 0.16
    interference_gain: float = 0.24
    retrieval_temperature: float = 0.35
    attention_temperature: float = 0.50
    retrieval_threshold: float = 0.62
    overload_threshold: float = 0.62
    dependency_scale: float = 5.0
    recovery_load_cost: float = 0.45
    recovery_time_cost: float = 0.20
    adaptation_rate: float = 0.04
    simulate_biology: bool = True
    population_decay: float = 0.22
    population_recurrent_gain: float = 0.28
    population_inhibition_gain: float = 0.16

    def __post_init__(self) -> None:
        """Reject invalid gains and thresholds before allocating processing state."""
        for name in (
            "retention_decay", "refresh_gain", "interference_gain",
            "retrieval_threshold", "overload_threshold", "recovery_load_cost",
            "recovery_time_cost", "adaptation_rate", "population_decay",
            "population_inhibition_gain",
        ):
            _validate_unit(name, getattr(self, name))
        for name in ("retrieval_temperature", "attention_temperature", "dependency_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.population_recurrent_gain) or self.population_recurrent_gain < 0:
            raise ValueError("population_recurrent_gain must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible configuration snapshot."""
        return asdict(self)


@dataclass(frozen=True)
class Point19Request:
    """One message increment plus explicit structural, contextual, and behavioral evidence."""

    sequence_id: str
    text: str | None = None
    units: Sequence[str] | None = None
    unit_annotations: Sequence[Mapping[str, Any]] = ()
    dependencies: Sequence[Mapping[str, Any]] = ()
    goals: Mapping[str, float] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    extra_task_load: float = 0.0
    source_reference: str | None = None

    def __post_init__(self) -> None:
        """Validate the request while preserving the distinction between input and inference."""
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be a non-empty string")
        if (self.text is None) == (self.units is None):
            raise ValueError("provide exactly one of text or units")
        if self.text is not None and not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.units is not None:
            if not self.units or any(not isinstance(item, str) or not item.strip() for item in self.units):
                raise ValueError("units must contain non-empty strings")
            object.__setattr__(self, "units", tuple(self.units))
        object.__setattr__(self, "unit_annotations", tuple(dict(item) for item in self.unit_annotations))
        object.__setattr__(self, "dependencies", tuple(dict(item) for item in self.dependencies))
        if self.unit_annotations and self.units is not None and len(self.unit_annotations) != len(self.units):
            raise ValueError("unit_annotations must have one item per supplied unit")
        if not isinstance(self.goals, Mapping) or not isinstance(self.context, Mapping):
            raise TypeError("goals and context must be mappings")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        for name, raw in {**dict(self.goals), **dict(self.metrics)}.items():
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        _validate_unit("extra_task_load", self.extra_task_load)
        dependency_ids = [str(item.get("identifier", f"d{i}")) for i, item in enumerate(self.dependencies)]
        if len(dependency_ids) != len(set(dependency_ids)):
            raise ValueError("dependency identifiers must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached request snapshot for provenance and run logging."""
        return {
            "sequence_id": self.sequence_id,
            "text": self.text,
            "units": list(self.units) if self.units is not None else None,
            "unit_annotations": [dict(item) for item in self.unit_annotations],
            "dependencies": [dict(item) for item in self.dependencies],
            "goals": dict(self.goals),
            "context": dict(self.context),
            "metrics": dict(self.metrics),
            "extra_task_load": self.extra_task_load,
            "source_reference": self.source_reference,
        }


@dataclass
class Representation:
    """Track an encoded item, its features, accessibility, source, and current chunk."""

    identifier: str
    text: str
    features: dict[str, Any]
    position: int
    source: str = "observed_input"
    accessibility: float = 1.0
    confidence: float = 0.65
    task_relevance: float = 0.5
    chunk_id: str | None = None
    interference: float = 0.0
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a detached representation snapshot."""
        return asdict(self)


@dataclass
class Dependency:
    """Represent an open relation and the features required to resolve it."""

    identifier: str
    required_features: dict[str, Any]
    cue_features: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.7
    candidate_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    open: bool = True
    failure_type: str = "unresolved"

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dependency snapshot."""
        return asdict(self)


@dataclass
class Point19State:
    """Maintain active items, open relations, situation evidence, and adaptive task parameters."""

    representations: dict[str, Representation] = field(default_factory=dict)
    dependencies: dict[str, Dependency] = field(default_factory=dict)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    event_model: list[dict[str, Any]] = field(default_factory=list)
    uncertainty: list[dict[str, Any]] = field(default_factory=list)
    goal_weights: dict[str, float] = field(default_factory=dict)
    estimated_load: float = 0.0
    overload_probability: float = 0.0
    adaptive_weights: dict[str, float] = field(default_factory=lambda: {
        "dependency": 0.18, "complexity": 0.14, "distance": 0.12,
        "interference": 0.18, "task": 0.14, "uncertainty": 0.12,
    })
    step_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full revisable state without exposing mutable internals."""
        return {
            "representations": {key: item.to_dict() for key, item in self.representations.items()},
            "dependencies": {key: item.to_dict() for key, item in self.dependencies.items()},
            "chunks": [dict(item) for item in self.chunks],
            "event_model": [dict(item) for item in self.event_model],
            "uncertainty": [dict(item) for item in self.uncertainty],
            "goal_weights": dict(self.goal_weights),
            "estimated_load": self.estimated_load,
            "overload_probability": self.overload_probability,
            "adaptive_weights": dict(self.adaptive_weights),
            "step_index": self.step_index,
        }


@dataclass(frozen=True)
class Point19Result:
    """Collect the 19-step trace, diagnostics, separate biological proxy, and final state."""

    sequence_id: str
    interpretation: Mapping[str, Any]
    load_assessment: Mapping[str, Any]
    retrieval: Mapping[str, Any]
    recovery: Mapping[str, Any]
    failure_diagnosis: Mapping[str, Any]
    coherence: Mapping[str, Any]
    adaptation: Mapping[str, Any]
    biological: Mapping[str, Any]
    operation_trace: tuple[Mapping[str, Any], ...]
    final_state: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-data result for serialization or downstream processing."""
        return {
            "schema": "ore.working_memory_overload.point19.1.0",
            "sequence_id": self.sequence_id,
            "interpretation": dict(self.interpretation),
            "load_assessment": dict(self.load_assessment),
            "retrieval": dict(self.retrieval),
            "recovery": dict(self.recovery),
            "failure_diagnosis": dict(self.failure_diagnosis),
            "coherence": dict(self.coherence),
            "adaptation": dict(self.adaptation),
            "biological": dict(self.biological),
            "operation_trace": [dict(item) for item in self.operation_trace],
            "final_state": dict(self.final_state),
            "metadata": dict(self.metadata),
        }
