"""Typed contracts for Point 22 cognitive heuristics and systematic reasoning errors."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


def _unit_interval(name: str, value: float) -> None:
    """Reject non-finite values and values outside [0, 1]."""
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")


@dataclass(frozen=True)
class JudgmentProblem:
    """Describe a judgment task, its evidence, and its environmental context."""

    problem_id: str
    question: str
    target_type: str = "probability"
    categories: tuple[str, ...] = ()
    examples: tuple[Mapping[str, Any], ...] = ()
    base_rates: Mapping[str, float] = field(default_factory=dict)
    features: Mapping[str, float] = field(default_factory=dict)
    prototypes: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    feature_weights: Mapping[str, float] = field(default_factory=dict)
    similarity_scores: Mapping[str, float] = field(default_factory=dict)
    candidate_anchors: tuple[float, ...] = ()
    fluency: float = 0.0
    evidence_confidence: float = 0.5
    time_pressure: float = 0.0
    emotional_salience: float = 0.0
    decision_cost: float = 0.0
    cue_validity: Mapping[str, float] = field(default_factory=dict)
    environment_stability: float = 0.5
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate numeric ranges and ensure the request has usable identity."""
        if not self.problem_id.strip() or not self.question.strip():
            raise ValueError("problem_id and question must be non-empty")
        if self.target_type not in {"probability", "frequency", "magnitude", "cause", "choice"}:
            raise ValueError("unsupported target_type")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("categories must be unique")
        for key, value in self.base_rates.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"base rate for {key!r} must be finite and non-negative")
        if self.base_rates and sum(float(v) for v in self.base_rates.values()) <= 0:
            raise ValueError("base_rates must contain at least one positive value")
        for name in ("fluency", "evidence_confidence", "time_pressure",
                     "emotional_salience", "decision_cost", "environment_stability"):
            _unit_interval(name, getattr(self, name))
        for key, value in self.similarity_scores.items():
            _unit_interval(f"similarity_scores[{key}]", value)
        for key, value in self.cue_validity.items():
            _unit_interval(f"cue_validity[{key}]", value)
        for key, value in self.feature_weights.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"feature weight for {key!r} must be finite and non-negative")
        for anchor in self.candidate_anchors:
            if not math.isfinite(float(anchor)):
                raise ValueError("candidate anchors must be finite numbers")


@dataclass(frozen=True)
class Point22Config:
    """Control strategy thresholds and learning rates; values are engineering parameters."""

    adjustment_fraction: float = 0.5
    fluency_weight: float = 0.2
    learning_rate: float = 0.1
    attention_temperature: float = 1.0
    high_risk_threshold: float = 0.7
    conflict_threshold: float = 0.25
    small_sample_threshold: int = 30
    simulate_biology: bool = True

    def __post_init__(self) -> None:
        """Validate bounded configuration values."""
        for name in ("adjustment_fraction", "fluency_weight", "learning_rate",
                     "high_risk_threshold", "conflict_threshold"):
            _unit_interval(name, getattr(self, name))
        if self.attention_temperature <= 0 or not math.isfinite(self.attention_temperature):
            raise ValueError("attention_temperature must be positive and finite")
        if self.small_sample_threshold < 2:
            raise ValueError("small_sample_threshold must be at least 2")


@dataclass
class Point22State:
    """Retain slowly updated cue weights and prediction history across judgments."""

    cue_weights: dict[str, float] = field(default_factory=lambda: {
        "availability": 1.0, "representativeness": 1.0, "base_rate": 1.0,
    })
    prediction_history: list[dict[str, Any]] = field(default_factory=list)
    judgment_history: list[dict[str, Any]] = field(default_factory=list)
    run_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return an independent, serialization-friendly state snapshot."""
        return {
            "cue_weights": dict(self.cue_weights),
            "prediction_history": [dict(item) for item in self.prediction_history],
            "judgment_history": [dict(item) for item in self.judgment_history],
            "run_count": self.run_count,
        }


@dataclass(frozen=True)
class Point22Result:
    """Expose intuitive and verified judgments, diagnostics, and both operation tracks."""

    problem_id: str
    initial_answer: Any
    verified_answer: Any
    probabilities: Mapping[str, float]
    confidence: float
    heuristic_used: tuple[str, ...]
    warnings: tuple[str, ...]
    assumptions: tuple[str, ...]
    diagnostics: Mapping[str, Any]
    sampling: Mapping[str, Any]
    causal_check: Mapping[str, Any]
    ecological_fit: Mapping[str, Any]
    feedback: Mapping[str, Any]
    decision: Mapping[str, Any]
    biological: Mapping[str, Any]
    operation_trace: tuple[Mapping[str, Any], ...]
    final_state: Mapping[str, Any]
    metadata: Mapping[str, Any]
