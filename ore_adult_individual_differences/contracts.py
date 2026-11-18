"""Contracts and persistent state for adult individual differences (Point 17).

Values describe a temporary person-task-context interaction, not a diagnosis, intelligence score,
or fixed label. Age and language history are contextual inputs, never deterministic group rules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _finite(name: str, value: float) -> float:
    """Convert a numeric input to a finite float or reject invalid values."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _unit(name: str, value: float) -> float:
    """Validate a normalized numeric value in the closed interval [0, 1]."""
    result = _finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(name + " must be in [0, 1]")
    return result


def _plain(value: Any) -> Any:
    """Convert nested dataclasses and common containers to JSON-compatible values."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, set):
        return sorted((_plain(item) for item in value), key=repr)
    if hasattr(value, "item") and callable(value.item):
        return _plain(value.item())
    return value


def _normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Validate a non-negative goal-weight vector and normalize it to sum to one."""
    clean = {str(key): _finite("goal_weight", value) for key, value in weights.items()}
    if any(value < 0.0 for value in clean.values()):
        raise ValueError("goal weights cannot be negative")
    total = sum(clean.values())
    if total <= 0.0:
        raise ValueError("at least one goal weight must be positive")
    return {key: value / total for key, value in clean.items()}


@dataclass(frozen=True, slots=True)
class AdultComprehensionExperience:
    """Represent one adult comprehension or learning task and its current conditions.

    Content units and candidate interpretations preserve order and provenance. Language,
    cultural, and age-related inputs describe demonstrated experience and context; they are
    not inferred from demographic labels.
    """

    experience_id: str
    content_units: Sequence[Any]
    domain: str
    task_goal: Mapping[str, float] = field(default_factory=lambda: {
        "detail": 0.30, "gist": 0.25, "relation": 0.25, "procedure": 0.10, "source": 0.10,
    })
    desired_output: str = "comprehension"
    language: str = "unspecified"
    language_cues: Mapping[str, float] = field(default_factory=dict)
    interpretation_candidates: Sequence[Mapping[str, Any]] = ()
    dependency_pairs: Sequence[Sequence[int]] = ()
    pace_pressure: float = 0.0
    distraction: float = 0.0
    fatigue: float = 0.0
    stress: float = 0.0
    motivation: float = 0.5
    external_support: float = 0.0
    time_budget_ms: float | None = None
    processing_time_observations_ms: Mapping[str, float] = field(default_factory=dict)
    task_context: Mapping[str, Any] = field(default_factory=dict)
    cultural_cues: Sequence[str] = ()
    cultural_conventions: Mapping[str, Any] = field(default_factory=dict)
    context_observations: Mapping[str, float] = field(default_factory=dict)
    memory_request: str = "balanced"
    age_years: float | None = None
    memory_evidence: Mapping[str, Any] = field(default_factory=dict)
    new_facts: Sequence[Mapping[str, Any]] = ()
    evidence_quality: float = 0.75
    provenance_completeness: float = 1.0
    expected_response: str = ""
    observed_response: str = ""
    performance_feedback: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identifiers and probabilities while detaching mutable event inputs."""
        if not str(self.experience_id).strip():
            raise ValueError("experience_id must be non-empty")
        if not str(self.domain).strip():
            raise ValueError("domain must be non-empty")
        if not self.content_units:
            raise ValueError("content_units must contain at least one unit")
        object.__setattr__(self, "domain", str(self.domain).strip().lower())
        object.__setattr__(self, "desired_output", str(self.desired_output).strip().lower())
        object.__setattr__(self, "language", str(self.language).strip().lower())
        object.__setattr__(self, "task_goal", _normalize_weights(self.task_goal))
        raw_units = (self.content_units,) if isinstance(self.content_units, str) else self.content_units
        object.__setattr__(self, "content_units", tuple(
            dict(value) if isinstance(value, Mapping) else value for value in raw_units
        ))
        object.__setattr__(self, "interpretation_candidates", tuple(
            dict(value) for value in self.interpretation_candidates
        ))
        pairs: list[tuple[int, int]] = []
        for pair in self.dependency_pairs:
            if len(pair) != 2:
                raise ValueError("each dependency pair must contain two indices")
            left, right = int(pair[0]), int(pair[1])
            if left < 0 or right < 0 or left == right:
                raise ValueError("dependency indices must be distinct and non-negative")
            pairs.append((left, right))
        object.__setattr__(self, "dependency_pairs", tuple(pairs))
        for name in (
            "language_cues", "processing_time_observations_ms", "task_context",
            "cultural_conventions", "context_observations", "memory_evidence",
            "performance_feedback", "metadata",
        ):
            object.__setattr__(self, name, dict(getattr(self, name)))
        object.__setattr__(self, "cultural_cues", tuple(str(x) for x in self.cultural_cues))
        object.__setattr__(self, "new_facts", tuple(dict(value) for value in self.new_facts))
        for name in (
            "pace_pressure", "distraction", "fatigue", "stress", "motivation",
            "external_support", "evidence_quality", "provenance_completeness",
        ):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        if self.time_budget_ms is not None:
            budget = _finite("time_budget_ms", self.time_budget_ms)
            if budget <= 0.0:
                raise ValueError("time_budget_ms must be positive")
            object.__setattr__(self, "time_budget_ms", budget)
        observations: dict[str, float] = {}
        for key, value in self.processing_time_observations_ms.items():
            numeric = _finite("processing_time_observations_ms", value)
            if numeric < 0.0:
                raise ValueError("processing time observations cannot be negative")
            observations[str(key)] = numeric
        object.__setattr__(self, "processing_time_observations_ms", observations)
        if self.age_years is not None:
            age = _finite("age_years", self.age_years)
            if not 18.0 <= age <= 120.0:
                raise ValueError("age_years must be in [18, 120] when supplied")
            object.__setattr__(self, "age_years", age)
        if self.memory_request not in {"gist", "detail", "source", "balanced"}:
            raise ValueError("memory_request must be gist, detail, source, or balanced")
        if not self.desired_output:
            raise ValueError("desired_output must be non-empty")


@dataclass(frozen=True, slots=True)
class AdultDifferencesConfig:
    """Configure model heuristics while avoiding age-based ability penalties."""

    attention_distraction_weight: float = 0.55
    attention_fatigue_weight: float = 0.35
    attention_stress_weight: float = 0.20
    motivation_weight: float = 0.15
    external_support_gain: float = 0.25
    wm_attention_mix: float = 0.50
    schema_match_threshold: float = 0.45
    chunk_confidence_threshold: float = 0.58
    load_dependency_weight: float = 0.50
    speed_decay_tau_ms: float = 1200.0
    ambiguity_entropy_threshold: float = 0.72
    ambiguity_margin_threshold: float = 0.10
    expertise_reversal_gain: float = 0.65
    detail_memory_bias: float = 0.0
    semantic_update_rate: float = 0.12
    confidence_uncertainty_penalty: float = 0.45
    confidence_inference_penalty: float = 0.20
    max_live_hypotheses: int = 5

    def __post_init__(self) -> None:
        """Validate normalized parameters and positive time/selection scales."""
        for name in (
            "attention_distraction_weight", "attention_fatigue_weight",
            "attention_stress_weight", "motivation_weight", "external_support_gain",
            "wm_attention_mix", "schema_match_threshold", "chunk_confidence_threshold",
            "ambiguity_entropy_threshold", "ambiguity_margin_threshold",
            "expertise_reversal_gain", "confidence_uncertainty_penalty",
            "confidence_inference_penalty",
        ):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        for name in ("load_dependency_weight", "speed_decay_tau_ms"):
            value = _finite(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(name + " must be positive")
            object.__setattr__(self, name, value)
        if not 1 <= self.max_live_hypotheses <= 50:
            raise ValueError("max_live_hypotheses must be in [1, 50]")
        object.__setattr__(self, "max_live_hypotheses", int(self.max_live_hypotheses))


@dataclass
class AdultLearnerState:
    """Maintain separate domain, language, cultural, and task-specific learner knowledge."""

    learner_id: str = "default"
    attention_control: float = 0.70
    working_memory_support: float = 0.70
    processing_speed: float = 0.70
    prior_knowledge: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    expertise_by_domain: dict[str, float] = field(default_factory=dict)
    language_proficiency: dict[str, float] = field(default_factory=dict)
    language_cue_reliability: dict[str, dict[str, float]] = field(default_factory=dict)
    cultural_experience: dict[str, dict[str, Any]] = field(default_factory=dict)
    knowledge_estimates: dict[str, dict[str, Any]] = field(default_factory=dict)
    semantic_associations: dict[str, float] = field(default_factory=dict)
    episodic_records: list[dict[str, Any]] = field(default_factory=list)
    component_speed: dict[str, float] = field(default_factory=dict)
    performance_history: dict[str, list[float]] = field(default_factory=dict)
    task_history: list[dict[str, Any]] = field(default_factory=list)
    experience_count: int = 0

    def __post_init__(self) -> None:
        """Validate learner baselines and normalize domain-specific experience estimates."""
        if not str(self.learner_id).strip():
            raise ValueError("learner_id must be non-empty")
        for name in ("attention_control", "working_memory_support", "processing_speed"):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        for name in ("expertise_by_domain", "language_proficiency", "component_speed",
                     "semantic_associations"):
            setattr(self, name, {str(k).lower(): _unit(name, v) for k, v in getattr(self, name).items()})
        self.language_cue_reliability = {
            str(language).lower(): {
                str(cue).lower(): _unit("language_cue_reliability", value)
                for cue, value in cues.items()
            }
            for language, cues in self.language_cue_reliability.items()
        }
        self.prior_knowledge = {
            str(domain).lower(): [dict(item) for item in values]
            for domain, values in self.prior_knowledge.items()
        }
        self.cultural_experience = {
            str(context).lower(): dict(values)
            for context, values in self.cultural_experience.items()
        }
        self.knowledge_estimates = {
            str(domain).lower(): dict(values)
            for domain, values in self.knowledge_estimates.items()
        }
        self.episodic_records = list(self.episodic_records)
        self.task_history = list(self.task_history)
        self.performance_history = {
            str(key): list(values) for key, values in self.performance_history.items()
        }
        self.experience_count = max(0, int(self.experience_count))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation of the learner state."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class AdultComprehensionResult:
    """Return all 15 stage results, evidence provenance, biological traces, and learner state."""

    experience_id: str
    task_goal: Mapping[str, Any]
    current_resources: Mapping[str, Any]
    segmented_units: Mapping[str, Any]
    retrieved_knowledge: Mapping[str, Any]
    chunking: Mapping[str, Any]
    working_memory_load: Mapping[str, Any]
    processing_schedule: Mapping[str, Any]
    language_interpretation: Mapping[str, Any]
    cultural_interpretation: Mapping[str, Any]
    integrated_representation: Mapping[str, Any]
    interference_monitor: Mapping[str, Any]
    adaptive_support: Mapping[str, Any]
    aging_memory: Mapping[str, Any]
    knowledge_update: Mapping[str, Any]
    calibrated_decision: Mapping[str, Any]
    operation_trace: tuple[Mapping[str, Any], ...]
    biological: Mapping[str, Any]
    final_state: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the complete Point 17 result in JSON-compatible form."""
        return _plain(asdict(self))
