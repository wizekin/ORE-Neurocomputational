"""Typed contracts for Point 18 attentional-failure processing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _unit_interval(name: str, value: float) -> float:
    """Return a finite probability-like model value in the inclusive unit interval."""
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return number


@dataclass
class InformationRequirement:
    """Describe a task-relevant content element and its current evidence coverage."""

    key: str
    importance: float = 1.0
    confidence: float = 0.0
    critical: bool = False
    depends_on: tuple[str, ...] = ()
    evidence: Any = None

    def __post_init__(self) -> None:
        """Validate identity, importance, confidence, and dependency references."""
        if not self.key.strip():
            raise ValueError("requirement key cannot be blank")
        if not math.isfinite(float(self.importance)) or self.importance < 0.0:
            raise ValueError("requirement importance must be finite and non-negative")
        self.confidence = _unit_interval("requirement confidence", self.confidence)
        self.depends_on = tuple(str(key) for key in self.depends_on if str(key))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible requirement snapshot."""
        return asdict(self)


@dataclass
class AttentionalState:
    """Maintain task-sensitive attentional estimates and revisable selection history."""

    alertness: float = 0.75
    fatigue: float = 0.0
    distraction: float = 0.0
    stability: float = 0.75
    lapse_probability: float = 0.0
    attention_history: list[float] = field(default_factory=list)
    lapse_history: list[dict[str, Any]] = field(default_factory=list)
    selection_history: dict[str, float] = field(default_factory=dict)
    context_policy: dict[str, dict[str, float]] = field(default_factory=dict)
    last_verified_position: int = -1
    experience_count: int = 0

    def __post_init__(self) -> None:
        """Validate state estimates and normalize mutable history containers."""
        for name in ("alertness", "fatigue", "distraction", "stability", "lapse_probability"):
            setattr(self, name, _unit_interval(name, getattr(self, name)))
        self.attention_history = [_unit_interval("attention history", x) for x in self.attention_history]
        if self.last_verified_position < -1:
            raise ValueError("last_verified_position must be -1 or greater")
        if self.experience_count < 0:
            raise ValueError("experience_count cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached state snapshot suitable for traces and JSON serialization."""
        return asdict(self)


@dataclass(frozen=True)
class AttentionalConfig:
    """Expose task-calibratable parameters rather than treating them as neural constants."""

    relevance_weight: float = 0.45
    urgency_weight: float = 0.20
    salience_weight: float = 0.15
    novelty_weight: float = 0.10
    history_weight: float = 0.10
    cost_weight: float = 0.20
    fatigue_effect: float = 0.50
    distraction_effect: float = 0.60
    lapse_threshold: float = 0.65
    lapse_hysteresis_samples: int = 2
    encoding_threshold: float = 0.55
    requirement_threshold: float = 0.75
    release_confidence_threshold: float = 0.75
    release_coverage_threshold: float = 0.85
    attention_window: int = 8
    attention_decay: float = 0.80
    working_memory_budget: float = 8.0
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        """Reject invalid model settings before the processing pipeline starts."""
        unit_fields = (
            "relevance_weight", "urgency_weight", "salience_weight", "novelty_weight",
            "history_weight", "cost_weight", "fatigue_effect", "distraction_effect",
            "lapse_threshold", "encoding_threshold", "requirement_threshold",
            "release_confidence_threshold", "release_coverage_threshold", "attention_decay",
        )
        for name in unit_fields:
            _unit_interval(name, getattr(self, name))
        if self.lapse_hysteresis_samples < 1 or self.attention_window < 1:
            raise ValueError("window and hysteresis sample counts must be positive")
        if not math.isfinite(self.working_memory_budget) or self.working_memory_budget <= 0:
            raise ValueError("working_memory_budget must be finite and positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")


@dataclass
class AttentionalExperience:
    """Package one task, information stream, competing inputs, and observable state metrics."""

    experience_id: str
    task_goal: str
    required_information: Sequence[InformationRequirement | Mapping[str, Any]]
    incoming_items: Sequence[Mapping[str, Any]]
    metrics: Mapping[str, float] = field(default_factory=dict)
    competing_inputs: Sequence[Mapping[str, Any]] = ()
    source_items: Sequence[Mapping[str, Any]] = ()
    available_recovery: Mapping[str, bool] = field(default_factory=dict)
    context_key: str = "default"
    goal_vector: Mapping[str, float] = field(default_factory=dict)
    feedback: Mapping[str, Any] = field(default_factory=dict)
    safety_critical: bool = False
    allow_partial_release: bool = True

    def __post_init__(self) -> None:
        """Validate identifiers and convert requirement mappings into typed contracts."""
        if not self.experience_id.strip():
            raise ValueError("experience_id cannot be blank")
        if not self.task_goal.strip():
            raise ValueError("task_goal cannot be blank")
        normalized: list[InformationRequirement] = []
        for item in self.required_information:
            normalized.append(
                item if isinstance(item, InformationRequirement)
                else InformationRequirement(**dict(item))
            )
        keys = [item.key for item in normalized]
        if len(keys) != len(set(keys)):
            raise ValueError("required information keys must be unique")
        self.required_information = tuple(normalized)
        self.incoming_items = tuple(dict(item) for item in self.incoming_items)
        self.competing_inputs = tuple(dict(item) for item in self.competing_inputs)
        self.source_items = tuple(dict(item) for item in self.source_items)
        self.metrics = {str(k): float(v) for k, v in self.metrics.items()}
        self.goal_vector = {str(k): float(v) for k, v in self.goal_vector.items()}
        self.available_recovery = {str(k): bool(v) for k, v in self.available_recovery.items()}
        self.feedback = dict(self.feedback)
        for item in (*self.incoming_items, *self.competing_inputs, *self.source_items):
            if "key" in item and not str(item["key"]).strip():
                raise ValueError("item keys cannot be blank")

    def to_dict(self) -> dict[str, Any]:
        """Return the event without sharing mutable nested mappings with the caller."""
        return {
            "experience_id": self.experience_id,
            "task_goal": self.task_goal,
            "required_information": [x.to_dict() for x in self.required_information],
            "incoming_items": [dict(x) for x in self.incoming_items],
            "metrics": dict(self.metrics),
            "competing_inputs": [dict(x) for x in self.competing_inputs],
            "source_items": [dict(x) for x in self.source_items],
            "available_recovery": dict(self.available_recovery),
            "context_key": self.context_key,
            "goal_vector": dict(self.goal_vector),
            "feedback": dict(self.feedback),
            "safety_critical": self.safety_critical,
            "allow_partial_release": self.allow_partial_release,
        }


@dataclass(frozen=True)
class AttentionalResult:
    """Collect all 17 outputs, biological activity, provenance, and final state."""

    experience_id: str
    task_requirements: Mapping[str, Any]
    state_estimate: Mapping[str, Any]
    competing_input_assessment: Mapping[str, Any]
    priority_map: Mapping[str, Any]
    sensory_bias: Mapping[str, Any]
    working_memory_gate: Mapping[str, Any]
    sustained_attention: Mapping[str, Any]
    lapse_event: Mapping[str, Any]
    missing_information: Mapping[str, Any]
    completeness_audit: Mapping[str, Any]
    recovery_plan: Mapping[str, Any]
    reoriented_task: Mapping[str, Any]
    reencoded_representation: Mapping[str, Any]
    repair_verification: Mapping[str, Any]
    future_attention_policy: Mapping[str, Any]
    preserved_uncertainty: Mapping[str, Any]
    release_decision: Mapping[str, Any]
    operation_trace: tuple[Mapping[str, Any], ...]
    biological: Mapping[str, Any]
    final_state: Mapping[str, Any]
