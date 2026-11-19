"""Validated contracts for Point 04 retrieval-practice and testing-effect processing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Mapping


def _text(name: str, value: str) -> str:
    """Validate and normalize a non-empty text field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _unit(name: str, value: float) -> float:
    """Validate a finite scalar in the closed unit interval."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _weights(name: str, values: Mapping[str, float], allowed: set[str]) -> dict[str, float]:
    """Validate weights for exactly the named target components or relations."""
    result = {_text(f"{name} key", key): _unit(f"{name}[{key}]", value)
              for key, value in values.items()}
    unknown = set(result) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown keys: {sorted(unknown)}")
    return result


def normalize_utc(value: str | datetime) -> str:
    """Normalize a timezone-aware timestamp to ISO-8601 UTC."""
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be valid ISO-8601") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ValueError("timestamp must be a datetime or ISO-8601 string")
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class LearningTarget:
    """Define the knowledge, relations, answer criterion, and intended retrieval operation."""

    target_id: str
    prompt: str
    components: tuple[str, ...]
    relations: tuple[str, ...] = ()
    component_weights: Mapping[str, float] = field(default_factory=dict)
    relation_weights: Mapping[str, float] = field(default_factory=dict)
    expected_order: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    target_kind: str = "conceptual_explanation"
    context: Mapping[str, Any] = field(default_factory=dict)
    criterion_source: str = "instructor_or_validated_answer_key"
    criterion_reliability: float = 1.0
    initial_encoding_quality: float | None = None
    valid_cue_types: tuple[str, ...] = (
        "definition", "example", "causal", "relational", "diagram", "application"
    )

    def __post_init__(self) -> None:
        """Validate target structure and ensure every scoring weight refers to a known unit."""
        object.__setattr__(self, "target_id", _text("target_id", self.target_id))
        object.__setattr__(self, "prompt", _text("prompt", self.prompt))
        components = tuple(_text("component", item) for item in self.components)
        relations = tuple(_text("relation", item) for item in self.relations)
        if not components:
            raise ValueError("components must contain at least one required content unit")
        if len(set(components)) != len(components) or len(set(relations)) != len(relations):
            raise ValueError("components and relations must not contain duplicates")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "component_weights",
                           _weights("component_weights", self.component_weights, set(components)))
        object.__setattr__(self, "relation_weights",
                           _weights("relation_weights", self.relation_weights, set(relations)))
        order = tuple(_text("expected_order item", item) for item in self.expected_order)
        if order and (set(order) - set(components)):
            raise ValueError("expected_order entries must be target components")
        object.__setattr__(self, "expected_order", order)
        object.__setattr__(self, "conditions",
                           tuple(_text("condition", item) for item in self.conditions))
        object.__setattr__(self, "target_kind", _text("target_kind", self.target_kind))
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "criterion_source", _text("criterion_source", self.criterion_source))
        object.__setattr__(self, "criterion_reliability",
                           _unit("criterion_reliability", self.criterion_reliability))
        if self.initial_encoding_quality is not None:
            object.__setattr__(self, "initial_encoding_quality",
                               _unit("initial_encoding_quality", self.initial_encoding_quality))
        cues = tuple(_text("valid_cue_type", item) for item in self.valid_cue_types)
        if not cues:
            raise ValueError("valid_cue_types cannot be empty")
        object.__setattr__(self, "valid_cue_types", cues)


@dataclass(frozen=True, slots=True)
class RetrievalAttempt:
    """Capture an immutable retrieval attempt before feedback or answer exposure."""

    cue: str
    response: str
    confidence: float
    cue_type: str = "free_recall"
    aid_level: int = 0
    response_time_ms: float = 0.0
    recalled_components: tuple[str, ...] = ()
    inferred_components: tuple[str, ...] = ()
    recalled_relations: tuple[str, ...] = ()
    component_status: Mapping[str, str] = field(default_factory=dict)
    relation_status: Mapping[str, str] = field(default_factory=dict)
    condition_status: Mapping[str, bool] = field(default_factory=dict)
    claim_confidence: Mapping[str, float] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    answer_leakage: float = 0.0
    ambiguity: float = 0.0
    guessed: bool = False

    def __post_init__(self) -> None:
        """Validate cue, response, confidence, evidence labels, and aid metadata."""
        object.__setattr__(self, "cue", _text("cue", self.cue))
        object.__setattr__(self, "response", str(self.response))
        object.__setattr__(self, "confidence", _unit("confidence", self.confidence))
        object.__setattr__(self, "cue_type", _text("cue_type", self.cue_type))
        if self.aid_level < 0:
            raise ValueError("aid_level must be non-negative")
        if not math.isfinite(float(self.response_time_ms)) or self.response_time_ms < 0:
            raise ValueError("response_time_ms must be finite and non-negative")
        object.__setattr__(self, "response_time_ms", float(self.response_time_ms))
        for name in ("recalled_components", "inferred_components", "recalled_relations"):
            object.__setattr__(self, name, tuple(_text(name, item) for item in getattr(self, name)))
        component_labels = {"correct", "missing", "substitution", "uncertain", "contradicted"}
        relation_labels = {"correct", "missing", "reversed", "incorrect", "uncertain"}
        component_status = {str(k): str(v) for k, v in self.component_status.items()}
        relation_status = {str(k): str(v) for k, v in self.relation_status.items()}
        if set(component_status.values()) - component_labels:
            raise ValueError("unsupported component status")
        if set(relation_status.values()) - relation_labels:
            raise ValueError("unsupported relation status")
        object.__setattr__(self, "component_status", component_status)
        object.__setattr__(self, "relation_status", relation_status)
        object.__setattr__(self, "condition_status",
                           {str(k): bool(v) for k, v in self.condition_status.items()})
        object.__setattr__(self, "claim_confidence",
                           {str(k): _unit(f"claim_confidence[{k}]", v)
                            for k, v in self.claim_confidence.items()})
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "answer_leakage",
                           _unit("answer_leakage", self.answer_leakage))
        object.__setattr__(self, "ambiguity", _unit("ambiguity", self.ambiguity))


@dataclass(frozen=True, slots=True)
class Feedback:
    """Represent corrective information while retaining provenance and verification status."""

    correct_representation: str
    explanation: str = ""
    source: str = "validated_answer_criterion"
    source_reliability: float = 1.0
    verified: bool = True
    corrections: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate feedback provenance and keep correction labels explicit."""
        object.__setattr__(self, "correct_representation",
                           _text("correct_representation", self.correct_representation))
        object.__setattr__(self, "explanation", str(self.explanation))
        object.__setattr__(self, "source", _text("source", self.source))
        object.__setattr__(self, "source_reliability",
                           _unit("source_reliability", self.source_reliability))
        object.__setattr__(self, "corrections",
                           {str(k): _text("correction", v) for k, v in self.corrections.items()})


@dataclass(frozen=True, slots=True)
class TestingEffectRequest:
    """Supply a target, first attempt, optional correction, retest, cue variants, and delayed probes."""

    target: LearningTarget
    initial_attempt: RetrievalAttempt
    feedback: Feedback | None = None
    retest_attempt: RetrievalAttempt | None = None
    alternative_attempts: tuple[RetrievalAttempt, ...] = ()
    delayed_attempts: tuple[RetrievalAttempt, ...] = ()
    current_time_utc: str | datetime = "2026-01-01T00:00:00Z"
    delay_after_hours: float = 24.0 * 24.0
    fatigue: float = 0.0
    available_time_minutes: float = 20.0

    def __post_init__(self) -> None:
        """Validate all optional observations and constrain fatigue, delay, and time budget."""
        if not isinstance(self.target, LearningTarget):
            raise ValueError("target must be a LearningTarget")
        if not isinstance(self.initial_attempt, RetrievalAttempt):
            raise ValueError("initial_attempt must be a RetrievalAttempt")
        if self.feedback is not None and not isinstance(self.feedback, Feedback):
            raise ValueError("feedback must be Feedback or None")
        for name in ("retest_attempt",):
            value = getattr(self, name)
            if value is not None and not isinstance(value, RetrievalAttempt):
                raise ValueError(f"{name} must be RetrievalAttempt or None")
        for name in ("alternative_attempts", "delayed_attempts"):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, RetrievalAttempt) for item in values):
                raise ValueError(f"{name} must contain RetrievalAttempt values")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "current_time_utc", normalize_utc(self.current_time_utc))
        if not math.isfinite(float(self.delay_after_hours)) or self.delay_after_hours < 0:
            raise ValueError("delay_after_hours must be finite and non-negative")
        if not math.isfinite(float(self.available_time_minutes)) or self.available_time_minutes <= 0:
            raise ValueError("available_time_minutes must be finite and positive")
        object.__setattr__(self, "fatigue", _unit("fatigue", self.fatigue))


@dataclass
class TargetMemoryState:
    """Maintain cue-specific access, error recurrence, and calibration observations for one target."""

    target_id: str
    successes: int = 0
    failures: int = 0
    accessibility_ema: float = 0.5
    cue_routes: dict[str, dict[str, float]] = field(default_factory=dict)
    error_counts: dict[str, int] = field(default_factory=dict)
    confidence_pairs: list[dict[str, float]] = field(default_factory=list)
    verified_relations: dict[str, float] = field(default_factory=dict)
    attempt_history: list[dict[str, Any]] = field(default_factory=list)
    delayed_accuracy: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly copy of the target-specific learning state."""
        return asdict(self)


@dataclass
class TestingEffectState:
    """Persist target-specific testing-effect estimates across independent processing calls."""

    targets: dict[str, TargetMemoryState] = field(default_factory=dict)
    process_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize all target memory states without exposing mutable internals."""
        return {"targets": {key: value.to_dict() for key, value in self.targets.items()},
                "process_count": self.process_count}


@dataclass(frozen=True, slots=True)
class TestingEffectResult:
    """Capture all 22 ordered testing-effect operations and the resulting learner model."""

    target_id: str
    operation_trace: tuple[Mapping[str, Any], ...]
    initial_assessment: Mapping[str, Any]
    retest_assessment: Mapping[str, Any] | None
    feedback_decision: Mapping[str, Any]
    future_priority: Mapping[str, Any]
    target_state: Mapping[str, Any]
    biological: Mapping[str, Any]
    final_state: Mapping[str, Any]
    output: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Require the complete Point 04 operation registry in source order."""
        expected = [f"P04-{index:02d}" for index in range(1, 23)]
        actual = [str(item.get("operation_id", "")) for item in self.operation_trace]
        if actual != expected:
            raise ValueError("Point 04 results must contain exactly 22 ordered operations")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete retrieval-practice result as plain nested dictionaries."""
        return asdict(self)
