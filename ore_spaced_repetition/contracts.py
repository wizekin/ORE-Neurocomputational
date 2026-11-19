"""Validated contracts for Point 03 spaced-repetition scheduling and memory-state estimation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Mapping


def _text(name: str, value: str) -> str:
    """Return normalized non-empty text or reject an invalid field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _unit(name: str, value: float) -> float:
    """Validate a finite probability or normalized score in [0, 1]."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _positive(name: str, value: float, allow_zero: bool = False) -> float:
    """Validate a finite positive duration or parameter."""
    number = float(value)
    valid = number >= 0 if allow_zero else number > 0
    if not math.isfinite(number) or not valid:
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {comparison}")
    return number


def normalize_utc(value: str | datetime) -> str:
    """Normalize a timezone-aware timestamp to ISO-8601 UTC."""
    if isinstance(value, str):
        candidate = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ValueError("timestamp must be a valid ISO-8601 value") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ValueError("timestamp must be a datetime or ISO-8601 string")
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class LearningItem:
    """Represent one learnable item while retaining components, source, and learning context."""

    item_id: str
    content: str
    components: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)
    difficulty: float = 0.5
    target_retention_hours: float = 24.0 * 30.0
    encoding_measurements: Mapping[str, float] = field(default_factory=dict)
    source: str = "unspecified_learning_source"
    created_at_utc: str | datetime | None = None

    def __post_init__(self) -> None:
        """Validate item content, learning measures, horizon, and temporal origin."""
        object.__setattr__(self, "item_id", _text("item_id", self.item_id))
        object.__setattr__(self, "content", _text("content", self.content))
        object.__setattr__(self, "components", tuple(_text("component", x) for x in self.components))
        object.__setattr__(self, "relations", tuple(_text("relation", x) for x in self.relations))
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "difficulty", _unit("difficulty", self.difficulty))
        object.__setattr__(self, "target_retention_hours",
                           _positive("target_retention_hours", self.target_retention_hours))
        measurements = {
            _text("measurement", key): _unit(f"encoding_measurements[{key}]", value)
            for key, value in self.encoding_measurements.items()
        }
        allowed = {"attention", "comprehension", "relational_organization", "meaningful_practice"}
        unknown = set(measurements) - allowed
        if unknown:
            raise ValueError(f"unsupported encoding measurement(s): {sorted(unknown)}")
        object.__setattr__(self, "encoding_measurements", measurements)
        object.__setattr__(self, "source", _text("source", self.source))
        if self.created_at_utc is not None:
            object.__setattr__(self, "created_at_utc", normalize_utc(self.created_at_utc))


@dataclass(frozen=True, slots=True)
class ReviewObservation:
    """Record one delayed retrieval attempt before/after feedback with separate quality channels."""

    occurred_at_utc: str | datetime
    outcome: str
    accuracy: float
    completeness: float = 1.0
    relation_accuracy: float = 1.0
    fluency: float = 0.5
    effort: float = 0.5
    response_mode: str = "free_recall"
    hint_level: float = 0.0
    feedback_provided: bool = False
    learner_response: str | None = None
    corrected_response: str | None = None
    subjective_difficulty: float | None = None

    def __post_init__(self) -> None:
        """Validate review timestamp, performance dimensions, and cue/feedback metadata."""
        object.__setattr__(self, "occurred_at_utc", normalize_utc(self.occurred_at_utc))
        if self.outcome not in {"success", "partial", "failure", "corrected_relearning", "recognition_only"}:
            raise ValueError("unsupported review outcome")
        for name in ("accuracy", "completeness", "relation_accuracy", "fluency",
                     "effort", "hint_level"):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        if self.response_mode not in {"free_recall", "cued_recall", "recognition", "procedure", "explanation"}:
            raise ValueError("unsupported response_mode")
        if self.response_mode == "recognition" and self.outcome == "success":
            object.__setattr__(self, "outcome", "recognition_only")
        if self.learner_response is not None:
            object.__setattr__(self, "learner_response", str(self.learner_response))
        if self.corrected_response is not None:
            object.__setattr__(self, "corrected_response", str(self.corrected_response))
        if self.subjective_difficulty is not None:
            object.__setattr__(self, "subjective_difficulty",
                               _unit("subjective_difficulty", self.subjective_difficulty))
        if self.outcome == "corrected_relearning" and not self.feedback_provided:
            raise ValueError("corrected_relearning requires feedback_provided=True")


@dataclass(frozen=True, slots=True)
class OfflineConsolidationObservation:
    """Describe optional offline/sleep context without treating sleep as a guaranteed memory boost."""

    observed: bool = False
    duration_hours: float = 0.0
    quality: float = 0.5
    interference: float = 0.0
    evidence_confidence: float = 0.0

    def __post_init__(self) -> None:
        """Validate optional offline context and retain explicit observation status."""
        object.__setattr__(self, "duration_hours", _positive("duration_hours", self.duration_hours, True))
        for name in ("quality", "interference", "evidence_confidence"):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        if not self.observed and self.duration_hours > 0:
            raise ValueError("duration_hours requires observed=True")


@dataclass(frozen=True, slots=True)
class SpacedRepetitionConfig:
    """Configure memory models and practical scheduling policies; defaults are tunable hypotheses."""

    forgetting_model: str = "exponential"
    minimum_interval_hours: float = 0.25
    maximum_interval_hours: float = 24.0 * 90.0
    initial_interval_fraction: float = 0.05
    target_recall_low: float = 0.55
    target_recall_high: float = 0.85
    target_recall_probability: float = 0.70
    learning_rate: float = 0.12
    base_stability_hours: float = 24.0
    minimum_stability_hours: float = 0.5
    maximum_stability_hours: float = 24.0 * 365.0
    stability_uncertainty_initial: float = 0.45
    uncertainty_floor: float = 0.05
    urgency_window_hours: float = 24.0
    calibration_window: int = 100
    maintenance_successes_required: int = 3
    maintenance_min_span_hours: float = 24.0 * 7.0
    maintenance_accuracy_threshold: float = 0.9
    consolidation_learning_rate: float = 0.02

    def __post_init__(self) -> None:
        """Reject invalid scheduler ranges and non-finite model parameters."""
        if self.forgetting_model not in {"exponential", "power_law"}:
            raise ValueError("forgetting_model must be exponential or power_law")
        for name in ("minimum_interval_hours", "maximum_interval_hours",
                     "base_stability_hours", "minimum_stability_hours",
                     "maximum_stability_hours", "urgency_window_hours"):
            _positive(name, getattr(self, name))
        if self.minimum_interval_hours > self.maximum_interval_hours:
            raise ValueError("minimum_interval_hours must not exceed maximum_interval_hours")
        if self.minimum_stability_hours > self.maximum_stability_hours:
            raise ValueError("minimum_stability_hours must not exceed maximum_stability_hours")
        if not 0 < self.initial_interval_fraction <= 1:
            raise ValueError("initial_interval_fraction must be in (0, 1]")
        for name in ("target_recall_low", "target_recall_high",
                     "target_recall_probability", "learning_rate",
                     "stability_uncertainty_initial", "uncertainty_floor",
                     "maintenance_accuracy_threshold", "consolidation_learning_rate"):
            _unit(name, getattr(self, name))
        if self.target_recall_low >= self.target_recall_high:
            raise ValueError("target_recall_low must be less than target_recall_high")
        if not self.target_recall_low < self.target_recall_probability < self.target_recall_high:
            raise ValueError("target_recall_probability must lie inside the target recall band")
        if self.calibration_window < 1 or self.maintenance_successes_required < 2:
            raise ValueError("calibration_window must be positive and maintenance needs at least two successes")
        _positive("maintenance_min_span_hours", self.maintenance_min_span_hours, True)


@dataclass
class MemoryRecord:
    """Maintain one item's learned state, review timeline, and model-calibration history."""

    item: LearningItem
    created_at_utc: str
    last_learning_at_utc: str
    encoding_quality: float
    encoding_uncertainty: float
    stability_hours: float
    memory_accessibility: float
    memory_uncertainty: float
    next_interval_hours: float
    next_due_at_utc: str
    review_history: list[dict[str, Any]] = field(default_factory=list)
    consolidation_history: list[dict[str, Any]] = field(default_factory=list)
    predicted_outcomes: list[dict[str, float]] = field(default_factory=list)
    provisional_maintenance: bool = False
    last_action: str = "initial_review_scheduled"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly copy of the memory item's complete model state."""
        return asdict(self)


@dataclass
class SpacedRepetitionState:
    """Persist learner-specific memory items and scheduler calibration observations."""

    learner_id: str
    items: dict[str, MemoryRecord] = field(default_factory=dict)
    process_count: int = 0

    def __post_init__(self) -> None:
        """Validate learner identity and item-map key consistency."""
        self.learner_id = _text("learner_id", self.learner_id)
        for key, value in self.items.items():
            if key != value.item.item_id:
                raise ValueError("memory-state item keys must match item IDs")

    def to_dict(self) -> dict[str, Any]:
        """Return all memory records in serialization-friendly form."""
        return {"learner_id": self.learner_id,
                "items": {key: value.to_dict() for key, value in self.items.items()},
                "process_count": self.process_count}


@dataclass(frozen=True, slots=True)
class SpacedRepetitionRequest:
    """Supply one learning item, optional delayed review outcome, time, and offline context."""

    learner_id: str
    item: LearningItem
    current_time_utc: str | datetime
    review_observation: ReviewObservation | None = None
    offline_consolidation: OfflineConsolidationObservation = field(
        default_factory=OfflineConsolidationObservation
    )

    def __post_init__(self) -> None:
        """Validate learner identity, timestamps, and optional review observations."""
        object.__setattr__(self, "learner_id", _text("learner_id", self.learner_id))
        object.__setattr__(self, "current_time_utc", normalize_utc(self.current_time_utc))
        if not isinstance(self.item, LearningItem):
            raise ValueError("item must be a LearningItem")
        if self.review_observation is not None and not isinstance(self.review_observation, ReviewObservation):
            raise ValueError("review_observation must be a ReviewObservation")
        if not isinstance(self.offline_consolidation, OfflineConsolidationObservation):
            raise ValueError("offline_consolidation must be an OfflineConsolidationObservation")
        if self.item.created_at_utc and self.item.created_at_utc > self.current_time_utc:
            raise ValueError("item creation time cannot be in the future")


@dataclass(frozen=True, slots=True)
class SpacedRepetitionResult:
    """Capture one item's scheduled state and exactly twelve source-ordered operation records."""

    learner_id: str
    item_id: str
    operation_trace: tuple[Mapping[str, Any], ...]
    memory_record: Mapping[str, Any]
    review_event: Mapping[str, Any] | None
    calibration: Mapping[str, Any]
    biological: Mapping[str, Any]
    final_state: Mapping[str, Any]
    output: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Require the complete Point 03 operation sequence."""
        expected = [f"P03-{index:02d}" for index in range(1, 13)]
        actual = [str(item.get("operation_id", "")) for item in self.operation_trace]
        if actual != expected:
            raise ValueError("Point 03 results must contain exactly 12 ordered operations")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete scheduler result as nested dictionaries."""
        return asdict(self)
