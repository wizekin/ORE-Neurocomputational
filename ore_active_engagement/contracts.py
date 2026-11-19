"""Validated contracts for the Point 01 active-engagement learning model."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _finite(name: str, value: float) -> float:
    """Convert a numeric value to a finite float or reject invalid model input."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit(name: str, value: float) -> float:
    """Validate a probability, reliability, or normalized strength in [0, 1]."""
    result = _finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _required_text(name: str, value: str) -> str:
    """Normalize a required text field and reject blank values."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class LearningSegment:
    """Represent one observed text, speech transcript, diagram caption, or other input segment."""

    segment_id: str
    text: str
    modality: str = "text"
    reliability: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate segment identity, evidence reliability, and modality metadata."""
        object.__setattr__(self, "segment_id", _required_text("segment_id", self.segment_id))
        object.__setattr__(self, "text", _required_text("text", self.text))
        if self.modality not in {"text", "speech_transcript", "diagram_caption", "observation", "other"}:
            raise ValueError("unsupported learning-segment modality")
        object.__setattr__(self, "reliability", _unit("reliability", self.reliability))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return the observed segment while retaining modality and source provenance."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActiveEngagementRequest:
    """Describe one active-learning episode and optional learner-generated outputs."""

    learner_id: str
    goal: str
    material: str | Sequence[str | LearningSegment]
    context: Mapping[str, Any] = field(default_factory=dict)
    task_type: str = "understand_concept"
    learner_explanation: str | None = None
    required_concepts: tuple[str, ...] = ()
    listener_level: str = "novice"
    enable_teaching: bool = True
    enable_imagery: bool = False
    max_questions: int = 4
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize the input stream and validate task, audience, and optional learner text."""
        object.__setattr__(self, "learner_id", _required_text("learner_id", self.learner_id))
        object.__setattr__(self, "goal", _required_text("goal", self.goal))
        if isinstance(self.material, str):
            normalized: tuple[LearningSegment, ...] = (
                LearningSegment("segment-001", self.material),
            )
        else:
            items: list[LearningSegment] = []
            for index, item in enumerate(self.material, start=1):
                if isinstance(item, LearningSegment):
                    items.append(item)
                elif isinstance(item, str):
                    items.append(LearningSegment(f"segment-{index:03d}", item))
                else:
                    raise ValueError("material items must be strings or LearningSegment instances")
            normalized = tuple(items)
        if not normalized:
            raise ValueError("material must contain at least one non-empty segment")
        if len({item.segment_id for item in normalized}) != len(normalized):
            raise ValueError("segment_id values must be unique")
        object.__setattr__(self, "material", normalized)
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.task_type not in {
            "understand_claim", "explain_process", "interpret_diagram",
            "follow_proof", "understand_narrative", "understand_concept",
        }:
            raise ValueError("unsupported task_type")
        if self.learner_explanation is not None:
            object.__setattr__(
                self, "learner_explanation",
                _required_text("learner_explanation", self.learner_explanation),
            )
        object.__setattr__(self, "required_concepts", tuple(
            _required_text("required_concept", item).casefold()
            for item in self.required_concepts
        ))
        if self.listener_level not in {"novice", "intermediate", "expert"}:
            raise ValueError("listener_level must be novice, intermediate, or expert")
        if self.max_questions < 1 or self.max_questions > 20:
            raise ValueError("max_questions must be between 1 and 20")


@dataclass(frozen=True, slots=True)
class ActiveEngagementConfig:
    """Set bounded mathematical weights and stopping criteria for the model."""

    attention_temperature: float = 0.75
    question_cost_weight: float = 0.15
    goal_weight: float = 0.50
    centrality_weight: float = 0.25
    uncertainty_weight: float = 0.25
    fidelity_weight: float = 0.25
    coverage_weight: float = 0.25
    coherence_weight: float = 0.20
    evidence_weight: float = 0.20
    contradiction_penalty: float = 0.10
    uncertainty_penalty: float = 0.10
    learning_rate: float = 0.08
    plasticity_gate: float = 0.75
    minimum_quality: float = 0.55
    maximum_repair_passes: int = 1
    embedding_dimensions: int = 64

    def __post_init__(self) -> None:
        """Reject non-finite, negative, or out-of-range configuration values."""
        if _finite("attention_temperature", self.attention_temperature) <= 0:
            raise ValueError("attention_temperature must be positive")
        for name in (
            "question_cost_weight", "goal_weight", "centrality_weight",
            "uncertainty_weight", "fidelity_weight", "coverage_weight",
            "coherence_weight", "evidence_weight", "contradiction_penalty",
            "uncertainty_penalty", "learning_rate", "plasticity_gate",
            "minimum_quality",
        ):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        if self.maximum_repair_passes < 0 or self.maximum_repair_passes > 5:
            raise ValueError("maximum_repair_passes must be between 0 and 5")
        if self.embedding_dimensions < 8 or self.embedding_dimensions > 4096:
            raise ValueError("embedding_dimensions must be between 8 and 4096")


@dataclass
class ActiveEngagementState:
    """Maintain learner-specific association strengths and previously encoded episodes."""

    learner_id: str
    associative_weights: dict[str, float] = field(default_factory=dict)
    episode_memory: list[dict[str, Any]] = field(default_factory=list)
    comprehension_history: list[float] = field(default_factory=list)
    process_count: int = 0

    def __post_init__(self) -> None:
        """Validate learner identity and normalize stored association strengths."""
        self.learner_id = _required_text("learner_id", self.learner_id)
        self.associative_weights = {
            str(key): _unit("associative_weight", value)
            for key, value in self.associative_weights.items()
        }
        self.episode_memory = list(self.episode_memory)
        self.comprehension_history = [
            _unit("comprehension_history", value)
            for value in self.comprehension_history
        ]
        if self.process_count < 0:
            raise ValueError("process_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return persistent learner state as a serialization-friendly mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActiveEngagementResult:
    """Capture the explanation, uncertainty, memory episode, biology, and 17-step trace."""

    learner_id: str
    operation_trace: tuple[Mapping[str, Any], ...]
    questions: tuple[Mapping[str, Any], ...]
    candidate_answers: tuple[Mapping[str, Any], ...]
    explanation: Mapping[str, Any]
    elaboration: Mapping[str, Any]
    quality: Mapping[str, Any]
    repair: Mapping[str, Any]
    teaching: Mapping[str, Any]
    imagery: Mapping[str, Any]
    episode_memory: Mapping[str, Any]
    plasticity: Mapping[str, Any]
    comprehension: Mapping[str, Any]
    closure: Mapping[str, Any]
    biological: Mapping[str, Any]
    final_state: Mapping[str, Any]
    output: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Require one source-ordered trace record for each of the 17 algorithmic stages."""
        _required_text("learner_id", self.learner_id)
        if len(self.operation_trace) != 17:
            raise ValueError("Point 01 results must contain exactly 17 operation trace records")
        expected = [f"P01-{index:02d}" for index in range(1, 18)]
        actual = [str(item.get("operation_id", "")) for item in self.operation_trace]
        if actual != expected:
            raise ValueError("operation_trace must preserve the Point 01 source order")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete learning result in a plain nested mapping."""
        return asdict(self)
