"""Typed contracts for the Point 15 language-specific/domain-general architecture.

Every representation distinguishes linguistic evidence, conceptual knowledge, controlled
reasoning, and social-pragmatic hypotheses. These are computational abstractions, not neural
measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _finite(name: str, value: float) -> float:
    """Convert a numeric value to a finite float or reject it."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _unit(name: str, value: float) -> float:
    """Validate a normalized value in the closed interval [0, 1]."""
    result = _finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(name + " must be in [0, 1]")
    return result


def _plain(value: Any) -> Any:
    """Convert nested dataclasses and containers into JSON-compatible structures."""
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


@dataclass(frozen=True, slots=True)
class LanguageEvent:
    """Describe one spoken-transcript or written-language task and its context."""

    event_id: str
    sensory_stream: str | Sequence[str]
    modality: str = "text"
    linguistic_context: Mapping[str, Any] = field(default_factory=dict)
    lexicon: Mapping[str, Any] = field(default_factory=dict)
    lexical_alternatives: Mapping[str, Sequence[str]] = field(default_factory=dict)
    construction_patterns: Sequence[Mapping[str, Any]] = ()
    conceptual_memory: Mapping[str, Any] = field(default_factory=dict)
    situational_context: Mapping[str, Any] = field(default_factory=dict)
    task_goal: str = "comprehend"
    speaker_model: Mapping[str, Any] = field(default_factory=dict)
    cognitive_state: Mapping[str, Any] = field(default_factory=dict)
    reasoning_rules: Sequence[Mapping[str, Any]] = ()
    expected_next_tokens: Sequence[str] = ()
    predicted_response: str = ""
    observed_response: str = ""
    new_evidence: Mapping[str, Any] = field(default_factory=dict)
    feedback: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identifiers, modality, probabilities, and detach mutable inputs."""
        if not str(self.event_id).strip():
            raise ValueError("event_id must be non-empty")
        if self.modality not in {"speech", "text"}:
            raise ValueError("modality must be 'speech' or 'text'")
        if not isinstance(self.sensory_stream, (str, tuple, list)):
            raise ValueError("sensory_stream must be text or an ordered sequence of text segments")
        if isinstance(self.sensory_stream, (tuple, list)):
            if not all(isinstance(item, str) for item in self.sensory_stream):
                raise ValueError("all sensory_stream segments must be strings")
            object.__setattr__(self, "sensory_stream", tuple(self.sensory_stream))
        else:
            object.__setattr__(self, "sensory_stream", str(self.sensory_stream))
        for name in (
            "linguistic_context", "lexicon", "lexical_alternatives", "conceptual_memory",
            "situational_context", "speaker_model", "cognitive_state", "new_evidence",
            "feedback", "metadata",
        ):
            object.__setattr__(self, name, dict(getattr(self, name)))
        object.__setattr__(self, "construction_patterns", tuple(dict(x) for x in self.construction_patterns))
        object.__setattr__(self, "reasoning_rules", tuple(dict(x) for x in self.reasoning_rules))
        object.__setattr__(self, "expected_next_tokens", tuple(str(x) for x in self.expected_next_tokens))
        if not str(self.task_goal).strip():
            raise ValueError("task_goal must be non-empty")
        object.__setattr__(self, "task_goal", str(self.task_goal).strip().lower())
        for name in ("predicted_response", "observed_response"):
            object.__setattr__(self, name, str(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class LanguageProcessingConfig:
    """Configure bounded model-level weighting, control allocation, and plasticity."""

    sensory_weight: float = 0.65
    context_weight: float = 0.35
    lexical_temperature: float = 0.35
    construction_temperature: float = 0.5
    concept_relevance_threshold: float = 0.15
    control_gamma: float = 6.0
    control_recruitment_threshold: float = 0.5
    pragmatic_threshold: float = 0.35
    revision_threshold: float = 0.35
    learning_rate: float = 0.12
    learning_reliability_threshold: float = 0.65
    repeated_evidence_threshold: int = 2
    max_candidates_per_token: int = 8
    max_revisions: int = 2

    def __post_init__(self) -> None:
        """Validate all rates and thresholds before a processing pipeline is run."""
        for name in (
            "sensory_weight", "context_weight", "concept_relevance_threshold",
            "control_recruitment_threshold", "pragmatic_threshold", "revision_threshold",
            "learning_rate", "learning_reliability_threshold",
        ):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        for name in ("lexical_temperature", "construction_temperature", "control_gamma"):
            value = _finite(name, getattr(self, name))
            if value <= 0:
                raise ValueError(name + " must be positive")
            object.__setattr__(self, name, value)
        if self.max_candidates_per_token < 1 or self.max_revisions < 0:
            raise ValueError("candidate and revision limits must be non-negative (candidate >= 1)")
        if self.repeated_evidence_threshold < 1:
            raise ValueError("repeated_evidence_threshold must be at least 1")


@dataclass
class LanguageLearnerState:
    """Retain gradual linguistic learning without collapsing it into one global memory score."""

    learner_id: str = "default"
    lexical_strengths: dict[str, float] = field(default_factory=dict)
    construction_strengths: dict[str, float] = field(default_factory=dict)
    evidence_counts: dict[str, int] = field(default_factory=dict)
    conceptual_strengths: dict[str, float] = field(default_factory=dict)
    episode_history: list[dict[str, Any]] = field(default_factory=list)
    process_count: int = 0

    def __post_init__(self) -> None:
        """Normalize persistent strengths and ensure evidence counts are non-negative."""
        if not str(self.learner_id).strip():
            raise ValueError("learner_id must be non-empty")
        for name in ("lexical_strengths", "construction_strengths", "conceptual_strengths"):
            values = {str(k): _unit(name, v) for k, v in getattr(self, name).items()}
            setattr(self, name, values)
        self.evidence_counts = {str(k): max(0, int(v)) for k, v in self.evidence_counts.items()}
        self.episode_history = list(self.episode_history)
        if self.process_count < 0:
            raise ValueError("process_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dictionary of the learner's accumulated language state."""
        return _plain(asdict(self))


@dataclass(frozen=True, slots=True)
class LanguageProcessingResult:
    """Return separated language, knowledge, control, reasoning, pragmatic, and task outputs."""

    event_id: str
    linguistic_representation: Mapping[str, Any]
    conceptual_interpretation: Mapping[str, Any]
    control_request: Mapping[str, Any]
    reasoning_result: Mapping[str, Any]
    pragmatic_interpretation: Mapping[str, Any]
    integrated_interpretation: Mapping[str, Any]
    mismatch_and_revision: Mapping[str, Any]
    task_result: Mapping[str, Any]
    learning_update: Mapping[str, Any]
    operation_trace: tuple[Mapping[str, Any], ...]
    biological: Mapping[str, Any]
    final_state: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the complete Point 15 result in JSON-compatible form."""
        return _plain(asdict(self))
