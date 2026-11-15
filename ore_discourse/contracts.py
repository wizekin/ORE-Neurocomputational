"""Typed request, configuration, and result contracts for ORE discourse processing.

The contracts keep caller-provided facts separate from heuristic inferences so
that downstream systems can inspect provenance and uncertainty.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _validate_feature_map(name: str, values: Mapping[str, Any]) -> None:
    """Validate a sparse feature map whose values are finite non-negative strengths."""
    for key, raw in values.items():
        value = float(raw)
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings")
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} values must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DiscourseRequest:
    """One ordered discourse sample with optional explicit linguistic and world evidence.

    Sentences are the surface discourse. Propositions, entity records, common ground,
    speaker beliefs, temporal evidence and prosody are optional caller-supplied
    evidence. Missing evidence is left unknown rather than silently invented.
    """

    sequence_id: str
    sentences: Sequence[str] | None = None
    propositions: Sequence[Mapping[str, Any]] = ()
    entities: Sequence[Mapping[str, Any]] = ()
    common_ground: Mapping[str, Any] = field(default_factory=dict)
    speaker_model: Mapping[str, Any] = field(default_factory=dict)
    prosody: Mapping[int, Mapping[str, float]] = field(default_factory=dict)
    temporal_evidence: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    language: str = "en"
    source_reference: str | None = None

    def __post_init__(self) -> None:
        """Reject malformed samples before any state or neural simulation is created."""
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be a non-empty string")
        if self.sentences is None and not self.propositions:
            raise ValueError("provide sentences, propositions, or both")
        if self.sentences is not None:
            if any(not isinstance(item, str) or not item.strip() for item in self.sentences):
                raise ValueError("sentences must contain non-empty strings")
            object.__setattr__(self, "sentences", tuple(self.sentences))
        object.__setattr__(self, "propositions", tuple(dict(item) for item in self.propositions))
        object.__setattr__(self, "entities", tuple(dict(item) for item in self.entities))
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("language must be a non-empty identifier")
        if not isinstance(self.common_ground, Mapping) or not isinstance(self.speaker_model, Mapping):
            raise TypeError("common_ground and speaker_model must be mappings")
        if not isinstance(self.context, Mapping):
            raise TypeError("context must be a mapping")
        for index, features in self.prosody.items():
            if not isinstance(index, int) or index < 0 or not isinstance(features, Mapping):
                raise ValueError("prosody must map non-negative sentence indices to feature mappings")
            _validate_feature_map(f"prosody[{index}]", features)
        for index, evidence in self.temporal_evidence.items():
            if not isinstance(index, int) or index < 0 or not isinstance(evidence, Mapping):
                raise ValueError("temporal_evidence must map non-negative sentence indices to mappings")


@dataclass(frozen=True, slots=True)
class DiscourseConfig:
    """Bounded inference and LIF reference parameters.

    Values are engineering settings for reproducible experiments, not measured
    physiological constants or empirically calibrated language probabilities.
    """

    max_sentences: int = 100
    max_entities: int = 500
    memory_decay: float = 0.08
    salience_recency_gain: float = 0.55
    salience_frequency_gain: float = 0.30
    salience_topic_gain: float = 0.15
    coherence_entity_gain: float = 0.35
    coherence_topic_gain: float = 0.30
    coherence_relation_gain: float = 0.20
    coherence_consistency_gain: float = 0.15
    intention_temperature: float = 0.80
    irony_threshold: float = 0.58
    request_threshold: float = 0.52
    recurrent_damping: float = 0.35
    simulate_biology: bool = True
    population_size: int = 12
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
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        """Validate every numerical control and the configured resource limits."""
        if self.max_sentences < 1 or self.max_entities < 1:
            raise ValueError("sentence and entity limits must be positive")
        if self.population_size < 2 or self.steps_per_operation < 1:
            raise ValueError("population_size must be >= 2 and steps_per_operation positive")
        for name in (
            "memory_decay", "salience_recency_gain", "salience_frequency_gain",
            "salience_topic_gain", "coherence_entity_gain", "coherence_topic_gain",
            "coherence_relation_gain", "coherence_consistency_gain", "request_threshold",
            "irony_threshold", "recurrent_damping",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in (
            "intention_temperature", "dt_ms", "membrane_tau_ms",
            "membrane_resistance_mohm", "current_gain_pa", "epsilon",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.refractory_ms < 0.0 or self.recurrent_gain_pa < 0.0 or self.inhibition_gain_pa < 0.0:
            raise ValueError("refractory and recurrent/inhibitory gains must be non-negative")
        if not self.rest_mv <= self.reset_mv < self.threshold_mv:
            raise ValueError("require rest_mv <= reset_mv < threshold_mv")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible copy of the configuration."""
        return asdict(self)


@dataclass(slots=True)
class DiscourseResult:
    """Inspectable output with distinct context, pragmatic, situation, and time tracks."""

    sequence_id: str
    sentences: list[str]
    propositions: list[dict[str, Any]]
    discourse_context: dict[str, Any]
    proposition_links: list[dict[str, Any]]
    speaker_intentions: list[dict[str, Any]]
    indirect_requests: list[dict[str, Any]]
    irony: list[dict[str, Any]]
    situation_model: dict[str, Any]
    temporal_relations: list[dict[str, Any]]
    coherence: dict[str, Any]
    biological: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result without leaking NumPy objects or non-finite numbers."""
        def clean(value: Any) -> Any:
            """Recursively convert nested values into JSON-compatible representations."""
            if isinstance(value, Mapping):
                return {str(key): clean(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [clean(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return repr(value)

        return clean({
            "schema": "ore.discourse.eight_stage.v2",
            "sequence_id": self.sequence_id,
            "sentences": self.sentences,
            "propositions": self.propositions,
            "discourse_context": self.discourse_context,
            "proposition_links": self.proposition_links,
            "speaker_intentions": self.speaker_intentions,
            "indirect_requests": self.indirect_requests,
            "irony": self.irony,
            "situation_model": self.situation_model,
            "temporal_relations": self.temporal_relations,
            "coherence": self.coherence,
            "biological": self.biological,
            "operation_trace": self.operation_trace,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        })
