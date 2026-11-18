"""Validated data contracts for novelty-modulated hippocampal consolidation.

All numerical variables are explicit model inputs or estimates. They are not claimed to be
direct observations of molecular state or literal arithmetic performed by neurons.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _finite_vector(name: str, values: Sequence[float]) -> tuple[float, ...]:
    """Convert a non-empty sequence to finite floating-point values."""
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} values must be finite")
    return result


def _unit(name: str, value: float) -> float:
    """Validate and return a finite scalar in the closed unit interval."""
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _nonnegative(name: str, value: float) -> float:
    """Validate and return a finite non-negative scalar."""
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _jsonable(value: Any) -> Any:
    """Convert dataclasses and nested values into JSON-compatible primitives."""
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _patterns(name: str, values: Sequence[Sequence[float]], width: int) -> tuple[tuple[float, ...], ...]:
    """Validate a collection of same-width feature patterns."""
    result = tuple(_finite_vector(f"{name}[{index}]", row) for index, row in enumerate(values))
    if any(len(row) != width for row in result):
        raise ValueError(f"every {name} vector must have width {width}")
    return result


@dataclass(slots=True)
class MemoryTrace:
    """A recently encoded hippocampal episode with labile synaptic state.

    The feature vector is an abstract hippocampal population code. Active synapses, their
    signed early-plasticity changes, tag eligibility, persistence, and cortical integration
    are tracked separately so that improved retrieval expression is not confused with a
    durable latent trace. Numerical fields are simulation variables, not biological units.
    """

    trace_id: str
    hippocampal_pattern: Sequence[float]
    encoding_time_ms: float
    active_synapses: Mapping[str, float]
    synaptic_changes: Mapping[str, float] = field(default_factory=dict)
    synaptic_tags: Mapping[str, float] = field(default_factory=dict)
    tag_times_ms: Mapping[str, float] = field(default_factory=dict)
    synaptic_persistence: Mapping[str, float] = field(default_factory=dict)
    synapse_feature_indices: Mapping[str, int] = field(default_factory=dict)
    pattern_support: Sequence[float] = ()
    hippocampal_persistence: float = 0.25
    cortical_integration: float = 0.10
    episodic_detail: float = 0.75
    semantic_structure: float = 0.15
    interference_risk: float = 0.0
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize the episode representation and validate every synapse-level state."""
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise ValueError("trace_id must be a non-empty string")
        pattern = _finite_vector("hippocampal_pattern", self.hippocampal_pattern)
        encoding_time = _nonnegative("encoding_time_ms", self.encoding_time_ms)
        active = {str(key): _unit(f"active_synapses[{key}]", value)
                  for key, value in self.active_synapses.items()}
        if not active:
            raise ValueError("active_synapses must contain at least one synapse")
        if any(not key for key in active):
            raise ValueError("synapse identifiers must be non-empty strings")

        changes = {str(key): float(value) for key, value in self.synaptic_changes.items()}
        if any(not math.isfinite(value) for value in changes.values()):
            raise ValueError("synaptic_changes values must be finite")
        changes = {key: changes.get(key, 0.05 * activity) for key, activity in active.items()}

        supplied_tags = bool(self.synaptic_tags)
        tags = {str(key): _unit(f"synaptic_tags[{key}]", value)
                for key, value in self.synaptic_tags.items()}
        # A missing tag map means the encoding-active synapses established initial tags.
        # An explicit map, including explicit zeros, is respected as supplied.
        tags = ({key: active[key] for key in active} if not supplied_tags
                else {key: tags.get(key, 0.0) for key in active})
        tag_times = {str(key): _nonnegative(f"tag_times_ms[{key}]", value)
                     for key, value in self.tag_times_ms.items()}
        tag_times = {key: tag_times.get(key, encoding_time) for key in active}
        persistence = {str(key): _unit(f"synaptic_persistence[{key}]", value)
                       for key, value in self.synaptic_persistence.items()}
        persistence = {key: persistence.get(key, 0.15) for key in active}

        indices = {str(key): int(value) for key, value in self.synapse_feature_indices.items()}
        if indices and set(indices) - set(active):
            raise ValueError("synapse_feature_indices contains an unknown synapse")
        if not indices:
            ordered = list(active)
            indices = {key: index % len(pattern) for index, key in enumerate(ordered)}
        if any(index < 0 or index >= len(pattern) for index in indices.values()):
            raise ValueError("synapse feature indices must address the hippocampal pattern")

        support = tuple(float(value) for value in self.pattern_support) if self.pattern_support else tuple(
            1.0 for _ in pattern
        )
        if len(support) != len(pattern) or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in support
        ):
            raise ValueError("pattern_support must match the pattern width and contain unit values")

        for name in (
            "hippocampal_persistence", "cortical_integration", "episodic_detail",
            "semantic_structure", "interference_risk",
        ):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        object.__setattr__(self, "hippocampal_pattern", pattern)
        object.__setattr__(self, "encoding_time_ms", encoding_time)
        object.__setattr__(self, "active_synapses", active)
        object.__setattr__(self, "synaptic_changes", changes)
        object.__setattr__(self, "synaptic_tags", tags)
        object.__setattr__(self, "tag_times_ms", tag_times)
        object.__setattr__(self, "synaptic_persistence", persistence)
        object.__setattr__(self, "synapse_feature_indices", indices)
        object.__setattr__(self, "pattern_support", support)
        self.provenance = [dict(item) for item in self.provenance]

    def to_dict(self) -> dict[str, Any]:
        """Return the complete trace state, including synapse-level provenance."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class PostLearningEvent:
    """A sensory or behavioural event occurring after an episode has been encoded.

    Features describe the event in the same feature space as prior-knowledge prototypes.
    Timestamp is relative to the simulation clock in milliseconds. Event intensity is kept
    separate from novelty because a strong but familiar event need not be novel.
    """

    event_id: str
    timestamp_ms: float
    features: Sequence[float]
    sensory_intensity: float = 0.5
    reward_value: float = 0.0

    def __post_init__(self) -> None:
        """Validate event identity, temporal order fields, feature vector, and scalar inputs."""
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        object.__setattr__(self, "timestamp_ms", _nonnegative("timestamp_ms", self.timestamp_ms))
        object.__setattr__(self, "features", _finite_vector("features", self.features))
        object.__setattr__(self, "sensory_intensity", _unit("sensory_intensity", self.sensory_intensity))
        object.__setattr__(self, "reward_value", _unit("reward_value", self.reward_value))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the supplied post-learning event without inferring hidden neural values."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class InternalState:
    """Prior knowledge and motivational context used to interpret post-learning events.

    Prior patterns are model prototypes, not a complete semantic memory. A missing prototype
    set makes novelty classification uncertain rather than automatically calling the event
    distinct novelty.
    """

    prior_knowledge_patterns: Sequence[Sequence[float]] = ()
    predicted_features: Sequence[float] | None = None
    goal_relevance: float = 0.5
    arousal: float = 0.5
    motivational_value: float = 0.5
    reward_expectancy: float = 0.5
    stress: float = 0.0

    def __post_init__(self) -> None:
        """Validate feature prototypes and normalized motivational-state variables."""
        predicted = None if self.predicted_features is None else _finite_vector(
            "predicted_features", self.predicted_features
        )
        width = len(predicted) if predicted is not None else (
            len(self.prior_knowledge_patterns[0]) if self.prior_knowledge_patterns else None
        )
        patterns = tuple(_finite_vector(f"prior_knowledge_patterns[{i}]", row)
                         for i, row in enumerate(self.prior_knowledge_patterns))
        if width is not None and any(len(row) != width for row in patterns):
            raise ValueError("prior-knowledge patterns must use one common feature width")
        if predicted is not None and any(len(row) != len(predicted) for row in patterns):
            raise ValueError("predicted_features and prior-knowledge patterns must have equal width")
        object.__setattr__(self, "predicted_features", predicted)
        object.__setattr__(self, "prior_knowledge_patterns", patterns)
        for name in ("goal_relevance", "arousal", "motivational_value", "reward_expectancy", "stress"):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        """Return prior-knowledge and current-state inputs for reproducible simulations."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class SleepEvidence:
    """Supplied sleep-state and replay proxies for parallel cortical systems consolidation.

    Rhythm strengths and replay scores are normalized evidence summaries. The processor does
    not fabricate slow oscillations, spindles, or sharp-wave ripples when they are absent.
    """

    sleep_state: str = "wake"
    state_confidence: float = 1.0
    slow_oscillation_strength: float = 0.0
    spindle_strength: float = 0.0
    ripple_strength: float = 0.0
    replay_strength: float = 0.0
    coordination_strength: float = 0.0
    prior_knowledge_support: float = 0.5

    def __post_init__(self) -> None:
        """Normalize sleep-state labels and validate supplied rhythm/replay evidence."""
        state = self.sleep_state.strip().lower()
        if state not in {"wake", "awake_rest", "nrem", "rem"}:
            raise ValueError("sleep_state must be wake, awake_rest, nrem, or rem")
        object.__setattr__(self, "sleep_state", state)
        for name in (
            "state_confidence", "slow_oscillation_strength", "spindle_strength",
            "ripple_strength", "replay_strength", "coordination_strength",
            "prior_knowledge_support",
        ):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the available sleep and replay evidence without adding synthetic events."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class CompetingTrace:
    """A separate episode that may compete with the target trace during retrieval."""

    trace_id: str
    hippocampal_pattern: Sequence[float]
    strength: float = 0.5
    conflict: float = 1.0

    def __post_init__(self) -> None:
        """Validate competing episode identity, representation, strength, and cue conflict."""
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise ValueError("trace_id must be a non-empty string")
        object.__setattr__(self, "hippocampal_pattern",
                           _finite_vector("hippocampal_pattern", self.hippocampal_pattern))
        object.__setattr__(self, "strength", _unit("strength", self.strength))
        object.__setattr__(self, "conflict", _unit("conflict", self.conflict))

    def to_dict(self) -> dict[str, Any]:
        """Return this competing trace as a JSON-compatible mapping."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class ConsolidationConfig:
    """Engineering parameters for a reproducible novelty/STC/sleep consolidation model.

    Time constants are in milliseconds. Unit counts and thresholds are simulation choices,
    not universal physiological constants or measurements of an individual brain.
    """

    input_size: int = 16
    vta_units: int = 24
    lc_units: int = 24
    cortical_units: int = 32
    tag_tau_ms: float = 180000.0
    temporal_eligibility_tau_ms: float = 240000.0
    dopamine_tau_ms: float = 90000.0
    noradrenaline_tau_ms: float = 60000.0
    resource_tau_ms: float = 180000.0
    novelty_threshold: float = 0.18
    common_novelty_similarity_threshold: float = 0.60
    d1_d5_kd: float = 0.35
    d1_d5_hill: float = 2.0
    adrenergic_kd: float = 0.30
    resource_gain: float = 0.85
    resource_threshold: float = 0.42
    capture_threshold: float = 0.20
    capture_slope: float = 7.0
    stabilization_rate: float = 0.22
    synaptic_decay_rate: float = 0.035
    max_synaptic_change: float = 1.0
    persistence_decay: float = 0.025
    retrieval_persistence_weight: float = 0.45
    retrieval_match_weight: float = 0.30
    retrieval_context_weight: float = 0.15
    retrieval_interference_weight: float = 0.35
    retrieval_temperature: float = 0.30
    interference_sensitivity: float = 1.0
    cortical_integration_rate: float = 0.18
    allow_awake_rest_replay: bool = True
    sleep_state_threshold: float = 0.60
    random_seed: int = 53

    def __post_init__(self) -> None:
        """Validate dimensions, kinetic constants, unit parameters, and bounded learning rates."""
        if min(self.input_size, self.vta_units, self.lc_units, self.cortical_units) < 2:
            raise ValueError("input_size and population sizes must be at least two")
        for name in (
            "tag_tau_ms", "temporal_eligibility_tau_ms", "dopamine_tau_ms",
            "noradrenaline_tau_ms", "resource_tau_ms", "d1_d5_kd",
            "d1_d5_hill", "adrenergic_kd", "capture_slope", "retrieval_temperature",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "novelty_threshold", "common_novelty_similarity_threshold", "resource_gain",
            "resource_threshold", "capture_threshold", "stabilization_rate",
            "synaptic_decay_rate", "max_synaptic_change", "persistence_decay",
            "retrieval_persistence_weight", "retrieval_match_weight",
            "retrieval_context_weight", "retrieval_interference_weight",
            "interference_sensitivity", "cortical_integration_rate",
            "sleep_state_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            if name not in {"max_synaptic_change", "interference_sensitivity"} and value > 1.0:
                raise ValueError(f"{name} must not exceed one")
        if self.novelty_threshold >= 1.0 or self.resource_threshold >= 1.0:
            raise ValueError("novelty and resource thresholds must be below one")

    def to_dict(self) -> dict[str, Any]:
        """Return every simulation parameter so runs can be reproduced."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class ConsolidationRequest:
    """One post-learning consolidation interval for a registered or new memory trace."""

    trace: MemoryTrace
    post_learning_events: Sequence[PostLearningEvent] = ()
    internal_state: InternalState = field(default_factory=InternalState)
    sleep_evidence: SleepEvidence | None = None
    competing_traces: Sequence[CompetingTrace] = ()
    retrieval_cue: Sequence[float] | None = None
    context_match: float = 0.5

    def __post_init__(self) -> None:
        """Validate all event, cue, competitor, and context dimensions against the target trace."""
        width = len(self.trace.hippocampal_pattern)
        events = tuple(self.post_learning_events)
        if any(len(event.features) != width for event in events):
            raise ValueError("post-learning event features must match the trace pattern width")
        if any(event.timestamp_ms < self.trace.encoding_time_ms for event in events):
            raise ValueError("post-learning events must occur at or after encoding")
        if self.internal_state.predicted_features is not None and (
            len(self.internal_state.predicted_features) != width
        ):
            raise ValueError("predicted_features must match the trace pattern width")
        if any(len(row) != width for row in self.internal_state.prior_knowledge_patterns):
            raise ValueError("prior-knowledge patterns must match the trace pattern width")
        competitors = tuple(self.competing_traces)
        if any(len(item.hippocampal_pattern) != width for item in competitors):
            raise ValueError("competing traces must match the target pattern width")
        cue = None if self.retrieval_cue is None else _finite_vector("retrieval_cue", self.retrieval_cue)
        if cue is not None and len(cue) != width:
            raise ValueError("retrieval_cue must match the trace pattern width")
        object.__setattr__(self, "post_learning_events", events)
        object.__setattr__(self, "competing_traces", competitors)
        object.__setattr__(self, "retrieval_cue", cue)
        object.__setattr__(self, "context_match", _unit("context_match", self.context_match))

    def to_dict(self) -> dict[str, Any]:
        """Return the complete request without converting missing evidence into invented values."""
        return _jsonable(asdict(self))


@dataclass(slots=True)
class ConsolidationResult:
    """Inspectable result for all 16 source-ordered Point 05 operations."""

    trace_id: str
    operation_trace: list[dict[str, Any]]
    final_trace: dict[str, Any]
    novelty: dict[str, Any]
    neuromodulation: dict[str, Any]
    synaptic_capture: dict[str, Any]
    retrieval: dict[str, Any]
    interference: dict[str, Any]
    sleep: dict[str, Any]
    multidimensional_state: dict[str, Any]
    biological: dict[str, Any]
    logger: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the final memory state, all operation outputs, and biological caveats."""
        return _jsonable(asdict(self))
