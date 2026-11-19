"""Typed requests, state, configuration, and results for Point 06 cognitive-load management."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _finite_unit(name: str, value: float) -> float:
    """Validate one finite normalized value and return it as a float."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return number


@dataclass(frozen=True)
class Point06Config:
    """Task-calibratable parameters; none is a universal human capacity or neural constant."""

    goal_temperature: float = 0.45
    chunk_similarity_threshold: float = 0.38
    interaction_gain: float = 0.35
    prior_knowledge_gain: float = 0.45
    goal_relevance_threshold: float = 0.20
    overload_threshold: float = 0.58
    utilization_low: float = 0.55
    utilization_high: float = 0.85
    scaffold_fade_rate: float = 0.12
    success_threshold: float = 0.80
    working_set_limit: int = 8
    simulate_biology: bool = True
    population_decay: float = 0.22
    population_recurrent_gain: float = 0.26
    population_inhibition_gain: float = 0.18

    def __post_init__(self) -> None:
        """Reject invalid parameters before the processing state is modified."""
        for name in (
            "chunk_similarity_threshold", "prior_knowledge_gain", "goal_relevance_threshold",
            "overload_threshold", "utilization_low", "utilization_high", "scaffold_fade_rate",
            "success_threshold", "population_decay", "population_inhibition_gain",
        ):
            _finite_unit(name, getattr(self, name))
        for name in ("goal_temperature",):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.interaction_gain) or self.interaction_gain < 0.0:
            raise ValueError("interaction_gain must be finite and non-negative")
        if not math.isfinite(self.population_recurrent_gain) or self.population_recurrent_gain < 0.0:
            raise ValueError("population_recurrent_gain must be finite and non-negative")
        if self.working_set_limit < 1:
            raise ValueError("working_set_limit must be at least one; it is an engineering limit")
        if self.utilization_low > self.utilization_high:
            raise ValueError("utilization_low cannot exceed utilization_high")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached serializable configuration snapshot."""
        return asdict(self)


@dataclass(frozen=True)
class Point06Request:
    """One learning episode with explicit material, prior knowledge, relations, and observations."""

    sequence_id: str
    goal: str
    text: str | None = None
    units: Sequence[str] | None = None
    unit_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    dependencies: Sequence[Mapping[str, Any]] = ()
    relations: Sequence[Mapping[str, Any]] = ()
    prior_knowledge: Sequence[str | Mapping[str, Any]] = ()
    goal_candidates: Mapping[str, float] = field(default_factory=dict)
    presentation_costs: Mapping[str, float] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    examples: Sequence[Mapping[str, Any]] = ()
    resource_capacity: float = 1.0
    learner_profile: Mapping[str, Any] = field(default_factory=dict)
    source_reference: str | None = None

    def __post_init__(self) -> None:
        """Validate source exclusivity and normalize caller-owned collections into stable copies."""
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be a non-empty string")
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("goal must be a non-empty string")
        if (self.text is None) == (self.units is None):
            raise ValueError("provide exactly one of text or units")
        if self.text is not None and not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.units is not None:
            normalized = tuple(str(item).strip() for item in self.units)
            if not normalized or any(not item for item in normalized):
                raise ValueError("units must contain non-empty strings")
            object.__setattr__(self, "units", normalized)
        if not math.isfinite(float(self.resource_capacity)) or not 0.0 < float(self.resource_capacity) <= 1.0:
            raise ValueError("resource_capacity must be finite and in (0, 1]")
        metadata = {str(key): dict(value) for key, value in self.unit_metadata.items()}
        object.__setattr__(self, "unit_metadata", metadata)
        object.__setattr__(self, "dependencies", tuple(dict(item) for item in self.dependencies))
        object.__setattr__(self, "relations", tuple(dict(item) for item in self.relations))
        normalized_memories: list[dict[str, Any]] = []
        for index, item in enumerate(self.prior_knowledge):
            normalized_memories.append(
                {"id": f"memory-{index + 1}", "text": item, "confidence": 0.7}
                if isinstance(item, str) else dict(item)
            )
        object.__setattr__(self, "prior_knowledge", tuple(normalized_memories))
        object.__setattr__(self, "goal_candidates", dict(self.goal_candidates))
        object.__setattr__(self, "presentation_costs", dict(self.presentation_costs))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "examples", tuple(dict(item) for item in self.examples))
        object.__setattr__(self, "learner_profile", dict(self.learner_profile))
        for group_name, values in (
            ("goal_candidates", self.goal_candidates),
            ("presentation_costs", self.presentation_costs),
            ("metrics", self.metrics),
        ):
            for key, raw in values.items():
                number = float(raw)
                if not math.isfinite(number):
                    raise ValueError(f"{group_name}.{key} must be finite")
        for index, edge in enumerate(self.dependencies):
            if not edge.get("source") or not edge.get("target"):
                raise ValueError(f"dependency {index} requires source and target")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly request snapshot without exposing mutable mappings."""
        return {
            "sequence_id": self.sequence_id, "goal": self.goal, "text": self.text,
            "units": list(self.units) if self.units is not None else None,
            "unit_metadata": {key: dict(value) for key, value in self.unit_metadata.items()},
            "dependencies": [dict(item) for item in self.dependencies],
            "relations": [dict(item) for item in self.relations],
            "prior_knowledge": [dict(item) for item in self.prior_knowledge],
            "goal_candidates": dict(self.goal_candidates),
            "presentation_costs": dict(self.presentation_costs), "metrics": dict(self.metrics),
            "examples": [dict(item) for item in self.examples],
            "resource_capacity": self.resource_capacity,
            "learner_profile": dict(self.learner_profile), "source_reference": self.source_reference,
        }


@dataclass
class CognitiveUnit:
    """Represent one source-grounded learning unit and its modelled processing attributes."""

    identifier: str
    text: str
    position: int
    features: tuple[str, ...]
    task_relevance: float = 0.0
    intrinsic_load: float = 0.0
    familiarity: float = 0.0
    chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached unit representation suitable for traces and SQLite metadata."""
        return asdict(self)


@dataclass
class Point06State:
    """Maintain the current goal, external map, support level, and repeated-performance evidence."""

    active_goal: str = ""
    active_subgoal: str | None = None
    active_working_set: list[str] = field(default_factory=list)
    completed_units: list[str] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    external_representation: dict[str, Any] = field(default_factory=dict)
    estimated_load: float = 0.0
    support_level: float = 0.65
    success_streak: int = 0
    process_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a detached snapshot of the learner model's revisable task state."""
        return {
            "active_goal": self.active_goal, "active_subgoal": self.active_subgoal,
            "active_working_set": list(self.active_working_set),
            "completed_units": list(self.completed_units),
            "chunks": [dict(item) for item in self.chunks],
            "external_representation": dict(self.external_representation),
            "estimated_load": self.estimated_load, "support_level": self.support_level,
            "success_streak": self.success_streak, "process_count": self.process_count,
        }


@dataclass(frozen=True)
class Point06Result:
    """Collect the 16-step cognitive-load trace, functional result, biology proxy, and final state."""

    sequence_id: str
    goal: str
    goal_selection: Mapping[str, Any]
    complexity: Mapping[str, Any]
    prerequisite_retrieval: Mapping[str, Any]
    chunking: Mapping[str, Any]
    presentation: Mapping[str, Any]
    external_representation: Mapping[str, Any]
    working_set: Mapping[str, Any]
    overload_monitor: Mapping[str, Any]
    adaptation: Mapping[str, Any]
    scaffolding: Mapping[str, Any]
    integration: Mapping[str, Any]
    consolidation: Mapping[str, Any]
    comprehension: Mapping[str, Any]
    biological: Mapping[str, Any]
    operation_trace: tuple[Mapping[str, Any], ...]
    final_state: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the complete result to plain dictionaries and lists for storage or transport."""
        return {
            "schema": "ore.cognitive_load.point06.1.0",
            "sequence_id": self.sequence_id, "goal": self.goal,
            "goal_selection": dict(self.goal_selection),
            "complexity": dict(self.complexity),
            "prerequisite_retrieval": dict(self.prerequisite_retrieval),
            "chunking": dict(self.chunking), "presentation": dict(self.presentation),
            "external_representation": dict(self.external_representation),
            "working_set": dict(self.working_set),
            "overload_monitor": dict(self.overload_monitor),
            "adaptation": dict(self.adaptation), "scaffolding": dict(self.scaffolding),
            "integration": dict(self.integration), "consolidation": dict(self.consolidation),
            "comprehension": dict(self.comprehension), "biological": dict(self.biological),
            "operation_trace": [dict(item) for item in self.operation_trace],
            "final_state": dict(self.final_state), "metadata": dict(self.metadata),
        }