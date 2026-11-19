"""Distributed rate-coded biological-reference network for Point 06 operations."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


def _sigmoid(value: float) -> float:
    """Compute a numerically stable logistic activation for a model population."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("population drive must be finite")
    if number >= 0.0:
        z = math.exp(-min(number, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(number, -700.0))
    return z / (1.0 + z)


def _unit(value: float) -> float:
    """Validate and bound one normalized population signal."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("population signals must be finite")
    return max(0.0, min(1.0, number))


@dataclass
class RatePopulation:
    """Track one abstract functional population's activation and short history."""

    name: str
    biological_scope: str
    activation: float = 0.0
    history: list[float] = field(default_factory=list)

    def update(
        self, drive: float, recurrent: float, inhibition: float, decay: float,
        recurrent_gain: float = 0.26, inhibition_gain: float = 0.18,
    ) -> float:
        """Update a bounded rate proxy using external drive, recurrence, competition, and persistence."""
        target = _sigmoid(1.8 * _unit(drive) + recurrent_gain * recurrent - inhibition_gain * inhibition)
        self.activation = _unit((1.0 - decay) * target + decay * self.activation)
        self.history.append(self.activation)
        del self.history[:-32]
        return self.activation


class DistributedCognitiveLoadNetwork:
    """Simulate coarse distributed functional interactions without claiming neural fidelity."""

    def __init__(self, config: Any) -> None:
        """Initialize candidate control, attention, semantic, memory, and monitoring populations."""
        descriptions = {
            "prefrontal_goal_control": "Goal maintenance and control contributions associated with prefrontal/frontoparietal systems",
            "parietal_attention_selection": "Task-dependent selection and allocation associated with parietal attention systems",
            "temporal_semantic_language": "Distributed lexical, language, and semantic representation systems",
            "hippocampal_associative_retrieval": "Medial-temporal relational retrieval and pattern-completion functions",
            "cortical_schema_integration": "Distributed cortical knowledge structures and schema-supported integration",
            "cingulo_opercular_monitoring": "Performance monitoring and control functions associated with cingulo-opercular interactions",
            "sensory_external_representation": "Visual/auditory association processing of external notes, diagrams, and other materials",
        }
        self.config = config
        self.populations = {name: RatePopulation(name, scope) for name, scope in descriptions.items()}
        self.connections = {
            "prefrontal_goal_control": {"parietal_attention_selection": 0.12, "cingulo_opercular_monitoring": 0.10},
            "parietal_attention_selection": {"prefrontal_goal_control": 0.12, "sensory_external_representation": 0.08},
            "temporal_semantic_language": {"cortical_schema_integration": 0.12, "hippocampal_associative_retrieval": 0.06},
            "hippocampal_associative_retrieval": {"temporal_semantic_language": 0.10, "cortical_schema_integration": 0.08},
            "cortical_schema_integration": {"temporal_semantic_language": 0.10, "prefrontal_goal_control": 0.08},
            "cingulo_opercular_monitoring": {"prefrontal_goal_control": 0.14, "parietal_attention_selection": 0.06},
            "sensory_external_representation": {"parietal_attention_selection": 0.10, "temporal_semantic_language": 0.05},
        }
        self.operation_count = 0

    def engage(self, operation: str, target: str, drive: float) -> dict[str, Any]:
        """Advance each population once and return the targeted activation plus network snapshot."""
        if target not in self.populations:
            raise KeyError(f"unknown population: {target}")
        bounded_drive = _unit(drive)
        previous = {name: population.activation for name, population in self.populations.items()}
        global_activity = sum(previous.values()) / max(1, len(previous))
        updated: dict[str, float] = {}
        for name, population in self.populations.items():
            recurrent = sum(weight * previous[source] for source, weight in self.connections[name].items())
            input_drive = bounded_drive if name == target else 0.0
            updated[name] = population.update(
                input_drive, recurrent, global_activity,
                self.config.population_decay,
                self.config.population_recurrent_gain,
                self.config.population_inhibition_gain,
            )
        self.operation_count += 1
        return {
            "model": "distributed_rate_coded_reference",
            "operation": operation,
            "target_population": target,
            "target_activation": updated[target],
            "population_activations": updated,
            "network_mean_activity": sum(updated.values()) / len(updated),
            "operation_count": self.operation_count,
            "scope": "Engineering proxy; not measured, anatomically fitted, or literal human neural activity.",
        }

    def snapshot(self) -> dict[str, Any]:
        """Return activity histories and explicit limitations of this biological-reference model."""
        return {
            "model": "distributed_rate_coded_reference",
            "operation_count": self.operation_count,
            "populations": {
                name: {"biological_scope": item.biological_scope,
                       "activation": item.activation, "history": list(item.history)}
                for name, item in self.populations.items()
            },
            "scientific_limitations": [
                "Population labels describe candidate functional correspondences, not isolated brain modules.",
                "Activations are simulated engineering variables, not neural measurements.",
                "The equations are not established literal neuronal computations or fitted physiological constants.",
                "The model does not reproduce the full complexity of human working memory or cognitive control.",
            ],
        }