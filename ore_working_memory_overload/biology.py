"""Distributed rate-coded reference network for Point 19 biological operations."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


def _sigmoid(value: float) -> float:
    """Compute a numerically stable logistic rate activation."""
    if value >= 0.0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def _bounded(value: float) -> float:
    """Validate and constrain one population activation to the unit interval."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("population signals must be finite")
    return max(0.0, min(1.0, number))


@dataclass
class RatePopulation:
    """Track one functional population's rate-coded state and recent activation history."""

    name: str
    biological_scope: str
    activation: float = 0.0
    history: list[float] = field(default_factory=list)

    def update(
        self, drive: float, recurrent: float, inhibition: float, decay: float,
        recurrent_gain: float = 0.30, inhibition_gain: float = 0.18,
    ) -> float:
        """Update activation using input, recurrence, competition, and leaky persistence."""
        target = _sigmoid(
            1.8 * _bounded(drive) + recurrent_gain * recurrent - inhibition_gain * inhibition
        )
        self.activation = _bounded((1.0 - decay) * target + decay * self.activation)
        self.history.append(self.activation)
        self.history = self.history[-32:]
        return self.activation


class DistributedWorkingMemoryNetwork:
    """Simulate coarse functional population interactions without claiming neural fidelity."""

    def __init__(self, config: Any) -> None:
        """Initialize functionally labelled populations and sparse recurrent connections."""
        descriptions = {
            "phonological_lexical_maintenance": "Distributed temporal lexical and phonological maintenance functions",
            "syntactic_dependency_workspace": "Frontotemporal language and parietal dependency-maintenance interactions",
            "semantic_event_integration": "Temporal semantic and distributed event/situation representations",
            "parietal_attention_selection": "Posterior parietal attention and retrieval-selection functions",
            "prefrontal_goal_control": "Lateral prefrontal and frontoparietal goal/control interactions",
            "cingulo_opercular_conflict_monitor": "Anterior cingulate and related performance/conflict monitoring functions",
            "medial_temporal_pattern_completion": "Hippocampal and medial-temporal cue-based associative retrieval functions",
        }
        self.config = config
        self.populations = {
            name: RatePopulation(name, scope) for name, scope in descriptions.items()
        }
        self.connections = {
            "phonological_lexical_maintenance": {"prefrontal_goal_control": 0.10, "syntactic_dependency_workspace": 0.12},
            "syntactic_dependency_workspace": {"phonological_lexical_maintenance": 0.08, "prefrontal_goal_control": 0.12, "semantic_event_integration": 0.10},
            "semantic_event_integration": {"syntactic_dependency_workspace": 0.10, "medial_temporal_pattern_completion": 0.08},
            "parietal_attention_selection": {"prefrontal_goal_control": 0.13, "phonological_lexical_maintenance": 0.08},
            "prefrontal_goal_control": {"parietal_attention_selection": 0.12, "cingulo_opercular_conflict_monitor": 0.10, "syntactic_dependency_workspace": 0.08},
            "cingulo_opercular_conflict_monitor": {"prefrontal_goal_control": 0.14, "semantic_event_integration": 0.06},
            "medial_temporal_pattern_completion": {"semantic_event_integration": 0.10, "phonological_lexical_maintenance": 0.06},
        }
        self.operation_count = 0

    def engage(self, operation: str, target: str, drive: float) -> dict[str, Any]:
        """Advance all populations once and return the targeted activity and network snapshot."""
        if target not in self.populations:
            raise KeyError(f"unknown population: {target}")
        bounded = _bounded(drive)
        previous = {name: item.activation for name, item in self.populations.items()}
        global_activity = sum(previous.values()) / max(1, len(previous))
        updated: dict[str, float] = {}
        for name, population in self.populations.items():
            recurrent = sum(
                weight * previous[source]
                for source, weight in self.connections[name].items()
            )
            input_drive = bounded if name == target else 0.0
            updated[name] = population.update(
                input_drive, recurrent, global_activity,
                self.config.population_decay,
                self.config.population_recurrent_gain,
                self.config.population_inhibition_gain,
            )
        self.operation_count += 1
        return {
            "operation": operation,
            "target_population": target,
            "target_activation": updated[target],
            "population_activations": updated,
            "network_mean_activity": sum(updated.values()) / len(updated),
            "operation_count": self.operation_count,
            "scope": "Rate-coded computational reference, not measured or anatomically fitted human neural activity.",
        }

    def snapshot(self) -> dict[str, Any]:
        """Return all population activities and the scientific limitations of the model."""
        return {
            "model": "distributed_rate_coded_reference",
            "operation_count": self.operation_count,
            "populations": {
                name: {
                    "biological_scope": item.biological_scope,
                    "activation": item.activation,
                    "history": list(item.history),
                }
                for name, item in self.populations.items()
            },
            "scientific_limitations": [
                "Population labels describe candidate functional correspondences, not isolated brain modules.",
                "Activations are simulated engineering variables, not neural measurements.",
                "The equations are not established literal neuronal computations or fitted physiological constants.",
            ],
        }
