"""Distributed population-rate reference model for Point 20 reconstructive-memory operations."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from .contracts import Point20Config


def _sigmoid(value: float) -> float:
    """Compute a numerically stable logistic population activation."""
    if value >= 0.0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def _unit(value: float) -> float:
    """Validate and constrain one normalized population signal."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("population signals must be finite")
    return max(0.0, min(1.0, number))


@dataclass
class FunctionalPopulation:
    """Track one distributed functional population and its recent rate-coded activity."""

    name: str
    biological_scope: str
    activation: float = 0.0
    history: list[float] = field(default_factory=list)

    def update(
        self,
        drive: float,
        recurrent: float,
        inhibition: float,
        salience_modulation: float,
        decay: float,
        recurrent_gain: float = 0.30,
        inhibition_gain: float = 0.18,
    ) -> float:
        """Update activation from external drive, recurrence, inhibition, salience, and decay."""
        net = (
            1.7 * _unit(drive)
            + recurrent_gain * recurrent
            - inhibition_gain * inhibition
            + 0.15 * _unit(salience_modulation)
        )
        target = _sigmoid(net)
        self.activation = _unit((1.0 - decay) * target + decay * self.activation)
        self.history.append(self.activation)
        self.history = self.history[-48:]
        return self.activation


class ReconstructiveMemoryNetwork:
    """Coordinate inspectable functional populations without claiming a literal neural simulation."""

    def __init__(self, config: Point20Config) -> None:
        """Initialize candidate neural-system populations and sparse recurrent interactions."""
        descriptions = {
            "hippocampal_pattern_completion": (
                "Hippocampal formation contributions to event-specific binding and cue-driven pattern completion"
            ),
            "medial_prefrontal_schema": (
                "Medial prefrontal interactions supporting schema, script, and prior-knowledge use"
            ),
            "lateral_prefrontal_source_control": (
                "Lateral prefrontal contributions to controlled retrieval, source evaluation, and conflict monitoring"
            ),
            "parietal_episodic_detail": (
                "Parietal and angular-gyrus contributions to attention/accessibility of episodic detail; task-dependent"
            ),
            "sensory_association_reinstatement": (
                "Modality-specific sensory association reinstatement for visual, auditory, and spatial detail"
            ),
            "temporal_semantic_memory": (
                "Temporal semantic-memory systems supplying concepts, scripts, typical features, and relations"
            ),
            "amygdala_salience_modulation": (
                "Amygdala and neuromodulatory influence on emotional salience and prioritization, not truth detection"
            ),
            "distributed_event_integration": (
                "Distributed hippocampal-cortical event integration across feature and relational representations"
            ),
        }
        self.config = config
        self.populations = {
            name: FunctionalPopulation(name=name, biological_scope=scope)
            for name, scope in descriptions.items()
        }
        self.connections = {
            "hippocampal_pattern_completion": {
                "sensory_association_reinstatement": 0.14,
                "parietal_episodic_detail": 0.10,
                "distributed_event_integration": 0.12,
            },
            "medial_prefrontal_schema": {
                "temporal_semantic_memory": 0.16,
                "lateral_prefrontal_source_control": 0.10,
                "distributed_event_integration": 0.08,
            },
            "lateral_prefrontal_source_control": {
                "hippocampal_pattern_completion": 0.10,
                "medial_prefrontal_schema": 0.08,
                "parietal_episodic_detail": 0.10,
                "distributed_event_integration": 0.10,
            },
            "parietal_episodic_detail": {
                "hippocampal_pattern_completion": 0.10,
                "sensory_association_reinstatement": 0.12,
                "lateral_prefrontal_source_control": 0.08,
            },
            "sensory_association_reinstatement": {
                "hippocampal_pattern_completion": 0.12,
                "parietal_episodic_detail": 0.10,
            },
            "temporal_semantic_memory": {
                "medial_prefrontal_schema": 0.14,
                "distributed_event_integration": 0.10,
            },
            "amygdala_salience_modulation": {
                "hippocampal_pattern_completion": 0.07,
                "lateral_prefrontal_source_control": 0.05,
                "parietal_episodic_detail": 0.06,
            },
            "distributed_event_integration": {
                "hippocampal_pattern_completion": 0.10,
                "medial_prefrontal_schema": 0.08,
                "lateral_prefrontal_source_control": 0.10,
                "temporal_semantic_memory": 0.10,
            },
        }
        self.operation_count = 0

    def advance(
        self,
        operation_id: str,
        target_populations: Mapping[str, float],
        salience: float = 0.0,
    ) -> dict[str, Any]:
        """Advance all populations once and return the targeted and whole-network activations."""
        unknown = set(target_populations) - set(self.populations)
        if unknown:
            raise KeyError(f"unknown biological populations: {sorted(unknown)}")
        drives = {name: _unit(value) for name, value in target_populations.items()}
        previous = {name: item.activation for name, item in self.populations.items()}
        global_activity = sum(previous.values()) / max(1, len(previous))
        next_values: dict[str, float] = {}
        for name, population in self.populations.items():
            recurrent = sum(
                weight * previous[source]
                for source, weight in self.connections[name].items()
            )
            next_values[name] = population.update(
                drives.get(name, 0.0),
                recurrent,
                global_activity,
                salience if name == "amygdala_salience_modulation" else 0.0,
                self.config.population_decay,
            )
        self.operation_count += 1
        return {
            "operation_id": operation_id,
            "target_populations": dict(drives),
            "population_activations": next_values,
            "mean_network_activity": sum(next_values.values()) / len(next_values),
            "operation_count": self.operation_count,
            "salience_modulation": _unit(salience),
            "scope": "Coarse distributed rate-coded reference, not measured or anatomically fitted neural activity.",
        }

    def snapshot(self) -> dict[str, Any]:
        """Return activity histories and explicit limitations of the biological-reference track."""
        return {
            "model": "distributed_reconstructive_memory_rate_network",
            "operation_count": self.operation_count,
            "populations": {
                name: {
                    "biological_scope": population.biological_scope,
                    "activation": population.activation,
                    "history": list(population.history),
                }
                for name, population in self.populations.items()
            },
            "scientific_limitations": [
                "Populations are candidate functional correspondences, not isolated one-region/one-function modules.",
                "Activations are engineering variables rather than measured human neural signals.",
                "The equations are not established literal neuronal computations or fitted physiological constants.",
                "Amygdala/salience modulation changes priority, not evidence truth or source reliability.",
                "The model does not assert that reconsolidation occurs on every retrieval.",
            ],
        }
