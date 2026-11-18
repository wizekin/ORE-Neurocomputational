"""Distributed rate-coded network surrogates for Point 18 attention operations."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


def _unit(value: float) -> float:
    """Validate and clamp one population activation."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("network activation must be finite")
    return max(0.0, min(1.0, number))


def _sigmoid(value: float) -> float:
    """Compute a stable logistic activation for a population-rate update."""
    if value >= 0.0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


@dataclass
class AttentionPopulation:
    """Track a distributed functional population with recurrence and inhibition."""

    name: str
    biological_scope: str
    input_gain: float = 1.6
    recurrent_gain: float = 0.30
    inhibition_gain: float = 0.20
    decay_rate: float = 0.20
    activation: float = 0.0
    update_count: int = 0
    history: list[float] = field(default_factory=list)

    def update(self, drive: float) -> dict[str, Any]:
        """Update bounded population activity from input, recurrence, inhibition, and decay."""
        before = _unit(self.activation)
        drive = _unit(drive)
        target = _sigmoid(
            self.input_gain * drive + self.recurrent_gain * before
            - self.inhibition_gain * before * before
        )
        after = _unit((1.0 - self.decay_rate) * target + self.decay_rate * before)
        self.activation = after
        self.update_count += 1
        self.history.append(after)
        self.history = self.history[-32:]
        return {
            "population": self.name,
            "activation_before": before,
            "input_drive": drive,
            "activation_after": after,
            "update_count": self.update_count,
            "parameters": {
                "input_gain": self.input_gain,
                "recurrent_gain": self.recurrent_gain,
                "inhibition_gain": self.inhibition_gain,
                "decay_rate": self.decay_rate,
            },
        }


class AttentionalBiologicalNetwork:
    """Coordinate inspectable population-rate proxies without asserting a literal brain simulation."""

    def __init__(self) -> None:
        """Initialize distributed network populations implicated in attentional functions."""
        definitions = {
            "top_down_goal_control": "Prefrontal and posterior parietal goal-maintenance interactions",
            "dorsal_attention_selection": "Dorsal attention and orienting systems",
            "ventral_attention_reorienting": "Ventral attention and stimulus-driven reorienting systems",
            "sensory_gain_modulation": "Task-modulated sensory cortical population activity",
            "working_memory_admission": "Frontoparietal working-memory gating and updating",
            "arousal_state_modulation": "Brainstem, basal forebrain, and thalamocortical arousal interactions",
            "sustained_attention_monitor": "Distributed sustained-attention and control dynamics",
            "lapse_detection_monitor": "Medial frontal performance-monitoring and control interactions",
            "evidence_reconstruction": "Residual sensory traces and source-supported reinstatement",
            "completeness_audit": "Distributed monitoring and working-memory comparison processes",
            "recovery_control": "Control systems supporting pacing, repetition, and recovery selection",
            "goal_reorientation": "Goal reinstatement and attentional reorientation",
            "hippocampal_reencoding": "Medial temporal binding and context reinstatement",
            "repair_verification": "Error monitoring and evidence comparison",
            "selection_history_learning": "Experience-dependent attentional priority modulation",
            "uncertainty_maintenance": "Working-memory maintenance of unresolved alternatives",
            "downstream_release_gate": "Computational safety gate; not a single anatomical circuit",
        }
        self.populations = {
            name: AttentionPopulation(name=name, biological_scope=scope)
            for name, scope in definitions.items()
        }

    def engage(self, name: str, operation: str, drive: float) -> dict[str, Any]:
        """Update one named network population and annotate the functional operation."""
        if name not in self.populations:
            raise KeyError(f"unknown attentional network population: {name}")
        result = self.populations[name].update(drive)
        result["operation"] = operation
        return result

    def snapshot(self) -> dict[str, Any]:
        """Return current network activations with explicit scientific scope limitations."""
        return {
            name: {
                "biological_scope": population.biological_scope,
                "activation": population.activation,
                "update_count": population.update_count,
                "history": list(population.history),
            }
            for name, population in self.populations.items()
        } | {
            "scientific_scope": (
                "Bounded distributed rate-coded surrogates, not direct neural measurements, "
                "biophysical simulations, or one-region/one-function mappings."
            )
        }
