"""Rate-coded biological surrogates for interacting language and domain-general systems.

The network dynamics are simplified, inspectable population-level abstractions. They are not
anatomical replicas, direct brain measurements, or claims that a single region performs a task.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


def _bounded(value: float) -> float:
    """Convert a finite scalar to the closed unit interval."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("network inputs must be finite")
    return max(0.0, min(1.0, numeric))


def _sigmoid(value: float) -> float:
    """Compute a numerically stable logistic population-activation function."""
    if value >= 0.0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


@dataclass
class NetworkActivity:
    """Simulate one distributed subsystem using bounded recurrent rate-coded activation."""

    name: str
    biological_scope: str
    input_gain: float = 1.7
    recurrent_gain: float = 0.3
    inhibition_gain: float = 0.2
    decay_rate: float = 0.2
    activation: float = 0.0
    engagement_count: int = 0
    last_operation: str = ""
    last_load: float = 0.0
    activity_history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate non-negative connection gains and normalized activity parameters."""
        for name in ("input_gain", "recurrent_gain", "inhibition_gain"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(name + " must be finite and non-negative")
            setattr(self, name, value)
        self.decay_rate = _bounded(self.decay_rate)
        self.activation = _bounded(self.activation)
        self.last_load = _bounded(self.last_load)

    def engage(self, operation: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Integrate input drive, recurrent excitation, inhibition, and activation decay."""
        data = dict(payload or {})
        self.engagement_count += 1
        self.last_operation = str(operation)
        drive_value = data.get(
            "activation_input",
            data.get("drive", data.get("load", data.get("demand_score",
                                                    data.get("control_allocation", 0.0)))),
        )
        try:
            drive = _bounded(drive_value)
        except (TypeError, ValueError):
            drive = 0.0
        previous = self.activation
        net_input = (
            self.input_gain * drive
            + self.recurrent_gain * previous
            - self.inhibition_gain * previous * previous
        )
        target_activation = _sigmoid(net_input)
        self.activation = _bounded(
            (1.0 - self.decay_rate) * target_activation + self.decay_rate * previous
        )
        self.last_load = drive
        row = {
            "operation": self.last_operation,
            "engagement_count": self.engagement_count,
            "drive_proxy": drive,
            "activation_before": previous,
            "activation_after": self.activation,
            "net_input": net_input,
            "target_activation": target_activation,
            "rate_equation": (
                "a_t=(1-decay)*sigmoid(input_gain*drive+recurrent_gain*a_prev"
                "-inhibition_gain*a_prev^2)+decay*a_prev"
            ),
            "evidence_boundary": "rate_coded_functional_surrogate_not_neural_measurement",
        }
        self.activity_history.append(row)
        return row

    def snapshot(self) -> dict[str, Any]:
        """Return activation state, model parameters, and recent operation diagnostics."""
        return {
            "network": self.name,
            "biological_scope": self.biological_scope,
            "engagement_count": self.engagement_count,
            "activation": self.activation,
            "last_operation": self.last_operation,
            "last_drive_proxy": self.last_load,
            "parameters": {
                "input_gain": self.input_gain,
                "recurrent_gain": self.recurrent_gain,
                "inhibition_gain": self.inhibition_gain,
                "decay_rate": self.decay_rate,
            },
            "activity_history": list(self.activity_history[-15:]),
        }


class BiologicalLanguageReasoningNetwork:
    """Coordinate distributed perceptual, language, semantic, control, and inference surrogates."""

    def __init__(self) -> None:
        """Initialize interacting networks with distinct rate-dynamics parameterizations."""
        self.perceptual = NetworkActivity(
            "auditory_visual_perceptual",
            "Auditory pathways for speech and visual/orthographic pathways for text",
            input_gain=1.9, recurrent_gain=0.12, inhibition_gain=0.28, decay_rate=0.18,
        )
        self.language_selective = NetworkActivity(
            "frontotemporal_language_selective",
            "Distributed frontal and temporal language-selective circuitry, commonly left-lateralized",
            input_gain=1.7, recurrent_gain=0.48, inhibition_gain=0.20, decay_rate=0.22,
        )
        self.conceptual_semantic = NetworkActivity(
            "distributed_conceptual_semantic",
            "Distributed modality-linked and associative conceptual knowledge systems",
            input_gain=1.55, recurrent_gain=0.38, inhibition_gain=0.24, decay_rate=0.25,
        )
        self.multiple_demand_control = NetworkActivity(
            "multiple_demand_control",
            "Bilateral frontal/parietal and distributed domain-general control systems",
            input_gain=2.1, recurrent_gain=0.24, inhibition_gain=0.34, decay_rate=0.12,
        )
        self.domain_general_reasoning = NetworkActivity(
            "domain_general_reasoning",
            "Distributed systems supporting deliberate inferential operations",
            input_gain=1.8, recurrent_gain=0.34, inhibition_gain=0.24, decay_rate=0.18,
        )
        self.social_pragmatic = NetworkActivity(
            "social_pragmatic_cognition",
            "Medial prefrontal, temporoparietal, posterior-midline, and related systems",
            input_gain=1.55, recurrent_gain=0.36, inhibition_gain=0.22, decay_rate=0.24,
        )
        self.integration_monitoring = NetworkActivity(
            "cross_network_integration_monitoring",
            "Interacting language, semantic, control, and social inference systems",
            input_gain=1.65, recurrent_gain=0.32, inhibition_gain=0.28, decay_rate=0.20,
        )
        self.language_production = NetworkActivity(
            "language_production_task_output",
            "Language production and task-related motor/control systems",
            input_gain=1.8, recurrent_gain=0.25, inhibition_gain=0.30, decay_rate=0.15,
        )
        self.hippocampal_cortical_learning = NetworkActivity(
            "hippocampal_cortical_learning",
            "Medial-temporal episodic binding and experience-dependent cortical learning",
            input_gain=1.4, recurrent_gain=0.55, inhibition_gain=0.12, decay_rate=0.35,
        )
        self._systems = {
            item.name: item for item in (
                self.perceptual, self.language_selective, self.conceptual_semantic,
                self.multiple_demand_control, self.domain_general_reasoning,
                self.social_pragmatic, self.integration_monitoring,
                self.language_production, self.hippocampal_cortical_learning,
            )
        }

    def engage(
        self,
        system: str,
        operation: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Engage a named network and reject unknown routing targets."""
        if system not in self._systems:
            raise KeyError("unknown biological subsystem: " + system)
        return self._systems[system].engage(operation, payload)

    def snapshot(self) -> dict[str, Any]:
        """Return all network activation states and explicit biological-model limitations."""
        return {
            **{name: network.snapshot() for name, network in self._systems.items()},
            "scientific_scope": (
                "Rate-coded distributed network approximations, not direct neural measurements, "
                "exact anatomical localization, or literal biological equations."
            ),
            "functional_separation": (
                "Language, conceptual retrieval, general control/reasoning, and social-pragmatic "
                "inference remain separately traceable while exchanging information."
            ),
        }
