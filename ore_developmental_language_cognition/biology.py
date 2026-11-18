"""Distributed developmental network surrogates with bounded rate-coded activity.

These dynamics are inspectable functional models. They are not anatomical replicas, neuronal
recordings, or claims that development follows one region-per-operation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


def _unit(value: float) -> float:
    """Validate and clamp a finite normalized activation value."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("activation must be finite")
    return max(0.0, min(1.0, number))


def _sigmoid(value: float) -> float:
    """Compute a numerically stable logistic activation function."""
    if value >= 0.0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


@dataclass
class DevelopmentalNetworkActivity:
    """Track one distributed network with input, recurrence, inhibition, and activity decay."""

    name: str
    biological_scope: str
    input_gain: float = 1.7
    recurrent_gain: float = 0.3
    inhibition_gain: float = 0.2
    decay_rate: float = 0.2
    activation: float = 0.0
    engagement_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate finite non-negative gains and normalized activity parameters."""
        for field_name in ("input_gain", "recurrent_gain", "inhibition_gain"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(field_name + " must be finite and non-negative")
            setattr(self, field_name, value)
        self.decay_rate = _unit(self.decay_rate)
        self.activation = _unit(self.activation)

    def engage(self, operation: str, drive: float) -> dict[str, Any]:
        """Update population-rate proxy using drive, recurrence, inhibition, and retained activity."""
        current_drive = _unit(drive)
        previous = self.activation
        net_input = (
            self.input_gain * current_drive
            + self.recurrent_gain * previous
            - self.inhibition_gain * previous * previous
        )
        target = _sigmoid(net_input)
        self.activation = _unit((1.0 - self.decay_rate) * target + self.decay_rate * previous)
        self.engagement_count += 1
        row = {
            "network": self.name, "operation": str(operation), "drive_proxy": current_drive,
            "activation_before": previous, "activation_after": self.activation,
            "net_input": net_input, "target_activation": target,
            "rate_equation": (
                "a_t=(1-decay)*sigmoid(input_gain*drive+recurrent_gain*a_prev"
                "-inhibition_gain*a_prev^2)+decay*a_prev"
            ),
            "biological_scope": "rate_coded_distributed_functional_surrogate",
        }
        self.history.append(row)
        return row

    def snapshot(self) -> dict[str, Any]:
        """Return recent activity and model parameters for the operation logger."""
        return {
            "name": self.name, "biological_scope": self.biological_scope,
            "activation": self.activation, "engagement_count": self.engagement_count,
            "parameters": {
                "input_gain": self.input_gain, "recurrent_gain": self.recurrent_gain,
                "inhibition_gain": self.inhibition_gain, "decay_rate": self.decay_rate,
            },
            "history": list(self.history[-17:]),
        }


class DevelopmentalBiologicalNetwork:
    """Coordinate distributed sensory, language, memory, control, and white-matter surrogates."""

    def __init__(self) -> None:
        """Initialize distinct developmental subsystems with different dynamical parameters."""
        definitions = (
            ("auditory_perception", "Cochlea/brainstem/thalamic/auditory cortical pathways", 1.9, 0.15, 0.25, 0.18),
            ("auditory_feature_analysis", "Auditory cortical multi-timescale feature analysis", 1.8, 0.22, 0.22, 0.16),
            ("superior_temporal_speech", "Superior temporal speech-sound and sequence processing", 1.7, 0.35, 0.20, 0.20),
            ("statistical_pattern_learning", "Distributed auditory and statistical-learning systems", 1.6, 0.42, 0.18, 0.24),
            ("multisensory_social_learning", "Visual, auditory, joint-attention, and social interaction systems", 1.5, 0.30, 0.22, 0.22),
            ("temporal_semantic_learning", "Temporal semantic and lexical knowledge networks", 1.55, 0.40, 0.22, 0.24),
            ("distributed_conceptual_semantic", "Distributed sensory/action-linked conceptual systems", 1.50, 0.43, 0.20, 0.25),
            ("hippocampal_episodic_memory", "Hippocampal and medial-temporal episodic binding and retrieval", 1.55, 0.50, 0.15, 0.28),
            ("frontoparietal_working_memory", "Parietal and frontoparietal maintenance and attention networks", 1.75, 0.36, 0.30, 0.18),
            ("prefrontal_executive_control", "Prefrontal and distributed frontoparietal executive control", 1.85, 0.38, 0.32, 0.16),
            ("activity_dependent_circuit_plasticity", "Distributed synaptic/circuit plasticity and refinement", 1.45, 0.48, 0.14, 0.30),
            ("white_matter_myelination", "Distributed white-matter pathway maturation and communication timing", 1.25, 0.55, 0.10, 0.34),
            ("hippocampal_prefrontal_integration", "Developing interaction between episodic memory and prefrontal control", 1.55, 0.50, 0.24, 0.24),
            ("frontoparietal_multistep_planning", "Frontoparietal working-memory and multi-step task coordination", 1.75, 0.42, 0.28, 0.19),
            ("developmental_calibration", "Distributed experience-dependent adaptation across developmental domains", 1.40, 0.44, 0.18, 0.27),
        )
        self.systems = {
            name: DevelopmentalNetworkActivity(name, scope, input_gain, recurrent, inhibition, decay)
            for name, scope, input_gain, recurrent, inhibition, decay in definitions
        }

    def engage(self, name: str, operation: str, drive: float) -> dict[str, Any]:
        """Engage one named network and reject unknown subsystem routing."""
        if name not in self.systems:
            raise KeyError("unknown developmental network: " + name)
        return self.systems[name].engage(operation, drive)

    def snapshot(self) -> dict[str, Any]:
        """Return all network activity states and the biological interpretation boundary."""
        return {
            **{name: network.snapshot() for name, network in self.systems.items()},
            "scientific_scope": (
                "Rate-coded distributed network surrogates, not direct neural measurements, "
                "anatomical replicas, or literal neural equations."
            ),
            "age_scope": (
                "Development is continuous and variable. Age is contextual metadata, not a diagnostic "
                "cutoff or a single maturity score."
            ),
        }
