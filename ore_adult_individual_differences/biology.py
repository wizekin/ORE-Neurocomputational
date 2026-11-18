"""Distributed neural-network surrogates for adult comprehension and individual differences.

These bounded rate-coded networks are inspectable functional abstractions, not neural recordings,
biophysical simulations, or one-region/one-operation anatomical claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


def _unit(value: float) -> float:
    """Validate and clamp a finite activation value to [0, 1]."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("activation must be finite")
    return max(0.0, min(1.0, number))


def _sigmoid(value: float) -> float:
    """Compute a numerically stable logistic population activation."""
    if value >= 0.0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


@dataclass
class AdultNetworkActivity:
    """Track one distributed system with input drive, recurrence, inhibition, and decay."""

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
        """Validate finite gains and normalized activity parameters."""
        for field_name in ("input_gain", "recurrent_gain", "inhibition_gain"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(field_name + " must be finite and non-negative")
            setattr(self, field_name, value)
        self.decay_rate = _unit(self.decay_rate)
        self.activation = _unit(self.activation)

    def engage(self, operation: str, drive: float) -> dict[str, Any]:
        """Update a rate-coded population using drive, recurrence, inhibition, and retained activity."""
        current_drive = _unit(drive)
        previous = self.activation
        net_input = (
            self.input_gain * current_drive
            + self.recurrent_gain * previous
            - self.inhibition_gain * previous * previous
        )
        target = _sigmoid(net_input)
        self.activation = _unit(
            (1.0 - self.decay_rate) * target + self.decay_rate * previous
        )
        self.engagement_count += 1
        row = {
            "network": self.name, "operation": str(operation),
            "drive_proxy": current_drive, "activation_before": previous,
            "activation_after": self.activation, "net_input": net_input,
            "target_activation": target,
            "rate_equation": (
                "a_t=(1-decay)*sigmoid(input_gain*drive+recurrent_gain*a_prev"
                "-inhibition_gain*a_prev^2)+decay*a_prev"
            ),
            "evidence_boundary": "distributed_rate_coded_functional_surrogate",
        }
        self.history.append(row)
        return row

    def snapshot(self) -> dict[str, Any]:
        """Return model parameters and the most recent network activity records."""
        return {
            "name": self.name, "biological_scope": self.biological_scope,
            "activation": self.activation, "engagement_count": self.engagement_count,
            "parameters": {
                "input_gain": self.input_gain, "recurrent_gain": self.recurrent_gain,
                "inhibition_gain": self.inhibition_gain, "decay_rate": self.decay_rate,
            },
            "history": list(self.history[-15:]),
        }


class AdultComprehensionBiologicalNetwork:
    """Coordinate distributed attention, semantic, language, memory, and control surrogates."""

    def __init__(self) -> None:
        """Initialize distinct but interacting systems with different rate-dynamics parameters."""
        definitions = (
            ("goal_attention_control", "Prefrontal and parietal goal maintenance and attentional selection", 1.90, 0.38, 0.34, 0.18),
            ("frontoparietal_working_memory", "Distributed frontoparietal maintenance, updating, and interference control", 1.85, 0.42, 0.35, 0.16),
            ("temporal_processing_coordination", "Distributed timing and coordination across perceptual and integration operations", 1.65, 0.34, 0.22, 0.22),
            ("cortical_semantic_knowledge", "Distributed cortical semantic and conceptual knowledge structures", 1.55, 0.48, 0.18, 0.25),
            ("expertise_schema_chunking", "Knowledge-driven schema activation and expert chunk recognition across systems", 1.60, 0.48, 0.20, 0.24),
            ("distributed_language_network", "Distributed frontotemporal language networks shaped by proficiency and language use", 1.75, 0.43, 0.22, 0.20),
            ("contextual_social_semantics", "Context-sensitive semantic and social knowledge systems", 1.45, 0.35, 0.18, 0.26),
            ("hippocampal_memory_binding", "Hippocampal and medial-temporal episodic binding and source retrieval", 1.65, 0.48, 0.18, 0.27),
            ("hypothesis_conflict_monitor", "Distributed monitoring and control of competing interpretations", 1.80, 0.36, 0.38, 0.18),
            ("adaptive_support_control", "Goal-directed adjustment of scaffolding and processing strategy", 1.55, 0.42, 0.24, 0.25),
            ("gist_detail_source_memory", "Interacting semantic gist, episodic detail, and source/context retrieval systems", 1.50, 0.46, 0.20, 0.28),
            ("prediction_error_knowledge_update", "Hippocampal-cortical learning, mismatch monitoring, and knowledge revision", 1.65, 0.45, 0.20, 0.26),
            ("confidence_decision_monitor", "Distributed metacognitive monitoring and action selection", 1.70, 0.40, 0.28, 0.22),
        )
        self.systems = {
            name: AdultNetworkActivity(name, scope, input_gain, recurrent, inhibition, decay)
            for name, scope, input_gain, recurrent, inhibition, decay in definitions
        }

    def engage(self, name: str, operation: str, drive: float) -> dict[str, Any]:
        """Engage a named network and reject unknown routing names."""
        if name not in self.systems:
            raise KeyError("unknown adult-comprehension network: " + name)
        return self.systems[name].engage(operation, drive)

    def snapshot(self) -> dict[str, Any]:
        """Return distributed network activity and the interpretation boundary."""
        return {
            **{name: item.snapshot() for name, item in self.systems.items()},
            "scientific_scope": (
                "Bounded rate-coded surrogates for interacting distributed networks; "
                "not direct neural measurements, biophysical simulations, or one-region mappings."
            ),
            "individual_difference_scope": (
                "The modeled state is task- and context-sensitive. It is not an intelligence, "
                "ability, cultural-group, bilingualism, or age-related diagnosis."
            ),
        }
