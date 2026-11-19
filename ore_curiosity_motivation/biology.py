"""Distributed, population-level biological reference model for Point 08 motivated learning.

The network is a transparent engineering abstraction, not a fitted human neural simulation.
Rates and connections are not measured physiological values.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


def _sigmoid(value: float) -> float:
    """Compute a stable saturating activation."""
    if value >= 0.0:
        z = math.exp(-min(value, 700.0))
        return 1.0/(1.0+z)
    z = math.exp(max(value, -700.0))
    return z/(1.0+z)


@dataclass(slots=True)
class NeuralPopulation:
    """Hold a bounded population-rate proxy and its first-order response time."""

    name: str
    rate: float = 0.05
    time_constant_ms: float = 80.0
    baseline: float = -1.8

    def __post_init__(self) -> None:
        """Validate population identity, bounded rate, and positive time constant."""
        if not self.name or not 0.0 <= self.rate <= 1.0 or self.time_constant_ms <= 0.0:
            raise ValueError("Invalid population configuration.")

    def update(self, drive: float, recurrent: float, dt_ms: float) -> float:
        """Advance a leaky rate equation toward a sigmoid target."""
        target = _sigmoid(self.baseline+drive+recurrent)
        alpha = 1.0-math.exp(-max(0.0, dt_ms)/self.time_constant_ms)
        self.rate = min(1.0, max(0.0, self.rate+alpha*(target-self.rate)))
        return self.rate

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe population state."""
        return {"name": self.name, "rate": self.rate, "time_constant_ms": self.time_constant_ms}


class DistributedMotivationNetwork:
    """Coordinate coarse goal, curiosity, valuation, attention, affect, memory, and plasticity populations."""

    POPULATIONS = (
        "prefrontal_goal_control", "anterior_cingulate_monitoring",
        "frontoparietal_attention", "midbrain_dopamine_proxy",
        "ventral_striatal_value", "hippocampal_memory",
        "amygdala_emotional_salience", "locus_coeruleus_arousal",
        "hypothalamic_stress_axis", "cortical_plasticity",
    )

    def __init__(self, dt_ms: float = 10.0, substeps: int = 4) -> None:
        """Initialize ten interacting coarse populations and a sparse signed connectivity matrix."""
        if dt_ms <= 0.0 or substeps < 1:
            raise ValueError("dt_ms and substeps must be positive.")
        self.dt_ms = float(dt_ms)
        self.substeps = int(substeps)
        taus = (95.0, 50.0, 65.0, 45.0, 75.0, 125.0, 60.0, 55.0, 180.0, 150.0)
        self.populations = {
            name: NeuralPopulation(name, time_constant_ms=taus[index])
            for index, name in enumerate(self.POPULATIONS)
        }
        self.connections = {
            ("prefrontal_goal_control", "frontoparietal_attention"): 0.30,
            ("anterior_cingulate_monitoring", "prefrontal_goal_control"): 0.24,
            ("frontoparietal_attention", "hippocampal_memory"): 0.22,
            ("hippocampal_memory", "cortical_plasticity"): 0.32,
            ("midbrain_dopamine_proxy", "ventral_striatal_value"): 0.38,
            ("ventral_striatal_value", "prefrontal_goal_control"): 0.18,
            ("locus_coeruleus_arousal", "frontoparietal_attention"): 0.20,
            ("amygdala_emotional_salience", "hippocampal_memory"): 0.20,
            ("hypothalamic_stress_axis", "prefrontal_goal_control"): -0.18,
            ("hypothalamic_stress_axis", "hippocampal_memory"): -0.12,
            ("prefrontal_goal_control", "cortical_plasticity"): 0.20,
            ("anterior_cingulate_monitoring", "locus_coeruleus_arousal"): 0.12,
        }
        self.steps = 0
        self.last_reward_prediction_error: float | None = None

    def reset(self) -> None:
        """Reset population activity and the optional outcome-prediction-error channel."""
        for population in self.populations.values():
            population.rate = 0.05
        self.steps = 0
        self.last_reward_prediction_error = None

    def advance(self, drives: Mapping[str, float],
                reward_prediction_error: float | None = None) -> dict[str, Any]:
        """Advance the network; dopamine proxy receives only explicitly observed outcome prediction error."""
        unknown = set(drives)-set(self.POPULATIONS)
        if unknown:
            raise KeyError(f"Unknown populations: {sorted(unknown)}")
        self.last_reward_prediction_error = reward_prediction_error
        clean = {name: min(3.0, max(-3.0, float(value))) for name, value in drives.items()}
        for _ in range(self.substeps):
            previous = {name: population.rate for name, population in self.populations.items()}
            for name, population in self.populations.items():
                recurrent = sum(weight*previous[source]
                                for (source, target), weight in self.connections.items()
                                if target == name)
                drive = clean.get(name, 0.0)
                if name == "midbrain_dopamine_proxy" and reward_prediction_error is not None:
                    drive += max(-1.0, min(1.0, float(reward_prediction_error)))
                population.update(drive, recurrent, self.dt_ms)
        self.steps += 1
        return {
            "population_rates": {name: population.rate for name, population in self.populations.items()},
            "step": self.steps,
            "reward_prediction_error": reward_prediction_error,
            "semantic_or_novelty_mismatch_is_not_reward_error": True,
            "biological_scope": "distributed population-rate engineering abstraction",
        }

    def snapshot(self) -> dict[str, Any]:
        """Return the network state and key limitations."""
        return {"populations": {name: population.snapshot()
                                for name, population in self.populations.items()},
                "connections": len(self.connections), "steps": self.steps,
                "last_reward_prediction_error": self.last_reward_prediction_error,
                "limitations": [
                    "No cellular, anatomical, or subject-specific physiological fitting.",
                    "Population rates and weights are engineering parameters, not measured neural activity.",
                    "Curiosity, dopamine, reward, attention, emotion, and memory are distributed and task-dependent.",
                    "A software softmax or logistic equation is not asserted to be a literal neural calculation.",
                ]}


def biological_drives_for_step(request: Any, outputs: Mapping[str, Any]) -> dict[str, float]:
    """Translate current functional outputs into coarse, explicitly labelled population drives."""
    curiosity = float(outputs.get("step_05_estimate_curiosity", {}).get("curiosity", 0.0))
    motivation = float(outputs.get("step_06_separate_motivation_sources", {}).get("total_motivation", 0.0))
    attention = float(outputs.get("step_10_allocate_attention", {}).get("attention_weights_sum", 0.0))
    threat = float(outputs.get("step_15_appraise_emotion", {}).get("threat_estimate", request.stress))
    engagement = float(outputs.get("step_16_integrate_motivation_and_emotion", {}).get("engagement", 0.0))
    disengagement = float(outputs.get("step_17_detect_disengagement", {}).get("disengagement_score", 0.0))
    encoding = float(outputs.get("step_13_encode_target_information", {}).get("mean_target_trace_strength", 0.0))
    gap = float(outputs.get("step_02_detect_novelty_and_information_gaps", {}).get("mean_information_gap", 0.0))
    consolidation = float(outputs.get("step_22_support_consolidation", {}).get("mean_consolidation_strength", 0.0))
    return {
        "prefrontal_goal_control": 0.20+0.65*motivation+0.15*engagement,
        "anterior_cingulate_monitoring": 0.10+0.60*gap+0.30*disengagement,
        "frontoparietal_attention": 0.10+0.60*attention+0.30*curiosity,
        "midbrain_dopamine_proxy": 0.05+0.35*curiosity+0.25*motivation,
        "ventral_striatal_value": 0.10+0.65*motivation,
        "hippocampal_memory": 0.10+0.55*encoding+0.25*consolidation,
        "amygdala_emotional_salience": 0.10+0.55*request.emotional_relevance+0.35*request.emotional_arousal,
        "locus_coeruleus_arousal": 0.10+0.55*request.attention_available+0.35*request.emotional_arousal,
        "hypothalamic_stress_axis": 0.10+0.80*threat,
        "cortical_plasticity": 0.05+0.55*encoding+0.25*engagement,
    }
