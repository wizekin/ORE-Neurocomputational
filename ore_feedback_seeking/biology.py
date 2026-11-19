"""Distributed population-rate reference model for Point 07 feedback processes.

This module is an engineering analogy for interacting cognitive systems. It is not
an anatomically fitted, cellular, or physiologically validated human brain model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


def _logistic(value: float) -> float:
    """Apply a stable logistic activation to a population's net input."""
    if value >= 0.0:
        exp_value = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(max(value, -700.0))
    return exp_value / (1.0 + exp_value)


@dataclass(slots=True)
class NeuralPopulation:
    """Hold one coarse functional population's rate and membrane-like time constant."""

    name: str
    rate: float = 0.05
    time_constant_ms: float = 80.0
    baseline: float = 0.0

    def __post_init__(self) -> None:
        """Validate the reference population's bounded firing-rate proxy."""
        if not self.name:
            raise ValueError("Population name must not be empty.")
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("Population rate must be in [0, 1].")
        if self.time_constant_ms <= 0.0:
            raise ValueError("time_constant_ms must be positive.")

    def update(self, drive: float, recurrent_input: float, dt_ms: float) -> float:
        """Advance a leaky rate variable toward a saturating target activation."""
        target = _logistic(self.baseline + drive + recurrent_input)
        alpha = 1.0 - math.exp(-max(0.0, dt_ms) / self.time_constant_ms)
        self.rate = min(1.0, max(0.0, self.rate + alpha * (target - self.rate)))
        return self.rate

    def reset(self) -> None:
        """Restore the population to its low-activity reference baseline."""
        self.rate = 0.05

    def snapshot(self) -> dict[str, float | str]:
        """Return the current rate and time constant for trace recording."""
        return {"name": self.name, "rate": self.rate,
                "time_constant_ms": self.time_constant_ms}


class DistributedFeedbackNetwork:
    """Simulate interacting monitoring, memory, language, attention, and control populations."""

    POPULATIONS = (
        "prefrontal_goal_control", "medial_frontal_error_monitoring",
        "frontoparietal_attention", "working_memory", "temporal_semantic",
        "hippocampal_relational_memory", "language_explanation",
        "amygdala_salience", "striatal_value_learning", "cortical_plasticity",
    )

    def __init__(self, dt_ms: float = 10.0, substeps: int = 4) -> None:
        """Initialize coarse populations and a sparse, signed recurrent connectivity map."""
        if dt_ms <= 0.0 or substeps < 1:
            raise ValueError("dt_ms and substeps must be positive.")
        self.dt_ms = float(dt_ms)
        self.substeps = int(substeps)
        time_constants = (95.0, 45.0, 60.0, 80.0, 110.0,
                          130.0, 75.0, 55.0, 90.0, 150.0)
        self.populations = {
            name: NeuralPopulation(name, time_constant_ms=time_constants[index], baseline=-1.8)
            for index, name in enumerate(self.POPULATIONS)
        }
        self.connections: dict[tuple[str, str], float] = {
            ("prefrontal_goal_control", "frontoparietal_attention"): 0.35,
            ("prefrontal_goal_control", "working_memory"): 0.28,
            ("medial_frontal_error_monitoring", "prefrontal_goal_control"): 0.32,
            ("medial_frontal_error_monitoring", "frontoparietal_attention"): 0.22,
            ("frontoparietal_attention", "working_memory"): 0.30,
            ("frontoparietal_attention", "temporal_semantic"): 0.18,
            ("temporal_semantic", "hippocampal_relational_memory"): 0.20,
            ("temporal_semantic", "language_explanation"): 0.35,
            ("hippocampal_relational_memory", "cortical_plasticity"): 0.28,
            ("language_explanation", "prefrontal_goal_control"): 0.16,
            ("amygdala_salience", "hippocampal_relational_memory"): 0.18,
            ("amygdala_salience", "prefrontal_goal_control"): -0.12,
            ("striatal_value_learning", "prefrontal_goal_control"): 0.16,
            ("prefrontal_goal_control", "cortical_plasticity"): 0.20,
        }
        self.steps = 0
        self.reward_prediction_error = 0.0

    def reset(self) -> None:
        """Reset population rates and counters before an independent simulation episode."""
        for population in self.populations.values():
            population.reset()
        self.steps = 0
        self.reward_prediction_error = 0.0

    def _recurrent_input(self, target: str, previous: Mapping[str, float]) -> float:
        """Sum the previous-step rates multiplied by signed connection weights."""
        return sum(weight * previous[source]
                   for (source, destination), weight in self.connections.items()
                   if destination == target)

    def advance(self, drives: Mapping[str, float], reward_prediction_error: float | None = None) -> dict[str, Any]:
        """Advance the distributed network while keeping semantic mismatch distinct from reward error."""
        unknown = set(drives) - set(self.POPULATIONS)
        if unknown:
            raise KeyError(f"Unknown biological populations: {sorted(unknown)}")
        clean_drives = {name: min(3.0, max(-3.0, float(value))) for name, value in drives.items()}
        # This is a phasic optional reward/outcome input; it is never inferred from
        # a factual mismatch or semantic correction and does not persist to the next step.
        self.reward_prediction_error = (
            float(reward_prediction_error) if reward_prediction_error is not None else 0.0
        )
        for _ in range(self.substeps):
            previous = {name: population.rate for name, population in self.populations.items()}
            for name, population in self.populations.items():
                external = clean_drives.get(name, 0.0)
                if name == "striatal_value_learning":
                    external += max(-1.0, min(1.0, self.reward_prediction_error))
                recurrent = self._recurrent_input(name, previous)
                population.update(external, recurrent, self.dt_ms)
        self.steps += 1
        snapshot = {name: population.rate for name, population in self.populations.items()}
        return {"population_rates": snapshot, "step": self.steps,
                "reward_prediction_error": self.reward_prediction_error,
                "semantic_error_is_reward_error": False,
                "biological_scope": "distributed population-rate engineering analogy"}

    def snapshot(self) -> dict[str, Any]:
        """Return all population definitions, current rates, and model-scope limitations."""
        return {"populations": {name: population.snapshot()
                                for name, population in self.populations.items()},
                "connections": len(self.connections), "steps": self.steps,
                "reward_prediction_error": self.reward_prediction_error,
                "limitations": [
                    "Not a cellular or anatomically fitted simulation.",
                    "Rates and connection weights are engineering parameters, not measured neural values.",
                    "Brain regions are represented as interacting functional populations.",
                    "Conceptual correctness mismatch is not treated as a dopamine reward-prediction error.",
                ]}


def biological_drives_for_step(step_number: int, request: Any, outputs: Mapping[str, Any]) -> dict[str, float]:
    """Translate functional stage measurements into coarse population input drives."""
    uncertainty = float(outputs.get("step_03_monitor_comprehension", {}).get("uncertainty", 0.0))
    incompleteness = 1.0 - float(outputs.get("step_01_define_target", {}).get("completeness", 1.0))
    source_quality = float(outputs.get("step_07_verify_feedback", {}).get("feedback_quality", 0.0))
    error_score = float(outputs.get("step_08_classify_error", {}).get("error_probabilities", {}).get(
        outputs.get("step_08_classify_error", {}).get("most_likely_error", ""), 0.0))
    load = float(outputs.get("step_15_control_feedback_load", {}).get("feedback_load", 0.0))
    drives = {
        "prefrontal_goal_control": 0.30 + 0.55 * incompleteness,
        "medial_frontal_error_monitoring": 0.15 + 0.75 * uncertainty + 0.30 * error_score,
        "frontoparietal_attention": 0.15 + 0.55 * uncertainty + 0.20 * request.importance,
        "working_memory": 0.10 + 0.55 * load + 0.20 * uncertainty,
        "temporal_semantic": 0.10 + 0.35 * source_quality + 0.20 * bool(request.supporting_concepts),
        "hippocampal_relational_memory": 0.10 + 0.40 * bool(request.reasoning_chain) + 0.35 * bool(request.supporting_concepts),
        "language_explanation": 0.10 + 0.45 * (step_number in {2, 6, 12, 13, 16}),
        "amygdala_salience": 0.10 + 0.40 * request.stakes + 0.25 * request.importance,
        "striatal_value_learning": 0.05 + 0.25 * request.importance,
        "cortical_plasticity": 0.05 + 0.55 * source_quality + 0.30 * bool(outputs.get("step_10_apply_targeted_correction", {}).get("correction_applied")),
    }
    return drives
